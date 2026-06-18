# ADR 0004 — Self-Contained Deployment (one binary, zero external services)

- **Status:** Proposed — **Rev 1** (critique-gate findings folded in; see §13 Addendum A, binding)
- **Date:** 2026-06-13 (Rev 1: 2026-06-14)
- **Owner:** solution-architect
- **Builds on:** ADR 0001 (`0001-bypass-escalation-ladder.md`), ADR 0002 (`0002-data-layer-acquisition.md`), ADR 0003 (`0003-single-ip-escalation-ladder.md`).
- **Constraint inherited (load-bearing):** ADR 0003 §1.1 — the scraper runs from **ONE static IP**, no rotation, ever. A hard ban is permanent. Reputation is a non-renewable budget. **This ADR's primary decision (how to replace SearXNG) is dominated by that constraint.**
- **Scope guard (GOAL.md §2):** sets contracts only; writes no feature code. New behavior is config-gated and the *default* path must run standalone.

> **Goal of this ADR:** make `./web-search-mcp` run end-to-end with **no docker, no external service, and no mandatory secret**, while preserving the coverage goal (GOAL.md G1 ≥ 90%). The single binary already exists (`crates/mcp-server`, bin `web-search-mcp`, stdio MCP via rmcp — verified `main.rs`/`server.rs`). What breaks "self-contained" is three runtime dependencies, addressed in §3 (search), §6 (models), §7 (config).

---

## 1. Context and forces in tension

### 1.1 What is actually NOT self-contained today (ground truth — verified by reading source)

| # | Dependency | Where | Why it breaks standalone |
|---|---|---|---|
| 1 | **SearXNG container** (`searxng/searxng` on :8080) | `CrawlerConfig.searxng_url` (`config.rs:36-40`); consumed in `engine.rs` `quick_search`/`instant_search` fast paths (`engine.rs:532`, `:755`) via `fetch_searxng_results` (`engine.rs:1609`) and `generate_search_seeds` (`engine.rs:1658`) | Requires the operator to `docker run -d -p 8080:8080 searxng/searxng` before the binary is useful on its primary path. This is the dominant blocker. |
| 2 | **ML models from HuggingFace** | `candle_embedder.rs:92-107` (`hf_hub::api::sync::Api` fetch of `config.json`/`tokenizer.json`/`model.safetensors` for `sentence-transformers/all-MiniLM-L6-v2`); cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` | First run needs network to populate the HF cache. Offline-from-cold fails the neural path. |
| 3 | **Hardcoded config path** | `server.rs:36` loads `config/default.toml` from the **process CWD**, no override | Run the binary from any other directory and it silently falls back to `Config::default()`. No env/CLI override exists. |

### 1.2 Two important nuances the design must respect

- **SearXNG is not load-bearing for correctness — only for the *fast path*.** `engine.rs` already has a **fallback**: when `searxng_url` is `None` or SearXNG returns 0 results, it calls `generate_search_seeds(query, None)` and crawls search-engine SERPs directly via the crawler (`engine.rs:566-600`, `:822-831`). So "remove SearXNG" = "make the existing fallback the primary path, reliably." It is a *quality/latency* regression risk, not a *capability* one.
- **The `force_cpu` footgun (verified, must flag).** `create_embedder` (`embedder/src/lib.rs:84-108`): when `config.embedder.force_cpu == true`, it **skips the neural embedder entirely** and returns `HashEmbedder` (feature-hashing, *no semantic matching* — confirmed at `lib.rs:85-87` + the "no semantic matching" log at `:113`). A user setting `force_cpu` to avoid a GPU silently degrades semantic search to keyword hashing. This is a config trap that the self-contained story must not inherit (§6.3).

### 1.3 Forces in tension

| Force | Pressure |
|---|---|
| **Zero external setup (this ADR's reason to exist)** | `./web-search-mcp` must produce useful results with no docker, no key, no pre-staged files. |
| **Single-IP reputation (ADR 0003, dominant)** | The search step itself now spends our one IP's reputation against the engines. SERP scraping is **high-frequency, high-value** traffic to a *small* set of hostile domains (Google/Bing/DDG) — exactly the reputation-fragile pattern ADR 0003 §1.1 is built to avoid. Moving search in-process concentrates ban risk on the surfaces we can least afford to lose. |
| **Coverage / quality (GOAL.md G1 ≥ 90%)** | SearXNG existed *because* direct SERP scraping trips bot-detection. Removing it must not collapse result quality. |
| **"Self-contained" definition** | Is an optional, **default-off** API key still "self-contained"? Decision: a binary that **runs fully without any key** is self-contained; an *optional* key that *improves* a default-working system does not break that property (§3, Option E). A binary that *requires* a key to function does. |
| **Offline-from-cold** | First run with no network should still **start** and degrade predictably, not panic. |

### 1.4 Architecturally significant requirements (ASRs)

- **ASR-SC1 (zero-setup default):** with an empty data dir, no config file, no env vars, and no docker, `./web-search-mcp` starts, serves MCP, and returns non-empty results for a common query — or degrades with an explicit, typed warning. No silent dead path.
- **ASR-SC2 (reputation-safe search):** the in-process search step obeys the ADR 0003 governor — SERP engines are live-origin domains, paced and breaker-gated like any other, **never** hammered. A SERP hard-ban must not be able to take the whole product down.
- **ASR-SC3 (no mandatory secret):** the default path uses no API key. Any key is opt-in, env-var *reference* (never a literal — api-security SECRETS rule), and only *augments* results.
- **ASR-SC4 (offline-from-cold safety):** model fetch failure → explicit warning + named-quality fallback, never a panic and never a *silent* drop to keyword-only.
- **ASR-SC5 (config portability):** config path overridable by env/CLI; embedded defaults are complete and correct so no file is required.

---

## 2. The crux: how to replace the SearXNG dependency

SearXNG aggregates Google+Bing+DDG server-side and returns clean JSON with **no CAPTCHA to us** because *its* host absorbs the bot-detection cost. Removing the container means the binary must obtain SERPs another way, in-process. The five candidates below are each evaluated against: **reliability**, **single-IP reputation cost** (the ADR 0003 dominant force), **needs key/container?**, **maintenance burden**, **coverage/quality**.

### 2.1 Options table (verified, June 2026)

| Option | Reliability | Single-IP reputation cost | Key / container? | Maintenance | Coverage/quality | Verdict |
|---|---|---|---|---|---|---|
| **A. In-process SERP scraping as PRIMARY** (DDG HTML/Lite, Bing, Google via the crawler + ADR 0003 ladder) | **Low-medium.** DuckDuckGo Lite is **actively rate-limited in 2025-2026** — confirmed `202 Ratelimit` errors in the wild; "needs residential proxies for reliable access" [DDG-1, DDG-2]. Google/Bing SERP HTML is JS-gated/CAPTCHA-prone (the original reason SearXNG was added). | **High and concentrated.** SERP engines are a tiny set of hostile domains hit on *every* query. A hard-ban here (permanent under single-IP, ADR 0003 §4.2) kills the primary search surface with no recovery. This is the worst-case reputation profile. | **No** (zero external) | Medium-high: SERP HTML parsers drift constantly (the `search_results.rs` parsers already carry "2025-2026 update" patches — verified `search_results.rs:89-101`, `:167-177`, `:321-344`). | Medium when it works; brittle. | **Necessary fallback, unsafe as sole primary.** |
| **B. Free search API** (Brave / Tavily / Serper / Mojeek) | **High** (vendor absorbs bot-detection). | **Zero to our IP** — vendor egresses. Best possible reputation profile. | **Yes, a key.** And the free tiers shifted in 2026: **Brave killed its free tier in Feb 2026** — now $5 metered credits + **credit card required**, no spend cap [BRAVE-1, BRAVE-2]. **Tavily** still has a genuine free tier: **1,000 credits/month, NO credit card, API key only, basic search = 1 credit, no rollover** [TAVILY-1, TAVILY-2]. Mojeek API is trial/quote-only, not a documented free tier [MOJEEK-1]. | Low (HTTP + JSON). | High and clean. | **Reintroduces a key — but optional, it is the best augmentation.** |
| **C. Embed/vendor a Rust metasearch lib in-process** (`websearch`, `h2m-search`) | Inherits whatever the lib's providers are. | **Same as A or B depending on provider** — these crates are *thin multiplexers*: their providers are the same scraped engines (DDG) **or** the same keyed APIs (Tavily/Brave/Exa/SerpAPI) [RUST-1, RUST-2]. They do **not** remove the underlying dependency — they wrap it. | Depends on provider chosen. | New third-party dep on the critical path; maturity unverified; duplicates logic we already own in `search_results.rs`. | No new capability over A/B. | **Rejected — adds a dependency without removing one.** Wrapping our existing scrapers in someone else's multiplexer trades code we control for code we don't, for zero net capability. |
| **D. Binary shells out to `docker run searxng/searxng`** | High (it *is* SearXNG). | Zero to our IP (SearXNG egresses) — but SearXNG itself then scrapes from… our host's IP. Reputation cost merely moved one hop, still our IP. | **Still needs docker.** Defeats the entire purpose. | Adds container lifecycle mgmt to the binary. | Same as today. | **Rejected — explicitly fails "zero external setup."** Documented only to close it out. |
| **E. HYBRID: reputation-safe in-process default + optional keyed-API augmentation + config switch (CHOSEN)** | **High by composition.** Default = the existing ADR-0003-governed crawler fallback (works with no setup); when a key is present, prefer the vendor API (reliable, zero-IP-cost) and fall back to the governed crawler on miss/exhaustion. | **Minimized.** With no key: SERP scraping is *demoted to one governed surface among many* (the crawler already seeds Wikipedia, arXiv, Reddit, HN-Algolia-JSON, StackExchange, and entity-official-domains directly — verified `generate_search_seeds`, `engine.rs:1690-1716`), so a SERP ban degrades, not kills. With a key: zero IP cost. | **No key required**; key optional. | Reuses 100% of existing `search_results.rs` + `generate_search_seeds`; adds one small keyed-API client + a source selector. | High; graceful. | **CHOSEN.** |

### 2.2 Decision — Option E (Hybrid), with the **keyless governed-crawler path as the default and the contractual floor**

**Rationale, grounded in the constraints:**

1. **Self-contained is satisfied without any key.** Option E's *default* (no key, no docker) is exactly the fallback path that already exists in `engine.rs` — we are promoting and hardening it, not inventing it. This directly satisfies ASR-SC1/ASR-SC3.
2. **The single-IP constraint (ADR 0003, the dominant force) forbids making raw SERP scraping the *sole* primary.** Option A concentrates permanent-ban risk on Google/Bing/DDG, the surfaces we hit on every query and can least afford to lose. Option E mitigates this two ways: (a) it **does not depend on SERP engines** — `generate_search_seeds` already produces direct, low-ban-risk seeds (Wikipedia article URLs, arXiv search, HN Algolia **JSON API**, StackExchange, Reddit, plus official-domain `site:` seeds), so even with every SERP engine banned the crawler still has entry points; (b) every SERP hit goes through the ADR 0003 governor (paced, budgeted, breaker-gated) — satisfying ASR-SC2.
3. **The optional key buys reliability at zero IP cost — the best augmentation, and it does not compromise "self-contained."** A binary that *runs fully without a key* is self-contained; an opt-in key that *improves* an already-working system is an enhancement, not a dependency (the §1.3 definition). **Tavily is the recommended optional provider** because it is the only verified June-2026 option with a true no-credit-card free tier (1,000 searches/month) [TAVILY-1, TAVILY-2]. **Brave is explicitly NOT recommended as the default optional** because its Feb-2026 change requires a credit card with no spend cap [BRAVE-1] — that is a footgun for a "just run it" tool.
4. **Options C and D are rejected for adding a dependency without removing the real one** (C wraps our own scrapers; D still needs docker).

**The contractual floor:** the product MUST be fully functional with **no key and no container**. The keyed API is strictly additive. This is the testable acceptance criterion (§9).

---

## 3. Target search architecture (data flow)

```
                         ./web-search-mcp   (single binary, stdio MCP)
                                  │
                   ┌──────────────┴───────────────┐
                   │   SearchSource selector       │   (NEW, replaces the hardcoded
                   │   resolve order, per config   │    `if searxng_url` branch)
                   └──────────────┬───────────────┘
                                  │
        ┌─────────────────────────┼──────────────────────────────────────────┐
        │ has key (opt-in, env)?  │                no key (DEFAULT)            │
        ▼                         │                                            ▼
  ┌───────────────────┐           │                          ┌────────────────────────────┐
  │ KEYED API client  │  zero-IP  │                          │ GOVERNED CRAWLER SEEDS       │
  │ (Tavily default;  │  cost     │                          │ generate_search_seeds(q,None)│
  │  Serper/SearXNG-  │           │                          │  • Wikipedia article + search│
  │  url also allowed)│           │                          │  • arXiv / HN-Algolia JSON    │
  └─────────┬─────────┘           │                          │  • Reddit / StackExchange     │
            │ results? ──yes──────┼─────────► SERP results    │  • official-domain `site:`    │
            │ no / quota / error  │           (url+title+snip)│  • Google/Bing/DDG/Brave SERP │
            ▼                     │                ▲          │    (LOW priority, ADR-0003    │
  ┌───────────────────┐          │                │          │     GOVERNED: paced, budgeted,│
  │  FALL THROUGH  ───────────────┘                │          │     breaker-gated, permanent- │
  │  to governed crawler (same as default)         │          │     denylist-aware)           │
  └───────────────────┐                            │          └──────────────┬───────────────┘
                       └────────────────────────────┘                         │
                                  results merged ──► crawl top-N (fetch_urls_concurrent)
                                                     ──► extract ──► index ──► rank (existing pipeline)
```

**Sync vs async:** the keyed-API call and the crawler seeds are both async HTTP; the selector races/falls-through, it does not block. **Data ownership:** unchanged — the engine owns the index/vectors/cache; the search source only produces `(url, title, snippet)` triples (the existing `SearxngResult` shape, `engine.rs:1601`, renamed in §8).

---

## 4. The search-source contract (what backend-engineer implements)

A single trait abstracts "where SERP-like results come from," so the engine stops branching on `searxng_url`:

```rust
/// A source of search-result triples for a query. Implementations MUST be
/// reputation-safe per ADR 0003 (live-origin sources go through the governor).
#[async_trait]
pub trait SearchSource: Send + Sync {
    /// Name for logging/telemetry (e.g. "tavily", "searxng", "crawler-seeds").
    fn name(&self) -> &str;
    /// Returns ranked result triples, or an empty Vec on miss (NOT an error —
    /// empty means "I had nothing", error means "I failed and the caller should
    /// fall through"). Never panics; never blocks the runtime.
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>>;
    /// True iff this source touches the live origin / our IP reputation
    /// (crawler-seeds = true; keyed API / SearXNG-url = false). The selector
    /// uses this only for telemetry + ordering, not correctness.
    fn spends_ip_reputation(&self) -> bool;
}

pub struct SearchHit {       // = today's SearxngResult, promoted to the shared type
    pub url: String,
    pub title: String,
    pub snippet: String,
}
```

**Selector resolution order (config-driven, §8 `search_source`):**
1. `keyed-api` (if `search_api_key_env` names a *set* env var) — zero IP cost; on quota/error/empty → fall through.
2. `searxng-url` (if `searxng_url` is set — back-compat for operators who *want* their own SearXNG) — zero IP cost to us; on empty/error → fall through.
3. `crawler-seeds` (ALWAYS available, the floor) — governed per ADR 0003.

**Hard rules for the implementer:**
- The selector MUST NOT make `keyed-api` or `searxng-url` *required*. If both are absent, `crawler-seeds` runs and the product works. (ASR-SC1, ASR-SC3.)
- Every `crawler-seeds` SERP fetch to Google/Bing/DDG/Brave is a **live-origin request** and MUST be admitted by the ADR 0003 `ReputationGovernor` (ADR 0003 C3) and respect the permanent denylist. Direct-data seeds (Wikipedia/arXiv/HN-JSON/StackExchange) are lower ban-risk but still governed. (ASR-SC2.)
- No secret literal anywhere: the key is read via `std::env::var(<search_api_key_env>)` at construction, never logged, never serialized. (api-security SECRETS; mirrors the existing `captcha_api_key_env` pattern, `config.rs:84-88`.)

### 4.1 Keyed-API client contract (Tavily default)

- **Request:** `POST https://api.tavily.com/search` with `{ "api_key": <from env>, "query": <q>, "max_results": <limit>, "search_depth": "basic" }` (basic = 1 credit [TAVILY-2]). Timeout ≤ 5s (matches the existing SearXNG client, `engine.rs:1618`).
- **Response → `Vec<SearchHit>`:** map `results[].url` / `.title` / `.content`.
- **Errors are non-fatal:** 401 (bad/expired key) / 429 (quota exhausted) / timeout → log a single WARN and **return `Ok(vec![])`** so the selector falls through to the crawler. A bad key must NEVER break search. (ASR-SC3.)
- **Provider is pluggable** via `search_api_provider` (default `"tavily"`); the contract is the trait, not the vendor. *Verify the exact request/response shape against `docs.tavily.com` at implementation time — Rule 1: do not ship the field names above unverified.*

---

## 5. Known issues to be AWARE of (recorded, NOT designed-for here)

Two benchmark findings the self-contained design must *not* worsen, folded in as motivation. **No fix is designed in this ADR** — they are flagged for the owning agent and a future ADR.

- **KI-1 — Cross-query index contamination (search-result isolation).** The engine accumulates *all* queries' crawls into ONE shared session index and serves later queries from it (index-first retrieval, `engine.rs:476-525`; daemon pre-caches into the same shared `url_cache`, `engine.rs:296-303`). Measured: a "react" query returned django/rust results from prior queries. **Relevance to self-contained:** making the keyless crawler path the default *increases* cross-query crawl volume into that shared index, so contamination pressure rises. The §3 selector and §8 config MUST NOT deepen the coupling (e.g. do not key the cache only by content-hash without a freshness/relevance gate). A proper fix (per-query index scoping or a relevance floor on index-first hits) is **owed to a separate ADR**.
- **KI-2 — Weak source-authority ranking.** Canonical homepages (tokio.rs, rust-lang.org) rank below blog tutorials. The crawler already *seeds* official domains (`entity_domain::detect_official_domains`, `engine.rs:1712`) but ranking does not preserve that authority signal. **Relevance:** the keyless default leans harder on these direct seeds, so the authority-ranking weakness is more visible without the API's pre-sorted relevance. Also **owed to a separate ADR** (ranking/authority), not solved here.

---

## 6. Model provisioning (dependency #2)

### 6.1 Options

| Option | Cold-start | Binary size | Offline-from-cold | Verdict |
|---|---|---|---|---|
| **Keep candle auto-fetch on first run** (current) | One-time network fetch to HF cache (`candle_embedder.rs:92-107`) | Small | **Fails** unless cache pre-warmed | Acceptable *with* a clear first-run message + named fallback. |
| **Vendor model files in the repo/release** | None | +~90 MB (MiniLM safetensors+tokenizer) per model, ×2 with cross-encoder | Works | Heavy; bloats every release for a one-time fetch. |
| **Auto-fetch + explicit offline fallback to HashEmbedder** (CHOSEN) | One-time fetch; on failure, named degradation | Small | **Starts**, degrades loudly | Honors ASR-SC4. |

### 6.2 Decision — keep auto-fetch, make the fallback **loud and named**, keep models out of the binary

The existing graceful fallback (`lib.rs:100-106`: candle init fail → HashEmbedder) is correct in shape but **too quiet** — it logs a WARN that is easy to miss, and the result is a silent collapse to keyword-only search. Required changes (contract, §8/§9):
- On candle init failure, emit a **distinct, structured, surface-able warning** (`model_fetch_failed`, with the model id and the reason) and set a server capability/health flag so the degraded state is observable, not buried.
- First-run download emits an **explicit one-line INFO** ("fetching embedding model ~90MB, one-time…") so a 30s first start is not mistaken for a hang.
- Models stay **out** of the binary (boring-by-default; a 200MB binary to save a one-time fetch is a bad trade). An **optional** `models_dir` config lets an air-gapped operator pre-stage files (`candle_embedder.rs:42-48` already loads a local dir if present — reuse it, do not rebuild it).

### 6.3 Fix the `force_cpu` footgun (must, per §1.2)

`force_cpu` currently means *two* things — "use CPU device" AND "skip the neural model and use HashEmbedder" (`lib.rs:85-87`). These must be **decoupled**:
- `force_cpu = true` MUST mean **run the neural model on CPU** (candle supports CPU inference — `candle_embedder.rs:33-36`, `Device::Cpu`), NOT drop to HashEmbedder.
- HashEmbedder is the fallback *only* when the neural model genuinely cannot load (offline-from-cold, ASR-SC4), and that path is the loud/named one from §6.2.
- This is an expand-and-contract behavior change (migration rule): keep parsing `force_cpu`, change its *meaning*; document it; the old "skip neural" behavior is removed because it silently destroys semantic quality, which is contrary to the product's purpose.

---

## 7. Config provisioning (dependency #3)

### 7.1 Decisions
- **Embedded defaults are the floor.** `Config::default()` already exists and is complete (`config.rs:344-432`), and `Config::load` already returns defaults when the file is absent (`config.rs:436-445`). So **no config file is required** today — the only gap is the *path*.
- **Make the config path overridable** (config-management EXTERNALIZE rule): resolution order — `--config <path>` CLI arg → `WEB_SEARCH_MCP_CONFIG` env var → `./config/default.toml` (current behavior) → embedded `Config::default()`. The hardcoded `Path::new("config/default.toml")` at `server.rs:36` is replaced by this resolver. This fixes the silent CWD-dependence in §1.1#3.
- **Validate at startup, fail fast** (config-management VALIDATE rule): if a config file is *named explicitly* (CLI/env) but missing or unparseable → **fail fast** with a clear message (do not silently fall back — an explicit path that doesn't load is an operator error). If no path is named, defaults are correct and silent.
- **Secure defaults** (already true and to be preserved): all ADR 0001/0002/0003 bypass surfaces default off (`config.rs` `#[serde(default)]` off-safe fields, guarded by the existing tests `config.rs:456-512`). The new `search_*` fields follow the same additive off-safe pattern.

---

## 8. Config surface (additive, off-safe, no secret literals)

All new fields `#[serde(default)]` so the shipped `config/default.toml` and `Config::default()` keep parsing (guarded by the existing additive-default tests, `config.rs:456-512` — extend them).

```rust
pub struct CrawlerConfig {
    // ... existing fields UNCHANGED. `searxng_url` is RETAINED (back-compat:
    //     operators who run their own SearXNG keep the zero-IP-cost path) ...

    /// Search source resolution. "auto" (DEFAULT) = keyed-api if a key is set,
    /// else searxng_url if set, else crawler-seeds. Other values pin one source:
    /// "crawler" | "tavily" | "searxng" — used for testing/forcing the floor.
    #[serde(default = "default_search_source")]
    pub search_source: String,                       // "auto"

    /// Optional keyed search-API provider. None/"" => no keyed source.
    /// "tavily" (recommended; true no-card free tier) | "serper" | ... .
    #[serde(default)]
    pub search_api_provider: Option<String>,         // None

    /// NAME of the env var holding the search-API key — NEVER the key itself
    /// (api-security SECRETS; mirrors captcha_api_key_env, config.rs:84). The key
    /// is read via std::env::var(<this name>) and never logged. None => keyless.
    #[serde(default)]
    pub search_api_key_env: Option<String>,          // None

    /// Bounded wait for the keyed-API request, seconds.
    #[serde(default = "default_search_api_timeout_secs")]
    pub search_api_timeout_secs: u64,                 // 5
}

pub struct EmbedderConfig {
    // ... existing fields UNCHANGED ...
    /// Optional directory of pre-staged model files for air-gapped/offline use.
    /// None => auto-fetch from HF on first run (default). When set and present,
    /// loaded instead of fetching (reuses candle_embedder.rs:42-48 local path).
    #[serde(default)]
    pub models_dir: Option<PathBuf>,                 // None
    // NOTE: `force_cpu` MEANING CHANGES (§6.3): now "neural on CPU", not
    //       "skip neural". Field name/type unchanged; behavior is migrated.
}
```

**Config-path resolver (new, in `mcp-server`, not a config field):** `--config` arg → `WEB_SEARCH_MCP_CONFIG` env → `config/default.toml` → embedded default (§7).

**Startup validation (extends the §7 fail-fast rules):**
- If `search_source == "tavily"` (pinned) but `search_api_key_env` is unset or names an unset env var → **fail fast** (explicit request for a source that can't run).
- If `search_source == "auto"` and no key/searxng_url → **INFO** ("running keyless: crawler-seed search; results via governed crawl") — not an error; this is the supported default.
- If an explicitly-named config path (CLI/env) fails to load → **fail fast** (§7).

---

## 9. Single-command run — what `./web-search-mcp` does with zero setup

The acceptance behavior (ASR-SC1) — **no docker, no key, no config file, empty data dir**:

1. **Config:** no `--config`, no env, no `config/default.toml` → embedded `Config::default()` (`config.rs:344`). `search_source="auto"`, no key, no SearXNG → **crawler-seeds** floor selected. INFO logged (§8).
2. **Models:** candle auto-fetches MiniLM + cross-encoder on first run (one-time ~network), INFO "fetching… one-time" (§6.2). If offline-from-cold → loud `model_fetch_failed` + HashEmbedder fallback, health flag set (§6.2). **No panic, no silent keyword-only.**
3. **Data dir:** `data/` created lazily; persistent index/vectors/cache (redb) initialize empty (`engine.rs:82-159` already handles absent dirs).
4. **Serve:** stdio MCP up (`main.rs:28-30`); daemon starts (`server.rs:45`).
5. **A query** (`quick_search`/`instant_search`): selector → crawler-seeds → `generate_search_seeds(q, None)` (Wikipedia/arXiv/HN-JSON/StackExchange/Reddit/official-domain + governed SERP) → governed crawl → extract → index → rank → results. **Non-empty for common queries; on a SERP ban, degrades via the non-SERP seeds, never dead-ends** (ASR-SC2).

With a Tavily key exported and `search_api_key_env="TAVILY_API_KEY"` set: step 5 prefers the zero-IP-cost API, falls through to the same crawler floor on quota/error.

---

## 10. Behavior at 10× / 100× load, migration, rollback

**10× / 100× (more queries):**
- **Keyed path:** Tavily free tier is **1,000 basic searches/month, no rollover** [TAVILY-2] — at 10×/100× query volume the free quota is exhausted and the selector **falls through to the governed crawler** (by design, §4). This is graceful: paid quota is a soft ceiling, not a wall.
- **Keyless path:** the bottleneck is the ADR 0003 single-IP reputation budget on SERP engines, not throughput. Per ADR 0003 §10, more queries to the same SERP domain queue behind the AIMD pacer / per-domain budget; when exhausted, SERP seeds are skipped and the **non-SERP direct seeds** (Wikipedia/arXiv/HN-JSON/StackExchange/official-domain) still serve. The system sheds load toward lower-coverage-but-safe sources rather than racing to a ban — the desired single-IP behavior.
- The single browser (R4) and shared session index (KI-1) remain the heavy bottlenecks ADR 0003 §10 already names; this ADR does not change them.

**Migration (expand-and-contract, reversible):**
- **Additive config** — every new field is `#[serde(default)]` off-safe; existing configs and `Config::default()` keep working (guarded by extending `config.rs:456-512` tests). The contract test: *legacy `config/default.toml` with no `search_*` keys parses and yields `search_source="auto"`, keyless.*
- **`searxng_url` is retained**, so an operator currently running their own SearXNG sees **no behavior change** until they remove it. This ADR does not delete their working path; it makes the *no-SearXNG* path the well-supported default.
- **`force_cpu` meaning change (§6.3)** is the one *behavioral* migration: it is documented, the field is unchanged, and the removed "skip-neural" behavior is intentionally dropped (it silently destroyed semantic quality). Reversible by reverting the embedder factory.

**Rollback:** all of §3/§4/§8 is gated by `search_source`. Setting `search_source="searxng"` + a `searxng_url` restores today's exact primary path. There is no irreversible element in this ADR (unlike ADR 0003's permanent denylist).

---

## 11. Residual risk / honest boundary (Rule 1)

1. **Keyless quality is lower than SearXNG-backed.** SearXNG returns pre-deduplicated, pre-ranked multi-engine results; the keyless crawler-seed path returns whatever the governed crawl reaches. Coverage on hostile, SERP-only queries is *worse* without a key — this is the real cost of "zero external setup," and it is the honest trade. The optional Tavily key closes most of this gap for free up to 1,000/mo.
2. **SERP scraping reputation risk is real even when governed.** DDG Lite is actively rate-limiting in 2026 [DDG-1]; the governor *paces and bounds* this but cannot make a hostile SERP engine reliable from one IP. The mitigation is that the product **does not depend** on any single SERP engine (the non-SERP seeds), so a SERP ban degrades rather than kills — but a query whose *only* good answers sit behind Google specifically will be weaker keyless.
3. **Tavily field-shape and free-tier terms are verified as of June 2026 [TAVILY-1, TAVILY-2] but vendor terms drift.** The implementer MUST re-verify the request/response schema and the free-tier allowance against `docs.tavily.com` before shipping (Rule 1 — do not ship §4.1's field names unverified). Provider is pluggable precisely so a terms change is a config swap, not a rewrite.
4. **KI-1 (index contamination) and KI-2 (authority ranking) are made more visible by the keyless default but are NOT fixed here (§5).** Each is owed a separate ADR. Shipping self-contained without addressing KI-1 means cross-query bleed persists; this is a known, accepted, recorded debt — not a silent one.
5. **Offline-from-cold yields keyword-only search.** If the very first run has no network, models can't fetch and HashEmbedder (no semantics) serves until a connected run populates the cache. This is loud and named (§6.2), but it *is* a degraded mode the operator must understand.

---

## 12. Citations (ground truth — verified June 2026)

- **[TAVILY-1]** Tavily free tier: 1,000 API credits/month, no credit card required — https://freetier.co/directory/products/tavily ; corroborated https://costbench.com/software/web-scraping/tavily/free-plan/
- **[TAVILY-2]** Tavily credit mechanics: basic search = 1 credit, advanced = 2, credits do not roll over; PAYG $0.008/credit — https://docs.tavily.com/documentation/api-credits ; https://www.firecrawl.dev/blog/tavily-pricing
- **[BRAVE-1]** Brave Search API killed its free tier (Feb 2026); now $5 metered credits, credit card required, overages billed with no spend cap — https://www.implicator.ai/brave-drops-free-search-api-tier-puts-all-developers-on-metered-billing/
- **[BRAVE-2]** Brave API pricing $5/1K requests, 50 req/sec, attribution-for-credit — https://costbench.com/software/ai-search-apis/brave-search-api/
- **[MOJEEK-1]** Mojeek Web Search API is trial/quote-based, no publicized free-tier price list — https://www.mojeek.com/services/search/web-search-api/
- **[DDG-1]** DuckDuckGo Lite actively rate-limited in 2025-2026 (`202 Ratelimit` in the wild) — https://github.com/open-webui/open-webui/issues/13935 ; https://github.com/open-webui/open-webui/discussions/6624
- **[DDG-2]** DDG scraping "needs residential proxies for reliable access," rate-limiting is the bot-detection mechanism — https://scraperly.com/scrape/duckduckgo
- **[RUST-1]** `websearch` crate multiplexes Google/Tavily/DuckDuckGo/Brave/Exa/SerpAPI/SearXNG (providers are the same scraped engines or keyed APIs we already have) — https://crates.io/crates/websearch
- **[RUST-2]** `h2m-search` crate: DuckDuckGo/Wikipedia/SearXNG/Brave/Tavily — same provider set — https://lib.rs/crates/h2m-search

**Source-code ground truth (this repo, read at authoring time):** `crates/common/src/config.rs` (config shape, `Config::default`, `Config::load`, additive-default tests); `crates/orchestrator/src/engine.rs` (`quick_search` :460, `instant_search` :741, `fetch_searxng_results` :1609, `generate_search_seeds` :1658, `SearxngResult` :1601, index-first :476); `crates/embedder/src/lib.rs:84-116` (`create_embedder` + `force_cpu` footgun); `crates/embedder/src/candle_embedder.rs:32-108` (HF auto-fetch + local-dir path); `crates/mcp-server/src/server.rs:36` (hardcoded config path); `crates/mcp-server/src/main.rs`; ADR 0003 (single-IP governor, contracts C2/C3).

**Flagged for verification before implementation (Rule 1):**
1. Tavily request/response schema + current free-tier terms — re-verify at `docs.tavily.com` (§4.1, §11.3).
2. That `force_cpu`-on-CPU candle inference performs acceptably (vs the old skip-to-hash) — measure before removing the old path (§6.3).
3. KI-1/KI-2 are recorded, not designed — each needs its own ADR before being called "addressed" (§5, §11.4).
```

---

## 13. Addendum A — critique-gate remediations (BINDING on the implementer)

Rev 1. The phase-4 critique gate (security-auditor + qa-engineer + testing-engineer, 2026-06-14) found that **two load-bearing reputation-safety claims of Option E are contradicted by the current code**, plus an unguarded SSRF surface and several testability seams. The original §1–§12 design intent stands; this addendum corrects the claims that were false-against-code and makes the fixes contractual. **Every item below is a precondition for implementation — backend-engineer MUST satisfy them, not treat them as optional.** Each carries the gate severity and the verifying test.

### A.1 — BLOCKER-1 (CRITICAL): the governor must be on the DEFAULT live-origin fetch path

**Finding (code-verified).** The `ReputationGovernor` (`admit` / `pace_delay` / permanent-denylist / breaker) is consulted **only inside `escalate()`**, which `Fetcher::fetch` calls **only when `enable_escalation == true`** (`crates/crawler/src/fetcher.rs:333-335`). That flag defaults `false` (`config.rs:107-108,:368`) and is **absent from the shipped `config/default.toml`**. On the default path, `fetch` falls through to the legacy retry loop → `fetch_once` → raw `client.get(url)` (`fetcher.rs:383-391`) with **no admit, no pacing, no denylist, no breaker**. ASR-SC2 ("every SERP hit goes through the governor", §4/§9.5) is therefore **asserted, not enforced** — and promoting the keyless crawler to the *primary* search surface (this ADR's purpose) sends high-frequency SERP traffic to the single static IP **ungoverned**, making permanent-ban risk *worse* than today, the exact opposite of the design goal.

**Decision (operator, 2026-06-14): Governor always-on for live-origin — Option (a).** Decouple reputation governance from `enable_escalation`. **Every live-origin fetch** (the legacy/default path in `fetch_once`, not just the escalation rungs) MUST pass `governor.is_permanently_denied` → `admit` → `pace_delay` → `record`, **regardless of `enable_escalation`**. The escalation ladder remains opt-in; *reputation safety does not*. Reputation safety must not be defeatable by leaving a flag off. (Rejected: flipping the self-contained default to `enable_escalation=true` — it couples the single-IP safety budget to a config value an operator can turn off and drags the whole ladder into the default profile.)

- **Scope note:** direct-data hosts (Wikipedia/arXiv/HN-JSON/StackExchange) are lower ban-risk but still live-origin → still governed (consistent with §4).
- **Contract test (must pass before merge):** on the **default** config (`enable_escalation=false`), a SERP fetch to `google.com`/`bing.com`/`duckduckgo.com`/`brave.com` is admitted through a spy `ReputationGovernor` (`admit` called); and a host on the permanent denylist is **not** fetched. **The test MUST assert behavior, not just instrumentation:** a governor returning `Admission::Deny` (or the denylist hit) actually **suppresses the socket** — calling `admit` and ignoring its verdict is a literal-satisfiable hole and must fail the test. RED check: revert the decoupling and this test fails.

### A.2 — BLOCKER-2 (CRITICAL): the SERP-ban degradation floor does not exist as coded

**Finding (code-verified).** The official/canonical-domain seeds are emitted as **Google SERP queries** — `https://www.google.com/search?q=site:{domain}+{q}` (`engine.rs:1715`), and `insert(0, …)` makes them *highest* priority. The ADR's degradation argument (§2.2 pt 2a, §9 step 5, §10 keyless) claims these official-domain seeds are a **SERP-independent floor** that survives a Google/Bing/DDG ban. They are not: if Google is hard-banned (the permanent single-IP failure ADR 0003 guards against), every official-domain seed dies with it. The "degrades, not kills" verdict that justifies choosing Option E over Option A rests on a floor that isn't there.

**Remediation (binding).** The implementer MUST add a genuinely SERP-independent seed for detected official domains — a **direct** `https://{domain}/` fetch (and/or the domain's native search endpoint where known), NOT routed through any SERP engine. The Google `site:` seed MAY remain as an *additional* (governed, lower-priority) seed, but MUST NOT be the *only* path to official-domain content.

- **Acceptance test (must pass):** with `google.com` + `bing.com` + `duckduckgo.com` + `brave.com` all on the permanent denylist, a query still returns non-empty results sourced only from direct-domain / Wikipedia / arXiv / HN-JSON / StackExchange / Reddit seeds, **and zero fetch attempts are made to the four denylisted SERP hosts.** This is the concrete form of ASR-SC2's failure path and the empirical proof of the §2.2 "degrades, not kills" claim.

### A.3 — HIGH: SSRF guard is a SearchSource-contract precondition

**Finding (code-verified).** Every search source emits attacker-influenceable `url` strings (SearXNG `results[].url`, SERP-HTML parser output, Tavily `results[].url`) fed directly into `fetch_urls_concurrent`/`crawl` → `client.get(url)`. The client has **no scheme allowlist and no internal-IP filter**, and follows up to **10 redirects** (`fetcher.rs:195`), so an `https://` seed can 302 to `http://169.254.169.254/…` (cloud metadata) or `http://127.0.0.1:<port>`. The `u.starts_with("http")` check (`engine.rs:1631`) does not stop loopback/link-local. This ADR widens the surface by making the multi-source keyless path the default and adding the Tavily source.

**Remediation (binding) — add to the §4 `SearchSource` contract and the fetch path:**
- Reject non-`http(s)` schemes.
- Resolve host and **deny RFC1918 / loopback (`127.0.0.0/8`, `::1`) / link-local (`169.254.0.0/16`, incl. `169.254.169.254`) / ULA (`fc00::/7`)**.
- **Re-apply the check on every redirect hop** — replace `Policy::limited(10)` with a custom redirect policy that re-validates each `Location`, not just the initial URL.
- **Anti-rebinding (re-gate note):** the resolved IP used for the deny check MUST be the IP the socket actually connects to (resolve-then-pin, or check at connect time). Resolving for the check and letting reqwest re-resolve for the connection is literally compliant but TOCTOU/DNS-rebinding-exploitable.
- **Test:** a `SearchSource` returning `http://169.254.169.254/latest/meta-data/`, an `https://` URL that 302s to a link-local address, **and a hostname whose A-record resolves to a link-local IP** are all refused before any socket to the internal address; a normal public URL passes. (Stub the redirect/DNS with a local server; no real metadata endpoint.)

### A.4 — MED: untrusted-text and keyed-client hardening (fold into the §4 / §4.1 contract)

- **Snippet/title sanitization (worsens KI-1).** `results[].content` → `snippet` (and `title`) MUST be length-bounded and control-char-stripped **before** entering the shared index/ranker. Untrusted text persists in the shared session index (KI-1, §5) and can surface under unrelated later queries. *Test:* an oversized / control-char snippet is truncated/cleaned before indexing.
- **Tavily client TLS/transport lock.** HTTPS-only, `danger_accept_invalid_certs(false)`, **no redirects**, host **hardcoded** to `https://api.tavily.com` — no config field may set the base URL (prevents key-exfil to an attacker endpoint). The keyed-client struct MUST NOT derive `Debug`/`Serialize` over the key field (or must redact it), and the request body MUST never be logged. *Tests:* `tavily_key_never_logged` (run with key `"SECRET123"`, assert no log line contains it); request-body shape test locks the verified schema.

### A.5 — MED: config-path and model-load trust

- **Config resolver (§7) — trust + observability.** `--config` / `WEB_SEARCH_MCP_CONFIG` are operator-trusted (in the TCB); **log the resolved absolute config path at startup** so a CWD-hijacked `config/default.toml` is observable, not silent. **Sanitize `Error::Config`** messages (currently surface raw path/parse error, `config.rs:441,443`) before they can reach any MCP client (error-handling: no internal paths to clients).
- **`models_dir` (§6.2) — trusted + verified.** Loading `model.safetensors` from a config-controlled dir has **no checksum/signature check** (`candle_embedder.rs:84-89`). Document `models_dir` as operator-trusted and **verify a pinned checksum/manifest for the expected model id before load — MANDATORY, not optional** (ml-inference WEIGHT INTEGRITY). Logging file sizes is **not** a substitute for the checksum (re-gate: the "at minimum log" escape clause is struck — size-logging leaves the integrity hole open for a config-controlled path). Additionally log the resolved absolute model path + file sizes alongside the verification.

### A.6 — Test seams the implementer MUST add up front (qa + testing-engineer)

These are structural — without them the §9 acceptance criteria are not honestly testable, and the testing-domain "no real external services" rule cannot be met.

1. **`create_embedder` health return channel (genuine structural blocker for §6.2 health flag).** Today it returns only `Box<dyn Embedder>` — no channel to signal "degraded". Change to `create_embedder(cfg) -> (Box<dyn Embedder>, EmbedderHealth)` (or set a shared `Arc<AtomicHealth>` the server reads for its capability flag). ASR-SC4's health flag is untestable until this exists.
2. **`model_fetch_failed` is a structured, named event** (not the current quiet `tracing::warn!`), carrying model id + sanitized reason (no paths/tokens). *Test:* force `CandleEmbedder::new` to Err via a deliberately-broken `models_dir` (offline seam — reuses the local-dir branch, no network) ⇒ no panic, `model_fetch_failed` emitted, health flag = degraded, HashEmbedder returned.
3. **`with_base(url)` on the Tavily client** (mirror `capsolver.rs:63`) so 401/429/timeout/2xx are tested against a local stub (`wiremock` dev-dep or a tokio stub); keep `build_body` + `parse_results` **pure**. Inject the timeout so the timeout test is sub-second. **`with_base` MUST be `#[cfg(test)]`-only (or test-crate-visible), NEVER reachable from config or any runtime path** — this resolves the tension with A.4's hardcoded-host rule (re-gate finding): a config-reachable base-URL setter would reintroduce the exact key-exfil surface A.4 closes.
4. **Pure `resolve_config_path(args, env) -> PathSource` and `validate_search_config(&Config) -> Result<()>`** (mirror `decide_gate`) so config-resolver (item 5) and startup-validation (item 8) are pure unit tests with no server boot.
5. **Mock-based ASR-SC1 / contractual-floor test.** Split ASR-SC1 into (a) a CI-safe contract test via a mock `SearchSource` — "no key + no `searxng_url` ⇒ selector resolves to `crawler-seeds`, `search()` returns `Ok` (never `Err`/panic), `generate_search_seeds` non-empty, server serves MCP" — and (b) an explicitly out-of-CI live smoke test. The "MUST work with no key and no container" floor (§2.2) needs this named test, not prose.
6. **Selector fall-through is tested with a `FakeSource`** (scripted `Hits`/`Empty`/`Err`) over an injected `Vec<Box<dyn SearchSource>>` — the selector MUST NOT construct real clients internally. Assert call **order** (`tavily`→`searxng`→`crawler-seeds`) and short-circuit via a shared call-log. Empty (`Ok(vec![])`) advances the chain identically to `Err` (the §4 miss-vs-failure contract).
7. **Additive-config contract test** extends `config.rs:456-512`: a legacy `[crawler]`/`[embedder]` TOML with **no** `search_*`/`models_dir` keys parses, yields `search_source="auto"`, providers `None`, `search_api_timeout_secs=5`, `models_dir=None`. Plus `shipped_default_toml_roundtrips` (the actual shipped file parses under the new struct — catches drift).

### A.7 — `force_cpu` migration: GREEN+RED + a named perf gate (§6.3)

The meaning change (skip-neural → neural-on-CPU) does not break parsing (field name/type unchanged), so the risk is a **silent behavioral/latency** change, not a parse failure. Required:
- **GREEN:** `force_cpu=true` + a loadable (fixture) model ⇒ `create_embedder` returns the **neural** embedder on `Device::Cpu` (assert `model_name()` is the MiniLM name / `dimensions()==384`), not `HashEmbedder`.
- **RED:** reverting §6.3 (restoring `if force_cpu {return Hash}`) makes the GREEN test fail.
- **Perf gate (ADR §6.3/§12 flagged-item #2):** measure CPU neural-inference latency against a **stated threshold**, committed as a benchmark result file, **before** the old fast path is removed. If CPU inference is unacceptably slow, retain an explicit `embedder="hash"` escape hatch so the capability isn't silently deleted. "Measure before removing" with no threshold and no file is insufficient (integrity rule: the perf claim must map to a committed result).

### A.8 — KI-1 non-worsening tripwire (accepted debt, bounded)

KI-1 (cross-query shared-index contamination, §5) stays deferred to its own ADR, but the keyless default **increases** contamination pressure, so shipping with *no guard* is not acceptable. Add a **regression tripwire** (not a fix): run query A ("django") then query B ("react"); assert B's top-N contains no result whose only provenance is A's crawl (or assert a contamination ratio ≤ a baseline captured now). This enforces §5's existing prohibition ("do not key the cache only by content-hash without a freshness/relevance gate"), which is currently unenforced, and gives the future KI-1 ADR a failing-test target.

### A.9 — Gate status: **GO** (re-gated 2026-06-14)

Rev 1 folds A.1–A.8 into the design as binding constraints. The security-auditor re-review (2026-06-14) returned **GO for backend-engineer**, with BLOCKER-1 (A.1), BLOCKER-2 (A.2), and SSRF (A.3) confirmed CLOSED / CLOSED-IF-TESTED, and three contract tightenings folded back in (now incorporated above):
- **A.1** — the contract test must assert a governor `Deny` verdict **suppresses the socket**, not merely that `admit` was called (literal-satisfiable hole closed).
- **A.5** — `models_dir` checksum verification is **mandatory**; the "at minimum log file sizes" escape clause is struck.
- **A.4/A.6.3** — `with_base` is **`#[cfg(test)]`-only**, never config-reachable, resolving the key-exfil tension with the hardcoded-host rule.
- **A.3 note** — host-deny must pin the resolved IP to the connected socket (anti-rebinding), and the test adds a DNS-resolves-to-link-local case.

Items the original gate found sound — env-var-name secret indirection (ASR-SC3), additive `#[serde(default)]` migration, rollback via `search_source` — are unchanged. **Next: backend-engineer implements §3/§4/§6/§7/§8 under the A.1–A.8 contract.**
