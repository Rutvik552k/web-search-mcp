# ADR 0001 — Anti-Bot Bypass Escalation Ladder & Block-Detection Classifier

- **Status:** Proposed
- **Date:** 2026-06-06
- **Owner:** solution-architect
- **Covers:** TASKS.md 1.1 (escalation ladder) and 1.3 (block-detection classifier spec)
- **Supersedes:** none (first ADR)
- **Implemented by (later phases):** 2.1–2.7, 3.1–3.4

> Scope guard (GOAL.md §2): bypass features are **opt-in via config, default-off where
> legally sensitive**, and only authorized against owned/permitted targets. This ADR sets
> contracts only; it writes no feature code. CAPTCHA-solving and robots bypass remain gated
> behind the Phase 5 legal gate.

---

## 1. Context and forces in tension

The crawler (`crates/crawler/src/fetcher.rs`) today does: cache lookup → retry loop with
exponential backoff → single plain `reqwest` GET (one static UA, one default header set,
no proxy) → SPA detection → optional headless browser fallback → cache insert. There is **one
fetch strategy**. Against bot-protected sites it has no path to recover: any 4xx/5xx becomes
`Error::Blocked` and the body is discarded *before* anyone can inspect why (fetcher.rs:207–212).

GOAL.md requires ≥90% coverage including a Cloudflare/DataDome/rate-limited tier (G1, G2). To
hit that we must (a) **recognize** *why* a fetch failed and (b) **escalate** through progressively
more powerful — and more expensive — strategies, stopping as soon as we get real content or hit a
budget. The forces in tension:

| Force | Pressure |
|---|---|
| **Coverage** | Must defeat Cloudflare/DataDome JS challenges + CAPTCHAs to reach G2. |
| **Cost** | Proxies cost $/GB; CAPTCHA solves cost $/solve (~$1–3 per 1000, provider-dependent — verify in 3.3). Headless = CPU/RAM/seconds. A naive "always use the strongest tool" blows the cost budget (user cost-tracking rules: warn >$5/op, halt >$50/session). |
| **Latency** | Plain HTTP ~100–800ms; headless render 2–15s; CAPTCHA solve 10–60s. The `fetch_urls_concurrent` fast-path caps per-request at 4s (crawler.rs:262) — heavy rungs cannot run there. |
| **Correctness of cache** | The moka cache (fetcher.rs:83–86) must **never** cache a challenge/block page, or it poisons future reads for 10 min. |
| **Not breaking SPA path** | The existing `is_spa` → browser fallback (fetcher.rs:135) must keep working and must not double-render when the ladder also invokes the browser. |
| **Resilience** | Every rung needs a timeout, bounded attempts, and a per-domain circuit breaker so a hostile domain cannot consume the whole crawl (user resilience rules). |

**Architecturally significant requirements (ASRs):**

- **ASR-1 Coverage**: blocked-tier success must improve materially vs baseline (G2; numeric target set by operator post-baseline — *unstated, operator owes per GOAL.md §5*).
- **ASR-2 Cost cap**: hard per-session and per-URL spend ceilings; CAPTCHA/proxy spend is opt-in and metered.
- **ASR-3 Latency budget**: each rung has a latency ceiling; the concurrent fast-path only runs cheap rungs.
- **ASR-4 Cache integrity**: only verified real content is cached.
- **ASR-5 Determinism/observability**: every escalation decision is logged with the classifier verdict and the signals that produced it (correlation by URL).

> **ASRs I could not resolve from the repo — operator input needed before Phase 2 implementation:**
> 1. G2 numeric target (GOAL.md §5). 2. Proxy provider + whether residential/datacenter (changes cost math). 3. CAPTCHA provider choice + per-session $ cap (Task 3.3). These do **not** block the ADR but block 2.x/3.x.

---

## 2. Options considered

### Option A — Flat "try everything" fallback chain
On any non-2xx or empty body, run every heavier strategy in fixed order until something works.

- **Pros:** trivial to implement; no classifier needed.
- **Cons:** wastes the most expensive resources on the cheapest failures (e.g., burns a CAPTCHA solve on a transient 500); no cost control; cannot satisfy ASR-2. Cannot distinguish "JS challenge" (needs browser) from "your IP is banned" (needs proxy) — so it often escalates down the wrong branch.

### Option B — Classifier-driven escalation ladder (CHOSEN)
A **block-detection classifier** inspects each response and emits a verdict. The verdict
**selects the next rung**, not just "go one step heavier." Each rung has a trigger condition,
a bounded attempt count, a latency ceiling, and (where it spends money) a cost gate.

- **Pros:** spends the minimum to win; satisfies ASR-1/2/3; the classifier verdict is the single
  observable that explains every escalation (ASR-5); maps cleanly onto the existing retry loop.
- **Cons:** the classifier is a heuristic and **will** misclassify some pages — anti-bot vendors
  change markers (CLAUDE.md Rule 1 flagged below). Mitigated by: verdicts are advisory (a wrong
  "RealContent" still gets a content-quality check before caching), signals are
  config-tunable, and we log raw signals so misses are debuggable.

### Option C — Outsource to a scraping API (ZenRows/ScrapingBee/Scrapfly)
- **Pros:** highest coverage, zero maintenance of fingerprints/challenges.
- **Cons:** violates GOAL.md §1 ("self-contained, API-free"). Rejected on the project's core constraint, not on merit.

**Decision: Option B.** It is the only option that meets the API-free constraint *and* the cost
cap. The ladder slots into the existing retry loop as a strategy selector; the classifier becomes
the function that today is the implicit `status.is_client_error()` check, but evidence-based.

---

## 3. The escalation ladder (Task 1.1)

Rungs are ordered cheapest→most-expensive. The classifier verdict (§4) **routes** to a rung; we do
not always climb one at a time. "Real content" at any rung short-circuits and returns.

```
                         ┌─────────────────────────────────────────────┐
   fetch(url) ──► cache  │  moka cache  ── HIT (real content only) ─────┼──► return
                  miss   └─────────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────────── escalation controller ─────────────────────────────┐
        │  per-URL budget: time_budget_ms, cost_budget_usd, attempts_remaining             │
        │  per-domain circuit breaker (open → skip heavy rungs, fail fast)                 │
        └──────────────────────────────────────────────────────────────────────────────────┘
                    │
   RUNG 0  Plain HTTP GET (current client) ──► classify ──► RealContent ──► (SPA? →RUNG3) ──► cache+return
                    │  verdict≠RealContent
                    ▼
   RUNG 1  UA / header rotation + fresh cookie jar ──► classify ──► RealContent ──► return
                    │  still blocked
                    ▼
   RUNG 2  Proxy rotation (sticky per domain) ──► classify ──► RealContent ──► return
                    │  still blocked
                    ▼
   RUNG 3  Headless browser (chromiumoxide) ──► classify ──► RealContent / cf_clearance ──► return
                    │  still challenge / detected as bot
                    ▼
   RUNG 4  Stealth headless + proxy + rotated fingerprint ──► classify ──► RealContent ──► return
                    │  CAPTCHA still present
                    ▼
   RUNG 5  CAPTCHA solve (opt-in, cost-gated) ──► inject token ──► classify ──► RealContent ──► return
                    │  exhausted / budget hit
                    ▼
                GIVE UP ──► typed Error (Blocked{class} / RateLimited / ChallengeUnsolved)  [NOT cached as content]
```

| Rung | Trigger (classifier verdict) | What changes | Latency budget | Cost | Max attempts | Give-up → |
|---|---|---|---|---|---|---|
| **0 HTTP** | always first | none (current client) | `request_timeout_secs` | free | 1 | rung 1 |
| **1 UA/header rotate** | `SoftBlock`, generic `Http403` (no vendor marker), `RateLimited` (after honoring `retry-after`) | swap to a full realistic browser header set from the pool (Task 2.1), new cookie jar; honor `Retry-After` with backoff+jitter | ≤ 2× base | free | `max_header_profiles` (default 2) | rung 2 |
| **2 Proxy rotate** | still blocked after rung 1; or `Cloudflare403`/`DataDome403` that is IP-reputation shaped (403 with vendor marker but **no** JS challenge); repeated `RateLimited` | route via next healthy proxy, sticky per domain; eject proxy on failure (Task 2.3) | ≤ 3× base | $ proxy bytes | `min(max_proxy_attempts, healthy_proxies)` (default 3) | rung 3 |
| **3 Headless** | `JsChallenge`; **or** `RealContent` + `is_spa` (the existing SPA path) | render with JS via existing `BrowserPool`; persist `cf_clearance`/`dd` cookies into the client cookie jar for reuse | ≤ `browser_timeout_secs` (default 15s) | CPU/RAM | 1 | rung 4 |
| **4 Stealth headless** | `JsChallenge` *and* rung 3 still returned a challenge (bot-detected) | stealth patches (Task 3.1: drop `navigator.webdriver`, real viewport/UA/locale) + proxy + rotated TLS/JA3 fingerprint (Task 2.2) | ≤ `browser_timeout_secs` | CPU/RAM + $ proxy | 1 | rung 5 |
| **5 CAPTCHA solve** | `Captcha{recaptcha\|hcaptcha\|turnstile}` detected (and `enable_captcha_solver` + cost budget OK) | extract sitekey, call solver provider, inject token, submit (Task 3.3) | ≤ `captcha_timeout_secs` (default 60s) | **$$ per solve** | 1 | GIVE UP |

**Give-up conditions (any one):** rung 5 exhausted; `attempts_remaining == 0`; `time_budget_ms`
exceeded; `cost_budget_usd` would be exceeded by the next rung; per-domain circuit breaker open.
On give-up return the **most informative typed error** the classifier produced — never a generic
one, and never cache it as content.

### How it slots into `fetcher.rs` without breaking cache or SPA path

The current `fetch()` (fetcher.rs:112–167) becomes a thin wrapper around a new escalation
controller. Three concrete changes — all additive, none breaks existing behavior when bypass
toggles are off:

1. **`fetch_once` must stop discarding bodies on error.** Today it returns `Err(Blocked)` at
   fetcher.rs:207 *before* reading the body, so the classifier would be blind. Change `fetch_once`
   to always capture `(status, headers, body)` into a raw struct and hand it to the classifier;
   the classifier — not the status code — decides success vs block. Plain-2xx pages are
   unaffected (they classify as `RealContent`). *This is the one behavior change to existing code;
   it is backward-compatible because a 2xx with content still yields `RealContent` → same result.*

2. **Cache insert is gated on verdict.** Move the `cache.insert` (fetcher.rs:148) so it runs
   **only** when the final verdict is `RealContent` (or browser-rendered real content). Challenge,
   CAPTCHA, soft-block, and rate-limited responses are **never** inserted (ASR-4). Optional bounded
   *negative cache* (short TTL, separate small moka map) records "gave up on this URL" to avoid
   re-hammering within one crawl — flag as optional, default off.

3. **SPA path unified, not duplicated.** The existing `is_spa` → `try_browser_fallback`
   (fetcher.rs:135) becomes **rung 3 reached via the `RealContent + is_spa` trigger**, using the
   same `BrowserPool`. The `JsChallenge` verdict also routes to rung 3, but with challenge-solve
   semantics (wait for `cf-mitigated` to clear / cookie to appear) rather than "did we get more
   text." One browser entry point, two triggers — no double render: a page is either a challenge
   *or* a SPA, the classifier disambiguates (a CF "Just a moment" page is `JsChallenge`, not SPA,
   even though `detect_spa` might also fire on its short body — **classifier wins**).

The retry loop's existing `RateLimited`/`Blocked` handling (fetcher.rs:151–159) is replaced by the
ladder; the 429/`Retry-After` honor logic moves into rung 1. **Add jitter** to the backoff
(fetcher.rs:127 is currently `base * 2^n` with no jitter) — required by resilience rules to avoid
thundering-herd retries across the 8–16 concurrent workers.

---

## 4. Block-detection classifier spec (Task 1.3)

A pure function over the raw response. **Signals are evaluated in priority order; first match
wins** (CAPTCHA before generic challenge before generic 403, because a CAPTCHA page is *also* a
403 with a CF marker — order matters).

```
classify(status: u16, headers: &HeaderMap, final_url: &str, body: &str) -> BlockClass
```

```rust
// Proposed type (crates/common or crawler). Carries the evidence for observability (ASR-5).
pub enum BlockClass {
    RealContent,                              // serve + cache
    Cloudflare403,                            // IP/edge block, no JS challenge → proxy
    JsChallenge { vendor: ChallengeVendor },  // Cloudflare IUAM / DataDome interstitial → browser
    Captcha { kind: CaptchaKind },            // recaptcha | hcaptcha | turnstile → solver
    SoftBlock,                                // 200 but empty/placeholder/"enable JS" → UA rotate→browser
    RateLimited { retry_after_secs: u64 },    // 429 / vendor rate signal
}
pub enum ChallengeVendor { Cloudflare, DataDome, Akamai /*verify in 1.2*/, Unknown }
pub enum CaptchaKind { Recaptcha, Hcaptcha, Turnstile }
```

### Per-class signals (priority order)

#### (1) CAPTCHA page — `Captcha{kind}`  *(check first)*
| Kind | Body / header signals |
|---|---|
| **reCAPTCHA** | script `src` contains `www.google.com/recaptcha/api.js` or `www.gstatic.com/recaptcha`; element `class="g-recaptcha"` with `data-sitekey`. |
| **hCaptcha** | script `src` contains `js.hcaptcha.com/1/api.js` or `hcaptcha.com/captcha`; element `class="h-captcha"` with `data-sitekey`. |
| **Turnstile** | script `src` contains `challenges.cloudflare.com/turnstile/v0/api.js`; element `class="cf-turnstile"` with `data-sitekey`. |

Cited: Cloudflare Turnstile docs, hCaptcha developer docs, Google reCAPTCHA docs, CapMonster CAPTCHA-identification guide. Status is usually 403 but **must not** be required (CAPTCHA can appear inline on a 200 form page).

#### (2) JS challenge interstitial — `JsChallenge{vendor}`
| Vendor | Signals |
|---|---|
| **Cloudflare** | Header `cf-mitigated: challenge` (**authoritative & most reliable** — only valid value is `challenge`); `content-type: text/html` always; status **403** (managed challenge, most common today) or **503** (legacy "I'm Under Attack"/IUAM — *historically 503; verify per target, CF has shifted to 403*); body markers: `_cf_chl_opt` / `window._cf_chl_opt`, path `/cdn-cgi/challenge-platform/`, `challenges.cloudflare.com`, visible text `Just a moment...` or `Checking your browser`, legacy `cf-browser-verification`; `server: cloudflare`; body small (<~50KB). |
| **DataDome** | Header `x-datadome` present (and/or `x-datadome-cid`); `datadome` token in a `Set-Cookie`; body contains a `dd` JS object / `geo.captcha-delivery.com` reference; status **403** (most common; can be 400–500, rarely 401); often asks to enable JS / solve CAPTCHA (if CAPTCHA markers also present, class is `Captcha` by priority). |

Cited: Cloudflare challenge-detection doc (`detect-response/`), Cloudflare "Just a moment"/`_cf_chl_opt` research, DataDome bypass guides (ZenRows/Scrapfly), DataDome docs (`rule-responses`). **CLAUDE.md Rule 1 flags to verify in Task 1.2/2.x:** Akamai and PerimeterX/HUMAN markers are **not yet grounded here** — treat `ChallengeVendor::Akamai` as a stub until verified; vendors rotate markers, so signals must live in config (§5), not be hard-coded constants.

#### (3) Generic edge 403 (no JS challenge, no CAPTCHA) — `Cloudflare403`
Status 403 **and** a CF/DataDome fingerprint (`server: cloudflare` or `cf-ray` header present, or
`x-datadome`) **but none** of the JS-challenge/CAPTCHA body markers above. Interpretation: IP/edge
reputation block, not a solvable challenge → **proxy rotation (rung 2)**, not browser. (A plain
non-CF 403 with no vendor marker → treat as `SoftBlock`/`Http403` → rung 1 UA rotation first.)

#### (4) Rate limited — `RateLimited{retry_after_secs}`
Status **429**; read `Retry-After` (seconds or HTTP-date) → `retry_after_secs` (default 30 if
absent, as today at fetcher.rs:194). Also classify here if a vendor returns 503 + a documented
rate header. Honor the wait (capped), then rung 1.

#### (5) Soft block / empty — `SoftBlock`
Status **200** but content is not real: `content_length` below `soft_block_min_bytes` (heuristic,
default ~500 visible chars via the existing `estimate_visible_text_len`); or body is a known
placeholder ("Please enable JavaScript", "Access denied", "Pardon our interruption" — DataDome
soft wall); or `<title>` matches a block pattern (`Access Denied`, `Attention Required`,
`Just a moment`). Distinct from a *legit* SPA: a SPA has `RealContent`+`is_spa` (empty mount div
+ heavy JS bundle) and routes to browser as a render, whereas SoftBlock has a block phrase. When
ambiguous, prefer `SoftBlock` only if a block phrase matches; otherwise `RealContent`+`is_spa`.

#### (6) Real content — `RealContent`  *(default / fall-through)*
2xx, no block marker, visible text ≥ threshold. If `detect_spa(body)` is true, set `is_spa` so the
controller routes to rung 3 as a **render** (not a challenge solve). Only this class is cached.

> **Confidence & tuning:** all numeric thresholds (`soft_block_min_bytes`, challenge body-size cap,
> default `retry_after`) and all marker string-lists are config-driven (§5) so the operator can
> tune against the real `benchmark/urls.jsonl` set without a recompile, and so vendor marker churn
> is a config change, not a code change.

---

## 5. Config additions to `CrawlerConfig` (Task 2.7 contract)

Additive only; every field `#[serde(default)]` with a safe/off default so existing `config.toml`
files keep parsing and current behavior is unchanged when bypass is off (config-management rule:
secure defaults, validate at startup). **No secrets in config values** — API keys are read from
**env var references**, never literals (api-security / secrets rules).

```rust
pub struct CrawlerConfig {
    // ... existing fields unchanged ...

    /// Master switch for the escalation ladder. Off => current single-strategy behavior.
    #[serde(default)] pub enable_escalation: bool,
    /// Highest rung the ladder may reach (0=HTTP .. 5=CAPTCHA). Default 3 (no proxy/CAPTCHA $).
    #[serde(default = "default_max_rung")] pub max_escalation_rung: u8,

    // -- Rung 1: UA / header rotation (Task 2.1) --
    #[serde(default)] pub user_agent_pool: Vec<String>,          // empty => use single `user_agent`
    #[serde(default = "default_header_profiles")] pub max_header_profiles: u8,

    // -- Rung 2: proxy pool (Task 2.3). Proxy URLs MAY embed creds => treat whole list as a secret;
    //    prefer `proxy_pool_env` naming an env var holding the list. --
    #[serde(default)] pub proxy_pool_env: Option<String>,        // env var name, NOT the proxies
    #[serde(default)] pub proxies: Vec<String>,                  // dev-only fallback; warn if set in prod
    #[serde(default = "default_max_proxy_attempts")] pub max_proxy_attempts: u8,
    #[serde(default = "default_proxy_sticky")] pub proxy_sticky_per_domain: bool,
    #[serde(default = "default_proxy_health_fail")] pub proxy_eject_after_failures: u8,

    // -- Rungs 3/4: browser & stealth (Tasks 3.1/2.2) --
    #[serde(default = "default_browser_timeout")] pub browser_timeout_secs: u64,
    #[serde(default)] pub enable_stealth: bool,
    #[serde(default)] pub fingerprint_profile: Option<String>,   // e.g. "chrome-124"; verify lib in 1.2

    // -- Rung 5: CAPTCHA solver (Task 3.3) — legally gated, default OFF --
    #[serde(default)] pub enable_captcha_solver: bool,
    #[serde(default)] pub captcha_provider: Option<String>,      // e.g. "2captcha" (verify API/pricing 3.3)
    #[serde(default)] pub captcha_api_key_env: Option<String>,   // env var NAME holding the key
    #[serde(default = "default_captcha_timeout")] pub captcha_timeout_secs: u64,

    // -- Cost / budget guards (ASR-2; user cost rules) --
    #[serde(default = "default_url_cost_budget")] pub per_url_cost_budget_usd: f64,   // e.g. 0.01
    #[serde(default = "default_session_cost_cap")] pub session_cost_cap_usd: f64,     // hard halt, e.g. 5.0
    #[serde(default = "default_url_time_budget")] pub per_url_time_budget_ms: u64,    // e.g. 20000

    // -- Circuit breaker per domain (resilience rule) --
    #[serde(default = "default_cb_threshold")] pub domain_breaker_fail_threshold: u32,
    #[serde(default = "default_cb_cooldown")] pub domain_breaker_cooldown_secs: u64,

    // -- Classifier tuning (so vendor-marker churn is config, not code) --
    #[serde(default = "default_soft_block_min_bytes")] pub soft_block_min_bytes: usize,
    #[serde(default)] pub block_marker_overrides: Option<PathBuf>, // optional TOML of extra markers

    // -- Optional negative cache (don't re-hammer give-ups within a crawl) --
    #[serde(default)] pub enable_negative_cache: bool,
    #[serde(default = "default_neg_cache_ttl")] pub negative_cache_ttl_secs: u64,
}
```

**Startup validation (config-management rule):** if `enable_captcha_solver` then
`captcha_api_key_env` must name a *set* env var or fail fast; if `max_escalation_rung >= 2` and no
proxy source configured, warn and clamp to rung 1; reject `session_cost_cap_usd <= 0`. Per the
GOAL legal gate, `enable_captcha_solver` and `respect_robots_txt=false` must be explicit opt-ins.

---

## 6. Failure modes and how each is bounded

| Failure mode | Bound / mitigation |
|---|---|
| **Classifier false-negative** (thinks a block is real content) | Content-quality gate before caching (visible-text ≥ threshold + no block phrase); only `RealContent` is cached. A poisoned cache entry self-heals at the 600s TTL. Raw signals logged for tuning. |
| **Classifier false-positive** (thinks real content is a block) | Wastes a rung but cannot loop: per-URL `attempts_remaining` + `time_budget_ms` cap. Worst case = one wasted cheap rung, then give-up returns content if rung 0 had it. |
| **Infinite/expensive escalation** | Hard `per_url_time_budget_ms`, `attempts_remaining`, and `max_escalation_rung`. Each rung has its own timeout (rung-3/4 `browser_timeout_secs`, rung-5 `captcha_timeout_secs`). |
| **Cost blowout** | Per-URL `per_url_cost_budget_usd` checked *before* entering a paid rung (2/5); global `session_cost_cap_usd` halts all paid rungs when reached (mirrors user rule: halt >$50; warn path logs running total). CAPTCHA + proxy spend metered per solve/byte. |
| **Hostile domain consuming the crawl** | Per-domain circuit breaker: after `domain_breaker_fail_threshold` blocks, breaker opens → that domain fast-fails (skips heavy rungs) for `domain_breaker_cooldown_secs`. Bulkhead: heavy rungs (browser) bounded by the single-browser pool so they can't exhaust workers. |
| **Browser hang / crash** (chromiumoxide) | Existing `tokio::time::timeout` around render (browser.rs:137) + page close in all paths; rung capped at 1 attempt. On `None`, controller falls through to next rung or give-up. |
| **Proxy dead / leaking** | Health-check + auto-eject after `proxy_eject_after_failures` (Task 2.3); sticky sessions reset on eject; if no healthy proxy, rung 2 is skipped, not retried forever. |
| **Cache poisoning by challenge page** | §3 change: cache insert gated on `RealContent` only. |
| **`retry-after` abuse** (huge value stalls crawl) | Cap honored wait (e.g. min(retry_after, ceiling)); if it exceeds `per_url_time_budget_ms`, requeue via frontier (as crawler.rs:157 already does) instead of blocking the worker. |
| **Thundering-herd retries** | Add jitter to backoff (fetcher.rs:127). |
| **Secret leakage** | API keys/proxy creds only via env-var references; never logged (log provider name + masked, never the key); never written to the moka cache or `FetchResult`. |
| **SSRF via operator/LLM-supplied URL** (deferred to 5.1) | Out of scope for this ADR; flagged so rung 2 (proxy) design in 2.3 keeps an allow/deny hook. |

### Behavior at 10× and 100× load
- **10×** (more URLs): the controller is per-URL and stateless except shared pools; scales with the
  existing 8–16 worker `FuturesUnordered`. Proxy pool and the single browser become the bottleneck —
  heavy rungs (3–5) must be **rate-limited separately** from cheap rungs so a flood of challenges
  doesn't serialize on one browser. The `fetch_urls_concurrent` fast-path stays cheap-rungs-only
  (its 4s cap forbids rungs 3–5).
- **100×**: single-node browser/CAPTCHA throughput is the hard ceiling (GOAL.md non-goal: not
  internet-scale/distributed). Bound by `session_cost_cap_usd` and the circuit breakers; the design
  degrades gracefully — heavy rungs shed load (return give-up) rather than queue unboundedly.

### Rollback / migration path
- **Reversible by config:** `enable_escalation=false` restores exact current behavior (single
  strategy). All new fields default off/safe, so dropping in the new code with an old `config.toml`
  is a no-op (config-only expand; no schema migration, no data store touched).
- **Reversible by code:** the one behavior change (fetcher.rs `fetch_once` capturing body on error)
  is gated — when `enable_escalation=false`, the controller collapses to "classify→ if not
  RealContent and status is error, return the same typed `Error` as today." Pure expand-and-contract:
  add the classifier path alongside the old path, flip the flag to migrate, remove the old path only
  after the benchmark (GATE 2) confirms parity-or-better.

---

## 7. Interfaces other agents must implement

These are the contracts Phase 2/3 engineers must honor (set here, not implemented here).

**C1 — Classifier (Task 2.4, pure, unit-testable):**
```rust
fn classify(status: u16, headers: &reqwest::header::HeaderMap,
            final_url: &str, body: &str, cfg: &ClassifierConfig) -> BlockClass;
```
- Deterministic; no I/O; priority order = CAPTCHA → JsChallenge → Cloudflare403 → RateLimited → SoftBlock → RealContent.
- Must return the **evidence** (which signal fired) for logging — either via a richer return or a `tracing` event keyed by URL.
- Marker lists come from `cfg`, seeded with the §4 grounded values; not hard-coded literals.

**C2 — Escalation controller (Task 2.5):** wraps `fetch()`. Inputs: `url`, `&CrawlerConfig`,
per-URL budget. Owns: rung selection per §3 table, budget/breaker enforcement per §6, cache-insert
gating per §3. Output: `Result<FetchResult>` (unchanged public signature of `Fetcher::fetch`) so
`crawler.rs` callers (`fetch_and_process`, `fetch_one`, `fetch_urls_concurrent`) need **no change**.

**C3 — `fetch_once` change:** capture `(status, headers, body)` for all statuses; hand to C1. Do
not early-return `Err` before classification.

**C4 — Error semantics (extend `crates/common/src/error.rs`):** add
`Error::ChallengeUnsolved { url, vendor }` and enrich give-up returns to carry the `BlockClass`
(e.g. `Blocked { url, status }` gains context or a sibling variant). Existing `RateLimited` /
`Blocked` retained for back-compat with `crawler.rs:155`.

**C5 — Observability:** every escalation emits a `tracing` event `{url, rung, verdict,
signals, latency_ms, cost_usd_delta}`; running session cost is a counter checked against
`session_cost_cap_usd`.

**C6 — Cache contract:** `cache.insert` iff final `BlockClass::RealContent`. Optional negative
cache is a *separate* bounded map, never merged with the content cache.

---

## Citations (CLAUDE.md Rule 1)

- Cloudflare, "Detect a Challenge Page response" — `cf-mitigated: challenge` (only valid value), `content-type: text/html`: https://developers.cloudflare.com/cloudflare-challenges/challenge-types/challenge-pages/detect-response/
- Cloudflare challenge body markers `_cf_chl_opt`, `/cdn-cgi/challenge-platform/`, "Just a moment", challenge pages <~50KB (community + research): https://developers.cloudflare.com/cloudflare-challenges/ , https://github.com/scaredos/cfresearch , https://blog.noah.ovh/cloudflare-js-challenge-1/
- Cloudflare 403 + `cf-mitigated: challenge` real-world: https://community.cloudflare.com/t/ads-txt-403-forbidden-error-cf-mitigated-challenge/507924
- DataDome block signals (`x-datadome` header, `datadome` Set-Cookie, `dd` script object, 403 most common): https://www.zenrows.com/blog/datadome-bypass , https://scrapfly.io/blog/posts/how-to-bypass-datadome-anti-scraping , https://docs.datadome.co/docs/rule-responses
- CAPTCHA markers (reCAPTCHA `g-recaptcha`/`api.js`, hCaptcha `h-captcha`/`js.hcaptcha.com`, Turnstile `cf-turnstile`/`challenges.cloudflare.com/turnstile/v0/api.js`): https://developers.cloudflare.com/turnstile/migration/recaptcha/ , https://docs.hcaptcha.com/ , https://developers.google.com/recaptcha/docs/display , https://capmonster.cloud/en/blog/identify-captcha-types/

**Flagged assumptions to verify before implementation (Rule 1):**
1. Legacy Cloudflare IUAM "Just a moment" status is **historically 503** but CF has largely moved to **403** — confirm per-target in the benchmark set, do not hard-depend on 503.
2. **Akamai / PerimeterX(HUMAN)** markers are not grounded in this ADR — `ChallengeVendor::Akamai` is a stub; verify markers in Task 1.2 before adding.
3. CAPTCHA solver **API shape and per-solve pricing** (Task 3.3) and **TLS/JA3 fingerprint Rust lib** (`rquest`/reqwest-impersonate, Task 1.2) are explicitly deferred to the research-gated tasks; this ADR assumes nothing about their APIs.
4. G2 numeric target, proxy provider/type, and CAPTCHA provider+cost cap are **operator inputs** (GOAL.md §5) owed before Phase 2/3.
```
