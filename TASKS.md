# TASKS — Path to the Goal

> Ordered plan to meet [GOAL.md](GOAL.md). Owners use the CLAUDE.md department
> routing map. Each phase ends in a **GATE** that must pass before the next starts.
> Subagents run in background (Rule 3); main agent drives the chain (Rule 2).
>
> Legend: ⛔ blocker · 🔬 research-gate (verify before code, Rule 1) · 🔒 security/legal

---

## Phase 0 — Orchestration & baseline (do first)

| # | Task | Owner |
|---|------|-------|
| 0.1 | Produce ordered specialist chain + gates for this whole effort | `team-orchestrator` |
| 0.2 | ⛔ Operator delivers `benchmark/urls.jsonl` (tiered, `blocked` flag) for G1/G2 coverage. `queries.jsonl` optional — G3 uses MCPBench | operator |
| 0.3 | Build **coverage** harness (G1/G2 only) | `testing-engineer` | ✅ DONE — `crates/benchmark` (rmcp client + mock-server self-test, 19 tests green). Schema locked in `benchmark/README.md`. Also computes nDCG@10/p@5 as a secondary to MCPBench |
| 0.5 | 🔬 Integrate **MCPBench** (https://github.com/modelscope/MCPBench) for G3. ✅ WIRED + handshake-VERIFIED 2026-06-18 (`benchmark/mcpbench/`: config + README + `verify_handshake.py`). MCPBench plugs via SSE (`supergateway` wraps our stdio server); 15 tools discovered, `instant_search` call returned isError=False. Tools map 1:1, no adapter. Dataset fields `unique_id`/`Prompt`/`Answer`; judge = hardcoded `openai/deepseek-v3` (operator key, eval-time only). **Full 600-QA run = operator-blocked:** (a) paid OpenAI-compatible judge key serving deepseek-v3, (b) confirm/pin a `supergateway` version that re-arms per-request (current build dies after 1 SSE session), (c) run SearXNG/keyed source (keyless floor hit a CAPTCHA), (d) run MCPBench under WSL/Linux (Win path mangling). G3 `score` → `firecrawl-compare --accuracy-ours/--accuracy-firecrawl` | `testing-engineer` |
| 0.4 | Run **baseline** on current `main`: coverage harness + MCPBench; record numbers in `benchmark/RESULTS.md` | `testing-engineer` |
| 0.6 | 🔬 **G4 Firecrawl comparison harness** — Firecrawl adapter in `crates/benchmark` runs `/v2/scrape`+`/v2/crawl` over the **same** `urls.jsonl` + MCPBench QA pairs; records both runs in `benchmark/RESULTS.firecrawl.md`. ✅ HARNESS DONE 2026-06-18 (`9c0df30`): `firecrawl.rs`/`compare.rs`/`lib.rs`/`bin/firecrawl-compare`, 37 tests green, contract-mocked, env-key only, never a runtime dep. Verified cloud v2 contract. | `testing-engineer` |
| 0.6a | **Self-host Firecrawl** (operator chose 2026-06-18: self-hosted AGPL Docker, not paid cloud → compute-vs-compute cost). Stand up `benchmark/firecrawl-selfhost/` compose; verify self-hosted API parity vs our v2 adapter; wire `FIRECRAWL_BASE_URL` if adapter lacks base-url env. ▶ devops-engineer running (parity verdict pending) → testing-engineer for any adapter delta | `devops-engineer` |
| 0.6b | ⛔ Run the **real G4 baseline**: operator runs self-hosted Firecrawl + `firecrawl-compare` over `urls.jsonl`, fills `RESULTS.firecrawl.md`. Needs 0.6a parity confirmed | operator |

**GATE 0:** harness reproducible; baseline coverage/accuracy numbers committed. No optimization before a baseline exists (Rule 1).

---

## Phase 1 — Architecture of the bypass layer ✅ DONE (awaiting GATE 1 approval)

| # | Task | Owner | Status |
|---|------|-------|--------|
| 1.1 | ADR escalation ladder + integration into fetcher.rs | `solution-architect` | ✅ `docs/adr/0001-bypass-escalation-ladder.md` — 6-rung **classifier-routed** ladder (verdict picks rung, not climb-one-by-one) |
| 1.2 | 🔬 Verify Rust bypass libs | `research-solution-architect` | ✅ `docs/research/0001-bypass-libs.md`. **Correction: `rquest` is DEPRECATED → use `wreq` 5.3.0 + wreq-util** (BoringSSL TLS+H2). Stealth = patch browser.rs (Runtime.enable leak). Proxy = client pool (no per-request). |
| 1.3 | Block-detection classifier spec | `solution-architect` | ✅ in ADR — `BlockClass` {RealContent, Cloudflare403, JsChallenge, Captcha, SoftBlock, RateLimited}, priority-ordered, config-driven markers |

**GATE 1 (operator approval needed):** ADR is status=Proposed. Approve before Phase 2.
Lib choices verified. ⚠️ **Scope reality (needs operator sign-off):** OSS-only achieves passive-fingerprint + most JS-challenge. DataDome/behavioral sites need a **commercial unblock API (tier-3)** — see decision 1.4.

| 1.4 | DECISION: scope | operator | ✅ RESOLVED 2026-06-06 — pursue scrape-any-data; tier-3 commercial unblocker optional fallback; legal guardrails now hard defaults (GOAL.md §2) |

---

## Phase 1.5 — Data-layer acquisition (Rung -1) — HIGHEST ROI, mostly unblocked

ADR `docs/adr/0002-data-layer-acquisition.md`. Acquire data from less-protected surfaces
before fighting blockers. In-band tasks need NO wreq/toolchain/proxy — **runnable now**.

| # | Task | Owner | Status |
|---|------|-------|--------|
| 1.5.1 | In-band hydration | `backend-engineer` | ✅ `extractor/src/hydration.rs`, 51 tests green, off-by-default. `enable_data_layer` flag |
| 1.5.2 | Structured-data promotion | `backend-engineer` | ✅ `consensus.rs` promote/salvage, additive (fills gaps, replaces body only if richer) |
| 1.5.3 | **Wire** `extract_page_with_config` into orchestrator call sites (~8) + enable flag for measured run; later add `DataLayerEngine` for out-of-band. Needs `fetch_once` "capture body on error" (shared w/ 2.4) | `backend-engineer` | ▶ NEXT (unblocked) |
| 1.5.4 | **Syndication** — RSS/Atom/JSON Feed discovery + extend existing `sitemap.rs` (news sitemaps) | `backend-engineer` | ✅ DONE 2026-06-13 — `feeds.rs` (RSS2.0/Atom1.0/JSONFeed1.1) + `sitemap.rs::parse_news_sitemap`; wired as ADR 0003 **R2** rung in `fetcher.rs`. Sitemap network-probe sub-surface still TODO |
| 1.5.5 | **Out-of-band**: internal JSON/GraphQL + mobile/legacy API probe (high potential, high build); AMP/print; Wayback CDX archive fallback | `backend-engineer` | after impersonation rung |

**GATE 1.5:** in-band (1.5.1/2) shows coverage lift on benchmark; trait merged additively, existing pipeline green.

---

---

## Phase 2 — Anti-blocking implementation (crawler/fetcher)

Each task ships with unit tests + a benchmark re-run delta. Implements ADR interfaces C1–C6.

| # | Task | Owner |
|---|------|-------|
| 2.0 | 🔬 Build spike | `backend-engineer` | ⚠️ **NO-GO (toolchain)**. Versions verified: **wreq=5.3.0 + wreq-util=2.2.6**, boring2 4.15.15, no openssl-sys clash. Spike at `spike-wreq/`. See 2.0a |
| 2.0a | ⛔ **OPERATOR: install Windows BoringSSL toolchain** — VS2022 Build Tools (C++ workload), cmake, NASM, LLVM/libclang; `rustup default stable-x86_64-pc-windows-msvc`. Then re-run spike → fingerprint smoke-test must read as Chrome | operator |
| 2.1 | **wreq emulation** — swap `reqwest`→`wreq` in `fetcher.rs`; **delete static UA + `default_headers()`** (lines 304–311), use `.emulation(Chrome*)`. Rotate profile per the pool | `backend-engineer` |
| 2.2 | **TLS/JA3+HTTP2** = handled by wreq emulation (folded into 2.1). Verify handshake on a live Cloudflare site | `backend-engineer` |
| 2.3 | **Proxy pool** — pool of pre-built wreq clients keyed (proxy, profile); sticky-by-host via `dashmap`; rotate on block; health-eject; creds from env only | `backend-engineer` |
| 2.4 | **Block-detection classifier** (ADR §2) — drives ladder. Stop discarding body on 4xx/5xx (`fetcher.rs:207`); gate `cache.insert` (line 148) to `RealContent` only | `backend-engineer` |
| 2.5 | **Escalation ladder** (ADR §1) — classifier-routed rung selection in fetch retry loop; add jitter to backoff (line 127) | `backend-engineer` |
| 2.6 | 🔒 **robots.txt bypass toggle** — opt-in override (operator chose full bypass) | `backend-engineer` |
| 2.7 | Config surface (~25 additive `#[serde(default)]` fields per ADR §3; off-by-default = current behavior) | `backend-engineer` |

**GATE 2:** unit tests green; coverage harness re-run shows non-protected + JS-SPA tiers improved vs baseline. GATE 2.0 (build spike) must pass before any of 2.1–2.7.

---

## Phase 3 — Stealth browser + CAPTCHA (hardest blockers)

| # | Task | Owner |
|---|------|-------|
| 3.1 | **Stealth headless** — patch `browser.rs`: defeat **CDP `Runtime.enable` leak** via `Page.createIsolatedWorld` + `addScriptToEvaluateOnNewDocument` (mask webdriver, inject window.chrome); `--headless=new`, real UA/locale, drop imagesEnabled tell, replace fixed 1500ms sleep w/ network-idle wait; verify vs detection test page | `backend-engineer` |
| 3.2 | **Cloudflare/DataDome challenge flow** — solve JS challenge in browser, persist `cf_clearance`/cookies, reuse for the domain | `backend-engineer` |
| 3.3 | 🔬🔒 **CAPTCHA solving** — integrate a solver provider (verify API + pricing) for reCAPTCHA/hCaptcha/Turnstile; trigger on CAPTCHA detection only; cost-cap per session | `backend-engineer` |
| 3.4 | Make browser path no longer feature-gated-off-by-default for the bypass build profile | `backend-engineer` |

**GATE 3:** blocked-subset (G2) success measured and reported; operator sets G2 target.

---

## Phase 4 — Accuracy tuning

| # | Task | Owner |
|---|------|-------|
| 4.1 | Run accuracy eval (G3) via **MCPBench**; identify failing query classes | `data-scientist` |
| 4.2 | Tune ranking pipeline (RRF weights, CE/ColBERT thresholds, primary-source boost) against eval — avoid overfitting to MCPBench set; hold out a split | `data-scientist` + `backend-engineer` |
| 4.3 | Re-run full harness; confirm coverage ≥ 90% (G1) + accuracy target (G3) | `testing-engineer` |
| 4.4 | **G4 head-to-head** — run G4 harness (0.6) ours vs Firecrawl on identical inputs; confirm **≥ +5 pts** on coverage/blocked/accuracy, **≤** Firecrawl P99 latency, **strictly lower** $/1k pages; numbers in `benchmark/RESULTS.firecrawl.md` | `testing-engineer` + `data-scientist` |

**GATE 4:** G1 ≥ 0.90, G2/G3 targets met, **and G4 margins met** (numbers in `benchmark/RESULTS.md` + `benchmark/RESULTS.firecrawl.md`).

---

## Phase 5 — Hardening, legal, release

| # | Task | Owner |
|---|------|-------|
| 5.1 | 🔒 Security review: proxy creds + CAPTCHA key handling (no secrets in code/logs/cache), SSRF on user-supplied URLs, prompt-injection on scraped content reaching LLM | `security-engineer` |
| 5.2 | 🔒 Compliance/legal review: robots bypass + CAPTCHA solving authorization, ToS risk, document allowed-use boundary | `compliance-officer` |
| 5.3 | Resilience: timeouts/circuit-breakers per escalation step, proxy-pool backpressure, cost guards | `site-reliability-engineer` |
| 5.4 | Release gate: changelog, version bump, README update, go/no-go | `release-manager` |

**GATE 5:** security + legal sign-off; all gates 0–4 green.

---

## Critical path

`0.2 (operator data) → 0.3/0.4 baseline → 1.x ADR+research → 2.x anti-block → 3.x stealth/CAPTCHA → 4.x accuracy → 5.x release`

**Top blockers:** (1) operator benchmark sets [0.2], (2) fingerprint-lib verification [1.2], (3) proxy creds + CAPTCHA provider [3.3]. Nothing downstream is verifiable without [0.2].

---

## Session resume — 2026-06-11

Single-IP escalation ladder built on the existing crawler. Operator constraint: **one static IP, no proxy/IP rotation** → ADR 0001 Rung 2 (proxy) deleted; strategy is *never-get-blocked*. New docs: ADR `docs/adr/0003-single-ip-escalation-ladder.md`, design `docs/design/0004-stealth-and-captcha-layers.md` + `0005-hybrid-escalation-controller.md`, compliance `docs/compliance/0001-r5-captcha-gate.md`.

**DONE this session (all `cargo test` green — 171 pass / 0 fail / 2 operator-run `#[ignore]` spikes; nothing committed):**
- ✅ Block-detection classifier (`crates/crawler/src/classifier.rs`) — ADR 0001 §4 markers, priority-ordered.
- ✅ R4 v1 stealth (`browser.rs` `fetch_with_stealth`) — drops automation/headless tells, preload, UA coherence, network-idle wait, cf_clearance capture.
- ✅ **R4 v2** — vendored+patched chromiumoxide (`vendor/chromiumoxide/`, `[patch.crates-io]`); removed auto `Runtime.enable` from `frame.rs` (the CF/DataDome leak). Guard test + `PATCH_NOTES.md`.
- ✅ R5 CAPTCHA (`crates/crawler/src/captcha/`) — `CaptchaSolver` trait + CapSolver/2Captcha + extract/inject/cost; live wiring with compliance conditions enforced as code (default-OFF + dual opt-in `enable_captcha_solver`+`captcha_run_opt_in`, audit, record-on-give-up, cost pre-check).
- ✅ Hybrid controller (`fetcher.rs`) — `enable_escalation` gate (off ⇒ unchanged), `fetch_raw_once`, classifier-routed rungs, R0 hydration salvage, cf_clearance reuse (`enable_clearance_replay` default-off).
- ✅ Governor (`crates/crawler/src/governor.rs`) — file-backed permanent denylist + AIMD pacer + soft breaker.
- ✅ Linker fix pinned in `.cargo/config.toml` (gnu `-lktmw32`).

**OPEN / NEXT SESSION:**
- 🚫 **wreq R1 impersonation — OPERATOR-BLOCKED** (TASK 2.0a): no cmake/nasm/clang/cl, rustup=gnu. Install VS2022 BuildTools(C++)+cmake+NASM+LLVM, `rustup default stable-x86_64-pc-windows-msvc`, re-run `spike-wreq/`.
- ⏳ **Live efficacy unproven** — operator runs from their IP: `cargo test -p web-search-crawler --features browser -- --ignored stealth_smoke` and `$env:CF_SPIKE_URL=...; ... --ignored clearance_replay_binding` (JA3/H2 cookie-binding question).
- ▶ **Next rung to build: R2/R3 alternative-surface** — RSS/Atom/JSON-Feed + Internet Archive CDX (zero-ban-risk coverage lever, highest-ROI remaining; ADR 0003).
- Minor: 6 cosmetic warnings; Sec-CH-UA metadata + isolated-world evaluate deferred.

---

## Session resume — 2026-06-13

**R2/R3 alternative-surface rung BUILT** (was the top OPEN item from 2026-06-11). All
working-tree, nothing committed. `cargo test -p web-search-common -p web-search-crawler`
= **208 + 2 pass / 0 fail / 1 ignored**; `cargo check --workspace` clean.

- **R3 archive** `crates/crawler/src/archive.rs` — Internet Archive CDX client. Pure
  `parse_cdx_newest` (column-by-name, `statuscode:200`, optional max-age, newest) +
  `snapshot_raw_url` (`…/web/<ts>id_/<url>` raw modifier) + async `fetch_archived`
  (non-fatal `Option`). Zero-ban-risk (web.archive.org only). 15 tests. Ground truth:
  IA wayback-cdx README + archive.org developer docs.
- **R2 feeds** `crates/crawler/src/feeds.rs` — RSS 2.0 / Atom 1.0 / JSON Feed 1.1
  discovery + parse. `discover_feeds` (scraper on HTML), `parse_feed`/`sniff_and_parse`
  (regex over raw XML — scraper/html5ever corrupts CDATA + void `<link>`),
  `find_item_for_url`, async `probe_feeds` (autodiscovery + well-known paths,
  governor-gated). 14 tests. Ground truth: rssboard, RFC 4287, jsonfeed.org.
- **News sitemap** `sitemap.rs::parse_news_sitemap` + `NewsEntry` + `is_news_sitemap`
  (regex for `<news:*>` namespace). 3 tests.
- **Wiring** `fetcher.rs` `run_escalation` SoftBlock|Cloudflare403 branch (replaced the
  R2/R3 stubs): reputation-first **R0→R3(zero-ban, no governor)→R2(live, governor-gated)
  →give_up**. New `RungIo::archive_fetch`+`alt_surface_probe` (mocked in `MockIo`); 5
  controller tests (recover, preference ordering, off-safe give-up, R3-miss→R2 fallthrough).
- **Config** ADR 0003 §6.2 fields added to `CrawlerConfig` (`enable_archive_fallback`,
  `archive_cdx_endpoint`/`_timeout_ms`/`_max_snapshot_age_days`/`_user_agent`,
  `enable_alt_surface`, `src_feed`, `src_sitemap`, `max_alt_probes`) — all off-safe,
  legacy-parse test extended. `Rung` enum gained R2/R3.

**NEXT:** (a) wire sitemap as an R2 sub-surface (config `src_sitemap` exists, only feeds
probed today); (b) live-network integration test for `probe_feeds`/`fetch_archived`
(contract-mocked client per no-real-external-services rule); (c) internal JSON/GraphQL/
mobile-API probe (ADR 0002 1.5.5); (d) still operator-blocked: wreq R1 impersonation
toolchain + live efficacy spikes (TASK 2.0a). Nothing committed — ask before commit.

---
_Generated 2026-06-06. Update status inline as phases complete._
