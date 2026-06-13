# Design 0004 — Stealth Headless (R4) & Commercial CAPTCHA Layer (R5)

- **Status:** Proposed (implementation spec — module layout, interfaces, exact CDP/API calls, config)
- **Date:** 2026-06-11
- **Owner:** backend-engineer
- **Implements:** ADR 0003 §3.1 rung **R4** (stealth headless) and rung **R5** (commercial unblock /
  CAPTCHA API). This doc does **not** restate the ladder, the AIMD governor, or the
  permanent-denylist breaker — it consumes them. Read ADR 0003 and ADR 0001 first.
- **Builds on (does not contradict):**
  - `docs/adr/0003-single-ip-escalation-ladder.md` — R4/R5 triggers, governor gate, C1–C7 contracts,
    permanent hard-stop. R4 fires on `JsChallenge{vendor}` (or `RealContent + is_spa`); R5 fires on
    `Captcha{kind}` (and `JsChallenge` R4 could not solve) **only** when `enable_unblock_api` /
    `enable_captcha_solver` + cost budget OK.
  - `docs/adr/0001-bypass-escalation-ladder.md` — `BlockClass` classifier (§4): `JsChallenge`→R4,
    `Captcha`→R5; the typed-error give-up contract; cache-insert-on-`RealContent`-only (C6).
  - `crates/crawler/src/browser.rs` — the existing `BrowserPool` we extend (fixed 1500ms sleep,
    `--blink-settings=imagesEnabled=false`; both are removed here).
  - `crates/crawler/src/fetcher.rs` — `try_browser_fallback` / `fetch_via_browser` invocation points.
  - `crates/common/src/config.rs` — `CrawlerConfig`; all new fields `#[serde(default)]`, off-safe.

> **Scope guard (GOAL.md §2, ADR 0001/0003):** everything here is **opt-in / default-OFF**. Nothing
> in this doc changes behavior unless the operator explicitly turns it on. This is a spec, not feature
> code — small illustrative Rust sketches show the CDP sequence and the solver trait; they are
> *shape*, not final implementation.

> **HONESTY NOTE (CLAUDE.md Rule 1).** The two deep-research passes that produced this ground truth
> (2026-06-11) had their **adversarial-verify step rate-limited; it ABSTAINED (votes 0–0).** The
> mechanisms below are **primary-source-grounded** (cited in §6) but the **real-world success rates
> are UNVERIFIED.** Every success-rate claim is marked *unproven*. Re-verify any load-bearing API
> detail against the live doc before coding (the chromiumoxide-specific facts in §1.2 were verified
> against docs.rs 0.7.0 on 2026-06-11 — see §6).

---

## 0. What was verified vs assumed before writing this

Verified live on 2026-06-11 (sources in §6):

- **chromiumoxide 0.7.0 auto-enables the Runtime domain.** `Page::enable_runtime()` is documented
  *"Activated by default"*; `Page::disable_runtime()` exists; `Page::execute()` sends a raw CDP
  command; `Page::evaluate_on_new_document()` = `Page.addScriptToEvaluateOnNewDocument`. There is
  **no** high-level `createIsolatedWorld` helper → it must be sent via `execute()` with
  `CreateIsolatedWorldParams`. **This makes the `Runtime.enable` leak our default-path problem and is
  the single biggest build risk** (§1.2, §7 risk R-1).
- The generated CDP types exist in `chromiumoxide_cdp`: `CreateIsolatedWorldParams`,
  `AddScriptToEvaluateOnNewDocumentParams`, `AddBindingParams`, each with `::builder()` and a
  `...Returns`.
- rebrowser-patches fix modes (the mechanism we port): **`addBinding` (default)**, `alwaysIsolated`,
  `enableDisable`, `0` (off).

Assumed / **unproven** (spikes in §7): R4 efficacy vs live CF/DataDome from one IP; cookie
fingerprint-binding for wreq replay (ADR 0003 §5); CapSolver/2Captcha real solve rates.

---

# PART 1 — R4: Stealth Headless (Runtime.enable-safe)

## 1.1 Module layout

Extend the existing `crates/crawler/src/browser.rs` rather than add a parallel browser stack
(ADR 0003 §7 Option R4-a: hand-patch — recommended; no fragile dependency). New submodule tree under
the crawler crate:

```
crates/crawler/src/
├── browser.rs                 # EXISTING BrowserPool — extended, see §4
└── stealth/
    ├── mod.rs                 # StealthConfig, RuntimeFixMode enum, public entry: prepare_page()
    ├── runtime_fix.rs         # the Runtime.enable-leak defeat (addBinding / isolated / enableDisable)
    ├── fingerprint.rs         # pre-nav script + Emulation overrides (UA, Sec-CH-UA, viewport, locale)
    ├── wait.rs                # network-idle / challenge-cleared wait (replaces the 1500ms sleep)
    └── clearance.rs           # cf_clearance / datadome cookie capture + per-domain persistence
```

`stealth` is compiled only under the existing `#[cfg(feature = "browser")]`. When the `browser`
feature is off, or `enable_stealth == false`, `prepare_page()` is a no-op and `BrowserPool` behaves
exactly as today.

**Public surface (mod.rs):**

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub enum RuntimeFixMode {
    AddBinding,     // DEFAULT: new binding in main world, call it, save its execution context id
    AlwaysIsolated, // Page.createIsolatedWorld per frame; evades MutationObserver; no main-world, no workers
    EnableDisable,  // Runtime.enable then immediately Runtime.disable; small leak window
    Off,            // do nothing (current vanilla behavior)
}

pub struct StealthConfig {
    pub mode: RuntimeFixMode,
    pub user_agent: String,           // MUST NOT contain "HeadlessChrome"
    pub accept_language: String,      // coherent with Sec-CH-UA / locale
    pub viewport: (u32, u32),
    pub challenge_wait: WaitConfig,
}

/// Called once per page, BEFORE navigation. Installs the runtime fix and the pre-nav
/// fingerprint script, applies Emulation overrides. Returns the execution-context handle
/// the caller uses for any evaluate().
#[cfg(feature = "browser")]
pub async fn prepare_page(page: &chromiumoxide::Page, cfg: &StealthConfig)
    -> Result<EvalContext, StealthError>;

/// Where evaluate() must run so it does not re-trigger the leak.
pub enum EvalContext {
    MainViaBinding { execution_context_id: i64 }, // AddBinding mode
    Isolated { execution_context_id: i64 },       // AlwaysIsolated mode
    Default,                                       // EnableDisable / Off — uses page default
}
```

## 1.2 Which stealth mode we adopt (and why)

**Default: `AddBinding`.** Rationale (ground truth #4, ADR 0003 §7):

- Keeps **main-world** access (we sometimes need real page globals for challenge state / hydration
  scraping under R0), and works with **iframes/workers** — Turnstile renders in an iframe, so a mode
  that loses iframe access is a liability.
- Avoids ever sending `Runtime.enable`, defeating the canonical `Runtime.enable → consoleAPICalled`
  leak that CF Turnstile + DataDome detect (ground truth #1).

**Per-challenge override: `AlwaysIsolated` for known challenge pages.** When the classifier verdict
is `JsChallenge{Cloudflare|DataDome}`, prefer `AlwaysIsolated`: the isolated world evades in-page
`MutationObserver`/`Error.stack` instrumentation that watches the main world. We lose main-world
variable access, but on a pure challenge page we only need to (a) wait for clearance and (b) read
cookies via CDP `Network.getCookies` — neither needs main-world JS. This is config-selectable, not
hardcoded (`stealth_challenge_mode` in §3).

`EnableDisable` is the **documented fallback** (small leak window) if `AddBinding` proves unstable on
this chromiumoxide version; `Off` reproduces today's behavior for A/B.

> **Why NOT JS-injection shims (chromiumoxide_stealth, puppeteer-extra-stealth style):** ABSENCE beats
> shimming. JS overwrites of `navigator.webdriver` are detectable via `.toString()` `[native code]`
> checks, `Error.stack` diffs, and `Proxy` exception-type probes (CreepJS). We do **not** JS-overwrite
> `navigator.webdriver`; we rely on it being absent under `--headless=new` + the runtime fix
> (ground truth #4). REJECT `chromiumoxide_stealth` for the leak.

## 1.3 The exact CDP call sequence (AddBinding default)

The ordering is load-bearing. The fix must be installed **before** any navigation and **before** any
`evaluate()`, and we must prevent the library from auto-firing `Runtime.enable`.

**Step 0 — do not let chromiumoxide auto-enable Runtime.** chromiumoxide 0.7.0 activates the Runtime
domain by default (§0). We never call `Page::evaluate*` on the default context for a stealth page
(that path assumes the auto-enabled runtime); instead we route all evaluation through the binding/
isolated context obtained below, and we treat the auto-enable as a build risk to confirm in the crate
source (§7 R-1). If the auto-enable cannot be suppressed at page creation, the documented workaround
is `EnableDisable` mode (accept the tiny window) or a patched chromiumoxide (vendored).

**Step 1 — pre-nav fingerprint script** via `Page.addScriptToEvaluateOnNewDocument` (runs in every
new frame *before* the page's own scripts; this is the only reliable pre-nav hook and is not a
JS-overwrite shim of `webdriver` — it sets coherent values that absence alone does not cover, e.g.
`window.chrome`, WebGL vendor/renderer, `navigator.permissions` query coherence).

**Step 2 — UA + client-hints coherence** via `Network.setUserAgentOverride` (carries
`userAgentMetadata` so `navigator.userAgent`, `Sec-CH-UA`, platform, and brands are *one coherent
set* — ASR-7). Drop the `HeadlessChrome` token. Also set `Emulation.setLocaleOverride` /
`Emulation.setTimezoneOverride` if configured.

**Step 3 — capture a main-world execution context WITHOUT Runtime.enable** (the `addBinding` trick):
`Runtime.addBinding{ name }` registers a binding callable from the page; combined with a tiny pre-nav
script that calls the binding once per new document, the binding callback carries the
`executionContextId` of the main world. We save that id and target all subsequent `Runtime.evaluate`
calls at it via `contextId`. (For `AlwaysIsolated`, replace Step 3 with `Page.createIsolatedWorld`
after navigation and save its returned `executionContextId`.)

**Step 4 — navigate**, then run the **challenge-cleared wait** (§1.5), then capture clearance cookies
(§1.6), then read `page.content()`.

Illustrative sketch (shape, not final — uses `page.execute(...)` for raw CDP; types from
`chromiumoxide_cdp`):

```rust
use chromiumoxide::cdp::browser_protocol::page::{
    AddScriptToEvaluateOnNewDocumentParams, CreateIsolatedWorldParams,
};
use chromiumoxide::cdp::browser_protocol::network::SetUserAgentOverrideParams;
use chromiumoxide::cdp::js_protocol::runtime::AddBindingParams;

const BINDING: &str = "__cc_ctx";

// Step 1: pre-nav fingerprint + binding-caller, installed before navigation.
let preload = format!(
    r#"
    // call the binding once per new document so the CDP side learns this frame's contextId
    try {{ window.{b} && window.{b}('ctx'); }} catch (e) {{}}
    // coherent values absence does not cover (NOT a webdriver shim):
    if (!window.chrome) {{ window.chrome = {{ runtime: {{}} }}; }}
    "#, b = BINDING);
page.execute(
    AddScriptToEvaluateOnNewDocumentParams::builder()
        .source(preload).build().unwrap()
).await?;

// Step 2: coherent UA + Sec-CH-UA. userAgentMetadata keeps brands/platform consistent.
page.execute(
    SetUserAgentOverrideParams::builder()
        .user_agent(cfg.user_agent.clone())      // no "HeadlessChrome"
        .accept_language(cfg.accept_language.clone())
        // .user_agent_metadata(...)             // brands/platform/mobile — fill from profile
        .build().unwrap()
).await?;

// Step 3: AddBinding — main-world contextId WITHOUT Runtime.enable.
page.execute(AddBindingParams::builder().name(BINDING).build().unwrap()).await?;
// The binding's Runtime.bindingCalled event (consumed on the CDP handler) yields
// executionContextId; save it into EvalContext::MainViaBinding.

// (AlwaysIsolated variant, AFTER navigation, instead of relying on main world)
// let iso = page.execute(
//     CreateIsolatedWorldParams::builder()
//         .frame_id(frame_id).world_name("cc_iso").grant_universal_access(true)
//         .build().unwrap()
// ).await?;
// let ctx_id = iso.execution_context_id;   // save into EvalContext::Isolated

// Step 4: navigate, wait (see §1.5), capture cookies (see §1.6), read content.
```

> **Rule 1 flag:** the precise field names (`accept_language` vs `accept-language` casing, presence of
> `user_agent_metadata` builder setter) must be confirmed against the chromiumoxide 0.7.0 generated
> source before coding; the *commands* are correct, the *builder ergonomics* are version-specific.

## 1.4 Detection-vector checklist (neutralize once the leak is fixed)

Each item states the vector, the chosen technique, and the priority. Order matters: the
`Runtime.enable` leak (#1) dominates — fixing it first is what gets us past CF/DataDome load-time
CAPTCHA; the rest harden against secondary fingerprinting.

| # | Vector | Technique | Priority |
|---|---|---|---|
| 1 | **`Runtime.enable → consoleAPICalled` CDP leak** | `addBinding` (default) / isolated world — §1.3 | **TOP** |
| 2 | `navigator.webdriver` | **absence** under `--headless=new` + runtime fix; do **not** JS-shim | high |
| 3 | `HeadlessChrome` UA token | drop it; coherent UA via `Network.setUserAgentOverride` | high |
| 4 | `Sec-CH-UA` / UA mismatch | `userAgentMetadata` set together with UA (one coherent unit) | high |
| 5 | `window.chrome` runtime object | set in pre-nav script (#1 hook) | medium |
| 6 | WebGL vendor/renderer | spoof to a real GPU pair in pre-nav script | medium |
| 7 | canvas / audio FP | leave default headless values (over-spoofing is itself a signal) — *do not over-invest* | low |
| 8 | `navigator.permissions` / `plugins` | coherence patch in pre-nav script | medium |
| 9 | `hardwareConcurrency` / `deviceMemory` | set to realistic values matching the profile | low |
| 10 | screen / viewport realism | `window_size` + Emulation; avoid 1280×720-exact default | medium |
| 11 | `--headless=new` (NOT old headless) | launch arg change in `browser.rs` (§4) | high |
| 12 | `imagesEnabled=false` (currently in browser.rs) | **REMOVE** — it is fingerprintable | high |
| 13 | fixed 1500ms sleep (currently in browser.rs) | **REPLACE** with network-idle / challenge-cleared wait (§1.5) | high |
| 14 | metronomic request timing | jitter (handled by the AIMD governor, ADR 0003 §4.1) | low |

**Behavioral realism (mouse Bézier, human timing):** explicitly **LOW priority / possibly theater**
against behavioral ML (ground truth, ADR 0003 §8). We implement network-idle waits and avoid
metronomic timing (the governor already jitters), but do not build a mouse-movement engine —
behavioral ML is the single-IP hard ceiling (§5), not something a Bézier curve defeats.

## 1.5 Network-idle / challenge-cleared wait (replaces the 1500ms sleep)

`browser.rs` currently does `sleep(1500ms)` after navigation. Replace with a bounded poll that exits
on the *first* satisfied condition, capped at `browser_timeout_secs` (ADR 0003 R4 budget, default
15s):

```rust
pub struct WaitConfig {
    pub max_wait: Duration,            // = browser_timeout_secs
    pub poll_interval: Duration,       // e.g. 250ms, jittered
    pub content_selector: Option<String>, // optional "real content present" selector
}

// Exit when ANY holds:
//  (a) cf-mitigated challenge cleared: the challenge markers (_cf_chl_opt,
//      "Just a moment", /cdn-cgi/challenge-platform/) are GONE from the DOM, OR
//  (b) a cf_clearance / datadome clearance cookie has APPEARED (via Network.getCookies), OR
//  (c) the optional content_selector matches (real content rendered), OR
//  (d) network has been idle (no in-flight requests) for `quiet_period` (~500ms).
// Else: time out -> return whatever we have; classifier decides (likely still JsChallenge -> R5).
```

Network-idle detection: subscribe to `Network.requestWillBeSent` / `Network.loadingFinished` /
`Network.loadingFailed` on the page's CDP event stream and track an in-flight counter; "idle" = counter
at 0 for `quiet_period`. (chromiumoxide 0.7.0 exposes `Page::event_listener::<T>()` for typed CDP
events; if listener ergonomics differ, fall back to polling `page.content()` for the absence of the
challenge markers — both are acceptable, the marker-absence check is the load-bearing one.)

The wait NEVER returns "challenge solved" by sleeping; it returns on positive evidence (cookie /
marker-gone / content) or times out. A timeout is a give-up at R4 → routes to R5 per ADR 0003.

## 1.6 cf_clearance / DataDome cookie persistence

Single-IP advantage (ground truth, ADR 0003 §5): our IP is **static**, so a minted `cf_clearance` /
`datadome` cookie is bound to {session + UA + IP} where IP never changes — cookie reuse is *more*
stable for us than for rotating setups. Capture once, reuse for the domain TTL.

```rust
// clearance.rs
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ClearanceCookie {
    pub domain: String,         // registrable domain
    pub name: String,           // "cf_clearance" | "datadome"
    pub value: String,
    pub user_agent: String,     // the UA it was minted under — MUST be replayed verbatim
    pub minted_at: i64,         // epoch secs
    pub ttl_secs: u64,          // conservative; CF clearance ~ up to 30m, re-verify
}

pub trait ClearanceStore: Send + Sync {
    fn get(&self, domain: &str) -> Option<ClearanceCookie>;   // None if expired/missing
    fn put(&self, c: ClearanceCookie);
    // Optional file-backing for cross-restart reuse; in-memory DashMap is the default.
}
```

Capture via CDP `Network.getCookies` after the wait (§1.5) detects clearance. Inject on a later
browser visit via `Network.setCookie` before navigation. **Reuse is keyed by registrable domain and
must replay the exact UA the cookie was minted under** (a UA mismatch invalidates the cookie).

## 1.7 The wreq-replay spike (ADR 0003 §5 — DO NOT build the R4→R2/R1 reuse path until confirmed)

ADR 0003 §5 flags the highest-leverage optimization: mint `cf_clearance` in the browser, then replay
it on the cheap **wreq** client (R1) so we drain the rest of a domain with one render instead of many.
**Unverified:** the cookie is widely reported bound to the client **TLS/JA3 + HTTP/2 fingerprint**, not
just UA+IP. If so, a cookie minted under chromiumoxide's fingerprint is **rejected** when presented by
wreq unless wreq's emulation profile produces a *matching* JA3/H2 fingerprint.

**Phase-3 spike (gates the reuse path):**
1. Mint `cf_clearance` in the stealth browser on a CF-protected benchmark domain.
2. Present it on the wreq client (R1) against the same domain; measure accept/reject.
3. If accepted → build R4→R1/R2 cookie reuse (the win).
4. If rejected → **fallback:** keep the solved session **inside the browser** (drain the API /
   re-request from the headless context, not wreq). Slower, more reputation-costly, but correct.

**Do not assume replay works.** This doc specifies the capture/persist struct (§1.6) but leaves the
*cross-client* reuse path gated behind this spike.

---

# PART 2 — R5: Commercial CAPTCHA / Unblock Layer

## 2.0 Decision: provider abstraction, NOT a self-hosted solver

We do **not** build a self-hosted CAPTCHA solver (ground truth #5/§6):

- OSS solvers are dead: GoodByeCaptcha unmaintained since 2020; uncaptcha is historical/dead;
  NopeCHA is closed-source.
- arXiv 2507.23091: SOTA audio models fail **93%+** on hard audio CAPTCHA (best 6.9% vs 52% human);
  audio solving only ever worked on reCAPTCHA v2 checkbox anyway.

Instead: a **provider abstraction** over commercial APIs (CapSolver, 2Captcha), default OFF,
cost-capped, compliance-gated.

## 2.1 Module layout

```
crates/crawler/src/captcha/
├── mod.rs            # CaptchaSolver trait, CaptchaKind/Request/Token, ProviderSelection, gating
├── capsolver.rs      # CapSolverClient (reqwest) — createTask / getTaskResult
├── twocaptcha.rs     # TwoCaptchaClient (reqwest) — same two-call async shape
├── extract.rs        # sitekey / action / cdata extraction from page HTML
├── inject.rs         # token injection (CDP path for browser; HTTP-only path documented)
└── meter.rs          # per-solve + cumulative USD metering, hard cap halt
```

No mature Rust crate exists (the `capsolver` crate is empty/unreliable) → call the HTTP API directly
with `reqwest` (already a dependency).

## 2.2 The `CaptchaSolver` trait

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptchaKind { RecaptchaV2, RecaptchaV3, Hcaptcha, Turnstile }

pub struct CaptchaRequest {
    pub kind: CaptchaKind,
    pub website_url: String,
    pub website_key: String,          // sitekey
    pub action: Option<String>,       // reCAPTCHA v3 action
    pub cdata: Option<String>,        // Turnstile cData
}

pub struct CaptchaToken {
    pub token: String,                // gRecaptchaResponse | token (Turnstile/hCaptcha)
    pub user_agent: Option<String>,   // MUST be replayed on the follow-up request if present
    pub cost_usd: f64,                // metered per solve
}

#[async_trait::async_trait]
pub trait CaptchaSolver: Send + Sync {
    /// Two-call async shape: createTask -> poll getTaskResult -> token.
    async fn solve(&self, req: CaptchaRequest) -> Result<CaptchaToken, CaptchaError>;
    fn provider_name(&self) -> &'static str;
}

#[derive(thiserror::Error, Debug)]
pub enum CaptchaError {
    #[error("provider error {code}: {message}")] Provider { code: String, message: String },
    #[error("timed out after {0:?}")] Timeout(std::time::Duration),
    #[error("cost cap would be exceeded")] CostCapExceeded,   // non-retriable
    #[error("disabled by config")] Disabled,                  // non-retriable
    #[error("transport: {0}")] Transport(String),             // retriable (bounded)
}
```

## 2.3 Provider impls outline (CapSolver + 2Captcha)

Both providers share a **two-call async shape**:

1. `POST createTask` with `{ clientKey, task: { type, websiteURL, websiteKey, [action], [cdata] } }`
   → `{ taskId }`.
2. Poll `getTaskResult { clientKey, taskId }` → `{ status: idle|processing|ready, solution }`.
   Backoff between polls (e.g. start 3s, cap at provider TTL); bounded total wait
   (`captcha_timeout_secs`, default 60s — ADR 0001 R5 budget). Token field is `gRecaptchaResponse`
   (reCAPTCHA) or `token` (Turnstile/hCaptcha). The response carries a **`userAgent` the caller MUST
   match** on the follow-up request.

Task types:
- **Turnstile:** CapSolver `AntiTurnstileTaskProxyLess` (token minted without our IP).
- **reCAPTCHA v2:** `ReCaptchaV2TaskProxyLess` (CapSolver) / `userrecaptcha` (2Captcha).
- **reCAPTCHA v3:** score-based; proxyless solvers struggle → treat as **low-confidence** (§2.7, §5).
- **hCaptcha:** `HCaptchaTaskProxyLess`.

```rust
// capsolver.rs — shape only
pub struct CapSolverClient { http: reqwest::Client, client_key: String, base: String /* docs.capsolver.com */ }

#[async_trait::async_trait]
impl CaptchaSolver for CapSolverClient {
    async fn solve(&self, req: CaptchaRequest) -> Result<CaptchaToken, CaptchaError> {
        // 1. POST {base}/createTask  -> task_id
        // 2. loop: POST {base}/getTaskResult ; match status { Ready => return, Processing => sleep+poll,
        //          _ => continue } until captcha_timeout_secs -> Err(Timeout)
        // 3. map solution.gRecaptchaResponse | solution.token -> CaptchaToken { token, user_agent, cost_usd }
        unimplemented!("Phase 3 — re-verify task `type` strings + JSON field names against live docs")
    }
    fn provider_name(&self) -> &'static str { "capsolver" }
}
```

`ProviderSelection` picks the impl from `captcha_provider` config; unknown/None → solver disabled.

## 2.4 extract → solve → inject → verify flow

1. **extract** (`extract.rs`): pull `website_key` + kind from the page HTML the classifier already
   captured (ADR 0001 `Captcha{kind}` already fired, so markers are present):
   - reCAPTCHA: `data-sitekey` on `.g-recaptcha`, or sitekey in the `api.js?render=` src.
   - hCaptcha: `data-sitekey` on `.h-captcha`.
   - Turnstile: `data-sitekey` on `.cf-turnstile`; capture `data-action` / `data-cdata` if present.
   Use `scraper` (already in the workspace) for attribute reads; regex only for the `api.js` src.
2. **solve**: `CaptchaSolver::solve(req)` → `CaptchaToken` (two-call async, §2.3).
3. **inject**:
   - **Browser path (R5-after-R4, preferred):** via CDP `Runtime.evaluate` (on the stealth
     `EvalContext`), set the hidden field — `g-recaptcha-response` / `h-captcha-response` /
     `cf-turnstile-response` — then trigger the form callback / submit. Replay the provider's
     `userAgent` (already matched if R4 set the same UA).
   - **HTTP-only path:** POST the form with the token field populated via the wreq client.
     **Documented limitation:** reCAPTCHA v3 / Turnstile often need the token consumed in-page by JS
     within the **2-minute TTL**, so the pure-HTTP submit frequently fails for those; HTTP-only is
     reliable mainly for classic v2 form posts. Prefer the browser path when R4 is available.
4. **verify**: re-classify the post-submit response (ADR 0001 classifier). `RealContent` → return +
   cache (C6). Still a challenge → R5 give-up → typed error (ADR 0001 C4 `ChallengeUnsolved`).

**Token lifetime:** tokens are **single-use, ~2-minute TTL** → **no caching**, solve-on-demand,
inject within 2 minutes. Single-IP fact: **reCAPTCHA tokens are largely IP-portable** — Google
`siteverify` `remoteip` is **optional** (§6) — so a proxyless-minted token validates from our IP.

## 2.5 Gating, cost cap, secret handling (HARD requirements)

Mirrors ADR 0001 §5/§6 and the user's api-security / cost-tracking / llm-safety rules:

- **Default OFF:** `enable_captcha_solver = false`. Solver is never constructed unless explicitly on.
- **Solve ONLY on `Captcha` verdict** (or `JsChallenge` R4 could not solve, per ADR 0003 R5 trigger).
  Never speculatively solve.
- **API key by env-var NAME only** (`captcha_api_key_env`), never a literal in config, never logged.
  `meter.rs` / clients log the **provider name and a masked tail**, never the key or the token value.
- **Per-session USD cost cap** (`session_cost_cap_usd`, ADR 0001) — `meter.rs` adds each solve's
  `cost_usd`; on reaching the cap it **hard-halts** all paid rungs and emits a WARN (user rule:
  warn > $5, halt > $50). Per-solve cost also checked against `per_url_cost_budget_usd` *before*
  calling the provider.
- **Per-solve + cumulative metering** emitted as a `tracing` event `{provider, kind, cost_usd,
  cumulative_usd}` (ADR 0001 C5; no secrets).
- **Startup validation:** if `enable_captcha_solver` then `captcha_api_key_env` must name a *set*
  env var, else fail fast (config-management rule). `enable_captcha_solver` is an explicit
  GOAL.md legal-gate opt-in.

## 2.6 Compliance note

CAPTCHA solving is the GOAL.md §2 / ADR 0001 Phase-5 legal-gated tier. This layer is authorized only
against owned/permitted targets, default OFF, cost-capped, and routed only on a confirmed `Captcha`
verdict. A permanently-denylisted domain (ADR 0003 §4.2 hard trip) must **never** reach R5 — the
governor refuses all live-origin rungs first. No token, key, or solved-page is cached or logged.

## 2.7 reCAPTCHA v3 caveat

v3 is **score-based (0.0–1.0)**, not pass/fail. Proxyless solvers struggle to earn a high score, and
the score also depends on the *requesting* session's reputation. Treat v3 solves as **low-confidence**:
attempt once, but expect failure on strict thresholds; document in observability that a v3 verdict is
likely to give up. Do not spend repeated solves chasing a v3 score.

---

# PART 3 — Config delta for `CrawlerConfig`

All additive, `#[serde(default)]`, off-safe. Existing fields and ADR 0001/0003 fields unchanged.
Reuses `session_cost_cap_usd` / `per_url_cost_budget_usd` / `browser_timeout_secs` from those ADRs
(do **not** duplicate them here).

```rust
pub struct CrawlerConfig {
    // ... existing + ADR 0001/0003 fields unchanged ...

    // -- R4 stealth headless (default OFF; no-op unless `browser` feature + this flag) --
    #[serde(default)] pub enable_stealth: bool,                       // master switch, default false
    #[serde(default = "d_runtime_fix")] pub stealth_runtime_fix_mode: RuntimeFixMode, // default AddBinding
    #[serde(default = "d_challenge_mode")] pub stealth_challenge_mode: RuntimeFixMode, // default AlwaysIsolated
    #[serde(default)] pub stealth_user_agent: Option<String>,         // None => derive coherent UA (no "HeadlessChrome")
    #[serde(default = "d_lang")] pub stealth_accept_language: String, // "en-US,en;q=0.9"
    #[serde(default = "d_vw")] pub stealth_viewport_w: u32,           // e.g. 1366 (avoid 1280 default)
    #[serde(default = "d_vh")] pub stealth_viewport_h: u32,           // e.g. 768
    #[serde(default = "d_idle")] pub stealth_network_idle_ms: u64,    // quiet period, e.g. 500
    #[serde(default = "d_poll")] pub stealth_poll_interval_ms: u64,   // e.g. 250 (jittered)
    #[serde(default)] pub stealth_content_selector: Option<String>,   // optional "content present" selector

    // -- cf_clearance / datadome cookie reuse (capture/persist; cross-client reuse GATED by §1.7 spike) --
    #[serde(default)] pub enable_clearance_reuse: bool,               // default false until spike confirms
    #[serde(default)] pub clearance_store_path: Option<PathBuf>,      // None => in-memory only
    #[serde(default = "d_clear_ttl")] pub clearance_cookie_ttl_secs: u64, // conservative, e.g. 1200

    // -- R5 commercial CAPTCHA solver (default OFF; LEGAL + COST gate) --
    #[serde(default)] pub enable_captcha_solver: bool,                // default false
    #[serde(default)] pub captcha_provider: Option<String>,           // "capsolver" | "2captcha"
    #[serde(default)] pub captcha_api_key_env: Option<String>,        // env var NAME, never the key
    #[serde(default = "d_captcha_to")] pub captcha_timeout_secs: u64,  // e.g. 60
    #[serde(default = "d_captcha_poll")] pub captcha_poll_interval_ms: u64, // e.g. 3000
    #[serde(default)] pub captcha_v3_low_confidence: bool,            // default false; if true, skip v3 entirely
}
```

`d_runtime_fix()` returns `RuntimeFixMode::AddBinding`; `d_challenge_mode()` returns
`RuntimeFixMode::AlwaysIsolated`. `RuntimeFixMode` derives `Deserialize` (camelCase).

**Startup validation (additive to ADR 0001/0003):**
- `enable_captcha_solver == true` ⇒ `captcha_provider` ∈ {capsolver, 2captcha} **and**
  `captcha_api_key_env` names a *set* env var, else **fail fast**.
- `enable_stealth == true` ⇒ the `browser` cargo feature must be compiled in, else **warn** (stealth
  is a no-op without it).
- `enable_clearance_reuse == true` ⇒ **warn** that cross-client (wreq) replay is unverified (§1.7);
  in-browser reuse is always safe.
- If `stealth_user_agent` contains `"HeadlessChrome"` ⇒ **fail fast** (defeats the whole point).

---

# PART 4 — Integration points (browser.rs / fetcher.rs) without breaking cache/SPA

The existing SPA render path (`fetcher.rs:135` `is_spa && enable_browser` → `try_browser_fallback`)
and `fetch_via_browser` (search-engine JS path) must keep working unchanged when `enable_stealth` is
off. Changes are additive and gated.

### 4.1 `browser.rs`

1. **Launch args (always-safe improvements, gated where fingerprintable):**
   - Replace `--headless` with `--headless=new` (checklist #11).
   - **Remove `--blink-settings=imagesEnabled=false`** (checklist #12, fingerprintable). If a perf
     knob is still wanted, gate it behind a separate non-default flag, off by default.
   - Keep `--no-sandbox`, `--disable-dev-shm-usage`, etc.
2. **`BrowserPool::fetch` gains a stealth-aware sibling** `fetch_stealth(url, timeout, &StealthConfig)`
   (or `fetch` takes an `Option<&StealthConfig>`): when `Some`, call `stealth::prepare_page(&page, cfg)`
   **before** navigation (the §1.3 sequence), then use the §1.5 wait **instead of** the 1500ms sleep,
   then capture clearance cookies (§1.6). When `None`, the path is byte-for-byte today's behavior
   (preserves SPA render + `fetch_via_browser`).
3. The 1500ms `sleep` is removed only on the stealth path initially; the non-stealth path may keep it
   until the network-idle wait is proven equivalent (de-risks the SPA path — checklist #13 applies to
   R4, not to the legacy SPA render until validated).

### 4.2 `fetcher.rs`

- **R4 entry:** the escalation controller (ADR 0003 C2) routes `JsChallenge{vendor}` (governor-admitted)
  to `BrowserPool::fetch_stealth` with `stealth_challenge_mode` (default `AlwaysIsolated`). The existing
  `RealContent + is_spa` SPA trigger still routes to the **non-stealth** `fetch` unless `enable_stealth`
  is on, in which case it uses `stealth_runtime_fix_mode` (default `AddBinding`) — one browser entry,
  two triggers, no double render (ADR 0001 §3 item 3 preserved; classifier disambiguates).
- **Cache gating unchanged (C6):** insert iff final verdict `RealContent`. A challenge page that R4
  fails to clear is **never** cached; it routes to R5 or gives up (ADR 0003 give-up).
- **R5 entry:** on `Captcha{kind}` (or R4-failed `JsChallenge`) with `enable_captcha_solver` + budget
  OK, the controller calls the `CaptchaSolver` (§2.4 flow). Inject via the live browser page if R4
  already has one open (preferred), else the documented HTTP-only path (§2.4 limitation). Verify →
  cache on `RealContent` only.
- **No change to public signatures:** `Fetcher::fetch(url) -> Result<FetchResult>` and the
  `crawler.rs` callers are untouched (ADR 0003 C2).

---

# PART 5 — Honest boundary: what one IP genuinely cannot do

Restates ADR 0003 §8 for the R4/R5 layers specifically. With `enable_captcha_solver = false`
(default), these are **unreachable** and the ladder correctly gives up with a typed error:

| Surface | Why one IP + OSS stealth can't | Only option |
|---|---|---|
| **Per-customer behavioral ML** (enterprise CF Bot Management, DataDome ML) | scores the *session's* behavior/reputation over time from one IP; no fingerprint or Bézier curve defeats a per-customer model | **R5 vendor egress** (their IP pool + farm) |
| **Active / interactive Turnstile + WASM challenges** | require genuine in-browser interaction/compute the headless context can't fake reliably; UNPROVEN that R4 clears these | **R5** `AntiTurnstileTask` |
| **reCAPTCHA v3 strict score** | score depends on session reputation built from one IP; proxyless solve earns a low score | **R5 v3 (low-confidence)** or give up |

R4 stealth is **necessary but not guaranteed** (its failure is exactly what routes to R5). R5 is the
**only** option for the behavioral-ML / active-Turnstile residual — and it is default OFF, so by
default that residual is an honest give-up, not a silent failure. Whether G1 ≥ 0.90 is reachable with
R5 off depends on the behavioral-ML share of `benchmark/urls.jsonl` (operator input owed, ADR 0003 §8).

---

# PART 6 — Citations (ground truth, 2026-06-11)

**R4 stealth — Runtime.enable CDP leak & defeat:**
- rebrowser, "How to fix Runtime.enable CDP detection…":
  https://rebrowser.net/blog/how-to-fix-runtime-enable-cdp-detection-of-puppeteer-playwright-and-other-automation-libraries
- rebrowser-patches (fix modes: addBinding default / alwaysIsolated / enableDisable / 0):
  https://github.com/rebrowser/rebrowser-patches
- rebrowser, "Sensitive CDP Methods": https://rebrowser.net/docs/sensitive-cdp-methods
- rebrowser bot-detector (the in-page tests): https://github.com/rebrowser/rebrowser-bot-detector
- Castle, "From Puppeteer stealth to Nodriver…": https://blog.castle.io/from-puppeteer-stealth-to-nodriver-how-anti-detect-frameworks-evolved-to-evade-bot-detection/
- Octobrowser, "CDP leaks in Puppeteer…": https://blog.octobrowser.net/cdp-leaks-in-puppeteer
- DataDome, "How New Headless Chrome & the CDP Signal Are Impacting Bot Detection":
  https://datadome.co/threat-research/how-new-headless-chrome-the-cdp-signal-are-impacting-bot-detection/

**chromiumoxide 0.7.0 API (verified live 2026-06-11):**
- Page methods (`evaluate`, `execute`, `enable_runtime` "Activated by default", `disable_runtime`,
  `evaluate_on_new_document`, `wait_for_navigation`): https://docs.rs/chromiumoxide/0.7.0/chromiumoxide/page/struct.Page.html
- CDP types (`CreateIsolatedWorldParams`, `AddScriptToEvaluateOnNewDocumentParams`, `AddBindingParams`):
  https://docs.rs/chromiumoxide_cdp/latest/chromiumoxide_cdp/ and https://docs.rs/chromiumoxide/latest/chromiumoxide/cdp/

**Rust browser-stealth options compared:**
- chaser-oxide: https://github.com/ccheshirecat/chaser-oxide — eoka: https://crates.io/crates/eoka

**cf_clearance binding (session + UA + IP; possibly + JA3/H2):**
- https://kameleo.io/glossary/cf-clearance-cookie — https://www.roundproxies.com/blog/cf-clearance/

**R5 CAPTCHA — provider APIs & single-IP facts:**
- CapSolver API: https://docs.capsolver.com/en/api/ — Turnstile: https://docs.capsolver.com/en/guide/captcha/cloudflare_turnstile/
- 2Captcha reCAPTCHA v2: https://2captcha.com/api-docs/recaptcha-v2
- Google siteverify (`remoteip` optional → tokens IP-portable): https://developers.google.com/recaptcha/docs/verify
- OSS solvers dead / audio fails 93%+: arXiv 2507.23091 (audio CAPTCHA SOTA failure); GoodByeCaptcha (unmaintained since 2020); uncaptcha (historical); NopeCHA (closed-source).

---

# PART 7 — Open questions / spikes (Rule 1)

| # | Spike / unknown | Why it matters | Gate |
|---|---|---|---|
| **R-1** | **Does chromiumoxide 0.7.0 auto-fire `Runtime.enable` at page creation, and can we suppress it?** `enable_runtime()` is documented "Activated by default." If it fires before our `addBinding` setup, the leak is present regardless of mode. | Invalidates `AddBinding`/`AlwaysIsolated` if unsuppressable; would force `EnableDisable` or a **vendored/patched chromiumoxide**. | **Read the crate source before coding R4.** Document the workaround if confirmed. |
| **R-2** | **R4 efficacy vs live CF/DataDome from one IP** — all stealth modes UNPROVEN against commercial WAF (adversarial-verify abstained). | Determines how often R4 succeeds vs falls through to paid R5. | Phase-3 stealth spike against `benchmark/urls.jsonl`. |
| **R-3** | **Cookie fingerprint-binding / wreq replay (§1.7)** — is `cf_clearance` bound to JA3/H2 such that wreq can't replay a browser-minted cookie? | Gates the high-leverage R4→R1/R2 reuse path. Fallback: keep session in-browser. | Phase-3 cookie spike. **Do not build cross-client reuse until confirmed.** |
| **R-4** | **CapSolver/2Captcha real solve rates + per-solve pricing** — mechanism grounded, success/cost UNPROVEN. | Cost-cap math and whether R5 is worth enabling. | Verify pricing + run a metered trial before enabling in prod. |
| **R-5** | **chromiumoxide builder ergonomics** — exact field setters on `SetUserAgentOverrideParams` (userAgentMetadata), `CreateIsolatedWorldParams`, event-listener API for network-idle. | Compile-time correctness of the §1.3 sketch. | Confirm against 0.7.0 generated source. |
| **R-6** | **Hard-ban classifier signal (ADR 0003 §4.2)** — "JsChallenge persists after R4 also failed" is the proposed hard-trip; must not false-trip and permanently denylist a reachable domain. | A false hard-trip permanently loses a domain. | Validate against benchmark before R4-failure feeds the permanent denylist. |
