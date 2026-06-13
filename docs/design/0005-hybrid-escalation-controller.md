# Design 0005 — Hybrid Escalation Controller (classifier-routed R0/R1/R4/R5)

- **Status:** Proposed (implementation-ready spec — module layout, exact signatures, fetcher.rs seam, contracts, tests)
- **Date:** 2026-06-11
- **Owner:** solution-architect
- **Implements:** the "hybrid" ADR 0003 §3 describes — wires the **just-shipped R4 stealth headless**
  (`crates/crawler/src/browser.rs` `fetch_with_stealth`) and the **in-progress R5 CAPTCHA solver**
  (`crates/crawler/src/captcha/*`) into ONE classifier-routed escalation controller inside
  `crates/crawler/src/fetcher.rs`.
- **Consumes (does not restate):**
  - `docs/adr/0001-bypass-escalation-ladder.md` — §4 `BlockClass` classifier spec + markers + priority
    order; §7 contracts C1–C6.
  - `docs/adr/0003-single-ip-escalation-ladder.md` — reputation-first ladder, AIMD governor,
    permanent denylist, contracts C1–C7. Single static IP, recover→prevent.
  - `docs/design/0004-stealth-and-captcha-layers.md` — R4 (`fetch_with_stealth`, `BrowserFetchResult`
    now carrying `cookies`) and R5 (`CaptchaSolver` trait, two-call provider shape) module shapes.

> **Scope guard.** This is a DESIGN / CONTRACT doc. It writes **no feature code** and edits **no
> source file**. A concurrent agent is editing `crates/crawler/src/captcha/*`,
> `crates/common/src/config.rs`, `crates/common/src/error.rs`, and `crates/crawler/src/lib.rs`. This
> doc therefore (a) lists only NEW config fields that do not collide with the stealth/captcha fields
> that agent is adding, and (b) describes the `lib.rs` change as "ADD a `mod classifier;` line", never
> "replace the file".

> **Off-safe invariant (expand-and-contract).** With `enable_escalation = false` (DEFAULT) the
> controller collapses to **byte-for-byte today's behavior**: cache → retry loop → single reqwest GET
> → `is_spa` browser fallback → cache insert, with the same `Error::Blocked` / `Error::RateLimited`
> early-returns. Nothing in this doc changes a single fetch unless the operator turns the flag on.

> **Rule 1 honesty.** Every load-bearing efficacy/binding claim inherited from ADR 0003 / Design 0004
> is still **unverified** (R4 vs live WAF, cf_clearance JA3/H2 binding, hard-ban classifier signal).
> This doc carries those flags forward; it does not resolve them.

---

## 1. BlockClass classifier — concrete implementation plan

The classifier is fully specced in ADR 0001 §4 but **not built**. This is the first thing the
implementer writes; the controller (§2) and the fetcher seam (§3) depend on it.

### 1.1 New file + lib.rs wiring

- **New file:** `crates/crawler/src/classifier.rs`.
- **lib.rs:** the implementer must **ADD** one line `pub mod classifier;` to
  `crates/crawler/src/lib.rs` (current mod list: `browser, captcha, fetcher, frontier,
  link_extractor, pagination, robots, search_results, sitemap, throttle, crawler`). Because that file
  is being edited concurrently, **add the single line; do not overwrite the file** — apply it as a
  one-line insertion after `pub mod captcha;`.

### 1.2 The enum (carry evidence for observability — ADR 0001 ASR-5 / C1)

```rust
// crates/crawler/src/classifier.rs

/// Verdict over a raw HTTP response. Pure, deterministic, no I/O (ADR 0001 C1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockClass {
    /// 2xx, no block marker, visible text >= threshold. The ONLY class cached (C6).
    /// `is_spa` is carried by FetchResult, not here; the controller reads both.
    RealContent,
    /// 403 + CF/DataDome fingerprint (server: cloudflare / cf-ray / x-datadome) but
    /// NO JS-challenge or CAPTCHA body markers. IP/edge reputation block.
    Cloudflare403,
    /// Cloudflare IUAM / DataDome interstitial -> route to R4 (stealth headless).
    JsChallenge { vendor: ChallengeVendor },
    /// CAPTCHA present -> route to R5 (solver) when enabled + budget OK.
    Captcha { kind: CaptchaKind },
    /// 200 but empty/placeholder/"enable JS"/block phrase. Not real content.
    SoftBlock,
    /// 429 / vendor rate signal.
    RateLimited { retry_after_secs: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChallengeVendor { Cloudflare, DataDome, Akamai /* stub, ADR 0001 §4(2) */, Unknown }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptchaKind { Recaptcha, Hcaptcha, Turnstile }

/// Which signal fired — emitted in the tracing event (ASR-5). Returned alongside
/// the class so the controller can log it without re-deriving.
#[derive(Debug, Clone)]
pub struct Verdict {
    pub class: BlockClass,
    pub signal: &'static str,   // e.g. "cf-mitigated", "_cf_chl_opt", "g-recaptcha", "soft:title"
}
```

> Note: `CaptchaKind` here is the **classifier's** narrower set (Recaptcha/Hcaptcha/Turnstile from
> ADR 0001 §4). The R5 `captcha::CaptchaKind` (RecaptchaV2/V3/Hcaptcha/Turnstile, Design 0004 §2.2) is
> a **separate** type the concurrent agent owns. The controller maps classifier-kind → solver-kind at
> the R5 boundary (§2.4); do **not** unify the two enums (avoids a cross-crate coupling the concurrent
> edit would collide with).

### 1.3 The signature (exact — ADR 0001 C1)

```rust
/// Pure, deterministic, no I/O. Marker lists come from `cfg`, seeded with the
/// ADR 0001 §4 grounded values — NOT hard-coded literals (so vendor marker churn
/// is a config change). Priority order is enforced INSIDE this fn.
pub fn classify(
    status: u16,
    headers: &reqwest::header::HeaderMap,
    final_url: &str,
    body: &str,
    cfg: &ClassifierConfig,
) -> Verdict;
```

`ClassifierConfig` is a small struct owned by `classifier.rs` (not `CrawlerConfig`), built once from
`CrawlerConfig` fields (§6 `classifier_marker_overrides`, `soft_block_min_bytes`) plus the §4 default
marker lists. Keeping it local avoids touching `config.rs` beyond the additive fields in §6.

```rust
pub struct ClassifierConfig {
    pub soft_block_min_bytes: usize,          // default ~500 visible chars
    pub challenge_body_size_cap: usize,       // CF challenge pages < ~50KB
    pub default_retry_after_secs: u64,        // 30 (matches today's fetcher.rs default)
    pub captcha_markers: CaptchaMarkers,      // seeded with ADR 0001 §4(1)
    pub challenge_markers: ChallengeMarkers,  // seeded with ADR 0001 §4(2)
    pub soft_block_phrases: Vec<String>,      // "enable javascript", "access denied", ...
}
```

### 1.4 Priority order (first match wins — ADR 0001 §4 / C1)

**`Captcha → JsChallenge → Cloudflare403 → RateLimited → SoftBlock → RealContent`.**

Order is load-bearing: a CAPTCHA page is *also* a 403 with a CF marker, so CAPTCHA must be checked
before the generic-403 branch, or it misroutes to R1 instead of R5.

### 1.5 Concrete marker checks (ground truth: ADR 0001 §4 + browser.rs `looks_like_challenge`)

Evaluate top-to-bottom; return on first match.

**(1) `Captcha { kind }`** — status NOT required (CAPTCHA can appear inline on a 200 form):
- **Turnstile:** script `src` contains `challenges.cloudflare.com/turnstile/v0/api.js`; element
  `class="cf-turnstile"` with `data-sitekey`.
- **hCaptcha:** `src` contains `js.hcaptcha.com/1/api.js` or `hcaptcha.com/captcha`;
  `class="h-captcha"` + `data-sitekey`.
- **reCAPTCHA:** `src` contains `www.google.com/recaptcha/api.js` or `www.gstatic.com/recaptcha`;
  `class="g-recaptcha"` + `data-sitekey`.

**(2) `JsChallenge { vendor }`** — interstitial, not a CAPTCHA:
- **Cloudflare** (vendor=Cloudflare): header **`cf-mitigated: challenge`** (authoritative; only valid
  value is `challenge`) — OR body markers `_cf_chl_opt` / `window._cf_chl_opt`, path
  `/cdn-cgi/challenge-platform/`, `challenges.cloudflare.com`, visible text `just a moment` /
  `checking your browser`, legacy `cf-browser-verification`; typically `server: cloudflare`,
  `content-type: text/html`, body `< challenge_body_size_cap`. Status 403 (managed) or 503 (legacy
  IUAM — do not hard-depend on 503).
- **DataDome** (vendor=DataDome): header `x-datadome` present (and/or `x-datadome-cid`); `datadome`
  token in `Set-Cookie`; body `dd` JS object / `geo.captcha-delivery.com`. Status usually 403. (If
  CAPTCHA markers ALSO present → already returned as `Captcha` in step 1 by priority.)
- (Akamai/PerimeterX markers remain **stubs**, ADR 0001 §4(2) flag — do not invent them.)

> These are exactly the four markers `browser.rs::looks_like_challenge` already uses
> (`just a moment`, `/cdn-cgi/challenge-platform/`, `_cf_chl_opt`, `cf-mitigated`). The classifier is
> the **header+body+status** superset; the browser helper is the DOM-only subset used during the R4
> wait. Keep them consistent but separate (different inputs).

**(3) `Cloudflare403`** — status 403 **and** a CF/DataDome fingerprint (`server: cloudflare` OR
`cf-ray` present OR `x-datadome`) **but none** of the step-1/step-2 body markers. → IP/edge block. (A
plain non-CF 403 with no vendor marker falls through to `SoftBlock`.)

**(4) `RateLimited { retry_after_secs }`** — status **429**; parse `Retry-After` (seconds or
HTTP-date) → secs, default `default_retry_after_secs` (30) if absent. Also here if a vendor returns
503 + a documented rate header.

**(5) `SoftBlock`** — status 200 (or non-CF non-429 error) but content not real: visible text
(`estimate_visible_text_len`, already in fetcher.rs:365 — reuse it, do not duplicate) below
`soft_block_min_bytes`; OR body contains a known placeholder phrase (`please enable javascript`,
`access denied`, `pardon our interruption`); OR `<title>` matches a block pattern (`access denied`,
`attention required`, `just a moment`). **Disambiguation vs SPA:** prefer `SoftBlock` only when a
block phrase matches; otherwise let it fall through to `RealContent` and let `is_spa` route the
render (classifier wins on challenge phrasing, SPA wins otherwise — ADR 0001 §4(5/6)).

**(6) `RealContent`** (fall-through) — 2xx, no marker, visible text ≥ threshold. The only cached
class. `FetchResult.is_spa` (set by existing `detect_spa`) decides whether the controller then routes
to R4 as a **render** (not a challenge solve).

---

## 2. The escalation controller flow

The controller is a private method on `Fetcher` (e.g. `escalate(url) -> Result<FetchResult>`) called
from `fetch()` only when `enable_escalation == true`. It routes the classifier verdict to the rungs
that **exist today**, and leaves R2/R3 as explicit TODO stubs.

### 2.1 What exists today vs what is a stub

| Rung | Status in this iteration | Backing code |
|---|---|---|
| **R0** cache + in-band hydration salvage | cache exists; **hydration salvage UNWIRED** | moka cache (fetcher.rs:121); `extractor::hydration::extract(html) -> Option<HydrationContent>` exists but is not called from the crawler |
| **R1** coherent GET | **reqwest GET** (wreq blocked on toolchain — stays reqwest) | `fetch_once` (fetcher.rs:176) |
| **R2** alternative-surface (sitemap/RSS/internal API) | **TODO STUB** — log + fall through | `sitemap.rs` exists but not wired as a rung |
| **R3** Internet Archive CDX | **TODO STUB** — log + fall through | none |
| **R4** stealth headless | **SHIPPED** | `BrowserPool::fetch` → `fetch_with_stealth` (browser.rs:323); `BrowserFetchResult.cookies` |
| **R5** CAPTCHA solver | **being built** | `crates/crawler/src/captcha/*` (`CaptchaSolver` trait) |

> R2/R3 are zero-/low-ban-risk rungs in ADR 0003 but are **not required for the hybrid to function**.
> They appear in the flow as `// TODO(R2)/(R3)` fall-through stubs so the rung numbering matches ADR
> 0003 and the wiring slot is reserved, but the hybrid does **not block on them**.

### 2.2 Decision pseudocode

```text
escalate(url):
    domain = registrable_domain(url)

    # ---- governor gate (minimal version, §5) ----
    if governor.is_permanently_denied(domain):
        # live-origin forbidden; R0/R3 only. R3 is a stub today, so:
        return Err(PermanentlyDenied { domain, reason })    # C4 (new variant)

    # ---- R0: in-band salvage on anything we already have ----
    # (this iteration: cache is checked in fetch() before escalate(); R0 hydration
    #  salvage runs on a body we obtain below, not as a pre-fetch step. See §2.3.)

    # ---- R1: coherent reqwest GET (live origin; governor-paced) ----
    admit = governor.admit(domain).await        # waits out AIMD pace + jitter
    if admit != Proceed:
        return give_up(SkipLive)                 # budget exhausted / soft breaker open
    raw = fetch_raw_once(url).await?             # (status, headers, body) for ALL statuses — §3
    verdict = classify(raw.status, &raw.headers, &raw.final_url, &raw.body, &cfg)
    governor.record(domain, &verdict.class, Rung::R1)
    emit_tracing(url, domain, Rung::R1, &verdict, latency, cost_delta=0)

    match verdict.class:

        RealContent:
            result = raw.into_fetch_result()
            if result.is_spa && enable_browser:
                # existing SPA path, NOT a challenge solve (ADR 0001 §3 item 3)
                result = render_via_browser(url, result).await   # R4 entry, render semantics
            # R0 salvage as enrichment on the (good) body:
            maybe_enrich_with_hydration(&mut result)             # §2.3
            return Ok(result)                                    # cache gating in fetch() — §3

        SoftBlock | Cloudflare403:
            # TODO(R2): alternative-surface probe (sitemap/RSS/internal API). STUB -> fall through.
            # TODO(R3): Internet Archive CDX. STUB -> fall through.
            # R0 salvage on the soft-blocked body (might still contain JSON-LD / hydration):
            if let Some(c) = maybe_salvage_hydration(&raw.body):
                return Ok(c)                                     # salvaged real content
            return give_up(verdict)                              # typed error; soft breaker may trip

        JsChallenge { vendor }:
            if !enable_browser || rung_above_max(R4):
                return give_up(verdict)
            admit = governor.admit(domain).await
            if admit != Proceed: return give_up(SkipLive)
            br = BrowserPool::fetch(url, browser_timeout).await   # -> fetch_with_stealth
            # re-classify the rendered body (it may now be RealContent or still a challenge)
            if let Some(br) = br:
                store_clearance_cookies(domain, &br.cookies)      # §4
                v2 = classify(200, &empty_headers(), &br.final_url, &br.body, &cfg)
                governor.record(domain, &v2.class, Rung::R4)
                if v2.class == RealContent:
                    return Ok(br.into_fetch_result())
                # R4 failed to clear -> this is the hard-ban candidate signal (§5 / ADR 0003 §4.2)
                if v2.class == Captcha { .. }:
                    fallthrough to R5
            # R4 produced nothing / still challenged:
            if can_reach_r5(verdict):
                fallthrough to R5
            return give_up_r4_failed(verdict)                     # feeds breaker_verdict() — §5

        Captcha { kind }:
            fallthrough to R5

        RateLimited { retry_after_secs }:
            honor_wait(min(retry_after_secs, ceiling)).await      # requeue via frontier if > budget
            governor.record(domain, &RateLimited, Rung::R1)       # AIMD multiplicative decrease
            return give_up(verdict)                               # caller retries via frontier

    # ---- R5: solver (only when enabled + verdict==Captcha + budget OK) ----
    R5:
        if !(config.enable_captcha_solver
             && matches!(verdict.class, Captcha{..})  # or JsChallenge R4 couldn't solve
             && meter.would_stay_under_cap(per_url_cost_budget, session_cost_cap)):
            return give_up(verdict)                               # honest give-up (Disabled / CostCap)
        token = solver.solve(map_to_solver_request(verdict, url, &raw.body)).await?
        meter.add(token.cost_usd)                                 # hard-halt on cap (§ Design 0004 2.5)
        submitted = inject_and_submit(url, token).await           # browser path preferred (Design 0004 §2.4)
        v3 = classify(submitted.status, &submitted.headers, &submitted.final_url, &submitted.body, &cfg)
        governor.record(domain, &v3.class, Rung::R5)
        if v3.class == RealContent: return Ok(submitted.into_fetch_result())
        return Err(ChallengeUnsolved { url, vendor })             # C4
```

### 2.3 R0 in-band hydration salvage (wiring the existing extractor)

`extractor::hydration::extract(html) -> Option<HydrationContent>` already exists but is **never
called** from the crawler. R0 wires it as a **pure, zero-request** step over a body we already have:

- On `RealContent`: optional **enrichment** (`maybe_enrich_with_hydration`) — if the hydration blob
  yields a richer title/body than the raw HTML, prefer it. Non-fatal, additive.
- On `SoftBlock`/`Cloudflare403`: **salvage** (`maybe_salvage_hydration`) — a soft-walled page may
  still embed JSON-LD / a hydration blob with the real content; if `extract` returns `Some`, treat it
  as `RealContent` and return it **without** spending another live request. This is the cheapest win
  and never touches the origin again.

R0 has **zero ban-risk** and runs on bytes already in hand. (It does not pre-empt R1 here because in
this iteration we only obtain a body by doing the R1 GET first; ADR 0003's "R0 before any live
request" applies once R3 archive can supply a body, which is a stub. Documented gap, not a blocker.)

### 2.4 Verdict → rung routing table (the hybrid, this iteration)

| Classifier verdict | Routes to | Rung exists? | On failure → |
|---|---|---|---|
| `RealContent` (not SPA) | return immediately | — | (cache + return) |
| `RealContent` + `is_spa` | **R4 render** (non-challenge semantics) | yes (shipped) | keep HTTP body |
| `SoftBlock` | R0 salvage → *(R2/R3 TODO)* → give up | R0 yes; R2/R3 stub | typed error |
| `Cloudflare403` | R0 salvage → *(R2/R3 TODO)* → give up | R0 yes; R2/R3 stub | typed error |
| `JsChallenge{vendor}` | **R4 stealth headless** | yes (shipped) | R5 (if enabled) else give up; R4-fail = hard-ban candidate (§5) |
| `Captcha{kind}` | **R5 solver** (gated) | being built | `ChallengeUnsolved` |
| `RateLimited{n}` | honor `Retry-After`, AIMD decrease, requeue | yes | give up / requeue |

---

## 3. The exact fetcher.rs integration seam (diff-plan, NOT code)

All changes are additive and gated on `enable_escalation`. `Fetcher::fetch` keeps its **public
signature unchanged** (`pub async fn fetch(&self, url: &str) -> Result<FetchResult>`) so the five
`crawler.rs` callers (crawler.rs:153, 178-via-`fetch_via_browser`, 230, 270, 364) need **no change**
(ADR 0001/0003 C2).

### Seam 1 — `fetch_once` captures `(status, headers, body)` for ALL statuses (ADR 0001 C3)

- **Location:** `fetch_once`, fetcher.rs:176–255. The early-returns at **fetcher.rs:196–212** (429 →
  `Err(RateLimited)`) and **fetcher.rs:214–219** (`is_client_error() || is_server_error()` →
  `Err(Blocked)`) currently **discard the body before it is read** (the `response.text()` at
  fetcher.rs:233 only runs on the success path).
- **Before:** on any 4xx/5xx the body is dropped and a typed `Err` is returned; the classifier would
  be blind.
- **After (escalation path only):** introduce a sibling `fetch_raw_once(url) -> Result<RawResponse>`
  where `RawResponse { status: u16, headers: HeaderMap, final_url: String, content_type: String,
  body: String, etag, last_modified, response_time_ms }`. It reads `response.text()` for **every**
  status (including 4xx/5xx) and returns the raw tuple to the controller, which calls `classify()`.
  `fetch_once` (the legacy function) stays **exactly as-is** for the `enable_escalation == false`
  path — do not modify its early-returns; add `fetch_raw_once` alongside it (expand-and-contract).
- **Why backward-compatible:** a 2xx page with content classifies as `RealContent` → identical
  result; the only new capability is that error bodies become inspectable.

### Seam 2 — `fetch()` branches on `enable_escalation` (fetcher.rs:119–174)

- **Before:** cache check (121–128) → retry loop (132–168) calling `fetch_once`, with the
  `is_spa` browser fallback (142–147), cache insert (150–155), and the
  `Error::Blocked { status }` non-retry early-return (160–164).
- **After:** keep the cache check (121–128) verbatim. Then:
  - `if !self.enable_escalation { /* existing retry loop, unchanged */ }` — the entire current
    132–174 block moves under this guard untouched.
  - `else { return self.escalate(url).await; }` — the new controller (§2) owns the retry/backoff,
    rung selection, and **its own** cache insert (Seam 3). The controller reuses
    `self.backoff_base_ms` / `self.max_retries` and ADDS jitter to the backoff
    (`base * 2^n * (1 ± jitter)`) per ADR 0001/0003 (current fetcher.rs:134 has no jitter).
- New `Fetcher` fields (constructor fetcher.rs:64–107): `enable_escalation: bool`,
  `classifier_cfg: ClassifierConfig`, `governor: Arc<dyn ReputationGovernor>` (minimal impl §5),
  `solver: Option<Arc<dyn CaptchaSolver>>` (None unless `enable_captcha_solver`), and the clearance
  store (§4). All derived from `CrawlerConfig` in `Fetcher::new`. When the flag is off these are
  constructed but never consulted.

### Seam 3 — cache insert gated on `RealContent` only (ADR 0001 C6)

- **Location:** the unconditional `self.cache.insert(...)` at **fetcher.rs:150–155** currently caches
  **every** `Ok(result)`, including a browser-rendered challenge page or a soft-wall that still
  returned 200.
- **After:** the legacy path (flag off) keeps inserting as today (its `Ok` only ever comes from
  `fetch_once`, which already rejects 4xx/5xx — acceptable parity). The **escalation path** inserts
  iff the FINAL verdict is `RealContent` (including R0-salvaged, R4-rendered, R5-verified real
  content). Challenge / CAPTCHA / soft-block / rate-limited / denied results are **never** inserted
  (ASR-4). Optional negative cache stays a separate bounded map, default off (not built this
  iteration).

### Seam 4 — `Error::Blocked` non-retry early-return is replaced by the controller (fetcher.rs:158–166)

- The current `match` arm that returns `Err` immediately on a 4xx `Error::Blocked` (160–164) is
  **bypassed** on the escalation path — the controller, not the status code, decides whether to climb
  a rung or give up. The legacy arm stays for the flag-off path.

> **No other file changes.** `crawler.rs` callers untouched (C2). `browser.rs` `fetch` /
> `fetch_with_stealth` consumed as-is (the controller calls `BrowserPool::fetch`, which already routes
> to stealth when configured). `captcha/*`, `config.rs`, `error.rs`, `lib.rs` are the concurrent
> agent's — this doc only states the *contract additions* it needs from them (§6, §7), it does not
> describe edits to them beyond the single `mod classifier;` line.

---

## 4. cf_clearance reuse

R4 (`fetch_with_stealth`) already returns captured cookies on `BrowserFetchResult.cookies`
(browser.rs:64, 479–488), and `is_clearance_cookie` flags `cf_clearance` / `datadome`
(browser.rs:622). This iteration adds **storage + a SAFE reuse rule**; it does **not** build
cross-client replay.

### 4.1 Storage — per-domain map with TTL

A small in-process store owned by `Fetcher` (mirrors the `ClearanceStore` trait shape in Design 0004
§1.6, but minimal):

```rust
struct ClearanceEntry {
    name: String,          // "cf_clearance" | "datadome"
    value: String,
    user_agent: String,    // the UA the cookie was minted under — MUST replay verbatim
    minted_at: Instant,
    ttl: Duration,         // conservative; clearance_cookie_ttl_secs (Design 0004 config), e.g. 1200s
}
// keyed by registrable domain; DashMap<String, ClearanceEntry>. In-memory only this iteration
// (clearance_store_path file-backing is deferred — Design 0004 already defines the field).
```

`store_clearance_cookies(domain, &cookies)` (called after R4, §2.2) filters with
`is_clearance_cookie`, stamps `minted_at` + the R4 UA, and inserts. `get` returns `None` past TTL.

### 4.2 The SAFE reuse rule (given the unverified JA3/H2 binding)

- **SAFE (build now): same-client / same-path reuse.** A clearance cookie minted in the **browser**
  may be re-injected into a **later browser visit** to the same domain (via CDP `Network.setCookie`
  before navigation) replaying the **exact minted UA**. This stays inside the browser fingerprint that
  earned it — no cross-fingerprint replay — so it is sound under single-IP (IP is stable; UA replayed
  verbatim).
- **GATED (do NOT build): cross-client replay onto the reqwest/wreq R1 client.** ADR 0003 §5 /
  Design 0004 §1.7 flag that `cf_clearance` is widely reported bound to the client **TLS/JA3 + HTTP/2
  fingerprint**, not just UA+IP. Presenting a browser-minted cookie on the reqwest client would
  **likely be rejected** and could itself be a tell. **Mark this path behind the Phase-3 spike;** the
  store is written so it *can* feed R1 later, but the controller MUST NOT inject clearance cookies
  into the reqwest client until the spike confirms acceptance. `enable_clearance_reuse` (Design 0004
  field, default false) gates even the safe in-browser reuse; with it off, R4 always solves fresh.

> Honest consequence: until the spike, R4 amortization is **per-browser-visit only** — we cannot
> drain a domain's cheap JSON API on the reqwest client using an R4-minted cookie. Slower and more
> reputation-costly, but correct (ADR 0003 §5 fallback).

---

## 5. Minimal-viable governor + denylist (this iteration vs full ADR 0003)

ADR 0003 §4 specifies a full AIMD pacer + two-mode breaker + persistent denylist. For THIS iteration
we build the **smallest version that honors the single-IP hard constraint** and defer the tuning
machinery.

### 5.1 Build now (minimal)

A `ReputationGovernor` impl (`crates/crawler/src/governor.rs`, new file; also needs a `mod governor;`
ADD to lib.rs alongside the classifier line) keyed by registrable domain in a `DashMap`:

- **`admit(domain)`** — (a) if permanently denied → `DeniedPermanent`; (b) apply a **per-domain pacing
  delay with jitter** before returning `Proceed` (reuse / subsume the existing `throttle.rs`
  `Throttle::wait`, which crawler.rs already calls — the governor's pace replaces its static rate);
  (c) enforce a simple **`per_domain_request_budget`** counter — when exhausted, `SkipLive`.
- **`record(domain, class, rung)`** — minimal AIMD: on `RealContent` nudge the domain's delay **down**
  (additive-increase of rate); on any soft signal (`RateLimited` / `SoftBlock` / `JsChallenge` where
  content used to flow) **halve** the rate (multiplicative-decrease). One soft counter per domain.
- **Soft breaker (in-memory):** after `soft_breaker_fail_threshold` soft signals, open for
  `domain_breaker_cooldown_secs` → `admit` returns `SkipLive` during cooldown; auto half-open after.
- **HARD trip → permanent denylist (REQUIRED — this is the non-negotiable single-IP behavior).** The
  controller calls a pure helper:

```rust
// classifier.rs or governor.rs — pure, stateless (ADR 0003 C1 addition)
pub enum BreakerTrip { None, SoftCooldown, HardPermanent }
pub fn breaker_verdict(class: &BlockClass, r4_attempted_and_blocked: bool, cfg: &GovernorConfig)
    -> BreakerTrip;
// HardPermanent iff: (Cloudflare403 | JsChallenge | DataDome-shaped 403) AND r4_attempted_and_blocked
//   (we presented a real browser and STILL got blocked => IP reputation, not a solvable challenge),
//   OR an operator-recorded C&D entry. A single 403 with R4 untried is SoftCooldown, never Hard.
```
  On `HardPermanent`, write `{domain, reason, classifier_verdict, first_seen_iso}` to the
  **persistent file-backed denylist** (append-only JSON-lines at `permanent_denylist_path`), loaded
  at startup, checked by `admit` before any live rung. This survives restart (the soft breaker does
  not). This is the ADR 0003 §4.2/§4.3/C7 hard requirement and the GOAL.md §2 mandate — it ships now.

### 5.2 Deferred to the full ADR 0003 version

- Full continuous AIMD with `aimd_increase_step_rps` / `aimd_decrease_factor` /
  `per_domain_max_concurrency` tuning knobs (this iteration uses a coarse halve/nudge on a single
  delay value).
- `denylist_blocks_archive` nuance (R3 archive is a stub anyway, so the §4.4 distinction is moot
  this iteration — a denylisted domain simply fast-fails).
- Per-domain concurrency control beyond the existing single-browser bottleneck.
- The hard-ban signal **validation** against the benchmark (ADR 0003 flag #2 / Design 0004 R-6): the
  `breaker_verdict` threshold ships **config-tunable and conservative** (R4-must-have-failed), but a
  false hard-trip permanently loses a reachable domain — so `permanent_denylist_path = None` (default)
  keeps it in-memory-only until validated, with a **startup WARN** (ADR 0003 §6 validation).

> Scope decision (one line): **build the permanent hard-stop denylist + a coarse halve/nudge pacer +
> soft cooldown + per-domain request budget now; defer the fine-grained AIMD tuning knobs and the
> archive-blackout nuance.** The denylist is the load-bearing single-IP safety mechanism and cannot be
> deferred; the AIMD precision can.

---

## 6. Config delta (additive, `#[serde(default)]`, off-safe — NEW fields only)

These are **separate from** the stealth/captcha fields the concurrent agent is adding to
`crates/common/src/config.rs` (already present: `enable_stealth`, `stealth_user_agent`,
`stealth_locale` at config.rs:50/55/59; `enable_captcha_solver`, `captcha_provider`,
`captcha_api_key_env`, … at config.rs:70+). To avoid collisions, this doc lists **only fields not yet
in the struct**. The implementer ADDs these; the concurrent agent owns the stealth/captcha block.

```rust
// ADD to CrawlerConfig (additive; do not touch existing fields)

/// Master switch for the classifier-routed escalation controller.
/// false (DEFAULT) => byte-for-byte current behavior (§3 expand-contract).
#[serde(default)] pub enable_escalation: bool,
/// Highest rung the controller may reach. 0=R0,1=R1,4=R4,5=R5 (ADR 0003 numbering;
/// R2/R3 are stubs). Default 4 (R4, no paid R5). Clamp to <=4 if enable_captcha_solver=false.
#[serde(default = "d_max_rung")] pub max_escalation_rung: u8,

// -- classifier marker overrides (so vendor marker churn is config, not code) --
/// Optional TOML/JSON path of extra/replacement marker strings merged into the
/// seeded ADR 0001 §4 lists. None => use seeded defaults only.
#[serde(default)] pub classifier_marker_overrides: Option<PathBuf>,
/// Min visible-text chars for RealContent vs SoftBlock. Default ~500.
#[serde(default = "d_soft_min")] pub soft_block_min_bytes: usize,

// -- minimal governor / permanent denylist (§5) --
/// Live-origin request cap per domain per session. Default e.g. 50.
#[serde(default = "d_req_budget")] pub per_domain_request_budget: u32,
/// Soft signals before the cooldown breaker opens. Default e.g. 5.
#[serde(default = "d_soft_thresh")] pub soft_breaker_fail_threshold: u32,
/// Soft-trip cooldown seconds. Default e.g. 300.
#[serde(default = "d_cooldown")] pub domain_breaker_cooldown_secs: u64,
/// Pacing jitter ratio (±). Default e.g. 0.3.
#[serde(default = "d_jitter")] pub pacing_jitter_ratio: f64,
/// Persistent, restart-surviving hard-stop denylist path.
/// None (DEFAULT) => in-memory only (startup WARN: a hard ban won't survive restart).
#[serde(default)] pub permanent_denylist_path: Option<PathBuf>,
```

**Startup validation (additive; config-management rule):**
- `enable_escalation == false` ⇒ none of the above are consulted (no-op).
- `max_escalation_rung >= 5` requires `enable_captcha_solver == true` (else clamp to 4 + WARN).
- `permanent_denylist_path == None` ⇒ **WARN** (hard ban won't survive restart — ADR 0003 §6).
- reject `pacing_jitter_ratio` outside `[0, 1)`.

> Reused from the existing/concurrent config (do **not** re-declare): `browser_timeout_secs` (via R4),
> `per_url_cost_budget_usd` / `session_cost_cap_usd` (R5 meter), `enable_captcha_solver` /
> `captcha_provider` / `captcha_api_key_env` (R5 gate), `enable_stealth` (R4),
> `enable_clearance_reuse` / `clearance_cookie_ttl_secs` / `clearance_store_path` (§4). If any of these
> are NOT yet in the struct when the implementer arrives, that is the concurrent agent's field to add
> per Design 0004 Part 3 — coordinate, do not duplicate.

---

## 7. Contracts the implementer must honor (mirrors ADR 0001 C1–C6 / ADR 0003 C7)

**H1 — Classifier (= ADR 0001 C1).** `classify(status, &HeaderMap, final_url, body, &cfg) -> Verdict`
in `crates/crawler/src/classifier.rs`. Pure, deterministic, no I/O. Priority order **Captcha →
JsChallenge → Cloudflare403 → RateLimited → SoftBlock → RealContent**. Markers from `cfg` (seeded with
§1.5 grounded values), never hard-coded. Returns the firing signal for the tracing event.

**H2 — Controller wraps `fetch` (= ADR 0003 C2).** `Fetcher::fetch` public signature **unchanged**;
`crawler.rs` callers unchanged. Controller owns rung selection (§2), governor consultation before
every live rung (R1/R4/R5), cache-insert gating (H5), and the §3 seam. `enable_escalation = false` ⇒
exact current behavior.

**H3 — `fetch_raw_once` (= ADR 0001 C3).** Captures `(status, headers, final_url, body)` for **all**
statuses (no early-return before reading the body). Added alongside the untouched `fetch_once`.

**H4 — Governor + permanent denylist (= ADR 0003 C3/C7).** `ReputationGovernor` trait
(`admit`/`record`/`is_permanently_denied`); minimal impl per §5. The hard trip writes a file-backed,
append-only, fsync'd JSON-lines denylist (schema `{domain, reason, classifier_verdict,
first_seen_iso}`), loaded at startup, checked before any live rung. **No secrets in it.** The denylist
is irreversible-by-design (operator edit is the only way back).

**H5 — Cache contract (= ADR 0001 C6).** `cache.insert` iff final verdict `RealContent` (incl.
R0-salvaged / R4-rendered / R5-verified). Challenge / CAPTCHA / soft-block / rate-limited / denied are
never cached as content.

**H6 — Error semantics (= ADR 0001 C4 / ADR 0003 C4).** The controller returns the **most-informative
typed error** on give-up. Needs (from the concurrent `error.rs` agent): a `ChallengeUnsolved { url,
vendor }` and a `PermanentlyDenied { domain, reason }` variant (the existing `Blocked` /
`RateLimited` / `Captcha` variants at error.rs:17/20/69 are retained). **State this as a contract
request to that agent — do not edit `error.rs` here.**

**H7 — Observability (= ADR 0001 C5 / ADR 0003 C5).** Every rung emits a `tracing` event
`{url, domain, rung, verdict.class, verdict.signal, latency_ms, cost_usd_delta}` plus governor state
`{pace_delay_ms, request_budget_remaining, breaker_state}`. A **hard trip emits a distinct,
alert-worthy event** `{domain, reason, classifier_verdict}` (compliance-audit artifact). No secrets,
no tokens, no keys, no PII in any event (logging / api-security rules).

**H8 — R5 gating (= Design 0004 §2.5).** R5 reached **only** when `enable_captcha_solver` AND
`verdict == Captcha{..}` (or JsChallenge R4 couldn't solve) AND the meter confirms the solve stays
under `per_url_cost_budget_usd` and `session_cost_cap_usd`. Never speculatively solve; never reach R5
for a permanently-denied domain (governor refuses first).

### What the hybrid can and cannot do from one IP (honest note)

- **Can:** classify *why* a fetch failed; route JsChallenge to R4 stealth; salvage in-band content
  (R0) from soft-walled bodies for free; pace per-domain and stop **permanently** on a confirmed hard
  ban (never re-contacting a banned origin — the single-IP safety property); reach R5 only when the
  operator explicitly funds and enables it.
- **Cannot (honest residual, ADR 0003 §8 / Design 0004 Part 5):** defeat **per-customer behavioral
  ML** or **active Turnstile/WASM** from one IP with OSS technique — R4 is *best-effort* (it does NOT
  defeat the CDP `Runtime.enable` leak; that is R4 v2). With `enable_captcha_solver = false` (default)
  those pages are an **honest typed give-up**, not a silent failure. R4→reqwest cookie replay is
  **unbuilt** (JA3/H2 binding unverified, §4). Whether G1 ≥ 0.90 is reachable with R5 off depends on
  the behavioral-ML share of `benchmark/urls.jsonl` (operator input owed).

---

## 8. Test plan (no live WAF — single-IP-polite)

All tests are offline. **No test calls a real WAF or live origin** (testing rule: NO REAL EXTERNAL
SERVICES). The only live check is the already-`#[ignore]`d `stealth_smoke_sannysoft` in browser.rs.

### 8.1 Classifier unit tests (`classifier.rs` `#[cfg(test)]`) — fixture responses

Build `(status, HeaderMap, final_url, body)` fixtures and assert the exact `Verdict.class` +
`.signal`. Cover the priority order and each branch:

- **Captcha (each kind):** body with `cf-turnstile` + `data-sitekey` → `Captcha{Turnstile}`,
  signal `cf-turnstile`; `g-recaptcha`+`api.js` → `Captcha{Recaptcha}`; `h-captcha` →
  `Captcha{Hcaptcha}`. **Priority test:** a 403 page that has BOTH `cf-mitigated: challenge` AND a
  `cf-turnstile` element must classify as `Captcha`, NOT `JsChallenge` (order proof).
- **JsChallenge:** header `cf-mitigated: challenge` only → `JsChallenge{Cloudflare}` signal
  `cf-mitigated`; body `_cf_chl_opt` with 403 + `server: cloudflare` → `JsChallenge{Cloudflare}`;
  `x-datadome` header + `dd` object → `JsChallenge{DataDome}`. 503 IUAM body markers →
  `JsChallenge{Cloudflare}` (proves no hard 503 dependency).
- **Cloudflare403:** 403 + `cf-ray` present + NO body markers → `Cloudflare403` (proves it does NOT
  misroute to JsChallenge/Captcha).
- **RateLimited:** 429 + `Retry-After: 120` → `RateLimited{120}`; 429 no header →
  `RateLimited{30}` (default); `Retry-After` HTTP-date parsed to secs.
- **SoftBlock:** 200 with `<title>Access Denied</title>` → `SoftBlock`; 200 tiny body below
  `soft_block_min_bytes` with a block phrase → `SoftBlock`.
- **RealContent:** 200 full article → `RealContent`; 200 SPA shell (`<div id="root"></div>` + bundle)
  → `RealContent` (NOT SoftBlock — proves the SPA/SoftBlock disambiguation; `is_spa` handled
  downstream).
- **Determinism:** same fixture → same verdict across N calls (pure-function property).
- **Config-driven markers:** an override marker added via `ClassifierConfig` fires; a marker removed
  from the list does not.

### 8.2 Controller routing tests (`fetcher.rs`/`governor.rs` `#[cfg(test)]`) — mocked rungs

Inject test doubles so no network is touched: a fake `fetch_raw_once` returning a scripted
`RawResponse`, a fake `BrowserPool::fetch` (R4) returning a scripted `BrowserFetchResult`, a fake
`CaptchaSolver` (R5) returning a scripted token, and the real classifier + minimal governor.

- **R1 RealContent** → returns immediately, **cache.insert called once** (H5).
- **R1 RealContent + is_spa** → R4 render invoked once, no double render, result cached.
- **JsChallenge → R4 clears** (fake R4 body classifies RealContent) → returned + cached; clearance
  cookies stored (§4 store has the entry).
- **JsChallenge → R4 fails** (fake R4 body still challenged) + `enable_captcha_solver=false` →
  give-up typed error, **nothing cached**, and `breaker_verdict` returns `HardPermanent` (R4 was
  attempted-and-blocked) → denylist entry written.
- **Captcha + solver disabled** → `ChallengeUnsolved`/give-up, R5 never constructed.
- **Captcha + solver enabled + cost OK** → fake solver token injected, post-submit classifies
  RealContent → returned + cached; meter incremented.
- **Captcha + solver enabled + cost cap hit** → R5 refused (CostCap), give-up, no solve attempted.
- **Permanently-denied domain** → `admit` returns `DeniedPermanent`, **no live rung runs**,
  `PermanentlyDenied` error.
- **`enable_escalation = false`** → controller path is never entered; assert identical
  `FetchResult`/`Error` to the legacy `fetch_once` path on the same fixtures (expand-contract parity).
- **SoftBlock + hydration salvage** → fake hydration returns `Some` on a soft-walled body → treated
  as RealContent, returned + cached, **no second live request** (assert `fetch_raw_once` call count).

### 8.3 Governor unit tests

- AIMD: `record(RealContent)` decreases delay; `record(RateLimited)` halves rate.
- Per-domain budget exhaustion → `admit` returns `SkipLive`.
- Soft breaker opens after `soft_breaker_fail_threshold`, auto half-opens after cooldown.
- **Permanent denylist round-trip:** hard trip writes the JSON-lines entry; a fresh governor loading
  the same file reports `is_permanently_denied == true` (restart-survival proof). No secrets in the
  file.
- `breaker_verdict`: single 403 with R4 untried → `SoftCooldown`; same 403 after R4 failed →
  `HardPermanent`.

---

## 9. Open questions / spikes carried forward (Rule 1 — not resolved here)

| # | Carried from | Gate |
|---|---|---|
| Q1 | R4 efficacy vs live CF/DataDome from one IP (R4 does NOT defeat the `Runtime.enable` leak — that's R4 v2) | Design 0004 R-2 stealth spike |
| Q2 | cf_clearance JA3/H2 binding / cross-client wreq replay (§4) — **do not build cross-client reuse** | ADR 0003 §5 / Design 0004 R-3 cookie spike |
| Q3 | Hard-ban classifier signal ("JsChallenge persists after R4 also failed") must not false-trip the permanent denylist | ADR 0003 §4.2 / Design 0004 R-6 — keep `permanent_denylist_path=None` default until validated |
| Q4 | R2 (alternative-surface) and R3 (Internet Archive CDX) are TODO stubs this iteration | future design; hybrid does not block on them |
| Q5 | wreq toolchain block — R1 stays reqwest for now | revisit when toolchain unblocks |
