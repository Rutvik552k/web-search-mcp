//! Headless browser fallback for JS-rendered pages.
//!
//! Uses chromiumoxide (Chrome DevTools Protocol) to render pages that
//! return empty/minimal content via plain HTTP (SPAs, CAPTCHA walls, etc.).
//!
//! Feature-gated behind `browser` — when disabled, `BrowserPool::new()` returns
//! a no-op that always falls back to HTTP.
//!
//! # Stealth layer (R4 v2 — Design 0004 Part 1, ADR 0003 R4)
//!
//! When `enable_stealth` is on, a sibling render path (`fetch_with_stealth`)
//! applies a curated launch profile and a pre-navigation fingerprint preload to
//! reduce the headless tells that Cloudflare / DataDome key on. The legacy
//! non-stealth path (`fetch_with_browser`) is kept byte-for-byte for the SPA
//! render + `fetch_via_browser` flows when stealth is off.
//!
//! ## R4 v2: the load-time `Runtime.enable` leak is now CLOSED at the source
//!
//! Stock chromiumoxide 0.7.0 AUTO-FIRES `Runtime.enable` eagerly during target
//! init, before any user code runs, and it is NOT suppressible through the
//! public API — the canonical `Runtime.enable -> executionContextCreated /
//! consoleAPICalled` CDP leak Cloudflare/DataDome/sannysoft fingerprint. R4 v1
//! could only ship a STOPGAP (`disable_runtime()` after page creation, which
//! merely shrinks an already-open window).
//!
//! R4 v2 replaces that with the real fix: a **vendored + patched** chromiumoxide
//! at `vendor/chromiumoxide/` (wired via `[patch.crates-io]` in the workspace
//! `Cargo.toml`) whose `FrameManager::init_commands` no longer queues
//! `Runtime.enable` at all (see `vendor/chromiumoxide/PATCH_NOTES.md`). The
//! load-time leak is therefore gone for EVERY chromiumoxide use in this
//! workspace — stealth AND the legacy SPA path. The
//! [`CHROMIUMOXIDE_RUNTIME_ENABLE_PATCH_APPLIED`] marker documents this at the
//! call site.
//!
//! Because `Runtime.evaluate` is a *command* (not an event subscription), it
//! still works without `Runtime.enable`, defaulting to the page's main
//! execution context — so `page.content()` / `page.evaluate()` are unaffected.
//! The R4 v1 `disable_runtime()` stopgap is consequently REMOVED (there is no
//! eager enable left to disable). The remaining reachable stealth wins are
//! retained: `--headless=new`, dropping `--enable-automation` /
//! `--enable-blink-features=IdleDetection` / `--blink-settings=imagesEnabled=false`,
//! a coherent UA via `Network.setUserAgentOverride`, an absence/coherence
//! fingerprint preload installed pre-navigation via
//! `Page.addScriptToEvaluateOnNewDocument`, a network-idle / challenge-cleared
//! wait replacing the fixed sleep, and clearance-cookie capture.
//!
//! ## Honest scope
//!
//! What R4 v2 CLOSES: the load-time `Runtime.enable` CDP leak (the main win).
//! What remains UNPROVEN here: live efficacy against real Cloudflare / DataDome
//! — that needs the `#[ignore]`d live detector smoke test run manually against
//! a real Chrome + network (see `stealth_smoke_sannysoft`). Sec-CH-UA client
//! hints (`user_agent_metadata`) are still deferred.

use std::sync::Arc;
use std::time::Duration;

#[cfg(feature = "browser")]
use tokio::sync::Mutex;

/// A current, real desktop Chrome User-Agent. Used when `stealth_user_agent` is
/// unset. Deliberately contains NO "HeadlessChrome" token (Design 0004
/// checklist #3). Bump alongside the pinned Chrome major when revisiting.
pub const DEFAULT_STEALTH_UA: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36";

/// Default `Accept-Language` / `navigator.languages` coherence value.
pub const DEFAULT_STEALTH_LOCALE: &str = "en-US,en;q=0.9";

/// Realistic stealth viewport. Avoids the fingerprintable exact 1280x720 that
/// the legacy non-stealth path uses (Design 0004 checklist #10).
const STEALTH_VIEWPORT: (u32, u32) = (1366, 768);

/// R4 v2 marker: asserts (by its presence + the offline `patch_marker_*` test)
/// that this crate is built against the VENDORED chromiumoxide whose
/// `FrameManager::init_commands` no longer auto-fires `Runtime.enable`. This is
/// a documentation/contract marker, not a runtime guarantee — the actual proof
/// lives in `vendor/chromiumoxide/src/handler/frame.rs` and is exercised by the
/// offline test `vendored_chromiumoxide_omits_runtime_enable`.
pub const CHROMIUMOXIDE_RUNTIME_ENABLE_PATCH_APPLIED: bool = true;

/// Result of browser-rendered page fetch.
#[derive(Debug, Clone)]
pub struct BrowserFetchResult {
    pub url: String,
    pub final_url: String,
    pub body: String,
    pub status: u16,
    /// Cookies captured from the rendered page (stealth path only; empty on the
    /// legacy path). Surfaced so a later rung can reuse a minted
    /// `cf_clearance` / `datadome` clearance cookie (Design 0004 §1.6). Wiring
    /// the persistence/reuse is NOT done here — this just surfaces them.
    pub cookies: Vec<(String, String)>,
    /// The actual User-Agent the stealth browser presented to the page (the UA
    /// passed to `SetUserAgentOverrideParams`). `Some` on the stealth path,
    /// `None` on the legacy non-stealth path (which does not override the UA).
    ///
    /// A clearance cookie is bound to the UA it was minted under, so any later
    /// reuse MUST replay this exact UA verbatim (Design 0004 §1.6 / Design 0005
    /// §4.1). Surfaced here so the escalation controller stores the *real* minted
    /// UA with the cookie instead of a `DEFAULT_STEALTH_UA` placeholder.
    pub user_agent: Option<String>,
}

/// Resolved stealth settings, derived once from `CrawlerConfig` at pool
/// construction. `None` inside `BrowserPool` means stealth is off and the
/// legacy path runs unchanged.
#[derive(Debug, Clone)]
pub struct StealthConfig {
    /// UA presented to the page (and via `Network.setUserAgentOverride`).
    pub user_agent: String,
    /// Accept-Language + `navigator.languages` coherence value.
    pub accept_language: String,
    /// Window / viewport size.
    pub viewport: (u32, u32),
}

impl StealthConfig {
    /// Build from config-supplied overrides, falling back to current real
    /// defaults. Never yields a UA containing "HeadlessChrome".
    pub fn from_parts(user_agent: Option<&str>, locale: Option<&str>) -> Self {
        Self {
            user_agent: user_agent
                .filter(|ua| !ua.is_empty())
                .unwrap_or(DEFAULT_STEALTH_UA)
                .to_string(),
            accept_language: locale
                .filter(|l| !l.is_empty())
                .unwrap_or(DEFAULT_STEALTH_LOCALE)
                .to_string(),
            viewport: STEALTH_VIEWPORT,
        }
    }
}

/// Lazy-initialized browser pool for SPA rendering.
///
/// Holds at most one browser process. Pages are created/destroyed per fetch.
/// When the `browser` feature is disabled, all methods gracefully return None.
pub struct BrowserPool {
    #[cfg(feature = "browser")]
    browser: Mutex<Option<BrowserState>>,
    /// Stealth settings; `Some` only when `enable_stealth` is on. When `None`,
    /// every fetch takes the legacy non-stealth path.
    #[cfg(feature = "browser")]
    stealth: Option<StealthConfig>,
    #[cfg(not(feature = "browser"))]
    _phantom: (),
}

#[cfg(feature = "browser")]
struct BrowserState {
    browser: chromiumoxide::Browser,
    _handle: tokio::task::JoinHandle<()>,
    /// Whether the launched browser was configured with the stealth arg
    /// profile. We must not mix a stealth-launched browser with a non-stealth
    /// fetch (or vice versa) since the launch flags differ; if a mismatch is
    /// ever requested we relaunch.
    stealth_launch: bool,
}

impl BrowserPool {
    /// Create a new browser pool. Does NOT launch browser yet (lazy init).
    ///
    /// Stealth is OFF: equivalent to the historical `BrowserPool::new()`. Kept
    /// so callers that do not have a config (and the `cfg(not(browser))` build)
    /// keep working.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            #[cfg(feature = "browser")]
            browser: Mutex::new(None),
            #[cfg(feature = "browser")]
            stealth: None,
            #[cfg(not(feature = "browser"))]
            _phantom: (),
        })
    }

    /// Create a browser pool with stealth settings threaded in from
    /// `CrawlerConfig`. When `enable_stealth` is false this is identical to
    /// `new()`. The `browser`-feature-off build ignores the arguments and
    /// behaves like `new()`.
    pub fn with_stealth(
        enable_stealth: bool,
        user_agent: Option<&str>,
        locale: Option<&str>,
    ) -> Arc<Self> {
        #[cfg(feature = "browser")]
        {
            let stealth = if enable_stealth {
                Some(StealthConfig::from_parts(user_agent, locale))
            } else {
                None
            };
            Arc::new(Self {
                browser: Mutex::new(None),
                stealth,
            })
        }
        #[cfg(not(feature = "browser"))]
        {
            if enable_stealth {
                tracing::warn!(
                    "enable_stealth=true but the `browser` feature is not compiled in; \
                     stealth is a no-op (Design 0004 Part 3 startup-validation note)"
                );
            }
            let _ = (user_agent, locale);
            Self::new()
        }
    }

    /// Fetch a URL using headless Chrome. Returns None if browser unavailable.
    ///
    /// Routes to the stealth path only when stealth is configured on this pool;
    /// otherwise the legacy path runs unchanged.
    pub async fn fetch(&self, url: &str, timeout: Duration) -> Option<BrowserFetchResult> {
        #[cfg(feature = "browser")]
        {
            if self.stealth.is_some() {
                self.fetch_with_stealth(url, timeout).await
            } else {
                self.fetch_with_browser(url, timeout).await
            }
        }
        #[cfg(not(feature = "browser"))]
        {
            let _ = (url, timeout);
            tracing::debug!("Browser feature not enabled, skipping browser fetch");
            None
        }
    }

    /// R5 token-injection path (Design 0004 §2.4 step 3, browser path). Navigate
    /// to `url` in a stealth page, run the challenge wait, evaluate `injection_js`
    /// (which writes the solved token into the page's `*-response` field and fires
    /// the form callback — built by [`crate::captcha::inject::build_injection_js`]),
    /// give the page a brief moment to act on it, then re-read the resulting body
    /// so the controller can re-classify it.
    ///
    /// SECURITY: `injection_js` embeds the single-use token (JSON-escaped by the
    /// builder). This method NEVER logs `injection_js` or the page body. The token
    /// is consumed in-page and is NOT persisted anywhere by this method.
    ///
    /// Returns `None` when the `browser` feature is off or the page could not be
    /// driven — the controller then gives up honestly (`ChallengeUnsolved`). This
    /// is a SINGLE bounded attempt: it does not loop or retry (compliance C-7).
    #[allow(unused_variables)]
    pub async fn solve_in_page(
        &self,
        url: &str,
        injection_js: &str,
        timeout: Duration,
    ) -> Option<BrowserFetchResult> {
        #[cfg(feature = "browser")]
        {
            // The injection path is only meaningful on the stealth browser (it is
            // where R4 minted the UA / session). If stealth is not configured we
            // cannot inject into a coherent session → give up honestly.
            if self.stealth.is_none() {
                tracing::debug!(url, "solve_in_page: stealth not configured; cannot inject");
                return None;
            }
            self.solve_in_page_stealth(url, injection_js, timeout).await
        }
        #[cfg(not(feature = "browser"))]
        {
            // No browser compiled in → R5 cannot inject the token into a page.
            // The controller treats `None` as an honest give-up.
            None
        }
    }

    /// Check if browser feature is available.
    pub fn is_available(&self) -> bool {
        cfg!(feature = "browser")
    }

    // -- Legacy non-stealth path (UNCHANGED behavior) -------------------------
    // Kept byte-for-byte (modulo the added `cookies: vec![]` on the result, and
    // the `stealth_launch: false` field) so the existing SPA render and
    // `fetch_via_browser` flows are unaffected when stealth is off.
    #[cfg(feature = "browser")]
    async fn fetch_with_browser(&self, url: &str, timeout: Duration) -> Option<BrowserFetchResult> {
        use chromiumoxide::browser::{Browser, BrowserConfig};
        use futures::StreamExt;

        // FIX #1 (SSRF): Chrome uses its OWN network stack, so the reqwest
        // `SsrfResolver` does NOT protect this navigation. `new_page(url)`
        // navigates Chrome straight at `url`, so gate it the same way the reqwest
        // path is gated (scheme + resolve-then-deny, fail closed). Strict policy:
        // the browser path only runs in production where loopback is denied.
        if let Err(e) = crate::ssrf::precheck_navigation(url, crate::ssrf::SsrfPolicy::strict()).await {
            tracing::warn!(url, reason = %e, "Browser: navigation blocked by SSRF pre-check");
            return None;
        }

        // Lazy-init browser
        let mut guard = self.browser.lock().await;
        if guard.as_ref().map(|s| s.stealth_launch).unwrap_or(false) {
            // A stealth-launched browser is live but a non-stealth fetch was
            // requested; drop it so we relaunch with the legacy flags.
            *guard = None;
        }
        if guard.is_none() {
            tracing::info!("Launching headless Chrome for SPA rendering...");
            match Browser::launch(
                BrowserConfig::builder()
                    .arg("--headless")
                    .arg("--disable-gpu")
                    .arg("--no-sandbox")
                    .arg("--disable-dev-shm-usage")
                    .arg("--disable-extensions")
                    .arg("--disable-background-networking")
                    .arg("--disable-sync")
                    .arg("--disable-translate")
                    .arg("--blink-settings=imagesEnabled=false")
                    .window_size(1280, 720)
                    .build()
                    .map_err(|e| {
                        tracing::warn!("Browser config error: {e}");
                        e
                    })
                    .ok()?,
            )
            .await
            {
                Ok((browser, mut handler)) => {
                    let handle = tokio::spawn(async move {
                        use futures::StreamExt;
                        loop {
                            match handler.next().await {
                                Some(_) => {}
                                None => break,
                            }
                        }
                    });
                    *guard = Some(BrowserState {
                        browser,
                        _handle: handle,
                        stealth_launch: false,
                    });
                    tracing::info!("Headless Chrome launched successfully");
                }
                Err(e) => {
                    tracing::warn!("Failed to launch headless Chrome: {e}");
                    tracing::warn!("Install Chrome/Chromium to enable browser rendering");
                    return None;
                }
            }
        }

        let state = guard.as_ref()?;

        // Create new page, navigate, wait for content
        let page = match state.browser.new_page(url).await {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(url, error = %e, "Browser: failed to create page");
                return None;
            }
        };

        // Wait for page load with timeout
        let result = tokio::time::timeout(timeout, async {
            // Wait for navigation to complete
            if let Err(e) = page.wait_for_navigation().await {
                tracing::debug!(url, error = %e, "Browser: navigation wait failed");
            }

            // Small delay for JS execution
            tokio::time::sleep(Duration::from_millis(1500)).await;

            // Get rendered HTML
            let body = page.content().await.ok()?;
            let final_url = page.url().await.ok().flatten()
                .map(|u| u.to_string())
                .unwrap_or_else(|| url.to_string());

            Some(BrowserFetchResult {
                url: url.to_string(),
                final_url,
                body,
                status: 200,
                cookies: Vec::new(),
                // Legacy non-stealth path does not override the UA, so there is
                // no minted UA to surface for clearance-cookie binding.
                user_agent: None,
            })
        })
        .await;

        // Close the page to free resources
        let _ = page.close().await;

        match result {
            Ok(Some(r)) => {
                tracing::info!(url, body_len = r.body.len(), "Browser: rendered page");
                Some(r)
            }
            Ok(None) => {
                tracing::warn!(url, "Browser: page returned no content");
                None
            }
            Err(_) => {
                tracing::warn!(url, timeout_ms = timeout.as_millis(), "Browser: timed out");
                None
            }
        }
    }

    // -- Stealth path (R4 v1) -------------------------------------------------
    #[cfg(feature = "browser")]
    async fn fetch_with_stealth(&self, url: &str, timeout: Duration) -> Option<BrowserFetchResult> {
        use chromiumoxide::browser::{Browser, BrowserConfig};
        use chromiumoxide::cdp::browser_protocol::network::SetUserAgentOverrideParams;
        use chromiumoxide::cdp::browser_protocol::page::AddScriptToEvaluateOnNewDocumentParams;
        use futures::StreamExt;

        // FIX #1 (SSRF): gate the stealth `page.goto(url)` (below) — Chrome's own
        // network stack bypasses the reqwest `SsrfResolver`. Fail closed before
        // launching anything.
        if let Err(e) = crate::ssrf::precheck_navigation(url, crate::ssrf::SsrfPolicy::strict()).await {
            tracing::warn!(url, reason = %e, "Stealth: navigation blocked by SSRF pre-check");
            return None;
        }

        let cfg = self.stealth.as_ref()?.clone();

        // Lazy-init / (re)launch with the stealth arg profile.
        let mut guard = self.browser.lock().await;
        if !guard.as_ref().map(|s| s.stealth_launch).unwrap_or(false) {
            // Either no browser yet, or a non-stealth browser is live; relaunch
            // with the curated stealth flags.
            *guard = None;
        }
        if guard.is_none() {
            tracing::info!("Launching stealth headless Chrome (R4 v1)...");
            // disable_default_args() drops chromiumoxide's DEFAULT_ARGS, which
            // include the `--enable-automation` and
            // `--enable-blink-features=IdleDetection` stealth tells
            // (browser.rs:1005,1008). We re-supply a curated list that OMITS
            // both, uses `--headless=new` (not bare `--headless`), and DROPS
            // `--blink-settings=imagesEnabled=false` (fingerprintable —
            // Design 0004 checklist #12). Sandbox/gpu/dev-shm args are kept.
            let mut builder = BrowserConfig::builder().disable_default_args();
            for arg in stealth_launch_args() {
                builder = builder.arg(arg);
            }
            let conf = match builder
                .window_size(cfg.viewport.0, cfg.viewport.1)
                .build()
            {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!("Stealth browser config error: {e}");
                    return None;
                }
            };
            match Browser::launch(conf).await {
                Ok((browser, mut handler)) => {
                    let handle = tokio::spawn(async move {
                        loop {
                            match handler.next().await {
                                Some(_) => {}
                                None => break,
                            }
                        }
                    });
                    *guard = Some(BrowserState {
                        browser,
                        _handle: handle,
                        stealth_launch: true,
                    });
                    tracing::info!("Stealth headless Chrome launched successfully");
                }
                Err(e) => {
                    tracing::warn!("Failed to launch stealth headless Chrome: {e}");
                    tracing::warn!("Install Chrome/Chromium to enable browser rendering");
                    return None;
                }
            }
        }

        let state = guard.as_ref()?;

        // STEP A — create the page at about:blank FIRST, so we can install the
        // pre-navigation preload + UA override BEFORE the real navigation.
        // chromiumoxide 0.7.0 has no `new_blank_page`; `new_page` always
        // navigates, so we navigate to about:blank then `goto(url)` ourselves.
        let page = match state.browser.new_page("about:blank").await {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(url, error = %e, "Stealth: failed to create page");
                return None;
            }
        };

        // STEP A.1 — R4 v2: the `disable_runtime()` stopgap is GONE. The
        // vendored chromiumoxide (`[patch.crates-io]`) no longer auto-fires
        // `Runtime.enable` at target init, so there is no eager enable left to
        // disable — the load-time `Runtime.enable -> executionContextCreated /
        // consoleAPICalled` leak never opens. `page.content()` / `page.evaluate`
        // below still work because `Runtime.evaluate` is a command that defaults
        // to the page's main context without needing `Runtime.enable`.
        debug_assert!(CHROMIUMOXIDE_RUNTIME_ENABLE_PATCH_APPLIED);

        // STEP B — pre-navigation fingerprint preload via
        // Page.addScriptToEvaluateOnNewDocument. Runs in every new frame BEFORE
        // the page's own scripts. This is best-effort and itself detectable via
        // `.toString()` "[native code]" checks; the absence-based fix (relying
        // on `navigator.webdriver` simply being absent under `--headless=new` +
        // a patched runtime) needs R4 v2. Kept minimal on purpose.
        if let Ok(params) = AddScriptToEvaluateOnNewDocumentParams::builder()
            .source(fingerprint_preload_script())
            .build()
        {
            if let Err(e) = page.execute(params).await {
                tracing::debug!(url, error = %e, "Stealth: preload install failed (non-fatal)");
            }
        }

        // STEP C — UA + Accept-Language coherence via
        // Network.setUserAgentOverride. We set what compiles cleanly on 0.7.0:
        // user_agent + accept_language. `user_agent_metadata` (Sec-CH-UA brands
        // / platform) is left as a TODO for R4 v2: building a coherent
        // UserAgentMetadata by hand on 0.7.0 is verbose and easy to get
        // INcoherent (an incoherent Sec-CH-UA is worse than none), so we defer
        // it rather than ship a mismatched client-hints set.
        // TODO(R4 v2): set user_agent_metadata with brands/platform matching the UA.
        match SetUserAgentOverrideParams::builder()
            .user_agent(cfg.user_agent.clone())
            .accept_language(cfg.accept_language.clone())
            .build()
        {
            Ok(ua_params) => {
                if let Err(e) = page.execute(ua_params).await {
                    tracing::debug!(url, error = %e, "Stealth: UA override failed (non-fatal)");
                }
            }
            Err(e) => {
                tracing::debug!(url, error = %e, "Stealth: UA override build failed (non-fatal)");
            }
        }

        // The exact UA presented via SetUserAgentOverrideParams above — surfaced
        // on the result so the clearance store binds the cookie to the real
        // minted UA (Design 0004 §1.6 / Design 0005 §4.1), not a placeholder.
        let minted_ua = cfg.user_agent.clone();

        // STEP D — navigate to the real URL, run the challenge-cleared /
        // network-idle wait (replaces the legacy fixed 1500ms sleep), capture
        // cookies, read content. All under the overall `timeout`.
        let result = tokio::time::timeout(timeout, async {
            if let Err(e) = page.goto(url).await {
                tracing::debug!(url, error = %e, "Stealth: goto failed");
                return None;
            }
            if let Err(e) = page.wait_for_navigation().await {
                tracing::debug!(url, error = %e, "Stealth: navigation wait failed");
            }

            // Bounded poll: exit on the FIRST satisfied condition — a clearance
            // cookie appears, the challenge markers are gone, OR the DOM has
            // been quiet for `QUIET_PERIOD`. Never returns "solved" by sleeping
            // a fixed amount; it returns on positive evidence or it times out
            // (the outer `timeout` caps it). Design 0004 §1.5.
            stealth_wait(&page, url).await;

            let body = page.content().await.ok()?;
            let final_url = page
                .url()
                .await
                .ok()
                .flatten()
                .unwrap_or_else(|| url.to_string());

            // Capture cookies (esp. cf_clearance / datadome) for a later rung
            // to reuse. Persistence/reuse wiring is out of scope (Design 0004
            // §1.6 / §1.7 — the cross-client replay path is gated by a spike).
            let cookies = match page.get_cookies().await {
                Ok(cs) => cs
                    .into_iter()
                    .map(|c| (c.name, c.value))
                    .collect::<Vec<_>>(),
                Err(e) => {
                    tracing::debug!(url, error = %e, "Stealth: cookie capture failed");
                    Vec::new()
                }
            };

            Some(BrowserFetchResult {
                url: url.to_string(),
                final_url,
                body,
                status: 200,
                cookies,
                user_agent: Some(minted_ua),
            })
        })
        .await;

        let _ = page.close().await;

        match result {
            Ok(Some(r)) => {
                let has_clearance = r
                    .cookies
                    .iter()
                    .any(|(n, _)| is_clearance_cookie(n));
                tracing::info!(
                    url,
                    body_len = r.body.len(),
                    cookies = r.cookies.len(),
                    has_clearance,
                    "Stealth: rendered page"
                );
                Some(r)
            }
            Ok(None) => {
                tracing::warn!(url, "Stealth: page returned no content");
                None
            }
            Err(_) => {
                tracing::warn!(url, timeout_ms = timeout.as_millis(), "Stealth: timed out");
                None
            }
        }
    }

    /// Stealth-path token injection (R5 browser path). Navigates, waits, runs the
    /// injection JS via `Page::evaluate`, gives the page a brief settle window,
    /// then re-reads the body. Single bounded attempt (compliance C-7). NEVER logs
    /// the injection JS or the body (token is secret-equivalent).
    #[cfg(feature = "browser")]
    async fn solve_in_page_stealth(
        &self,
        url: &str,
        injection_js: &str,
        timeout: Duration,
    ) -> Option<BrowserFetchResult> {
        use chromiumoxide::browser::{Browser, BrowserConfig};
        use chromiumoxide::cdp::browser_protocol::network::SetUserAgentOverrideParams;
        use chromiumoxide::cdp::browser_protocol::page::AddScriptToEvaluateOnNewDocumentParams;
        use futures::StreamExt;

        // FIX #1 (SSRF): gate the `page.goto(url)` below — Chrome bypasses the
        // reqwest `SsrfResolver`. Fail closed before launching anything.
        if let Err(e) = crate::ssrf::precheck_navigation(url, crate::ssrf::SsrfPolicy::strict()).await {
            tracing::warn!(url, reason = %e, "solve_in_page: navigation blocked by SSRF pre-check");
            return None;
        }

        let cfg = self.stealth.as_ref()?.clone();

        let mut guard = self.browser.lock().await;
        if !guard.as_ref().map(|s| s.stealth_launch).unwrap_or(false) {
            *guard = None;
        }
        if guard.is_none() {
            let mut builder = BrowserConfig::builder().disable_default_args();
            for arg in stealth_launch_args() {
                builder = builder.arg(arg);
            }
            let conf = builder.window_size(cfg.viewport.0, cfg.viewport.1).build().ok()?;
            match Browser::launch(conf).await {
                Ok((browser, mut handler)) => {
                    let handle = tokio::spawn(async move {
                        loop {
                            match handler.next().await {
                                Some(_) => {}
                                None => break,
                            }
                        }
                    });
                    *guard = Some(BrowserState { browser, _handle: handle, stealth_launch: true });
                }
                Err(e) => {
                    tracing::warn!(url, error = %e, "solve_in_page: stealth launch failed");
                    return None;
                }
            }
        }
        let state = guard.as_ref()?;
        let page = state.browser.new_page("about:blank").await.ok()?;
        // R4 v2: no `disable_runtime()` stopgap — the vendored chromiumoxide
        // never auto-fires `Runtime.enable` (see module docs / PATCH_NOTES.md).
        debug_assert!(CHROMIUMOXIDE_RUNTIME_ENABLE_PATCH_APPLIED);
        if let Ok(params) = AddScriptToEvaluateOnNewDocumentParams::builder()
            .source(fingerprint_preload_script())
            .build()
        {
            let _ = page.execute(params).await;
        }
        if let Ok(ua_params) = SetUserAgentOverrideParams::builder()
            .user_agent(cfg.user_agent.clone())
            .accept_language(cfg.accept_language.clone())
            .build()
        {
            let _ = page.execute(ua_params).await;
        }
        let minted_ua = cfg.user_agent.clone();

        let result = tokio::time::timeout(timeout, async {
            if page.goto(url).await.is_err() {
                return None;
            }
            let _ = page.wait_for_navigation().await;
            stealth_wait(&page, url).await;

            // Inject the solved token + fire the page callback. We evaluate the
            // builder-produced IIFE; its return value is ignored. NEVER logged.
            //
            // R4 v2 note: with `Runtime.enable` removed in the vendored crate,
            // `page.evaluate` runs in the page's MAIN execution context via a
            // context-less `Runtime.evaluate` command (chromiumoxide passes
            // `contextId: None` because `execution_context()` is now `None`).
            // Running the token injection in the main world is REQUIRED here —
            // the page's own `*-response` field + form callback live in the main
            // world, so an isolated world would not reach them. The leak we
            // cared about (the load-time auto `Runtime.enable`) is already gone;
            // this per-call evaluate does not re-introduce it.
            if let Err(e) = page.evaluate(injection_js).await {
                tracing::debug!(url, error = %e, "solve_in_page: injection evaluate failed (non-fatal)");
            }
            // Brief settle window for the page's own callback / navigation to act
            // on the injected token (bounded, single attempt — no loop).
            tokio::time::sleep(Duration::from_millis(1500)).await;
            let _ = page.wait_for_navigation().await;

            let body = page.content().await.ok()?;
            let final_url = page.url().await.ok().flatten().unwrap_or_else(|| url.to_string());
            let cookies = match page.get_cookies().await {
                Ok(cs) => cs.into_iter().map(|c| (c.name, c.value)).collect::<Vec<_>>(),
                Err(_) => Vec::new(),
            };
            Some(BrowserFetchResult {
                url: url.to_string(),
                final_url,
                body,
                status: 200,
                cookies,
                user_agent: Some(minted_ua),
            })
        })
        .await;

        let _ = page.close().await;
        match result {
            Ok(Some(r)) => Some(r),
            _ => None,
        }
    }
}

// -- Pure helpers (unit-tested below) ----------------------------------------

/// Curated launch args for the stealth profile. Replaces chromiumoxide's
/// DEFAULT_ARGS (dropped via `disable_default_args()`).
///
/// Design 0004 §4.1 / checklist:
/// - `--headless=new` NOT bare `--headless` (#11)
/// - OMITS `--enable-automation` and `--enable-blink-features=IdleDetection` (#1/#11 tells)
/// - DROPS `--blink-settings=imagesEnabled=false` (#12, fingerprintable)
/// - keeps sandbox/gpu/dev-shm and other benign hardening flags.
fn stealth_launch_args() -> Vec<&'static str> {
    vec![
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-extensions",
        "--disable-background-networking",
        "--disable-background-timer-throttling",
        "--disable-backgrounding-occluded-windows",
        "--disable-renderer-backgrounding",
        "--disable-sync",
        "--disable-translate",
        "--disable-default-apps",
        "--disable-hang-monitor",
        "--disable-popup-blocking",
        "--disable-prompt-on-repost",
        "--no-first-run",
        "--force-color-profile=srgb",
        "--password-store=basic",
        "--use-mock-keychain",
        "--lang=en-US",
    ]
}

/// The pre-navigation fingerprint preload. Best-effort coherence patches that
/// ABSENCE alone does not cover. NOTE: every overwrite here is detectable via
/// `.toString()`/`Proxy` probes; the robust absence-based fix needs R4 v2. Kept
/// minimal deliberately (over-spoofing is itself a signal — Design 0004 #7).
fn fingerprint_preload_script() -> String {
    // Wrapped in an IIFE; each patch is independently try/caught so one failure
    // does not abort the rest. Comments are intentional documentation of WHY.
    r#"
(() => {
  try {
    // navigator.webdriver: under --headless=new it should be absent/false. We
    // delete the own-prop and force the prototype getter to undefined as a
    // belt-and-suspenders measure. (Absence is the real fix; this is a stopgap.)
    try { delete Object.getPrototypeOf(navigator).webdriver; } catch (e) {}
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
  } catch (e) {}

  try {
    // window.chrome: real Chrome exposes a chrome.runtime object; headless does
    // not. Provide a plausible minimal shape.
    if (!window.chrome) {
      window.chrome = { runtime: {} };
    }
  } catch (e) {}

  try {
    // navigator.permissions.query coherence: headless reports 'denied' for
    // notifications while Notification.permission is 'default' — an inconsistency
    // detectors check. Align query() with Notification.permission.
    const orig = navigator.permissions && navigator.permissions.query;
    if (orig) {
      navigator.permissions.query = (params) =>
        params && params.name === 'notifications'
          ? Promise.resolve({ state: Notification.permission })
          : orig.call(navigator.permissions, params);
    }
  } catch (e) {}

  try {
    // navigator.plugins / mimeTypes: empty arrays are a headless tell. Provide a
    // small plausible non-empty length.
    Object.defineProperty(navigator, 'plugins', {
      get: () => [1, 2, 3],
    });
  } catch (e) {}

  try {
    // navigator.languages: coherent with the Accept-Language we set via CDP.
    Object.defineProperty(navigator, 'languages', {
      get: () => ['en-US', 'en'],
    });
  } catch (e) {}
})();
"#
    .to_string()
}

/// True if a cookie name is a known WAF clearance cookie worth reusing.
fn is_clearance_cookie(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n == "cf_clearance" || n == "datadome"
}

/// Cloudflare / DataDome interstitial markers. While ANY of these is present in
/// the DOM, the challenge is not cleared. Design 0004 §1.5(a).
#[cfg(feature = "browser")]
const CHALLENGE_MARKERS: [&str; 4] = [
    "just a moment",
    "/cdn-cgi/challenge-platform/",
    "_cf_chl_opt",
    "cf-mitigated",
];

/// Heuristic: does this rendered HTML still look like an unsolved challenge?
fn looks_like_challenge(body_lower: &str) -> bool {
    const MARKERS: [&str; 4] = [
        "just a moment",
        "/cdn-cgi/challenge-platform/",
        "_cf_chl_opt",
        "cf-mitigated",
    ];
    MARKERS.iter().any(|m| body_lower.contains(m))
}

/// Bounded challenge-cleared / network-quiet wait (Design 0004 §1.5).
///
/// Replaces the legacy fixed `sleep(1500ms)`. Polls the page on a short
/// interval and returns as soon as ANY of the following holds:
///   (a) a `cf_clearance` / `datadome` clearance cookie has appeared, OR
///   (b) the challenge markers are GONE from the DOM AND the DOM has been
///       stable (unchanged length) for `QUIET_PERIOD` (proxy for network-idle:
///       0.7.0's typed event-listener ergonomics are awkward, so we use the
///       load-bearing marker-absence + DOM-stability check the spec sanctions
///       as the acceptable fallback).
/// The OUTER `tokio::time::timeout` caps total time at the rung budget; this
/// loop simply exits early on positive evidence rather than always sleeping.
#[cfg(feature = "browser")]
async fn stealth_wait(page: &chromiumoxide::Page, url: &str) {
    const POLL_INTERVAL: Duration = Duration::from_millis(250);
    const QUIET_PERIOD: Duration = Duration::from_millis(500);
    let _ = CHALLENGE_MARKERS; // documented marker set; logic uses looks_like_challenge

    let mut last_len: Option<usize> = None;
    let mut stable_since: Option<std::time::Instant> = None;

    loop {
        // (a) clearance cookie present?
        if let Ok(cookies) = page.get_cookies().await {
            if cookies.iter().any(|c| is_clearance_cookie(&c.name)) {
                tracing::debug!(url, "Stealth wait: clearance cookie appeared");
                return;
            }
        }

        // (b) challenge gone + DOM stable for the quiet period?
        if let Ok(body) = page.content().await {
            let body_lower = body.to_lowercase();
            let challenged = looks_like_challenge(&body_lower);
            let len = body.len();

            if !challenged {
                match last_len {
                    Some(prev) if prev == len => {
                        let since = *stable_since.get_or_insert_with(std::time::Instant::now);
                        if since.elapsed() >= QUIET_PERIOD {
                            tracing::debug!(url, "Stealth wait: DOM quiet, no challenge markers");
                            return;
                        }
                    }
                    _ => {
                        // length changed (still rendering) — reset stability clock
                        stable_since = None;
                    }
                }
                last_len = Some(len);
            } else {
                // still challenged — keep waiting (reset stability)
                last_len = Some(len);
                stable_since = None;
            }
        }

        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_creates_without_browser_feature() {
        let pool = BrowserPool::new();
        // Should not panic regardless of feature flag
        assert!(!pool.is_available() || pool.is_available());
    }

    #[test]
    fn with_stealth_disabled_constructs() {
        // enable_stealth=false must construct cleanly and never panic.
        let pool = BrowserPool::with_stealth(false, None, None);
        assert!(!pool.is_available() || pool.is_available());
    }

    #[test]
    fn with_stealth_enabled_constructs() {
        let pool = BrowserPool::with_stealth(
            true,
            Some("Mozilla/5.0 ... Chrome/124.0.0.0 Safari/537.36"),
            Some("de-DE,de;q=0.9"),
        );
        assert!(!pool.is_available() || pool.is_available());
    }

    #[test]
    fn stealth_config_uses_defaults_when_unset() {
        let cfg = StealthConfig::from_parts(None, None);
        assert_eq!(cfg.user_agent, DEFAULT_STEALTH_UA);
        assert_eq!(cfg.accept_language, DEFAULT_STEALTH_LOCALE);
        assert_eq!(cfg.viewport, STEALTH_VIEWPORT);
    }

    #[test]
    fn stealth_config_uses_overrides() {
        let cfg = StealthConfig::from_parts(Some("MyUA/1.0"), Some("fr-FR,fr;q=0.9"));
        assert_eq!(cfg.user_agent, "MyUA/1.0");
        assert_eq!(cfg.accept_language, "fr-FR,fr;q=0.9");
    }

    #[test]
    fn stealth_config_empty_overrides_fall_back_to_defaults() {
        // Empty strings must NOT be accepted as a UA/locale.
        let cfg = StealthConfig::from_parts(Some(""), Some(""));
        assert_eq!(cfg.user_agent, DEFAULT_STEALTH_UA);
        assert_eq!(cfg.accept_language, DEFAULT_STEALTH_LOCALE);
    }

    #[test]
    fn default_ua_has_no_headless_token() {
        // The whole point of the stealth UA: never carry "HeadlessChrome".
        assert!(!DEFAULT_STEALTH_UA.contains("HeadlessChrome"));
        assert!(DEFAULT_STEALTH_UA.contains("Chrome/"));
    }

    #[test]
    fn stealth_args_omit_automation_tells() {
        let args = stealth_launch_args();
        // The two stealth tells from chromiumoxide DEFAULT_ARGS must be absent.
        assert!(!args.iter().any(|a| a.contains("--enable-automation")));
        assert!(!args
            .iter()
            .any(|a| a.contains("IdleDetection")));
        // imagesEnabled=false is fingerprintable and must be dropped.
        assert!(!args.iter().any(|a| a.contains("imagesEnabled")));
    }

    #[test]
    fn stealth_args_use_new_headless_and_keep_sandbox() {
        let args = stealth_launch_args();
        assert!(args.contains(&"--headless=new"));
        // bare --headless must NOT appear (it would re-enable old headless).
        assert!(!args.contains(&"--headless"));
        assert!(args.contains(&"--no-sandbox"));
        assert!(args.contains(&"--disable-dev-shm-usage"));
        assert!(args.contains(&"--disable-gpu"));
    }

    #[test]
    fn preload_script_patches_expected_surfaces() {
        let s = fingerprint_preload_script();
        assert!(s.contains("webdriver"));
        assert!(s.contains("window.chrome"));
        assert!(s.contains("permissions"));
        assert!(s.contains("plugins"));
        assert!(s.contains("languages"));
        // Must be wrapped so it cannot throw at top level.
        assert!(s.contains("try"));
    }

    #[test]
    fn clearance_cookie_detection() {
        assert!(is_clearance_cookie("cf_clearance"));
        assert!(is_clearance_cookie("CF_Clearance")); // case-insensitive
        assert!(is_clearance_cookie("datadome"));
        assert!(!is_clearance_cookie("session_id"));
        assert!(!is_clearance_cookie("__cfduid"));
    }

    #[test]
    fn challenge_marker_detection() {
        assert!(looks_like_challenge("<title>just a moment...</title>"));
        assert!(looks_like_challenge(
            r#"<script src="/cdn-cgi/challenge-platform/h/b/orchestrate"></script>"#
        ));
        assert!(looks_like_challenge("window._cf_chl_opt = {}"));
        assert!(!looks_like_challenge(
            "<html><body><h1>real article content here</h1></body></html>"
        ));
    }

    #[test]
    fn runtime_enable_patch_marker_is_set() {
        // Offline contract check: the crate advertises that it is built against
        // the vendored chromiumoxide whose init chain omits `Runtime.enable`.
        assert!(CHROMIUMOXIDE_RUNTIME_ENABLE_PATCH_APPLIED);
    }

    /// Offline proof that the RESOLVED chromiumoxide is the vendored fork with
    /// the R4 v2 patch applied — i.e. that `FrameManager::init_commands` no
    /// longer queues `Runtime.enable`. We can't introspect the compiled crate's
    /// CDP chain at runtime, so we assert on the patched SOURCE that the build
    /// actually compiled against (the `[patch.crates-io]` path). If a
    /// chromiumoxide version bump silently dropped the patch (re-vendor not
    /// re-applied), this fails in CI — offline, no Chrome/network needed.
    ///
    /// The path is workspace-root-relative: this test file lives at
    /// `crates/crawler/src/browser.rs`, so the vendored crate is two dirs up
    /// from the crate root (`CARGO_MANIFEST_DIR` = `crates/crawler`).
    #[test]
    fn vendored_chromiumoxide_omits_runtime_enable() {
        use std::path::Path;
        let manifest_dir = env!("CARGO_MANIFEST_DIR"); // .../crates/crawler
        let frame_rs = Path::new(manifest_dir)
            .join("..")
            .join("..")
            .join("vendor")
            .join("chromiumoxide")
            .join("src")
            .join("handler")
            .join("frame.rs");
        let src = std::fs::read_to_string(&frame_rs).unwrap_or_else(|e| {
            panic!(
                "R4 v2: cannot read vendored chromiumoxide frame.rs at {}: {e}. \
                 The [patch.crates-io] vendored fork must be present.",
                frame_rs.display()
            )
        });

        // Narrow to the init_commands fn body so we don't match the explanatory
        // comment block (which intentionally mentions the removed command).
        let body_start = src
            .find("pub fn init_commands(")
            .expect("init_commands fn must exist in vendored frame.rs");
        let body = &src[body_start..];
        let body_end = body
            .find("pub fn main_frame(")
            .map(|i| body_start + i)
            .unwrap_or(src.len());
        let init_body = &src[body_start..body_end];

        // The R4 v2 patch removes the construction of `runtime::EnableParams`
        // and its push into the CommandChain. Neither token may appear as live
        // code in the init_commands body. (The doc comment uses backticked
        // `Runtime.enable` / `runtime::EnableParams` in prose — that's fine; we
        // assert on the actual constructor call form, which only ever appeared
        // as `runtime::EnableParams::default()`.)
        assert!(
            !init_body.contains("runtime::EnableParams::default()"),
            "R4 v2 REGRESSION: vendored chromiumoxide init_commands still \
             constructs `runtime::EnableParams::default()` — the auto-fired \
             Runtime.enable leak is back. Re-apply the patch (see \
             vendor/chromiumoxide/PATCH_NOTES.md)."
        );
        assert!(
            !init_body.contains("enable_runtime.identifier()"),
            "R4 v2 REGRESSION: vendored chromiumoxide init_commands still pushes \
             the Runtime.enable command into the CommandChain."
        );

        // Sanity: the KEPT commands must still be present (we didn't gut Page.*).
        assert!(
            init_body.contains("page::EnableParams::default()"),
            "Page.enable must be KEPT (drives lifecycle/navigation events)."
        );
        assert!(
            init_body.contains("SetLifecycleEventsEnabledParams::new(true)"),
            "Page.setLifecycleEventsEnabled(true) must be KEPT."
        );
    }

    /// LIVE stealth-detector smoke test. IGNORED: needs a real Chrome binary,
    /// network egress, and is single-IP-polite (we must NOT hammer real
    /// Cloudflare/DataDome). R4 v2 IS ACTIVE for this path: the vendored
    /// chromiumoxide no longer auto-fires `Runtime.enable`, so against
    /// rebrowser-bot-detector the "Runtime.enable leak" check should now read
    /// GREEN (the load-time leak is closed); the remaining checks (Sec-CH-UA,
    /// `.toString()` probes) are still best-effort. Run manually against
    /// bot.sannysoft.com or rebrowser-bot-detector with:
    ///   cargo test -p web-search-crawler --features browser -- --ignored stealth_smoke
    #[tokio::test]
    #[ignore = "needs Chrome + network; single-IP-polite; not for CI"]
    async fn stealth_smoke_sannysoft() {
        let pool = BrowserPool::with_stealth(true, None, None);
        if !pool.is_available() {
            return;
        }
        // bot.sannysoft.com is a passive detector page (no real WAF / no abuse).
        let res = pool
            .fetch("https://bot.sannysoft.com/", Duration::from_secs(20))
            .await;
        // We only assert we got SOMETHING back; scoring the page is a manual
        // visual/parse step, deliberately not asserted in an automated test.
        assert!(res.is_some(), "expected a rendered result from the detector page");
        let res = res.unwrap();
        assert!(res.body.len() > 1000, "expected a non-trivial rendered body");
    }
}
