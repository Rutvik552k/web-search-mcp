use crate::browser::BrowserPool;
use crate::captcha::{CostMeter, CostMeterHandle};
use crate::classifier::{self, BlockClass, ChallengeVendor, ClassifierConfig, Verdict};
use crate::governor::{
    Admission, GovernorConfig, MinimalGovernor, ReputationGovernor, Rung, breaker_verdict,
    BreakerTrip,
};
use dashmap::DashMap;
use moka::future::Cache;
use reqwest::header::HeaderMap;
use reqwest::{Client, StatusCode};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use web_search_common::config::CrawlerConfig;
use web_search_common::{Error, Result};

/// Result of fetching a URL.
#[derive(Debug, Clone)]
pub struct FetchResult {
    pub url: String,
    pub final_url: String,
    pub status: u16,
    pub content_type: String,
    pub body: String,
    pub response_time_ms: u64,
    pub content_length: usize,
    pub is_spa: bool,
    /// Whether this result was served from cache
    pub from_cache: bool,
    /// ETag header for conditional requests
    pub etag: Option<String>,
    /// Last-Modified header for conditional requests
    pub last_modified: Option<String>,
}

/// Cache statistics.
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub entries: u64,
}

/// Raw HTTP response captured for ALL statuses (Design 0005 H3 / ADR 0001 C3).
/// Unlike `fetch_once`, `fetch_raw_once` never early-returns before reading the
/// body, so the classifier can inspect 4xx/5xx error bodies.
#[derive(Debug, Clone)]
pub(crate) struct RawResponse {
    status: u16,
    headers: HeaderMap,
    final_url: String,
    content_type: String,
    body: String,
    etag: Option<String>,
    last_modified: Option<String>,
    response_time_ms: u64,
}

impl RawResponse {
    fn into_fetch_result(self, url: &str) -> FetchResult {
        let content_length = self.body.len();
        let is_spa = detect_spa(&self.body);
        FetchResult {
            url: url.to_string(),
            final_url: self.final_url,
            status: self.status,
            content_type: self.content_type,
            body: self.body,
            response_time_ms: self.response_time_ms,
            content_length,
            is_spa,
            from_cache: false,
            etag: self.etag,
            last_modified: self.last_modified,
        }
    }
}

/// A stored clearance cookie (Design 0005 §4.1). In-memory only this iteration.
///
/// Two reuse paths:
///   - SAFE (always): same-browser reuse — the cookie + minted UA stay inside the
///     browser fingerprint that earned it (no cross-fingerprint replay).
///   - GATED (Design 0005 §4.2 / §7 R-3, Design 0004 §1.7): cross-client replay
///     onto the reqwest client, attaching this cookie + the EXACT minted UA to a
///     plain reqwest GET. Built but gated behind `enable_clearance_replay`
///     (default OFF) because `cf_clearance` is widely reported bound to the
///     client TLS/JA3 + HTTP/2 fingerprint — a browser-minted cookie MAY be
///     rejected by the reqwest client. Until the operator-run
///     `clearance_replay_binding` spike confirms replayability, the honest
///     expectation is the cookie is fingerprint-bound and the fallback is
///     keep-session-in-browser. Single-IP satisfies the IP-binding automatically.
///
/// NEVER logged (cookie value is secret-equivalent).
#[derive(Debug, Clone)]
struct ClearanceEntry {
    name: String,
    value: String,
    /// The UA the cookie was minted under — MUST replay verbatim if reused.
    user_agent: String,
    minted_at: Instant,
    ttl: Duration,
}

/// HTTP fetcher with retry, headers, SPA detection, response caching, and browser fallback.
pub struct Fetcher {
    client: Client,
    max_retries: u32,
    backoff_base_ms: u64,
    /// LRU cache with TTL for HTTP responses
    cache: Cache<String, CachedResponse>,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    /// Headless browser pool for SPA/JS rendering fallback
    browser_pool: Arc<BrowserPool>,
    /// Whether to attempt browser fallback on SPA-detected pages
    enable_browser: bool,

    // -- Hybrid escalation controller (Design 0005). All consulted ONLY when
    //    `enable_escalation == true`; constructed-but-unused when off. --
    /// Master switch. false (default) => byte-for-byte legacy `fetch()`.
    enable_escalation: bool,
    /// Highest rung the controller may reach (clamped to <=4 unless solver on).
    max_escalation_rung: u8,
    /// Whether the R5 solver is enabled (gates the Captcha rung).
    enable_captcha_solver: bool,
    /// Per-run opt-in for the live R5 solve path (compliance gate 0001 C-1).
    /// SEPARATE from `enable_captcha_solver`; BOTH must be true to solve.
    captcha_run_opt_in: bool,
    /// R5 solver params (provider/env-name/timeout/cap), used to lazily build a
    /// live solver on the R5 path. Default-off: solver only constructed when both
    /// gates pass. The API key is NEVER stored here — only the env-var NAME.
    captcha_provider: Option<String>,
    captcha_api_key_env: Option<String>,
    captcha_timeout_secs: u64,
    /// Session-wide cost meter (compliance gate 0001 C-10 / K-2). Shared with the
    /// solver client so the pre-check and the paid call enforce ONE cap.
    cost_meter: CostMeter,
    /// The configured per-session USD cap (passed to `build_solver`'s gate).
    captcha_session_cost_cap_usd: f64,
    /// Conservative per-solve cost estimate for the C-10 pre-check (matches the
    /// solver client's own up-front reservation).
    solve_cost_estimate_usd: f64,
    /// Seeded classifier marker config.
    classifier_cfg: ClassifierConfig,
    /// Per-domain reputation governor + permanent denylist.
    governor: Arc<MinimalGovernor>,
    /// Browser render timeout for R4.
    browser_timeout: Duration,
    /// Cap on honored Retry-After (seconds) before we requeue rather than block.
    retry_after_ceiling_secs: u64,
    /// Per-domain clearance-cookie store (Design 0005 §4). In-memory only.
    clearance: DashMap<String, ClearanceEntry>,
    clearance_ttl: Duration,
    /// Cross-client clearance replay (Design 0005 §4.2 / R-3). Default false.
    /// When false, a minted clearance cookie is only reused on the SAFE
    /// same-browser path; it is NEVER attached to the reqwest client. When true,
    /// the unverified R4→R1 replay is attempted (cookie + minted UA on the
    /// reqwest GET) — gated because the cookie may be TLS/JA3-bound.
    enable_clearance_replay: bool,

    // -- R2/R3 alternative-surface + archive rungs (ADR 0003 §3.1 / §6.2). All
    //    off-safe; consulted only on the escalation path. --
    /// R3 Internet Archive CDX fallback enabled.
    enable_archive_fallback: bool,
    /// Prebuilt archive config (endpoint/timeout/age/UA) from `CrawlerConfig`.
    archive_cfg: crate::archive::ArchiveConfig,
    /// R2 alternative-surface (feeds) probe enabled.
    enable_alt_surface: bool,
    /// R2 sub-surface: RSS/Atom/JSON-Feed discovery + parse.
    src_feed: bool,
    /// Per-candidate timeout for the R2 feed probe (ms).
    alt_probe_timeout_ms: u64,
    /// Max total network probes the R2 rung may make per URL.
    max_alt_probes: usize,
}

/// Cached HTTP response with conditional request headers.
#[derive(Clone, Debug)]
struct CachedResponse {
    result: FetchResult,
    /// Stored for future conditional GET requests (If-None-Match)
    #[allow(dead_code)]
    etag: Option<String>,
    /// Stored for future conditional GET requests (If-Modified-Since)
    #[allow(dead_code)]
    last_modified: Option<String>,
}

impl Fetcher {
    pub fn new(config: &CrawlerConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.request_timeout_secs))
            .user_agent(&config.user_agent)
            .redirect(reqwest::redirect::Policy::limited(10))
            .cookie_store(true)
            .gzip(true)
            .brotli(true)
            .pool_max_idle_per_host(20)
            .tcp_nodelay(true)
            .tcp_keepalive(Duration::from_secs(30))
            .default_headers(Self::default_headers())
            .build()
            .map_err(|e| Error::Crawl {
                url: String::new(),
                reason: format!("client build: {e}"),
            })?;

        // LRU cache: max 2000 entries, 10 minute TTL
        let cache = Cache::builder()
            .max_capacity(2000)
            .time_to_live(Duration::from_secs(600))
            .build();

        // Thread stealth settings (R4 v1) in from config. When
        // `enable_stealth` is false this is identical to `BrowserPool::new()`
        // and the legacy non-stealth render path runs unchanged.
        let browser_pool = BrowserPool::with_stealth(
            config.enable_stealth,
            config.stealth_user_agent.as_deref(),
            config.stealth_locale.as_deref(),
        );

        // -- Hybrid escalation controller wiring (Design 0005 §3 Seam 2). All
        //    off-safe; only consulted when `enable_escalation` is true. --
        // Clamp the max rung to R4 when the solver is disabled (Design 0005 §9).
        let max_escalation_rung = if !config.enable_captcha_solver && config.max_escalation_rung >= 5
        {
            if config.enable_escalation {
                tracing::warn!(
                    requested = config.max_escalation_rung,
                    "enable_captcha_solver=false; clamping max_escalation_rung to 4 (Design 0005 §9)"
                );
            }
            4
        } else {
            config.max_escalation_rung
        };

        // Cross-client clearance replay is the UNVERIFIED path (Design 0005
        // §4.2 / R-3). Warn loudly when an operator turns it on so the
        // fingerprint-binding risk is explicit — until the gated spike confirms
        // replayability, the honest expectation is the cookie MAY be JA3/H2-bound
        // and the reqwest client will be rejected.
        if config.enable_escalation && config.enable_clearance_replay {
            tracing::warn!(
                "enable_clearance_replay=true: cross-client cf_clearance/datadome replay onto the \
                 reqwest client is UNVERIFIED (cookie may be TLS/JA3+HTTP2-bound — Design 0005 §4.2 \
                 / R-3). Run the `clearance_replay_binding` spike to confirm before relying on it."
            );
        }

        let classifier_cfg = ClassifierConfig::seeded(config.soft_block_min_bytes, 30);

        let governor = MinimalGovernor::new(GovernorConfig::from_parts(
            config.per_domain_request_budget,
            config.soft_breaker_fail_threshold,
            config.domain_breaker_cooldown_secs,
            config.pacing_jitter_ratio,
            config.permanent_denylist_path.clone(),
        ));

        Ok(Self {
            client,
            max_retries: config.max_retries,
            backoff_base_ms: config.backoff_base_ms,
            cache,
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            browser_pool,
            enable_browser: config.enable_browser,
            enable_escalation: config.enable_escalation,
            max_escalation_rung,
            enable_captcha_solver: config.enable_captcha_solver,
            captcha_run_opt_in: config.captcha_run_opt_in,
            captcha_provider: config.captcha_provider.clone(),
            captcha_api_key_env: config.captcha_api_key_env.clone(),
            captcha_timeout_secs: config.captcha_timeout_secs,
            cost_meter: CostMeter::new(config.captcha_session_cost_cap_usd),
            captcha_session_cost_cap_usd: config.captcha_session_cost_cap_usd,
            solve_cost_estimate_usd: 0.003,
            classifier_cfg,
            governor,
            browser_timeout: Duration::from_secs(config.request_timeout_secs.max(5)),
            retry_after_ceiling_secs: 120,
            clearance: DashMap::new(),
            clearance_ttl: Duration::from_secs(1200),
            enable_clearance_replay: config.enable_clearance_replay,
            enable_archive_fallback: config.enable_archive_fallback,
            archive_cfg: crate::archive::ArchiveConfig {
                cdx_endpoint: config.archive_cdx_endpoint.clone(),
                timeout_ms: config.archive_timeout_ms,
                max_snapshot_age_days: config.archive_max_snapshot_age_days,
                // Fall back to the crawler's UA when no dedicated archive UA is set.
                user_agent: config
                    .archive_user_agent
                    .clone()
                    .or_else(|| Some(config.user_agent.clone())),
            },
            enable_alt_surface: config.enable_alt_surface,
            src_feed: config.src_feed,
            // Per-candidate feed-probe timeout: the request timeout, in ms.
            alt_probe_timeout_ms: config.request_timeout_secs.saturating_mul(1000),
            max_alt_probes: config.max_alt_probes as usize,
        })
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            hits: self.cache_hits.load(Ordering::Relaxed),
            misses: self.cache_misses.load(Ordering::Relaxed),
            entries: self.cache.entry_count(),
        }
    }

    /// Fetch a URL with caching, conditional requests, retries, and backoff.
    pub async fn fetch(&self, url: &str) -> Result<FetchResult> {
        // Check cache first
        if let Some(cached) = self.cache.get(url).await {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(url, "Cache hit");
            let mut result = cached.result.clone();
            result.from_cache = true;
            return Ok(result);
        }
        self.cache_misses.fetch_add(1, Ordering::Relaxed);

        // Hybrid escalation controller (Design 0005 §3 Seam 2). When the flag is
        // OFF (default) we fall through to the EXACT legacy retry loop below —
        // byte-for-byte today's behavior. When ON, the controller owns rung
        // selection, governor consultation, and its own cache insert.
        if self.enable_escalation {
            return self.escalate(url).await;
        }

        let mut last_err = None;

        for attempt in 0..=self.max_retries {
            if attempt > 0 {
                let delay = self.backoff_base_ms * 2u64.pow(attempt - 1);
                tracing::debug!(url, attempt, delay_ms = delay, "Retrying");
                tokio::time::sleep(Duration::from_millis(delay)).await;
            }

            match self.fetch_once(url).await {
                Ok(result) => {
                    // If SPA detected and browser enabled, try browser fallback
                    let result = if result.is_spa && self.enable_browser {
                        tracing::info!(url, "SPA detected, attempting browser render");
                        self.try_browser_fallback(url, result).await
                    } else {
                        result
                    };

                    // Cache the response
                    let cached = CachedResponse {
                        etag: result.etag.clone(),
                        last_modified: result.last_modified.clone(),
                        result: result.clone(),
                    };
                    self.cache.insert(url.to_string(), cached).await;
                    return Ok(result);
                }
                Err(e) => {
                    // Don't retry 4xx (except 429)
                    if let Error::Blocked { status, .. } = &e {
                        if *status != 429 && *status < 500 {
                            return Err(e);
                        }
                    }
                    last_err = Some(e);
                }
            }
        }

        Err(last_err.unwrap_or_else(|| Error::Crawl {
            url: url.to_string(),
            reason: "max retries exceeded".into(),
        }))
    }

    async fn fetch_once(&self, url: &str) -> Result<FetchResult> {
        let start = Instant::now();

        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| Error::Http(e.to_string()))?;

        let status = response.status();
        let final_url = response.url().to_string();
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("text/html")
            .to_string();

        // Handle error status codes
        if status == StatusCode::TOO_MANY_REQUESTS {
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(30);

            let domain = url::Url::parse(url)
                .map(|u| u.host_str().unwrap_or("unknown").to_string())
                .unwrap_or_else(|_| "unknown".to_string());

            return Err(Error::RateLimited {
                domain,
                retry_after_secs: retry_after,
            });
        }

        if status.is_client_error() || status.is_server_error() {
            return Err(Error::Blocked {
                url: url.to_string(),
                status: status.as_u16(),
            });
        }

        // Capture conditional request headers for future cache validation
        let etag = response
            .headers()
            .get("etag")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        let last_modified = response
            .headers()
            .get("last-modified")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        let body = response
            .text()
            .await
            .map_err(|e| Error::Http(e.to_string()))?;

        let elapsed = start.elapsed().as_millis() as u64;
        let content_length = body.len();
        let is_spa = detect_spa(&body);

        Ok(FetchResult {
            url: url.to_string(),
            final_url,
            status: status.as_u16(),
            content_type,
            body,
            response_time_ms: elapsed,
            content_length,
            is_spa,
            from_cache: false,
            etag,
            last_modified,
        })
    }

    /// Capture `(status, headers, body)` for ALL statuses (Design 0005 H3 / ADR
    /// 0001 C3). Added ALONGSIDE the untouched `fetch_once` (expand-contract):
    /// this never early-returns on 4xx/5xx, so the classifier can inspect error
    /// bodies. Only used on the escalation path.
    async fn fetch_raw_once(&self, url: &str) -> Result<RawResponse> {
        let start = Instant::now();

        let mut req = self.client.get(url);

        // Cross-client clearance replay (Design 0005 §4.2 / R-3 — GATED). Only
        // when `enable_clearance_replay` is on AND a live, not-expired clearance
        // entry exists for this domain do we attach the minted cookie + replay
        // the EXACT minted UA on the reqwest client. With the flag off (default)
        // this whole block is skipped and the clearance store is reused only on
        // the SAFE same-browser path — current behavior unchanged.
        //
        // The cookie is widely reported TLS/JA3+HTTP2-bound, so this may be
        // rejected by the reqwest client (different fingerprint). We log only the
        // domain + that a replay was ATTEMPTED — never the cookie value or UA
        // (no secrets/tokens in logs).
        if self.enable_clearance_replay {
            let domain = registrable_domain(url);
            if let Some(entry) = self.live_clearance(&domain) {
                if let Some(deco) = clearance_request_decoration(&entry) {
                    req = req
                        .header(reqwest::header::COOKIE, deco.cookie_header)
                        // Replay the minted UA VERBATIM — a UA mismatch
                        // invalidates the cookie even when the cookie itself is
                        // accepted (Design 0004 §1.6).
                        .header(reqwest::header::USER_AGENT, deco.user_agent);
                    tracing::debug!(
                        domain = %domain,
                        replay_attempted = true,
                        "clearance replay: attaching minted clearance cookie + UA to reqwest GET"
                    );
                }
            }
        }

        let response = req
            .send()
            .await
            .map_err(|e| Error::Http(e.to_string()))?;

        let status = response.status().as_u16();
        let final_url = response.url().to_string();
        let headers = response.headers().clone();
        let content_type = headers
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("text/html")
            .to_string();
        let etag = headers.get("etag").and_then(|v| v.to_str().ok()).map(|s| s.to_string());
        let last_modified = headers
            .get("last-modified")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        // Read the body for EVERY status (the key difference from fetch_once).
        let body = response.text().await.map_err(|e| Error::Http(e.to_string()))?;
        let response_time_ms = start.elapsed().as_millis() as u64;

        Ok(RawResponse {
            status,
            headers,
            final_url,
            content_type,
            body,
            etag,
            last_modified,
            response_time_ms,
        })
    }

    /// The escalation controller (Design 0005 §2). Called from `fetch()` only
    /// when `enable_escalation == true`. Routes the classifier verdict to the
    /// rungs that exist today (R0/R1/R4/R5), consults the governor before every
    /// live rung, caches only `RealContent`, and returns the most-informative
    /// typed error on give-up.
    async fn escalate(&self, url: &str) -> Result<FetchResult> {
        // Compliance gate 0001 G-1: compute the authenticated-context flag ONCE
        // from the URL userinfo + the OUTGOING request headers (the default
        // header set the reqwest client would send). If the request carries auth
        // material, R5 is refused downstream (never solve an auth-wall CAPTCHA).
        let authenticated_context = is_authenticated_context(url, &Self::default_headers());
        let meter_handle = self.cost_meter.handle();

        let result = run_escalation(self, &EscalationParams {
            url,
            classifier_cfg: &self.classifier_cfg,
            governor: self.governor.as_ref(),
            enable_browser: self.enable_browser,
            enable_captcha_solver: self.enable_captcha_solver,
            max_rung: self.max_escalation_rung,
            enable_archive_fallback: self.enable_archive_fallback,
            enable_alt_surface: self.enable_alt_surface,
            retry_after_ceiling_secs: self.retry_after_ceiling_secs,
            captcha_run_opt_in: self.captcha_run_opt_in,
            authenticated_context,
            cost_meter: Some(&meter_handle),
            solve_cost_estimate_usd: self.solve_cost_estimate_usd,
        })
        .await?;

        // Cache insert gated on RealContent only (Design 0005 H5 / ADR 0001 C6).
        // `run_escalation` only ever returns Ok on RealContent (incl. R0-salvage,
        // R4-render, R5-verify), so any Ok here is cacheable.
        let cached = CachedResponse {
            etag: result.etag.clone(),
            last_modified: result.last_modified.clone(),
            result: result.clone(),
        };
        self.cache.insert(url.to_string(), cached).await;
        Ok(result)
    }

    /// Store clearance cookies minted by R4 (Design 0005 §4.1). The cookie is
    /// stamped with the EXACT minted UA (surfaced by `BrowserFetchResult`) so any
    /// later reuse can replay it verbatim. NEVER logged (cookie value is a
    /// secret-equivalent token).
    fn store_clearance_cookies(&self, domain: &str, cookies: &[(String, String)], ua: &str) {
        for (name, value) in cookies {
            let n = name.to_ascii_lowercase();
            if n == "cf_clearance" || n == "datadome" {
                self.clearance.insert(
                    domain.to_string(),
                    ClearanceEntry {
                        name: name.clone(),
                        value: value.clone(),
                        user_agent: ua.to_string(),
                        minted_at: Instant::now(),
                        ttl: self.clearance_ttl,
                    },
                );
            }
        }
    }

    /// Return the live (not-expired) clearance entry for a domain, or `None` if
    /// absent or past TTL. Pure read; the expired-entry sweep is lazy (a stale
    /// entry is simply ignored here and overwritten on the next mint).
    fn live_clearance(&self, domain: &str) -> Option<ClearanceEntry> {
        let entry = self.clearance.get(domain)?;
        if entry.minted_at.elapsed() >= entry.ttl {
            return None;
        }
        Some(entry.clone())
    }

    /// Try browser fallback when SPA is detected.
    /// Replaces the HTTP result body with browser-rendered HTML.
    async fn try_browser_fallback(&self, url: &str, http_result: FetchResult) -> FetchResult {
        match self.browser_pool.fetch(url, Duration::from_secs(15)).await {
            Some(browser_result) if browser_result.body.len() > http_result.body.len() / 2 => {
                tracing::info!(
                    url,
                    http_len = http_result.body.len(),
                    browser_len = browser_result.body.len(),
                    "Browser rendered more content, using browser result"
                );
                FetchResult {
                    url: http_result.url,
                    final_url: browser_result.final_url,
                    status: http_result.status,
                    content_type: http_result.content_type,
                    body: browser_result.body,
                    response_time_ms: http_result.response_time_ms,
                    content_length: http_result.content_length,
                    is_spa: false, // rendered, no longer SPA
                    from_cache: false,
                    etag: http_result.etag,
                    last_modified: http_result.last_modified,
                }
            }
            _ => {
                tracing::debug!(url, "Browser fallback didn't improve content, keeping HTTP result");
                http_result
            }
        }
    }

    /// Explicitly fetch a URL via browser (for search engines that need JS).
    /// Used by crawler when search parser returns 0 results.
    pub async fn fetch_via_browser(&self, url: &str) -> Option<FetchResult> {
        if !self.enable_browser {
            return None;
        }
        let result = self.browser_pool.fetch(url, Duration::from_secs(20)).await?;
        Some(FetchResult {
            url: url.to_string(),
            final_url: result.final_url,
            status: result.status,
            content_type: "text/html".to_string(),
            body: result.body,
            response_time_ms: 0,
            content_length: 0,
            is_spa: false,
            from_cache: false,
            etag: None,
            last_modified: None,
        })
    }

    fn default_headers() -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8".parse().unwrap());
        headers.insert("accept-language", "en-US,en;q=0.9".parse().unwrap());
        headers.insert("accept-encoding", "gzip, deflate, br".parse().unwrap());
        headers.insert("cache-control", "no-cache".parse().unwrap());
        headers
    }
}

// ── Escalation controller (Design 0005 §2) ──────────────────────────────────
//
// The routing logic is a free async function over a `RungIo` trait so it is
// UNIT-TESTABLE with mocked rungs (Design 0005 §8.2) — no live network, browser,
// or solver. `Fetcher` provides the live `RungIo` impl; tests provide a fake.

/// Outcome of an R4/R5 rung attempt: the rendered/submitted body + headers so
/// the controller can re-classify it.
pub(crate) struct RungOutcome {
    pub final_url: String,
    pub body: String,
    pub status: u16,
    /// Clearance cookies captured (R4 only); empty otherwise. Stored by the live
    /// impl via `store_clearance_cookies`; retained on the struct for the gated
    /// same-browser reuse path (Design 0005 §4) which is not yet wired.
    #[allow(dead_code)]
    pub cookies: Vec<(String, String)>,
    /// The UA the rung ran under (for clearance-cookie binding). Retained for the
    /// gated reuse path (replay-verbatim rule, Design 0005 §4.2).
    #[allow(dead_code)]
    pub user_agent: String,
}

/// The I/O operations the controller needs, abstracted for testability.
#[allow(async_fn_in_trait)]
pub(crate) trait RungIo {
    /// R1: a single coherent live GET capturing all statuses.
    async fn raw_get(&self, url: &str) -> Result<RawResponse>;
    /// R4: stealth/browser render. `None` if browser unavailable / produced nothing.
    async fn browser_render(&self, url: &str) -> Option<RungOutcome>;
    /// R5: extract sitekey from `body`, solve, inject, re-fetch. `None` when the
    /// solver is unavailable; `Some(Err)` on a genuine solve/submit failure.
    async fn solve_captcha(&self, url: &str, body: &str) -> Option<Result<RungOutcome>>;
    /// R0: pure, zero-request in-band hydration salvage over a body in hand.
    fn hydration_salvage(&self, url: &str, raw: &RawResponse) -> Option<FetchResult>;
    /// R3: fetch the newest Internet Archive snapshot of `url`. Zero-ban-risk —
    /// queries `web.archive.org`, NEVER the live target origin. `None` when the
    /// archive rung is disabled or no usable snapshot exists. Non-fatal.
    async fn archive_fetch(&self, url: &str) -> Option<RungOutcome>;
    /// R2: alternative-surface probe (RSS/Atom/JSON-Feed) for `url`, using the
    /// already-fetched page `body` for feed autodiscovery. Hits the SAME origin
    /// (well-known feed paths) so the controller governs/paces it. `None` when the
    /// rung is disabled or no feed item recovers the page. Non-fatal.
    async fn alt_surface_probe(&self, url: &str, body: &str) -> Option<RungOutcome>;
    /// Sleep helper (mocked in tests so they don't actually wait).
    async fn sleep(&self, dur: Duration);
    /// Record-keeping hook so tests can assert call counts.
    fn note(&self, _what: &str) {}
}

pub(crate) struct EscalationParams<'a> {
    pub url: &'a str,
    pub classifier_cfg: &'a ClassifierConfig,
    pub governor: &'a MinimalGovernor,
    pub enable_browser: bool,
    pub enable_captcha_solver: bool,
    pub max_rung: u8,
    /// R3 Internet Archive CDX fallback enabled (ADR 0003 §6.2). Off-safe.
    pub enable_archive_fallback: bool,
    /// R2 alternative-surface (feeds) probe enabled (ADR 0003 §6.2). Off-safe.
    pub enable_alt_surface: bool,
    pub retry_after_ceiling_secs: u64,
    /// Compliance gate 0001 C-1: per-run opt-in, SEPARATE from
    /// `enable_captcha_solver`. Both must be true for any live solve.
    pub captcha_run_opt_in: bool,
    /// Compliance gate 0001 G-1: whether the outgoing request carries auth
    /// material (computed once at controller entry from the URL + outgoing
    /// headers via [`is_authenticated_context`]). When true, R5 is REFUSED.
    pub authenticated_context: bool,
    /// Compliance gate 0001 C-10: per-URL/session cost pre-check. The meter is
    /// reserved BEFORE the paid solve; if it would exceed the cap the solve is
    /// refused. `None` when no solver is wired (tests/off path) — then the gate
    /// treats cost as "would exceed" only if a solve is actually attempted with a
    /// solver present; with `None` the controller never reaches the paid call.
    pub cost_meter: Option<&'a CostMeterHandle>,
    /// Conservative per-solve estimate reserved against the cap (C-10). Matches
    /// the solver client's own up-front reservation so the pre-check is coherent.
    pub solve_cost_estimate_usd: f64,
}

fn registrable_domain(url: &str) -> String {
    url::Url::parse(url)
        .ok()
        .and_then(|u| u.host_str().map(|h| h.to_string()))
        .unwrap_or_else(|| "unknown".to_string())
}

/// Compliance gate 0001 **G-1 (Critical) — public + unauthenticated only**.
///
/// Returns `true` when the request/target carries ANY authentication material,
/// in which case R5 MUST be refused (never solve a CAPTCHA gating a login/auth
/// wall — gate 0001 §4 permanent prohibition). The checks are:
///
///   1. **URL userinfo** — `https://user:pass@host/…` (basic-auth in the URL).
///   2. **`Authorization` header** present (any scheme: Bearer/Basic/…).
///   3. **`Cookie` header** carrying a session/auth-looking cookie — name
///      contains `session`/`sess`/`auth`/`token`/`sid`/`login`/`jwt`, or any of
///      the common framework session cookie names.
///
/// HONEST RESIDUAL (documented, not overclaimed): cookie-based auth detection is
/// a NAME heuristic — a site may use an unguessable session-cookie name we do not
/// match, and a benign cookie may share one of these substrings (a false-refuse,
/// which fails SAFE — we decline to solve). This is the strongest feasible
/// in-code check without parsing every site's auth scheme; the residual is a
/// missed-detection on exotic cookie names, which the operator-level
/// public-target precondition (gate 0001 C-1/C-4) must still cover. We err toward
/// refusing (no solve) on any signal.
pub(crate) fn is_authenticated_context(url: &str, headers: &HeaderMap) -> bool {
    // 1. userinfo in the URL (basic-auth credentials).
    if let Ok(u) = url::Url::parse(url) {
        if !u.username().is_empty() || u.password().is_some() {
            return true;
        }
    }
    // 2. Authorization header (any scheme).
    if headers.contains_key(reqwest::header::AUTHORIZATION) {
        return true;
    }
    // 3. Session/auth-looking cookie.
    if let Some(cookie) = headers.get(reqwest::header::COOKIE).and_then(|v| v.to_str().ok()) {
        if cookie_looks_authenticated(cookie) {
            return true;
        }
    }
    false
}

/// Whether a `Cookie:` header value carries a session/auth-looking cookie. Pure;
/// case-insensitive substring match on the cookie NAMES only (never logs values).
fn cookie_looks_authenticated(cookie_header: &str) -> bool {
    const NAME_SIGNALS: &[&str] =
        &["session", "sess", "auth", "token", "sid", "login", "jwt", "csrf_token"];
    const EXACT_NAMES: &[&str] = &[
        "phpsessid",
        "jsessionid",
        "asp.net_sessionid",
        "connect.sid",
        "_session_id",
        "laravel_session",
        "wordpress_logged_in",
    ];
    for pair in cookie_header.split(';') {
        let name = pair.split('=').next().unwrap_or("").trim().to_ascii_lowercase();
        if name.is_empty() {
            continue;
        }
        if EXACT_NAMES.contains(&name.as_str()) {
            return true;
        }
        if NAME_SIGNALS.iter().any(|s| name.contains(s)) {
            return true;
        }
    }
    false
}

/// Compliance gate 0001 **G-2 (High) — structured per-solve audit event**.
///
/// Emits EXACTLY ONE structured `tracing` event per R5 solve attempt, on the
/// dedicated `audit` target so it can be routed/retained separately. Carries ONLY
/// the gate-0001 C-5 field set: `{domain, captcha_kind, provider, outcome,
/// cost_usd, cumulative_cost_usd, timestamp}`. It NEVER carries the token, API
/// key, cookie, query string, page body, or any PII (api-security NO SECRETS IN
/// LOGS; logging AUDIT TRAIL). `domain` is the registrable host (no query
/// string). `outcome` is one of: solved | failed | refused.
pub(crate) fn emit_solve_audit(
    domain: &str,
    captcha_kind: &str,
    provider: &str,
    outcome: &str,
    cost_usd: f64,
    cumulative_cost_usd: f64,
) {
    tracing::info!(
        target: "audit",
        domain = domain,
        captcha_kind = captcha_kind,
        provider = provider,
        outcome = outcome,
        cost_usd = cost_usd,
        cumulative_cost_usd = cumulative_cost_usd,
        timestamp = now_unix_secs(),
        "captcha solve audit"
    );
}

fn now_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The headers to attach to a reqwest GET when replaying a clearance cookie
/// cross-client (Design 0005 §4.2). Pure data so the decoration logic is
/// unit-testable without a live client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ClearanceDecoration {
    /// The `Cookie:` header value, e.g. `cf_clearance=<value>`.
    pub cookie_header: String,
    /// The minted UA to replay verbatim as the `User-Agent` header.
    pub user_agent: String,
}

/// Build the cross-client replay decoration from a stored clearance entry
/// (Design 0005 §4.2). Returns `None` when the entry has no cookie value or no
/// minted UA to replay (a UA is REQUIRED — replaying the cookie without the
/// exact minting UA invalidates it, Design 0004 §1.6). Pure; no I/O, no logging.
fn clearance_request_decoration(entry: &ClearanceEntry) -> Option<ClearanceDecoration> {
    if entry.value.is_empty() || entry.user_agent.is_empty() {
        return None;
    }
    Some(ClearanceDecoration {
        cookie_header: format!("{}={}", entry.name, entry.value),
        user_agent: entry.user_agent.clone(),
    })
}

fn vendor_label(v: ChallengeVendor) -> &'static str {
    match v {
        ChallengeVendor::Cloudflare => "cloudflare",
        ChallengeVendor::DataDome => "datadome",
        ChallengeVendor::Akamai => "akamai",
        ChallengeVendor::Unknown => "unknown",
    }
}

/// Turn an R2/R3 [`RungOutcome`] into a `FetchResult` (mirrors the R4 path's
/// RawResponse construction). The recovered body is treated as `text/html`.
fn rung_outcome_to_result(out: RungOutcome, url: &str) -> FetchResult {
    RawResponse {
        status: out.status,
        headers: HeaderMap::new(),
        final_url: out.final_url,
        content_type: "text/html".into(),
        body: out.body,
        etag: None,
        last_modified: None,
        response_time_ms: 0,
    }
    .into_fetch_result(url)
}

/// The classifier-routed escalation flow (Design 0005 §2.2 / §2.4 table).
pub(crate) async fn run_escalation<E: RungIo>(
    io: &E,
    p: &EscalationParams<'_>,
) -> Result<FetchResult> {
    let url = p.url;
    let domain = registrable_domain(url);

    // ---- governor gate: permanently-denied domains never touch the origin ----
    if p.governor.is_permanently_denied(&domain) {
        return Err(Error::PermanentlyDenied { domain, reason: "hard-ban".into() });
    }

    // ---- R1: governor-paced coherent live GET ----
    match p.governor.admit(&domain) {
        Admission::DeniedPermanent => {
            return Err(Error::PermanentlyDenied { domain, reason: "hard-ban".into() });
        }
        Admission::SkipLive(reason) => {
            tracing::info!(url, domain = %domain, ?reason, "escalation: skipping live rung");
            return Err(Error::Blocked { url: url.to_string(), status: 0 });
        }
        Admission::Proceed => {}
    }
    io.sleep(p.governor.pace_delay(&domain)).await;

    let raw = io.raw_get(url).await?;
    let verdict = classifier::classify(raw.status, &raw.headers, &raw.final_url, &raw.body, p.classifier_cfg);
    p.governor.record(&domain, &verdict.class, Rung::R1, false);
    emit_verdict(url, &domain, "R1", &verdict);

    match &verdict.class {
        BlockClass::RealContent => {
            let mut result = raw.clone().into_fetch_result(url);
            // SPA render via R4 (NON-challenge semantics) — existing SPA path.
            if result.is_spa && p.enable_browser && p.max_rung >= 4 {
                if let Admission::Proceed = p.governor.admit(&domain) {
                    io.sleep(p.governor.pace_delay(&domain)).await;
                    if let Some(out) = io.browser_render(url).await {
                        // Keep the larger body (mirror legacy try_browser_fallback).
                        if out.body.len() > result.body.len() / 2 {
                            result.final_url = out.final_url;
                            result.body = out.body;
                            result.content_length = result.body.len();
                            result.is_spa = false;
                        }
                    }
                }
            }
            // R0 enrichment on the good body (additive, non-fatal).
            if let Some(salvaged) = io.hydration_salvage(url, &raw) {
                if salvaged.body.chars().count() > result.body.chars().count() {
                    result.body = salvaged.body;
                    result.content_length = result.body.len();
                }
            }
            Ok(result)
        }

        BlockClass::SoftBlock | BlockClass::Cloudflare403 => {
            // Reputation-first recovery ladder (ADR 0003 §3.1): spend the cheapest,
            // lowest-ban-risk surfaces before giving up.
            //   R0 in-band salvage (free, body in hand)
            //   → R3 archive (zero-ban: never touches the origin)
            //   → R2 alternative-surface feeds (same origin, governor-gated)
            //   → give up.

            // R0 salvage on the (soft-blocked) body — might still carry content.
            if let Some(salvaged) = io.hydration_salvage(url, &raw) {
                tracing::info!(url, domain = %domain, "escalation: R0 hydration salvage recovered content");
                return Ok(salvaged);
            }

            // R3 archive fallback — IP-safe (web.archive.org only). No governor
            // consultation: it does not touch the target origin's reputation.
            if p.enable_archive_fallback {
                if let Some(out) = io.archive_fetch(url).await {
                    let v = classifier::classify(
                        out.status, &HeaderMap::new(), &out.final_url, &out.body, p.classifier_cfg,
                    );
                    emit_verdict(url, &domain, "R3", &v);
                    if matches!(v.class, BlockClass::RealContent) {
                        tracing::info!(url, domain = %domain, "escalation: R3 archive snapshot recovered content");
                        return Ok(rung_outcome_to_result(out, url));
                    }
                }
            }

            // R2 alternative-surface (feeds) — same origin, so governor-gated and
            // paced like any live-origin rung (ASR-6 reputation budget).
            if p.enable_alt_surface {
                if let Admission::Proceed = p.governor.admit(&domain) {
                    io.sleep(p.governor.pace_delay(&domain)).await;
                    if let Some(out) = io.alt_surface_probe(url, &raw.body).await {
                        let v = classifier::classify(
                            out.status, &HeaderMap::new(), &out.final_url, &out.body, p.classifier_cfg,
                        );
                        p.governor.record(&domain, &v.class, Rung::R2, false);
                        emit_verdict(url, &domain, "R2", &v);
                        if matches!(v.class, BlockClass::RealContent) {
                            tracing::info!(url, domain = %domain, "escalation: R2 alt-surface recovered content");
                            return Ok(rung_outcome_to_result(out, url));
                        }
                    }
                }
            }

            give_up(io, p.governor, &domain, &verdict, url, false)
        }

        BlockClass::JsChallenge { vendor } => {
            let vendor = *vendor;
            if !p.enable_browser || p.max_rung < 4 {
                return give_up(io, p.governor, &domain, &verdict, url, false);
            }
            match p.governor.admit(&domain) {
                Admission::Proceed => {}
                _ => return give_up(io, p.governor, &domain, &verdict, url, false),
            }
            io.sleep(p.governor.pace_delay(&domain)).await;
            let r4 = io.browser_render(url).await;
            if let Some(out) = r4 {
                let v2 = classifier::classify(200, &HeaderMap::new(), &out.final_url, &out.body, p.classifier_cfg);
                p.governor.record(&domain, &v2.class, Rung::R4, false);
                emit_verdict(url, &domain, "R4", &v2);
                match &v2.class {
                    BlockClass::RealContent => {
                        let res = RawResponse {
                            status: out.status,
                            headers: HeaderMap::new(),
                            final_url: out.final_url,
                            content_type: "text/html".into(),
                            body: out.body,
                            etag: None,
                            last_modified: None,
                            response_time_ms: 0,
                        };
                        return Ok(res.into_fetch_result(url));
                    }
                    BlockClass::Captcha { kind } => {
                        // R4 surfaced a CAPTCHA → fall through to R5 on this body.
                        let kind_label = captcha_kind_label(*kind);
                        return run_r5(io, p, &domain, url, &out.body, vendor, kind_label).await;
                    }
                    _ => {
                        // R4 ran a real browser and STILL blocked => hard-ban
                        // candidate (Design 0005 §5 / ADR 0003 §4.2).
                        return give_up(io, p.governor, &domain, &v2, url, true);
                    }
                }
            }
            // R4 produced nothing — try R5 on the original challenge body, else give up.
            run_r5(io, p, &domain, url, &raw.body, vendor, "unknown").await
        }

        BlockClass::Captcha { kind } => {
            let kind_label = captcha_kind_label(*kind);
            run_r5(io, p, &domain, url, &raw.body, ChallengeVendor::Cloudflare, kind_label).await
        }

        BlockClass::RateLimited { retry_after_secs } => {
            let wait = (*retry_after_secs).min(p.retry_after_ceiling_secs);
            io.sleep(Duration::from_secs(wait)).await;
            // record already applied the multiplicative decrease; caller retries
            // via the frontier. Give up this attempt with the rate-limit error.
            Err(Error::RateLimited { domain, retry_after_secs: *retry_after_secs })
        }
    }
}

/// Human label for a classifier CAPTCHA kind (audit field only — no secrets).
fn captcha_kind_label(kind: crate::classifier::CaptchaKind) -> &'static str {
    use crate::classifier::CaptchaKind as K;
    match kind {
        K::Recaptcha => "recaptcha",
        K::Hcaptcha => "hcaptcha",
        K::Turnstile => "turnstile",
    }
}

/// Record an R5 give-up to the governor's PERMANENT denylist (compliance gate
/// 0001 **G-3 — record-on-give-up**). A domain on which R5 failed/refused is
/// written so R5 can NEVER re-attempt it (mirrors the never-rotate-back-in rule).
/// The `admit`/`is_permanently_denied` check at controller entry then short-
/// circuits any later R5 entry for the domain. `reason` is a fixed label, never a
/// secret. Idempotent (the governor dedups).
fn record_r5_give_up(io: &impl RungIo, governor: &MinimalGovernor, domain: &str, reason: &str) {
    governor.record_hard_ban(domain, reason, "R5-give-up");
    io.note("r5-give-up-denylist");
}

/// Provider name for the audit event, derived from gate state WITHOUT
/// constructing a solver or reading secrets. When the live solver is wired this
/// reflects the configured provider; for the gated/off path it is `"none"`.
/// (The live `solve_captcha` impl carries the real provider; the controller only
/// needs a label for the refused/disabled audit lines.)
fn audit_provider(p: &EscalationParams<'_>) -> &'static str {
    if p.enable_captcha_solver && p.captcha_run_opt_in {
        "configured"
    } else {
        "none"
    }
}

/// R5 solver rung (Design 0005 §2.2 R5 / H8) + compliance gate 0001 conditions.
///
/// Gate composition (ALL must hold to reach the paid solve — default-deny):
///   - **C-1** `enable_captcha_solver` AND `captcha_run_opt_in` (per-run opt-in,
///     separate from the config flag).
///   - **K-4/H8** `max_rung >= 5`.
///   - **G-1** NOT `authenticated_context` (refuse auth-wall CAPTCHAs).
///   - **C-10** `CostMeter::try_reserve(estimate)` succeeds (per-URL/session cost
///     pre-check) BEFORE the paid call.
///   - governor admits (not denied/skip).
///
/// On any refusal or failure: emit the G-2 audit event (`outcome=refused|failed`)
/// and record the domain to the permanent denylist (**G-3**), then return a typed
/// give-up. **C-7**: exactly one solve attempt — no loop.
///
/// **G-4 (no raw PII retention):** this rung returns only the transformed
/// `FetchResult` (which the extractor pipeline turns into derivatives downstream).
/// It performs NO raw-body cache insert on the R5 path — caching is gated to a
/// final `RealContent` verdict in `Fetcher::escalate`, and a solved-gated raw body
/// is never written to any durable store here.
#[allow(clippy::too_many_arguments)]
async fn run_r5<E: RungIo>(
    io: &E,
    p: &EscalationParams<'_>,
    domain: &str,
    url: &str,
    body: &str,
    vendor: ChallengeVendor,
    kind_label: &str,
) -> Result<FetchResult> {
    let provider = audit_provider(p);
    let cumulative = p.cost_meter.map(|m| m.spent_usd()).unwrap_or(0.0);

    let challenge_unsolved = || Error::ChallengeUnsolved {
        url: url.to_string(),
        vendor: vendor_label(vendor).to_string(),
    };

    // ── C-1 + H8: BOTH the config flag AND the per-run opt-in, plus rung budget.
    // Default-deny. A disabled/opt-out state is an honest give-up — NO denylist
    // write (we never attempted a solve; the domain may be solvable later once
    // the operator opts in).
    if !p.enable_captcha_solver || !p.captcha_run_opt_in || p.max_rung < 5 {
        return Err(challenge_unsolved());
    }

    // ── G-1 (Critical): public + unauthenticated only. Refuse auth-wall CAPTCHAs.
    // This is a hard prohibition (gate 0001 §4) — audit + denylist so we never
    // re-attempt this domain, then refuse.
    if p.authenticated_context {
        emit_solve_audit(domain, kind_label, provider, "refused", 0.0, cumulative);
        record_r5_give_up(io, p.governor, domain, "auth-context-refused");
        tracing::warn!(domain, "R5 REFUSED: authenticated context (gate 0001 G-1 / §4)");
        return Err(challenge_unsolved());
    }

    // ── governor admission (denylist already checked at controller entry; this
    // also enforces budget/breaker before a paid rung).
    match p.governor.admit(domain) {
        Admission::Proceed => {}
        Admission::DeniedPermanent => {
            return Err(Error::PermanentlyDenied {
                domain: domain.to_string(),
                reason: "hard-ban".into(),
            });
        }
        Admission::SkipLive(_) => {
            emit_solve_audit(domain, kind_label, provider, "refused", 0.0, cumulative);
            return Err(challenge_unsolved());
        }
    }

    // ── C-10: per-URL/session cost PRE-CHECK before the paid solve call. Reserve
    // the estimate; if it would exceed the cap, refuse + audit outcome=refused
    // and record the give-up (G-3). The reservation is consumed by the solve
    // below (the solver client does NOT double-reserve on the live path — it
    // reserves its own estimate; to avoid a double-charge we refund here and let
    // the client reserve). We only use this as the pre-check guard.
    if let Some(meter) = p.cost_meter {
        match meter.try_reserve(p.solve_cost_estimate_usd) {
            Ok(()) => {
                // Pre-check passed; refund immediately so the solver client's own
                // up-front reservation is the single source of truth for spend
                // (no double counting). The cap is still enforced because the
                // client reserves the same estimate before its paid call.
                meter.refund(p.solve_cost_estimate_usd);
            }
            Err(_) => {
                let cumulative = meter.spent_usd();
                emit_solve_audit(domain, kind_label, provider, "refused", 0.0, cumulative);
                record_r5_give_up(io, p.governor, domain, "cost-cap-refused");
                tracing::warn!(domain, "R5 REFUSED: per-URL cost pre-check would exceed cap (C-10)");
                return Err(challenge_unsolved());
            }
        }
    }

    io.sleep(p.governor.pace_delay(domain)).await;

    // ── C-7: SINGLE bounded attempt. No retry loop.
    match io.solve_captcha(url, body).await {
        Some(Ok(out)) => {
            let v3 = classifier::classify(out.status, &HeaderMap::new(), &out.final_url, &out.body, p.classifier_cfg);
            p.governor.record(domain, &v3.class, Rung::R5, false);
            emit_verdict(url, domain, "R5", &v3);
            let spent = p.cost_meter.map(|m| m.spent_usd()).unwrap_or(0.0);
            if v3.class == BlockClass::RealContent {
                emit_solve_audit(domain, kind_label, provider, "solved", p.solve_cost_estimate_usd, spent);
                let res = RawResponse {
                    status: out.status,
                    headers: HeaderMap::new(),
                    final_url: out.final_url,
                    content_type: "text/html".into(),
                    body: out.body,
                    etag: None,
                    last_modified: None,
                    response_time_ms: 0,
                };
                return Ok(res.into_fetch_result(url));
            }
            // Solved a token but the page is still challenged → give up + record
            // the domain (G-3) so R5 never re-attempts it.
            emit_solve_audit(domain, kind_label, provider, "failed", p.solve_cost_estimate_usd, spent);
            record_r5_give_up(io, p.governor, domain, "challenge-unsolved");
            Err(challenge_unsolved())
        }
        Some(Err(e)) => {
            let spent = p.cost_meter.map(|m| m.spent_usd()).unwrap_or(0.0);
            emit_solve_audit(domain, kind_label, provider, "failed", spent, spent);
            record_r5_give_up(io, p.governor, domain, "solve-error");
            Err(e)
        }
        None => {
            // No solver wired (e.g. browser feature off / live path declined to
            // inject). Honest give-up + record (G-3).
            emit_solve_audit(domain, kind_label, provider, "failed", 0.0, cumulative);
            record_r5_give_up(io, p.governor, domain, "no-solver");
            Err(challenge_unsolved())
        }
    }
}

/// Give up with the most-informative typed error; trip the breaker per the pure
/// `breaker_verdict` decision and write the permanent denylist on a hard trip.
fn give_up<E: RungIo>(
    io: &E,
    governor: &MinimalGovernor,
    domain: &str,
    verdict: &Verdict,
    url: &str,
    r4_attempted_and_blocked: bool,
) -> Result<FetchResult> {
    let gcfg = GovernorConfig::from_parts(0, 0, 0, 0.0, None);
    let trip = breaker_verdict(&verdict.class, r4_attempted_and_blocked, &gcfg);
    if trip == BreakerTrip::HardPermanent {
        let verdict_label = format!("{:?}", verdict.class);
        governor.record_hard_ban(domain, "hard-ban", &verdict_label);
        io.note("hard-ban");
        return Err(Error::PermanentlyDenied { domain: domain.to_string(), reason: "hard-ban".into() });
    }
    match &verdict.class {
        BlockClass::JsChallenge { vendor } => Err(Error::ChallengeUnsolved {
            url: url.to_string(),
            vendor: vendor_label(*vendor).to_string(),
        }),
        BlockClass::Captcha { .. } => Err(Error::ChallengeUnsolved {
            url: url.to_string(),
            vendor: "captcha".into(),
        }),
        BlockClass::RateLimited { retry_after_secs } => Err(Error::RateLimited {
            domain: domain.to_string(),
            retry_after_secs: *retry_after_secs,
        }),
        _ => Err(Error::Blocked { url: url.to_string(), status: 403 }),
    }
}

/// Observability event (Design 0005 H7). No secrets, tokens, keys, or PII —
/// only url/domain/rung/verdict-class/signal.
fn emit_verdict(url: &str, domain: &str, rung: &str, verdict: &Verdict) {
    tracing::debug!(
        url,
        domain,
        rung,
        verdict = ?verdict.class,
        signal = verdict.signal,
        "escalation verdict"
    );
}

// ── Live `RungIo` impl for `Fetcher` ─────────────────────────────────────────

impl RungIo for Fetcher {
    async fn raw_get(&self, url: &str) -> Result<RawResponse> {
        self.fetch_raw_once(url).await
    }

    async fn browser_render(&self, url: &str) -> Option<RungOutcome> {
        let br = self.browser_pool.fetch(url, self.browser_timeout).await?;
        let domain = registrable_domain(url);
        // Bind the stored clearance cookie to the ACTUAL minted UA surfaced by
        // the stealth browser (Design 0005 §4.1) — not a `DEFAULT_STEALTH_UA`
        // placeholder. Falls back to the default only on the legacy non-stealth
        // path, which surfaces `None` and never mints a clearance cookie anyway.
        // Never logged (cookie value is secret-equivalent).
        let ua = br
            .user_agent
            .clone()
            .unwrap_or_else(|| crate::browser::DEFAULT_STEALTH_UA.to_string());
        self.store_clearance_cookies(&domain, &br.cookies, &ua);
        Some(RungOutcome {
            final_url: br.final_url,
            body: br.body,
            status: br.status,
            cookies: br.cookies,
            user_agent: ua,
        })
    }

    async fn solve_captcha(&self, url: &str, body: &str) -> Option<Result<RungOutcome>> {
        // LIVE R5 solve flow (Design 0004 §2.4 / compliance gate 0001). The
        // controller (`run_r5`) has ALREADY enforced the gate composition
        // (C-1 config+opt-in, G-1 auth, C-10 cost pre-check, governor admit,
        // C-7 single attempt) before reaching here. This method performs the
        // mechanical extract → build_solver → solve → inject → re-fetch.
        //
        // Returns:
        //   - `None`  → no live solver available / cannot inject (browser off,
        //               no sitekey, disabled). The controller treats this as an
        //               honest give-up (`ChallengeUnsolved`) and records G-3.
        //   - `Some(Ok(outcome))`  → solved+injected; the controller re-classifies.
        //   - `Some(Err(e))`       → genuine solve/submit failure (typed).
        //
        // Secrets: the API key is read by env-var NAME only (via build_solver);
        // the token + injection JS are NEVER logged.

        // 1) Detect the CAPTCHA on the (classifier-confirmed) body. No sitekey →
        //    nothing to solve → honest give-up.
        let detected = crate::captcha::extract::detect_captcha(body)?;

        // 2) Build the solver from config (default-off; env-keyed). The cost
        //    meter is SHARED so the client's up-front reservation enforces the
        //    same session cap the controller pre-checked (C-10 / K-2). A misconfig
        //    or disabled state returns None/Err → give up honestly.
        let params = crate::captcha::SolverParams {
            enable_captcha_solver: self.enable_captcha_solver,
            provider: self.captcha_provider.as_deref(),
            api_key_env: self.captcha_api_key_env.as_deref(),
            timeout_secs: self.captcha_timeout_secs,
            session_cost_cap_usd: self.captcha_session_cost_cap_usd,
        };
        let solver = match crate::captcha::build_solver(&params, self.cost_meter.handle()) {
            Ok(Some(s)) => s,
            Ok(None) => return None, // solver disabled → give up
            Err(e) => return Some(Err(e)),
        };

        // 3) Build the provider request from the detected markers.
        let req = crate::captcha::CaptchaRequest {
            kind: detected.kind,
            site_key: detected.site_key,
            page_url: url.to_string(),
            action: detected.action,
            cdata: detected.cdata,
        };

        // 4) Solve (single bounded attempt — the client polls internally up to
        //    its own timeout; the controller does not loop). The client reserves
        //    cost up-front and hard-halts at the cap.
        let token = match solver.solve(&req).await {
            Ok(t) => t,
            Err(e) => return Some(Err(e.into_common(solver.provider()))),
        };

        // 5) Inject the token into a stealth page + re-fetch (browser path). If
        //    the browser feature is off, `solve_in_page` returns None → R5 cannot
        //    inject → honest give-up (the token is never persisted; it expires).
        let injection_js = crate::captcha::inject::build_injection_js(detected.kind, &token);
        let br = self
            .browser_pool
            .solve_in_page(url, &injection_js, self.browser_timeout)
            .await?;

        let ua = br
            .user_agent
            .clone()
            .unwrap_or_else(|| crate::browser::DEFAULT_STEALTH_UA.to_string());
        // G-4: we return only the transformed outcome; the raw solved-gated body
        // is not persisted here (caching is gated to a final RealContent verdict
        // in `escalate`, and the token is dropped at end of scope).
        Some(Ok(RungOutcome {
            final_url: br.final_url,
            body: br.body,
            status: br.status,
            cookies: br.cookies,
            user_agent: ua,
        }))
    }

    fn hydration_salvage(&self, url: &str, raw: &RawResponse) -> Option<FetchResult> {
        // R0: pure, zero-request in-band salvage via the extractor crate (Design
        // 0005 §2.3). Acyclic dep (extractor has no crawler dep).
        let h = web_search_extractor::hydration::extract(&raw.body)?;
        if h.body_text.chars().count() < 200 {
            return None;
        }
        let mut html = String::new();
        if let Some(t) = &h.title {
            html.push_str(&format!("<h1>{t}</h1>"));
        }
        html.push_str(&format!("<p>{}</p>", h.body_text));
        Some(FetchResult {
            url: url.to_string(),
            final_url: raw.final_url.clone(),
            status: raw.status,
            content_type: "text/html".into(),
            body: html,
            response_time_ms: raw.response_time_ms,
            content_length: h.body_text.len(),
            is_spa: false,
            from_cache: false,
            etag: raw.etag.clone(),
            last_modified: raw.last_modified.clone(),
        })
    }

    async fn archive_fetch(&self, url: &str) -> Option<RungOutcome> {
        // R3: Internet Archive CDX fallback. Disabled => no-op. Reuses the shared
        // reqwest client; the module is non-fatal (returns None on any miss).
        if !self.enable_archive_fallback {
            return None;
        }
        let page = crate::archive::fetch_archived(&self.client, url, &self.archive_cfg).await?;
        Some(RungOutcome {
            final_url: page.final_url,
            body: page.body,
            status: page.status,
            cookies: Vec::new(),
            user_agent: String::new(),
        })
    }

    async fn alt_surface_probe(&self, url: &str, body: &str) -> Option<RungOutcome> {
        // R2: alternative-surface (RSS/Atom/JSON-Feed) probe. Disabled => no-op.
        if !self.enable_alt_surface || !self.src_feed {
            return None;
        }
        let item = crate::feeds::probe_feeds(
            &self.client,
            url,
            body,
            self.alt_probe_timeout_ms,
            self.max_alt_probes,
        )
        .await?;
        // Synthesize an HTML document from the recovered feed item so the
        // classifier + extractor see it as a normal article body.
        let recovered = feed_item_to_html(&item);
        if recovered.trim().is_empty() {
            return None;
        }
        Some(RungOutcome {
            // The feed item's own link is the canonical article URL; fall back to
            // the requested URL when the feed omitted it.
            final_url: if item.link.is_empty() { url.to_string() } else { item.link.clone() },
            body: recovered,
            status: 200,
            cookies: Vec::new(),
            user_agent: String::new(),
        })
    }

    async fn sleep(&self, dur: Duration) {
        tokio::time::sleep(dur).await;
    }
}

/// Wrap a recovered [`crate::feeds::FeedItem`] into a minimal HTML document.
/// Prefers the full `content_html`; falls back to the summary. The title becomes
/// an `<h1>` so downstream extraction keeps it. Pure.
fn feed_item_to_html(item: &crate::feeds::FeedItem) -> String {
    let body = item
        .content_html
        .clone()
        .or_else(|| item.summary.clone())
        .unwrap_or_default();
    let title = &item.title;
    format!("<html><body><article><h1>{title}</h1>{body}</article></body></html>")
}

/// Detect if page is a Single Page Application (JS-rendered).
///
/// Signals:
/// - Very short body with <script> tags
/// - Empty <div id="root"> or <div id="app">
/// - <noscript> containing meaningful content
fn detect_spa(body: &str) -> bool {
    let body_lower = body.to_lowercase();

    // Check for SPA mount points with no content
    let spa_markers = [
        r#"<div id="root"></div>"#,
        r#"<div id="app"></div>"#,
        r#"<div id="__next"></div>"#,
        r#"<div id="root"> </div>"#,
    ];
    for marker in &spa_markers {
        if body_lower.contains(marker) {
            return true;
        }
    }

    // Short body text but has script tags
    let text_len = estimate_visible_text_len(&body_lower);
    let has_scripts = body_lower.contains("<script");
    if text_len < 500 && has_scripts && body.len() > 5000 {
        return true;
    }

    // Has meaningful noscript content (suggests JS-dependent)
    if let Some(start) = body_lower.find("<noscript>") {
        if let Some(end) = body_lower[start..].find("</noscript>") {
            let noscript_content = &body_lower[start + 10..start + end];
            let noscript_text_len = estimate_visible_text_len(noscript_content);
            if noscript_text_len > 100 {
                return true;
            }
        }
    }

    false
}

/// Rough estimate of visible text length (strip tags).
fn estimate_visible_text_len(html: &str) -> usize {
    let mut in_tag = false;
    let in_script = false;
    let mut text_len = 0;

    for ch in html.chars() {
        match ch {
            '<' => {
                in_tag = true;
            }
            '>' => {
                in_tag = false;
            }
            _ if !in_tag && !in_script => {
                if !ch.is_whitespace() {
                    text_len += 1;
                }
            }
            _ => {}
        }
    }
    text_len
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_spa_empty_root() {
        assert!(detect_spa(r#"<html><body><div id="root"></div><script src="bundle.js"></script></body></html>"#));
    }

    #[test]
    fn detect_spa_normal_page() {
        assert!(!detect_spa(
            "<html><body><h1>Hello World</h1><p>This is a normal page with lots of content that should not be detected as a SPA because it has substantial visible text content.</p></body></html>"
        ));
    }

    #[test]
    fn detect_spa_next_app() {
        assert!(detect_spa(r#"<html><body><div id="__next"></div><script defer src="/_next/static/chunks/main.js"></script></body></html>"#));
    }

    #[test]
    fn estimate_text_len_works() {
        assert_eq!(estimate_visible_text_len("<p>Hello</p>"), 5);
        assert_eq!(estimate_visible_text_len("<div><span>AB</span></div>"), 2);
        assert_eq!(estimate_visible_text_len("plain text"), 9);
    }

    // ── Controller routing tests with MOCKED rungs (Design 0005 §8.2) ────────
    //
    // No live network / browser / solver. A `MockIo` returns scripted rung
    // outcomes and counts calls; the real classifier + minimal governor run.
    use std::cell::Cell;
    use std::sync::atomic::AtomicUsize;

    struct MockIo {
        raw: RawResponse,
        r2: Option<RungOutcome>,
        r3: Option<RungOutcome>,
        r4: Option<RungOutcome>,
        r5: Option<Result<RungOutcome>>,
        salvage: Option<FetchResult>,
        raw_calls: AtomicUsize,
        r2_calls: AtomicUsize,
        r3_calls: AtomicUsize,
        r4_calls: AtomicUsize,
        r5_calls: AtomicUsize,
        hard_bans: AtomicUsize,
        r5_give_up_denylists: AtomicUsize,
    }

    impl MockIo {
        fn new(raw: RawResponse) -> Self {
            Self {
                raw,
                r2: None,
                r3: None,
                r4: None,
                r5: None,
                salvage: None,
                raw_calls: AtomicUsize::new(0),
                r2_calls: AtomicUsize::new(0),
                r3_calls: AtomicUsize::new(0),
                r4_calls: AtomicUsize::new(0),
                r5_calls: AtomicUsize::new(0),
                hard_bans: AtomicUsize::new(0),
                r5_give_up_denylists: AtomicUsize::new(0),
            }
        }
    }

    /// Clone a `RungOutcome` (it derives no `Clone` — fields are cloned by hand,
    /// mirroring `browser_render`'s mapping).
    fn clone_outcome(o: &RungOutcome) -> RungOutcome {
        RungOutcome {
            final_url: o.final_url.clone(),
            body: o.body.clone(),
            status: o.status,
            cookies: o.cookies.clone(),
            user_agent: o.user_agent.clone(),
        }
    }

    impl RungIo for MockIo {
        async fn raw_get(&self, _url: &str) -> Result<RawResponse> {
            self.raw_calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.raw.clone())
        }
        async fn browser_render(&self, _url: &str) -> Option<RungOutcome> {
            self.r4_calls.fetch_add(1, Ordering::SeqCst);
            self.r4.as_ref().map(|o| RungOutcome {
                final_url: o.final_url.clone(),
                body: o.body.clone(),
                status: o.status,
                cookies: o.cookies.clone(),
                user_agent: o.user_agent.clone(),
            })
        }
        async fn solve_captcha(&self, _url: &str, _body: &str) -> Option<Result<RungOutcome>> {
            self.r5_calls.fetch_add(1, Ordering::SeqCst);
            match &self.r5 {
                None => None,
                Some(Ok(o)) => Some(Ok(RungOutcome {
                    final_url: o.final_url.clone(),
                    body: o.body.clone(),
                    status: o.status,
                    cookies: o.cookies.clone(),
                    user_agent: o.user_agent.clone(),
                })),
                Some(Err(_)) => Some(Err(Error::ChallengeUnsolved {
                    url: "x".into(),
                    vendor: "captcha".into(),
                })),
            }
        }
        fn hydration_salvage(&self, _url: &str, _raw: &RawResponse) -> Option<FetchResult> {
            self.salvage.clone()
        }
        async fn archive_fetch(&self, _url: &str) -> Option<RungOutcome> {
            self.r3_calls.fetch_add(1, Ordering::SeqCst);
            self.r3.as_ref().map(clone_outcome)
        }
        async fn alt_surface_probe(&self, _url: &str, _body: &str) -> Option<RungOutcome> {
            self.r2_calls.fetch_add(1, Ordering::SeqCst);
            self.r2.as_ref().map(clone_outcome)
        }
        async fn sleep(&self, _dur: Duration) { /* no-op in tests */ }
        fn note(&self, what: &str) {
            match what {
                "hard-ban" => {
                    self.hard_bans.fetch_add(1, Ordering::SeqCst);
                }
                "r5-give-up-denylist" => {
                    self.r5_give_up_denylists.fetch_add(1, Ordering::SeqCst);
                }
                _ => {}
            }
        }
    }

    fn raw(status: u16, headers: &[(&str, &str)], body: &str) -> RawResponse {
        let mut h = HeaderMap::new();
        for (k, v) in headers {
            h.insert(
                reqwest::header::HeaderName::from_bytes(k.as_bytes()).unwrap(),
                reqwest::header::HeaderValue::from_str(v).unwrap(),
            );
        }
        RawResponse {
            status,
            headers: h,
            final_url: "https://example.com/page".into(),
            content_type: "text/html".into(),
            body: body.to_string(),
            etag: None,
            last_modified: None,
            response_time_ms: 1,
        }
    }

    fn outcome(body: &str) -> RungOutcome {
        RungOutcome {
            final_url: "https://example.com/page".into(),
            body: body.to_string(),
            status: 200,
            cookies: vec![("cf_clearance".into(), "abc".into())],
            user_agent: "ua".into(),
        }
    }

    fn gov(max_rung: u8) -> (Arc<MinimalGovernor>, ClassifierConfig, u8) {
        let g = MinimalGovernor::new(GovernorConfig::from_parts(50, 5, 300, 0.3, None));
        (g, ClassifierConfig::seeded(500, 30), max_rung)
    }

    fn article_body() -> String {
        format!(
            "<html><body><article><h1>T</h1><p>{}</p></article></body></html>",
            "Real article body content here. ".repeat(40)
        )
    }

    #[allow(clippy::too_many_arguments)]
    async fn run<E: RungIo>(
        io: &E,
        g: &MinimalGovernor,
        c: &ClassifierConfig,
        max_rung: u8,
        enable_browser: bool,
        enable_solver: bool,
    ) -> Result<FetchResult> {
        // Default to opted-in so the legacy routing tests that EXPECT a solve to
        // proceed (e.g. captcha_solver_enabled_solves_and_returns) still pass; the
        // per-run-opt-in gate has dedicated tests below that toggle it off.
        run_with(io, g, c, max_rung, enable_browser, enable_solver, RunOpts::default()).await
    }

    /// Knobs for the compliance-gate tests (gate 0001 C-1 / G-1 / C-10) plus the
    /// R2/R3 rung toggles (ADR 0003 §6.2).
    #[derive(Clone, Copy)]
    struct RunOpts<'a> {
        run_opt_in: bool,
        authenticated: bool,
        cost_meter: Option<&'a CostMeterHandle>,
        estimate: f64,
        enable_archive: bool,
        enable_alt: bool,
    }
    impl Default for RunOpts<'_> {
        fn default() -> Self {
            Self {
                run_opt_in: true,
                authenticated: false,
                cost_meter: None,
                estimate: 0.003,
                enable_archive: false,
                enable_alt: false,
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_with<E: RungIo>(
        io: &E,
        g: &MinimalGovernor,
        c: &ClassifierConfig,
        max_rung: u8,
        enable_browser: bool,
        enable_solver: bool,
        opts: RunOpts<'_>,
    ) -> Result<FetchResult> {
        let p = EscalationParams {
            url: "https://example.com/page",
            classifier_cfg: c,
            governor: g,
            enable_browser,
            enable_captcha_solver: enable_solver,
            max_rung,
            enable_archive_fallback: opts.enable_archive,
            enable_alt_surface: opts.enable_alt,
            retry_after_ceiling_secs: 120,
            captcha_run_opt_in: opts.run_opt_in,
            authenticated_context: opts.authenticated,
            cost_meter: opts.cost_meter,
            solve_cost_estimate_usd: opts.estimate,
        };
        run_escalation(io, &p).await
    }

    // ── R2/R3 alternative-surface + archive rung tests (ADR 0003 §3.1) ────────

    /// A thin body classifies as SoftBlock; with no R0 salvage, the R3 archive
    /// snapshot (zero-ban-risk) recovers real content.
    #[tokio::test]
    async fn soft_block_r3_archive_recovers() {
        let mut io = MockIo::new(raw(200, &[], "blocked"));
        io.r3 = Some(outcome(&article_body()));
        let (g, c, mr) = gov(4);
        let opts = RunOpts { enable_archive: true, ..RunOpts::default() };
        let res = run_with(&io, &g, &c, mr, false, false, opts).await.unwrap();
        assert!(res.body.contains("Real article body"));
        assert_eq!(io.r3_calls.load(Ordering::SeqCst), 1, "R3 archive must be tried");
        assert_eq!(io.r2_calls.load(Ordering::SeqCst), 0, "R2 not reached once R3 wins");
    }

    /// With archive disabled, the R2 alternative-surface (feeds) probe recovers
    /// the content on the same origin (governor-gated).
    #[tokio::test]
    async fn soft_block_r2_alt_surface_recovers() {
        let mut io = MockIo::new(raw(200, &[], "blocked"));
        io.r2 = Some(outcome(&article_body()));
        let (g, c, mr) = gov(4);
        let opts = RunOpts { enable_alt: true, ..RunOpts::default() };
        let res = run_with(&io, &g, &c, mr, false, false, opts).await.unwrap();
        assert!(res.body.contains("Real article body"));
        assert_eq!(io.r3_calls.load(Ordering::SeqCst), 0, "archive disabled => not called");
        assert_eq!(io.r2_calls.load(Ordering::SeqCst), 1, "R2 must be tried");
    }

    /// Reputation-first ordering: when BOTH rungs are enabled and the zero-ban-risk
    /// R3 archive recovers content, the live-origin R2 probe is never reached.
    #[tokio::test]
    async fn soft_block_r3_preferred_over_r2() {
        let mut io = MockIo::new(raw(200, &[], "blocked"));
        io.r3 = Some(outcome(&article_body()));
        io.r2 = Some(outcome(&article_body()));
        let (g, c, mr) = gov(4);
        let opts = RunOpts { enable_archive: true, enable_alt: true, ..RunOpts::default() };
        let res = run_with(&io, &g, &c, mr, false, false, opts).await.unwrap();
        assert!(res.body.contains("Real article body"));
        assert_eq!(io.r3_calls.load(Ordering::SeqCst), 1);
        assert_eq!(io.r2_calls.load(Ordering::SeqCst), 0, "R2 must NOT run when R3 already won");
    }

    /// With both rungs disabled (defaults), a soft block gives up without ever
    /// touching the R2/R3 surfaces — preserving off-safe behavior.
    #[tokio::test]
    async fn soft_block_rungs_disabled_give_up() {
        let io = MockIo::new(raw(200, &[], "blocked"));
        let (g, c, mr) = gov(4);
        let err = run(&io, &g, &c, mr, false, false).await.unwrap_err();
        assert!(matches!(err, Error::Blocked { .. }));
        assert_eq!(io.r3_calls.load(Ordering::SeqCst), 0);
        assert_eq!(io.r2_calls.load(Ordering::SeqCst), 0);
    }

    /// If R3 archive returns a still-blocked snapshot (not RealContent), the
    /// controller falls through to R2.
    #[tokio::test]
    async fn soft_block_r3_miss_falls_through_to_r2() {
        let mut io = MockIo::new(raw(200, &[], "blocked"));
        io.r3 = Some(outcome("blocked")); // archive snapshot also thin => not RealContent
        io.r2 = Some(outcome(&article_body()));
        let (g, c, mr) = gov(4);
        let opts = RunOpts { enable_archive: true, enable_alt: true, ..RunOpts::default() };
        let res = run_with(&io, &g, &c, mr, false, false, opts).await.unwrap();
        assert!(res.body.contains("Real article body"));
        assert_eq!(io.r3_calls.load(Ordering::SeqCst), 1, "R3 attempted");
        assert_eq!(io.r2_calls.load(Ordering::SeqCst), 1, "R2 reached after R3 miss");
    }

    #[tokio::test]
    async fn r1_real_content_returns_immediately() {
        let io = MockIo::new(raw(200, &[], &article_body()));
        let (g, c, mr) = gov(4);
        let res = run(&io, &g, &c, mr, false, false).await.unwrap();
        assert!(res.body.contains("Real article body"));
        assert_eq!(io.raw_calls.load(Ordering::SeqCst), 1);
        assert_eq!(io.r4_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn r1_spa_invokes_r4_render_once() {
        // A realistic small SPA shell: the empty-root marker triggers detect_spa
        // regardless of length, and is short enough that the R4-rendered article
        // body comfortably exceeds the half-length keep heuristic.
        let shell = r#"<html><body><div id="root"></div><script src="b.js"></script></body></html>"#;
        let mut io = MockIo::new(raw(200, &[], shell));
        io.r4 = Some(outcome(&article_body()));
        let (g, c, mr) = gov(4);
        let res = run(&io, &g, &c, mr, true, false).await.unwrap();
        assert_eq!(io.r4_calls.load(Ordering::SeqCst), 1, "exactly one render");
        assert!(res.body.contains("Real article body"));
    }

    #[tokio::test]
    async fn js_challenge_r4_clears_returns_and_stores() {
        // R1 sees a CF challenge; R4 renders real content.
        let challenge = r#"<html><head><title>Just a moment...</title></head>
            <body><script>window._cf_chl_opt={}</script></body></html>"#;
        let mut io = MockIo::new(raw(403, &[("server", "cloudflare")], challenge));
        io.r4 = Some(outcome(&article_body()));
        let (g, c, mr) = gov(4);
        let res = run(&io, &g, &c, mr, true, false).await.unwrap();
        assert!(res.body.contains("Real article body"));
        assert_eq!(io.r4_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn js_challenge_r4_fails_solver_off_hard_bans() {
        // R1 sees CF challenge; R4 returns STILL a challenge; solver off.
        let challenge = r#"<html><head><title>Just a moment...</title></head>
            <body><script>window._cf_chl_opt={}</script></body></html>"#;
        let mut io = MockIo::new(raw(403, &[("server", "cloudflare")], challenge));
        io.r4 = Some(outcome(challenge)); // still challenged after R4
        let (g, c, mr) = gov(4);
        let err = run(&io, &g, &c, mr, true, false).await.unwrap_err();
        assert!(matches!(err, Error::PermanentlyDenied { .. }), "got {err:?}");
        // hard-ban recorded -> denylist + breaker
        assert_eq!(io.hard_bans.load(Ordering::SeqCst), 1);
        assert!(g.is_permanently_denied("example.com"));
    }

    #[tokio::test]
    async fn captcha_solver_disabled_gives_challenge_unsolved() {
        let body = r#"<div class="cf-turnstile" data-sitekey="k"></div>"#;
        let io = MockIo::new(raw(403, &[], body));
        let (g, c, mr) = gov(4);
        let err = run(&io, &g, &c, mr, true, false).await.unwrap_err();
        assert!(matches!(err, Error::ChallengeUnsolved { .. }), "got {err:?}");
        // solver never invoked (gated before solve_captcha)
        assert_eq!(io.r5_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn captcha_solver_enabled_solves_and_returns() {
        let body = r#"<div class="cf-turnstile" data-sitekey="k"></div>"#;
        let mut io = MockIo::new(raw(403, &[], body));
        io.r5 = Some(Ok(outcome(&article_body())));
        let (g, c, _mr) = gov(5);
        let res = run(&io, &g, &c, 5, true, true).await.unwrap();
        assert!(res.body.contains("Real article body"));
        assert_eq!(io.r5_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn permanently_denied_domain_no_live_rung() {
        let io = MockIo::new(raw(200, &[], &article_body()));
        let (g, c, mr) = gov(4);
        g.record_hard_ban("example.com", "hard-ban", "Cloudflare403");
        let err = run(&io, &g, &c, mr, true, true).await.unwrap_err();
        assert!(matches!(err, Error::PermanentlyDenied { .. }));
        assert_eq!(io.raw_calls.load(Ordering::SeqCst), 0, "no live GET for denied domain");
    }

    #[tokio::test]
    async fn soft_block_hydration_salvage_no_second_request() {
        // Soft-blocked body, but R0 salvage recovers content -> RealContent, and
        // NO second live request is made.
        let mut io = MockIo::new(raw(200, &[], "<html><body>Access denied</body></html>"));
        io.salvage = Some(FetchResult {
            url: "https://example.com/page".into(),
            final_url: "https://example.com/page".into(),
            status: 200,
            content_type: "text/html".into(),
            body: "<h1>Salvaged</h1><p>recovered content</p>".into(),
            response_time_ms: 0,
            content_length: 20,
            is_spa: false,
            from_cache: false,
            etag: None,
            last_modified: None,
        });
        let (g, c, mr) = gov(4);
        let res = run(&io, &g, &c, mr, true, false).await.unwrap();
        assert!(res.body.contains("Salvaged"));
        assert_eq!(io.raw_calls.load(Ordering::SeqCst), 1, "exactly one live request");
    }

    #[tokio::test]
    async fn rate_limited_honors_retry_after() {
        // 429 with Retry-After:5 -> RateLimited error carrying the value; the
        // controller sleeps the (mocked, no-op) wait and gives up for requeue.
        let io = MockIo::new(raw(429, &[("retry-after", "5")], "slow down"));
        let (g, c, mr) = gov(4);
        let err = run(&io, &g, &c, mr, true, false).await.unwrap_err();
        match err {
            Error::RateLimited { retry_after_secs, .. } => assert_eq!(retry_after_secs, 5),
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    // ── Compliance gate 0001 conditions (OFFLINE, mock solver) ──────────────
    //
    // These prove G-1 (auth refusal), C-1 (per-run opt-in), C-10 (cost
    // pre-check), G-3 (record-on-give-up → denylist), and the gate composition
    // (config-off OR opt-in-absent OR over-budget OR authed → NO solve). The
    // mock `solve_captcha` NEVER calls a real provider.

    fn turnstile_body() -> &'static str {
        r#"<div class="cf-turnstile" data-sitekey="0xKEY"></div>"#
    }

    /// C-1: solver ENABLED but per-run opt-in ABSENT → NO solve (default-deny).
    #[tokio::test]
    async fn captcha_opt_in_absent_no_solve() {
        let mut io = MockIo::new(raw(403, &[], turnstile_body()));
        io.r5 = Some(Ok(outcome(&article_body())));
        let (g, c, _mr) = gov(5);
        let opts = RunOpts { run_opt_in: false, ..Default::default() };
        let err = run_with(&io, &g, &c, 5, true, true, opts).await.unwrap_err();
        assert!(matches!(err, Error::ChallengeUnsolved { .. }), "got {err:?}");
        assert_eq!(io.r5_calls.load(Ordering::SeqCst), 0, "solver must NOT be invoked without opt-in");
        // opt-out is NOT a give-up that bans (solvable later once opted in).
        assert_eq!(io.r5_give_up_denylists.load(Ordering::SeqCst), 0);
    }

    /// C-1: config flag OFF (even if opt-in true) → NO solve.
    #[tokio::test]
    async fn captcha_config_off_no_solve_even_with_opt_in() {
        let mut io = MockIo::new(raw(403, &[], turnstile_body()));
        io.r5 = Some(Ok(outcome(&article_body())));
        let (g, c, _mr) = gov(5);
        let opts = RunOpts { run_opt_in: true, ..Default::default() };
        // enable_solver = false
        let err = run_with(&io, &g, &c, 5, true, false, opts).await.unwrap_err();
        assert!(matches!(err, Error::ChallengeUnsolved { .. }), "got {err:?}");
        assert_eq!(io.r5_calls.load(Ordering::SeqCst), 0);
    }

    /// C-1 happy path: BOTH flags on → solve proceeds.
    #[tokio::test]
    async fn captcha_both_gates_on_solves() {
        let mut io = MockIo::new(raw(403, &[], turnstile_body()));
        io.r5 = Some(Ok(outcome(&article_body())));
        let (g, c, _mr) = gov(5);
        let opts = RunOpts { run_opt_in: true, ..Default::default() };
        let res = run_with(&io, &g, &c, 5, true, true, opts).await.unwrap();
        assert!(res.body.contains("Real article body"));
        assert_eq!(io.r5_calls.load(Ordering::SeqCst), 1);
    }

    /// G-1: authenticated context → REFUSE, NO solve, AND record to denylist.
    #[tokio::test]
    async fn captcha_authenticated_context_refused_and_denylisted() {
        let mut io = MockIo::new(raw(403, &[], turnstile_body()));
        io.r5 = Some(Ok(outcome(&article_body())));
        let (g, c, _mr) = gov(5);
        let opts = RunOpts { run_opt_in: true, authenticated: true, ..Default::default() };
        let err = run_with(&io, &g, &c, 5, true, true, opts).await.unwrap_err();
        assert!(matches!(err, Error::ChallengeUnsolved { .. }), "got {err:?}");
        assert_eq!(io.r5_calls.load(Ordering::SeqCst), 0, "auth-wall CAPTCHA must NEVER be solved (G-1)");
        // G-3: domain recorded so R5 never re-attempts it.
        assert_eq!(io.r5_give_up_denylists.load(Ordering::SeqCst), 1);
        assert!(g.is_permanently_denied("example.com"));
    }

    /// C-10: cost pre-check would exceed the cap → REFUSE, NO solve, denylist.
    #[tokio::test]
    async fn captcha_over_budget_refused() {
        let mut io = MockIo::new(raw(403, &[], turnstile_body()));
        io.r5 = Some(Ok(outcome(&article_body())));
        let (g, c, _mr) = gov(5);
        // A meter already at its cap: any reservation exceeds.
        let meter = CostMeter::new(0.0);
        let handle = meter.handle();
        let opts = RunOpts { run_opt_in: true, cost_meter: Some(&handle), estimate: 0.003, ..Default::default() };
        let err = run_with(&io, &g, &c, 5, true, true, opts).await.unwrap_err();
        assert!(matches!(err, Error::ChallengeUnsolved { .. }), "got {err:?}");
        assert_eq!(io.r5_calls.load(Ordering::SeqCst), 0, "over-budget must NOT invoke the solver (C-10)");
        assert_eq!(io.r5_give_up_denylists.load(Ordering::SeqCst), 1);
        assert!(g.is_permanently_denied("example.com"));
    }

    /// C-10 happy path: cost pre-check passes (under cap) → solve proceeds, and
    /// the pre-check is refunded (no double-charge; the meter is untouched by the
    /// mock solver which does not spend).
    #[tokio::test]
    async fn captcha_under_budget_solves_no_double_charge() {
        let mut io = MockIo::new(raw(403, &[], turnstile_body()));
        io.r5 = Some(Ok(outcome(&article_body())));
        let (g, c, _mr) = gov(5);
        let meter = CostMeter::new(5.0);
        let handle = meter.handle();
        let opts = RunOpts { run_opt_in: true, cost_meter: Some(&handle), estimate: 0.003, ..Default::default() };
        let res = run_with(&io, &g, &c, 5, true, true, opts).await.unwrap();
        assert!(res.body.contains("Real article body"));
        assert_eq!(io.r5_calls.load(Ordering::SeqCst), 1);
        // Pre-check reserved then refunded; the mock solver does not spend → 0.
        assert_eq!(meter.spent_usd(), 0.0, "pre-check must refund so the client is the sole spender");
    }

    /// G-3: solver INVOKED but the post-submit page is still challenged → give up
    /// AND record the domain to the denylist so R5 never re-attempts it.
    #[tokio::test]
    async fn captcha_solve_fails_records_denylist() {
        let mut io = MockIo::new(raw(403, &[], turnstile_body()));
        // Solver returns a body that is STILL a CAPTCHA (not RealContent).
        io.r5 = Some(Ok(outcome(turnstile_body())));
        let (g, c, _mr) = gov(5);
        let opts = RunOpts { run_opt_in: true, ..Default::default() };
        let err = run_with(&io, &g, &c, 5, true, true, opts).await.unwrap_err();
        assert!(matches!(err, Error::ChallengeUnsolved { .. }), "got {err:?}");
        assert_eq!(io.r5_calls.load(Ordering::SeqCst), 1, "exactly one solve attempt (C-7)");
        assert_eq!(io.r5_give_up_denylists.load(Ordering::SeqCst), 1, "G-3: give-up must denylist");
        assert!(g.is_permanently_denied("example.com"));
    }

    /// G-3 verification: once denylisted by an R5 give-up, a SUBSEQUENT escalate
    /// for the same domain short-circuits at the governor BEFORE any live rung —
    /// proving the denylist check happens before re-entering R5.
    #[tokio::test]
    async fn denylisted_after_give_up_blocks_reentry() {
        let (g, c, _mr) = gov(5);
        // First run: solve fails → domain denylisted.
        {
            let mut io = MockIo::new(raw(403, &[], turnstile_body()));
            io.r5 = Some(Ok(outcome(turnstile_body())));
            let opts = RunOpts { run_opt_in: true, ..Default::default() };
            let _ = run_with(&io, &g, &c, 5, true, true, opts).await;
        }
        assert!(g.is_permanently_denied("example.com"));
        // Second run: a fresh attempt must NOT touch the origin (no live GET).
        let io2 = MockIo::new(raw(200, &[], &article_body()));
        let opts = RunOpts { run_opt_in: true, ..Default::default() };
        let err = run_with(&io2, &g, &c, 5, true, true, opts).await.unwrap_err();
        assert!(matches!(err, Error::PermanentlyDenied { .. }), "got {err:?}");
        assert_eq!(io2.raw_calls.load(Ordering::SeqCst), 0, "denylist must block re-entry before R1");
    }

    // ── is_authenticated_context (gate 0001 G-1) unit tests ─────────────────

    fn hmap(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(
                reqwest::header::HeaderName::from_bytes(k.as_bytes()).unwrap(),
                reqwest::header::HeaderValue::from_str(v).unwrap(),
            );
        }
        h
    }

    #[test]
    fn auth_context_userinfo_in_url_is_authed() {
        assert!(is_authenticated_context("https://user:pass@example.com/x", &HeaderMap::new()));
        assert!(is_authenticated_context("https://user@example.com/x", &HeaderMap::new()));
    }

    #[test]
    fn auth_context_authorization_header_is_authed() {
        let h = hmap(&[("authorization", "Bearer abc.def.ghi")]);
        assert!(is_authenticated_context("https://example.com/x", &h));
        let h2 = hmap(&[("authorization", "Basic dXNlcjpwYXNz")]);
        assert!(is_authenticated_context("https://example.com/x", &h2));
    }

    #[test]
    fn auth_context_session_cookie_is_authed() {
        for c in [
            "sessionid=abc",
            "PHPSESSID=xyz",
            "JSESSIONID=1",
            "auth_token=t",
            "sid=9",
            "connect.sid=s",
            "laravel_session=l",
        ] {
            let h = hmap(&[("cookie", c)]);
            assert!(is_authenticated_context("https://example.com/x", &h), "cookie {c:?} should be authed");
        }
    }

    #[test]
    fn auth_context_public_request_is_not_authed() {
        // No userinfo, no Authorization, no session-looking cookie.
        assert!(!is_authenticated_context("https://example.com/article", &HeaderMap::new()));
        let benign = hmap(&[("cookie", "theme=dark; consent=1; locale=en")]);
        assert!(!is_authenticated_context("https://example.com/article", &benign));
    }

    // ── G-2 audit event field-set assertion (no secrets) ────────────────────
    //
    // We cannot easily intercept tracing output in a unit test without a
    // subscriber harness, so we assert the CONTRACT structurally: the audit
    // emitter takes ONLY the G-2 fields by signature (domain, kind, provider,
    // outcome, cost, cumulative) — there is no parameter for a token, key,
    // cookie, URL, query string, or body. This compile-time shape is the
    // guarantee that no secret can be passed. We exercise it to ensure it runs
    // without panicking and that the values are the sanitized labels only.
    #[test]
    fn audit_event_carries_only_safe_fields() {
        // Domain is the registrable host (no query string); kind/provider/outcome
        // are fixed labels; costs are f64. No secret-bearing parameter exists.
        emit_solve_audit("example.com", "turnstile", "configured", "refused", 0.0, 0.0);
        emit_solve_audit("example.com", "recaptcha", "configured", "solved", 0.003, 0.003);
        // audit_provider never leaks a key — only a coarse label.
        let p_on = EscalationParams {
            url: "https://example.com/x",
            classifier_cfg: &ClassifierConfig::seeded(500, 30),
            governor: &MinimalGovernor::new(GovernorConfig::from_parts(50, 5, 300, 0.3, None)),
            enable_browser: true,
            enable_captcha_solver: true,
            max_rung: 5,
            enable_archive_fallback: false,
            enable_alt_surface: false,
            retry_after_ceiling_secs: 120,
            captcha_run_opt_in: true,
            authenticated_context: false,
            cost_meter: None,
            solve_cost_estimate_usd: 0.003,
        };
        assert_eq!(audit_provider(&p_on), "configured");
        let p_off = EscalationParams { enable_captcha_solver: false, ..p_on };
        assert_eq!(audit_provider(&p_off), "none");
    }

    /// Expand-contract parity: with escalation OFF, the controller path is never
    /// entered. We assert the gate flag itself, since the legacy path is a thin
    /// wrapper and is exercised by the existing fetch_once tests.
    #[test]
    fn escalation_flag_default_off() {
        let cfg = web_search_common::config::Config::default().crawler;
        assert!(!cfg.enable_escalation, "escalation must default OFF (expand-contract)");
        // Cross-client replay (the unverified path) must default OFF too.
        assert!(!cfg.enable_clearance_replay, "clearance replay must default OFF (Design 0005 §4.2)");
        let _ = Cell::new(0); // silence unused import in some cfgs
    }

    // ── Clearance store + cross-client replay decoration (Design 0005 §4) ────
    //
    // OFFLINE unit tests for the store (insert / expire / UA-match) and the pure
    // request-decoration helper (cookie + UA attached). No live client, no
    // network. The live binding question is the separate `#[ignore]` spike below.

    /// Build a `Fetcher` from a config so we exercise the REAL store/getter on a
    /// real struct (no live fetch is performed). `enable_browser=false` and the
    /// escalation flag is irrelevant for the in-memory store tests.
    fn test_fetcher(enable_clearance_replay: bool) -> Fetcher {
        let mut cfg = web_search_common::config::Config::default().crawler;
        cfg.enable_clearance_replay = enable_clearance_replay;
        Fetcher::new(&cfg).expect("fetcher builds")
    }

    #[test]
    fn clearance_store_insert_and_live_match() {
        let f = test_fetcher(false);
        f.store_clearance_cookies(
            "example.com",
            &[("cf_clearance".into(), "TOKEN123".into())],
            "Mozilla/5.0 MintedUA",
        );
        let entry = f.live_clearance("example.com").expect("entry present + live");
        assert_eq!(entry.name, "cf_clearance");
        assert_eq!(entry.value, "TOKEN123");
        // UA-match: the stored UA is the exact minted UA, replayed verbatim.
        assert_eq!(entry.user_agent, "Mozilla/5.0 MintedUA");
        // A domain with no minted cookie has no entry.
        assert!(f.live_clearance("other.com").is_none());
    }

    #[test]
    fn clearance_store_only_keeps_clearance_cookies() {
        let f = test_fetcher(false);
        // Non-clearance cookies must NOT be stored (only cf_clearance/datadome).
        f.store_clearance_cookies(
            "example.com",
            &[("session_id".into(), "abc".into()), ("__cfduid".into(), "x".into())],
            "ua",
        );
        assert!(f.live_clearance("example.com").is_none());
        // datadome IS a clearance cookie.
        f.store_clearance_cookies("dd.com", &[("datadome".into(), "DDTOK".into())], "ua");
        assert_eq!(f.live_clearance("dd.com").unwrap().name, "datadome");
    }

    #[test]
    fn clearance_store_expires_past_ttl() {
        let f = test_fetcher(false);
        // Insert a manually-expired entry (minted_at in the past, tiny TTL).
        f.clearance.insert(
            "example.com".into(),
            ClearanceEntry {
                name: "cf_clearance".into(),
                value: "TOKEN".into(),
                user_agent: "ua".into(),
                minted_at: Instant::now() - Duration::from_secs(10),
                ttl: Duration::from_secs(1),
            },
        );
        // Past TTL => treated as absent.
        assert!(f.live_clearance("example.com").is_none(), "expired entry must not be live");
    }

    #[test]
    fn clearance_decoration_attaches_cookie_and_ua() {
        let entry = ClearanceEntry {
            name: "cf_clearance".into(),
            value: "TOKEN123".into(),
            user_agent: "Mozilla/5.0 MintedUA".into(),
            minted_at: Instant::now(),
            ttl: Duration::from_secs(1200),
        };
        let deco = clearance_request_decoration(&entry).expect("decoration built");
        // Cookie header is `name=value`.
        assert_eq!(deco.cookie_header, "cf_clearance=TOKEN123");
        // UA replayed verbatim — a mismatch would invalidate the cookie.
        assert_eq!(deco.user_agent, "Mozilla/5.0 MintedUA");
    }

    #[test]
    fn clearance_decoration_requires_value_and_ua() {
        // No cookie value => no decoration.
        let no_value = ClearanceEntry {
            name: "cf_clearance".into(),
            value: String::new(),
            user_agent: "ua".into(),
            minted_at: Instant::now(),
            ttl: Duration::from_secs(1200),
        };
        assert!(clearance_request_decoration(&no_value).is_none());
        // No minted UA => no decoration (replaying without the minting UA
        // invalidates the cookie; we refuse rather than send a mismatched UA).
        let no_ua = ClearanceEntry {
            name: "cf_clearance".into(),
            value: "TOKEN".into(),
            user_agent: String::new(),
            minted_at: Instant::now(),
            ttl: Duration::from_secs(1200),
        };
        assert!(clearance_request_decoration(&no_ua).is_none());
    }

    // ── GATED LIVE SPIKE — operator-run, NOT for CI (Design 0005 §4.2 / R-3) ──
    //
    // This is the UNVERIFIED-binding spike that gates `enable_clearance_replay`.
    // It is `#[ignore]` and must NEVER run in CI. Running it requires ALL of:
    //   - a real Chrome/Chromium binary + the `browser` cargo feature,
    //   - network egress to a KNOWN Cloudflare-protected URL (set CF_SPIKE_URL),
    //   - acceptance that it makes a SMALL number of real requests to a live WAF.
    //
    // It is single-IP-POLITE: exactly ONE browser solve + exactly ONE reqwest
    // replay GET to the same URL. It does NOT loop, retry, or hammer. Our IP is
    // static so the cf_clearance IP-binding is satisfied automatically; the ONLY
    // thing under test is whether the cookie is ALSO bound to the TLS JA3/HTTP2
    // fingerprint such that a browser-minted cookie is REJECTED when replayed by
    // the reqwest client (different TLS fingerprint).
    //
    // VERDICT: the test PRINTS whether the reqwest replay returned real content
    // (cookie is replayable across clients => build the R4→R1 reuse path) or a
    // fresh challenge (cookie is fingerprint-bound => keep-session-in-browser is
    // the fallback, `enable_clearance_replay` stays default-off). It asserts only
    // that the spike ran end-to-end, never the verdict (which is the finding).
    //
    // INVOKE:
    //   $env:CF_SPIKE_URL="https://<a-known-cloudflare-protected-url>"
    //   cargo test -p web-search-crawler --features browser -- --ignored --nocapture clearance_replay_binding
    #[cfg(feature = "browser")]
    #[tokio::test]
    #[ignore = "live CF target + network; single-IP-polite (1 solve + 1 replay); operator-run; not for CI"]
    async fn clearance_replay_binding_spike() {
        use crate::browser::BrowserPool;

        let target = match std::env::var("CF_SPIKE_URL") {
            Ok(u) if !u.is_empty() => u,
            _ => {
                eprintln!(
                    "[clearance-spike] SKIP: set CF_SPIKE_URL to a known Cloudflare-protected URL to run."
                );
                return;
            }
        };

        // STEP 1 — mint cf_clearance in the stealth browser (solve the challenge).
        let pool = BrowserPool::with_stealth(true, None, None);
        if !pool.is_available() {
            eprintln!("[clearance-spike] SKIP: browser feature/binary unavailable.");
            return;
        }
        let br = pool
            .fetch(&target, Duration::from_secs(45))
            .await
            .expect("[clearance-spike] browser produced no result");

        let clearance = br
            .cookies
            .iter()
            .find(|(n, _)| {
                let n = n.to_ascii_lowercase();
                n == "cf_clearance" || n == "datadome"
            })
            .cloned();
        let minted_ua = br
            .user_agent
            .clone()
            .expect("[clearance-spike] stealth path must surface the minted UA");

        let (cookie_name, cookie_value) = match clearance {
            Some(c) => c,
            None => {
                eprintln!(
                    "[clearance-spike] INCONCLUSIVE: browser did not obtain a cf_clearance/datadome \
                     cookie (challenge not solved). Cannot test replay."
                );
                return;
            }
        };

        // STEP 2 — replay the cookie + EXACT minted UA on a plain reqwest client
        // (a DIFFERENT TLS/JA3 fingerprint from the browser). Exactly ONE GET.
        let client = reqwest::Client::builder()
            .user_agent(minted_ua.clone())
            .build()
            .expect("reqwest client");
        let resp = client
            .get(&target)
            .header(reqwest::header::COOKIE, format!("{cookie_name}={cookie_value}"))
            .send()
            .await
            .expect("[clearance-spike] replay GET failed at transport layer");

        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();
        let body_lower = body.to_lowercase();

        // STEP 3 — VERDICT. A fresh challenge in the replayed response means the
        // cookie is fingerprint-bound (reqwest's TLS fingerprint differs from the
        // browser's); real content means the cookie replayed across clients.
        let still_challenged = ["just a moment", "/cdn-cgi/challenge-platform/", "_cf_chl_opt"]
            .iter()
            .any(|m| body_lower.contains(m));

        eprintln!("[clearance-spike] ===== cf_clearance cross-client replay VERDICT =====");
        eprintln!("[clearance-spike] replayed cookie: {cookie_name} (value redacted)");
        eprintln!("[clearance-spike] reqwest replay status: {status}, body_len: {}", body.len());
        if still_challenged || status == 403 {
            eprintln!(
                "[clearance-spike] VERDICT: REJECTED — fresh challenge / 403 on replay. The cookie \
                 is FINGERPRINT-BOUND (TLS JA3/HTTP2). Keep `enable_clearance_replay` default-OFF; \
                 fallback is keep-session-in-browser (Design 0005 §4.2 / R-3)."
            );
        } else {
            eprintln!(
                "[clearance-spike] VERDICT: ACCEPTED — real content on the reqwest replay. The \
                 cookie is REPLAYABLE across clients (session+UA+IP, not JA3-bound). The R4→R1 \
                 reuse path is viable; `enable_clearance_replay` can be turned on (Design 0005 §4.2)."
            );
        }
        eprintln!("[clearance-spike] =============================================================");

        // We assert only that the spike ran end-to-end; the verdict is the
        // finding, printed above, not an automated pass/fail.
        assert!(!cookie_value.is_empty(), "spike obtained a non-empty clearance cookie");
    }
}
