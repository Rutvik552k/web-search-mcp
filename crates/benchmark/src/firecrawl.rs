//! Firecrawl baseline adapter for the G4 head-to-head comparison (TASKS.md 0.6,
//! GOAL.md §G4). BASELINE-COMPARISON ONLY — this module is never wired into the
//! server runtime or any default build. The `benchmark` binary is the sole
//! caller, and only when an operator-supplied key is present.
//!
//! # Verified Firecrawl API contract (verified 2026-06-18)
//!
//! Sources (official docs + repo):
//!   * Scrape:  https://docs.firecrawl.dev/api-reference/endpoint/scrape
//!   * Crawl:   https://docs.firecrawl.dev/api-reference/endpoint/crawl-post
//!   * Repo:    https://github.com/firecrawl/firecrawl
//!   * Pricing: https://www.firecrawl.dev/pricing
//!
//! Current API version is **v2** (base `https://api.firecrawl.dev/v2`). Auth is
//! `Authorization: Bearer <FIRECRAWL_API_KEY>` on every call.
//!
//! ## `POST /v2/scrape`
//! Request body: `{ "url": "<abs url>", "formats": ["markdown"], ... }`.
//! Response (success):
//! ```json
//! { "success": true,
//!   "data": { "markdown": "# ...", "html": "...",
//!             "metadata": { "title": "...", "sourceURL": "...", "statusCode": 200 } } }
//! ```
//!
//! ## `POST /v2/crawl` (async start)
//! Request body: `{ "url": "...", "limit": N, "scrapeOptions": { "formats": ["markdown"] } }`.
//! Response: `{ "success": true, "id": "<job id>", "url": "<status url>" }`.
//!
//! ## `GET /v2/crawl/{id}` (poll)
//! Response: `{ "status": "completed"|"scraping"|"failed", "total": N,
//!             "completed": M, "creditsUsed": K, "data": [ { "markdown": "...",
//!             "metadata": { ... } }, ... ] }`. Poll until `status == "completed"`.
//!
//! ## Pricing → cost metric (verified 2026-06-18, firecrawl.dev/pricing)
//! 1 credit = 1 scraped/crawled page (base). Standard plan = $83 / 100,000
//! credits ⇒ **$0.83 per 1,000 pages** (cheapest sustained main tier; Hobby
//! $16/5,000 = $3.20/1k is smaller-volume). We use the Standard rate as the
//! representative $/1k for the comparison and record the basis explicitly.
//! See [`COST_PER_1K_PAGES_USD`].
//!
//! UNVERIFIED / flagged for operator escalation:
//!   * Firecrawl Terms of Service may restrict using the service to benchmark
//!     against competitors. The real-key run is operator-invoked precisely so
//!     the operator owns that decision — see report.
//!   * Enhanced/stealth proxy and JSON extraction cost extra credits (4/page).
//!     Our cost figure assumes the base 1-credit path (markdown only), matching
//!     how we run the comparison. If the operator enables stealth proxy to clear
//!     the blocked subset, multiply the blocked-subset $/page by ~4.

use std::time::Duration;

use serde::Deserialize;

/// Default Firecrawl v2 API base. Override via [`FirecrawlConfig::base_url`]
/// (the contract-mocked tests point this at a local mock server).
pub const DEFAULT_BASE_URL: &str = "https://api.firecrawl.dev/v2";

/// Environment variable the operator sets to enable the baseline run. Read at
/// runtime only; never hardcoded, never logged, never cached to disk.
pub const API_KEY_ENV: &str = "FIRECRAWL_API_KEY";

/// $ per 1,000 pages for Firecrawl, computed from verified pricing
/// (2026-06-18): Standard plan $83 / 100,000 credits, 1 credit = 1 page.
/// Basis recorded in the report so the operator can re-derive it.
pub const COST_PER_1K_PAGES_USD: f64 = 0.83;

/// How we describe the cost basis in the report (single source of truth).
pub const COST_BASIS_NOTE: &str = "Firecrawl Standard plan $83 / 100,000 credits, \
    1 credit = 1 page (base markdown path) ⇒ $0.83 / 1k pages (verified 2026-06-18, \
    firecrawl.dev/pricing). Enhanced/stealth proxy = 4 credits/page.";

// ── Verified response shapes (deserialize only the fields we score) ──────────

/// `POST /v2/scrape` success body. Unknown fields are ignored so the adapter is
/// tolerant of additive contract changes (microservices rule: tolerate unknown).
#[derive(Debug, Deserialize)]
pub struct ScrapeResponse {
    #[serde(default)]
    pub success: bool,
    #[serde(default)]
    pub data: Option<ScrapeData>,
    /// Present on error bodies (`{ "success": false, "error": "..." }`).
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ScrapeData {
    #[serde(default)]
    pub markdown: Option<String>,
    #[serde(default)]
    pub html: Option<String>,
    #[serde(default)]
    pub metadata: Option<PageMetadata>,
}

#[derive(Debug, Deserialize)]
pub struct PageMetadata {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(rename = "sourceURL", default)]
    pub source_url: Option<String>,
    #[serde(rename = "statusCode", default)]
    pub status_code: Option<u16>,
}

/// `POST /v2/crawl` start-job body.
#[derive(Debug, Deserialize)]
pub struct CrawlStartResponse {
    #[serde(default)]
    pub success: bool,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
}

/// `GET /v2/crawl/{id}` poll body.
#[derive(Debug, Deserialize)]
pub struct CrawlStatusResponse {
    /// "scraping" | "completed" | "failed" | "cancelled".
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub total: u64,
    #[serde(default)]
    pub completed: u64,
    #[serde(rename = "creditsUsed", default)]
    pub credits_used: u64,
    #[serde(default)]
    pub data: Vec<ScrapeData>,
}

impl CrawlStatusResponse {
    pub fn is_terminal(&self) -> bool {
        matches!(self.status.as_str(), "completed" | "failed" | "cancelled")
    }
    pub fn is_completed(&self) -> bool {
        self.status == "completed"
    }
}

/// The single page-level outcome the harness scores. Mirrors the coverage rule
/// in `metrics.rs`: a page is clean when its extracted text passes
/// `metrics::is_clean`. We pull text from `markdown` (Firecrawl's primary
/// LLM-ready output), falling back to `html` stripped of tags is NOT done here —
/// we score markdown only, the like-for-like "clean main content" Firecrawl
/// advertises, so the comparison is apples-to-apples with our `body_text`.
#[derive(Debug, Clone)]
pub struct FirecrawlPage {
    pub url: String,
    /// Extracted main-content text (markdown). Empty string on failure.
    pub text: String,
    /// HTTP status Firecrawl reported for the upstream page, if any.
    pub status_code: Option<u16>,
    /// Set when the scrape failed (call error, non-2xx, success:false).
    pub error: Option<String>,
}

// ── Parsing helpers (pure, unit-tested) ──────────────────────────────────────

/// Parse a `/v2/scrape` response body into a [`FirecrawlPage`] for `url`.
/// Pure function over the verified contract — the core of the contract test.
pub fn parse_scrape(url: &str, body: &str) -> FirecrawlPage {
    match serde_json::from_str::<ScrapeResponse>(body) {
        Ok(resp) if resp.success => {
            let data = resp.data;
            let text = data
                .as_ref()
                .and_then(|d| d.markdown.clone())
                .unwrap_or_default();
            let status_code = data
                .as_ref()
                .and_then(|d| d.metadata.as_ref())
                .and_then(|m| m.status_code);
            FirecrawlPage {
                url: url.to_string(),
                text,
                status_code,
                error: None,
            }
        }
        Ok(resp) => FirecrawlPage {
            url: url.to_string(),
            text: String::new(),
            status_code: None,
            error: Some(
                resp.error
                    .unwrap_or_else(|| "scrape returned success=false".to_string()),
            ),
        },
        Err(e) => FirecrawlPage {
            url: url.to_string(),
            text: String::new(),
            status_code: None,
            error: Some(format!("unparseable scrape body: {e}")),
        },
    }
}

/// Parse a `/v2/crawl/{id}` poll body. Pure; the core of the crawl contract test.
pub fn parse_crawl_status(body: &str) -> Result<CrawlStatusResponse, String> {
    serde_json::from_str::<CrawlStatusResponse>(body)
        .map_err(|e| format!("unparseable crawl status body: {e}"))
}

/// Map a completed crawl-status payload to per-page outcomes.
pub fn crawl_status_to_pages(resp: &CrawlStatusResponse) -> Vec<FirecrawlPage> {
    resp.data
        .iter()
        .map(|d| {
            let url = d
                .metadata
                .as_ref()
                .and_then(|m| m.source_url.clone())
                .unwrap_or_default();
            FirecrawlPage {
                url,
                text: d.markdown.clone().unwrap_or_default(),
                status_code: d.metadata.as_ref().and_then(|m| m.status_code),
                error: None,
            }
        })
        .collect()
}

// ── Live client (operator-invoked path only; never a CI test target) ─────────

/// Configuration for a live Firecrawl run. Built only when the env key is
/// present; see [`FirecrawlClient::from_env`].
#[derive(Clone)]
pub struct FirecrawlConfig {
    base_url: String,
    /// API key. Held in memory only — never written to disk, never logged.
    api_key: String,
    request_timeout: Duration,
    /// Crawl poll interval / overall budget.
    poll_interval: Duration,
    poll_max: Duration,
}

impl FirecrawlConfig {
    /// Build from the operator key in [`API_KEY_ENV`]. Returns `None` (clean
    /// skip) when the key is absent — the harness then runs ours-only and writes
    /// a placeholder Firecrawl column. Never panics, never logs the key.
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var(API_KEY_ENV).ok()?;
        let api_key = api_key.trim().to_string();
        if api_key.is_empty() {
            return None;
        }
        Some(Self {
            base_url: std::env::var("FIRECRAWL_BASE_URL")
                .unwrap_or_else(|_| DEFAULT_BASE_URL.to_string()),
            api_key,
            request_timeout: Duration::from_secs(120),
            poll_interval: Duration::from_secs(3),
            poll_max: Duration::from_secs(600),
        })
    }

    /// Construct an explicit config (used by tests pointing at a local mock).
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            request_timeout: Duration::from_secs(30),
            poll_interval: Duration::from_millis(50),
            poll_max: Duration::from_secs(10),
        }
    }
}

/// Live Firecrawl HTTP client. Used only on the operator-invoked real run.
pub struct FirecrawlClient {
    cfg: FirecrawlConfig,
    http: reqwest::Client,
}

impl FirecrawlClient {
    pub fn new(cfg: FirecrawlConfig) -> anyhow::Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(cfg.request_timeout)
            .build()?;
        Ok(Self { cfg, http })
    }

    /// Build from the operator env key. `Ok(None)` ⇒ no key ⇒ clean skip.
    pub fn from_env() -> anyhow::Result<Option<Self>> {
        match FirecrawlConfig::from_env() {
            Some(cfg) => Ok(Some(Self::new(cfg)?)),
            None => Ok(None),
        }
    }

    /// Scrape a single URL via `POST /v2/scrape`. Network errors / non-2xx are
    /// folded into `FirecrawlPage.error` (a failed page is a coverage miss, not
    /// a harness crash — one bad URL must not wedge the run).
    pub async fn scrape(&self, url: &str) -> FirecrawlPage {
        let endpoint = format!("{}/scrape", self.cfg.base_url);
        let body = serde_json::json!({ "url": url, "formats": ["markdown"], "onlyMainContent": true });
        let resp = self
            .http
            .post(&endpoint)
            .bearer_auth(&self.cfg.api_key)
            .json(&body)
            .send()
            .await;
        match resp {
            Ok(r) => {
                let status = r.status();
                match r.text().await {
                    Ok(text) if status.is_success() => parse_scrape(url, &text),
                    Ok(text) => FirecrawlPage {
                        url: url.to_string(),
                        text: String::new(),
                        status_code: Some(status.as_u16()),
                        // Do NOT echo the raw body unbounded; truncate. Never
                        // includes the key (key is only in the request header).
                        error: Some(format!("HTTP {}: {}", status.as_u16(), truncate(&text, 160))),
                    },
                    Err(e) => FirecrawlPage {
                        url: url.to_string(),
                        text: String::new(),
                        status_code: Some(status.as_u16()),
                        error: Some(format!("read body failed: {e}")),
                    },
                }
            }
            Err(e) => FirecrawlPage {
                url: url.to_string(),
                text: String::new(),
                status_code: None,
                error: Some(format!("request failed: {e}")),
            },
        }
    }

    /// Start a crawl job and poll until terminal. Returns all crawled pages.
    /// Used for the multi-page crawl comparison; the per-URL coverage run uses
    /// [`scrape`] (one page per `urls.jsonl` line, like-for-like with ours).
    pub async fn crawl(&self, url: &str, limit: u64) -> anyhow::Result<Vec<FirecrawlPage>> {
        let start_ep = format!("{}/crawl", self.cfg.base_url);
        let body = serde_json::json!({
            "url": url,
            "limit": limit,
            "scrapeOptions": { "formats": ["markdown"], "onlyMainContent": true }
        });
        let start_text = self
            .http
            .post(&start_ep)
            .bearer_auth(&self.cfg.api_key)
            .json(&body)
            .send()
            .await?
            .text()
            .await?;
        let start: CrawlStartResponse = serde_json::from_str(&start_text)
            .map_err(|e| anyhow::anyhow!("unparseable crawl-start body: {e}"))?;
        let job_id = start
            .id
            .filter(|_| start.success)
            .ok_or_else(|| anyhow::anyhow!("crawl start failed: {:?}", start.error))?;

        let poll_ep = format!("{}/crawl/{}", self.cfg.base_url, job_id);
        let deadline = std::time::Instant::now() + self.cfg.poll_max;
        loop {
            let text = self
                .http
                .get(&poll_ep)
                .bearer_auth(&self.cfg.api_key)
                .send()
                .await?
                .text()
                .await?;
            let status = parse_crawl_status(&text).map_err(|e| anyhow::anyhow!(e))?;
            if status.is_completed() {
                return Ok(crawl_status_to_pages(&status));
            }
            if status.is_terminal() {
                anyhow::bail!("crawl ended non-completed: status={}", status.status);
            }
            if std::time::Instant::now() >= deadline {
                anyhow::bail!("crawl poll timed out after {:?}", self.cfg.poll_max);
            }
            tokio::time::sleep(self.cfg.poll_interval).await;
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    let one_line = s.replace('\n', " ");
    if one_line.chars().count() <= max {
        one_line
    } else {
        let t: String = one_line.chars().take(max).collect();
        format!("{t}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Fixtures captured from the VERIFIED contract (docs.firecrawl.dev,
    // 2026-06-18). NO real network — this is the contract-verified mock per the
    // testing rule (no real external services in tests).
    const SCRAPE_OK: &str = r##"{
        "success": true,
        "data": {
            "markdown": "# Example Domain\n\nThis domain is for use in illustrative examples in documents. You may use this domain in literature without prior coordination or asking for permission. More body text follows to clear the clean threshold comfortably and prove the extractor returned the right article rather than navigation chrome boilerplate text here.",
            "html": "<h1>Example Domain</h1>",
            "metadata": { "title": "Example Domain", "sourceURL": "https://example.com/", "statusCode": 200 }
        }
    }"##;

    const SCRAPE_BLOCKED: &str = r##"{
        "success": true,
        "data": {
            "markdown": "Attention Required! Sorry, you have been blocked.",
            "metadata": { "title": "Just a moment...", "sourceURL": "https://protected.example.com/", "statusCode": 403 }
        }
    }"##;

    const SCRAPE_ERR: &str = r##"{ "success": false, "error": "This website is no longer supported" }"##;

    const CRAWL_RUNNING: &str = r##"{ "status": "scraping", "total": 10, "completed": 3, "creditsUsed": 3, "data": [] }"##;

    const CRAWL_DONE: &str = r##"{
        "status": "completed", "total": 2, "completed": 2, "creditsUsed": 2,
        "data": [
            { "markdown": "Page one body text that is long enough to pass the two hundred character clean-content threshold so that this counts as a clean page in the coverage metric, padding padding padding padding padding here we go.", "metadata": { "sourceURL": "https://site.example/a", "statusCode": 200 } },
            { "markdown": "short", "metadata": { "sourceURL": "https://site.example/b", "statusCode": 200 } }
        ]
    }"##;

    #[test]
    fn parse_scrape_extracts_markdown_and_status() {
        let page = parse_scrape("https://example.com/", SCRAPE_OK);
        assert!(page.error.is_none());
        assert_eq!(page.status_code, Some(200));
        assert!(page.text.contains("Example Domain"));
        // The scored text must clear the same clean threshold our pipeline uses.
        assert!(crate::metrics::is_clean(&page.text, Some("Example Domain")));
    }

    #[test]
    fn parse_scrape_blocked_page_text_is_short_and_marker_absent() {
        let page = parse_scrape("https://protected.example.com/", SCRAPE_BLOCKED);
        assert!(page.error.is_none()); // Firecrawl returned success=true...
        assert_eq!(page.status_code, Some(403));
        // ...but the body is a block notice: not clean main content.
        assert!(!crate::metrics::is_clean(&page.text, Some("Notion")));
    }

    #[test]
    fn parse_scrape_error_body_is_a_miss() {
        let page = parse_scrape("https://gone.example/", SCRAPE_ERR);
        assert!(page.error.is_some());
        assert!(page.text.is_empty());
        assert!(!crate::metrics::is_clean(&page.text, None));
    }

    #[test]
    fn parse_scrape_garbage_body_does_not_panic() {
        let page = parse_scrape("https://x.example/", "not json at all");
        assert!(page.error.is_some());
        assert!(page.text.is_empty());
    }

    #[test]
    fn parse_crawl_status_running_is_not_terminal() {
        let s = parse_crawl_status(CRAWL_RUNNING).expect("parse");
        assert_eq!(s.status, "scraping");
        assert!(!s.is_terminal());
        assert!(!s.is_completed());
        assert_eq!(s.completed, 3);
        assert_eq!(s.credits_used, 3);
    }

    #[test]
    fn parse_crawl_status_completed_yields_pages() {
        let s = parse_crawl_status(CRAWL_DONE).expect("parse");
        assert!(s.is_completed());
        assert_eq!(s.credits_used, 2);
        let pages = crawl_status_to_pages(&s);
        assert_eq!(pages.len(), 2);
        assert_eq!(pages[0].url, "https://site.example/a");
        // First page clean, second too short.
        assert!(crate::metrics::is_clean(&pages[0].text, None));
        assert!(!crate::metrics::is_clean(&pages[1].text, None));
    }

    #[test]
    fn parse_crawl_status_garbage_errors_cleanly() {
        assert!(parse_crawl_status("}{").is_err());
    }

    // Env-var tests mutate process-global state; serialize them so they cannot
    // race each other (deterministic, no flakiness at the source).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn from_env_skips_cleanly_when_key_absent() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Save/restore so we don't disturb a real operator env in the same proc.
        let prev = std::env::var(API_KEY_ENV).ok();
        unsafe { std::env::remove_var(API_KEY_ENV) };
        assert!(FirecrawlConfig::from_env().is_none());
        assert!(FirecrawlClient::from_env().expect("no error").is_none());
        if let Some(v) = prev {
            unsafe { std::env::set_var(API_KEY_ENV, v) };
        }
    }

    #[test]
    fn from_env_blank_key_is_treated_as_absent() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var(API_KEY_ENV).ok();
        unsafe { std::env::set_var(API_KEY_ENV, "   ") };
        assert!(FirecrawlConfig::from_env().is_none());
        match prev {
            Some(v) => unsafe { std::env::set_var(API_KEY_ENV, v) },
            None => unsafe { std::env::remove_var(API_KEY_ENV) },
        }
    }

    #[test]
    fn cost_per_1k_matches_verified_pricing() {
        // Standard: $83 / 100_000 credits, 1 credit = 1 page.
        let derived = 83.0 / 100_000.0 * 1000.0;
        assert!((COST_PER_1K_PAGES_USD - derived).abs() < 1e-9);
    }
}
