use crate::browser::BrowserPool;
use moka::future::Cache;
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

        let browser_pool = BrowserPool::new();

        Ok(Self {
            client,
            max_retries: config.max_retries,
            backoff_base_ms: config.backoff_base_ms,
            cache,
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            browser_pool,
            enable_browser: config.enable_browser,
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
}
