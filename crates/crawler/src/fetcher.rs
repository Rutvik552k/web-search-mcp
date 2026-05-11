use reqwest::{Client, StatusCode};
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
}

/// HTTP fetcher with retry, headers, and SPA detection.
pub struct Fetcher {
    client: Client,
    max_retries: u32,
    backoff_base_ms: u64,
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

        Ok(Self {
            client,
            max_retries: config.max_retries,
            backoff_base_ms: config.backoff_base_ms,
        })
    }

    /// Fetch a URL with retries and exponential backoff.
    pub async fn fetch(&self, url: &str) -> Result<FetchResult> {
        let mut last_err = None;

        for attempt in 0..=self.max_retries {
            if attempt > 0 {
                let delay = self.backoff_base_ms * 2u64.pow(attempt - 1);
                tracing::debug!(url, attempt, delay_ms = delay, "Retrying");
                tokio::time::sleep(Duration::from_millis(delay)).await;
            }

            match self.fetch_once(url).await {
                Ok(result) => return Ok(result),
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
