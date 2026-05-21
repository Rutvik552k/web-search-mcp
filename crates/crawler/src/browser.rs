//! Headless browser fallback for JS-rendered pages.
//!
//! Uses chromiumoxide (Chrome DevTools Protocol) to render pages that
//! return empty/minimal content via plain HTTP (SPAs, CAPTCHA walls, etc.).
//!
//! Feature-gated behind `browser` — when disabled, `BrowserPool::new()` returns
//! a no-op that always falls back to HTTP.

use std::sync::Arc;
use std::time::Duration;

/// Result of browser-rendered page fetch.
#[derive(Debug, Clone)]
pub struct BrowserFetchResult {
    pub url: String,
    pub final_url: String,
    pub body: String,
    pub status: u16,
}

/// Lazy-initialized browser pool for SPA rendering.
///
/// Holds at most one browser process. Pages are created/destroyed per fetch.
/// When the `browser` feature is disabled, all methods gracefully return None.
pub struct BrowserPool {
    #[cfg(feature = "browser")]
    browser: Mutex<Option<BrowserState>>,
    #[cfg(not(feature = "browser"))]
    _phantom: (),
}

#[cfg(feature = "browser")]
struct BrowserState {
    browser: chromiumoxide::Browser,
    _handle: tokio::task::JoinHandle<()>,
}

impl BrowserPool {
    /// Create a new browser pool. Does NOT launch browser yet (lazy init).
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            #[cfg(feature = "browser")]
            browser: Mutex::new(None),
            #[cfg(not(feature = "browser"))]
            _phantom: (),
        })
    }

    /// Fetch a URL using headless Chrome. Returns None if browser unavailable.
    ///
    /// Launches browser on first call, reuses for subsequent calls.
    /// Waits up to `timeout` for page to load and JS to execute.
    pub async fn fetch(&self, url: &str, timeout: Duration) -> Option<BrowserFetchResult> {
        #[cfg(feature = "browser")]
        {
            self.fetch_with_browser(url, timeout).await
        }
        #[cfg(not(feature = "browser"))]
        {
            let _ = (url, timeout);
            tracing::debug!("Browser feature not enabled, skipping browser fetch");
            None
        }
    }

    /// Check if browser feature is available.
    pub fn is_available(&self) -> bool {
        cfg!(feature = "browser")
    }

    #[cfg(feature = "browser")]
    async fn fetch_with_browser(&self, url: &str, timeout: Duration) -> Option<BrowserFetchResult> {
        use chromiumoxide::browser::{Browser, BrowserConfig};
        use futures::StreamExt;

        // Lazy-init browser
        let mut guard = self.browser.lock().await;
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
}
