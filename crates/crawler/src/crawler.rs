use futures::stream::FuturesUnordered;
use futures::StreamExt;
use std::sync::Arc;
use std::time::{Duration, Instant};
use web_search_common::config::CrawlerConfig;
use web_search_common::{Error, Result};

use crate::fetcher::Fetcher;
use crate::frontier::UrlFrontier;
use crate::link_extractor;
use crate::pagination;
use crate::robots::RobotsCache;
use crate::throttle::Throttle;

/// Crawled page ready for extraction and indexing.
#[derive(Debug, Clone)]
pub struct CrawledPage {
    pub url: String,
    pub final_url: String,
    pub body: String,
    pub content_type: String,
    pub status: u16,
    pub response_time_ms: u64,
    pub depth: u8,
    pub is_spa: bool,
    pub links: Vec<link_extractor::ExtractedLink>,
}

/// Multi-worker web crawler with rate limiting and robots.txt compliance.
pub struct Crawler {
    fetcher: Arc<Fetcher>,
    frontier: Arc<UrlFrontier>,
    throttle: Arc<Throttle>,
    robots: Arc<RobotsCache>,
    config: CrawlerConfig,
}

impl Crawler {
    pub fn new(config: CrawlerConfig) -> Result<Self> {
        let fetcher = Arc::new(Fetcher::new(&config)?);
        let frontier = Arc::new(UrlFrontier::new(50)); // default max 50 per domain
        let throttle = Arc::new(Throttle::new(config.requests_per_second_per_domain));
        let robots = Arc::new(RobotsCache::new(&config.user_agent));

        Ok(Self {
            fetcher,
            frontier,
            throttle,
            robots,
            config,
        })
    }

    /// Crawl starting from seed URLs with FuturesUnordered for max concurrency.
    ///
    /// Uses a streaming approach: pages are processed as they complete,
    /// not in batches. This eliminates head-of-line blocking where one slow
    /// fetch delays the entire batch.
    pub async fn crawl(
        &self,
        seeds: &[&str],
        max_pages: usize,
        max_depth: u8,
        time_limit: Duration,
    ) -> Vec<CrawledPage> {
        let start = Instant::now();
        let concurrency = self.config.num_workers.max(8).min(16); // 8-16 workers

        // Add seeds to frontier
        self.frontier.add_seeds(seeds);

        let mut crawled: Vec<CrawledPage> = Vec::new();
        let mut futures = FuturesUnordered::new();
        let mut consecutive_idle = 0u32;

        loop {
            // Check limits
            if crawled.len() >= max_pages || start.elapsed() >= time_limit {
                break;
            }

            // Fill up to concurrency with new tasks from frontier
            while futures.len() < concurrency {
                match self.frontier.pop() {
                    Some(entry) if entry.depth <= max_depth => {
                        if self.config.respect_robots_txt && !self.robots.is_allowed(&entry.url) {
                            continue;
                        }
                        let fetcher = self.fetcher.clone();
                        let frontier = self.frontier.clone();
                        let throttle = self.throttle.clone();

                        futures.push(tokio::spawn(async move {
                            throttle.wait(&entry.domain).await;
                            Self::fetch_and_process(fetcher, frontier, entry).await
                        }));
                    }
                    Some(_) => {} // too deep
                    None => break,
                }
            }

            if futures.is_empty() {
                consecutive_idle += 1;
                if consecutive_idle > 8 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(300)).await;
                continue;
            }
            consecutive_idle = 0;

            // Process next completed future (non-blocking drain)
            match tokio::time::timeout(Duration::from_millis(100), futures.next()).await {
                Ok(Some(Ok(Some(page)))) => {
                    tracing::debug!(url = %page.final_url, total = crawled.len() + 1, "Page crawled");
                    crawled.push(page);
                }
                Ok(Some(Ok(None))) => {} // search page or error, no content page
                Ok(Some(Err(e))) => {
                    tracing::debug!(error = %e, "Task panicked");
                }
                Ok(None) => {} // stream exhausted
                Err(_) => {} // timeout, loop continues to fill more futures
            }
        }

        // Drain remaining in-flight futures
        while let Some(result) = futures.next().await {
            if crawled.len() >= max_pages { break; }
            if let Ok(Some(page)) = result {
                crawled.push(page);
            }
        }

        tracing::info!(
            total = crawled.len(),
            elapsed_ms = start.elapsed().as_millis(),
            "Crawl complete"
        );

        crawled
    }

    /// Fetch a single URL, handle search result parsing, return content page or None.
    async fn fetch_and_process(
        fetcher: Arc<Fetcher>,
        frontier: Arc<UrlFrontier>,
        entry: crate::frontier::PrioritizedUrl,
    ) -> Option<CrawledPage> {
        tracing::debug!(url = %entry.url, depth = entry.depth, "Fetching");

        let result = match fetcher.fetch(&entry.url).await {
            Ok(r) => r,
            Err(Error::RateLimited { domain, .. }) => {
                tracing::warn!(domain, "Rate limited");
                frontier.push(&entry.url, entry.depth);
                return None;
            }
            Err(e) => {
                tracing::warn!(url = %entry.url, error = %e, "Fetch failed");
                return None;
            }
        };

        // Check if search results page
        if let Some(search_results) = crate::search_results::parse_search_results(&result.final_url, &result.body) {
            if !search_results.is_empty() {
                tracing::info!(url = %entry.url, results = search_results.len(), "Parsed search results — following links");
                for sr in &search_results {
                    frontier.push(&sr.url, entry.depth);
                }
                return None;
            }
            // Search engine returned 0 — try browser fallback
            if result.body.len() > 2000 {
                tracing::info!(url = %entry.url, "Search parser returned 0, trying browser fallback");
                if let Some(browser_result) = fetcher.fetch_via_browser(&entry.url).await {
                    if let Some(browser_search_results) = crate::search_results::parse_search_results(&browser_result.final_url, &browser_result.body) {
                        tracing::info!(url = %entry.url, results = browser_search_results.len(), "Browser fallback: parsed search results");
                        for sr in &browser_search_results {
                            frontier.push(&sr.url, entry.depth);
                        }
                    }
                }
            }
            return None;
        }

        // Check JSON API
        if result.content_type.to_lowercase().contains("json") {
            if let Some(search_results) = crate::search_results::try_parse_json_api(&result.body) {
                tracing::info!(url = %entry.url, results = search_results.len(), "Parsed JSON API — following links");
                for sr in &search_results {
                    frontier.push(&sr.url, entry.depth);
                }
                return None;
            }
        }

        // Extract links and return content page
        let links = link_extractor::extract_links(&result.body, &result.final_url);
        for link in &links {
            if !link.is_external || entry.depth == 0 {
                frontier.push(&link.url, entry.depth + 1);
            }
        }

        Some(CrawledPage {
            url: entry.url,
            final_url: result.final_url,
            body: result.body,
            content_type: result.content_type,
            status: result.status,
            response_time_ms: result.response_time_ms,
            depth: entry.depth,
            is_spa: result.is_spa,
            links,
        })
    }

    /// Fetch a single URL (for atomic `fetch_page` tool).
    pub async fn fetch_one(&self, url: &str) -> Result<CrawledPage> {
        let domain = url::Url::parse(url)
            .map(|u| u.host_str().unwrap_or("").to_string())
            .unwrap_or_default();

        self.throttle.wait(&domain).await;

        let result = self.fetcher.fetch(url).await?;
        let links = link_extractor::extract_links(&result.body, &result.final_url);

        Ok(CrawledPage {
            url: url.to_string(),
            final_url: result.final_url,
            body: result.body,
            content_type: result.content_type,
            status: result.status,
            response_time_ms: result.response_time_ms,
            depth: 0,
            is_spa: result.is_spa,
            links,
        })
    }

    /// Follow pagination from a starting URL.
    pub async fn paginate(&self, start_url: &str, max_pages: u32) -> Vec<CrawledPage> {
        let mut pages = Vec::new();
        let mut current_url = start_url.to_string();

        for page_num in 0..max_pages {
            match self.fetch_one(&current_url).await {
                Ok(page) => {
                    // Try to find next page link
                    let next = link_extractor::find_next_page_link(&page.body, &page.final_url)
                        .or_else(|| {
                            let pattern = pagination::detect_pagination(&page.final_url, &page.body);
                            pagination::next_page_url(&page.final_url, &pattern, page_num)
                        });

                    pages.push(page);

                    match next {
                        Some(next_url) => {
                            tracing::debug!(page = page_num + 1, next = %next_url, "Following pagination");
                            current_url = next_url;
                        }
                        None => {
                            tracing::debug!(page = page_num + 1, "No more pages");
                            break;
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(url = %current_url, error = %e, "Pagination fetch failed");
                    break;
                }
            }
        }

        pages
    }

    /// Access the frontier for external URL management.
    pub fn frontier(&self) -> &UrlFrontier {
        &self.frontier
    }

    async fn maybe_fetch_robots(&self, domain: &str) {
        if !self.config.respect_robots_txt {
            return;
        }
        // Only fetch if not cached
        if self.robots.is_allowed(&format!("https://{domain}/robots-check")) {
            let robots_url = format!("https://{domain}/robots.txt");
            if let Ok(result) = self.fetcher.fetch(&robots_url).await {
                if result.status == 200 {
                    self.robots.parse_and_cache(domain, &result.body);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> CrawlerConfig {
        CrawlerConfig {
            num_workers: 2,
            max_concurrent_connections: 10,
            requests_per_second_per_domain: 10.0,
            request_timeout_secs: 10,
            user_agent: "TestBot/1.0".to_string(),
            respect_robots_txt: false,
            enable_browser: false,
            max_retries: 1,
            backoff_base_ms: 100,
        }
    }

    #[test]
    fn crawler_creates_successfully() {
        let crawler = Crawler::new(test_config());
        assert!(crawler.is_ok());
    }

    #[test]
    fn frontier_accessible() {
        let crawler = Crawler::new(test_config()).unwrap();
        assert!(crawler.frontier().is_empty());
    }
}
