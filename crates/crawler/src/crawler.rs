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

    /// Crawl starting from seed URLs with concurrent fetching.
    ///
    /// Pops batches of URLs from the frontier and fetches them in parallel
    /// (up to `concurrency` at a time). Stops when frontier is empty,
    /// `max_pages` reached, or `time_limit` exceeded.
    pub async fn crawl(
        &self,
        seeds: &[&str],
        max_pages: usize,
        max_depth: u8,
        time_limit: Duration,
    ) -> Vec<CrawledPage> {
        use tokio::sync::Semaphore;

        let start = Instant::now();
        let concurrency = self.config.num_workers.max(4);
        let semaphore = Arc::new(Semaphore::new(concurrency));

        // Add seeds to frontier
        self.frontier.add_seeds(seeds);

        let crawled: Arc<tokio::sync::Mutex<Vec<CrawledPage>>> =
            Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let mut consecutive_empty = 0;
        let mut handles: Vec<tokio::task::JoinHandle<()>> = Vec::new();

        loop {
            // Check limits
            {
                let current = crawled.lock().await.len();
                if current >= max_pages || start.elapsed() >= time_limit {
                    break;
                }
            }

            // Pop a batch of URLs from frontier
            let mut batch = Vec::new();
            for _ in 0..concurrency {
                match self.frontier.pop() {
                    Some(entry) if entry.depth <= max_depth => batch.push(entry),
                    Some(_) => {} // too deep, skip
                    None => break,
                }
            }

            if batch.is_empty() {
                consecutive_empty += 1;
                if consecutive_empty > 3 {
                    break;
                }
                // Wait for in-flight fetches to potentially add new URLs
                tokio::time::sleep(Duration::from_millis(200)).await;
                continue;
            }
            consecutive_empty = 0;

            // Spawn concurrent fetch tasks for the batch
            for entry in batch {
                // Check robots.txt
                if self.config.respect_robots_txt && !self.robots.is_allowed(&entry.url) {
                    continue;
                }

                let sem = semaphore.clone();
                let fetcher = self.fetcher.clone();
                let frontier = self.frontier.clone();
                let throttle = self.throttle.clone();
                let crawled = crawled.clone();
                let max_pages = max_pages;

                handles.push(tokio::spawn(async move {
                    // Acquire semaphore slot
                    let _permit = match sem.acquire().await {
                        Ok(p) => p,
                        Err(_) => return,
                    };

                    // Check if we already have enough
                    if crawled.lock().await.len() >= max_pages {
                        return;
                    }

                    // Throttle per domain
                    throttle.wait(&entry.domain).await;

                    tracing::debug!(url = %entry.url, depth = entry.depth, "Fetching");
                    match fetcher.fetch(&entry.url).await {
                        Ok(result) => {
                            // Check if search results page
                            if let Some(search_results) = crate::search_results::parse_search_results(&result.final_url, &result.body) {
                                tracing::info!(
                                    url = %entry.url,
                                    results = search_results.len(),
                                    "Parsed search results — following links"
                                );
                                for sr in &search_results {
                                    frontier.push(&sr.url, entry.depth);
                                }
                                return;
                            }

                            // Extract links
                            let links = link_extractor::extract_links(&result.body, &result.final_url);
                            for link in &links {
                                if !link.is_external || entry.depth == 0 {
                                    frontier.push(&link.url, entry.depth + 1);
                                }
                            }

                            let page = CrawledPage {
                                url: entry.url.clone(),
                                final_url: result.final_url,
                                body: result.body,
                                content_type: result.content_type,
                                status: result.status,
                                response_time_ms: result.response_time_ms,
                                depth: entry.depth,
                                is_spa: result.is_spa,
                                links,
                            };

                            let mut pages = crawled.lock().await;
                            pages.push(page);
                            tracing::debug!(
                                url = %entry.url,
                                pages = pages.len(),
                                "Page crawled"
                            );
                        }
                        Err(Error::RateLimited { domain, retry_after_secs }) => {
                            tracing::warn!(domain, retry_after_secs, "Rate limited");
                            throttle.apply_backoff(&domain);
                            frontier.push(&entry.url, entry.depth);
                        }
                        Err(e) => {
                            tracing::warn!(url = %entry.url, error = %e, "Fetch failed");
                        }
                    }
                }));
            }

            // Clean up completed handles periodically
            handles.retain(|h| !h.is_finished());
        }

        // Wait for all in-flight fetches
        for handle in handles {
            let _ = handle.await;
        }

        let result = match Arc::try_unwrap(crawled) {
            Ok(mutex) => mutex.into_inner(),
            Err(arc) => arc.blocking_lock().clone(),
        };

        tracing::info!(
            total = result.len(),
            elapsed_ms = start.elapsed().as_millis(),
            "Crawl complete"
        );

        result
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
