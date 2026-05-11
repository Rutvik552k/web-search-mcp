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

    /// Crawl starting from seed URLs. Returns crawled pages via channel.
    ///
    /// Spawns `num_workers` concurrent fetchers. Stops when frontier is empty
    /// or `max_pages` reached or `time_limit` exceeded.
    pub async fn crawl(
        &self,
        seeds: &[&str],
        max_pages: usize,
        max_depth: u8,
        time_limit: Duration,
    ) -> Vec<CrawledPage> {
        let start = Instant::now();

        // Add seeds to frontier
        self.frontier.add_seeds(seeds);

        let mut crawled: Vec<CrawledPage> = Vec::new();
        let mut consecutive_empty = 0;

        while crawled.len() < max_pages && start.elapsed() < time_limit {
            // Pop next URL from frontier
            let entry = match self.frontier.pop() {
                Some(e) => e,
                None => {
                    consecutive_empty += 1;
                    if consecutive_empty > 3 {
                        break; // frontier exhausted
                    }
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    continue;
                }
            };
            consecutive_empty = 0;

            // Skip if depth exceeded
            if entry.depth > max_depth {
                continue;
            }

            // Check robots.txt
            if self.config.respect_robots_txt && !self.robots.is_allowed(&entry.url) {
                tracing::debug!(url = %entry.url, "Blocked by robots.txt");
                continue;
            }

            // Throttle
            self.throttle.wait(&entry.domain).await;

            // Fetch
            tracing::debug!(url = %entry.url, depth = entry.depth, "Fetching");
            match self.fetcher.fetch(&entry.url).await {
                Ok(result) => {
                    // Fetch robots.txt for domain if not cached (first visit)
                    self.maybe_fetch_robots(&entry.domain).await;

                    // Extract links and add to frontier
                    let links = link_extractor::extract_links(&result.body, &result.final_url);
                    for link in &links {
                        if !link.is_external || entry.depth == 0 {
                            // Follow internal links and external links from seeds
                            self.frontier.push(&link.url, entry.depth + 1);
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

                    crawled.push(page);
                    tracing::debug!(
                        url = %entry.url,
                        pages = crawled.len(),
                        frontier = self.frontier.len(),
                        "Page crawled"
                    );
                }
                Err(Error::RateLimited { domain, retry_after_secs }) => {
                    tracing::warn!(domain, retry_after_secs, "Rate limited, applying backoff");
                    self.throttle.apply_backoff(&domain);
                    // Re-queue the URL
                    self.frontier.push(&entry.url, entry.depth);
                }
                Err(e) => {
                    tracing::warn!(url = %entry.url, error = %e, "Fetch failed");
                }
            }
        }

        tracing::info!(
            total = crawled.len(),
            elapsed_ms = start.elapsed().as_millis(),
            "Crawl complete"
        );

        crawled
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
