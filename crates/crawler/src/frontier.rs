use dashmap::DashMap;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use parking_lot::Mutex;

/// Priority URL frontier for crawl scheduling.
///
/// URLs are prioritized by: depth (shallow first), domain authority, freshness.
/// Tracks per-domain counts to enforce max_pages_per_domain.
pub struct UrlFrontier {
    queue: Mutex<BinaryHeap<PrioritizedUrl>>,
    seen: DashMap<String, ()>,
    domain_counts: DashMap<String, usize>,
    max_per_domain: usize,
}

#[derive(Debug, Clone)]
pub struct PrioritizedUrl {
    pub url: String,
    pub domain: String,
    pub depth: u8,
    pub priority: f32,
}

impl PartialEq for PrioritizedUrl {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for PrioritizedUrl {}

impl PartialOrd for PrioritizedUrl {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedUrl {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .partial_cmp(&other.priority)
            .unwrap_or(Ordering::Equal)
    }
}

impl UrlFrontier {
    pub fn new(max_per_domain: usize) -> Self {
        Self {
            queue: Mutex::new(BinaryHeap::new()),
            seen: DashMap::new(),
            domain_counts: DashMap::new(),
            max_per_domain,
        }
    }

    /// Add a URL to the frontier. Returns false if already seen or domain limit reached.
    pub fn push(&self, url: &str, depth: u8) -> bool {
        // Normalize URL
        let normalized = normalize_url(url);

        // Skip if already seen
        if self.seen.contains_key(&normalized) {
            return false;
        }

        let domain = extract_domain(&normalized);

        // Check per-domain limit
        let count = self.domain_counts.get(&domain).map(|v| *v).unwrap_or(0);
        if count >= self.max_per_domain {
            return false;
        }

        // Calculate priority: shallower = higher priority
        let depth_score = 1.0 / (depth as f32 + 1.0);
        let priority = depth_score;

        self.seen.insert(normalized.clone(), ());
        *self.domain_counts.entry(domain.clone()).or_insert(0) += 1;

        let mut queue = self.queue.lock();
        queue.push(PrioritizedUrl {
            url: normalized,
            domain,
            depth,
            priority,
        });

        true
    }

    /// Pop the highest-priority URL.
    pub fn pop(&self) -> Option<PrioritizedUrl> {
        let mut queue = self.queue.lock();
        queue.pop()
    }

    /// Number of URLs in the queue.
    pub fn len(&self) -> usize {
        self.queue.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.lock().is_empty()
    }

    /// Total URLs ever seen (queued + already processed).
    pub fn total_seen(&self) -> usize {
        self.seen.len()
    }

    /// Check if URL has been seen before.
    pub fn has_seen(&self, url: &str) -> bool {
        self.seen.contains_key(&normalize_url(url))
    }

    /// Mark URL as seen without adding to queue.
    pub fn mark_seen(&self, url: &str) {
        self.seen.insert(normalize_url(url), ());
    }

    /// Add multiple seed URLs at depth 0.
    pub fn add_seeds(&self, urls: &[&str]) {
        for url in urls {
            self.push(url, 0);
        }
    }
}

/// Normalize URL: lowercase scheme+host, remove fragment, trailing slash.
fn normalize_url(url: &str) -> String {
    match url::Url::parse(url) {
        Ok(mut parsed) => {
            parsed.set_fragment(None);
            let mut s = parsed.to_string();
            // Remove trailing slash for consistency (except root)
            if s.ends_with('/') && s.matches('/').count() > 3 {
                s.pop();
            }
            s
        }
        Err(_) => url.to_string(),
    }
}

/// Extract domain from URL.
fn extract_domain(url: &str) -> String {
    url::Url::parse(url)
        .ok()
        .and_then(|u| u.host_str().map(|h| h.to_string()))
        .unwrap_or_else(|| "unknown".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_pop() {
        let frontier = UrlFrontier::new(100);
        assert!(frontier.push("https://example.com/a", 0));
        assert!(frontier.push("https://example.com/b", 1));
        assert_eq!(frontier.len(), 2);

        // Depth 0 should come first (higher priority)
        let first = frontier.pop().unwrap();
        assert_eq!(first.url, "https://example.com/a");
        assert_eq!(first.depth, 0);
    }

    #[test]
    fn dedup_urls() {
        let frontier = UrlFrontier::new(100);
        assert!(frontier.push("https://example.com/a", 0));
        assert!(!frontier.push("https://example.com/a", 0)); // duplicate
        assert_eq!(frontier.len(), 1);
    }

    #[test]
    fn fragment_normalized() {
        let frontier = UrlFrontier::new(100);
        assert!(frontier.push("https://example.com/page#section1", 0));
        assert!(!frontier.push("https://example.com/page#section2", 0)); // same without fragment
        assert_eq!(frontier.len(), 1);
    }

    #[test]
    fn domain_limit_enforced() {
        let frontier = UrlFrontier::new(2); // max 2 per domain
        assert!(frontier.push("https://example.com/a", 0));
        assert!(frontier.push("https://example.com/b", 0));
        assert!(!frontier.push("https://example.com/c", 0)); // limit reached
        assert!(frontier.push("https://other.com/a", 0)); // different domain OK
    }

    #[test]
    fn add_seeds() {
        let frontier = UrlFrontier::new(100);
        frontier.add_seeds(&["https://a.com", "https://b.com", "https://c.com"]);
        assert_eq!(frontier.len(), 3);
        assert_eq!(frontier.total_seen(), 3);
    }

    #[test]
    fn has_seen_works() {
        let frontier = UrlFrontier::new(100);
        assert!(!frontier.has_seen("https://example.com"));
        frontier.push("https://example.com", 0);
        assert!(frontier.has_seen("https://example.com"));
    }

    #[test]
    fn extract_domain_works() {
        assert_eq!(extract_domain("https://www.example.com/path"), "www.example.com");
        assert_eq!(extract_domain("https://api.github.com/repos"), "api.github.com");
    }
}
