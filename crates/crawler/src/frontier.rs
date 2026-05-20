use dashmap::DashMap;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use parking_lot::Mutex;

/// Priority URL frontier for crawl scheduling.
///
/// URLs scored by: depth (shallow first) + source authority + domain diversity.
/// Tracks per-domain counts to enforce max_pages_per_domain.
pub struct UrlFrontier {
    queue: Mutex<BinaryHeap<PrioritizedUrl>>,
    seen: DashMap<String, ()>,
    domain_counts: DashMap<String, usize>,
    max_per_domain: usize,
}

/// Known high-authority domains get priority boost in the frontier.
fn domain_authority_bonus(domain: &str) -> f32 {
    // Tier 1: academic, government, major reference
    let tier1 = [".gov", ".edu", "nature.com", "arxiv.org", "pubmed.", "ieee.org", "who.int", "nasa.gov"];
    if tier1.iter().any(|t| domain.contains(t)) { return 0.3; }

    // Tier 2: established tech/news
    let tier2 = ["wikipedia.org", "stackoverflow.com", "github.com", "bbc.", "reuters.",
                  "nytimes.com", "docs.rs", "developer.mozilla.org", "rust-lang.org"];
    if tier2.iter().any(|t| domain.contains(t)) { return 0.2; }

    // Tier 3: known content sites
    let tier3 = ["medium.com", "dev.to", "reddit.com", "hackernews", "substack.com"];
    if tier3.iter().any(|t| domain.contains(t)) { return 0.1; }

    0.0
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

    /// Add a URL with optional anchor text and query for relevance scoring.
    pub fn push_with_context(&self, url: &str, depth: u8, anchor_text: &str, query: &str) -> bool {
        let normalized = normalize_url(url);
        if self.seen.contains_key(&normalized) {
            return false;
        }
        let domain = extract_domain(&normalized);
        let count = self.domain_counts.get(&domain).map(|v| *v).unwrap_or(0);
        if count >= self.max_per_domain {
            return false;
        }

        // Composite priority: depth + authority + diversity + URL relevance prediction
        let depth_score = 1.0 / (depth as f32 + 1.0);
        let authority = domain_authority_bonus(&domain);
        let diversity = 1.0 / (count as f32 + 1.0);

        // URL relevance prediction: score URL without fetching the page.
        // Uses anchor text overlap with query + URL path token matching.
        let url_relevance = predict_url_relevance(url, anchor_text, query);

        let priority = depth_score + authority + diversity * 0.2 + url_relevance * 0.4;

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

        // Composite priority: depth + authority + domain diversity
        let depth_score = 1.0 / (depth as f32 + 1.0);     // 1.0 at depth 0, 0.5 at depth 1
        let authority = domain_authority_bonus(&domain);     // 0.0 - 0.3
        let diversity = 1.0 / (count as f32 + 1.0);         // high when domain is fresh (few pages)
        let priority = depth_score + authority + diversity * 0.2;

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

/// Predict URL relevance to a query without fetching the page.
///
/// Algorithm (from "Fast Webpage Classification Using URL Features"):
/// 1. Tokenize URL path into words (split on /-_.)
/// 2. Tokenize anchor text into words
/// 3. Tokenize query into words
/// 4. Score = |anchor ∩ query| / |query| + |url_path ∩ query| / |query| * 0.5
///
/// This gives a [0.0, 1.5] relevance prediction score.
/// No network I/O required — pure string analysis.
fn predict_url_relevance(url: &str, anchor_text: &str, query: &str) -> f32 {
    if query.is_empty() {
        return 0.0;
    }

    let query_words: std::collections::HashSet<String> = query
        .to_lowercase()
        .split_whitespace()
        .filter(|w| w.len() >= 2)
        .map(|w| w.to_string())
        .collect();

    if query_words.is_empty() {
        return 0.0;
    }

    // Score anchor text overlap with query (strongest signal)
    let anchor_words: std::collections::HashSet<String> = anchor_text
        .to_lowercase()
        .split_whitespace()
        .filter(|w| w.len() >= 2)
        .map(|w| w.to_string())
        .collect();
    let anchor_overlap = query_words.intersection(&anchor_words).count() as f32;
    let anchor_score = anchor_overlap / query_words.len() as f32;

    // Score URL path token overlap with query (weaker signal)
    let url_path = url::Url::parse(url)
        .map(|u| u.path().to_string())
        .unwrap_or_default();
    let path_words: std::collections::HashSet<String> = url_path
        .to_lowercase()
        .split(|c: char| c == '/' || c == '-' || c == '_' || c == '.')
        .filter(|w| w.len() >= 2)
        .map(|w| w.to_string())
        .collect();
    let path_overlap = query_words.intersection(&path_words).count() as f32;
    let path_score = path_overlap / query_words.len() as f32;

    anchor_score + path_score * 0.5
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
