use dashmap::DashMap;
use sha2::{Digest, Sha256};
use web_search_common::Result;

use crate::simhash;

/// Multi-level deduplication store with TTL.
///
/// Level 1: Exact — SHA-256 content hash (permanent)
/// Level 2: Near  — SimHash 64-bit fingerprint, hamming ≤ threshold (permanent)
/// Level 3: URL   — exact URL dedup with TTL (expires after url_ttl_secs)
pub struct DedupStore {
    /// SHA-256 content hashes (permanent — same content is always duplicate)
    exact_hashes: DashMap<String, String>, // hash → url
    /// SimHash fingerprints (permanent)
    simhash_fps: DashMap<String, u64>, // url → simhash
    /// Seen URLs with timestamp for TTL expiry
    seen_urls: DashMap<String, u64>, // url → unix epoch when seen
    /// Hamming distance threshold for near-duplicate
    simhash_threshold: u32,
    /// URL dedup TTL in seconds (0 = never expire)
    url_ttl_secs: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DedupResult {
    /// Never seen before
    Unique,
    /// Exact duplicate (same content hash)
    ExactDuplicate { original_url: String },
    /// Near-duplicate (similar content, small hamming distance)
    NearDuplicate {
        original_url: String,
        hamming_distance: u32,
    },
    /// Same URL already crawled
    UrlDuplicate,
}

impl DedupStore {
    pub fn new(simhash_threshold: u32) -> Self {
        Self {
            exact_hashes: DashMap::new(),
            simhash_fps: DashMap::new(),
            seen_urls: DashMap::new(),
            simhash_threshold,
            url_ttl_secs: 3600, // 1 hour default
        }
    }

    /// Create with custom URL TTL.
    pub fn with_ttl(simhash_threshold: u32, url_ttl_secs: u64) -> Self {
        Self {
            exact_hashes: DashMap::new(),
            simhash_fps: DashMap::new(),
            seen_urls: DashMap::new(),
            simhash_threshold,
            url_ttl_secs,
        }
    }

    fn now_epoch() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    fn is_url_expired(&self, timestamp: u64) -> bool {
        if self.url_ttl_secs == 0 { return false; }
        Self::now_epoch().saturating_sub(timestamp) > self.url_ttl_secs
    }

    /// Check if content is a duplicate. If unique, registers it.
    /// URL dedup respects TTL — expired URLs are re-crawlable.
    pub fn check_and_register(&self, url: &str, content: &str) -> DedupResult {
        // Level 3: URL dedup (with TTL)
        if let Some(entry) = self.seen_urls.get(url) {
            if !self.is_url_expired(*entry.value()) {
                return DedupResult::UrlDuplicate;
            }
            // Expired — remove and re-process
            drop(entry);
            self.seen_urls.remove(url);
        }

        // Level 1: Exact hash
        let content_hash = sha256_hex(content);
        if let Some(entry) = self.exact_hashes.get(&content_hash) {
            return DedupResult::ExactDuplicate {
                original_url: entry.value().clone(),
            };
        }

        // Level 2: SimHash near-duplicate
        let fingerprint = simhash::simhash(content);
        for entry in self.simhash_fps.iter() {
            let distance = simhash::hamming_distance(fingerprint, *entry.value());
            if distance <= self.simhash_threshold {
                return DedupResult::NearDuplicate {
                    original_url: entry.key().clone(),
                    hamming_distance: distance,
                };
            }
        }

        // Not a duplicate — register it with current timestamp
        self.seen_urls.insert(url.to_string(), Self::now_epoch());
        self.exact_hashes
            .insert(content_hash, url.to_string());
        self.simhash_fps.insert(url.to_string(), fingerprint);

        DedupResult::Unique
    }

    /// Check URL-only dedup without content. Respects TTL.
    pub fn has_url(&self, url: &str) -> bool {
        match self.seen_urls.get(url) {
            Some(entry) => !self.is_url_expired(*entry.value()),
            None => false,
        }
    }

    /// Register a URL as seen (without content check).
    pub fn mark_url_seen(&self, url: &str) {
        self.seen_urls.insert(url.to_string(), Self::now_epoch());
    }

    /// Number of unique documents tracked.
    pub fn len(&self) -> usize {
        self.seen_urls.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seen_urls.is_empty()
    }

    /// Clear all dedup state.
    pub fn clear(&self) {
        self.exact_hashes.clear();
        self.simhash_fps.clear();
        self.seen_urls.clear();
    }

    /// Save dedup state to a JSON file.
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        use std::io::Write;
        // Save URLs with timestamps for TTL persistence
        let urls: Vec<(String, u64)> = self.seen_urls.iter()
            .map(|e| (e.key().clone(), *e.value()))
            .collect();
        let hashes: Vec<(String, String)> = self
            .exact_hashes
            .iter()
            .map(|e| (e.key().clone(), e.value().clone()))
            .collect();
        let fps: Vec<(String, u64)> = self
            .simhash_fps
            .iter()
            .map(|e| (e.key().clone(), *e.value()))
            .collect();

        let data = serde_json::json!({
            "threshold": self.simhash_threshold,
            "url_ttl_secs": self.url_ttl_secs,
            "urls_v2": urls,
            "hashes": hashes,
            "fingerprints": fps,
        });

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut f = std::fs::File::create(path)?;
        f.write_all(serde_json::to_vec(&data)?.as_slice())?;
        Ok(())
    }

    /// Load dedup state from a JSON file. Supports both v1 (no TTL) and v2 (with TTL) formats.
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let data = std::fs::read(path)?;
        let parsed: serde_json::Value = serde_json::from_slice(&data)?;

        let threshold = parsed["threshold"].as_u64().unwrap_or(3) as u32;
        let ttl = parsed["url_ttl_secs"].as_u64().unwrap_or(3600);
        let store = DedupStore::with_ttl(threshold, ttl);

        // v2 format: urls_v2 = [(url, timestamp), ...]
        if let Some(urls) = parsed["urls_v2"].as_array() {
            let now = Self::now_epoch();
            for u in urls {
                if let (Some(url), Some(ts)) = (u[0].as_str(), u[1].as_u64()) {
                    // Skip expired URLs on load
                    if store.url_ttl_secs > 0 && now.saturating_sub(ts) > store.url_ttl_secs {
                        continue;
                    }
                    store.seen_urls.insert(url.to_string(), ts);
                }
            }
        } else if let Some(urls) = parsed["urls"].as_array() {
            // v1 legacy format: urls = [url, ...] — assign current timestamp
            let now = Self::now_epoch();
            for u in urls {
                if let Some(s) = u.as_str() {
                    store.seen_urls.insert(s.to_string(), now);
                }
            }
        }

        if let Some(hashes) = parsed["hashes"].as_array() {
            for h in hashes {
                if let (Some(k), Some(v)) = (h[0].as_str(), h[1].as_str()) {
                    store.exact_hashes.insert(k.to_string(), v.to_string());
                }
            }
        }
        if let Some(fps) = parsed["fingerprints"].as_array() {
            for f in fps {
                if let (Some(k), Some(v)) = (f[0].as_str(), f[1].as_u64()) {
                    store.simhash_fps.insert(k.to_string(), v);
                }
            }
        }

        Ok(store)
    }

    /// Load from file if exists, otherwise create new.
    pub fn open_or_create(path: &std::path::Path, threshold: u32) -> Self {
        match Self::load(path) {
            Ok(s) => {
                tracing::info!(urls = s.len(), "Loaded dedup state from disk");
                s
            }
            Err(_) => Self::new(threshold),
        }
    }
}

fn sha256_hex(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unique_content_passes() {
        let store = DedupStore::new(3);
        let result = store.check_and_register(
            "https://a.com",
            "This is unique content about quantum physics",
        );
        assert_eq!(result, DedupResult::Unique);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn exact_duplicate_detected() {
        let store = DedupStore::new(3);
        let content = "Exact same content on two different URLs";

        let r1 = store.check_and_register("https://a.com", content);
        assert_eq!(r1, DedupResult::Unique);

        let r2 = store.check_and_register("https://b.com", content);
        assert!(matches!(r2, DedupResult::ExactDuplicate { .. }));
        if let DedupResult::ExactDuplicate { original_url } = r2 {
            assert_eq!(original_url, "https://a.com");
        }
    }

    #[test]
    fn url_duplicate_detected() {
        let store = DedupStore::new(3);
        store.check_and_register("https://a.com", "first content");
        let r2 = store.check_and_register("https://a.com", "different content");
        assert_eq!(r2, DedupResult::UrlDuplicate);
    }

    #[test]
    fn different_content_passes() {
        let store = DedupStore::new(3);
        store.check_and_register(
            "https://a.com",
            "quantum physics experiments at the large hadron collider discovering new particles",
        );
        let r2 = store.check_and_register(
            "https://b.com",
            "chocolate cake recipe with cream cheese frosting and vanilla extract",
        );
        assert_eq!(r2, DedupResult::Unique);
    }

    #[test]
    fn has_url_works() {
        let store = DedupStore::new(3);
        assert!(!store.has_url("https://a.com"));
        store.mark_url_seen("https://a.com");
        assert!(store.has_url("https://a.com"));
    }

    #[test]
    fn clear_resets_state() {
        let store = DedupStore::new(3);
        store.check_and_register("https://a.com", "content");
        assert_eq!(store.len(), 1);
        store.clear();
        assert_eq!(store.len(), 0);
    }
}
