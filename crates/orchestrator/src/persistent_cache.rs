// SPDX-License-Identifier: MIT

//! Persistent cache backed by redb (pure-Rust embedded KV store).
//!
//! Architecture: DashMap (hot, <1μs) front + redb (warm, <1ms) back.
//! Write-behind: dirty entries flushed to redb every 30 seconds.
//! On startup: redb entries loaded into DashMap for instant access.

use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing;

// ── Table Definitions ──────────────────────────────────────────────

const EMBEDDING_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("embeddings");
const SCORE_TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("ce_scores");
const URL_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("url_extractions");
const QUERY_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("query_results");

// ── Serializable Cache Entries ─────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct EmbeddingEntry {
    vector: Vec<f32>,
    stored_at: u64, // unix epoch secs
}

#[derive(Serialize, Deserialize)]
struct ScoreEntry {
    score: f32,
    stored_at: u64,
}

#[derive(Serialize, Deserialize)]
pub struct UrlCacheEntry {
    pub url: String,
    pub domain: String,
    pub title: String,
    pub body_text: String,
    pub body_for_embed: String,
    pub source_tier: u8, // serialized SourceTier ordinal
    pub stored_at: u64,
}

#[derive(Serialize, Deserialize)]
pub struct QueryCacheEntry {
    pub query_embedding: Vec<f32>,
    pub response_json: Vec<u8>, // bincode-serialized SearchResponse
    pub stored_at: u64,
}

// ── Persistent Cache ───────────────────────────────────────────────

pub struct PersistentCache {
    db: Database,
}

impl PersistentCache {
    /// Open or create the cache database at the given path.
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let db = Database::create(path)?;

        // Ensure tables exist by opening them in a write transaction
        let write_txn = db.begin_write()?;
        write_txn.open_table(EMBEDDING_TABLE)?;
        write_txn.open_table(SCORE_TABLE)?;
        write_txn.open_table(URL_TABLE)?;
        write_txn.open_table(QUERY_TABLE)?;
        write_txn.commit()?;

        tracing::info!(path = %path.display(), "Persistent cache opened");
        Ok(Self { db })
    }

    // ── Embedding Cache ────────────────────────────────────────────

    /// Load all embeddings from disk into a DashMap.
    pub fn load_embeddings(&self) -> anyhow::Result<Vec<(String, Vec<f32>)>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(EMBEDDING_TABLE)?;
        let mut entries = Vec::new();

        for result in table.iter()? {
            let (key, value) = result?;
            if let Ok(entry) = bincode::deserialize::<EmbeddingEntry>(value.value()) {
                entries.push((key.value().to_string(), entry.vector));
            }
        }

        tracing::info!(count = entries.len(), "Loaded embeddings from persistent cache");
        Ok(entries)
    }

    /// Flush embedding entries to disk in a single transaction.
    pub fn flush_embeddings(&self, entries: &[(&str, &[f32])]) -> anyhow::Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let now = now_epoch();
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(EMBEDDING_TABLE)?;
            for (hash, vector) in entries {
                let entry = EmbeddingEntry {
                    vector: vector.to_vec(),
                    stored_at: now,
                };
                let bytes = bincode::serialize(&entry)?;
                table.insert(*hash, bytes.as_slice())?;
            }
        }
        write_txn.commit()?;
        tracing::debug!(count = entries.len(), "Flushed embeddings to disk");
        Ok(())
    }

    // ── Cross-Encoder Score Cache ──────────────────────────────────

    /// Load all CE scores from disk.
    pub fn load_scores(&self) -> anyhow::Result<Vec<((u64, u64), f32)>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(SCORE_TABLE)?;
        let mut entries = Vec::new();

        for result in table.iter()? {
            let (key, value) = result?;
            let key_bytes = key.value();
            if key_bytes.len() == 16 {
                let query_hash = u64::from_le_bytes(key_bytes[..8].try_into().unwrap());
                let doc_hash = u64::from_le_bytes(key_bytes[8..].try_into().unwrap());
                if let Ok(entry) = bincode::deserialize::<ScoreEntry>(value.value()) {
                    entries.push(((query_hash, doc_hash), entry.score));
                }
            }
        }

        tracing::info!(count = entries.len(), "Loaded CE scores from persistent cache");
        Ok(entries)
    }

    /// Flush CE score entries to disk in a single transaction.
    pub fn flush_scores(&self, entries: &[((u64, u64), f32)]) -> anyhow::Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let now = now_epoch();
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(SCORE_TABLE)?;
            for ((qh, dh), score) in entries {
                let mut key_bytes = [0u8; 16];
                key_bytes[..8].copy_from_slice(&qh.to_le_bytes());
                key_bytes[8..].copy_from_slice(&dh.to_le_bytes());
                let entry = ScoreEntry {
                    score: *score,
                    stored_at: now,
                };
                let bytes = bincode::serialize(&entry)?;
                table.insert(key_bytes.as_slice(), bytes.as_slice())?;
            }
        }
        write_txn.commit()?;
        tracing::debug!(count = entries.len(), "Flushed CE scores to disk");
        Ok(())
    }

    // ── URL Extraction Cache ───────────────────────────────────────

    /// Load URL cache entries from disk (respecting TTL).
    pub fn load_url_entries(&self, ttl_secs: u64) -> anyhow::Result<Vec<(String, UrlCacheEntry)>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(URL_TABLE)?;
        let cutoff = now_epoch().saturating_sub(ttl_secs);
        let mut entries = Vec::new();

        for result in table.iter()? {
            let (key, value) = result?;
            if let Ok(entry) = bincode::deserialize::<UrlCacheEntry>(value.value()) {
                if entry.stored_at > cutoff {
                    entries.push((key.value().to_string(), entry));
                }
            }
        }

        tracing::info!(count = entries.len(), "Loaded URL cache entries from persistent cache");
        Ok(entries)
    }

    /// Flush URL cache entries to disk.
    pub fn flush_url_entries(&self, entries: &[(&str, &UrlCacheEntry)]) -> anyhow::Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(URL_TABLE)?;
            for (url, entry) in entries {
                let bytes = bincode::serialize(entry)?;
                table.insert(*url, bytes.as_slice())?;
            }
        }
        write_txn.commit()?;
        tracing::debug!(count = entries.len(), "Flushed URL cache entries to disk");
        Ok(())
    }

    // ── Eviction ───────────────────────────────────────────────────

    /// Remove expired entries from all tables.
    pub fn evict_expired(&self, embedding_max: usize, score_max: usize, url_ttl_secs: u64) -> anyhow::Result<()> {
        let url_cutoff = now_epoch().saturating_sub(url_ttl_secs);

        let write_txn = self.db.begin_write()?;

        // Evict expired URL entries
        {
            let mut table = write_txn.open_table(URL_TABLE)?;
            let mut to_remove = Vec::new();
            for result in table.iter()? {
                let (key, value) = result?;
                if let Ok(entry) = bincode::deserialize::<UrlCacheEntry>(value.value()) {
                    if entry.stored_at < url_cutoff {
                        to_remove.push(key.value().to_string());
                    }
                }
            }
            for key in &to_remove {
                table.remove(key.as_str())?;
            }
            if !to_remove.is_empty() {
                tracing::debug!(count = to_remove.len(), "Evicted expired URL cache entries");
            }
        }

        // Cap embedding table size (remove oldest if over limit)
        {
            let table = write_txn.open_table(EMBEDDING_TABLE)?;
            let count = table.len()? as usize;
            if count > embedding_max {
                let excess = count - embedding_max;
                let mut oldest: Vec<(String, u64)> = Vec::new();
                for result in table.iter()? {
                    let (key, value) = result?;
                    if let Ok(entry) = bincode::deserialize::<EmbeddingEntry>(value.value()) {
                        oldest.push((key.value().to_string(), entry.stored_at));
                    }
                }
                oldest.sort_by_key(|(_, ts)| *ts);
                drop(table);
                let mut table = write_txn.open_table(EMBEDDING_TABLE)?;
                for (key, _) in oldest.iter().take(excess) {
                    table.remove(key.as_str())?;
                }
                tracing::debug!(removed = excess, "Capped embedding cache");
            }
        }

        // Cap score table
        {
            let table = write_txn.open_table(SCORE_TABLE)?;
            let count = table.len()? as usize;
            if count > score_max {
                let excess = count - score_max;
                let mut oldest: Vec<(Vec<u8>, u64)> = Vec::new();
                for result in table.iter()? {
                    let (key, value) = result?;
                    if let Ok(entry) = bincode::deserialize::<ScoreEntry>(value.value()) {
                        oldest.push((key.value().to_vec(), entry.stored_at));
                    }
                }
                oldest.sort_by_key(|(_, ts)| *ts);
                drop(table);
                let mut table = write_txn.open_table(SCORE_TABLE)?;
                for (key, _) in oldest.iter().take(excess) {
                    table.remove(key.as_slice())?;
                }
                tracing::debug!(removed = excess, "Capped score cache");
            }
        }

        write_txn.commit()?;
        Ok(())
    }

    /// Get database stats for logging.
    pub fn stats(&self) -> anyhow::Result<CacheStats> {
        let read_txn = self.db.begin_read()?;
        Ok(CacheStats {
            embeddings: read_txn.open_table(EMBEDDING_TABLE)?.len()? as usize,
            scores: read_txn.open_table(SCORE_TABLE)?.len()? as usize,
            urls: read_txn.open_table(URL_TABLE)?.len()? as usize,
            queries: read_txn.open_table(QUERY_TABLE)?.len()? as usize,
        })
    }
}

#[derive(Debug)]
pub struct CacheStats {
    pub embeddings: usize,
    pub scores: usize,
    pub urls: usize,
    pub queries: usize,
}

/// Spawn a background flush task that periodically writes dirty DashMap entries to redb.
pub fn spawn_flush_task(
    cache: Arc<PersistentCache>,
    embedding_cache: dashmap::DashMap<String, Vec<f32>>,
    interval_secs: u64,
) -> tokio::task::JoinHandle<()> {
    let embedding_cache = Arc::new(embedding_cache);
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));
        // Track what we've already flushed via a snapshot of known keys
        let mut last_flush_count = 0_usize;

        loop {
            interval.tick().await;

            // Flush new embedding entries
            let current_count = embedding_cache.len();
            if current_count > last_flush_count {
                let entries: Vec<(String, Vec<f32>)> = embedding_cache
                    .iter()
                    .map(|r| (r.key().clone(), r.value().clone()))
                    .collect();

                let pairs: Vec<(&str, &[f32])> = entries
                    .iter()
                    .map(|(k, v)| (k.as_str(), v.as_slice()))
                    .collect();

                if let Err(e) = cache.flush_embeddings(&pairs) {
                    tracing::warn!(error = %e, "Failed to flush embeddings to disk");
                }
                last_flush_count = current_count;
            }

            // Periodic eviction
            if let Err(e) = cache.evict_expired(10_000, 50_000, 4 * 3600) {
                tracing::warn!(error = %e, "Failed to evict expired cache entries");
            }
        }
    })
}

fn now_epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_cache(name: &str) -> (PersistentCache, PathBuf) {
        let dir = std::env::temp_dir().join(format!("redb_test_{}_{}", std::process::id(), name));
        let path = dir.join("test_cache.redb");
        std::fs::create_dir_all(&dir).ok();
        let cache = PersistentCache::open(&path).unwrap();
        (cache, path)
    }

    #[test]
    fn embedding_round_trip() {
        let (cache, path) = temp_cache("embedding");

        let v1 = vec![1.0_f32, 2.0, 3.0];
        let v2 = vec![4.0_f32, 5.0, 6.0];
        let entries = vec![
            ("hash_abc", v1.as_slice()),
            ("hash_def", v2.as_slice()),
        ];
        cache.flush_embeddings(&entries).unwrap();

        let loaded = cache.load_embeddings().unwrap();
        assert_eq!(loaded.len(), 2);
        assert!(loaded.iter().any(|(k, v)| k == "hash_abc" && v == &[1.0, 2.0, 3.0]));

        // Cleanup
        drop(cache);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn score_round_trip() {
        let (cache, path) = temp_cache("score");

        let entries = vec![((123_u64, 456_u64), 0.95_f32)];
        cache.flush_scores(&entries).unwrap();

        let loaded = cache.load_scores().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0], ((123, 456), 0.95));

        drop(cache);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn stats_work() {
        let (cache, path) = temp_cache("stats");
        let stats = cache.stats().unwrap();
        assert_eq!(stats.embeddings, 0);
        assert_eq!(stats.scores, 0);

        drop(cache);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }
}
