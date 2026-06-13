use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::{Duration, Instant};
use web_search_common::config::Config;
use web_search_common::models::*;
use web_search_common::Result;
use crate::persistent_cache::PersistentCache;
use crate::synthesis;
use web_search_crawler::Crawler;
use web_search_embedder::{self, Embedder};
use web_search_extractor::{self, consensus};
use web_search_indexer::dedup::{DedupResult, DedupStore};
use web_search_indexer::hnsw::HnswIndex;
use web_search_indexer::text_index::TextIndex;
use web_search_ranker::authority;
use web_search_ranker::pipeline::{RankCandidate, RankingPipeline};

/// Progress update sent during long-running operations.
#[derive(Debug, Clone)]
pub struct ProgressUpdate {
    /// Progress fraction (0.0 to 1.0)
    pub progress: f32,
    /// Total expected steps (optional)
    pub total: Option<u64>,
    /// Current step
    pub current: u64,
    /// Human-readable message
    pub message: String,
}

/// Semantic query cache entry — stores response with query embedding for similarity matching.
struct QueryCacheEntry {
    query_embedding: Vec<f32>,
    response: SearchResponse,
    cached_at: Instant,
}

/// Cached extraction result for a URL — avoids re-crawling + re-extracting.
#[derive(Clone)]
struct CachedExtraction {
    candidate: RankCandidate,
    body_for_embed: String,
    cached_at: Instant,
}

/// Central search engine coordinating all components.
///
/// Owns the crawler, extractor, indexer, embedder, and ranker.
/// Implements the 5 smart tools + 8 atomic tools.
pub struct SearchEngine {
    crawler: Crawler,
    text_index: TextIndex,
    vector_index: HnswIndex,
    dedup: DedupStore,
    embedder: Box<dyn Embedder>,
    pipeline: RankingPipeline,
    _config: Config,
    /// Optional progress callback for long-running operations
    progress_tx: Option<tokio::sync::broadcast::Sender<ProgressUpdate>>,
    /// Semantic query cache: similar queries return cached results instantly.
    /// Key insight: "Rust benefits 2026" and "advantages of Rust language" should
    /// hit the same cache because their embeddings are similar (cosine > 0.85).
    query_cache: tokio::sync::Mutex<Vec<QueryCacheEntry>>,
    /// Content-hash → embedding vector cache. Avoids re-embedding identical content.
    embedding_cache: dashmap::DashMap<String, Vec<f32>>,
    /// URL → extracted content cache. Avoids re-crawling + re-extracting same URLs.
    /// TTL: 30 minutes. Eliminates most of the 8.3s crawl time on repeated queries.
    url_cache: dashmap::DashMap<String, CachedExtraction>,
    /// Persistent disk cache (redb). Survives restarts.
    persistent_cache: Option<Arc<PersistentCache>>,
    /// Background flush task handle — flushes DashMap → redb every 30s.
    _flush_handle: Option<tokio::task::JoinHandle<()>>,
    /// Background crawl daemon — pre-indexes content before queries arrive.
    daemon: std::sync::OnceLock<crate::daemon::CrawlDaemon>,
}

impl SearchEngine {
    /// Create a new SearchEngine from config.
    ///
    /// Uses persistent disk-backed storage when data_dir exists.
    /// Falls back to in-memory index if disk path unavailable.
    pub fn new(config: Config) -> Result<Self> {
        let crawler = Crawler::new(config.crawler.clone())?;

        // Use persistent index if data_dir is configured
        let data_dir = &config.server.data_dir;
        let index_path = data_dir.join("index");
        let vector_path = data_dir.join("vectors").join("hnsw.json");

        let text_index = if data_dir.as_os_str().is_empty() {
            TextIndex::in_memory(config.indexer.tantivy_heap_size)?
        } else {
            match TextIndex::open(&index_path, config.indexer.tantivy_heap_size) {
                Ok(idx) => {
                    tracing::info!(path = %index_path.display(), docs = idx.num_docs(), "Opened persistent text index");
                    idx
                }
                Err(e) => {
                    tracing::warn!("Failed to open persistent index, using in-memory: {e}");
                    TextIndex::in_memory(config.indexer.tantivy_heap_size)?
                }
            }
        };

        let vector_index = HnswIndex::open_or_create(&vector_path, config.embedder.embedding_dim);
        tracing::info!(vectors = vector_index.len(), "Vector index ready");

        let dedup_path = data_dir.join("dedup.json");
        let dedup = DedupStore::open_or_create(&dedup_path, config.indexer.simhash_threshold);
        tracing::info!(urls = dedup.len(), "Dedup state ready");
        let embedder = web_search_embedder::create_embedder(&config.embedder);
        let pipeline = RankingPipeline::new(config.ranker.clone());

        let (progress_tx, _) = tokio::sync::broadcast::channel(32);

        // Open persistent cache (redb) — warm-load embeddings + CE scores into DashMap
        let embedding_cache = dashmap::DashMap::new();
        let cache_path = data_dir.join("cache.redb");
        let persistent_cache = if !data_dir.as_os_str().is_empty() {
            match PersistentCache::open(&cache_path) {
                Ok(pc) => {
                    // Warm-load embeddings
                    match pc.load_embeddings() {
                        Ok(entries) => {
                            for (hash, vec) in entries {
                                embedding_cache.insert(hash, vec);
                            }
                        }
                        Err(e) => tracing::warn!(error = %e, "Failed to load embeddings from disk"),
                    }
                    // Warm-load CE scores into ranker's DashMap
                    match pc.load_scores() {
                        Ok(entries) => {
                            let sc = pipeline.score_cache();
                            for (key, score) in &entries {
                                sc.insert(*key, *score);
                            }
                            tracing::info!(scores = entries.len(), "Loaded CE scores from persistent cache");
                        }
                        Err(e) => tracing::warn!(error = %e, "Failed to load CE scores from disk"),
                    }
                    if let Ok(stats) = pc.stats() {
                        tracing::info!(
                            embeddings = stats.embeddings,
                            scores = stats.scores,
                            urls = stats.urls,
                            "Persistent cache stats"
                        );
                    }
                    Some(Arc::new(pc))
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to open persistent cache, running without");
                    None
                }
            }
        } else {
            None
        };

        // Spawn background flush task (embeddings + CE scores → redb every 30s)
        let _flush_handle = if let Some(ref pc) = persistent_cache {
            let pc = Arc::clone(pc);
            let emb_cache = embedding_cache.clone();
            let score_cache = pipeline.score_cache().clone();
            Some(tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(30));
                let mut last_emb_count = 0_usize;
                let mut last_score_count = 0_usize;
                loop {
                    interval.tick().await;

                    // Flush new embeddings
                    let cur_emb = emb_cache.len();
                    if cur_emb > last_emb_count {
                        let entries: Vec<(String, Vec<f32>)> = emb_cache
                            .iter()
                            .map(|r| (r.key().clone(), r.value().clone()))
                            .collect();
                        let pairs: Vec<(&str, &[f32])> = entries
                            .iter()
                            .map(|(k, v)| (k.as_str(), v.as_slice()))
                            .collect();
                        if let Err(e) = pc.flush_embeddings(&pairs) {
                            tracing::warn!(error = %e, "Flush embeddings failed");
                        }
                        last_emb_count = cur_emb;
                    }

                    // Flush new CE scores
                    let cur_scores = score_cache.len();
                    if cur_scores > last_score_count {
                        let entries: Vec<((u64, u64), f32)> = score_cache
                            .iter()
                            .map(|r| (*r.key(), *r.value()))
                            .collect();
                        if let Err(e) = pc.flush_scores(&entries) {
                            tracing::warn!(error = %e, "Flush CE scores failed");
                        }
                        last_score_count = cur_scores;
                    }

                    // Periodic eviction
                    if let Err(e) = pc.evict_expired(10_000, 50_000, 4 * 3600) {
                        tracing::warn!(error = %e, "Cache eviction failed");
                    }
                }
            }))
        } else {
            None
        };

        // Note: daemon is spawned lazily via start_daemon() after SearchEngine
        // is wrapped in Arc, since it needs Arc<Crawler> from the engine.

        Ok(Self {
            crawler,
            text_index,
            vector_index,
            dedup,
            embedder,
            pipeline,
            _config: config,
            progress_tx: Some(progress_tx),
            query_cache: tokio::sync::Mutex::new(Vec::new()),
            embedding_cache,
            url_cache: dashmap::DashMap::new(),
            persistent_cache,
            _flush_handle,
            daemon: std::sync::OnceLock::new(),
        })
    }

    /// Pre-warm ML models by running a dummy embedding.
    ///
    /// On first call the embedder may need to load weights into memory (CandleEmbedder
    /// downloads + mmap, ONNX session init, etc.).  Calling this at startup moves that
    /// latency out of the first real request path.
    pub async fn warmup(&self) -> Result<()> {
        let start = Instant::now();
        // Trigger the embedder's model load by embedding a throwaway sentence.
        let _ = self.embedder.embed(&["warmup ping"]).await?;
        tracing::info!(elapsed_ms = start.elapsed().as_millis() as u64, "Embedder warmed up");
        Ok(())
    }

    /// Start the background crawl daemon.
    ///
    /// Must be called after the engine is wrapped in Arc.
    /// Pages crawled by the daemon populate the URL cache so next query
    /// hitting the same URL skips crawl entirely.
    pub fn start_daemon(self: &Arc<Self>) {
        use crate::daemon::CrawlDaemon;

        let crawler = match Crawler::new(self._config.crawler.clone()) {
            Ok(c) => Arc::new(c),
            Err(e) => {
                tracing::warn!(error = %e, "Failed to create crawler for daemon");
                return;
            }
        };
        let cancel = tokio_util::sync::CancellationToken::new();
        let url_cache = self.url_cache.clone();

        let daemon = CrawlDaemon::spawn(
            crawler,
            move |page| {
                let extraction = web_search_extractor::consensus::extract_page(
                    &page.body,
                    &page.final_url,
                );
                if extraction.body_text.len() < 50 {
                    return;
                }

                let domain = url::Url::parse(&page.final_url)
                    .ok()
                    .and_then(|u| u.host_str().map(|h| h.to_string()))
                    .unwrap_or_default();
                let source_tier = web_search_ranker::authority::classify_domain(&domain);

                let candidate = web_search_ranker::pipeline::RankCandidate {
                    url: page.final_url.clone(),
                    domain,
                    title: extraction.title.clone().unwrap_or_default(),
                    body_text: extraction.body_text.clone(),
                    published_date: extraction.published_date,
                    source_tier,
                    bm25_score: None,
                    vector_score: None,
                    bm25_rank: None,
                    vector_rank: None,
                    embedding: None,
                };

                url_cache.insert(
                    page.final_url.clone(),
                    CachedExtraction {
                        candidate,
                        body_for_embed: extraction.body_text,
                        cached_at: Instant::now(),
                    },
                );

                tracing::debug!(url = %page.final_url, "Daemon: pre-cached URL");
            },
            cancel,
        );

        let _ = self.daemon.set(daemon);
        tracing::info!("Background crawl daemon started");
    }

    /// Feed discovered links to the background daemon for pre-indexing.
    async fn enqueue_links_for_daemon(
        &self,
        pages: &[web_search_crawler::crawler::CrawledPage],
    ) {
        if let Some(daemon) = self.daemon.get() {
            let urls: Vec<(String, crate::daemon::CrawlPriority)> = pages
                .iter()
                .flat_map(|p| p.links.iter())
                .filter(|l| l.is_external)
                .take(20) // cap to avoid flooding
                .map(|l| (l.url.clone(), crate::daemon::CrawlPriority::HighQualityLink))
                .collect();
            if !urls.is_empty() {
                daemon.enqueue_batch(urls).await;
            }
        }
    }

    /// Subscribe to progress updates for long-running operations.
    pub fn subscribe_progress(&self) -> Option<tokio::sync::broadcast::Receiver<ProgressUpdate>> {
        self.progress_tx.as_ref().map(|tx| tx.subscribe())
    }

    /// Send a progress update (best-effort, ignores if no subscribers).
    fn send_progress(&self, current: u64, total: Option<u64>, message: &str) {
        if let Some(tx) = &self.progress_tx {
            let progress = total
                .map(|t| if t > 0 { current as f32 / t as f32 } else { 0.0 })
                .unwrap_or(0.0);
            let _ = tx.send(ProgressUpdate {
                progress,
                total,
                current,
                message: message.to_string(),
            });
        }
    }

    // ── Smart Tools ──────────────────────────────────────────────────

    /// Deep research: multi-wave crawl → extract → index → rank.
    pub async fn deep_research(
        &self,
        query: &str,
        max_pages: usize,
        max_depth: u8,
        time_limit_secs: u64,
    ) -> Result<SearchResponse> {
        let start = Instant::now();
        let time_limit = Duration::from_secs(time_limit_secs);

        tracing::info!(query, max_pages, max_depth, "Starting deep research");

        // Generate seed URLs from query + reformulated variants
        let query_variants = crate::query::reformulate_query(query);
        let seeds: Vec<String> = query_variants
            .iter()
            .flat_map(|q| generate_search_seeds(q, self._config.crawler.searxng_url.as_deref()))
            .collect();
        let seed_refs: Vec<&str> = seeds.iter().map(|s| s.as_str()).collect();
        tracing::info!(seed_count = seeds.len(), variants = query_variants.len(), "Generated search seeds");
        self.send_progress(1, Some(5), "Generated search seeds, starting wave 1 crawl");

        // Wave 1: Broad crawl
        let wave1_limit = (max_pages / 2).max(10);
        let wave1_time = Duration::from_secs(time_limit_secs / 2);
        let pages = self.crawler.crawl(&seed_refs, wave1_limit, max_depth, wave1_time).await;

        tracing::info!(wave1_pages = pages.len(), "Wave 1 complete");
        self.send_progress(2, Some(5), &format!("Wave 1 complete: {} pages crawled, extracting content", pages.len()));

        // Process crawled pages (batch extract + batch embed)
        let mut candidates = self.process_pages_batch(&pages).await;

        // Feed discovered links to background daemon for pre-indexing
        self.enqueue_links_for_daemon(&pages).await;

        // Wave 2: Follow best links from wave 1 results
        if start.elapsed() < time_limit && candidates.len() < max_pages {
            let remaining_time = time_limit.saturating_sub(start.elapsed());
            let remaining_pages = max_pages.saturating_sub(candidates.len());

            // Collect promising URLs from wave 1 pages
            let wave2_urls: Vec<String> = pages.iter()
                .flat_map(|p| p.links.iter())
                .filter(|l| !self.dedup.has_url(&l.url))
                .take(remaining_pages * 2)
                .map(|l| l.url.clone())
                .collect();

            if !wave2_urls.is_empty() {
                let wave2_refs: Vec<&str> = wave2_urls.iter().map(|s| s.as_str()).collect();
                let wave2_pages = self.crawler.crawl(
                    &wave2_refs,
                    remaining_pages,
                    max_depth,
                    remaining_time,
                ).await;

                tracing::info!(wave2_pages = wave2_pages.len(), "Wave 2 complete");
                self.send_progress(3, Some(5), &format!("Wave 2 complete: {} additional pages", wave2_pages.len()));

                let wave2_candidates = self.process_pages_batch(&wave2_pages).await;
                candidates.extend(wave2_candidates);
            }
        }

        // Commit index and save vectors to disk
        self.text_index.commit()?;
        self.save_vectors();
        self.send_progress(4, Some(5), &format!("Indexed {} candidates, running ranking pipeline", candidates.len()));

        // Run hybrid search: BM25 + vector ranks merged onto candidates
        self.apply_hybrid_ranks(&mut candidates, query).await;

        // Run ranking pipeline
        let top_k = 10;
        let mut response = self.pipeline.rank(candidates, query, top_k);

        // Apply query-focused MMR synthesis across results
        self.apply_synthesis(&mut response, query);

        self.send_progress(5, Some(5), &format!(
            "Deep research complete: {} results from {} pages in {}ms",
            response.results.len(), pages.len(), start.elapsed().as_millis()
        ));

        tracing::info!(
            total_crawled = pages.len(),
            results = response.results.len(),
            elapsed_ms = start.elapsed().as_millis(),
            "Deep research complete"
        );

        Ok(response)
    }

    /// Quick search: adaptive crawl + extract + rank.
    ///
    /// Adapts crawl parameters based on query type:
    /// - Factual: minimal crawl (20 pages, 0 depth, 8s) — answers are short
    /// - News: moderate crawl (30 pages, 0 depth, 10s) — recency matters
    /// - Technical: moderate crawl (40 pages, 1 depth, 12s) — follow docs
    /// - Research: deeper crawl (50 pages, 1 depth, 15s) — need multiple sources
    /// - General: default crawl (50 pages, 1 depth, 15s)
    pub async fn quick_search(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<SearchResponse> {
        let pipeline_start = Instant::now();

        // Check semantic query cache first
        if let Some(cached) = self.check_query_cache(query).await {
            tracing::info!(query, "Semantic cache hit — returning cached results");
            return Ok(cached);
        }

        // INDEX-FIRST: search existing index before crawling.
        // If we have enough fresh results in the persistent index, skip crawling entirely.
        // This is the single biggest latency win — eliminates 44% of query time.
        let index_results = self.text_index.search(query, max_results * 2).unwrap_or_default();
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let freshness_threshold = 24 * 3600; // 24 hours
        let fresh_results: Vec<_> = index_results.iter()
            .filter(|r| r.indexed_at > 0 && now_secs.saturating_sub(r.indexed_at) < freshness_threshold)
            .filter(|r| r.score > 0.5)
            .collect();

        if fresh_results.len() >= max_results {
            tracing::info!(
                query,
                fresh = fresh_results.len(),
                total_indexed = index_results.len(),
                "Index-first hit — returning pre-indexed results, skipping crawl"
            );

            let results: Vec<RankedResult> = fresh_results.iter()
                .take(max_results)
                .map(|r| {
                    let tier = authority::classify_domain(&r.domain);
                    RankedResult {
                        content: r.title.clone(), // index stores title, not full content
                        url: r.url.clone(),
                        title: r.title.clone(),
                        confidence: 0.7,
                        verification: VerificationStatus::Partial,
                        claims: vec![],
                        contradictions: vec![],
                        source_tier: tier,
                        freshness: None,
                        relevance_score: r.score,
                    }
                })
                .collect();

            let response = SearchResponse {
                results,
                synthesis: vec![],
                warnings: vec!["Results from pre-built index (no live crawl)".into()],
                coverage_score: 0.7,
                total_pages_crawled: 0,
                total_time_ms: pipeline_start.elapsed().as_millis() as u64,
                query: query.to_string(),
            };
            self.cache_query_response(query, &response).await;
            return Ok(response);
        }

        // FAST PATH: When SearXNG is configured, fetch JSON API directly (~300ms),
        // then crawl only the result URLs concurrently (~2-3s total).
        // This bypasses the slow frontier queue and skips fetching 15+ SERP pages.
        let query_type = web_search_ranker::query_type::detect_query_type(query);

        let (pages, used_fast_path) = if let Some(ref searxng_url) = self._config.crawler.searxng_url {
            let searxng_results = fetch_searxng_results(searxng_url, query).await;

            if !searxng_results.is_empty() {
                // Crawl only the top-N result URLs concurrently
                let max_fetch = match query_type {
                    web_search_common::models::QueryType::Factual  => 8,
                    web_search_common::models::QueryType::News     => 10,
                    web_search_common::models::QueryType::Technical => 12,
                    web_search_common::models::QueryType::Research  => 15,
                    web_search_common::models::QueryType::General   => 10,
                };
                let urls: Vec<&str> = searxng_results.iter()
                    .take(max_fetch)
                    .map(|r| r.url.as_str())
                    .collect();

                tracing::info!(
                    query,
                    query_type = ?query_type,
                    searxng_results = searxng_results.len(),
                    fetching = urls.len(),
                    "SearXNG fast path: concurrent fetch"
                );

                let pages = self.crawler.fetch_urls_concurrent(
                    &urls,
                    Duration::from_secs(5),
                ).await;
                (pages, true)
            } else {
                tracing::info!("SearXNG returned 0 results, falling back to full crawl");
                (Vec::new(), false)
            }
        } else {
            (Vec::new(), false)
        };

        // SLOW PATH: fallback when SearXNG unavailable or returned 0
        let pages = if !used_fast_path || pages.is_empty() {
            let (max_pages, depth, timeout_secs) = match query_type {
                web_search_common::models::QueryType::Factual  => (10, 0, 5),
                web_search_common::models::QueryType::News     => (15, 0, 8),
                web_search_common::models::QueryType::Technical => (20, 1, 10),
                web_search_common::models::QueryType::Research  => (30, 1, 12),
                web_search_common::models::QueryType::General   => (25, 1, 10),
            };

            tracing::info!(
                query,
                query_type = ?query_type,
                max_pages,
                depth,
                timeout_secs,
                "Fallback: full crawl with frontier"
            );

            let seeds = generate_search_seeds(query, self._config.crawler.searxng_url.as_deref());
            let seed_refs: Vec<&str> = seeds.iter().map(|s| s.as_str()).collect();

            self.crawler.crawl(
                &seed_refs,
                max_pages,
                depth,
                Duration::from_secs(timeout_secs),
            ).await
        } else {
            pages
        };

        // Process pages: extract + index. Skip embedding on fast path (BM25-only ranking).
        let mut candidates = if used_fast_path {
            // Fast path: extract + index only, no embedding (saves ~6s).
            // BM25 ranking is sufficient when SearXNG already did relevance sorting.
            self.process_pages_extract_only(&pages).await
        } else {
            self.process_pages_batch(&pages).await
        };
        self.enqueue_links_for_daemon(&pages).await;

        let total_budget = if used_fast_path {
            Duration::from_secs(12) // fast path: 12s total budget
        } else {
            Duration::from_secs(20) // slow path: 20s total budget
        };

        if pipeline_start.elapsed() > total_budget && !candidates.is_empty() {
            // Over budget — skip full ranking, return basic results
            tracing::warn!(
                elapsed_ms = pipeline_start.elapsed().as_millis(),
                budget_ms = total_budget.as_millis(),
                candidates = candidates.len(),
                "Time budget exceeded — returning partial results (no full ranking)"
            );
            let results: Vec<RankedResult> = candidates.iter()
                .take(max_results)
                .map(|c| RankedResult {
                    content: web_search_extractor::extract_snippet(&c.body_text, query, 1500),
                    url: c.url.clone(),
                    title: c.title.clone(),
                    confidence: 0.5,
                    verification: VerificationStatus::Unverified,
                    claims: vec![],
                    contradictions: vec![],
                    source_tier: c.source_tier,
                    freshness: c.published_date,
                    relevance_score: c.bm25_score.unwrap_or(0.0),
                })
                .collect();
            return Ok(SearchResponse {
                results,
                synthesis: vec![],
                warnings: vec!["Results returned under time pressure — verification skipped".into()],
                coverage_score: 0.3,
                total_pages_crawled: pages.len(),
                total_time_ms: pipeline_start.elapsed().as_millis() as u64,
                query: query.to_string(),
            });
        }

        self.text_index.commit()?;

        if used_fast_path {
            // Fast path: BM25-only hybrid ranks (skip vector search — no embeddings).
            // Still runs the full ranking pipeline for quality (CE + NLI + diversity).
            let bm25_limit = candidates.len().max(50);
            if let Ok(bm25_results) = self.text_index.search(query, bm25_limit) {
                let url_to_idx: std::collections::HashMap<String, usize> = candidates.iter()
                    .enumerate()
                    .map(|(i, c)| (c.url.clone(), i))
                    .collect();
                for (rank, result) in bm25_results.iter().enumerate() {
                    if let Some(&idx) = url_to_idx.get(&result.url) {
                        candidates[idx].bm25_rank = Some(rank + 1);
                        candidates[idx].bm25_score = Some(result.score);
                    }
                }
            }
        } else {
            self.save_vectors();
            self.apply_hybrid_ranks(&mut candidates, query).await;
        }

        let mut response = if used_fast_path {
            // Fast path: skip expensive CE/NLI pipeline.
            // Sort by BM25 score, build results directly.
            candidates.sort_by(|a, b| {
                b.bm25_score.unwrap_or(0.0).partial_cmp(&a.bm25_score.unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let results: Vec<RankedResult> = candidates.iter()
                .take(max_results)
                .map(|c| {
                    RankedResult {
                        content: web_search_extractor::extract_snippet(&c.body_text, query, 1500),
                        url: c.url.clone(),
                        title: c.title.clone(),
                        confidence: c.bm25_score.map(|s| (s * 0.1).min(0.95)).unwrap_or(0.5),
                        verification: VerificationStatus::Unverified,
                        claims: vec![],
                        contradictions: vec![],
                        source_tier: c.source_tier,
                        freshness: c.published_date,
                        relevance_score: c.bm25_score.unwrap_or(0.0),
                    }
                })
                .collect();
            SearchResponse {
                results,
                synthesis: vec![],
                warnings: vec![],
                coverage_score: 0.7,
                total_pages_crawled: pages.len(),
                total_time_ms: pipeline_start.elapsed().as_millis() as u64,
                query: query.to_string(),
            }
        } else {
            let mut resp = self.pipeline.rank(candidates, query, max_results);
            self.apply_synthesis(&mut resp, query);
            resp
        };

        // Synthesis for fast path
        if used_fast_path && !response.results.is_empty() {
            self.apply_synthesis(&mut response, query);
        }

        self.cache_query_response(query, &response).await;

        tracing::info!(
            total_ms = pipeline_start.elapsed().as_millis(),
            results = response.results.len(),
            fast_path = used_fast_path,
            "Quick search pipeline complete"
        );

        Ok(response)
    }

    /// Instant search: ultra-fast path (~0.5-2s).
    ///
    /// When SearXNG configured:
    ///   1. Check semantic cache (instant if hit)
    ///   2. Fetch SearXNG JSON (~300ms) — get title + snippet for each result
    ///   3. Rank snippets directly by query-term overlap — NO crawling at all
    ///   4. Return results in <1s
    ///
    /// Fallback (no SearXNG):
    ///   1. Crawl with 3s timeout, extract, return unranked
    pub async fn instant_search(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<SearchResponse> {
        let start = Instant::now();

        // 1. Semantic cache
        if let Some(cached) = self.check_query_cache(query).await {
            tracing::info!(query, elapsed_ms = start.elapsed().as_millis(), "instant_search: cache hit");
            return Ok(cached);
        }

        // 2. SearXNG snippet-only mode — no crawling at all
        if let Some(ref searxng_url) = self._config.crawler.searxng_url {
            let searxng_results = fetch_searxng_results(searxng_url, query).await;

            if !searxng_results.is_empty() {
                // Rank by query-term overlap in title + snippet
                let query_lower = query.to_lowercase();
                let query_terms: Vec<&str> = query_lower.split_whitespace()
                    .filter(|w| w.len() > 2)
                    .collect();

                let mut results: Vec<RankedResult> = searxng_results.iter()
                    .map(|sr| {
                        let domain = url::Url::parse(&sr.url)
                            .ok()
                            .and_then(|u| u.host_str().map(|h| h.to_string()))
                            .unwrap_or_default();
                        let tier = authority::classify_domain(&domain);

                        // Score by query-term density in title + snippet
                        let combined = format!("{} {}", sr.title.to_lowercase(), sr.snippet.to_lowercase());
                        let term_hits = query_terms.iter()
                            .filter(|qt| combined.contains(*qt))
                            .count();
                        let relevance = term_hits as f32 / query_terms.len().max(1) as f32;

                        RankedResult {
                            content: if sr.snippet.is_empty() { sr.title.clone() } else { sr.snippet.clone() },
                            url: sr.url.clone(),
                            title: sr.title.clone(),
                            confidence: 0.6,
                            verification: VerificationStatus::Unverified,
                            claims: vec![],
                            contradictions: vec![],
                            source_tier: tier,
                            freshness: None,
                            relevance_score: relevance,
                        }
                    })
                    .collect();

                // Sort by relevance (query-term overlap)
                results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score)
                    .unwrap_or(std::cmp::Ordering::Equal));
                results.truncate(max_results);

                let response = SearchResponse {
                    results,
                    synthesis: vec![],
                    warnings: vec![],
                    coverage_score: 0.6,
                    total_pages_crawled: 0,
                    total_time_ms: start.elapsed().as_millis() as u64,
                    query: query.to_string(),
                };

                self.cache_query_response(query, &response).await;

                tracing::info!(
                    results = response.results.len(),
                    elapsed_ms = response.total_time_ms,
                    "instant_search complete (SearXNG snippet-only)"
                );

                return Ok(response);
            }
        }

        // 3. Fallback: crawl with tight limits
        let seeds = generate_search_seeds(query, self._config.crawler.searxng_url.as_deref());
        let seed_refs: Vec<&str> = seeds.iter().map(|s| s.as_str()).collect();

        let pages = self.crawler.crawl(
            &seed_refs,
            max_results.max(5),
            0,
            Duration::from_secs(3),
        ).await;

        if pages.is_empty() {
            return Ok(SearchResponse {
                results: vec![],
                synthesis: vec![],
                warnings: vec!["No pages fetched within timeout".into()],
                coverage_score: 0.0,
                total_pages_crawled: 0,
                total_time_ms: start.elapsed().as_millis() as u64,
                query: query.to_string(),
            });
        }

        let mut results: Vec<RankedResult> = Vec::new();
        for page in &pages {
            let extraction = consensus::extract_page(&page.body, &page.final_url);
            if extraction.body_text.len() < 50 { continue; }

            let domain = extract_domain(&page.final_url);
            let tier = authority::classify_domain(&domain);

            results.push(RankedResult {
                content: web_search_extractor::extract_snippet(&extraction.body_text, query, 1500),
                url: page.final_url.clone(),
                title: extraction.title.unwrap_or_default(),
                confidence: extraction.extraction_confidence as f32 * 0.01,
                verification: VerificationStatus::Unverified,
                claims: vec![],
                contradictions: vec![],
                source_tier: tier,
                freshness: extraction.published_date,
                relevance_score: extraction.extraction_confidence as f32 * 0.01,
            });
        }

        results.truncate(max_results);

        let response = SearchResponse {
            results,
            synthesis: vec![],
            warnings: vec![],
            coverage_score: 0.5,
            total_pages_crawled: pages.len(),
            total_time_ms: start.elapsed().as_millis() as u64,
            query: query.to_string(),
        };

        self.cache_query_response(query, &response).await;

        tracing::info!(
            results = response.results.len(),
            elapsed_ms = response.total_time_ms,
            "instant_search complete (crawl fallback)"
        );

        Ok(response)
    }

    /// Streaming search: returns progressive results via mpsc channel.
    ///
    /// Two-tier deadline:
    /// - Partial results at 3s (lightly ranked, from fastest sources)
    /// - Refined results at 10s (full pipeline: cross-encoder, verification, synthesis)
    ///
    /// Callers receive `SearchEvent`s and can display results as they arrive.
    pub async fn streaming_search(
        &self,
        query: &str,
        config: crate::streaming::StreamConfig,
    ) -> tokio::sync::mpsc::Receiver<crate::streaming::SearchEvent> {
        use crate::streaming::*;

        let (tx, rx) = create_event_channel(16);
        let start = Instant::now();

        // Phase 1: fast crawl for partial results
        emit_progress(&tx, "crawl", "fetching initial pages");

        let seeds = generate_search_seeds(query, self._config.crawler.searxng_url.as_deref());
        let seed_refs: Vec<&str> = seeds.iter().map(|s| s.as_str()).collect();

        let pages = self.crawler.crawl(
            &seed_refs,
            15,  // small batch
            0,   // no link-following
            config.partial_deadline,
        ).await;

        emit_progress(&tx, "extract", &format!("processing {} pages", pages.len()));

        // Quick extraction for partial results (no embedding/indexing)
        let mut partial: Vec<RankedResult> = Vec::new();
        for page in &pages {
            let extraction = web_search_extractor::consensus::extract_page(&page.body, &page.final_url);
            if extraction.body_text.len() < 50 { continue; }

            let domain = extract_domain(&page.final_url);
            let tier = web_search_ranker::authority::classify_domain(&domain);
            partial.push(RankedResult {
                content: web_search_extractor::extract_snippet(
                    &extraction.body_text, query, 1500,
                ),
                url: page.final_url.clone(),
                title: extraction.title.unwrap_or_default(),
                confidence: extraction.extraction_confidence as f32 * 0.01,
                verification: VerificationStatus::Unverified,
                claims: vec![],
                contradictions: vec![],
                source_tier: tier,
                freshness: extraction.published_date,
                relevance_score: extraction.extraction_confidence as f32 * 0.01,
            });
        }
        partial.truncate(config.partial_max);

        // Emit partial results
        if !partial.is_empty() {
            emit_partial(&tx, partial, start);
        }

        // Phase 2: full pipeline for refined results (if time remains)
        if !past_deadline(start, config.hard_deadline) {
            emit_progress(&tx, "index", "embedding and indexing");

            let mut candidates = self.process_pages_batch(&pages).await;
            self.enqueue_links_for_daemon(&pages).await;

            if !past_deadline(start, config.hard_deadline) {
                self.text_index.commit().ok();
                self.save_vectors();
                self.apply_hybrid_ranks(&mut candidates, query).await;

                emit_progress(&tx, "rank", "running ranking pipeline");
                let mut response = self.pipeline.rank(candidates, query, config.final_max);
                self.apply_synthesis(&mut response, query);

                emit_refined(
                    &tx,
                    response.results,
                    response.synthesis,
                    start,
                ).await;
            }
        }

        emit_complete(&tx, pages.len(), start).await;
        rx
    }

    /// Verify a claim by searching for supporting/contradicting evidence.
    pub async fn verify_claim(
        &self,
        claim: &str,
        min_sources: usize,
    ) -> Result<SearchResponse> {
        // Search for the claim
        let mut response = self.quick_search(claim, min_sources * 2).await?;

        // Add verification-specific warning if insufficient sources
        if response.results.len() < min_sources {
            response.warnings.push(format!(
                "Only found {} sources (requested minimum: {})",
                response.results.len(),
                min_sources
            ));
        }

        Ok(response)
    }

    // ── Atomic Tools ─────────────────────────────────────────────────

    /// Fetch a single page.
    pub async fn fetch_page(&self, url: &str) -> Result<String> {
        let page = self.crawler.fetch_one(url).await?;
        Ok(serde_json::to_string_pretty(&serde_json::json!({
            "url": page.url,
            "final_url": page.final_url,
            "status": page.status,
            "content_type": page.content_type,
            "content_length": page.body.len(),
            "response_time_ms": page.response_time_ms,
            "is_spa": page.is_spa,
            "body_preview": &page.body[..page.body.len().min(2000)],
        }))?)
    }

    /// Extract clean content from a URL.
    pub async fn extract(&self, url: &str) -> Result<String> {
        let page = self.crawler.fetch_one(url).await?;
        let extraction = consensus::extract_page(&page.body, &page.final_url);
        Ok(serde_json::to_string_pretty(&serde_json::json!({
            "url": page.final_url,
            "title": extraction.title,
            "author": extraction.author,
            "published_date": extraction.published_date,
            "body_text": extraction.body_text,
            "headings": extraction.headings,
            "tables": extraction.tables,
            "language": extraction.language,
            "extraction_confidence": extraction.extraction_confidence,
        }))?)
    }

    /// Follow links from a URL matching a pattern.
    pub async fn follow_links(
        &self,
        url: &str,
        pattern: Option<&str>,
        _depth: u8,
    ) -> Result<String> {
        let page = self.crawler.fetch_one(url).await?;

        let links: Vec<_> = page.links.iter()
            .filter(|l| {
                if let Some(p) = pattern {
                    l.url.contains(p) || l.anchor_text.to_lowercase().contains(&p.to_lowercase())
                } else {
                    true
                }
            })
            .take(50)
            .collect();

        Ok(serde_json::to_string_pretty(&serde_json::json!({
            "source_url": url,
            "total_links": page.links.len(),
            "matching_links": links.len(),
            "links": links.iter().map(|l| serde_json::json!({
                "url": l.url,
                "anchor_text": l.anchor_text,
                "is_external": l.is_external,
            })).collect::<Vec<_>>(),
        }))?)
    }

    /// Follow pagination from a starting URL.
    pub async fn paginate(&self, url: &str, max_pages: u32) -> Result<String> {
        let pages = self.crawler.paginate(url, max_pages).await;

        let results: Vec<serde_json::Value> = pages.iter().map(|p| {
            let extraction = consensus::extract_page(&p.body, &p.final_url);
            serde_json::json!({
                "url": p.final_url,
                "title": extraction.title,
                "body_preview": &extraction.body_text[..extraction.body_text.len().min(500)],
                "links_found": p.links.len(),
            })
        }).collect();

        Ok(serde_json::to_string_pretty(&serde_json::json!({
            "start_url": url,
            "pages_fetched": pages.len(),
            "pages": results,
        }))?)
    }

    /// Search the local index.
    pub async fn search_index(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<String> {
        let results = self.text_index.search(query, max_results)?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let stale_threshold = 24 * 3600; // 24 hours

        let mut warnings = Vec::new();
        let stale_count = results.iter()
            .filter(|r| r.indexed_at > 0 && now.saturating_sub(r.indexed_at) > stale_threshold)
            .count();
        if stale_count > 0 {
            warnings.push(format!(
                "{stale_count}/{} results indexed >24h ago — content may be outdated",
                results.len()
            ));
        }

        Ok(serde_json::to_string_pretty(&serde_json::json!({
            "query": query,
            "total_docs_in_index": self.text_index.num_docs(),
            "warnings": warnings,
            "results": results.iter().map(|r| {
                let age_secs = if r.indexed_at > 0 { now.saturating_sub(r.indexed_at) } else { 0 };
                serde_json::json!({
                    "url": r.url,
                    "title": r.title,
                    "domain": r.domain,
                    "score": r.score,
                    "indexed_at": r.indexed_at,
                    "age_hours": age_secs / 3600,
                })
            }).collect::<Vec<_>>(),
        }))?)
    }

    /// Find semantically similar content.
    pub async fn find_similar(
        &self,
        text: &str,
        top_k: usize,
    ) -> Result<String> {
        let query_vec = self.embedder.embed_one(text).await?;
        let results = self.vector_index.search(&query_vec, top_k)?;
        Ok(serde_json::to_string_pretty(&serde_json::json!({
            "query_preview": &text[..text.len().min(100)],
            "total_vectors": self.vector_index.len(),
            "results": results.iter().map(|r| serde_json::json!({
                "url": r.doc_id,
                "similarity": r.score,
            })).collect::<Vec<_>>(),
        }))?)
    }

    /// Get link graph for a URL.
    pub async fn get_link_graph(
        &self,
        url: &str,
        _depth: u8,
    ) -> Result<String> {
        let page = self.crawler.fetch_one(url).await?;

        let outgoing: Vec<serde_json::Value> = page.links.iter()
            .take(100)
            .map(|l| serde_json::json!({
                "url": l.url,
                "anchor_text": l.anchor_text,
                "is_external": l.is_external,
            }))
            .collect();

        Ok(serde_json::to_string_pretty(&serde_json::json!({
            "url": url,
            "outgoing_links": outgoing.len(),
            "links": outgoing,
        }))?)
    }

    // ── Semantic Query Cache ─────────────────────────────────────

    /// Check if a semantically similar query was recently searched.
    /// Returns cached response if cosine similarity > 0.85 and age < 30 minutes.
    async fn check_query_cache(&self, query: &str) -> Option<SearchResponse> {
        let query_vec = self.embedder.embed_one(query).await.ok()?;
        let cache = self.query_cache.lock().await;
        let now = Instant::now();
        let ttl = Duration::from_secs(4 * 3600); // 4 hours (was 30 min)

        let mut best_sim = 0.0_f32;
        let mut best_response = None;

        for entry in cache.iter() {
            if now.duration_since(entry.cached_at) > ttl {
                continue; // expired
            }
            let sim = web_search_embedder::cosine_similarity(&query_vec, &entry.query_embedding);
            // Threshold 0.92 (was 0.85) — tighter to avoid false cache hits
            if sim > 0.92 && sim > best_sim {
                best_sim = sim;
                best_response = Some(entry.response.clone());
            }
        }

        if let Some(ref resp) = best_response {
            tracing::info!(
                similarity = best_sim,
                cached_results = resp.results.len(),
                "Semantic cache hit"
            );
        }
        best_response
    }

    /// Cache a query response for future similar queries.
    async fn cache_query_response(&self, query: &str, response: &SearchResponse) {
        if response.results.is_empty() {
            return; // don't cache empty results
        }
        if let Ok(query_vec) = self.embedder.embed_one(query).await {
            let mut cache = self.query_cache.lock().await;
            // Evict expired entries and cap at 200
            let now = Instant::now();
            let ttl = Duration::from_secs(4 * 3600); // 4 hours
            cache.retain(|e| now.duration_since(e.cached_at) < ttl);
            if cache.len() >= 100 {
                cache.remove(0); // LRU eviction
            }
            cache.push(QueryCacheEntry {
                query_embedding: query_vec,
                response: response.clone(),
                cached_at: now,
            });
            tracing::info!(cache_size = cache.len(), "Query response cached");
        }
    }

    /// Apply query-focused MMR synthesis to a search response.
    /// Uses query relevance + novelty to select top-5 most informative sentences.
    fn apply_synthesis(&self, response: &mut SearchResponse, query: &str) {
        let docs: Vec<(&str, &str, &str)> = response.results.iter()
            .map(|r| (r.content.as_str(), r.url.as_str(), r.title.as_str()))
            .collect();

        let scored = synthesis::synthesize_tfidf(&docs, 5, query);
        response.synthesis = scored.into_iter()
            .map(|s| SynthesizedSentence {
                text: s.text,
                score: s.score,
                source_url: s.source_url,
                source_title: s.source_title,
            })
            .collect();
    }

    // ── Internal ─────────────────────────────────────────────────────

    /// Save vector index + dedup state to disk (best-effort).
    fn save_vectors(&self) {
        let vector_path = self._config.server.data_dir.join("vectors").join("hnsw.json");
        if let Err(e) = self.vector_index.save(&vector_path) {
            tracing::warn!("Failed to save vector index: {e}");
        }
        let dedup_path = self._config.server.data_dir.join("dedup.json");
        if let Err(e) = self.dedup.save(&dedup_path) {
            tracing::warn!("Failed to save dedup state: {e}");
        }
    }

    /// Final flush of all in-memory caches to disk. Call on shutdown.
    pub fn flush_to_disk(&self) {
        self.save_vectors();

        if let Some(ref pc) = self.persistent_cache {
            // Flush embeddings
            let entries: Vec<(String, Vec<f32>)> = self.embedding_cache
                .iter()
                .map(|r| (r.key().clone(), r.value().clone()))
                .collect();
            let pairs: Vec<(&str, &[f32])> = entries
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_slice()))
                .collect();
            if let Err(e) = pc.flush_embeddings(&pairs) {
                tracing::warn!(error = %e, "Shutdown: flush embeddings failed");
            }

            // Flush CE scores
            let scores: Vec<((u64, u64), f32)> = self.pipeline.score_cache()
                .iter()
                .map(|r| (*r.key(), *r.value()))
                .collect();
            if let Err(e) = pc.flush_scores(&scores) {
                tracing::warn!(error = %e, "Shutdown: flush CE scores failed");
            }

            if let Ok(stats) = pc.stats() {
                tracing::info!(
                    embeddings = stats.embeddings,
                    scores = stats.scores,
                    urls = stats.urls,
                    "Shutdown: persistent cache flushed"
                );
            }
        }
    }

    /// Apply hybrid BM25 + vector ranks to candidates after indexing.
    ///
    /// Queries both tantivy (BM25) and HNSW (vector) indexes,
    /// then sets bm25_rank/vector_rank/scores on matching candidates.
    async fn apply_hybrid_ranks(&self, candidates: &mut [RankCandidate], query: &str) {
        // Build URL→index map for O(1) lookup instead of O(n) linear scan
        let url_to_idx: std::collections::HashMap<String, usize> = candidates.iter()
            .enumerate()
            .map(|(i, c)| (c.url.clone(), i))
            .collect();

        // BM25 search
        let bm25_limit = candidates.len().max(50);
        if let Ok(bm25_results) = self.text_index.search(query, bm25_limit) {
            for (rank, result) in bm25_results.iter().enumerate() {
                if let Some(&idx) = url_to_idx.get(&result.url) {
                    candidates[idx].bm25_rank = Some(rank + 1);
                    candidates[idx].bm25_score = Some(result.score);
                }
            }
        }

        // Vector search
        if let Ok(query_vec) = self.embedder.embed_one(query).await {
            let vec_limit = candidates.len().max(50);
            if let Ok(vec_results) = self.vector_index.search(&query_vec, vec_limit) {
                for (rank, result) in vec_results.iter().enumerate() {
                    if let Some(&idx) = url_to_idx.get(&result.doc_id) {
                        candidates[idx].vector_rank = Some(rank + 1);
                        candidates[idx].vector_score = Some(result.score);
                    }
                }
            }
        }

        let ranked = candidates.iter().filter(|c| c.bm25_rank.is_some() || c.vector_rank.is_some()).count();
        tracing::info!(ranked, total = candidates.len(), "Hybrid ranks applied");
    }

    /// Process crawled pages in bulk: extract, dedup, index, then batch embed.
    ///
    /// Two-phase approach:
    /// Phase 1: Extract + dedup + index (fast, no ML) — per page
    /// Phase 2: Batch embed all texts at once (1 ML call instead of N)
    /// Fast extract-only processing: extract + dedup + index, NO embedding.
    /// Used on SearXNG fast path — BM25 ranking only, saves ~6s embedding time.
    async fn process_pages_extract_only(
        &self,
        pages: &[web_search_crawler::crawler::CrawledPage],
    ) -> Vec<RankCandidate> {
        let start = std::time::Instant::now();
        let mut candidates = Vec::new();

        for crawled in pages {
            if let Some((candidate, _body)) = self.extract_and_index(crawled) {
                candidates.push(candidate);
            }
        }

        tracing::info!(
            pages = pages.len(),
            candidates = candidates.len(),
            elapsed_ms = start.elapsed().as_millis(),
            "Extract-only processing (no embedding)"
        );
        candidates
    }

    async fn process_pages_batch(
        &self,
        pages: &[web_search_crawler::crawler::CrawledPage],
    ) -> Vec<RankCandidate> {
        let extract_start = std::time::Instant::now();

        // Phase 1: Extract + dedup + index (no embedding yet)
        let mut pre_candidates: Vec<(RankCandidate, String)> = Vec::new(); // (candidate, body_text for embedding)

        for crawled in pages {
            if let Some((candidate, body_for_embed)) = self.extract_and_index(crawled) {
                pre_candidates.push((candidate, body_for_embed));
            }
        }

        let extract_ms = extract_start.elapsed().as_millis();
        tracing::info!(
            pages = pages.len(),
            candidates = pre_candidates.len(),
            extract_ms,
            "Phase 1: extraction + indexing complete"
        );

        if pre_candidates.is_empty() {
            return Vec::new();
        }

        // Phase 2: Batch embed — check content-hash cache first, embed only misses
        let embed_start = std::time::Instant::now();

        // Separate cached vs uncached
        let mut cache_hits = 0_usize;
        let mut need_embed: Vec<(usize, String)> = Vec::new(); // (index, text)
        for (i, (candidate, body)) in pre_candidates.iter_mut().enumerate() {
            let hash = sha256_short(body);
            if let Some(cached_vec) = self.embedding_cache.get(&hash) {
                candidate.embedding = Some(cached_vec.clone());
                if let Err(e) = self.vector_index.insert(&candidate.url, cached_vec.clone()) {
                    tracing::warn!(url = %candidate.url, error = %e, "Vector insert failed (cached)");
                }
                cache_hits += 1;
            } else {
                need_embed.push((i, body.clone()));
            }
        }

        let text_count = need_embed.len();
        if !need_embed.is_empty() {
            let texts: Vec<&str> = need_embed.iter().map(|(_, t)| t.as_str()).collect();
            match self.embedder.embed(&texts).await {
                Ok(embeddings) => {
                    for (j, (orig_i, body)) in need_embed.iter().enumerate() {
                        if let Some(vec) = embeddings.get(j) {
                            let (candidate, _) = &mut pre_candidates[*orig_i];
                            if let Err(e) = self.vector_index.insert(&candidate.url, vec.clone()) {
                                tracing::warn!(url = %candidate.url, error = %e, "Vector insert failed");
                            }
                            candidate.embedding = Some(vec.clone());
                            // Populate cache
                            let hash = sha256_short(body);
                            self.embedding_cache.insert(hash, vec.clone());
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Batch embedding failed");
                }
            }
        }

        let embed_ms = embed_start.elapsed().as_millis();
        tracing::info!(
            total = pre_candidates.len(),
            cache_hits,
            embedded = text_count,
            embed_ms,
            "Phase 2: embedding complete"
        );

        pre_candidates.into_iter().map(|(c, _)| c).collect()
    }

    /// Phase 1 of page processing: extract, dedup, index. No ML.
    /// Returns (candidate_without_embedding, body_text_for_embedding).
    /// Checks URL cache first — skips re-extraction for recently seen URLs.
    fn extract_and_index(
        &self,
        crawled: &web_search_crawler::crawler::CrawledPage,
    ) -> Option<(RankCandidate, String)> {
        // Check URL cache (4 hour TTL — content doesn't change that fast)
        let ttl = Duration::from_secs(4 * 3600);
        if let Some(cached) = self.url_cache.get(&crawled.final_url) {
            if cached.cached_at.elapsed() < ttl {
                tracing::debug!(url = %crawled.final_url, "URL cache hit — skipping extraction");
                return Some((cached.candidate.clone(), cached.body_for_embed.clone()));
            } else {
                drop(cached);
                self.url_cache.remove(&crawled.final_url);
            }
        }

        // Skip non-HTML
        let ct = crawled.content_type.to_lowercase();
        if !ct.contains("html") && !ct.contains("text/plain") && !ct.contains("xml") {
            if ct.contains("json") {
                tracing::debug!(url = %crawled.final_url, "Skipping JSON content (not indexable)");
            } else {
                tracing::debug!(url = %crawled.final_url, content_type = %ct, "Skipping non-HTML content");
            }
            return None;
        }

        // Dedup check
        let dedup_result = self.dedup.check_and_register(&crawled.final_url, &crawled.body);
        match dedup_result {
            DedupResult::Unique => {}
            other => {
                tracing::debug!(url = %crawled.final_url, result = ?other, "Skipping duplicate");
                return None;
            }
        }

        // Extract content
        let extraction = consensus::extract_page(&crawled.body, &crawled.final_url);

        if extraction.body_text.len() < 30 {
            tracing::debug!(url = %crawled.final_url, len = extraction.body_text.len(), "Skipping: too little content");
            return None;
        }

        if is_low_quality_content(&extraction.body_text, &crawled.final_url) {
            tracing::debug!(url = %crawled.final_url, "Skipping: low quality / error page");
            return None;
        }

        // Skip search engine result pages (SERP) — they list other pages, not real content
        if consensus::is_serp_page(&crawled.final_url, &extraction.body_text) {
            tracing::debug!(url = %crawled.final_url, "Skipping: SERP page");
            return None;
        }

        // Skip non-English pages — English queries often return Somali, Turkish, Tagalog Wikipedia etc.
        if let Some(ref lang) = extraction.language {
            if !lang.starts_with("en") {
                tracing::debug!(url = %crawled.final_url, language = %lang, "Skipping: non-English page (lang attribute)");
                return None;
            }
        } else {
            // No lang attribute — use stop-word heuristic on first 500 chars of body text
            let sample: String = extraction.body_text.chars().take(500).collect();
            let sample_lower = sample.to_lowercase();
            let stop_words = ["the", "is", "and", "of", "to"];
            let hit_count = stop_words.iter()
                .filter(|w| {
                    // Match whole words only to avoid false positives (e.g. "to" inside "together")
                    sample_lower.split_whitespace().any(|token| token == **w)
                })
                .count();
            if hit_count < 2 {
                tracing::debug!(
                    url = %crawled.final_url,
                    stop_word_hits = hit_count,
                    "Skipping: likely non-English page (stop-word heuristic)"
                );
                return None;
            }
        }

        // Build Page model + index in tantivy
        let content_hash = sha256_short(&crawled.body);
        let domain = extract_domain(&crawled.final_url);
        let page = consensus::to_page(
            &extraction,
            &crawled.final_url,
            &domain,
            &content_hash,
            crawled.status,
            crawled.response_time_ms,
            crawled.body.len(),
        );

        if let Err(e) = self.text_index.add_page(&page, None) {
            tracing::warn!(url = %crawled.final_url, error = %e, "Index add failed");
        }

        let source_tier = authority::classify_domain(&domain);
        let body_for_embed = extraction.body_text.clone();

        let candidate = RankCandidate {
            url: crawled.final_url.clone(),
            domain,
            title: extraction.title.unwrap_or_default(),
            body_text: extraction.body_text,
            published_date: extraction.published_date,
            source_tier,
            bm25_score: None,
            vector_score: None,
            bm25_rank: None,
            vector_rank: None,
            embedding: None, // filled in Phase 2
        };

        // Cache for future queries
        self.url_cache.insert(crawled.final_url.clone(), CachedExtraction {
            candidate: candidate.clone(),
            body_for_embed: body_for_embed.clone(),
            cached_at: Instant::now(),
        });

        // Evict oldest if cache too large (cap at 500 URLs)
        if self.url_cache.len() > 500 {
            // Remove ~50 oldest entries
            let mut oldest: Vec<(String, Instant)> = self.url_cache.iter()
                .map(|e| (e.key().clone(), e.value().cached_at))
                .collect();
            oldest.sort_by_key(|(_k, t)| *t);
            for (key, _) in oldest.iter().take(50) {
                self.url_cache.remove(key);
            }
        }

        Some((candidate, body_for_embed))
    }
}

/// Generate seed search URLs from a query.
///
/// Uses multiple search engines and direct knowledge sources to maximize
/// coverage without relying on any single API.
/// Result from SearXNG JSON API — contains URL, title, and snippet.
/// Used for snippet-only ranking (instant_search) and fast URL extraction.
#[derive(Debug, Clone)]
struct SearxngResult {
    url: String,
    title: String,
    snippet: String,
}

/// Fetch SearXNG JSON API directly and return parsed results.
/// Single HTTP request, ~200-500ms. Returns up to 20 results with snippets.
async fn fetch_searxng_results(
    searxng_url: &str,
    query: &str,
) -> Vec<SearxngResult> {
    let encoded = urlencoding_simple(query);
    let base = searxng_url.trim_end_matches('/');
    let api_url = format!("{base}/search?q={encoded}&format=json&categories=general&pageno=1");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap_or_default();

    match client.get(&api_url).send().await {
        Ok(resp) => {
            if let Ok(text) = resp.text().await {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                    if let Some(items) = json["results"].as_array() {
                        let mut results = Vec::new();
                        let mut seen = std::collections::HashSet::new();
                        for item in items {
                            let url = match item["url"].as_str() {
                                Some(u) if u.starts_with("http") => u,
                                _ => continue,
                            };
                            if !seen.insert(url.to_string()) { continue; }
                            results.push(SearxngResult {
                                url: url.to_string(),
                                title: item["title"].as_str().unwrap_or("").to_string(),
                                snippet: item["content"].as_str().unwrap_or("").to_string(),
                            });
                        }
                        tracing::info!(
                            results = results.len(),
                            elapsed_source = "searxng",
                            "SearXNG fast-path: got results"
                        );
                        return results;
                    }
                }
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, "SearXNG fast-path fetch failed");
        }
    }
    Vec::new()
}

fn generate_search_seeds(query: &str, searxng_url: Option<&str>) -> Vec<String> {
    let encoded = urlencoding_simple(query);
    let _encoded_dash = query.to_lowercase().replace(' ', "-");
    let encoded_underscore = query.to_lowercase().replace(' ', "_");

    let mut seeds = Vec::new();

    // SearXNG metasearch — primary source when configured.
    // Aggregates Google+Bing+DDG in one request, returns clean JSON, no CAPTCHA.
    // Self-hosted: docker run -d -p 8080:8080 searxng/searxng
    if let Some(base) = searxng_url {
        let base = base.trim_end_matches('/');
        // Query across multiple categories for maximum coverage
        seeds.push(format!("{base}/search?q={encoded}&format=json&categories=general&pageno=1"));
        seeds.push(format!("{base}/search?q={encoded}&format=json&categories=general&pageno=2"));
        seeds.push(format!("{base}/search?q={encoded}&format=json&categories=science"));
        seeds.push(format!("{base}/search?q={encoded}&format=json&categories=it"));
        seeds.push(format!("{base}/search?q={encoded}&format=json&categories=news"));
    }

    // Fallback public SearXNG instances (may 403 — best-effort)
    if searxng_url.is_none() {
        for instance in &[
            "https://search.sapti.me",
            "https://searx.tiekoetter.com",
            "https://search.inetol.net",
            "https://searx.be",
        ] {
            seeds.push(format!("{instance}/search?q={encoded}&format=json&categories=general"));
        }
    }

    seeds.extend([
        // Major search engines (fallback — often blocked/JS-rendered)
        format!("https://www.google.com/search?q={encoded}&num=20"),
        format!("https://www.bing.com/search?q={encoded}&count=20"),
        // Alternative search engines (HTML versions, no API key)
        format!("https://search.brave.com/search?q={encoded}"),
        format!("https://www.mojeek.com/search?q={encoded}"),
        format!("https://html.duckduckgo.com/html/?q={encoded}"),
        // Wikipedia — direct article attempt + search
        format!("https://en.wikipedia.org/wiki/{encoded_underscore}"),
        format!("https://en.wikipedia.org/w/index.php?search={encoded}&title=Special:Search"),
        // Reddit discussions
        format!("https://old.reddit.com/search/?q={encoded}&sort=relevance&t=year"),
        // StackExchange
        format!("https://stackexchange.com/search?q={encoded}"),
        // ArXiv for academic papers
        format!("https://arxiv.org/search/?query={encoded}&searchtype=all"),
        // Hacker News — use Algolia API (returns JSON, SPA frontend is unscrapable)
        format!("https://hn.algolia.com/api/v1/search?query={encoded}&tags=story"),
    ]);

    // Add official/canonical domain seeds for detected entities
    let official_domains = web_search_ranker::entity_domain::detect_official_domains(query);
    for domain in &official_domains {
        // Site-specific search on Google to find relevant pages on the official domain
        seeds.insert(0, format!("https://www.google.com/search?q=site%3A{domain}+{encoded}"));
    }

    // Add topic-specific sources based on query keywords
    let q = query.to_lowercase();
    if q.contains("code") || q.contains("programming") || q.contains("software") || q.contains("api") {
        seeds.push(format!("https://github.com/search?q={encoded}&type=repositories"));
        seeds.push(format!("https://docs.rs/releases/search?query={encoded}"));
    }
    if q.contains("science") || q.contains("research") || q.contains("study") || q.contains("effect") || q.contains("impact") {
        seeds.push(format!("https://scholar.google.com/scholar?q={encoded}"));
        seeds.push(format!("https://pubmed.ncbi.nlm.nih.gov/?term={encoded}"));
    }
    if q.contains("news") || q.contains("latest") || q.contains("today") {
        seeds.push(format!("https://news.google.com/search?q={encoded}"));
    }

    seeds
}

fn urlencoding_simple(s: &str) -> String {
    s.replace(' ', "+")
        .replace('&', "%26")
        .replace('=', "%3D")
        .replace('?', "%3F")
        .replace('#', "%23")
}

fn sha256_short(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())[..16].to_string()
}

/// Detect low-quality content: error pages, empty search results, CAPTCHAs, etc.
fn is_low_quality_content(body: &str, url: &str) -> bool {
    let lower = body.to_lowercase();
    let word_count = body.split_whitespace().count();

    // Too short to be useful content (after extraction)
    // Relaxed from 30 — some legitimate pages (API docs, changelogs) are short
    if word_count < 15 {
        return true;
    }

    // "No results" pages from search engines
    let no_result_phrases = [
        "no results matching",
        "no results found",
        "your search did not match",
        "did not return any results",
        "0 results",
        "no documents match",
        "there were no results",
    ];
    if no_result_phrases.iter().any(|p| lower.contains(p)) {
        // If "no results" is prominent (first 500 chars), it's an error page
        let head: String = lower.chars().take(500).collect();
        if no_result_phrases.iter().any(|p| head.contains(p)) {
            return true;
        }
    }

    // CAPTCHA / access denied pages
    let block_phrases = [
        "please verify you are a human",
        "complete the captcha",
        "access denied",
        "403 forbidden",
        "enable javascript to continue",
        "please enable cookies",
        "checking your browser",
        "just a moment...",
    ];
    // Only reject CAPTCHA pages if they're truly just a challenge page
    // Raised from 200 — some real pages mention "enable javascript" in footers
    if block_phrases.iter().any(|p| lower.contains(p)) && word_count < 500 {
        // Double-check: if the page has substantial content beyond the block phrase, keep it
        let block_count = block_phrases.iter().filter(|p| lower.contains(*p)).count();
        if block_count >= 2 || word_count < 100 {
            return true;
        }
    }

    // Search result listing pages that leaked through (Wikidata, generic search pages)
    let url_lower = url.to_lowercase();
    let is_search_page = url_lower.contains("special:search")
        || url_lower.contains("wikidata.org/w/index.php?search")
        || (url_lower.contains("search") && url_lower.contains("title=Special"));

    if is_search_page {
        return true;
    }

    // Google News homepage / aggregator pages (not actual articles)
    if url_lower.contains("news.google.com") && !url_lower.contains("/articles/") && !url_lower.contains("/stories/") {
        return true;
    }

    // General news aggregator / index page detection
    let aggregator_phrases = [
        "top stories", "trending now", "more headlines",
        "latest news", "breaking news", "see more headlines",
        "follow this topic", "chevron_right",
    ];
    let aggregator_hits = aggregator_phrases.iter().filter(|p| lower.contains(*p)).count();
    if aggregator_hits >= 3 {
        return true;
    }

    // Very high ratio of boilerplate signals (navigation text, footers, etc.)
    let nav_phrases = ["privacy policy", "terms of use", "cookie policy", "sign in", "log in", "create account"];
    let nav_count = nav_phrases.iter().filter(|p| lower.contains(*p)).count();
    if nav_count >= 3 && word_count < 100 {
        return true;
    }

    false
}

fn extract_domain(url: &str) -> String {
    url::Url::parse(url)
        .ok()
        .and_then(|u| u.host_str().map(|h| h.to_string()))
        .unwrap_or_else(|| "unknown".to_string())
}

impl Drop for SearchEngine {
    fn drop(&mut self) {
        self.flush_to_disk();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_seeds_produces_urls() {
        let seeds = generate_search_seeds("rust programming language", None);
        assert!(seeds.len() >= 5, "Should generate multiple seed URLs");
        // Should include major search engines
        let all = seeds.join(" ");
        assert!(all.contains("brave.com") || all.contains("mojeek.com"));
        assert!(all.contains("wikipedia.org"));
        assert!(all.contains("rust+programming+language") || all.contains("rust_programming_language"));
    }

    #[test]
    fn url_encoding_works() {
        assert_eq!(urlencoding_simple("hello world"), "hello+world");
        assert_eq!(urlencoding_simple("a&b=c"), "a%26b%3Dc");
    }

    #[test]
    fn sha256_short_deterministic() {
        let h1 = sha256_short("test content");
        let h2 = sha256_short("test content");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 16);
    }

    #[test]
    fn extract_domain_works() {
        assert_eq!(extract_domain("https://www.example.com/path"), "www.example.com");
        assert_eq!(extract_domain("invalid"), "unknown");
    }

    #[tokio::test]
    async fn engine_creates_with_default_config() {
        let config = Config::default();
        let engine = SearchEngine::new(config);
        assert!(engine.is_ok());
    }
}
