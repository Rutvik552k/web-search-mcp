use sha2::{Digest, Sha256};
use std::time::{Duration, Instant};
use web_search_common::config::Config;
use web_search_common::models::*;
use web_search_common::Result;
use web_search_crawler::Crawler;
use web_search_embedder::{self, Embedder};
use web_search_extractor::{self, consensus};
use web_search_indexer::dedup::{DedupResult, DedupStore};
use web_search_indexer::hnsw::HnswIndex;
use web_search_indexer::text_index::TextIndex;
use web_search_ranker::authority;
use web_search_ranker::pipeline::{RankCandidate, RankingPipeline};

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
}

impl SearchEngine {
    /// Create a new SearchEngine from config.
    pub fn new(config: Config) -> Result<Self> {
        let crawler = Crawler::new(config.crawler.clone())?;

        let text_index = TextIndex::in_memory(config.indexer.tantivy_heap_size)?;
        let vector_index = HnswIndex::new(config.embedder.embedding_dim);
        let dedup = DedupStore::new(config.indexer.simhash_threshold);
        let embedder = web_search_embedder::create_embedder(&config.embedder);
        let pipeline = RankingPipeline::new(config.ranker.clone());

        Ok(Self {
            crawler,
            text_index,
            vector_index,
            dedup,
            embedder,
            pipeline,
            _config: config,
        })
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

        // Generate seed URLs from query
        let seeds = generate_search_seeds(query);
        let seed_refs: Vec<&str> = seeds.iter().map(|s| s.as_str()).collect();

        // Wave 1: Broad crawl
        let wave1_limit = (max_pages / 2).max(10);
        let wave1_time = Duration::from_secs(time_limit_secs / 2);
        let pages = self.crawler.crawl(&seed_refs, wave1_limit, max_depth, wave1_time).await;

        tracing::info!(wave1_pages = pages.len(), "Wave 1 complete");

        // Process crawled pages
        let mut candidates = Vec::new();
        for page in &pages {
            if let Some(candidate) = self.process_crawled_page(page).await {
                candidates.push(candidate);
            }
        }

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

                for page in &wave2_pages {
                    if let Some(candidate) = self.process_crawled_page(page).await {
                        candidates.push(candidate);
                    }
                }
            }
        }

        // Commit index
        self.text_index.commit()?;

        // Run ranking pipeline
        let top_k = 10;
        let response = self.pipeline.rank(candidates, query, top_k);

        tracing::info!(
            total_crawled = pages.len(),
            results = response.results.len(),
            elapsed_ms = start.elapsed().as_millis(),
            "Deep research complete"
        );

        Ok(response)
    }

    /// Quick search: single-wave crawl, fast ranking.
    pub async fn quick_search(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<SearchResponse> {
        let seeds = generate_search_seeds(query);
        let seed_refs: Vec<&str> = seeds.iter().map(|s| s.as_str()).collect();

        let pages = self.crawler.crawl(
            &seed_refs,
            50,
            1, // shallow
            Duration::from_secs(15),
        ).await;

        let mut candidates = Vec::new();
        for page in &pages {
            if let Some(candidate) = self.process_crawled_page(page).await {
                candidates.push(candidate);
            }
        }

        self.text_index.commit()?;
        Ok(self.pipeline.rank(candidates, query, max_results))
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
        Ok(serde_json::to_string_pretty(&serde_json::json!({
            "query": query,
            "total_docs_in_index": self.text_index.num_docs(),
            "results": results.iter().map(|r| serde_json::json!({
                "url": r.url,
                "title": r.title,
                "domain": r.domain,
                "score": r.score,
            })).collect::<Vec<_>>(),
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

    // ── Internal ─────────────────────────────────────────────────────

    /// Process a crawled page: extract, dedup, index, embed.
    async fn process_crawled_page(
        &self,
        crawled: &web_search_crawler::crawler::CrawledPage,
    ) -> Option<RankCandidate> {
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

        if extraction.body_text.len() < 50 {
            tracing::debug!(url = %crawled.final_url, "Skipping: too little content");
            return None;
        }

        // Build Page model
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

        // Index in tantivy
        if let Err(e) = self.text_index.add_page(&page) {
            tracing::warn!(url = %crawled.final_url, error = %e, "Index add failed");
        }

        // Embed and add to vector index
        let embedding = match self.embedder.embed_one(&extraction.body_text).await {
            Ok(vec) => {
                if let Err(e) = self.vector_index.insert(&crawled.final_url, vec.clone()) {
                    tracing::warn!(url = %crawled.final_url, error = %e, "Vector insert failed");
                }
                Some(vec)
            }
            Err(e) => {
                tracing::debug!(url = %crawled.final_url, error = %e, "Embedding failed");
                None
            }
        };

        // Build ranking candidate
        let source_tier = authority::classify_domain(&domain);

        Some(RankCandidate {
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
            embedding,
        })
    }
}

/// Generate seed search URLs from a query.
///
/// Uses multiple search engines and direct knowledge sources to maximize
/// coverage without relying on any single API.
fn generate_search_seeds(query: &str) -> Vec<String> {
    let encoded = urlencoding_simple(query);
    let encoded_dash = query.to_lowercase().replace(' ', "-");
    let encoded_underscore = query.to_lowercase().replace(' ', "_");

    let mut seeds = vec![
        // Search engines (HTML versions, no API key)
        format!("https://search.brave.com/search?q={encoded}"),
        format!("https://www.mojeek.com/search?q={encoded}"),
        format!("https://html.duckduckgo.com/html/?q={encoded}"),
        // Wikipedia — direct article attempt + search
        format!("https://en.wikipedia.org/wiki/{encoded_underscore}"),
        format!("https://en.wikipedia.org/w/index.php?search={encoded}&title=Special:Search"),
        // Wikidata for structured knowledge
        format!("https://www.wikidata.org/w/index.php?search={encoded}&ns0=1"),
        // Reddit discussions
        format!("https://old.reddit.com/search/?q={encoded}&sort=relevance&t=year"),
        // StackExchange
        format!("https://stackexchange.com/search?q={encoded}"),
        // ArXiv for academic papers
        format!("https://arxiv.org/search/?query={encoded}&searchtype=all"),
        // Hacker News
        format!("https://hn.algolia.com/?q={encoded}"),
    ];

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
    if q.contains("news") || q.contains("latest") || q.contains("today") || q.contains("2024") || q.contains("2025") || q.contains("2026") {
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
    fn generate_seeds_produces_urls() {
        let seeds = generate_search_seeds("rust programming language");
        assert!(!seeds.is_empty());
        assert!(seeds[0].contains("duckduckgo"));
        assert!(seeds[0].contains("rust+programming+language"));
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

    #[test]
    fn engine_creates_with_default_config() {
        let config = Config::default();
        let engine = SearchEngine::new(config);
        assert!(engine.is_ok());
    }
}
