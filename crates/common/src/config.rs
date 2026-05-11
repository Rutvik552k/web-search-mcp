use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub crawler: CrawlerConfig,
    pub extractor: ExtractorConfig,
    pub indexer: IndexerConfig,
    pub embedder: EmbedderConfig,
    pub ranker: RankerConfig,
    pub server: ServerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrawlerConfig {
    /// Number of concurrent crawler workers
    pub num_workers: usize,
    /// Maximum concurrent connections total
    pub max_concurrent_connections: usize,
    /// Per-domain requests per second limit
    pub requests_per_second_per_domain: f64,
    /// Default timeout per request in seconds
    pub request_timeout_secs: u64,
    /// User-Agent string
    pub user_agent: String,
    /// Respect robots.txt
    pub respect_robots_txt: bool,
    /// Enable headless browser for JS-heavy pages
    pub enable_browser: bool,
    /// Maximum retry attempts per URL
    pub max_retries: u32,
    /// Backoff base in milliseconds for retries
    pub backoff_base_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractorConfig {
    /// Enable ML-based content classification (requires ONNX model)
    pub enable_ml_classifier: bool,
    /// Path to content classifier ONNX model
    pub classifier_model_path: PathBuf,
    /// Minimum consensus votes to keep a content block (2 or 3)
    pub min_consensus_votes: u8,
    /// Maximum text chunk size in tokens for embedding
    pub chunk_size_tokens: usize,
    /// Chunk overlap ratio (0.0 - 0.5)
    pub chunk_overlap_ratio: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexerConfig {
    /// Path to tantivy index directory
    pub index_path: PathBuf,
    /// Path to HNSW vector index directory
    pub vector_index_path: PathBuf,
    /// Heap size for tantivy writer in bytes
    pub tantivy_heap_size: usize,
    /// HNSW M parameter (max connections per node)
    pub hnsw_m: usize,
    /// HNSW ef_construction parameter
    pub hnsw_ef_construction: usize,
    /// SimHash hamming distance threshold for near-duplicate
    pub simhash_threshold: u32,
    /// Enable Product Quantization for vectors
    pub enable_pq: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedderConfig {
    /// Path to embedding ONNX model
    pub embedding_model_path: PathBuf,
    /// Embedding dimensions (384 for MiniLM-L6)
    pub embedding_dim: usize,
    /// Batch size for embedding generation
    pub batch_size: usize,
    /// Force CPU even if GPU available
    pub force_cpu: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankerConfig {
    /// Path to cross-encoder ONNX model
    pub cross_encoder_model_path: PathBuf,
    /// BM25 top-K for Stage 1
    pub bm25_top_k: usize,
    /// HNSW top-K for Stage 1
    pub hnsw_top_k: usize,
    /// Cross-encoder top-K for Stage 2 output
    pub rerank_top_k: usize,
    /// Minimum sources for claim verification
    pub min_verification_sources: usize,
    /// MMR lambda (0.0 = pure diversity, 1.0 = pure relevance)
    pub mmr_lambda: f32,
    /// Maximum results per domain in final output
    pub max_results_per_domain: usize,
    /// Minimum unique source organizations in final output
    pub min_unique_orgs: usize,
    /// Path to source tiers config
    pub source_tiers_path: PathBuf,
    /// Minimum relevance score for final results (0.0 - 1.0).
    /// Results below this are filtered after Stage 2 cross-encoder reranking.
    #[serde(default = "default_min_relevance_score")]
    pub min_relevance_score: f32,
}

fn default_min_relevance_score() -> f32 {
    0.35
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Server name reported via MCP
    pub name: String,
    /// Server version
    pub version: String,
    /// Data directory for runtime storage
    pub data_dir: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        let data_dir = PathBuf::from("data");
        Self {
            crawler: CrawlerConfig {
                num_workers: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).max(4),
                max_concurrent_connections: 500,
                requests_per_second_per_domain: 2.0,
                request_timeout_secs: 30,
                user_agent: "WebSearchMCP/0.1 (research-bot)".to_string(),
                respect_robots_txt: true,
                enable_browser: false,
                max_retries: 3,
                backoff_base_ms: 1000,
            },
            extractor: ExtractorConfig {
                enable_ml_classifier: false,
                classifier_model_path: PathBuf::from("models/content-clf.onnx"),
                min_consensus_votes: 2,
                chunk_size_tokens: 512,
                chunk_overlap_ratio: 0.2,
            },
            indexer: IndexerConfig {
                index_path: data_dir.join("index"),
                vector_index_path: data_dir.join("vectors"),
                tantivy_heap_size: 50_000_000, // 50MB
                hnsw_m: 16,
                hnsw_ef_construction: 200,
                simhash_threshold: 3,
                enable_pq: false,
            },
            embedder: EmbedderConfig {
                embedding_model_path: PathBuf::from("models/minilm-l6-v2.onnx"),
                embedding_dim: 384,
                batch_size: 32,
                force_cpu: false,
            },
            ranker: RankerConfig {
                cross_encoder_model_path: PathBuf::from("models/cross-encoder.onnx"),
                bm25_top_k: 200,
                hnsw_top_k: 200,
                rerank_top_k: 50,
                min_verification_sources: 3,
                mmr_lambda: 0.7,
                max_results_per_domain: 2,
                min_unique_orgs: 3,
                source_tiers_path: PathBuf::from("config/source_tiers.toml"),
                min_relevance_score: 0.35,
            },
            server: ServerConfig {
                name: "web-search-mcp".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                data_dir,
            },
        }
    }
}

impl Config {
    /// Load config from a TOML file, falling back to defaults for missing fields.
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(path)
            .map_err(|e| Error::Config(format!("failed to read config: {e}")))?;
        let config: Self = toml::from_str(&content)
            .map_err(|e| Error::Config(format!("failed to parse config: {e}")))?;
        Ok(config)
    }
}
