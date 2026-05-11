use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    // Crawling errors
    #[error("crawl failed for {url}: {reason}")]
    Crawl { url: String, reason: String },

    #[error("request failed: {0}")]
    Http(String),

    #[error("url parse error: {0}")]
    UrlParse(#[from] url::ParseError),

    #[error("blocked by site: {url} (status {status})")]
    Blocked { url: String, status: u16 },

    #[error("rate limited: {domain} (retry after {retry_after_secs}s)")]
    RateLimited {
        domain: String,
        retry_after_secs: u64,
    },

    #[error("robots.txt disallows: {url}")]
    RobotsDisallowed { url: String },

    // Extraction errors
    #[error("extraction failed for {url}: {reason}")]
    Extraction { url: String, reason: String },

    #[error("unsupported content type: {content_type}")]
    UnsupportedContentType { content_type: String },

    // Indexing errors
    #[error("index error: {0}")]
    Index(String),

    #[error("tantivy error: {0}")]
    Tantivy(String),

    // Embedding errors
    #[error("embedding error: {0}")]
    Embedding(String),

    #[error("model not found: {path}")]
    ModelNotFound { path: String },

    // Ranking errors
    #[error("ranking error: {0}")]
    Ranking(String),

    // Orchestration errors
    #[error("research budget exhausted: {reason}")]
    BudgetExhausted { reason: String },

    #[error("timeout after {elapsed_secs}s (limit: {limit_secs}s)")]
    Timeout {
        elapsed_secs: u64,
        limit_secs: u64,
    },

    // General
    #[error("config error: {0}")]
    Config(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("{0}")]
    Other(String),
}
