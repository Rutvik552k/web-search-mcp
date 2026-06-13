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

    // CAPTCHA solver (R5 — Design 0004 Part 2, ADR 0003 R5).
    // `reason` is a sanitized, secret-free description (provider error code/text,
    // "timeout", "cost cap exceeded", etc.). The API key and the solved token are
    // NEVER placed in this message (api-security ERROR HANDLING + NO SECRETS IN
    // LOGS; llm-safety NO SECRETS IN PROMPTS).
    #[error("captcha solve failed via {provider}: {reason}")]
    Captcha { provider: String, reason: String },

    // Hybrid escalation controller (Design 0005 H6 / ADR 0001 C4 / ADR 0003 C4).
    // `vendor` is the challenge vendor name (e.g. "cloudflare", "datadome",
    // "turnstile") — never a secret/cookie/token. Returned when the controller
    // climbed to the solver/stealth rung and still could not clear the
    // challenge. Distinct from `Blocked` (a raw status) so callers can tell
    // "we tried to solve and failed" from "the origin returned an error".
    #[error("challenge unsolved for {url} (vendor {vendor})")]
    ChallengeUnsolved { url: String, vendor: String },

    // Returned when a live-origin fetch is REFUSED because the domain is on the
    // permanent denylist (ADR 0003 §4.2 single-IP hard-stop). Lets callers
    // distinguish "we deliberately never contacted this origin" from "we tried
    // and were blocked". `reason` is a fixed enum-ish label
    // (e.g. "hard-ban", "explicit-block", "cease-and-desist") — never a secret.
    #[error("permanently denied: {domain} ({reason})")]
    PermanentlyDenied { domain: String, reason: String },

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
