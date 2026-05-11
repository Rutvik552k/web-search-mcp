use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A crawled and extracted web page.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Page {
    pub url: String,
    pub domain: String,
    pub title: Option<String>,
    pub author: Option<String>,
    pub published_date: Option<DateTime<Utc>>,
    pub body_text: String,
    pub headings: Vec<Heading>,
    pub links: Vec<Link>,
    pub tables: Vec<Table>,
    pub metadata: PageMetadata,
    pub content_hash: String,
    pub crawled_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Heading {
    pub level: u8, // 1-6
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Link {
    pub url: String,
    pub anchor_text: String,
    pub is_external: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Table {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageMetadata {
    pub language: Option<String>,
    pub description: Option<String>,
    pub content_type: String,
    pub status_code: u16,
    pub response_time_ms: u64,
    pub content_length: usize,
    pub extraction_confidence: f32, // 0.0 - 1.0 from consensus voting
    pub json_ld: Option<serde_json::Value>,
    pub open_graph: Option<OpenGraphData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenGraphData {
    pub og_title: Option<String>,
    pub og_description: Option<String>,
    pub og_image: Option<String>,
    pub og_type: Option<String>,
    pub og_site_name: Option<String>,
}

/// An atomic claim extracted from a page, with source attribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claim {
    pub text: String,
    pub source_url: String,
    pub source_span: (usize, usize), // start, end byte offsets in body_text
    pub confidence: f32,
    pub verification: VerificationStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationStatus {
    /// 3+ independent sources confirm this claim
    Verified,
    /// 2 sources confirm
    Partial,
    /// Only 1 source (the original)
    Unverified,
    /// Other sources contradict this claim
    Contested,
}

/// A contradiction detected between sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contradiction {
    pub claim_a: String,
    pub source_a: String,
    pub claim_b: String,
    pub source_b: String,
    pub severity: ContradictionSeverity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContradictionSeverity {
    /// Direct factual contradiction (e.g., different numbers)
    Hard,
    /// Nuanced disagreement (e.g., different interpretation)
    Soft,
    /// Temporal difference (was true, no longer true)
    Temporal,
}

/// An entity extracted via NER.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub name: String,
    pub entity_type: EntityType,
    pub source_url: String,
    pub mentions: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityType {
    Person,
    Organization,
    Location,
    Date,
    Number,
    Event,
    Product,
    Other,
}

/// Source credibility tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SourceTier {
    /// .gov, .edu, major journals, WHO, NASA
    Tier1 = 1,
    /// Established news, official docs, tech documentation
    Tier2 = 2,
    /// Blogs, forums, wikis, community sites
    Tier3 = 3,
    /// Unknown, user-generated, unverified
    Tier4 = 4,
}

impl SourceTier {
    pub fn weight(&self) -> f32 {
        match self {
            SourceTier::Tier1 => 1.3,
            SourceTier::Tier2 => 1.1,
            SourceTier::Tier3 => 0.9,
            SourceTier::Tier4 => 0.7,
        }
    }
}

/// A single ranked search result returned to the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedResult {
    pub content: String,
    pub url: String,
    pub title: String,
    pub confidence: f32,
    pub verification: VerificationStatus,
    pub claims: Vec<Claim>,
    pub contradictions: Vec<Contradiction>,
    pub source_tier: SourceTier,
    pub freshness: Option<DateTime<Utc>>,
    pub relevance_score: f32,
}

/// The complete response from a search/research operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<RankedResult>,
    pub warnings: Vec<String>,
    pub coverage_score: f32,
    pub total_pages_crawled: usize,
    pub total_time_ms: u64,
    pub query: String,
}

/// Query type detected from keywords for adaptive ranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryType {
    /// Breaking news, current events
    News,
    /// Academic, scientific, deep analysis
    Research,
    /// Simple factual question
    Factual,
    /// Code, API, technical documentation
    Technical,
    /// General/unknown
    General,
}

/// Budget constraints for research operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchBudget {
    pub max_pages: usize,
    pub time_limit_secs: u64,
    pub max_depth: u8,
    pub max_pages_per_domain: usize,
}

impl Default for ResearchBudget {
    fn default() -> Self {
        Self {
            max_pages: 500,
            time_limit_secs: 120,
            max_depth: 3,
            max_pages_per_domain: 50,
        }
    }
}
