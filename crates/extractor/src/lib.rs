pub mod chunker;
pub mod metadata;
pub mod readability;
pub mod trafilatura;
pub mod consensus;

pub use consensus::extract_page;

use chrono::{DateTime, Utc};
use web_search_common::models::{Heading, Link, Table, OpenGraphData};

/// Intermediate extraction result from a single pass.
#[derive(Debug, Clone)]
pub struct ExtractionPass {
    pub title: Option<String>,
    pub body_text: String,
    pub headings: Vec<Heading>,
    pub confidence: f32,
}

/// Complete extraction result after consensus voting.
#[derive(Debug, Clone)]
pub struct ExtractionResult {
    pub title: Option<String>,
    pub author: Option<String>,
    pub published_date: Option<DateTime<Utc>>,
    pub body_text: String,
    pub headings: Vec<Heading>,
    pub links: Vec<Link>,
    pub tables: Vec<Table>,
    pub language: Option<String>,
    pub description: Option<String>,
    pub json_ld: Option<serde_json::Value>,
    pub open_graph: Option<OpenGraphData>,
    pub extraction_confidence: f32,
}
