use std::path::Path;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy};
use web_search_common::models::Page;
use web_search_common::{Error, Result};

use crate::schema::build_schema;

/// Full-text search index backed by tantivy (mmap).
pub struct TextIndex {
    index: Index,
    reader: IndexReader,
    writer: parking_lot::Mutex<IndexWriter>,
    schema: Schema,
}

/// A document retrieved from the text index.
#[derive(Debug, Clone)]
pub struct TextSearchResult {
    pub doc_id: u64,
    pub url: String,
    pub title: String,
    pub domain: String,
    pub score: f32,
    pub source_tier: u64,
}

impl TextIndex {
    /// Open or create a tantivy index at the given path.
    pub fn open(path: &Path, heap_size: usize) -> Result<Self> {
        let schema = build_schema();

        let index = if path.exists() && path.join("meta.json").exists() {
            Index::open_in_dir(path).map_err(|e| Error::Index(format!("open index: {e}")))?
        } else {
            std::fs::create_dir_all(path)?;
            Index::create_in_dir(path, schema.clone())
                .map_err(|e| Error::Index(format!("create index: {e}")))?
        };

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .map_err(|e| Error::Index(format!("reader: {e}")))?;

        let writer = index
            .writer(heap_size)
            .map_err(|e| Error::Index(format!("writer: {e}")))?;

        Ok(Self {
            index,
            reader,
            writer: parking_lot::Mutex::new(writer),
            schema,
        })
    }

    /// Create an in-memory index (for testing).
    pub fn in_memory(heap_size: usize) -> Result<Self> {
        let schema = build_schema();
        let index = Index::create_from_tempdir(schema.clone())
            .map_err(|e| Error::Index(format!("create temp index: {e}")))?;

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(|e| Error::Index(format!("reader: {e}")))?;

        let writer = index
            .writer(heap_size)
            .map_err(|e| Error::Index(format!("writer: {e}")))?;

        Ok(Self {
            index,
            reader,
            writer: parking_lot::Mutex::new(writer),
            schema,
        })
    }

    /// Index a page.
    pub fn add_page(&self, page: &Page) -> Result<u64> {
        let url_field = self.schema.get_field("url").unwrap();
        let title_field = self.schema.get_field("title").unwrap();
        let domain_field = self.schema.get_field("domain").unwrap();
        let body_field = self.schema.get_field("body").unwrap();
        let content_hash_field = self.schema.get_field("content_hash").unwrap();
        let source_tier_field = self.schema.get_field("source_tier").unwrap();
        let confidence_field = self.schema.get_field("extraction_confidence").unwrap();

        let title = page.title.as_deref().unwrap_or("");

        let writer = self.writer.lock();
        let stamp = writer.add_document(doc!(
            url_field => page.url.as_str(),
            title_field => title,
            domain_field => page.domain.as_str(),
            body_field => page.body_text.as_str(),
            content_hash_field => page.content_hash.as_bytes(),
            source_tier_field => 4u64, // default tier 4, can be updated
            confidence_field => page.metadata.extraction_confidence as f64,
        )).map_err(|e| Error::Index(format!("add doc: {e}")))?;

        Ok(stamp)
    }

    /// Commit pending writes.
    pub fn commit(&self) -> Result<()> {
        let mut writer = self.writer.lock();
        writer
            .commit()
            .map_err(|e| Error::Index(format!("commit: {e}")))?;
        self.reader
            .reload()
            .map_err(|e| Error::Index(format!("reload: {e}")))?;
        Ok(())
    }

    /// Search using BM25. Returns top-K results.
    pub fn search(&self, query_str: &str, top_k: usize) -> Result<Vec<TextSearchResult>> {
        let url_field = self.schema.get_field("url").unwrap();
        let title_field = self.schema.get_field("title").unwrap();
        let domain_field = self.schema.get_field("domain").unwrap();
        let body_field = self.schema.get_field("body").unwrap();
        let source_tier_field = self.schema.get_field("source_tier").unwrap();

        let searcher = self.reader.searcher();
        let query_parser = QueryParser::for_index(&self.index, vec![title_field, body_field]);

        let query = query_parser
            .parse_query(query_str)
            .map_err(|e| Error::Index(format!("parse query: {e}")))?;

        let top_docs = searcher
            .search(&query, &TopDocs::with_limit(top_k))
            .map_err(|e| Error::Index(format!("search: {e}")))?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let doc = searcher
                .doc::<TantivyDocument>(doc_address)
                .map_err(|e| Error::Index(format!("fetch doc: {e}")))?;

            let url = doc
                .get_first(url_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let title = doc
                .get_first(title_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let domain = doc
                .get_first(domain_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let source_tier = doc
                .get_first(source_tier_field)
                .and_then(|v| v.as_u64())
                .unwrap_or(4);

            results.push(TextSearchResult {
                doc_id: doc_address.doc_id as u64,
                url,
                title,
                domain,
                score,
                source_tier,
            });
        }

        Ok(results)
    }

    /// Get total number of documents in the index.
    pub fn num_docs(&self) -> u64 {
        let searcher = self.reader.searcher();
        searcher.num_docs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use web_search_common::models::*;

    fn make_page(url: &str, title: &str, body: &str) -> Page {
        Page {
            url: url.to_string(),
            domain: url::Url::parse(url)
                .map(|u| u.host_str().unwrap_or("").to_string())
                .unwrap_or_default(),
            title: Some(title.to_string()),
            author: None,
            published_date: None,
            body_text: body.to_string(),
            headings: vec![],
            links: vec![],
            tables: vec![],
            metadata: PageMetadata {
                language: None,
                description: None,
                content_type: "text/html".to_string(),
                status_code: 200,
                response_time_ms: 100,
                content_length: body.len(),
                extraction_confidence: 0.95,
                json_ld: None,
                open_graph: None,
            },
            content_hash: "abc123".to_string(),
            crawled_at: Utc::now(),
        }
    }

    #[test]
    fn index_and_search() {
        let idx = TextIndex::in_memory(15_000_000).unwrap();

        let page1 = make_page(
            "https://example.com/rust",
            "Rust Programming Language",
            "Rust is a systems programming language focused on safety and performance",
        );
        let page2 = make_page(
            "https://example.com/python",
            "Python Programming",
            "Python is a high-level interpreted language used for data science",
        );

        idx.add_page(&page1).unwrap();
        idx.add_page(&page2).unwrap();
        idx.commit().unwrap();

        assert_eq!(idx.num_docs(), 2);

        let results = idx.search("rust safety performance", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].url, "https://example.com/rust");
    }

    #[test]
    fn search_returns_ranked_by_relevance() {
        let idx = TextIndex::in_memory(15_000_000).unwrap();

        let pages = vec![
            make_page("https://a.com", "Climate", "climate change global warming temperature rise"),
            make_page("https://b.com", "Cooking", "recipe chocolate cake baking dessert"),
            make_page("https://c.com", "Weather", "climate patterns weather forecasting prediction"),
        ];

        for p in &pages {
            idx.add_page(p).unwrap();
        }
        idx.commit().unwrap();

        let results = idx.search("climate change", 10).unwrap();
        assert!(results.len() >= 1);
        // First result should be the most relevant (climate change article)
        assert!(
            results[0].url == "https://a.com" || results[0].url == "https://c.com",
            "Top result should be climate-related, got {}",
            results[0].url
        );
    }

    #[test]
    fn empty_index_returns_no_results() {
        let idx = TextIndex::in_memory(15_000_000).unwrap();
        let results = idx.search("anything", 10).unwrap();
        assert!(results.is_empty());
    }
}
