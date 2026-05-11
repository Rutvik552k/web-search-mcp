use web_search_common::Result;
use web_search_embedder::Embedder;

use crate::hnsw::{HnswIndex, VectorSearchResult};
use crate::text_index::{TextIndex, TextSearchResult};

/// Unified search result from hybrid retrieval.
#[derive(Debug, Clone)]
pub struct HybridSearchResult {
    pub doc_id: String,
    pub url: String,
    pub title: String,
    pub domain: String,
    pub isr_score: f32,
    pub bm25_rank: Option<usize>,
    pub vector_rank: Option<usize>,
    pub bm25_score: Option<f32>,
    pub vector_score: Option<f32>,
}

/// Perform hybrid search: BM25 + vector, fused with ISR.
///
/// Stage 1 of the ranking pipeline:
/// - BM25 via tantivy → top K
/// - HNSW via embedder → top K
/// - Merge via ISR: score = Σ 1/rank²
pub async fn hybrid_search(
    text_index: &TextIndex,
    vector_index: &HnswIndex,
    embedder: &dyn Embedder,
    query: &str,
    top_k: usize,
) -> Result<Vec<HybridSearchResult>> {
    // BM25 search
    let bm25_results = text_index.search(query, top_k)?;

    // Vector search
    let query_vec = embedder.embed_one(query).await?;
    let vector_results = vector_index.search(&query_vec, top_k)?;

    // ISR fusion
    let fused = isr_fuse(&bm25_results, &vector_results);

    Ok(fused)
}

/// ISR (Inverse Square Rank) fusion of BM25 and vector results.
///
/// score = Σ 1/rank² for each ranking list the doc appears in.
/// Steeper decay than RRF — top results dominate, better P@10.
fn isr_fuse(
    bm25: &[TextSearchResult],
    vector: &[VectorSearchResult],
) -> Vec<HybridSearchResult> {
    use std::collections::HashMap;

    struct DocInfo {
        url: String,
        title: String,
        domain: String,
        isr_score: f32,
        bm25_rank: Option<usize>,
        vector_rank: Option<usize>,
        bm25_score: Option<f32>,
        vector_score: Option<f32>,
    }

    let mut docs: HashMap<String, DocInfo> = HashMap::new();

    // Add BM25 results
    for (rank, result) in bm25.iter().enumerate() {
        let rank_1 = (rank + 1) as f32;
        let isr = 1.0 / (rank_1 * rank_1);

        let entry = docs.entry(result.url.clone()).or_insert_with(|| DocInfo {
            url: result.url.clone(),
            title: result.title.clone(),
            domain: result.domain.clone(),
            isr_score: 0.0,
            bm25_rank: None,
            vector_rank: None,
            bm25_score: None,
            vector_score: None,
        });
        entry.isr_score += isr;
        entry.bm25_rank = Some(rank + 1);
        entry.bm25_score = Some(result.score);
    }

    // Add vector results
    for (rank, result) in vector.iter().enumerate() {
        let rank_1 = (rank + 1) as f32;
        let isr = 1.0 / (rank_1 * rank_1);

        let entry = docs.entry(result.doc_id.clone()).or_insert_with(|| DocInfo {
            url: result.doc_id.clone(),
            title: String::new(),
            domain: String::new(),
            isr_score: 0.0,
            bm25_rank: None,
            vector_rank: None,
            bm25_score: None,
            vector_score: None,
        });
        entry.isr_score += isr;
        entry.vector_rank = Some(rank + 1);
        entry.vector_score = Some(result.score);
    }

    // Sort by ISR score descending
    let mut results: Vec<HybridSearchResult> = docs
        .into_values()
        .map(|d| HybridSearchResult {
            doc_id: d.url.clone(),
            url: d.url,
            title: d.title,
            domain: d.domain,
            isr_score: d.isr_score,
            bm25_rank: d.bm25_rank,
            vector_rank: d.vector_rank,
            bm25_score: d.bm25_score,
            vector_score: d.vector_score,
        })
        .collect();

    results.sort_by(|a, b| {
        b.isr_score
            .partial_cmp(&a.isr_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isr_fuse_ranks_consensus_higher() {
        let bm25 = vec![
            TextSearchResult { doc_id: 0, url: "a.com".into(), title: "A".into(), domain: "a.com".into(), score: 5.0, source_tier: 4 },
            TextSearchResult { doc_id: 1, url: "b.com".into(), title: "B".into(), domain: "b.com".into(), score: 4.0, source_tier: 4 },
            TextSearchResult { doc_id: 2, url: "c.com".into(), title: "C".into(), domain: "c.com".into(), score: 3.0, source_tier: 4 },
        ];
        let vector = vec![
            VectorSearchResult { doc_id: "b.com".into(), score: 0.95 },
            VectorSearchResult { doc_id: "a.com".into(), score: 0.90 },
            VectorSearchResult { doc_id: "d.com".into(), score: 0.85 },
        ];

        let fused = isr_fuse(&bm25, &vector);

        // a.com: BM25 rank 1 (1/1=1.0) + vector rank 2 (1/4=0.25) = 1.25
        // b.com: BM25 rank 2 (1/4=0.25) + vector rank 1 (1/1=1.0) = 1.25
        // Both should be at top (tied score)
        let top_urls: Vec<&str> = fused.iter().take(2).map(|r| r.url.as_str()).collect();
        assert!(top_urls.contains(&"a.com"));
        assert!(top_urls.contains(&"b.com"));

        // d.com only in vector results, should be lower
        let d_pos = fused.iter().position(|r| r.url == "d.com").unwrap();
        assert!(d_pos >= 2);
    }

    #[test]
    fn isr_fuse_preserves_metadata() {
        let bm25 = vec![TextSearchResult {
            doc_id: 0,
            url: "test.com".into(),
            title: "Test Page".into(),
            domain: "test.com".into(),
            score: 10.0,
            source_tier: 2,
        }];
        let vector = vec![];

        let fused = isr_fuse(&bm25, &vector);
        assert_eq!(fused[0].title, "Test Page");
        assert_eq!(fused[0].bm25_rank, Some(1));
        assert!(fused[0].vector_rank.is_none());
    }
}
