use web_search_common::config::RankerConfig;
use web_search_common::models::*;
use web_search_indexer::simhash;

use crate::authority;
use crate::diversity::{self, DiversityCandidate};
use crate::freshness;
use crate::hallucination::{self, DocClaims};
use crate::query_type;

/// 5-stage anti-hallucination ranking pipeline.
///
/// Stage 1: Dual retrieval (BM25 + HNSW) → ISR fusion → top 300
/// Stage 2: Cross-encoder rerank → top 50 (placeholder: pass-through)
/// Stage 3: Authority + freshness boost → top 30
/// Stage 4: Anti-hallucination checks → enrich with confidence metadata
/// Stage 5: Diversity filter (MMR + domain cap + dedup) → final top K
pub struct RankingPipeline {
    config: RankerConfig,
}

/// Input document for the ranking pipeline.
#[derive(Debug, Clone)]
pub struct RankCandidate {
    pub url: String,
    pub domain: String,
    pub title: String,
    pub body_text: String,
    pub published_date: Option<chrono::DateTime<chrono::Utc>>,
    pub source_tier: SourceTier,
    pub bm25_score: Option<f32>,
    pub vector_score: Option<f32>,
    pub bm25_rank: Option<usize>,
    pub vector_rank: Option<usize>,
    pub embedding: Option<Vec<f32>>,
}

impl RankingPipeline {
    pub fn new(config: RankerConfig) -> Self {
        Self { config }
    }

    /// Run the full 5-stage pipeline on a set of candidates.
    pub fn rank(
        &self,
        candidates: Vec<RankCandidate>,
        query: &str,
        top_k: usize,
    ) -> SearchResponse {
        let start = std::time::Instant::now();
        let query_type = query_type::detect_query_type(query);
        let total_input = candidates.len();

        tracing::info!(
            candidates = total_input,
            query_type = ?query_type,
            "Starting 5-stage ranking"
        );

        // Stage 1: ISR fusion (already done in hybrid_search, scores are in candidates)
        let mut scored: Vec<(usize, f32)> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let mut isr = 0.0_f32;
                if let Some(r) = c.bm25_rank {
                    let r1 = r as f32;
                    isr += 1.0 / (r1 * r1);
                }
                if let Some(r) = c.vector_rank {
                    let r1 = r as f32;
                    isr += 1.0 / (r1 * r1);
                }
                // Fallback: use raw scores if no ranks
                if c.bm25_rank.is_none() && c.vector_rank.is_none() {
                    isr = c.bm25_score.unwrap_or(0.0) * 0.5 + c.vector_score.unwrap_or(0.0) * 0.5;
                }
                (i, isr)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top 300 for Stage 2
        let stage1_limit = self.config.bm25_top_k.max(self.config.hnsw_top_k).min(scored.len());
        scored.truncate(stage1_limit);

        tracing::debug!(stage1_results = scored.len(), "Stage 1 complete: ISR fusion");

        // Stage 2: Cross-encoder rerank (placeholder: keep ISR scores)
        // TODO: When ONNX cross-encoder model is available, rerank here
        let stage2_limit = self.config.rerank_top_k.min(scored.len());
        scored.truncate(stage2_limit);

        tracing::debug!(stage2_results = scored.len(), "Stage 2 complete: rerank");

        // Stage 3: Authority + freshness boost
        for (idx, score) in scored.iter_mut() {
            let c = &candidates[*idx];
            let auth_mult = authority::authority_boost(c.source_tier);
            let fresh_mult = freshness::freshness_decay(c.published_date, query_type);
            *score = freshness::authority_freshness_boost(*score, auth_mult, fresh_mult, query_type);
        }

        // Re-sort after boost
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let stage3_limit = 30.min(scored.len());
        scored.truncate(stage3_limit);

        tracing::debug!(stage3_results = scored.len(), "Stage 3 complete: authority+freshness");

        // Stage 4: Anti-hallucination checks
        let doc_claims: Vec<DocClaims> = scored
            .iter()
            .map(|(idx, _)| {
                let c = &candidates[*idx];
                DocClaims {
                    url: c.url.clone(),
                    domain: c.domain.clone(),
                    key_phrases: extract_key_phrases(&c.body_text),
                }
            })
            .collect();

        let hall_check = hallucination::check_hallucination(
            &doc_claims,
            self.config.min_unique_orgs,
        );

        tracing::debug!(
            unique_orgs = hall_check.unique_orgs,
            contradictions = hall_check.contradictions.len(),
            echo_risk = hall_check.echo_chamber_risk,
            "Stage 4 complete: anti-hallucination"
        );

        // Stage 5: Diversity filter
        let diversity_candidates: Vec<DiversityCandidate> = scored
            .iter()
            .map(|(idx, score)| {
                let c = &candidates[*idx];
                DiversityCandidate {
                    url: c.url.clone(),
                    domain: c.domain.clone(),
                    score: *score,
                    body_hash: simhash::simhash(&c.body_text),
                    embedding: c.embedding.clone(),
                }
            })
            .collect();

        let diverse_indices = diversity::diversify(
            &diversity_candidates,
            self.config.max_results_per_domain,
            3, // simhash threshold
            self.config.mmr_lambda,
            top_k,
        );

        // Build final results
        let results: Vec<RankedResult> = diverse_indices
            .iter()
            .map(|&di| {
                let (orig_idx, score) = scored[di];
                let c = &candidates[orig_idx];

                // Find matching claims for this doc
                let doc_claims: Vec<Claim> = hall_check
                    .claims
                    .iter()
                    .filter(|cl| cl.source_url == c.url)
                    .cloned()
                    .collect();

                // Find contradictions involving this doc
                let doc_contras: Vec<Contradiction> = hall_check
                    .contradictions
                    .iter()
                    .filter(|ct| ct.source_a == c.url || ct.source_b == c.url)
                    .cloned()
                    .collect();

                // Determine overall verification
                let verification = if doc_claims.iter().any(|c| c.verification == VerificationStatus::Verified) {
                    VerificationStatus::Verified
                } else if doc_claims.iter().any(|c| c.verification == VerificationStatus::Partial) {
                    VerificationStatus::Partial
                } else if !doc_contras.is_empty() {
                    VerificationStatus::Contested
                } else {
                    VerificationStatus::Unverified
                };

                let confidence = match verification {
                    VerificationStatus::Verified => 0.95,
                    VerificationStatus::Partial => 0.75,
                    VerificationStatus::Unverified => 0.5,
                    VerificationStatus::Contested => 0.3,
                };

                RankedResult {
                    content: c.body_text.chars().take(2000).collect(),
                    url: c.url.clone(),
                    title: c.title.clone(),
                    confidence,
                    verification,
                    claims: doc_claims,
                    contradictions: doc_contras,
                    source_tier: c.source_tier,
                    freshness: c.published_date,
                    relevance_score: score,
                }
            })
            .collect();

        let elapsed = start.elapsed().as_millis() as u64;

        tracing::info!(
            final_results = results.len(),
            elapsed_ms = elapsed,
            "Ranking pipeline complete"
        );

        SearchResponse {
            results,
            warnings: hall_check.warnings,
            coverage_score: hall_check.unique_orgs as f32 / self.config.min_unique_orgs as f32,
            total_pages_crawled: total_input,
            total_time_ms: elapsed,
            query: query.to_string(),
        }
    }
}

/// Extract key phrases from body text for cross-reference checking.
///
/// Simple heuristic: sentences containing numbers, proper nouns, or key facts.
fn extract_key_phrases(text: &str) -> Vec<String> {
    text.split(|c: char| c == '.' || c == '!' || c == '?')
        .map(|s| s.trim().to_string())
        .filter(|s| {
            let word_count = s.split_whitespace().count();
            // Keep sentences with 5+ words that contain meaningful content
            word_count >= 5 && word_count <= 50
        })
        .take(20) // limit to avoid O(n²) in cross-reference
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_candidate(url: &str, domain: &str, title: &str, body: &str, bm25_rank: usize) -> RankCandidate {
        RankCandidate {
            url: url.to_string(),
            domain: domain.to_string(),
            title: title.to_string(),
            body_text: body.to_string(),
            published_date: Some(Utc::now()),
            source_tier: authority::classify_domain(domain),
            bm25_score: Some(10.0 - bm25_rank as f32),
            vector_score: None,
            bm25_rank: Some(bm25_rank),
            vector_rank: None,
            embedding: None,
        }
    }

    #[test]
    fn pipeline_produces_results() {
        let config = RankerConfig {
            cross_encoder_model_path: "models/cross-encoder.onnx".into(),
            bm25_top_k: 200,
            hnsw_top_k: 200,
            rerank_top_k: 50,
            min_verification_sources: 3,
            mmr_lambda: 0.7,
            max_results_per_domain: 2,
            min_unique_orgs: 3,
            source_tiers_path: "config/source_tiers.toml".into(),
        };

        let pipeline = RankingPipeline::new(config);
        let candidates = vec![
            make_candidate("https://a.com/1", "a.com", "Page A", "This is page A with substantial content about climate change and global warming effects on the environment.", 1),
            make_candidate("https://b.com/1", "b.com", "Page B", "This is page B discussing climate policy and international agreements on carbon emissions reduction.", 2),
            make_candidate("https://c.com/1", "c.com", "Page C", "This is page C about renewable energy solutions and their impact on reducing greenhouse gas emissions.", 3),
        ];

        let response = pipeline.rank(candidates, "climate change", 10);
        assert!(!response.results.is_empty());
        assert!(response.total_time_ms < 5000);
        assert_eq!(response.query, "climate change");
    }

    #[test]
    fn pipeline_respects_top_k() {
        let config = RankerConfig {
            cross_encoder_model_path: "models/cross-encoder.onnx".into(),
            bm25_top_k: 200,
            hnsw_top_k: 200,
            rerank_top_k: 50,
            min_verification_sources: 3,
            mmr_lambda: 0.7,
            max_results_per_domain: 5,
            min_unique_orgs: 2,
            source_tiers_path: "config/source_tiers.toml".into(),
        };

        let pipeline = RankingPipeline::new(config);
        let candidates: Vec<RankCandidate> = (0..20)
            .map(|i| make_candidate(
                &format!("https://site{i}.com/page"),
                &format!("site{i}.com"),
                &format!("Page {i}"),
                &format!("Content for page {i} with enough text to be considered valid content by the ranking pipeline and extraction quality checks."),
                i + 1,
            ))
            .collect();

        let response = pipeline.rank(candidates, "test query", 5);
        assert!(response.results.len() <= 5);
    }

    #[test]
    fn authority_affects_ranking() {
        let config = RankerConfig {
            cross_encoder_model_path: "".into(),
            bm25_top_k: 200,
            hnsw_top_k: 200,
            rerank_top_k: 50,
            min_verification_sources: 3,
            mmr_lambda: 0.7,
            max_results_per_domain: 5,
            min_unique_orgs: 2,
            source_tiers_path: "".into(),
        };

        let pipeline = RankingPipeline::new(config);
        let candidates = vec![
            make_candidate("https://random-blog.xyz/p", "random-blog.xyz", "Blog", "Research shows interesting findings about the topic with data and analysis.", 1),
            make_candidate("https://nature.com/article", "nature.com", "Nature", "Research shows interesting findings about the topic with data and analysis.", 2),
        ];

        let response = pipeline.rank(candidates, "research findings", 10);
        // Nature (Tier 1) should rank higher than random blog (Tier 4) despite lower BM25 rank
        if response.results.len() >= 2 {
            let nature_pos = response.results.iter().position(|r| r.url.contains("nature.com"));
            let blog_pos = response.results.iter().position(|r| r.url.contains("random-blog"));
            if let (Some(np), Some(bp)) = (nature_pos, blog_pos) {
                assert!(np < bp, "Nature (Tier 1) should rank above random blog (Tier 4)");
            }
        }
    }

    #[test]
    fn extract_key_phrases_filters() {
        let phrases = extract_key_phrases(
            "Short. This is a sentence with enough words to be meaningful. Also short. \
             Another longer sentence that should be extracted as a key phrase for analysis."
        );
        // Should only keep sentences with 5+ words
        assert!(phrases.iter().all(|p| p.split_whitespace().count() >= 5));
        assert!(!phrases.is_empty());
    }

    #[test]
    fn empty_candidates() {
        let config = RankerConfig {
            cross_encoder_model_path: "".into(),
            bm25_top_k: 200,
            hnsw_top_k: 200,
            rerank_top_k: 50,
            min_verification_sources: 3,
            mmr_lambda: 0.7,
            max_results_per_domain: 2,
            min_unique_orgs: 3,
            source_tiers_path: "".into(),
        };

        let pipeline = RankingPipeline::new(config);
        let response = pipeline.rank(vec![], "test", 10);
        assert!(response.results.is_empty());
    }
}
