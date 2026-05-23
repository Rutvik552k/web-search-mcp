use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use web_search_common::config::RankerConfig;
use web_search_common::models::*;
use web_search_embedder::cross_encoder::{CrossEncoderScore, NliLabel};
use web_search_embedder::CrossEncoder;
use web_search_indexer::simhash;

use crate::authority;
use crate::diversity::{self, DiversityCandidate};
use crate::entity_domain;
use crate::freshness;
use crate::hallucination::{self, DocClaims};
use crate::query_type;

/// Backend-agnostic cross-encoder wrapper.
/// Tries ONNX Runtime first (5-10x faster), falls back to Candle.
enum Reranker {
    #[cfg(feature = "onnx")]
    Onnx(web_search_embedder::onnx_cross_encoder::OnnxCrossEncoder),
    Candle(CrossEncoder),
}

impl Reranker {
    fn score_pairs(&self, pairs: &[(&str, &str)]) -> web_search_common::Result<Vec<CrossEncoderScore>> {
        match self {
            #[cfg(feature = "onnx")]
            Reranker::Onnx(m) => m.score_pairs(pairs),
            Reranker::Candle(m) => m.score_pairs(pairs),
        }
    }

    fn classify_from_logits(&self, logits: &[f32]) -> (NliLabel, f32) {
        match self {
            #[cfg(feature = "onnx")]
            Reranker::Onnx(m) => m.classify_from_logits(logits),
            Reranker::Candle(m) => m.classify_from_logits(logits),
        }
    }

    fn model_id(&self) -> &str {
        match self {
            #[cfg(feature = "onnx")]
            Reranker::Onnx(m) => m.model_id(),
            Reranker::Candle(m) => m.model_id(),
        }
    }
}

/// 5-stage anti-hallucination ranking pipeline.
///
/// Stage 1: Dual retrieval (BM25 + HNSW) → ISR fusion → top 300
/// Stage 2: Cross-encoder rerank → top 50
/// Stage 3: Authority + freshness boost → top 30
/// Stage 4: Anti-hallucination checks → enrich with confidence metadata
/// Stage 5: Diversity filter (MMR + domain cap + dedup) → final top K
pub struct RankingPipeline {
    config: RankerConfig,
    /// Cross-encoder for Stage 2 reranking (ONNX or Candle backend).
    /// Falls back to this when ColBERT is unavailable. Also used for Stage 4 NLI.
    reranker: Option<Arc<Reranker>>,
    /// ColBERT late-interaction reranker for Stage 2 (160-400x faster than CE).
    #[cfg(feature = "onnx")]
    colbert: Option<Arc<web_search_embedder::colbert::ColBertReranker>>,
    /// NLI model for Stage 4 contradiction detection
    nli_model: Option<Arc<Reranker>>,
    /// Cache: (query_hash, doc_content_hash) -> reranker score.
    score_cache: dashmap::DashMap<(u64, u64), f32>,
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
        // Load ColBERT for Stage 2 (160-400x faster than cross-encoder)
        #[cfg(feature = "onnx")]
        let colbert = {
            match web_search_embedder::colbert::ColBertReranker::new() {
                Ok(c) => {
                    tracing::info!("ColBERT reranker loaded — Stage 2 will use MaxSim");
                    Some(Arc::new(c))
                }
                Err(e) => {
                    tracing::info!("ColBERT unavailable, falling back to cross-encoder: {e}");
                    None
                }
            }
        };

        // Load cross-encoder: fallback for Stage 2 + used for Stage 4 NLI
        let reranker = Self::load_reranker("cross-encoder/ms-marco-MiniLM-L-6-v2");

        // Load NLI model: same priority — ONNX > Candle > None
        let nli_model = Self::load_reranker("cross-encoder/nli-MiniLM2-L6-H768");

        Self {
            config,
            reranker,
            #[cfg(feature = "onnx")]
            colbert,
            nli_model,
            score_cache: dashmap::DashMap::new(),
        }
    }

    /// Access the cross-encoder score cache for persistence (load/flush to disk).
    pub fn score_cache(&self) -> &dashmap::DashMap<(u64, u64), f32> {
        &self.score_cache
    }

    /// Try loading a cross-encoder model. ONNX first (fast), Candle fallback.
    fn load_reranker(model_id: &str) -> Option<Arc<Reranker>> {
        // Try ONNX Runtime first — graph optimization gives 5-10x speedup
        #[cfg(feature = "onnx")]
        {
            match web_search_embedder::onnx_cross_encoder::OnnxCrossEncoder::new(model_id) {
                Ok(m) => {
                    tracing::info!(model = m.model_id(), "ONNX reranker loaded");
                    return Some(Arc::new(Reranker::Onnx(m)));
                }
                Err(e) => {
                    tracing::info!("ONNX reranker unavailable for {model_id}, trying Candle: {e}");
                }
            }
        }

        // Candle fallback
        match CrossEncoder::new(model_id) {
            Ok(m) => {
                tracing::info!(model = m.model_id(), "Candle reranker loaded");
                Some(Arc::new(Reranker::Candle(m)))
            }
            Err(e) => {
                tracing::warn!("No reranker available for {model_id}: {e}");
                None
            }
        }
    }

    /// Stage 2 via ColBERT MaxSim. Returns true if ColBERT was used.
    #[allow(unused_variables)]
    fn stage2_colbert(
        &self,
        candidates: &[RankCandidate],
        query: &str,
        scored: &mut Vec<(usize, f32)>,
    ) -> bool {
        #[cfg(feature = "onnx")]
        {
            if let Some(ref colbert) = self.colbert {
                let start = std::time::Instant::now();
                let query_hash = hash_str(query);
                let mut cache_hits = 0_usize;

                // Check cache first, collect uncached
                let mut cached_results: Vec<(usize, f32)> = Vec::new();
                let mut need_score: Vec<(usize, usize, String)> = Vec::new(); // (pos_in_scored, candidate_idx, truncated)

                for (pos, &(idx, rrf_score)) in scored.iter().enumerate() {
                    let body = &candidates[idx].body_text;
                    let truncated: String = body.chars().take(512).collect();
                    let doc_hash = hash_str(&truncated);
                    let cache_key = (query_hash, doc_hash);

                    if let Some(cached) = self.score_cache.get(&cache_key) {
                        let blended = 0.3 * rrf_score + 0.7 * *cached;
                        cached_results.push((idx, blended));
                        cache_hits += 1;
                    } else {
                        need_score.push((pos, idx, truncated));
                    }
                }

                // Score uncached docs with ColBERT
                if !need_score.is_empty() {
                    let docs: Vec<&str> = need_score.iter().map(|(_, _, t)| t.as_str()).collect();
                    match colbert.score_documents(query, &docs) {
                        Ok(scores) => {
                            for (i, score) in scores.iter().enumerate() {
                                if i >= need_score.len() { break; }
                                let (pos, idx, ref truncated) = need_score[i];
                                let doc_hash = hash_str(truncated);
                                self.score_cache.insert((query_hash, doc_hash), *score);
                                let rrf_score = scored[pos].1;
                                let blended = 0.3 * rrf_score + 0.7 * score;
                                cached_results.push((idx, blended));
                            }
                        }
                        Err(e) => {
                            tracing::warn!("ColBERT scoring failed, falling back: {e}");
                            return false;
                        }
                    }
                }

                cached_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                tracing::debug!(
                    scored = cached_results.len(),
                    cache_hits,
                    elapsed_ms = start.elapsed().as_millis(),
                    "Stage 2 complete (ColBERT MaxSim)"
                );

                *scored = cached_results;
                return true;
            }
        }
        false
    }

    /// Stage 2 via cross-encoder (fallback when ColBERT unavailable).
    fn stage2_cross_encoder(
        &self,
        candidates: &[RankCandidate],
        query: &str,
        scored: &mut Vec<(usize, f32)>,
    ) {
        if let Some(reranker) = &self.reranker {
            let query_hash = hash_str(query);
            let early_term_min = 15.min(scored.len());
            let batch_size = 8;
            let mut ce_scores: Vec<(usize, f32)> = Vec::with_capacity(scored.len());
            let mut max_ce_score = 0.0_f32;
            let mut terminated_early = false;
            let mut cache_hits = 0_usize;

            for chunk_start in (0..scored.len()).step_by(batch_size) {
                let chunk_end = (chunk_start + batch_size).min(scored.len());
                let chunk = &scored[chunk_start..chunk_end];

                let mut need_inference: Vec<(usize, usize, String)> = Vec::new();
                for (ci, &(idx, rrf_score)) in chunk.iter().enumerate() {
                    let body = &candidates[idx].body_text;
                    let truncated: String = body.chars().take(256).collect();
                    let doc_hash = hash_str(&truncated);
                    let cache_key = (query_hash, doc_hash);

                    if let Some(cached) = self.score_cache.get(&cache_key) {
                        let blended = 0.3 * rrf_score + 0.7 * *cached;
                        ce_scores.push((idx, blended));
                        max_ce_score = max_ce_score.max(*cached);
                        cache_hits += 1;
                    } else {
                        need_inference.push((ci, idx, truncated));
                    }
                }

                if !need_inference.is_empty() {
                    let pairs: Vec<(&str, &str)> = need_inference.iter()
                        .map(|(_, _, doc)| (query, doc.as_str()))
                        .collect();

                    match reranker.score_pairs(&pairs) {
                        Ok(scores) => {
                            let mut batch_max = 0.0_f32;
                            for (i, ce_score) in scores.iter().enumerate() {
                                if i >= need_inference.len() { break; }
                                let (ci, idx, ref truncated) = need_inference[i];
                                let global_i = chunk_start + ci;
                                if global_i < scored.len() {
                                    let doc_hash = hash_str(truncated);
                                    self.score_cache.insert((query_hash, doc_hash), ce_score.score);
                                    let blended = 0.3 * scored[global_i].1 + 0.7 * ce_score.score;
                                    ce_scores.push((idx, blended));
                                    batch_max = batch_max.max(ce_score.score);
                                    max_ce_score = max_ce_score.max(ce_score.score);
                                }
                            }

                            if ce_scores.len() >= early_term_min && batch_max < max_ce_score * 0.6 {
                                tracing::debug!(
                                    scored = ce_scores.len(),
                                    total = scored.len(),
                                    "Stage 2: early termination (CE)"
                                );
                                terminated_early = true;
                                break;
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Cross-encoder batch failed: {e}");
                            for &(ci, idx, _) in &need_inference {
                                let global_i = chunk_start + ci;
                                if global_i < scored.len() {
                                    ce_scores.push((idx, scored[global_i].1));
                                }
                            }
                        }
                    }
                }
            }

            *scored = ce_scores;
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            tracing::debug!(
                scored = scored.len(),
                cache_hits,
                early_term = terminated_early,
                "Stage 2 complete (cross-encoder)"
            );
        }
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

        // Stage 0: Query-term relevance gate (free filter, ~0ms).
        // Reject candidates whose body text contains zero distinctive query terms.
        // Catches obviously irrelevant pages (Reddit homepage, Wikipedia login, etc.)
        // Filter out common English words that match too broadly.
        const STOP_WORDS: &[&str] = &[
            "the", "and", "for", "are", "what", "how", "why", "when", "where", "which",
            "with", "from", "that", "this", "have", "has", "had", "will", "would", "could",
            "should", "can", "not", "but", "its", "was", "were", "been", "being", "does",
            "between", "about", "into", "over", "after", "before", "during", "each",
            "differences", "difference", "compared", "comparison", "performance", "pricing",
            "best", "most", "more", "less", "than", "very", "also", "other",
            "workloads", "workload", "using", "used",
        ];
        let query_terms: Vec<String> = query
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .map(|w| w.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|w| !w.is_empty() && !STOP_WORDS.contains(&w.as_str()))
            .collect();

        let candidates: Vec<RankCandidate> = if query_terms.len() >= 2 {
            let before = candidates.len();
            // Require at least 30% of distinctive query terms (min 2) to appear in title+body
            let min_hits = (query_terms.len() as f32 * 0.3).ceil().max(2.0) as usize;
            let filtered: Vec<RankCandidate> = candidates
                .into_iter()
                .filter(|c| {
                    let body_lower = c.body_text.to_lowercase();
                    let title_lower = c.title.to_lowercase();
                    let combined = format!("{} {}", title_lower, body_lower);
                    let hits = query_terms.iter()
                        .filter(|qt| combined.contains(qt.as_str()))
                        .count();
                    hits >= min_hits
                })
                .collect();
            let removed = before - filtered.len();
            if removed > 0 {
                tracing::debug!(
                    removed, kept = filtered.len(), min_hits,
                    terms = ?query_terms,
                    "Stage 0: query-term relevance gate"
                );
            }
            filtered
        } else {
            candidates
        };

        let total_input = candidates.len();

        tracing::info!(
            candidates = total_input,
            query_type = ?query_type,
            "Starting 5-stage ranking"
        );

        // Stage 1: Reciprocal Rank Fusion (RRF)
        //
        // Standard RRF formula: score(d) = Σ 1/(k + rank_i(d))
        // where k=60 is the standard constant (Cormack et al., 2009).
        // Consistently 15-30% better than linear score combination.
        // Replaces previous ISR (inverse-square rank) fusion.
        const RRF_K: f32 = 60.0;
        let mut scored: Vec<(usize, f32)> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let mut rrf = 0.0_f32;
                if let Some(r) = c.bm25_rank {
                    rrf += 1.0 / (RRF_K + r as f32);
                }
                if let Some(r) = c.vector_rank {
                    rrf += 1.0 / (RRF_K + r as f32);
                }
                // Fallback: use raw scores if no ranks available
                if c.bm25_rank.is_none() && c.vector_rank.is_none() {
                    rrf = c.bm25_score.unwrap_or(0.0) * 0.5 + c.vector_score.unwrap_or(0.0) * 0.5;
                }
                (i, rrf)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Tighter funnel: top-50 for Stage 2 (was 300). Google does 1000→100→10.
        // At our scale (10K-100K docs), 50 candidates is plenty.
        let stage1_limit = 50.min(scored.len());
        scored.truncate(stage1_limit);

        tracing::debug!(stage1_results = scored.len(), "Stage 1 complete: RRF fusion");

        // Stage 2: Rerank with ColBERT MaxSim (fast) or cross-encoder (fallback)
        //
        // ColBERT: ~5-10ms for 50 docs (late interaction, pre-normalized token embeddings)
        // Cross-encoder: ~8s for 50 docs (full attention, sequential pairs)
        // Cache: (query_hash, doc_hash) → score. Skips inference on cache hit.
        let used_colbert = self.stage2_colbert(&candidates, query, &mut scored);
        if !used_colbert {
            self.stage2_cross_encoder(&candidates, query, &mut scored);
        }

        let stage2_limit = self.config.rerank_top_k.min(scored.len());
        scored.truncate(stage2_limit);

        // Filter out results below minimum relevance score,
        // but always keep at least top_k results as a safety floor.
        if self.config.min_relevance_score > 0.0 && scored.len() > top_k {
            let before = scored.len();
            let threshold = self.config.min_relevance_score;
            // Keep results above threshold OR the top_k best results (whichever yields more)
            let above_threshold: Vec<_> = scored.iter()
                .filter(|(_, score)| *score >= threshold)
                .cloned()
                .collect();
            if above_threshold.len() >= top_k {
                scored = above_threshold;
            } else {
                scored.truncate(top_k.max(above_threshold.len()));
            }
            let filtered = before - scored.len();
            if filtered > 0 {
                tracing::debug!(filtered, threshold, kept = scored.len(), "Low-relevance filter (safety floor active)");
            }
        }

        tracing::debug!(stage2_results = scored.len(), "Stage 2 complete");

        // Stage 3: Authority + freshness boost
        for (idx, score) in scored.iter_mut() {
            let c = &candidates[*idx];
            let auth_mult = authority::authority_boost(c.source_tier);
            let fresh_mult = freshness::freshness_decay(c.published_date, query_type);
            *score = freshness::authority_freshness_boost(*score, auth_mult, fresh_mult, query_type);
        }

        // Stage 3b: Primary source boost — promote results from canonical/official domains
        let official_domains = entity_domain::detect_official_domains(query);
        if !official_domains.is_empty() {
            let mut boosted_count = 0_usize;
            for (idx, score) in scored.iter_mut() {
                let c = &candidates[*idx];
                let ps_boost = entity_domain::primary_source_boost(query, &c.domain, &official_domains);
                if ps_boost > 1.0 {
                    *score *= ps_boost;
                    boosted_count += 1;
                }
            }
            if boosted_count > 0 {
                tracing::info!(
                    boosted_count,
                    official_domains = ?official_domains,
                    "Primary source boost applied"
                );
            }
        }

        // Re-sort after boost
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let stage3_limit = 30.min(scored.len());
        scored.truncate(stage3_limit);

        // Stage 3c: Query anchor penalty — demote results that completely miss original query terms.
        // This prevents query drift where reformulated queries (e.g. "B200") dilute results
        // for the original query (e.g. "NVIDIA H200 GPU specs").
        let query_words: Vec<String> = query
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .filter(|w| w.len() >= 4)
            .collect();

        if !query_words.is_empty() {
            let mut penalized_count = 0_usize;
            for (idx, score) in scored.iter_mut() {
                let c = &candidates[*idx];
                let title_lower = c.title.to_lowercase();
                let body_prefix: String = c.body_text.chars().take(500).collect::<String>().to_lowercase();
                let has_query_word = query_words.iter().any(|qw| {
                    title_lower.contains(qw.as_str()) || body_prefix.contains(qw.as_str())
                });
                if !has_query_word {
                    *score *= 0.5;
                    penalized_count += 1;
                }
            }
            if penalized_count > 0 {
                tracing::info!(penalized_count, "Query anchor penalty applied — demoted results missing all original query terms");
                // Re-sort after penalty
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
        }

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

        let mut hall_check = hallucination::check_hallucination(
            &doc_claims,
            self.config.min_unique_orgs,
        );

        // NLI-based contradiction detection (if model available)
        if let Some(nli) = &self.nli_model {
            let nli_contradictions = self.run_nli_checks(nli, &scored, &candidates);
            if !nli_contradictions.is_empty() {
                tracing::info!(count = nli_contradictions.len(), "NLI contradictions detected");
                hall_check.contradictions.extend(nli_contradictions);
                hall_check.warnings.push(format!(
                    "NLI model detected semantic contradictions across sources"
                ));
            }
        }

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

                // Confidence blends verification status with source tier
                let base_confidence: f32 = match verification {
                    VerificationStatus::Verified => 0.9,
                    VerificationStatus::Partial => 0.7,
                    VerificationStatus::Unverified => 0.5,
                    VerificationStatus::Contested => 0.3,
                };
                // Tier boost: Tier1 gets +0.1, Tier2 +0.05, Tier3 +0, Tier4 -0.05
                let tier_adj: f32 = match c.source_tier {
                    SourceTier::Tier1 => 0.1,
                    SourceTier::Tier2 => 0.05,
                    SourceTier::Tier3 => 0.0,
                    SourceTier::Tier4 => -0.05,
                };
                let confidence = (base_confidence + tier_adj).clamp(0.1, 0.99);

                RankedResult {
                    content: web_search_extractor::extract_snippet(&c.body_text, query, 1500),
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

        // xQuAD-style subtopic diversification:
        // Re-order results to maximize coverage of different subtopics.
        // Extract key bigrams from each result as "subtopics", then greedily
        // reorder so each next result covers the most uncovered subtopics.
        let results = xquad_diversify(results, query);

        let elapsed = start.elapsed().as_millis() as u64;

        tracing::info!(
            final_results = results.len(),
            elapsed_ms = elapsed,
            "Ranking pipeline complete"
        );

        SearchResponse {
            results,
            synthesis: vec![], // populated by orchestrator after ranking
            warnings: hall_check.warnings,
            coverage_score: hall_check.unique_orgs as f32 / self.config.min_unique_orgs as f32,
            total_pages_crawled: total_input,
            total_time_ms: elapsed,
            query: query.to_string(),
        }
    }
}

impl RankingPipeline {
    /// Run NLI-based contradiction detection across top documents.
    ///
    /// Cleans text artifacts (citations, brackets) before NLI.
    /// Reduced scope: top 5 docs → max 10 pairs (was top 10 → 45 pairs).
    /// Uses single batched `score_pairs` call instead of individual `classify_nli` calls.
    fn run_nli_checks(
        &self,
        nli: &Reranker,
        scored: &[(usize, f32)],
        candidates: &[RankCandidate],
    ) -> Vec<Contradiction> {
        let mut contradictions = Vec::new();

        // Extract first meaningful sentence from top 5 docs
        let doc_sentences: Vec<(String, String)> = scored.iter()
            .take(5)
            .filter_map(|(idx, _)| {
                let c = &candidates[*idx];
                let first_sentence = c.body_text
                    .split(|ch: char| ch == '.' || ch == '!' || ch == '?')
                    .find(|s| {
                        let cleaned = clean_text_for_nli(s);
                        cleaned.split_whitespace().count() >= 8
                    })
                    .map(|s| clean_text_for_nli(s))?;

                if first_sentence.split_whitespace().count() < 6 {
                    return None;
                }
                Some((first_sentence, c.url.clone()))
            })
            .collect();

        if doc_sentences.len() < 2 {
            return contradictions;
        }

        // Build all pairs
        let mut pairs: Vec<(&str, &str)> = Vec::new();
        let mut pair_meta: Vec<(usize, usize)> = Vec::new();

        for i in 0..doc_sentences.len() {
            for j in (i + 1)..doc_sentences.len() {
                pairs.push((&doc_sentences[i].0, &doc_sentences[j].0));
                pair_meta.push((i, j));
            }
        }

        // Single batched model call
        match nli.score_pairs(&pairs) {
            Ok(scores) => {
                for (k, score) in scores.iter().enumerate() {
                    if k >= pair_meta.len() { break; }
                    let logits = &score.logits;
                    if logits.len() < 3 { continue; }

                    // Use classify_nli on only those pairs that look potentially contradictory
                    // Quick pre-filter: if no logit is dominant, skip
                    let max_logit = logits.iter().take(3).cloned().fold(f32::NEG_INFINITY, f32::max);
                    let min_logit = logits.iter().take(3).cloned().fold(f32::INFINITY, f32::min);
                    if max_logit - min_logit < 0.5 { continue; } // all similar → likely neutral

                    let (i, j) = pair_meta[k];
                    // Use pre-computed logits — no second inference call
                    let (label, conf) = nli.classify_from_logits(logits);
                    if label == NliLabel::Contradiction && conf > 0.7 {
                        contradictions.push(Contradiction {
                            claim_a: doc_sentences[i].0.clone(),
                            source_a: doc_sentences[i].1.clone(),
                            claim_b: doc_sentences[j].0.clone(),
                            source_b: doc_sentences[j].1.clone(),
                            severity: if conf > 0.9 {
                                ContradictionSeverity::Hard
                            } else {
                                ContradictionSeverity::Soft
                            },
                        });
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Batched NLI scoring failed: {e}");
            }
        }

        contradictions
    }
}

/// Clean text for NLI: strip citation brackets, HTML artifacts, excess whitespace.
fn clean_text_for_nli(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut in_bracket = false;

    for ch in text.chars() {
        match ch {
            '[' => in_bracket = true,
            ']' => { in_bracket = false; continue; }
            _ if in_bracket => continue,
            _ => result.push(ch),
        }
    }

    // Remove leftover artifacts
    result = result
        .replace("  ", " ")
        .replace(" ,", ",")
        .replace(" .", ".");

    result.trim().to_string()
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

/// xQuAD-style result diversification.
///
/// Algorithm (from Santos et al., "Explicit Search Result Diversification"):
/// 1. Extract "subtopics" from each result as key bigrams not in the query
/// 2. Greedily reorder: each step picks the result that covers the most
///    uncovered subtopics, weighted by relevance score
/// 3. This ensures the final result set covers multiple aspects of the query
///
/// Complexity: O(k * n * t) where k=results, n=subtopics, t=terms per result
fn xquad_diversify(mut results: Vec<RankedResult>, query: &str) -> Vec<RankedResult> {
    if results.len() <= 2 {
        return results;
    }

    let query_words: std::collections::HashSet<String> = query
        .to_lowercase()
        .split_whitespace()
        .map(|w| w.to_string())
        .collect();

    // Extract subtopic bigrams from each result
    let subtopics: Vec<std::collections::HashSet<String>> = results.iter()
        .map(|r| {
            let words: Vec<String> = r.content
                .to_lowercase()
                .split(|c: char| !c.is_alphanumeric())
                .filter(|w| w.len() >= 3 && !query_words.contains(*w))
                .map(|w| w.to_string())
                .collect();
            // Use bigrams as subtopics
            words.windows(2)
                .map(|pair| format!("{}_{}", pair[0], pair[1]))
                .collect()
        })
        .collect();

    // Greedy diversification
    let mut selected: Vec<usize> = Vec::with_capacity(results.len());
    let mut covered: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut remaining: Vec<usize> = (0..results.len()).collect();

    while !remaining.is_empty() {
        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for (pos, &ri) in remaining.iter().enumerate() {
            let new_coverage = subtopics[ri].iter()
                .filter(|t| !covered.contains(*t))
                .count() as f32;
            // Score = relevance * (1 + new_subtopic_coverage * 0.1)
            let score = results[ri].relevance_score * (1.0 + new_coverage * 0.1);
            if score > best_score {
                best_score = score;
                best_idx = pos;
            }
        }

        let chosen = remaining.remove(best_idx);
        covered.extend(subtopics[chosen].iter().cloned());
        selected.push(chosen);
    }

    // Reorder results by selected order
    let mut diversified = Vec::with_capacity(results.len());
    // We need to take ownership — use indices to extract from original
    let mut slots: Vec<Option<RankedResult>> = results.drain(..).map(Some).collect();
    for i in selected {
        if let Some(r) = slots[i].take() {
            diversified.push(r);
        }
    }
    diversified
}

/// Fast string hash for cache keys. Not cryptographic — just for deduplication.
fn hash_str(s: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
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
            min_relevance_score: 0.0,
        };

        let pipeline = RankingPipeline::new(config);
        let candidates = vec![
            make_candidate("https://a.com/1", "a.com", "Page A", "This is page A with substantial content about climate change and global warming effects on the environment.", 1),
            make_candidate("https://b.com/1", "b.com", "Page B", "This is page B discussing climate policy and international agreements on carbon emissions reduction.", 2),
            make_candidate("https://c.com/1", "c.com", "Page C", "This is page C about renewable energy solutions and their impact on reducing greenhouse gas emissions.", 3),
        ];

        let response = pipeline.rank(candidates, "climate change", 10);
        assert!(!response.results.is_empty());
        // First run includes model download (~15s), subsequent runs are <1s
        assert!(response.total_time_ms < 120_000);
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
            min_relevance_score: 0.0,
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
            min_relevance_score: 0.0,
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
            min_relevance_score: 0.0,
        };

        let pipeline = RankingPipeline::new(config);
        let response = pipeline.rank(vec![], "test", 10);
        assert!(response.results.is_empty());
    }
}
