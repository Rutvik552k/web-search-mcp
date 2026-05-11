/// Inverse Square Rank fusion: score = Σ 1/rank²
/// Steeper decay than RRF, better P@10 for LLM grounding.
pub fn isr_fuse(ranked_lists: &[Vec<(String, f32)>]) -> Vec<(String, f32)> {
    use std::collections::HashMap;

    let mut scores: HashMap<String, f32> = HashMap::new();

    for list in ranked_lists {
        for (rank, (doc_id, _score)) in list.iter().enumerate() {
            let rank_1based = (rank + 1) as f32;
            let isr_score = 1.0 / (rank_1based * rank_1based);
            *scores.entry(doc_id.clone()).or_insert(0.0) += isr_score;
        }
    }

    let mut results: Vec<(String, f32)> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Maximal Marginal Relevance for diversity.
/// MMR(d) = λ * rel(d, query) - (1-λ) * max_sim(d, selected)
pub fn mmr_select(
    candidates: &[(String, f32)],       // (doc_id, relevance_score)
    similarity_fn: &dyn Fn(&str, &str) -> f32, // pairwise similarity
    lambda: f32,
    top_k: usize,
) -> Vec<(String, f32)> {
    let mut selected: Vec<(String, f32)> = Vec::with_capacity(top_k);
    let mut remaining: Vec<(String, f32)> = candidates.to_vec();

    while selected.len() < top_k && !remaining.is_empty() {
        let mut best_idx = 0;
        let mut best_mmr = f32::NEG_INFINITY;

        for (i, (doc_id, relevance)) in remaining.iter().enumerate() {
            let max_sim_to_selected = selected
                .iter()
                .map(|(sel_id, _)| similarity_fn(doc_id, sel_id))
                .fold(0.0_f32, f32::max);

            let mmr = lambda * relevance - (1.0 - lambda) * max_sim_to_selected;

            if mmr > best_mmr {
                best_mmr = mmr;
                best_idx = i;
            }
        }

        selected.push(remaining.remove(best_idx));
    }

    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isr_ranks_consensus_higher() {
        let list_a = vec![
            ("doc1".to_string(), 0.9),
            ("doc2".to_string(), 0.8),
            ("doc3".to_string(), 0.7),
        ];
        let list_b = vec![
            ("doc2".to_string(), 0.95),
            ("doc1".to_string(), 0.85),
            ("doc4".to_string(), 0.6),
        ];

        let fused = isr_fuse(&[list_a, list_b]);

        // doc1 and doc2 appear in both lists, should be top
        assert!(fused[0].0 == "doc1" || fused[0].0 == "doc2");
        assert!(fused[1].0 == "doc1" || fused[1].0 == "doc2");
    }

    #[test]
    fn mmr_promotes_diversity() {
        let candidates = vec![
            ("a".to_string(), 1.0),
            ("b".to_string(), 0.6),
            ("c".to_string(), 0.5),
        ];

        // a and b are very similar, c is different
        let sim = |x: &str, y: &str| -> f32 {
            if (x == "a" && y == "b") || (x == "b" && y == "a") {
                0.95 // very similar
            } else {
                0.1 // dissimilar
            }
        };

        // λ=0.5: balance relevance and diversity equally
        let selected = mmr_select(&candidates, &sim, 0.5, 2);
        // First pick: "a" (highest relevance)
        assert_eq!(selected[0].0, "a");
        // Second pick: "c" beats "b" because b's high similarity to a gets penalized
        // b MMR: 0.5*0.6 - 0.5*0.95 = 0.3 - 0.475 = -0.175
        // c MMR: 0.5*0.5 - 0.5*0.1  = 0.25 - 0.05 = 0.2
        assert_eq!(selected[1].0, "c");
    }
}
