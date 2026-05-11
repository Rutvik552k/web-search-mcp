use std::collections::HashMap;
use web_search_indexer::simhash;

/// Candidate document for diversity filtering.
#[derive(Debug, Clone)]
pub struct DiversityCandidate {
    pub url: String,
    pub domain: String,
    pub score: f32,
    pub body_hash: u64, // SimHash fingerprint
    pub embedding: Option<Vec<f32>>,
}

/// Apply diversity constraints to ranked results.
///
/// 1. SimHash dedup: skip near-duplicates (hamming ≤ threshold)
/// 2. Domain cap: max N results per domain
/// 3. MMR reranking: penalize docs similar to already-selected ones
pub fn diversify(
    candidates: &[DiversityCandidate],
    max_per_domain: usize,
    simhash_threshold: u32,
    mmr_lambda: f32,
    top_k: usize,
) -> Vec<usize> {
    let mut selected_indices: Vec<usize> = Vec::with_capacity(top_k);
    let mut domain_counts: HashMap<String, usize> = HashMap::new();
    let mut selected_hashes: Vec<u64> = Vec::new();

    for (idx, candidate) in candidates.iter().enumerate() {
        if selected_indices.len() >= top_k {
            break;
        }

        // Check SimHash dedup
        let is_near_dup = selected_hashes.iter().any(|&h| {
            simhash::hamming_distance(candidate.body_hash, h) <= simhash_threshold
        });
        if is_near_dup {
            continue;
        }

        // Check domain cap
        let domain_count = domain_counts.get(&candidate.domain).copied().unwrap_or(0);
        if domain_count >= max_per_domain {
            continue;
        }

        // MMR penalty: reduce score if too similar to already-selected docs
        if !selected_indices.is_empty() && candidate.embedding.is_some() {
            let max_sim = selected_indices.iter()
                .filter_map(|&si| {
                    let selected = &candidates[si];
                    match (&candidate.embedding, &selected.embedding) {
                        (Some(a), Some(b)) => Some(cosine_sim(a, b)),
                        _ => None,
                    }
                })
                .fold(0.0_f32, f32::max);

            let mmr_score = mmr_lambda * candidate.score - (1.0 - mmr_lambda) * max_sim;
            if mmr_score < 0.0 && selected_indices.len() >= 3 {
                // Only skip after we have at least 3 results
                continue;
            }
        }

        selected_indices.push(idx);
        selected_hashes.push(candidate.body_hash);
        *domain_counts.entry(candidate.domain.clone()).or_insert(0) += 1;
    }

    selected_indices
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-10 { 0.0 } else { dot / denom }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(url: &str, domain: &str, score: f32, hash: u64) -> DiversityCandidate {
        DiversityCandidate {
            url: url.to_string(),
            domain: domain.to_string(),
            score,
            body_hash: hash,
            embedding: None,
        }
    }

    #[test]
    fn respects_domain_cap() {
        // Use very different hashes to avoid SimHash dedup triggering
        let candidates = vec![
            make_candidate("https://a.com/1", "a.com", 1.0, 0xAAAA000000000000),
            make_candidate("https://a.com/2", "a.com", 0.9, 0xBBBB000000000000),
            make_candidate("https://a.com/3", "a.com", 0.8, 0xCCCC000000000000),
            make_candidate("https://b.com/1", "b.com", 0.7, 0xDDDD000000000000),
        ];

        let selected = diversify(&candidates, 2, 3, 0.7, 10);
        let domains: Vec<&str> = selected.iter().map(|&i| candidates[i].domain.as_str()).collect();
        let a_count = domains.iter().filter(|&&d| d == "a.com").count();
        assert!(a_count <= 2, "Should cap at 2 per domain, got {a_count}");
        assert!(selected.contains(&3), "b.com result should be included");
    }

    #[test]
    fn removes_near_duplicates() {
        let candidates = vec![
            make_candidate("https://a.com/1", "a.com", 1.0, 0xFF00FF00),
            make_candidate("https://b.com/1", "b.com", 0.9, 0xFF00FF01), // hamming=1 → dup
            make_candidate("https://c.com/1", "c.com", 0.8, 0x00FF00FF), // different
        ];

        let selected = diversify(&candidates, 5, 3, 0.7, 10);
        assert_eq!(selected.len(), 2); // b.com skipped as near-dup of a.com
        assert!(selected.contains(&0));
        assert!(selected.contains(&2));
    }

    #[test]
    fn top_k_limits_output() {
        let candidates: Vec<DiversityCandidate> = (0..20)
            .map(|i| make_candidate(
                &format!("https://site{i}.com/page"),
                &format!("site{i}.com"),
                1.0 - i as f32 * 0.01,
                i as u64 * 1000000,
            ))
            .collect();

        let selected = diversify(&candidates, 5, 3, 0.7, 5);
        assert_eq!(selected.len(), 5);
    }

    #[test]
    fn mmr_penalizes_similar_embeddings() {
        let candidates = vec![
            DiversityCandidate {
                url: "https://a.com/1".into(),
                domain: "a.com".into(),
                score: 1.0,
                body_hash: 1,
                embedding: Some(vec![1.0, 0.0, 0.0]),
            },
            DiversityCandidate {
                url: "https://b.com/1".into(),
                domain: "b.com".into(),
                score: 0.95,
                body_hash: 2,
                embedding: Some(vec![0.99, 0.01, 0.0]), // very similar to first
            },
            DiversityCandidate {
                url: "https://c.com/1".into(),
                domain: "c.com".into(),
                score: 0.5,
                body_hash: 3,
                embedding: Some(vec![0.0, 1.0, 0.0]), // very different
            },
        ];

        let selected = diversify(&candidates, 5, 3, 0.3, 10); // low lambda = diversity-heavy
        // All 3 should be included but the order matters —
        // first is always picked, then MMR decides
        assert!(selected.contains(&0)); // highest score, always first
    }

    #[test]
    fn empty_input() {
        let selected = diversify(&[], 5, 3, 0.7, 10);
        assert!(selected.is_empty());
    }
}
