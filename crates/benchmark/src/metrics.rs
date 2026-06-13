//! Pure scoring functions for the coverage + accuracy benchmark.
//!
//! Everything here is deterministic and side-effect free so it can be covered
//! by fast unit tests (the base of the test pyramid). The harness in `main.rs`
//! does the I/O; this module does the math.

/// Minimum number of (non-boilerplate) characters the extractor must return for
/// a page to count as "clean main content" per GOAL.md G1.
pub const MIN_CLEAN_CHARS: usize = 200;

/// Normalize a URL for label matching in G3.
///
/// We compare ranked-result URLs against the operator's `relevant_urls` labels.
/// Trivial differences (scheme, leading `www.`, a trailing slash, a `#fragment`)
/// must not cause a false miss. We deliberately keep the query string, since two
/// pages can legitimately differ only by query.
pub fn normalize_url(raw: &str) -> String {
    let mut s = raw.trim().to_lowercase();

    // Drop fragment.
    if let Some(idx) = s.find('#') {
        s.truncate(idx);
    }
    // Strip scheme.
    for scheme in ["https://", "http://"] {
        if let Some(rest) = s.strip_prefix(scheme) {
            s = rest.to_string();
            break;
        }
    }
    // Strip leading www.
    if let Some(rest) = s.strip_prefix("www.") {
        s = rest.to_string();
    }
    // Strip a single trailing slash (but keep "/" if that's all that remains).
    if s.len() > 1 {
        s = s.trim_end_matches('/').to_string();
    }
    s
}

/// Decide whether an extraction counts as "clean main content" (G1).
///
/// `body_text` is the extractor's main-content output (already boilerplate-
/// stripped by the consensus extractor). It is clean when it has at least
/// [`MIN_CLEAN_CHARS`] characters AND, if the label provides `expect_contains`,
/// the body actually contains that marker string (case-insensitive).
pub fn is_clean(body_text: &str, expect_contains: Option<&str>) -> bool {
    let long_enough = body_text.chars().count() >= MIN_CLEAN_CHARS;
    if !long_enough {
        return false;
    }
    match expect_contains {
        Some(marker) if !marker.is_empty() => body_text
            .to_lowercase()
            .contains(&marker.to_lowercase()),
        _ => true,
    }
}

/// Discounted Cumulative Gain over the first `k` items of a binary relevance
/// vector given in rank order: `DCG@k = sum_i rel_i / log2(i + 2)`.
pub fn dcg_at_k(rels_in_rank_order: &[u8], k: usize) -> f64 {
    rels_in_rank_order
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, &rel)| rel as f64 / ((i + 2) as f64).log2())
        .sum()
}

/// nDCG@k for binary relevance. The ideal ranking places every relevant item
/// first. Returns 0.0 when there are no relevant items (IDCG == 0).
pub fn ndcg_at_k(rels_in_rank_order: &[u8], k: usize) -> f64 {
    let dcg = dcg_at_k(rels_in_rank_order, k);

    let total_relevant = rels_in_rank_order.iter().filter(|&&r| r > 0).count();
    let ideal: Vec<u8> = std::iter::repeat(1u8).take(total_relevant).collect();
    let idcg = dcg_at_k(&ideal, k);

    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

/// Precision@k for binary relevance: relevant-items-in-top-k / k.
///
/// Denominator is always `k` (the conventional definition), so a query whose
/// label set has fewer than `k` relevant URLs cannot reach 1.0. This is
/// intentional and documented in benchmark/README.md.
pub fn precision_at_k(rels_in_rank_order: &[u8], k: usize) -> f64 {
    if k == 0 {
        return 0.0;
    }
    let hits: u32 = rels_in_rank_order
        .iter()
        .take(k)
        .map(|&r| r as u32)
        .sum();
    hits as f64 / k as f64
}

/// Convert an ordered list of result URLs into a binary relevance vector by
/// matching (normalized) against a set of relevant URLs.
pub fn relevance_vector(ranked_urls: &[String], relevant_urls: &[String]) -> Vec<u8> {
    let relevant: std::collections::HashSet<String> =
        relevant_urls.iter().map(|u| normalize_url(u)).collect();
    ranked_urls
        .iter()
        .map(|u| if relevant.contains(&normalize_url(u)) { 1 } else { 0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── normalize_url ────────────────────────────────────────────────
    #[test]
    fn normalize_strips_scheme_www_and_trailing_slash() {
        assert_eq!(normalize_url("https://www.Tokio.rs/"), "tokio.rs");
        assert_eq!(normalize_url("http://tokio.rs"), "tokio.rs");
    }

    #[test]
    fn normalize_drops_fragment_but_keeps_query() {
        assert_eq!(normalize_url("https://a.com/p?x=1#frag"), "a.com/p?x=1");
    }

    #[test]
    fn normalize_root_slash_is_preserved_not_emptied() {
        // A lone "/" must not be stripped to "".
        assert_eq!(normalize_url("https://a.com/"), "a.com");
        assert_eq!(normalize_url("/"), "/");
    }

    #[test]
    fn normalize_matches_scheme_and_www_variants() {
        assert_eq!(
            normalize_url("https://www.djangoproject.com/"),
            normalize_url("http://djangoproject.com")
        );
    }

    // ── is_clean ─────────────────────────────────────────────────────
    #[test]
    fn clean_requires_min_chars() {
        let short = "x".repeat(MIN_CLEAN_CHARS - 1);
        let exact = "x".repeat(MIN_CLEAN_CHARS);
        assert!(!is_clean(&short, None));
        assert!(is_clean(&exact, None));
    }

    #[test]
    fn clean_counts_chars_not_bytes() {
        // 200 multi-byte chars => >200 bytes but exactly 200 chars => clean.
        let multibyte = "é".repeat(MIN_CLEAN_CHARS);
        assert_eq!(multibyte.chars().count(), MIN_CLEAN_CHARS);
        assert!(is_clean(&multibyte, None));
    }

    #[test]
    fn clean_honors_expect_contains_case_insensitive() {
        let body = format!("{} Example Domain reference text", "y".repeat(MIN_CLEAN_CHARS));
        assert!(is_clean(&body, Some("example domain")));
        assert!(!is_clean(&body, Some("not present here")));
    }

    #[test]
    fn clean_fails_when_long_but_marker_missing() {
        let body = "z".repeat(500);
        assert!(!is_clean(&body, Some("premium insight")));
    }

    #[test]
    fn empty_expect_contains_is_ignored() {
        let body = "w".repeat(300);
        assert!(is_clean(&body, Some("")));
    }

    // ── dcg / ndcg ───────────────────────────────────────────────────
    #[test]
    fn dcg_known_value() {
        // rel = [1,0,1] => 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5 = 1.5
        let dcg = dcg_at_k(&[1, 0, 1], 10);
        assert!((dcg - 1.5).abs() < 1e-9, "dcg was {dcg}");
    }

    #[test]
    fn ndcg_perfect_ranking_is_one() {
        // All relevant items already at the top.
        assert!((ndcg_at_k(&[1, 1, 0, 0], 10) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ndcg_no_relevant_is_zero() {
        assert_eq!(ndcg_at_k(&[0, 0, 0], 10), 0.0);
    }

    #[test]
    fn ndcg_known_value_for_rank1_and_rank3() {
        // ranked rel = [1,0,1,0,0]; 2 relevant total.
        // DCG  = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
        // IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309 = 1.6309
        // nDCG = 1.5 / 1.6309 = 0.9197
        let n = ndcg_at_k(&[1, 0, 1, 0, 0], 10);
        assert!((n - 0.9197).abs() < 1e-3, "ndcg was {n}");
    }

    #[test]
    fn ndcg_at_k_respects_cutoff() {
        // Relevant item sits at rank 11 — outside @10, so DCG@10 = 0.
        let mut rels = vec![0u8; 10];
        rels.push(1);
        assert_eq!(ndcg_at_k(&rels, 10), 0.0);
    }

    // ── precision ────────────────────────────────────────────────────
    #[test]
    fn precision_at_5_basic() {
        // 2 hits in top 5 => 0.4
        assert!((precision_at_k(&[1, 0, 1, 0, 0, 1], 5) - 0.4).abs() < 1e-9);
    }

    #[test]
    fn precision_denominator_is_k_even_with_few_results() {
        // Only 2 results returned, both relevant => 2/5 = 0.4, not 1.0.
        assert!((precision_at_k(&[1, 1], 5) - 0.4).abs() < 1e-9);
    }

    #[test]
    fn precision_k_zero_is_zero() {
        assert_eq!(precision_at_k(&[1, 1], 0), 0.0);
    }

    // ── relevance_vector ─────────────────────────────────────────────
    #[test]
    fn relevance_vector_matches_normalized_urls() {
        let ranked = vec![
            "https://tokio.rs/".to_string(),
            "https://some-blog.example/post".to_string(),
            "http://www.docs.rs/tokio".to_string(),
        ];
        let relevant = vec![
            "https://tokio.rs".to_string(),
            "https://docs.rs/tokio".to_string(),
        ];
        assert_eq!(relevance_vector(&ranked, &relevant), vec![1, 0, 1]);
    }

    #[test]
    fn relevance_vector_empty_labels_all_zero() {
        let ranked = vec!["https://a.com".to_string()];
        assert_eq!(relevance_vector(&ranked, &[]), vec![0]);
    }
}
