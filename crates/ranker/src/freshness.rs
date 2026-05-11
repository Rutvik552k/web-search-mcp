use chrono::{DateTime, Utc};
use web_search_common::models::QueryType;

/// Compute freshness decay multiplier for a document.
///
/// Uses exponential decay: `decay = e^(-λ * days_old)`
/// λ adapts based on query type:
/// - News: λ=0.1 (half-life ~7 days, aggressive decay)
/// - Research: λ=0.005 (half-life ~139 days, slow decay)
/// - Technical: λ=0.01 (half-life ~70 days)
/// - Factual/General: λ=0.01 (half-life ~70 days)
///
/// Returns multiplier in [0.0, 1.0].
pub fn freshness_decay(
    published_date: Option<DateTime<Utc>>,
    query_type: QueryType,
) -> f32 {
    let date = match published_date {
        Some(d) => d,
        None => return 0.8, // unknown date: mild penalty
    };

    let days_old = (Utc::now() - date).num_days().max(0) as f64;

    let lambda = match query_type {
        QueryType::News => 0.1,       // half-life ~7 days
        QueryType::Research => 0.005, // half-life ~139 days
        QueryType::Technical => 0.01, // half-life ~70 days
        QueryType::Factual => 0.01,   // half-life ~70 days
        QueryType::General => 0.01,   // half-life ~70 days
    };

    let decay = (-lambda * days_old).exp() as f32;
    decay.clamp(0.01, 1.0) // never fully zero
}

/// Compute the combined Stage 3 boost (authority + freshness).
///
/// Used after cross-encoder rerank to adjust scores based on
/// source quality and temporal relevance.
pub fn authority_freshness_boost(
    base_score: f32,
    authority_multiplier: f32,
    freshness_multiplier: f32,
    query_type: QueryType,
) -> f32 {
    // Adaptive weights based on query type
    let (auth_weight, fresh_weight) = match query_type {
        QueryType::News => (0.2, 0.4),      // freshness matters most
        QueryType::Research => (0.4, 0.1),   // authority matters most
        QueryType::Technical => (0.3, 0.1),  // authority > freshness
        QueryType::Factual => (0.25, 0.15),  // balanced
        QueryType::General => (0.2, 0.2),    // balanced
    };

    let auth_boost = 1.0 + auth_weight * (authority_multiplier - 1.0);
    let fresh_boost = 1.0 + fresh_weight * (freshness_multiplier - 1.0);

    base_score * auth_boost * fresh_boost
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn recent_doc_high_score() {
        let now = Utc::now();
        let decay = freshness_decay(Some(now), QueryType::General);
        assert!(decay > 0.99, "decay={decay}");
    }

    #[test]
    fn old_doc_low_score_for_news() {
        let old = Utc::now() - Duration::days(30);
        let decay = freshness_decay(Some(old), QueryType::News);
        assert!(decay < 0.1, "decay={decay}, 30-day old news should decay heavily");
    }

    #[test]
    fn old_doc_moderate_for_research() {
        let old = Utc::now() - Duration::days(30);
        let decay = freshness_decay(Some(old), QueryType::Research);
        assert!(decay > 0.8, "decay={decay}, 30-day old research should barely decay");
    }

    #[test]
    fn unknown_date_mild_penalty() {
        let decay = freshness_decay(None, QueryType::General);
        assert_eq!(decay, 0.8);
    }

    #[test]
    fn decay_never_zero() {
        let ancient = Utc::now() - Duration::days(10000);
        let decay = freshness_decay(Some(ancient), QueryType::News);
        assert!(decay >= 0.01, "decay={decay} should never be zero");
    }

    #[test]
    fn authority_freshness_boost_news() {
        let score = authority_freshness_boost(1.0, 1.2, 0.5, QueryType::News);
        // Authority boost small, freshness penalty significant
        assert!(score < 1.0, "score={score}, news with old content should be penalized");
    }

    #[test]
    fn authority_freshness_boost_research() {
        let score = authority_freshness_boost(1.0, 1.2, 0.5, QueryType::Research);
        // Authority boost significant, freshness penalty small
        // 1.0 * (1 + 0.4*0.2) * (1 + 0.1*-0.5) = 1.08 * 0.95 = 1.026
        assert!(score > 0.9, "score={score}, research with high authority should stay high");
    }

    #[test]
    fn boost_preserves_zero() {
        let score = authority_freshness_boost(0.0, 1.5, 1.0, QueryType::General);
        assert_eq!(score, 0.0);
    }
}
