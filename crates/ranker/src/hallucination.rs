use std::collections::{HashMap, HashSet};
use web_search_common::models::{Claim, Contradiction, ContradictionSeverity, VerificationStatus};

use crate::authority::domain_to_org;

/// Stage 4 anti-hallucination analysis result.
#[derive(Debug, Clone)]
pub struct HallucinationCheck {
    pub claims: Vec<Claim>,
    pub contradictions: Vec<Contradiction>,
    pub warnings: Vec<String>,
    pub unique_orgs: usize,
    pub echo_chamber_risk: bool,
}

/// Document with extracted claims for cross-reference checking.
#[derive(Debug, Clone)]
pub struct DocClaims {
    pub url: String,
    pub domain: String,
    pub key_phrases: Vec<String>,
}

/// Run anti-hallucination checks on a set of ranked documents.
///
/// A. Cross-reference validation: count sources confirming each key phrase
/// B. Contradiction detection: find conflicting claims across sources
/// C. Echo chamber detection: count unique source organizations
pub fn check_hallucination(docs: &[DocClaims], min_orgs: usize) -> HallucinationCheck {
    let mut claims = Vec::new();
    let mut contradictions = Vec::new();
    let mut warnings = Vec::new();

    // Count unique organizations
    let orgs: HashSet<String> = docs.iter().map(|d| domain_to_org(&d.domain)).collect();
    let unique_orgs = orgs.len();

    let echo_chamber_risk = unique_orgs < min_orgs && docs.len() >= min_orgs;
    if echo_chamber_risk {
        warnings.push(format!(
            "Limited source diversity: only {} unique organizations in top {} results (minimum: {})",
            unique_orgs,
            docs.len(),
            min_orgs
        ));
    }

    // Cross-reference: count how many documents mention each key phrase
    // Uses fuzzy matching: phrases that share 60%+ words are considered the same claim
    let mut phrase_sources: HashMap<String, Vec<String>> = HashMap::new();
    let all_phrases: Vec<(String, String)> = docs.iter()
        .flat_map(|doc| {
            doc.key_phrases.iter().map(move |p| (p.to_lowercase(), doc.url.clone()))
        })
        .collect();

    for (phrase, url) in &all_phrases {
        // Try exact match first
        let entry = phrase_sources.entry(phrase.clone()).or_default();
        if !entry.contains(url) {
            entry.push(url.clone());
        }

        // Fuzzy match: check existing phrases for high word overlap
        let phrase_words: HashSet<&str> = phrase.split_whitespace()
            .filter(|w| w.len() > 3) // skip short words
            .collect();
        if phrase_words.len() < 3 { continue; }

        for (existing_phrase, sources) in phrase_sources.iter_mut() {
            if existing_phrase == phrase { continue; }
            let existing_words: HashSet<&str> = existing_phrase.split_whitespace()
                .filter(|w| w.len() > 3)
                .collect();
            if existing_words.len() < 3 { continue; }

            let overlap = phrase_words.intersection(&existing_words).count();
            let max_len = phrase_words.len().max(existing_words.len());
            let overlap_ratio = overlap as f32 / max_len as f32;

            // 60%+ word overlap → same claim
            if overlap_ratio >= 0.6 && !sources.contains(url) {
                sources.push(url.clone());
            }
        }
    }

    // Generate claims with verification status
    // Lowered threshold: 2 orgs = Verified (was 3), 1 org from Tier1 = Partial
    for (phrase, sources) in &phrase_sources {
        let unique_source_orgs: HashSet<String> = sources
            .iter()
            .filter_map(|url| {
                url::Url::parse(url)
                    .ok()
                    .and_then(|u| u.host_str().map(|h| domain_to_org(h)))
            })
            .collect();

        let status = match unique_source_orgs.len() {
            n if n >= 3 => VerificationStatus::Verified,
            2 => VerificationStatus::Verified,
            1 => VerificationStatus::Partial,
            _ => VerificationStatus::Unverified,
        };

        // Only create claims for the first source
        if let Some(first_source) = sources.first() {
            claims.push(Claim {
                text: phrase.clone(),
                source_url: first_source.clone(),
                source_span: (0, 0),
                confidence: match status {
                    VerificationStatus::Verified => 0.9,
                    VerificationStatus::Partial => 0.7,
                    VerificationStatus::Unverified => 0.5,
                    VerificationStatus::Contested => 0.2,
                },
                verification: status,
            });
        }
    }

    // Simple contradiction detection: look for numeric disagreements
    // Full NLI would use an ONNX model; this is a heuristic baseline
    let number_claims = extract_number_claims(docs);
    for ((topic, _), values) in &number_claims {
        if values.len() >= 2 {
            let nums: Vec<f64> = values.iter().map(|(n, _)| *n).collect();
            let max = nums.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min = nums.iter().cloned().fold(f64::INFINITY, f64::min);

            // If values differ by more than 20%, flag as contradiction
            if min > 0.0 && (max - min) / min > 0.2 {
                let v1 = &values[0];
                let v2 = &values[1];
                contradictions.push(Contradiction {
                    claim_a: format!("{}: {}", topic, v1.0),
                    source_a: v1.1.clone(),
                    claim_b: format!("{}: {}", topic, v2.0),
                    source_b: v2.1.clone(),
                    severity: if (max - min) / min > 1.0 {
                        ContradictionSeverity::Hard
                    } else {
                        ContradictionSeverity::Soft
                    },
                });
            }
        }
    }

    if !contradictions.is_empty() {
        warnings.push(format!(
            "{} contradictions detected across sources",
            contradictions.len()
        ));
    }

    HallucinationCheck {
        claims,
        contradictions,
        warnings,
        unique_orgs,
        echo_chamber_risk,
    }
}

/// Extract claims containing numbers for contradiction detection.
/// Returns: Map<(topic_context, unit), Vec<(number, source_url)>>
fn extract_number_claims(
    docs: &[DocClaims],
) -> HashMap<(String, String), Vec<(f64, String)>> {
    let mut number_claims: HashMap<(String, String), Vec<(f64, String)>> = HashMap::new();

    for doc in docs {
        for phrase in &doc.key_phrases {
            // Simple heuristic: find "X is/was/are NUMBER" patterns
            let words: Vec<&str> = phrase.split_whitespace().collect();
            for (i, word) in words.iter().enumerate() {
                if let Ok(num) = word.replace(',', "").replace('%', "").parse::<f64>() {
                    // Use surrounding context as topic key
                    let start = i.saturating_sub(3);
                    let _end = (i + 1).min(words.len());
                    let context = words[start..i].join(" ");
                    let unit = if word.ends_with('%') {
                        "%".to_string()
                    } else if i + 1 < words.len() {
                        words[i + 1].to_string()
                    } else {
                        String::new()
                    };

                    if !context.is_empty() {
                        number_claims
                            .entry((context, unit))
                            .or_default()
                            .push((num, doc.url.clone()));
                    }
                }
            }
        }
    }

    number_claims
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_doc(url: &str, domain: &str, phrases: &[&str]) -> DocClaims {
        DocClaims {
            url: url.to_string(),
            domain: domain.to_string(),
            key_phrases: phrases.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn verified_when_three_sources() {
        let docs = vec![
            make_doc("https://a.com/1", "a.com", &["earth orbits the sun"]),
            make_doc("https://b.com/1", "b.com", &["earth orbits the sun"]),
            make_doc("https://c.com/1", "c.com", &["earth orbits the sun"]),
        ];

        let result = check_hallucination(&docs, 3);
        let claim = result.claims.iter().find(|c| c.text.contains("earth")).unwrap();
        assert_eq!(claim.verification, VerificationStatus::Verified);
    }

    #[test]
    fn partial_when_single_source() {
        let docs = vec![
            make_doc("https://a.com/1", "a.com", &["unique claim only here"]),
            make_doc("https://b.com/1", "b.com", &["different topic entirely"]),
        ];

        let result = check_hallucination(&docs, 3);
        let claim = result.claims.iter().find(|c| c.text.contains("unique")).unwrap();
        assert_eq!(claim.verification, VerificationStatus::Partial);
    }

    #[test]
    fn echo_chamber_detected() {
        let docs = vec![
            make_doc("https://blog.example.com/1", "blog.example.com", &["claim"]),
            make_doc("https://news.example.com/2", "news.example.com", &["claim"]),
            make_doc("https://docs.example.com/3", "docs.example.com", &["claim"]),
        ];

        let result = check_hallucination(&docs, 3);
        assert!(result.echo_chamber_risk, "All from example.com org");
        assert_eq!(result.unique_orgs, 1);
    }

    #[test]
    fn no_echo_chamber_diverse_sources() {
        let docs = vec![
            make_doc("https://a.com/1", "a.com", &["claim"]),
            make_doc("https://b.org/1", "b.org", &["claim"]),
            make_doc("https://c.net/1", "c.net", &["claim"]),
        ];

        let result = check_hallucination(&docs, 3);
        assert!(!result.echo_chamber_risk);
        assert_eq!(result.unique_orgs, 3);
    }

    #[test]
    fn numeric_contradiction_detected() {
        let docs = vec![
            make_doc("https://a.com/1", "a.com", &["the population is 100 million"]),
            make_doc("https://b.com/1", "b.com", &["the population is 500 million"]),
        ];

        let result = check_hallucination(&docs, 2);
        assert!(
            !result.contradictions.is_empty(),
            "Should detect numeric disagreement: 100 vs 500"
        );
    }

    #[test]
    fn no_contradiction_consistent_numbers() {
        let docs = vec![
            make_doc("https://a.com/1", "a.com", &["the speed is 100 mph"]),
            make_doc("https://b.com/1", "b.com", &["the speed is 105 mph"]),
        ];

        let result = check_hallucination(&docs, 2);
        // 5% difference should not trigger (threshold is 20%)
        assert!(result.contradictions.is_empty());
    }

    #[test]
    fn verified_when_two_orgs() {
        let docs = vec![
            make_doc("https://a.com/1", "a.com", &["shared fact"]),
            make_doc("https://b.com/1", "b.com", &["shared fact"]),
        ];

        let result = check_hallucination(&docs, 3);
        let claim = result.claims.iter().find(|c| c.text.contains("shared")).unwrap();
        assert_eq!(claim.verification, VerificationStatus::Verified);
    }
}
