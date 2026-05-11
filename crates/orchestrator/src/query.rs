/// Query reformulation: expand a query with synonyms and related terms
/// to improve recall across diverse sources.
pub fn reformulate_query(query: &str) -> Vec<String> {
    let mut queries = vec![query.to_string()];

    let q = query.to_lowercase();
    let words: Vec<&str> = q.split_whitespace().collect();

    // Apply synonym expansions
    let _expanded = query.to_string();
    for &(term, synonyms) in SYNONYMS.iter() {
        if q.contains(term) {
            for syn in synonyms {
                let variant = q.replace(term, syn);
                if variant != q && !queries.contains(&variant) {
                    queries.push(variant);
                }
            }
        }
    }

    // Generate "what is X" variant for short queries
    if words.len() <= 3 && !q.starts_with("what") && !q.starts_with("how") {
        queries.push(format!("what is {q}"));
    }

    // Generate site-specific variants for technical queries
    if is_technical(&q) {
        queries.push(format!("{q} documentation"));
        queries.push(format!("{q} tutorial"));
    }

    // Limit to avoid crawling too many seed sets
    queries.truncate(4);
    queries
}

fn is_technical(q: &str) -> bool {
    let tech_terms = [
        "api", "code", "programming", "library", "framework",
        "install", "setup", "config", "error", "debug",
        "function", "method", "class", "module", "package",
    ];
    tech_terms.iter().any(|t| q.contains(t))
}

/// Common synonym pairs for query expansion.
const SYNONYMS: &[(&str, &[&str])] = &[
    ("impact", &["effect", "influence"]),
    ("effect", &["impact", "consequence"]),
    ("benefit", &["advantage", "positive effect"]),
    ("drawback", &["disadvantage", "limitation"]),
    ("fast", &["quick", "rapid", "high-performance"]),
    ("slow", &["sluggish", "poor performance"]),
    ("improve", &["enhance", "optimize", "boost"]),
    ("reduce", &["decrease", "minimize", "lower"]),
    ("cause", &["lead to", "result in"]),
    ("prevent", &["avoid", "mitigate"]),
    ("compare", &["versus", "vs", "difference between"]),
    ("best", &["top", "recommended", "optimal"]),
    ("latest", &["recent", "new", "current"]),
    ("research", &["study", "analysis", "investigation"]),
    ("climate change", &["global warming", "climate crisis"]),
    ("machine learning", &["ML", "artificial intelligence"]),
    ("artificial intelligence", &["AI", "machine learning"]),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_query_unchanged() {
        let queries = reformulate_query("rust programming");
        assert!(queries.contains(&"rust programming".to_string()));
    }

    #[test]
    fn synonym_expansion() {
        let queries = reformulate_query("impact of climate change");
        assert!(queries.len() > 1);
        // Should have "effect" variant
        assert!(queries.iter().any(|q| q.contains("effect")));
    }

    #[test]
    fn short_query_gets_what_is() {
        let queries = reformulate_query("quantum computing");
        assert!(queries.iter().any(|q| q.contains("what is")));
    }

    #[test]
    fn technical_query_gets_docs() {
        let queries = reformulate_query("rust api");
        assert!(queries.iter().any(|q| q.contains("documentation") || q.contains("tutorial")));
    }

    #[test]
    fn limited_to_4() {
        let queries = reformulate_query("impact of artificial intelligence on climate change");
        assert!(queries.len() <= 4);
    }
}
