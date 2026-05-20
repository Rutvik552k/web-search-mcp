/// Query reformulation: expand a query with synonyms, related terms,
/// question variants, and domain-specific expansions.
pub fn reformulate_query(query: &str) -> Vec<String> {
    let mut queries = vec![query.to_string()];

    let q = query.to_lowercase();
    let words: Vec<&str> = q.split_whitespace().collect();

    // Apply synonym expansions
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

    // Generate question variants for factual queries
    if !q.contains('?') && words.len() >= 2 {
        if q.starts_with("why") || q.starts_with("how") || q.starts_with("what") {
            // Already a question, add declarative variant
            let declarative = q.replace("why ", "").replace("how ", "").replace("what is ", "");
            if !queries.contains(&declarative) {
                queries.push(declarative);
            }
        }
    }

    // Generate site-specific variants for technical queries
    if is_technical(&q) {
        queries.push(format!("{q} documentation"));
        queries.push(format!("{q} tutorial"));
    }

    // Generate domain-specific expansions
    if is_scientific(&q) {
        queries.push(format!("{q} research paper"));
    }
    if is_medical(&q) {
        queries.push(format!("{q} clinical study"));
    }
    if is_news_query(&q) {
        queries.push(format!("{q} 2026")); // add current year
    }

    // Limit to avoid crawling too many seed sets
    queries.truncate(6);
    queries
}

fn is_technical(q: &str) -> bool {
    let tech_terms = [
        "api", "code", "programming", "library", "framework",
        "install", "setup", "config", "error", "debug",
        "function", "method", "class", "module", "package",
        "rust", "python", "javascript", "typescript", "docker",
        "kubernetes", "database", "sql", "linux", "git",
    ];
    tech_terms.iter().any(|t| q.contains(t))
}

fn is_scientific(q: &str) -> bool {
    let terms = [
        "study", "research", "hypothesis", "experiment", "data",
        "analysis", "correlation", "evidence", "peer-reviewed",
        "journal", "methodology", "findings", "meta-analysis",
    ];
    terms.iter().any(|t| q.contains(t))
}

fn is_medical(q: &str) -> bool {
    let terms = [
        "treatment", "diagnosis", "symptoms", "disease", "therapy",
        "clinical", "patient", "drug", "medication", "health",
        "cancer", "diabetes", "cardiac", "vaccine",
    ];
    terms.iter().any(|t| q.contains(t))
}

fn is_news_query(q: &str) -> bool {
    let terms = [
        "news", "latest", "today", "breaking", "update",
        "announcement", "recent", "current", "happening",
    ];
    terms.iter().any(|t| q.contains(t))
}

/// Common synonym pairs for query expansion — 40+ mappings.
const SYNONYMS: &[(&str, &[&str])] = &[
    // Cause & effect
    ("impact", &["effect", "influence"]),
    ("effect", &["impact", "consequence"]),
    ("cause", &["lead to", "result in"]),
    ("prevent", &["avoid", "mitigate"]),
    // Evaluation
    ("benefit", &["advantage", "positive effect"]),
    ("drawback", &["disadvantage", "limitation"]),
    ("risk", &["danger", "threat", "vulnerability"]),
    ("opportunity", &["potential", "prospect"]),
    // Performance
    ("fast", &["quick", "rapid", "high-performance"]),
    ("slow", &["sluggish", "poor performance"]),
    ("efficient", &["optimized", "streamlined"]),
    ("scalable", &["extensible", "high-throughput"]),
    // Actions
    ("improve", &["enhance", "optimize", "boost"]),
    ("reduce", &["decrease", "minimize", "lower"]),
    ("create", &["build", "develop", "implement"]),
    ("remove", &["delete", "eliminate", "drop"]),
    ("fix", &["resolve", "patch", "repair"]),
    ("deploy", &["release", "ship", "launch"]),
    // Comparison
    ("compare", &["versus", "vs", "difference between"]),
    ("best", &["top", "recommended", "optimal"]),
    ("alternative", &["replacement", "substitute", "instead of"]),
    // Recency
    ("latest", &["recent", "new", "current"]),
    ("outdated", &["deprecated", "legacy", "obsolete"]),
    // Research
    ("research", &["study", "analysis", "investigation"]),
    ("evidence", &["data", "proof", "findings"]),
    // Domain synonyms
    ("climate change", &["global warming", "climate crisis"]),
    ("machine learning", &["ML", "artificial intelligence"]),
    ("artificial intelligence", &["AI", "machine learning"]),
    ("deep learning", &["neural networks", "DL"]),
    ("natural language processing", &["NLP", "text analysis"]),
    ("cybersecurity", &["information security", "infosec"]),
    ("cryptocurrency", &["crypto", "digital currency"]),
    ("blockchain", &["distributed ledger", "DLT"]),
    ("open source", &["OSS", "FOSS"]),
    ("cloud computing", &["cloud infrastructure", "IaaS"]),
    ("container", &["docker", "containerization"]),
    ("microservices", &["service-oriented", "distributed services"]),
    ("devops", &["CI/CD", "continuous delivery"]),
    ("startup", &["early-stage company", "venture"]),
    ("remote work", &["work from home", "distributed team"]),
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
    fn limited_to_6() {
        let queries = reformulate_query("impact of artificial intelligence on climate change");
        assert!(queries.len() <= 6);
    }

    #[test]
    fn medical_query_expansion() {
        let queries = reformulate_query("diabetes treatment options");
        assert!(queries.iter().any(|q| q.contains("clinical study")));
    }

    #[test]
    fn news_query_adds_year() {
        let queries = reformulate_query("latest tech news");
        assert!(queries.iter().any(|q| q.contains("2026")));
    }
}
