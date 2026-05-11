use web_search_common::models::QueryType;

/// Detect query type from keywords for adaptive ranking weights.
pub fn detect_query_type(query: &str) -> QueryType {
    let q = query.to_lowercase();

    let news_signals = [
        "latest", "today", "breaking", "news", "update", "announce",
        "report", "yesterday", "this week", "recent",
    ];
    let research_signals = [
        "study", "research", "paper", "analysis", "impact", "effect",
        "evidence", "systematic", "review", "meta-analysis", "journal",
    ];
    let technical_signals = [
        "api", "code", "function", "error", "bug", "implement", "library",
        "framework", "documentation", "tutorial", "how to", "example",
        "syntax", "install", "config", "setup",
    ];
    let factual_signals = [
        "what is", "who is", "when did", "where is", "how many",
        "definition", "meaning", "capital of", "population",
    ];

    let news_count = news_signals.iter().filter(|s| q.contains(*s)).count();
    let research_count = research_signals.iter().filter(|s| q.contains(*s)).count();
    let technical_count = technical_signals.iter().filter(|s| q.contains(*s)).count();
    let factual_count = factual_signals.iter().filter(|s| q.contains(*s)).count();

    let max = news_count.max(research_count).max(technical_count).max(factual_count);

    if max == 0 {
        return QueryType::General;
    }

    if news_count == max {
        QueryType::News
    } else if research_count == max {
        QueryType::Research
    } else if technical_count == max {
        QueryType::Technical
    } else {
        QueryType::Factual
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_news() {
        assert_eq!(detect_query_type("latest news on climate change"), QueryType::News);
    }

    #[test]
    fn detects_research() {
        assert_eq!(detect_query_type("research on impact of microplastics"), QueryType::Research);
    }

    #[test]
    fn detects_technical() {
        assert_eq!(detect_query_type("how to install rust library"), QueryType::Technical);
    }

    #[test]
    fn detects_factual() {
        assert_eq!(detect_query_type("what is the capital of France"), QueryType::Factual);
    }

    #[test]
    fn defaults_to_general() {
        assert_eq!(detect_query_type("cats and dogs"), QueryType::General);
    }
}
