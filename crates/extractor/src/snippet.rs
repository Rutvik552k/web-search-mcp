/// Extract the most query-relevant snippet from body text.
///
/// Finds the sentence window with highest overlap to query terms,
/// returns a ~300 char snippet centered on that window.
pub fn extract_snippet(body: &str, query: &str, max_len: usize) -> String {
    if body.is_empty() || query.is_empty() {
        return body.chars().take(max_len).collect();
    }

    let query_terms: Vec<String> = query
        .to_lowercase()
        .split_whitespace()
        .filter(|w| w.len() > 2)
        .map(|w| w.to_string())
        .collect();

    if query_terms.is_empty() {
        return body.chars().take(max_len).collect();
    }

    let sentences: Vec<&str> = body
        .split(|c: char| c == '.' || c == '!' || c == '?' || c == '\n')
        .map(|s| s.trim())
        .filter(|s| s.len() > 20)
        .collect();

    if sentences.is_empty() {
        return body.chars().take(max_len).collect();
    }

    // Score each sentence by query term overlap
    let mut best_idx = 0;
    let mut best_score = 0;

    for (i, sentence) in sentences.iter().enumerate() {
        let lower = sentence.to_lowercase();
        let score: usize = query_terms
            .iter()
            .filter(|term| lower.contains(term.as_str()))
            .count();
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }

    // Build snippet from best sentence + neighbors
    let start = best_idx.saturating_sub(1);
    let end = (best_idx + 3).min(sentences.len());
    let snippet: String = sentences[start..end].join(". ");

    if snippet.len() > max_len {
        format!("{}...", &snippet[..max_len])
    } else {
        snippet
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_relevant_sentence() {
        let body = "The weather is nice today. Rust programming language provides memory safety without garbage collection. Python is popular for data science. Java runs on many platforms.";
        let snippet = extract_snippet(body, "rust memory safety", 300);
        assert!(snippet.contains("Rust programming"));
        assert!(snippet.contains("memory safety"));
    }

    #[test]
    fn handles_empty_query() {
        let body = "Some content here that is long enough to be useful.";
        let snippet = extract_snippet(body, "", 100);
        assert!(!snippet.is_empty());
    }

    #[test]
    fn respects_max_len() {
        let body = "A".repeat(1000);
        let snippet = extract_snippet(&body, "test", 200);
        assert!(snippet.len() <= 203); // 200 + "..."
    }

    #[test]
    fn handles_short_body() {
        let snippet = extract_snippet("Short.", "query", 300);
        assert_eq!(snippet, "Short.");
    }
}
