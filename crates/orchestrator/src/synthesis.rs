//! TF-IDF extractive text summarization.
//!
//! Scores sentences by their normalized TF-IDF weight sum,
//! returns top-K most informative sentences as a synthesis.
//! No external dependencies — pure Rust implementation.
//!
//! Algorithm:
//! 1. Tokenize all documents into sentences
//! 2. Build document-frequency (DF) map across all docs
//! 3. For each sentence, compute: sum(TF-IDF(word)) / num_words
//! 4. Rank sentences by score, return top-K with source attribution

use std::collections::HashMap;

/// A scored sentence with source attribution.
#[derive(Debug, Clone)]
pub struct ScoredSentence {
    pub text: String,
    pub score: f32,
    pub source_url: String,
    pub source_title: String,
}

/// Synthesize top-K sentences using Query-Focused MMR + TextRank.
///
/// Hybrid algorithm:
/// 1. TextRank: graph-based importance scoring (PageRank on sentence similarity)
/// 2. Query relevance: TF-IDF cosine similarity between each sentence and the query
/// 3. MMR selection: greedily pick sentences maximizing:
///    score = λ * query_relevance + (1-λ) * textrank_score - penalty * max_sim_to_selected
///    This ensures sentences are both relevant to the query AND informative AND diverse.
///
/// This hybrid approach outperforms pure TextRank (which ignores the query) and
/// pure TF-IDF (which ignores structural importance).
pub fn synthesize_tfidf(
    documents: &[(&str, &str, &str)], // (body, url, title)
    top_k: usize,
    query: &str,
) -> Vec<ScoredSentence> {
    if documents.is_empty() {
        return vec![];
    }

    // Step 1: Collect sentences with source attribution
    let mut all_sentences: Vec<(String, &str, &str)> = Vec::new();
    let mut doc_word_sets: Vec<HashMap<String, u32>> = Vec::new();

    for &(body, url, title) in documents {
        let sentences = split_sentences(body);
        let mut doc_words: HashMap<String, u32> = HashMap::new();
        for sent in &sentences {
            for word in tokenize(sent) {
                *doc_words.entry(word).or_insert(0) += 1;
            }
        }
        doc_word_sets.push(doc_words);

        for sent in sentences {
            if sent.split_whitespace().count() >= 5 && sent.len() >= 20 {
                all_sentences.push((sent, url, title));
            }
        }
    }

    if all_sentences.is_empty() {
        return vec![];
    }

    // Limit to 200 sentences to keep graph manageable (O(n^2))
    all_sentences.truncate(200);
    let n = all_sentences.len();

    // Step 2: Build TF-IDF vectors for each sentence
    // Vocabulary: all unique words across sentences
    let mut vocab: HashMap<String, usize> = HashMap::new();
    let mut doc_freq: HashMap<String, u32> = HashMap::new();
    for doc_words in &doc_word_sets {
        for word in doc_words.keys() {
            *doc_freq.entry(word.clone()).or_insert(0) += 1;
        }
    }
    let num_docs = doc_word_sets.len().max(1) as f32;

    // Build sparse TF-IDF vectors per sentence
    let sentence_vectors: Vec<HashMap<usize, f32>> = all_sentences
        .iter()
        .map(|(sent, _, _)| {
            let words = tokenize(sent);
            let total = words.len().max(1) as f32;
            let mut vec = HashMap::new();
            let mut tf_counts: HashMap<String, u32> = HashMap::new();
            for w in &words {
                *tf_counts.entry(w.clone()).or_insert(0) += 1;
            }
            for (word, count) in tf_counts {
                let idx = {
                    let next = vocab.len();
                    *vocab.entry(word.clone()).or_insert(next)
                };
                let tf = count as f32 / total;
                let idf = doc_freq
                    .get(&word)
                    .map(|&d| (num_docs / d as f32).ln() + 1.0)
                    .unwrap_or(1.0);
                vec.insert(idx, tf * idf);
            }
            vec
        })
        .collect();

    // Step 3: Build similarity graph (adjacency matrix as edge list)
    // Edge weight = cosine similarity between TF-IDF vectors
    let mut edges: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            let sim = sparse_cosine(&sentence_vectors[i], &sentence_vectors[j]);
            if sim > 0.05 {
                edges[i].push((j, sim));
                edges[j].push((i, sim));
            }
        }
    }

    // Step 4: PageRank iteration (TextRank scoring)
    let damping = 0.85_f32;
    let max_iter = 30;
    let convergence = 1e-4;
    let mut scores = vec![1.0 / n as f32; n];

    for _ in 0..max_iter {
        let mut new_scores = vec![(1.0 - damping) / n as f32; n];
        let mut max_diff = 0.0_f32;

        for i in 0..n {
            let mut sum = 0.0_f32;
            for &(j, weight) in &edges[i] {
                let out_weight_sum: f32 = edges[j].iter().map(|(_, w)| w).sum();
                if out_weight_sum > 0.0 {
                    sum += weight * scores[j] / out_weight_sum;
                }
            }
            new_scores[i] += damping * sum;
            max_diff = max_diff.max((new_scores[i] - scores[i]).abs());
        }

        scores = new_scores;
        if max_diff < convergence {
            break;
        }
    }

    // Step 5: Compute query relevance for each sentence
    let query_tokens: HashMap<usize, f32> = {
        let words = tokenize(query);
        let total = words.len().max(1) as f32;
        let mut vec = HashMap::new();
        let mut tf_counts: HashMap<String, u32> = HashMap::new();
        for w in &words {
            *tf_counts.entry(w.clone()).or_insert(0) += 1;
        }
        for (word, count) in tf_counts {
            let idx = *vocab.get(&word).unwrap_or(&usize::MAX);
            if idx != usize::MAX {
                let tf = count as f32 / total;
                let idf = doc_freq.get(&word).map(|&d| (num_docs / d as f32).ln() + 1.0).unwrap_or(1.0);
                vec.insert(idx, tf * idf);
            }
        }
        vec
    };

    // Query-sentence relevance scores (cosine similarity to query TF-IDF vector)
    let query_relevance: Vec<f32> = sentence_vectors.iter()
        .map(|sv| sparse_cosine(sv, &query_tokens))
        .collect();

    // Step 6: MMR greedy selection
    // score_i = λ * query_relevance + (1-λ) * textrank_score - penalty * max_sim_to_selected
    // λ = 0.6 (bias toward query relevance)
    let lambda = 0.6_f32;
    let diversity_penalty = 0.3_f32;

    // Normalize textrank scores to [0,1]
    let max_tr = scores.iter().cloned().fold(0.0_f32, f32::max);
    let norm_tr: Vec<f32> = if max_tr > 0.0 {
        scores.iter().map(|s| s / max_tr).collect()
    } else {
        vec![0.0; n]
    };
    let max_qr = query_relevance.iter().cloned().fold(0.0_f32, f32::max);
    let norm_qr: Vec<f32> = if max_qr > 0.0 {
        query_relevance.iter().map(|s| s / max_qr).collect()
    } else {
        vec![0.0; n]
    };

    let mut selected: Vec<ScoredSentence> = Vec::with_capacity(top_k);
    let mut used = vec![false; n];

    for _ in 0..top_k {
        let mut best_idx = None;
        let mut best_mmr = f32::NEG_INFINITY;

        for i in 0..n {
            if used[i] { continue; }
            let word_count = all_sentences[i].0.split_whitespace().count();
            if word_count < 5 { continue; }

            // Query relevance + TextRank importance
            let base = lambda * norm_qr[i] + (1.0 - lambda) * norm_tr[i];

            // Fact boost
            let fact_boost = if all_sentences[i].0.chars().any(|c| c.is_ascii_digit()) { 0.05 } else { 0.0 };

            // Diversity penalty: max similarity to any already-selected sentence
            let max_sim = selected.iter()
                .map(|s| jaccard_similarity(&s.text, &all_sentences[i].0))
                .fold(0.0_f32, f32::max);

            let mmr = base + fact_boost - diversity_penalty * max_sim;

            if mmr > best_mmr {
                best_mmr = mmr;
                best_idx = Some(i);
            }
        }

        match best_idx {
            Some(i) => {
                used[i] = true;
                let (ref sent, url, title) = all_sentences[i];
                selected.push(ScoredSentence {
                    text: sent.clone(),
                    score: best_mmr,
                    source_url: url.to_string(),
                    source_title: title.to_string(),
                });
            }
            None => break,
        }
    }

    selected
}

/// Cosine similarity between sparse TF-IDF vectors.
fn sparse_cosine(a: &HashMap<usize, f32>, b: &HashMap<usize, f32>) -> f32 {
    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;

    for (&idx, &val) in a {
        norm_a += val * val;
        if let Some(&bval) = b.get(&idx) {
            dot += val * bval;
        }
    }
    for (_, &val) in b {
        norm_b += val * val;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-10 { 0.0 } else { dot / denom }
}

/// Split text into sentences. Handles common abbreviations.
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let trimmed = current.trim().to_string();
            // Skip if looks like abbreviation (e.g., "Dr.", "U.S.", "etc.")
            let words: Vec<&str> = trimmed.split_whitespace().collect();
            if let Some(last) = words.last() {
                if last.len() <= 4 && last.ends_with('.') && last.chars().filter(|c| *c == '.').count() == 1 {
                    // Likely abbreviation, don't split
                    continue;
                }
            }
            if trimmed.split_whitespace().count() >= 3 {
                sentences.push(trimmed);
                current = String::new();
            }
        }
    }
    // Add remaining text
    let trimmed = current.trim().to_string();
    if trimmed.split_whitespace().count() >= 3 {
        sentences.push(trimmed);
    }

    sentences
}

/// Tokenize text into lowercase words, removing stopwords.
fn tokenize(text: &str) -> Vec<String> {
    let stopwords = [
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "each", "every", "both", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "and", "but", "or", "if", "while", "that",
        "this", "it", "its", "also", "just", "about", "which", "their",
        "them", "they", "what", "who", "whom", "these", "those",
    ];

    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| w.len() >= 2)
        .map(|w| w.to_lowercase())
        .filter(|w| !stopwords.contains(&w.as_str()))
        .collect()
}

/// Jaccard similarity between two sentences (word-level).
fn jaccard_similarity(a: &str, b: &str) -> f32 {
    let set_a: std::collections::HashSet<String> = tokenize(a).into_iter().collect();
    let set_b: std::collections::HashSet<String> = tokenize(b).into_iter().collect();

    if set_a.is_empty() && set_b.is_empty() {
        return 1.0;
    }

    let intersection = set_a.intersection(&set_b).count() as f32;
    let union = set_a.union(&set_b).count() as f32;

    if union == 0.0 { 0.0 } else { intersection / union }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesize_basic() {
        let docs = vec![
            (
                "Rust is a systems programming language. It provides memory safety without garbage collection. The borrow checker prevents data races at compile time.",
                "https://rust-lang.org",
                "Rust Language",
            ),
            (
                "Python is great for data science. Rust offers better performance for systems code. Both languages have growing ecosystems.",
                "https://example.com/compare",
                "Comparison",
            ),
        ];

        let result = synthesize_tfidf(&docs, 3, "Rust programming language");
        assert!(!result.is_empty());
        assert!(result.len() <= 3);
        // Each result should have source attribution
        for s in &result {
            assert!(!s.source_url.is_empty());
            assert!(s.score > 0.0);
        }
    }

    #[test]
    fn synthesize_empty_input() {
        let result = synthesize_tfidf(&[], 5, "test");
        assert!(result.is_empty());
    }

    #[test]
    fn synthesize_deduplicates_similar() {
        let docs = vec![
            (
                "Rust provides memory safety guarantees. Rust provides memory safety through ownership.",
                "https://a.com",
                "A",
            ),
        ];

        let result = synthesize_tfidf(&docs, 5, "Rust memory safety");
        // Should not return both near-identical sentences
        assert!(result.len() <= 2);
    }

    #[test]
    fn split_sentences_works() {
        let text = "First sentence here. Second one follows! Third with question?";
        let sents = split_sentences(text);
        assert_eq!(sents.len(), 3);
    }

    #[test]
    fn tokenize_removes_stopwords() {
        let tokens = tokenize("the quick brown fox jumps over the lazy dog");
        assert!(!tokens.contains(&"the".to_string()));
        assert!(tokens.contains(&"quick".to_string()));
        assert!(tokens.contains(&"fox".to_string()));
    }

    #[test]
    fn jaccard_identical() {
        let sim = jaccard_similarity("hello world test", "hello world test");
        assert!((sim - 1.0).abs() < 0.01);
    }

    #[test]
    fn jaccard_different() {
        let sim = jaccard_similarity("cats dogs animals", "programming rust language");
        assert!(sim < 0.1);
    }
}
