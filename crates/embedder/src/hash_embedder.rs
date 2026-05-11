use async_trait::async_trait;
use siphasher::sip::SipHasher;
use std::hash::{Hash, Hasher};
use web_search_common::Result;

use crate::Embedder;

/// Feature-hashing embedder that generates dense vectors without any ML model.
///
/// Uses locality-sensitive hashing: text → word n-grams → hash each n-gram →
/// accumulate into fixed-dim vector → L2 normalize.
///
/// Not as good as neural embeddings but:
/// - Zero model downloads
/// - Works on any platform
/// - Instant startup
/// - Deterministic
/// - Captures word co-occurrence patterns via n-gram overlap
pub struct HashEmbedder {
    dim: usize,
    ngram_sizes: Vec<usize>,
}

impl HashEmbedder {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            ngram_sizes: vec![1, 2, 3], // unigrams, bigrams, trigrams
        }
    }

    fn embed_text(&self, text: &str) -> Vec<f32> {
        let mut vector = vec![0.0_f32; self.dim];

        let words = tokenize(text);
        if words.is_empty() {
            return vector;
        }

        // Generate n-gram features and hash into vector dimensions
        for &n in &self.ngram_sizes {
            let weight = ngram_weight(n);

            if words.len() < n {
                continue;
            }

            for window in words.windows(n) {
                let ngram = window.join(" ");

                // Use two different hash seeds to get dimension index and sign
                let h1 = hash_with_seed(&ngram, 0);
                let h2 = hash_with_seed(&ngram, 1);

                let idx = (h1 as usize) % self.dim;
                let sign = if h2 & 1 == 0 { 1.0 } else { -1.0 };

                vector[idx] += sign * weight;
            }
        }

        // L2 normalize
        l2_normalize(&mut vector);

        vector
    }
}

#[async_trait]
impl Embedder for HashEmbedder {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.embed_text(t)).collect())
    }

    fn dimensions(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        "hash-embedder"
    }
}

/// Simple whitespace + punctuation tokenizer.
/// Lowercases, strips non-alphanumeric, filters stopwords.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty() && w.len() > 1)
        .filter(|w| !is_stopword(w))
        .map(|w| w.to_string())
        .collect()
}

/// Weight for n-gram size: unigrams=1.0, bigrams=1.5, trigrams=2.0
/// Higher weight for longer n-grams captures more context.
fn ngram_weight(n: usize) -> f32 {
    match n {
        1 => 1.0,
        2 => 1.5,
        3 => 2.0,
        _ => 1.0,
    }
}

fn hash_with_seed(text: &str, seed: u64) -> u64 {
    let mut hasher = SipHasher::new_with_keys(seed, seed.wrapping_mul(0x517cc1b727220a95));
    text.hash(&mut hasher);
    hasher.finish()
}

fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn is_stopword(word: &str) -> bool {
    matches!(
        word,
        "the" | "is" | "at" | "which" | "on" | "a" | "an" | "and" | "or" | "but"
        | "in" | "with" | "to" | "for" | "of" | "not" | "no" | "can" | "had"
        | "has" | "have" | "it" | "its" | "was" | "were" | "will" | "be" | "been"
        | "being" | "do" | "does" | "did" | "this" | "that" | "these" | "those"
        | "am" | "are" | "if" | "by" | "from" | "up" | "out" | "so" | "than"
        | "too" | "very" | "just" | "about" | "into" | "through" | "during"
        | "before" | "after" | "above" | "below" | "between" | "each" | "all"
        | "both" | "few" | "more" | "most" | "other" | "some" | "such" | "only"
        | "own" | "same" | "then" | "when" | "where" | "why" | "how" | "what"
        | "who" | "whom" | "he" | "she" | "they" | "we" | "you" | "me" | "him"
        | "her" | "us" | "them" | "my" | "your" | "his" | "our" | "their"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosine_similarity;

    #[tokio::test]
    async fn identical_texts_have_similarity_one() {
        let emb = HashEmbedder::new(384);
        let vecs = emb.embed(&["hello world", "hello world"]).await.unwrap();
        let sim = cosine_similarity(&vecs[0], &vecs[1]);
        assert!((sim - 1.0).abs() < 1e-5, "sim={sim}");
    }

    #[tokio::test]
    async fn similar_texts_have_high_similarity() {
        let emb = HashEmbedder::new(384);
        let vecs = emb
            .embed(&[
                "machine learning algorithms improve prediction accuracy",
                "deep learning models enhance predictive performance",
            ])
            .await
            .unwrap();
        let sim = cosine_similarity(&vecs[0], &vecs[1]);
        // Related texts should have positive similarity
        assert!(sim > 0.0, "sim={sim} should be > 0");
    }

    #[tokio::test]
    async fn unrelated_texts_have_low_similarity() {
        let emb = HashEmbedder::new(384);
        let vecs = emb
            .embed(&[
                "quantum physics experiments at the large hadron collider",
                "chocolate cake recipe with cream cheese frosting",
            ])
            .await
            .unwrap();
        let sim = cosine_similarity(&vecs[0], &vecs[1]);
        // Unrelated texts should have low similarity
        assert!(sim.abs() < 0.5, "sim={sim} should be near 0");
    }

    #[tokio::test]
    async fn vectors_are_normalized() {
        let emb = HashEmbedder::new(384);
        let vecs = emb
            .embed(&["test normalization of output vectors"])
            .await
            .unwrap();
        let norm: f32 = vecs[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm={norm} should be 1.0");
    }

    #[tokio::test]
    async fn correct_dimensions() {
        let emb = HashEmbedder::new(384);
        assert_eq!(emb.dimensions(), 384);
        let vecs = emb.embed(&["test"]).await.unwrap();
        assert_eq!(vecs[0].len(), 384);
    }

    #[tokio::test]
    async fn empty_text_returns_zero_vector() {
        let emb = HashEmbedder::new(384);
        let vecs = emb.embed(&[""]).await.unwrap();
        assert!(vecs[0].iter().all(|&x| x == 0.0));
    }

    #[tokio::test]
    async fn batch_embedding() {
        let emb = HashEmbedder::new(384);
        let texts = vec!["first document", "second document", "third document"];
        let vecs = emb.embed(&texts).await.unwrap();
        assert_eq!(vecs.len(), 3);
        for v in &vecs {
            assert_eq!(v.len(), 384);
        }
    }
}
