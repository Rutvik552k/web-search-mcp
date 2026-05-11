mod hash_embedder;

#[cfg(feature = "candle")]
mod candle_embedder;

use async_trait::async_trait;
use web_search_common::Result;

pub use hash_embedder::HashEmbedder;

#[cfg(feature = "candle")]
pub use candle_embedder::CandleEmbedder;

/// Trait abstraction for embedding engines.
///
/// Implementations:
/// - `HashEmbedder` — feature hashing, zero deps, works everywhere (default)
/// - `CandleEmbedder` — neural embeddings via candle (feature: "candle")
/// - `OnnxEmbedder` — ONNX Runtime (feature: "onnx", MSVC targets only)
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Generate embeddings for a batch of texts.
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Generate embedding for a single text.
    async fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let mut results = self.embed(&[text]).await?;
        results
            .pop()
            .ok_or_else(|| web_search_common::Error::Embedding("empty result".into()))
    }

    /// Return the dimensionality of embeddings.
    fn dimensions(&self) -> usize;

    /// Return the model/engine name for logging.
    fn model_name(&self) -> &str;
}

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have same length");

    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        dot / denom
    }
}

/// Create the best available embedder based on compiled features and config.
pub fn create_embedder(config: &web_search_common::config::EmbedderConfig) -> Box<dyn Embedder> {
    #[cfg(feature = "candle")]
    {
        match CandleEmbedder::new(config) {
            Ok(e) => {
                tracing::info!(model = e.model_name(), dim = e.dimensions(), "Using CandleEmbedder");
                return Box::new(e);
            }
            Err(e) => {
                tracing::warn!("CandleEmbedder init failed, falling back to HashEmbedder: {e}");
            }
        }
    }

    let e = HashEmbedder::new(config.embedding_dim);
    tracing::info!(dim = e.dimensions(), "Using HashEmbedder (feature hashing)");
    Box::new(e)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn cosine_opposite_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6);
    }
}
