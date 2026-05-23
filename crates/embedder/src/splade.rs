// SPDX-License-Identifier: MIT

//! SPLADE sparse encoder via ONNX Runtime.
//!
//! Converts text to sparse term weights (vocab-sized vector).
//! Used to augment Tantivy BM25 with semantic term expansion:
//! "car" activates "vehicle", "automobile", "sedan" etc.
//!
//! Model: prithivida/Splade_PP_en_v1 (ONNX, ~532MB)
//! Output: top-K (term_text, weight) pairs per document.

use hf_hub::{api::sync::Api, Repo, RepoType};
use ort::session::Session;
use ort::value::Tensor;
use std::sync::Mutex;
use tokenizers::Tokenizer;
use web_search_common::{Error, Result};

const MODEL_ID: &str = "prithivida/Splade_PP_en_v1";
const VOCAB_SIZE: usize = 30_522;

/// SPLADE sparse encoder.
///
/// Produces weighted term vectors: each vocabulary token gets a relevance weight.
/// Top-K terms are returned for indexing in Tantivy.
pub struct SpladeEncoder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    /// Reverse vocab: token_id → token_text
    id_to_token: Vec<String>,
    /// Max tokens to include in sparse output
    max_terms: usize,
    /// Minimum weight threshold for a term to be included
    min_weight: f32,
}

impl SpladeEncoder {
    /// Load SPLADE model from HuggingFace ONNX export.
    pub fn new() -> Result<Self> {
        Self::with_config(100, 0.1)
    }

    /// Load with custom term limits.
    pub fn with_config(max_terms: usize, min_weight: f32) -> Result<Self> {
        tracing::info!(model = MODEL_ID, "Loading SPLADE encoder (ONNX)...");

        let api = Api::new().map_err(|e| Error::Embedding(format!("HF API: {e}")))?;
        let repo = api.repo(Repo::new(MODEL_ID.to_string(), RepoType::Model));

        // Download tokenizer
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| Error::Embedding(format!("fetch tokenizer: {e}")))?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Embedding(format!("load tokenizer: {e}")))?;

        // Build reverse vocab
        let vocab = tokenizer.get_vocab(true);
        let mut id_to_token = vec![String::new(); vocab.len().max(VOCAB_SIZE)];
        for (token, id) in &vocab {
            if (*id as usize) < id_to_token.len() {
                id_to_token[*id as usize] = token.clone();
            }
        }

        // Try ONNX model paths
        let onnx_paths = [
            "onnx/model.onnx",
            "model.onnx",
        ];
        let mut onnx_path = None;
        for path in onnx_paths {
            match repo.get(path) {
                Ok(p) => {
                    onnx_path = Some(p);
                    break;
                }
                Err(_) => continue,
            }
        }
        let onnx_path = onnx_path.ok_or_else(|| {
            Error::Embedding(format!("No ONNX model found in {MODEL_ID}"))
        })?;

        // Build ONNX session
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        let session = Session::builder()
            .map_err(|e| Error::Embedding(format!("ort builder: {e}")))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| Error::Embedding(format!("ort opt: {e}")))?
            .with_intra_threads(num_threads)
            .map_err(|e| Error::Embedding(format!("ort threads: {e}")))?
            .commit_from_file(&onnx_path)
            .map_err(|e| Error::Embedding(format!("ort load: {e}")))?;

        tracing::info!(
            model = MODEL_ID,
            vocab_size = id_to_token.len(),
            max_terms,
            threads = num_threads,
            "SPLADE encoder loaded"
        );

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            id_to_token,
            max_terms,
            min_weight,
        })
    }

    /// Encode text into sparse term weights.
    ///
    /// Returns top-K (token_text, weight) pairs sorted by weight descending.
    /// SPLADE pooling: log(1 + ReLU(max_over_tokens(logits)))
    pub fn encode(&self, text: &str) -> Result<Vec<(String, f32)>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| Error::Embedding(format!("tokenize: {e}")))?;

        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        let type_ids = encoding.get_type_ids();
        let seq_len = ids.len();

        // Build input tensors
        let shape = [1, seq_len];
        let input_ids: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = mask.iter().map(|&x| x as i64).collect();
        let token_type_ids: Vec<i64> = type_ids.iter().map(|&x| x as i64).collect();

        let input_ids_t = Tensor::from_array((shape, input_ids))
            .map_err(|e| Error::Embedding(format!("tensor: {e}")))?;
        let attention_mask_t = Tensor::from_array((shape, attention_mask.clone()))
            .map_err(|e| Error::Embedding(format!("tensor: {e}")))?;
        let token_type_ids_t = Tensor::from_array((shape, token_type_ids))
            .map_err(|e| Error::Embedding(format!("tensor: {e}")))?;

        // Run inference
        let mut session = self
            .session
            .lock()
            .map_err(|e| Error::Embedding(format!("lock: {e}")))?;

        let outputs = session
            .run(
                ort::inputs![
                    "input_ids" => input_ids_t,
                    "attention_mask" => attention_mask_t,
                    "token_type_ids" => token_type_ids_t
                ]
                .map_err(|e| Error::Embedding(format!("inputs: {e}")))?,
            )
            .map_err(|e| Error::Embedding(format!("run: {e}")))?;

        // Extract logits: [1, seq_len, vocab_size]
        let output = outputs
            .get(0)
            .ok_or_else(|| Error::Embedding("no output".into()))?;

        let (_shape, data) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Embedding(format!("extract: {e}")))?;

        // SPLADE pooling: for each vocab token, take max over sequence positions,
        // apply attention mask, then log(1 + ReLU(x))
        let vocab_size = if data.len() >= seq_len * VOCAB_SIZE {
            VOCAB_SIZE
        } else {
            // Smaller model or different output shape
            data.len() / seq_len
        };

        let mut sparse = vec![0.0_f32; vocab_size];
        for t in 0..seq_len {
            if attention_mask[t] == 0 {
                continue;
            }
            let offset = t * vocab_size;
            for v in 0..vocab_size {
                if offset + v < data.len() {
                    sparse[v] = sparse[v].max(data[offset + v]);
                }
            }
        }

        // Apply log(1 + ReLU(x)) and collect non-zero terms
        let mut terms: Vec<(String, f32)> = Vec::new();
        for (v, &val) in sparse.iter().enumerate() {
            let activated = (1.0 + val.max(0.0)).ln();
            if activated >= self.min_weight && v < self.id_to_token.len() {
                let token = &self.id_to_token[v];
                // Skip special tokens and single-char tokens
                if !token.is_empty()
                    && !token.starts_with('[')
                    && !token.starts_with('#')
                    && token.len() > 1
                {
                    terms.push((token.clone(), activated));
                }
            }
        }

        // Sort by weight descending, take top-K
        terms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        terms.truncate(self.max_terms);

        Ok(terms)
    }

    /// Convert sparse terms to a Tantivy-friendly string.
    ///
    /// Repeats each term proportionally to its weight so that BM25's TF component
    /// naturally approximates SPLADE dot-product scoring.
    ///
    /// Example: [("vehicle", 3.2), ("car", 2.1)] → "vehicle vehicle vehicle car car"
    pub fn terms_to_tantivy_string(terms: &[(String, f32)]) -> String {
        let mut parts = Vec::new();
        for (term, weight) in terms {
            let repeats = weight.round().max(1.0) as usize;
            for _ in 0..repeats.min(10) {
                // Cap at 10 repeats to avoid bloating index
                parts.push(term.as_str());
            }
        }
        parts.join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tantivy_string_conversion() {
        let terms = vec![
            ("vehicle".to_string(), 3.2),
            ("car".to_string(), 1.8),
        ];
        let result = SpladeEncoder::terms_to_tantivy_string(&terms);
        // "vehicle" repeated 3 times, "car" repeated 2 times
        assert_eq!(result.matches("vehicle").count(), 3);
        assert_eq!(result.matches("car").count(), 2);
    }

    #[test]
    fn tantivy_string_empty() {
        let terms: Vec<(String, f32)> = vec![];
        let result = SpladeEncoder::terms_to_tantivy_string(&terms);
        assert!(result.is_empty());
    }

    #[test]
    fn tantivy_string_caps_repeats() {
        let terms = vec![("important".to_string(), 15.0)];
        let result = SpladeEncoder::terms_to_tantivy_string(&terms);
        // Should cap at 10 repeats
        assert_eq!(result.matches("important").count(), 10);
    }
}
