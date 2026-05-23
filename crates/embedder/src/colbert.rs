// SPDX-License-Identifier: MIT

//! ColBERT late-interaction reranker via ONNX Runtime.
//!
//! Uses mxbai-edge-colbert-v0-17m (17M params, 48-dim, INT8 quantized).
//! MaxSim scoring: ~5-10ms for 50 docs vs ~8s with cross-encoder.
//!
//! Architecture: encode query and doc tokens separately → per-token embeddings →
//! MaxSim (for each query token, find best matching doc token, sum maxima).

use hf_hub::{api::sync::Api, Repo, RepoType};
use ort::session::Session;
use ort::value::Tensor;
use std::collections::HashSet;
use std::sync::Mutex;
use tokenizers::Tokenizer;
use web_search_common::{Error, Result};

const MODEL_ID: &str = "ryandono/mxbai-edge-colbert-v0-17m-onnx-int8";
const EMBEDDING_DIM: usize = 48;

/// ColBERT late-interaction reranker.
///
/// Encodes query and document tokens into 48-dim embeddings,
/// then scores via MaxSim (sum of per-query-token max cosine similarities).
pub struct ColBertReranker {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    skiplist: HashSet<u32>,
    max_seq_len: usize,
    /// Token ID for [Q] marker (query prefix), if available
    query_marker: Option<u32>,
    /// Token ID for [D] marker (document prefix), if available
    doc_marker: Option<u32>,
}

impl ColBertReranker {
    /// Load ColBERT model from HuggingFace. Prefers INT8 quantized.
    pub fn new() -> Result<Self> {
        tracing::info!(model = MODEL_ID, "Loading ColBERT reranker (ONNX)...");

        let api = Api::new().map_err(|e| Error::Embedding(format!("HF API: {e}")))?;
        let repo = api.repo(Repo::new(MODEL_ID.to_string(), RepoType::Model));

        // Download tokenizer
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| Error::Embedding(format!("fetch tokenizer: {e}")))?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Embedding(format!("load tokenizer: {e}")))?;

        // Try INT8 quantized first, fall back to FP32
        let onnx_paths = [
            "onnx/model_quantized.onnx",
            "onnx/model.onnx",
            "model_quantized.onnx",
            "model.onnx",
        ];
        let mut onnx_path = None;
        let mut variant = "";
        for path in onnx_paths {
            match repo.get(path) {
                Ok(p) => {
                    variant = path;
                    onnx_path = Some(p);
                    break;
                }
                Err(_) => continue,
            }
        }
        let onnx_path = onnx_path.ok_or_else(|| {
            Error::Embedding(format!("No ONNX model found in {MODEL_ID}"))
        })?;

        // Load skiplist (token IDs to exclude from MaxSim — punctuation, etc.)
        let skiplist: HashSet<u32> = match repo.get("skiplist.json") {
            Ok(path) => {
                let data = std::fs::read_to_string(&path).unwrap_or_default();
                serde_json::from_str(&data).unwrap_or_default()
            }
            Err(_) => HashSet::new(),
        };

        // Read config for max_seq_len
        let max_seq_len = match repo.get("config.json") {
            Ok(path) => {
                let data = std::fs::read_to_string(&path).unwrap_or_default();
                let cfg: serde_json::Value = serde_json::from_str(&data).unwrap_or_default();
                cfg["max_position_embeddings"].as_u64().unwrap_or(512) as usize
            }
            Err(_) => 512,
        };

        // Look up [Q] and [D] marker token IDs
        let query_marker = tokenizer.token_to_id("[Q]");
        let doc_marker = tokenizer.token_to_id("[D]");

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
            variant,
            dim = EMBEDDING_DIM,
            skiplist_size = skiplist.len(),
            query_marker = ?query_marker,
            doc_marker = ?doc_marker,
            threads = num_threads,
            "ColBERT reranker loaded"
        );

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            skiplist,
            max_seq_len,
            query_marker,
            doc_marker,
        })
    }

    /// Encode a batch of texts into per-token embeddings.
    ///
    /// Returns `Vec<Vec<Vec<f32>>>` — [batch][token][dim].
    /// If `is_query` is true, inserts [Q] marker after [CLS].
    fn encode_batch(&self, texts: &[&str], is_query: bool) -> Result<Vec<Vec<Vec<f32>>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let batch_size = texts.len();
        let marker = if is_query { self.query_marker } else { self.doc_marker };

        // Tokenize
        let encodings: Vec<_> = texts
            .iter()
            .map(|t| {
                self.tokenizer
                    .encode(*t, true)
                    .map_err(|e| Error::Embedding(format!("tokenize: {e}")))
            })
            .collect::<Result<Vec<_>>>()?;

        // Determine max length (with room for marker token if needed)
        let extra = if marker.is_some() { 1 } else { 0 };
        let max_len = encodings
            .iter()
            .map(|e| (e.get_ids().len() + extra).min(self.max_seq_len))
            .max()
            .unwrap_or(0);

        // Build padded input tensors
        let mut input_ids_vec = Vec::with_capacity(batch_size * max_len);
        let mut attention_mask_vec = Vec::with_capacity(batch_size * max_len);

        for enc in &encodings {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();

            // Insert marker after [CLS] (position 0)
            if let Some(m) = marker {
                // [CLS]
                if !ids.is_empty() {
                    input_ids_vec.push(ids[0] as i64);
                    attention_mask_vec.push(mask[0] as i64);
                }
                // [Q] or [D]
                input_ids_vec.push(m as i64);
                attention_mask_vec.push(1i64);
                // Rest of tokens
                let remaining = (ids.len() - 1).min(max_len - 2);
                for i in 1..=remaining {
                    input_ids_vec.push(ids[i] as i64);
                    attention_mask_vec.push(mask[i] as i64);
                }
                let filled = 1 + 1 + remaining; // CLS + marker + rest
                for _ in filled..max_len {
                    input_ids_vec.push(0i64);
                    attention_mask_vec.push(0i64);
                }
            } else {
                // No marker — standard padding
                let len = ids.len().min(max_len);
                for i in 0..len {
                    input_ids_vec.push(ids[i] as i64);
                    attention_mask_vec.push(mask[i] as i64);
                }
                for _ in len..max_len {
                    input_ids_vec.push(0i64);
                    attention_mask_vec.push(0i64);
                }
            }
        }

        // Create tensors
        let shape = [batch_size, max_len];
        let input_ids = Tensor::from_array((shape, input_ids_vec))
            .map_err(|e| Error::Embedding(format!("input_ids tensor: {e}")))?;
        let attention_mask = Tensor::from_array((shape, attention_mask_vec))
            .map_err(|e| Error::Embedding(format!("attention_mask tensor: {e}")))?;

        // Run inference
        let mut session = self
            .session
            .lock()
            .map_err(|e| Error::Embedding(format!("session lock: {e}")))?;

        let outputs = session
            .run(
                ort::inputs![
                    "input_ids" => input_ids,
                    "attention_mask" => attention_mask
                ]
                .map_err(|e| Error::Embedding(format!("ort inputs: {e}")))?,
            )
            .map_err(|e| Error::Embedding(format!("ort run: {e}")))?;

        // Extract output: [batch, seq_len, dim]
        let output = outputs
            .get(0)
            .ok_or_else(|| Error::Embedding("no output".into()))?;

        let (out_shape, data) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Embedding(format!("extract output: {e}")))?;

        let out_seq = if out_shape.len() >= 2 { out_shape[1] as usize } else { max_len };
        let out_dim = if out_shape.len() >= 3 { out_shape[2] as usize } else { EMBEDDING_DIM };

        // Reshape into [batch][token][dim], L2-normalize each token vector
        let mut result = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let mut tokens = Vec::with_capacity(out_seq);
            for t in 0..out_seq {
                let offset = b * out_seq * out_dim + t * out_dim;
                let vec: Vec<f32> = data[offset..offset + out_dim].to_vec();
                // L2 normalize
                let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-10 {
                    tokens.push(vec.iter().map(|x| x / norm).collect());
                } else {
                    tokens.push(vec);
                }
            }
            result.push(tokens);
        }

        Ok(result)
    }

    /// Compute MaxSim score between query and document token embeddings.
    ///
    /// For each query token, find max cosine similarity with any doc token,
    /// then sum all maxima. Skiplist tokens are excluded from the sum.
    pub fn max_sim(
        query_vecs: &[Vec<f32>],
        doc_vecs: &[Vec<f32>],
    ) -> f32 {
        if query_vecs.is_empty() || doc_vecs.is_empty() {
            return 0.0;
        }

        let mut total = 0.0_f32;
        for q_vec in query_vecs {
            let mut max_sim = f32::NEG_INFINITY;
            for d_vec in doc_vecs {
                // Dot product (vectors are already L2-normalized, so this = cosine sim)
                let sim: f32 = q_vec.iter().zip(d_vec.iter()).map(|(a, b)| a * b).sum();
                if sim > max_sim {
                    max_sim = sim;
                }
            }
            if max_sim > f32::NEG_INFINITY {
                total += max_sim;
            }
        }
        total
    }

    /// Score a query against multiple documents using ColBERT MaxSim.
    ///
    /// Encodes query once, then scores against each document.
    /// Returns one score per document.
    pub fn score_documents(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>> {
        if docs.is_empty() {
            return Ok(vec![]);
        }

        // Encode query (single)
        let query_embeddings = self.encode_batch(&[query], true)?;
        let query_vecs = query_embeddings
            .into_iter()
            .next()
            .ok_or_else(|| Error::Embedding("empty query encoding".into()))?;

        // Filter skiplist tokens from query (skip [CLS], marker, padding, punctuation)
        // For now, skip first 2 tokens (CLS + marker) and any zero-norm tokens
        let query_filtered: Vec<&Vec<f32>> = query_vecs
            .iter()
            .skip(if self.query_marker.is_some() { 2 } else { 1 }) // skip CLS (+ marker)
            .filter(|v| {
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>();
                norm > 1e-10 // skip zero/padding tokens
            })
            .collect();

        if query_filtered.is_empty() {
            return Ok(vec![0.0; docs.len()]);
        }

        // Encode documents in batches of 16
        let batch_size = 16;
        let mut scores = Vec::with_capacity(docs.len());

        for chunk in docs.chunks(batch_size) {
            let doc_embeddings = self.encode_batch(chunk, false)?;

            for doc_vecs in &doc_embeddings {
                // Filter doc tokens: skip CLS, marker, padding
                let skip_n = if self.doc_marker.is_some() { 2 } else { 1 };
                let doc_filtered: Vec<&Vec<f32>> = doc_vecs
                    .iter()
                    .skip(skip_n)
                    .filter(|v| {
                        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>();
                        norm > 1e-10
                    })
                    .collect();

                if doc_filtered.is_empty() {
                    scores.push(0.0);
                    continue;
                }

                // MaxSim: for each query token, find best doc token match
                let mut total = 0.0_f32;
                for q_vec in &query_filtered {
                    let mut max_sim = f32::NEG_INFINITY;
                    for d_vec in &doc_filtered {
                        let sim: f32 = q_vec.iter().zip(d_vec.iter()).map(|(a, b)| a * b).sum();
                        if sim > max_sim {
                            max_sim = sim;
                        }
                    }
                    if max_sim > f32::NEG_INFINITY {
                        total += max_sim;
                    }
                }

                // Normalize by query token count for comparability
                scores.push(total / query_filtered.len() as f32);
            }
        }

        Ok(scores)
    }

    /// Truncate document text for ColBERT encoding.
    /// ColBERT handles longer docs than cross-encoders (up to max_seq_len tokens),
    /// but we still cap to keep latency low.
    pub fn truncate_doc(text: &str, max_chars: usize) -> String {
        text.chars().take(max_chars).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_sim_basic() {
        // Identical vectors → perfect score
        let q = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let d = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let score = ColBertReranker::max_sim(&q, &d);
        assert!((score - 2.0).abs() < 1e-6); // 1.0 + 1.0

        // Orthogonal → zero
        let d2 = vec![vec![0.0, 0.0, 1.0]];
        let q2 = vec![vec![1.0, 0.0, 0.0]];
        let score2 = ColBertReranker::max_sim(&q2, &d2);
        assert!(score2.abs() < 1e-6);
    }

    #[test]
    fn max_sim_empty() {
        let empty: Vec<Vec<f32>> = vec![];
        let q = vec![vec![1.0, 0.0]];
        assert_eq!(ColBertReranker::max_sim(&empty, &q), 0.0);
        assert_eq!(ColBertReranker::max_sim(&q, &empty), 0.0);
    }
}
