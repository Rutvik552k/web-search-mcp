//! ONNX Runtime cross-encoder for fast reranking and NLI.
//!
//! 5-10x faster than Candle FP32 on CPU due to graph optimization,
//! operator fusion, and INT8 quantization (23 MB vs 91 MB).
//!
//! Model priority: quantized INT8 → optimized FP32 → baseline FP32.
//! Gated behind the `onnx` feature flag.

use hf_hub::{api::sync::Api, Repo, RepoType};
use ort::session::Session;
use ort::value::Tensor;
use std::sync::Mutex;
use tokenizers::Tokenizer;
use web_search_common::{Error, Result};

use crate::cross_encoder::{CrossEncoderScore, NliLabel};

/// ONNX Runtime cross-encoder. Same interface as candle CrossEncoder
/// but backed by ONNX Runtime with graph optimizations and INT8 quantization.
pub struct OnnxCrossEncoder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    model_id: String,
    max_seq_len: usize,
    num_labels: usize,
    id2label: std::collections::HashMap<usize, String>,
}

/// ONNX model variants in order of preference (fastest first).
const ONNX_MODEL_PATHS: &[&str] = &[
    // INT8 quantized — 4x smaller, fastest on x86-64
    "onnx/model_quint8_avx2.onnx",
    // Optimized FP32 (graph fusion, constant folding)
    "onnx/model_O3.onnx",
    "onnx/model_O2.onnx",
    // Baseline FP32
    "onnx/model.onnx",
    // Root-level fallback
    "model.onnx",
];

impl OnnxCrossEncoder {
    /// Load cross-encoder from ONNX model on HuggingFace.
    ///
    /// Tries quantized INT8 first (4x smaller, fastest), falls back through
    /// optimized FP32 variants, then baseline FP32.
    pub fn new(model_id: &str) -> Result<Self> {
        tracing::info!(model = model_id, "Loading ONNX cross-encoder...");

        let api = Api::new().map_err(|e| Error::Embedding(format!("HF API: {e}")))?;
        let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

        // Download config + tokenizer
        let config_path = repo
            .get("config.json")
            .map_err(|e| Error::Embedding(format!("fetch config: {e}")))?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| Error::Embedding(format!("fetch tokenizer: {e}")))?;

        // Try ONNX model paths in priority order
        let mut onnx_path = None;
        let mut onnx_variant = "";
        for path in ONNX_MODEL_PATHS {
            match repo.get(path) {
                Ok(p) => {
                    onnx_variant = path;
                    onnx_path = Some(p);
                    break;
                }
                Err(_) => continue,
            }
        }
        let onnx_path = onnx_path.ok_or_else(|| {
            Error::Embedding(format!(
                "No ONNX model found for {model_id}. \
                 Export with: optimum-cli export onnx --model {model_id} onnx_output/"
            ))
        })?;

        // Parse config
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| Error::Embedding(format!("read config: {e}")))?;
        let config_json: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| Error::Embedding(format!("parse config: {e}")))?;

        let num_labels = config_json["num_labels"]
            .as_u64()
            .or_else(|| config_json["id2label"].as_object().map(|m| m.len() as u64))
            .unwrap_or(1) as usize;

        let max_seq_len = config_json["max_position_embeddings"]
            .as_u64()
            .unwrap_or(512) as usize;

        let id2label: std::collections::HashMap<usize, String> = config_json["id2label"]
            .as_object()
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| {
                        let idx = k.parse::<usize>().ok()?;
                        let label = v.as_str()?.to_lowercase();
                        Some((idx, label))
                    })
                    .collect()
            })
            .unwrap_or_default();

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Embedding(format!("load tokenizer: {e}")))?;

        // Build ONNX session with maximum optimization
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        let session = Session::builder()
            .map_err(|e| Error::Embedding(format!("ort session builder: {e}")))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| Error::Embedding(format!("ort optimization: {e}")))?
            .with_intra_threads(num_threads)
            .map_err(|e| Error::Embedding(format!("ort threads: {e}")))?
            .commit_from_file(&onnx_path)
            .map_err(|e| Error::Embedding(format!("ort load model: {e}")))?;

        tracing::info!(
            model = model_id,
            variant = onnx_variant,
            num_labels,
            threads = num_threads,
            "ONNX cross-encoder loaded"
        );

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            model_id: model_id.to_string(),
            max_seq_len,
            num_labels,
            id2label,
        })
    }

    /// Score a batch of (text_a, text_b) pairs.
    ///
    /// ONNX model includes classification head — outputs logits directly.
    /// No manual classifier.weight projection needed (unlike candle path).
    pub fn score_pairs(&self, pairs: &[(&str, &str)]) -> Result<Vec<CrossEncoderScore>> {
        if pairs.is_empty() {
            return Ok(vec![]);
        }

        let batch_size = pairs.len();

        // Tokenize pairs — cross-encoder uses [CLS] text_a [SEP] text_b [SEP]
        let encodings: Vec<_> = pairs
            .iter()
            .map(|(a, b)| {
                self.tokenizer
                    .encode((*a, *b), true)
                    .map_err(|e| Error::Embedding(format!("tokenize pair: {e}")))
            })
            .collect::<Result<Vec<_>>>()?;

        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0)
            .min(self.max_seq_len);

        // Build padded input vectors
        let mut input_ids_vec = Vec::with_capacity(batch_size * max_len);
        let mut attention_mask_vec = Vec::with_capacity(batch_size * max_len);
        let mut token_type_ids_vec = Vec::with_capacity(batch_size * max_len);

        for enc in &encodings {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let types = enc.get_type_ids();
            let len = ids.len().min(max_len);

            for i in 0..len {
                input_ids_vec.push(ids[i] as i64);
                attention_mask_vec.push(mask[i] as i64);
                token_type_ids_vec.push(types[i] as i64);
            }
            for _ in len..max_len {
                input_ids_vec.push(0i64);
                attention_mask_vec.push(0i64);
                token_type_ids_vec.push(0i64);
            }
        }

        // Create ort Tensors using Tensor::from_array((shape, data))
        let shape = [batch_size, max_len];
        let input_ids = Tensor::from_array((shape, input_ids_vec))
            .map_err(|e| Error::Embedding(format!("input_ids tensor: {e}")))?;
        let attention_mask = Tensor::from_array((shape, attention_mask_vec))
            .map_err(|e| Error::Embedding(format!("attention_mask tensor: {e}")))?;
        let token_type_ids = Tensor::from_array((shape, token_type_ids_vec))
            .map_err(|e| Error::Embedding(format!("token_type_ids tensor: {e}")))?;

        // Run inference
        let mut session = self
            .session
            .lock()
            .map_err(|e| Error::Embedding(format!("session lock: {e}")))?;

        let outputs = session
            .run(
                ort::inputs![
                    "input_ids" => input_ids,
                    "attention_mask" => attention_mask,
                    "token_type_ids" => token_type_ids
                ]
                .map_err(|e| Error::Embedding(format!("ort inputs: {e}")))?,
            )
            .map_err(|e| Error::Embedding(format!("ort run: {e}")))?;

        // Extract logits — ONNX model outputs [batch, num_labels] directly
        let logits_value = outputs
            .get(0)
            .ok_or_else(|| Error::Embedding("no output from ONNX model".into()))?;

        let (shape, logits_data) = logits_value
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Embedding(format!("extract logits: {e}")))?;

        let out_cols = if shape.len() >= 2 { shape[1] as usize } else { self.num_labels };

        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let offset = i * out_cols;
            let logits: Vec<f32> = logits_data[offset..offset + out_cols].to_vec();

            let score = if out_cols == 1 {
                sigmoid(logits[0])
            } else {
                let sm = softmax(&logits);
                sm.get(2).copied().unwrap_or(0.0)
            };

            results.push(CrossEncoderScore { logits, score });
        }

        Ok(results)
    }

    /// Classify NLI label from pre-computed logits.
    pub fn classify_from_logits(&self, logits: &[f32]) -> (NliLabel, f32) {
        if logits.len() < 3 {
            return (NliLabel::Neutral, 0.5);
        }

        let sm = softmax(logits);

        let mut contradiction_idx = 0;
        let mut entailment_idx = 2;
        let mut neutral_idx = 1;

        for (idx, label) in &self.id2label {
            match label.as_str() {
                "contradiction" => contradiction_idx = *idx,
                "entailment" => entailment_idx = *idx,
                "neutral" => neutral_idx = *idx,
                _ => {}
            }
        }

        if sm.len() > contradiction_idx.max(entailment_idx).max(neutral_idx) {
            let c = sm[contradiction_idx];
            let e = sm[entailment_idx];
            let n = sm[neutral_idx];

            if c > e && c > n {
                (NliLabel::Contradiction, c)
            } else if e > c && e > n {
                (NliLabel::Entailment, e)
            } else {
                (NliLabel::Neutral, n)
            }
        } else {
            (NliLabel::Neutral, sm[1])
        }
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}
