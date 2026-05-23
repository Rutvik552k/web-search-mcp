#[cfg(feature = "candle")]
mod inner {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use candle_transformers::models::bert::{BertModel, Config as BertConfig};
    use hf_hub::{api::sync::Api, Repo, RepoType};
    use std::sync::Mutex;
    use tokenizers::Tokenizer;
    use web_search_common::{Error, Result};

    /// Cross-encoder that scores (query, document) pairs using full attention.
    ///
    /// Used for:
    /// - Stage 2 reranking (ms-marco-MiniLM-L-6-v2)
    /// - Stage 4 NLI contradiction detection (nli-MiniLM2-L6-H768)
    ///
    /// Architecture: BertModel encoder + linear classification head.
    /// The classification head (classifier.weight, classifier.bias) is loaded
    /// separately since candle's BertModel doesn't include it.
    pub struct CrossEncoder {
        model: Mutex<BertModel>,
        tokenizer: Tokenizer,
        device: Device,
        model_id: String,
        max_seq_len: usize,
        num_labels: usize,
        /// Final projection: weight matrix [num_labels, hidden_size]
        classifier_weight: Tensor,
        /// Final projection: bias vector [num_labels]
        classifier_bias: Tensor,
        /// Optional dense pre-projection for MLP classification heads (NLI models).
        /// When present: logits = out_proj(tanh(dense(cls_output)))
        classifier_dense: Option<Tensor>,
        classifier_dense_bias: Option<Tensor>,
        /// BERT pooler: tanh(dense(cls_hidden_state)). Applied before classifier.
        /// BertForSequenceClassification uses this; candle's BertModel doesn't include it.
        pooler_weight: Option<Tensor>,
        pooler_bias: Option<Tensor>,
        /// Label index mapping for NLI: label_name → index
        id2label: std::collections::HashMap<usize, String>,
    }

    /// Score output from cross-encoder.
    #[derive(Debug, Clone)]
    pub struct CrossEncoderScore {
        /// Raw logits from the model
        pub logits: Vec<f32>,
        /// Primary score (sigmoid of first logit for reranking, or softmax for NLI)
        pub score: f32,
    }

    /// NLI classification result.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum NliLabel {
        Contradiction,
        Neutral,
        Entailment,
    }

    impl CrossEncoder {
        /// Load a cross-encoder model from HuggingFace Hub.
        ///
        /// For reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2` (1 output = relevance score)
        /// For NLI: `cross-encoder/nli-deberta-v3-small` (3 outputs = contradiction/neutral/entailment)
        pub fn new(model_id: &str) -> Result<Self> {
            let device = Device::Cpu;

            tracing::info!(model = model_id, "Loading cross-encoder model...");

            let api = Api::new().map_err(|e| Error::Embedding(format!("HF API: {e}")))?;
            let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

            let config_path = repo.get("config.json")
                .map_err(|e| Error::Embedding(format!("fetch config: {e}")))?;
            let tokenizer_path = repo.get("tokenizer.json")
                .map_err(|e| Error::Embedding(format!("fetch tokenizer: {e}")))?;

            // Try safetensors first, fall back to pytorch_model.bin conversion
            let weights_path = repo.get("model.safetensors")
                .or_else(|_| {
                    tracing::info!("model.safetensors not found, trying pytorch_model.bin");
                    repo.get("pytorch_model.bin")
                })
                .map_err(|e| Error::Embedding(format!("fetch weights: {e}")))?;

            // Parse config
            let config_str = std::fs::read_to_string(&config_path)
                .map_err(|e| Error::Embedding(format!("read config: {e}")))?;
            let bert_config: BertConfig = serde_json::from_str(&config_str)
                .map_err(|e| Error::Embedding(format!("parse config: {e}")))?;

            // Detect num_labels from config JSON
            let config_json: serde_json::Value = serde_json::from_str(&config_str)
                .map_err(|e| Error::Embedding(format!("parse config json: {e}")))?;
            let num_labels = config_json["num_labels"]
                .as_u64()
                .or_else(|| config_json["id2label"].as_object().map(|m| m.len() as u64))
                .unwrap_or(1) as usize;

            // Parse label mapping for NLI models (order varies between models)
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

            // Load weights — support both safetensors and pytorch format
            let weights_str = weights_path.to_string_lossy().to_string();
            let vb = if weights_str.ends_with(".safetensors") {
                unsafe {
                    VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                        .map_err(|e| Error::Embedding(format!("load safetensors: {e}")))?
                }
            } else {
                // pytorch_model.bin — use candle's pth loader
                VarBuilder::from_pth(&weights_path, DType::F32, &device)
                    .map_err(|e| Error::Embedding(format!("load pytorch weights: {e}")))?
            };

            // Clone VarBuilder before BertModel consumes it — we need it
            // to load the classification head weights separately.
            let vb_head = vb.clone();

            let model = BertModel::load(vb, &bert_config)
                .map_err(|e| Error::Embedding(format!("load model: {e}")))?;

            // Load classification head: classifier.weight [num_labels, hidden_size]
            // and classifier.bias [num_labels]. These are NOT part of candle's BertModel.
            let hidden_size = bert_config.hidden_size;
            // Load classification head. Two architectures:
            // 1. Single linear: classifier.weight [num_labels, hidden] (ms-marco reranker)
            // 2. Two-layer MLP: classifier.dense [hidden, hidden] + classifier.out_proj [num_labels, hidden] (NLI models)
            //
            // For architecture 2, we compose into a single effective weight matrix:
            // out_proj @ tanh(dense @ x + dense_bias) + out_proj_bias
            // But for simplicity, we just load the out_proj (final projection) and
            // apply dense+tanh as a pre-step in scoring.
            let (classifier_weight, classifier_bias, classifier_dense, classifier_dense_bias) = {
                let cls = vb_head.pp("classifier");

                // Try single-linear first: classifier.weight [num_labels, hidden]
                let single_w = vb_head.get((num_labels, hidden_size), "classifier.weight");
                let single_b = vb_head.get(num_labels, "classifier.bias");

                if let (Ok(w), Ok(b)) = (single_w, single_b) {
                    tracing::info!("Classification head loaded (single linear): [{num_labels}, {hidden_size}]");
                    (w, b, None, None)
                } else {
                    // Try two-layer MLP: classifier.dense + classifier.out_proj
                    let dense_w = cls.get((hidden_size, hidden_size), "dense.weight");
                    let dense_b = cls.get(hidden_size, "dense.bias");
                    let proj_w = cls.get((num_labels, hidden_size), "out_proj.weight");
                    let proj_b = cls.get(num_labels, "out_proj.bias");

                    match (dense_w, dense_b, proj_w, proj_b) {
                        (Ok(dw), Ok(db), Ok(pw), Ok(pb)) => {
                            tracing::info!(
                                "Classification head loaded (MLP): dense [{hidden_size}, {hidden_size}] + out_proj [{num_labels}, {hidden_size}]"
                            );
                            (pw, pb, Some(dw), Some(db))
                        }
                        _ => {
                            tracing::warn!(
                                "Classification head not found — using random init. Scores WILL be degraded."
                            );
                            let w = Tensor::randn(0f32, 0.02, (num_labels, hidden_size), &device)
                                .map_err(|e| Error::Embedding(format!("init weight: {e}")))?;
                            let b = Tensor::zeros(num_labels, DType::F32, &device)
                                .map_err(|e| Error::Embedding(format!("init bias: {e}")))?;
                            (w, b, None, None)
                        }
                    }
                }
            };

            // Load BERT pooler: bert.pooler.dense.weight [hidden, hidden] + bias.
            // BertForSequenceClassification applies pooler(cls) before classifier.
            // Candle's BertModel doesn't include it.
            let pooler = vb_head.pp("bert").pp("pooler").pp("dense");
            let (pooler_weight, pooler_bias) = match (
                pooler.get((hidden_size, hidden_size), "weight"),
                pooler.get(hidden_size, "bias"),
            ) {
                (Ok(w), Ok(b)) => {
                    tracing::info!("Pooler loaded: [{hidden_size}, {hidden_size}]");
                    (Some(w), Some(b))
                }
                _ => {
                    tracing::debug!("No pooler weights found (may be intentional)");
                    (None, None)
                }
            };

            tracing::info!(
                model = model_id,
                num_labels,
                max_seq = bert_config.max_position_embeddings,
                has_pooler = pooler_weight.is_some(),
                "Cross-encoder loaded with classification head"
            );

            Ok(Self {
                model: Mutex::new(model),
                tokenizer,
                device,
                model_id: model_id.to_string(),
                max_seq_len: bert_config.max_position_embeddings,
                num_labels,
                classifier_weight,
                classifier_bias,
                classifier_dense,
                classifier_dense_bias,
                pooler_weight,
                pooler_bias,
                id2label,
            })
        }

        /// Score a batch of (text_a, text_b) pairs.
        ///
        /// For reranking: text_a = query, text_b = document
        /// For NLI: text_a = premise, text_b = hypothesis
        pub fn score_pairs(&self, pairs: &[(&str, &str)]) -> Result<Vec<CrossEncoderScore>> {
            if pairs.is_empty() {
                return Ok(vec![]);
            }

            let batch_size = pairs.len();

            // Tokenize pairs — cross-encoder uses [CLS] text_a [SEP] text_b [SEP]
            let encodings: Vec<_> = pairs.iter().map(|(a, b)| {
                self.tokenizer.encode((*a, *b), true)
                    .map_err(|e| Error::Embedding(format!("tokenize pair: {e}")))
            }).collect::<Result<Vec<_>>>()?;

            let max_len = encodings.iter()
                .map(|e| e.get_ids().len())
                .max()
                .unwrap_or(0)
                .min(self.max_seq_len);

            // Build padded tensors
            let mut input_ids = Vec::with_capacity(batch_size * max_len);
            let mut token_type_ids = Vec::with_capacity(batch_size * max_len);

            for enc in &encodings {
                let ids = enc.get_ids();
                let types = enc.get_type_ids();
                let len = ids.len().min(max_len);

                input_ids.extend_from_slice(&ids[..len]);
                token_type_ids.extend_from_slice(&types[..len]);

                for _ in len..max_len {
                    input_ids.push(0);
                    token_type_ids.push(0);
                }
            }

            let input_ids_t = Tensor::new(input_ids.as_slice(), &self.device)
                .and_then(|t| t.reshape(&[batch_size, max_len]))
                .map_err(|e| Error::Embedding(format!("input tensor: {e}")))?;

            let token_type_ids_t = Tensor::new(token_type_ids.as_slice(), &self.device)
                .and_then(|t| t.reshape(&[batch_size, max_len]))
                .map_err(|e| Error::Embedding(format!("type tensor: {e}")))?;

            // Forward pass
            let model = self.model.lock()
                .map_err(|e| Error::Embedding(format!("model lock: {e}")))?;

            let output = model.forward(&input_ids_t, &token_type_ids_t, None)
                .map_err(|e| Error::Embedding(format!("forward: {e}")))?;

            // Extract [CLS] token hidden state: [batch, hidden_size]
            let cls_raw = output
                .narrow(1, 0, 1)
                .and_then(|t| t.squeeze(1))
                .map_err(|e| Error::Embedding(format!("cls extract: {e}")))?;

            // Apply pooler if present: pooled = tanh(dense(cls_raw))
            // BertForSequenceClassification applies this before the classifier.
            let cls_output = if let (Some(pw), Some(pb)) =
                (&self.pooler_weight, &self.pooler_bias)
            {
                let pw_t = pw.t()
                    .map_err(|e| Error::Embedding(format!("transpose pooler: {e}")))?;
                cls_raw
                    .matmul(&pw_t)
                    .and_then(|t| t.broadcast_add(pb))
                    .and_then(|t| t.tanh())
                    .map_err(|e| Error::Embedding(format!("pooler: {e}")))?
            } else {
                cls_raw
            };

            // Apply classification head.
            // Single linear: logits = cls @ weight^T + bias
            // MLP (NLI): logits = out_proj(tanh(dense(cls) + dense_bias))
            let head_input = if let (Some(dense_w), Some(dense_b)) =
                (&self.classifier_dense, &self.classifier_dense_bias)
            {
                // MLP path: dense projection + tanh activation
                let dense_wt = dense_w.t()
                    .map_err(|e| Error::Embedding(format!("transpose dense: {e}")))?;
                cls_output
                    .matmul(&dense_wt)
                    .and_then(|t| t.broadcast_add(dense_b))
                    .and_then(|t| t.tanh())
                    .map_err(|e| Error::Embedding(format!("dense layer: {e}")))?
            } else {
                cls_output
            };

            let weight_t = self.classifier_weight.t()
                .map_err(|e| Error::Embedding(format!("transpose classifier weight: {e}")))?;
            let logits_tensor = head_input
                .matmul(&weight_t)
                .and_then(|t| t.broadcast_add(&self.classifier_bias))
                .map_err(|e| Error::Embedding(format!("classifier head: {e}")))?;

            // Convert to scores
            let mut results = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let logits: Vec<f32> = logits_tensor.get(i)
                    .and_then(|t| t.to_vec1::<f32>())
                    .unwrap_or_else(|_| vec![0.0; self.num_labels]);

                let score = if self.num_labels == 1 {
                    // Reranking: use raw logit directly for ranking.
                    // ms-marco models use Identity activation (not sigmoid).
                    // Raw logits give better discrimination (e.g., +8.6 vs -4.3).
                    // Normalize to [0,1] via sigmoid only for display/caching.
                    sigmoid(logits[0])
                } else {
                    // NLI: softmax → entailment probability
                    let sm = softmax(&logits);
                    sm.get(2).copied().unwrap_or(0.0)
                };

                results.push(CrossEncoderScore { logits, score });
            }

            Ok(results)
        }

        /// Classify NLI label for a single pair.
        ///
        /// Uses model's id2label mapping to correctly identify which index
        /// corresponds to contradiction/entailment/neutral (varies between models).
        pub fn classify_nli(&self, premise: &str, hypothesis: &str) -> Result<(NliLabel, f32)> {
            let scores = self.score_pairs(&[(premise, hypothesis)])?;
            let score = scores.into_iter().next()
                .ok_or_else(|| Error::Embedding("empty NLI result".into()))?;

            let logits = &score.logits;
            if logits.len() < 3 {
                return Ok((NliLabel::Neutral, 0.5));
            }

            let sm = softmax(logits);

            // Find which index maps to which label using id2label
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

            let (label, conf) = if sm.len() > contradiction_idx.max(entailment_idx).max(neutral_idx) {
                let c_score = sm[contradiction_idx];
                let e_score = sm[entailment_idx];
                let n_score = sm[neutral_idx];

                if c_score > e_score && c_score > n_score {
                    (NliLabel::Contradiction, c_score)
                } else if e_score > c_score && e_score > n_score {
                    (NliLabel::Entailment, e_score)
                } else {
                    (NliLabel::Neutral, n_score)
                }
            } else {
                // Fallback: hardcoded order
                if sm[0] > sm[1] && sm[0] > sm[2] {
                    (NliLabel::Contradiction, sm[0])
                } else if sm[2] > sm[0] && sm[2] > sm[1] {
                    (NliLabel::Entailment, sm[2])
                } else {
                    (NliLabel::Neutral, sm[1])
                }
            };

            Ok((label, conf))
        }

        /// Classify NLI label from pre-computed logits (no model inference).
        ///
        /// Same label resolution as `classify_nli` but skips the forward pass.
        /// Use after `score_pairs()` to avoid double inference.
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
                let c_score = sm[contradiction_idx];
                let e_score = sm[entailment_idx];
                let n_score = sm[neutral_idx];

                if c_score > e_score && c_score > n_score {
                    (NliLabel::Contradiction, c_score)
                } else if e_score > c_score && e_score > n_score {
                    (NliLabel::Entailment, e_score)
                } else {
                    (NliLabel::Neutral, n_score)
                }
            } else {
                if sm[0] > sm[1] && sm[0] > sm[2] {
                    (NliLabel::Contradiction, sm[0])
                } else if sm[2] > sm[0] && sm[2] > sm[1] {
                    (NliLabel::Entailment, sm[2])
                } else {
                    (NliLabel::Neutral, sm[1])
                }
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
}

#[cfg(feature = "candle")]
pub use inner::*;

/// Stub for when candle is not available.
#[cfg(not(feature = "candle"))]
pub mod stub {
    use web_search_common::{Error, Result};

    pub struct CrossEncoder;

    #[derive(Debug, Clone)]
    pub struct CrossEncoderScore {
        pub logits: Vec<f32>,
        pub score: f32,
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum NliLabel {
        Contradiction,
        Neutral,
        Entailment,
    }

    impl CrossEncoder {
        pub fn new(_model_id: &str) -> Result<Self> {
            Err(Error::Embedding("Cross-encoder requires 'candle' feature".into()))
        }

        pub fn score_pairs(&self, _pairs: &[(&str, &str)]) -> Result<Vec<CrossEncoderScore>> {
            Ok(vec![])
        }

        pub fn classify_nli(&self, _premise: &str, _hypothesis: &str) -> Result<(NliLabel, f32)> {
            Ok((NliLabel::Neutral, 0.5))
        }

        pub fn classify_from_logits(&self, _logits: &[f32]) -> (NliLabel, f32) {
            (NliLabel::Neutral, 0.5)
        }

        pub fn model_id(&self) -> &str { "none" }
    }
}

#[cfg(not(feature = "candle"))]
pub use stub::*;
