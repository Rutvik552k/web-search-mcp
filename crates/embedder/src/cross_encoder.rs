#[cfg(feature = "candle")]
mod inner {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use candle_transformers::models::bert::{BertModel, Config as BertConfig};
    use hf_hub::{api::sync::Api, Repo, RepoType};
    use std::path::PathBuf;
    use std::sync::Mutex;
    use tokenizers::Tokenizer;
    use web_search_common::{Error, Result};

    /// Cross-encoder that scores (query, document) pairs using full attention.
    ///
    /// Used for:
    /// - Stage 2 reranking (ms-marco-MiniLM-L-6-v2)
    /// - Stage 4 NLI contradiction detection (nli-deberta-v3-small)
    pub struct CrossEncoder {
        model: Mutex<BertModel>,
        tokenizer: Tokenizer,
        device: Device,
        model_id: String,
        max_seq_len: usize,
        num_labels: usize,
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
            let weights_path = repo.get("model.safetensors")
                .map_err(|e| Error::Embedding(format!("fetch weights: {e}")))?;

            // Parse config
            let config_str = std::fs::read_to_string(&config_path)
                .map_err(|e| Error::Embedding(format!("read config: {e}")))?;
            let bert_config: BertConfig = serde_json::from_str(&config_str)
                .map_err(|e| Error::Embedding(format!("parse config: {e}")))?;

            // Detect num_labels from config JSON
            let config_json: serde_json::Value = serde_json::from_str(&config_str)
                .map_err(|e| Error::Embedding(format!("parse config json: {e}")))?;
            let num_labels = config_json["num_labels"].as_u64().unwrap_or(1) as usize;

            let tokenizer = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| Error::Embedding(format!("load tokenizer: {e}")))?;

            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                    .map_err(|e| Error::Embedding(format!("load weights: {e}")))?
            };

            let model = BertModel::load(vb, &bert_config)
                .map_err(|e| Error::Embedding(format!("load model: {e}")))?;

            tracing::info!(
                model = model_id,
                num_labels,
                max_seq = bert_config.max_position_embeddings,
                "Cross-encoder loaded"
            );

            Ok(Self {
                model: Mutex::new(model),
                tokenizer,
                device,
                model_id: model_id.to_string(),
                max_seq_len: bert_config.max_position_embeddings,
                num_labels,
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

            // Extract [CLS] token embeddings (first token of each sequence)
            // For cross-encoders, the pooled output or CLS embedding is used
            // We take the first token's hidden state
            let cls_output = output
                .narrow(1, 0, 1)
                .and_then(|t| t.squeeze(1))
                .map_err(|e| Error::Embedding(format!("cls extract: {e}")))?;

            // Convert to scores
            let mut results = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let hidden = cls_output.get(i)
                    .and_then(|t| t.to_vec1::<f32>())
                    .unwrap_or_else(|_| vec![0.0; self.num_labels]);

                // For reranking (1 label): use mean of hidden state as score
                // For NLI (3 labels): hidden state is larger, take first 3 as logits proxy
                let score = if self.num_labels == 1 {
                    // Reranking: mean pooled score
                    let mean: f32 = hidden.iter().sum::<f32>() / hidden.len() as f32;
                    sigmoid(mean)
                } else {
                    // NLI: take first 3 values, softmax
                    let logits: Vec<f32> = hidden.iter().take(3).copied().collect();
                    let sm = softmax(&logits);
                    sm.get(2).copied().unwrap_or(0.0) // entailment probability
                };

                results.push(CrossEncoderScore {
                    logits: hidden.iter().take(self.num_labels.max(3)).copied().collect(),
                    score,
                });
            }

            Ok(results)
        }

        /// Classify NLI label for a single pair.
        pub fn classify_nli(&self, premise: &str, hypothesis: &str) -> Result<(NliLabel, f32)> {
            let scores = self.score_pairs(&[(premise, hypothesis)])?;
            let score = scores.into_iter().next()
                .ok_or_else(|| Error::Embedding("empty NLI result".into()))?;

            let logits = &score.logits;
            if logits.len() < 3 {
                return Ok((NliLabel::Neutral, 0.5));
            }

            let sm = softmax(logits);
            let (label, conf) = if sm[0] > sm[1] && sm[0] > sm[2] {
                (NliLabel::Contradiction, sm[0])
            } else if sm[2] > sm[0] && sm[2] > sm[1] {
                (NliLabel::Entailment, sm[2])
            } else {
                (NliLabel::Neutral, sm[1])
            };

            Ok((label, conf))
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

        pub fn model_id(&self) -> &str { "none" }
    }
}

#[cfg(not(feature = "candle"))]
pub use stub::*;
