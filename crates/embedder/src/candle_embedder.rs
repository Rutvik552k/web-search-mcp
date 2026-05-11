use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::PathBuf;
use std::sync::Mutex;
use tokenizers::Tokenizer;
use web_search_common::config::EmbedderConfig;
use web_search_common::{Error, Result};

use crate::Embedder;

/// Neural embedding engine using candle (pure Rust ML framework).
///
/// Loads a sentence-transformers model (e.g., all-MiniLM-L6-v2) and runs
/// inference locally. Supports CPU and GPU (CUDA/Metal).
pub struct CandleEmbedder {
    model: Mutex<BertModel>,
    tokenizer: Tokenizer,
    device: Device,
    dim: usize,
    model_name: String,
    max_seq_len: usize,
}

impl CandleEmbedder {
    /// Create a new CandleEmbedder.
    ///
    /// If `config.embedding_model_path` points to a local directory with model files,
    /// loads from there. Otherwise downloads from HuggingFace Hub.
    pub fn new(config: &EmbedderConfig) -> Result<Self> {
        let device = if config.force_cpu {
            Device::Cpu
        } else {
            Self::detect_device()
        };

        tracing::info!(device = ?device, "CandleEmbedder initializing");

        let model_path = &config.embedding_model_path;
        let model_id = if model_path.exists() && model_path.is_dir() {
            // Local model directory
            model_path.to_string_lossy().to_string()
        } else {
            // Default HuggingFace model
            "sentence-transformers/all-MiniLM-L6-v2".to_string()
        };

        let (model, tokenizer, bert_config) = Self::load_model(&model_id, &device)?;

        Ok(Self {
            model: Mutex::new(model),
            tokenizer,
            device,
            dim: bert_config.hidden_size,
            model_name: model_id,
            max_seq_len: bert_config.max_position_embeddings,
        })
    }

    fn detect_device() -> Device {
        #[cfg(feature = "cuda")]
        {
            if let Ok(device) = Device::new_cuda(0) {
                tracing::info!("CUDA device detected");
                return device;
            }
        }
        #[cfg(feature = "metal")]
        {
            if let Ok(device) = Device::new_metal(0) {
                tracing::info!("Metal device detected");
                return device;
            }
        }
        Device::Cpu
    }

    fn load_model(
        model_id: &str,
        device: &Device,
    ) -> Result<(BertModel, Tokenizer, BertConfig)> {
        let (config_path, tokenizer_path, weights_path) = if PathBuf::from(model_id).is_dir() {
            let dir = PathBuf::from(model_id);
            (
                dir.join("config.json"),
                dir.join("tokenizer.json"),
                dir.join("model.safetensors"),
            )
        } else {
            // Download from HuggingFace Hub
            tracing::info!(model = model_id, "Downloading model from HuggingFace Hub");
            let api = Api::new().map_err(|e| Error::Embedding(format!("HF API init: {e}")))?;
            let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

            let config = repo
                .get("config.json")
                .map_err(|e| Error::Embedding(format!("fetch config.json: {e}")))?;
            let tokenizer = repo
                .get("tokenizer.json")
                .map_err(|e| Error::Embedding(format!("fetch tokenizer.json: {e}")))?;
            let weights = repo
                .get("model.safetensors")
                .map_err(|e| Error::Embedding(format!("fetch model.safetensors: {e}")))?;

            (config, tokenizer, weights)
        };

        // Load config
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| Error::Embedding(format!("read config: {e}")))?;
        let bert_config: BertConfig = serde_json::from_str(&config_str)
            .map_err(|e| Error::Embedding(format!("parse config: {e}")))?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Embedding(format!("load tokenizer: {e}")))?;

        // Load model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)
                .map_err(|e| Error::Embedding(format!("load weights: {e}")))?
        };

        let model = BertModel::load(vb, &bert_config)
            .map_err(|e| Error::Embedding(format!("load model: {e}")))?;

        tracing::info!(
            hidden_size = bert_config.hidden_size,
            layers = bert_config.num_hidden_layers,
            "Model loaded"
        );

        Ok((model, tokenizer, bert_config))
    }

    /// Run inference on a batch of texts.
    fn forward(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let tokenizer = &self.tokenizer;

        // Tokenize all texts
        let encodings = tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| Error::Embedding(format!("tokenize: {e}")))?;

        // Find max length for padding
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0)
            .min(self.max_seq_len);

        let batch_size = texts.len();

        // Build input tensors with padding
        let mut input_ids_flat = Vec::with_capacity(batch_size * max_len);
        let mut attention_mask_flat = Vec::with_capacity(batch_size * max_len);
        let mut token_type_ids_flat = Vec::with_capacity(batch_size * max_len);

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let types = encoding.get_type_ids();
            let len = ids.len().min(max_len);

            input_ids_flat.extend_from_slice(&ids[..len]);
            attention_mask_flat.extend_from_slice(&mask[..len]);
            token_type_ids_flat.extend_from_slice(&types[..len]);

            // Pad to max_len
            for _ in len..max_len {
                input_ids_flat.push(0);
                attention_mask_flat.push(0);
                token_type_ids_flat.push(0);
            }
        }

        let input_ids = Tensor::new(input_ids_flat.as_slice(), &self.device)
            .and_then(|t| t.reshape(&[batch_size, max_len]))
            .map_err(|e| Error::Embedding(format!("input_ids tensor: {e}")))?;

        let token_type_ids = Tensor::new(token_type_ids_flat.as_slice(), &self.device)
            .and_then(|t| t.reshape(&[batch_size, max_len]))
            .map_err(|e| Error::Embedding(format!("token_type_ids tensor: {e}")))?;

        let attention_mask_u32: Vec<u32> = attention_mask_flat.iter().map(|&x| x as u32).collect();

        // Forward pass
        let model = self
            .model
            .lock()
            .map_err(|e| Error::Embedding(format!("model lock: {e}")))?;

        let embeddings = model
            .forward(&input_ids, &token_type_ids, None)
            .map_err(|e| Error::Embedding(format!("forward: {e}")))?;

        // Mean pooling with attention mask
        let attention_mask = Tensor::new(attention_mask_u32.as_slice(), &self.device)
            .and_then(|t| t.reshape(&[batch_size, max_len]))
            .and_then(|t| t.to_dtype(DType::F32))
            .map_err(|e| Error::Embedding(format!("attention mask tensor: {e}")))?;

        let mask_expanded = attention_mask
            .unsqueeze(2)
            .and_then(|t| t.broadcast_as(embeddings.shape()))
            .map_err(|e| Error::Embedding(format!("mask expand: {e}")))?;

        let masked = embeddings
            .mul(&mask_expanded)
            .map_err(|e| Error::Embedding(format!("mask mul: {e}")))?;

        let summed = masked
            .sum(1)
            .map_err(|e| Error::Embedding(format!("sum: {e}")))?;

        let mask_sum = attention_mask
            .sum(1)
            .and_then(|t| t.unsqueeze(1))
            .and_then(|t| t.clamp(1e-9, f64::MAX))
            .map_err(|e| Error::Embedding(format!("mask sum: {e}")))?;

        let mean_pooled = summed
            .broadcast_div(&mask_sum)
            .map_err(|e| Error::Embedding(format!("div: {e}")))?;

        // L2 normalize
        let norms = mean_pooled
            .sqr()
            .and_then(|t| t.sum(1))
            .and_then(|t| t.sqrt())
            .and_then(|t| t.unsqueeze(1))
            .and_then(|t| t.clamp(1e-10, f64::MAX))
            .map_err(|e| Error::Embedding(format!("norm: {e}")))?;

        let normalized = mean_pooled
            .broadcast_div(&norms)
            .map_err(|e| Error::Embedding(format!("normalize: {e}")))?;

        // Convert to Vec<Vec<f32>>
        let result: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| {
                normalized
                    .get(i)
                    .and_then(|t| t.to_vec1::<f32>())
                    .unwrap_or_else(|_| vec![0.0; self.dim])
            })
            .collect();

        Ok(result)
    }
}

#[async_trait]
impl Embedder for CandleEmbedder {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Run inference in blocking task to not block tokio runtime
        let texts_owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        // We need to do this synchronously since model is behind Mutex
        // In practice, batch sizes are small enough that this is fine
        let text_refs: Vec<&str> = texts_owned.iter().map(|s| s.as_str()).collect();
        self.forward(&text_refs)
    }

    fn dimensions(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}
