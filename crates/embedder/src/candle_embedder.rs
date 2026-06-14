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

/// Lowercase hex encoding of a byte slice (avoids a `hex` crate dep).
fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

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

        // Resolution order for where to load model files from (ADR 0004 §6.2/A.5):
        //   1. `models_dir` (config-controlled, pre-staged, air-gapped) — loaded
        //      from THERE and CHECKSUM-VERIFIED before load (A.5, fail-closed).
        //   2. `embedding_model_path` if it is an existing local dir.
        //   3. HF Hub auto-fetch (hf_hub provides its own integrity).
        let model_id = if let Some(dir) = config.models_dir.as_ref() {
            if dir.exists() && dir.is_dir() {
                dir.to_string_lossy().to_string()
            } else {
                // models_dir was named but is absent — fail closed rather than
                // silently fetching unverified weights from the network.
                return Err(Error::Embedding(format!(
                    "models_dir is set but does not exist or is not a directory: {}",
                    dir.display()
                )));
            }
        } else {
            let model_path = &config.embedding_model_path;
            if model_path.exists() && model_path.is_dir() {
                model_path.to_string_lossy().to_string()
            } else {
                "sentence-transformers/all-MiniLM-L6-v2".to_string()
            }
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
            let config_path = dir.join("config.json");
            let tokenizer_path = dir.join("tokenizer.json");
            let weights_path = dir.join("model.safetensors");

            // MANDATORY checksum verification before loading weights from a
            // config-controlled directory (ADR 0004 A.5; ml-inference WEIGHT
            // INTEGRITY). Fail CLOSED on missing manifest or any mismatch — do
            // not load unverified weights into the model.
            Self::verify_checksums(&dir, &[&config_path, &tokenizer_path, &weights_path])?;

            // RISK-ACCEPT #5 (security audit, ADR 0004 A.5): there is a TOCTOU
            // window between this checksum verification and the mmap/read of the
            // same files below — a local actor with write access to `models_dir`
            // could swap a file after the hash passes. Accepted under the
            // operator-trusted `models_dir` model: the directory is operator-staged
            // and on a trust boundary the operator already controls (same trust as
            // the binary itself); closing the window would require holding an
            // fd + hashing the fd, which is out of scope here. No behavior change.

            (config_path, tokenizer_path, weights_path)
        } else {
            // Download from HuggingFace Hub. First-run fetch is ~90MB and can
            // take ~30s on a cold cache — emit ONE explicit one-time INFO so it
            // is not mistaken for a hang (ADR 0004 §6.2).
            tracing::info!(
                model = model_id,
                "fetching embedding model (~90MB) from HuggingFace, one-time — \
                 subsequent runs use the local cache; this may take ~30s on first start"
            );
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

    /// MANDATORY SHA-256 verification of pre-staged model files in a
    /// config-controlled `models_dir` (ADR 0004 A.5).
    ///
    /// Requires a manifest `checksums.sha256` in `dir`, listing one entry per
    /// expected file in the common `"<hex>  <filename>"` format (sha256sum-style;
    /// also tolerates `"<filename>=<hex>"`). For each file passed in `files` we
    /// compute its SHA-256 and compare to the manifest entry keyed by file name.
    ///
    /// Fails CLOSED: a missing manifest, a missing entry, or ANY mismatch returns
    /// `Err` and the weights are NOT loaded (ml-inference WEIGHT INTEGRITY). The
    /// resolved absolute path + per-file size are logged ALONGSIDE the
    /// verification (not as a substitute for it).
    fn verify_checksums(dir: &std::path::Path, files: &[&std::path::Path]) -> Result<()> {
        use sha2::{Digest, Sha256};

        let manifest_path = dir.join("checksums.sha256");
        let manifest = std::fs::read_to_string(&manifest_path).map_err(|_| {
            Error::Embedding(
                "models_dir checksum manifest 'checksums.sha256' is missing — \
                 refusing to load unverified model weights (ADR 0004 A.5)"
                    .to_string(),
            )
        })?;

        // Parse manifest into filename -> expected lowercase hex.
        let mut expected: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        for line in manifest.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            // "<hex>  <filename>"  (sha256sum) OR "<filename>=<hex>".
            let (name, hex) = if let Some((n, h)) = line.split_once('=') {
                (n.trim(), h.trim())
            } else {
                let mut parts = line.split_whitespace();
                let h = parts.next().unwrap_or("");
                // filename may contain spaces; rejoin the remainder.
                let n = line[h.len()..].trim().trim_start_matches('*');
                (n, h)
            };
            if !name.is_empty() && !hex.is_empty() {
                // Key by the file name only (basename), case-insensitive hex.
                let base = std::path::Path::new(name)
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| name.to_string());
                expected.insert(base, hex.to_ascii_lowercase());
            }
        }

        for file in files {
            let name = file
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();

            let abs = std::fs::canonicalize(file).unwrap_or_else(|_| file.to_path_buf());
            let bytes = std::fs::read(file).map_err(|_| {
                Error::Embedding(format!(
                    "models_dir is missing required file '{name}' — refusing to load"
                ))
            })?;
            let size = bytes.len();

            let want = expected.get(&name).ok_or_else(|| {
                Error::Embedding(format!(
                    "checksum manifest has no entry for '{name}' — fail closed (ADR 0004 A.5)"
                ))
            })?;

            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            let got = hex_lower(&hasher.finalize());

            // Log resolved absolute path + size ALONGSIDE verification.
            tracing::info!(
                file = %name,
                path = %abs.display(),
                size_bytes = size,
                "verifying staged model file checksum"
            );

            if &got != want {
                return Err(Error::Embedding(format!(
                    "checksum mismatch for '{name}': manifest and file disagree — \
                     refusing to load (ADR 0004 A.5 WEIGHT INTEGRITY)"
                )));
            }
        }

        tracing::info!(
            dir = %std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf()).display(),
            files = files.len(),
            "models_dir checksums verified"
        );
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::io::Write;

    fn sha256_hex(bytes: &[u8]) -> String {
        let mut h = Sha256::new();
        h.update(bytes);
        hex_lower(&h.finalize())
    }

    /// Stage config.json/tokenizer.json/model.safetensors with arbitrary bytes.
    /// Returns (tempdir, the three file paths). If `manifest` is Some, writes
    /// checksums.sha256 with the given (filename, hex) entries.
    fn stage(
        contents: &[(&str, &[u8])],
        manifest: Option<&[(&str, &str)]>,
    ) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        for (name, bytes) in contents {
            let mut f = std::fs::File::create(dir.path().join(name)).unwrap();
            f.write_all(bytes).unwrap();
        }
        if let Some(entries) = manifest {
            let mut m = std::fs::File::create(dir.path().join("checksums.sha256")).unwrap();
            for (name, hex) in entries {
                // sha256sum format: "<hex>  <filename>"
                writeln!(m, "{hex}  {name}").unwrap();
            }
        }
        let p = dir.path().to_path_buf();
        (dir, p)
    }

    fn files(dir: &std::path::Path) -> [std::path::PathBuf; 3] {
        [
            dir.join("config.json"),
            dir.join("tokenizer.json"),
            dir.join("model.safetensors"),
        ]
    }

    /// A.5: correct manifest hashes => verification passes.
    #[test]
    fn a5_checksum_correct_passes() {
        let c = b"{\"hidden_size\":384}";
        let t = b"tokenizer-bytes";
        let w = b"weights-bytes";
        let manifest = [
            ("config.json", sha256_hex(c)),
            ("tokenizer.json", sha256_hex(t)),
            ("model.safetensors", sha256_hex(w)),
        ];
        let mref: Vec<(&str, &str)> =
            manifest.iter().map(|(n, h)| (*n, h.as_str())).collect();
        let (_d, dir) = stage(
            &[("config.json", c), ("tokenizer.json", t), ("model.safetensors", w)],
            Some(&mref),
        );
        let fs = files(&dir);
        let refs: Vec<&std::path::Path> = fs.iter().map(|p| p.as_path()).collect();
        CandleEmbedder::verify_checksums(&dir, &refs).expect("correct hashes must pass");
    }

    /// A.5: a WRONG hash => verification fails closed (Err).
    #[test]
    fn a5_checksum_mismatch_fails_closed() {
        let c = b"{\"hidden_size\":384}";
        let t = b"tokenizer-bytes";
        let w = b"weights-bytes";
        let bad = "0".repeat(64);
        let manifest = [
            ("config.json", sha256_hex(c)),
            ("tokenizer.json", sha256_hex(t)),
            ("model.safetensors", bad.clone()), // WRONG
        ];
        let mref: Vec<(&str, &str)> =
            manifest.iter().map(|(n, h)| (*n, h.as_str())).collect();
        let (_d, dir) = stage(
            &[("config.json", c), ("tokenizer.json", t), ("model.safetensors", w)],
            Some(&mref),
        );
        let fs = files(&dir);
        let refs: Vec<&std::path::Path> = fs.iter().map(|p| p.as_path()).collect();
        let res = CandleEmbedder::verify_checksums(&dir, &refs);
        assert!(res.is_err(), "mismatched hash must fail closed");
    }

    /// A.5: MISSING manifest => fail closed (no manifest = no unverified load).
    #[test]
    fn a5_missing_manifest_fails_closed() {
        let (_d, dir) = stage(
            &[
                ("config.json", b"x"),
                ("tokenizer.json", b"y"),
                ("model.safetensors", b"z"),
            ],
            None, // no checksums.sha256
        );
        let fs = files(&dir);
        let refs: Vec<&std::path::Path> = fs.iter().map(|p| p.as_path()).collect();
        let res = CandleEmbedder::verify_checksums(&dir, &refs);
        assert!(res.is_err(), "missing manifest must fail closed");
    }

    /// A.5: manifest present but missing an entry for one file => fail closed.
    #[test]
    fn a5_missing_entry_fails_closed() {
        let c = b"cc";
        let t = b"tt";
        let w = b"ww";
        // Only config + tokenizer listed; weights entry absent.
        let manifest = [
            ("config.json", sha256_hex(c)),
            ("tokenizer.json", sha256_hex(t)),
        ];
        let mref: Vec<(&str, &str)> =
            manifest.iter().map(|(n, h)| (*n, h.as_str())).collect();
        let (_d, dir) = stage(
            &[("config.json", c), ("tokenizer.json", t), ("model.safetensors", w)],
            Some(&mref),
        );
        let fs = files(&dir);
        let refs: Vec<&std::path::Path> = fs.iter().map(|p| p.as_path()).collect();
        let res = CandleEmbedder::verify_checksums(&dir, &refs);
        assert!(res.is_err(), "missing manifest entry must fail closed");
    }
}
