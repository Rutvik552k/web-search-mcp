mod hash_embedder;
pub mod cross_encoder;

#[cfg(feature = "onnx")]
mod onnx_embedder;

#[cfg(feature = "onnx")]
pub mod onnx_cross_encoder;

#[cfg(feature = "onnx")]
pub mod colbert;

#[cfg(feature = "onnx")]
pub mod splade;

#[cfg(feature = "candle")]
mod candle_embedder;

use async_trait::async_trait;
use web_search_common::Result;

pub use hash_embedder::HashEmbedder;
pub use cross_encoder::{CrossEncoder, CrossEncoderScore, NliLabel};

#[cfg(feature = "onnx")]
pub use onnx_embedder::OnnxEmbedder;

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

/// Observable health of the embedder created by [`create_embedder`].
///
/// Distinguishes a healthy neural embedder from a degraded keyword-only one,
/// and (for the degraded case) carries the *reason* so the server can surface
/// it (ADR 0004 A.6.1). The reason is pre-sanitized — no filesystem paths, no
/// HF tokens (A.5/A.6.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbedderHealth {
    /// Neural embedder loaded and serving semantic embeddings.
    Neural {
        /// Model id / name (e.g. the MiniLM model).
        model: String,
    },
    /// Degraded to keyword-hashing. `reason` explains WHY (sanitized).
    Degraded {
        /// Sanitized, surface-able reason for the degradation.
        reason: String,
    },
}

impl EmbedderHealth {
    /// True iff a neural embedder is serving (not degraded to keyword-only).
    pub fn is_neural(&self) -> bool {
        matches!(self, EmbedderHealth::Neural { .. })
    }
}

/// Sanitize a candle/load error before it reaches a structured event or the
/// health channel: strip anything that could carry a filesystem path or an HF
/// token (A.5/A.6.2). We keep only the first line and drop substrings that look
/// like paths/URLs/tokens. Errs on the side of terseness.
fn sanitize_reason(raw: &str) -> String {
    let first_line = raw.lines().next().unwrap_or(raw);
    let cleaned: String = first_line
        .split_whitespace()
        .filter(|tok| {
            // Drop tokens that look like a path, URL, or bearer/HF token.
            let looks_like_path = tok.contains('/') || tok.contains('\\') || tok.contains(':');
            let looks_like_token = tok.starts_with("hf_") || tok.len() > 40;
            !(looks_like_path || looks_like_token)
        })
        .collect::<Vec<_>>()
        .join(" ");
    let cleaned = cleaned.trim();
    if cleaned.is_empty() {
        "model load failed".to_string()
    } else {
        cleaned.to_string()
    }
}

/// Create the best available embedder based on compiled features and config,
/// returning the embedder together with its observable [`EmbedderHealth`]
/// (ADR 0004 A.6.1 — the health return channel makes the degraded state
/// surface-able by the server).
///
/// Backend resolution (ADR 0004 §6.3 / A.7):
/// - `embedder_backend = Some("hash")` => keyword-hashing, by explicit choice
///   (the old "fast, no-neural" capability; health = `Degraded { "hash backend
///   selected by config" }`). This is the ONLY way to deliberately skip neural.
/// - `embedder_backend = Some("neural")` => force candle; init failure is fatal
///   (health is never degraded — we panic-free-return the error path's hash
///   only if candle is not compiled, otherwise propagate as Degraded with the
///   reason so the server still starts; see below).
/// - `None` (DEFAULT) / anything else => auto: ALWAYS ATTEMPT the neural
///   embedder (candle), and degrade to keyword hashing ONLY when candle
///   genuinely cannot load (offline/load failure) or the feature isn't compiled.
///
/// IMPORTANT (A.7, §6.3): `force_cpu` does NOT short-circuit to HashEmbedder.
/// `force_cpu` now means "run the neural model on CPU" and is honored inside
/// `CandleEmbedder::new` (`Device::Cpu`). The old skip-neural branch is removed
/// because it silently destroyed semantic quality (ADR 0004 §1.2, §6.3).
pub fn create_embedder(
    config: &web_search_common::config::EmbedderConfig,
) -> (Box<dyn Embedder>, EmbedderHealth) {
    // Explicit escape hatch: "hash" => keyword hashing by choice (A.7).
    if matches!(config.embedder_backend.as_deref(), Some("hash")) {
        let e = HashEmbedder::new(config.embedding_dim);
        tracing::info!(
            dim = e.dimensions(),
            backend = "hash",
            "HashEmbedder selected by config (embedder_backend=\"hash\") — keyword hashing, no semantic matching"
        );
        return (
            Box::new(e),
            EmbedderHealth::Degraded {
                reason: "hash backend selected by config".to_string(),
            },
        );
    }

    let force_neural = matches!(config.embedder_backend.as_deref(), Some("neural"));

    // Auto / neural: ALWAYS attempt the neural embedder. `force_cpu` only picks
    // the CPU device inside CandleEmbedder::new — it never skips neural here.
    #[cfg(feature = "candle")]
    {
        tracing::info!(force_cpu = config.force_cpu, "Initializing neural embedder (candle)...");
        match CandleEmbedder::new(config) {
            Ok(e) => {
                let model = e.model_name().to_string();
                tracing::info!(
                    model = %model,
                    dim = e.dimensions(),
                    "Neural embedder ready (CandleEmbedder)"
                );
                return (Box::new(e), EmbedderHealth::Neural { model });
            }
            Err(e) => {
                // Loud, named, structured fallback event (A.6.2). Sanitize the
                // reason: no filesystem paths, no HF tokens.
                //
                // RISK-ACCEPT #6 (security audit): when a NAMED `models_dir`
                // checksum verification fails (tamper/mismatch), this path
                // currently DEGRADES to HashEmbedder rather than aborting the
                // process. Accepted: the failure is loud (the `model_fetch_failed`
                // event below) and surfaced to the operator via the
                // `EmbedderHealth::Degraded { reason }` health channel, so a
                // tampered/absent staged model never silently loads — it visibly
                // drops to keyword-only. Hard-aborting on a staged-model integrity
                // failure is a possible future hardening but is out of scope here.
                let reason = sanitize_reason(&e.to_string());
                let model_id = "sentence-transformers/all-MiniLM-L6-v2";
                tracing::warn!(
                    event = "model_fetch_failed",
                    model = %model_id,
                    reason = %reason,
                    "neural embedder unavailable — degraded to keyword-only"
                );
                if force_neural {
                    tracing::warn!(
                        backend = "neural",
                        "embedder_backend=\"neural\" requested but candle failed to load; \
                         starting degraded (keyword-only) rather than aborting the server"
                    );
                }
                let hash = HashEmbedder::new(config.embedding_dim);
                return (Box::new(hash), EmbedderHealth::Degraded { reason });
            }
        }
    }

    // candle feature NOT compiled: there is no neural path to attempt.
    #[cfg(not(feature = "candle"))]
    {
        let reason = if force_neural {
            "neural backend requested but candle feature not compiled".to_string()
        } else {
            "candle feature not compiled".to_string()
        };
        // Emit the same named event so the degraded state is observable even
        // when the binary was built without candle (A.6.2).
        tracing::warn!(
            event = "model_fetch_failed",
            model = "sentence-transformers/all-MiniLM-L6-v2",
            reason = %reason,
            "neural embedder unavailable — degraded to keyword-only"
        );
        let e = HashEmbedder::new(config.embedding_dim);
        tracing::info!(
            dim = e.dimensions(),
            backend = "hash",
            "Using HashEmbedder (feature hashing — no semantic matching)"
        );
        (Box::new(e), EmbedderHealth::Degraded { reason })
    }
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

    // ---- ADR 0004 Wave 3 embedder-hardening tests ----

    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use tracing::field::{Field, Visit};
    use tracing::subscriber::with_default;
    use tracing::Subscriber;
    use tracing_subscriber::layer::{Context, SubscriberExt};
    use tracing_subscriber::registry::LookupSpan;
    use tracing_subscriber::Layer;

    /// Tracing layer that flips a flag when a `model_fetch_failed` event fires
    /// (capture without a network/`tracing-test` dep).
    struct FetchFailedSpy {
        seen: Arc<AtomicBool>,
    }
    struct EventVisitor {
        is_fetch_failed: bool,
    }
    impl Visit for EventVisitor {
        fn record_debug(&mut self, _f: &Field, _v: &dyn std::fmt::Debug) {}
        fn record_str(&mut self, field: &Field, value: &str) {
            if field.name() == "event" && value == "model_fetch_failed" {
                self.is_fetch_failed = true;
            }
        }
    }
    impl<S> Layer<S> for FetchFailedSpy
    where
        S: Subscriber + for<'a> LookupSpan<'a>,
    {
        fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
            let mut v = EventVisitor { is_fetch_failed: false };
            event.record(&mut v);
            if v.is_fetch_failed {
                self.seen.store(true, Ordering::SeqCst);
            }
        }
    }

    fn cfg_with(
        models_dir: Option<std::path::PathBuf>,
        force_cpu: bool,
        backend: Option<&str>,
    ) -> web_search_common::config::EmbedderConfig {
        let mut c = web_search_common::config::Config::default().embedder;
        c.models_dir = models_dir;
        c.force_cpu = force_cpu;
        c.embedder_backend = backend.map(|s| s.to_string());
        c
    }

    /// A.6.1: `embedder_backend="hash"` => HashEmbedder + a distinct
    /// hash-by-choice Degraded health, returned via the tuple.
    #[test]
    fn a61_hash_backend_by_choice() {
        let (emb, health) = create_embedder(&cfg_with(None, false, Some("hash")));
        assert_eq!(emb.model_name(), "hash-embedder");
        match health {
            EmbedderHealth::Degraded { reason } => {
                assert!(reason.contains("hash backend selected by config"), "{reason}");
            }
            EmbedderHealth::Neural { .. } => panic!("hash backend must be Degraded"),
        }
    }

    /// A.7: `force_cpu=true` MUST NOT short-circuit to HashEmbedder. With a
    /// bogus models_dir (so candle cannot load) the path must be
    /// attempt-neural-then-degrade, and the health reason must reflect WHY it is
    /// hash (a load failure), NOT "skipped by force_cpu".
    ///
    /// RED: restoring the old `if force_cpu { return Hash }` short-circuit makes
    /// this fail — the reason would be a skip note, not a load-failure reason,
    /// and no `model_fetch_failed` event would fire on the force_cpu path.
    #[test]
    fn a7_force_cpu_does_not_skip_neural() {
        let seen = Arc::new(AtomicBool::new(false));
        let sub = tracing_subscriber::registry().with(FetchFailedSpy { seen: seen.clone() });
        with_default(sub, || {
            let bogus = std::path::PathBuf::from("/nonexistent_models_dir_a7_xyz");
            let (emb, health) = create_embedder(&cfg_with(Some(bogus), true, None));
            // Degraded because candle genuinely could not load (bad dir),
            // NOT because force_cpu skipped neural.
            match health {
                EmbedderHealth::Degraded { reason } => {
                    let r = reason.to_lowercase();
                    assert!(
                        !r.contains("force_cpu") && !r.contains("skip"),
                        "reason must reflect a load failure, not a force_cpu skip: {reason}"
                    );
                }
                EmbedderHealth::Neural { .. } => {
                    // Only legitimate if candle somehow loaded a default model;
                    // then force_cpu certainly did not skip neural — also pass.
                }
            }
            let _ = emb;
        });
        // When candle is compiled, the bogus dir forces a load failure and the
        // named event must fire. When candle is NOT compiled, the not-compiled
        // branch also emits the named event. Either way it must be observable.
        #[cfg(feature = "candle")]
        assert!(seen.load(Ordering::SeqCst), "model_fetch_failed must fire on candle load failure");
    }

    /// A.6.2: a load failure (bogus models_dir) must NOT panic, must return a
    /// HashEmbedder, set Degraded health, and emit a `model_fetch_failed` event.
    /// (No candle network needed — the bad-dir seam fails before any fetch.)
    #[test]
    fn a62_named_fallback_on_load_failure() {
        let seen = Arc::new(AtomicBool::new(false));
        let sub = tracing_subscriber::registry().with(FetchFailedSpy { seen: seen.clone() });
        with_default(sub, || {
            let bogus = std::path::PathBuf::from("/nonexistent_models_dir_a62_qrs");
            let (emb, health) = create_embedder(&cfg_with(Some(bogus), false, None));
            assert_eq!(emb.model_name(), "hash-embedder", "must degrade to HashEmbedder");
            assert!(matches!(health, EmbedderHealth::Degraded { .. }));
        });
        assert!(
            seen.load(Ordering::SeqCst),
            "model_fetch_failed event must be emitted on degrade"
        );
    }

    #[test]
    fn sanitize_reason_strips_paths_and_tokens() {
        let raw = "load weights: /home/user/models/model.safetensors not found hf_abcDEF123token";
        let s = sanitize_reason(raw);
        assert!(!s.contains('/'), "no paths: {s}");
        assert!(!s.contains("hf_"), "no HF token: {s}");
        assert!(!s.is_empty());
    }

    /// A.7 perf gate — CPU neural-inference latency vs a stated threshold.
    ///
    /// `#[ignore]` + `#[cfg(feature="candle")]`: this is NOT a CI test (it needs
    /// candle + a cached/fetchable MiniLM model — testing-rule "no real external
    /// services" keeps it out of the default run). Run explicitly with:
    ///   cargo test -p web-search-embedder --features candle -- --ignored a7_perf_gate_cpu_inference
    ///
    /// THRESHOLD (stated): CPU mean latency for a 384-d MiniLM embed of a single
    /// short sentence must be < 250 ms (warm). This is the gate the ADR §6.3/A.7
    /// requires "before the old fast path is removed" — the `embedder_backend=
    /// "hash"` escape hatch preserves the fast capability regardless, de-risking
    /// it. The measured number is recorded in benchmark/RESULTS.embedder-cpu.md.
    #[cfg(feature = "candle")]
    #[ignore = "perf gate — run explicitly with --ignored; needs candle + cached MiniLM"]
    #[tokio::test]
    async fn a7_perf_gate_cpu_inference() {
        const THRESHOLD_MS: f64 = 250.0;
        let cfg = cfg_with(None, /*force_cpu=*/ true, Some("neural"));
        let (emb, health) = create_embedder(&cfg);
        assert!(health.is_neural(), "perf gate requires the neural embedder to load: {health:?}");
        assert_eq!(emb.dimensions(), 384);
        assert_eq!(emb.model_name(), "sentence-transformers/all-MiniLM-L6-v2");

        // Warmup (load mmap caches, first forward).
        let _ = emb.embed(&["warmup sentence for the cpu perf gate"]).await.unwrap();

        let runs = 20;
        let start = std::time::Instant::now();
        for _ in 0..runs {
            let _ = emb.embed(&["the quick brown fox jumps over the lazy dog"]).await.unwrap();
        }
        let mean_ms = start.elapsed().as_secs_f64() * 1000.0 / runs as f64;
        eprintln!("A.7 perf gate: CPU MiniLM mean embed latency = {mean_ms:.1} ms over {runs} runs (threshold {THRESHOLD_MS} ms)");
        assert!(
            mean_ms < THRESHOLD_MS,
            "CPU neural inference too slow ({mean_ms:.1} ms >= {THRESHOLD_MS} ms) — keep embedder_backend=\"hash\" escape hatch"
        );
    }
}
