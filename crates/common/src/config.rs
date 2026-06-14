use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub crawler: CrawlerConfig,
    pub extractor: ExtractorConfig,
    pub indexer: IndexerConfig,
    pub embedder: EmbedderConfig,
    pub ranker: RankerConfig,
    pub server: ServerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrawlerConfig {
    /// Number of concurrent crawler workers
    pub num_workers: usize,
    /// Maximum concurrent connections total
    pub max_concurrent_connections: usize,
    /// Per-domain requests per second limit
    pub requests_per_second_per_domain: f64,
    /// Default timeout per request in seconds
    pub request_timeout_secs: u64,
    /// User-Agent string
    pub user_agent: String,
    /// Respect robots.txt
    pub respect_robots_txt: bool,
    /// Enable headless browser for JS-heavy pages
    pub enable_browser: bool,
    /// Maximum retry attempts per URL
    pub max_retries: u32,
    /// Backoff base in milliseconds for retries
    pub backoff_base_ms: u64,
    /// SearXNG instance URL for metasearch (e.g., "http://localhost:8080").
    /// When set, used as primary search source (JSON API, no CAPTCHA).
    /// Run: docker run -d -p 8080:8080 searxng/searxng
    #[serde(default)]
    pub searxng_url: Option<String>,

    // -- Self-contained search source (ADR 0004 §8) --------------------------
    // All additive / off-safe. The DEFAULT ("auto" with no key/searxng_url)
    // resolves to the governed crawler-seeds floor — the product runs fully
    // standalone with no key and no container (ADR 0004 §2.2).
    /// Search source resolution. "auto" (DEFAULT) = keyed-api if a key is set,
    /// else searxng_url if set, else crawler-seeds. Other values pin one source:
    /// "crawler" | "tavily" | "searxng" — used for testing/forcing the floor.
    #[serde(default = "default_search_source")]
    pub search_source: String,
    /// Optional keyed search-API provider. None/"" => no keyed source.
    /// "tavily" (recommended; true no-card free tier) | "serper" | ... .
    #[serde(default)]
    pub search_api_provider: Option<String>,
    /// NAME of the env var holding the search-API key — NEVER the key itself
    /// (api-security SECRETS MANAGEMENT; mirrors captcha_api_key_env). The key is
    /// read via `std::env::var(<this name>)` at construction time and is never
    /// logged. None => keyless.
    #[serde(default)]
    pub search_api_key_env: Option<String>,
    /// Bounded wait for the keyed-API request, in seconds. Default 5.
    #[serde(default = "default_search_api_timeout_secs")]
    pub search_api_timeout_secs: u64,

    // -- R4 stealth headless (Design 0004 Part 1; ADR 0003 R4) ----------------
    // All default-OFF / off-safe. When `enable_stealth == false` the browser
    // path is byte-for-byte the legacy SPA-render behavior. These only take
    // effect when the `browser` cargo feature is also compiled in.
    /// Master switch for the stealth headless layer. Default false.
    /// When false, BrowserPool behaves exactly as before (no CDP stealth, no
    /// preload script, fixed-sleep wait).
    #[serde(default)]
    pub enable_stealth: bool,
    /// User-Agent presented by the stealth browser. `None` => use a current
    /// real Chrome UA constant (see crawler::browser::DEFAULT_STEALTH_UA).
    /// MUST NOT contain "HeadlessChrome" (Design 0004 §1.3 / checklist #3).
    #[serde(default)]
    pub stealth_user_agent: Option<String>,
    /// Accept-Language / `navigator.languages` coherence value for the stealth
    /// browser. `None` => "en-US,en;q=0.9".
    #[serde(default)]
    pub stealth_locale: Option<String>,

    // -- R5 commercial CAPTCHA solver (Design 0004 Part 2; ADR 0003 R5) -------
    // All default-OFF / off-safe. The solver is NEVER constructed unless
    // `enable_captcha_solver` is true AND a known provider AND a SET env var are
    // all present (Design 0004 §2.5 hard gate). Automated CAPTCHA solving is the
    // GOAL.md §2 / ADR 0001 Phase-5 legal-gated tier — keep these off until that
    // compliance review authorizes a shipped config.
    /// Master switch for the commercial CAPTCHA solver layer. Default false.
    /// LEGAL + COST gate (Design 0004 §2.5/§2.6).
    #[serde(default)]
    pub enable_captcha_solver: bool,
    /// PER-RUN opt-in for the live R5 solve path (compliance gate 0001 C-1).
    /// This is SEPARATE from, and IN ADDITION TO, `enable_captcha_solver`. A
    /// shipped config with `enable_captcha_solver = true` is necessary but NOT
    /// sufficient: a live solve also requires this per-session/per-run opt-in to
    /// be set by the caller, so a single persisted global flag cannot silently
    /// turn every crawl into a paid-solve crawl. Default FALSE (default-deny).
    /// Solving requires BOTH `enable_captcha_solver` AND `captcha_run_opt_in`.
    #[serde(default)]
    pub captcha_run_opt_in: bool,
    /// Provider selection: "capsolver" | "2captcha". `None` => solver disabled.
    #[serde(default)]
    pub captcha_provider: Option<String>,
    /// NAME of the environment variable that holds the provider API key — never
    /// the key itself (api-security SECRETS MANAGEMENT; Design 0004 §2.5). The
    /// key is read via `std::env::var(<this name>)` at construction time and is
    /// never logged.
    #[serde(default)]
    pub captcha_api_key_env: Option<String>,
    /// Total bounded wait per solve, in seconds (createTask + all polls). Caps
    /// the two-call async flow (Design 0004 §2.3). Default 120.
    #[serde(default = "default_captcha_timeout_secs")]
    pub captcha_timeout_secs: u64,
    /// Hard per-SESSION USD spend cap for the CAPTCHA solver. Every solve is
    /// metered through `CostMeter`; reaching this cap hard-halts all paid solves
    /// (cost-tracking BUDGET GUARD; Design 0004 §2.5). Must be > 0 when the
    /// solver is enabled (rejected at startup otherwise). Default 5.0.
    #[serde(default = "default_captcha_session_cost_cap_usd")]
    pub captcha_session_cost_cap_usd: f64,

    // -- Hybrid escalation controller (Design 0005 §9; ADR 0001 §5 / ADR 0003 §6)
    // All default-OFF / off-safe. With `enable_escalation == false` (DEFAULT)
    // NONE of these fields are consulted and `Fetcher::fetch` is byte-for-byte
    // the legacy path. These are SEPARATE from the stealth/captcha fields above
    // and must not collide with them.
    /// Master switch for the classifier-routed escalation controller. Default
    /// false => byte-for-byte current behavior (Design 0005 §3 expand-contract).
    #[serde(default)]
    pub enable_escalation: bool,
    /// Highest rung the controller may reach. 0=R0,1=R1,4=R4,5=R5 (ADR 0003
    /// numbering; R2/R3 are stubs). Default 4 (R4 stealth, no paid R5). The
    /// controller clamps to <=4 when `enable_captcha_solver == false`.
    #[serde(default = "default_max_escalation_rung")]
    pub max_escalation_rung: u8,
    /// Optional TOML/JSON path of extra/replacement classifier marker strings
    /// merged into the seeded ADR 0001 §4 lists. None => seeded defaults only.
    #[serde(default)]
    pub classifier_marker_overrides: Option<PathBuf>,
    /// Min visible-text chars for RealContent vs SoftBlock. Default ~500.
    #[serde(default = "default_soft_block_min_bytes")]
    pub soft_block_min_bytes: usize,
    /// Live-origin request cap per domain per session. Default 50.
    #[serde(default = "default_per_domain_request_budget")]
    pub per_domain_request_budget: u32,
    /// Soft signals before the cooldown breaker opens. Default 5.
    #[serde(default = "default_soft_breaker_fail_threshold")]
    pub soft_breaker_fail_threshold: u32,
    /// Soft-trip cooldown seconds. Default 300.
    #[serde(default = "default_domain_breaker_cooldown_secs")]
    pub domain_breaker_cooldown_secs: u64,
    /// Pacing jitter ratio (±). Must be in [0, 1). Default 0.3.
    #[serde(default = "default_pacing_jitter_ratio")]
    pub pacing_jitter_ratio: f64,
    /// Persistent, restart-surviving hard-stop denylist path (ADR 0003 §4.3).
    /// None (DEFAULT) => in-memory only (a hard ban won't survive restart;
    /// the controller emits a startup WARN — Design 0005 §5.2 / ADR 0003 §6).
    #[serde(default)]
    pub permanent_denylist_path: Option<PathBuf>,

    // -- R3 archive fallback (Internet Archive CDX — ADR 0003 §6.2) -----------
    // Zero-ban-risk rung: queries web.archive.org only, never the live target
    // origin. All default-OFF / off-safe; consulted only on the escalation path.
    /// Master switch for the R3 Internet Archive CDX fallback. Default false.
    #[serde(default)]
    pub enable_archive_fallback: bool,
    /// CDX query endpoint. Default the public IA CDX server.
    #[serde(default = "default_archive_cdx_endpoint")]
    pub archive_cdx_endpoint: String,
    /// Bounded wait for the CDX query + snapshot fetch, in ms. Default 4000.
    #[serde(default = "default_archive_timeout_ms")]
    pub archive_timeout_ms: u64,
    /// Reject snapshots older than this many days. None (DEFAULT) => any age.
    #[serde(default)]
    pub archive_max_snapshot_age_days: Option<u32>,
    /// Polite UA presented to web.archive.org. None => the crawler `user_agent`.
    #[serde(default)]
    pub archive_user_agent: Option<String>,

    // -- R2 alternative-surface probe (feeds/sitemap — ADR 0003 §6.2) ---------
    // Low-ban-risk rung: static syndication surfaces (same origin, rarely
    // WAF-challenged). All default-OFF master, sub-surfaces default-on.
    /// Master switch for the R2 alternative-surface probe. Default false.
    #[serde(default)]
    pub enable_alt_surface: bool,
    /// Sub-surface: RSS/Atom/JSON-Feed discovery + parse. Only active when
    /// `enable_alt_surface` is also on. Default true.
    #[serde(default = "default_true")]
    pub src_feed: bool,
    /// Sub-surface: sitemap (incl. news sitemap) discovery. Only active when
    /// `enable_alt_surface` is also on. Default true.
    #[serde(default = "default_true")]
    pub src_sitemap: bool,
    /// Max total network probes the R2 rung may make per URL. Default 3.
    #[serde(default = "default_max_alt_probes")]
    pub max_alt_probes: u8,

    /// Cross-client clearance-cookie replay (Design 0005 §4.2 / §7 R-3, Design
    /// 0004 §1.7). Default FALSE — this is the UNVERIFIED path.
    ///
    /// When TRUE, a `cf_clearance` / `datadome` cookie minted in the stealth
    /// browser (R4) is replayed onto the plain reqwest client (R1): the matching
    /// cookie AND the exact User-Agent it was minted under are attached to the
    /// reqwest GET. This is the high-leverage R4→R1 reuse path — but the cookie
    /// is widely reported bound to the client TLS/JA3 + HTTP/2 fingerprint (not
    /// just session + UA + IP), in which case a browser-minted cookie is
    /// REJECTED by the reqwest client (different TLS fingerprint) and the replay
    /// is at best wasted and at worst a tell.
    ///
    /// The honest expectation until the gated spike (`clearance_replay_binding`
    /// in fetcher.rs, operator-run) confirms otherwise is: the cookie MAY be
    /// fingerprint-bound, so this stays default-OFF and the fallback is to keep
    /// the solved session inside the browser (same-client SAFE reuse only).
    /// Single-IP satisfies the IP-binding automatically (our IP is static).
    #[serde(default)]
    pub enable_clearance_replay: bool,
}

fn default_captcha_timeout_secs() -> u64 {
    120
}

fn default_search_source() -> String {
    "auto".to_string()
}

fn default_search_api_timeout_secs() -> u64 {
    5
}

fn default_captcha_session_cost_cap_usd() -> f64 {
    5.0
}

fn default_max_escalation_rung() -> u8 {
    4
}

fn default_soft_block_min_bytes() -> usize {
    500
}

fn default_per_domain_request_budget() -> u32 {
    50
}

fn default_soft_breaker_fail_threshold() -> u32 {
    5
}

fn default_domain_breaker_cooldown_secs() -> u64 {
    300
}

fn default_pacing_jitter_ratio() -> f64 {
    0.3
}

fn default_archive_cdx_endpoint() -> String {
    "http://web.archive.org/cdx/search/cdx".to_string()
}

fn default_archive_timeout_ms() -> u64 {
    4000
}

fn default_max_alt_probes() -> u8 {
    3
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractorConfig {
    /// Enable ML-based content classification (requires ONNX model)
    pub enable_ml_classifier: bool,
    /// Path to content classifier ONNX model
    pub classifier_model_path: PathBuf,
    /// Minimum consensus votes to keep a content block (2 or 3)
    pub min_consensus_votes: u8,
    /// Maximum text chunk size in tokens for embedding
    pub chunk_size_tokens: usize,
    /// Chunk overlap ratio (0.0 - 0.5)
    pub chunk_overlap_ratio: f32,
    /// Master switch for in-band data-layer acquisition (ADR 0002, Rung -1 IN-band:
    /// hydration-state salvage + structured-data promotion). OFF = byte-for-byte
    /// identical to pre-data-layer extraction behavior.
    #[serde(default)]
    pub enable_data_layer: bool,
    /// Sub-flag: parse `__NEXT_DATA__`/`__NUXT__`/`__INITIAL_STATE__`/`__APOLLO_STATE__`/
    /// RSC flight blobs to salvage thin/soft-blocked pages. Only active when
    /// `enable_data_layer` is also on.
    #[serde(default = "default_true")]
    pub data_layer_hydration: bool,
    /// Sub-flag: promote JSON-LD `Article.articleBody` / OG fields to primary content
    /// when richer than the consensus extraction. Only active when `enable_data_layer`
    /// is also on.
    #[serde(default = "default_true")]
    pub data_layer_structured_promotion: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexerConfig {
    /// Path to tantivy index directory
    pub index_path: PathBuf,
    /// Path to HNSW vector index directory
    pub vector_index_path: PathBuf,
    /// Heap size for tantivy writer in bytes
    pub tantivy_heap_size: usize,
    /// HNSW M parameter (max connections per node)
    pub hnsw_m: usize,
    /// HNSW ef_construction parameter
    pub hnsw_ef_construction: usize,
    /// SimHash hamming distance threshold for near-duplicate
    pub simhash_threshold: u32,
    /// Enable Product Quantization for vectors
    pub enable_pq: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedderConfig {
    /// Path to embedding ONNX model
    pub embedding_model_path: PathBuf,
    /// Embedding dimensions (384 for MiniLM-L6)
    pub embedding_dim: usize,
    /// Batch size for embedding generation
    pub batch_size: usize,
    /// Force CPU even if GPU available.
    ///
    /// MEANING CHANGED (ADR 0004 §6.3 / A.7): `force_cpu` now means "run the
    /// neural model on CPU" — it NO LONGER skips the neural embedder and drops
    /// to keyword hashing. The old "skip neural" behavior silently destroyed
    /// semantic quality and was removed. To explicitly opt into keyword-only
    /// hashing, set `embedder_backend = "hash"` (below), not `force_cpu`.
    pub force_cpu: bool,
    /// Optional directory of pre-staged model files for air-gapped/offline use
    /// (ADR 0004 §6.2). None => auto-fetch from HF on first run (default). When
    /// set and present, loaded instead of fetching. Files in this directory are
    /// checksum-verified against `checksums.sha256` before load (ADR 0004 A.5).
    #[serde(default)]
    pub models_dir: Option<PathBuf>,
    /// Embedder backend selection (ADR 0004 §6.3 / A.7 escape hatch).
    /// - `None` (DEFAULT) => auto: attempt the neural embedder (candle), and
    ///   fall back to keyword hashing ONLY if the neural model genuinely cannot
    ///   load (offline/load failure) or the `candle` feature isn't compiled.
    /// - `Some("neural")` => force the neural embedder; init failure is fatal.
    /// - `Some("hash")` => explicitly use the keyword-hashing embedder (the old
    ///   "fast, no-neural" capability — preserved as an explicit opt-in so the
    ///   §6.3 meaning change to `force_cpu` does not silently delete it).
    #[serde(default)]
    pub embedder_backend: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankerConfig {
    /// Path to cross-encoder ONNX model
    pub cross_encoder_model_path: PathBuf,
    /// BM25 top-K for Stage 1
    pub bm25_top_k: usize,
    /// HNSW top-K for Stage 1
    pub hnsw_top_k: usize,
    /// Cross-encoder top-K for Stage 2 output
    pub rerank_top_k: usize,
    /// Minimum sources for claim verification
    pub min_verification_sources: usize,
    /// MMR lambda (0.0 = pure diversity, 1.0 = pure relevance)
    pub mmr_lambda: f32,
    /// Maximum results per domain in final output
    pub max_results_per_domain: usize,
    /// Minimum unique source organizations in final output
    pub min_unique_orgs: usize,
    /// Path to source tiers config
    pub source_tiers_path: PathBuf,
    /// Minimum relevance score for final results (0.0 - 1.0).
    /// Results below this are filtered after Stage 2 cross-encoder reranking.
    #[serde(default = "default_min_relevance_score")]
    pub min_relevance_score: f32,
}

fn default_min_relevance_score() -> f32 {
    0.35
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Server name reported via MCP
    pub name: String,
    /// Server version
    pub version: String,
    /// Data directory for runtime storage
    pub data_dir: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        let data_dir = PathBuf::from("data");
        Self {
            crawler: CrawlerConfig {
                num_workers: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).max(4),
                max_concurrent_connections: 500,
                requests_per_second_per_domain: 2.0,
                request_timeout_secs: 30,
                user_agent: "WebSearchMCP/0.1 (research-bot)".to_string(),
                respect_robots_txt: true,
                enable_browser: false,
                max_retries: 3,
                backoff_base_ms: 1000,
                searxng_url: None,
                search_source: default_search_source(),
                search_api_provider: None,
                search_api_key_env: None,
                search_api_timeout_secs: default_search_api_timeout_secs(),
                enable_stealth: false,
                stealth_user_agent: None,
                stealth_locale: None,
                enable_captcha_solver: false,
                captcha_run_opt_in: false,
                captcha_provider: None,
                captcha_api_key_env: None,
                captcha_timeout_secs: default_captcha_timeout_secs(),
                captcha_session_cost_cap_usd: default_captcha_session_cost_cap_usd(),
                enable_escalation: false,
                max_escalation_rung: default_max_escalation_rung(),
                classifier_marker_overrides: None,
                soft_block_min_bytes: default_soft_block_min_bytes(),
                per_domain_request_budget: default_per_domain_request_budget(),
                soft_breaker_fail_threshold: default_soft_breaker_fail_threshold(),
                domain_breaker_cooldown_secs: default_domain_breaker_cooldown_secs(),
                pacing_jitter_ratio: default_pacing_jitter_ratio(),
                permanent_denylist_path: None,
                enable_archive_fallback: false,
                archive_cdx_endpoint: default_archive_cdx_endpoint(),
                archive_timeout_ms: default_archive_timeout_ms(),
                archive_max_snapshot_age_days: None,
                archive_user_agent: None,
                enable_alt_surface: false,
                src_feed: true,
                src_sitemap: true,
                max_alt_probes: default_max_alt_probes(),
                enable_clearance_replay: false,
            },
            extractor: ExtractorConfig {
                enable_ml_classifier: false,
                classifier_model_path: PathBuf::from("models/content-clf.onnx"),
                min_consensus_votes: 2,
                chunk_size_tokens: 512,
                chunk_overlap_ratio: 0.2,
                enable_data_layer: false,
                data_layer_hydration: true,
                data_layer_structured_promotion: true,
            },
            indexer: IndexerConfig {
                index_path: data_dir.join("index"),
                vector_index_path: data_dir.join("vectors"),
                tantivy_heap_size: 50_000_000, // 50MB
                hnsw_m: 16,
                hnsw_ef_construction: 200,
                simhash_threshold: 3,
                enable_pq: false,
            },
            embedder: EmbedderConfig {
                embedding_model_path: PathBuf::from("models/minilm-l6-v2.onnx"),
                embedding_dim: 384,
                batch_size: 32,
                force_cpu: false,
                models_dir: None,
                embedder_backend: None,
            },
            ranker: RankerConfig {
                cross_encoder_model_path: PathBuf::from("models/cross-encoder.onnx"),
                bm25_top_k: 200,
                hnsw_top_k: 200,
                rerank_top_k: 50,
                min_verification_sources: 3,
                mmr_lambda: 0.7,
                max_results_per_domain: 2,
                min_unique_orgs: 3,
                source_tiers_path: PathBuf::from("config/source_tiers.toml"),
                min_relevance_score: 0.35,
            },
            server: ServerConfig {
                name: "web-search-mcp".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                data_dir,
            },
        }
    }
}

impl Config {
    /// Load config from a TOML file, falling back to defaults for missing fields.
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(path)
            .map_err(|e| Error::Config(format!("failed to read config: {e}")))?;
        let config: Self = toml::from_str(&content)
            .map_err(|e| Error::Config(format!("failed to parse config: {e}")))?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pre-existing `config/default.toml` (which has NO captcha keys) must
    /// keep parsing, and the new R5 fields must default off-safe. This guards
    /// the additive-`#[serde(default)]` contract (Design 0004 Part 3).
    #[test]
    fn legacy_crawler_section_parses_with_captcha_defaults_off() {
        // A `[crawler]` table WITHOUT any captcha_* / stealth_* keys — i.e. the
        // shape of the shipped config/default.toml prior to R5.
        let toml_src = r#"
            num_workers = 8
            max_concurrent_connections = 500
            requests_per_second_per_domain = 2.0
            request_timeout_secs = 30
            user_agent = "WebSearchMCP/0.1 (research-bot)"
            respect_robots_txt = true
            enable_browser = false
            max_retries = 3
            backoff_base_ms = 1000
        "#;
        let cfg: CrawlerConfig = toml::from_str(toml_src).expect("legacy crawler config must parse");
        // R5 solver must be off-safe by default.
        assert!(!cfg.enable_captcha_solver);
        // Per-run opt-in (compliance gate 0001 C-1) must default-deny even when
        // the legacy config omits the key entirely.
        assert!(!cfg.captcha_run_opt_in);
        assert!(cfg.captcha_provider.is_none());
        assert!(cfg.captcha_api_key_env.is_none());
        assert_eq!(cfg.captcha_timeout_secs, 120);
        assert_eq!(cfg.captcha_session_cost_cap_usd, 5.0);
        // Hybrid escalation controller (Design 0005 §9) must be off-safe.
        assert!(!cfg.enable_escalation);
        assert_eq!(cfg.max_escalation_rung, 4);
        assert!(cfg.classifier_marker_overrides.is_none());
        assert_eq!(cfg.soft_block_min_bytes, 500);
        assert_eq!(cfg.per_domain_request_budget, 50);
        assert_eq!(cfg.soft_breaker_fail_threshold, 5);
        assert_eq!(cfg.domain_breaker_cooldown_secs, 300);
        assert!((cfg.pacing_jitter_ratio - 0.3).abs() < f64::EPSILON);
        assert!(cfg.permanent_denylist_path.is_none());
        // R3 archive fallback (ADR 0003 §6.2) must be off-safe with IA defaults.
        assert!(!cfg.enable_archive_fallback);
        assert_eq!(cfg.archive_cdx_endpoint, "http://web.archive.org/cdx/search/cdx");
        assert_eq!(cfg.archive_timeout_ms, 4000);
        assert!(cfg.archive_max_snapshot_age_days.is_none());
        assert!(cfg.archive_user_agent.is_none());
        // R2 alternative-surface (ADR 0003 §6.2): master off, sub-surfaces on.
        assert!(!cfg.enable_alt_surface);
        assert!(cfg.src_feed);
        assert!(cfg.src_sitemap);
        assert_eq!(cfg.max_alt_probes, 3);
        // Cross-client clearance replay (Design 0005 §4.2 / R-3) is the
        // unverified path — must default OFF.
        assert!(!cfg.enable_clearance_replay);
    }

    #[test]
    fn default_config_has_captcha_solver_off() {
        let cfg = Config::default();
        assert!(!cfg.crawler.enable_captcha_solver);
        assert!(!cfg.crawler.captcha_run_opt_in);
        assert!(cfg.crawler.captcha_provider.is_none());
    }

    /// ADR 0004 §8 / Addendum A.6.7: a legacy `[crawler]`/`[embedder]` TOML with
    /// NO `search_*` / `models_dir` keys must keep parsing and yield the
    /// off-safe self-contained defaults (search_source="auto", keyless,
    /// timeout=5, models_dir=None). Guards the additive-`#[serde(default)]`
    /// contract for the self-contained-deploy fields.
    #[test]
    fn legacy_config_parses_with_search_defaults_auto() {
        let crawler_src = r#"
            num_workers = 8
            max_concurrent_connections = 500
            requests_per_second_per_domain = 2.0
            request_timeout_secs = 30
            user_agent = "WebSearchMCP/0.1 (research-bot)"
            respect_robots_txt = true
            enable_browser = false
            max_retries = 3
            backoff_base_ms = 1000
        "#;
        let crawler: CrawlerConfig =
            toml::from_str(crawler_src).expect("legacy crawler config must parse");
        assert_eq!(crawler.search_source, "auto");
        assert!(crawler.search_api_provider.is_none());
        assert!(crawler.search_api_key_env.is_none());
        assert_eq!(crawler.search_api_timeout_secs, 5);

        let embedder_src = r#"
            embedding_model_path = "models/minilm-l6-v2.onnx"
            embedding_dim = 384
            batch_size = 32
            force_cpu = false
        "#;
        let embedder: EmbedderConfig =
            toml::from_str(embedder_src).expect("legacy embedder config must parse");
        assert!(embedder.models_dir.is_none());
        // ADR 0004 A.7: the escape-hatch field must default None (auto =
        // attempt-neural-then-degrade) and a legacy TOML without the key parses.
        assert!(embedder.embedder_backend.is_none());
    }

    /// ADR 0004 Addendum A.6.7: the *actual shipped* `config/default.toml` (which
    /// has NO `search_*` / `models_dir` keys) must parse under the new struct.
    /// Catches struct/file drift — if a required field is added without a serde
    /// default, this fails.
    #[test]
    fn shipped_default_toml_roundtrips() {
        // CARGO_MANIFEST_DIR = crates/common ; the shipped file is at the repo
        // root under config/default.toml (two levels up).
        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let path = manifest_dir.join("../../config/default.toml");
        let content = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read shipped config {}: {e}", path.display()));
        let cfg: Config = toml::from_str(&content)
            .unwrap_or_else(|e| panic!("shipped config/default.toml must parse: {e}"));
        // The shipped file carries no search_* keys → serde defaults apply.
        assert_eq!(cfg.crawler.search_source, "auto");
        assert!(cfg.crawler.search_api_provider.is_none());
        assert!(cfg.crawler.search_api_key_env.is_none());
        assert_eq!(cfg.crawler.search_api_timeout_secs, 5);
        assert!(cfg.embedder.models_dir.is_none());
        assert!(cfg.embedder.embedder_backend.is_none());
    }

    /// ADR 0004 A.7: `Config::default()` must carry the escape-hatch field as
    /// None (auto). Guards against a non-None default silently forcing a backend.
    #[test]
    fn default_config_embedder_backend_is_none() {
        let cfg = Config::default();
        assert!(cfg.embedder.embedder_backend.is_none());
    }
}
