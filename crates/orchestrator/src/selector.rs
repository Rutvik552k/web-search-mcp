//! Search-source selector + config builder (ADR 0004 §3/§4, Addendum A.6.6).
//!
//! The selector replaces the engine's hardcoded `if let Some(searxng_url)`
//! branch. It holds an INJECTED, ordered `Vec<Box<dyn SearchSource>>` and walks
//! it: the first source that returns `Ok(non-empty)` wins; a miss
//! (`Ok(vec![])`) or a failure (`Err`) advances to the next; if all are
//! exhausted it returns an empty Vec and the caller uses the frontier floor
//! (`generate_search_seeds`).
//!
//! Per A.6.6 the selector does NOT construct clients itself — that is the job
//! of the separate `build_sources_from_config` builder, keeping the selector a
//! pure ordering/fall-through machine that is trivially testable with a
//! `FakeSource`.

use crate::search_source::{SearchHit, SearchSource};
use web_search_common::config::CrawlerConfig;

/// Ordered, injected list of search sources. Resolves a query to the first
/// non-empty source, falling through misses and failures.
pub struct SearchSelector {
    sources: Vec<Box<dyn SearchSource>>,
}

impl SearchSelector {
    /// Construct from an already-built, ordered source list (A.6.6 — sources are
    /// injected; the selector never constructs clients).
    pub fn new(sources: Vec<Box<dyn SearchSource>>) -> Self {
        Self { sources }
    }

    /// True when no sources are configured — the caller goes straight to the
    /// frontier floor.
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// Walk the sources in order. Return the first `Ok(non-empty)` hit list.
    /// Misses (`Ok(vec![])`) and failures (`Err`, logged with the source name)
    /// both advance to the next source. All exhausted ⇒ empty Vec (caller uses
    /// the frontier floor — `generate_search_seeds`).
    pub async fn resolve(&self, query: &str, limit: usize) -> Vec<SearchHit> {
        for source in &self.sources {
            match source.search(query, limit).await {
                Ok(hits) if !hits.is_empty() => {
                    tracing::info!(
                        source = source.name(),
                        hits = hits.len(),
                        "search source resolved"
                    );
                    return hits;
                }
                Ok(_) => {
                    // Miss — advance (identical to Err per §4, distinct only for
                    // telemetry).
                    tracing::debug!(source = source.name(), "search source miss; advancing");
                }
                Err(_) => {
                    // Failure — advance. Name the source; never log the error
                    // body (may carry transport detail).
                    tracing::warn!(source = source.name(), "search source failed; advancing");
                }
            }
        }
        Vec::new()
    }
}

/// SearXNG `SearchSource` — wraps the engine's existing `fetch_searxng_results`
/// (back-compat for operators running their own SearXNG; zero IP cost to us).
/// Applies the same A.4 sanitization to its hits before they enter the index.
pub struct SearxngSource {
    base_url: String,
}

impl SearxngSource {
    pub fn new(base_url: String) -> Self {
        Self { base_url }
    }
}

#[async_trait::async_trait]
impl SearchSource for SearxngSource {
    fn name(&self) -> &str {
        "searxng"
    }

    async fn search(&self, query: &str, _limit: usize) -> anyhow::Result<Vec<SearchHit>> {
        let raw = crate::engine::fetch_searxng_results(&self.base_url, query).await;
        // A.4 / FIX #7: sanitize url + title/snippet before they enter the
        // shared index (url control-char strip is consistent with the text
        // fields; idempotent if already stripped at the source).
        Ok(raw
            .into_iter()
            .map(|h| SearchHit {
                url: crate::tavily::sanitize_url(&h.url),
                title: crate::tavily::sanitize(&h.title),
                snippet: crate::tavily::sanitize(&h.snippet),
            })
            .collect())
    }

    fn spends_ip_reputation(&self) -> bool {
        false
    }
}

/// Build the ordered source list from config (A.6.6 — separate from the
/// selector). `env_lookup` injects the env-var read so this stays pure and
/// testable (Wave-1 injected-closure pattern) — production passes
/// `std::env::var`.
///
/// Resolution per `search_source`:
/// - `"auto"` (default): tavily (if `search_api_key_env` names a SET env var
///   AND provider=="tavily"), THEN searxng (if `searxng_url` set). Order:
///   tavily, then searxng. If the result is empty, emit the §8 keyless INFO.
/// - `"tavily"` pinned: only TavilySource (if a key is resolvable).
/// - `"searxng"` pinned: only SearxngSource (if `searxng_url` set).
/// - `"crawler"` pinned: empty (forces the frontier floor).
pub fn build_sources_from_config_with<F>(cfg: &CrawlerConfig, env_lookup: F) -> Vec<Box<dyn SearchSource>>
where
    F: Fn(&str) -> Option<String>,
{
    let timeout = cfg.search_api_timeout_secs;
    let mut sources: Vec<Box<dyn SearchSource>> = Vec::new();

    let tavily_key = || -> Option<String> {
        // Provider must be tavily (or unset, treated as the default tavily) and
        // the named env var must be SET and non-empty.
        let provider_ok = cfg
            .search_api_provider
            .as_deref()
            .map(|p| p.eq_ignore_ascii_case("tavily"))
            .unwrap_or(true);
        if !provider_ok {
            return None;
        }
        let env_name = cfg.search_api_key_env.as_deref()?;
        env_lookup(env_name).filter(|v| !v.is_empty())
    };

    match cfg.search_source.as_str() {
        "tavily" => {
            if let Some(key) = tavily_key() {
                sources.push(Box::new(crate::tavily::TavilySource::new(key, timeout)));
            }
        }
        "searxng" => {
            if let Some(url) = cfg.searxng_url.clone() {
                sources.push(Box::new(SearxngSource::new(url)));
            }
        }
        "crawler" => {
            // Pinned floor — no sources; the engine uses generate_search_seeds.
        }
        // "auto" (default) and any unknown value resolve to auto behavior.
        _ => {
            if let Some(key) = tavily_key() {
                sources.push(Box::new(crate::tavily::TavilySource::new(key, timeout)));
            }
            if let Some(url) = cfg.searxng_url.clone() {
                sources.push(Box::new(SearxngSource::new(url)));
            }
            if sources.is_empty() {
                // §8 keyless default INFO — the supported standalone path.
                tracing::info!("running keyless: crawler-seed search; results via governed crawl");
            }
        }
    }

    sources
}

/// Production wrapper: reads real env vars via `std::env::var`.
pub fn build_sources_from_config(cfg: &CrawlerConfig) -> Vec<Box<dyn SearchSource>> {
    build_sources_from_config_with(cfg, |name| std::env::var(name).ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    enum Outcome {
        Hits(Vec<SearchHit>),
        Empty,
        Err,
    }

    /// Scripted source recording its name into a shared call-log on each call,
    /// so tests assert resolution ORDER + short-circuit.
    struct FakeSource {
        name: &'static str,
        outcome: Outcome,
        log: Arc<Mutex<Vec<&'static str>>>,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl SearchSource for FakeSource {
        fn name(&self) -> &str {
            self.name
        }
        async fn search(&self, _q: &str, _l: usize) -> anyhow::Result<Vec<SearchHit>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.log.lock().unwrap().push(self.name);
            match &self.outcome {
                Outcome::Hits(h) => Ok(h.clone()),
                Outcome::Empty => Ok(vec![]),
                Outcome::Err => Err(anyhow::anyhow!("scripted failure")),
            }
        }
        fn spends_ip_reputation(&self) -> bool {
            false
        }
    }

    fn hit(url: &str) -> SearchHit {
        SearchHit { url: url.into(), title: "t".into(), snippet: "s".into() }
    }

    fn fake(
        name: &'static str,
        outcome: Outcome,
        log: &Arc<Mutex<Vec<&'static str>>>,
    ) -> Box<dyn SearchSource> {
        Box::new(FakeSource {
            name,
            outcome,
            log: log.clone(),
            calls: Arc::new(AtomicUsize::new(0)),
        })
    }

    #[tokio::test]
    async fn resolve_returns_first_nonempty_and_short_circuits() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let sel = SearchSelector::new(vec![
            fake("tavily", Outcome::Hits(vec![hit("https://t.com")]), &log),
            fake("searxng", Outcome::Hits(vec![hit("https://s.com")]), &log),
        ]);
        let hits = sel.resolve("q", 5).await;
        assert_eq!(hits, vec![hit("https://t.com")]);
        // Short-circuit: searxng never called.
        assert_eq!(*log.lock().unwrap(), vec!["tavily"]);
    }

    #[tokio::test]
    async fn resolve_order_is_tavily_then_searxng_then_crawler() {
        let log = Arc::new(Mutex::new(Vec::new()));
        // tavily misses, searxng errors, crawler(=last fake) hits.
        let sel = SearchSelector::new(vec![
            fake("tavily", Outcome::Empty, &log),
            fake("searxng", Outcome::Err, &log),
            fake("crawler", Outcome::Hits(vec![hit("https://c.com")]), &log),
        ]);
        let hits = sel.resolve("q", 5).await;
        assert_eq!(hits, vec![hit("https://c.com")]);
        assert_eq!(*log.lock().unwrap(), vec!["tavily", "searxng", "crawler"]);
    }

    #[tokio::test]
    async fn miss_advances_identically_to_error() {
        // Two chains: [Empty, Hits] and [Err, Hits] must both reach the 2nd source.
        for first in [Outcome::Empty, Outcome::Err] {
            let log = Arc::new(Mutex::new(Vec::new()));
            let sel = SearchSelector::new(vec![
                fake("first", first, &log),
                fake("second", Outcome::Hits(vec![hit("https://x.com")]), &log),
            ]);
            let hits = sel.resolve("q", 5).await;
            assert_eq!(hits, vec![hit("https://x.com")]);
            assert_eq!(*log.lock().unwrap(), vec!["first", "second"]);
        }
    }

    #[tokio::test]
    async fn all_exhausted_returns_empty() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let sel = SearchSelector::new(vec![
            fake("a", Outcome::Empty, &log),
            fake("b", Outcome::Err, &log),
        ]);
        assert!(sel.resolve("q", 5).await.is_empty());
        assert_eq!(*log.lock().unwrap(), vec!["a", "b"]);
    }

    // -- build_sources_from_config (injected env, no real env reads) ----------

    fn base_cfg() -> CrawlerConfig {
        web_search_common::config::Config::default().crawler
    }

    #[test]
    fn auto_keyless_no_searxng_is_empty() {
        let cfg = base_cfg(); // search_source="auto", no key env, no searxng_url
        let sources = build_sources_from_config_with(&cfg, |_| None);
        assert!(sources.is_empty(), "keyless auto must yield empty → floor");
    }

    #[test]
    fn auto_with_key_env_set_pushes_tavily() {
        let mut cfg = base_cfg();
        cfg.search_api_provider = Some("tavily".into());
        cfg.search_api_key_env = Some("TAVILY_API_KEY".into());
        let sources = build_sources_from_config_with(&cfg, |name| {
            if name == "TAVILY_API_KEY" { Some("SECRET".into()) } else { None }
        });
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].name(), "tavily");
    }

    #[test]
    fn auto_key_env_named_but_unset_is_empty() {
        let mut cfg = base_cfg();
        cfg.search_api_provider = Some("tavily".into());
        cfg.search_api_key_env = Some("TAVILY_API_KEY".into());
        // env_lookup returns None → no tavily.
        let sources = build_sources_from_config_with(&cfg, |_| None);
        assert!(sources.is_empty());
    }

    #[test]
    fn auto_key_and_searxng_orders_tavily_then_searxng() {
        let mut cfg = base_cfg();
        cfg.search_api_provider = Some("tavily".into());
        cfg.search_api_key_env = Some("K".into());
        cfg.searxng_url = Some("http://localhost:8080".into());
        let sources = build_sources_from_config_with(&cfg, |_| Some("SECRET".into()));
        assert_eq!(sources.len(), 2);
        assert_eq!(sources[0].name(), "tavily");
        assert_eq!(sources[1].name(), "searxng");
    }

    #[test]
    fn pinned_crawler_is_empty_even_with_key_and_searxng() {
        let mut cfg = base_cfg();
        cfg.search_source = "crawler".into();
        cfg.search_api_provider = Some("tavily".into());
        cfg.search_api_key_env = Some("K".into());
        cfg.searxng_url = Some("http://localhost:8080".into());
        let sources = build_sources_from_config_with(&cfg, |_| Some("SECRET".into()));
        assert!(sources.is_empty(), "pinned crawler forces the floor");
    }

    #[test]
    fn pinned_tavily_only_tavily() {
        let mut cfg = base_cfg();
        cfg.search_source = "tavily".into();
        cfg.search_api_provider = Some("tavily".into());
        cfg.search_api_key_env = Some("K".into());
        cfg.searxng_url = Some("http://localhost:8080".into());
        let sources = build_sources_from_config_with(&cfg, |_| Some("SECRET".into()));
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].name(), "tavily");
    }

    #[test]
    fn pinned_searxng_only_searxng() {
        let mut cfg = base_cfg();
        cfg.search_source = "searxng".into();
        cfg.searxng_url = Some("http://localhost:8080".into());
        let sources = build_sources_from_config_with(&cfg, |_| None);
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].name(), "searxng");
    }

    #[test]
    fn auto_non_tavily_provider_does_not_push_tavily() {
        let mut cfg = base_cfg();
        cfg.search_api_provider = Some("serper".into());
        cfg.search_api_key_env = Some("K".into());
        let sources = build_sources_from_config_with(&cfg, |_| Some("SECRET".into()));
        assert!(sources.is_empty(), "non-tavily provider not yet supported → no source");
    }
}
