//! WAVE 4 cross-cutting integration tests (ADR 0004 §13 Addendum).
//!
//! These tie Waves 1–3 together at the lowest honest seam, with ZERO real
//! external services (no live SERP / HF / Tavily / SearXNG / network). Fakes,
//! local in-memory indices, and injected env closures only.
//!
//! Coverage map:
//!   * A.6.5 — mock ASR-SC1 / contractual floor  → `a65_*` tests below.
//!   * A.8   — KI-1 non-worsening tripwire        → `a8_*` tests below.
//!
//! A.2 is covered by:
//!   * the seed-survival unit tests in `engine.rs` (`a2_*`, private fn access), and
//!   * the crawler-crate governor tests (`a2_*` in `fetcher.rs`) proving the four
//!     SERP hosts make zero socket. (Split across crates because the orchestrator
//!     cannot inject a seeded-denylist crawler without a network crawl — see the
//!     report for the honest gap note.)

use web_search_common::config::{Config, CrawlerConfig};
use web_search_orchestrator::selector::{build_sources_from_config_with, SearchSelector};

// ── A.6.5 — mock ASR-SC1 / contractual floor ────────────────────────────────
//
// ADR 0004 §13 A.6.5: no Tavily key + no `searxng_url` + `search_source="auto"`
// ⇒ the selector resolves to the floor (returns an EMPTY hit list, never Err,
// never panics) and `generate_search_seeds` is non-empty so the floor has entry
// points. This is the CI-safe contract form of ASR-SC1 (the live smoke is
// explicitly OUT of CI). All env reads are injected (`|_| None`) — no real env,
// no network.

fn default_crawler() -> CrawlerConfig {
    Config::default().crawler
}

#[test]
fn a65_default_config_is_the_keyless_floor() {
    // The shipped default: search_source="auto", no provider, no key env, no
    // searxng_url. This is the §9 zero-setup acceptance config.
    let cfg = default_crawler();
    assert_eq!(cfg.search_source, "auto", "default must be auto");
    assert!(cfg.search_api_key_env.is_none(), "default must be keyless");
    assert!(cfg.searxng_url.is_none(), "default must have no searxng_url");
}

#[test]
fn a65_auto_keyless_builds_empty_source_list() {
    // build_sources_from_config_with with an EMPTY env (|_| None) ⇒ no keyed
    // source, no searxng source ⇒ empty Vec ⇒ the engine uses the frontier floor.
    let cfg = default_crawler();
    let sources = build_sources_from_config_with(&cfg, |_| None);
    assert!(
        sources.is_empty(),
        "A.6.5: keyless auto must produce an empty source list (→ floor)"
    );
}

#[tokio::test]
async fn a65_empty_selector_resolves_to_floor_without_error_or_panic() {
    // The §8 keyless behavior: a selector built from an empty source list returns
    // an EMPTY hit list (never Err, never panic). The empty result is the signal
    // the engine uses to drop to generate_search_seeds.
    let cfg = default_crawler();
    let selector = SearchSelector::new(build_sources_from_config_with(&cfg, |_| None));
    assert!(selector.is_empty(), "no sources configured");

    // resolve() returns Vec (not Result) — proving it cannot Err; assert empty.
    let hits = selector.resolve("how to learn rust", 10).await;
    assert!(
        hits.is_empty(),
        "A.6.5: keyless floor must resolve to an empty hit list, not fabricate hits"
    );
}

#[tokio::test]
async fn a65_explicit_empty_selector_is_safe() {
    // The literal `SearchSelector::new(vec![])` form named in the task: the floor
    // contract holds even when constructed directly with no sources.
    let selector = SearchSelector::new(vec![]);
    assert!(selector.is_empty());
    let hits = selector.resolve("anything at all", 5).await;
    assert!(hits.is_empty(), "explicit empty selector must yield empty, no panic");
}

// ── A.8 — KI-1 non-worsening tripwire ───────────────────────────────────────
//
// ADR 0004 §13 A.8 + §5 (KI-1): the engine accumulates ALL queries' crawls into
// ONE shared session index and serves later queries from it (index-first
// retrieval, engine.rs quick_search). Cross-query bleed = query B's top-N
// contains a doc whose ONLY provenance is query A's corpus.
//
// LOWEST REAL SEAM (no network): the shared session index *is* a
// `web_search_indexer::TextIndex`. We reproduce the exact shared-index condition
// the engine creates — two queries, DISJOINT corpora, one shared index — and put
// a TRIPWIRE on the contamination ratio. This is a regression guard, NOT a fix:
// it fails the future KI-1 ADR's target if bleed worsens past the captured
// baseline.

use web_search_common::models::{Page, PageMetadata};
use web_search_indexer::TextIndex;

/// Build a minimal `Page` fixture with a controllable body. `tag` is embedded in
/// the body so we can trace provenance: a result whose body came only from corpus
/// A carries A's tag terms and none of B's.
fn page(url: &str, domain: &str, title: &str, body: &str) -> Page {
    Page {
        url: url.to_string(),
        domain: domain.to_string(),
        title: Some(title.to_string()),
        author: None,
        published_date: None,
        body_text: body.to_string(),
        headings: vec![],
        links: vec![],
        tables: vec![],
        metadata: PageMetadata {
            language: Some("en".into()),
            description: None,
            content_type: "text/html".into(),
            status_code: 200,
            response_time_ms: 1,
            content_length: body.len(),
            extraction_confidence: 0.9,
            json_ld: None,
            open_graph: None,
        },
        content_hash: format!("{:x}", md5_like(url)),
        crawled_at: chrono::Utc::now(),
    }
}

/// Tiny deterministic non-crypto hash for unique content_hash per URL (avoids a
/// crate dep just for fixtures).
fn md5_like(s: &str) -> u64 {
    let mut h: u64 = 1469598103934665603;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628257);
    }
    h
}

/// Domains that belong to corpus A (the "django" query) and corpus B ("react").
/// Provenance is judged by domain: a B-query result on an A-only domain is bleed.
const CORPUS_A_DOMAINS: [&str; 3] = ["djangoproject.com", "django-a.example", "djangogirls.org"];

fn is_corpus_a_domain(domain: &str) -> bool {
    CORPUS_A_DOMAINS.iter().any(|d| domain == *d)
}

/// Reproduce the engine's shared-index condition and measure cross-query bleed.
/// Returns (top_n_len, contaminated_count) for query B.
fn measure_bleed_shared_index() -> (usize, usize) {
    let idx = TextIndex::in_memory(15_000_000).expect("in-memory index");

    // Corpus A — crawled for query "django" (the PRIOR query). Disjoint topic.
    idx.add_page(
        &page(
            "https://djangoproject.com/docs",
            "djangoproject.com",
            "Django web framework documentation",
            "Django is a high-level Python web framework for the django orm and django views \
             python backend server-side templating migrations admin",
        ),
        None,
    )
    .unwrap();
    idx.add_page(
        &page(
            "https://django-a.example/tutorial",
            "django-a.example",
            "Django tutorial models and querysets",
            "django models queryset python web backend orm migrations django admin server",
        ),
        None,
    )
    .unwrap();
    idx.add_page(
        &page(
            "https://djangogirls.org/intro",
            "djangogirls.org",
            "Django Girls intro to django",
            "django python web framework backend tutorial beginners orm views templates",
        ),
        None,
    )
    .unwrap();

    // Corpus B — crawled for query "react" (the CURRENT query). Disjoint topic.
    idx.add_page(
        &page(
            "https://react.dev/learn",
            "react.dev",
            "React JavaScript library for UIs",
            "react is a javascript library for building user interfaces with react hooks jsx \
             components state props frontend rendering virtual dom react components",
        ),
        None,
    )
    .unwrap();
    idx.add_page(
        &page(
            "https://react-b.example/hooks",
            "react-b.example",
            "React hooks guide",
            "react hooks usestate useeffect components jsx frontend javascript ui rendering react",
        ),
        None,
    )
    .unwrap();

    idx.commit().unwrap();

    // Query B mirrors the engine's index-first retrieval (quick_search line ~507):
    // search the shared index with B's query.
    let top_n = 5usize;
    let results = idx.search("react hooks javascript components", top_n).unwrap();

    let contaminated = results
        .iter()
        .filter(|r| is_corpus_a_domain(&r.domain))
        .count();

    (results.len(), contaminated)
}

#[test]
fn a8_ki1_tripwire_bleed_within_baseline() {
    // CAPTURED BASELINE (recorded 2026-06-14 on this BM25 seam): with disjoint
    // corpora, a B-topic query returns ZERO corpus-A documents — BM25 term
    // overlap alone does not surface "django" docs for a "react" query. The
    // engine's KI-1 bug is NOT in BM25 relevance; it is in the index-first
    // *freshness/score gate* serving cross-query docs. This tripwire pins the
    // BM25-layer contamination ratio at the captured baseline so a future change
    // that loosens the relevance gate (worsening KI-1) fails here.
    //
    // Baseline contamination ratio for query B at the index layer = 0.0.
    const BASELINE_CONTAMINATION: f64 = 0.0;

    let (top_n, contaminated) = measure_bleed_shared_index();
    assert!(top_n > 0, "query B must return results (sanity)");

    let ratio = contaminated as f64 / top_n as f64;
    assert!(
        ratio <= BASELINE_CONTAMINATION,
        "A.8 KI-1 TRIPWIRE: query-B contamination ratio {ratio} ({contaminated}/{top_n}) \
         exceeds captured baseline {BASELINE_CONTAMINATION}. Cross-query index bleed has \
         WORSENED. This is the KI-1 regression guard — investigate the index-first \
         relevance/freshness gate (ADR 0004 §5/§11.4) before raising this baseline."
    );
}

#[test]
#[ignore = "RED-CHECK scaffold: run manually to confirm the tripwire fires on worsened bleed"]
fn a8_ki1_red_check_worsened_bleed_would_fire() {
    // RED proof: if cross-query bleed WORSENS (a corpus-A django doc stuffed with
    // corpus-B react terms surfaces for the react query), the contamination ratio
    // exceeds the 0.0 baseline and the real `a8_ki1_tripwire_bleed_within_baseline`
    // assertion would FAIL. This demonstrates the tripwire is not vacuously green.
    let idx = TextIndex::in_memory(15_000_000).unwrap();
    idx.add_page(
        &page(
            "https://djangoproject.com/x",
            "djangoproject.com",
            "react react react",
            "react hooks javascript components react react react react react react react",
        ),
        None,
    )
    .unwrap();
    idx.add_page(
        &page(
            "https://react.dev/learn",
            "react.dev",
            "react",
            "react hooks javascript components react",
        ),
        None,
    )
    .unwrap();
    idx.commit().unwrap();

    let results = idx.search("react hooks javascript components", 5).unwrap();
    let contaminated = results.iter().filter(|r| is_corpus_a_domain(&r.domain)).count();
    let ratio = contaminated as f64 / results.len() as f64;
    assert!(
        ratio > 0.0,
        "RED-CHECK: worsened bleed must exceed baseline 0.0 (got {ratio}); \
         if this fails the tripwire would be vacuous"
    );
}

#[test]
fn a8_ki1_tripwire_b_results_are_react_provenance() {
    // Stronger provenance assertion: every result query B serves must come from a
    // corpus-B (react) domain — none whose ONLY provenance is corpus A.
    let idx = TextIndex::in_memory(15_000_000).expect("in-memory index");
    measure_bleed_shared_index(); // (exercises the helper for parity)

    // Rebuild inline to assert per-result provenance directly.
    idx.add_page(
        &page(
            "https://djangoproject.com/docs",
            "djangoproject.com",
            "Django docs",
            "django python web framework orm views backend server migrations admin templating",
        ),
        None,
    )
    .unwrap();
    idx.add_page(
        &page(
            "https://react.dev/learn",
            "react.dev",
            "React docs",
            "react javascript hooks jsx components state props frontend ui rendering react",
        ),
        None,
    )
    .unwrap();
    idx.commit().unwrap();

    let results = idx.search("react hooks javascript", 5).unwrap();
    assert!(!results.is_empty(), "B query returns results");
    for r in &results {
        assert!(
            !is_corpus_a_domain(&r.domain),
            "A.8: query-B result {} has corpus-A-only provenance (cross-query bleed)",
            r.url
        );
    }
}
