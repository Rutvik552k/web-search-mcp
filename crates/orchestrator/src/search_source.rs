//! Search-source contract (ADR 0004 §4).
//!
//! A single trait abstracts "where SERP-like results come from", so the engine
//! stops branching on `searxng_url`. Implementations MUST be reputation-safe per
//! ADR 0003 (live-origin sources go through the governor) — that wiring is a
//! later wave; this module introduces only the contract types.
//!
//! # Miss vs. failure (load-bearing contract, ADR 0004 §4)
//!
//! - `Ok(vec![])` means **"I had nothing"** (a *miss*). The selector advances to
//!   the next source.
//! - `Err(_)` means **"I failed"** (transport/quota/parse error). The selector
//!   ALSO advances to the next source — but the distinction is preserved for
//!   telemetry. A keyed-API failure (bad key, 429 quota) MUST surface as a miss
//!   or a sanitized error, NEVER break the whole search (ASR-SC3).
//!
//! Implementations MUST never panic and MUST never block the runtime.

use async_trait::async_trait;

/// One search-result triple for a query — the shared shape every `SearchSource`
/// produces. This is the public promotion of `engine.rs`'s private
/// `SearxngResult` (ADR 0004 §3/§8); the engine's private type is retained for
/// now and rewired in a later wave.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchHit {
    /// Result URL. Attacker-influenceable — the fetch path MUST SSRF-guard this
    /// before opening a socket (ADR 0004 Addendum A.3). Not validated here.
    pub url: String,
    /// Result title.
    pub title: String,
    /// Result snippet / summary text.
    pub snippet: String,
}

/// A source of search-result triples for a query. Implementations MUST be
/// reputation-safe per ADR 0003 (live-origin sources go through the governor).
#[async_trait]
pub trait SearchSource: Send + Sync {
    /// Name for logging/telemetry (e.g. "tavily", "searxng", "crawler-seeds").
    fn name(&self) -> &str;

    /// Returns ranked result triples for `query`, capped at `limit`.
    ///
    /// Contract:
    /// - `Ok(vec![])` = **miss** ("I had nothing") — distinct from an error.
    /// - `Err(_)` = **failure** ("I failed; caller should fall through").
    ///
    /// In BOTH cases the selector falls through to the next source; the
    /// distinction is for telemetry, not correctness. Never panics; never blocks
    /// the runtime.
    async fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<SearchHit>>;

    /// True iff this source touches the live origin / our IP reputation
    /// (crawler-seeds = true; keyed API / SearXNG-url = false). The selector uses
    /// this only for telemetry + ordering, not correctness (ADR 0004 §4).
    fn spends_ip_reputation(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Scripted test source proving the trait is object-safe and exercising the
    /// miss-vs-failure contract. Returns whatever it is told to return and counts
    /// invocations so a selector test (later wave) can assert call order.
    struct FakeSource {
        name: &'static str,
        spends_ip: bool,
        outcome: Outcome,
        calls: Arc<AtomicUsize>,
    }

    #[derive(Clone)]
    enum Outcome {
        Hits(Vec<SearchHit>),
        Empty,
        Err,
    }

    #[async_trait]
    impl SearchSource for FakeSource {
        fn name(&self) -> &str {
            self.name
        }
        async fn search(&self, _query: &str, _limit: usize) -> anyhow::Result<Vec<SearchHit>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            match &self.outcome {
                Outcome::Hits(h) => Ok(h.clone()),
                Outcome::Empty => Ok(vec![]),
                Outcome::Err => Err(anyhow::anyhow!("scripted failure")),
            }
        }
        fn spends_ip_reputation(&self) -> bool {
            self.spends_ip
        }
    }

    fn hit(url: &str) -> SearchHit {
        SearchHit {
            url: url.to_string(),
            title: "t".to_string(),
            snippet: "s".to_string(),
        }
    }

    /// The trait MUST be object-safe so the selector can hold
    /// `Vec<Box<dyn SearchSource>>` (ADR 0004 Addendum A.6.6).
    #[tokio::test]
    async fn trait_is_object_safe_via_boxed_dyn() {
        let calls = Arc::new(AtomicUsize::new(0));
        let src: Box<dyn SearchSource> = Box::new(FakeSource {
            name: "fake",
            spends_ip: true,
            outcome: Outcome::Hits(vec![hit("https://example.com")]),
            calls: calls.clone(),
        });
        assert_eq!(src.name(), "fake");
        assert!(src.spends_ip_reputation());
        let got = src.search("q", 5).await.unwrap();
        assert_eq!(got, vec![hit("https://example.com")]);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    /// `Ok(vec![])` is a MISS and is distinct from `Err` — both are observable
    /// and neither panics (ADR 0004 §4 miss-vs-failure contract).
    #[tokio::test]
    async fn empty_is_a_miss_distinct_from_error() {
        let calls = Arc::new(AtomicUsize::new(0));

        let miss: Box<dyn SearchSource> = Box::new(FakeSource {
            name: "miss",
            spends_ip: false,
            outcome: Outcome::Empty,
            calls: calls.clone(),
        });
        let res = miss.search("q", 5).await;
        assert!(res.is_ok(), "Empty must be Ok, not Err");
        assert!(res.unwrap().is_empty(), "Empty must yield an empty Vec");

        let fail: Box<dyn SearchSource> = Box::new(FakeSource {
            name: "fail",
            spends_ip: false,
            outcome: Outcome::Err,
            calls: calls.clone(),
        });
        let res = fail.search("q", 5).await;
        assert!(res.is_err(), "Err must be Err, distinct from the empty miss");
    }
}
