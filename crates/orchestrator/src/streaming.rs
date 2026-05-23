// SPDX-License-Identifier: MIT

//! Streaming search pipeline with progressive results.
//!
//! Returns partial results as they become available via mpsc channel.
//! Two-tier deadline: partial results at 3s, hard stop at 10s.
//!
//! Pipeline stages run concurrently via channels:
//! crawl → extract → embed → rank → client

use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use web_search_common::models::*;

/// Events emitted by the streaming pipeline.
#[derive(Debug, Clone)]
pub enum SearchEvent {
    /// First batch of results available (fast, lightly ranked)
    PartialResults {
        results: Vec<RankedResult>,
        elapsed_ms: u64,
    },
    /// Refined results after full ranking pipeline
    RefinedResults {
        results: Vec<RankedResult>,
        synthesis: Vec<SynthesizedSentence>,
        elapsed_ms: u64,
    },
    /// Pipeline status update
    Progress {
        stage: String,
        detail: String,
    },
    /// Pipeline complete — no more events
    Complete {
        total_pages_crawled: usize,
        total_time_ms: u64,
    },
}

/// Configuration for streaming search.
pub struct StreamConfig {
    /// Deadline for first partial results (default: 3s)
    pub partial_deadline: Duration,
    /// Hard deadline — stop everything (default: 10s)
    pub hard_deadline: Duration,
    /// Max results in partial batch
    pub partial_max: usize,
    /// Max results in final batch
    pub final_max: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            partial_deadline: Duration::from_secs(3),
            hard_deadline: Duration::from_secs(10),
            partial_max: 5,
            final_max: 10,
        }
    }
}

/// Launch a streaming search that sends events to the returned receiver.
///
/// The caller can consume events as they arrive:
/// ```ignore
/// let mut rx = engine.streaming_search("query", config).await;
/// while let Some(event) = rx.recv().await {
///     match event {
///         SearchEvent::PartialResults { results, .. } => show_partial(results),
///         SearchEvent::RefinedResults { results, .. } => show_final(results),
///         SearchEvent::Complete { .. } => break,
///         _ => {}
///     }
/// }
/// ```
pub fn create_event_channel(buffer: usize) -> (mpsc::Sender<SearchEvent>, mpsc::Receiver<SearchEvent>) {
    mpsc::channel(buffer)
}

/// Helper to emit a progress event (best-effort, non-blocking).
pub fn emit_progress(tx: &mpsc::Sender<SearchEvent>, stage: &str, detail: &str) {
    let _ = tx.try_send(SearchEvent::Progress {
        stage: stage.to_string(),
        detail: detail.to_string(),
    });
}

/// Helper to emit partial results (best-effort, non-blocking).
pub fn emit_partial(tx: &mpsc::Sender<SearchEvent>, results: Vec<RankedResult>, start: Instant) {
    let _ = tx.try_send(SearchEvent::PartialResults {
        results,
        elapsed_ms: start.elapsed().as_millis() as u64,
    });
}

/// Helper to emit refined results.
pub async fn emit_refined(
    tx: &mpsc::Sender<SearchEvent>,
    results: Vec<RankedResult>,
    synthesis: Vec<SynthesizedSentence>,
    start: Instant,
) {
    let _ = tx
        .send(SearchEvent::RefinedResults {
            results,
            synthesis,
            elapsed_ms: start.elapsed().as_millis() as u64,
        })
        .await;
}

/// Helper to emit completion.
pub async fn emit_complete(
    tx: &mpsc::Sender<SearchEvent>,
    total_pages: usize,
    start: Instant,
) {
    let _ = tx
        .send(SearchEvent::Complete {
            total_pages_crawled: total_pages,
            total_time_ms: start.elapsed().as_millis() as u64,
        })
        .await;
}

/// Check if we've exceeded the hard deadline.
pub fn past_deadline(start: Instant, deadline: Duration) -> bool {
    start.elapsed() >= deadline
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn event_channel_works() {
        let (tx, mut rx) = create_event_channel(8);

        emit_progress(&tx, "crawl", "fetching 10 pages");
        let event = rx.recv().await.unwrap();
        match event {
            SearchEvent::Progress { stage, detail } => {
                assert_eq!(stage, "crawl");
                assert_eq!(detail, "fetching 10 pages");
            }
            _ => panic!("wrong event type"),
        }
    }

    #[tokio::test]
    async fn partial_results_emitted() {
        let (tx, mut rx) = create_event_channel(8);
        let start = Instant::now();

        let results = vec![RankedResult {
            content: "test".into(),
            url: "https://example.com".into(),
            title: "Test".into(),
            confidence: 0.8,
            verification: VerificationStatus::Unverified,
            claims: vec![],
            contradictions: vec![],
            source_tier: SourceTier::Tier1,
            freshness: None,
            relevance_score: 0.9,
        }];

        emit_partial(&tx, results, start);
        let event = rx.recv().await.unwrap();
        match event {
            SearchEvent::PartialResults { results, .. } => {
                assert_eq!(results.len(), 1);
                assert_eq!(results[0].url, "https://example.com");
            }
            _ => panic!("wrong event type"),
        }
    }

    #[test]
    fn deadline_check() {
        let start = Instant::now();
        assert!(!past_deadline(start, Duration::from_secs(10)));
        // Can't easily test true case without sleeping
    }

    #[test]
    fn default_config() {
        let cfg = StreamConfig::default();
        assert_eq!(cfg.partial_deadline, Duration::from_secs(3));
        assert_eq!(cfg.hard_deadline, Duration::from_secs(10));
        assert_eq!(cfg.partial_max, 5);
        assert_eq!(cfg.final_max, 10);
    }
}
