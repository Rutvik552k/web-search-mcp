//! G4 head-to-head comparison core (TASKS.md 0.6 / 4.4, GOAL.md §G4).
//!
//! Pure aggregation + report-building over two coverage runs (ours vs Firecrawl)
//! on the SAME `urls.jsonl`. The I/O (driving our MCP server, calling Firecrawl)
//! lives in the `firecrawl-compare` binary; the math + report live here so they
//! are unit-tested deterministically (base of the pyramid).
//!
//! G4 targets (GOAL.md §G4):
//!   * coverage, blocked-subset, accuracy: ours − Firecrawl ≥ **+5 pts**
//!   * P99 latency per page: ours **≤** Firecrawl
//!   * $/1k pages: ours **strictly lower**

use crate::firecrawl;

/// One page outcome on one side (ours or Firecrawl), already scored clean/miss.
#[derive(Debug, Clone)]
pub struct PageRun {
    pub url: String,
    pub tier: String,
    pub blocked: bool,
    pub clean: bool,
    pub latency_ms: f64,
    pub detail: String,
}

/// Aggregated coverage metrics for one side over the URL set.
#[derive(Debug, Clone)]
pub struct SideSummary {
    pub total: usize,
    pub clean: usize,
    pub blocked_total: usize,
    pub blocked_clean: usize,
    pub p99_latency_ms: f64,
    /// $/1k pages. `None` for ours (API-free → compute-only; reported as a note,
    /// not a dollar figure unless the operator supplies a compute cost basis).
    pub cost_per_1k_usd: Option<f64>,
}

impl SideSummary {
    pub fn coverage(&self) -> f64 {
        if self.total == 0 { 0.0 } else { self.clean as f64 / self.total as f64 }
    }
    pub fn blocked_success(&self) -> f64 {
        if self.blocked_total == 0 {
            0.0
        } else {
            self.blocked_clean as f64 / self.blocked_total as f64
        }
    }
}

/// P99 latency over a sample (nearest-rank method on the sorted sample).
/// Returns 0.0 for an empty sample. Deterministic — no clock, pure input.
pub fn p99(latencies_ms: &[f64]) -> f64 {
    if latencies_ms.is_empty() {
        return 0.0;
    }
    let mut v: Vec<f64> = latencies_ms.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Nearest-rank: ceil(p/100 * N), 1-indexed.
    let rank = ((0.99 * v.len() as f64).ceil() as usize).max(1);
    v[rank - 1]
}

/// Build a [`SideSummary`] from raw page runs.
pub fn summarize(runs: &[PageRun], cost_per_1k_usd: Option<f64>) -> SideSummary {
    let total = runs.len();
    let clean = runs.iter().filter(|r| r.clean).count();
    let blocked: Vec<&PageRun> = runs.iter().filter(|r| r.blocked).collect();
    let blocked_total = blocked.len();
    let blocked_clean = blocked.iter().filter(|r| r.clean).count();
    let latencies: Vec<f64> = runs.iter().map(|r| r.latency_ms).collect();
    SideSummary {
        total,
        clean,
        blocked_total,
        blocked_clean,
        p99_latency_ms: p99(&latencies),
        cost_per_1k_usd,
    }
}

/// One G4 delta line with its pass/fail verdict.
#[derive(Debug, Clone)]
pub struct Delta {
    pub name: String,
    pub ours: String,
    pub firecrawl: String,
    pub delta: String,
    pub target: String,
    pub pass: Option<bool>, // None = cannot evaluate (e.g. placeholder run)
}

fn verdict(pass: Option<bool>) -> &'static str {
    match pass {
        Some(true) => "✅ PASS",
        Some(false) => "❌ FAIL",
        None => "⚠️ N/A (no real Firecrawl run)",
    }
}

/// Optional accuracy pair (G3 — MCPBench). Supplied by the operator/MCPBench
/// run; the coverage harness does not compute it, so it is wired in separately.
#[derive(Debug, Clone, Copy)]
pub struct AccuracyPair {
    pub ours: f64,
    pub firecrawl: f64,
}

/// Compute the four G4 deltas. `firecrawl` is `None` on a placeholder run (no
/// key) — then percentage-point deltas are reported as N/A but ours-side numbers
/// are still shown. `accuracy` is `None` until an MCPBench G3 pair is provided.
pub fn compute_deltas(
    ours: &SideSummary,
    firecrawl: Option<&SideSummary>,
    accuracy: Option<AccuracyPair>,
) -> Vec<Delta> {
    let mut out = Vec::new();
    let pct = |x: f64| format!("{:.1}%", x * 100.0);

    // 1. Coverage (≥ +5 pts).
    out.push(match firecrawl {
        Some(fc) => {
            let d = (ours.coverage() - fc.coverage()) * 100.0;
            Delta {
                name: "Coverage (G1)".into(),
                ours: pct(ours.coverage()),
                firecrawl: pct(fc.coverage()),
                delta: format!("{d:+.1} pts"),
                target: "≥ +5 pts".into(),
                pass: Some(d >= 5.0),
            }
        }
        None => Delta {
            name: "Coverage (G1)".into(),
            ours: pct(ours.coverage()),
            firecrawl: "—".into(),
            delta: "—".into(),
            target: "≥ +5 pts".into(),
            pass: None,
        },
    });

    // 2. Blocked-subset (≥ +5 pts).
    out.push(match firecrawl {
        Some(fc) => {
            let d = (ours.blocked_success() - fc.blocked_success()) * 100.0;
            Delta {
                name: "Blocked-subset (G2)".into(),
                ours: pct(ours.blocked_success()),
                firecrawl: pct(fc.blocked_success()),
                delta: format!("{d:+.1} pts"),
                target: "≥ +5 pts".into(),
                pass: Some(d >= 5.0),
            }
        }
        None => Delta {
            name: "Blocked-subset (G2)".into(),
            ours: pct(ours.blocked_success()),
            firecrawl: "—".into(),
            delta: "—".into(),
            target: "≥ +5 pts".into(),
            pass: None,
        },
    });

    // 3. MCPBench accuracy (≥ +5 pts) — only when supplied.
    out.push(match accuracy {
        Some(a) => {
            let d = (a.ours - a.firecrawl) * 100.0;
            Delta {
                name: "MCPBench accuracy (G3)".into(),
                ours: pct(a.ours),
                firecrawl: pct(a.firecrawl),
                delta: format!("{d:+.1} pts"),
                target: "≥ +5 pts".into(),
                pass: Some(d >= 5.0),
            }
        }
        None => Delta {
            name: "MCPBench accuracy (G3)".into(),
            ours: "(run MCPBench)".into(),
            firecrawl: "(run MCPBench)".into(),
            delta: "—".into(),
            target: "≥ +5 pts".into(),
            pass: None,
        },
    });

    // 4a. P99 latency per page (ours ≤ Firecrawl).
    out.push(match firecrawl {
        Some(fc) => Delta {
            name: "P99 latency / page".into(),
            ours: format!("{:.0} ms", ours.p99_latency_ms),
            firecrawl: format!("{:.0} ms", fc.p99_latency_ms),
            delta: format!("{:+.0} ms", ours.p99_latency_ms - fc.p99_latency_ms),
            target: "≤ Firecrawl".into(),
            pass: Some(ours.p99_latency_ms <= fc.p99_latency_ms),
        },
        None => Delta {
            name: "P99 latency / page".into(),
            ours: format!("{:.0} ms", ours.p99_latency_ms),
            firecrawl: "—".into(),
            delta: "—".into(),
            target: "≤ Firecrawl".into(),
            pass: None,
        },
    });

    // 4b. $/1k pages (ours strictly lower). Ours is API-free → compute-only.
    let ours_cost = ours
        .cost_per_1k_usd
        .map(|c| format!("${c:.2}"))
        .unwrap_or_else(|| "compute-only (API-free)".into());
    out.push(match (firecrawl.and_then(|f| f.cost_per_1k_usd), ours.cost_per_1k_usd) {
        (Some(fc_cost), Some(our_cost)) => Delta {
            name: "$ / 1k pages".into(),
            ours: format!("${our_cost:.2}"),
            firecrawl: format!("${fc_cost:.2}"),
            delta: format!("{:+.2}", our_cost - fc_cost),
            target: "strictly lower".into(),
            pass: Some(our_cost < fc_cost),
        },
        (Some(fc_cost), None) => Delta {
            name: "$ / 1k pages".into(),
            ours: ours_cost,
            firecrawl: format!("${fc_cost:.2}"),
            // API-free server has no per-page API fee; strictly lower holds by
            // construction once a compute basis ≥ 0 is below Firecrawl's fee,
            // but we do NOT assert PASS without a measured compute-cost figure.
            delta: "API-free vs paid API".into(),
            target: "strictly lower".into(),
            pass: None,
        },
        _ => Delta {
            name: "$ / 1k pages".into(),
            ours: ours_cost,
            firecrawl: "—".into(),
            delta: "—".into(),
            target: "strictly lower".into(),
            pass: None,
        },
    });

    out
}

/// Render the full `RESULTS.firecrawl.md` report.
#[allow(clippy::too_many_arguments)]
pub fn build_report(
    now_rfc3339: &str,
    git_sha: &str,
    urls_path: &str,
    server: &str,
    ours: &SideSummary,
    ours_runs: &[PageRun],
    firecrawl: Option<&SideSummary>,
    firecrawl_runs: &[PageRun],
    accuracy: Option<AccuracyPair>,
    firecrawl_skipped_reason: Option<&str>,
) -> String {
    let mut out = String::new();
    out.push_str("# G4 — Firecrawl Head-to-Head Comparison\n\n");
    out.push_str(&format!("- **Generated:** {now_rfc3339}\n"));
    out.push_str(&format!("- **Git SHA:** `{git_sha}`\n"));
    out.push_str(&format!("- **Inputs:** `{urls_path}` (identical set, both sides)\n"));
    out.push_str(&format!("- **Our server:** `{server}`\n"));
    // Reflect the ACTUAL base the run hit (self-hosted localhost for the G4
    // baseline, per GOAL.md §G4), not the cloud default — so the report does not
    // misrepresent a self-host run as a cloud-API run.
    let fc_base = std::env::var("FIRECRAWL_BASE_URL")
        .unwrap_or_else(|_| firecrawl::DEFAULT_BASE_URL.to_string());
    let fc_is_selfhost = fc_base.contains("localhost") || fc_base.contains("127.0.0.1");
    out.push_str(&format!(
        "- **Firecrawl:** v2 API (`{}`){}, baseline-comparison only — never a runtime dependency (GOAL.md Mission: API-free)\n",
        fc_base,
        if fc_is_selfhost { " — **self-hosted locally** (no Fire-engine)" } else { "" }
    ));
    out.push_str(&format!("- **Cost basis:** {}\n\n", firecrawl::COST_BASIS_NOTE));

    out.push_str(
        "> Firecrawl API contract verified 2026-06-18 against docs.firecrawl.dev \
         (scrape, crawl-post) + firecrawl.dev/pricing + github.com/firecrawl/firecrawl. \
         Adapter parses the verified contract via contract-mocked tests; the real-key \
         run is operator-invoked (no real external service in CI).\n\n",
    );

    if server.contains("mock") {
        out.push_str(
            "> ⚠️ **OURS-SIDE = MOCK SERVER.** This report's ours-side ran the hermetic \
             `mock-mcp-server` (no ML models / network), so the ours coverage numbers are \
             self-test placeholders, NOT a real measurement. Point `--server` at the real \
             `web-search-mcp` release binary for a true ours-side run.\n\n",
        );
    }

    if let Some(reason) = firecrawl_skipped_reason {
        out.push_str(&format!(
            "> ⚠️ **PLACEHOLDER RUN — Firecrawl not executed.** {reason}\n>\n\
             > The four G4 deltas below show ours-side numbers; Firecrawl columns and\n\
             > pass/fail are marked N/A until the operator provides `FIRECRAWL_API_KEY`\n\
             > and re-runs on identical inputs/hardware/date.\n\n",
        ));
    } else if fc_is_selfhost {
        // GOAL.md §G4 blocked-tier caveat — REQUIRED on any self-host run.
        out.push_str(
            "> ⚠️ **BLOCKED-TIER CAVEAT (GOAL.md §G4).** This Firecrawl side ran \
             **self-hosted**, which has **no Fire-engine**. Self-host is therefore weaker on \
             bot-protected pages than Firecrawl's paid cloud. The blocked-subset (G2) number \
             here is a **FLOOR for Firecrawl, not its cloud ceiling** — a G2 win over self-host \
             is **not** a win over cloud Firecrawl. A cloud-equivalent blocked-tier comparison \
             would require the paid cloud key (operator decision — TASKS 0.6b).\n\n",
        );
    }

    // ── G4 deltas summary table ──
    let deltas = compute_deltas(ours, firecrawl, accuracy);
    out.push_str("## G4 deltas (ours − Firecrawl)\n\n");
    out.push_str("| Metric | Ours | Firecrawl | Delta | Target | Verdict |\n");
    out.push_str("|--------|------|-----------|-------|--------|--------|\n");
    for d in &deltas {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} |\n",
            d.name, d.ours, d.firecrawl, d.delta, d.target, verdict(d.pass)
        ));
    }
    out.push('\n');

    let all_pass = deltas.iter().all(|d| d.pass == Some(true));
    let any_eval = deltas.iter().any(|d| d.pass.is_some());
    out.push_str(&format!(
        "**GATE 4 (G4 margins): {}**\n\n",
        if !any_eval || firecrawl.is_none() {
            "⚠️ NOT EVALUABLE — placeholder run (provide FIRECRAWL_API_KEY + MCPBench G3 pair)"
        } else if all_pass {
            "✅ all margins met"
        } else {
            "❌ one or more margins not met"
        }
    ));

    // ── Side-by-side coverage ──
    out.push_str("## Coverage detail\n\n");
    out.push_str("| | Ours | Firecrawl |\n|--|------|-----------|\n");
    out.push_str(&format!(
        "| Coverage (G1) | {}/{} = {:.1}% | {} |\n",
        ours.clean,
        ours.total,
        ours.coverage() * 100.0,
        firecrawl
            .map(|f| format!("{}/{} = {:.1}%", f.clean, f.total, f.coverage() * 100.0))
            .unwrap_or_else(|| "—".into()),
    ));
    out.push_str(&format!(
        "| Blocked-subset (G2) | {}/{} = {:.1}% | {} |\n",
        ours.blocked_clean,
        ours.blocked_total,
        ours.blocked_success() * 100.0,
        firecrawl
            .map(|f| format!("{}/{} = {:.1}%", f.blocked_clean, f.blocked_total, f.blocked_success() * 100.0))
            .unwrap_or_else(|| "—".into()),
    ));
    out.push_str(&format!(
        "| P99 latency/page | {:.0} ms | {} |\n",
        ours.p99_latency_ms,
        firecrawl
            .map(|f| format!("{:.0} ms", f.p99_latency_ms))
            .unwrap_or_else(|| "—".into()),
    ));
    out.push('\n');

    // ── Per-URL detail, both sides ──
    out.push_str("## Per-URL detail\n\n");
    out.push_str("### Ours\n\n");
    push_url_table(&mut out, ours_runs);
    out.push_str("\n### Firecrawl\n\n");
    if firecrawl_runs.is_empty() {
        out.push_str("_No Firecrawl run (placeholder)._\n");
    } else {
        push_url_table(&mut out, firecrawl_runs);
    }
    out.push('\n');

    out
}

fn push_url_table(out: &mut String, runs: &[PageRun]) {
    out.push_str("| URL | Tier | Blocked | Clean | Latency (ms) | Detail |\n");
    out.push_str("|-----|------|---------|-------|--------------|--------|\n");
    for r in runs {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {:.0} | {} |\n",
            r.url,
            r.tier,
            if r.blocked { "yes" } else { "no" },
            if r.clean { "✅" } else { "❌" },
            r.latency_ms,
            r.detail,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(url: &str, blocked: bool, clean: bool, lat: f64) -> PageRun {
        PageRun {
            url: url.into(),
            tier: if blocked { "cloudflare".into() } else { "static".into() },
            blocked,
            clean,
            latency_ms: lat,
            detail: String::new(),
        }
    }

    #[test]
    fn p99_nearest_rank_basic() {
        let xs: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        // ceil(0.99*100)=99 → 99th value (1-indexed) = 99.0
        assert_eq!(p99(&xs), 99.0);
        assert_eq!(p99(&[]), 0.0);
        assert_eq!(p99(&[42.0]), 42.0);
    }

    #[test]
    fn summarize_counts_clean_and_blocked() {
        let runs = vec![
            run("a", false, true, 100.0),
            run("b", false, false, 200.0),
            run("c", true, true, 300.0),
            run("d", true, false, 400.0),
        ];
        let s = summarize(&runs, None);
        assert_eq!(s.total, 4);
        assert_eq!(s.clean, 2);
        assert_eq!(s.blocked_total, 2);
        assert_eq!(s.blocked_clean, 1);
        assert!((s.coverage() - 0.5).abs() < 1e-9);
        assert!((s.blocked_success() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn deltas_pass_when_ours_beats_by_5pts_and_cheaper_and_faster() {
        // Ours: 95% cov, 60% blocked, 80ms p99, $0 (compute basis 0 supplied).
        let ours = SideSummary {
            total: 100, clean: 95, blocked_total: 10, blocked_clean: 6,
            p99_latency_ms: 80.0, cost_per_1k_usd: Some(0.05),
        };
        // Firecrawl: 80% cov, 40% blocked, 200ms p99, $0.83.
        let fc = SideSummary {
            total: 100, clean: 80, blocked_total: 10, blocked_clean: 4,
            p99_latency_ms: 200.0, cost_per_1k_usd: Some(0.83),
        };
        let acc = AccuracyPair { ours: 0.90, firecrawl: 0.80 };
        let deltas = compute_deltas(&ours, Some(&fc), Some(acc));
        assert!(deltas.iter().all(|d| d.pass == Some(true)),
            "all G4 margins should pass: {:?}",
            deltas.iter().map(|d| (d.name.clone(), d.pass)).collect::<Vec<_>>());
    }

    #[test]
    fn deltas_fail_when_margin_under_5pts() {
        let ours = SideSummary {
            total: 100, clean: 83, blocked_total: 10, blocked_clean: 5,
            p99_latency_ms: 100.0, cost_per_1k_usd: Some(0.10),
        };
        let fc = SideSummary {
            total: 100, clean: 80, blocked_total: 10, blocked_clean: 4,
            p99_latency_ms: 200.0, cost_per_1k_usd: Some(0.83),
        };
        // Coverage delta = +3 pts < 5 → coverage FAILs.
        let deltas = compute_deltas(&ours, Some(&fc), None);
        let cov = deltas.iter().find(|d| d.name.starts_with("Coverage")).unwrap();
        assert_eq!(cov.pass, Some(false));
    }

    #[test]
    fn deltas_na_on_placeholder_run() {
        let ours = SideSummary {
            total: 8, clean: 6, blocked_total: 2, blocked_clean: 0,
            p99_latency_ms: 500.0, cost_per_1k_usd: None,
        };
        let deltas = compute_deltas(&ours, None, None);
        // No Firecrawl side → every delta is N/A (None), ours numbers still set.
        assert!(deltas.iter().all(|d| d.pass.is_none()));
        let cov = deltas.iter().find(|d| d.name.starts_with("Coverage")).unwrap();
        assert_eq!(cov.ours, "75.0%");
    }

    #[test]
    fn report_marks_placeholder_and_is_not_evaluable() {
        let ours = summarize(&[run("a", false, true, 10.0)], None);
        let report = build_report(
            "2026-06-18T00:00:00Z", "deadbeef", "benchmark/urls.jsonl",
            "target/release/web-search-mcp", &ours, &[run("a", false, true, 10.0)],
            None, &[], None, Some("FIRECRAWL_API_KEY not set."),
        );
        assert!(report.contains("PLACEHOLDER RUN"));
        assert!(report.contains("NOT EVALUABLE"));
        assert!(report.contains("baseline-comparison only"));
        // Real server path (no "mock") ⇒ no mock-server warning.
        assert!(!report.contains("OURS-SIDE = MOCK SERVER"));
    }

    #[test]
    fn report_flags_mock_server_ours_side() {
        let ours = summarize(&[run("a", false, false, 10.0)], None);
        let report = build_report(
            "2026-06-18T00:00:00Z", "abc", "benchmark/urls.jsonl",
            "target/debug/mock-mcp-server", &ours, &[], None, &[], None,
            Some("no key"),
        );
        assert!(report.contains("OURS-SIDE = MOCK SERVER"));
    }

    #[test]
    fn report_all_pass_renders_gate_pass() {
        let ours = SideSummary {
            total: 10, clean: 10, blocked_total: 2, blocked_clean: 2,
            p99_latency_ms: 50.0, cost_per_1k_usd: Some(0.01),
        };
        let fc = SideSummary {
            total: 10, clean: 8, blocked_total: 2, blocked_clean: 1,
            p99_latency_ms: 300.0, cost_per_1k_usd: Some(0.83),
        };
        let report = build_report(
            "2026-06-18T00:00:00Z", "abc", "benchmark/urls.jsonl", "srv",
            &ours, &[], Some(&fc), &[],
            Some(AccuracyPair { ours: 0.9, firecrawl: 0.8 }), None,
        );
        assert!(report.contains("✅ all margins met"));
    }
}
