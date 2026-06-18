//! G4 head-to-head comparison runner (TASKS.md 0.6 / 4.4, GOAL.md §G4).
//!
//! Runs OUR MCP server and (if `FIRECRAWL_API_KEY` is set) Firecrawl over the
//! SAME `urls.jsonl`, on the same machine, same invocation, and writes
//! `benchmark/RESULTS.firecrawl.md` with both runs side-by-side + the four G4
//! deltas and their pass/fail verdicts.
//!
//! Firecrawl is **baseline-comparison only** (GOAL.md Mission: API-free): its
//! adapter is never wired into the server runtime. The key is read from env at
//! runtime, never hardcoded/logged/cached. If the key is absent the Firecrawl
//! run is skipped cleanly and a clearly-marked PLACEHOLDER report is written.
//!
//! Run (ours + Firecrawl, operator-invoked):
//!   FIRECRAWL_API_KEY=fc-... \
//!   cargo run -p web-search-benchmark --bin firecrawl-compare -- \
//!       --urls benchmark/urls.jsonl \
//!       --server target/release/web-search-mcp \
//!       --out benchmark/RESULTS.firecrawl.md
//!
//! Ours-only / placeholder (no key): same command without FIRECRAWL_API_KEY.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use rmcp::ServiceExt;
use rmcp::model::CallToolRequestParams;
use rmcp::transport::{ConfigureCommandExt, TokioChildProcess};
use serde::Deserialize;

use web_search_benchmark::compare::{self, AccuracyPair, PageRun};
use web_search_benchmark::firecrawl::{self, FirecrawlClient};
use web_search_benchmark::metrics;

#[derive(Debug, Deserialize)]
struct UrlSpec {
    url: String,
    tier: String,
    #[serde(default)]
    blocked: bool,
    #[serde(default)]
    expect_contains: Option<String>,
}

struct Args {
    urls: String,
    server: String,
    server_args: Vec<String>,
    workdir: String,
    out: String,
    extract_tool: String,
    call_timeout_secs: u64,
    /// Optional G3 accuracy pair from a prior MCPBench run (ours,firecrawl).
    accuracy_ours: Option<f64>,
    accuracy_firecrawl: Option<f64>,
    /// Optional measured $/1k for ours (compute-only basis). Default: none →
    /// reported as "compute-only (API-free)" without asserting the $ verdict.
    ours_cost_per_1k: Option<f64>,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut a = Args {
            urls: "benchmark/urls.jsonl".into(),
            server: "target/release/web-search-mcp".into(),
            server_args: Vec::new(),
            workdir: ".".into(),
            out: "benchmark/RESULTS.firecrawl.md".into(),
            extract_tool: "extract".into(),
            call_timeout_secs: 120,
            accuracy_ours: None,
            accuracy_firecrawl: None,
            ours_cost_per_1k: None,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let mut next = || it.next().context(format!("missing value for {flag}"));
            match flag.as_str() {
                "--urls" => a.urls = next()?,
                "--server" => a.server = next()?,
                "--server-arg" => a.server_args.push(next()?),
                "--workdir" => a.workdir = next()?,
                "--out" => a.out = next()?,
                "--extract-tool" => a.extract_tool = next()?,
                "--call-timeout-secs" => a.call_timeout_secs = next()?.parse()?,
                "--accuracy-ours" => a.accuracy_ours = Some(next()?.parse()?),
                "--accuracy-firecrawl" => a.accuracy_firecrawl = Some(next()?.parse()?),
                "--ours-cost-per-1k" => a.ours_cost_per_1k = Some(next()?.parse()?),
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                other => anyhow::bail!("unknown flag: {other} (try --help)"),
            }
        }
        Ok(a)
    }
}

fn print_help() {
    eprintln!(
        "firecrawl-compare — G4 head-to-head (ours vs Firecrawl) on identical urls.jsonl\n\n\
         USAGE:\n  firecrawl-compare [--urls FILE] [--server BIN] [options]\n\n\
         OPTIONS:\n\
         \x20 --urls FILE              urls.jsonl (same set as G1/G2) [default: benchmark/urls.jsonl]\n\
         \x20 --server BIN             our MCP server [default: target/release/web-search-mcp]\n\
         \x20 --server-arg ARG         extra arg for our server (repeatable)\n\
         \x20 --workdir DIR            cwd for our server [default: .]\n\
         \x20 --out FILE               report [default: benchmark/RESULTS.firecrawl.md]\n\
         \x20 --extract-tool NAME      our extract tool [default: extract]\n\
         \x20 --call-timeout-secs N    per-call timeout [default: 120]\n\
         \x20 --accuracy-ours F        G3 (MCPBench) accuracy for ours, 0..1 (optional)\n\
         \x20 --accuracy-firecrawl F   G3 (MCPBench) accuracy for Firecrawl, 0..1 (optional)\n\
         \x20 --ours-cost-per-1k F     measured $/1k pages for ours (compute basis; optional)\n\n\
         Firecrawl runs ONLY when FIRECRAWL_API_KEY is set (baseline-only, never a runtime dep).\n\
         No key ⇒ clean skip + clearly-marked PLACEHOLDER report."
    );
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_target(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let args = Args::parse()?;
    let specs = read_jsonl::<UrlSpec>(&args.urls)
        .with_context(|| format!("reading urls file {}", args.urls))?;
    eprintln!("[g4] {} url(s) from {}", specs.len(), args.urls);

    // ── Our side: drive the MCP server over the same set ──────────────
    let ours_runs = run_ours(&args, &specs).await?;
    let ours_summary = compare::summarize(&ours_runs, args.ours_cost_per_1k);

    // ── Firecrawl side: operator-invoked path only ────────────────────
    let (fc_runs, fc_summary, skip_reason) = match FirecrawlClient::from_env()? {
        Some(client) => {
            eprintln!("[g4] FIRECRAWL_API_KEY present — running Firecrawl baseline");
            let runs = run_firecrawl(&client, &specs).await;
            let summary =
                compare::summarize(&runs, Some(firecrawl::COST_PER_1K_PAGES_USD));
            (runs, Some(summary), None)
        }
        None => {
            let reason = format!(
                "{} not set (or empty). Firecrawl is baseline-comparison only; \
                 set it to run the real baseline.",
                firecrawl::API_KEY_ENV
            );
            eprintln!("[g4] {reason} → writing PLACEHOLDER report (ours-only)");
            (Vec::new(), None, Some(reason))
        }
    };

    // ── G3 accuracy pair (from a prior MCPBench run, if supplied) ─────
    let accuracy = match (args.accuracy_ours, args.accuracy_firecrawl) {
        (Some(o), Some(f)) => Some(AccuracyPair { ours: o, firecrawl: f }),
        _ => None,
    };

    let report = compare::build_report(
        &chrono::Utc::now().to_rfc3339(),
        &git_sha(),
        &args.urls,
        &args.server,
        &ours_summary,
        &ours_runs,
        fc_summary.as_ref(),
        &fc_runs,
        accuracy,
        skip_reason.as_deref(),
    );

    if let Some(parent) = std::path::Path::new(&args.out).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&args.out, &report).with_context(|| format!("writing {}", args.out))?;
    eprintln!("[g4] wrote {}", args.out);
    println!("{report}");
    Ok(())
}

// ── Our side ─────────────────────────────────────────────────────────────────

async fn run_ours(args: &Args, specs: &[UrlSpec]) -> Result<Vec<PageRun>> {
    eprintln!("[g4] spawning our server: {} {:?}", args.server, args.server_args);
    let server_bin = args.server.clone();
    let server_args = args.server_args.clone();
    let workdir = args.workdir.clone();
    let transport = TokioChildProcess::new(tokio::process::Command::new(&server_bin).configure(
        |cmd| {
            cmd.args(&server_args);
            cmd.current_dir(&workdir);
        },
    ))
    .with_context(|| format!("failed to spawn server binary `{server_bin}`"))?;
    let client = ().serve(transport).await.context("MCP handshake failed")?;
    let timeout = Duration::from_secs(args.call_timeout_secs);

    let mut runs = Vec::new();
    for spec in specs {
        let mut map = serde_json::Map::new();
        map.insert("url".into(), serde_json::Value::String(spec.url.clone()));
        let started = Instant::now();
        let (clean, detail) = match call_tool_text(&client, &args.extract_tool, map, timeout).await {
            Ok(text) => match serde_json::from_str::<serde_json::Value>(&text) {
                Ok(json) => {
                    let body = json.get("body_text").and_then(|v| v.as_str()).unwrap_or("");
                    (
                        metrics::is_clean(body, spec.expect_contains.as_deref()),
                        format!("body_chars={}", body.chars().count()),
                    )
                }
                Err(_) => (false, truncate(&text, 80)),
            },
            Err(e) => (false, format!("call_error: {e}")),
        };
        let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
        eprintln!("  [ours] {} -> {}", spec.url, if clean { "CLEAN" } else { "miss" });
        runs.push(PageRun {
            url: spec.url.clone(),
            tier: spec.tier.clone(),
            blocked: spec.blocked,
            clean,
            latency_ms,
            detail,
        });
    }
    let _ = client.cancel().await;
    Ok(runs)
}

async fn call_tool_text(
    client: &rmcp::service::RunningService<rmcp::RoleClient, ()>,
    tool: &str,
    arguments: serde_json::Map<String, serde_json::Value>,
    timeout: Duration,
) -> Result<String> {
    let params = CallToolRequestParams {
        name: tool.to_string().into(),
        arguments: Some(arguments),
        meta: None,
        task: None,
    };
    let result = tokio::time::timeout(timeout, client.call_tool(params))
        .await
        .context("tool call timed out")?
        .context("tool call failed")?;
    let text = result
        .content
        .iter()
        .find_map(|c| c.as_text().map(|t| t.text.clone()))
        .unwrap_or_default();
    if result.is_error.unwrap_or(false) {
        anyhow::bail!("tool reported error: {}", truncate(&text, 120));
    }
    Ok(text)
}

// ── Firecrawl side (operator-invoked) ─────────────────────────────────────────

async fn run_firecrawl(client: &FirecrawlClient, specs: &[UrlSpec]) -> Vec<PageRun> {
    let mut runs = Vec::new();
    for spec in specs {
        let started = Instant::now();
        let page = client.scrape(&spec.url).await;
        let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
        let clean = metrics::is_clean(&page.text, spec.expect_contains.as_deref());
        let detail = match &page.error {
            Some(e) => truncate(e, 80),
            None => format!(
                "chars={} status={}",
                page.text.chars().count(),
                page.status_code.map(|s| s.to_string()).unwrap_or_else(|| "?".into())
            ),
        };
        eprintln!("  [firecrawl] {} -> {}", spec.url, if clean { "CLEAN" } else { "miss" });
        runs.push(PageRun {
            url: spec.url.clone(),
            tier: spec.tier.clone(),
            blocked: spec.blocked,
            clean,
            latency_ms,
            detail,
        });
    }
    runs
}

// ── Helpers ────────────────────────────────────────────────────────────────────

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &str) -> Result<Vec<T>> {
    let content = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for (i, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("//") {
            continue;
        }
        let item: T = serde_json::from_str(trimmed)
            .with_context(|| format!("{path}:{} invalid JSON", i + 1))?;
        out.push(item);
    }
    Ok(out)
}

fn git_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn truncate(s: &str, max: usize) -> String {
    let one_line = s.replace('\n', " ");
    if one_line.chars().count() <= max {
        one_line
    } else {
        let t: String = one_line.chars().take(max).collect();
        format!("{t}…")
    }
}
