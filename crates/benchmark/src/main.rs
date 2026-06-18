//! Coverage + accuracy benchmark harness for the web-search MCP server.
//!
//! Implements TASKS.md task 0.3 against GOAL.md criteria:
//!   * G1 — coverage% of pages with clean main content (+ per-tier breakdown)
//!   * G2 — success rate on the `blocked: true` subset
//!   * G3 — nDCG@10 and precision@5 of ranked search results vs labels
//!
//! It spawns (or attaches to) an MCP server over stdio, drives it with the
//! standard JSON-RPC 2.0 / MCP handshake via the `rmcp` client, and writes
//! `benchmark/RESULTS.md` with a timestamp and the current git SHA.
//!
//! Run:
//!   cargo run -p web-search-benchmark --bin benchmark -- \
//!       --urls benchmark/urls.jsonl \
//!       --queries benchmark/queries.jsonl \
//!       --server target/release/web-search-mcp \
//!       --out benchmark/RESULTS.md
//!
//! Self-test (hermetic, no network/models) uses the mock server — see
//! benchmark/README.md.

use web_search_benchmark::metrics;

use std::collections::BTreeMap;
use std::time::Duration;

use anyhow::{Context, Result};
use rmcp::ServiceExt;
use rmcp::model::CallToolRequestParams;
use rmcp::transport::{ConfigureCommandExt, TokioChildProcess};
use serde::Deserialize;

const NDCG_K: usize = 10;
const PRECISION_K: usize = 5;

// ── Input schemas (locked — keep in sync with benchmark/README.md) ──────────

/// One line of `urls.jsonl` (G1/G2).
#[derive(Debug, Deserialize)]
struct UrlSpec {
    url: String,
    /// static | spa | cloudflare | datadome | ratelimited | paywall | login
    tier: String,
    #[serde(default)]
    blocked: bool,
    /// Optional marker that MUST appear in the extracted body for it to count
    /// as clean (verifies the extractor returned the right article, not chrome).
    #[serde(default)]
    expect_contains: Option<String>,
}

/// One line of `queries.jsonl` (G3).
#[derive(Debug, Deserialize)]
struct QuerySpec {
    query: String,
    #[serde(default)]
    relevant_urls: Vec<String>,
    #[serde(default)]
    #[allow(dead_code)] // notes are documentation for the operator, not scored
    notes: Option<String>,
}

// ── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    urls: Option<String>,
    queries: Option<String>,
    server: String,
    server_args: Vec<String>,
    workdir: String,
    out: String,
    search_tool: String,
    extract_tool: String,
    call_timeout_secs: u64,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut a = Args {
            urls: None,
            queries: None,
            server: "target/release/web-search-mcp".to_string(),
            server_args: Vec::new(),
            workdir: ".".to_string(),
            out: "benchmark/RESULTS.md".to_string(),
            search_tool: "quick_search".to_string(),
            extract_tool: "extract".to_string(),
            call_timeout_secs: 60,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let mut next = || it.next().context(format!("missing value for {flag}"));
            match flag.as_str() {
                "--urls" => a.urls = Some(next()?),
                "--queries" => a.queries = Some(next()?),
                "--server" => a.server = next()?,
                "--server-arg" => a.server_args.push(next()?),
                "--workdir" => a.workdir = next()?,
                "--out" => a.out = next()?,
                "--search-tool" => a.search_tool = next()?,
                "--extract-tool" => a.extract_tool = next()?,
                "--call-timeout-secs" => a.call_timeout_secs = next()?.parse()?,
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                other => anyhow::bail!("unknown flag: {other} (try --help)"),
            }
        }
        if a.urls.is_none() && a.queries.is_none() {
            anyhow::bail!("provide at least one of --urls / --queries (try --help)");
        }
        Ok(a)
    }
}

fn print_help() {
    eprintln!(
        "benchmark — coverage + accuracy harness for the web-search MCP server\n\n\
         USAGE:\n  benchmark [--urls FILE] [--queries FILE] [--server BIN] [options]\n\n\
         OPTIONS:\n\
         \x20 --urls FILE              urls.jsonl  (G1/G2 coverage)\n\
         \x20 --queries FILE           queries.jsonl (G3 accuracy)\n\
         \x20 --server BIN             MCP server binary to spawn [default: target/release/web-search-mcp]\n\
         \x20 --server-arg ARG         extra arg passed to the server (repeatable)\n\
         \x20 --workdir DIR            cwd for the spawned server [default: .]\n\
         \x20 --out FILE               results markdown [default: benchmark/RESULTS.md]\n\
         \x20 --search-tool NAME       tool for G3 [default: quick_search]\n\
         \x20 --extract-tool NAME      tool for G1/G2 [default: extract]\n\
         \x20 --call-timeout-secs N    per-tool-call timeout [default: 60]"
    );
}

// ── Aggregates ──────────────────────────────────────────────────────────────

#[derive(Default, Clone)]
struct TierStat {
    total: usize,
    clean: usize,
}

struct UrlOutcome {
    url: String,
    tier: String,
    blocked: bool,
    clean: bool,
    detail: String,
}

struct QueryOutcome {
    query: String,
    ndcg: f64,
    precision: f64,
    num_relevant: usize,
    num_results: usize,
    error: Option<String>,
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

    // Spawn the MCP server as a child process and complete the MCP handshake.
    eprintln!("[harness] spawning server: {} {:?}", args.server, args.server_args);
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
    eprintln!("[harness] connected. server info: {:?}", client.peer_info());

    let timeout = Duration::from_secs(args.call_timeout_secs);

    // ── G1/G2 — coverage ─────────────────────────────────────────────
    let mut url_outcomes: Vec<UrlOutcome> = Vec::new();
    if let Some(path) = &args.urls {
        let specs = read_jsonl::<UrlSpec>(path)
            .with_context(|| format!("reading urls file {path}"))?;
        eprintln!("[harness] G1/G2: {} url(s) via tool `{}`", specs.len(), args.extract_tool);
        for spec in specs {
            let outcome = run_url(&client, &args.extract_tool, &spec, timeout).await;
            eprintln!(
                "  [{}] {} -> {}",
                spec_tier_label(&outcome.tier, outcome.blocked),
                outcome.url,
                if outcome.clean { "CLEAN" } else { "miss" }
            );
            url_outcomes.push(outcome);
        }
    }

    // ── G3 — accuracy ────────────────────────────────────────────────
    let mut query_outcomes: Vec<QueryOutcome> = Vec::new();
    if let Some(path) = &args.queries {
        let specs = read_jsonl::<QuerySpec>(path)
            .with_context(|| format!("reading queries file {path}"))?;
        eprintln!("[harness] G3: {} quer(y/ies) via tool `{}`", specs.len(), args.search_tool);
        for spec in specs {
            let outcome = run_query(&client, &args.search_tool, &spec, timeout).await;
            eprintln!(
                "  q=\"{}\" nDCG@{NDCG_K}={:.4} P@{PRECISION_K}={:.4}{}",
                outcome.query,
                outcome.ndcg,
                outcome.precision,
                outcome
                    .error
                    .as_ref()
                    .map(|e| format!(" ERROR: {e}"))
                    .unwrap_or_default()
            );
            query_outcomes.push(outcome);
        }
    }

    // Clean shutdown of the child server.
    let _ = client.cancel().await;

    let report = build_report(&args, &url_outcomes, &query_outcomes);
    if let Some(parent) = std::path::Path::new(&args.out).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&args.out, &report).with_context(|| format!("writing {}", args.out))?;
    eprintln!("[harness] wrote {}", args.out);
    println!("{report}");
    Ok(())
}

// ── Per-item drivers ─────────────────────────────────────────────────────────

async fn run_url(
    client: &rmcp::service::RunningService<rmcp::RoleClient, ()>,
    tool: &str,
    spec: &UrlSpec,
    timeout: Duration,
) -> UrlOutcome {
    let mut map = serde_json::Map::new();
    map.insert("url".into(), serde_json::Value::String(spec.url.clone()));

    let (clean, detail) = match call_tool_text(client, tool, map, timeout).await {
        Ok(text) => match serde_json::from_str::<serde_json::Value>(&text) {
            Ok(json) => {
                let body = json
                    .get("body_text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let clean = metrics::is_clean(body, spec.expect_contains.as_deref());
                (
                    clean,
                    format!("body_chars={}", body.chars().count()),
                )
            }
            // Tool returned non-JSON (e.g. an "[ERROR] ..." string) => not clean.
            Err(_) => (false, truncate(&text, 80)),
        },
        Err(e) => (false, format!("call_error: {e}")),
    };

    UrlOutcome {
        url: spec.url.clone(),
        tier: spec.tier.clone(),
        blocked: spec.blocked,
        clean,
        detail,
    }
}

async fn run_query(
    client: &rmcp::service::RunningService<rmcp::RoleClient, ()>,
    tool: &str,
    spec: &QuerySpec,
    timeout: Duration,
) -> QueryOutcome {
    let mut map = serde_json::Map::new();
    map.insert("query".into(), serde_json::Value::String(spec.query.clone()));
    map.insert("max_results".into(), serde_json::Value::from(NDCG_K as u64));

    match call_tool_text(client, tool, map, timeout).await {
        Ok(text) => {
            let ranked = parse_result_urls(&text);
            // BENCH_DUMP=1 → print the ranked result URLs (and a hit marker vs the
            // labels) to stderr, so a real query's output can be eyeballed.
            if std::env::var("BENCH_DUMP").is_ok() {
                let rels = metrics::relevance_vector(&ranked, &spec.relevant_urls);
                eprintln!("[dump] query: {}", spec.query);
                for (i, u) in ranked.iter().enumerate() {
                    let hit = rels.get(i).copied().unwrap_or(0) > 0;
                    eprintln!("[dump]  {:>2}. {} {}", i + 1, if hit { "✓" } else { " " }, u);
                }
            }
            let rels = metrics::relevance_vector(&ranked, &spec.relevant_urls);
            QueryOutcome {
                query: spec.query.clone(),
                ndcg: metrics::ndcg_at_k(&rels, NDCG_K),
                precision: metrics::precision_at_k(&rels, PRECISION_K),
                num_relevant: spec.relevant_urls.len(),
                num_results: ranked.len(),
                error: None,
            }
        }
        Err(e) => QueryOutcome {
            query: spec.query.clone(),
            ndcg: 0.0,
            precision: 0.0,
            num_relevant: spec.relevant_urls.len(),
            num_results: 0,
            error: Some(e.to_string()),
        },
    }
}

/// Call a tool and return its first text content block, erroring if the tool
/// signalled `is_error`, the call timed out, or no text came back.
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

/// Extract ranked result URLs from a search tool's JSON payload.
/// Search tools return `{ "results": [ { "url": "..." }, ... ] }`.
fn parse_result_urls(text: &str) -> Vec<String> {
    let json: serde_json::Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    json.get("results")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|r| r.get("url").and_then(|u| u.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

// ── Reporting ────────────────────────────────────────────────────────────────

fn build_report(
    args: &Args,
    urls: &[UrlOutcome],
    queries: &[QueryOutcome],
) -> String {
    let now = chrono::Utc::now().to_rfc3339();
    let git_sha = git_sha();

    let mut out = String::new();
    out.push_str("# Benchmark Results\n\n");
    out.push_str(&format!("- **Generated:** {now}\n"));
    out.push_str(&format!("- **Git SHA:** `{git_sha}`\n"));
    out.push_str(&format!("- **Server:** `{}`", args.server));
    if !args.server_args.is_empty() {
        out.push_str(&format!(" {:?}", args.server_args));
    }
    out.push('\n');
    out.push_str(&format!(
        "- **Tools:** extract=`{}`, search=`{}`\n",
        args.extract_tool, args.search_tool
    ));
    out.push_str(&format!(
        "- **Metrics:** coverage clean-threshold = {} chars; nDCG@{NDCG_K}; precision@{PRECISION_K}\n\n",
        metrics::MIN_CLEAN_CHARS
    ));
    out.push_str("> Baseline run. G1 target = 0.90. G2/G3 targets are operator-set after baseline (GOAL.md Phase 4 gate).\n\n");

    // ── G1/G2 ──
    if !urls.is_empty() {
        let total = urls.len();
        let clean = urls.iter().filter(|u| u.clean).count();
        let coverage = clean as f64 / total as f64;
        out.push_str("## G1 — Coverage\n\n");
        out.push_str(&format!(
            "**coverage = {clean}/{total} = {:.4} ({:.1}%)** — target ≥ 0.90 {}\n\n",
            coverage,
            coverage * 100.0,
            if coverage >= 0.90 { "✅ PASS" } else { "❌ below target" }
        ));

        // Per-tier breakdown.
        let mut tiers: BTreeMap<String, TierStat> = BTreeMap::new();
        for u in urls {
            let e = tiers.entry(u.tier.clone()).or_default();
            e.total += 1;
            if u.clean {
                e.clean += 1;
            }
        }
        out.push_str("### Per-tier breakdown\n\n");
        out.push_str("| Tier | Clean | Total | Coverage |\n|------|-------|-------|----------|\n");
        for (tier, s) in &tiers {
            out.push_str(&format!(
                "| {tier} | {} | {} | {:.1}% |\n",
                s.clean,
                s.total,
                100.0 * s.clean as f64 / s.total as f64
            ));
        }
        out.push('\n');

        // ── G2 blocked subset ──
        let blocked: Vec<&UrlOutcome> = urls.iter().filter(|u| u.blocked).collect();
        out.push_str("## G2 — Blocked-subset success\n\n");
        if blocked.is_empty() {
            out.push_str("_No URLs tagged `blocked: true` in this set._\n\n");
        } else {
            let bclean = blocked.iter().filter(|u| u.clean).count();
            out.push_str(&format!(
                "**blocked success = {bclean}/{} = {:.1}%** — target: operator-set (Phase 3 gate)\n\n",
                blocked.len(),
                100.0 * bclean as f64 / blocked.len() as f64
            ));
        }

        // Per-URL detail.
        out.push_str("### Per-URL detail\n\n");
        out.push_str("| URL | Tier | Blocked | Clean | Detail |\n|-----|------|---------|-------|--------|\n");
        for u in urls {
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} |\n",
                u.url,
                u.tier,
                if u.blocked { "yes" } else { "no" },
                if u.clean { "✅" } else { "❌" },
                u.detail
            ));
        }
        out.push('\n');
    }

    // ── G3 ──
    if !queries.is_empty() {
        let n = queries.len() as f64;
        let mean_ndcg = queries.iter().map(|q| q.ndcg).sum::<f64>() / n;
        let mean_p = queries.iter().map(|q| q.precision).sum::<f64>() / n;
        out.push_str("## G3 — Accuracy\n\n");
        out.push_str(&format!(
            "**mean nDCG@{NDCG_K} = {mean_ndcg:.4}**, **mean precision@{PRECISION_K} = {mean_p:.4}** (over {} quer{}) — targets operator-set\n\n",
            queries.len(),
            if queries.len() == 1 { "y" } else { "ies" }
        ));
        out.push_str("### Per-query detail\n\n");
        out.push_str(&format!(
            "| Query | nDCG@{NDCG_K} | P@{PRECISION_K} | #relevant | #results | Note |\n|-------|--------|------|-----------|----------|------|\n"
        ));
        for q in queries {
            out.push_str(&format!(
                "| {} | {:.4} | {:.4} | {} | {} | {} |\n",
                q.query,
                q.ndcg,
                q.precision,
                q.num_relevant,
                q.num_results,
                q.error.clone().unwrap_or_default()
            ));
        }
        out.push('\n');
    }

    out
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &str) -> Result<Vec<T>> {
    let content = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for (i, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        // Tolerate blank lines and `#` / `//` comments (sample files annotate).
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
        let truncated: String = one_line.chars().take(max).collect();
        format!("{truncated}…")
    }
}

fn spec_tier_label(tier: &str, blocked: bool) -> String {
    if blocked {
        format!("{tier},blocked")
    } else {
        tier.to_string()
    }
}
