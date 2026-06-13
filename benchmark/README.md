# Benchmark Harness — Coverage + Accuracy

Implements [TASKS.md](../TASKS.md) task **0.3** against [GOAL.md](../GOAL.md)
criteria **G1** (coverage ≥ 90%), **G2** (blocked-subset success), and **G3**
(nDCG@10 + precision@5). It drives the MCP server over stdio (JSON-RPC 2.0) and
writes [`RESULTS.md`](RESULTS.md) with a timestamp and git SHA.

> **Operator: this section is the contract for your input files.** Match these
> schemas exactly. See [`urls.sample.jsonl`](urls.sample.jsonl) and
> [`queries.sample.jsonl`](queries.sample.jsonl) for ready-to-copy examples.

---

## 1. Input schemas (LOCKED)

Both files are **JSON Lines**: one JSON object per line, UTF-8. The harness
ignores blank lines and lines starting with `#` or `//` (so you may annotate),
but a real data file needs none of that.

### `benchmark/urls.jsonl` — for G1 (coverage) and G2 (blocked subset)

| Field             | Type    | Required | Meaning |
|-------------------|---------|----------|---------|
| `url`             | string  | **yes**  | Absolute URL to fetch + extract. |
| `tier`            | string  | **yes**  | One of: `static`, `spa`, `cloudflare`, `datadome`, `ratelimited`, `paywall`, `login`. Drives the per-tier breakdown. |
| `blocked`         | bool    | **yes**  | `true` if this URL sits behind a bot-protection / anti-scraping wall. Defines the G2 subset. |
| `expect_contains` | string  | no       | A short marker string that MUST appear in the extracted main content for the page to count as clean. Use it to prove the extractor returned the *right article*, not navigation chrome. Omit if you only care about length. |

```json
{"url": "https://example.com/", "tier": "static", "blocked": false, "expect_contains": "Example Domain"}
{"url": "https://shop.example/p/123", "tier": "cloudflare", "blocked": true, "expect_contains": "Add to cart"}
```

**Coverage rule (G1):** a page is *clean* when the extractor's `body_text` has
≥ **200 characters** (boilerplate already stripped by the consensus extractor)
**and**, if `expect_contains` is present, the body contains that marker
(case-insensitive). `coverage = clean_pages / total_pages`.

**Tier guidance** — aim for spread so the per-tier breakdown is meaningful:
- `static` — plain server-rendered HTML.
- `spa` — JS-rendered single-page app (needs browser fallback).
- `cloudflare` / `datadome` — JS-challenge / bot-detection vendors.
- `ratelimited` — returns 429 / throttles aggressively.
- `paywall` / `login` — content gated behind subscription or auth.

Set `blocked: true` for `cloudflare`, `datadome`, `ratelimited`, `paywall`,
`login` entries you expect to fight back; `false` for `static`/`spa`.

### `benchmark/queries.jsonl` — for G3 (accuracy)

| Field           | Type            | Required | Meaning |
|-----------------|-----------------|----------|---------|
| `query`         | string          | **yes**  | The search query to issue. |
| `relevant_urls` | array of string | **yes**  | Ground-truth relevant URLs (binary relevance). Order does not matter. |
| `notes`         | string          | no       | Free-text rationale. Not scored; for human review. |

```json
{"query": "rust async runtime", "relevant_urls": ["https://tokio.rs/", "https://docs.rs/tokio"], "notes": "tokio is canonical"}
```

**Accuracy rules (G3):**
- **nDCG@10** — binary gain `rel_i / log2(i+2)` over the top 10 ranked results;
  normalized by the ideal ranking (all relevant first).
- **precision@5** — relevant-in-top-5 **/ 5**. The denominator is always 5, so a
  query whose label set has fewer than 5 relevant URLs cannot reach 1.0 — this
  is the conventional definition and is intentional.
- **URL matching** is normalized: scheme, a leading `www.`, a trailing `/`, and
  any `#fragment` are ignored; the query string is kept. So
  `http://djangoproject.com` matches `https://www.djangoproject.com/`.

---

## 2. What the harness does

1. Spawns the MCP server binary as a child process and completes the MCP
   handshake via the `rmcp` client (stdio JSON-RPC 2.0).
2. **G1/G2:** for each `urls.jsonl` line, calls the `extract` tool, reads
   `body_text`, applies the coverage rule, and aggregates overall + per-tier +
   blocked-subset rates.
3. **G3:** for each `queries.jsonl` line, calls the search tool (default
   `quick_search`), reads the ranked `results[].url`, and computes nDCG@10 and
   precision@5 vs `relevant_urls`.
4. Writes `benchmark/RESULTS.md` with every number, a timestamp, and the git SHA.

Per-call timeouts and clean child shutdown are built in, so one hung URL cannot
wedge the run.

---

## 3. Why a workspace crate (not a script)

This is a Rust workspace and the server speaks MCP via `rmcp`. A workspace crate
(`crates/benchmark`) reuses the exact same `rmcp` client + child-process
transport the protocol expects, gets compile-time checking against the protocol
types, and runs with a single reproducible `cargo run -p web-search-benchmark`
— no separate runtime (Python/Node) or hand-rolled JSON-RPC framing to drift
out of sync with the server.

The scoring math lives in `src/metrics.rs` as pure functions with a unit-test
suite (the base of the test pyramid). A second binary, `mock-mcp-server`,
provides a hermetic server for the self-test (no network, no ML models) so the
harness can be validated deterministically and offline.

---

## 4. Running

### Real baseline (operator data + real server)

```bash
# 1. Build the real MCP server (downloads ML models on first run)
cargo build --release --bin web-search-mcp

# 2. Run the harness against operator data
cargo run -p web-search-benchmark --bin benchmark -- \
    --urls benchmark/urls.jsonl \
    --queries benchmark/queries.jsonl \
    --server target/release/web-search-mcp \
    --out benchmark/RESULTS.md
```

Useful flags: `--search-tool deep_research` (slower, full pipeline),
`--extract-tool fetch_page`, `--call-timeout-secs 120`, `--server-arg <arg>`
(repeatable), `--workdir <dir>` (cwd for the server, so it finds
`config/default.toml`). Run `cargo run -p web-search-benchmark -- --help` for all.

### Hermetic self-test (no network, no models)

Verifies the full JSON-RPC round-trip + metric math against the sample files:

```bash
# Build both bins
cargo build -p web-search-benchmark

# Run the harness against the mock server + sample data
cargo run -p web-search-benchmark --bin benchmark -- \
    --urls benchmark/urls.sample.jsonl \
    --queries benchmark/queries.sample.jsonl \
    --server target/debug/mock-mcp-server \
    --out benchmark/RESULTS.sample.md
```

Expected self-test numbers (deterministic):
`coverage = 2/3 (66.7%)`, blocked subset `0/1`,
`mean nDCG@10 ≈ 0.7853`, `mean precision@5 = 0.4000`.

### Unit tests (metric math)

```bash
cargo test -p web-search-benchmark
```

---

## 5. Files

| Path | Purpose |
|------|---------|
| `urls.jsonl` | **Operator-provided** G1/G2 input (not committed by harness). |
| `queries.jsonl` | **Operator-provided** G3 input. |
| `urls.sample.jsonl` | Format example (3 lines). |
| `queries.sample.jsonl` | Format example (2 lines). |
| `RESULTS.md` | Generated baseline report. |
| `../crates/benchmark/src/main.rs` | Harness (MCP client + aggregation + report). |
| `../crates/benchmark/src/metrics.rs` | Pure scoring functions + unit tests. |
| `../crates/benchmark/src/bin/mock_mcp_server.rs` | Hermetic mock server for the self-test. |
