# G4 — Firecrawl Head-to-Head Comparison

- **Generated:** 2026-06-18T13:40:37.798623400+00:00
- **Git SHA:** `cd2ba77623c50e380761cc034c5ffebb445421a3`
- **Inputs:** `benchmark/urls.jsonl` (identical set, both sides)
- **Our server:** `target/debug/mock-mcp-server`
- **Firecrawl:** v2 API (`https://api.firecrawl.dev/v2`), baseline-comparison only — never a runtime dependency (GOAL.md Mission: API-free)
- **Cost basis:** Firecrawl Standard plan $83 / 100,000 credits, 1 credit = 1 page (base markdown path) ⇒ $0.83 / 1k pages (verified 2026-06-18, firecrawl.dev/pricing). Enhanced/stealth proxy = 4 credits/page.

> Firecrawl API contract verified 2026-06-18 against docs.firecrawl.dev (scrape, crawl-post) + firecrawl.dev/pricing + github.com/firecrawl/firecrawl. Adapter parses the verified contract via contract-mocked tests; the real-key run is operator-invoked (no real external service in CI).

> ⚠️ **OURS-SIDE = MOCK SERVER.** This report's ours-side ran the hermetic `mock-mcp-server` (no ML models / network), so the ours coverage numbers are self-test placeholders, NOT a real measurement. Point `--server` at the real `web-search-mcp` release binary for a true ours-side run.

> ⚠️ **PLACEHOLDER RUN — Firecrawl not executed.** FIRECRAWL_API_KEY not set (or empty). Firecrawl is baseline-comparison only; set it to run the real baseline.
>
> The four G4 deltas below show ours-side numbers; Firecrawl columns and
> pass/fail are marked N/A until the operator provides `FIRECRAWL_API_KEY`
> and re-runs on identical inputs/hardware/date.

## G4 deltas (ours − Firecrawl)

| Metric | Ours | Firecrawl | Delta | Target | Verdict |
|--------|------|-----------|-------|--------|--------|
| Coverage (G1) | 0.0% | — | — | ≥ +5 pts | ⚠️ N/A (no real Firecrawl run) |
| Blocked-subset (G2) | 0.0% | — | — | ≥ +5 pts | ⚠️ N/A (no real Firecrawl run) |
| MCPBench accuracy (G3) | (run MCPBench) | (run MCPBench) | — | ≥ +5 pts | ⚠️ N/A (no real Firecrawl run) |
| P99 latency / page | 5 ms | — | — | ≤ Firecrawl | ⚠️ N/A (no real Firecrawl run) |
| $ / 1k pages | compute-only (API-free) | — | — | strictly lower | ⚠️ N/A (no real Firecrawl run) |

**GATE 4 (G4 margins): ⚠️ NOT EVALUABLE — placeholder run (provide FIRECRAWL_API_KEY + MCPBench G3 pair)**

## Coverage detail

| | Ours | Firecrawl |
|--|------|-----------|
| Coverage (G1) | 0/8 = 0.0% | — |
| Blocked-subset (G2) | 0/2 = 0.0% | — |
| P99 latency/page | 5 ms | — |

## Per-URL detail

### Ours

| URL | Tier | Blocked | Clean | Latency (ms) | Detail |
|-----|------|---------|-------|--------------|--------|
| https://www.rust-lang.org/ | static | no | ❌ | 5 | body_chars=3 |
| https://en.wikipedia.org/wiki/Rust_(programming_language) | static | no | ❌ | 2 | body_chars=3 |
| https://doc.rust-lang.org/book/ | static | no | ❌ | 1 | body_chars=3 |
| https://blog.rust-lang.org/ | static | no | ❌ | 2 | body_chars=3 |
| https://nextjs.org/ | spa | no | ❌ | 2 | body_chars=3 |
| https://react.dev/ | spa | no | ❌ | 2 | body_chars=3 |
| https://www.g2.com/products/notion/reviews | cloudflare | yes | ❌ | 2 | body_chars=3 |
| https://www.crunchbase.com/organization/openai | cloudflare | yes | ❌ | 2 | body_chars=3 |

### Firecrawl

_No Firecrawl run (placeholder)._

