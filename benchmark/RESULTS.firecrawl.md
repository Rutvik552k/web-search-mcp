# G4 — Firecrawl Head-to-Head Comparison

- **Generated:** 2026-06-18T14:23:02.385908500+00:00
- **Git SHA:** `c67dac0556692852aa54efe7b8871d8771820c8f`
- **Inputs:** `benchmark/urls.jsonl` (identical set, both sides)
- **Our server:** `target/release/web-search-mcp.exe`
- **Firecrawl:** v2 API (`http://localhost:3002/v2`) — **self-hosted locally** (no Fire-engine), baseline-comparison only — never a runtime dependency (GOAL.md Mission: API-free)
- **Cost basis:** Firecrawl Standard plan $83 / 100,000 credits, 1 credit = 1 page (base markdown path) ⇒ $0.83 / 1k pages (verified 2026-06-18, firecrawl.dev/pricing). Enhanced/stealth proxy = 4 credits/page.

> Firecrawl API contract verified 2026-06-18 against docs.firecrawl.dev (scrape, crawl-post) + firecrawl.dev/pricing + github.com/firecrawl/firecrawl. Adapter parses the verified contract via contract-mocked tests; the real-key run is operator-invoked (no real external service in CI).

> ⚠️ **BLOCKED-TIER CAVEAT (GOAL.md §G4).** This Firecrawl side ran **self-hosted**, which has **no Fire-engine**. Self-host is therefore weaker on bot-protected pages than Firecrawl's paid cloud. The blocked-subset (G2) number here is a **FLOOR for Firecrawl, not its cloud ceiling** — a G2 win over self-host is **not** a win over cloud Firecrawl. A cloud-equivalent blocked-tier comparison would require the paid cloud key (operator decision — TASKS 0.6b).

## G4 deltas (ours − Firecrawl)

| Metric | Ours | Firecrawl | Delta | Target | Verdict |
|--------|------|-----------|-------|--------|--------|
| Coverage (G1) | 62.5% | 75.0% | -12.5 pts | ≥ +5 pts | ❌ FAIL |
| Blocked-subset (G2) | 0.0% | 0.0% | +0.0 pts | ≥ +5 pts | ❌ FAIL |
| MCPBench accuracy (G3) | (run MCPBench) | (run MCPBench) | — | ≥ +5 pts | ⚠️ N/A (no real Firecrawl run) |
| P99 latency / page | 565 ms | 7339 ms | -6774 ms | ≤ Firecrawl | ✅ PASS |
| $ / 1k pages | compute-only (API-free) | $0.83 | API-free vs paid API | strictly lower | ⚠️ N/A (no real Firecrawl run) |

**GATE 4 (G4 margins): ❌ one or more margins not met**

## Coverage detail

| | Ours | Firecrawl |
|--|------|-----------|
| Coverage (G1) | 5/8 = 62.5% | 6/8 = 75.0% |
| Blocked-subset (G2) | 0/2 = 0.0% | 0/2 = 0.0% |
| P99 latency/page | 565 ms | 7339 ms |

## Per-URL detail

### Ours

| URL | Tier | Blocked | Clean | Latency (ms) | Detail |
|-----|------|---------|-------|--------------|--------|
| https://www.rust-lang.org/ | static | no | ✅ | 565 | body_chars=2257 |
| https://en.wikipedia.org/wiki/Rust_(programming_language) | static | no | ❌ | 434 | body_chars=1531 |
| https://doc.rust-lang.org/book/ | static | no | ✅ | 228 | body_chars=944 |
| https://blog.rust-lang.org/ | static | no | ✅ | 119 | body_chars=552 |
| https://nextjs.org/ | spa | no | ✅ | 288 | body_chars=1623 |
| https://react.dev/ | spa | no | ✅ | 241 | body_chars=3751 |
| https://www.g2.com/products/notion/reviews | cloudflare | yes | ❌ | 130 | call_error: tool reported error: [ERROR] blocked by site: https://www.g2.com/products/notion/reviews (status 403) |
| https://www.crunchbase.com/organization/openai | cloudflare | yes | ❌ | 273 | call_error: tool reported error: [ERROR] blocked by site: https://www.crunchbase.com/organization/openai (status 403) |

### Firecrawl

| URL | Tier | Blocked | Clean | Latency (ms) | Detail |
|-----|------|---------|-------|--------------|--------|
| https://www.rust-lang.org/ | static | no | ✅ | 1195 | chars=3466 status=200 |
| https://en.wikipedia.org/wiki/Rust_(programming_language) | static | no | ✅ | 3508 | chars=215385 status=200 |
| https://doc.rust-lang.org/book/ | static | no | ✅ | 1354 | chars=1672 status=200 |
| https://blog.rust-lang.org/ | static | no | ✅ | 7339 | chars=40595 status=200 |
| https://nextjs.org/ | spa | no | ✅ | 2263 | chars=15690 status=200 |
| https://react.dev/ | spa | no | ✅ | 1961 | chars=16638 status=200 |
| https://www.g2.com/products/notion/reviews | cloudflare | yes | ❌ | 2258 | HTTP 500: {"success":false,"code":"SCRAPE_RETRY_LIMIT","error":"Scrape aborted a… |
| https://www.crunchbase.com/organization/openai | cloudflare | yes | ❌ | 1581 | HTTP 500: {"success":false,"code":"SCRAPE_RETRY_LIMIT","error":"Scrape aborted a… |

