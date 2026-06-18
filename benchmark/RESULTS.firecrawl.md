# G4 — Firecrawl Head-to-Head Comparison

- **Generated:** 2026-06-18T22:03:55.564244100+00:00
- **Git SHA:** `c026dbbbb299a1fb3a27370ccdad6cadd3efb585`
- **Inputs:** `benchmark/urls.jsonl` (identical set, both sides)
- **Our server:** `target/release/web-search-mcp.exe`
- **Firecrawl:** v2 API (`http://localhost:3002/v2`) — **self-hosted locally** (no Fire-engine), baseline-comparison only — never a runtime dependency (GOAL.md Mission: API-free)
- **Cost basis:** Firecrawl Standard plan $83 / 100,000 credits, 1 credit = 1 page (base markdown path) ⇒ $0.83 / 1k pages (verified 2026-06-18, firecrawl.dev/pricing). Enhanced/stealth proxy = 4 credits/page.

> Firecrawl API contract verified 2026-06-18 against docs.firecrawl.dev (scrape, crawl-post) + firecrawl.dev/pricing + github.com/firecrawl/firecrawl. Adapter parses the verified contract via contract-mocked tests; the real-key run is operator-invoked (no real external service in CI).

> ⚠️ **BLOCKED-TIER CAVEAT (GOAL.md §G4).** This Firecrawl side ran **self-hosted**, which has **no Fire-engine**. Self-host is therefore weaker on bot-protected pages than Firecrawl's paid cloud. The blocked-subset (G2) number here is a **FLOOR for Firecrawl, not its cloud ceiling** — a G2 win over self-host is **not** a win over cloud Firecrawl. A cloud-equivalent blocked-tier comparison would require the paid cloud key (operator decision — TASKS 0.6b).

## G4 deltas (ours − Firecrawl)

| Metric | Ours | Firecrawl | Delta | Target | Verdict |
|--------|------|-----------|-------|--------|--------|
| Coverage (G1) | 75.0% | 75.0% | +0.0 pts | ≥ +5 pts | ❌ FAIL |
| Blocked-subset (G2) | 0.0% | 0.0% | +0.0 pts | ≥ +5 pts | ❌ FAIL |
| MCPBench accuracy (G3) | (run MCPBench) | (run MCPBench) | — | ≥ +5 pts | ⚠️ N/A (no real Firecrawl run) |
| P99 latency / page | 1327 ms | 4045 ms | -2718 ms | ≤ Firecrawl | ✅ PASS |
| $ / 1k pages | compute-only (API-free) | $0.83 | API-free vs paid API | strictly lower | ⚠️ N/A (no real Firecrawl run) |

**GATE 4 (G4 margins): ❌ one or more margins not met**

## Coverage detail

| | Ours | Firecrawl |
|--|------|-----------|
| Coverage (G1) | 6/8 = 75.0% | 6/8 = 75.0% |
| Blocked-subset (G2) | 0/2 = 0.0% | 0/2 = 0.0% |
| P99 latency/page | 1327 ms | 4045 ms |

## Per-URL detail

### Ours

| URL | Tier | Blocked | Clean | Latency (ms) | Detail |
|-----|------|---------|-------|--------------|--------|
| https://www.rust-lang.org/ | static | no | ✅ | 1327 | body_chars=2257 |
| https://en.wikipedia.org/wiki/Rust_(programming_language) | static | no | ✅ | 882 | body_chars=74869 |
| https://doc.rust-lang.org/book/ | static | no | ✅ | 765 | body_chars=944 |
| https://blog.rust-lang.org/ | static | no | ✅ | 590 | body_chars=12390 |
| https://nextjs.org/ | spa | no | ✅ | 682 | body_chars=1623 |
| https://react.dev/ | spa | no | ✅ | 944 | body_chars=3751 |
| https://www.g2.com/products/notion/reviews | cloudflare | yes | ❌ | 623 | call_error: tool reported error: [ERROR] blocked by site: https://www.g2.com/products/notion/reviews (status 403) |
| https://www.crunchbase.com/organization/openai | cloudflare | yes | ❌ | 754 | call_error: tool reported error: [ERROR] blocked by site: https://www.crunchbase.com/organization/openai (status 403) |

### Firecrawl

| URL | Tier | Blocked | Clean | Latency (ms) | Detail |
|-----|------|---------|-------|--------------|--------|
| https://www.rust-lang.org/ | static | no | ✅ | 1733 | chars=3466 status=200 |
| https://en.wikipedia.org/wiki/Rust_(programming_language) | static | no | ✅ | 1957 | chars=236274 status=200 |
| https://doc.rust-lang.org/book/ | static | no | ✅ | 635 | chars=1672 status=200 |
| https://blog.rust-lang.org/ | static | no | ✅ | 4045 | chars=40595 status=200 |
| https://nextjs.org/ | spa | no | ✅ | 1470 | chars=15267 status=200 |
| https://react.dev/ | spa | no | ✅ | 2107 | chars=16638 status=200 |
| https://www.g2.com/products/notion/reviews | cloudflare | yes | ❌ | 1871 | HTTP 500: {"success":false,"code":"SCRAPE_RETRY_LIMIT","error":"Scrape aborted a… |
| https://www.crunchbase.com/organization/openai | cloudflare | yes | ❌ | 1011 | HTTP 500: {"success":false,"code":"SCRAPE_RETRY_LIMIT","error":"Scrape aborted a… |

