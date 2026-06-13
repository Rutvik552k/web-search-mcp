# Benchmark Results

- **Generated:** 2026-06-06T20:25:50.238536400+00:00
- **Git SHA:** `ffd795775bf4a5f31a212f46b6fd9c1e83e50740`
- **Server:** `target/debug/mock-mcp-server.exe`
- **Tools:** extract=`extract`, search=`quick_search`
- **Metrics:** coverage clean-threshold = 200 chars; nDCG@10; precision@5

> Baseline run. G1 target = 0.90. G2/G3 targets are operator-set after baseline (GOAL.md Phase 4 gate).

## G1 — Coverage

**coverage = 2/3 = 0.6667 (66.7%)** — target ≥ 0.90 ❌ below target

### Per-tier breakdown

| Tier | Clean | Total | Coverage |
|------|-------|-------|----------|
| cloudflare | 0 | 1 | 0.0% |
| spa | 1 | 1 | 100.0% |
| static | 1 | 1 | 100.0% |

## G2 — Blocked-subset success

**blocked success = 0/1 = 0.0%** — target: operator-set (Phase 3 gate)

### Per-URL detail

| URL | Tier | Blocked | Clean | Detail |
|-----|------|---------|-------|--------|
| https://example.com/ | static | no | ✅ | body_chars=337 |
| https://spa.example.com/app | spa | no | ✅ | body_chars=346 |
| https://protected.example.com/article | cloudflare | yes | ❌ | body_chars=52 |

## G3 — Accuracy

**mean nDCG@10 = 0.7853**, **mean precision@5 = 0.4000** (over 2 queries) — targets operator-set

### Per-query detail

| Query | nDCG@10 | P@5 | #relevant | #results | Note |
|-------|--------|------|-----------|----------|------|
| rust async runtime | 0.9197 | 0.4000 | 2 | 5 |  |
| python web framework | 0.6509 | 0.4000 | 2 | 5 |  |

