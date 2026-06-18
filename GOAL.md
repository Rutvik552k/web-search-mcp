# GOAL — Ground Truth

> Authoritative definition of "done" for the web-search MCP server.
> Every task in [TASKS.md](TASKS.md) traces back to a criterion here.
> Per CLAUDE.md Rule 1: a claim counts only when backed by a benchmark result,
> command output, or doc/source citation — never opinion.

---

## 1. Mission statement

A self-contained, API-free web-search MCP server that can **scrape any website —
with or without anti-bot blockers — at high accuracy and 90%+ resource coverage**,
and that **beats Firecrawl head-to-head** (see G4) while requiring no third-party
scrape API.

## 2. Scope & authorization (read first)

Operator decision (2026-06-06): pursue **scrape-any-data**. Strategy, cheapest→hardest:
1. **Data-layer first (Rung -1)** — acquire data from less-protected surfaces
   (hydration blobs, JSON-LD, internal/mobile APIs, RSS, sitemaps, archive) before
   fighting the protected HTML. See `docs/adr/0002-data-layer-acquisition.md`.
2. **HTTP impersonation** — wreq TLS/HTTP2 fingerprint (Chrome). `docs/research/0001`.
3. **Stealth headless** — defeat CDP `Runtime.enable` detection.
4. **Proxy rotation** + **CAPTCHA solve**.
5. **Commercial unblock API (tier-3)** — optional, for DataDome/behavioral sites OSS can't beat.

### Reality (verified, `docs/research/0001`)
"ANY site" is the target, not a guarantee: OSS reliably beats passive fingerprinting
+ many JS challenges; DataDome/behavioral sites may need tier-3 or stay out of reach.

### Legal guardrails (verified, `docs/research/0002` — NOT legal advice; get counsel)
These are **hard defaults**, not optional, because they neutralize the real risks
(ToS breach, GDPR/PII, CFAA-via-circumvention):
- **Public + unauthenticated only.** Never log in / never cross an auth wall.
- **Block or C&D = permanent per-domain hard stop. Never rotate IPs back in.**
  (Rotating to evade a known block is the strongest unauthorized-access fact pattern —
  *Power Ventures*, *3taps*.)
- **No PII retention by default**; minimize, pseudonymize, honor opt-out, retention limits.
- Polite rate limits even when ignoring robots; treat robots/CAPTCHA as opt-out signals.
- Store transformed/attributed derivatives, not raw copies; honor EU TDM opt-outs.
- Takedown + erasure pipeline + audit log.
- Aggressive bypass features **opt-in via config**; compliance gate (TASKS Phase 5) before enabled.

## 3. Measurable acceptance criteria (the "ground truth")

The goal is **not testable until these artifacts exist** and are committed.
Targets are claimed only after the harness reports them.

### G1 — Coverage ≥ 90%
- **Artifact:** operator-provided benchmark URL set (`benchmark/urls.jsonl`),
  spanning tiers: static HTML, JS-SPA, Cloudflare/DataDome-protected, rate-limited,
  paywalled/login-walled.
- **Metric:** `coverage = pages_with_clean_main_content / total_urls`.
  "Clean main-content" = extractor returns ≥ 200 chars of non-boilerplate text
  matching the page's primary article (manually spot-verified on a 20-URL sample).
- **Target:** ≥ 0.90 over the full set.

### G2 — Blocked-subset success ≥ (operator-defined %)
- **Artifact:** the bot-protected subset of `benchmark/urls.jsonl` (tagged `blocked: true`).
- **Metric:** success rate on that subset alone.
- **Target:** set by operator once baseline is measured (Phase 4 gate).

### G3 — High accuracy
- **Tool:** [MCPBench](https://github.com/modelscope/MCPBench) (modelscope) — verified
  it evaluates web-search MCP servers on **task-completion accuracy, latency, token
  consumption** over 600 built-in QA pairs (`{unique_id, prompt, answer}`, Frames/news/tech).
  Plug our server via its JSON config (local cmd), run `evaluation_websearch.sh`.
- **Metric:** MCPBench accuracy (+ latency, token use). Optionally extend with custom
  QA pairs in MCPBench format.
- **Target:** set by operator after baseline (Phase 4 gate).
- **Note:** MCPBench does NOT measure coverage or blocker bypass — those are G1/G2,
  measured by our own harness. MCPBench only covers G3.

### G4 — Beat Firecrawl head-to-head
- **Baseline:** [Firecrawl](https://github.com/firecrawl/firecrawl) (`/scrape` + `/crawl`),
  a leading commercial+OSS crawl/scrape API. Run it over the **same** `benchmark/urls.jsonl`
  and the **same** MCPBench QA pairs as G1–G3, recording its results as the comparison line.
- **Metric:** our score minus Firecrawl's score on each of:
  1. Coverage (G1 metric),
  2. Blocked-subset success (G2 metric),
  3. MCPBench accuracy (G3 metric),
  4. Latency / cost — P99 latency per page **and** compute $ per 1k pages. Firecrawl runs
     **self-hosted** (AGPL-3.0, Docker) on the same hardware, so both sides are compute-only;
     cost compares like-for-like (no paid cloud tier).
- **Target:** **≥ +5 percentage points** over Firecrawl on each of coverage, blocked-subset,
  and accuracy; **≤ Firecrawl** on P99 latency and **≤ Firecrawl** on compute $/1k pages.
- **Artifact:** head-to-head comparison report (`benchmark/RESULTS.firecrawl.md`) produced by the
  benchmark harness, with both runs on identical inputs, same hardware, same date.
- **Note:** Firecrawl runs **self-hosted locally** for the comparison (operator decision
  2026-06-18) — not the paid cloud API. It is **baseline-comparison only**, a separate
  process queried over HTTP, never a runtime dependency of our server (Mission stays API-free).
  AGPL-3.0: unmodified, not redistributed, not exposed to third parties → no source-disclosure
  obligation on our code, no contamination.
- **⚠️ Blocked-tier caveat:** self-hosted Firecrawl has **no Fire-engine**, so it is weaker on
  bot-protected pages than the paid cloud. The G2 (blocked-subset) number from this baseline is a
  **floor** for Firecrawl, not its cloud ceiling — `RESULTS.firecrawl.md` must state this, and a
  G2 win over self-host is **not** a win over cloud Firecrawl. Cloud-equivalent blocked-tier
  comparison would need the paid cloud key (operator decision — see TASKS 0.6b).

## 4. Non-goals (current)

- Distributed/multi-node crawling at internet scale.
- Real-time index of the whole web (index is query-driven + daemon pre-fetch).
- Hosted SaaS / multi-tenant (single-operator tool).

## 5. Open dependencies on operator

- Provide `benchmark/urls.jsonl` coverage set for G1/G2 (criterion 2b). **Required** —
  MCPBench cannot measure coverage/blocker bypass.
- `queries.jsonl` no longer required: G3 uses MCPBench's 600 built-in QA pairs
  (optionally extend with custom QA in MCPBench format).
- Set G2 / G3 numeric targets after baseline run.
- Run the self-hosted Firecrawl Docker stack for the G4 baseline comparison (baseline-only,
  not a runtime dep; setup in `benchmark/firecrawl-selfhost/`).
- Provide proxy-pool credentials and CAPTCHA-solver API key (or approve a provider).

---
_Last updated: 2026-06-18 (added G4: beat Firecrawl head-to-head, operator sign-off).
Change this file only with operator sign-off — it is the contract._
