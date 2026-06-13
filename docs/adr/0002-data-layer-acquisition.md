# ADR 0002 — Data-Layer-First Acquisition ("Rung -1")

- **Status:** Proposed
- **Date:** 2026-06-06
- **Owner:** research-solution-architect
- **Relates to:** ADR 0001 (escalation ladder). This ADR inserts a new rung *below* Rung 0.
- **Backs:** GOAL.md G1 (coverage ≥90%) and G2 (blocked-subset success) — by *avoiding* the fight on the protected HTML page where a cheaper surface carries the same data.
- **Implemented by (later phases):** new `data_layer` module in `crates/crawler`; extractor hooks in `crates/extractor`.

> Scope guard (GOAL.md §2): all out-of-band sources are opt-in via config. The cloaking
> angle (Googlebot UA) and archive fallback that ignores a site's `robots`/AI-block intent are
> gated behind the **Phase 5 legal gate**, default OFF — same treatment as CAPTCHA-solving in ADR 0001.

---

## 1. Thesis and context

ADR 0001 fights the fingerprint: it climbs from plain HTTP → impersonation → proxy → headless →
CAPTCHA on the **protected HTML document**. Every rung above 0 costs money, latency, or both, and
the hardest rungs (DataDome behavioral ML) are explicitly *not guaranteed* even with the full
stack (Research 0001 headline).

The cheaper, more reliable move is frequently **not to fetch the protected HTML at all**, but to
acquire the *same content* from one of the site's own less-protected surfaces:

- The structured data is often **already in the Rung-0 HTML response** (hydration blobs, JSON-LD) —
  zero extra requests, it was sitting in a body we currently throw away on a soft block.
- The site's **own JSON/GraphQL/mobile API** returns the same records as clean JSON, and that
  surface is routinely **less defended** than the document path (verified below).
- **Syndication** (RSS/Atom/JSON Feed, sitemaps, news sitemaps) and **alternate renderings**
  (AMP, print, `m.` mobile) are designed for machine consumption and rarely carry the JS challenge.
- **Cache/archive** surfaces hold a copy when the live origin is hostile.

This is **Rung -1**: tried *before* (and for blocked-prone domains, *in parallel with*) the wreq
impersonation rung, because it is cheaper and often wins outright. It does not replace the ladder —
it short-circuits it when an easier surface exists, and falls through to ADR 0001's ladder when it
does not.

### Why an internal API is often weaker than the page it backs (the load-bearing claim)

Two independent mechanisms, both verified:

1. **Mobile/native and legacy APIs lag the web WAF.** Web frontends get WAF updates "daily";
   mobile API backends are pinned by app-store release inertia and must keep **legacy endpoints**
   alive for old app versions, which "often lack the sophisticated anti-bot protections added in
   later iterations." The same data "flowing freely … on the user's smartphone" while the web
   front is "a formidable wall of defense" is a documented asymmetry.[mobile-api-dev][datadome-mobile][shopee]
2. **The challenge interstitial is served on document navigations, not XHR.** Cloudflare/DataDome
   render the "Just a moment" / interstitial on the HTML `document` request; a site's own
   `fetch`/XHR JSON endpoints must remain callable for the frontend to function, so they are
   commonly either exempted or gated only by a **clearance cookie obtained once**. The honest
   counter-case: a same-origin API behind the *same* WAF rule set can be equally protected
   (vendor guidance assumes you "still handle the same Cloudflare layers")[scrapfly-cf][scrapeops-cf] —
   so this is a *frequently-true*, not *always-true*, property. **Flag (Rule 1):** treat per-domain
   API weakness as a hypothesis to confirm against `benchmark/urls.jsonl`, not a guarantee.

A powerful combined pattern follows from (2): pay the headless cost **once** (ADR 0001 Rung 3) to
mint a `cf_clearance` cookie, then drain the cheap JSON API on plain HTTP reusing that cookie.

---

## 2. The technique table (detect / extract / bypass / risk)

"Bypass est." = rough estimate of how often the surface yields usable content on a
Cloudflare/DataDome-protected target, with the *reason*. **These are heuristics, not measured
numbers** (Rule 1: the project has no G2 baseline yet — GOAL.md §5). Confirm against the benchmark.

Band: **IN** = parsed from the Rung-0 body we already have (0 extra requests). **OUT** = one or
more extra cheap GETs to an alternative surface.

| # | Technique | Band | Detect (how we know it's available) | Extract | Bypass est. & why | Failure / risk modes |
|---|---|---|---|---|---|---|
| 1 | **Hydration state** — `__NEXT_DATA__`, `self.__next_f.push`, `__NUXT__`, `window.__INITIAL_STATE__`, `__APOLLO_STATE__`, Redux preload | IN | Substring scan of body for the marker tokens; for Next, `<script id="__NEXT_DATA__" type="application/json">` | Parse the JSON (Next: `props.pageProps`; Apollo: normalized cache; flight: concatenate `__next_f` chunks) → map to title/body/author/date | **High when the page is SSR'd at all** — the full record is in the first HTML response even on "JS-heavy" sites; survives soft-blocks that return real HTML. Useless if the block replaced the body with a challenge page (then it's just not present → fall through) | RSC flight format (`__next_f`) is chunked + React-wire-serialized, harder to parse than legacy `__NEXT_DATA__`; shape is per-app (no universal schema); empty on pure CSR pages | [trickster][brightdata-next][njsparser][nuxt-apollo][apollo-ssr] |
| 2 | **Internal JSON / GraphQL API** — the XHR/fetch endpoints the frontend calls | OUT | `_next/data/<buildId>/<route>.json` (buildId from technique 1); common paths `/api/`, `/wp-json/wp/v2/posts`, `/graphql`, `?format=json`, `.json` suffix; GraphQL introspection probe | Plain GET (often no special headers); shape JSON → fields. Reuse `cf_clearance` from a one-time browser solve if gated | **Medium-high.** Mobile/legacy variants frequently unprotected; same-origin XHR often skips the interstitial. **But** discovery is per-site and the endpoint *may* share the WAF | Discovery is manual/heuristic (no universal map); endpoint may require auth token, signed params, or POST body; can be equally protected; schema drift | [mobile-api-dev][datadome-mobile][shopee][scrapfly-cf][scrapeops-cf] |
| 3 | **Structured-data surfaces** — JSON-LD, microdata, OpenGraph, oEmbed | IN (JSON-LD/OG/microdata); OUT (oEmbed) | JSON-LD `<script type="application/ld+json">` (**already extracted** in `metadata.rs`); OG `meta[property^=og:]` (**already extracted**); oEmbed `<link rel=alternate type=application/json+oembed>` | JSON-LD `Article`/`NewsArticle` `articleBody`/`headline`/`author`/`datePublished`; oEmbed: GET endpoint → JSON `html`/`title` | **High for in-band** (publishers ship JSON-LD for SEO; full `articleBody` is common on news). oEmbed lower coverage (mostly media/embeds, often summary not full body) | JSON-LD often has headline+date but **truncated/absent body**; multiple `@graph` nodes need disambiguation; oEmbed rarely returns full article text | [oembed-docs][oembed-extractor][smashing-oembed] (JSON-LD/OG already in repo) |
| 4 | **Syndication** — RSS/Atom/JSON Feed, sitemap.xml, sitemap index, news sitemaps | OUT | `<link rel=alternate type=application/rss+xml\|atom+xml\|feed+json>`; well-known `/feed`, `/rss`, `/sitemap.xml`, `/sitemap_index.xml`, `/robots.txt` `Sitemap:` line | Parse feed entries (`<content:encoded>` often holds **full body**); sitemap → URL list (**`sitemap.rs` already parses both**) | **High for discovery + freshness**; feed `content:encoded` sometimes carries the full article (bypasses the page entirely). Feeds/sitemaps are static XML, almost never challenged | Many feeds are summary-only (no full body); feeds cover recent items only; sitemap gives URLs not content (still must fetch each) | [rssboard-disc][whatwg-feed][rssdiscovery][feed-extractor] (sitemaps already impl) |
| 5 | **Alternate renderings** — AMP, print/reader, `m.` mobile, mobile REST | OUT | `<link rel=amphtml href>`; try `?amp=1`, `/amp/`; `m.<host>`; print params `?print=1`/`?output=print` | Fetch the variant → run normal `extract_page` (AMP/print are clean minimal HTML) | **Medium.** AMP/print pages are stripped, easy to extract, and historically less aggressively protected; many sites still emit `rel=amphtml` (Squarespace left tags in place after Feb-2025 UI removal) | AMP is deprecated/declining — fewer sites emit it each year; `m.` subdomains largely gone (responsive design); variant may 404 or redirect back to protected canonical | [amp-discovery][squarespace-amp] |
| 6 | **Cache / archive** — Wayback Machine, archive.today, Bing cache | OUT | Wayback CDX API: `web.archive.org/cdx/search/cdx?url=<u>&filter=statuscode:200&output=json`; snapshot raw HTML via `…/<ts>id_/<url>` | CDX → newest 200 snapshot ts → fetch `id_` raw HTML → `extract_page`. Bing still serves `cache:` | **Medium, and declining as a fallback.** Free public CDX API; great when origin is hostile. **But** 241+ news sites (NYT, USA Today, Guardian, Reddit) now block the IA crawler / filter article pages — so coverage on exactly the high-value protected sites is shrinking | Stale content (snapshot age); gaps for never-archived or `noarchive` pages; IA rate-limits (503 → back off); Google `cache:` **removed Sept 2024 — do not use**; archive.today has no clean API | [google-cache-dead][wayback-api][wayback-cdx][ia-blocked][ia-blocked2] |
| 7 | **Cloaking angle** — fetch as Googlebot/Bingbot UA | OUT | Heuristic: site is a paywall/news domain known to serve crawlers; no positive detection | Set `User-Agent: Googlebot…`; some paywalls gate on **UA only** and serve full content | **Low-to-medium and fragile.** Works only where the site checks UA *without* reverse-DNS IP verification (a documented but shrinking class); defenders verify Googlebot via reverse+forward DNS and block fake bots | **High detection/abuse risk:** fake-Googlebot detection is standard (DataDome ships it); arguably ToS/cloaking-policy violating; 2025 trend is cryptographic crawler auth (Web Bot Auth) that this cannot forge | [scrapfly-gbot][hn-paywall][datadome-fakebot][webbotauth] |

---

## 3. The Rung -1 design

### 3.1 Two integration points, not one

Data-layer acquisition happens at **two** distinct moments, because in-band and out-of-band have
different costs and triggers:

```
                         ┌──────────────────── Fetcher::fetch (escalation controller, ADR 0001) ─────────────────────┐
  fetch(url) ─► cache ─► │  RUNG 0  plain HTTP GET ──► classify(status,headers,body)                                   │
                         │     │                                                                                       │
                         │     ├─ RealContent ──────────────► [IN-BAND data-layer enrich] ──► extractor ──► cache+ret  │ (free)
                         │     │                                                                                       │
                         │     └─ Blocked/Challenge/SoftBlock/thin ──► ┌──────── RUNG -1: out-of-band DataLayerSources ───────┐
                         │                                             │  try in CostClass order (or race w/ Rung 1 if cfg):   │
                         │                                             │   Structured(in-band on whatever body we got)         │
                         │                                             │   → Feed/Sitemap → Internal API → AMP/print/mobile    │
                         │                                             │   → Archive → (Cloaking, gated)                       │
                         │                                             │  first high-confidence hit ─► extractor ─► cache+ret  │
                         │                                             └───────────────────────────────────────────────────────┘
                         │                                                  │ all sources miss / low confidence                 │
                         │                                                  ▼                                                   │
                         │   RUNG 1 wreq impersonation ─► RUNG 2 proxy ─► RUNG 3 headless ─► … (ADR 0001 ladder unchanged)      │
                         └────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

- **In-band** (`techniques 1, 3-JSON-LD/OG, hydration`) is *free* — it only reads a body we already
  fetched. It runs as an **extractor enrichment on every successful Rung-0 page** (improves G1
  accuracy), and as the **first thing tried** on a soft-block body (the block may have left the
  hydration blob intact).
- **Out-of-band** (`techniques 2, 4, 5, 6, 7`) is **Rung -1 proper**: extra cheap GETs tried *before*
  spending on wreq/proxy/headless, because a feed fetch (~one static-XML GET) is cheaper and more
  reliable than a BoringSSL impersonation client + proxy bytes.

### 3.2 The `DataLayerSource` trait

New module `crates/crawler/src/data_layer/mod.rs`. The trait is deliberately small and mirrors the
detect→extract flow.

```rust
use async_trait::async_trait;          // already idiomatic in tokio crates; add if absent
use web_search_common::Result;

/// Relative cost, used to order Rung -1 attempts cheapest-first.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CostClass {
    InBand,      // 0 network requests — parse the body we already have
    CheapGet,    // 1 GET to a static/alt surface (feed, sitemap, AMP, archive)
    ProbeGet,    // discovery probe + GET (internal API, oEmbed, GraphQL)
    Gated,       // legally/ethically gated (cloaking) — off unless opted in
}

/// Verdict from a source's cheap detection step.
pub enum Detection {
    /// Confident this source applies; carries any hint extracted during detection
    /// (e.g. the AMP href, the feed URL, the Next.js buildId) to avoid re-parsing.
    Applicable { hint: Option<String> },
    NotApplicable,
}

/// What a successful acquisition yields. All variants funnel into the existing extractor/indexer.
pub enum DataLayerPayload {
    /// Already-structured article fields → mapped straight into ExtractionResult,
    /// skipping trafilatura/readability.
    Structured(StructuredContent),
    /// Clean HTML (AMP/print/cache snapshot) → run through existing consensus::extract_page.
    Html { html: String, final_url: String },
    /// Raw internal-API JSON → generic shaper or site adapter → ExtractionResult.
    Json(serde_json::Value),
    /// URLs discovered (feed index / sitemap) → enqueue to frontier; not content itself.
    Links(Vec<String>),
}

pub struct StructuredContent {
    pub title: Option<String>,
    pub body_text: String,
    pub author: Option<String>,
    pub published: Option<String>,   // ISO-8601; parsed downstream by metadata::parse_date
    pub language: Option<String>,
}

pub struct DataLayerHit {
    pub source: &'static str,
    pub payload: DataLayerPayload,
    /// 0.0-1.0. Controller accepts a hit only at/above cfg.data_layer_min_confidence.
    pub confidence: f32,
}

/// Read-only context handed to every source. `rung0_body`/`rung0_headers` are Some
/// whenever we have a Rung-0 response (even a soft-blocked one); None when Rung 0 errored
/// before a body (then only URL-derived sources can act).
pub struct DataLayerCtx<'a> {
    pub url: &'a str,
    pub rung0_body: Option<&'a str>,
    pub rung0_content_type: Option<&'a str>,
    pub cfg: &'a DataLayerConfig,
}

/// Abstraction over the Fetcher's plain GET so sources can sub-fetch alt surfaces
/// through the SAME cache/throttle/cookie-jar without a circular dependency on Fetcher.
#[async_trait]
pub trait RawFetch: Send + Sync {
    /// Plain GET (optionally with an override UA, e.g. Googlebot for the gated source).
    async fn get(&self, url: &str, ua_override: Option<&str>) -> Result<RawResponse>;
}
pub struct RawResponse { pub status: u16, pub final_url: String, pub content_type: String, pub body: String }

#[async_trait]
pub trait DataLayerSource: Send + Sync {
    fn name(&self) -> &'static str;
    fn cost(&self) -> CostClass;
    /// Cheap; in-band sources MUST NOT do network I/O here.
    fn detect(&self, ctx: &DataLayerCtx) -> Detection;
    /// Acquire content. May issue sub-fetches via `fetch`. `hint` is the Detection hint.
    async fn acquire(&self, ctx: &DataLayerCtx, hint: Option<&str>, fetch: &dyn RawFetch)
        -> Result<DataLayerHit>;
}
```

### 3.3 The orchestrator over the sources

```rust
pub struct DataLayerEngine { sources: Vec<Box<dyn DataLayerSource>> }

impl DataLayerEngine {
    /// In-band only: run on a good Rung-0 body to enrich extraction. Never network I/O.
    pub fn enrich(&self, ctx: &DataLayerCtx) -> Option<DataLayerHit> { /* CostClass::InBand, best confidence */ }

    /// Rung -1: called when Rung 0 did not yield RealContent. Tries sources cheapest-first,
    /// returns the first hit ≥ cfg.data_layer_min_confidence. Each source bounded by
    /// cfg.data_layer_per_source_timeout_ms; total bounded by the ADR-0001 per-URL time budget.
    pub async fn acquire(&self, ctx: &DataLayerCtx, fetch: &dyn RawFetch) -> Option<DataLayerHit> {
        let mut srcs: Vec<_> = self.sources.iter()
            .filter(|s| s.cost() != CostClass::Gated || ctx.cfg.enable_cloaking_source)
            .filter(|s| matches!(s.detect(ctx), Detection::Applicable{..}))
            .collect();
        srcs.sort_by_key(|s| s.cost());          // InBand < CheapGet < ProbeGet
        for s in srcs { /* timeout(acquire); if hit.confidence ≥ min → return Some(hit) */ }
        None
    }
}
```

Sources registered in cost order: `HydrationStateSource`, `StructuredDataSource` (InBand) →
`FeedSource`, `SitemapSource`, `AmpPrintSource`, `ArchiveSource` (CheapGet) → `InternalApiSource`,
`OEmbedSource` (ProbeGet) → `CrawlerUaSource` (Gated).

### 3.4 How a hit feeds the existing extractor/indexer

The existing path is `crawler.rs::fetch_and_process` → `extractor::consensus::extract_page(html,
url)` → `to_page(...)` → indexer. A `DataLayerHit` joins that path with **no change to the indexer**:

- `DataLayerPayload::Html{..}` → `extract_page(html, final_url)` exactly as today.
- `DataLayerPayload::Structured` / `Json` → a new thin `extractor` helper builds an `ExtractionResult`
  directly (title/body_text/author/published/language already known), then `to_page(...)`. Trafilatura
  /readability are skipped — we already have clean structured text, so we *avoid* the lossy
  boilerplate-stripping passes. Re-use `metadata::parse_date` for the date string and
  `consensus::clean_body_text` for final hygiene.
- `DataLayerPayload::Links` → push to `frontier` (same as the existing search-results / sitemap path
  in `fetch_and_process`).

In-band JSON-LD/OG is **already** parsed by `metadata::extract_metadata` and rides on
`ExtractionResult.json_ld` / `.open_graph`; technique 3's only new work is *promoting* a JSON-LD
`articleBody` into `body_text` when the consensus extractors come up thin — a small change in
`consensus::extract_page`, not a new pipeline.

### 3.5 Integration points in the existing code (additive, behind a flag)

1. **`fetcher.rs` — stop discarding the body on error (shared with ADR 0001 C3).** `fetch_once`
   (fetcher.rs:207-212) returns `Err(Blocked)` *before* reading the body. Capture
   `(status, headers, body)` for all statuses so the soft-block / challenge body is available to
   in-band sources. Backward-compatible: a 2xx still returns the same `FetchResult`.

2. **`fetcher.rs::fetch` — insert the two hooks** (fetcher.rs:132-149), gated on
   `cfg.enable_data_layer`:
   - On a good Rung-0 result: `engine.enrich(ctx)` and merge any higher-confidence body before
     cache insert.
   - On a non-RealContent result, **before** returning `Err` / before climbing the ladder:
     `engine.acquire(ctx, raw_fetch).await`; on `Some(hit)` build a `FetchResult` from the hit
     (with a synthetic `status=200`, `from_cache=false`) and return it. **Cache contract unchanged
     (ADR 0001 C6):** only a real content hit is cached; a `Links`-only hit is not cached as content.
   - `Fetcher` implements `RawFetch` for its own plain GET so sources sub-fetch through the existing
     cache/cookie jar. The single browser pool is *not* invoked by Rung -1 (that stays Rung 3+).

3. **`crawler.rs` — no signature change.** `fetch_and_process` / `fetch_one` /
   `fetch_urls_concurrent` keep calling `fetcher.fetch(url)`; Rung -1 is invisible to them
   (contract C2 in ADR 0001 preserved). The existing JSON-content branch
   (crawler.rs:191-199) and sitemap parsing (`sitemap.rs`) are *generalized* by `InternalApiSource`
   / `SitemapSource` rather than duplicated.

4. **`extractor` — promote structured body.** Small change in `consensus::extract_page`: if
   `traf`/`read` confidence is below threshold but `json_ld` carries an `articleBody`, use it; add a
   `from_structured(StructuredContent)` constructor for the `Structured`/`Json` payloads.

### 3.6 Ordering vs the wreq HTTP rung (the "-1" justification)

Rung -1 sources outrank Rung 1 (wreq impersonation) because they are cheaper on every axis:

| Step | Extra network | $ cost | Latency (typical) | Reliability driver |
|---|---|---|---|---|
| In-band (1,3) | 0 | 0 | ~1-5 ms (parse) | content already present |
| Feed / sitemap / AMP / archive (4,5,6) | 1 GET | 0 (free hosts) | ~100-600 ms | static XML/HTML, rarely challenged |
| Internal API / oEmbed (2) | 1 probe + 1 GET | 0 | ~200-900 ms | weaker API surface |
| **wreq impersonation (ADR-0001 Rung 1)** | 1 GET w/ heavy client | 0 (CPU) | ~100-800 ms + client build | beats passive fingerprinting only |
| Proxy / headless / CAPTCHA (Rung 2-5) | 1+ | $$ / CPU | 2-60 s | the expensive last resort |

For **blocked-prone domains** (per-domain circuit-breaker history from ADR 0001), config
`data_layer_race=true` runs the cheapest CheapGet sources **concurrently** with Rung 1 and takes
whichever returns usable content first — trading one wasted cheap request for tail-latency.

---

## 4. Config additions (`DataLayerConfig`, additive, secure defaults)

Additive to `CrawlerConfig`; every field `#[serde(default)]` so existing `config.toml` keeps parsing
and behavior is unchanged when off (config-management rule). No secrets here.

```rust
#[serde(default)] pub enable_data_layer: bool,                 // master switch (in-band + Rung -1)
#[serde(default = "d_true")] pub data_layer_in_band: bool,     // free enrichment; safe default ON when enabled
#[serde(default = "d_conf")] pub data_layer_min_confidence: f32,   // accept hit ≥ this (e.g. 0.6)
#[serde(default = "d_src_to")] pub data_layer_per_source_timeout_ms: u64, // e.g. 1500
#[serde(default)] pub data_layer_race: bool,                   // race CheapGet sources with Rung 1
// per-source enables (so operator tunes coverage vs request volume):
#[serde(default = "d_true")] pub src_hydration: bool,
#[serde(default = "d_true")] pub src_structured: bool,
#[serde(default = "d_true")] pub src_feed: bool,
#[serde(default = "d_true")] pub src_sitemap: bool,
#[serde(default = "d_true")] pub src_amp_print: bool,
#[serde(default)] pub src_internal_api: bool,                  // off by default (probe volume + per-site)
#[serde(default)] pub src_oembed: bool,
#[serde(default)] pub src_archive: bool,                       // off: staleness + IA-block trend
#[serde(default)] pub enable_cloaking_source: bool,            // LEGAL GATE — off; Phase 5
#[serde(default)] pub archive_user_agent: Option<String>,      // identify ourselves to IA politely
```

**Startup validation:** if `enable_cloaking_source` then require an explicit acknowledgement flag
(mirror ADR 0001's CAPTCHA gate); if `src_archive` warn that coverage on blocked news domains is
declining (IA blocks). `respect_robots_txt` continues to govern whether we honor a site's crawl
directives on *its own* surfaces.

---

## 5. Failure modes and bounds

| Failure mode | Bound / mitigation |
|---|---|
| Hydration blob present but garbage/empty (CSR page) | Source returns low confidence → controller rejects → falls through to ladder. No loop. |
| Internal-API discovery wrong (404/401/HTML challenge) | Per-source timeout + the response is re-classified; a challenge body just means NotApplicable → next source. |
| Structured data has headline but **no body** | `confidence` reflects body length; below `data_layer_min_confidence` → not accepted as the page, but still rides as metadata. |
| Archive snapshot stale / `noarchive` / IA 503 | CDX `filter=statuscode:200` + newest-first; on 503 back off (don't hammer IA); staleness flagged in `tracing` (operator can reject by age later). |
| Cloaking source detected as fake Googlebot | Default OFF; when on, it is last and one-shot; a block just falls through. Documented ToS risk; legal gate. |
| Source fan-out inflates request volume | Sources gated individually; ProbeGet/archive off by default; total bounded by ADR-0001 per-URL time budget; `data_layer_race` is opt-in. |
| Cache poisoning by a Links-only or low-conf hit | C6 unchanged: only RealContent (incl. accepted data-layer body) is cached; Links go to frontier, never the content cache. |
| Duplicate fetch (alt surface == canonical) | Sub-fetches go through the existing moka cache keyed by URL; AMP/print resolve to distinct URLs so no self-recursion. |

**At 10×/100× load:** in-band adds only CPU (microseconds of parsing) and *reduces* heavy-rung load
by satisfying pages early — it is strictly de-risking. Out-of-band adds at most a bounded number of
cheap GETs per blocked URL; archive/IA is the one external dependency and is rate-limit-sensitive, so
it is off by default and must honor IA back-off. None of the Rung -1 sources touch the single browser
pool, so they cannot serialize on it (the ADR-0001 bottleneck).

---

## 6. ROI-ordered implementation recommendation

Ordered by (coverage lift × reliability) ÷ build cost. First two are near-free wins that also raise
**G1 accuracy on already-fetched pages**, not just G2 bypass.

1. **Technique 1 — Hydration state (IN-band).** Highest ROI. Zero extra requests, big lift on
   SSR'd Next/Nuxt/Apollo sites, and it salvages content from soft-blocked bodies we currently
   discard. Build: a body scanner + Next.js `__NEXT_DATA__`/flight + generic `__INITIAL_STATE__`
   parser. Depends only on the ADR-0001 "capture body on error" change.
2. **Technique 3 — Structured-data promotion (IN-band).** Mostly *already built* — `metadata.rs`
   parses JSON-LD/OG. Only work: promote `articleBody` into `body_text` when consensus is thin.
   Near-zero cost, immediate G1 gain on news/article sites.
3. **Technique 4 — Syndication (OUT, CheapGet).** Sitemaps are *already parsed* (`sitemap.rs`);
   add `<link rel=alternate>` feed discovery + a feed parser (`content:encoded` full bodies). Static
   XML almost never challenged → reliable Rung -1 win, and feeds/sitemaps also feed the frontier.
4. **Technique 5 — AMP/print (OUT, CheapGet).** Cheap detect (`rel=amphtml`) + reuse `extract_page`.
   Declining surface, so medium priority and expect erosion over time.
5. **Technique 2 — Internal JSON/GraphQL API (OUT, ProbeGet).** Highest *potential* (the thesis's
   strongest case) but highest build cost and per-site fragility; needs the `_next/data` buildId
   path (from technique 1) + generic shapers + optional one-time `cf_clearance` reuse. Default OFF;
   implement after 1-4 prove the plumbing.
6. **Technique 6 — Cache/archive (OUT, CheapGet).** Useful fallback but value is *declining* (IA
   blocks on exactly the high-value sites; Google cache dead). Default OFF; implement as a
   last-resort source.
7. **Technique 7 — Cloaking (OUT, Gated).** Lowest priority, highest risk, shrinking efficacy
   (reverse-DNS verification, Web Bot Auth). Legal-gated, default OFF; implement only if the
   operator explicitly authorizes for owned/permitted targets.

**Build order summary:** 1 → 3 → 4 (these three are cheap and lift G1 *and* G2) → then 5 → 2 →
6 → 7 behind gates. Measure each against `benchmark/urls.jsonl` before enabling the next, per the
ADR-0001 GATE-2 discipline.

---

## 7. Honest limitations & when this changes

- Rung -1 wins **when the data exists on an easier surface**. For a pure-CSR, API-gated,
  feed-less, never-archived, DataDome-behavioral site, every Rung -1 source misses and we fall
  straight through to the ADR-0001 ladder — Rung -1 then cost only a few cheap GETs (bounded).
- The "API is weaker" property is *frequent, not guaranteed* (§1). If the benchmark shows a target's
  API is equally protected, `src_internal_api` should be left off for that domain.
- Archive and cloaking are **structurally declining** (publisher IA-blocks, dead Google cache,
  crawler cryptographic auth). They are fallbacks, not foundations — revisit if the trend reverses.
- This ADR sets contracts and ordering; it writes no feature code. Bypass-rate cells are estimates
  (Rule 1) pending the operator's `benchmark/urls.jsonl` baseline (GOAL.md §5).

---

## Citations

- [mobile-api-dev] Reverse Engineering Mobile APIs — legacy/mobile endpoints lag web WAF: https://dev.to/deepak_mishra_35863517037/reverse-engineering-mobile-apis-the-path-of-least-resistance-23fc
- [datadome-mobile] DataDome, mobile app API bot protection (asymmetry acknowledged): https://datadome.co/guides/api-protection/bot-protection-mobile-app-api/
- [shopee] Shopee scraping via native mobile API interception: https://www.bluetickconsultants.com/shopee-scraping-via-mobile-application-native-app-api-interception/
- [scrapfly-cf] Scrapfly, How to Bypass Cloudflare 2026 (API still behind same layers — counter-case): https://scrapfly.io/blog/posts/how-to-bypass-cloudflare-anti-scraping
- [scrapeops-cf] ScrapeOps, How to Bypass Cloudflare 2026: https://scrapeops.io/web-scraping-playbook/how-to-bypass-cloudflare/
- [trickster] Scraping Next.js websites in 2025 (`__NEXT_DATA__`, `__next_f` flight): https://www.trickster.dev/post/scraping-nextjs-web-sites-in-2025/
- [brightdata-next] Web Scraping With Next.js 2026: https://brightdata.com/blog/how-tos/web-scraping-with-next-js
- [njsparser] njsparser — Next.js flight-data parser: https://github.com/novitae/njsparser
- [nuxt-apollo] nuxt-modules/apollo `window.__NUXT__` state hydration: https://github.com/nuxt-modules/apollo/issues/17
- [apollo-ssr] Apollo SSR (`__APOLLO_STATE__` rehydration): https://www.apollographql.com/docs/react/features/server-side-rendering/
- [oembed-docs] oEmbed spec / discovery `rel=alternate type=application/json+oembed`: https://oembed.org/docs/
- [oembed-extractor] oembed-extractor: https://github.com/extractus/oembed-extractor
- [smashing-oembed] Smashing, Programmatically Discovering oEmbed: https://www.smashingmagazine.com/2019/11/programmatically-discovering-sharing-code-oembed/
- [rssboard-disc] RSS Autodiscovery (`link rel=alternate type=application/rss+xml`): https://www.rssboard.org/rss-autodiscovery
- [whatwg-feed] WHATWG, Feed Autodiscovery: https://blog.whatwg.org/feed-autodiscovery
- [rssdiscovery] node-rssdiscovery (RSS/Atom/RDF discovery): https://github.com/danmactough/node-rssdiscovery
- [feed-extractor] feed-extractor (RSS/Atom/JSON Feed → normalized): https://github.com/extractus/feed-extractor
- [amp-discovery] amp.dev, Make your pages discoverable (`rel=amphtml`): https://amp.dev/documentation/guides-and-tutorials/optimize-and-measure/discovery
- [squarespace-amp] Squarespace AMP removal Feb 2025 left `rel=amphtml` tags in place: https://www.collaborada.com/blog/amp
- [google-cache-dead] Search Engine Land, Google killed cache feature (Sept 2024, permanent): https://searchengineland.com/google-search-completely-kills-the-cache-feature-446904
- [wayback-api] Internet Archive Wayback Machine APIs: https://archive.org/help/wayback_api.php
- [wayback-cdx] Wayback CDX Server API (filter=statuscode:200, output=json): https://github.com/internetarchive/wayback/tree/master/wayback-cdx-server
- [ia-blocked] Nieman Lab, news publishers limiting IA access (241 sites block IA bots): https://www.niemanlab.org/2026/01/news-publishers-limit-internet-archive-access-due-to-ai-scraping-concerns/
- [ia-blocked2] Reddit blocks Wayback Machine (Aug 2025): https://www.sdxcentral.com/news/reddit-blocks-wayback-machine-to-stop-ai-dev-scrape/
- [scrapfly-gbot] Scrapfly, Googlebot user-agent string: https://dev.to/scrapfly_dev/what-is-googlebot-user-agent-string-fko
- [hn-paywall] HN, How Google's crawler bypasses paywalls (UA-only gating): https://news.ycombinator.com/item?id=11134798
- [datadome-fakebot] DataDome, stopping fake Googlebots (reverse-DNS verification): https://datadome.co/learning-center/scrapers-bad-bots-steal-content/
- [webbotauth] Web Bot Auth — cryptographic crawler verification (2026): https://seojuice.com/blog/web-bot-auth-googlebot-verification-2026/

**Flagged for verification before/with implementation (Rule 1):**
1. All "Bypass est." cells are heuristic — confirm per-domain against `benchmark/urls.jsonl`; the
   "internal API is weaker" property is frequent, not guaranteed (§1, [scrapfly-cf] counter-case).
2. Next.js **RSC flight** (`self.__next_f.push`) parsing is materially harder than legacy
   `__NEXT_DATA__`; budget a parser spike ([njsparser] is the reference, Python).
3. archive.today has **no clean public API** (not confirmed in this research) — treat as best-effort
   HTML scrape only; Wayback CDX is the grounded archive path.
4. Cloaking efficacy is shrinking (Web Bot Auth [webbotauth]); do not build it expecting durability.
