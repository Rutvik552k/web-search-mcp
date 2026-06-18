# Self-hosted Firecrawl — G4 benchmark baseline (operator-run)

Local Firecrawl so the G4 head-to-head harness hits a **local** instance instead
of the paid cloud API. **Baseline-comparison infra only** — not part of the
`web_search_mcp` server runtime or any shipped artifact. The harness adapter
(`crates/benchmark/src/firecrawl.rs`) is never wired into the server.

All facts below verified **2026-06-18** against the upstream repo at
`main` (commit-current). Sources at the bottom.

---

## 1. Prerequisites (operator's machine)

- Docker + the Docker Compose plugin (`docker compose`, not legacy `docker-compose`).
- This step is **operator-run**: Docker availability cannot be verified from the
  agent sandbox. Run the commands below on a machine with Docker installed.

## 2. Run it

```bash
cd benchmark/firecrawl-selfhost
cp .env.example .env          # defaults already give a working no-paid-deps stack
docker compose up -d          # pulls digest-pinned GHCR images (no source build)
docker compose ps             # all services should be Up; rabbitmq healthy
```

First boot pulls images and initializes the Postgres queue; give it ~30–60s.
Watch readiness with:

```bash
docker compose logs -f api    # wait for the API to report listening on :3002
```

**Resulting local API base URL:** `http://localhost:3002`
Queue/admin UI: `http://localhost:3002/admin/CHANGEME/queues` (key = `BULL_AUTH_KEY`).

## 3. Smoke test — prove `/v2/scrape` returns markdown

Self-hosted with `USE_DB_AUTHENTICATION=false`, **no Bearer token is required**
(API keys are only needed against cloud `api.firecrawl.dev`). A token is still
accepted if sent, so the harness sending `Authorization: Bearer <anything>` is
harmless.

```bash
# v2 scrape — the path + body our adapter uses:
curl -s -X POST http://localhost:3002/v2/scrape \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://example.com","formats":["markdown"],"onlyMainContent":true}' \
  | head -c 800
```

Expected shape (matches our adapter's `ScrapeResponse`):

```json
{ "success": true,
  "data": { "markdown": "# Example Domain\n\n...",
            "metadata": { "title": "Example Domain",
                          "sourceURL": "https://example.com",
                          "statusCode": 200 } } }
```

If you see `markdown` text in `data.markdown`, the baseline is working.

## 4. Resource footprint

5 containers: `api`, `playwright-service`, `redis`, `rabbitmq`, `nuq-postgres`.

| Service            | Image (pinned by digest)                 | CPU limit | Mem limit |
|--------------------|------------------------------------------|-----------|-----------|
| api                | ghcr.io/firecrawl/firecrawl:2.10.19      | 4.0       | 8G        |
| playwright-service | ghcr.io/firecrawl/playwright-service     | 2.0       | 4G        |
| redis              | redis:7-alpine                           | —         | small     |
| rabbitmq           | rabbitmq:3-management                    | —         | ~0.5G     |
| nuq-postgres       | ghcr.io/firecrawl/nuq-postgres           | —         | small     |

Upstream baseline guidance: **4 GB RAM minimum, 8 GB recommended, 2+ CPU cores.**
Plan for ~6–8 GB RAM in practice with the browser pool warm. No GPU. The compose
caps api/playwright so a runaway crawl can't exhaust the host; raise the caps if
you have headroom.

## 5. Teardown

```bash
docker compose down           # stop + remove containers + network
docker compose down -v        # also drop volumes (none persisted here, but safe)
docker image prune            # optional: reclaim the pulled images
```

## 6. Pinning / reproducibility

Images are pinned by **digest** (`@sha256:...`) so the same compose always
resolves the same artifact. `firecrawl` has semver tags (pinned to `2.10.19`'s
digest); `playwright-service` and `nuq-postgres` only publish `latest` upstream,
so they are pinned by the `latest` digest as of 2026-06-18. To bump: re-resolve
digests from `ghcr.io/v2/firecrawl/<image>/manifests/<tag>` and update the
`@sha256` pins + tag comments in `docker-compose.yaml`.

## 7. Feature degradation without paid deps (verified)

The default stack runs **basic scrape + crawl with markdown** with no paid
dependencies. What you give up vs. cloud:

- **No Fire-engine** (self-host has no access): weaker handling of IP blocks /
  bot detection / heavy anti-scraping. Protected pages fail more often than
  cloud. This is the main quality gap to note in the G4 write-up.
- **No LLM features** without `OPENAI_API_KEY`: `json` / `summary` / `extract`
  formats and LLM extras are unavailable. **Markdown scrape/crawl is
  unaffected** — that is exactly the path the harness exercises.
- **No proxy** unless you set `PROXY_*`: no IP rotation; more blocks on hostile
  sites. Optional; leaving it off keeps the comparison apples-to-apples on the
  markdown path.
- **No Supabase/DB auth**: no persistent multi-tenant auth, no advanced logging.
  Irrelevant for a single-operator local benchmark. You'll see a benign
  "bypassing authentication" warning in logs — expected.

---

## API parity vs. our v2 adapter — VERDICT: MATCH

The self-hosted image **does expose `/v2`**. Upstream `apps/api/src/index.ts`
mounts `app.use("/v2", v2Router)`, and `v2Router` registers `POST /scrape`,
`POST /crawl`, and `GET /crawl/:jobId` — i.e. the exact endpoints our adapter
calls: `/v2/scrape`, `/v2/crawl`, `/v2/crawl/{id}`.

Response shapes match what `parse_scrape` / `parse_crawl_status` deserialize:
the v2 `ScrapeResponse` success branch is `{ success: true, data: Document }`
where `Document.markdown?: string` and `Document.metadata` carries
`sourceURL?: string` and `statusCode: number` — the fields our `ScrapeData` /
`PageMetadata` read. Error bodies are `{ success: false, error: ... }`, which our
tolerant deserializer already handles.

> Caveat (call out in the G4 write-up): the upstream **SELF_HOST.md** example
> still uses `POST /v1/crawl`. `/v1` exists too, but our adapter targets `/v2`,
> and `/v2` is mounted in the self-hosted build — so no adapter change is needed.
> Self-host has historically lagged cloud; this is pinned to image `2.10.19`. If
> a future image drops/changes `/v2`, re-verify before re-running.

## Pointing the harness at this instance

Our adapter reads a base-URL env: `FirecrawlConfig::from_env()` uses
`FIRECRAWL_BASE_URL` (falling back to the cloud default) and `FIRECRAWL_API_KEY`.
The runner only builds the Firecrawl client when `FIRECRAWL_API_KEY` is **set
and non-empty** — so for the local instance you must set a *dummy* key to
activate the path (self-host ignores it). Exact invocation:

```bash
FIRECRAWL_BASE_URL=http://localhost:3002/v2 \
FIRECRAWL_API_KEY=local-selfhost-dummy \
cargo run -p web-search-benchmark --bin firecrawl-compare -- \
    --urls benchmark/urls.jsonl \
    --server target/release/web-search-mcp \
    --out benchmark/RESULTS.firecrawl.selfhost.md
```

Note the base URL **includes `/v2`** — the adapter appends `/scrape`,
`/crawl`, `/crawl/{id}` to `base_url`, so `base_url` must be the version root.
