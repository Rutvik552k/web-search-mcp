# Web Search MCP Server

A self-contained, API-free web search engine built in Rust that runs as an MCP (Model Context Protocol) server. Crawls, indexes, and ranks web content directly — no Google API, no Bing API, no external search dependencies. Uses neural ML models for semantic understanding, cross-encoder reranking, and NLI-based contradiction detection to provide verified, grounded results that prevent LLM hallucination.

## What It Does

- **Crawls the web directly** — no search API keys needed, parses 9 search engines (Brave, Mojeek, DuckDuckGo, Wikipedia, ArXiv, Reddit, HN, Google Scholar, PubMed)
- **Parallel crawling** — concurrent batch fetching with per-domain rate limiting
- **Extracts clean content** — 2-pass consensus extraction (Readability + Trafilatura-inspired) with CSS/JS sanitization
- **Neural embeddings** — all-MiniLM-L6-v2 via Candle (pure Rust, no Python) for semantic search
- **Cross-encoder reranking** — ms-marco-MiniLM-L-6-v2 for token-level relevance scoring
- **NLI contradiction detection** — nli-deberta-v3-small classifies claim pairs as entailment/contradiction/neutral
- **5-stage anti-hallucination pipeline** — ISR fusion, cross-encoder rerank, authority scoring, NLI contradiction detection, diversity filtering
- **Persistent storage** — tantivy index + HNSW vectors + dedup state survive across sessions
- **Query reformulation** — synonym expansion for better recall
- **Works with any LLM** — standard MCP protocol over stdio

## Architecture

```
                         MCP Server (stdio, concurrent)
                              |
                         Orchestrator
                     /    |    |    \
               Crawler  Extractor  Indexer  Ranker
                  |        |         |        |
              Parallel   2-Pass    Tantivy   5-Stage Pipeline
              HTTP Fetch  Consensus  HNSW     Cross-Encoder Rerank
              Search      CSS/JS    SimHash   NLI Contradictions
              Result      Sanitize  Dedup     Authority + Freshness
              Parser      Snippets  Persist   MMR Diversity
              Sitemap     Metadata    |
              Robots.txt  Chunker  Embedder
              Pagination          MiniLM-L6 (Candle)
```

### 8 Crates

| Crate | Purpose |
|-------|---------|
| `common` | Data models, config, errors, logging |
| `crawler` | Parallel crawler with search result parsing, SPA detection, robots.txt, rate limiting, pagination, sitemap |
| `extractor` | Readability + Trafilatura extraction, CSS/JS sanitization, metadata, JSON-LD, tables, snippet extraction, chunking |
| `indexer` | Tantivy full-text (disk-backed) + HNSW vectors (persistent) + SimHash/dedup (persistent) |
| `embedder` | CandleEmbedder (neural, default) + HashEmbedder (fallback) + CrossEncoder (reranking + NLI) |
| `ranker` | 5-stage anti-hallucination ranking pipeline with ML models |
| `orchestrator` | Research flow engine with query reformulation, hybrid search wiring |
| `mcp-server` | MCP protocol layer with 13 tool definitions, concurrent tool calls |

## 13 MCP Tools

### Smart Tools (high-level research)

| Tool | Description |
|------|-------------|
| `deep_research` | Multi-wave crawl across hundreds of pages. Query reformulation with synonym expansion. Follows search result links, pagination. Returns verified results with confidence scores, claim attribution, and NLI-based contradiction detection |
| `quick_search` | Fast single-wave search (~15s). Good for simple factual questions |
| `explore_topic` | Discovery mode — broadly explores a topic, builds entity connections |
| `verify_claim` | Fact-checking — searches for evidence supporting or contradicting a claim. Reports source diversity and contradiction severity |
| `compare_sources` | Side-by-side content comparison from multiple URLs on a specific aspect |

### Atomic Tools (fine-grained control)

| Tool | Description |
|------|-------------|
| `fetch_page` | Fetch raw content from a URL with retry, anti-blocking headers, and SPA detection |
| `extract` | Extract clean text using 2-pass consensus with CSS/JS sanitization |
| `follow_links` | Follow links from a page, optionally filtered by URL pattern or anchor text |
| `paginate` | Auto-detect and follow pagination (5 patterns: query param, offset, cursor, path segment, rel=next) |
| `search_index` | Query the persistent local full-text index from previously crawled content |
| `find_similar` | Find semantically similar content using neural vector embeddings (MiniLM-L6) |
| `get_entities` | Extract named entities (persons, organizations, locations, dates) with type classification |
| `get_link_graph` | Map outgoing links from a URL with anchor text and external/internal classification |

## Anti-Hallucination Pipeline

All 5 stages now fully operational with ML models:

```
Stage 1: Dual Retrieval                                   ~8ms
         BM25 (tantivy) + Vector (HNSW/MiniLM-L6)
         Merge via ISR (1/rank²) — steeper than RRF
         
Stage 2: Cross-Encoder Rerank                             ~50ms
         ms-marco-MiniLM-L-6-v2 (auto-downloaded, ~80MB)
         Token-level attention scoring on (query, doc) pairs
         Blend: 0.3 × ISR + 0.7 × cross-encoder score
         
Stage 3: Authority + Freshness                            ~1ms
         TrustRank via domain tier classification
         Adaptive freshness decay per query type
         News: λ=0.1 (7-day half-life)
         Research: λ=0.005 (139-day half-life)
         
Stage 4: Anti-Hallucination Layer                         ~15ms
         A. Cross-reference validation (3+ sources = Verified)
         B. NLI contradiction detection (nli-deberta-v3-small, ~140MB)
            Pairwise entailment/contradiction/neutral classification
            Confidence threshold: 0.7 for flagging
         C. Echo chamber detection (unique source organizations)
         D. Numeric contradiction heuristic (20% threshold)
         E. Claim-source attribution mapping
         
Stage 5: Diversity Filter                                 ~3ms
         Query-relevant snippet extraction
         SimHash near-duplicate removal (hamming ≤ 3)
         MMR reranking (λ=0.7 relevance vs diversity)
         Max 2 results per domain
         Min 3 unique source organizations
```

### ML Models (auto-downloaded on first run)

| Model | HuggingFace ID | Size | Purpose |
|-------|---------------|------|---------|
| MiniLM-L6-v2 | `sentence-transformers/all-MiniLM-L6-v2` | ~22MB | Bi-encoder embeddings for semantic search |
| MS MARCO MiniLM | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~80MB | Stage 2 cross-encoder reranking |
| NLI DeBERTa v3 | `cross-encoder/nli-deberta-v3-small` | ~140MB | Stage 4 contradiction detection |

Total first-run download: ~242MB. Cached at `~/.cache/huggingface/` — subsequent runs are instant.

All models run on CPU via Candle (pure Rust). No Python, no PyTorch, no GPU required.

### Source Tiers

| Tier | Examples | Weight |
|------|---------|--------|
| Tier 1 | .gov, .edu, Nature, ArXiv, WHO, NASA, PubMed, IEEE | 1.3x |
| Tier 2 | BBC, Reuters, Wikipedia, StackOverflow, GitHub, MDN | 1.1x |
| Tier 3 | Medium, Reddit, Dev.to, Substack, HackerNews | 0.9x |
| Tier 4 | Unknown / unverified | 0.7x |

## Requirements

- **Rust 1.85+** (edition 2024)
- **No API keys** — the server crawls the web directly
- **No GPU required** — all ML models run on CPU via Candle
- **Internet** — required for first-run model download (~242MB) and web crawling
- **~500MB disk** — for model cache + persistent search index

## Installation

### Build from source

```bash
git clone https://github.com/Rutvik552k/web-search-mcp.git
cd web-search-mcp
cargo build --release
```

The binary will be at `target/release/web-search-mcp` (or `web-search-mcp.exe` on Windows).

First run downloads ML models automatically from HuggingFace (~242MB).

### Minimal build (no ML models)

For a lightweight build without neural models (uses hash-based embeddings, no reranking or NLI):

```bash
cargo build --release --no-default-features -p web-search-embedder
cargo build --release
```

## Usage

### With Claude Code

Add to your `.claude/.mcp.json`:

```json
{
  "mcpServers": {
    "web-search": {
      "command": "/path/to/web-search-mcp"
    }
  }
}
```

On Windows:

```json
{
  "mcpServers": {
    "web-search": {
      "command": "C:/path/to/web-search-mcp.exe"
    }
  }
}
```

Then in Claude Code, the tools are available automatically:

```
Use quick_search to find information about quantum computing
Use deep_research to investigate the impact of microplastics on marine life
Use verify_claim to check if "The Great Wall of China is visible from space"
Use extract to get clean content from https://en.wikipedia.org/wiki/Rust_(programming_language)
Use compare_sources to compare Rust and Go getting started guides
```

### With Any MCP Client

The server communicates over **stdio** using JSON-RPC 2.0 (MCP protocol). Any MCP-compatible client works:

```bash
# Start the server (logs to stderr, MCP on stdout)
./target/release/web-search-mcp
```

Example initialization:

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"my-app","version":"1.0"}}}
```

Example tool calls:

```json
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"quick_search","arguments":{"query":"Rust vs Go performance","max_results":5}}}

{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"verify_claim","arguments":{"claim":"Python is faster than C++","min_sources":5}}}

{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"extract","arguments":{"url":"https://en.wikipedia.org/wiki/Quantum_computing"}}}
```

### Standalone Testing

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}
{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"extract","arguments":{"url":"https://en.wikipedia.org/wiki/Rust_(programming_language)"}}}' | timeout 20 ./target/release/web-search-mcp
```

## Configuration

Default config at `config/default.toml`:

```toml
[crawler]
num_workers = 8                          # concurrent fetch tasks
requests_per_second_per_domain = 2.0     # politeness rate limit
request_timeout_secs = 30
respect_robots_txt = true

[indexer]
index_path = "data/index"               # persistent tantivy index
vector_index_path = "data/vectors"       # persistent HNSW vectors
simhash_threshold = 3                    # near-duplicate hamming distance

[ranker]
bm25_top_k = 200           # Stage 1 BM25 candidates
hnsw_top_k = 200           # Stage 1 vector candidates
rerank_top_k = 50          # Stage 2 cross-encoder output
mmr_lambda = 0.7           # diversity vs relevance (0=diverse, 1=relevant)
max_results_per_domain = 2 # prevent single-source dominance
min_unique_orgs = 3        # echo chamber detection threshold

[embedder]
embedding_dim = 384        # MiniLM-L6 dimensions
batch_size = 32
force_cpu = false          # set true to skip neural embedder

[server]
data_dir = "data"          # persistent storage directory
```

Source credibility tiers at `config/source_tiers.toml`.

## Response Format

All search tools return structured JSON with verification metadata:

```json
{
  "results": [
    {
      "content": "query-relevant snippet from the article...",
      "url": "https://source.com/article",
      "title": "Article Title",
      "confidence": 0.95,
      "verification": "Verified",
      "claims": [
        {
          "text": "specific claim extracted from the article",
          "source_url": "https://source.com/article",
          "confidence": 0.95,
          "verification": "Verified"
        }
      ],
      "contradictions": [
        {
          "claim_a": "Source A says X",
          "source_a": "https://a.com/article",
          "claim_b": "Source B says Y",
          "source_b": "https://b.com/article",
          "severity": "Hard"
        }
      ],
      "source_tier": "Tier1",
      "freshness": "2026-05-10T00:00:00Z",
      "relevance_score": 1.25
    }
  ],
  "warnings": [
    "NLI model detected semantic contradictions across sources",
    "Limited source diversity: only 2 unique organizations"
  ],
  "coverage_score": 0.85,
  "total_pages_crawled": 50,
  "total_time_ms": 77,
  "query": "original search query"
}
```

### Verification Levels

| Status | Meaning | Confidence |
|--------|---------|------------|
| `Verified` | 3+ independent source organizations confirm | 0.95 |
| `Partial` | 2 sources confirm | 0.75 |
| `Unverified` | Single source only | 0.50 |
| `Contested` | NLI model detected contradictions | 0.30 |

### Contradiction Severity

| Severity | Meaning |
|----------|---------|
| `Hard` | Direct factual contradiction (NLI confidence > 0.9 or numeric disagreement > 100%) |
| `Soft` | Nuanced disagreement (NLI confidence 0.7-0.9 or numeric disagreement 20-100%) |
| `Temporal` | Information changed over time |

## Persistent Storage

The server maintains persistent state across sessions:

```
data/
├── index/          # Tantivy full-text search index (mmap-backed)
├── vectors/
│   └── hnsw.json   # HNSW vector embeddings (384-dim per document)
└── dedup.json      # URL seen set + SimHash fingerprints + content hashes
```

- First search builds the index from scratch
- Subsequent searches query existing index AND crawl new content
- Index grows over time as more content is crawled
- Delete `data/` directory to reset

## Search Engine Coverage

The crawler parses search result pages from 9 engines and follows actual result links:

| Engine | What's Extracted |
|--------|-----------------|
| Brave Search | Result links (class `l1`) |
| Mojeek | Result links (class `title`) |
| DuckDuckGo | Result links (decoded redirect URLs) |
| Wikipedia | Article links from search results |
| ArXiv | Paper abstract links (`/abs/`) |
| Reddit (old) | Comment thread links (`/comments/`) |
| Hacker News | External links from stories |
| Google Scholar | Paper links |
| PubMed | Article links (numeric IDs) |

Query reformulation generates synonym variants (e.g., "impact" → "effect", "climate change" → "global warming") and searches multiple engines per variant.

## Tests

```bash
cargo test
```

166 tests across all crates:

- Search result parser (Brave, Mojeek, Wikipedia, ArXiv — 7 tests)
- Content extraction accuracy + CSS sanitization (10 tests)
- Snippet extraction relevance (4 tests)
- SimHash near-duplicate detection (3 tests)
- HNSW vector search (5 tests)
- Tantivy full-text index (3 tests)
- Dedup store (exact + near + URL — 5 tests)
- ISR fusion + MMR diversity (4 tests)
- Query type detection (5 tests)
- Authority classification + domain-to-org mapping (11 tests)
- Freshness decay curves per query type (8 tests)
- Anti-hallucination checks (cross-reference, contradiction, echo chamber — 7 tests)
- URL frontier dedup, priority, domain limits (7 tests)
- Pagination pattern detection + URL generation (7 tests)
- Robots.txt parsing + cache (7 tests)
- Link extraction and resolution (7 tests)
- Query reformulation with synonyms (5 tests)
- Sitemap XML parsing (3 tests)
- Cosine similarity + embeddings (10 tests)

## Project Stats

| Metric | Value |
|--------|-------|
| Language | Rust (edition 2024) |
| Total lines of code | ~16,000 |
| Crates | 8 |
| MCP tools | 13 (5 smart + 8 atomic) |
| Tests | 166 passing |
| ML models | 3 (auto-downloaded) |
| Binary size | ~27MB (release) |
| Ranking pipeline latency | ~77ms |
| External API dependencies | 0 |

## License

**Source Available — Non-Commercial Use Only**

This software is free for personal, educational, academic, research, and development use. Commercial use requires written permission from the admin.

See [LICENSE](LICENSE) for full terms.

For commercial licensing inquiries: [GitHub](https://github.com/Rutvik552k)
