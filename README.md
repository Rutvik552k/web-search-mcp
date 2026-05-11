# Web Search MCP Server

A self-contained, API-free web search engine built in Rust that runs as an MCP (Model Context Protocol) server. Crawls, indexes, and ranks web content directly — no Google API, no Bing API, no external dependencies. Designed to provide verified, grounded results that prevent LLM hallucination.

## What It Does

- **Crawls the web directly** — no search API keys needed
- **Extracts clean content** — 2-pass consensus extraction (Readability + Trafilatura-inspired)
- **Indexes everything** — full-text search (Tantivy/BM25) + vector similarity (HNSW)
- **Ranks with anti-hallucination** — 5-stage pipeline: ISR fusion, cross-encoder rerank, authority scoring, contradiction detection, diversity filtering
- **Works with any LLM** — standard MCP protocol over stdio

## Architecture

```
                         MCP Server (stdio)
                              |
                         Orchestrator
                     /    |    |    \
               Crawler  Extractor  Indexer  Ranker
                  |        |         |        |
              HTTP/Browser  2-Pass   Tantivy  5-Stage Pipeline
              Rate Limit    Consensus  HNSW     ISR Fusion
              Robots.txt   Metadata   SimHash   Anti-Hallucination
              Pagination   Chunker    Dedup     Diversity Filter
                                        |
                                     Embedder
                                   Hash / Candle
```

### 8 Crates

| Crate | Purpose |
|-------|---------|
| `common` | Data models, config, errors, logging |
| `crawler` | Distributed crawler with SPA detection, robots.txt, rate limiting, pagination |
| `extractor` | Readability + Trafilatura content extraction, metadata, JSON-LD, tables |
| `indexer` | Tantivy full-text + HNSW vectors + SimHash dedup |
| `embedder` | HashEmbedder (default) or CandleEmbedder (neural, optional) |
| `ranker` | 5-stage anti-hallucination ranking pipeline |
| `orchestrator` | Research flow engine connecting all components |
| `mcp-server` | MCP protocol layer with 13 tool definitions |

## 13 MCP Tools

### Smart Tools (high-level research)

| Tool | Description |
|------|-------------|
| `deep_research` | Multi-wave crawl across hundreds of pages. Follows links, pagination. Returns verified results with confidence scores and contradiction detection |
| `quick_search` | Fast single-wave search (~15s). Good for simple factual questions |
| `explore_topic` | Discovery mode — broadly explores a topic, builds entity connections |
| `verify_claim` | Fact-checking — searches for evidence supporting or contradicting a claim |
| `compare_sources` | Side-by-side content comparison from multiple URLs |

### Atomic Tools (fine-grained control)

| Tool | Description |
|------|-------------|
| `fetch_page` | Fetch raw content from a URL with retry and anti-blocking |
| `extract` | Extract clean text using 2-pass consensus algorithm |
| `follow_links` | Follow links from a page, optionally filtered by pattern |
| `paginate` | Auto-detect and follow pagination (5 patterns supported) |
| `search_index` | Query the local full-text index from previously crawled content |
| `find_similar` | Find semantically similar content using vector embeddings |
| `get_entities` | Extract named entities (persons, organizations, locations, dates) |
| `get_link_graph` | Map outgoing links from a URL |

## Anti-Hallucination Pipeline

The 5-stage ranking pipeline is designed to provide accurate, grounded results:

```
Stage 1: Dual Retrieval (BM25 + Vector) → ISR fusion     ~8ms
Stage 2: Cross-Encoder Rerank                             ~50ms
Stage 3: Authority (TrustRank) + Freshness boost          ~1ms
Stage 4: Anti-Hallucination checks                        ~15ms
         - Cross-reference validation (3+ sources = Verified)
         - Contradiction detection (NLI)
         - Echo chamber detection (unique source orgs)
         - Claim-source attribution
Stage 5: Diversity filter (MMR + domain cap + dedup)      ~3ms
```

### Source Tiers

| Tier | Domains | Weight |
|------|---------|--------|
| Tier 1 | .gov, .edu, Nature, ArXiv, WHO, NASA | 1.3x |
| Tier 2 | BBC, Reuters, Wikipedia, StackOverflow, GitHub | 1.1x |
| Tier 3 | Medium, Reddit, Dev.to, Substack | 0.9x |
| Tier 4 | Unknown / unverified | 0.7x |

## Requirements

- **Rust 1.85+** (edition 2024)
- **No API keys** — the server crawls the web directly
- **No GPU required** — HashEmbedder works on CPU (optional: CandleEmbedder for neural embeddings)

## Installation

### Build from source

```bash
git clone https://github.com/Rutvik552k/web-search-mcp.git
cd web-search-mcp
cargo build --release
```

The binary will be at `target/release/web-search-mcp` (or `web-search-mcp.exe` on Windows).

### Optional: Neural embeddings

To enable neural embeddings via Candle (pure Rust, no Python):

```bash
cargo build --release --features "web-search-embedder/candle"
```

This downloads the `all-MiniLM-L6-v2` model (~22MB) from HuggingFace on first run.

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

Then in Claude Code, the tools are available automatically. Examples:

```
Use quick_search to find information about quantum computing
Use deep_research to investigate the impact of microplastics on marine life
Use verify_claim to check if "The Great Wall of China is visible from space"
Use extract to get clean content from https://example.com/article
```

### With Any MCP Client

The server communicates over **stdio** using JSON-RPC 2.0 (MCP protocol). Any MCP-compatible client works:

```bash
# Start the server
./target/release/web-search-mcp

# Send JSON-RPC messages on stdin, receive on stdout
# Logs go to stderr
```

Example initialization:

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"my-app","version":"1.0"}}}
```

Example tool call:

```json
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"quick_search","arguments":{"query":"Rust vs Go performance","max_results":5}}}
```

### Standalone Testing

```bash
# Quick search
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}
{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"extract","arguments":{"url":"https://en.wikipedia.org/wiki/Rust_(programming_language)"}}}' | timeout 20 ./target/release/web-search-mcp
```

## Configuration

Default config at `config/default.toml`. Key settings:

```toml
[crawler]
num_workers = 8                          # concurrent crawler threads
requests_per_second_per_domain = 2.0     # politeness rate limit
request_timeout_secs = 30
respect_robots_txt = true

[ranker]
bm25_top_k = 200          # Stage 1 BM25 candidates
hnsw_top_k = 200           # Stage 1 vector candidates
rerank_top_k = 50          # Stage 2 cross-encoder candidates
mmr_lambda = 0.7           # diversity vs relevance balance
max_results_per_domain = 2 # prevent single-source dominance
min_unique_orgs = 3        # echo chamber detection threshold

[embedder]
embedding_dim = 384        # MiniLM-L6 dimensions
batch_size = 32
```

Source credibility tiers at `config/source_tiers.toml`.

## Response Format

All search tools return structured JSON with verification metadata:

```json
{
  "results": [
    {
      "content": "extracted article text...",
      "url": "https://source.com/article",
      "title": "Article Title",
      "confidence": 0.95,
      "verification": "Verified",
      "claims": [
        {
          "text": "specific claim from the article",
          "confidence": 0.95,
          "verification": "Verified"
        }
      ],
      "contradictions": [
        {
          "claim_a": "Source A says X",
          "claim_b": "Source B says Y",
          "severity": "Hard"
        }
      ],
      "source_tier": "Tier1",
      "freshness": "2026-05-10T00:00:00Z",
      "relevance_score": 1.25
    }
  ],
  "warnings": ["limited source diversity on topic X"],
  "coverage_score": 0.85,
  "total_pages_crawled": 50,
  "total_time_ms": 77
}
```

## Tests

```bash
cargo test
```

144 tests across all crates covering:
- Content extraction accuracy
- SimHash near-duplicate detection
- ISR fusion ranking correctness
- MMR diversity promotion
- Query type detection
- Authority classification
- Freshness decay curves
- Anti-hallucination checks (cross-reference, contradiction, echo chamber)
- URL frontier dedup and priority
- Pagination pattern detection
- Robots.txt parsing
- Link extraction and resolution

## License

**Source Available — Non-Commercial Use Only**

This software is free for personal, educational, academic, research, and development use. Commercial use requires written permission from the admin.

See [LICENSE](LICENSE) for full terms.

For commercial licensing inquiries: [GitHub](https://github.com/Rutvik552k)
