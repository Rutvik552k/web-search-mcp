use rmcp::model::Tool;
use std::sync::Arc;

/// Define all 13 MCP tools — 5 smart + 8 atomic.
pub fn tool_definitions() -> Vec<Tool> {
    vec![
        // ── Smart Tools (high-level intent) ──────────────────────────────
        make_tool(
            "deep_research",
            "Multi-wave deep research: crawls hundreds of pages across multiple waves, follows links and pagination, indexes content, ranks with anti-hallucination pipeline. Returns verified results with confidence scores, claim attribution, and contradiction detection.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Research query" },
                    "depth": { "type": "integer", "description": "Crawl depth (1-5)", "default": 3 },
                    "max_pages": { "type": "integer", "description": "Maximum pages to crawl", "default": 500 },
                    "time_limit_secs": { "type": "integer", "description": "Time budget in seconds", "default": 120 }
                },
                "required": ["query"]
            }),
        ),
        make_tool(
            "quick_search",
            "Fast single-wave search: crawls ~50 pages, extracts and ranks. Returns top results in 5-15 seconds.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query" },
                    "max_results": { "type": "integer", "description": "Number of results", "default": 10 }
                },
                "required": ["query"]
            }),
        ),
        make_tool(
            "explore_topic",
            "Discovery mode: broadly explores a topic, follows links, builds entity graph. Returns topic map with key sources.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "topic": { "type": "string", "description": "Topic to explore" },
                    "breadth": { "type": "string", "enum": ["narrow", "medium", "wide"], "default": "medium" }
                },
                "required": ["topic"]
            }),
        ),
        make_tool(
            "verify_claim",
            "Fact-checking: searches for evidence supporting or contradicting a claim. Returns evidence with confidence.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "claim": { "type": "string", "description": "Claim to verify" },
                    "min_sources": { "type": "integer", "description": "Minimum sources to check", "default": 5 }
                },
                "required": ["claim"]
            }),
        ),
        make_tool(
            "compare_sources",
            "Side-by-side comparison of content from multiple URLs on a specific aspect.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "urls": { "type": "array", "items": { "type": "string" }, "description": "URLs to compare" },
                    "aspect": { "type": "string", "description": "Aspect to compare on" }
                },
                "required": ["urls", "aspect"]
            }),
        ),
        // ── Atomic Tools (fine-grained control) ──────────────────────────
        make_tool(
            "fetch_page",
            "Fetch a single URL and return raw content. Handles anti-blocking and retries.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "url": { "type": "string", "description": "URL to fetch" }
                },
                "required": ["url"]
            }),
        ),
        make_tool(
            "extract",
            "Extract clean text from a URL using 3-pass consensus. Returns structured content.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "url": { "type": "string", "description": "URL to extract from" }
                },
                "required": ["url"]
            }),
        ),
        make_tool(
            "follow_links",
            "Follow links from a page matching a pattern.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "url": { "type": "string", "description": "Starting URL" },
                    "pattern": { "type": "string", "description": "URL pattern or CSS selector" },
                    "depth": { "type": "integer", "default": 1 }
                },
                "required": ["url"]
            }),
        ),
        make_tool(
            "paginate",
            "Auto-detect and follow pagination on a page.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "url": { "type": "string", "description": "URL of first page" },
                    "max_pages": { "type": "integer", "default": 10 }
                },
                "required": ["url"]
            }),
        ),
        make_tool(
            "search_index",
            "Query the local search index from previously crawled content.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query" },
                    "max_results": { "type": "integer", "default": 20 }
                },
                "required": ["query"]
            }),
        ),
        make_tool(
            "find_similar",
            "Find semantically similar content using vector embeddings.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "Text to find similar content for" },
                    "top_k": { "type": "integer", "default": 10 }
                },
                "required": ["text"]
            }),
        ),
        make_tool(
            "get_entities",
            "Extract named entities (persons, organizations, locations, dates) from text.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "Text to extract entities from" }
                },
                "required": ["text"]
            }),
        ),
        make_tool(
            "get_link_graph",
            "Build a link graph showing outgoing and incoming links for a URL.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "url": { "type": "string", "description": "URL to analyze" },
                    "depth": { "type": "integer", "default": 1 }
                },
                "required": ["url"]
            }),
        ),
    ]
}

fn make_tool(name: &str, description: &str, schema: serde_json::Value) -> Tool {
    let schema_obj: serde_json::Map<String, serde_json::Value> =
        serde_json::from_value(schema).expect("valid tool schema");
    Tool::new(
        name.to_string(),
        description.to_string(),
        Arc::new(schema_obj),
    )
}
