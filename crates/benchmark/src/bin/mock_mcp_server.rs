//! Hermetic mock MCP server for the benchmark harness self-test.
//!
//! It speaks the SAME MCP stdio JSON-RPC protocol as the real `web-search-mcp`
//! server but returns deterministic canned responses — no network, no ML model
//! downloads, no flakiness. Its only purpose is to let `benchmark` exercise the
//! full handshake + tool-call round-trip and verify the metric math end-to-end.
//!
//! Canned data is aligned with benchmark/urls.sample.jsonl and
//! benchmark/queries.sample.jsonl so the self-test produces known numbers:
//!   coverage = 2/3, blocked subset = 0/1,
//!   mean nDCG@10 ≈ 0.7853, mean precision@5 = 0.4000.

use std::sync::Arc;

use rmcp::ServiceExt;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, Content, ListToolsResult, PaginatedRequestParams,
    ServerCapabilities, ServerInfo, Tool,
};
use rmcp::service::{RequestContext, RoleServer};
use rmcp::{ErrorData as McpError, ServerHandler};

#[derive(Clone)]
struct MockServer;

impl ServerHandler for MockServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some("Mock web-search MCP server (benchmark self-test)".into()),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..ServerInfo::default()
        }
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        let schema_url: serde_json::Map<String, serde_json::Value> = serde_json::from_value(
            serde_json::json!({
                "type": "object",
                "properties": { "url": { "type": "string" } },
                "required": ["url"]
            }),
        )
        .unwrap();
        let schema_query: serde_json::Map<String, serde_json::Value> = serde_json::from_value(
            serde_json::json!({
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": ["query"]
            }),
        )
        .unwrap();
        Ok(ListToolsResult {
            tools: vec![
                Tool::new("extract", "Mock extract", Arc::new(schema_url)),
                Tool::new("quick_search", "Mock search", Arc::new(schema_query)),
            ],
            next_cursor: None,
            meta: None,
        })
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let args = request.arguments.unwrap_or_default();
        match request.name.as_ref() {
            "extract" => {
                let url = args.get("url").and_then(|v| v.as_str()).unwrap_or("");
                Ok(CallToolResult::success(vec![Content::text(mock_extract(url))]))
            }
            "quick_search" => {
                let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
                Ok(CallToolResult::success(vec![Content::text(mock_search(query))]))
            }
            other => Ok(CallToolResult::error(vec![Content::text(format!(
                "[ERROR] unknown tool: {other}"
            ))])),
        }
    }
}

/// Canned extraction keyed by URL substring (mirrors urls.sample.jsonl).
fn mock_extract(url: &str) -> String {
    let (title, body) = if url.contains("protected.example.com") {
        // Simulated Cloudflare block: short body, no expected marker.
        ("Attention Required! | Cloudflare", "Please enable cookies. Sorry, you have been blocked.".to_string())
    } else if url.contains("spa.example.com") {
        // JS-SPA rendered to clean text, no expect_contains marker required.
        (
            "Demo SPA Dashboard",
            format!(
                "{} The single-page application rendered its main view with charts, \
                 tables, and a detailed narrative describing quarterly metrics and trends.",
                "Welcome to the dashboard. ".repeat(8)
            ),
        )
    } else if url.contains("example.com") {
        // Static HTML article containing the expected marker.
        (
            "Example Domain",
            format!(
                "Example Domain. This domain is for use in illustrative examples in documents. \
                 You may use this domain in literature without prior coordination or asking for \
                 permission. {}",
                "More illustrative body text follows here. ".repeat(4)
            ),
        )
    } else {
        ("Unknown", "n/a".to_string())
    };

    serde_json::json!({
        "url": url,
        "title": title,
        "author": serde_json::Value::Null,
        "published_date": serde_json::Value::Null,
        "body_text": body,
        "headings": [],
        "tables": [],
        "language": "en",
        "extraction_confidence": 0.9,
    })
    .to_string()
}

/// Canned ranked results keyed by query substring (mirrors queries.sample.jsonl).
fn mock_search(query: &str) -> String {
    let urls: Vec<&str> = if query.contains("rust async") {
        vec![
            "https://tokio.rs/",                  // relevant (rank 1)
            "https://some-blog.example/rust",     // not
            "https://docs.rs/tokio",              // relevant (rank 3)
            "https://news.example/rust-async",    // not
            "https://forum.example/threads/42",   // not
        ]
    } else if query.contains("python web") {
        vec![
            "https://www.w3schools.com/python",        // not
            "https://flask.palletsprojects.com/",      // relevant (rank 2)
            "https://realpython.com/flask-vs-django",  // not
            "https://www.djangoproject.com/",          // relevant (rank 4)
            "https://geeksforgeeks.org/python-web",    // not
        ]
    } else {
        vec![]
    };

    let results: Vec<serde_json::Value> = urls
        .iter()
        .map(|u| {
            serde_json::json!({
                "url": u,
                "title": u,
                "content": "snippet",
                "confidence": 0.8,
                "relevance_score": 1.0,
            })
        })
        .collect();

    serde_json::json!({
        "query": query,
        "results": results,
        "synthesis": [],
        "warnings": [],
        "total_pages_crawled": results_len(&results),
        "total_time_ms": 1,
    })
    .to_string()
}

fn results_len(v: &[serde_json::Value]) -> usize {
    v.len()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Logs to stderr — stdout is reserved for the MCP stdio transport.
    eprintln!("[mock-mcp-server] starting");
    let transport = rmcp::transport::io::stdio();
    let service = MockServer.serve(transport).await?;
    service.waiting().await?;
    Ok(())
}
