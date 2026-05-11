use rmcp::model::{
    CallToolRequestParams, CallToolResult, Content, ListToolsResult, PaginatedRequestParams,
    ServerCapabilities, ServerInfo,
};
use rmcp::service::{RequestContext, RoleServer};
use rmcp::{ErrorData as McpError, ServerHandler};
use web_search_common::config::Config;
use web_search_orchestrator::SearchEngine;

use crate::tools::tool_definitions;

pub struct WebSearchServer {
    engine: SearchEngine,
}

impl WebSearchServer {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let config = Config::load(std::path::Path::new("config/default.toml"))
            .unwrap_or_else(|_| Config::default());

        let engine = SearchEngine::new(config)?;

        tracing::info!("WebSearchServer initialized");
        Ok(Self { engine })
    }
}

impl ServerHandler for WebSearchServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "Web Search MCP Server — a self-contained research engine. \
                 Crawls, indexes, and ranks web content directly without external APIs. \
                 Provides verified results with confidence scores and contradiction detection \
                 to prevent LLM hallucination.\n\n\
                 Smart tools:\n\
                 - deep_research: Multi-wave crawl + rank (thorough, 2min)\n\
                 - quick_search: Fast single-wave search (5-15s)\n\
                 - explore_topic: Discovery mode with entity graph\n\
                 - verify_claim: Fact-checking with evidence\n\
                 - compare_sources: Side-by-side URL comparison\n\n\
                 Atomic tools:\n\
                 - fetch_page, extract, follow_links, paginate\n\
                 - search_index, find_similar, get_entities, get_link_graph"
                    .into(),
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..ServerInfo::default()
        }
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> std::result::Result<ListToolsResult, McpError> {
        Ok(ListToolsResult {
            tools: tool_definitions(),
            next_cursor: None,
            meta: None,
        })
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> std::result::Result<CallToolResult, McpError> {
        let name = &request.name;
        let args = request.arguments.unwrap_or_default();

        tracing::info!(tool = %name, "Tool called");

        match web_search_orchestrator::tools::dispatch_tool(&self.engine, name, &args).await {
            Ok(result) => {
                tracing::info!(tool = %name, result_len = result.len(), "Tool succeeded");
                Ok(CallToolResult::success(vec![Content::text(result)]))
            }
            Err(e) => {
                tracing::error!(tool = %name, error = %e, "Tool failed");
                Ok(CallToolResult::error(vec![Content::text(format!(
                    "[ERROR] {e}"
                ))]))
            }
        }
    }
}
