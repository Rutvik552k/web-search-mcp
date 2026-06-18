use std::sync::Arc;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, Content, ListToolsResult, PaginatedRequestParams,
    ServerCapabilities, ServerInfo,
};
use rmcp::service::{RequestContext, RoleServer};
use rmcp::{ErrorData as McpError, ServerHandler};
use web_search_common::config::Config;
use web_search_orchestrator::SearchEngine;

/// Log progress updates to stderr as structured JSON for MCP-aware clients.
fn spawn_progress_logger(engine: &SearchEngine) {
    if let Some(mut rx) = engine.subscribe_progress() {
        tokio::spawn(async move {
            while let Ok(update) = rx.recv().await {
                tracing::info!(
                    progress = update.progress,
                    current = update.current,
                    total = ?update.total,
                    message = %update.message,
                    "progress"
                );
            }
        });
    }
}

use crate::config_resolver::{resolve_config_path, validate_search_config, ConfigSource};
use crate::tools::tool_definitions;

pub struct WebSearchServer {
    engine: Arc<SearchEngine>,
}

impl WebSearchServer {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let config = Self::load_config()?;

        let engine = Arc::new(SearchEngine::new(config)?);

        // Start progress logger for long-running operations
        spawn_progress_logger(&engine);

        // Start background crawl daemon — pre-indexes content before queries arrive
        engine.start_daemon();

        tracing::info!("WebSearchServer initialized");
        Ok(Self { engine })
    }

    /// Resolve the config path (CLI/env/default), load it with the §7 fail-fast
    /// rule, log the resolved ABSOLUTE path (Addendum A.5), and validate the
    /// self-contained search config (§8). Operator-facing: the startup error here
    /// goes to stderr/logs and MAY include the path — it never reaches an MCP
    /// client.
    fn load_config() -> Result<Config, Box<dyn std::error::Error>> {
        let source = resolve_config_path(
            &std::env::args().collect::<Vec<_>>(),
            |k| std::env::var(k).ok(),
        );
        let path = source.path();
        // Log the resolved ABSOLUTE path so a CWD-hijacked default file is
        // observable, not silent (ADR 0004 Addendum A.5).
        let abs = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        tracing::info!(config_path = %abs.display(), explicit = source.is_explicit(), "resolving config");

        let config = match &source {
            // Explicit path: missing or unparseable is an operator error → fail
            // fast, never silently fall back (ADR 0004 §7).
            ConfigSource::Explicit(p) => {
                if !p.exists() {
                    return Err(format!(
                        "config file named explicitly (--config / WEB_SEARCH_MCP_CONFIG) \
                         not found: {}",
                        abs.display()
                    )
                    .into());
                }
                Config::load(p).map_err(|e| -> Box<dyn std::error::Error> {
                    format!("failed to load explicitly-named config {}: {e}", abs.display()).into()
                })?
            }
            // Default path: missing => embedded defaults (current behavior).
            ConfigSource::Default(p) => Config::load(p).unwrap_or_else(|e| {
                tracing::warn!(error = %e, "config/default.toml failed to load; using embedded defaults");
                Config::default()
            }),
        };

        // §8 startup validation — fail fast on a pinned source that can't run or
        // an unknown provider.
        validate_search_config(&config, |k| std::env::var(k).ok())
            .map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;

        Ok(config)
    }

    /// Pre-warm ML models so the first real request doesn't pay cold-start cost.
    pub async fn warmup(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.engine.warmup().await?;
        Ok(())
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
                 - instant_search: Ultra-fast cached search (~1-2s)\n\
                 - streaming_search: Progressive results (partial ~3s, refined ~10s)\n\
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
