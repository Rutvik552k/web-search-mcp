use rmcp::ServiceExt;
use tracing_subscriber::{fmt, EnvFilter};

mod config_resolver;
mod server;
mod tools;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Log to stderr — stdout reserved for MCP stdio transport
    fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with_writer(std::io::stderr)
        .with_target(false)
        .init();

    tracing::info!("Starting Web Search MCP Server v{}", env!("CARGO_PKG_VERSION"));

    let server = server::WebSearchServer::new()?;

    // Pre-warm ML models in background — first real query won't pay cold-start cost
    let warmup_start = std::time::Instant::now();
    if let Err(e) = server.warmup().await {
        tracing::warn!("Warmup failed (non-fatal): {e}");
    } else {
        tracing::info!(elapsed_ms = warmup_start.elapsed().as_millis() as u64, "ML models warmed up");
    }

    let transport = rmcp::transport::io::stdio();
    let service = server.serve(transport).await?;
    service.waiting().await?;

    tracing::info!("MCP server shutting down — flushing caches to disk");
    // Drop triggers flush_to_disk via SearchEngine's Drop impl

    Ok(())
}
