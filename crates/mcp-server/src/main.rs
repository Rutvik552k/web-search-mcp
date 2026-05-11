use rmcp::ServiceExt;
use tracing_subscriber::{fmt, EnvFilter};

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
    let transport = rmcp::transport::io::stdio();
    let service = server.serve(transport).await?;
    service.waiting().await?;

    Ok(())
}
