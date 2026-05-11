use tracing_subscriber::{fmt, EnvFilter};

/// Initialize logging to stderr (stdout is reserved for MCP stdio transport).
pub fn init() {
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .with_target(false)
        .init();
}
