#!/usr/bin/env bash
# Launch the re-arming stdio->SSE gateway for our web-search MCP server, for the
# MCPBench keyed accuracy run. MUST be run under WSL/Linux (not Git Bash) — it
# spawns the Windows .exe via WSL interop (/mnt/c/...).
#
# Replaces `npx -y supergateway` (which dies after one SSE session — see
# rearm_gateway.mjs header). Listens on port 8005, /sse, matching
# configs/mcp_config_websearch_ours.json (MCPBench builds
# http://localhost:8005/sse from run_config[0].port).
#
# Usage (WSL terminal):
#   bash launch_rearm_gateway.sh
# Leave it running; in a second WSL terminal run the eval (see README).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSM="${WSM:-/mnt/c/Users/rutvi/Downloads/projects/web_search_mcp/target/release/web-search-mcp.exe}"
PORT="${PORT:-8005}"

if [[ ! -f "$WSM" ]]; then
  echo "FATAL: server binary not found at $WSM" >&2
  echo "Set WSM=/mnt/c/.../web-search-mcp.exe or rebuild the release binary." >&2
  exit 1
fi

cd "$HERE"   # so node resolves @modelcontextprotocol/sdk + express from ./node_modules
echo "[launch] rearm gateway -> port $PORT, stdio=$WSM"
exec node rearm_gateway.mjs \
  --stdio "$WSM" \
  --port "$PORT" \
  --baseUrl "http://localhost:$PORT" \
  --ssePath /sse \
  --messagePath /message \
  --logLevel info
