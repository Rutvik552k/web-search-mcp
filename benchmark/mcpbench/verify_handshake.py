#!/usr/bin/env python3
"""Minimal MCPBench-equivalent SSE handshake verifier.

Mirrors what MCPBench's langProBe/async_mcp_client.py + synced_mcp_client.py do:
  1. connect to the MCP server over SSE at http://localhost:<port>/sse
  2. initialize the MCP session
  3. list_tools()           <- MCPBench uses this to build the system prompt
  4. call_tool(name, args)  <- MCPBench uses this when the judged LLM emits <tool>

This does NOT need any LLM key — it exercises ONLY the transport + tool surface,
which is the furthest verification possible without the operator's judge key.

Usage:
    python verify_handshake.py --url http://localhost:8005/sse \
        --tool instant_search --query "what is rust programming language"
"""
import argparse
import asyncio
import json
import sys

from mcp import ClientSession
from mcp.client.sse import sse_client


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8005/sse")
    ap.add_argument("--tool", default="instant_search")
    ap.add_argument("--query", default="what is the rust programming language")
    ap.add_argument("--call", action="store_true",
                    help="also issue a real call_tool (may hit the network)")
    args = ap.parse_args()

    print(f"[verify] connecting SSE -> {args.url}", flush=True)
    async with sse_client(args.url) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print(f"[verify] initialize OK -> server={init.serverInfo.name} "
                  f"v{init.serverInfo.version}", flush=True)

            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            print(f"[verify] list_tools OK -> {len(names)} tools: {names}",
                  flush=True)

            if args.tool not in names:
                print(f"[verify] FAIL: expected tool {args.tool!r} not exposed",
                      file=sys.stderr)
                return 2

            # show the schema MCPBench will render into the system prompt
            spec = next(t for t in tools.tools if t.name == args.tool)
            print(f"[verify] schema[{args.tool}] = "
                  f"{json.dumps(spec.inputSchema)}", flush=True)

            if args.call:
                print(f"[verify] call_tool({args.tool}, query={args.query!r}) ...",
                      flush=True)
                res = await session.call_tool(args.tool, {"query": args.query})
                text = "".join(
                    c.text for c in res.content if getattr(c, "text", None)
                )
                print(f"[verify] call_tool returned {len(text)} chars, "
                      f"isError={res.isError}", flush=True)
                print("[verify] first 600 chars:\n" + text[:600], flush=True)

            print("[verify] HANDSHAKE OK", flush=True)
            return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
