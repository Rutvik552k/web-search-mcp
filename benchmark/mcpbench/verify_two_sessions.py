#!/usr/bin/env python3
"""Prove a stdio->SSE gateway survives TWO sequential SSE sessions.

This is the exact failure mode MCPBench triggers: langProBe/synced_mcp_client.py
opens a FRESH SSE session (connect -> initialize -> list_tools/call_tool ->
close) per question. Stock supergateway reuses one MCP `Server` across SSE
connections, so the 2nd `GET /sse` throws "Already connected to a transport" and
the gateway dies — a 600-QA run would halt at Q1.

This driver opens session #1 (initialize + list_tools), CLOSES it, then opens a
SECOND independent session (initialize + list_tools + optional call_tool). If the
gateway re-arms correctly, BOTH sessions succeed. If it has the stock bug,
session #2 fails to connect / list_tools.

Exit 0 = both sessions OK (gateway re-arms). Non-zero = re-arm bug present.

Usage:
    python verify_two_sessions.py --url http://localhost:8011/sse \
        --tool instant_search --query "rust programming language" --call
"""
import argparse
import asyncio
import sys

from mcp import ClientSession
from mcp.client.sse import sse_client


async def one_session(url: str, tool: str, query: str, do_call: bool, n: int) -> int:
    print(f"[s{n}] connecting SSE -> {url}", flush=True)
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print(f"[s{n}] initialize OK -> {init.serverInfo.name} "
                  f"v{init.serverInfo.version}", flush=True)
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            print(f"[s{n}] list_tools OK -> {len(names)} tools", flush=True)
            if tool not in names:
                print(f"[s{n}] FAIL: tool {tool!r} not exposed", file=sys.stderr)
                return 2
            if do_call:
                res = await session.call_tool(tool, {"query": query})
                text = "".join(c.text for c in res.content
                               if getattr(c, "text", None))
                print(f"[s{n}] call_tool OK -> {len(text)} chars, "
                      f"isError={res.isError}", flush=True)
    print(f"[s{n}] session closed cleanly", flush=True)
    return 0


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8011/sse")
    ap.add_argument("--tool", default="instant_search")
    ap.add_argument("--query", default="rust programming language")
    ap.add_argument("--call", action="store_true",
                    help="also issue call_tool in each session (hits network)")
    ap.add_argument("--gap", type=float, default=1.0,
                    help="seconds between sessions (let session 1 fully close)")
    args = ap.parse_args()

    rc1 = await one_session(args.url, args.tool, args.query, args.call, 1)
    if rc1 != 0:
        print("[result] SESSION 1 FAILED", file=sys.stderr)
        return rc1

    await asyncio.sleep(args.gap)

    try:
        rc2 = await one_session(args.url, args.tool, args.query, args.call, 2)
    except Exception as e:  # noqa: BLE001 - we want to surface the re-arm crash
        print(f"[s2] EXCEPTION on second session: {e!r}", file=sys.stderr)
        print("[result] RE-ARM BUG: gateway did NOT survive a 2nd SSE session",
              file=sys.stderr)
        return 3
    if rc2 != 0:
        print("[result] SESSION 2 FAILED", file=sys.stderr)
        return rc2

    print("[result] BOTH SESSIONS OK — gateway re-arms across sequential SSE "
          "sessions", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
