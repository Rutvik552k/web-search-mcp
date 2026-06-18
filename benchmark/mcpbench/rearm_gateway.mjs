#!/usr/bin/env node
/**
 * rearm_gateway.mjs — supergateway-compatible stdio→SSE bridge that SURVIVES
 * multiple sequential SSE sessions.
 *
 * WHY THIS EXISTS
 * ---------------
 * Stock `supergateway` (all versions through 3.4.3, latest as of 2026-06)
 * creates ONE `@modelcontextprotocol/sdk` `Server` instance at startup and
 * calls `server.connect(newTransport)` on EVERY `GET /sse`
 * (dist/gateways/stdioToSse.js). The MCP SDK `Protocol.connect()` refuses to
 * bind a second transport to an already-connected server and throws:
 *
 *     Already connected to a transport. Call close() before connecting to a
 *     new transport.
 *
 * So the FIRST SSE session works and the SECOND one crashes the gateway. This
 * is the same defect class as MCP-SuperAssistant issues #183 / #194. MCPBench
 * (langProBe/synced_mcp_client.py) opens a FRESH SSE session per
 * list_tools/call_tool, so on stock supergateway a 600-QA run dies at Q1.
 *
 * This wrapper instead spins up a FRESH `Server` + a FRESH child stdio process
 * PER SSE connection, and tears both down on close. No `Server`/transport is
 * ever reused, so sequential sessions are fully isolated and the gateway never
 * enters the "already connected" state. CLI surface mirrors supergateway so
 * MCPBench's launcher / our docs need no other change.
 *
 * USAGE (drop-in for `npx -y supergateway --stdio ...`):
 *   node rearm_gateway.mjs --stdio "<command>" \
 *       --port 8005 --baseUrl http://localhost:8005 \
 *       --ssePath /sse --messagePath /message [--logLevel info]
 *
 * Requires the MCP SDK + express (installed via `npm i` in this dir, or resolved
 * from a sibling supergateway install). Node >= 18.
 */
import express from 'express';
import bodyParser from 'body-parser';
import { spawn } from 'node:child_process';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js';

function parseArgs(argv) {
  const a = {
    stdio: null,
    port: 8005,
    baseUrl: 'http://localhost:8005',
    ssePath: '/sse',
    messagePath: '/message',
    logLevel: 'info',
  };
  for (let i = 0; i < argv.length; i++) {
    const f = argv[i];
    const next = () => argv[++i];
    switch (f) {
      case '--stdio': a.stdio = next(); break;
      case '--port': a.port = parseInt(next(), 10); break;
      case '--baseUrl': a.baseUrl = next(); break;
      case '--ssePath': a.ssePath = next(); break;
      case '--messagePath': a.messagePath = next(); break;
      case '--logLevel': a.logLevel = next(); break;
      case '--header': next(); break; // accepted + ignored (parity)
      case '--cors': break;
      default: break; // tolerate unknown flags for supergateway parity
    }
  }
  if (!a.stdio) {
    console.error('[rearm] FATAL: --stdio "<command>" is required');
    process.exit(2);
  }
  return a;
}

const args = parseArgs(process.argv.slice(2));
const log = (...m) => { if (args.logLevel !== 'none') console.error('[rearm]', ...m); };

const app = express();
// body-parse everything EXCEPT the message path (SSE transport reads that raw)
app.use((req, res, nextFn) => {
  if (req.path === args.messagePath) return nextFn();
  return bodyParser.json()(req, res, nextFn);
});

app.get('/healthz', (_req, res) => res.send('ok'));

// sessionId -> { transport, server, child }
const sessions = {};

app.get(args.ssePath, async (req, res) => {
  log(`New SSE connection from ${req.ip}`);

  // FRESH child stdio process for THIS session (full isolation).
  const child = spawn(args.stdio, { shell: true });

  // FRESH Server instance for THIS session — never reused, so connect() is
  // only ever called once per Server (the bug in stock supergateway).
  const server = new Server({ name: 'rearm-gateway', version: '1.0.0' }, { capabilities: {} });
  const transport = new SSEServerTransport(`${args.baseUrl}${args.messagePath}`, res);
  await server.connect(transport);
  const sessionId = transport.sessionId;
  sessions[sessionId] = { transport, server, child };
  log(`session ${sessionId} opened (child pid ${child.pid})`);

  // SSE (client) -> child stdin
  transport.onmessage = (msg) => {
    child.stdin.write(JSON.stringify(msg) + '\n');
  };

  // child stdout -> SSE (only this session's transport)
  let buffer = '';
  child.stdout.on('data', (chunk) => {
    buffer += chunk.toString('utf8');
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() ?? '';
    for (const line of lines) {
      if (!line.trim()) continue;
      try { transport.send(JSON.parse(line)); }
      catch { log(`child non-JSON: ${line.slice(0, 200)}`); }
    }
  });
  child.stderr.on('data', (c) => log(`child[${sessionId}] stderr: ${c.toString('utf8').trim()}`));
  child.on('exit', (code, sig) => log(`child[${sessionId}] exited code=${code} sig=${sig}`));

  const cleanup = () => {
    if (!sessions[sessionId]) return;
    log(`session ${sessionId} closing — killing child pid ${child.pid}`);
    try { child.kill(); } catch {}
    delete sessions[sessionId];
  };
  transport.onclose = cleanup;
  transport.onerror = (e) => { log(`session ${sessionId} transport error: ${e}`); cleanup(); };
  req.on('close', cleanup);
});

app.post(args.messagePath, async (req, res) => {
  const sessionId = req.query.sessionId;
  if (!sessionId) return res.status(400).send('Missing sessionId parameter');
  const s = sessions[sessionId];
  if (s?.transport?.handlePostMessage) {
    await s.transport.handlePostMessage(req, res);
  } else {
    res.status(503).send(`No active SSE connection for session ${sessionId}`);
  }
});

app.listen(args.port, () => {
  log(`Listening on port ${args.port}`);
  log(`SSE endpoint: ${args.baseUrl}${args.ssePath}`);
  log(`POST messages: ${args.baseUrl}${args.messagePath}`);
});

for (const sig of ['SIGINT', 'SIGTERM']) {
  process.on(sig, () => {
    log(`${sig} — tearing down ${Object.keys(sessions).length} session(s)`);
    for (const id of Object.keys(sessions)) {
      try { sessions[id].child.kill(); } catch {}
    }
    process.exit(0);
  });
}
