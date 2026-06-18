# G3 accuracy via MCPBench

Measures **G3 (answer accuracy)** for our web-search MCP server using
[MCPBench](https://github.com/modelscope/MCPBench) (modelscope), `WebSearch`
benchmark. MCPBench drives an LLM agent that is given OUR tools, lets it answer
600 hard multi-hop questions, and grades each answer with an LLM judge. The
score is the fraction of answers judged correct — that fraction **is G3**.

> Verified against MCPBench `main` @ commit pushed 2025-09-03 (repo
> `modelscope/MCPBench`), inspected via `gh api` on 2026-06-18. Source files
> cited inline below.

---

## How MCPBench reaches our server (verified)

MCPBench does **not** speak stdio to the MCP server directly. Its client
(`langProBe/synced_mcp_client.py` → `async_mcp_client.connect_to_sse_server`)
connects over **SSE** at `http://localhost:<port>/sse`. A local stdio MCP server
is wrapped into SSE by **`npx supergateway`**, launched by
`launch_mcps_as_sse.sh`:

```sh
npx -y supergateway --stdio "$ARGS $COMMAND" \
  --port "$PORT" --baseUrl "http://localhost:$PORT" \
  --ssePath /sse --messagePath /message ...
```

`$COMMAND`, `$ARGS`, `$PORT` come from each `mcp_pool[].run_config[0]` entry in
the JSON config. So the data flow is:

```
MCPBench (langProBe/evaluation.py)
  └─ agent LLM picks a tool by name  (program_utils.build_system_content -> list_tools)
       └─ SyncedMcpClient  --SSE-->  http://localhost:8005/sse
            └─ supergateway  --stdio-->  ./target/release/web-search-mcp   (OUR server)
```

Tool discovery and calls are standard MCP: `list_tools()` then
`call_tool(name, args)` (`program_utils.mcp_calling`). Our tool names + JSON
schemas are consumed verbatim into the agent's system prompt — **no tool-shape
adapter is needed** (see "Tool mapping" below).

---

## Prerequisites

- **Our server, release build** (faster than debug):
  `cargo build -p mcp-server --release` → `target/release/web-search-mcp`
- **Node** (for `npx supergateway`) — `node -v` ≥ 18.
- **Python 3.10+** and MCPBench's deps: `pip install -r requirements.txt`
  inside the cloned MCPBench (`dspy>=2.6`, `mcp`, `openai`, `dashscope`, ...).
- A **judge/agent LLM key** — see next section. This is supplied by the
  operator at run time; it is **not** a dependency of our server.

### Recommended: run a SearXNG instance for a fair G3

Our keyless floor still answers, but for representative accuracy give the server
a real search source. Either run SearXNG locally (default
`searxng_url = http://localhost:8080` in `config/default.toml`):

```sh
docker run -d -p 8080:8080 searxng/searxng
# enable JSON: settings.yml -> search.formats = ["json"]
```

…or point the server at a keyed source via its own config (`--config` /
`WEB_SEARCH_MCP_CONFIG`, ADR 0004). The judge key below is separate from any
search-source key.

---

## The LLM the operator MUST supply (judge + agent)

MCPBench uses **one OpenAI-compatible chat endpoint for BOTH** the answering
agent and the answer judge:

- **Agent LLM** — `--lm`, `--lm_api_base`, `--lm_api_key`
  (`langProBe/evaluation.py`, threaded into `ProcessManager` and called by
  `program_utils.call_lm` via the `openai` SDK; model string must be
  `openai/<model>`).
- **Judge LLM** — `langProBe/mcp_program.py :: evaluate_prediction` **hardcodes**
  `manager.model = "openai/deepseek-v3"` and reuses the SAME `lm_api_key` /
  `lm_api_base`. The judge prompt is in
  `evaluation_utils.evaluate_final_answer` ("回答对关键信息就算正确… 只需要返回
  True或False"); a `True` verdict ⇒ that QA pair scores 1.

The stock `evaluation_websearch.sh` targets Alibaba DashScope's
OpenAI-compatible gateway:

```
--lm=openai/deepseek-v3
--lm_api_base=https://dashscope.aliyuncs.com/compatible-mode/v1
--lm_api_key=xxx          <-- operator replaces with a real key
```

**Operator action:** set a real key. Because the judge model is pinned to
`deepseek-v3`, the supplied endpoint must serve a `deepseek-v3` model
(DashScope does). Any OpenAI-compatible base that serves both your chosen agent
model and `deepseek-v3` works.

> COST / SAFETY: ~600 QA × multi-step agent loops + 600 judge calls. This is a
> paid LLM run owned by the operator. No key is committed here. Do a small
> `dataset_mode=test` run first (2 items) to confirm spend and wiring.

---

## Tool mapping (ours → what MCPBench needs)

MCPBench needs no specific tool names — the agent LLM reads whatever
`list_tools()` returns and chooses. Verified our server exposes **15 tools**;
the ones relevant to WebSearch QA:

| Our tool          | Use in MCPBench WebSearch | Input schema (required) |
|-------------------|---------------------------|-------------------------|
| `instant_search`  | fast factual lookup       | `query` (str)           |
| `quick_search`    | single-wave search        | `query` (str)           |
| `deep_research`   | multi-hop hard questions  | `query` (str)           |
| `streaming_search`| progressive search        | `query` (str)           |
| `extract`         | read a specific URL       | `url` (str)             |
| `fetch_page`      | raw page fetch            | `url` (str)             |

The WebSearch questions are multi-hop (e.g. constraint-chaining over Wikipedia),
so `deep_research` / `quick_search` are the natural picks; the agent decides.

---

## Setup + run

```sh
# 1. clone MCPBench
git clone https://github.com/modelscope/MCPBench.git
cd MCPBench
pip install -r requirements.txt

# 2. drop our config into MCPBench/configs/ and edit the absolute path + port
cp <this repo>/benchmark/mcpbench/mcp_config_websearch_ours.json \
   configs/mcp_config_websearch_ours.json
#   -> set run_config[0].command to the ABSOLUTE path of target/release/web-search-mcp

# 3. start our server as SSE (one of the two below)
#    (a) MCPBench's launcher reads the config and starts supergateway for you:
./launch_mcps_as_sse.sh mcp_config_websearch_ours.json
#    (b) or start it yourself (what we verified):
npx -y supergateway --stdio "/abs/path/to/target/release/web-search-mcp" \
    --port 8005 --baseUrl http://localhost:8005 \
    --ssePath /sse --messagePath /message --logLevel info

# 4. wait until our server logs "ML models warmed up" / "WebSearchServer initialized"
#    (first boot fetches the ~90MB MiniLM embedder once).

# 5. run the eval (set a REAL judge key). Smoke first:
DSPY_CACHEDIR=evaluation_mcp/.dspy_cache python -c "
import multiprocessing as mp; mp.set_start_method('spawn', True)
from langProBe.evaluation import main; main()" \
  --benchmark=WebSearch --dataset_mode=test \
  --dataset_path=langProBe/WebSearch/data/websearch_600.jsonl \
  --file_path=evaluation_websearch_ours \
  --lm=openai/deepseek-v3 \
  --lm_api_base=https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --lm_api_key=$WSM_JUDGE_KEY \
  --num_threads=1 \
  --config=configs/mcp_config_websearch_ours.json

# 6. full 600-QA run: same command with --dataset_mode=full
```

> `evaluation_websearch.sh` points `--dataset_path` at `websearch_test.jsonl`
> (not in the repo). Use the shipped **`websearch_600.jsonl`** (600 pairs) or
> `websearch_300.jsonl`. `dataset_mode`: `full` = all, `test` = 2 (smoke),
> `tiny` = 200, `lite` = 500 (`langProBe/benchmark.py :: dataset_size`).

### Dataset format (verified, `websearch_600.jsonl`)

JSONL; the loader (`benchmark.py :: MCPBench.init_dataset`) reads exactly:

```json
{"unique_id": 0, "Prompt": "...question...", "Answer": "Jane Ballou",
 "reasoning_types": "...", "wiki_links": "[...]"}
```

→ mapped to `dspy.Example(id=unique_id, question=Prompt, answer=Answer)`.
(Note: capital `Prompt`/`Answer`, not `prompt`/`answer`.) `reasoning_types` and
`wiki_links` are ignored by the scorer.

### Where results land + how to read G3

Under `--file_path` (e.g. `evaluation_websearch_ours/`):
- `MCP_WEBSEARCH_MCPPredict.txt` — `score,cost,input_tokens,output_tokens`
- `evaluation_results.csv`, `<name>.stat`, `evaluation_records.csv`
- per-question transcripts under `logs/websearch_messages_*.jsonl`

**`score` is G3** — fraction of QA pairs the judge marked correct (the metric is
`mcp_metric = pred.success`, averaged by dspy `Evaluate`). dspy may print it as a
percentage; G3 as a 0..1 fraction = `score/100` if shown as a percent.

---

## Feeding G3 into G4 (firecrawl-compare)

`crates/benchmark/src/bin/firecrawl_compare.rs` accepts the MCPBench accuracy
numbers as optional flags and folds them into the G4 head-to-head report:

```sh
cargo run -p web-search-benchmark --bin firecrawl-compare -- \
  --urls benchmark/urls.jsonl \
  --server target/release/web-search-mcp \
  --accuracy-ours 0.NN \
  --accuracy-firecrawl 0.MM \
  --out benchmark/RESULTS.firecrawl.md
```

- `--accuracy-ours` = G3 from THIS MCPBench run against our config.
- `--accuracy-firecrawl` = G3 from an identical MCPBench run using the Firecrawl
  config (`configs/mcp_config_template.json` "Local run example", i.e.
  `npx -y firecrawl-mcp` with `FIRECRAWL_API_KEY`), same dataset + same judge.

Both must be 0..1. The report only includes the accuracy block when **both** are
supplied (`firecrawl_compare.rs` builds `AccuracyPair` only if both are `Some`).

---

## Verification evidence (run 2026-06-18, this repo)

Done with the **debug** binary on Windows; release works identically.

- supergateway launched our stdio server and exposed SSE on `:8007`.
- `verify_handshake.py` (mirrors MCPBench's SSE client) over
  `http://localhost:8007/sse`:
  - `initialize` OK (server `rmcp v0.16.0`)
  - `list_tools` → **15 tools**, schemas intact
  - `call_tool(instant_search, {"query":"rust programming language"})` →
    `isError=False`, 3695-char structured JSON result.

Reproduce:

```sh
npx -y supergateway --stdio "<abs path>/web-search-mcp(.exe)" \
  --port 8007 --baseUrl http://localhost:8007 --ssePath /sse --messagePath /message
python benchmark/mcpbench/verify_handshake.py \
  --url http://localhost:8007/sse --tool instant_search \
  --query "rust programming language" --call
```

What is **operator-run** (needs the judge key / paid LLM): the actual
600-QA accuracy run in step 5–6 above. Everything up to and including a real
`call_tool` is verified here.

---

## PREP STATE (2026-06-18) — harness made keyed-run-ready

Everything below was set up + verified **without a key** so the operator's
single keyed run works first try. External MCPBench clone lives at WSL path
**`~/MCPBench`** (`\\wsl$\Ubuntu\home\rutvik\MCPBench`), HEAD
`5f397445370e6cb44dfdfc5680a48f128a75d349` (2025-09-03). WSL venv:
**`~/mcpbench-venv`** (Ubuntu, Python 3.12).

### 1. supergateway re-arm bug — FIXED with a custom gateway (not a version pin)

The bug is **structural in supergateway through 3.4.3 (latest)**, not fixed in
any published version. Root cause (read in `dist/gateways/stdioToSse.js`): ONE
`@modelcontextprotocol/sdk` `Server` is created at startup and `server.connect()`
is called on EVERY `GET /sse`. The SDK's `Protocol.connect()` throws
`Already connected to a transport` on the 2nd connect. Reproduced under WSL:
session 1 OK → session 2 = `RemoteProtocolError: Server disconnected` +
`Error: Already connected to a transport ... protocol.js:217` in the gateway log.

**Fix:** `rearm_gateway.mjs` (this dir) — a supergateway-CLI-compatible drop-in
that spawns a FRESH `Server` + FRESH child stdio process **per SSE connection**
and tears both down on close. No `Server`/transport is ever reused. Deps
(`@modelcontextprotocol/sdk@1.29.0`, `express@4`, `body-parser@1`) installed in
`benchmark/mcpbench/node_modules` (run gateway from this dir so node resolves
them). Launch with `launch_rearm_gateway.sh`.

**2-session proof** (`verify_two_sessions.py`, mirrors MCPBench's fresh-session-
per-call pattern):
- stock supergateway@3.4.3 → session 2 FAILS (exit 3, "Already connected").
- `rearm_gateway.mjs` → BOTH sessions do initialize + list_tools(15) +
  call_tool(instant_search, isError=False) and close cleanly (exit 0).
- Confirmed under the REAL MCPBench loop (dataset_mode=test, fake key): the
  gateway opened **2 sequential sessions** (distinct child pids), one per QA pair.

### 2. Judge model patched to OpenAI

`api.openai.com` serves `gpt-4o-mini`, NOT `deepseek-v3`. Both the agent
(`--lm`) and judge use `program_utils.call_lm` →
`OpenAI(base_url=lm_api_base, api_key=lm_api_key)` then `model.split('/')`
(prefix must be `openai`). The judge model is hardcoded separately.

**Patched files/lines (in the WSL clone `~/MCPBench`):**
- `langProBe/mcp_program.py:120` — `answer_eval_manager.model = "openai/deepseek-v3"`
  → `"openai/gpt-4o-mini"`. Original saved as `langProBe/mcp_program.py.orig`.
- Agent model + base + key are passed as CLI flags (no edit): `--lm=openai/gpt-4o-mini`,
  `--lm_api_base=https://api.openai.com/v1`, `--lm_api_key=$OPENAI_API_KEY`.

That single line is the only judge edit needed — one OpenAI key + base serves
both agent and judge with `gpt-4o-mini`.

### 3. dspy MUST be pinned `<3` (would otherwise break the keyed run)

MCPBench `requirements.txt` says `dspy>=2.6` but was authored for 2.6.x. dspy
**3.x removed `Evaluate(return_outputs=...)`** → `ValueError: return_outputs is
no longer supported` at `benchmark.py:209`, killing the run before any question.
Also `pandas` is imported (`analysis.py`) but missing from requirements.txt.
**Both fixed in the venv:** `dspy==2.6.27` pinned, `pandas` installed. Verified
the full eval loop runs to completion under dspy 2.6.27 (only the fake-key 401
remains, as expected).

### 4. Linux/WSL invocation (Git Bash mangles paths — do NOT use it)

Git Bash rewrites `--ssePath /sse` → `C:/Program Files/Git/sse` and cannot spawn
the `.exe` cleanly (`shell:true` → "system cannot find the path"). Run under
**WSL/Linux**. WSL interop launches the Windows `.exe` via `/mnt/c/...` and it
responds to MCP correctly (verified). Config
`configs/mcp_config_websearch_ours.json` uses the WSL path
`/mnt/c/Users/rutvi/Downloads/projects/web_search_mcp/target/release/web-search-mcp.exe`
and `port: 8005`; MCPBench builds `http://localhost:8005/sse` from it
(`program_utils.py:166`) and does NOT auto-spawn the gateway — you launch it.

### EXACT KEYED RUN (operator runs this; key from env, never written anywhere)

**Terminal A (WSL) — start the re-arm gateway, leave it running:**
```bash
cd /mnt/c/Users/rutvi/Downloads/projects/web_search_mcp/benchmark/mcpbench
bash launch_rearm_gateway.sh        # listens on :8005 /sse; re-arms per session
```

**Terminal B (WSL) — SMOKE first (2 pairs), then FULL (600). `OPENAI_API_KEY`
comes from your env; it is NOT in any file:**
```bash
cd ~/MCPBench
# --- SMOKE: dataset_mode=test = 2 QA pairs (confirm spend + wiring) ---
DSPY_CACHEDIR=evaluation_mcp/.dspy_cache ~/mcpbench-venv/bin/python -c "
import multiprocessing as mp; mp.set_start_method('spawn', True)
from langProBe.evaluation import main; main()" \
  --benchmark=WebSearch --dataset_mode=test \
  --dataset_path=langProBe/WebSearch/data/websearch_600.jsonl \
  --file_path=evaluation_websearch_ours \
  --lm=openai/gpt-4o-mini \
  --lm_api_base=https://api.openai.com/v1 \
  --lm_api_key="$OPENAI_API_KEY" \
  --num_threads=1 \
  --config=configs/mcp_config_websearch_ours.json

# --- FULL: dataset_mode=full = all 600 QA pairs (G3) ---
#     same command, change ONLY: --dataset_mode=full
```
> `dataset_mode=test` = 2 pairs (`benchmark.py:19` dataset_size, the in-place
> re-trim to 50 is a no-op on a 2-item set). `full` = all 600.
> `score` in `evaluation_websearch_ours/MCP_WEBSEARCH_MCPPredict.txt` is **G3**.
> Then fold G3 into G4: see "Feeding G3 into G4" above
> (`--accuracy-ours` / `--accuracy-firecrawl`).

> COST/SAFETY: ~600 agent loops + 600 judge calls on gpt-4o-mini = a paid
> OpenAI run YOU own. Run the SMOKE first and check the bill before `full`.

### Other notes
- **serverInfo name is `rmcp`.** Cosmetic — MCPBench keys the server by the JSON
  config `name`, not MCP serverInfo.
- A Firecrawl-side G3 (for the G4 accuracy delta) needs an identical MCPBench run
  pointed at a Firecrawl MCP config + the same judge — out of scope for this prep.
