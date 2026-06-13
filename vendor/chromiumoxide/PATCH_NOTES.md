# Vendored chromiumoxide 0.7.0 — R4 v2 patch notes

This is a **vendored, patched fork** of the published `chromiumoxide` 0.7.0
crate. It is wired into the workspace via `[patch.crates-io]` in the repo-root
`Cargo.toml`:

```toml
[patch.crates-io]
chromiumoxide = { path = "vendor/chromiumoxide" }
```

`crates/crawler/Cargo.toml` keeps its dependency as `chromiumoxide = "0.7"`
(optional, behind the `browser` feature); the patch transparently redirects that
to this local path. The published sub-crates `chromiumoxide_cdp` and
`chromiumoxide_types` (0.7) are **left as published deps** — only the top
`chromiumoxide` crate is forked.

## What was changed and why

### The single functional change: remove the auto-fired `Runtime.enable`

**File:** `src/handler/frame.rs`, fn `FrameManager::init_commands`.

Upstream queued a `CommandChain` of four commands at every target (page) init,
**before any user code runs**:

1. `Page.enable`
2. `Page.getFrameTree`
3. `Page.setLifecycleEventsEnabled(true)`
4. **`Runtime.enable`**  ← REMOVED

`Runtime.enable` turns on the Runtime event stream
(`Runtime.executionContextCreated`, `Runtime.consoleAPICalled`, ...). Because it
is fired eagerly at load time and is **not suppressible through chromiumoxide's
public API**, it is a reliable load-time fingerprint that Cloudflare, DataDome,
`bot.sannysoft.com`, and `rebrowser-bot-detector` use to flag a CDP/automation-
driven browser (the canonical `Runtime.enable -> consoleAPICalled` CDP leak).

The R4 v2 patch removes the 4th entry (and the `runtime::EnableParams::default()`
that built it, plus the now-unused `js_protocol::runtime` *module* import — the
`js_protocol::runtime::*` glob that supplies the runtime *event* types is kept).

### Commands KEPT

`Page.enable`, `Page.getFrameTree`, and `Page.setLifecycleEventsEnabled(true)`
are retained: they drive the frame tree and the page lifecycle events that
`Page::wait_for_navigation()` depends on, and they are **not** the load-time
signal detectors key on. `NetworkManager::init_commands` (`Network.enable`) and
`page_init_commands` (`Performance`/`Log` enable) are untouched.

## Does removing `Runtime.enable` break anything?

No, for this workspace's usage. The reasoning, verified against the 0.7.0
source:

- `Runtime.evaluate` is a **command**, not an event subscription. It works
  without `Runtime.enable`. When called with no `contextId` it runs in the
  page's **default (main) execution context**.
- `Page::evaluate` (page.rs ~1034-1059) calls `self.execution_context().await?`
  and, if it is `None`, sends `EvaluateParams { context_id: None, .. }`. With the
  runtime event stream off, `execution_context()` is **always** `None` (the
  `executionContextCreated` event that would populate it never arrives), so
  `evaluate()` transparently targets the default context.
- `Page::content()` (page.rs ~1200) is implemented on top of `evaluate()`, so it
  keeps working on both the legacy SPA path and the stealth path.
- `goto`, `wait_for_navigation`, `get_cookies`, `addScriptToEvaluateOnNewDocument`,
  `setUserAgentOverride` do not depend on the Runtime event stream.

### Known behavioral trade-off

- `Page::execution_context()` / `secondary_execution_context()` now always
  return `Ok(None)` (no `executionContextCreated`/`Destroyed` event tracking).
- The eager utility **isolated world** (`__chromiumoxide_utility_world__`) is
  still requested at init via `Page.createIsolatedWorld`, but its returned
  context id is never captured (again, no event), so secondary-world evaluate is
  not available. The crawler does not use it.

If a future consumer needs an explicit execution-context id, drive
`Page.createIsolatedWorld` directly and use the `executionContextId` from its
**command response** (which does not require `Runtime.enable`).

## How to re-apply on a chromiumoxide version bump

1. Download/extract the new `chromiumoxide` crate source into
   `vendor/chromiumoxide/` (overwrite). Delete cargo cache artifacts:
   `.cargo-ok`, `.cargo_vcs_info.json`, and the published `Cargo.lock`.
2. Open `src/handler/frame.rs`, find `fn init_commands`.
3. Remove the `Runtime.enable` entry from the `CommandChain` vec, remove the
   `let enable_runtime = runtime::EnableParams::default();` line, and re-add the
   `// R4 v2 patch:` comment block explaining the removal.
4. If `runtime::EnableParams` was the only use of the `js_protocol::runtime`
   *module* import, drop that import (keep the `runtime::*` glob if the event
   types are still referenced — `cargo check` will tell you).
5. Bump the version pin in `crates/crawler/Cargo.toml` if the major changed.
6. Run the offline guard test — it fails if the patch was not re-applied:
   `cargo test -p web-search-crawler --features browser vendored_chromiumoxide_omits_runtime_enable`
7. Re-run `cargo check -p web-search-crawler --features browser` and
   `cargo check --workspace`.

## Maintenance cost

This fork must be re-synced manually whenever chromiumoxide is upgraded. The
guard test `vendored_chromiumoxide_omits_runtime_enable` (in
`crates/crawler/src/browser.rs`) makes a forgotten re-apply a **loud CI
failure** rather than a silent regression of the leak.

## Scope of the fix (honest)

- **CLOSED:** the load-time auto-fired `Runtime.enable` CDP leak, for *every*
  chromiumoxide use in this workspace (stealth and legacy SPA paths).
- **UNPROVEN here:** live efficacy against real Cloudflare / DataDome. That
  requires running the `#[ignore]`d `stealth_smoke_sannysoft` test manually
  against a real Chrome + network. Sec-CH-UA client hints
  (`user_agent_metadata`) remain deferred.
