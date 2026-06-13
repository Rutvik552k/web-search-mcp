# Research 0001 — Bot-Protection Bypass Stack (verified)

> Owner: research-solution-architect. Verified against crates.io / GitHub / docs.rs, June 2026.
> Backs TASKS.md task 1.2. Per CLAUDE.md Rule 1 every recommendation carries a citation;
> unverified items are flagged for a pre-merge build spike.

## Headline

No single self-hosted OSS tool scrapes "ANY website incl. bot-protected." Two layers, two defenses:
- **Passive fingerprinting** (TLS JA3/JA4, HTTP/2 SETTINGS, header order, client hints) → TLS-impersonating HTTP client. Cheap, no browser. **Reliably defeated.**
- **Active JS challenges** (Cloudflare Managed/Turnstile, DataDome interactive) → real browser with hidden CDP surface. Slow, **not guaranteed** (DataDome = per-site ML + behavioral).

→ Two-tier escalation: HTTP-impersonation first, browser-stealth fallback. Maps onto existing `Fetcher → BrowserPool`. **DataDome/behavioral sites need a commercial tier-3** (out of OSS scope).

## Area 1 — TLS/HTTP2 impersonation → `wreq` 5.3.0 (+ wreq-util)

- `rquest` (formerly reqwest-impersonate) is **DEPRECATED** → repo renamed `rquest-deprecated`. Do not adopt.
- Successor: **`wreq`** by same author — hard fork of reqwest, BoringSSL, impersonates Chrome 100–147 / Firefox / Safari / Opera / OkHttp (TLS **and** HTTP/2). Apache-2.0.
- Stable **5.3.0** (docs.rs). 6.0.0-rc exists — don't pin rc.
- Citations: https://github.com/0x676e67/rquest-deprecated · https://github.com/0x676e67/wreq · https://docs.rs/wreq · https://github.com/0x676e67/wreq-util · https://github.com/rust-unofficial/awesome-rust/pull/1822

**Integration (fetcher.rs):** builder ports ~1:1. **Delete `.user_agent()` + `Self::default_headers()` (lines 304–311)** — `.emulation(Emulation::Chrome137)` injects browser-consistent UA, accept-*, header ORDER, sec-ch-ua. The hand-rolled static headers ARE the current fingerprint that gets blocked. Rest (timeout/redirect/cookie_store/gzip/brotli/pool/tcp) mirrors reqwest. Add config field for emulation profile.

**Risks (pre-merge checklist):**
1. `cargo tree -i openssl-sys` must return nothing (boring-sys vs openssl-sys symbol clash). On Windows reqwest defaults to SChannel so mixing wreq(boring)+reqwest(schannel) is OK if no openssl-sys.
2. Windows 11 BoringSSL build needs MSVC C++ tools + cmake + NASM/perl + libclang. Build a 1-file spike first.
3. wreq-util ↔ wreq 5.3.0 version pairing UNVERIFIED (wreq-util latest seen = 3.0.0-rc.12). Pin after `cargo build` confirms.
4. Confirm exact wreq feature names (cookies/gzip/brotli/socks) on docs.rs before pinning.
5. Smoke-test migrated client vs https://tls.peet.ws/api/all + a known Cloudflare site → fingerprint reads as Chrome.

## Area 2 — Stealth headless (browser.rs JS-challenge fallback)

Vanilla chromiumoxide trivially detected. Decisive 2026 vector = **CDP `Runtime.enable` leak** (automation auto-fires it; page JS detects). Setting `navigator.webdriver=false` alone NOT sufficient.

**Option A (recommended): hand-patch browser.rs.** Keep chromiumoxide 0.7.0 (MIT/Apache, maintained). Inside `fetch_with_browser`:
1. `Page.addScriptToEvaluateOnNewDocument` (runs before page JS): mask `navigator.webdriver`, inject plausible `window.chrome`, patch platform/hardwareConcurrency.
2. Evaluate via `Page.createIsolatedWorld` not default `Runtime.evaluate` → avoids `Runtime.enable` leak.
3. `--headless=new`, real UA/locale/window; drop `--blink-settings=imagesEnabled=false` (fingerprintable) on stealth path. Replace fixed 1500ms sleep (metronomic tell) with network-idle/selector wait.
- Citations: https://rebrowser.net/blog/how-to-fix-runtime-enable-cdp-detection-of-puppeteer-playwright-and-other-automation-libraries · https://rebrowser.net/docs/sensitive-cdp-methods · https://github.com/rebrowser/rebrowser-patches · https://docs.rs/crate/chromiumoxide/latest

**Option B (optional): `chaser-oxide` 0.2.0** — chromiumoxide fork w/ CDP patching already done. Risk: experimental, single-maintainer, no published CF/DataDome proof. Pin exactly if used.
Avoid `chromiumoxide_stealth` (JS-injection only, doesn't fix Runtime.enable leak).

**Hard limit:** DataDome ≈ per-site ML + behavioral. Stealth passing site A can fail site B. "Full bypass of ANY site" not deliverable OSS-only. Citations: https://spider.cloud/blog/bypass-cloudflare-datadome-perimeterx-2026 · https://scrapfly.io/blog/posts/how-to-bypass-cloudflare-anti-scraping

## Area 3 — Proxy rotation with wreq

- Verified: wreq supports HTTP/HTTPS/SOCKS4/5(h) + auth, **client-level only** (`ClientBuilder::proxy`). **No per-request proxy** (same as reqwest).
- Pattern: replace single `client` with **pool of pre-built clients** keyed (proxy, emulation_profile). Building a wreq client is heavy (BoringSSL ctx) — pool, don't rebuild per request.
- **Sticky sessions:** route by hash(target host) → same client/proxy so cookies + TLS session stay coherent. `dashmap` (already workspace dep) `host -> client_index`.
- **Rotation:** on 403/429/challenge in fetch retry loop, advance to next client before next attempt (not just backoff).
- **Health:** per-client AtomicU64 success/fail; eject after K consecutive fails; probe before reinstating; never drop to zero live.
- Secrets: proxy creds from env/secret store, never in source/logs.
- Citations: https://docs.rs/wreq · https://roundproxies.com/blog/wreq-util/ · https://roundproxies.com/blog/how-to-setup-proxies-in-rust/

## Cargo deps

```toml
# workspace [workspace.dependencies]
reqwest = { version = "0.12", features = ["json","cookies","gzip","brotli"] }  # keep for non-crawler crates
wreq = "=5.3.0"        # VERIFIED resolves (spike 2.0)
wreq-util = "=2.2.6"   # VERIFIED pairing for wreq 5.3 (NOT "3"/rc.12 — that needs wreq 6.0-rc). boring stack: boring2 4.15.15
# chaser-oxide = "=0.2.0"  # optional Area-2 Option B only
```
Add wreq + wreq-util to crates/crawler/Cargo.toml; keep chromiumoxide for `browser` feature. Leave reqwest on orchestrator/mcp-server unless openssl-sys conflict.

## Contingency
If wreq maintenance lapses (single author): `ratcurl`/`hyprcurl` (curl-impersonate FFI). Note only, not current rec.
