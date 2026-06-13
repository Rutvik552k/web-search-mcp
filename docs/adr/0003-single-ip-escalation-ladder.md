# ADR 0003 — Single-IP Escalation Ladder (replaces proxy rotation)

- **Status:** Proposed
- **Date:** 2026-06-11
- **Owner:** solution-architect
- **Amends:** ADR 0001 (`docs/adr/0001-bypass-escalation-ladder.md`) — **deletes Rung 2 (proxy rotation)** and the proxy-health-eject breaker; re-orders the ladder.
- **Builds on:** ADR 0002 (`docs/adr/0002-data-layer-acquisition.md`) — promotes the data-layer surfaces from a "Rung -1" short-circuit into first-class **early, IP-safe rungs**.
- **Reuses unchanged from ADR 0001:** the `BlockClass` classifier (§4) and its priority order, the typed-error give-up contract, and the cache-insert-on-`RealContent`-only rule (C6).
- **Implemented by (later phases):** Phase 2 (revised — proxy tasks 2.3 dropped), Phase 3.

> **Scope guard (GOAL.md §2):** unchanged from ADR 0001/0002 — bypass surfaces are opt-in via
> config, default-off where legally sensitive. This ADR sets contracts only; it writes no feature
> code. The single new *hard default* is the **permanent per-domain hard-stop** on a confirmed block
> / C&D, which GOAL.md §2 already mandates ("Block or C&D = permanent per-domain hard stop. Never
> rotate IPs back in.") — this ADR makes that mandate the load-bearing design constraint.

---

## 1. Context and forces in tension

### 1.1 The new hard constraint

**Operator decision (2026-06-11):** the scraper runs from **ONE static IP**. No proxy rotation, no
IP rotation, ever. This **deletes Rung 2 (proxy rotate)** from ADR 0001's ladder and removes the
proxy pool (ADR 0001 §5 `proxy_*` fields, Task 2.3) from the design.

This is not merely a cost cut — it **inverts the strategy**. ADR 0001's ladder is a
**recover-after-block** design: when an IP gets blocked, Rung 2 rotates to a fresh IP and continues.
Under a single static IP that escape hatch is gone. A confirmed hard ban on our one IP is
**permanent and unrecoverable** for that domain (and, for shared-WAF vendors keyed on IP reputation,
potentially degrades us across *other* domains behind the same vendor). The design must therefore
shift to **never-get-blocked**: prefer surfaces that never touch the protected origin, and when we
must touch it, spend reputation as a scarce, non-renewable budget.

This also aligns the engineering design with the GOAL.md §2 **legal** guardrail (rotating IPs to
evade a known block is the strongest unauthorized-access fact pattern — *Power Ventures*, *3taps*).
Single-IP is simultaneously the safest *legal* posture and now the *only* operational posture.

### 1.2 Forces in tension (revised)

| Force | Pressure under single-IP |
|---|---|
| **Coverage (G1 ≥ 90%, G2)** | Must still defeat Cloudflare/DataDome JS challenges to reach G2 — but now without the proxy escape hatch. The burden shifts onto (a) IP-safe alternative surfaces (ADR 0002) and (b) *coherent* impersonation that never trips the WAF in the first place. |
| **IP reputation (NEW, dominant)** | One IP. A hard ban is permanent. Reputation is a non-renewable budget spent on every live-origin request. This force now **outranks latency and (non-paid) cost** in rung ordering. |
| **Cost** | Proxy $/GB is gone. Remaining paid surface is the optional tier-3 commercial unblock API (R5). Headless = CPU/RAM/seconds. Internet Archive CDX is free. |
| **Latency** | Unchanged shape: HTTP ~100–800ms; archive CDX ~200–900ms; headless 2–15s; tier-3 API seconds. But latency now *yields* to reputation — we will accept a slower IP-safe surface over a faster live-origin hit. |
| **Cache integrity** | Unchanged (ADR 0001): never cache a challenge/block page. |
| **SPA path** | Unchanged (ADR 0001 §3): the existing `is_spa` → browser fallback (fetcher.rs:135) must keep working and must not double-render. |
| **Resilience** | Every live-origin rung needs timeout + bounded attempts + a per-domain breaker — but the breaker semantics **change**: a hard ban trips a *permanent* hard-stop, not a cooldown (§4). |

### 1.3 Architecturally significant requirements (ASRs)

Carries ADR 0001's ASR-1..ASR-5 (coverage, cost cap, latency budget, cache integrity,
observability) **plus**:

- **ASR-6 Reputation preservation (NEW, top priority):** the design must minimize the probability of
  a permanent IP ban. Concretely: (a) prefer rungs that never hit the live protected origin; (b) on
  the live origin, pace adaptively per-domain and back off hard on soft signals *before* a hard ban
  occurs; (c) on a confirmed hard ban / explicit block / C&D, **stop permanently** for that domain
  and never retry it (GOAL.md §2).
- **ASR-7 Coherent impersonation (NEW):** when we do hit the live origin, the client must be
  *coherent* across TLS + HTTP/2 + headers + UA (one synthetic layer flags the whole session — see
  §6 ground truth #1). A static fingerprint string is insufficient; protocol-level
  extension/SETTINGS control (wreq, verified in research/0001) is required.

> **ASRs still owed by the operator (unchanged from ADR 0001 §1, GOAL.md §5):** G2 numeric target;
> tier-3 commercial unblock provider choice + per-session $ cap. These do **not** block this ADR but
> block the paid R5 rung.

---

## 2. Options considered

### Option A — Keep ADR 0001's ladder, just delete Rung 2, leave ordering as-is
Strip proxy rotation; otherwise climb HTTP → UA-rotate → headless → CAPTCHA on the live origin as
before.

- **Pros:** minimal change to ADR 0001.
- **Cons:** **fails ASR-6.** Every rung still hammers the live protected origin, and with the proxy
  escape hatch gone, repeated live-origin attempts on a hostile domain now *race toward a permanent
  ban with no recovery*. UA-rotation-on-one-IP is exactly the "behavioral signal orthogonal to
  fingerprint" that high-volume 403 patterns trip (§6 #3) — it spends reputation for little gain.
  Leaves the cheap, zero-ban-risk surfaces (archive, feeds) as a late afterthought.

### Option B — Reputation-first re-ordering: IP-safe surfaces early, live-origin gated by a reputation budget (CHOSEN)
Re-order the ladder so the **zero-ban-risk** surfaces (in-band salvage, archive/CDX) sit *early* and
the **live-origin** rungs (coherent impersonation, headless) are *gated behind a per-domain
reputation budget* and a permanent hard-stop breaker. End at the optional tier-3 API.

- **Pros:** directly serves ASR-6 — we exhaust ban-free options before spending reputation, and we
  cap reputation spend per domain. Folds ADR 0002's data-layer surfaces (already the
  highest-coverage, most-IP-safe tactic per §6 #2) into their natural early position. The classifier
  (ADR 0001 §4) is unchanged — it still *routes*, only the routing targets and ordering change.
- **Cons:** the archive surface (CDX) is gappy at per-URL granularity (§6 #2c) so it cannot be the
  *only* answer; live-origin rungs are still required for fresh/unarchived content, so we still spend
  *some* reputation. More moving parts than Option A. Mitigated: each live-origin rung is bounded by
  the reputation governor (§4), and the residual unreachable set is honestly bounded (§8).

### Option C — Pure data-layer + archive only; never touch the live protected origin
Refuse all live-origin requests on protected domains; serve only what archive/feeds/in-band yield.

- **Pros:** *zero* ban risk — ASR-6 trivially satisfied.
- **Cons:** **fails G1/G2.** CDX coverage is gappy per-URL and stale; feeds are summary-only; in-band
  needs a body we can only get by fetching. Large classes of fresh, unarchived, CSR pages become
  permanently unreachable. Throws away coverage we *can* safely get with paced coherent impersonation
  on well-behaved domains.

**Decision: Option B.** It is the only option that honors the single-IP/reputation constraint
(ASR-6) *without* surrendering the coverage goal (G1/G2). It reuses ADR 0001's classifier verbatim
and ADR 0002's surfaces, re-composing them under a reputation-first ordering and a permanent-stop
breaker.

---

## 3. The revised escalation ladder

Rungs are ordered by **(ban-risk, then cost, then latency)** — reputation-first per ASR-6, a
deliberate change from ADR 0001's pure cost ordering. The classifier verdict (ADR 0001 §4,
**unchanged**) still **routes** to a rung; "real content" at any rung short-circuits and returns.

**Rung renumbering note (read carefully):** this ADR re-numbers the ladder. The old ADR 0001 Rung 2
(proxy) is **deleted**. ADR 0001's in-band/data-layer work (ADR 0002 "Rung -1") becomes the new
**R0**. ADR 0001's old Rung 1 (UA/header rotate) is **absorbed into R1** as profile selection within
the coherent wreq client (per research/0001: rotate the wreq *emulation profile*, which carries a
coherent TLS+H2+header+UA set together — never rotate a header set independently of the fingerprint,
that is the incoherence ASR-7 forbids). Old Rungs 3/4/5 become R4/R4-stealth/R5.

```
                         ┌─────────────────────────────────────────────┐
   fetch(url) ──► cache  │  moka cache  ── HIT (real content only) ─────┼──► return
                  miss   └─────────────────────────────────────────────┘
                    │
                    ▼
   ┌──────────── escalation controller + PER-DOMAIN REPUTATION GOVERNOR (§4) ────────────┐
   │  per-URL budget: time_budget_ms, cost_budget_usd, attempts_remaining                  │
   │  per-domain: reputation budget (live-origin request allowance), AIMD pacer,           │
   │              PERMANENT hard-stop denylist (survives restart)                          │
   └───────────────────────────────────────────────────────────────────────────────────────┘
                    │
   ── ZERO-BAN-RISK rungs (do NOT touch the live protected origin) ──────────────────────────
                    │
   R0  IN-BAND salvage (cache + ADR-0002 hydration/JSON-LD on body we ALREADY have)
       │   free, 0 new requests. RealContent ──► (SPA? → R4 render) ──► cache+return
       │   verdict ≠ RealContent  (or no body yet)
       ▼
   R3  ARCHIVE fallback — Internet Archive CDX / Wayback (NEVER hits origin → IP-safe)
       │   RealContent (snapshot) ──► extract ──► cache+return
       │   miss / stale / gappy
       ▼
   ── LIVE-ORIGIN rungs (spend IP reputation — GATED by governor §4) ─────────────────────────
                    │  reputation budget OK + breaker not tripped
                    ▼
   R1  COHERENT wreq impersonation GET (adaptive-paced, profile-rotated) ──► classify
       │   RealContent ──► return
       │   blocked / soft-block
       ▼
   R2  (NEW — replaces proxy) ALTERNATIVE-SURFACE probe: sitemap/RSS/Atom/JSON-Feed +
       │   internal JSON / GraphQL / mobile-API probe (same-origin XHR often unchallenged)
       │   RealContent (JSON/feed) ──► extract ──► cache+return
       │   miss
       ▼
   R4  STEALTH HEADLESS (Runtime.enable-safe) — solve JS challenge in browser;
       │   persist cf_clearance / datadome cookie; reuse for the domain (§5 OPEN QUESTION)
       │   RealContent ──► return
       │   challenge unsolved / bot-detected
       ▼
   R5  COMMERCIAL UNBLOCK API (tier-3, opt-in, cost-gated) ──► classify ──► RealContent ──► return
       │   exhausted / budget hit / breaker tripped
       ▼
   GIVE UP ──► typed Error (Blocked{class} / ChallengeUnsolved / RateLimited)  [NOT cached]
       └─ if verdict is a HARD ban / explicit block / C&D ──► PERMANENT per-domain hard-stop (§4)
```

### 3.1 Rung table

For each rung: trigger (which `BlockClass` from ADR 0001 §4 routes here), what changes, latency
budget, **ban-risk**, max attempts, and give-up→next.

| Rung | Touches live origin? | Trigger (classifier verdict) | What changes | Latency budget | Ban-risk | Max attempts | Give-up → |
|---|---|---|---|---|---|---|---|
| **R0 In-band salvage** | **No** | always first (runs on cache body or any body already fetched). On a *good* body it is enrichment; on a soft-blocked body it salvages. | parse cache hit; ADR-0002 in-band sources (`hydration`, JSON-LD/OG promotion) on whatever body exists | ~1–5 ms (parse) | **none** | 1 | R3 |
| **R3 Archive (CDX/Wayback)** | **No** (hits `web.archive.org`, never the target origin) | Rung R0 missed **and** `enable_archive_fallback`. Especially when classifier last saw `Cloudflare403`/`JsChallenge`/`SoftBlock` (origin hostile → go around it). | query CDX `web.archive.org/cdx/search/cdx?url=<u>&filter=statuscode:200&output=json`, newest 200 snapshot, fetch `…/<ts>id_/<url>` raw HTML → `extract_page` | ≤ `archive_timeout_ms` (e.g. 4000) | **none** to *origin*; IA rate-limit risk (back off on 503, never hammer) | 1 | R1 |
| **R1 Coherent impersonation** | **Yes** | `SoftBlock`, generic `Http403`/`Cloudflare403` with no JS challenge, or first live attempt. Governor must approve (budget + breaker). | single coherent wreq GET with a Chrome emulation **profile** (TLS+H2+headers+UA as one unit, ASR-7); rotate *profile* (not headers alone) on retry; adaptive pace via §4 governor; honor `Retry-After` | ≤ 2× base | **low** (one well-formed request; the dominant risk is *rate/behavioral*, paced by §4) | `max_profiles` (default 2) | R2 |
| **R2 Alternative-surface (NEW)** | **Yes** (same origin, but cheaper/less-defended surface) | R1 blocked, or `Cloudflare403`/`DataDome403` shaped as edge block (the slot that used to route to *proxy*). | ADR-0002 out-of-band sources: sitemap / RSS / Atom / JSON-Feed discovery + internal JSON/GraphQL/mobile-API probe; reuse any `cf_clearance` cookie minted by a prior R4 (§5) | ≤ 3× base | **low** (static XML rarely challenged; API probe is 1–2 GETs, governor-paced) | `max_alt_probes` (default 3) | R4 |
| **R4 Stealth headless** | **Yes** | `JsChallenge { vendor }`; **or** `RealContent + is_spa` (the existing SPA render path, ADR 0001 §3). Governor must approve. | render with JS via `BrowserPool`; **Runtime.enable-safe** stealth (§7); on challenge, wait for `cf-mitigated` clear / cookie; persist `cf_clearance`/`datadome` cookie for the domain (§5) | ≤ `browser_timeout_secs` (default 15s) | **elevated** (full live render; behavioral surface largest) | 1 | R5 |
| **R5 Commercial unblock API** | **Yes (via vendor)** | `JsChallenge`/`Captcha`/`DataDome` that R4 could not solve, **and** `enable_unblock_api` + cost budget OK. The honest tier for behavioral-ML / active Turnstile (§6 #5). | hand URL to vendor unblock API; vendor returns rendered HTML/JSON → `extract_page` | ≤ `unblock_timeout_secs` | n/a to *our* IP (vendor's egress); **$$ per call** | 1 | GIVE UP |

**Which rungs touch the live protected origin (ban risk):** R1, R2, R4 hit the origin directly
(governor-gated). R0 and R3 do **not** (zero ban risk). R5 reaches the origin only via the vendor's
own egress, not our IP.

**Give-up conditions (any one):** R5 exhausted/disabled; `attempts_remaining == 0`;
`per_url_time_budget_ms` exceeded; next rung would exceed `per_url_cost_budget_usd` or
`session_cost_cap_usd`; **per-domain breaker tripped** (cooldown *or* permanent — §4). On give-up
return the most-informative typed error (ADR 0001 C4); never cache it as content. **If the verdict
is a hard ban / explicit block / C&D, additionally write the domain to the permanent denylist (§4)
so it is never attempted again — this restart-surviving stop is the single biggest behavioral
difference from ADR 0001.**

### 3.2 How it slots into `fetcher.rs` (additive; preserves ADR 0001 C1–C6)

Unchanged from ADR 0001 §3 except for ordering and the deleted proxy rung:

1. **`fetch_once` still captures `(status, headers, body)` for all statuses** (ADR 0001 C3 / ADR
   0002 §3.5.1) — currently `fetch_once` early-returns `Err(Blocked)` at fetcher.rs:208 before
   reading the body. This change is shared with both prior ADRs; no new requirement here.
2. **Cache insert gated on `RealContent`** (ADR 0001 C6, unchanged) — fetcher.rs:148.
3. **SPA path unified** (ADR 0001 §3 item 3, unchanged) — `is_spa` → R4 render via the same
   `BrowserPool` (fetcher.rs:135–137); classifier disambiguates challenge vs SPA.
4. **The escalation controller consults the reputation governor (§4) before every live-origin rung
   (R1/R2/R4/R5).** If the domain is on the permanent denylist, the controller fast-fails *before*
   R0 even runs a live request (R0/R3 may still serve archived/in-band content for a denylisted
   domain — they never touch the origin, so a permanent *origin* stop does not forbid them; this is
   an intentional nuance, see §4.4).
5. **The static `Throttle` (throttle.rs) is subsumed by the AIMD governor (§4).** The current
   `Throttle::apply_backoff` (drops a domain to 0.1 rps) is the crude ancestor of the AIMD
   multiplicative-decrease; the governor replaces it with a continuous per-domain rate that
   additively increases on success and multiplicatively decreases on soft signals.

The retry loop's backoff (fetcher.rs:127, `base * 2^n` with no jitter) gains jitter (ADR 0001 §3) —
still required.

---

## 4. Adaptive per-domain reputation governor (replaces proxy health-eject)

ADR 0001 governed liveness with a **proxy health-eject** (drop a dead proxy, rotate to the next) and
a **cooldown circuit breaker** (open for `domain_breaker_cooldown_secs`, then retry). Under
single-IP there is no proxy to eject and **a cooldown-then-retry breaker is actively dangerous**:
retrying a domain that just hard-banned our only IP cannot recover it and may *deepen* the ban. The
governor replaces both.

### 4.1 AIMD pacer (per domain)

An **Additive-Increase / Multiplicative-Decrease** controller on per-domain live-origin
concurrency and request rate — the same control law TCP uses for a shared scarce resource, here
applied to "WAF tolerance for our one IP." Empirically self-tuning, because **specific rate
thresholds are unknowable** (§6 #3: the widely-cited Cloudflare "10 req/2min" was *refuted*; pacing
must be empirical, never hardcoded).

- **Additive increase:** on each `RealContent` from a domain, nudge that domain's allowed rate up by
  `aimd_increase_step` (additive) and/or concurrency by 1, up to `per_domain_max_concurrency` /
  `per_domain_max_rate`.
- **Multiplicative decrease:** on any **soft** signal — `RateLimited` (429), `SoftBlock`, a
  `JsChallenge` appearing where content used to flow, or a 403 burst — multiply the domain's rate by
  `aimd_decrease_factor` (e.g. 0.5) and shrink concurrency. This is the "slow down *before* a hard
  ban" mechanism (ASR-6b).
- **Jitter:** every live request waits `base_delay * (1 ± jitter_ratio)` to avoid a periodic
  signature (behavioral signals are orthogonal to fingerprint — §6 #3).
- **Per-domain request budget:** `per_domain_request_budget` caps total live-origin requests to a
  domain per session; when exhausted, live rungs are skipped (R0/R3 archive/in-band may still serve).
  This bounds total reputation spend even if no block is ever observed.

Because rate-limiting *can* be keyed on JA3/JA4 (enterprise Bot Management; default is still
IP-keyed — §6 #3), and we run a *stable* coherent profile per domain, our impersonated fingerprint
is effectively its **own throttle bucket** — another reason to pace it as a scarce resource rather
than rotate it.

### 4.2 Per-domain circuit breaker — two trip modes

| Trip mode | Trigger | State | Recovery |
|---|---|---|---|
| **Soft (cooldown)** | `soft_breaker_fail_threshold` soft signals (429/SoftBlock/transient challenge) in the window | open → skip live rungs for `domain_breaker_cooldown_secs`; R0/R3 still allowed | auto half-open after cooldown, one probe; AIMD rate stays decreased |
| **HARD (permanent)** | classifier verdict indicates a **confirmed hard ban**, an **explicit block page**, or an operator-recorded **C&D** (GOAL.md §2). | open **permanently**; domain written to the persistent denylist (§4.3); all *live-origin* rungs (R1/R2/R4/R5) refused for this domain forever | **none** — never retried, never "rotated back in" (GOAL.md §2). Only an explicit operator action (remove from denylist file) can clear it. |

**Why permanent, not cooldown (the core single-IP difference):** with proxy rotation (ADR 0001),
a hard ban on one IP was recoverable by switching IPs, so a cooldown-then-retry-via-new-proxy made
sense. With one non-rotatable IP, a hard ban is terminal for that origin; retrying after a cooldown
(a) cannot succeed and (b) is the exact "evade a known block" behavior GOAL.md §2 forbids on both
legal and reputation grounds. So the hard-trip is a **terminal absorbing state**.

> **Classifier mapping (Rule 1 flag):** ADR 0001's `BlockClass` does **not** today distinguish a
> *transient* challenge from a *confirmed hard ban*. The hard-trip needs a definable signal.
> Proposed (to verify in Phase 2 against the benchmark, do **not** assume): a hard ban =
> `Cloudflare403`/`DataDome403`/`Http403` that **persists after R4 stealth headless also fails**
> (i.e., we presented a real browser and still got blocked → it is IP/account reputation, not a
> solvable challenge), **or** an explicit operator C&D entry. A single 403 is *soft* (cooldown),
> not hard. This threshold is config-tunable and must be validated empirically before it can
> permanently denylist a domain — a false hard-trip permanently loses a reachable domain, so the bar
> is deliberately high.

### 4.3 Permanent denylist (persistence)

A per-domain denylist that **survives process restart** (the soft cooldown breaker is in-memory and
does not). Minimal, file-backed (no DB dependency, consistent with GOAL.md §4 single-operator
non-goal): append-only JSON-lines at a configured path, loaded at startup, written on every hard
trip. Each entry: `{ domain, reason (HardBan|ExplicitBlock|CeaseAndDesist), classifier_verdict,
first_seen_iso, note }`. The controller checks it before any live-origin rung. This is also the
GOAL.md §2 / Phase 5.2 compliance audit artifact.

### 4.4 Denylist semantics nuance

The permanent stop forbids **live-origin** access (R1/R2/R4/R5) — it does **not** forbid serving the
domain's content from **zero-ban-risk** surfaces (R0 in-band on an already-cached body, R3 archive
snapshot), because those never touch the banned origin and therefore neither violate the legal
guardrail nor risk the IP. This preserves some coverage on banned domains without ever re-contacting
them. (If the operator wants a *total* blackout including archive, a `denylist_blocks_archive=true`
flag covers it; default false.)

---

## 5. Cookie / session reuse — OPEN QUESTION (Phase-3 spike, do not assume)

The ladder's R4→R2 reuse pattern (mint a `cf_clearance`/`datadome` cookie once in the stealth
browser, then drain the cheap JSON API / re-request on the wreq client reusing that cookie) is the
highest-leverage way to amortize the expensive headless rung across many cheap requests — **and**
it minimizes reputation spend (one render instead of many). ADR 0002 §1 relies on the same pattern.

**Unverified assumption (CLAUDE.md Rule 1):** it is **not confirmed** that a `cf_clearance` /
DataDome cookie minted by the headless browser can be **replayed by the wreq client**. Cloudflare
clearance cookies are widely reported to be **bound to the client's TLS/JA3 + HTTP/2 fingerprint**
(and IP); if so, a cookie obtained under chromiumoxide's fingerprint will be **rejected** when
presented by the wreq client unless wreq's emulation profile produces a *matching* JA3/H2
fingerprint — which is exactly the coherence ASR-7 demands but has **not been measured** for this
browser↔wreq pair.

**Phase-3 spike (gates R4→R2 cookie reuse):** mint a `cf_clearance` in the stealth browser, present
it on the wreq client against the same Cloudflare-protected benchmark domain, and measure whether it
is accepted. **Do not build the R4→R2 reuse path until this spike confirms it.** If cookies are
fingerprint-bound and the browser/wreq fingerprints cannot be aligned, the fallback is: keep the
*solved session inside the browser* (drain the API from the headless context, not the wreq client) —
slower and more reputation-costly, but correct. This is a single-IP-amplified version of ADR 0001's
already-flagged "persist `cf_clearance` for reuse" claim.

---

## 6. Config delta vs ADR 0001 §5

All fields additive, `#[serde(default)]`, off-safe (config-management rule: secure defaults,
validate at startup). No secrets as literals — keys/endpoints that could carry creds are env-var
*references*. This is a **delta**: it lists what is removed vs ADR 0001 §5 and what is added. The
classifier-tuning and budget fields from ADR 0001 §5 (`per_url_*`, `session_cost_cap_usd`,
`soft_block_min_bytes`, `block_marker_overrides`, negative cache) are **retained unchanged**.

### 6.1 REMOVE (proxy rotation is deleted)

```
- proxy_pool_env
- proxies
- max_proxy_attempts
- proxy_sticky_per_domain
- proxy_eject_after_failures
```

(ADR 0001 §5's startup-validation clause "if max_escalation_rung >= 2 and no proxy source, clamp" is
also removed; rung numbering is redefined in §3.)

### 6.2 ADD

```rust
pub struct CrawlerConfig {
    // ... existing fields unchanged; ADR 0001 §5 budget/classifier fields retained ...

    // -- Single-IP posture (NEW; documents the hard constraint in config) --
    /// Hard guard: if true (DEFAULT), any proxy/IP-rotation field is rejected at startup.
    #[serde(default = "d_true")] pub single_ip_mode: bool,

    // -- Adaptive per-domain pacing / AIMD governor (replaces proxy health-eject) --
    #[serde(default = "d_max_conc")]   pub per_domain_max_concurrency: u32,   // e.g. 2
    #[serde(default = "d_target_rate")] pub per_domain_target_rate_rps: f64,  // start point, e.g. 1.0
    #[serde(default = "d_min_rate")]    pub per_domain_min_rate_rps: f64,     // floor, e.g. 0.05
    #[serde(default = "d_max_rate")]    pub per_domain_max_rate_rps: f64,     // ceil, e.g. 4.0
    #[serde(default = "d_aimd_inc")]    pub aimd_increase_step_rps: f64,      // additive, e.g. 0.25
    #[serde(default = "d_aimd_dec")]    pub aimd_decrease_factor: f64,        // multiplicative, e.g. 0.5
    #[serde(default = "d_jitter")]      pub pacing_jitter_ratio: f64,         // e.g. 0.3 (±30%)
    #[serde(default = "d_req_budget")]  pub per_domain_request_budget: u32,   // live-origin cap/session

    // -- Circuit breaker: soft cooldown retained; HARD trip is permanent (NEW) --
    #[serde(default = "d_soft_thresh")] pub soft_breaker_fail_threshold: u32, // soft signals → cooldown
    #[serde(default = "d_cb_cooldown")] pub domain_breaker_cooldown_secs: u64,// soft-trip cooldown
    /// Persistent, restart-surviving permanent hard-stop denylist.
    #[serde(default)] pub permanent_denylist_path: Option<PathBuf>,           // None => in-memory only (warn)
    /// If true, a permanently denylisted domain is ALSO blocked from archive/in-band (total blackout).
    #[serde(default)] pub denylist_blocks_archive: bool,                      // default false (§4.4)

    // -- R3 archive fallback (Internet Archive CDX — IP-safe) --
    #[serde(default)] pub enable_archive_fallback: bool,                      // default OFF (ADR 0002 trend)
    #[serde(default = "d_cdx")] pub archive_cdx_endpoint: String,             // web.archive.org/cdx/search/cdx
    #[serde(default = "d_arch_to")] pub archive_timeout_ms: u64,              // e.g. 4000
    #[serde(default)] pub archive_max_snapshot_age_days: Option<u32>,         // None => any age; reject staler
    #[serde(default)] pub archive_user_agent: Option<String>,                // polite IA identification

    // -- R2 alternative-surface toggles (ADR 0002 out-of-band, promoted to a rung) --
    #[serde(default = "d_true")] pub src_feed: bool,                         // RSS/Atom/JSON-Feed
    #[serde(default = "d_true")] pub src_sitemap: bool,
    #[serde(default)]            pub src_internal_api: bool,                  // probe volume → default OFF
    #[serde(default = "d_alt_probes")] pub max_alt_probes: u8,               // e.g. 3

    // -- R1 coherent impersonation profiles (NOT independent header rotation) --
    #[serde(default)] pub emulation_profiles: Vec<String>,                   // e.g. ["chrome-124","chrome-131"]; empty => single
    #[serde(default = "d_max_profiles")] pub max_profiles: u8,               // e.g. 2

    // -- R5 commercial unblock API (tier-3; opt-in, cost-gated; replaces CAPTCHA-only R5) --
    #[serde(default)] pub enable_unblock_api: bool,                          // LEGAL/COST gate, default OFF
    #[serde(default)] pub unblock_provider: Option<String>,                  // verify API+pricing in Phase 3
    #[serde(default)] pub unblock_api_key_env: Option<String>,               // env var NAME, never the key
    #[serde(default = "d_unblock_to")] pub unblock_timeout_secs: u64,
}
```

**Startup validation (config-management rule):**
- If `single_ip_mode` (default) **and** any removed `proxy_*`-shaped field is present in the loaded
  config → **fail fast** with a message pointing at this ADR. (Catches stale ADR-0001 configs.)
- If `enable_unblock_api` then `unblock_api_key_env` must name a *set* env var, else fail fast.
- If `permanent_denylist_path` is `None` → **warn** (a hard ban will not survive restart; on the next
  run we could re-contact a domain we are legally/operationally bound to never touch again — GOAL.md
  §2). Recommend setting it.
- Reject `aimd_decrease_factor` outside `(0,1)`; reject `per_domain_min_rate_rps > per_domain_max_rate_rps`.
- `enable_unblock_api` and `respect_robots_txt=false` remain explicit opt-ins (GOAL.md legal gate).

---

## 7. Stealth headless — Runtime.enable-safe (R4)

Unchanged in intent from ADR 0001 Rung 4 / TASKS 3.1, restated because R4 is now the *only*
live-origin challenge-solver below the paid tier (no proxy rung sits beside it). Two options
(present both, per ground truth #4 — both are immature):

- **Option R4-a (recommended start): hand-patch `browser.rs`.** Defeat the canonical
  `Runtime.enable → consoleAPICalled` CDP leak via `Page.createIsolatedWorld` +
  `addScriptToEvaluateOnNewDocument` (drop `navigator.webdriver`, sync UA/`Sec-CH-UA`, real
  viewport/locale, `--headless=new`, replace the fixed sleep with a network-idle wait). This is the
  plan already recorded in research/0001 Area 2 and TASKS 3.1.
- **Option R4-b: adopt a transport-layer-CDP-stealth crate** — `eoka` (crates.io 0.3.15) or
  `chaser-oxide` (0.2.3), which do transport-layer CDP blocking + isolated worlds + UA/`Sec-CH-UA`
  sync (the layer JS-injection stealth like `chromiumoxide_stealth` cannot reach).

> **Rule 1 flag — both options UNPROVEN vs commercial WAF from a single IP.** `eoka` is verified only
> against open-source detectors (sannysoft, rebrowser 6/6, creepjs 33%); neither crate has a verified
> result against live Cloudflare/DataDome enterprise Bot Management from one IP. Treat R4 as
> *necessary but not guaranteed*; its failure is precisely what routes to R5. Verify in the Phase-3
> stealth spike before relying on it.

---

## 8. Honest coverage estimate (which rung recovers what; residual)

Per CLAUDE.md Rule 1 these are **architectural estimates, not measured numbers** — the project has no
G2 baseline yet (GOAL.md §5). Confirm every cell against `benchmark/urls.jsonl`. They restate the
grounded properties from §6 ground truth, not invented rates.

| Site class | Recovered by | Confidence | Ban-risk to recover |
|---|---|---|---|
| Static HTML, no protection | R1 (coherent GET) — often R0 if cached | high | low |
| SSR'd JS-SPA (Next/Nuxt/Apollo) with hydration blob | **R0 in-band** (ADR 0002 #1) — content is in the first body | high | **none** |
| News/article with JSON-LD `articleBody` | **R0 in-band** (ADR 0002 #3) | high | **none** |
| "Basic protection" (passive fingerprinting) | R1 coherent wreq (clears ~30–40% of "basic" per §6 #1; curl_cffi-class ~80% on mixed) | medium | low |
| Cloudflare/DataDome **JS challenge** (solvable) | R4 stealth headless (Runtime.enable-safe) | medium (R4 unproven vs commercial WAF, §7) | elevated |
| Same-origin data behind a weaker XHR/JSON or mobile API | **R2 alternative-surface** (ADR 0002 #2 — frequent, not guaranteed) | medium | low |
| Origin hostile but page is **archived** | **R3 archive/CDX** — IP-safe; gappy per-URL, good for major sites (§6 #2c) | medium (coverage gaps + staleness; IA-block trend, ADR 0002) | **none** |
| Pure-CSR + API-gated + feed-less + never-archived + **behavioral-ML / active Turnstile** | **R5 commercial unblock API only** (§6 #5) | — | n/a (vendor egress) |

**Residual that single-IP genuinely cannot reach without tier-3:** per-customer **behavioral ML**
and **active Turnstile/WASM** challenges (§6 #5) are *genuinely impossible* from one IP with OSS
technique alone. With `enable_unblock_api=false` (default), these pages are **unreachable** and the
ladder correctly gives up with a typed error. This is the explicit, honest boundary GOAL.md §2
already states ("ANY site is the target, not a guarantee"). Hitting G1 ≥ 0.90 therefore depends on
the *composition* of `benchmark/urls.jsonl`: if the behavioral-ML subset exceeds ~10% of the set and
R5 stays off, 90% is unreachable by construction — an **operator input owed** (which subset, and
whether tier-3 is funded for G2). Flagged, not assumed.

---

## 9. Interfaces / contracts (Phase 2/3 must honor) — mirrors ADR 0001 §7 C1–C6

**C1 — Classifier (UNCHANGED from ADR 0001 C1).** Same `classify(status, headers, final_url, body,
cfg) -> BlockClass`, same priority order. **One addition:** the controller must be able to ask "does
this verdict, *after R4 also failed*, constitute a HARD ban?" — proposed as a separate pure helper so
the classifier itself stays stateless:
```rust
/// Pure. Decides the breaker trip mode for a give-up verdict, given whether the
/// live browser rung (R4) was already attempted and also blocked.
fn breaker_verdict(class: &BlockClass, r4_attempted_and_blocked: bool, cfg: &GovernorConfig)
    -> BreakerTrip;  // BreakerTrip { None, SoftCooldown, HardPermanent }
```

**C2 — Escalation controller (REVISED from ADR 0001 C2).** Same public signature
(`Fetcher::fetch(url) -> Result<FetchResult>`; callers in `crawler.rs` unchanged). Owns: the §3.1
rung selection and ordering (R0→R3→R1→R2→R4→R5), **reputation-governor consultation before every
live-origin rung**, breaker enforcement (soft + hard), and cache-insert gating (C6). **Must refuse
all live-origin rungs for a domain on the permanent denylist** and route only to R0/R3 (unless
`denylist_blocks_archive`).

**C3 — Reputation governor (NEW; replaces ADR 0001's proxy health + cooldown breaker).** A per-domain
component, keyed by registrable domain:
```rust
trait ReputationGovernor: Send + Sync {
    /// Called before a live-origin rung. Waits out the AIMD pace + jitter, then returns
    /// whether the rung may proceed (budget left, breaker not open, not permanently denied).
    async fn admit(&self, domain: &str) -> Admission;   // Admission { Proceed, SkipLive(reason), DeniedPermanent }
    /// Feed the classifier verdict back so AIMD can adjust and the breaker can trip.
    fn record(&self, domain: &str, outcome: &BlockClass, rung: Rung);
    /// True iff domain is on the persistent permanent denylist.
    fn is_permanently_denied(&self, domain: &str) -> bool;
}
```
- AIMD math per §4.1; trip modes per §4.2; persistence per §4.3.
- The existing `Throttle` (throttle.rs) is **replaced** by the AIMD rate inside the governor (its
  `apply_backoff` becomes the multiplicative-decrease).

**C4 — Error semantics (extends ADR 0001 C4).** Retain `Error::ChallengeUnsolved { url, vendor }`
and the `BlockClass`-carrying give-up. **Add** `Error::PermanentlyDenied { domain, reason }` returned
when a live fetch is refused due to the permanent denylist, so callers can distinguish "we chose
never to contact this" from "we tried and failed."

**C5 — Observability (extends ADR 0001 C5).** Every escalation emits `{url, domain, rung, verdict,
signals, latency_ms, cost_usd_delta}` **plus governor state** `{aimd_rate_rps, concurrency,
request_budget_remaining, breaker_state}`. A **hard trip emits a distinct, alert-worthy event**
`{domain, reason, classifier_verdict}` (it is permanent and a compliance-audit artifact, Phase 5.2).
Running session cost still checked against `session_cost_cap_usd`.

**C6 — Cache contract (UNCHANGED from ADR 0001 C6).** `cache.insert` iff final
`BlockClass::RealContent` (including R0/R2/R3/R5-derived real content). Challenge/block/denied
results never cached as content. Optional negative cache unchanged.

**C7 — Persistent denylist contract (NEW).** File-backed, append-only, loaded at startup, fsync'd on
write; schema per §4.3. Read by C2/C3 before any live rung. The single writer is the governor; format
is JSON-lines so it doubles as the Phase 5.2 audit log. No secrets in it.

---

## 10. Behavior at 10× / 100× load, rollback

**10× (more URLs):** the controller is per-URL; the governor is per-domain shared state in a
`DashMap` (like the existing `Throttle`). More URLs to the *same* domain do **not** increase that
domain's reputation spend beyond `per_domain_request_budget` — they queue behind the AIMD pacer,
which is the desired single-IP behavior (reputation is the bottleneck, not throughput). The single
browser (R4) remains the heavy-rung bottleneck (ADR 0001 §6) and must stay rate-limited separately
from cheap rungs; R0/R3 (in-band/archive) do not touch it.

**100×:** unchanged ceiling from ADR 0001 — single-node browser/tier-3 throughput is the hard limit
(GOAL.md non-goal: not internet-scale). Under single-IP the *reputation* budget shed-loads
gracefully: when a domain's budget is exhausted, live rungs are skipped and only R0/R3 serve, rather
than the system racing toward a ban. Degradation is "serve archived/in-band or give up," never
"hammer the origin."

**Rollback / migration:**
- **Reversible by config:** all new fields default off/safe; `enable_escalation=false` (ADR 0001)
  still restores single-strategy behavior. With escalation on but archive/alt/unblock off and a
  generous budget, behavior approximates ADR 0001-minus-proxy.
- **vs ADR 0001:** this is a *Proposed* amendment; if rejected at GATE 1, ADR 0001 stands. If
  accepted, ADR 0001's Rung 2 / proxy config is never implemented (Phase 2 Task 2.3 is dropped).
- **The one irreversible-by-design element is the permanent denylist** — by construction a hard trip
  is terminal. Operator override = manual edit/removal of the denylist file (the only path back,
  intentionally a human decision per GOAL.md §2).

---

## Citations (ground truth, deep-research pass 2026-06-11 — adversarially verified)

1. TLS/HTTP2/JA3/JA4 impersonation necessary-but-not-sufficient; coherence across layers; Chrome
   110+ ClientHello randomization (~10^12 perms); wreq provides protocol-level extension/SETTINGS
   control:
   - https://github.com/0x676e67/wreq
   - https://blog.cloudflare.com/per-customer-bot-defenses/
   - https://scrapfly.io/blog/posts/how-to-bypass-cloudflare-anti-scraping
2. Alternative-surface / data-layer is highest-coverage + most IP-safe; Internet Archive CDX never
   touches the live origin; hydration is per-framework/per-site; CDX per-URL coverage is gappy:
   - https://github.com/novitae/njsparser
   - https://github.com/vercel/next.js/discussions/42170
   - https://github.com/internetarchive/wayback/tree/master/wayback-cdx-server
3. Pacing preserves reputation, spoofing does not; specific Cloudflare rate numbers REFUTED (no
   "10 req/2min" in docs) → pacing must be adaptive/empirical; rate-limit can be JA3/JA4-keyed
   (enterprise add-on; default IP-keyed); bot-score < 30 = automation:
   - https://developers.cloudflare.com/waf/rate-limiting-rules/best-practices/
   - https://developers.cloudflare.com/waf/rate-limiting-rules/parameters/
   - https://developers.cloudflare.com/bots/concepts/bot-score/
4. Transport-layer CDP stealth defeats the `Runtime.enable → consoleAPICalled` leak JS-injection
   stealth cannot hide; `eoka` / `chaser-oxide` are immature/experimental, verified only vs OSS
   detectors; hand-patch (`Page.createIsolatedWorld` + `addScriptToEvaluateOnNewDocument`) is the
   alternative:
   - https://rebrowser.net
   - https://crates.io/crates/eoka
   - https://github.com/ccheshirecat/chaser-oxide
5. Genuinely impossible from one IP: per-customer behavioral ML + active Turnstile/WASM challenge →
   commercial unblock API (tier-3) is the only reliable option, must stay opt-in/config-gated:
   - (ground truth #5, this research pass) — consistent with GOAL.md §2 rung 5 and ADR 0001 Option C.

**Flagged assumptions to verify before implementation (CLAUDE.md Rule 1):**
1. **Cookie fingerprint-binding (§5):** that a browser-minted `cf_clearance`/DataDome cookie can be
   replayed by the wreq client is **unverified**; gate the R4→R2 reuse path behind a Phase-3 spike.
   Do not build it assuming it works.
2. **Hard-ban classifier signal (§4.2):** ADR 0001's `BlockClass` does not yet distinguish a
   transient challenge from a confirmed hard ban; the proposed "403 persists after R4 also failed"
   rule must be validated against the benchmark before it is allowed to *permanently* denylist a
   domain (a false hard-trip permanently loses a reachable domain).
3. **R4 stealth efficacy (§7):** `eoka`/`chaser-oxide` and the hand-patch are **unproven vs
   commercial WAF from a single IP**; verify in the Phase-3 stealth spike.
4. **Coverage cells (§8)** are architectural estimates, not measured; confirm against
   `benchmark/urls.jsonl`. Whether G1 ≥ 0.90 is reachable with R5 off depends on the behavioral-ML
   share of the benchmark set — **operator input owed** (GOAL.md §5).
5. **Tier-3 provider + per-session $ cap (R5)** remain operator inputs (carried from ADR 0001 §1).
