# Compliance Gate 0001 — LIVE CAPTCHA-Solving Path (Ladder Rung R5)

- **Status:** **GO-WITH-CONDITIONS** (conditional authorization to wire/enable the live R5 solve path)
- **Date:** 2026-06-11
- **Owner:** compliance-officer (read-only assessment)
- **Gate required by:** GOAL.md §2 ("Aggressive bypass features opt-in via config; compliance gate (TASKS Phase 5) before enabled") and TASKS.md task 5.2.
- **Scope of this decision:** authorization, conditions, and permanent prohibitions for wiring the *already-built, default-OFF* R5 abstraction (`crates/crawler/src/captcha/`) into the live fetch path. This document does **not** modify product code; the conditions below are the implementation checklist the `backend-engineer` must bake into the live wiring.
- **NOT LEGAL ADVICE.** This is an engineering-compliance assessment. Several items below are flagged **NEEDS LEGAL COUNSEL**; those must be resolved by counsel, not by this doc or by an engineer.
- **Inputs read:** `GOAL.md` §2; `TASKS.md` Phase 5; `docs/adr/0001-bypass-escalation-ladder.md`; `docs/design/0004-stealth-and-captcha-layers.md`; `crates/crawler/src/captcha/mod.rs`; `crates/crawler/src/captcha/cost.rs`; `crates/crawler/src/captcha/inject.rs`; `crates/crawler/src/governor.rs`; `crates/common/src/config.rs`.

---

## 1. The central question — does "treat CAPTCHA as opt-out" forbid solving it?

GOAL.md §2 contains two lines that pull in opposite directions, **in the same section**:

- **Line A (2026-06-06 operator decision):** "pursue **scrape-any-data**" with a ladder whose top rung is "**Proxy rotation + CAPTCHA solve**" and a tier-3 commercial unblocker as an **opt-in fallback**.
- **Line B (legal guardrails, same section):** "Polite rate limits even when ignoring robots; **treat robots/CAPTCHA as opt-out signals.**"

These are reconcilable only under a narrow reading, and I adopt that narrow reading as the gate condition:

**A CAPTCHA is an opt-out signal at the *domain-reputation* level, not a puzzle to be defeated wherever it appears.** The operator's scrape-any-data decision authorizes *attempting* a solve as an explicit, opt-in, cost-capped fallback on a confirmed `Captcha` verdict for a **public, unauthenticated** resource. It does **not** authorize treating a CAPTCHA as a routine obstacle to grind through indefinitely. The reconciliation that keeps both lines true:

1. **A CAPTCHA encountered while NOT logged in, on a public page, is a friction control** — the operator may, opt-in, attempt one solve as the ladder's last rung (Line A). This is a single bounded attempt, not a campaign.
2. **A CAPTCHA that *gates a login / sits behind an auth wall* is an opt-out we must honor absolutely** — solving it would cross the "public + unauthenticated only" hard default and is the single strongest "circumvention of an access control" fact pattern. Permanently off-limits (§4).
3. **A repeated / hard block is an opt-out we must honor permanently** — the moment a domain hard-blocks (the governor's `HardPermanent` trip), it is denylisted and **never re-attempted by any rung, including R5**. This is the existing `permanent_denylist_path` mechanism in `governor.rs`, and it is what makes "treat CAPTCHA as opt-out" true at the level that matters: we do not rotate/retry our way back in after a domain has said no. This directly honors the GOAL.md §2 "Block or C&D = permanent per-domain hard stop. Never rotate IPs back in" line and the *Power Ventures / 3taps* fact pattern cited there.

**Resolution:** Automated CAPTCHA solving is **permissible at all, but only in the narrow public-unauthenticated, single-attempt, opt-in, denylist-respecting band** described above. The "opt-out" line is satisfied by (a) never solving a CAPTCHA that protects authenticated access, and (b) treating any hard block as a permanent per-domain stop that R5 can never re-enter. Outside that band, solving is **not** authorized. This narrow band is exactly what the conditions in §3 enforce.

> **NEEDS LEGAL COUNSEL (C-1):** Whether a single opt-in solve of a CAPTCHA on a *public, unauthenticated* page is itself "circumvention of a technological access-control measure" under CFAA / DMCA §1201 / equivalent is a legal-interpretation question. *hiQ v. LinkedIn* (9th Cir.) suggests scraping *public* data is not CFAA "unauthorized access," but a CAPTCHA is arguably an access control and the law is jurisdiction- and fact-specific. This gate authorizes the **engineering posture** (narrow band, default-off, denylist-respecting); it does **not** opine that the act is lawful in any given jurisdiction. Counsel must confirm before the operator flips the flag in any production/commercial use.

---

## 2. Exposure map (cite the specific concern per item)

| # | Exposure | Specific concern | Type | Where it bites in this codebase |
|---|---|---|---|---|
| E-1 | **CFAA-style circumvention** (US) / Computer Misuse Act (UK) / equivalent unauthorized-access statutes | Solving a CAPTCHA to reach content can be framed as defeating an access-control measure. Risk is **low** for public+unauthenticated pages (*hiQ v. LinkedIn*), **high** the moment the CAPTCHA gates a login or a known-restricted endpoint (*Power Ventures*, *3taps* — circumventing a block to regain access). | **NEEDS LEGAL COUNSEL** (see C-1) | Mitigated structurally by §4 (auth walls permanently off-limits) + the governor denylist (`governor.rs`). |
| E-2 | **Target-site ToS breach** | Most site ToS prohibit automated access / circumventing anti-bot measures. Breach is generally a **contract** matter (account termination, C&D, civil claim), not criminal — but a C&D converts a tolerated scrape into a *Power Ventures* unauthorized-access posture. | **NEEDS LEGAL COUNSEL** | The "C&D = permanent hard stop" line in GOAL.md §2 must be operationalized: a C&D'd domain goes on the permanent denylist (currently only auto-tripped on post-R4 hard block — see Gap G-3). |
| E-3 | **CAPTCHA-solver provider's OWN ToS** | CapSolver / 2Captcha (the two providers in `captcha/mod.rs`) impose their own acceptable-use terms; some prohibit certain target classes (e.g. solving CAPTCHAs that gate financial / government / account-takeover flows). Using the provider outside *its* AUP is an independent breach against the provider. | **NEEDS LEGAL COUNSEL** (review the chosen provider's live AUP before enabling) | `captcha_provider` config selects the provider; the provider's AUP is not encoded anywhere — must be reviewed out-of-band. |
| E-4 | **GDPR / PII if the solve gates access to personal data** | If solving a CAPTCHA yields a page containing personal data of EU/UK data subjects, the operator becomes a **controller** processing that data and needs a lawful basis (Art. 6), must honor data-subject rights (Arts. 15–21), and must respect TDM opt-outs (DSM Directive Art. 4). GOAL.md §2 already commits to "no PII retention by default." | **NEEDS LEGAL COUNSEL** for lawful-basis/controller determination; **ENGINEERING GAP** for the retention control | No PII-classification or PII-suppression exists on the R5-gated content path (Gap G-4). GOAL.md §2 "no PII retention by default; minimize, pseudonymize, honor opt-out, retention limits" is a stated default but is **not enforced in code** on solved-gated pages. |
| E-5 | **Cross-border transfer** (solve request + page content) | The solve request sends the target `website_url` + `site_key` to a third-country provider (CapSolver/2Captcha). The URL itself can be personal data / reveal a data subject. If page content with EU personal data is processed, transfer rules (GDPR Ch. V) apply. | **NEEDS LEGAL COUNSEL** | `captcha/capsolver.rs` / `twocaptcha.rs` send `websiteURL` + `websiteKey` to the provider — confirmed by the trait contract in `mod.rs`. No DPA with the provider is referenced. |
| E-6 | **Secret leakage (provider API key)** | A leaked solver key is financial exposure (others spend your balance) and an audit failure. | **ENGINEERING GAP — already mitigated** | `captcha/mod.rs` reads the key by env-var NAME only (`captcha_api_key_env`), never a literal, never logged (logs provider name only). `build_solver` fails fast if the env var is unset. **Clean.** |
| E-7 | **Cost blowout** | Paid solves can run away. | **ENGINEERING GAP — already mitigated** | `captcha/cost.rs` `CostMeter` reserves before each solve, hard-halts at `captcha_session_cost_cap_usd` (default $5), one-time WARN at $5 threshold, refund path. Concurrent solvers share one total. **Clean.** |

**Materially-compliant areas already built (clean findings — worth stating explicitly):**
- **Default-OFF gating** — `decide_gate` returns `Disabled` unless master switch + known provider + set env var (`captcha/mod.rs`). Verified by 8 unit tests.
- **Secret handling** — env-var-name indirection, never logged (E-6).
- **Cost cap** — hard halt + WARN, unit-tested (E-7).
- **Permanent denylist** — file-backed, restart-surviving, no secrets in the entry schema (`governor.rs` `DenylistEntry` = domain/reason/verdict/timestamp). This is the spine of "never rotate back in."
- **Token hygiene** — `CaptchaToken` doc-comment marks it single-use/~2-min TTL, "never cache it"; `inject.rs` JSON-escapes the token and never logs it.

---

## 3. Decision: **GO-WITH-CONDITIONS**

The R5 abstraction as built (`crates/crawler/src/captcha/`) is compliance-sound for the **off** state and for the cost/secret dimensions. It is **NOT yet safe to wire live** because the access-boundary controls that GOAL.md §2 promises (public-only, auth-wall prohibition, CAPTCHA-domain opt-out, PII non-retention, denylist on CAPTCHA give-up) are **stated in prose but not enforced in the live path**. Therefore: **conditional GO** — the live wiring may proceed **only** when every MUST condition below is implemented and verified by tests. Each condition is written to be a concrete implementation checklist item for the `backend-engineer` (task 5.2 → live-wiring task).

### 3.1 Conditions already satisfied by the built module (verify they remain true at wiring time — do not regress)

- **K-1 Default-OFF.** `enable_captcha_solver` defaults `false`; solver never constructed otherwise. *(config.rs:70, captcha/mod.rs `decide_gate`)* — **MUST NOT regress.**
- **K-2 Hard per-session cost cap.** `captcha_session_cost_cap_usd` (default $5), reserved-before-call, hard halt. *(cost.rs)* — **MUST NOT regress;** the live wiring MUST call `try_reserve` **before** the provider call, not after.
- **K-3 API key by env-var name only, never logged.** *(captcha/mod.rs `build_solver`)* — **MUST NOT regress.**
- **K-4 Solve only on a confirmed `Captcha` verdict.** The ladder must route to R5 **only** from the classifier `Captcha{kind}` verdict (or an R4-failed `JsChallenge` per ADR 0003), never speculatively. *(Design 0004 §2.5)* — **MUST** be honored by the controller wiring.
- **K-5 Permanent denylist short-circuits before any live rung.** `governor.admit()` returns `DeniedPermanent` first; a denylisted domain MUST never reach R5. *(governor.rs:251)* — **MUST** be the first check in the R5 entry path.

### 3.2 Conditions the `backend-engineer` MUST implement in the live wiring (these are NEW — the gate is conditional on them)

- **C-1 (MUST) Explicit per-run opt-in, distinct from the build-time config flag.** `enable_captcha_solver=true` in a shipped config is necessary but **not sufficient**. The live solve must additionally require a per-invocation opt-in (e.g. an explicit `allow_captcha_solve: true` on the search/scrape request, default false) so a single global flag cannot silently turn every crawl into a paid-solve crawl. Default-deny if absent.
- **C-2 (MUST) Public + unauthenticated targets only — enforced, not assumed.** Before any R5 solve, the controller MUST assert the request carries **no** credentials, cookies that denote a session/login, `Authorization` header, or basic-auth in the URL. If any auth material is present on the request or the target is reached via a login flow, R5 is **refused** (typed error), not attempted. This operationalizes GOAL.md §2 "Never log in / never cross an auth wall." **(See Gap G-1 — no such guard exists today.)**
- **C-3 (MUST) Per-domain CAPTCHA opt-out / never re-attempt a hard-blocked domain.** A domain that returns a hard block, a C&D, or for which an R5 solve has already failed once MUST be recorded so R5 is never attempted on it again. Today the governor hard-trips the permanent denylist only on a **post-R4 JsChallenge/Cloudflare403** (`breaker_verdict` → `HardPermanent`). The wiring MUST extend this so that **(a)** an R5 give-up (`ChallengeUnsolved`) on a domain adds it to a per-domain "do-not-solve" set (at minimum a soft per-session set; preferably the permanent denylist), and **(b)** a manual/operator C&D entry can be added to `permanent_denylist_path`. **(See Gap G-3.)**
- **C-4 (MUST) Honor a domain denylist file before solving.** R5 entry MUST consult `permanent_denylist_path` (via `governor.is_permanently_denied`) AND an operator-editable explicit "captcha denylist" so the operator can pre-list domains that must never be solved (e.g. anything the operator knows is auth-gated, regulated, or has objected). Default-deny on match.
- **C-5 (MUST) Audit-log every solve attempt and outcome — no secrets/tokens/PII.** Emit a structured `tracing` event for every R5 attempt with **exactly**: `{domain (registrable), captcha_kind, provider, outcome (solved|failed|refused|cost_capped), cost_usd, cumulative_usd, timestamp}`. MUST NOT include: the solved **token**, the API **key**, full **URL with query string** (log registrable domain or path-without-query only, since query strings can carry PII/identifiers), any **page content**, or any **PII**. This is the GOAL.md §2 "audit log" + logging-domain "AUDIT TRAIL"/"NO SECRETS IN LOGS" requirement. The existing meter event (cost.rs) covers cost; the wiring MUST add the domain/kind/outcome event. **(See Gap G-2.)**
- **C-6 (MUST) No PII retention from solved-gated pages.** Content obtained *after* an R5 solve MUST flow through the same "no raw retention / store transformed-attributed derivative" path GOAL.md §2 promises, and MUST NOT be written to the persistent cache or any durable store as a raw copy keyed to a person. At minimum: do not persist raw HTML of an R5-gated page beyond the in-memory request lifecycle; if any retention occurs it must be the transformed/extracted derivative only. **(See Gap G-4 — this control does not exist yet and is the weakest area.)**
- **C-7 (MUST) Single bounded attempt; no solve campaigns.** One solve attempt per `Captcha` verdict per page, bounded by `captcha_timeout_secs`. On failure → typed `ChallengeUnsolved` give-up and route to C-3 (record the domain), never an immediate re-solve loop. reCAPTCHA v3 MUST be treated low-confidence (attempt once at most, or skip if `captcha_v3_low_confidence`) per Design 0004 §2.7.
- **C-8 (MUST) Provider AUP pre-check is an operator/legal precondition, surfaced at startup.** Startup MUST emit a WARN naming the selected `captcha_provider` and stating that enabling it asserts the operator has reviewed and accepted that provider's acceptable-use policy (E-3). (The existing `build_solver` log line already notes "legal/compliance gate is the operator's responsibility" — extend it to name the AUP obligation.)
- **C-9 (SHOULD) Robots/TDM coherence.** `respect_robots_txt` defaults `true` (config.rs:29). If an operator has set `respect_robots_txt=false` AND `enable_captcha_solver=true`, startup SHOULD emit a WARN that the combination maximizes ToS/CFAA exposure, and the per-domain opt-out (C-3/C-4) becomes the only remaining opt-out signal — so it MUST be enabled. (Honoring EU TDM opt-out signals is a GOAL.md §2 commitment; flag any domain advertising a TDM reservation as captcha-denylisted.) **NEEDS LEGAL COUNSEL** on TDM opt-out scope.
- **C-10 (MUST) Cost-cap pre-check wired to the request budget too.** Per-URL cost budget MUST be checked in addition to the session cap, so a single page cannot consume the whole session budget in repeated solves (defense-in-depth with C-7's single-attempt rule).

### 3.3 Open gaps this gate surfaces (the conditions above close them; listed so the wiring task is unambiguous)

| Gap | Finding | Control cited | Risk |
|---|---|---|---|
| **G-1** | No code asserts the target is unauthenticated before a heavy/solve rung; "public+unauthenticated only" is prose in GOAL.md §2, not an enforced precondition. | GOAL.md §2 (hard default); CFAA fact-pattern (Power Ventures) | **Critical** — this is the line whose breach turns low-risk public scraping into the strongest unauthorized-access posture. Close via C-2/§4. |
| **G-2** | No per-solve audit event (domain/kind/outcome/timestamp) exists; only the cost meter logs USD. | GOAL.md §2 ("audit log"); logging-domain AUDIT TRAIL | **High** — without it there is no evidence trail for which domains were solved, which a regulator/auditor would expect. Close via C-5. |
| **G-3** | `permanent_denylist` only auto-trips on post-R4 JS block; a **CAPTCHA give-up** does not denylist the domain, so R5 could be re-attempted on the same domain on a later run. | GOAL.md §2 ("treat CAPTCHA as opt-out", "never rotate back in") | **High** — directly weakens the opt-out reconciliation in §1. Close via C-3. |
| **G-4** | No PII detection/suppression or raw-retention guard on R5-gated content. | GDPR Arts. 5(1)(c) minimization, 5(1)(e) storage limitation, 17 erasure; GOAL.md §2 ("no PII retention by default") | **High** | Close via C-6; lawful-basis question is **NEEDS LEGAL COUNSEL** (E-4). |

---

## 4. Permanently off-limits — regardless of any config, flag, or per-run opt-in

These are **hard prohibitions**. No combination of `enable_captcha_solver`, per-run opt-in, or operator config may enable them. The wiring MUST refuse (typed error), not attempt, when any of these is detected:

1. **CAPTCHA that gates a login / authentication wall.** If the CAPTCHA sits on a sign-in, registration, password-reset, MFA, or any flow that establishes an authenticated session — **never solve.** This is the "public + unauthenticated only" hard default (GOAL.md §2) and the highest-risk CFAA surface (C-1/E-1).
2. **Any target requiring credentials to reach.** Requests carrying cookies/tokens/`Authorization`/basic-auth, or reached by submitting credentials — **never solve** (C-2).
3. **Domains on the permanent denylist or that issued a block/C&D.** Once hard-blocked, never re-attempted by any rung including R5 (GOAL.md §2 "never rotate IPs back in"; *Power Ventures*, *3taps*). The governor's `DeniedPermanent` is terminal.
4. **CAPTCHAs gating known-PII / regulated endpoints** — account dashboards, health-record portals (HIPAA §164.312 access controls), payment/checkout flows (PCI DSS — never automate access to a cardholder-data surface), government identity services. Even if technically public, treat as off-limits absent explicit counsel sign-off. **NEEDS LEGAL COUNSEL** before any exception.
5. **Solving on behalf of account-takeover / fraud / spam flows** — categorically prohibited and a near-universal solver-provider AUP violation (E-3).
6. **Caching, logging, or persisting the solved token or the provider API key** — single-use secret-equivalents; never written to cache, logs, `FetchResult`, or the denylist file (already structurally enforced; MUST remain true).

---

## 5. Summary

- **Decision:** **GO-WITH-CONDITIONS.** The built R5 module is compliance-clean in its OFF state and in the secret/cost dimensions; the live wiring is authorized **only** after the §3.2 conditions are implemented and tested, and subject to the §4 permanent prohibitions and the §1/§2 legal-counsel items.
- **Central-question resolution:** CAPTCHA solving is permissible only in a narrow band — public, unauthenticated, single opt-in attempt, denylist-respecting. "Treat CAPTCHA as opt-out" is honored by (a) never solving auth-gating CAPTCHAs and (b) permanently denylisting any hard-blocked domain. Outside that band, not authorized.
- **Frameworks/exposure assessed:** CFAA-style circumvention (E-1), target ToS (E-2), solver-provider AUP (E-3), GDPR/PII incl. controller status + DSM TDM (E-4), cross-border transfer (E-5), secret handling (E-6 clean), cost (E-7 clean).
- **Findings by severity:** 1 Critical (G-1 unauthenticated-only not enforced), 3 High (G-2 audit log, G-3 denylist on CAPTCHA give-up, G-4 PII non-retention), plus 5 **NEEDS LEGAL COUNSEL** items (C-1 single-solve legality, E-2 ToS/C&D, E-3 provider AUP, E-4 lawful basis, E-5 transfer/DPA, C-9 TDM scope).
- **Top blocking items before the flag may be enabled in production:** G-1/C-2 (enforce unauthenticated-only) and the §4 auth-wall prohibition — these are non-negotiable; then G-2/C-5 (audit log), G-3/C-3 (denylist on give-up), G-4/C-6 (no PII retention). Legal-counsel items C-1/E-3/E-4 must be cleared by the operator's counsel before any commercial/production run.

### Condition checklist (implementation hand-off to backend-engineer, task 5.2 → live wiring)

- [ ] **K-1..K-5** verify default-OFF, cost cap reserve-before-call, env-only key, solve-only-on-`Captcha`-verdict, denylist-checked-first — do not regress.
- [ ] **C-1** per-run opt-in (`allow_captcha_solve`, default false), separate from the config flag.
- [ ] **C-2** enforce public + unauthenticated only; refuse R5 if any credentials/session present (closes G-1).
- [ ] **C-3** record domain on R5 give-up so it is never re-solved; support operator C&D denylist entries (closes G-3).
- [ ] **C-4** consult permanent denylist + operator captcha-denylist before solving; default-deny on match.
- [ ] **C-5** structured per-solve audit event {domain, kind, provider, outcome, cost, cumulative, timestamp}; no token/key/query-string/PII (closes G-2).
- [ ] **C-6** no raw PII retention from R5-gated pages; persist transformed derivative only (closes G-4).
- [ ] **C-7** single bounded attempt per verdict; v3 low-confidence; no solve loops.
- [ ] **C-8** startup WARN naming provider + AUP-acceptance obligation.
- [ ] **C-9** WARN on `respect_robots_txt=false` + solver-on; flag TDM-opt-out domains as denylisted.
- [ ] **C-10** per-URL cost budget pre-check in addition to session cap.
- [ ] **§4** hard-refuse: auth-wall CAPTCHAs, credentialed targets, denylisted/C&D'd domains, known-PII/regulated endpoints, ATO/fraud flows, token/key persistence.
- [ ] **Legal (operator):** clear C-1, E-2, E-3, E-4, E-5, C-9 with counsel before enabling in production.
