# CLAUDE.md — JARVIS Project Operating Rules

These rules OVERRIDE default behavior. Follow them exactly.

---

## Rule 1 — Ground Research Before Anything

Every decision, recommendation, and implementation MUST be grounded in verified
research before it is acted on. No guessing, no "it should work," no assuming an
API/library/pattern behaves a certain way.

- Web-search to verify APIs, library versions, model behavior, and patterns
  **before** writing code that depends on them.
- Read the source / official docs / repo when integrating anything external.
- Cite the ground truth (doc link, repo file, command output, test result) that
  backs the decision.
- If research cannot confirm a claim, say so explicitly and stop — do not
  proceed on an unverified assumption.
- Verify before recommend: never swap an agreed-upon decision without research +
  asking the user first.

This applies to the main agent **and** every subagent. A department subagent's
point of view is only valid when backed by ground truth, not opinion.

---

## Rule 2 — Subagents Work As A Team By Department

All subagents operate as one team. Each gives its point of view, **backed by
ground truth**, for the department it owns.

### How it works

1. **Main agent triages.** For every user query, the main agent first decides
   which department(s) the work belongs to.
2. **Main agent assigns.** It routes the work to the matching department
   subagent(s) to execute — it does not do specialist work itself when a
   department owns it.
3. **Subagent executes + reports POV.** The subagent does the work and returns
   its findings/decision with the ground-truth evidence behind it.
4. **Cross-department work** (a query spanning many departments) goes to
   `team-orchestrator` first to produce the ordered chain of specialists and the
   gates between them; the main agent then runs each specialist in turn and feeds
   results forward. (Subagents cannot spawn subagents — main agent drives the chain.)
5. **Main agent synthesizes** the department POVs into the final answer for the user.

### Department → Subagent routing map

| Department / concern | Subagent |
|---|---|
| Orchestration of multi-department work | `team-orchestrator` |
| Architecture, service boundaries, ADRs, build-vs-buy | `solution-architect` |
| Hard algorithm/data-structure design, perf/scale-critical components, research-to-production, "is there a smarter way" — formalize→survey prior art→design→benchmark | `research-solution-architect` |
| Server-side services, APIs, business logic, queues | `backend-engineer` |
| Web UI (React/TS), accessibility, state, perf | `frontend-engineer` |
| End-to-end vertical slices (API→UI), MVPs, prototypes | `fullstack-engineer` |
| Shared UI components, design tokens, primitives | `design-system-engineer` |
| User flows, IA, prototypes, unhappy-path UX | `ux-flow-designer` |
| Cloud architecture, k8s, multi-region, cost | `cloud-engineer` |
| CI/CD, IaC, container build/deploy, env parity | `devops-engineer` |
| Reliability, SLOs, observability, incident response | `site-reliability-engineer` |
| Adversarial security review (read-only) | `security-auditor` |
| Security hardening, threat modeling, fixes | `security-engineer` |
| Regulatory/compliance gap review (read-only) | `compliance-officer` |
| Functional/exploratory/release QA testing | `qa-engineer` |
| Automated test suites (unit/integration/e2e/load) | `testing-engineer` |
| Computer-vision systems and pipelines | `computer-vision-engineer` |
| Productionizing models, training pipelines, eval gates, serving | `ml-engineer` |
| Data pipelines, ETL, schema/quality validation, leakage checks | `data-engineer` |
| Experiment design, metrics, statistical analysis, A/B readouts | `data-scientist` |
| Sourced research, prior-art, feasibility, dossiers | `research-agent` |
| Understanding legacy/undocumented code, reverse-eng | `reverse-engineering-agent` |
| Delivery planning, milestones, dependencies, risk | `project-manager` |
| Release gates, changelog, versioning, go/no-go | `release-manager` |

When a query does not clearly map to a department, the main agent picks the
closest-fit department and states why, or asks the user to disambiguate.

---

## Rule 3 — Subagents Always Run In Background

Always dispatch subagents with background execution (`run_in_background`) so the
main agent is NEVER blocked waiting on one. Launch the subagent, then keep doing
productive work (other tasks, status reporting, monitoring) — act on each
subagent's result when its completion notification fires. Never sit idle waiting
for a subagent to finish.

## Rule 4 — Software Development Lifecycle (SDLC) Runs Every Conversation

Every piece of work — feature, fix, refactor, or change — MUST move through the
software development lifecycle below. The lifecycle is **always on**: the
relevant lifecycle-engineer departments are engaged for every conversation, not
just large ones. Small work runs a lightweight pass through the same phases; it
never skips them.

### The lifecycle (the loop)

Work flows through these phases in order, and **loops** — the cycle repeats until
the work meets its acceptance criteria. A failed gate sends the work back to an
earlier phase, not forward.

1. **Requirements & triage** — clarify what is being asked and the acceptance
   criteria. Main agent decides which department(s) own the work (Rule 2).
2. **Research & ground truth** — the owning engineer gathers verified ground
   truth (docs, source, repo, command/test output) BEFORE any design or code.
   This is mandatory per Rule 1: no design or implementation begins until the
   engineer can cite the ground source backing it.
3. **Design & architecture** — `solution-architect` (and
   `research-solution-architect` for hard/perf-critical problems) produces the
   design/ADR, every claim backed by the ground truth from phase 2.
4. **Architecture critique (gate)** — `testing-engineer` (with `qa-engineer` for
   functional risk, `security-auditor` for adversarial risk) **critiques the
   architecture using facts** — not opinion. The critique must cite evidence
   (spec, benchmark, prior incident, failing case, source) for every objection.
   The design does not proceed to implementation until this critique is answered.
5. **Implementation** — the owning engineer department builds the solution.
   **Every implementing engineer MUST have a ground source before writing code**
   (Rule 1): the API/library/pattern behavior must be verified, cited, and
   recorded. No implementation on assumption.
6. **Test & verify** — `testing-engineer` writes/extends automated suites
   (unit/integration/e2e/load); `qa-engineer` runs functional/exploratory/edge
   testing against the acceptance criteria. Tests must run and show output —
   "it should work" is not acceptance.
7. **Review & gates** — code review, `security-engineer`/`security-auditor`
   sign-off, `release-manager` go/no-go where a release is involved.
8. **Release & observe** — ship through the release gates, confirm monitoring,
   keep a rehearsed rollback. Findings feed back into phase 1 → the loop closes.

### Two hard requirements inside the loop

- **Testing engineers critique the architecture with facts.** The
  `testing-engineer` (and the QA/security review departments) act as an
  adversarial, evidence-backed check on the design at phase 4 AND on the built
  solution at phase 6. Every critique cites ground truth — a spec, a benchmark, a
  failing test, a prior incident, or source. No "I think" objections; no praise
  padding.
- **Building engineers must hold ground source before implementing.** Any
  engineer department that develops a solution (backend, frontend, fullstack,
  ml, data, cloud, devops, design-system, etc.) MUST have verified the relevant
  ground source — official docs, source code, repo, or confirmed test output —
  BEFORE implementation begins, and MUST cite it. This is Rule 1 applied at the
  implementation phase: research-then-build, never build-then-hope.

The loop is the default operating mode for the project. Engage it every
conversation; size the rigor to the task, but never drop a phase.


---

## Reminder

- Research before code. Verify before recommend. Test before ship.
- Never commit or push without asking.
- Ground truth over opinion — for every department, every time.


## goal
- Desined web search mcp server that can scrap any website with and without any blockers. It must have high accuracy and cover 90%+ resources.