# Research 0002 — Legal Risks of Aggressive Scraping & Concrete Mitigations

> Owner: research agent. Compiled June 2026 for the operator of this MCP scraping tool.
> Every load-bearing claim carries a URL. Established law vs. contested/unsettled law are
> labeled. Backs the "scrape ANY website" goal with a grounded risk map.

## ⚠️ NOT LEGAL ADVICE

I am not a lawyer and this is not legal advice. This is a sourced landscape map so the
operator can make informed choices. **Engage qualified counsel (ideally with internet/data
law experience in your and your targets' jurisdictions) before running this tool in
production against third-party sites.** Outcomes are fact-specific and the law is unsettled
in key areas flagged below.

---

## Question & Decision Context

What are the concrete legal risks of aggressive web scraping (US + EU focus), and what
specific, implementable mitigations reduce each risk and by how much? This informs design
defaults and operating policy for an authorized scraping tool intended to target arbitrary
sites.

---

## Summary of Findings (the answer up front)

1. **CFAA (US criminal/civil "hacking" statute) is now a weak weapon against scraping of
   genuinely public, unauthenticated pages — but a strong one once access is revoked or a
   technical barrier is circumvented.** After *Van Buren* (2021) and *hiQ* (2022), scraping
   public data is "unlikely" to be unauthorized access. But a cease-and-desist (C&D) plus IP
   block, then continued access via IP rotation, has repeatedly been held to be CFAA
   violation (*Power Ventures*, *3taps*). **Confidence: High. Recency: 2021–2022 binding;
   stable.**

2. **The real, live litigation risk for public-data scraping is breach of contract (ToS),
   not CFAA.** hiQ won the CFAA fight but *lost* on breach of LinkedIn's User Agreement and
   settled with a permanent injunction to stop and delete everything. **Confidence: High.
   Recency: late 2022.**

3. **Privacy law (GDPR/CCPA) applies to scraped *personal* data even when it is public.**
   "It was public" is not a defense under GDPR. Regulators have fined scrapers of public
   data heavily (Clearview ~€100M across EU; CNIL fined KASPR €240k in Dec 2024 for scraping
   LinkedIn contacts). **Confidence: High. Recency: 2024.**

4. **Copyright + EU database rights bite when you copy/redistribute substantial protected
   content.** US fair use is narrowing for commercial verbatim copying (*Thomson Reuters v.
   Ross*, 2025). The EU grants sui generis database rights on top of copyright.
   **Confidence: Medium–High. Recency: 2024–2025, fast-moving (AI-training cases).**

5. **IP rotation to evade a block is the single biggest legal-exposure amplifier in this
   tool.** It converts a weak ToS dispute into a plausible CFAA "unauthorized access" claim,
   because circumventing a technical barrier (especially after a C&D) is exactly the trigger
   courts have used. **Confidence: High for the post-C&D/post-block case; Medium for
   rotation with no prior revocation.**

6. **Lowest-risk posture exists and is implementable:** public/unauthenticated only, no
   login walls, drop PII at ingest, polite rate limits, honor robots/opt-out/C&D, store
   transformed derivatives + attribution rather than raw copies, retention limits, and a
   fast takedown + per-domain blocklist process. Details in the matrix and final section.

---

## Detailed Findings

### 1. US — Computer Fraud and Abuse Act (CFAA)

**Established fact:**
- *Van Buren v. United States* (SCOTUS, June 3, 2021): the CFAA's "exceeds authorized
  access" clause uses a narrow "gates-up-or-down" rule — you violate it only by accessing
  areas of a system that are **off-limits to you**, not by accessing data you may access but
  for a disallowed purpose or in violation of a use policy. Opinion:
  https://www.supremecourt.gov/opinions/20pdf/19-783_k53l.pdf · analysis (Proskauer):
  https://newmedialaw.proskauer.com/2021/06/06/supreme-court-ends-long-running-circuit-split-over-cfaa-exceeds-authorized-access-issue-adopting-a-narrow-interpretation-that-will-reverberate-in-scraping-disputes-and-litigation-ov/
- *hiQ Labs v. LinkedIn* (9th Cir., Apr 18, 2022, on remand): when a network **generally
  permits public access**, accessing that public data is **unlikely to be "without
  authorization"** under the CFAA. Opinion:
  https://cdn.ca9.uscourts.gov/datastore/opinions/2022/04/18/17-16783.pdf · summary (Practical
  Law): https://uk.practicallaw.thomsonreuters.com/w-035-2799
- **Revocation/technical-barrier trigger:** *Facebook v. Power Ventures* (9th Cir.) — after a
  C&D and IP blocking, continued access by **changing IP addresses** was a CFAA violation;
  access "without authorization" occurs when permission is **explicitly revoked**. EFF case
  page: https://www.eff.org/cases/facebook-v-power-ventures · Finnegan analysis:
  https://www.finnegan.com/en/insights/articles/the-computer-fraud-and-abuse-act-and-third-party-web-scrapers.html
- *Craigslist v. 3taps* (N.D. Cal.): after a C&D and IP block, using **proxy servers and
  rotating IPs** to bypass the block was unauthorized access under the CFAA. Same Finnegan/
  Bloomberg Law analyses above; IAPP on C&D-creating-liability:
  https://iapp.org/news/a/can-a-cease-and-desist-notice-create-cfaa-liability-scrapers-beware

**Contested / unsettled:**
- *Van Buren* expressly left open whether authorization can be limited **only** by
  technological barriers (passwords) or **also** by contract/use policies. Courts still
  split on whether ToS-only violations can ever ground CFAA. (Van Buren opinion, fn. 8;
  CRS analysis: https://www.congress.gov/crs-product/LSB10616)
- *hiQ* is 9th Circuit law; other circuits are not bound. Risk varies by where suit is filed.

**Takeaway:** Scraping public, unauthenticated pages with no prior C&D/block ≈ low CFAA
risk. Scraping authenticated/login-walled content, or continuing after a C&D or block
(especially by rotating IPs), ≈ real CFAA risk.

### 2. Contract (ToS), robots.txt, Trespass to Chattels

**Established fact:**
- **Clickwrap** agreements (affirmative "I agree") are reliably enforced. **Browsewrap**
  (terms merely posted, no click) is weaker but **enforceable against sophisticated
  commercial parties** with actual or constructive knowledge of the terms. Quinn Emanuel
  landscape: https://www.quinnemanuel.com/the-firm/publications/the-legal-landscape-of-web-scraping/
  · Proskauer: https://newmedialaw.proskauer.com/2019/07/09/web-scraping-decisions-consider-contract-cause-of-action/
- **Breach of contract is the live winner.** hiQ ultimately **lost** on LinkedIn's User
  Agreement and (Dec 2022) settled into a permanent injunction: stop scraping, delete all
  data/code/algorithms. Zwillgen lessons:
  https://www.zwillgen.com/alternative-data/hiq-v-linkedin-wrapped-up-web-scraping-lessons-learned/
  · Privacy World: https://www.privacyworld.blog/2022/12/linkedins-data-scraping-battle-with-hiq-labs-ends-with-proposed-judgment/
- **Trespass to chattels** requires **actual harm/impairment** to the target's servers
  (bandwidth/performance degradation), not mere access. LegalClarity:
  https://legalclarity.org/is-web-scraping-legal-a-look-at-the-law/

**Contested / unsettled:**
- **robots.txt has no statute behind it.** Ignoring it is **not itself a cause of action**,
  but it is evidence of "knowing/unauthorized" conduct and is read into the ToS/legitimate-
  expectations analysis (and is explicitly named by CNIL as an opt-out signal — see §3).
- **Copyright preemption of contract claims:** a 2024 N.D. Cal. decision took a **broad view
  that copyright can preempt** a breach-of-contract scraping claim where the contract right
  is equivalent to copyright — cutting the other way for site owners. Skadden:
  https://www.skadden.com/insights/publications/2024/05/district-court-adopts-broad-view
  (This is unsettled and not nationally binding.)

**Takeaway:** Contract is the most reliable claim a target can bring for public data.
Accepting a clickwrap (i.e., creating an account/logging in) dramatically strengthens it.

### 3. Data Protection — GDPR & CCPA/CPRA

**Established fact (GDPR):**
- GDPR applies to **personal data regardless of public availability**. "It was public" is
  not a lawful basis. CNIL focus sheet (primary regulator guidance):
  https://www.cnil.fr/en/legal-basis-legitimate-interest-focus-sheet-measures-implement-case-data-collection-web-scraping
- **Legitimate interest** (Art. 6(1)(f)) is the usual basis for commercial scraping but
  requires a documented **balancing test** vs. data-subject rights and **reasonable
  expectations**. (CNIL, above.)
- **CNIL's required/recommended measures** (load-bearing for mitigations below): define
  collection criteria in advance; filter out unnecessary categories; **exclude sites that
  structurally contain sensitive data** (health, minors, etc.); delete irrelevant data
  immediately; **respect robots.txt and CAPTCHA as opposition signals**; offer a **prior
  opt-out**; **pseudonymize/anonymize immediately**; use per-item random pseudonyms to
  prevent re-identification; be transparent (publish scraped-site lists). (CNIL, above.)
- **Enforcement is real against public-data scraping:** CNIL fined **KASPR €240,000 (Dec
  2024)** for scraping LinkedIn contact details — including data users had restricted.
  (Reported via the GDPR/CCPA compliance summary:
  https://iswebscrapinglegal.com/blog/gdpr-ccpa-web-scraping/) Clearview AI fined ~**€100M
  total** across the EU (Netherlands **€30.5M, Sept 2024**; France/Italy/Greece €20M each;
  UK £7.5M) for scraping public photos. TechCrunch:
  https://techcrunch.com/2024/09/03/clearview-ai-hit-with-its-largest-gdpr-fine-yet-as-dutch-regulator-considers-holding-execs-personally-liable/
  · Hacker News: https://thehackernews.com/2024/09/clearview-ai-faces-305m-fine-for.html

**Established fact (CCPA/CPRA):**
- The CPRA **expanded** the "publicly available" exclusion to cover info a consumer
  **lawfully made available to the general public** (incl. social media not restricted to a
  specific audience) and widely distributed media. Such info is **exempt from CCPA rights
  (deletion, opt-out)**. Byte Back: https://www.bytebacklaw.com/2022/01/how-do-the-cpra-cpa-vcdpa-treat-publicly-available-information/
  · CA AG: https://oag.ca.gov/privacy/ccpa
- But the exclusion turns on a **reasonable belief the consumer made it public and did not
  restrict it** — restricted/inferred/sensitive data is **not** covered. (Same sources.)

**Takeaway:** GDPR is the highest-severity exposure for personal data and is **not** cured
by publicness. CCPA is more forgiving for truly-public data but not for restricted/sensitive
data. The cleanest mitigation is to **not retain personal data at all** where the use case
allows.

### 4. Intellectual Property — Copyright, EU Database Rights, Hot News

**Established fact:**
- Scraped text/images can be **copyrighted**; copying + redistribution of protected
  expression infringes. Verbatim commercial copying is **disfavored under fair use** —
  *Thomson Reuters v. Ross* (2025) rejected a fair-use defense for copying Westlaw headnotes
  to build a competing commercial product. Wolf Greenfield:
  https://wolfgreenfield.com/articles/fair-use-in-ai-copyright-litigation-a-surprising-turn-in-thomson-reuters-v.-ross
  · TechCrunch: https://techcrunch.com/2025/02/17/what-the-us-first-major-ai-copyright-ruling-might-mean-for-ip-law/
- **EU sui generis database rights** (Database Directive 96/9/EC) protect databases
  representing substantial investment; extracting/re-utilizing a **substantial part** can
  infringe — independent of copyright. Recognized in EU, plus Mexico, South Korea.
  GroupBWT overview: https://groupbwt.com/blog/is-web-scraping-legal/
- **EU text-and-data-mining (TDM) exceptions** (DSM Directive 2019/790, Arts. 3–4): research/
  non-commercial mining is broadly permitted; commercial mining is permitted **only if the
  rightsholder has not opted out** (machine-readable reservation). A German court let LAION
  rely on the TDM research exception. Morrison Foerster:
  https://www.mofo.com/resources/insights/241004-to-scrape-or-not-to-scrape-first-court-decision

**Contested / unsettled:**
- **Fair use for AI training** generally is in active litigation; *Ross* was narrow (not a
  generative model) and is not the last word.
- **"Hot news" misappropriation** (US, *NBA v. Motorola*) is a narrow, rarely-successful
  state-law doctrine for time-sensitive factual data (scores, prices). Low but nonzero risk
  for real-time data products. (Quinn Emanuel landscape, above.)

**Takeaway:** Storing/serving **raw copies** of protected content is the risky act. Serving
**transformed/derived** data with attribution, and honoring TDM opt-outs, sharply lowers IP
risk.

### 5. Jurisdiction

**Established fact / well-supported:**
- **CFAA is US federal**, but circuits differ; *hiQ* protection is strongest in the 9th
  Circuit. Forum matters. (Sources in §1.)
- **GDPR has extraterritorial reach** (Art. 3): it applies if you target/monitor EU data
  subjects regardless of where the operator sits — Clearview (a US company) was fined and is
  now facing criminal complaints in the EU. The Register:
  https://www.theregister.com/2025/10/28/noyb_criminal_charges_clearview/
- **Database rights** exist in the EU/UK/Mexico/Korea but **not** as a standalone right in
  the US. (GroupBWT, above.)
- Three locations independently drive risk: **operator location** (who can be sued/where
  judgments enforce), **target/server location** (which ToS/IP/database law applies), and
  **data-subject location** (which privacy law applies, via GDPR/CCPA extraterritoriality).

**Takeaway:** You cannot "proxy your way out" of jurisdiction. Choosing proxy exit nodes to
appear local to a target does **not** reduce GDPR/contract exposure and (per §6) may *add*
CFAA/circumvention exposure.

### 6. IP Rotation Specifically (operator's question)

**Established fact:** Rotating IPs to **evade an IP-based block** is the fact pattern that
courts have repeatedly used to find CFAA "unauthorized access":
- *Facebook v. Power Ventures*: changing IPs to circumvent Facebook's IP screening after a
  C&D = CFAA violation. https://www.eff.org/cases/facebook-v-power-ventures
- *Craigslist v. 3taps*: proxies + rotating IPs to bypass Craigslist's block after a C&D =
  unauthorized access. (Finnegan/Bloomberg Law analyses, §1.)
- Circumventing a technical access barrier is treated as evidence that access was **not
  authorized** — it is the affirmative act, more than the scraping itself, that creates
  liability. Help Net Security (older but on-point):
  https://www.helpnetsecurity.com/2013/08/20/is-evading-an-ip-address-block-to-access-a-website-against-the-law/

**Contested / unsettled:** Whether **rotating IPs with no prior C&D and no specific block of
you** (e.g., generic distribution to avoid rate-based throttling) is "circumventing a
barrier." There is a meaningful legal difference between (a) load-distribution before any
revocation and (b) deliberately defeating a block aimed at you. (b) is clearly risky; (a) is
grayer and less litigated.

**Bottom line on rotation:** It **increases** legal exposure whenever it is used to defeat a
block or continue after a C&D. The safest rule the tool can encode: **a block or C&D is a
hard stop for that domain — never rotate to get back in.**

---

## Risk → Mitigation Matrix

Strength = how much the mitigation reduces that specific risk. ✅ strong · 🟡 partial ·
➖ minimal/indirect.

| Mitigation (implement in tool/ops) | CFAA | Contract/ToS | GDPR/CCPA | Copyright/DB rights | Strength notes |
|---|---|---|---|---|---|
| **Public/unauthenticated pages only; never log in** | ✅ | ✅ | 🟡 | ➖ | Keeps you in *hiQ* "public access" safe harbor; avoids accepting clickwrap. Strongest single control. |
| **Avoid login-walled / paywalled / restricted content** | ✅ | ✅ | ✅ | 🟡 | Restricted = "off-limits" (CFAA) + accepted ToS + not "publicly available" (CCPA) + not reasonable-expectation (GDPR). |
| **Drop/strip PII at ingest (don't store personal data)** | ➖ | ➖ | ✅ | ➖ | Removes the highest-severity exposure (GDPR fines). If no use case needs it, this is the best GDPR control. |
| **Data minimization + sensitive-site exclusion list** | ➖ | ➖ | ✅ | ➖ | Directly tracks CNIL's required measures (health/minors/genealogy/etc.). |
| **Honor robots.txt + CAPTCHA as opt-out signals** | 🟡 | 🟡 | ✅ | ➖ | Not legally required in US per se, but CNIL treats robots/CAPTCHA as opposition signals; also reduces "knowing/unauthorized" evidence. |
| **Polite rate limits / no server impairment** | 🟡 | 🟡 | ➖ | ➖ | Defeats trespass-to-chattels (needs actual harm) and reduces "aggressive/unauthorized" framing. |
| **Honor opt-out / DNT / provide prior objection** | ➖ | ➖ | ✅ | ➖ | CNIL explicitly recommends a prior opt-out mechanism for legitimate-interest scraping. |
| **Hard stop on C&D or IP block per domain (NEVER rotate back in)** | ✅ | ✅ | 🟡 | ➖ | Avoids the *Power Ventures*/*3taps* trigger — the biggest CFAA risk reducer for this tool. |
| **Do NOT rotate IPs to defeat a block** | ✅ | ➖ | ➖ | ➖ | Rotation-to-evade is the affirmative act courts penalize. See §6. |
| **Store transformed/derived data + attribution, not raw copies** | ➖ | 🟡 | 🟡 | ✅ | Lowers copyright/DB-right exposure; transformation aids (does not guarantee) fair use. |
| **Honor EU TDM machine-readable opt-out reservations** | ➖ | ➖ | ➖ | ✅ | DSM Art. 4 commercial TDM is lawful only absent rightsholder reservation. |
| **Retention limits / auto-delete** | ➖ | 🟡 | ✅ | 🟡 | GDPR storage-limitation; also limits damages exposure across the board. |
| **Fast takedown-response process + audit log** | 🟡 | ✅ | ✅ | ✅ | Mitigates damages/injunction risk everywhere; demonstrates good faith; handles GDPR erasure + DMCA. |
| **Pseudonymize immediately; per-item random IDs** | ➖ | ➖ | ✅ | ➖ | CNIL anti-re-identification measure. |
| **Geo-aware proxy exit nodes** | ⚠️ | ➖ | ➖ | ➖ | Does NOT reduce risk and can *increase* it if used to evade blocks (§6) or to obscure circumvention. Not a mitigation. |

⚠️ = listed because the operator raised it: choosing proxy exit-node jurisdiction is **not**
a legal mitigation. It does not defeat GDPR/contract extraterritoriality and, when used to
appear unblocked, aligns with the circumvention fact pattern courts penalize.

---

## Conflicts & Caveats

- **CFAA circuit split persists.** *hiQ* binds the 9th Circuit; other forums may treat ToS
  or technical barriers more aggressively. *Van Buren* deliberately left the contract-vs-
  technical-barrier question open. (§1)
- **Contract vs. copyright preemption pulls both ways.** Some 2024 authority lets copyright
  *preempt* a site's contract claim (helps scrapers); most authority still enforces ToS
  contracts (hurts scrapers). Unsettled. (§2)
- **AI-training fair use is in flux.** *Ross* is narrow; broader generative-AI cases are
  pending and could move the line either way. (§4)
- **Secondary sources used for two specifics** (KASPR fine amount; some CCPA framing) —
  the KASPR €240k figure traces to CNIL's own 2024 actions but I cited it via a compliance
  summary; treat the *fact of enforcement* as High confidence and the *exact figure* as
  Medium until confirmed against CNIL's decision register.
- **No advice on UK DPA, Canada PIPEDA, Brazil LGPD, or sector rules** (HIPAA, FCRA) — out
  of scope here but relevant if you scrape health/credit data or target those regions.

---

## Open Questions (what further research/counsel would resolve)

1. In which **jurisdictions/forums** will this tool's operator and primary targets sit?
   That determines whether *hiQ* protection applies and whether database rights exist.
2. Will any use case **retain personal data**? If yes, a documented GDPR legitimate-interest
   balancing test (LIA/DPIA) is needed; if no, most GDPR risk evaporates.
3. Will outputs be **redistributed/commercialized** vs. internal analysis only? Drives the
   copyright/fair-use and database-right analysis.
4. Exact text/figure of the **CNIL KASPR decision** (confirm against CNIL register).
5. Does the product touch **real-time time-sensitive data** (prices, scores)? If so, assess
   hot-news misappropriation with counsel.

---

## Lowest-Risk Operating Posture (recommended default)

Encode these as the tool's defaults; require explicit, logged operator override to deviate:

1. **Public + unauthenticated only.** No logging in, no clickwrap acceptance, no paywall/
   login-wall bypass. (Biggest CFAA + contract reducer.)
2. **Treat C&D or IP block as a permanent per-domain hard stop.** Never rotate IPs, change
   identities, or otherwise circumvent to regain access. (Kills the *Power Ventures*/*3taps*
   trigger — the dominant CFAA risk for this tool.)
3. **Default to no PII retention.** Strip personal data at ingest; if a use case needs it,
   require a documented legitimate-interest assessment, sensitive-site exclusion list,
   pseudonymization, opt-out, and retention limits (CNIL measures).
4. **Be polite.** Conservative rate limits, respect robots.txt and CAPTCHAs as opt-out
   signals. (Defeats trespass-to-chattels; reduces "aggressive/unauthorized" framing; aligns
   with CNIL.)
5. **Store derived/transformed data + source attribution, not raw copies.** Honor EU TDM
   machine-readable opt-outs. (Lowers copyright/DB-right exposure.)
6. **Build a takedown + erasure pipeline and an audit log** of what was fetched, when, under
   what robots/ToS state. (Limits damages/injunction exposure; demonstrates good faith;
   handles GDPR/DMCA requests.)
7. **Get counsel before targeting any specific high-value site** (LinkedIn-class platforms,
   anything with personal/sensitive data, or any target in an unfamiliar jurisdiction).

This posture does not make scraping "legal everywhere" — nothing can, given unsettled law —
but it keeps the tool inside the most defensible zone the current case law and regulator
guidance describe, and it removes the two highest-amplitude risks (login-wall access and
block-evasion via IP rotation).

---

## Sources

1. *Van Buren v. United States*, SCOTUS opinion (2021) — https://www.supremecourt.gov/opinions/20pdf/19-783_k53l.pdf
2. CRS, "Van Buren v. United States" analysis — https://www.congress.gov/crs-product/LSB10616
3. Proskauer, SCOTUS narrows CFAA "exceeds authorized access" — https://newmedialaw.proskauer.com/2021/06/06/supreme-court-ends-long-running-circuit-split-over-cfaa-exceeds-authorized-access-issue-adopting-a-narrow-interpretation-that-will-reverberate-in-scraping-disputes-and-litigation-ov/
4. *hiQ Labs v. LinkedIn*, 9th Cir. opinion (Apr 18, 2022) — https://cdn.ca9.uscourts.gov/datastore/opinions/2022/04/18/17-16783.pdf
5. Thomson Reuters Practical Law, hiQ on-remand summary — https://uk.practicallaw.thomsonreuters.com/w-035-2799
6. Zwillgen, "hiQ v. LinkedIn Wrapped Up: Web Scraping Lessons" — https://www.zwillgen.com/alternative-data/hiq-v-linkedin-wrapped-up-web-scraping-lessons-learned/
7. Privacy World, hiQ/LinkedIn final judgment (Dec 2022) — https://www.privacyworld.blog/2022/12/linkedins-data-scraping-battle-with-hiq-labs-ends-with-proposed-judgment/
8. EFF, *Facebook v. Power Ventures* case page — https://www.eff.org/cases/facebook-v-power-ventures
9. Finnegan, CFAA and third-party web scrapers — https://www.finnegan.com/en/insights/articles/the-computer-fraud-and-abuse-act-and-third-party-web-scrapers.html
10. IAPP, "Can a cease-and-desist create CFAA liability?" — https://iapp.org/news/a/can-a-cease-and-desist-notice-create-cfaa-liability-scrapers-beware
11. Help Net Security, evading IP blocks and the law — https://www.helpnetsecurity.com/2013/08/20/is-evading-an-ip-address-block-to-access-a-website-against-the-law/
12. Quinn Emanuel, "The Legal Landscape of Web Scraping" — https://www.quinnemanuel.com/the-firm/publications/the-legal-landscape-of-web-scraping/
13. Proskauer, web scraping & contract causes of action — https://newmedialaw.proskauer.com/2019/07/09/web-scraping-decisions-consider-contract-cause-of-action/
14. LegalClarity, "Is Web Scraping Legal?" (trespass to chattels) — https://legalclarity.org/is-web-scraping-legal-a-look-at-the-law/
15. Skadden, broad view of copyright preemption in scraping (2024) — https://www.skadden.com/insights/publications/2024/05/district-court-adopts-broad-view
16. CNIL, legitimate-interest focus sheet on web scraping measures — https://www.cnil.fr/en/legal-basis-legitimate-interest-focus-sheet-measures-implement-case-data-collection-web-scraping
17. iswebscrapinglegal, GDPR/CCPA web scraping guide (KASPR fine) — https://iswebscrapinglegal.com/blog/gdpr-ccpa-web-scraping/
18. TechCrunch, Clearview AI €30.5M Dutch fine (Sept 2024) — https://techcrunch.com/2024/09/03/clearview-ai-hit-with-its-largest-gdpr-fine-yet-as-dutch-regulator-considers-holding-execs-personally-liable/
19. The Hacker News, Clearview AI €30.5M fine — https://thehackernews.com/2024/09/clearview-ai-faces-305m-fine-for.html
20. The Register, criminal complaint vs Clearview AI (Oct 2025) — https://www.theregister.com/2025/10/28/noyb_criminal_charges_clearview/
21. Byte Back, CPRA/CCPA "publicly available" treatment — https://www.bytebacklaw.com/2022/01/how-do-the-cpra-cpa-vcdpa-treat-publicly-available-information/
22. California AG, CCPA overview — https://oag.ca.gov/privacy/ccpa
23. Wolf Greenfield, *Thomson Reuters v. Ross* fair-use analysis — https://wolfgreenfield.com/articles/fair-use-in-ai-copyright-litigation-a-surprising-turn-in-thomson-reuters-v.-ross
24. TechCrunch, first major US AI copyright ruling (Ross) — https://techcrunch.com/2025/02/17/what-the-us-first-major-ai-copyright-ruling-might-mean-for-ip-law/
25. Morrison Foerster, first EU TDM-exception decision (LAION, Germany) — https://www.mofo.com/resources/insights/241004-to-scrape-or-not-to-scrape-first-court-decision
26. GroupBWT, web scraping legal issues / sui generis database rights — https://groupbwt.com/blog/is-web-scraping-legal/
