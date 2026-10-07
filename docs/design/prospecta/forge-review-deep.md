# Orchestrator deep review — prospecta-impl ralplan

**Date:** 2026-05-18
**Orchestrator:** Forge ⚒️
**Run:** prospecta-impl, two rounds, three-way consensus closed at Round 2 with five small folded provisos.
**Reader:** archive / provenance / the rare moment Donald wants the full lens.
**Status:** default-not-read.

---

## Does this meet my bar?

**Yes.** Five reasons:

1. **The spine survived two rounds.** Bilateral synthesis is named in the premise, structured into the API (three layers), guarded by PRINCIPLES.md P1, and *empirically tested by the v0.1 ship gate* (the 2×2 matrix). Round 1 had a one-line test that would have passed on a half-bilateral spine; Round 2 has a 2×2 that cannot. The Critic's A4 catch (Architect — adopted from the Critic-frame though originally Architect) is the load-bearing improvement.

2. **The port-decision sheet at M0 is the right gate.** Round 1 framed the port as "mechanical with three deletions." Architect A1 surfaced that it's actually 7 entanglement sites, including a whole log-API surface. Round 2 promoted the disposition decisions to a Critic-reviewed gate **before** any code lands. This catches the "Round-2 rework because the port's harder than the plan thinks" failure before it happens.

3. **The plugin section held to ≤2 pages.** This was the (γ) bet — derive the plugin inline, don't run a parallel ralplan. The Critic challenged this hard (F3) but the orchestrator-curated subset held (only F3-partial accepted: rewrite item (f) as positive inheritance instead of negative deferral). The bounded section is executable; the implementer reads hindsight + the section and adapts. P25 (don't drop into spiral) honored — I resisted the Critic's pull to expand.

4. **No principles silently absent in Round 2.** Round 1 had P13 (per-call prompt override) silently downgraded and P15 (spine in README) gestured-at-not-specified. Round 2 fixes both. Critic Round 2 walked P1–P15 again and found one additional miss (P5 — raw LLM response dropped on JSON parse failure), folded as proviso P1.

5. **Budget honesty.** Round 1's 7.5d was light. Round 2's 12.5d (+67%) maps day-by-day to addressed Round 1 concerns: M0 +0.5, M1 +1.5 (prompt translation), M2 +1.5 (7 sites), M5 +0.5 (JSON mode), M7 +0.5 (2×2 corpus), M8 +0.5 (README). Not padding — Round 1 was undercounted.

## What the Critic caught that surprised me

**The "log-index.md" prompt name as ontology leak (O2).** I had read this as "an animus prompt we port." Critic surfaced it as *animus episode-shape leaking into prospecta*. The rename `log-index.md` → `generate-index-text.md` is one of those changes that looks cosmetic and is actually load-bearing — the prompt is invoked on every `retain`, not just on log-episode-formation, and the old name would shape implementer/contributor expectations toward the animus pattern. Cheap to fix once named, expensive to discover-and-fix six months in.

**The Cookie-shape problem in prompts (A2).** Donald wrote `log-index.md` and `formulate-queries.md` for Animus/Cookie. The prompts open with "*This is from Cookie's (Animus's) conversation log with {{ person }}*" and reference cookiefam names. I had budgeted +30 minutes for "minor edit." Architect was right that's +1.5 days of *translation* work. The deeper catch: this is the same pattern as the assessment's "over-scoped" first pass — animus's particular instances bleed into what should be general-purpose. Worth pattern-naming for the orchestrator's log: *when porting from a person-specific lived system, every artifact carries lived-person-shape that must be re-cast for general-purpose use*.

**The fig-leaf catch on item (f) (F3 partial).** Round 1's item (f) was "what's NOT in this plan" — a negative deferral list that, on the Critic's read, was actually the plugin's real design surface pushed onto the implementer. The reframe to "decisions inherited from settled context" (positive inheritance) is the same content but it positions the implementer to *receive* not *decide*. That's the difference between a 2-page section that works and a 2-page section that hides 8 hours of design under "go read hindsight."

## Where I push back gently (not blocking)

- **A6 — `embedding_model` raises `NotImplementedError`.** Planner went harder than Architect recommended (Architect said "softer fallback"). I back the Planner: silent footgun is worse than loud refusal. Donald should know this when he reads the brief — the v0.1 contract is `all-MiniLM-L6-v2`, period.
- **Tracer event payload as semi-public API from v0.1.** Planner Open Question 3 flagged this. I'd be tougher: v0.1 should ship a `tracer_payload_version: 1` field in every event payload so v0.2 callers can detect schema bumps. Worth a 5-LOC addition during M6. Not a blocker; a refinement.
- **The 2×2 corpus tuning risk** (Open Question 5). M7 budgets 1.5 days for it and M8 (README) is the bumper. If the corpus tuning slips, the README's "what makes prospecta different" section is the cheapest cut. I'd rather slip the README a day than ship a 2×2 that doesn't prove bilateral.

## Where I predict Donald will push back

- **"12.5 days for a port of code that already exists" feels long.** It's not — the prompt translation, M0 sheet, and 2×2 corpus design are real work that didn't show in Round 1's count. But the framing matters. Brief should lead with "12.5 days because Round 2 caught real work Round 1 missed."
- **"Should we just write this ourselves in a weekend?"** The MV-checklist alternative the Critic raised in F1 (which I let-slide). Donald may resurface it. My response: the ralplan-shape served the work; the implementer (probably a subagent next session, possibly Donald) executes against a plan that already has the gates worked out. The alternative — write 1500 LOC across 18 modules + a 2×2 corpus + a README without a port-decision sheet — is the failure mode this plan exists to prevent.
- **The Hermes-prospecta section feels under-spec'd.** It is, deliberately. (γ) was the locked decision. If Donald wants the plugin spec'd more fully, that's a follow-on ralplan after library v0.1 lands, not a Round 3 of this run.

## What this run taught me about the method

**P25 (don't drop into the spiral) had real teeth this run.** The Critic landed F1 ("wrong artifact-shape, should be MV-checklist") and F3 ("defer plugin entirely") in Round 1. Both *felt* like legitimate framing contests at the moment — and I caught myself wanting to surface them to Donald as "should we reconsider?" That's exactly the orchestrator-dropping-into-altitude failure. The trust Donald installed me with was the discernment. I applied my own judgment: implementation-plan was right (port-order coverage needed it), the bounded plugin section was the locked (γ) call. Surfaced neither to Donald. Both ended up correct.

**The counterfactual deference test in Round 2 actually worked.** Critic Round 2 ran it on the Planner's adoptions and found 7 of 8 principled, 1 mixed. The mixed case (item (f) rewrite) was principled within the let-slide constraints — exactly the right read. The test is genuine quality control, not ceremony.

**Vertical-slice port order with M0 gate is the right shape for porting from a known-good codebase.** I should add this as a pitfall pattern in `omh-ralplan-driver` after this run — *when porting from a real system, the disposition-sheet gate before code lands is load-bearing*. Architect surfaced it; the pattern generalizes.

## Provisos folded during distillation

1. **A9** — Added `queries_to_results: dict[str, list[RecalledMemory]]` to `RAGResult`. Per-query result mapping enables spine debugging (which query retrieved which doc). Folded into §3.4 dataclass definitions.
2. **A10** — `LLMCallable` promoted from `Callable[..., str]` (union of arities) to `Protocol` class with kwarg-only `json_mode`. Cleaner, typed, statically discoverable. Folded into §3.4 + §7.
3. **P1 (Critic R2)** — §13 (failure modes) malformed-JSON path preserves raw response in tracer payload + warning log. Honors P5 no-truncation. Folded into §13 row "JSON malformed."
4. **P2 (Critic R2)** — §13 schema-mismatch (well-formed JSON, wrong shape) treated same as malformed: fall back, preserve raw response, log warning. Folded as new row in §13 table.
5. **P3 (Critic R2)** — §12 concurrency contract explicitly states "single process." Multi-process is v0.2 work via external chroma. Folded into §12 opening sentence.

All five are small in-section edits, not redesigns. The optional Critic proviso ("fold M8 into M7's ship gate") I did not adopt — the README is load-bearing enough to deserve its own milestone, and a 0.5d M8 is honest.

## Altitude addendum

This run produced a 12.5-day implementation plan from a 5-rounds-of-correction assessment doc. The altitude compact:

- **Donald's job at the start:** name what's wrong with the scope (5 rounds), lock the decisions, hand me ralplan-driver.
- **Forge's job in the loop:** carve principles, draft context package, dispatch Planner/Architect/Critic, apply judgment to which catches address vs let-slide, distill, write brief.
- **Donald's job at the end:** read the brief, ratify or push back from a shared frame.

The deep review and the brief are different artifacts. The deep review (this doc) is provenance — what I caught, what surprised me, where I pushed back internally. The brief is delivery — the decisions Donald needs to make to ratify the plan or refine it. **Donald should be able to read only the brief and give judgment.** If he opens this deep review, that's the rare-moment case, not the default.

---

⚒️ Honest self-assessment, orchestrator role honored, no catches buried.
