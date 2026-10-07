# Brief — Prospecta Implementation Plan

**Date:** 2026-05-18
**Author:** Forge ⚒️
**Audience:** Donald, for ratification or push-back.
**Read time:** ~5 minutes.
**Deeper provenance (default-not-read):** `forge-review-deep.md` next door.

---

## TL;DR

Two-round ralplan closed at consensus. **Canonical plan at `~/src/witt3rd/prospecta/docs/design/prospecta/plan.md`.** ~12.5 engineer-days for `prospecta` v0.1, with a bounded 2-page `hermes-prospecta` derivation inside. Nine milestones, each with a runnable demo + test gate. The v0.1 ship gate is a 2×2 bilateral-synthesis integration test that proves both halves of the spine contribute (not just "the spine helps").

No structural surprises. Round 2 grew the budget from 7.5d to 12.5d — that's honest correction for work Round 1 missed (prompt *translation* not copy, port-decision sheet, JSON-mode for read-side spine).

---

## Decisions you need to make

### Decision 1 — Ratify the canonical plan and authorize implementation?

**My take:** Yes, ratify as-is. The plan is design-ready and executable. None of the open items in §14 are blockers — they're risks the M0 gate or M7 bumper handle.

**Alternative:** Push back on the 12.5d budget; request a tighter v0.1 cut (e.g., ship without tracer, without README polish, without 2×2 corpus tuning — call that v0.1-lite and re-budget). Trade-off: 3-4 days saved at the cost of P1 (spine in README is the value-proposition document) and P15 honors getting deferred. I do not recommend this — the budget growth is real work, not padding.

**Recommendation: ratify.**

### Decision 2 — Implementer disposition?

**My take:** Delegate to a subagent (probably Claude Code or Codex) running against `plan.md` as the spec, with `PRINCIPLES.md` as the load-bearing input. I'd run the implementation in two delegate_task batches: M0 + M1 as one batch (port-decision sheet + scaffolding + prompt translation, surface back to you for the M0 sheet review), then M2–M8 as a second batch with the M0 sheet locked.

**Alternative A:** You implement yourself. You'd be faster on the port pieces (you know animus's index.py cold) but slower on the 2×2 corpus design. Net: comparable.

**Alternative B:** Defer implementation entirely; canonical plan sits on disk until you want to spend the time. Acceptable — the plan is durable.

**Recommendation: subagent batch dispatch, surface M0 to you for review before M1 starts.**

### Decision 3 — Where do prospecta and hermes-prospecta live as repos?

**My take:** I created `~/src/witt3rd/prospecta/` and `~/src/witt3rd/hermes-prospecta/` as empty repo roots. The library plan lives inside the prospecta repo at `docs/design/prospecta/`. **Confirm this is where you want them, or move them before M1 starts.**

**Alternative:** Promote to a Nous-org or witt3rd-org GitHub repo before M1 if you intend to publish. (M0/M1 don't depend on remote state; this can wait until M8 ship gate.)

**Recommendation: leave at `~/src/witt3rd/` for now; promote to GitHub when v0.1 is ready to publish.**

---

## Where I deviated from our pre-dispatch conversation

Three deviations, all explicit and small:

1. **Budget.** We talked about "library is the substantial artifact; plugin is a Saturday once library lands." 12.5d ≠ a Saturday for the library, but it's close to "a working week and a half." The Saturday line was about the plugin specifically — that's still right; the plugin is ~300-500 LOC of glue once library v0.1 is stable. The library itself is bigger than I implied pre-dispatch.

2. **`embedding_model` parameter.** Pre-dispatch I'd said "expose as `Memory(embedding_model=...)` with warn-on-mismatch." The Architect's A6 + Planner's adoption ended up at "raises `NotImplementedError` if user-supplied in v0.1." Honest: chroma's silent-on-fresh-index behavior makes warn-on-mismatch a footgun. v0.1 hardcodes `all-MiniLM-L6-v2`; v0.2 implements real migration. **Worth confirming you accept this.**

3. **Tracer / observability primitive.** Pre-dispatch I didn't name observability. The Critic's M1 catch was right — debugging a bilateral-synthesis miss in production without a tracer means spelunking. Added as v0.1 (six named events, default no-op, ~30 LOC). Cheap addition, real value.

---

## Where the plan landed

Vertical-slice port order with a port-decision-sheet **gate** before code lands. M0 surfaces every entanglement site in animus's `index.py` and `queries.py` with explicit disposition (delete / shim / generalize / preserve). M1 does prompt translation (not copy — animus prompts are Cookie-shaped and need generalization). M2 ports classical RAG end-to-end without the spine. M3 adds the write-side spine (`generate_index_text` + `retain`). M4–M5 add the read-side spine with JSON mode. M6 wires sweeper + CLI + tracer. M7 is the 2×2 bilateral ship gate. M8 ships the README. The bounded plugin section (§9 of the plan) names what the implementer inherits from the locked decisions and hindsight's reference shape; no plugin work happens until library v0.1 is stable.

---

## Open questions left (not blocking)

1. **The 7th `index.py` entanglement site** — Architect named seven, my reading found six. M0 sheet surfaces this; if the seventh is structurally load-bearing rather than deletable, M2 budget gets a second adjustment.
2. **Anthropic JSON-mode adapter complexity** — Hermes and OpenAI are clean; raw Anthropic SDK callers need ~10 lines (tool-use coercion) rather than 3. Documented in README; acceptable.
3. **2×2 corpus tuning risk** — designing the corpus so ON/OFF and OFF/ON each show *partial* improvement (not full, not zero) requires iteration. M7 budgets for it; M8 README is the bumper.

---

## Next steps

If you ratify, my proposed sequence:

1. You confirm Decisions 1–3 above.
2. I dispatch M0 + M1 as the first implementation batch (port-decision sheet + scaffolding + prompt translation). ~3 days of work.
3. I surface the M0 sheet to you when it's ready — quick walk before M2 starts.
4. M2–M8 dispatched as second batch, ~9.5 days of work. I surface back at the M7 ship gate (2×2 bilateral test results).
5. Library v0.1 tagged. Hermes plugin follow-on (the actual Saturday) starts.

If you push back on any of the three decisions, name which and I refine.

---

⚒️ Forge — brief for ratification, prospecta-impl ralplan 2026-05-18.
