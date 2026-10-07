# Brief — Prospecta Schema + Plan v2 (Postgres Substrate)

**Date:** 2026-05-18
**Author:** Forge ⚒️
**Audience:** Donald, for ratification or push-back.
**Read time:** ~5 minutes.
**Deeper provenance (default-not-read):** `forge-review-deep-v2.md` next door.

---

## TL;DR

You ratified the substrate pivot (six decisions) and asked for the (γ) hybrid path: inline-revise the plan, run a focused schema ralplan. Both done. **Three canonical artifacts now on disk** at `~/src/witt3rd/prospecta/docs/design/prospecta/`:

- `decision-record-1.md` — substrate pivot decisions (six locked, 2026-05-18)
- `schema.md` — canonical schema stance (ralplan-blessed, R2 APPROVE_WITH_PROVISO, ~9 provisos folded including 1 real bug caught)
- `plan-v2.md` — full implementation plan with substrate pivot folded; supersedes `plan.md`

**Total budget: ~14.5 working days** for prospecta v0.1 (was 12.5 on chroma; +2 for Postgres bring-up, SQL translation in M2, Postgres tracer sink in M6). 10 milestones (added M-1: Postgres bring-up + schema migration). The bilateral spine, bounded plugin section, and 2×2 integration test all carry forward intact from plan v1.

The schema ralplan caught a real bug (advisory-lock constant overflowed signed bigint — would have crashed first migration in production). Worth its own runtime cost.

---

## Decisions you need to make

### Decision 1 — Ratify plan-v2 and authorize implementation?

**My take:** Yes, ratify. Plan-v2 is design-ready end-to-end. Substrate pivot is honored, schema is rigorous, milestones are concrete, deferred items are explicit. None of the open questions in §15 are blockers.

**Alternative:** Push back on the 14.5d budget. Same dynamics as before — could cut M-1 corners (skip docker-compose, assume operator brings Postgres) for ~0.5d; could defer Postgres sink to v0.2 in M6 for ~0.5d. I do not recommend either; both leave operator pain.

**Recommendation: ratify.**

### Decision 2 — Implementer disposition unchanged?

You said omh-ralph-driver for execution. That's the path. My proposed sequence:

1. **M-1 + M0 + M1 dispatched together** as the first ralph run (~4.5d of work — Postgres bring-up + port-decision sheet + scaffolding + prompt translation). Surface M0 sheet to you for review before M2 dispatches.
2. **M2 dispatched separately** (~4d alone) because the SQL translation of `animus/memory/index.py` is the substantial intellectual work. Worth standalone review.
3. **M3–M8 as a final batch** (~6d).

I'll let omh-ralph-driver split this however its iron-law calls for it; the above is just my recommended batch shape.

**Recommendation: omh-ralph-driver execution, my proposed batch shape, you can adjust.**

### Decision 3 — Anything to push back on in the substrate pivot itself?

The six decisions in `decision-record-1.md` were ratified, but the schema ralplan surfaced two design positions you should consciously bless before we ship:

**(α) Per-bank vector dimensionality.** One bank = one embedder = one dim, forever (or until you carve a new bank and migrate). The Critic raised this as a possible flexibility cost (per-row dim would let one bank evolve). I held the per-bank position because it's simpler, the migration path (`create new bank → reindex → swap bank_id`) is well-understood, and hindsight does the same. **Confirm you accept this trade.**

**(β) Replace-on-source-match for re-retain.** Same `documents.source` + same caller = delete old `memory_items`, regenerate. Different source on same `content_hash` = `DocumentSourceConflictError`. The Critic considered no-op-on-collision; I went with replace because it bounds storage growth and matches caller intent (re-retaining same source means "I want this updated"). **Confirm you accept this trade.**

Neither needs a separate decision call; just naming them so you ratify with eyes open.

---

## Where I deviated from our pre-dispatch conversation

Just one: I let-slide three Critic catches (C2 `corpus_path` on documents, C4 `prompt_caller_supplied` boolean, C6/C7/C8 prose-only doc additions). All v0.2-flavored or doc-flavored, not load-bearing for v0.1. Documented in the schema consensus summary at `~/forge/.omh/plans/ralplan-prospecta-schema.md`.

Otherwise: all six substrate decisions held, the schema rigor honored, hindsight's seed material lifted where transferable.

---

## Where the plan landed

10 milestones, 14.5 working days. M-1 stands up Postgres (docker-compose + migrations + advisory lock) and validates the bank lifecycle. M0 sheets the port disposition. M1 scaffolds + translates the three Cookie-shaped prompts. M2 ports `animus/memory/index.py` translated to **SQL via psycopg** instead of chroma calls — this is the substantial intellectual work, ~4d alone. M3 adds the write-side spine (`generate_index_text` + `retain` + replace-on-source-match). M4–M5 add the read-side spine with JSON mode. M6 wires sweeper + CLI + tracer + Postgres sink. M7 is the 2×2 bilateral ship gate (now Postgres-backed, with hybrid retrieval as the implicit safety net). M8 ships the README with the "what makes prospecta different" section now naming pgvector + hybrid + bank_id_template.

The Hermes plugin section in `plan-v2.md` §10 stays bounded at ≤2 pages, six items, positive-inheritance framing. Item (a) `__init__.py` skeleton now wires the embed callable via the new wizard step.

---

## Open questions left (not blocking)

1. **Wizard UX for `embed_provider`.** Six top-level provider choices (sentence_transformers / openai / openai_compatible / + future). Setup wizard cascades. Donald can refine during M8.
2. **`bank_id` resolution for shared Postgres** when multiple Hermes profiles point at one DB. Hindsight's template syntax inherited; verify in M-1 with Cookie's-bank + Forge's-bank case.
3. **`pgvector` minimum version pin.** Pin `>= 0.5` (HNSW support); verify against Azure Flexible Server's shipped version during M8 deployment doc work.
4. **Azure deployment doc** needs live-testing against Flexible Server. M8 deliverable.

---

## Next steps

If you ratify Decision 1 (and bless α + β):

1. omh-ralph-driver loaded with `plan-v2.md` + `schema.md` + `decision-record-1.md` + `PRINCIPLES.md` as core context.
2. M-1 + M0 + M1 dispatched as first batch (~4.5d).
3. M0 sheet surfaces to you for review before M2.
4. M2 dispatched as standalone (~4d, the substantial SQL-translation work).
5. M3–M8 final batch (~6d). M7 ship gate surfaces to you (2×2 bilateral test results).
6. Library v0.1 tagged. Hermes plugin follow-on starts.

If you push back on any decision, name which.

---

## Artifacts on disk

```
~/src/witt3rd/prospecta/
├── PRINCIPLES.md
├── docs/design/prospecta/
│   ├── context.md                          # original plan ralplan context
│   ├── plan.md                             # SUPERSEDED — chroma plan
│   ├── stance-planner.md, *-v2.md          # R1+R2 provenance, plan ralplan
│   ├── review-architect*.md                # ”
│   ├── review-critic*.md                   # ”
│   ├── brief.md                            # the prior brief (chroma plan)
│   ├── forge-review-deep.md                # the prior deep review
│   ├── decision-record-1.md                # NEW: substrate pivot
│   ├── schema-context.md                   # NEW: schema ralplan context
│   ├── schema-planner.md, *-v2.md          # R1+R2 provenance, schema ralplan
│   ├── schema-architect*.md                # ”
│   ├── schema-critic*.md                   # ”
│   ├── schema.md                           # CANONICAL: schema stance
│   └── plan-v2.md                          # CANONICAL: full plan with substrate pivot
└── (M-1 will scaffold the actual code from here)

~/forge/.omh/plans/
├── ralplan-prospecta-impl.md               # consensus summary, original plan
└── ralplan-prospecta-schema.md             # NEW: consensus summary, schema
```

⚒️ Forge — brief for ratification, prospecta-impl + prospecta-schema ralplans 2026-05-18.
