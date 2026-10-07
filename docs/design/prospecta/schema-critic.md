# Schema Critic — Prospecta v0.1 Schema Stance (R1 Critic)

**Domain:** `~/src/witt3rd/prospecta/` — schema slice
**Authored:** 2026-05-18 by Forge ⚒️ (Critic role, R1)
**Input:** `schema-planner.md` (self-graded APPROVE), `schema-context.md`, `decision-record-1.md`, `PRINCIPLES.md`, hindsight v0.6.1 models
**Charter:** contest the *framing*, not compliance-check the DDL.

---

## Verdict: **APPROVE_WITH_RESERVATIONS**

The stance is structurally sound. The spine is centered (P1 honored), Postgres+pgvector substrate is leveraged honestly, hindsight lineage is cited at every divergence, and the 10 contested questions are locked with rationale that survives adversarial inversion. The reservations below are not framing-killers — they are dimensions the planner under-named or under-modeled, several of which Donald has flagged in adjacent contexts (versioning, prompt provenance, operator ergonomics around multi-corpus banks). Critic recommends **R2 with revised stance** if M1 + M5 + F2 cluster lands as load-bearing; otherwise distill as-is with the open questions promoted to first-class §"Known limitations" in plan-v2.md.

---

## Framing contests

### F1 — `documents.original_text` + `memory_items.original_chunk` = the same bytes twice

The planner correctly defends `original_chunk` against the join-on-recall failure mode (§"What prospecta diverges on"). What it does *not* name: when a 50KB document yields 5 memory_items, the chunks together approximately reconstruct `documents.original_text`, so the source bytes live in storage **twice** — once in `documents.original_text` (the dedup-keyed canonical), once distributed across `memory_items.original_chunk` for the same document.

This isn't catastrophic — pgvector storage dominates byte cost — but it's worth naming because:

- It creates a **silent consistency obligation**: nothing enforces that `original_chunk` substrings are *actually* substrings of `original_text`. A library bug that mutates one but not the other is undetectable at the schema level.
- It muddies the question "what is the source of truth for source bytes?" The planner answers `documents.original_text` (line 65: "P5: no truncation, source of truth"), but then duplicates that truth into `memory_items` for read-side efficiency. Both true; neither named together.

**Counter-proposal:** keep the duplication (the planner's efficiency argument is correct) but add a CHECK or schema-level comment naming `documents.original_text` as canonical and `memory_items.original_chunk` as a denormalized read-side projection. The decision is right; the documentation of the asymmetry isn't load-bearing in the DDL itself.

### F2 — `bank_id` carries multi-tenancy AND multi-corpus, and the schema notices only the first

`bank_id` is correctly the multi-tenancy primitive (Decision 4, off-limits). But the schema implicitly assumes **one bank ⇔ one corpus**. The only place `corpus_path` exists is `sweeper_state.corpus_path` — and even there it's PK material, suggesting a bank *can* have multiple corpus paths, but `documents` has no corpus_path column. Sweeper state knows about corpora; the substrate does not.

Donald's named ops query (schema-context §6): "is the sweeper healthy for `~/animus/substrate/`?" implies a sweeper-per-corpus-path mental model. But if Donald wants to ask "which documents in this bank came from `~/animus/substrate/` vs. `~/forge/`?", the answer today is **`documents.source` LIKE prefix-matching** — string-prefix archaeology on every query.

**Counter-proposal:** add a nullable `documents.corpus_path TEXT` column (with index) in v0.1. Cost: one column, one index, one library-side wire-up at retain time. Benefit: makes "corpus" a first-class join axis, makes sweeper_state and documents share vocabulary, and gives Donald the per-subdirectory queries he'll want within six months. This is *not* re-opening bank_id; it is acknowledging that bank is the *tenant* primitive and corpus is the *source-grouping* primitive within a tenant, which the schema already half-models in sweeper_state.

### F3 — The "altitude" argument: could v0.1 ship with 4 tables?

The planner's eight tables (nine with `prospecta_schema_version`) are each individually defended. But the gestalt question — could v0.1 ship with `banks` + `memory_items` + one `events` table + `sweeper_state` and add the typed event tables in v0.2 once we know which ops queries fire? — is not answered.

The planner's anti-polymorphic argument (§"Why not one polymorphic events table") is *partially* correct: typed columns matter when the column is queried as an SQL predicate. But `recall_events.trace` is already JSONB; `retain_events.raw_llm_response` is just TEXT. Most of the typed columns *are* either timestamps or JSONB or TEXT — the typing is shallow.

Counter-argument that the planner could have made: Donald's ops queries (schema-context §6) include "median formulate duration" and "recall events count by mode last 7d." These are predicate queries on `duration_ms` and `mode` — *those* columns benefit from typing. The polymorphic table would force `duration_ms` into JSONB extraction, which is the real cost. Naming this explicitly would strengthen the stance.

**Lean:** keep four typed event tables. But the planner should name that the *predicates Donald queries by* are what drives typing — not abstract "queryability." Three sentences in the rationale.

---

## Ontology contests

### O1 — `documents` is load-bearing, not animus log.jsonl smuggling

Critic position: `documents` is correctly first-class. Three real jobs it does that `memory_items` alone cannot:
1. Dedup key (`UNIQUE (bank_id, content_hash)`).
2. Single canonical home for `original_text` (P5 anchor).
3. Per-document tags/metadata distinct from per-question tags.

The planner does not foreground these; they are inferable from the DDL. **No change required**, but the README should name documents as the per-source-unit primitive distinct from per-question memory_items, so a contributor reading the SQL sees the spine in the *relationship* (one document → many anticipated questions).

### O2 — Dropped `tag_groups`/`mental_models` may be slightly too aggressive

The planner drops hindsight's `tag_groups` (mental-model trigger machinery) per off-limits. Correct on KG removal. But hindsight's `tag_groups` also served a lighter purpose: **bundling tags for caller-friendly recall** (`tags_match: 'all'` against a saved group). The current schema makes the caller pass the explicit tags array every time.

This is a thin, recoverable gap. v0.2 can add a `tag_aliases` table (name → tag set) without schema upheaval. **No v0.1 action**, but worth naming in §"Future extensions" so we don't pretend the drop was zero-cost.

### O3 — Granularity (one row per question) is correct, but the planner under-defends against hindsight parity

Hindsight's `MemoryItem` is one-row-per-fact-extracted (`memory_item.py`). Prospecta's `memory_items` is one-row-per-anticipated-question. **Same table name, different unit.** This is fine, but it means an operator coming from hindsight will be subtly miscalibrated — the row count semantics differ. The planner names "we converge" with hindsight; it would be honest to also name where we **don't converge** at the row-semantics level. Two sentences in §"Hindsight-shape lineage."

### O4 — Per-bank embedding dim is correct; per-row dim was right to reject

Planner's argument (§"Why per-bank not per-row") is sound. Per-row `dim` column would require N HNSW indexes anyway (pgvector requires fixed dim per index), buying nothing. Critic concurs. **No change.**

---

## Missing dimensions

### M1 — Versioning / re-retain semantics are under-specified

What happens when `retain(source='foo.md', ...)` runs twice on a file whose `original_text` changed?

The schema says: `UNIQUE (bank_id, content_hash)` on documents → new content_hash → new document row. The *old* document row and its memory_items are now orphan — they still exist, still match recall queries, still consume HNSW slots. The planner's `update_mode='replace'` semantic (§"Dedup story") is described as library-side delete-then-insert keyed by `(document_id, content_hash_of_original_chunk)` — but `document_id` changed when content changed.

**Result:** prospecta's storage grows monotonically on re-retain, unless the library explicitly deletes prior documents by `source` before inserting. The schema doesn't enforce this; the library will have to.

**Counter-proposal:** either (a) `UNIQUE (bank_id, source)` on documents and let the library overwrite, (b) add a `documents.superseded_by UUID` column with the chain, or (c) explicitly name "documents are versioned by content_hash; library is responsible for soft-superseding prior versions on retain-by-source" in plan-v2.md. (c) is cheapest, (a) is most honest, (b) is over-engineering. **Lean (a) or (c).** This is a **must-name** before consensus.

### M2 — `prompt_override` per-call not captured

Plan §3.4 names `prompt_override` as a per-call parameter (P4 caller-wins). The schema's `formulate_events.parse_fallback` is a boolean about *output* parsing — but there's no column for *which prompt was used*. If a caller overrides the formulate prompt, the event log says "json_mode_used=true" but nothing about whether the prompt was library-default or caller-supplied.

**Counter-proposal:** add `formulate_events.prompt_caller_supplied BOOLEAN` (mirroring `retain_events.index_text_caller_supplied`) and similarly on `llm_calls`. One bit per row. Makes spine-vs-caller behavior auditable end-to-end. **Should-fix.**

### M3 — Soft-delete absent

The schema has hard `ON DELETE CASCADE`. Deleting a bank wipes documents, memory_items, all events. There is no `deleted_at` column anywhere.

For v0.1 this is defensible (operator deletes deliberately, observability runs while live). But the planner does not name this as a deliberate choice — it just is. **Should-name:** "v0.1 deletion is hard, auditability lives in event logs until the bank is dropped; v0.2 may add soft-delete on documents if rollback semantics are needed." One sentence. Not a blocker.

### M4 — Embedding-model migration has prose, not schema

Planner says: "rebuild the bank's vectors against a new embedder, alter `banks.embedding_dim`, drop/rebuild HNSW. This is a documented v0.2 operation." But `banks` has no `embedding_model_version` column distinct from `embedding_model_id`. If Donald switches from `text-embedding-3-large@2024-01` to `text-embedding-3-large@2024-06`, the schema can't tell the embeddings were silently re-baked.

**Counter-proposal:** `banks.embedding_model_id` already exists as TEXT and is caller-opaque — the convention to put `model@version` in that field can be documented. Or add `embedding_model_version` as a separate column. **Lean:** document the convention. No DDL change. Name it.

### M5 — Bank stats / dashboards are computed-on-the-fly only

Donald named "robust logging and statistics for ops." The schema supports this *by query* (typed event tables + composite indexes). It does **not** ship a `bank_stats` materialized view, summary table, or rollup. For 100K–1M memory_items and modest event volume this is fine. For Donald-running-this-for-a-year-with-cron-sweepers, the event tables will grow unbounded.

**Counter-proposal:** name event-table retention/rollup as a v0.2 concern explicitly. Don't ship a materialized view in v0.1 (premature). But name that the schema as-shipped has **no event-log GC** — that retain_events and llm_calls grow forever. Operator will need to know. **Must-name.**

### M6 — Content size bounds

What if someone retains a 1MB markdown document? `documents.original_text TEXT` accepts it. The library chunks it; HNSW is fine. But Postgres TOAST kicks in on rows > 2KB and there's no bound check. For v0.1, this is correct (no premature limit). But the test strategy should include a "large document" test in plan-v2.md §7. **Should-add to test plan, not schema.**

### M7 — Sweeper state granularity

Per `(bank_id, corpus_path)`. If `corpus_path = '~/animus/substrate/'` and Donald wants to know "did the sweeper finish `~/animus/substrate/wisdom/`?" — it can't tell him. This is the same as F2: corpus-as-prefix-string vs. corpus-as-tree.

For v0.1, single-level corpus is fine. **No change**, but the same prose that addresses F2 covers this.

---

## Simplicity contests

### S1 — RRF `k = 60` locked, but per-side `limit = 50` also locked

Planner exposes `limit` (final result count) and `mode` to callers. Hides `k` and per-side `50`. This is correct restraint (P12 honest config surface). The lock is justified by the literature.

However: **the SQL hardcodes `60` in the CTE** (line 335-336). If `hybrid_rrf_k` is a library-config knob, the SQL must take it as a parameter, not literal-substitute it. The planner's example is illustrative, but the **must-fix** is: either name the parameter (`$rrf_k`) in the SQL or commit to k=60 being a library-build-time constant, not a runtime config. The current state — "library config exists but SQL is literal" — is incoherent.

**Counter-proposal:** parameterize `$rrf_k` in the SQL; default-to-60 in library; expose as library init param (not per-call). One-line SQL change. **Must-fix.**

### S2 — Hand-rolled SQL migrations vs Alembic

Planner's argument is sound: Alembic is surface, schema is small, hand-rolled SQL is debuggable. Critic concurs. The one risk: when v0.2 adds 3 columns to memory_items and a backfill, hand-rolled SQL with a `prospecta_schema_version` table and ordered files works fine for forward-only — but **rollback** is operator-by-hand. Name this as a known limitation. One sentence. **Should-name.**

---

## Principle audit (compact)

| # | Planner claim | Critic verdict |
|---|---|---|
| P1 | spine in `memory_items.content` + `llm_generated` flag | **HONORED**. Spine is centered. Critic recommends: add SQL comment on `memory_items.content` ("LLM-anticipated question form; bilateral-synthesis spine") so a DDL reader sees the spine without README. **P15-extension below.** |
| P2 | no Hermes coupling | **HONORED**. `bank_id` is TEXT, no Hermes types in DDL. ✓ |
| P3 | embedder/LLM are caller-injected | **HONORED with gap**: schema records `embedding_model_id` (good, M4) but does *not* record which LLM model produced `raw_llm_response`. **Should-add `llm_calls.model_id TEXT`** (caller-opaque) so spine-attribution is auditable. **Minor.** |
| P4 | caller-wins recorded via `llm_generated`, `index_text_caller_supplied` | **MOSTLY HONORED**, gap on prompt_override (M2). |
| P5 | full raw responses persisted | **MOSTLY HONORED**. `retain_events.raw_llm_response` and `formulate_events.raw_response` are present. But **`llm_calls` does NOT carry raw response**, only `prompt_name` + `duration_ms` + `error`. If the spine fires through `llm_calls` (which it will, for `generate-index-text`), the *raw call* is logged but the *content* is in retain_events. Fine if cross-referenced — but it should be named: `llm_calls` is a meta-log; raw text lives in the prompt-specific event table. **Should-document.** |
| P6 | schema doesn't heuristic LLM job | **HONORED**. ✓ |
| P7 | single write path (retain + sweeper both upsert memory_items) | **HONORED at schema level**; library-side responsibility. ✓ |
| P8 | superseded by Decision 1+3 | **OK** — "zero-config = docker compose up" is honest. |
| P9 | thin plugin | **HONORED at schema** (plugin only sets bank_id template). ✓ |
| P10 | reuses psycopg | **HONORED**. ✓ |
| P11 | tests with real Postgres | **HONORED in stance**. Critic asks: does the test plan cover *every table* with a real SQL exercise, not just import + create? Plan-v2 §7 must enumerate per-table assertions. **Should-verify in plan-v2.** |
| P12 | honest config surface | **HONORED** at config; **partially missed** at SQL: k=60 literal in CTE (S1). |
| P13 | prompts ship with library | schema-silent; **OK**. |
| P14 | sweeper is safety net | **HONORED via `sweeper_state` table**. Critic asks: does the table's column set make "sweeper may lag" honest? Yes — `last_pass_ended_at` can be hours old without affecting hot path. ✓ But name in `sweeper_state` SQL comment: "sweeper is best-effort safety net; staleness is expected." **P15-extension.** |
| P15 | spine documented in README | **HONORED in README scope** but **schema-level DDL comments are absent**. A contributor reading `0001_initial.sql` cold sees no `COMMENT ON COLUMN memory_items.content IS '...spine...'`. **Should-add COMMENT ON COLUMN statements** for the four spine columns + `llm_generated`. ~10 lines of DDL. Makes the schema self-documenting. **Should-fix.** |

---

## Specific counter-proposals (concrete)

| ID | Change | Cost | Benefit |
|---|---|---|---|
| C1 | Parameterize `$rrf_k` in hybrid SQL (or commit to compile-time constant) | 1 line SQL | Removes incoherence between config knob and literal SQL |
| C2 | Add `documents.corpus_path TEXT` + index | 1 column, 1 index | First-class corpus axis; resolves F2 + M7 |
| C3 | Document re-retain semantics (M1) — pick (a), (b), or (c) | prose | Prevents monotonic storage growth surprise |
| C4 | Add `formulate_events.prompt_caller_supplied BOOLEAN` and `llm_calls.prompt_caller_supplied BOOLEAN` | 2 columns | P4 audit symmetry with `index_text_caller_supplied` |
| C5 | `COMMENT ON COLUMN` for spine columns in 0001_initial.sql | 10 lines DDL | P15 schema-level |
| C6 | Document event-table unbounded growth as v0.2 concern (M5) | prose | Operator expectation-setting |
| C7 | Add `llm_calls.model_id TEXT` (caller-opaque) | 1 column | P3 audit completeness |
| C8 | Document `embedding_model_id` convention as `model@version` (M4) | prose | No DDL change; convention-only |

---

## Must-fix before consensus

- **C1** (RRF k parameterization) — current state is incoherent between config-knob claim and literal-60 SQL.
- **C3** (re-retain semantics) — silent monotonic growth is a real operator surprise; must be named in plan-v2.md even if the chosen answer is "library handles it."
- **C5** (COMMENT ON COLUMN for spine) — P15 honored only in README is a partial honor; the SQL must speak the spine too.

These are three small changes. None re-opens off-limits.

---

## Should-fix (R2 candidates)

- **C2** (corpus_path column) — strongly recommended; resolves a real Donald query Class.
- **C4** (prompt_caller_supplied bits) — symmetric audit completeness.
- **C6**, **C7**, **C8** — prose-only; document and ship.

---

## Things I'd let slide

- The `documents.original_text` + `memory_items.original_chunk` duplication (F1). Planner's efficiency argument is correct. Storage asymmetry is named; not load-bearing.
- 8 tables vs 4 (F3). Planner's defense holds if predicate-typed-columns rationale is added.
- Dropped `tag_groups` (O2). v0.2 work, not v0.1 blocker.
- Soft-delete absence (M3). Defensible for v0.1; name as known limitation.
- Hand-rolled migrations (S2). Right call; document rollback limitation.
- HNSW vs IVFFlat (planner Q3). Locked correctly.
- Per-bank embedding dim (O4). Correct as locked.

---

## Closing — what the stance is and isn't

The stance is structurally honest about the spine, correctly leverages pgvector + Postgres-native FTS, lifts hindsight where transferable, diverges where prospecta is simpler, and locks decisions with citation. It does not violate any P1–P15. It does not re-open off-limits.

What it under-models: **the time dimension of operator life** — re-retain (M1), event-log growth (M5), embedder migration (M4), corpus subdirectories (F2/M7). These are not v0.1-killers; they are six-month-out predictable friction that the stance can cheaply pre-empt by naming.

Critic recommends: **R2 with the three must-fixes folded in, the should-fixes considered, and the let-slides documented as known limitations in plan-v2.md §"Known limitations" so future-Forge reading the plan cold sees the dimensions the stance deliberately punted.**

If Planner accepts must-fixes → APPROVE → distill into `schema-stance.md` → fold into `plan-v2.md` §5.

⚒️ Forge — Critic R1, prospecta schema stance, 2026-05-18.
