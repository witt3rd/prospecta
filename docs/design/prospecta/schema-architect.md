# Schema Architect Review — Prospecta v0.1 Postgres+pgvector

**Reviewer:** Forge ⚒️ (Architect role, R1)
**Authored:** 2026-05-18
**Reviewing:** `schema-planner.md` (Planner R1, self-graded APPROVE)
**Inputs:** `schema-context.md`, `decision-record-1.md`, `PRINCIPLES.md`, hindsight-client v0.6.1 seed material, parent `plan.md`

---

## Verdict

**APPROVE_WITH_RESERVATIONS.**

The stance is architecturally sound on every load-bearing axis — granularity, dim strategy, hybrid pattern, index choice, migration shape, test isolation. Six concerns are real but small; none is a structural fault. They are the kind of cleanup that costs an hour now and avoids a v0.2 migration. None justify another planner round; they fold into a touch-up before the schema-stance distills into `plan-v2.md` §5.

The Planner's self-grade is defensible. I would not have given full APPROVE because A1 (dead config field) and A4 (lost sweep audit) are surfaces that will accumulate cost if shipped as-is, and A2 (dim enforcement gap) deserves a sentence of operator-facing documentation that the stance doesn't currently carry. But the bones are right.

---

## Strengths

1. **Bilateral spine is structurally first-class, not annotated.** `memory_items.content` *is* the LLM-anticipated question form; `original_chunk` carries the source; both columns are mandatory; `llm_generated` makes the spine auditable. The schema cannot be misread as content-space RAG. P1 is enforced by shape, not by docstring.

2. **Per-bank vector dim via partial HNSW + library-mediated dim assertion is the right call.** Donald's home-BGE / cloud-OpenAI split is supported physically (one HNSW per bank), and cross-embedder-within-bank is made *impossible-by-construction* rather than guarded by convention. The partial-index pattern composes with bank-scoped recall (every query has `WHERE bank_id = ...`) so the planner picks the right index automatically. This is the single best decision in the stance.

3. **RRF SQL is canonical and correct.** I traced the FULL OUTER JOIN by hand: both-sides-hit produces `1/(60+sem_rank) + 1/(60+lex_rank)`; semantic-only produces `1/(60+sem_rank) + 0`; lexical-only the symmetric case. COALESCE is correct on the join keys. `k=60` matches the literature (Cormack et al. 2009, and every pgvector hybrid-search guide through 2025). HNSW `m=16, ef_construction=64` matches pgvector's documented defaults.

4. **Event tables are typed, not polymorphic.** The Planner's rejection of a discriminator-on-`type` events table is correctly motivated by P5 (raw responses queryable, not JSONB-extracted) and Donald's stated ops queries (schema-context §6). Typed tables with composite `(bank_id, created_at DESC)` indexes will answer Donald's "median formulate duration" query without a JSONB extraction step.

5. **Hand-rolled SQL migrations over Alembic.** Right call. The schema is small and stable; Alembic's autogenerate machinery solves a problem prospecta doesn't have (declarative-model → migration diff), and it would pull SQLAlchemy by convention. P12 honored.

6. **Test scaffold gets DB isolation right.** Per-test bank scoping on a session-scoped testcontainer is the correct trade between speed and fidelity. Per-test schema would re-run migrations (multi-second overhead); transaction-rollback would break the HNSW recall side. The Planner identified the failure modes of the alternatives, not just chose blindly.

---

## Concerns

### A1 — Dead config: `banks.fts_config` exists but cannot be honored in v0.1

**What's wrong.** `banks.fts_config REGCONFIG NOT NULL DEFAULT 'english'` is added to the schema, but `memory_items.content_tsv` is a STORED generated column hardcoded to `to_tsvector('english', content)`. Postgres generated columns cannot reference another table, so the per-bank config is silently ignored. The column is configuration noise — exactly the failure mode P12 (honest config surface) warns against.

**Why it matters.** Operators reading the schema will conclude that per-bank FTS language is supported; it isn't. When a caller sets `banks.fts_config = 'simple'` for a code-heavy bank, the tsvector still uses `english` stemming, recall behavior diverges from configured behavior, and the bug surfaces only in lexical recall quality (not as an error). This is a substrate honesty issue, not a perf issue.

**Suggested fix.** Pick one of two paths and commit:
- **(α) Remove `banks.fts_config` for v0.1.** Document `english` as the only supported FTS language. Add the column back in v0.2 when the trigger-based per-bank tsvector lands. Simpler, more honest.
- **(β) Implement now via trigger.** `BEFORE INSERT OR UPDATE` trigger sets `content_tsv` from `to_tsvector(banks.fts_config, NEW.content)` with a bank join. Two writes per insert; trigger maintains correctness. Acceptable but more machinery.

Lean (α) for v0.1 — matches "minimize bespoke" stance. The column is the contest, not the trigger.

### A2 — Vector dim is enforced by the library, not by the schema; this is undocumented in the stance

**What's wrong.** `memory_items.embedding` is declared `vector NOT NULL` — *unsized*. The dim assertion lives in library code on insert; the partial HNSW index uses `(embedding::vector(1024))` cast, which fails *at index-use time* if a row has wrong dim, not at insert time. A caller using psycopg directly (skipping the library) can insert any dim into a bank, silently breaking the bank's HNSW index on next maintenance.

**Why it matters.** PRINCIPLES.md §P2 ("Standalone first, plugin second") implies the library is one of N possible callers — animus, scripts, notebooks, future agents may bypass `Memory()` and write SQL directly. The schema is the contract; the library is one consumer of that contract. The stance currently relies on the library being the only writer.

**Suggested fix.** Two complementary moves:
1. **CHECK constraint** on `memory_items`: `CHECK (vector_dims(embedding) = (SELECT embedding_dim FROM banks WHERE bank_id = memory_items.bank_id))` — actually pgvector doesn't allow subqueries in CHECK; the cleaner pattern is a `BEFORE INSERT OR UPDATE` trigger that raises if dim doesn't match `banks.embedding_dim`. Cheap, schema-side, enforced for all writers.
2. **Document the migration story** in the stance itself, not just as an "v0.2 operation" footnote. Operators need to read: "switching embedders in a bank requires (a) `ALTER TABLE banks SET embedding_dim = N`, (b) `DELETE FROM memory_items WHERE bank_id = ...`, (c) re-embed and re-insert, (d) `DROP INDEX memory_items_emb_hnsw_<bank>` and recreate." This is operationally tractable but unobvious; it deserves explicit treatment in the stance, not deferral.

### A3 — `content_hash` re-retain semantics are underspecified in the schema

**What's wrong.** `documents` has `UNIQUE (bank_id, content_hash)`, but the stance doesn't show the upsert path. The §`memory_items` deep-dive mentions "delete-then-insert by `(document_id, content_hash_of_original_chunk)` — practical handling lives in the library," but there is no `content_hash_of_original_chunk` column on `memory_items`. The library either computes the hash on every retain (acceptable but undocumented) or the delete-then-insert key is fuzzy (problematic).

Concretely: if a caller calls `retain(source='X.md', original_text=...)` twice with identical content (same content_hash), the document insert hits the unique constraint. What happens? The stance is silent. Three plausible behaviors:
- **No-op** (return existing document_id, skip memory_items work entirely).
- **Update memory_items** in place (treating it as `update_mode='replace'`).
- **Error** (caller must explicitly choose update_mode).

**Why it matters.** Sweeper re-runs will frequently re-retain unchanged files. The behavior on collision is the hot path for sweeper correctness — and it's not specified in the schema stance.

**Suggested fix.** Add a §"Re-retain semantics" subsection to the stance with two named cases:
- **Identical content** (`(bank_id, content_hash)` matches existing document): no-op. Library reads existing `document_id`, skips embedding work, returns the existing `RetainResponse`.
- **Same source, different content** (caller-supplied `source` matches, content_hash differs): library deletes prior document's memory_items (cascade) and inserts new document + memory_items. This is `update_mode='replace'` at document granularity, distinct from per-memory_item replace.

This is a one-paragraph addition, not a re-design. It belongs in the schema stance because it's about the unique constraint's operational semantics.

### A4 — Collapsing `sweep_passes` into `sweeper_state` loses append-only audit; the Planner's justification doesn't hold

**What's wrong.** The Planner's argument for the collapse is: "history is recoverable from `retain_events` filtered by source." This is false in two ways:
1. The sweeper does work even when nothing changes (file scan, hash check, no retain fired). `retain_events` only records actual retains. A "sweeper ran, found 1000 files, indexed 0 new, pruned 3" pass leaves zero retain_events rows but is real operational data.
2. `last_pass_errors` and `last_pass_files_pruned` are state columns. Their history is lost on every upsert. Donald's likely ops query "did sweeper error rate spike last week?" cannot be answered.

**Why it matters.** P14 says the sweeper is the safety net; observability of the safety net is what makes it trustworthy. Collapsing history into current-state is the failure mode P14 implicitly warns against — silent degradation of the drift-detection layer.

**Suggested fix.** Add `sweep_events` back. Keep `sweeper_state` for the resume case (current pass position per corpus_path). Two tables, distinct concerns:
- `sweeper_state` — one row per `(bank_id, corpus_path)`, upserted, **current** pass status (for resume + dashboard "is sweeper healthy right now?").
- `sweep_events` — append-only, one row per completed pass, with `files_seen / indexed / pruned / errors / duration_ms`. Mirrors the four other event tables. Indexed `(bank_id, created_at DESC)`.

This is 30 LOC of migration. The "five tables become four" simplification claim doesn't survive the audit; the right comparison is "two tables with clean concerns vs one table with conflated concerns." Two tables is the cleaner answer.

### A5 — Migration runner needs advisory locks for concurrent-startup safety

**What's wrong.** The `run_migrations()` helper outlined in §Migrations does `SELECT MAX(version) → for each pending: BEGIN; exec; INSERT version; COMMIT`. No locking around the read-max-then-apply window. Two processes calling `Memory(...)` simultaneously on a fresh DB will both see `MAX(version) = 0`, both try to apply `0001_initial.sql`, the second will fail with `relation "banks" already exists` or similar.

**Why it matters.** This is the multi-process case Decision 1 explicitly promised to support natively ("Multi-process clients sharing one Postgres works natively"). Migration startup is the exact moment that promise gets tested. The failure isn't catastrophic (the second process crashes loudly) but it's user-hostile and easily prevented.

**Suggested fix.** Wrap `run_migrations()` in `pg_advisory_lock`:
```python
def run_migrations(conn):
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (PROSPECTA_MIGRATION_LOCK_KEY,))
        # ... existing logic ...
```
`PROSPECTA_MIGRATION_LOCK_KEY` is a fixed bigint (e.g., hash of `"prospecta-migrations"`). Concurrent processes serialize on the lock; second one sees the updated version on its read and no-ops correctly. Fifteen LOC, eliminates a real bug class.

### A6 — `scores.semantic` / `scores.lexical` are nullable in the API contract; the JSON example shows numeric values, masking the real shape

**What's wrong.** The `RecallResult` example shows:
```json
"scores": { "semantic": 0.812, "lexical": 0.443, "rrf": 0.0234 }
```
But the SQL returns `s.sem_score` and `l.lex_score` directly without COALESCE; when an item appears on only one side of the FULL OUTER JOIN, the missing side's score is NULL. So in practice `scores.lexical` can be `null` for a semantic-only hit, and the API contract has not declared that.

**Why it matters.** Caller code that parses `RecallResult` will see a numeric type in the example and write `result.scores.lexical > 0.3` — which throws on null. The contract should either document the nullability explicitly or COALESCE to 0 in SQL.

**Suggested fix.** COALESCE to 0 in the SQL (`COALESCE(s.sem_score, 0.0) AS sem_score`) and document the meaning in the contract: "Side scores are 0 when the item did not appear in that side's candidate pool. Only `rrf` is guaranteed > 0." Minimal change; preserves the cleaner type contract.

---

## Load-bearing decisions validated

These survive review without reservation:

- **Per-bank dim, partial HNSW index** — correct, operationally clean, makes cross-embedder-within-bank physically impossible (which is what we want).
- **HNSW over IVFFlat at v0.1** — correct for the 100K–1M row workload Donald named; concurrent-write friendliness matters when sweeper + retain hit the same bank.
- **RRF `k=60` not exposed per-call** — right P12 call; the literature converges; exposing it would invite micro-tuning we cannot justify.
- **Per-side limit (50) internal, `limit` exposed** — right call about which knob is the user's and which is the implementation's.
- **Hand-rolled SQL migrations** — right call; Alembic would be ceremony.
- **Per-test bank scoping on session testcontainer** — right call; the failure modes of the alternatives are correctly named.
- **Typed event tables (4, not polymorphic-1)** — right call; P5 + ops-query analysis dominate the proliferation cost.
- **`bank_id` as TEXT not UUID** — right call (convergent with hindsight; operators write strings).
- **`update_mode` per memory_item** — correct hindsight parity.

## Load-bearing decisions to reconsider

None at the structural level. The six concerns above are surface fixes, not architectural pivots. The Planner correctly identified six legitimately-deferred questions for R2/Critic; my concerns are mostly in adjacent territory (operational safety, contract precision, dead config) rather than in the questions the Planner left open.

One adjacent observation: the stance is silent on **`pg_trgm` index usage**. The Planner creates the extension but uses no index from it. Either drop the extension creation (less surface, P12) or document where the trigram index would land (concrete tertiary fallback when both semantic and lexical return zero). Leaving the extension created-but-unused is a minor case of A1's pattern. Lean: drop until evidence justifies it.

---

## Recommended changes (ordered by importance)

1. **A4 — Restore `sweep_events` as append-only audit alongside `sweeper_state`.** Highest impact; the simplification claim doesn't survive audit and the P14 observability cost is real.

2. **A5 — Wrap `run_migrations()` in `pg_advisory_xact_lock`.** Highest cost-per-LOC fix; 15 LOC eliminates a bug class promised against by Decision 1.

3. **A2 — Add dim-check trigger on `memory_items` + document the embedder-migration sequence.** Closes the bypass-the-library hole and makes the v0.2 operation operator-tractable from day one.

4. **A1 — Drop `banks.fts_config` for v0.1 (lean α).** Most honest path; trigger-based per-bank FTS lands in v0.2 with the multi-language pressure that justifies it. Removes a P12 violation.

5. **A3 — Add §"Re-retain semantics" to the stance specifying the (bank_id, content_hash) collision behavior.** One-paragraph addition; closes a real hot-path ambiguity for the sweeper.

6. **A6 — COALESCE side scores to 0 in SQL; document `RecallResult.scores` nullability rule.** Smallest fix; eliminates a foreseeable caller bug.

7. **Adjacent — Drop `CREATE EXTENSION pg_trgm` until an index needs it.** P12 honesty; restore when evidence justifies.

Six concerns, one adjacent observation, ~60 total LOC of migration + ~one page of doc revisions. None requires re-planning. The stance is ready to distill into `plan-v2.md` §5 with these touch-ups folded in; if Critic surfaces no META-level reframe, R2 is unnecessary.

⚒️ Forge — Architect R1, prospecta schema review, 2026-05-18.
