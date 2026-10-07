# Schema Architect Review — Prospecta v0.1 Postgres+pgvector (R2)

**Reviewer:** Forge ⚒️ (Architect role, R2)
**Authored:** 2026-05-18
**Reviewing:** `schema-planner-v2.md` (Planner R2, self-graded APPROVE)
**Inputs:** R1 stance (`schema-planner.md`), R1 Architect (`schema-architect.md`), R1 Critic (`schema-critic.md`), `PRINCIPLES.md`, `schema-context.md`, `decision-record-1.md`

---

## Verdict

**APPROVE_WITH_PROVISO.**

R1's six catches are all honored at the right altitude. R2 chose the right path on every fork (drop `fts_config`, restore `sweep_passes`, replace-on-source-match, parameterize `rrf_k`, dim trigger, COALESCE on scores, COMMENT ON COLUMN at the DDL surface). The stance is structurally ready to distill into `plan-v2.md` §5.

One real bug surfaced in R2's new material — the advisory-lock key literal overflows `bigint`. It's a fifteen-character fix, not a design issue, but it has to be corrected before the stance ships into the plan or the migration runner will crash on first call. Two smaller cleanups (idiomatic dim function; documented intent for the sweep_passes ↔ sweeper_state co-write contract) are nice-to-have, not blockers.

R2 introduced no new architectural problems. The trigger, the §7 re-retain contract, and the table-split for sweep state are all the right shapes.

---

## R1 status table

| # | Concern | Status | Citation from v2 |
|---|---|---|---|
| A1 | `banks.fts_config` dead config | **ADDRESSED** | §1: *"`fts_config` is dead config. Status: ADDRESSED. Treatment: column removed from `banks` DDL."* §3 banks DDL confirms the column is gone. |
| A2 | Vector dim not schema-enforced | **ADDRESSED** | §3 ships `CREATE OR REPLACE FUNCTION assert_memory_item_embedding_dim()` and `CREATE TRIGGER memory_items_embedding_dim_check BEFORE INSERT OR UPDATE OF embedding, bank_id ON memory_items`. §1: *"a `BEFORE INSERT OR UPDATE` trigger on `memory_items` reads `banks.embedding_dim` for the row's `bank_id` and raises if `array_length(embedding::real[], 1) <> embedding_dim`."* |
| A3 | Re-retain semantics unspecified | **ADDRESSED** | New §7 *Re-retain semantics*: *"replace-on-source-match, error-on-source-divergence."* Three named cases (no existing row; matching source; divergent source → `DocumentSourceConflictError`), plus three NULL-source edge cases enumerated explicitly. |
| A4 | `sweep_passes` audit lost via collapse | **ADDRESSED** | §3 reinstates `sweep_passes` as append-only table; §9 documents the split: *"`sweeper_state` is the current-state cache (one row per `(bank_id, corpus_path)`, upserted); `sweep_passes` is the append-only history (one row per pass, including errors and prune counts)."* |
| A5 | Migration runner missing advisory lock | **PARTIAL** | §8 documents the intent: *"The runner takes a Postgres session-level advisory lock... `SELECT pg_advisory_xact_lock(MIGRATE_LOCK_KEY)`."* But the literal `MIGRATE_LOCK_KEY = 0x70726F73706563_74616` overflows `bigint` (see A7 below). The mechanism is right; the constant is wrong. |
| A6 | RRF JSON output nullability | **ADDRESSED** | §6 RRF CTE wraps scores in `COALESCE(s.sem_score, 0)` and `COALESCE(l.lex_score, 0)`; §10 verifies *"`scores.lexical` ... never null — COALESCE in SQL ensures 0 on missing side."* |

Five clean ADDRESSED, one PARTIAL where the design is right but the chosen constant doesn't fit the data type it lands in.

---

## New checks (R2 surface)

### 1. Dim-check trigger correctness

The trigger function (§3) is sound. Three observations:

- **Fires on UPDATE as well as INSERT.** ✓ `BEFORE INSERT OR UPDATE OF embedding, bank_id`. The column-scoped UPDATE is the right call — it skips re-validation on metadata-only updates (tags, context), keeping the hot path narrow.
- **Race conditions with concurrent INSERT.** The trigger reads `banks.embedding_dim` per insert. If a re-embedding operation runs `UPDATE banks SET embedding_dim = N` while inserts are mid-flight, the trigger's read could see either old or new value. §1 documents the migration sequence as drop-index → re-embed → update-banks → rebuild-index, which implicitly serializes (the library is the only writer, and re-embed is a long operation under operator control). Acceptable — but worth a `LOCK TABLE banks IN EXCLUSIVE MODE` in the documented migration sequence to make the serialization explicit rather than convention. Not a blocker; flag for the operator notes that will land in plan-v2.md.
- **Idiomatic choice.** `array_length(NEW.embedding::real[], 1)` works but converts the vector to a real array on every insert. pgvector ships `vector_dims(vector)` which returns the dimensionality without the array roundtrip. Substituting it shaves microseconds per insert and avoids the cast. Trivial diff:

  ```sql
  actual_dim := vector_dims(NEW.embedding);
  ```

  Worth folding in during distillation. Not load-bearing.

- **FK guard.** The trigger explicitly checks `IF expected_dim IS NULL` and raises with a clear message before the FK violation surfaces. Good defensive layering — the operator sees `bank_id = % has no banks row` instead of a generic FK error from end-of-statement.

### 2. Re-retain DELETE-then-INSERT and `created_at` preservation

§7 specifies: *"The document `id` is preserved... `memory_items` are rewritten."* The document row's `created_at` survives (UPDATE doesn't touch it; `updated_at` advances). The `memory_items` rows get fresh `created_at` because they're new rows.

Is this coherent? Yes, deliberately:

- The **document** is the durable identity (preserves external references, `id`, `created_at`).
- The **memory_items** are spine *output* — when the spine prompt or chunking changes, the items get refreshed; their `created_at` reflecting the regeneration moment is the honest signal, not the document's original retention time. Items aren't versioned snapshots; they're the current spine projection.

The C3 concern about unbounded growth is closed cleanly: same content from same source → bounded N items per document (replaced, not appended). The choice loses per-item history across regenerations, but the design doc is explicit that `memory_items` is current-projection, not audit. Any historical audit needed for spine drift lives in `retain_events.raw_llm_response` (P5 — full preservation of the spine LLM's response is retained event-side, not entity-side). That separation is right.

One small gap: §7 says *"`(memory_items) cascades from the document row's CASCADE — actually we delete by `document_id` directly so the document row stays)."* The parenthetical aside is correct but reads as in-flight thinking; cleanup pass should tighten the prose. Not architectural.

### 3. `sweep_passes` ↔ `sweeper_state` co-write contract

The split is honest (history table + current-state cache). What the schema doesn't enforce: that they update together.

Two write paths could drift:

- Sweeper completes a pass → INSERT into `sweep_passes`, then UPSERT into `sweeper_state`. If the second statement fails (transient FK or trigger error on `sweeper_state`'s PK), `sweep_passes` has the row but `sweeper_state` doesn't — the dashboard claims the sweeper hasn't run recently when it has.
- Conversely, a future code path could write `sweeper_state` without an audit row (mistake, partial migration, manual operator action).

The fix could be schema-side: a trigger on `sweep_passes` that upserts the corresponding `sweeper_state` row on insert. That moves the co-write from library convention to schema invariant. Not load-bearing for v0.1 — the library is the only writer and a single transaction wrapping both writes is sufficient. But the stance should name the transactional contract explicitly (one paragraph in §9: "Both writes happen in one transaction in the library; future schema-level enforcement is open for v0.2 if a second writer appears"). Worth adding to plan-v2.md.

**Verdict:** schema is honest about the split; the cross-table consistency contract is implicit. Document it.

### 4. Advisory lock integer — *real bug*

§8 ships:

```python
MIGRATE_LOCK_KEY = 0x70726F73706563_74616
```

Python `_` is a digit separator. Stripped, the literal is `0x70726F7370656374616`, which is 19 hex digits = 76 bits. The comment claims `"prospecta" derived 64-bit constant`. ASCII "prospecta" is 9 bytes = 72 bits, already too large for signed `bigint` (max 0x7FFFFFFFFFFFFFFF, 63 bits + sign). The literal as written is *larger still* — it has 9 ASCII bytes plus a trailing nibble.

`pg_advisory_xact_lock(bigint)` takes signed 64-bit. psycopg will raise `OverflowError` or Postgres will return `bigint out of range` when this value crosses the wire on the first migration run. The mechanism is correct; the chosen constant doesn't fit.

**Fix (one line, before distillation):**

```python
# Pick any stable 63-bit value. Two reasonable choices:

# (a) Hash of "prospecta-migrations", truncated to 63 bits:
MIGRATE_LOCK_KEY = 0x507238CC1F7E5A2D  # e.g., int(hashlib.sha256(b"prospecta-migrations").hexdigest()[:16], 16) & 0x7FFFFFFFFFFFFFFF

# (b) The first 8 bytes of "prospect" as ASCII bigint (drops the 'a'):
MIGRATE_LOCK_KEY = 0x70726F7370656374  # "prospect" — fits 64 bits
```

Either works. Lean (a) for non-ambiguity (no one will mistake a hash for a typo); document the derivation in a comment so future readers know it's stable. The `pg_advisory_xact_lock(bigint, bigint)` two-arg form (32-bit classid + 32-bit objid) is also available if the design wants self-documenting splits ("prospecta" namespace + "migrations" object); not necessary, just an option.

This is the only must-fix in R2.

### 5. `$rrf_k` parameter binding

The §6 example uses Bash-style `$rrf_k`, `$query_embedding`, `$bank_id`. Postgres prepared-statement syntax is `$1, $2, ...` (numbered) or, with psycopg, `%s` positional / `%(name)s` named. The `$identifier` form isn't valid Postgres parameter syntax — it's design-doc pseudocode.

For an internal stance this is acceptable (illustrative, not executable), but the prose in §6 explicitly claims *"the literal in the SQL was inconsistent with the library-level config; parameter form makes both consistent."* That sentence reads as if the example SQL is the parameterized form, when it's actually a different pseudocode convention. A future reader implementing the library could ship `f"... + 1.0 / ($rrf_k + ...)"` thinking the string substitution is the binding.

**Fix:** add one sentence to §6 — *"Example SQL uses `$rrf_k` as placeholder notation for clarity; actual psycopg binding uses `%(rrf_k)s` or numbered `$1`-style as the library prefers."* Or convert the example to `%(rrf_k)s` and drop the note. Either closes the gap.

### 6. `COMMENT ON COLUMN` visibility

✓ The SQL `COMMENT ON COLUMN` statements are inline in §3 for `banks.embedding_dim`, `documents.original_text`, `documents.content_hash`, `memory_items.content`, `memory_items.original_chunk`, `memory_items.llm_generated`. Operators running `\d+ memory_items` in psql will see them. P15 is honored at the DDL surface, not only in README. This is exactly what R1's C5 asked for and what R2's §13 audit table claims.

### 7. PRINCIPLES P1–P15 walk in the R2 frame

Walking each principle for *newly silent* failure modes (not re-litigating R1):

- **P1 (bilateral spine).** Strengthened — `COMMENT ON COLUMN content` is at the DDL surface. ✓
- **P2 (standalone first).** Schema is library-owned, no Hermes coupling. ✓
- **P3 (LLM injected).** `llm_calls.prompt_name` records the call without binding a provider. ✓
- **P4 (caller wins).** `llm_generated` boolean and `index_text_caller_supplied` in `retain_events` preserve audit. The re-retain replace-on-source-match in §7 honors caller intent rather than no-op-by-default. ✓
- **P5 (no truncation).** All TEXT columns. ✓ But: the dim-check trigger only validates dim, not magnitude/normalization. Not a P5 issue per se — magnitude isn't truncation — but worth naming once: prospecta assumes normalized embeddings (cosine ops). Document in operator notes (§14 already notes cosine is locked).
- **P6 (let LLM cook).** ✓
- **P7 (single write path).** *Slight new concern.* The re-retain §7 path is `DELETE memory_items + UPDATE documents + re-run spine`. The sweeper's path is `index_single_file → retain()`. Both flow through the library's `retain()`. ✓ But §7 says "library logic, not schema." Verify in `plan-v2.md` that there's only one `retain()` implementation, not a separate `re_retain()`. Single-verb contract preserved.
- **P8 (zero-config).** Superseded by docker-compose decision. ✓
- **P9 (thin adapter).** ✓
- **P10 (surface adjacent).** Migration runner reuses `pg_advisory_xact_lock`. ✓ (once the constant is fixed)
- **P11 (tests over prose).** §11 adds dim-trigger, re-retain (three cases), scores-numeric, advisory-lock tests. ✓ Add a fifth: `pg_trgm` removal is testable via `SELECT * FROM pg_extension WHERE extname = 'pg_trgm'` returning zero rows after initial migration — minor, ensures the cleanup is durable across re-applies.
- **P12 (honest config surface).** R2 removed `fts_config` and `pg_trgm`; `rrf_k` is parameterized but not surfaced per-call. ✓
- **P13 (prompts with library).** ✓
- **P14 (sweeper safety net).** `sweep_passes` restored. ✓
- **P15 (spine documented).** ✓ at DDL surface via `COMMENT ON COLUMN`.

No newly-silent principle. R2 is principle-coherent.

---

## What R2 made worse

Nothing structurally. The added surface (dim trigger, re-retain section, advisory lock, sweep_passes restoration) is all warranted — each closes a real gap from R1. The trigger adds one `banks` row lookup per `memory_items` insert; the cost is microsecond-scale because `banks` is tiny and buffer-cached, as §3 notes.

The advisory lock constant being wrong is *regression-shaped* (R1 didn't have a lock at all; R2 added one with a bug), but it's a typo at the implementation surface, not a design step backward. The mechanism R2 chose is the right one.

---

## Recommended changes for distillation

In priority order:

1. **Fix `MIGRATE_LOCK_KEY` to fit `bigint`** (15 min, blocker). Drop a comment explaining the derivation so future readers don't try to "restore" the over-long ASCII value.
2. **Switch dim function to `vector_dims(NEW.embedding)`** (1 line, idiomatic).
3. **Document the `sweep_passes` ↔ `sweeper_state` transactional co-write contract in §9** (one paragraph — library writes both in a single transaction; future schema-level enforcement deferred to v0.2 if a second writer appears).
4. **Clarify the `$rrf_k` notation in §6** (one sentence — illustrative placeholder, actual binding via psycopg conventions).
5. **Add `LOCK TABLE banks IN EXCLUSIVE MODE` to the documented embedder-migration sequence** (in plan-v2.md operator notes) so the dim-change race is explicit, not convention.
6. **Tighten §7's parenthetical aside on cascade behavior** (prose cleanup).

None of these requires R3 from the Planner. Items 1–4 are mechanical; 5–6 land in plan-v2.md anyway. The schema stance is otherwise ready to distill.

⚒️ Forge — Architect R2, prospecta schema review, 2026-05-18.
