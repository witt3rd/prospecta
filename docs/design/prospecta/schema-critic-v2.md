# Schema Critic — Prospecta v0.1 Schema Stance (R2 Critic)

**Domain:** `~/src/witt3rd/prospecta/` — schema slice
**Authored:** 2026-05-18 by Forge ⚒️ (Critic role, R2)
**Input:** R2 stance (`schema-planner-v2.md`), R1 stance, R1 Critic (self), R1 Architect, `PRINCIPLES.md`
**Charter:** counterfactual deference, R1 over-prescription audit, new silently-absent.

---

## Verdict: **APPROVE_WITH_PROVISO**

R2 honors every must-fix from R1 and resolves the three orchestrator-curated catches the framing required. The proviso is narrow: one test in §11 is hand-wavy as written (advisory-lock concurrency), and one R1 catch (C1, `$rrf_k` parameterization) looks on re-read like Critic over-prescription rather than load-bearing principle. Neither blocks consensus; both deserve naming before plan-v2 distills.

Off-limits items (substrate decisions, spine, KG drop, C2 `corpus_path`, C4 `prompt_caller_supplied`, C6–C8 prose docs) untouched. Good.

---

## Test 1 — Counterfactual deference, per adoption

The question: for each adoption, would the Planner's defense have been written if Critic had proposed a different alternative? If grounds transfer → deferential. If grounds resist → principled.

### A1 — `banks.fts_config` drop → **STRONG PRINCIPLED**

The Planner's defense names a specific structural inconsistency: the generated column hardcodes `to_tsvector('english', content)`, so a `fts_config` column on `banks` promises configurability the schema cannot honor. Counterfactual: had Critic proposed "keep `fts_config` with a default of `'english'` and a trigger that swaps the generated column when it changes," the Planner would have had to defend the trigger overhead or reject it on simplicity grounds — neither of which matches the actual R2 defense. The defense **requires the orphan reading**; it does not transfer. Adoption is principled, not deference.

### A2 — Dim-check trigger → **PRINCIPLED**

Defense ("schema-enforced contract, not library convention") would also support an alternative — e.g., a library-side assertion at `INSERT` time — but the Planner names a specific reason for schema-level enforcement: a library bug or out-of-band SQL write cannot bypass the check. That reasoning does not transfer to a library-side check. **Grounds resist the counterfactual.** Principled, though not strongly — a CHECK constraint isn't an option (CHECK can't reference other tables), so the trigger is the only schema-level path. The Planner could have noted that explicitly.

### A3 + C3 — Replace-on-source-match → **STRONG PRINCIPLED** (this is the load-bearing test)

The decisive counterfactual: had Critic proposed **no-op-on-collision** (preserve existing, ignore re-retain), would the Planner have defended replace as it did?

The R2 defense names the no-op path explicitly and rejects it: *"No-op feels safer but silently divorces the bank from caller intent. If the caller is re-retaining because their spine prompt changed, no-op leaves the bank stale with no signal. Replace gives the caller agency."* (§7, "Why not no-op").

This is a substantive principled argument that **requires the replace semantics**. The "source-match invariance" reading is there in §7's three rules: same hash + same source = replace (caller's bookkeeping is consistent, allow rewrite); same hash + divergent source = raise (source-of-truth audit trail must not silently merge). The Planner's defense does NOT transfer to no-op; it specifically refutes it.

This is the catch where R2 most clearly does load-bearing work. The locked behavior is not the deferential one and the rationale stands on its own.

### A4 — Restore `sweep_passes` → **STRONG PRINCIPLED**

R1 collapsed `sweep_passes` into `sweeper_state`; R2 restores it and names R1's mistake directly (verdict §15: *"the sweep_passes collapse in R1 was a P12-flavored simplification that lost the P14 history affordance"*).

The defense in §9 names three specific affordances the state cache cannot provide: history queries ("4 errors in a row?"), prune accountability ("what did it delete on 2026-05-10?"), and failure-mode debugging (state cache overwrites the last_error; the log preserves every error). These are concrete, P14-grounded reasons.

Counterfactual: had Critic proposed "keep `sweeper_state` only and add a JSONB `recent_errors` array column," the Planner would have had to refute it — and the §9 affordances do refute it (a bounded array loses history; an unbounded JSONB is just a worse table). **Grounds resist.** Principled.

### A5 — `pg_advisory_xact_lock` → **DEFERENTIAL** (and that's fine)

This is fifteen lines of runner code. The defense is one sentence: two concurrent invocations serialize. No principled counterfactual exists; the only alternative would be "skip serialization, accept the race," which is uncontroversially worse. Adoption is mechanical and correct. Deferential is the honest verdict here.

### A6 — COALESCE on scores → **DEFERENTIAL**

Mechanical SQL fix. Three `COALESCE(..., 0)` calls. JSON contract honors A6. There's no principled defense to write — it's "yes, NULL in JSON is worse than 0." Adoption is correct and deferential.

### C1 — `$rrf_k` parameterization → **MILDLY PRINCIPLED, BUT POSSIBLY OVER-PRESCRIBED IN R1**

See Test 2. The defense in §6 names the R1 incoherence (literal `60` in SQL while prose claimed "configurable"). The fix removes the incoherence. The defense transfers — Critic could have demanded "lock at 60 as a true compile-time constant, drop the prose claim" and the Planner would have adopted that just as easily. **Some grounds transfer.**

### C5 — `COMMENT ON COLUMN` → **DEFERENTIAL**

Ten lines of DDL. P15-at-the-DDL-surface is uncontroversial. Adoption is mechanical and correct.

### Summary

| Adoption | Verdict |
|---|---|
| A1 (fts_config drop) | Strong principled |
| A2 (dim-check trigger) | Principled |
| **A3+C3 (replace-on-source-match)** | **Strong principled (load-bearing)** |
| A4 (sweep_passes restored) | Strong principled |
| A5 (advisory lock) | Deferential (correct) |
| A6 (COALESCE) | Deferential (correct) |
| C1 ($rrf_k parameterization) | Mildly principled (see Test 2) |
| C5 (COMMENT ON COLUMN) | Deferential (correct) |

The Planner did real work on A1, A2, A3+C3, A4. The deferential adoptions (A5, A6, C5) are deferential because the alternatives are uncontroversially worse — that's the right shape, not deference-as-smoothing.

---

## Test 2 — Did R1 Critic over-prescribe on C1?

**Yes, mildly.**

C1 was named in R1 as a "must-fix" with the rationale: *"current state is incoherent between config-knob claim and literal-60 SQL."* The incoherence was real, but the prescribed fix (parameterize `$rrf_k`, expose as `Memory(rrf_k=60)`) was one of two valid resolutions. The other — commit to k=60 as a true library-build-time constant, drop the prose claim about configurability — would have been simpler:

- One fewer prepared-statement parameter to plumb.
- One fewer init kwarg on `Memory(...)`.
- No risk of someone setting `rrf_k=0` and dividing by something undefined.
- The literature (Cormack et al.) treats k=60 as a defensible-everywhere constant; per-deployment tuning is not load-bearing for v0.1.

R1 Critic prescribed the **more flexible** of the two resolutions without naming that flexibility was the choice. The honest R1 framing would have been: *"either parameterize OR commit to constant; pick one."* The Planner adopted the parameterized path, which is fine but not load-bearing. The right call would have been to leave the choice to the Planner with both paths named.

This is a minor over-prescription, not a framing error. The R2 stance carries the parameter cleanly; the cost is one extra knob. Reflection-not-rework: name in plan-v2 §"Known limitations" that `rrf_k` exposure is hedge-shaped, and v0.2 may collapse to a constant if no caller exercises it.

---

## Test 3 — Did R2 make anything worse?

### Event-table count: 5 event + 1 state cache

Going from R1's 4 event tables to R2's 5 (with `sweep_passes` restored) is **not bloat.** The append-only history serves a different question than the state cache (§9 names three: history, prune accountability, failure-mode debugging). The bloat reading would only land if `sweep_passes` and `sweeper_state` carried overlapping columns redundantly — they don't. The state cache is one row per `(bank_id, corpus_path)`, denormalized for dashboard reads; the history is row-per-pass, indexed for time-bounded scans.

The R2 §9 explicitly names that R1 was wrong to collapse them. Verdict: not worse. Slightly more DDL (~80 LOC migration); affordance-real.

### Dim-check trigger PL/pgSQL — test fixture complication

**Mild concern, named here for proviso.** Introducing a plpgsql function into `0001_initial.sql` means CI Postgres containers need `plpgsql` enabled — which is default in every official image, so practically zero friction. But test fixtures that use **stripped-down Postgres** (e.g., `embedded-postgres` for unit-test isolation) may not ship plpgsql by default. Worth a one-line check in the test plan that the chosen test container (testcontainers-python on `pgvector/pgvector` is the obvious choice) includes plpgsql.

Not load-bearing. Document in plan-v2 §7 test infrastructure.

### `COMMENT ON COLUMN` coverage

R2 adds COMMENT to:

- `banks.embedding_dim` — dim contract documented
- `documents.original_text` — P5 anchor documented
- `documents.content_hash` — re-retain pointer documented
- `memory_items.content` — spine (P1) documented
- `memory_items.original_chunk` — P5 documented
- `memory_items.llm_generated` — caller-wins audit documented

**Missed columns where comment would have load-bearing read:**

- `memory_items.embedding` — the semantic side of the spine. Worth: *"Vector representation of `content` (the question form) in the per-bank embedding space; dim enforced by trigger; never a vector of `original_chunk`."*
- `memory_items.content_tsv` — generated column from `content`, not from `original_chunk`. A contributor could reasonably wonder which side is FTS-indexed; the SQL answers it, but a COMMENT seals it.
- `documents.source` — given the §7 re-retain rules turn on source identity, the column carrying the rule deserves a one-liner: *"Source identifier (filesystem path, URL, opaque caller string). See re-retain semantics for collision behavior."*

These three are P15-shaped. Not blockers; **proviso candidate** for distillation.

---

## Test 4 — NEW silently-absent walk (P1–P15)

R1's silently-absent catches landed; R2 doesn't open new gaps but a few tensions deserve naming.

### P4 (caller wins on every override) — **GAP NAMED**

R2 exposes `rrf_k` only at `Memory(rrf_k=60)` init, not at per-`recall()` call. Justified as P12 (config noise stays off the verb signature). But P4 says **caller wins on every override** — the principle reads as per-call, not per-instance.

The tension: `Memory(rrf_k=...)` is one-knob-per-Memory-instance. If two callers in the same process want different RRF behavior (e.g., one wants k=10 for sharper top-of-list, another wants k=200 for flatter fusion), they need two `Memory` instances. That's awkward for the "caller wins" reading.

**Not a must-fix** — the realistic-frequency of per-call RRF override is ~zero, and the §6 hint that `rrf_k` is a prepared-statement parameter means future plumbing to `recall(rrf_k=...)` is trivial. But the §6 framing claims P4 is honored when actually P12 is being held above P4 here.

Honest fix: name in plan-v2 §"Known limitations" that `rrf_k` is per-instance not per-call, and that this is a P12-over-P4 trade for v0.1 ergonomics. One sentence. **Proviso candidate.**

### P5 (no truncation) — **HONORED, but re-retain deserves one line**

Replace-on-source-match deletes `memory_items` rows and rewrites them. The question: does this silently truncate?

**No.** `documents.original_text` is preserved (UPDATE not DELETE on the documents row). `memory_items` are derived projections of `original_text` through the spine; deleting derived rows when the derivation changes is not truncation. The principle protects source bytes (P5 anchor is `documents.original_text`); the spine outputs are by-construction regeneratable from the source plus the prompt.

Worth naming explicitly in §7 (one line): *"Replace-on-source-match preserves `documents.original_text` and `documents.id`; only derived `memory_items` rows are deleted-and-rewritten. P5 holds because original bytes are never lost."*

### P11 (tests over prose) — **MOSTLY TESTABLE; ONE HAND-WAVY**

R2 adds four tests in §11. Walking each:

1. **Dim-check trigger** — *"insert a vector of wrong dim, assert exception."* Clean. One-liner pytest. ✓
2. **Re-retain semantics, three cases** — same source replaces; divergent source raises; NULL-source upgrade. All testable with deterministic fixtures. ✓
3. **`scores` numeric-not-null** — *"recall on a bank where lexical returns zero results, assert `scores.lexical == 0` not `None`."* Testable: retain with content that won't match a chosen lexical query. ✓
4. **Advisory-lock concurrency** — *"spawn two concurrent migration runners, assert no race."* **Hand-wavy as written.** What does "no race" mean operationally? Two processes calling `run_migrations` against a fresh database — desired observable: exactly one set of inserts to `prospecta_schema_version`, both processes return cleanly, no duplicate-key errors. This needs a timing harness (subprocess + barrier or `multiprocessing` with shared event) and is genuinely fiddly to make non-flaky.

**Proviso candidate:** sharpen the advisory-lock test description in §11 before distillation. Either:
- (a) commit to a specific harness shape (subprocess + sync barrier, assert exactly one process inserts the rows), or
- (b) downgrade to "tested in operator notes, not CI" if the test is flaky enough that CI gain doesn't justify cost.

P11 says tests over prose — a hand-wavy test description is prose-shaped.

### P12 (honest config surface) — **HONORED, with R2 hedge noted**

Eight tables, no Alembic, no orphan `fts_config`, `rrf_k` parameterized-not-exposed-per-call. R2 holds P12 cleanly. The `rrf_k` hedge (parameterized at SQL, single knob at Memory init) is P12-shaped: minimal surface, real configurability when needed.

### P15 (spine documented) — **STRENGTHENED**

R1 honored P15 in README only; R2 puts COMMENT ON COLUMN at the SQL surface. As noted in Test 3, three more columns (`memory_items.embedding`, `memory_items.content_tsv`, `documents.source`) would benefit from one-line comments. Proviso, not blocker.

### Other principles — no new silently-absent

P1, P2, P3, P6, P7, P8, P9, P10, P13, P14: walked, no new gaps in R2. The structural shape is the same as R1 with the must-fixes folded in; nothing new fell out of frame.

---

## Provisos (small, none blocking)

1. **Name the C1 over-prescription** in plan-v2 §"Known limitations": `Memory(rrf_k=...)` is a hedge-shaped exposure; v0.2 may collapse to a constant if no caller exercises it.
2. **Add COMMENT ON COLUMN** for `memory_items.embedding`, `memory_items.content_tsv`, `documents.source` — three more lines of DDL, completes the P15-at-DDL surface story.
3. **Name P4-over-P12 trade** for `rrf_k`: it's per-instance, not per-call. One sentence in plan-v2 §"Known limitations."
4. **Sharpen the advisory-lock test** in §11: commit to a specific harness shape or downgrade out of CI. Prose-shaped tests violate P11.
5. **One line in §7** affirming P5 holds across replace-on-source-match: `documents.original_text` preserved; only derived rows rewritten.
6. **One sentence in plan-v2 §7 test infra**: confirm chosen Postgres container ships plpgsql (default for `pgvector/pgvector`, worth naming).

None of these reopen off-limits. All six are distillation-time tightenings, not R3 material.

---

## Closing

R2 is solid. The orchestrator-curated catches were real and are honored at the right altitudes (schema for dim-check, comments, `sweep_passes`; runner for advisory lock; library for replace-on-source-match + `rrf_k` config). The load-bearing test (Test 1, A3+C3 counterfactual) lands on **strong principled**: replace-semantics is defended against no-op explicitly and the defense does not transfer.

R1 Critic over-prescribed mildly on C1 — both flexibility-paths should have been named. Acknowledging here rather than relitigating.

R2 did not make anything worse. The `sweep_passes` restoration is affordance-real, not bloat. The dim-check trigger is the only schema-level enforcement path available. The COMMENT ON COLUMN coverage is good and three additions complete the P15-at-DDL story.

The four new tests in §11 are mostly clean; one (advisory-lock concurrency) is prose-shaped and deserves a sharper description before distillation.

**APPROVE_WITH_PROVISO.** Fold the six small provisos into `schema-stance.md` distillation. No R3 needed.

⚒️ Forge — Critic R2, prospecta schema stance, 2026-05-18.
