# Prospecta v0.1 — Architect Review (Round 2)

**Role:** Architect (Round 2)
**Reviewer:** Forge ⚒️
**Date:** 2026-05-18
**Subject:** `stance-planner-v2.md` — revised implementation plan in response to Round 1 Architect (A1–A8) and Critic (F/O/M/S) catches.

---

## Verdict

**APPROVE_WITH_PROVISO.**

Round 2 closes the eight A-items I raised and the orchestrator-curated M/O/F catches from the Critic. The plan is materially stronger: budget is honest, the 2×2 matrix proves what the original test only suggested, the concurrency contract and llm failure-mode table are now first-class spec sections rather than ambient assumptions, and the plugin section's item (f) flipped from defensive negation to positive inheritance. None of the Round 1 concerns is left as IGNORED. The Planner's self-graded honesty about what *isn't* in v2 yet (M0 sheet content unauthored, 7th `index.py` site unenumerated, README undrafted) is the right posture — these are correctly bumped to first-deliverable-of-the-milestone rather than papered over.

The proviso is small and named below: three fold-ins for distillation that tighten v2 without warranting Round 3.

---

## Round 1 concern status (A1–A8)

| # | Status | Citation |
|---|--------|----------|
| A1 — `index.py` 7 entanglement sites; M2 light | **ADDRESSED** (gate, not in-line fix) | "(a) promoted a port-decision sheet to a new **M0 gate** before scaffolding; (b) widened M2 from 2 days to 3.5 days; (c) sub-dim 11 below enumerates all 7 sites with explicit disposition decisions (lean: delete log-API surface for v0.1)." |
| A2 — Prompts are Cookie-shaped, port budget light | **ADDRESSED** | "reframed as **prompt translation**, not port. Added 1.5 days to M1 (now 2.5 days). Prompts renamed (see O2). Per-prompt translation deltas spec'd in §M1." |
| A3 — `formulate_queries` fallback path unspec'd | **ADDRESSED** | §14 row: "formulate_queries (JSON-mode parse failure) … **Falls back** to single-query degraded mode: `[Query(text=message)]`. Tracer event records `parse_fallback=True`. Logged as warning." |
| A4 — Bilateral test does not prove bilateral; needs 2×2 | **ADDRESSED** | §7 introduces the OFF/OFF, ON/OFF, OFF/ON, ON/ON matrix with assertions "1. `both-on > write-only`… 2. `both-on > read-only`… 3. `both-on > off-off`… 4. `off-off` fails on the kelly-birthday-arc case." |
| A5 — `_debug_use_index_text` kwarg leaks test surface | **ADDRESSED** | "removed from `Memory.__init__`. Test-only paths live in `prospecta/_test_helpers.py`, not imported from `__init__.py`." Confirmed in §4.4 — debug kwargs absent from the locked signature. |
| A6 — `embedding_model` warn-on-mismatch chroma subtlety | **ADDRESSED** (harder than I recommended; honest) | "v0.1 raises `NotImplementedError` if `embedding_model` is user-supplied. Real support pushed to v0.2." This is option (b) from my A6 — the Planner picked the honest path over Critic's softer "warn and accept default." |
| A7 — Shared chroma client thread-safety | **ADDRESSED** | §13: "all chroma writes — from sweeper, from `retain`, from `index_single_file`, from `prune_stale` — go through a single `threading.RLock` owned by the `Memory` instance." |
| A8 — Line-parser is a P6 violation; JSON-mode to v0.1 | **ADDRESSED** | §8: "`formulate_queries` uses JSON mode. Output schema: `{\"queries\": [{\"text\": \"...\"}, ...]}`. Parsed cleanly; no regex line-parsing." Critic's stronger reading prevailed over my "three more test cases" softpedal. Correct. |

**No A-item left as PARTIAL or IGNORED.** A1 is "addressed by gate" not "addressed by enumeration" — that is the architecturally correct move (the sheet can't be authored without reading the file end-to-end, and the M0 deliverable is exactly that read). The Planner explicitly names the residual risk in §15.1 and §Open question.

---

## New checks for Round 2

### 1. M0 port-decision-sheet gate — executable as a gate?

**Mostly yes, with one tightening needed.** §6/M0 spec: "every animus-coupling site … site name, lines, disposition (delete / shim / generalize / preserve), rationale." Deliverable is concrete (a `.md` at `docs/design/prospecta/port-decision-sheet.md`); review surface is named (Critic blesses); budget is 0.5d. The four-verdict disposition vocabulary is the right shape — it forces a decision per site rather than letting site-by-site judgment leak into M2 coding.

What's missing: the sheet is described by **process** but not seeded with even a **partial table** in v2. The 7 sites Architect named are *referenced* but only six are enumerated in v2's §15.1 ("I have enumerated 6 … the 7th likely lives in chunker-coupling or `index_status()`"). A reviewer cannot tell whether the gate will land in 0.5d or 1.5d without seeing the table-shape worked once. Lean: M0 should ship a **seeded** table in v2 itself — the six sites the Planner has identified, with leans, leaving the 7th as `TBD-by-grep`. That converts M0 from "0.5d of unknown shape" to "0.5d of finish-the-table." Not blocking; fold-in for distillation.

### 2. 2×2 bilateral matrix — does it actually prove bilateral?

**Yes.** The two load-bearing assertions are correct:
- `both-on > write-only` proves the **read-side** spine is load-bearing (if read-side were decorative, write-only would equal both-on).
- `both-on > read-only` proves the **write-side** spine is load-bearing (mirror).

The OFF/OFF and the `both-on > off-off` assertion (retained from Round 1) are the floor and the ceiling. The corpus-widening cost (~8 files, ~1hr design) is acknowledged. The fact that ON/OFF and OFF/ON each show *partial* improvement is what makes the discriminator clean — and the §15.5 open question correctly flags that corpus-tuning to land both partials is the genuine risk.

One small thing: the assertions are stated in §7 as inequalities on "a discriminating metric (top-1 hit rate over a small query set, or NDCG@5)." Lean: pick one (top-1 hit rate is simpler for v0.1; NDCG@5 is more sensitive but needs graded relevance judgments). Either is defensible; ambiguity in v2 is a 5-minute fold-in.

### 3. `LLMCallable` widened with `json_mode=True` — composes cleanly?

**Yes, with a type-signature quibble.** §4.4 declares:

```python
LLMCallable = Callable[[list[dict]], str] | Callable[[list[dict], bool], str]
```

A union of arities is unusual. Mypy will accept it; readers will trip on it. The §8 adapter examples use kwarg call (`llm(messages, json_mode=False)`), which is the right shape. Lean: declare the type as `Callable[..., str]` with a docstring contract, or use a `Protocol` with an optional `json_mode` kwarg. The union-of-arities formulation will produce confusing type errors when adapters that don't take `json_mode` are passed.

Hermes adapter composes cleanly — `ctx.llm.complete_structured` is the right tool and the §8 wrapper is ~5 lines as advertised. The plugin-llm-access doc I checked supports `complete_structured(response_format=...)`, so no surprises there. Animus's `invoke_prompt` would need a `json_mode` branch in its adapter; that's the caller's concern, not the library's.

### 4. Tracer / observability primitive — payload schemas implementable?

**Yes, all six events are implementable from existing call sites.** The payload schemas in §12 use cheap inputs (chars, counts, ms) that don't require new instrumentation in the ported code. Six call sites at ~5 lines each is the ~30 LOC budget claimed; that's realistic.

Two notes:
- The Critic's M1 catch was specifically about **per-query → results mapping** (`RAGResult.trace: dict`). The Planner answered with **event-based tracing** (boundary hooks). These are different mechanisms — events tell you *when* something happened; the per-query mapping tells you *which query matched which document*. The latter is what you need to debug "spine returned the wrong doc." Lean: add `RAGResult.queries_to_results: dict[Query, list[RecalledMemory]] | None = None`, populated when `recall_synth(..., trace=True)`. Tracer events and result-trace are sibling tools, not substitutes. Fold-in for distillation.
- `llm_call` event's `prompt_name` field implies the library passes a prompt name through to the tracer. That requires threading a string from `_index_text`, `_formulate`, `_rag` to the tracer call site. Implementable, but mention it in M1's milestone deliverable so it doesn't surprise the porter.

### 5. Concurrency contract — does the RLock actually solve the races?

**Solves retain-vs-retain and sweeper-vs-retain. Does not solve multi-process.** §13 is precise about what it covers: "A `Memory` instance is safe to call from multiple threads. … single per-instance `threading.RLock` around all chroma writes." Reentrant lock means no deadlock between sweeper (one thread) and retain (another thread) within a single process. Correct.

What's not covered:
- **Multi-process** access to the same `index_dir`. Two `Memory` instances in two processes (e.g., a cron + a notebook) share the same SQLite-backed chroma store but not the same Python lock. Chroma 1.x's internal SQLite locking handles the database-level race, but the file-on-disk race in `retain_dir` (filename collision on auto-generated names) is not covered. v0.1's honest answer is "single-process assumption." §13 says "If the caller wants true write parallelism, that's an explicit v0.2 design" — fold a sentence in clarifying that multi-process is also v0.2.

No new deadlock risk. RLock is reentrant; sweeper and retain are on different threads, and chroma reads (which `recall` uses) are unlocked. No path acquires the lock, then re-enters via a non-RLock-aware reentrancy, then waits on itself.

### 6. `llm` failure-mode table — any wrong propagation rules?

**No wrong rows.** The table at §14 is correct. The two that matter for Hermes:

- Row "Sweeper `index_single_file` → llm raises → skip + warn + continue." This is what Hermes prefetch semantics expects: best-effort, never crash the sweep. Correct.
- Row "`recall_synth` → formulate_queries LLM raises → Exception propagates." This is the *opposite* of Hermes plugin prefetch semantics ("prefetch failures are caught and logged"). But §10 item (f) handles this correctly: "Error semantics propagate (inherited from M4 failure-mode spec, §14): `retain` failures bubble to the agent (caller decides retry); `prefetch` failures are caught and logged (per Hermes plugin contract, prefetch is best-effort)." The plugin wraps; the library propagates. That's the right division.

One row could be clarified: "`retain` (LLM generates index_text) → llm raises → Exception propagates to caller. The file is NOT written." The "NOT written" guarantee implies an order-of-operations: generate `index_text` *before* writing the file. If the port from `animus/memory/index_text.py` writes the file first then generates index_text, this row is a lie. Quick check: animus's `retain` flow does index_text → write → upsert. Preserving that order in prospecta is what makes the row honest. Worth noting in M3's deliverable as an invariant.

### 7. README content spec — delivers on P15?

**Yes.** §11 spec puts the spine in the lead-paragraph (~200 words), positions against LlamaIndex/Khoj/mem0 explicitly, and ships three API-level code examples plus honest caveats. P15 says "leads with bilateral synthesis. It says *this is what makes us different from LlamaIndex / Khoj / mem0*. Not buried in §4 of an architecture doc." §11 honors this structurally.

The "Round-1-of-prose" question: is the structure executable? Yes — six numbered sections, each named, each scope-bounded, total ~600 words, 0.5d budget. A README this size is one focused session of writing. The §15.4 open question correctly flags that the competitor-takedown framing risks reading poorly — Donald reviewing the draft is the right gate.

### 8. Budget growth 7.5d → 12.5d (+67%) — honest or padded?

**Honest.** Per-milestone reconciliation:
- M0 (NEW): +0.5d → maps to A1.
- M1 widened 1d → 2.5d (+1.5d): maps to A2 (prompt translation) + O2 (rename) + JSON-mode instruction in prompt.
- M2 widened 2d → 3.5d (+1.5d): maps to A1 (7-site port work).
- M5 widened 0.5d → 1d (+0.5d): maps to A8 (JSON mode + fallback path).
- M6 (sweeper) widened slightly: maps to tracer wiring (M1-Critic catch).
- M7 widened 1d → 1.5d (+0.5d): maps to A4 (2×2 corpus design).
- M8 (NEW): +0.5d → maps to M7-Critic / P15 (README).

Total: 0.5 + 1.5 + 1.5 + 0.5 + ~0 + 0.5 + 0.5 = **+5.0d**, matches the claimed delta. No phantom days. Each added day maps to a named Round 1 catch.

### 9. Walking P1–P15 against the new frame — silently absent?

- **P1 (spine is the spine):** v2 strengthens via 2×2 matrix. ✓
- **P2 (standalone first):** unchanged. ✓
- **P3 (LLM as injected callable):** v2 widens signature with `json_mode`; still provider-blind. ✓
- **P4 (caller wins on override):** v2 restores per-call prompt overrides (Critic M2). ✓
- **P5 (full content, no truncation):** unchanged. ✓
- **P6 (let the LLM cook):** v2 retires the line-parser. Critic's win. ✓
- **P7 (single write path):** preserved in M2 disposition table. ✓
- **P8 (embedded chroma zero-config):** unchanged. ✓
- **P9 (plugin is thin adapter):** §10 reframed positively; still ≤2 pages. ✓
- **P10 (surface adjacent mechanisms):** v2 leans on `complete_structured` rather than reimplementing JSON-mode. ✓
- **P11 (tests over prose):** each milestone still gates on tests. ✓
- **P12 (honest config):** unchanged. ✓
- **P13 (prompts caller-overridable):** v2 restores per-call. ✓
- **P14 (sweeper is safety net):** v2 §13 sharpens. ✓
- **P15 (spine in README):** v2 M8 + §11 delivers. ✓

**No principle silently dropped.** The only structural-coherence concern is what the Critic M1 originally caught and v2 partially answered: spine *debugging* (per-query → results mapping) is now event-based at boundaries, but not result-shaped at the call return. That's a P11-shaped gap (tests over prose; debugging requires structure) more than a P-principle violation. Captured as A9 below.

---

## New concerns (A9, A10)

### A9 — Tracer events ≠ per-query result mapping; spine misfires remain hard to debug

**What:** Critic M1 asked for `RAGResult.trace: dict | None` carrying per-query → results map. v2 answered with an event-based `Tracer` callable that fires at boundaries (`recall`, `formulate_queries`, `llm_call`, etc.) with payloads like `{"n_queries": int, "n_results": int}`. These are sibling tools, not substitutes. When `recall_synth` returns the wrong document, the tracer tells you "3 queries → 5 results in 240ms"; it does *not* tell you which query matched which document or at what rank.

**Why it matters:** P1 says the spine is *the* differentiator. Spine debugging — "why did this corpus + this question retrieve the wrong doc?" — is the v0.1 caller's most important diagnostic loop. Event tracing supports metrics and observability; per-query-result mapping supports the actual debugging question. Without it, debug-by-spelunking returns through the chroma metadata layer.

**Fix:** Add `RAGResult.queries_to_results: dict[Query, list[RecalledMemory]] | None = None`, populated when `recall_synth(..., trace=True)`. ~10 LOC, no signature break (new field on existing dataclass). Fold into M4 deliverable.

### A10 — `LLMCallable` union-of-arities will produce confusing type errors

**What:** `Callable[[list[dict]], str] | Callable[[list[dict], bool], str]` is a union of two callable signatures with different arities. The §8 adapter examples use `json_mode` as a kwarg, not a positional. Type-checkers will accept the union but produce unhelpful errors when an adapter that doesn't accept `json_mode` is passed — the error will read "incompatible callable" not "missing json_mode parameter."

**Why it matters:** Adapter ergonomics. The library's audience is callers wiring up their own LLM; a confusing type signature at the boundary is the friction point that makes a five-line adapter feel like ten.

**Fix:** Use a `Protocol`:

```python
class LLMCallable(Protocol):
    def __call__(self, messages: list[dict], *, json_mode: bool = False) -> str: ...
```

Cleaner static typing, kwarg-only `json_mode` matches the §8 examples, no union. ~3 LOC swap in `_types.py`. Fold into M1.

---

## Load-bearing decisions newly committed in v2

1. **M0 as a Critic-gated artifact** — port decisions land in writing before any code moves. The right architectural shape; converts "we'll figure it out as we go" into "we'll figure it out, write it down, and have it reviewed."
2. **JSON mode in v0.1, not v0.2** — the `LLMCallable` widening is small but real. The plan now depends on every adapter supporting JSON mode (Hermes has it; OpenAI has it; raw Anthropic needs tool-use coercion, flagged in §15.2 as +10 LOC adapter complexity).
3. **`NotImplementedError` on `embedding_model`** — picks honesty over silent fallback. Means the locked dep surface does *not* gain sentence-transformers in v0.1, and v0.2 has a real migration story to write.
4. **Per-Memory `threading.RLock`** — committed as the v0.1 concurrency model. Multi-process is v0.2; this is the right call but should be documented.
5. **Per-call prompt overrides on three methods** — `retain`, `formulate_queries`, `recall_synth`. P13 is now fully honored.
6. **2×2 matrix is the ship gate** — v2 explicitly: "This 2×2 matrix passing is the v0.1 ship gate." Stronger lock than Round 1's single-axis test.

---

## Anything Round 2 made worse?

**Nothing structural.** Two ergonomic regressions worth naming:

- **`LLMCallable` type-union** (A10 above) is uglier than Round 1's single signature. Worth the JSON-mode capability; cleanly fixed by Protocol.
- **The "9 milestones" shape** (M0..M8) is closer to Critic's S2 anti-pattern (collapse adjacent milestones) than Round 1 was. v2 added 2 milestones; the gating value of M0 (gate-before-code) is real, but M8 (ship-the-README) is closer to a sub-task of M7 than a true milestone. Not blocking; the bureaucracy is mild.

Neither is structural. Both fold cleanly into distillation.

---

## Recommended changes (distillation fold-ins; no Round 3)

Ordered by tightening value:

1. **Add `RAGResult.queries_to_results` field** (A9). Closes the spine-debugging gap that Critic M1 originally raised. ~10 LOC, M4 deliverable.
2. **Swap `LLMCallable` union-of-arities for a `Protocol`** (A10). ~3 LOC, M1 deliverable.
3. **Seed the M0 sheet with the 6 sites you've identified**, leaving the 7th as `TBD-by-grep`. Converts M0 from "0.5d of unknown shape" to "0.5d of finish-the-table." Land in v2.5 before M0 opens.
4. **Pick one discriminating metric for §7** (top-1 hit rate or NDCG@5). 5-min decision.
5. **§13: add a sentence on multi-process** ("v0.1 assumes single-process access to `index_dir`; multi-process is v0.2"). Documents the limit honestly.
6. **§14: name the `retain` order-of-operations invariant** (generate index_text → write file → upsert). Makes the "file NOT written on llm failure" row honest.
7. **§12: add `prompt_name` threading note** to M1's milestone deliverable so the porter knows to thread it.

None of these warrants Round 3. All are five-minute fold-ins for a v2.5 distillation pass; the plan is shippable without them but tighter with them.

---

## Closing

Round 2 closed every A-item and the orchestrator-curated subset of the Critic's catches without introducing new structural problems. The two new concerns I'm raising (A9, A10) are ergonomic and fold cleanly. The plan is now what an implementation plan should look like: a port-decision sheet as a first-class gate, a 2×2 test that proves what the project's name claims, a concurrency contract that says what's safe and what isn't, a failure-mode table that reads like a spec rather than vibes, a tracer that supports production debugging, a README that ships on day one.

The single most valuable change between Round 1 and Round 2: the 2×2 matrix. The Round 1 test would have passed on a half-bilateral spine. The Round 2 test cannot. That is the v0.1 ship gate the principles always implied.

The single most architecturally honest change: the `NotImplementedError` on `embedding_model`. The Planner picked the option that ships a footgun-free v0.1 over the option that ships a footgun with a warning. Correct.

Approve. Fold A9, A10, and the §M0/§7/§13/§14 tightenings in distillation. Ship.

⚒️
