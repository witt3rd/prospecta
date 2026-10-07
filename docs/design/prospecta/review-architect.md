# Prospecta v0.1 — Architect Review (Round 1)

**Role:** Architect (Round 1)
**Reviewer:** Forge ⚒️
**Date:** 2026-05-18
**Subject:** `stance-planner.md` — implementation plan for prospecta v0.1

---

## Verdict

**APPROVE_WITH_RESERVATIONS.**

The plan is executable and the milestone order is sound. The five premise verifications are evidence-backed at file/line granularity, which is what an implementation plan should look like — not aspirational, not vague. The bilateral-synthesis test is specified concretely enough to write. The bounded plugin section honors the ≤2-page cap with all six required items.

The reservations are not about whether v0.1 can ship along this plan; they are about a small number of load-bearing places where the plan under-budgets work, under-specifies a contract, or carries an animus assumption into prospecta without flagging it. Each is named below as `A1..A8` with a concrete fix.

None of the concerns is structural enough to send back. All can be folded into the implementer's working notes before M1 opens.

---

## Strengths (what the plan got right)

- **Premise verifications carry file/line evidence**, not assertions. The grep-of-39-hits-isolating-three-sites for `memory/index.py` is the right shape for "is this really portable?" The chromadb `PersistentClient` vs `HttpClient` claim is verified by API-surface inspection, not by hope.
- **Vertical-slice port order is correct.** M2 standing up classical-RAG end-to-end before M3 layers the write-side spine and M5 layers the read-side spine honors P11 (tests as the lock) better than bottom-up would: every milestone closes with a runnable demo plus a passing test gate. Failure modes surface early.
- **The decoupling step for `formulate_queries` is named as its own milestone** (M5), not buried inside `recall_synth`. That's the right architectural move given the partial-verification on Premise 2.
- **Three-layer API composes cleanly.** `recall(queries=[...])` → `formulate_queries(message)` + `recall(...)` → `recall_synth(message)` is a real progression, not a fake one. Each layer is independently usable. Caller-wins-on-override (P4) is concrete: `retain(index_text=...)` and `recall(queries=[...])` both let the caller supply the spine-side artifact rather than have the library generate it.
- **`LLMCallable = Callable[[list[dict]], str]` is the right shape.** Hermes adapter is genuinely 3 lines; animus adapter is ~10; tests mock with `lambda messages: "..."`. The rejected alternatives (b/c/d) are correctly rejected for the stated reasons.
- **v0.1/v0.2 cut lines are explicit and named.** Async, JSON-mode, multi-corpus, observability, server lifecycle, embedding migration — each is fake-as-named, not hand-waved.
- **The bounded plugin section is actually bounded.** Six items, ~2 pages, with the `__init__.py` skeleton showing the shape (not the implementation), and explicit "what's NOT in this plan" enumeration so the implementer doesn't go scope-shopping.
- **Sweeper-down does not block hot path** is structurally correct (P14): `retain()` and `index_single_file()` are direct calls, not queued behind the sweeper. The sweeper is genuinely a drift safety net, not a critical-path component.

---

## Concerns

### A1 — `memory/index.py` is more entangled than the plan accounts for (port budget under-stated)

**What:** The plan claims `_extract_person_from_path` (line 78), an optional `path_contains` filter param, and "one `meta["person"] = person` assignment at lines 650-653" cover the animus-isms. A grep of the file shows `person` woven through *seven* distinct sites in the file, not three: lines 78, 210/224/248/271-273, 650-653, 736-742, 763-764, 780-822 (an entire `add_log_entry(person=...)` API surface with `entry_id = f"{person}__{ts}"`), 892, 907, 1037-1058, 1086-1087. The log-write and log-recall API of `SemanticIndex` takes `person` as a required positional parameter and uses it for entry-id construction and metadata filtering.

**Why it matters:** Prospecta has no concept of "person." The `add_log_entry` / log-recall surface is genuinely animus engagement-pipeline scaffolding, not generic indexing. The planner's 1-day budget for M2 covers the read-side `path_contains` work cleanly, but the *write-side* log API needs either generalization (`person` → optional `scope_key`) or deletion. If deletion: any code path in the rest of the file that calls these methods must be cleaned up. If generalization: the metadata schema changes and downstream `recall` filtering changes shape.

**Fix:** Before M2 opens, the implementer runs `grep -nE "person|log_entry|recall_log" memory/index.py` and produces a one-page port-decision sheet: for each `person`-related site, *delete / generalize / keep*. Budget M2 at **3 days, not 2** to absorb the log-API decision. Alternative: defer the log-write API to v0.2 entirely (prospecta v0.1 retains via file-on-disk + `retain_dir`, which doesn't need `add_log_entry`'s in-chroma-only path) and delete those methods in M2.

---

### A2 — `log-index.md` prompt is animus-bound at the *narrative* layer, not just the surface layer

**What:** The plan says "copy three prompts" and "edit `formulate-queries.md` to drop scope prefixes." Reading `log-index.md` directly: the prompt opens with `"This is from Cookie's (Animus's) conversation log with {{ person }}"` and binds `Them = {{ person }}`, `Me = Cookie/Animus`. Required Jinja2 variables are `content`, `person`, optionally `tags`. The entire framing is *conversation-log-between-Cookie-and-a-person*, which is the wrong frame for prospecta's general-purpose `retain(content, ...)`.

`formulate-queries.md` has the same issue at the example layer: the vocabulary-translation examples reference *Yaya*, *Kelly*, *Donald*, *the wizard*, *🖤 black heart pattern*, *judo strategy*. These are not idiomatic translation hints for a general retrieval library; they're animus's lived substrate.

**Why it matters:** P15 says the spine is the *only* thing prospecta exists to support. The two prompts that drive the spine are not "shipped as-is and tweaked"; they are load-bearing artifacts that need substantive prospecta-shaped rewriting. The plan budgets +30min for this in M1. Realistic is +2-3 hours per prompt with editorial care, plus a regenerate of the M7 LLM recordings if you ship the prompts later.

**Fix:** Reframe both prompts at M1 with prospecta's general frame: "you are generating index text for an arbitrary content corpus" and "you are formulating queries for a retrieval library." Replace person-bound examples with corpus-bound examples (a software-architecture corpus, a journal corpus, a meeting-notes corpus — three minimal generic examples). Document required Jinja2 vars per prompt explicitly in M1's deliverable. Budget M1 at **1.5 days, not 1**.

---

### A3 — `formulate_queries` decoupling (M5) is correctly identified but slightly under-scoped

**What:** The plan names the decoupling step correctly (drop `ClassifiedInbound`, `_QueriedToken`, proof-token system, `scope` axis, person-fallback). M5 budget is half a day. Reading `queries.py`: the LLM call is 5 lines, `_parse_scoped_queries` is 35 LOC and ports clean once scope-prefix-stripping is done. What's missing from the budget: the *fallback path* on LLM failure currently constructs `f"{classified.person} {content}"` as a single ScopedQuery. Prospecta has no person; the fallback shape needs replacement (likely just `[Query(text=message)]`). Not hard, but worth naming explicitly in the milestone.

**Why it matters:** Silent fallbacks are the kind of thing that compiles and ships but degrades the spine without the test catching it. M7 doesn't exercise the LLM-failure path.

**Fix:** Add to M5's test gate: one fixture where the mock LLM raises an exception, assert that `formulate_queries` returns `[Query(text=message)]` (or whatever the chosen fallback is) and logs a warning. Half-day budget still holds; just name the fallback shape in the milestone deliverable.

---

### A4 — Bilateral test is well-spec'd but does not isolate which *side* of the spine is load-bearing

**What:** M7 has two test cases: `spine_off` (both `_debug_use_index_text=False` and `_debug_use_formulate=False`) and `spine_on` (both True). The negative is asserted (spine-off retrieves the red herring) and the positive is asserted (spine-on retrieves the right document). What's not asserted: that **both** sides of the spine are doing work. If `index_text` is generated but `formulate_queries` is bypassed (or vice versa), does the test still pass? If yes, half the spine is decorative.

**Why it matters:** The whole P1/P15 frame is that *bilateral* synthesis is load-bearing — both write-side and read-side together. If the test only proves "at least one side helps," the spine's actual claim is under-tested. This matters more in the long run than for v0.1 shipping, but the test corpus is designed once and lives forever; making it isolate sides is cheap to do now.

**Fix:** Add two more test cases at M7: `spine_index_only` (write-side spine on, read-side off) and `spine_formulate_only` (write-side off, read-side on). Assert the document of interest is reached in *both* mixed configurations as well — if not, the corpus needs sharpening until it does. Worst case, the test produces a 2x2 matrix: off/off (wrong doc), index-only (right doc), formulate-only (right doc), on/on (right doc). That matrix is the actual proof the spine is bilateral. ~1 hour additional work in M7.

---

### A5 — `_debug_use_index_text` / `_debug_use_formulate` as kwargs on `Memory.__init__` leak test surface into production API

**What:** The planner flagged this in Open Question #5 and leaned toward a separate `_test_helpers.py` module. I agree with the lean. As written in the M7 test code, those kwargs sit on `Memory.__init__` and are reachable from production code paths.

**Why it matters:** "Test-only" kwargs on the public constructor have a way of becoming "I needed to debug something in production so I set this flag" within six months. The plan's `__init__` signature is otherwise locked; leaking debug kwargs there violates the locking.

**Fix:** Move spine-toggle to a private helper. One option: `prospecta._test_helpers.build_memory_with_spine_toggle(...)` that constructs a `Memory` and monkey-patches its `_index_text._generate_index_text` and `_formulate.formulate_queries` to passthrough/no-op. Test imports from `_test_helpers`; production never sees it. The locked `Memory.__init__` signature does not change.

---

### A6 — `embedding_model` warn-on-mismatch contract is named but not specified

**What:** Plan says v0.1 "warns on mismatch when `Memory(embedding_model=...)` differs from collection metadata." The fix-the-warning detail is right ("specify the warning's exact message and where it fires") and acknowledged in §"what an ideal plan would have." But the mechanics matter for whether the warning fires at all: chromadb 1.x stores embedding-function metadata on the collection. If a user passes `embedding_model="sentence-transformers/all-mpnet-base-v2"` to prospecta but chroma's collection was built with the default onnx all-MiniLM-L6-v2 and `Memory.__init__` does not actually pass an embedding function down to `get_or_create_collection`, the user's parameter is silently ignored and the warning never fires (because chroma is happily using the default and the collection metadata still matches that default).

**Why it matters:** "Expose with warn-on-mismatch" is correct as policy, but the implementation has a real subtlety. If implemented naively (just store `embedding_model` on `Memory` and compare to chroma's metadata at construction), the comparison only works *after* a collection exists. First run with `embedding_model="X"` against a fresh index will silently set up onnx-default and the warning never fires — until run #2 where the user passes "X" again and now `get_or_create_collection` returns the existing collection whose metadata says "default."

**Fix:** Either (a) actually pass an `embedding_function` down to chroma when `embedding_model` is supplied (which means depending on sentence-transformers explicitly, not transitively — affects the locked dep surface), or (b) explicitly defer real `embedding_model` support to v0.2 and have v0.1's parameter raise `NotImplementedError("custom embedding_model deferred to v0.2; v0.1 uses chromadb default")`. Option (b) is honest and respects the locked dep surface. Recommend (b).

---

### A7 — `chromadb` thread-safety contract between sweeper and hot path is undocumented

**What:** Plan says sweeper is a daemon thread, hot-path `retain()` is a direct call. Both can call `index_single_file()` simultaneously. Chromadb 1.x `PersistentClient` is generally thread-safe for concurrent reads/writes on the same client instance, but the plan does not state that `Memory` holds a *single* chroma client instance (vs. constructing one per call). If two clients race against the same SQLite-backed persistent store, the failure mode is silent corruption rather than a clean error.

**Why it matters:** Sweeper-doesn't-block-hot-path (P14) is correct in spirit. The implementation detail is that sweeper and hot path must share the same client. Not stating this is the kind of omission that produces a hard-to-reproduce intermittent failure six months in.

**Fix:** Add one sentence to the sweeper sub-dimension: "`Memory` constructs and holds a single chroma client instance; `_sweeper.py` and all hot-path methods share it. Chroma client access is thread-safe in 1.x; we rely on chroma's internal locking, not our own." Add a test in M6 that exercises concurrent retain + sweep on the same file. ~30 min addition to M6.

---

### A8 — JSON-mode deferral is correct but the line-parser is the weakest link in the spine

**What:** Plan defers JSON-mode to v0.2 on the grounds that line-parsing handles `formulate_queries` output. Looking at `_parse_scoped_queries`: it strips code blocks, splits on newlines, drops comment-shaped lines, and regex-matches `[scope] text`. Drop the scope prefix and you have a line-list. This is fine *for the canonical case*. It is brittle in three documented cases (per the prompt itself): when the LLM emits prose preamble before the list, when it numbers the list (`1. query one`), or when it wraps in unexpected delimiters.

**Why it matters:** The bilateral test (M7) uses recorded LLM responses, so it will not surface parser brittleness. Production callers using different LLM backends (animus uses gemini-flash; the Hermes plugin will get whatever `ctx.llm.complete` defaults to) may hit edge cases the parser does not handle.

**Fix:** Two options. (Cheap) Add to M5's test gate three additional mock-LLM responses that exercise the brittle cases (numbered list, prose preamble, no code block). Assert parser returns reasonable queries or a clean fallback. (Expensive) Bring JSON-mode into v0.1 via `complete_structured` (already documented as available on `ctx.llm`). I recommend cheap: the structured-call path adds adapter complexity and reopens the `LLMCallable` shape question. The locked shape is correct for v0.1; the line-parser just needs three more test cases.

---

## Load-bearing decisions validated

These I checked against the constraints and concur with the planner:

- **Port order is vertical-slice, not bottom-up.** Right call. M2's classical-RAG-end-to-end gate is the earliest possible escape hatch for "the port has a hidden coupling we didn't see."
- **`LLMCallable = Callable[[list[dict]], str]`.** Correct shape. JSON-mode tension is real but addressable via test discipline (A8), not API change.
- **`Memory` is the single public class.** Correct. Internal modules underscore-prefixed; public surface is one import.
- **Caller-wins-on-override is preserved at all three layers** (`recall(queries=...)`, `retain(index_text=...)`, `prompts_dir=...`). Concrete in signatures.
- **chromadb pin `>=1.0,<2.0`.** Correct. Animus's pin is `>=1.0.0`; chroma 1.x has been stable; 2.x is unreleased. Tighter pinning (e.g., `>=1.2,<1.3`) would gain nothing and lose patch-level updates.
- **Argparse, not Click/Typer.** Correct for a 4-subcommand CLI. Zero-dep wins.
- **Sweeper as daemon thread + `threading.Event`.** Correct given prospecta has no async surface in v0.1. Adding asyncio for the sweeper alone bifurcates the library.
- **Three-prompt baseline + per-call override (P13).** Correct. Override is the answer for callers who want different prompts; no in-library prompt-tuning loop.

## Load-bearing decisions to reconsider

- **`index_dir` defaulting to `~/.local/share/prospecta/`.** Plan implies this in the CLI. The CLI's `PROSPECTA_HOME` env var is correct. But `Memory()` (library use) requires `index_dir` as a positional kwarg, no default — is that intentional? If yes, the README example should show it explicitly. If no, the library needs a default-resolution path consistent with the CLI. Recommend: keep `index_dir` required at the library layer; *the CLI* resolves a default from env. That keeps the library deterministic and the CLI ergonomic.

- **`_log_entry` API in `index.py`.** Per A1, decide *delete vs generalize* before M2. If deleted, the log-write and log-recall paths get pruned in M2. If generalized, the `Memory` public surface gains a method that does not currently appear in the locked signatures. Either way, decide at M2 boundary, not at M3.

- **Whether `retain()` requires `llm` to be set.** Per Open Question #1, the planner leans "lazy-load `llm`, raise loudly when used." I concur — for the CLI's `search`, `index`, `config` paths this is the difference between a usable library without an LLM and a hard requirement. Lock the lean.

---

## Interface soundness

### Between library and caller-supplied `llm`

The shape `Callable[[list[dict]], str]` composes cleanly with the three target adapters (Hermes, animus, notebook, test mocks). The cost is that:

1. **Provider-specific knobs (`max_tokens`, `temperature`) leak to the caller's wrapper.** Plan acknowledges this. Correct trade-off: the library should not negotiate inference; the caller wraps its preferred defaults. animus's `invoke_prompt(name, vars, max_tokens=8192)` pattern is preserved by the animus adapter setting `max_tokens=8192` inside its own wrapper.

2. **Role-stuffing happens in the library.** Prospecta's prompts use `system` + `user` role separation. The library produces the `[{"role": "system", "content": ...}, {"role": "user", "content": ...}]` list; the adapter passes it through. This is right.

3. **JSON-mode is not on the contract.** A8 above. The contract's purity is worth the v0.2 deferral. The integration test should pin the parser robustness so the deferral does not become a regression.

### Between library and Hermes plugin

The plugin adapter is genuinely thin (per Premise 4): one lambda wraps `ctx.llm.complete(messages=..., purpose="prospecta").text`. The bounded plugin section's `__init__.py` skeleton is correct in shape. Two small things to add:

- `prefetch()` returning the synthesis text is right, but the plugin's `prefetch_enabled=True` default plus the library's `recall_synth(message=...)` cost (one `formulate_queries` LLM call + one `recall_synth` LLM call) means **two LLM calls per user turn** in the Hermes plugin's default config. Hindsight's reference plugin retains turn observations and recalls via search (one call total). This is a real cost increase and worth flagging in the plugin section's config schema (the `prefetch: bool` default may want to be `false` for cost-sensitive callers, with an explicit doc note about the cost trade-off).
- `sync_turn` is `pass` (agent-driven retain). Good — matches the "Notes API collapsed to `retain()`" cut line. The plugin section says this in passing; recommend making it explicit in the config schema docs so plugin users do not expect Hindsight-style automatic turn-capture.

---

## Test-gate soundness — do the milestones actually gate?

| Milestone | Demo runnable? | Gate testable in CI? | Hidden cross-milestone dependency? |
|---|---|---|---|
| M1 — Scaffolding + prompts + types | Yes (`import prospecta`) | Yes (template render unit test) | None |
| M2 — Classical-RAG vertical slice | Yes (CLI `index` + `search`) | Yes (integration test w/ literal substring) | None — but see A1 on log-write API decision |
| M3 — Write-side spine | Yes (Python `retain()` call) | Yes (mock LLM, chroma assertion) | None |
| M4 — `recall_synth` w/ stub `formulate_queries` | Yes (Python call w/ mock LLM) | Yes (prompt-was-called assertion) | M5's eventual replacement is non-breaking |
| M5 — `formulate_queries` decoupled | Yes (Python call) | Yes (mock LLM, parser tests) | None, but A3 fallback needs test |
| M6 — Sweeper + CLI | Yes (REPL + sweeper-on test) | Yes (timed integration test) | A7: shared chroma client must be in place |
| M7 — Bilateral synthesis | Yes (`pytest test_bilateral_synthesis.py -v`) | Yes — *if* LLM recordings exist | Recordings depend on M5's prompts being final |

Two observations on the gating chain:

1. **M7 recordings have a one-time chicken-and-egg.** The first time someone runs M7, they need `PROSPECTA_TEST_LLM=record` against a real LLM with an API key. The plan acknowledges this. Recommend: M7's deliverable explicitly includes the recorded fixtures committed to the repo, recorded once by the implementer during M7. CI thereafter runs `replay`. This is in the plan but worth making first-class in the M7 deliverable list.

2. **M2 → M3 carries a decision (A1).** The plan does not name the `add_log_entry` decision as a milestone deliverable. Recommend adding to M2's deliverables: *"a port-decision sheet covering the seven `person`-related sites in `index.py`, with each marked delete / generalize / defer."* That doc is the gate between M2 and M3 — without it, M3 inherits an ambiguous foundation.

---

## CODING.md alignment

The plan is broadly aligned with animus's operating philosophy. Specific checks:

- **Structures-over-prose / tests-gate-not-docs:** Honored. M7 is the gate that makes v0.1 shippable; the bilateral-synthesis test, not the README, is the lock.
- **No-truncation:** Honored. `RecalledMemory.content: str  # FULL content, no truncation (P5)`. The bilateral test's body-assertion ("cello" or "vents" in synthesis) exercises full-content delivery.
- **Let-the-LLM-cook:** Honored. `formulate_queries` is one LLM call + line parsing. No Python heuristics where a prompt suffices. `index_text` generation is one LLM call. No multi-stage chain.
- **Surgical edits to known sections:** Mostly honored. A2 (prompt rewriting) is the place this is weakest — the plan budgets surface-level edits where the editorial work is more substantive. A1 (log-write API decision) is similar: surgical iff the decision is made up front.

---

## Recommended changes (ordered by importance)

1. **A1 — Port-decision sheet for `index.py`'s seven `person` sites at M2 boundary.** Without this, M2's classical-RAG gate ships on an ambiguous foundation. Recommend deciding *delete `add_log_entry` and log-recall surface entirely* and defer log-as-first-class to v0.2; prospecta v0.1 retains via file-on-disk only.

2. **A2 — Reframe both LLM prompts at M1.** `log-index.md` and `formulate-queries.md` need substantive rewriting away from Cookie/animus framing. Budget M1 at 1.5 days, not 1.

3. **A4 — 2x2 spine-isolation matrix in M7.** Add `spine_index_only` and `spine_formulate_only` cases. ~1 hour. Materially strengthens the v0.1 ship gate.

4. **A5 — Spine-toggle moves to `_test_helpers.py`, off the public `Memory.__init__`.** Locked signature stays locked.

5. **A6 — Defer real `embedding_model` to v0.2; v0.1 raises on non-default.** Avoids silent default-overrides in production.

6. **A8 — Three parser-robustness mock-LLM tests added to M5's gate.** Pins the deferral of JSON-mode against regression.

7. **A7 — One sentence on shared chroma client + a concurrency test in M6.**

8. **A3 — `formulate_queries` LLM-failure fallback shape named explicitly in M5.**

None blocks dispatch to the Critic round. All can be absorbed into the M1 prep without re-planning.

---

## Closing

The plan is implementation-shaped. The premise verifications are rigorous; the port order honors P11; the bilateral test is specified concretely; the plugin section is bounded. The reservations are the kind of catches a Round-1 review is supposed to surface — under-budgeted prompt work, an unstated decision boundary, a small leak of test surface into production API. With the eight A-items folded in, the plan ships v0.1 in the stated ~7.5 working days.

The single most consequential change is **A1** — the `index.py` port is more entangled than the planner's grep summary admits, and the decision sheet must precede M2 closing. The single most valuable addition is **A4** — the spine-isolation matrix turns M7 from "spine helps" into "*bilateral* spine is load-bearing," which is the actual P1/P15 claim.

⚒️
