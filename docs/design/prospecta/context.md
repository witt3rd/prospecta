# Context — Prospecta Implementation Plan

**Domain:** `~/src/witt3rd/prospecta/` (library) + `~/src/witt3rd/hermes-prospecta/` (adapter, derived)
**Run type:** Single ralplan covering the library implementation plan; the plugin plan is derived post-distillation, not a separate ralplan.
**Artifact type:** **Implementation-plan-shaped, not design-stance-shaped, not requirements-shaped.** The five locked design decisions and the spine are settled (see Constraints). This run produces an executable plan: module structure, migration order, test gates, dependencies, milestones.
**Authored:** 2026-05-18 by Forge ⚒️.

---

## The design question

**What is the minimum-viable, test-gated implementation plan to bring up `prospecta` (the standalone library) from empty repo to a working v0.1, and to derive `hermes-prospecta` from prospecta's locked API?**

Expansion:

- **Prospecta** is a pip-installable Python library packaging chroma + chunker + indexer + RAG-with-synthesis + retain + background sweeper, with **bilateral LLM-mediated retrieval** as the load-bearing pattern. No Hermes dependency.

- **Hermes-prospecta** is a thin adapter (~300-500 LOC) implementing Hermes's `MemoryProvider` ABC by delegating to prospecta. Matches the mem0 / hindsight / honcho factoring pattern.

- This plan should be MV-shaped: what's in v0.1 (the minimum to validate the spine), what's deferred to v0.2 (sweeper tuning, advanced strategies), what's never in scope (radiate, episode formation, REFLECT.md — animus keeps those).

- The plan must enumerate **port-from-animus order** for the modules listed in §"What's ported" below, with test gates at each step so we don't merge a half-ported `index.py` against a non-existent `chunker.py`.

---

## What's settled going in (constraints — NOT contests)

The assessment doc at `/home/dt/forge/wiki/2026-05-18_memory-library-and-plugin-assessment.md` ran through five rounds of correction with Donald and produced **five locked decisions**. They are inputs to this run, not contests. The Critic must NOT re-open them (P11 — don't re-litigate settled questions).

1. **Names:** library = `prospecta`, plugin = `hermes-prospecta`.
2. **Chroma lifecycle:** embedded by default (`chromadb.PersistentClient`), external opt-in via config.
3. **Indexable file set:** exactly animus's set at v1 — `{.md, .py, .yaml, .yml, .txt}` + `log.jsonl`. `.memoryignore` for exclusion. Frontmatter `index_text:` override first-class.
4. **Retain policy:** (c.3) — caller-supplied `index_text` wins; library auto-generates via LLM if absent. Hermes plugin `retain` tool defaults to auto-generation.
5. **Prefetch in Hermes plugin:** ON by default. Operator can disable via `plugins.entries.prospecta.prefetch: false`.

Additionally settled:

- **The spine.** Bilateral LLM-mediated retrieval is the load-bearing intellectual contribution (PRINCIPLES.md P1). Both write-side `index_text` and read-side `formulate_queries` are non-negotiable. This is not a position to debate — it's the *reason prospecta exists*.

- **Factoring.** Two artifacts (library + plugin), library has no Hermes dependency (P2). The Hermes plugin is one of N possible adapters.

- **API layers.** Three: low-level `recall(queries=[...])`, mid-level `formulate_queries(message)` + `recall(...)`, high-level `recall_synth(message)`. Animus uses low-level; Hermes plugin uses high-level.

---

## Out of scope

This run does NOT design or plan:

- **Radiate** — animus's surgical-substrate-write side. Stays in animus.
- **Episode formation, boundary detection, REFLECT.md** — caller-side concerns. Animus keeps them.
- **Notes API** (`memory/notes.py`) — collapses to `retain()`; no separate CRUD surface.
- **Conversation log writing** (`memory/log.py`) — caller writes its own logs; prospecta indexes them.
- **Multi-tier memory** (working / episodic / semantic / procedural splits). Prospecta is one substrate; tiered systems layer on top.
- **The plugin's full implementation** — derived as a follow-on doc from this run's library API.
- **Production deployment / CI / publishing to PyPI** — out of v0.1.
- **Provider routing, OAuth, credential management** — prospecta takes an `llm` callable; routing is the caller's problem.

---

## Premises to verify (pre-Planner) — P13

The Planner MUST verify each before designing around it. If a premise is wrong, the Planner says so and reworks.

- **Claim:** "animus's `memory/index.py` (1158 LOC) can be ported nearly as-is, dropping only animus-specific bits (`path_contains='person/<p>'` scoping)."
  **Source:** assessment doc §"Ports from animus"
  **Evidence supplied:** read of the file by Forge during assessment
  **Verify by opening:** `/home/dt/src/witt3rd/animus/script/animus/animus/memory/index.py` — confirm animus-specific code is isolated to identifiable sections, not woven through.
  **If true:** ~1000 LOC of index code lands in v0.1 with mechanical port work.
  **If false:** ~1000 LOC of fresh implementation OR scoped rework. Plan changes shape.

- **Claim:** "animus's `formulate_queries` lives in `animus/rel/queries.py` and is portable to the library."
  **Source:** assessment + research report
  **Verify by opening:** `/home/dt/src/witt3rd/animus/script/animus/animus/rel/queries.py` — confirm function shape, prompt-loading, and dependence on `animus.rel`-specific types (which would need decoupling).
  **If true:** spine read-side ports cleanly.
  **If false:** spine read-side needs rebuild against decoupled inputs. Plan adds a "decouple ScopedQuery from animus.rel" step.

- **Claim:** "the three load-bearing prompts (`rag-synthesize`, `formulate-queries`, `log-index`) exist at `~/src/witt3rd/animus/script/animus/animus/llm/prompt/`."
  **Verify by opening:** that directory; confirm all three named prompt files exist.
  **If true:** prompts ship inside prospecta package with minor edits (remove animus-specific {{ person }} interpolation in `formulate-queries` if present).
  **If false:** prompts need to be authored from scratch — adds writing time + tuning iterations.

- **Claim:** "`ctx.llm.complete()` in Hermes gives the plugin everything it needs — no provider-specific imports required."
  **Source:** Hermes docs `plugin-llm-access.md`
  **Verify by opening:** `/home/dt/src/ext/hermes-agent/website/docs/developer-guide/plugin-llm-access.md` — confirm the API surface for chat + JSON, sync + async, and the trust-gate behavior.
  **If true:** plugin's `llm` callable is a one-liner wrapping `ctx.llm.complete`.
  **If false:** plugin needs its own client-construction logic. Plan grows.

- **Claim:** "`chromadb.PersistentClient` can run in-process without a server and supports the same collection/upsert/query API as `HttpClient`."
  **Verify by opening:** `chromadb` docs or `~/src/witt3rd/animus/.venv/lib/python3.13/site-packages/chromadb/` import surface.
  **If true:** embedded-default is genuinely one line of config difference.
  **If false:** embedded mode requires a different code path. Plan adds an abstraction.

---

## Required reading (open these — DO NOT work from summaries)

Absolute paths only. Subagents have no `cd`.

### Principles + directives

- `/home/dt/src/witt3rd/prospecta/PRINCIPLES.md` — load-bearing, all 15 entries are inputs.
- `/home/dt/src/witt3rd/animus/CODING.md` — operating philosophy; cited by P5, P6, P11.

### Settled design (constraints)

- `/home/dt/forge/wiki/2026-05-18_memory-library-and-plugin-assessment.md` — five locked decisions in §4, ports list in §2.4, plugin shape in §3.

### Animus modules to port (in port-order)

- `/home/dt/src/witt3rd/animus/script/animus/animus/memory/__init__.py` — public surface
- `/home/dt/src/witt3rd/animus/script/animus/animus/memory/config.py` (28 LOC)
- `/home/dt/src/witt3rd/animus/script/animus/animus/memory/parser.py` (72 LOC) — frontmatter
- `/home/dt/src/witt3rd/animus/script/animus/animus/memory/chunker.py` (149 LOC)
- `/home/dt/src/witt3rd/animus/script/animus/animus/memory/ignore.py` (119 LOC) — `.memoryignore`
- `/home/dt/src/witt3rd/animus/script/animus/animus/memory/index_text.py` (44 LOC) — spine write-side helper
- `/home/dt/src/witt3rd/animus/script/animus/animus/memory/index.py` (1158 LOC) — SemanticIndex; the big one
- `/home/dt/src/witt3rd/animus/script/animus/animus/memory/rag.py` (132 LOC)
- `/home/dt/src/witt3rd/animus/script/animus/animus/memory/recall.py` (201 LOC)
- `/home/dt/src/witt3rd/animus/script/animus/animus/memory/server.py` (56 LOC) — optional chroma launcher
- `/home/dt/src/witt3rd/animus/script/animus/animus/rel/queries.py` — `formulate_queries` (spine read-side)
- `/home/dt/src/witt3rd/animus/script/animus/animus/memory/commands.py` (777 LOC) — CLI surface (reference, not direct port — prospecta's CLI will be cleaner)

### Animus modules NOT to port (read for context only — confirm they're not deps of the ports above)

- `/home/dt/src/witt3rd/animus/script/animus/animus/memory/log.py`
- `/home/dt/src/witt3rd/animus/script/animus/animus/memory/episode.py`
- `/home/dt/src/witt3rd/animus/script/animus/animus/memory/boundary.py`
- `/home/dt/src/witt3rd/animus/script/animus/animus/memory/radiate.py`
- `/home/dt/src/witt3rd/animus/script/animus/animus/memory/dedup.py`
- `/home/dt/src/witt3rd/animus/script/animus/animus/memory/integrate.py`
- `/home/dt/src/witt3rd/animus/script/animus/animus/memory/notes.py`

### Prompts (animus side, to port)

- `/home/dt/src/witt3rd/animus/script/animus/animus/llm/prompt/rag-synthesize.md` (verify exists)
- `/home/dt/src/witt3rd/animus/script/animus/animus/llm/prompt/log-index.md` (verify exists)
- prompt for `formulate_queries` — find it: search `~/src/witt3rd/animus/script/animus/animus/llm/prompt/` for `formulate*`

### Hermes plugin contract (for the derived plugin plan)

- `/home/dt/src/ext/hermes-agent/website/docs/developer-guide/memory-provider-plugin.md`
- `/home/dt/src/ext/hermes-agent/website/docs/developer-guide/plugin-llm-access.md`
- `/home/dt/src/ext/hermes-agent/agent/memory_provider.py` — the ABC
- `/home/dt/src/ext/hermes-agent/agent/memory_manager.py` — for understanding lifecycle dispatch

### Hermes reference plugins (for plugin-plan derivation)

- `/home/dt/src/ext/hermes-agent/plugins/memory/hindsight/__init__.py` — ~1700 LOC, similar shape (local-client + tools)
- `/home/dt/src/ext/hermes-agent/plugins/memory/mem0/__init__.py` — ~360 LOC, minimal cloud reference
- `/home/dt/src/ext/hermes-agent/plugins/memory/honcho/__init__.py` — ~1300 LOC, with CLI subcommands
- `/home/dt/src/ext/hermes-agent/plugins/memory/honcho/cli.py` — reference for prospecta CLI subcommands

### The context package itself

- `/home/dt/src/witt3rd/prospecta/docs/design/prospecta/context.md` — *this file*

---

## Ground-truth freshness (P9)

- **Hermes-agent:** `/home/dt/src/ext/hermes-agent/` — `git fetch && git describe` not run for this session; Planner should verify the checkout is recent (within last 4 weeks) before designing against memory-provider APIs.
- **Animus:** `/home/dt/src/witt3rd/animus/` — this is Donald's own repo, HEAD is the reference. No upstream to check.
- **Both prospecta repos:** newly created empty dirs. No git history to verify.

---

## Sub-dimensions the plan must address

1. **Repository scaffolding.** Package layout (`prospecta/`, `tests/`, `docs/`, `prompts/`), `pyproject.toml` (deps, entry points, Python version), `README.md` (leads with the spine per P15), license, `.gitignore`.

2. **Dependency surface.** **LOCKED 2026-05-18 pre-dispatch:** `chromadb` (embedded mode), `jinja2` (prompt templating — animus prompts use it; porting verbatim saves rework), `pyyaml`, `pytest` (dev). What's NOT a dep: no Hermes, no LlamaIndex, no provider SDKs (`openai`/`anthropic`/`litellm`), no `tiktoken`, no explicit `sentence-transformers` (transitive via chromadb embedded). The Planner verifies via animus's `pyproject.toml` whether the chroma pin should be loose or tight (see Contest 8).

3. **Module structure of `prospecta/`.** Internal layout — does it mirror animus's `memory/` 1:1, or get reorganized? Where do prompts live? Where does the `llm` callable interface get defined?

4. **Public API surface.** Class names, function signatures, return types. The three-layer API from §1.6 of the assessment.

5. **Port order with test gates.** Which modules first, what test gates each step, how we ensure incremental greenness instead of "merge it all and pray."

6. **Bilateral synthesis end-to-end test.** The integration test that proves the spine works — write something with `retain`, recall it via `recall_synth` with a query that doesn't lexically match the content, confirm question-space matching succeeds. This is the single most load-bearing test in v0.1.

7. **Background sweeper threading.** Daemon thread, atexit shutdown, cadence config, drift-only contract (P14). How tested?

8. **CLI surface.** `prospecta index --resume/--rebuild/--prune/--status`, `prospecta search QUERY`, `prospecta config`. What's the entry point? Click? argparse?

9. **`hermes-prospecta` derived plan — BOUNDED (≤2 pages).** **LOCKED 2026-05-18 pre-dispatch:** the plugin section in this plan is bounded. It must contain ONLY: (a) `__init__.py` skeleton outline (~30 lines showing the MemoryProvider-subclass-with-delegation pattern); (b) `plugin.yaml` content; (c) `cli.py` subcommand map (one line per command, no implementations); (d) `get_config_schema()` field list (≤6 fields); (e) the integration test approach (one paragraph, pointing at `tests/agent/test_memory_plugin_e2e.py` as the pattern to mirror); (f) a "what's NOT in this plan" list — full `prefetch()` / `sync_turn()` / error-handling semantics derived mechanically from the locked decisions + hindsight's reference pattern; the implementer reads hindsight's `__init__.py` and adapts. **Hard limit ≤2 pages.** If the Planner produces more, the Critic flags as P25 (depth-theater on mechanical work).

10. **v0.1 scope cut.** What's in, what's deferred to v0.2. Plan must name explicit cut lines, not leave them implicit.

11. **Milestones / merge gates.** Ordered milestones (M1 → M2 → ...) with what each demonstrates. Each milestone closes with a runnable demo + a test gate.

---

## Contested questions (seeds — Critic, do full audit, don't stop here)

1. **(META) Are these the right sub-dimensions for an implementation plan?** Is something missing — provenance/changelog, version policy, prompt-tuning loop, observability/logging? Or is a sub-dimension fake (something better deferred to v0.2)?

2. **Port-order: bottom-up vs top-down.** Plan A: port `parser → chunker → ignore → index_text → index → rag → recall` (leaves stitching public API for last). Plan B: stand up `Memory` class as a stub, fill methods one at a time. Plan C: vertical slice — `retain` + `search` end-to-end first, then `recall_synth`, then `formulate_queries`. Each has different test-gate shapes. Which honors P11 (tests are the lock) best?

3. **Where do prompts live?** `prospecta/prompts/*.md` shipped as package data? Or imported via `importlib.resources`? Or rendered via Jinja2 (like animus) vs plain string templates? Tradeoff: animus uses Jinja2 and prompt-template loading; copying that gives less rework but adds a Jinja dep.

4. **`llm` callable interface.** What's the exact signature the caller provides? Options:
   - (a) `llm(messages: list[dict]) -> str` (chat-completion-shaped)
   - (b) `llm(prompt: str) -> str` (single-prompt-shaped)
   - (c) `llm(messages: list[dict], *, max_tokens=None, temperature=None) -> str`
   - (d) Protocol class with separate `complete` and `complete_structured` methods (Hermes-shaped)
   The shape affects how easy it is to wrap `ctx.llm.complete`, how easy to mock in tests, how much the library can do (JSON-mode for formulate_queries?).

5. **Sweeper architecture.** Daemon thread with sleep-poll loop (animus pattern)? `threading.Event` for clean shutdown? Or `asyncio` task if we expect async callers? Or — provocative — *no* in-library sweeper, library exposes `sweep()` and callers schedule it (cron, systemd, plain script)?

6. **Embedding model handling.** Embedded chroma uses `all-MiniLM-L6-v2` by default (chroma's onnx default). Do we expose this as `Memory(embedding_model=...)`? Or fix it for v0.1? Switching models requires rebuilding the index (assessment §10); the library should at least *warn* on mismatch.

7. **Test corpus.** What's the smoke-test corpus shape? Hand-crafted ~10 markdown files in `tests/fixtures/`? Or generated? The bilateral-synthesis test (§7 above) needs a corpus where lexical query match would FAIL but question-space match succeeds.

8. **Version pin policy for chroma.** Chroma's API has shifted between minor versions (animus pins). Pin tightly (`chromadb>=0.5,<0.6`)? Pin loosely (`chromadb>=0.5`)? The Planner has to pick — and probably depends on animus's current pin.

9. **What does v0.1 NOT do that v0.2 should?** Likely: ~~external chroma~~ (settled — opt-in from v0.1), async API, multi-corpus aliasing, prompt-override-per-call as a first-class config, observability hooks. Plan should name the cut line explicitly.

10. **CLI library choice.** Click (animus uses) or argparse (zero-dep)? Or Typer? Affects packaging; argparse is stdlib but more verbose.

---

## What "done" looks like for this run

The plan produced by this run must:

- Enumerate explicit milestones (M1, M2, ...) with what each demonstrates and what tests gate it.
- Name an explicit port order with rationale.
- Specify the bilateral-synthesis integration test in enough detail that the implementer can write it without further design.
- Lock the `llm` callable signature.
- Specify v0.1 cut lines vs v0.2 deferrals — explicitly.
- Include a section deriving `hermes-prospecta` from the library API: `__init__.py` skeleton outline, `plugin.yaml`, `cli.py` map, config schema, MemoryProvider method-by-method behavior. NOT full code; a plan the implementer can execute.
- Pass the question: "what would an ideal version of this plan have that mine doesn't?" — the Critic must ask this and the Planner must address it in Round 2 if applicable.

The Critic must specifically also test:

- Is the spine adequately preserved? Could the plan accidentally produce a v0.1 that ships content-space RAG?
- Does the plan honor caller-wins-on-override (P4)? E.g., is `formulate_queries` exposed at the low-level API, not buried inside `recall_synth`?
- Is the plugin treated as a thin adapter, or does it accumulate library logic?

---

⚒️ Forge — context package for prospecta implementation ralplan, 2026-05-18.
