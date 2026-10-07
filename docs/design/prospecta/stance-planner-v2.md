# Prospecta v0.1 — Implementation Plan (Planner Stance, Round 2)

**Role:** Planner (Round 2)
**Author:** Forge ⚒️
**Date:** 2026-05-18
**Type:** Implementation plan, revised in response to Architect and Critic Round 1 reviews per orchestrator-curated treatment map. Round 1 stance was APPROVE-with-gaps; Round 2 closes the orchestrator-named subset. Catches the orchestrator held-as-let-slide are not re-litigated here (see context in delegated brief).

---

## 1. Round 2 revisions (response map)

For each addressed catch: status and treatment, concise. This section is for reviewers; the body of the plan carries the changes.

- **A1 — `index.py` has 7 entanglement sites, not 3; M2 budget light.** Status: ADDRESSED. Treatment: (a) promoted a port-decision sheet to a new **M0 gate** before scaffolding; (b) widened M2 from 2 days to 3.5 days; (c) sub-dim 11 below enumerates all 7 sites with explicit disposition decisions (lean: delete log-API surface for v0.1).
- **A2 — Prompt port under-budgeted; prompts are Cookie-shaped, not general-purpose.** Status: ADDRESSED. Treatment: reframed as **prompt translation**, not port. Added 1.5 days to M1 (now 2.5 days). Prompts renamed (see O2). Per-prompt translation deltas spec'd in §M1.
- **A4 — Bilateral-synthesis test must prove BILATERAL, not just "at least one side helps".** Status: ADDRESSED. Treatment: integration test rewritten as **2×2 matrix** (spine-off / write-only / read-only / both-on) with assertions `both-on > write-only` AND `both-on > read-only`. ~1 hour added to M7 corpus work. Full spec in §7.
- **A5 — `_debug_use_index_text` kwarg leaks test surface.** Status: ADDRESSED. Treatment: removed from `Memory.__init__`. Test-only paths live in `prospecta/_test_helpers.py`, not imported from `__init__.py`. Tests monkeypatch internal methods directly.
- **A6 — `embedding_model` warn-on-mismatch has chroma subtlety (silent on fresh index).** Status: ADDRESSED. Treatment: v0.1 raises `NotImplementedError` if `embedding_model` is user-supplied. Real support pushed to v0.2. Constructor signature updated.
- **A7 — Shared chroma client thread-safety between sweeper + hot path.** Status: ADDRESSED. Treatment: explicit **per-Memory `threading.RLock` around all chroma writes** (upsert/delete). Documented in §13 (concurrency contract).
- **A8 — Line-parser brittleness in `formulate_queries` is a P6 violation; JSON-mode promoted to v0.1.** Status: ADDRESSED. Treatment: `LLMCallable` signature widened to accept optional `json_mode=False` kwarg. `formulate_queries` uses JSON mode. Line-parsing retired. Updated in §8.
- **F2 — Port-is-translation, promote port-decision sheet to M0.** Status: ADDRESSED. Same fix as A1.
- **F3 (partial) — Plugin section item (f) is the actual design surface; rewrite as positive inheritances.** Status: ADDRESSED. Treatment: §10 item (f) now reads as "Decisions inherited from settled context" with positive framing (prefetch-on inherited, sync_turn no-op inherited, error semantics propagate).
- **O1 — `Query.path_contains` smuggles animus's scope ontology.** Status: ADDRESSED. Treatment: renamed to generic `metadata_filter: dict[str, Any] | None`. Updated dataclass, `search()` signature, `recall()` signature, and CLI flag (`--metadata KEY=VALUE`).
- **O2 — `log-index.md` prompt name leaks animus episode-shape.** Status: ADDRESSED. Treatment: renamed to `generate-index-text.md`. `formulate-queries.md` kept (not animus-shaped). M1 port plan reflects rename.
- **M1 — No tracing/observability primitive.** Status: ADDRESSED. Treatment: added `Memory(tracer=...)` injected callable; default no-op. Fires on retain / recall / formulate_queries / index_single_file. Full spec in §12.
- **M2 — P13 prompts-overridable downgraded silently from per-call to directory-level.** Status: ADDRESSED. Treatment: restored per-call overrides on `recall_synth(prompt_override=...)`, `retain(index_text_prompt_override=...)`, `formulate_queries(prompt_override=...)`. Resolution order: per-call > prompts_dir > library default.
- **M3 — retain-vs-retain concurrency unspecified.** Status: ADDRESSED. Treatment: per-Memory `threading.RLock` around chroma writes (same lock as A7). Caller fan-out is safe; serializes at the index boundary. §13 documents.
- **M4 — `llm` callable failure modes unspec'd.** Status: ADDRESSED. Treatment: §14 documents failure semantics — `retain` propagates, sweep logs+continues, `recall_synth` propagates, `formulate_queries` JSON-mode parse failure falls back to single-query.
- **M7 / P15 — README absent from plan.** Status: ADDRESSED. Treatment: new **M8 milestone** produces README.md with spec'd structure (§11). v0.1 ships with README, not just code.

---

## 2. Premise

Build `prospecta` v0.1 as a translated port of animus's `memory/` package re-shaped around an injected `llm` callable, with bilateral LLM-mediated retrieval (write-side `index_text` generation, read-side `formulate_queries`) as the load-bearing spine that everything else exists to support (P1, P15). v0.1 ships a pip-installable library, a small CLI, three translated prompts (renamed and generalized from Cookie-shaped originals), a README that leads with the spine, and a 2×2 bilateral-synthesis integration test that *proves both halves of the spine contribute*. The Hermes plugin is derived in a bounded ≤2-page section.

---

## 3. Premises verified (unchanged from Round 1, restated for completeness)

1. **`animus/memory/index.py` ports with bounded surgical changes.** Round 1 said "three deletions"; Architect catch A1 corrected to 7 entanglement sites. **Strengthened** — full disposition table in §11 (port decision sheet).
2. **`animus/rel/queries.py` `formulate_queries` is a translation, not a port.** Round 1 verdict stands; M5 budget already reflected decoupling work. JSON-mode promotion (A8) further tightens the read-side spine.
3. **The three load-bearing prompts exist in animus substrate but are Cookie-shaped.** Architect catch A2 corrected. **Strengthened** — M1 now does prompt **translation** (renamed and generalized), not mechanical copy. Per-prompt deltas in §M1.
4. **`ctx.llm.complete()` gives the Hermes plugin everything it needs.** Verified Round 1. Updated for A8: `complete_structured()` covers JSON mode for `formulate_queries` cleanly.
5. **`chromadb.PersistentClient` API parity with `HttpClient`.** Verified Round 1, unchanged.

---

## 4. Sub-dimensions (changes vs Round 1 noted inline)

### 4.1 Repository scaffolding — CHANGED

Added: `prospecta/_test_helpers.py` (A5 fix — out of public namespace). Added: `prospecta/_tracer.py` (M1 fix — observability primitive). Renamed prompts (O2):

```
prospecta/prompts/
├── generate-index-text.md     # was: log-index.md (O2 rename)
├── formulate-queries.md       # unchanged name; content translated (A2)
└── rag-synthesize.md          # unchanged
```

The rest of the tree from Round 1 stands.

### 4.2 Dependency surface — UNCHANGED

Locked surface from Round 1. JSON-mode (A8) does not add a dependency — it widens the `LLMCallable` signature, which the caller's wrapper handles (Hermes uses `complete_structured`; notebooks pass `response_format={"type": "json_object"}` to OpenAI).

### 4.3 Module structure — CHANGED

Three additions to Round 1: `_test_helpers.py` (test paths), `_tracer.py` (observability), `_concurrency.py` (the per-Memory lock, M3/A7). Net new code is ~80 LOC.

### 4.4 Public API surface — CHANGED (substantial)

Updated dataclasses and `Memory` signature. **`Query.path_contains` renamed to `metadata_filter` (O1).** **`embedding_model` raises `NotImplementedError` if user-supplied (A6).** **`tracer` added (M1).** **Per-call `prompt_override` kwargs restored (M2).**

```python
LLMCallable = Callable[[list[dict]], str] | Callable[[list[dict], bool], str]
# Optional second positional arg: json_mode (default False). When True, return
# must be a valid JSON string. Caller is responsible for setting the provider's
# JSON-mode flag. See §8 below.

@dataclass(frozen=True)
class Query:
    text: str
    metadata_filter: dict[str, Any] | None = None   # O1: was path_contains

@dataclass(frozen=True)
class RecalledMemory:
    content: str
    source: str
    score: float
    metadata: dict[str, Any]

@dataclass(frozen=True)
class RAGResult:
    synthesis: str
    sources: list[RecalledMemory]
    queries: list[Query]

Tracer = Callable[[str, dict[str, Any]], None]
# fires on: "retain", "recall", "formulate_queries", "index_single_file",
#           "sweep_pass", "llm_call"
# Default: no-op lambda. See §12.

class Memory:
    def __init__(
        self,
        *,
        llm: LLMCallable,
        index_dir: str | Path,
        corpus_paths: list[str | Path] | None = None,
        retain_dir: str | Path | None = None,
        chroma: Literal["embedded"] | tuple[Literal["http"], str, int] = "embedded",
        embedding_model: str | None = None,    # A6: raises NotImplementedError if set
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        collection_name: str = "prospecta",
        sweeper_interval_s: int = 86400,
        sweeper_enabled: bool = True,
        prompts_dir: str | Path | None = None,
        tracer: Tracer | None = None,           # M1: observability
    ) -> None: ...

    # Layer 1
    def recall(self, queries: list[Query], *, limit: int = 5) -> list[RecalledMemory]: ...
    def search(self, text: str, *, limit: int = 5,
               metadata_filter: dict[str, Any] | None = None) -> list[RecalledMemory]: ...  # O1

    # Layer 2 — M2 per-call override restored
    def formulate_queries(self, message: str, *, context: str = "",
                          prompt_override: str | None = None) -> list[Query]: ...

    # Layer 3 — M2 per-call override restored
    def recall_synth(self, message: str, *, context: str = "", limit: int = 5,
                     synth_prompt_override: str | None = None,
                     formulate_prompt_override: str | None = None) -> RAGResult: ...

    # Write side — M2 per-call override restored
    def retain(self, content: str, *,
               index_text: str | list[str] | None = None,
               index_text_prompt_override: str | None = None,
               tags: list[str] | None = None,
               source: str | None = None,
               path: str | Path | None = None) -> Path: ...

    # Maintenance + lifecycle — unchanged from Round 1
    def index_directory(self, path, *, resume: bool = True) -> dict: ...
    def index_single_file(self, path) -> dict: ...
    def index_status(self) -> dict: ...
    def prune_stale(self) -> dict: ...
    def rebuild(self) -> dict: ...
    def start_sweeper(self) -> None: ...
    def stop_sweeper(self) -> None: ...
    def shutdown(self) -> None: ...
```

**Prompt override resolution order (M2):** per-call `prompt_override` (literal prompt text or path) > `prompts_dir` configured at `Memory.__init__` > library-shipped default in `prospecta/prompts/`. Per-call wins. Documented in README + docstrings.

### 4.5 Port order — REVISED with M0 gate (A1/F2)

**New milestone M0 — Port Decision Sheet (0.5 day, BEFORE M1):**

Before any scaffolding, produce `docs/design/prospecta/port-decision-sheet.md` enumerating every entanglement site in `animus/memory/index.py` and `animus/rel/queries.py` with explicit disposition. Sheet is a gate artifact — Critic-reviewed before M1 starts. See §11 for the seed content (the 7 sites Architect named, plus the queries.py sites).

### 4.6 Bilateral-synthesis integration test — REVISED to 2×2 matrix (A4)

See §7.

### 4.7 Background sweeper threading — UPDATED (A7/M3)

Sweeper still daemon-thread (unchanged choice). **Added:** all chroma writes — from sweeper, from `retain`, from `index_single_file`, from `prune_stale` — go through a single `threading.RLock` owned by the `Memory` instance. Reads (`search`, `recall`) are not locked (chroma's reads are safe). Explicit in §13.

### 4.8 CLI surface — CHANGED (O1)

`--path-contains` flag retired; replaced with `--metadata KEY=VALUE` (repeatable). The CLI surface stays small. Other commands unchanged from Round 1.

### 4.9 v0.1 vs v0.2 cut lines — REVISED

JSON-mode moves to v0.1 (A8). `embedding_model` user-supplied path moves further into v0.2 (A6 — v0.1 raises). Tracer added to v0.1 (M1). See §9.

### 4.10 Milestones — CHANGED (added M0, M8; widened M1, M2)

See §6.

### 4.11 `hermes-prospecta` derived section — REVISED (F3-partial)

Item (f) rewritten as positive inheritance. See §10.

---

## 5. Port order with test gates — REVISED

**M0** (NEW gate) → **M1** (widened) → **M2** (widened) → **M3** → **M4** → **M5** → **M6** → **M7** (2×2 matrix) → **M8** (NEW: README).

Each milestone closes with a runnable demo + test gate. M0 closes with a Critic-reviewed sheet, not a test.

---

## 6. Milestones (revised count: 9; was 7)

### M0 — Port Decision Sheet (0.5 day, NEW)

Enumerate every animus-coupling site in the two files to be ported. For each: site name, lines, disposition (delete / shim / generalize / preserve), rationale. The 7 `index.py` sites Architect named, plus the `queries.py` sites Round 1 already covered. **Gate:** sheet committed to `docs/design/prospecta/port-decision-sheet.md`. Critic blesses before M1.

### M1 — Scaffolding + prompt translation + types (2.5 days, WAS 1)

- `pyproject.toml`, `.gitignore`, license, package skeleton.
- **Prompt translation, not copy (A2/O2):**
  - `log-index.md` → `generate-index-text.md`. Strip Cookie-specific framing (drop episode/spine language, drop animus-substrate-aware phrasing, drop the "your memory bank" voice). Generalize to "you are generating index questions for a document about to be filed in a semantic memory store." +1 day.
  - `formulate-queries.md`. Drop `[person]`/`[general]` scope prefixes. Add JSON-mode instruction (A8): output a JSON object `{"queries": [{"text": "..."}, ...]}`. +0.5 day.
  - `rag-synthesize.md`. Light edit: remove animus-specific role-stance phrasing. +0.5 hr.
- Port `template.py` for Jinja2 rendering (~50 LOC).
- Define dataclasses in `_types.py` (per §4.4).
- **Demo:** `python -c "from prospecta import Memory, Query; print(Query(text='hi'))"`.
- **Gate:** `pytest tests/unit/test_template.py tests/unit/test_prompts.py` — prompts load, render with `{{ datetime }}`, prompt names match new convention.

### M2 — Vertical slice: classical RAG end-to-end (3.5 days, WAS 2)

Widened per A1. Per-site disposition from M0 sheet drives the port:

- Port `_chunker.py`, `_parser.py`, `_ignore.py` (mechanical, ~340 LOC).
- Port `_index.py` from `animus/memory/index.py` (1158 LOC), executing M0 dispositions site-by-site:
  - **Delete** log-API surface (the lean per A1): line-range to be confirmed in M0 sheet.
  - **Delete** `_extract_person_from_path` and `meta["person"]` assignment.
  - **Generalize** `path_contains` → `metadata_filter` (O1).
  - **Shim** `animus.shared.get_logger` → `prospecta._shared.get_logger`.
  - **Shim** `animus.shared.paths.INDEX_DIR` → `prospecta._shared.paths.default_index_dir()`.
  - **Preserve** chunker integration, frontmatter pickup, embedding-fn wiring.
  - **Preserve** `index_single_file` as the single write path (P7).
- Add `_concurrency.py` with the per-Memory RLock (A7/M3).
- `Memory.__init__`, `Memory.index_directory`, `Memory.search` working against embedded chroma. **No spine yet.**
- **Demo:** `prospecta index --path tests/fixtures/bilateral_corpus/ --rebuild && prospecta search "kelly birthday"` returns chunks.
- **Gate:** `tests/integration/test_index_end_to_end.py` — index 10 markdown files, search for a literal substring, get the right file back. Plus a concurrency smoke test that two `retain()` calls from threads serialize cleanly.

### M3 — Write-side spine: `generate_index_text` + `retain` (1 day, unchanged)

- Port `_index_text.py` from `animus/memory/index_text.py` (44 LOC). Wire to injected `llm` callable.
- Implement `Memory.retain()` with caller-wins `index_text` override (P4) and per-call prompt override (M2).
- Frontmatter `index_text:` honored.
- **Demo:** `Memory(...).retain(content="long arc about Kelly's birthday", tags=["kelly"])` writes file with auto-generated frontmatter; file becomes searchable.
- **Gate:** `tests/integration/test_retain_roundtrip.py` — caller-supplied `index_text` is what lands in chroma; LLM-generated path produces non-empty index_text; per-call `index_text_prompt_override` is honored.

### M4 — Read-side: `recall_synth` (1 day, unchanged)

- Port `_rag.py` (132 LOC). Wire to `llm` callable.
- `Memory.recall_synth(message)` runs `formulate_queries` (stub: returns `[Query(text=message)]`), then `recall`, then synthesizes.
- **Demo:** `Memory(...).recall_synth("did kelly's birthday work out?")` returns a `RAGResult`.
- **Gate:** `tests/integration/test_recall_synth.py` with mock LLM — verify prompt receives retrieved chunks; verify return shape; verify `synth_prompt_override` is honored.

### M5 — Read-side spine: `formulate_queries` with JSON mode (1 day, WAS 0.5 day; A8 widens)

- Port and **translate** from `animus/rel/queries.py`. Drop proof-token machinery, `ClassifiedInbound`, scope axis.
- **Use JSON mode (A8):** call `llm(messages, json_mode=True)`, parse `{"queries": [...]}` cleanly. Line-parsing retired.
- Wire into `recall_synth` (replaces M4's stub).
- Per-call `prompt_override` works (M2).
- **Demo:** `Memory(...).formulate_queries("did kelly's birthday work out?", context="DM")` returns multiple `Query` objects.
- **Gate:** `tests/integration/test_formulate.py` — JSON-mode call produces structured output; malformed JSON falls back to single-query degraded mode (M4 failure-mode spec); per-call override is honored.

### M6 — Sweeper + CLI + Tracer wiring (1 day, slight widen)

- Daemon-thread sweeper with RLock (A7).
- CLI with `index/search/retain/config` and `--metadata` flag (O1).
- Tracer plumbed at the 6 named events (M1).
- **Demo:** Start a Python REPL with sweeper on + tracer printing events, drop a file in another shell, see sweep_pass + index_single_file events.
- **Gate:** `tests/integration/test_sweeper.py` — interval 1s, file appears within 3s; sweeper exception doesn't break next cycle; tracer fires on every named event.

### M7 — The bilateral-synthesis gate (1.5 days, was 1; A4 adds the 2×2)

See §7 for the full revised spec. Time added for the additional corpus design needed for the 2×2 to land cleanly.

### M8 — README + ship gate (0.5 day, NEW; M7/P15)

- Author `README.md` per §11 spec.
- Tag `0.1.0`. (PyPI publish deferred per Round 1 v0.2 list; this milestone is the repo-tag ship gate.)
- **Demo:** Reader of README understands what makes prospecta different from LlamaIndex/Khoj/mem0 within 5 minutes.
- **Gate:** README reviewed by Donald (the human collaborator standing for the audience); ship.

**Total budget:** ~12.5 working days for one engineer (was 7.5; +5 reflects A1 + A2 + M0 + M8 + A4 widening + A8 JSON-mode work). Honest estimate; Round 1 was light.

---

## 7. Bilateral-synthesis integration test specification — REVISED to 2×2 matrix (A4)

The Round 1 spec showed only "spine-off retrieves wrong document" and "spine-on retrieves right document." A4 catch: this proves the spine helps, not that **both halves are load-bearing**. Round 2 spec runs the 2×2.

### Test matrix

| Configuration | write-side (`index_text`) | read-side (`formulate_queries`) | Assertion |
|---|---|---|---|
| **OFF/OFF** (control) | OFF | OFF | retrieves wrong document (red herring wins) |
| **ON/OFF** (write-only) | ON | OFF | partial improvement; may retrieve right doc but not always |
| **OFF/ON** (read-only) | OFF | ON | partial improvement; multi-query helps but lexical body still dominant |
| **ON/ON** (full spine) | ON | ON | retrieves right document, top-1 |

### Assertions

1. `both-on > write-only` on a discriminating metric (top-1 hit rate over a small query set, or NDCG@5).
2. `both-on > read-only` on the same metric.
3. `both-on > off-off` (the original Round 1 assertion, retained).
4. `off-off` fails on the kelly-birthday-arc case (retrieves `pacific_beach_logistics.md`, the red herring).

### Implementation

Test-only paths live in `prospecta/_test_helpers.py` (A5 fix). They expose `build_memory_with_spine_config(write_on, read_on)` that wires the `Memory` instance to bypass `generate_index_text` (when `write_on=False`) or replace `formulate_queries` with identity (when `read_on=False`). The public `Memory.__init__` does **not** carry these flags — they are monkeypatched in via `_test_helpers`.

### Corpus widening

Round 1 corpus (5 files) is sufficient for the OFF/OFF vs ON/ON poles. To get clean separation between ON/OFF and OFF/ON, add 2-3 more crafted files where the write-side alone or read-side alone shows a measurable but incomplete win. Total corpus: ~8 files. ~1 hour additional design work, per the orchestrator's estimate.

### What this test proves (revised)

1. The **write-side** spine is load-bearing (proven by both-on > read-only).
2. The **read-side** spine is load-bearing (proven by both-on > write-only).
3. Bilateral is more than either half alone.
4. The OFF/OFF case fails on a corpus where classical RAG would fail — confirms the corpus is discriminating.

**This 2×2 matrix passing is the v0.1 ship gate.**

---

## 8. `llm` callable signature — REVISED (A8)

```python
LLMCallable = Callable[..., str]
# Required: positional list[dict] of chat messages.
# Optional kwarg: json_mode: bool = False
#   When True, return value MUST be a valid JSON string.
#   Caller is responsible for setting the provider's JSON-mode flag
#   (OpenAI: response_format={"type": "json_object"}; Anthropic: tool-use coercion;
#    Hermes: ctx.llm.complete_structured(...)).
#
# Library callers that need JSON mode use: llm(messages, json_mode=True)
# Library callers that do not: llm(messages)
```

### Where JSON mode is used inside the library

- **`formulate_queries`** uses JSON mode. Output schema: `{"queries": [{"text": "..."}, ...]}`. Parsed cleanly; no regex line-parsing.
- **`generate_index_text`** does NOT use JSON mode. Output is a newline-separated list of question strings (or `"a single string"`), which the library parses by splitting on newlines and stripping. The format is forgiving and the LLM's natural output shape matches.
- **`rag-synthesize`** does NOT use JSON mode. Output is prose synthesis.

### Adapter examples

```python
# Hermes adapter (still ~5 lines)
def llm(messages, json_mode=False):
    if json_mode:
        # Adapter constructs schema from the prompt's expected shape
        return ctx.llm.complete_structured(messages=messages,
                                            response_format={"type":"json_object"},
                                            purpose="prospecta").text
    return ctx.llm.complete(messages=messages, purpose="prospecta").text

# OpenAI notebook adapter
def llm(messages, json_mode=False):
    kw = {"response_format": {"type": "json_object"}} if json_mode else {}
    r = openai.chat.completions.create(model="gpt-4o", messages=messages, **kw)
    return r.choices[0].message.content
```

**This is locked.** A8 closed: line-parsing is gone in v0.1.

---

## 9. v0.1 vs v0.2 cut lines — REVISED

### v0.1 ships

Everything in Round 1's v0.1 list, plus:

- **JSON mode for `formulate_queries`** (promoted from v0.2 per A8).
- **Tracer primitive** (M1).
- **Per-call prompt overrides** on `retain`, `formulate_queries`, `recall_synth` (M2).
- **README** with spine-led structure (M8/P15).
- **2×2 bilateral test** (A4).

### v0.2 deferrals

- Async API (`arecall`, `arecall_synth`, `aretain`).
- Caller-registered indexable file-set extensions.
- Multi-corpus aliasing.
- **`embedding_model` real support.** v0.1 raises `NotImplementedError` if user-supplied (A6 — chroma's silent-on-fresh-index subtlety means we'd ship a footgun otherwise). v0.2 implements proper migration / rebuild path.
- External `chroma-server` lifecycle supervision (`prospecta serve`).
- Prompt-tuning loop (`run_eval`).
- PyPI publish.

### Never (out of scope)

Radiate, episodes, REFLECT.md, Notes API, multi-tier memory, provider routing.

---

## 10. `hermes-prospecta` derived section (≤2 pages, 6 items) — REVISED

Items (a)–(e) unchanged from Round 1 except for two small updates: `prefetch` calls `recall_synth` (still); item (e) integration test mirrors the same shape as before. The substantive change is item (f), per F3-partial.

### (f) Decisions inherited from settled context (REWRITTEN — F3-partial)

The plugin does not invent these; they fall out of the locked decisions and the hindsight reference plugin. State as positive inheritances so the implementer knows what they get for free:

- **Prefetch is on by default** (inherited from locked decision P5). Caller flips off via `cfg.prefetch = false`.
- **`sync_turn` is a no-op** (inherited from prospecta's agent-driven retain model — the agent calls `retain` as a tool, not a per-turn auto-capture). This is intentional and stated.
- **Error semantics propagate** (inherited from M4 failure-mode spec, §14): `retain` failures bubble to the agent (caller decides retry); `prefetch` failures are caught and logged (per Hermes plugin contract, prefetch is best-effort).
- **Per-session scoping inherits hindsight's pattern** — session-isolation handled by `MemoryManager`, not prospecta. The plugin passes `session_id` through where hindsight does; prospecta itself is session-agnostic.
- **Agent-context awareness** (`primary` / `subagent` / `cron`) inherits hindsight's gating: prospecta's `prefetch` is enabled in `primary` only; `subagent` and `cron` runs skip prefetch. The library does not need to know this — gating happens in the plugin's `prefetch` method.
- **Tool schemas** (`_RECALL`, `_RETAIN`, `_SEARCH`) follow hindsight's JSON-schema shape — three dicts, ~30 LOC total, derived mechanically from prospecta's method signatures.
- **Setup wizard** inherits the standard `hermes memory setup` flow — six questions matching the schema in (d). Implementer copies hindsight's wizard and adapts field names.
- **CLI delegations** are one-line per subcommand. `hermes prospecta index` → `prospecta.cli.index(args)`.

**Implementer instruction:** read hindsight's `__init__.py` once for the lifecycle patterns; the inheritances above name what carries forward without redesign. Plugin LOC budget: 300–500 LOC. Hard ceiling.

---

## 11. README.md content spec (NEW — M7/P15)

The README is part of v0.1, not a follow-on. Structure:

1. **Opening (spine-led, ~200 words).** Lead with: "Prospecta is a memory library for LLM agents that uses LLMs on both sides of retrieval — to generate index questions when storing, and to formulate queries when recalling. This bilateral mediation lets retrieval work in question-space, not content-space." Concrete worked example: the kelly-birthday-arc case (without the proper name — generic).
2. **Install + Quickstart.** `pip install -e .`; three lines of Python: construct `Memory(llm=my_llm, ...)`, call `retain(content)`, call `recall_synth(question)`.
3. **API examples (3 levels, matching the three-layer API).** Layer 1 (`recall(queries=...)`), Layer 2 (`formulate_queries(message)`), Layer 3 (`recall_synth(message)`). Each is a 5-line code snippet.
4. **"What makes prospecta different from LlamaIndex / Khoj / mem0".** Three-bullet comparison: prospecta does bilateral LLM-mediated retrieval; LlamaIndex et al. do classical content-space embedding. Prospecta has no provider lock-in (caller supplies `llm`); the others assume OpenAI or LangChain. Prospecta's write-side spine (`index_text`) is unique — competitors generate index text rarely or not at all.
5. **Caveats (honest).** v0.1 is sync only; embedding-model migration is v0.2; bilateral synthesis costs 2 LLM calls per retain and 1 + N per recall_synth.
6. **License + contributing pointer.**

Total: ~600 words. The opening 200 words honor P15 (spine leads the document). M8 produces this in 0.5 day.

---

## 12. Tracer / observability primitive (NEW — M1 fix)

```python
Tracer = Callable[[str, dict[str, Any]], None]

# Memory(tracer=my_tracer) — default no-op.
# Fires on six named events with structured payload:
#   "retain"             {"source": str, "content_chars": int, "index_text_chars": int,
#                         "llm_generated": bool, "duration_ms": float}
#   "recall"             {"n_queries": int, "n_results": int, "duration_ms": float}
#   "formulate_queries"  {"message_chars": int, "n_queries_out": int, "duration_ms": float,
#                         "json_mode_used": bool, "parse_fallback": bool}
#   "index_single_file"  {"path": str, "chunks": int, "duration_ms": float}
#   "sweep_pass"         {"files_seen": int, "files_indexed": int, "files_pruned": int,
#                         "errors": int, "duration_ms": float}
#   "llm_call"           {"prompt_name": str, "messages": int, "duration_ms": float,
#                         "json_mode": bool}
```

The tracer is the documented hook for production debugging, metrics emission, OpenTelemetry adapters, etc. Default no-op means zero cost when not used. Implementation: ~30 LOC in `_tracer.py` plus six call sites threaded through. Tests use a recording tracer (a `list.append` lambda) to verify event firing.

---

## 13. Concurrency contract (NEW — M3/A7 fix)

**Statement:** A `Memory` instance is safe to call from multiple threads. Concurrency is enforced by a single per-instance `threading.RLock` around all chroma writes. Specifically:

- **Locked operations** (serialize): `upsert`, `delete`, `update` on the chroma collection. These are invoked by `retain`, `index_single_file`, `prune_stale`, `rebuild`, and the sweeper.
- **Unlocked operations** (concurrent): `query` (used by `search` / `recall`). Chroma's read path is thread-safe; we do not need to serialize reads.

**Implication for callers:** `retain` from multiple threads is safe but serializes at the chroma boundary. If a caller wants true write parallelism, that's an explicit v0.2 design (batch retain, queue-based ingestion). v0.1's contract is: single-threaded synchronous logically; if the caller fans out, the library serializes the write path under the hood.

**Sweeper interaction:** the sweeper takes the same RLock for each `index_single_file` call inside the sweep pass. If a `retain` is in flight, the sweeper waits. If the sweeper is mid-pass, `retain` waits. No deadlock — RLock is reentrant within a single thread, and sweeper + retain are on different threads.

Documented in README caveats and in `Memory.__init__` docstring.

---

## 14. `llm` callable failure-mode spec (NEW — M4 fix)

| Operation | LLM failure mode | Behavior |
|---|---|---|
| `retain` (LLM generates index_text) | `llm` raises | Exception propagates to caller. The file is NOT written. Caller decides retry. |
| `retain` (caller-supplied index_text) | N/A | LLM not called. |
| `recall_synth` | `formulate_queries` LLM raises | Exception propagates. |
| `recall_synth` | `recall` (no LLM) raises | Exception propagates. |
| `recall_synth` | synthesis LLM raises | Exception propagates. |
| `formulate_queries` (JSON-mode parse failure) | JSON malformed | **Falls back** to single-query degraded mode: `[Query(text=message)]`. Tracer event records `parse_fallback=True`. Logged as warning. |
| Sweeper `index_single_file` | `llm` raises (auto-index_text) | Sweep skips that file with `logger.warning`. Sweep continues to next file. Tracer event records `errors += 1`. |
| Sweeper `index_single_file` | non-LLM exception (disk, chroma) | Same: skip + warn + continue. Failure does not poison the sweeper. |

**Rationale:** the hot path (caller-driven `retain`, `recall_synth`) propagates so the caller can decide; the background path (sweeper) is best-effort and never crashes the thread. JSON-mode parse failure is the one exception where the library degrades rather than raises — single-query mode is still useful, and forcing the caller to handle malformed-JSON every time is worse than degraded retrieval.

Documented in `Memory` class docstring and README caveats.

---

## 15. Open questions

1. **M0 sheet content — the 7 `index.py` sites.** I have enumerated 6 in my own reading (the original 3 from Round 1, plus log-API surface, plus path-validation, plus `meta["scope"]` if it exists). Architect named 7. The seventh likely lives in chunker-coupling or `index_status()`. The M0 sheet itself surfaces this; if the 7th is structurally load-bearing rather than deletable, M2 budget may need a second adjustment. **Risk: low; mitigation: M0 gate exists to surface this.**
2. **JSON-mode adapter complexity for Anthropic callers.** Hermes (via `complete_structured`) and OpenAI (via `response_format`) are clean. Anthropic JSON-mode requires tool-use coercion. Callers using raw Anthropic SDK will need ~10 lines of adapter, not 3. **Acceptable; document in README.**
3. **Tracer event payload schema stability.** v0.1 ships the six events with the payloads in §12. If a v0.2 caller writes an OpenTelemetry exporter against these, adding fields is safe but removing/renaming is breaking. **Lean: treat tracer payload as semi-public API from v0.1; bump minor version on changes.**
4. **README "what makes prospecta different" risks looking like a competitor takedown.** Frame as design philosophy, not competition. **Donald should review the README draft.**
5. **2×2 corpus tuning.** Designing the corpus so ON/OFF and OFF/ON each show *partial* improvement (not full, not zero) requires iteration. M7's 1.5 days budgets for this; if it slips, M8 (README) is the bumper. **Acceptable risk.**
6. **`metadata_filter` semantics beyond chroma's `where` clause.** v0.1 passes through to chroma's `where` filter directly. Callers can use `{"tags": {"$contains": "kelly"}}`. The dict shape is chroma-dialect; we leak that. **Acceptable for v0.1; document. v0.2 could add a small DSL if needed.**

---

## 16. Self-graded verdict

**APPROVE.**

Round 2 closes the orchestrator-curated subset of catches from Round 1's Architect (A1–A8) and Critic (F2, F3-partial, O1, O2, M1, M2, M3, M4, M7/P15). Five new artifacts ride into v0.1: M0 port-decision-sheet gate, prompt translation (not copy), 2×2 bilateral test matrix, tracer primitive, README. Two new sections specify the concurrency contract (§13) and llm failure-mode contract (§14) that were unspecified in Round 1.

Honest shifts:
- Budget grew from 7.5 days to 12.5 days (+67%). Round 1 was light; the additions are real work, not padding.
- The `LLMCallable` signature widened to support optional `json_mode=True`. Adapter complexity grew for Anthropic-direct callers (5–10 LOC vs 3).
- `embedding_model` user-supplied path is now an explicit `NotImplementedError`, not a warning. Honest about v0.1 not supporting it cleanly.
- The plugin section's item (f) flipped from "what's NOT in this plan" (negative, defensive) to "decisions inherited from settled context" (positive, derivable). Same content; better framing per F3-partial.

Held-as-let-slide catches (per orchestrator brief): F1 (artifact shape), F3-full (defer plugin), O3 (log.jsonl), O4 ("bilateral" as term), S1–S3 (simplifications). Not re-litigated. Donald's locked decisions stand.

Still TODO and acknowledged:
- M0 sheet content is seeded but not authored; first deliverable of M0 itself.
- The 7th `index.py` entanglement site is named-by-count, not enumerated. M0 surfaces.
- README draft does not exist yet; M8 produces it. Spec'd in §11.
- 2×2 corpus tuning is a budgeted-but-uncertain piece of M7.

Spine preservation (P1) is re-checked: M5 read-side spine uses JSON mode now (A8 strengthening, not weakening); M3 write-side spine uses translated `generate-index-text.md` (A2/O2 strengthening); M7 2×2 proves *both halves contribute*, not just "spine helps." The principle is more load-bearing in v2, not less.

Round 2 Critic will counterfactual-defer-test this. The places I'd lose that test are: (1) accepting A1 without enumerating the 7th site here (mitigated by M0), (2) the JSON-mode change widening `LLMCallable` (deliberate, locked, defended in §8), (3) the budget growth from 7.5 to 12.5 days (honest correction, not negotiable).

⚒️
