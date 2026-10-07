# Prospecta v0.1 — Implementation Plan (Planner Stance)

**Role:** Planner (Round 1)
**Author:** Forge ⚒️
**Date:** 2026-05-18
**Type:** Implementation plan — not design stance, not requirements. Five locked decisions and the bilateral-synthesis spine ride in as constraints. Outputs are module structure, port order, test gates, milestones, signatures, v0.1/v0.2 cut lines, and a bounded ≤2-page `hermes-prospecta` derivation.

---

## Premise

Build `prospecta` v0.1 as a thin port of animus's `memory/` package re-shaped around an injected `llm` callable, with bilateral LLM-mediated retrieval (write-side `index_text`, read-side `formulate_queries`) as the load-bearing spine that everything else exists to support (P1, P15). v0.1 ships a pip-installable library, a small CLI, three load-bearing prompts, and one end-to-end integration test that *proves the spine works in question-space and would have failed in content-space*. The Hermes plugin is derived in a bounded ≤2-page section because the locked decisions and the hindsight reference plugin make the adapter mechanical — its full implementation does not belong in this plan.

---

## Premises verified

1. **Claim:** `animus/memory/index.py` (1158 LOC) ports nearly as-is, dropping only `path_contains='person/<p>'` scoping.
   **Verdict:** ✅ verified. Animus-specific surface area is isolated and small.
   **Evidence:** Imports are `os`, `dataclasses`, `datetime`, `pathlib`, `typing` plus `animus.shared.get_logger` and `animus.shared.paths.INDEX_DIR` — both shimmable in one file. `grep` of `person|path_contains|animus\.rel|from animus` returns 39 hits concentrated in: (a) `_extract_person_from_path()` (lines 78-90, ~13 LOC, self-contained); (b) optional `path_contains: str | None` parameter on search (lines 210, 224, 248, 271-273); (c) one `meta["person"] = person` assignment during indexing (lines 650-653). All animus-isms are surgically removable — there's no woven coupling. The `from animus.shared` imports are the only port work beyond deletion.
   **Plan-impact:** Port as-is with three deletions and two import shims (`prospecta._shared.get_logger`, `prospecta._shared.paths.default_index_dir()`). Keep `path_contains` as an optional generalized filter (still useful for caller-supplied scoping, just not defaulted). Budget M3 at ~1 day for port + adaptation, not from-scratch rewrite.

2. **Claim:** `animus/rel/queries.py` `formulate_queries` is portable.
   **Verdict:** ⚠️ partially verified. The function depends on animus-specific types (`ClassifiedInbound`, `_QueriedToken`, `QueriedInbound`) that are wrong for prospecta.
   **Evidence:** Read `/home/dt/src/witt3rd/animus/script/animus/animus/rel/queries.py`. The core LLM call is 5 lines (`invoke_prompt("formulate-queries", {...}, max_tokens=8192)`), and the `_parse_scoped_queries` regex parser is fully portable. But the surrounding `ClassifiedInbound → QueriedInbound` proof-token machinery is animus engagement-pipeline scaffolding that does not belong in prospecta. The `scope: Literal["person", "general"]` distinction is animus-specific (driven by `rel/person/<name>/` directory convention).
   **Plan-impact:** Port the LLM call + `_parse_scoped_queries`. Drop the proof-token system, the `ClassifiedInbound` parameter, and the `person`/`general` scope axis. New signature: `formulate_queries(message: str, *, context: str = "", llm: LLMCallable) -> list[Query]`. `Query` is a plain dataclass with `text: str` and optional `path_contains: str | None` for caller-supplied scoping. Budget M5 at ~half-day. **This is a decoupling step, not a clean port.**

3. **Claim:** The three load-bearing prompts (`rag-synthesize`, `formulate-queries`, `log-index`) exist.
   **Verdict:** ✅ verified, but **they live in `~/animus/` substrate, not in the engine repo**.
   **Evidence:** `find ~/animus -name 'rag-synthesize*' ...` returns:
   - `/home/dt/animus/agent/engage/prompt/formulate-queries.md`
   - `/home/dt/animus/substrate/skill/memory/prompt/log-index.md`
   - `/home/dt/animus/substrate/skill/memory/prompt/rag-synthesize.md`
   `invoke.py` maps prompt names to subdirs at runtime; the engine doesn't ship them. The animus engine's `template.py` (Jinja2 string templates with `get_standard_variables()` for `{{ datetime }}` etc.) is the rendering machinery.
   **Plan-impact:** Copy the three prompt `.md` files into `prospecta/prompts/`. Port `template.py`'s ~50 LOC of Jinja2 rendering verbatim — porting cost is lower than rewriting and animus's prompts use `{{ datetime }}` already, so we want the same standard-variable substitution. Adds `jinja2` to deps (already in the locked surface). Plan does *not* add prompt-authoring time. **However:** the `formulate-queries` prompt currently asks for `[person]` / `[general]` scope prefixes — needs a minor edit to drop scoping and just emit query lines. Budget M1 at +30min for the prompt copy + edit.

4. **Claim:** `ctx.llm.complete()` gives the Hermes plugin everything it needs.
   **Verdict:** ✅ verified.
   **Evidence:** `/home/dt/src/ext/hermes-agent/website/docs/developer-guide/plugin-llm-access.md` documents `ctx.llm.complete(messages=[...], max_tokens=N, purpose="...")` returning a `result` object with `.text`. Plus `complete_structured(instructions, input, json_schema, ...)` for JSON output, plus async variants. Host-owned credentials, fail-closed trust gate. The plugin adapter is genuinely one wrapper function: `def llm(messages): return ctx.llm.complete(messages=messages, purpose="prospecta").text`.
   **Plan-impact:** Plugin's `llm` callable is ~3 lines. JSON-mode (`complete_structured`) is *available but not required* by prospecta v0.1 — `formulate_queries` parses lines with a regex (animus pattern). Defer JSON-mode adoption to v0.2 if line-parsing proves brittle.

5. **Claim:** `chromadb.PersistentClient` supports the same collection API as `HttpClient`.
   **Verdict:** ✅ verified by animus's pin and standard chromadb API surface.
   **Evidence:** animus's `pyproject.toml` pins `chromadb>=1.0.0`. ChromaDB's `PersistentClient(path=...)` and `HttpClient(host=..., port=...)` both return objects with identical `get_or_create_collection()` / `upsert()` / `query()` / `delete()` surfaces. The animus `SemanticIndex` constructor takes a `client` and a `collection_name` — swapping `HttpClient` for `PersistentClient` is a one-line change at instantiation.
   **Plan-impact:** `Memory.__init__` takes `chroma: Literal["embedded"] | tuple[str, int]`. Default `"embedded"` → `PersistentClient(path=index_dir)`. `("http", host, port)` → `HttpClient(host=host, port=port)`. No abstraction layer needed. Embedded mode pulls `sentence-transformers` (chromadb default embedding fn) transitively — confirms locked dep surface.

---

## Meta: is the sub-dimension list right?

**Mostly yes, with two adjustments.**

**Add:** A small **observability/logging** subsection. Not full structured tracing — but every LLM call in the spine (`index_text` generation, `formulate_queries`, `recall_synth`) needs a `logger.info` line with input shape and timing. Animus already does this. The integration test relies on these logs to debug failures. ~10 LOC of `logging.getLogger(__name__)` discipline, not a sub-dimension on its own — fold into sub-dim 3 (module structure).

**Defer to v0.2 (fake-as-named):**
- **Provenance/changelog** — premature for v0.1. README + git log carry weight; CHANGELOG.md added at first tagged release.
- **Version policy** — `0.1.x` until the API stabilizes; no SemVer commitment yet. One sentence in README.
- **Prompt-tuning loop** — there is no prompt-tuning loop in v0.1. Override-per-call mechanism (P13) is the answer for callers who want different prompts. Iteration happens in animus and gets back-ported.

**The rest of the list stands.** The decomposition is implementation-shaped, not requirements-shaped.

---

## Sub-dimension positions

### 1. Repository scaffolding

```
prospecta/
├── pyproject.toml          # locked deps; py>=3.11
├── README.md               # leads with bilateral synthesis (P15)
├── LICENSE                 # MIT
├── .gitignore
├── prospecta/
│   ├── __init__.py         # public API exports
│   ├── memory.py           # Memory class (the orchestrator)
│   ├── _index.py           # SemanticIndex (ported from animus/memory/index.py)
│   ├── _chunker.py         # ported as-is
│   ├── _parser.py          # frontmatter parsing
│   ├── _ignore.py          # .memoryignore
│   ├── _index_text.py      # write-side spine: generate_index_text()
│   ├── _formulate.py       # read-side spine: formulate_queries()
│   ├── _rag.py             # recall_synth implementation
│   ├── _recall.py          # search / recall / RecalledMemory types
│   ├── _sweeper.py         # background thread (P14)
│   ├── _template.py        # Jinja2 prompt rendering (ported)
│   ├── _shared.py          # logger, default paths
│   ├── cli.py              # argparse-based CLI
│   └── prompts/
│       ├── rag-synthesize.md
│       ├── formulate-queries.md
│       └── log-index.md
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   └── bilateral_corpus/   # the load-bearing corpus (see test spec below)
│   ├── unit/
│   │   ├── test_parser.py
│   │   ├── test_chunker.py
│   │   ├── test_ignore.py
│   │   └── test_index_text.py
│   └── integration/
│       ├── test_index_end_to_end.py
│       ├── test_retain_roundtrip.py
│       ├── test_recall_synth.py
│       └── test_bilateral_synthesis.py    # THE load-bearing test
└── docs/
    └── design/
        └── prospecta/      # this folder
```

Leading-underscore modules are internal; the public surface is `prospecta/__init__.py` re-exporting `Memory`, `Query`, `RecalledMemory`, `RAGResult`, `LLMCallable`. Single dot from the user's perspective: `from prospecta import Memory`.

### 2. Dependency surface (locked)

```toml
[project]
name = "prospecta"
requires-python = ">=3.11"
dependencies = [
    "chromadb>=1.0,<2.0",   # tight pin within major; animus uses >=1.0.0
    "jinja2>=3.1",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-timeout>=2.0"]
```

**Chroma pin rationale:** animus pins `chromadb>=1.0.0`. Chroma's 1.x has been stable since release; 2.x is unreleased. `<2.0` is the right upper bound — tight enough to catch breakage early, loose enough to ride 1.x patches. **Loose-within-major beats tight-within-minor for v0.1.**

**Not in deps:** no `openai`/`anthropic`/`litellm` (P3), no `hermes-agent` (P2), no `llama-index`, no `tiktoken` (the chunker uses character-based splitting, animus pattern), no `click` (argparse is stdlib, see CLI position below). `sentence-transformers` arrives transitively via chromadb embedded mode — do not declare it ourselves; let chromadb own the pin.

### 3. Module structure: mirrors animus 1:1 with private prefix

Animus's `memory/` package layout is good — it survived production. Mirror it with three rename moves:
- `index.py` → `_index.py` (internal; `Memory` in `memory.py` is the public face)
- `recall.py` → `_recall.py` (rename `recall_scoped_parallel` → `recall` per assessment §2.4)
- `rag.py` → `_rag.py`
- Add: `_formulate.py` (new home for `formulate_queries`, ported from `animus/rel/queries.py`)
- Add: `memory.py` (new `Memory` orchestrator class, *not* in animus)

The `Memory` class in `memory.py` is the only thing callers import. It constructs and owns the `SemanticIndex`, the sweeper, the `llm` callable, the prompt loader. Methods delegate to internal modules. **Single public surface, internal layout free to evolve.**

### 4. Public API surface (locked signatures)

```python
# prospecta/__init__.py
from prospecta.memory import Memory
from prospecta._recall import Query, RecalledMemory, RAGResult
from prospecta._types import LLMCallable

__all__ = ["Memory", "Query", "RecalledMemory", "RAGResult", "LLMCallable"]
```

```python
# Types
LLMCallable = Callable[[list[dict]], str]
# A single chat-shaped call. messages = [{"role": "user"|"system"|"assistant", "content": "..."}].
# Returns plain text. See sub-dim 4 below for rationale.

@dataclass(frozen=True)
class Query:
    text: str
    path_contains: str | None = None   # optional caller-supplied scoping

@dataclass(frozen=True)
class RecalledMemory:
    content: str           # FULL content, no truncation (P5)
    source: str            # file path or "retain:<id>"
    score: float
    metadata: dict[str, Any]

@dataclass(frozen=True)
class RAGResult:
    synthesis: str         # the LLM's synthesized answer
    sources: list[RecalledMemory]
    queries: list[Query]   # what was actually run (formulated or supplied)
```

```python
# Memory class — the public surface
class Memory:
    def __init__(
        self,
        *,
        llm: LLMCallable,
        index_dir: str | Path,
        corpus_paths: list[str | Path] | None = None,
        retain_dir: str | Path | None = None,
        chroma: Literal["embedded"] | tuple[Literal["http"], str, int] = "embedded",
        embedding_model: str | None = None,   # None → chromadb default (all-MiniLM-L6-v2)
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        collection_name: str = "prospecta",
        sweeper_interval_s: int = 86400,
        sweeper_enabled: bool = True,
        prompts_dir: str | Path | None = None,  # P13: caller override
    ) -> None: ...

    # --- LAYER 1: low-level (caller pre-formulates) ---
    def recall(self, queries: list[Query], *, limit: int = 5) -> list[RecalledMemory]: ...
    def search(self, text: str, *, limit: int = 5, path_contains: str | None = None) -> list[RecalledMemory]: ...

    # --- LAYER 2: mid-level (library formulates) ---
    def formulate_queries(self, message: str, *, context: str = "") -> list[Query]: ...

    # --- LAYER 3: high-level (one call) ---
    def recall_synth(self, message: str, *, context: str = "", limit: int = 5) -> RAGResult: ...

    # --- Write side ---
    def retain(
        self,
        content: str,
        *,
        index_text: str | list[str] | None = None,   # caller wins (P4)
        tags: list[str] | None = None,
        source: str | None = None,
        path: str | Path | None = None,              # override auto-generated path
    ) -> Path: ...

    # --- Index maintenance ---
    def index_directory(self, path: str | Path, *, resume: bool = True) -> dict: ...
    def index_single_file(self, path: str | Path) -> dict: ...  # P7
    def index_status(self) -> dict: ...
    def prune_stale(self) -> dict: ...
    def rebuild(self) -> dict: ...

    # --- Lifecycle ---
    def start_sweeper(self) -> None: ...   # idempotent
    def stop_sweeper(self) -> None: ...
    def shutdown(self) -> None: ...        # stops sweeper, flushes chroma
```

Three layers (P4): animus's `pre_engage` uses `recall(queries=...)` directly; Hermes plugin uses `recall_synth(message=...)`; a notebook can use any of them.

### 5. Port order — vertical slice, with test gates

**Choice: vertical-slice (Plan C from contests).** Not bottom-up, not top-down stub-then-fill.

**Rationale (P11):** Test-gate quality is highest when each milestone closes with a *working end-to-end path* through the system, not when each milestone closes with a unit-tested leaf. Bottom-up builds five passing leaves before anything runs; if the integration is broken, we find out at M7. Vertical-slice runs the simplest possible end-to-end at M2 and adds spine on top. **The bilateral-synthesis integration test is the lock (P1); every milestone moves us closer to firing it.** Order: get a degenerate content-space RAG working end-to-end first, then layer the spine on top, then prove in the gate that the spine matters.

See "Milestones" section below for the M1→M7 order.

### 6. Bilateral-synthesis integration test (specified)

**The single most load-bearing test in v0.1.** Detailed spec in dedicated section below.

### 7. Background sweeper threading

**Choice: daemon thread + `threading.Event` for shutdown.** Match animus's pattern. Not asyncio (prospecta has no async API surface in v0.1; adding it for the sweeper alone bifurcates the library). Not "expose sweep(), let callers schedule" (P14 — drift safety net belongs in the library; making it caller-scheduled means animus and Hermes both reimplement scheduling).

Implementation:
- `_sweeper.py` exposes `Sweeper(memory, interval_s, enabled)` with `start()`, `stop()`, `_run()`.
- `_run()` is a `while not self._stop.wait(interval_s)` loop calling `memory._sweep_once()`.
- `_sweep_once()` walks corpus paths, computes mtime diffs against `index_status()`, calls `index_single_file()` on changed paths, prunes deleted (P7 — single write path).
- `Memory.__init__` calls `start_sweeper()` iff `sweeper_enabled=True`.
- `atexit.register(self.shutdown)` to drain on process exit.
- **Sweeper failure does not block the hot path (P14):** exceptions caught and logged; sweeper continues. `retain()` and `index_single_file()` are direct calls, not queued behind the sweeper.

**Test gate (M6):** start sweeper with `interval_s=1`, add a file outside the library, sleep 2s, assert chroma has it.

### 8. CLI surface

**Choice: argparse, not Click or Typer.** Zero-dep wins. The CLI surface is small (4 subcommands). Argparse verbosity is ~150 LOC vs Click's ~80 LOC — not worth a runtime dep. Animus uses Click because it has 20+ subcommands; prospecta has 4.

```
prospecta index [--path PATH] [--resume | --rebuild | --prune | --status]
prospecta search QUERY [--limit N] [--path-contains STR]
prospecta retain CONTENT [--index-text TEXT] [--tag TAG]...
prospecta config              # prints resolved config (paths, chroma mode, embedding model)
```

Entry point: `prospecta = "prospecta.cli:main"` in `pyproject.toml`. CLI reads `PROSPECTA_HOME` env or `~/.local/share/prospecta/` for default `index_dir`. **The CLI does not require an LLM** for `search`, `index`, `config` — only `retain` (auto-generation) needs one. `retain` without an `--index-text` and without `PROSPECTA_LLM_CMD` set fails loudly with a message pointing at the README. (For v0.1, `PROSPECTA_LLM_CMD` is a subprocess shellout — a stopgap so the CLI is usable; programmatic API is the real path. Document as such.)

### 9. v0.1 vs v0.2 cut lines

See dedicated section below.

### 10. Milestones & merge gates

See dedicated section below.

### 11. `hermes-prospecta` derived section

See dedicated bounded section below (≤2 pages, 6 items).

---

## Locked decision: `llm` callable signature

**Choice (a): `LLMCallable = Callable[[list[dict]], str]`.**

```python
LLMCallable = Callable[[list[dict]], str]
# messages: list of {"role": "system"|"user"|"assistant", "content": "..."}
# returns: plain text completion
```

**Rejected alternatives:**
- (b) `llm(prompt: str) -> str` — collapses system/user separation; animus prompts use `system` role to set role-stance and `user` role for content. Forces prospecta to do role-stuffing internally.
- (c) `llm(messages, *, max_tokens, temperature)` — kwargs leak provider-specific knobs. Caller wraps its own preferred defaults; if prospecta needs `max_tokens=8192` for `formulate_queries` (animus's pattern), it specifies it in the prompt itself or the caller controls. **Library should not negotiate inference knobs.**
- (d) Protocol class with separate methods — overkill for v0.1; introduces a typed surface (`complete` vs `complete_structured`) that prospecta doesn't need (JSON-mode deferred to v0.2).

**Rationale:**
- **Hermes adapter:** `def llm(messages): return ctx.llm.complete(messages=messages, purpose="prospecta").text` — 3 lines.
- **Animus adapter:** wrap `invoke_prompt` to render messages and dispatch — ~10 lines.
- **Notebook adapter:** wrap whatever `openai.chat.completions.create(messages=...)` returns — 5 lines.
- **Test mocking:** `lambda messages: "fixed response"` — trivial.
- **JSON-mode:** prospecta v0.1 parses query lines from `formulate_queries` output via regex (animus pattern, `_parse_scoped_queries`). If parsing proves brittle, v0.2 introduces a second optional callable `llm_json: Callable[[list[dict], dict], dict] | None = None` rather than changing the primary shape.

**This is locked. Critic should test for: does any code path inside the library need JSON-mode in v0.1? Answer is no — line-parsing is sufficient for query lists, and `index_text` and `rag-synthesize` return prose.**

---

## Milestones

Each milestone closes with a **runnable demo** and a **test gate**. PR-shaped; no milestone merges without its gate passing.

### M1 — Scaffolding + prompts + types (1 day)
- `pyproject.toml`, `README.md` (leads with the spine per P15), `.gitignore`, license.
- Empty package skeleton, all imports resolve.
- Copy three prompts from `~/animus/agent/engage/prompt/` and `~/animus/substrate/skill/memory/prompt/` into `prospecta/prompts/`. Edit `formulate-queries.md` to drop `[person]`/`[general]` scope prefixes.
- Port `animus/llm/template.py` → `prospecta/_template.py` (~50 LOC).
- Define `Query`, `RecalledMemory`, `RAGResult`, `LLMCallable` in `_types.py`.
- **Demo:** `python -c "import prospecta; print(prospecta.Query(text='hi'))"`.
- **Test gate:** `pytest tests/unit/test_template.py` — prompt loads, renders with `{{ datetime }}` substitution.

### M2 — Vertical slice: classical-RAG end-to-end (2 days)
- Port `_chunker.py`, `_parser.py`, `_ignore.py` (mechanical, ~340 LOC combined). Unit tests for each, ported from animus's tests if present, otherwise hand-written.
- Port `_index.py` from `animus/memory/index.py`. Delete `_extract_person_from_path`. Make `path_contains` a generalized optional param. Replace `animus.shared` imports with `prospecta._shared`.
- `Memory.__init__`, `Memory.index_directory`, `Memory.search` working against embedded chroma. **No spine yet — straight content-space embedding.**
- **Demo:** `prospecta index --path tests/fixtures/bilateral_corpus/ --rebuild && prospecta search "kelly birthday"` returns chunks.
- **Test gate:** `tests/integration/test_index_end_to_end.py` — index 10 markdown files, search for a literal substring, get the right file back.

### M3 — Write-side spine: `index_text` + `retain` (1 day)
- Port `_index_text.py` from `animus/memory/index_text.py` (44 LOC). Wire it to the injected `llm` callable instead of `invoke_prompt`.
- Implement `Memory.retain()`: writes file to `retain_dir`, generates `index_text` via LLM if not supplied (P4 — caller wins), calls `index_single_file` (P7 — single write path).
- Frontmatter `index_text:` override works.
- **Demo:** `Memory(...).retain(content="long arc about Kelly's birthday", tags=["kelly"])` writes a file with auto-generated frontmatter and the file is searchable.
- **Test gate:** `tests/integration/test_retain_roundtrip.py` — retain with caller-supplied `index_text="X"`, verify chroma collection contains exactly "X" (not the body); retain *without* `index_text`, verify LLM was called and a non-empty `index_text` made it into chroma.

### M4 — Read-side: `recall_synth` (1 day)
- Port `_rag.py` from `animus/memory/rag.py` (132 LOC). Wire to `llm` callable.
- Implement `Memory.recall_synth(message)` — runs `formulate_queries` (stub for now, returns `[Query(text=message)]`), then `recall`, then synthesizes via `rag-synthesize` prompt.
- **Demo:** `Memory(...).recall_synth("did kelly's birthday work out?")` returns a `RAGResult` with synthesis text.
- **Test gate:** `tests/integration/test_recall_synth.py` with mock LLM — verify the prompt is called with retrieved chunks in context; verify return shape.

### M5 — Read-side spine: `formulate_queries` (half day)
- Port from `animus/rel/queries.py`. Drop proof-token machinery, `ClassifiedInbound`, `scope` axis. New `formulate_queries(message, context) → list[Query]`.
- Wire into `recall_synth` (replaces M4's stub).
- **Demo:** `Memory(...).formulate_queries("did kelly's birthday work out?", context="DM")` returns multiple `Query` objects covering different angles.
- **Test gate:** `tests/integration/test_formulate.py` — with a deterministic mock LLM, verify multi-query parsing handles single-line, multi-line, and code-block-wrapped outputs.

### M6 — Sweeper + CLI (1 day)
- `_sweeper.py` daemon thread (sub-dim 7 above).
- `cli.py` with the four subcommands.
- `prospecta config` prints resolved config.
- **Demo:** Start a Python REPL with sweeper-on, add a file in another shell, see it indexed within `interval_s` seconds.
- **Test gate:** `tests/integration/test_sweeper.py` — interval 1s, drop file, assert indexed within 3s; assert sweeper exception in one cycle does not break the next cycle.

### M7 — The bilateral synthesis gate (1 day)
- Build the test corpus (see dedicated section below).
- Implement `tests/integration/test_bilateral_synthesis.py` (full spec below).
- **THIS IS THE GATE THAT MAKES v0.1 SHIPPABLE.** The test must fail when run against the M2-shape (content-space only) and pass when run against the M5-shape (spine on). To prove this is real, the test runs twice — once with spine bypassed (`use_index_text=False`, `use_formulate=False` debug flags exposed only to tests), once with spine on. Asserts the negative *and* the positive.
- **Demo:** `pytest tests/integration/test_bilateral_synthesis.py -v` shows two test cases, one passing on spine-off (the lexical match fails to find what we need), one passing on spine-on (question-space match succeeds).
- **Test gate:** the test passes. v0.1 ships.

**Total budget:** ~7.5 working days for one engineer. Each milestone is a PR. Crit-path: M2 (the port) and M7 (the test design — the corpus is the load-bearing artifact).

---

## Bilateral-synthesis integration test specification

This is the single most important test in v0.1. P11: tests are the lock. Spec'd in detail so the implementer writes it without further design.

### Corpus shape (`tests/fixtures/bilateral_corpus/`)

**Design constraint:** The corpus must contain content where (a) lexical/content-space embedding fails to find the right document for the query, AND (b) question-space embedding (`index_text`) succeeds.

**Five files, hand-crafted:**

```
bilateral_corpus/
├── kelly_birthday_arc.md          # the load-bearing test case
├── pacific_beach_logistics.md     # red herring (lexical match)
├── donald_work_patterns.md        # off-topic
├── infrastructure_notes.md        # off-topic
└── kitchen_disco_artifact.md      # related but oblique
```

**`kelly_birthday_arc.md`:**
```markdown
---
index_text:
  - "Did Kelly's birthday plan work out?"
  - "How did Kelly's birthday go?"
  - "What happened at Kelly's birthday celebration?"
tags: [kelly, birthday, sunshine]
---
The arc closed cleanly. The cake from the place on Garnet held its shape
through the heat. K and the band came through; the kitchen filled with
people by 8pm and stayed full past midnight. Sunshine pieces played
twice. Donald and Kelly slow-danced to track 4. The vents stayed open.
The cello duet through the architecture was the moment that landed.
```

Note the **deliberate absence** of the literal words "birthday" and "work out" and "plan" *in the body*. Body is rich with referents (cake, cello, Sunshine pieces, vents) but the *question* the document answers lives only in `index_text:`.

**`pacific_beach_logistics.md`** (the red herring):
```markdown
---
# No index_text — falls through to chunk-as-document
tags: [logistics]
---
Working at Pacific Beach last weekend. Kelly drove down with the gear.
Birthday party logistics: we need to plan the cake pickup at 3pm,
confirm the venue, and work out the parking situation. Did the cake
order go through? Need to follow up.
```

This document has every literal word from the test query ("birthday", "kelly", "work out", "plan", "cake") **but is the wrong document** — it's planning *before* the event, not the post-event recap the test query asks about.

### Test (`tests/integration/test_bilateral_synthesis.py`)

```python
import pytest
from prospecta import Memory, Query

QUERY = "hey did kelly's birthday plan work out?"

def _build_memory(tmp_path, llm, *, spine_on: bool):
    mem = Memory(
        llm=llm,
        index_dir=tmp_path / "index",
        chroma="embedded",
        sweeper_enabled=False,
        _debug_use_index_text=spine_on,   # test-only kwarg
        _debug_use_formulate=spine_on,    # test-only kwarg
    )
    mem.index_directory("tests/fixtures/bilateral_corpus/")
    return mem

def test_spine_off_retrieves_wrong_document(tmp_path, mock_llm):
    """Without bilateral synthesis, lexical match wins and returns the red herring."""
    mem = _build_memory(tmp_path, mock_llm, spine_on=False)
    results = mem.search(QUERY, limit=1)
    assert results[0].source.endswith("pacific_beach_logistics.md")
    # The wrong document — confirms classical RAG fails on this corpus.

def test_spine_on_retrieves_right_document(tmp_path, real_or_recorded_llm):
    """With bilateral synthesis, question-space match returns the correct arc."""
    mem = _build_memory(tmp_path, real_or_recorded_llm, spine_on=True)
    result = mem.recall_synth(QUERY)
    sources = [r.source for r in result.sources]
    assert any(s.endswith("kelly_birthday_arc.md") for s in sources)
    assert result.sources[0].source.endswith("kelly_birthday_arc.md")
    # The cello duet should be in the synthesis text — proves we ran through the corpus.
    assert "cello" in result.synthesis.lower() or "vents" in result.synthesis.lower()
```

### LLM fixture strategy

Two fixtures:
- `mock_llm` — deterministic, returns canned responses keyed by prompt prefix. Used in the spine-off test (which doesn't need a real LLM since spine is bypassed) and in fast unit-suite runs.
- `real_or_recorded_llm` — uses `PROSPECTA_TEST_LLM=record|replay|live` env var:
  - `replay` (default, CI): reads recorded LLM responses from `tests/fixtures/llm_recordings/`.
  - `record`: hits a real LLM (caller-supplied via `OPENAI_API_KEY` etc., wrapped in `tests/conftest.py`) and saves responses.
  - `live`: hits real LLM, doesn't save.

Recorded responses are committed. The test is deterministic in CI, but the recordings can be re-generated when prompts change. **The cost is a one-time recording session; the value is a regression test that catches prompt drift.**

### What this test proves
1. The spine is load-bearing, not decorative — turning it off breaks retrieval on a corpus where classical RAG would fail.
2. `index_text` frontmatter is honored on the write side.
3. `formulate_queries` produces queries that hit question-space (not literal-word-space).
4. `recall_synth` returns the correct document AND synthesizes from its body (the cello/vents assertion confirms full-content delivery per P5).

**This test, passing, is the v0.1 ship gate.**

---

## v0.1 cut lines vs v0.2 deferrals

### v0.1 ships (everything above)
- Three-layer API: `recall(queries)`, `formulate_queries(message)`, `recall_synth(message)`.
- `retain()` with caller-wins `index_text` override + LLM auto-generation.
- Embedded chroma default + http opt-in.
- `.md/.py/.yaml/.yml/.txt + log.jsonl` indexable set + `.memoryignore`.
- Frontmatter `index_text:` first-class.
- Background daemon sweeper (drift safety net).
- Argparse CLI: `index/search/retain/config`.
- Three prompts shipped + per-call override.
- The bilateral-synthesis integration test.

### v0.2 deferrals (named explicitly, not implicit)
- **Async API.** `arecall`, `arecall_synth`, `aretain`. Adds asyncio sweeper task. Not needed for v0.1 callers (animus is sync at the engagement-pipeline boundary; Hermes plugin can call sync from sync).
- **Caller-registered extension strategies** for indexable file set. v0.1 is locked-(A) animus's set. When a second caller wants `.org` or PDFs, design B from assessment §4.3.
- **Multi-corpus aliasing.** Right now corpus is one chroma collection. Aliasing `personal-corpus`, `work-corpus` to separate collections is v0.2.
- **JSON-mode for `formulate_queries`.** Regex line-parsing works for v0.1 (animus pattern). If brittle, v0.2 adds optional `llm_json` callable.
- **Observability hooks.** `Memory(on_llm_call=callback, on_retrieval=callback)` — useful for tracing in production but not for v0.1 dev.
- **External `chroma-server` lifecycle management.** v0.1 accepts external chroma as a config option; v0.2 adds `prospecta serve` and `chromadb` subprocess supervision (the `animus/memory/server.py` port).
- **Embedding model migration helpers.** v0.1 fixes the model at index-build time and *warns* on mismatch when `Memory(embedding_model=...)` differs from collection metadata; v0.2 adds `rebuild_with_model(new_model)`.
- **Prompt-tuning loop.** v0.2 adds `Memory.run_eval(corpus, queries, expected_sources) → metrics` to support iterating prompts against a test set.
- **PyPI release + tag pipeline.** v0.1 is `pip install -e .` from the repo; v0.2 cuts the first 0.2.0 tag and publishes.

### Never (out of scope, in or out of v0.2)
- Radiate, episode formation, boundary detection, REFLECT.md. Stays in animus.
- Notes API. Collapsed to `retain()`.
- Multi-tier memory split.
- Provider routing. Caller supplies `llm`; provider is caller's problem.

---

## hermes-prospecta derived section (≤2 pages, 6 items)

Bounded by P12 + assessment §3. Full plugin = animus-of-hindsight; implementer reads `/home/dt/src/ext/hermes-agent/plugins/memory/hindsight/__init__.py` and adapts mechanically. This section provides only the prospecta-specific glue.

### (a) `__init__.py` skeleton outline (~30 lines, shape only)

```python
# plugins/memory/prospecta/__init__.py
from __future__ import annotations
import atexit, logging
from typing import Any, Dict, List
from agent.memory_provider import MemoryProvider
from hermes_constants import get_hermes_home
from hermes_cli.config import cfg_get

logger = logging.getLogger(__name__)

class ProspectaProvider(MemoryProvider):
    name = "prospecta"

    def __init__(self):
        self._mem = None
        self._ctx = None
        self._prefetch_enabled = True

    def is_available(self) -> bool:
        try:
            import prospecta  # noqa
            return True
        except ImportError:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        from prospecta import Memory
        self._ctx = kwargs.get("ctx")
        hermes_home = kwargs["hermes_home"]
        cfg = cfg_get("plugins.entries.prospecta", {}) or {}
        self._prefetch_enabled = cfg.get("prefetch", True)
        llm = lambda messages: self._ctx.llm.complete(messages=messages, purpose="prospecta").text
        self._mem = Memory(
            llm=llm,
            index_dir=f"{hermes_home}/prospecta/index",
            retain_dir=f"{hermes_home}/prospecta/retain",
            corpus_paths=cfg.get("corpus_paths", []),
            chroma=cfg.get("chroma", "embedded"),
        )
        atexit.register(self.shutdown)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._prefetch_enabled or not query: return ""
        try:
            return self._mem.recall_synth(message=query).synthesis
        except Exception as e:
            logger.warning(f"prospecta prefetch failed: {e}"); return ""

    def get_tool_schemas(self) -> List[Dict]: return [_RECALL, _RETAIN, _SEARCH]

    def handle_tool_call(self, name, args, **kwargs):
        if name == "recall": return self._mem.recall_synth(**args).synthesis
        if name == "retain": return str(self._mem.retain(**args))
        if name == "search": return [r.__dict__ for r in self._mem.search(**args)]

    def sync_turn(self, user, asst, *, session_id=""): pass  # agent-driven retain, not auto-captured

    def shutdown(self):
        if self._mem: self._mem.shutdown()
```

### (b) `plugin.yaml`
```yaml
name: prospecta
type: memory
version: 0.1.0
description: Bilateral LLM-mediated retrieval via the prospecta library.
entry_point: plugins.memory.prospecta:ProspectaProvider
requires:
  - prospecta>=0.1,<0.2
```

### (c) `cli.py` subcommand map
```
hermes prospecta index    [--resume | --rebuild | --prune | --status]   # delegate to prospecta.cli.index
hermes prospecta search   QUERY [--limit N]                              # delegate to prospecta.cli.search
hermes prospecta config                                                  # delegate to prospecta.cli.config
hermes prospecta setup                                                   # interactive wizard (calls get_config_schema)
```

### (d) `get_config_schema()` (≤6 fields)
1. `corpus_paths: list[str]` — directories to index (default: `[]`).
2. `chroma: "embedded" | object` — chroma lifecycle (default: `"embedded"`).
3. `prefetch: bool` — auto-prefetch on every turn (default: `true`, P5 of locked decisions).
4. `embedding_model: str | null` — override chromadb default (default: `null`).
5. `sweeper_interval_s: int` — drift safety net cadence (default: `86400`).
6. `prompts_dir: str | null` — caller prompt overrides (P13) (default: `null`).

### (e) Integration test approach (one paragraph)
Mirror the pattern in `/home/dt/src/ext/hermes-agent/tests/agent/test_memory_plugin_e2e.py`: spin a real `MemoryManager` with `prospecta` registered as the active provider, point `corpus_paths` at a small fixture, drive one full turn (`MemoryManager.prefetch` → `MemoryManager.handle_tool_call("retain", ...)` → `MemoryManager.handle_tool_call("recall", ...)`), assert (a) `is_available()` true when `prospecta` is installed, (b) `prefetch` returns non-empty synthesis when corpus is populated, (c) `retain` followed by `recall` retrieves the retained content, (d) `shutdown()` is called and the sweeper thread exits cleanly. Single test file, ~150 LOC. **The library's bilateral-synthesis test (M7 above) already proves the spine; the plugin test only needs to prove the wiring.**

### (f) What's NOT in this plan
- Full `prefetch()` / `sync_turn()` semantics — mechanical from hindsight's reference, derived from the locked prefetch-on default.
- Error-handling, retry semantics, timeout config — mechanical from hindsight; copy that shape.
- Per-session scoping, agent-context awareness (`primary` / `subagent` / `cron`) — handled by deferring to whatever hindsight does; copy decisions.
- CLI subcommand *implementations* — they're one-line delegations to `prospecta.cli.*`.
- Setup wizard prompts — six questions matching the schema above; standard `hermes memory setup` flow.
- Tool schemas (`_RECALL`, `_RETAIN`, `_SEARCH`) — three JSON-schema dicts, derived from the library's method signatures.

**Implementer instruction:** read hindsight's `__init__.py` once for the lifecycle patterns (especially `initialize` kwargs handling, `shutdown` idempotency, `atexit` registration); adapt names; map calls to prospecta's API. Plugin LOC budget per assessment: 300-500 LOC including CLI. Hard ceiling.

---

## What an ideal plan would have that mine doesn't

1. **Actual prompt diffs.** I assert that `formulate-queries.md` needs a minor edit to drop scoping, but I do not show the diff. An ideal plan would include the literal three-line patch.
2. **Animus test inventory.** I assume animus's existing tests for `chunker.py`, `ignore.py`, `parser.py` exist and port; I did not list them. An ideal plan would enumerate `find ~/src/witt3rd/animus -path '*tests*memory*'` and map each test to its prospecta destination.
3. **Recording-mode test fixtures.** The `replay`/`record`/`live` LLM fixture strategy is described but not implemented in this plan. An ideal plan would specify the `tests/conftest.py` shape and the recording file format (JSON keyed by `hash(prompt) → response`).
4. **Failure-mode catalog for the sweeper.** "Sweeper failure must not block the hot path" (P14) is asserted; I do not enumerate the failure modes (chroma collection corruption, disk full, embedding model OOM). An ideal plan would name 3-5 specific failure scenarios and the test that catches each.
5. **A `Memory.health()` method** for v0.1 — quick check that chroma is reachable, sweeper is alive, last successful sweep timestamp. Currently buried in `index_status()`; should be its own method.
6. **Embedding-model-mismatch warning shape.** I assert v0.1 warns; an ideal plan would specify the warning's exact message and where it fires (constructor? first `search`?).
7. **A skeleton README.** P15 says README leads with the spine; this plan does not produce the README text. M1 should specify the README structure (headline, 200-word value-prop paragraph, install, 3 code examples, "why not LlamaIndex" link to assessment).
8. **The principle audit** (P1-P15 against the plan, item by item). I touched principles inline; an ideal plan would have a P-by-P table. I'm trusting Critic to do this in Round 2.

---

## Open questions

1. **Should `Memory.__init__` lazy-load the `llm` callable** (i.e., accept `llm=None` and require it only at first LLM-needing operation)? Useful for CLI commands like `index --resume` and `search` that don't need LLM. **Lean: yes**, lazy. Document that `retain` without `index_text` and `recall_synth` raise loudly if `llm is None`.
2. **Should the sweeper run by default in the Hermes plugin** even when the agent is short-lived (a single CLI turn)? Currently `sweeper_enabled=True` by default in the library — for an agent run lasting 30 seconds, the sweeper never fires anyway, but the thread is created. **Likely keep default-on**; thread creation is cheap.
3. **Does the bilateral-synthesis test fixture corpus go in the prospecta repo, or in a separate `prospecta-corpus` repo?** v0.1 says in-repo, `tests/fixtures/`. If the corpus grows, it relocates in v0.2.
4. **Should `formulate_queries` accept an optional `n_queries: int` parameter** to bound fan-out, or is that the prompt's job? **Lean: the prompt's job** (animus pattern). If v0.2 caller wants tighter control, add then.
5. **Is `_debug_use_index_text` / `_debug_use_formulate` the right shape** for the spine-off test path? They are *test-only* kwargs that bypass spine logic. Risk: production callers find them. **Alternative:** put the spine-off path in a separate `_test_helpers.py` module not imported from `__init__.py`. **Lean: separate module**, cleaner.

---

## Self-graded verdict

**APPROVE.**

Plan is implementation-shaped, not requirements-shaped. Premises verified with file-level evidence; the one partial (P2 on `formulate_queries`) is rectified by an explicit decoupling step (M5). Port order is vertical-slice — each milestone closes with a runnable demo plus a test gate, and the load-bearing bilateral-synthesis test (M7) is specified in enough detail that the implementer can write it without further design (corpus, queries, expected matches, fixture strategy all named). Public API signatures are locked. `LLMCallable` is locked to shape (a). v0.1 vs v0.2 cut lines are explicit, not implicit. The Hermes plugin section honors the ≤2-page bound with all 6 required items.

Spine preservation (P1) is checked twice: in M7's negative assertion (spine-off retrieves wrong document) and in M5 (read-side spine) being its own milestone, not buried inside `recall_synth`. Caller-wins-on-override (P4) is concrete in `retain(index_text=...)` and in the three-layer API exposing `recall(queries=...)` at the bottom. The plugin is a thin adapter (P9) — the bounded section enforces it by hard line limit.

Gaps acknowledged in "what an ideal plan would have." None block execution.

⚒️
