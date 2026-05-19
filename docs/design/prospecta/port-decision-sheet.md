# Port Decision Sheet — animus → prospecta

**Date:** 2026-05-19
**Author:** Forge ⚒️, with Donald
**Scope:** `animus/memory/index.py` (1158 LOC) + `animus/rel/queries.py`
**Target:** prospecta v0.1 (`prospecta/_index.py`, `prospecta/_formulate.py`)
**M0 milestone gate** per `plan-v2.md` §6 and `decision-record-1.md` (post-addendum).
**Consumed by:** T9 (vertical-slice port) and T13 (read-side spine).

---

## Disposition verbs

- **DELETE** — animus-domain code prospecta does not need. Drop from the port.
- **GENERALIZE** — animus-specific shape with a generic equivalent in prospecta. Drop the specifics, keep the mechanic.
- **SHIM** — relies on `animus.shared.*` or similar; replace with prospecta-internal equivalent.
- **PRESERVE** — generic and load-bearing; ports as-is or with minimal edits.
- **REWRITE** — generic concept, but the underlying mechanism changes substrate (chroma → SQL). Behavior preserved; implementation new.

---

## Header / module-level (`animus/memory/index.py`)

### Site 1.1 — `from animus.shared import get_logger` (L22)

**Disposition: SHIM.**

Replace with `import logging; logger = logging.getLogger(__name__)`. Standard library equivalent.

### Site 1.2 — `from animus.shared.paths import INDEX_DIR` (L23)

**Disposition: DELETE.**

INDEX_DIR was animus's `~/animus/index/` chroma persistent directory. Prospecta uses `database_url` (Decision 1). No equivalent constant needed; remove the import and every reference.

### Site 1.3 — Lazy chromadb loader (`_get_chromadb`, L64-75)

**Disposition: DELETE.**

No chroma in prospecta. Replace with direct psycopg use via `prospecta.db.ConnectionPool` (T2).

### Site 1.4 — `from .notes import Note` (L60)

**Disposition: DELETE.**

Animus's `Note` is its rich note-with-relationships type from `memory/notes.py`, which we already locked as out-of-scope (not ported). T8's port replaced it with parser-internal `ParsedNote`. T9 uses `ParsedDocument` from the new ParserPlugin Protocol (decision-record-1 addendum 2026-05-19).

---

## Class `SemanticIndex` (~L120-340)

The chroma-shaped public API. Replaced wholesale by `Memory` methods backed by `prospecta.db.queries`.

### Site 2 — Chroma client + collection management (L120-150, 161, 193, 236, 250, 297-313, 333)

**Disposition: REWRITE.**

All `self._client` / `self._collection` / `collection.upsert(...)` / `collection.query(...)` / `collection.delete(...)` / `collection.count()` calls become SQL via `prospecta.db.queries`. Concrete mapping:

| Animus call | Prospecta replacement |
|---|---|
| `collection.upsert(documents=[index_text], metadatas=[meta], ids=[id])` | `INSERT INTO documents + memory_items ... ON CONFLICT (bank_id, content_hash)` per schema.md §6 (replace-on-source-match) |
| `collection.query(query_texts=[q], n_results=N, where=W)` | Hybrid RRF CTE from schema.md §5 (semantic + lexical fused) OR `semantic_search` / `lexical_search` for mode-locked calls |
| `collection.delete(ids=[...])` | `DELETE FROM memory_items WHERE id = ANY(%s)` |
| `collection.count()` | `SELECT COUNT(*) FROM memory_items WHERE bank_id = %s` |
| `client.delete_collection(...)` | `DELETE FROM banks WHERE bank_id = %s` (CASCADE to documents → memory_items) |
| `get_or_create_collection(...)` | `Memory.create_bank()` from T4 (already shipped) |

**This is the substantial mechanical work of T9.** ~400 LOC of chroma plumbing translates to ~150 LOC of psycopg + SQL (Postgres does more per call; less Python orchestration needed).

### Site 3 — `index_note(self, note: Note)` and `index_all(...)` (L151-204)

**Disposition: DELETE (signature) / PRESERVE-AS-CONCEPT (behavior).**

The `Note`-typed methods don't survive — `Note` was animus's notes-CRUD type. The *behavior* (one row per index_text, document+metadata structure) is what T11 (`retain`) and T9 (`index_directory`) provide via `ParsedDocument` flow.

Effectively, `Memory.retain(content, index_text=..., ...)` from T11 replaces `index_note`. `Memory.index_directory(path)` from T9 replaces `index_all` walking a notes dir.

---

## `search()` (L205-296)

### Site 4 — Search method signature + behavior

**Disposition: REWRITE + GENERALIZE.**

The shape ports cleanly, but the substrate-specific bits change:

- `path_contains: str | None` → **GENERALIZE** to `metadata_filter: dict[str, Any] | None` per O1 fold. Postgres `WHERE` can do real filtering (no need for the `request_n = n_results * 3` over-fetch + Python post-filter that animus did at L248, L271-273).
- `where=None, where_document=None` (chroma-shape filters) → replaced by `metadata_filter` + `tags`/`tags_match` per `Query` dataclass (T6, already shipped).
- `n_results` → `limit`.
- Distance → similarity formula `1 / (1 + distance)` — chroma-specific; pgvector returns cosine distance directly via `<=>` operator. Replace with `1 - (embedding <=> query_embedding)` or equivalent (schema.md §5 has the canonical SQL).

Return type: animus returns `list[SearchResult]` (chroma-shape). Prospecta returns `list[RecalledMemory]` (T6, already shipped with the right shape including `scores` dict per A6 COALESCE).

---

## `index_directory()` and helpers (L507-776)

The most entangled section. Strict per-site dispositions:

### Site 5.1 — `_extract_person_from_path()` (L77-92)

**Disposition: DELETE.**

Animus's `rel/person/<name>/` substrate convention. Prospecta has `bank_id`. No person concept anywhere.

### Site 5.2 — Person metadata injection in `index_directory` (L650-668)

**Disposition: DELETE.**

`if person: meta["person"] = person` and the surrounding extraction logic. Drop entirely.

### Site 5.3 — Person from frontmatter override (L736-768)

**Disposition: DELETE.**

`person = frontmatter.get("person")` + fallback to path extraction. Drop entirely. The frontmatter we DO honor is `index_text:` (preserved in Site 5.4) and `tags:` (generic, preserve).

### Site 5.4 — Markdown frontmatter `index_text:` handling (L688-776, `_index_note`)

**Disposition: PRESERVE-AS-BUNDLED-PARSER.**

This is HALF THE BILATERAL SPINE on the write side. Animus's frontmatter convention is exactly what prospecta needs — the markdown file ships with caller-supplied questions; library indexes those as `memory_items.content` (the LLM-anticipated question form) with `documents.original_text` preserving the body.

**Translation per the 2026-05-19 ParserPlugin addendum:** this code becomes the bundled markdown `ParserPlugin.parse()` implementation. It yields `ParsedDocument(original_text=body, index_text=frontmatter.get("index_text"), metadata=..., tags=...)`. Library dispatches to it for `.md` files automatically.

**The single string vs list handling (L719-722) ports verbatim.** Both shapes remain first-class.

### Site 5.5 — Log.jsonl handling (L542, L606-611, `_index_log_jsonl` L845-936)

**Disposition: DELETE.**

Animus's specific conversation-log shape. Per the 2026-05-19 ParserPlugin addendum: callers who want to index jsonl conversation logs ship their own `ParserPlugin` for `*.jsonl` (or whatever their format is). The library doesn't bundle one.

The Hermes plugin will likely ship its own ParserPlugin for conversation logs (when that's wanted), but that's hermes-prospecta's concern, not prospecta's.

### Site 5.6 — `index_episode()` method (L778-844)

**Disposition: DELETE.**

Episode formation is animus's engage-pipeline concern (the `process_episode` path that runs `detect_episode_boundary` etc.). Prospecta doesn't have episodes. `Memory.retain()` from T11 handles the equivalent: caller hands prospecta a content blob, library generates index_text and stores.

### Site 5.7 — Standard-file chunking dispatch (L545-560, `chunk_file` calls)

**Disposition: REWRITE + RESHAPE.**

Animus dispatched by extension: `.md` (without frontmatter) → chunker, `.py / .yaml / .yml / .txt` → chunker. The whole "code-and-config-files-also-indexed" pattern was animus indexing its own substrate including its codebase.

**Per the 2026-05-19 addendum:** prospecta does NOT dispatch by extension to chunker for `.py` / `.yaml` / `.txt`. Those file types reach the library ONLY via a caller-supplied ParserPlugin. The bundled markdown parser handles `.md`. Everything else is plugin territory.

What DOES survive: the chunker itself (`prospecta._chunker`, T8 already shipped) — the bundled markdown parser uses it for `.md` files WITHOUT `index_text:` frontmatter (chunk the body, each chunk becomes a memory_item with `content == chunk text`).

### Site 5.8 — Crawl + ignore (L475 `from .ignore import should_ignore`, L1027)

**Disposition: PRESERVE.**

`_crawl_files(root)` + `should_ignore` via `prospecta._ignore` (T8 already shipped). Generic. The crawler is library-owned (per ParserPlugin design); the ignore patterns are the same `.memoryignore` mechanic.

The `INDEXABLE_EXTENSIONS` constant ports as the union of:
- `[".md"]` (bundled markdown parser)
- All `file_patterns` from registered ParserPlugins

So if no plugins registered: `.md` only. If a caller registers a JSONL plugin with `file_patterns=["*.jsonl"]`, the crawler picks those up too.

---

## `index_single_file()` (L938-1102)

### Site 6 — Per-file index method

**Disposition: REWRITE.**

Same as Site 5.7 — dispatches by file pattern to the right parser. P7 (single write path through `index_single_file`) holds: `retain()` from T11 calls this, the sweeper from T14 calls this, the CLI from T15 calls this.

Person extraction at L1037-1086 — **DELETE** all of it. Per Site 5.

---

## `get_status()` (L317-374)

### Site 7 — Status method

**Disposition: GENERALIZE + REWRITE.**

Animus signature: `get_status(graph_ids: set[str] | None = None) -> dict`. The `graph_ids` parameter is the set of notes the rel-curation heartbeat knows about (used to compute `missing_notes`). **DELETE** that parameter — animus-specific.

Prospecta signature: `get_status() -> dict`. Returns:
- `collection` → `bank_id` (or all banks if not scoped)
- `count` → `SELECT COUNT(*) FROM memory_items WHERE bank_id = ?`
- `storage_path` → DELETE (no such concept; database_url is opaque)
- `embedding` → DELETE ("server-side" was chroma-specific)
- `exists` → trivially true if migration ran
- `stale_entries` → count of memory_items pointing at files no longer on disk (per source path tracking; design detail for T9)
- `missing_notes` / `indexed_notes` → DELETE (rel-curation concern)

What it becomes is roughly `Memory.bank_stats(bank_id)` from T4 (already shipped) plus a richer ops-stats query that hits the event tables. We'll likely expand `bank_stats` rather than replicate `get_status`.

---

## `prune_stale()` (L375-505)

### Site 8 — Prune method

**Disposition: GENERALIZE + REWRITE.**

`documents.source` is a caller-supplied opaque string. The library MUST NOT introspect or heuristic-classify it. No "is it a path? is it a URL?" inference inside `prune_stale` or anywhere else in the library.

`Memory.prune_stale()` in v0.1 is therefore *not* "delete rows whose source no longer exists on disk" — that would require the library to interpret what `source` means. Instead:

- **`Memory.prune_stale()`** — removes rows the *caller* has flagged as stale. The semantics of "stale" live in the caller, not the library. v0.1 minimum: removes rows where a caller has set a `stale: true` metadata key, or `Memory.remove_documents(sources: list[str])` is called explicitly with a caller-supplied list.
- **The sweeper (T14)** — which IS filesystem-aware by design (it walks configured corpus paths) — carries the path semantics. After its walk it computes which `documents.source` values are absent on disk and calls `Memory.remove_documents([...])` with that list. The sweeper has filesystem knowledge; the library doesn't.

`dry_run` parameter survives in both `prune_stale` and `remove_documents`. The 5000-id pagination chroma needed (L213) is DELETED — Postgres has no such limit; standard SQL works.

**Substrate-opacity is the discipline.** Hindsight uses `documents.source` polymorphically; prospecta does the same. The library never names what kinds of sources exist — only the caller and the sweeper know that.

---

## `clear()` (L311-314)

### Site 9 — Clear method

**Disposition: REWRITE.**

`client.delete_collection(COLLECTION_NAME)` → `DELETE FROM banks WHERE bank_id = %s` (CASCADE to documents → memory_items). Optionally `Memory.rebuild()` from plan-v2 §3.4 that drops+recreates the bank's HNSW index.

---

## `clear_chunks()` (L1114-1158)

### Site 10 — Clear chunks method

**Disposition: DELETE.**

Animus distinguishes "notes" from "chunks" in its collection (chunks are the chunker output for non-frontmatter files). Prospecta's `memory_items` table is uniform — no need to separately clear "chunks." `clear()` handles everything.

---

## `animus/rel/queries.py` — `formulate_queries`

Single file, all of it dispositioned at once.

### Site 11.1 — `formulate_queries(classified: ClassifiedInbound, ...) → list[ScopedQuery]`

**Disposition: GENERALIZE + REWRITE.**

- **Signature change:** `formulate_queries(message: str, *, context: str = "", llm: LLMCallable, prompt_override: str | None = None) -> list[Query]`. The `ClassifiedInbound` parameter dies (it's animus's engage-pipeline type containing classification metadata + the inbound message; prospecta just takes the message + optional context string).
- **Output:** animus's `ScopedQuery(text, scope: Literal["person", "general"])` → prospecta's `Query` from T6 (text, metadata_filter, tags, tags_match). The `scope` axis dies entirely.
- **Mechanism:** call `llm(messages, json_mode=True)` with the `formulate-queries.md` prompt from T7 (already translated). Parse `{"queries": [{"text": "..."}, ...]}` JSON output. Fall back to `[Query(text=message)]` on malformed JSON or schema mismatch, fire tracer event with `parse_fallback=True` (per schema.md §13 / T13 spec).

### Site 11.2 — Proof-token machinery (`_ClassifiedToken`, `_QueriedToken`, the `__slots__` sealing)

**Disposition: DELETE.**

Animus uses proof-token classes to enforce pipeline ordering (you can't construct `QueriedInbound` without calling `formulate_queries`, which proves classification happened first). Prospecta has no such pipeline; its API is direct. Drop the entire proof-token apparatus.

### Site 11.3 — `recall_scoped_parallel(...)` and `ThreadPoolExecutor` fan-out

**Disposition: GENERALIZE.**

The parallel-fan-out behavior is structural (run N recall queries concurrently, return list). Port the mechanic, drop the `scope` parameter. This becomes `Memory.recall(queries: list[Query])` internal implementation (T12, M4) — likely just a `ThreadPoolExecutor.map` over per-query SQL calls.

### Site 11.4 — Regex line-parsing of LLM output

**Disposition: DELETE.**

Animus parses `formulate_queries` output via regex over `[person]:` / `[general]:` line prefixes. Replaced by JSON-mode parsing (A8 fold). Schema.md §5 — the `formulate-queries.md` prompt now mandates `{"queries": [{"text": "..."}]}` JSON output.

---

## Summary tally

| Disposition | Site count | Approximate LOC |
|---|---|---|
| DELETE | 11 (Sites 1.2, 1.3, 1.4, 5.1, 5.2, 5.3, 5.5, 5.6, 7-partial, 10, 11.2, 11.4) | ~400 |
| GENERALIZE | 4 (Sites 4, 7-partial, 11.1, 11.3) | ~150 |
| SHIM | 1 (Site 1.1) | ~5 |
| PRESERVE | 2 (Site 5.4, 5.8) | ~150 |
| REWRITE | 5 (Sites 2, 3, 5.7, 6, 8, 9) | ~250 (animus) → ~150 (prospecta SQL-shaped) |

**Net:** ~1158 LOC animus → ~400-500 LOC prospecta `_index.py`, plus ~150 LOC `_formulate.py`. Substantially smaller because the entire animus-domain layer (person, episodes, log.jsonl, code-indexing) is gone, and the chroma-plumbing-to-Postgres mapping shrinks Python (Postgres does more per call).

---

## Test gates this sheet enables

T9 (vertical-slice) can now be implemented against:

1. **Concrete delete list** — the executor knows exactly which animus functions/methods to drop.
2. **Concrete rewrite list** — every chroma op has a named SQL replacement, sourced from schema.md §5/§6.
3. **Concrete preserve list** — the markdown `index_text:` parser ports as a bundled `ParserPlugin`; the chunker/ignore/parser modules from T8 are the helpers.
4. **The ParserPlugin Protocol** (decision-record-1 addendum) is the architectural seam between library-owned crawl and caller-owned parsing.

T13 (read-side spine) can be implemented against:

1. **Site 11.1** — the new signature for `formulate_queries`.
2. **Site 11.2** — proof-token machinery is gone.
3. **Site 11.3** — ThreadPoolExecutor fan-out stays.
4. **Site 11.4** — JSON-mode replaces regex parsing.

---

## Open questions surfaced by this sheet (for T9/T13 executors)

1. **Should `Memory.search()` expose all three modes (`hybrid` / `semantic` / `lexical`) in v0.1, or is `hybrid` enough?** Plan-v2 §3.4 says all three. T9 should implement all three; the cost is small (three small SQL functions sharing the same dispatching logic).

2. **Should the sweeper (T14) inherit `prune_stale()` from this port, or build its own?** Lean: `Memory.prune_stale()` is the canonical implementation; the sweeper calls it. CLI also calls it.

3. **`Memory.clear()` / `Memory.rebuild()` — v0.1 or v0.2?** Plan-v2 §3.4 has both in the v0.1 API. T9 ships them.

4. **Stale source tracking for `prune_stale`** — `documents.source` is a caller-supplied opaque string. The library MUST NOT introspect or classify it (don't check "does it look like a path?"). Filesystem-aware stale detection lives in the sweeper (T14), which walks corpus paths and calls `Memory.remove_documents([sources_gone])`. v0.1 `Memory.prune_stale()` removes only caller-flagged stale rows; it does not interpret `source`.

---

⚒️ Sheet locked 2026-05-19. T9 + T13 dispatch against this.
