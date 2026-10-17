# AGENTS.md: prospecta

Long-term memory for agents on Postgres + pgvector: verbatim notes written as chunks (no model at write
time), found by one SQL query (semantic + stemmed + exact) plus a cross-encoder, returned as cited evidence.
`README.md` is the overview; `docs/postgres-hybrid.md` is the one normative design; `PRINCIPLES.md` holds
the principles; `CHANGELOG.md` the release notes.

## Tests

`uv run --with pytest python -m pytest -q`. Postgres-backed tests start a pgvector testcontainer, or use a
throwaway database through `PROSPECTA_TEST_PG_URL` (`docs/testing.md`); never point that at a live bank.
Docs and skills links and front matter are checked without a database: `python tests/test_docs_links.py`
and `python -m pytest tests/test_docs_links.py tests/test_skills.py`.

## Invariants a small change can break

- No caps on content or evidence; a limit is physical, a design constant, or score-based (`docs/limits.md`).
- The write path calls no model. Recall degrades and says so; it never raises into the caller (P7, P22).
- An applied migration is never edited. Migration 0020 rewrites `memory_items`: read `docs/exact-tsv.md`.
- No real names or private note text in tests, panels, docs, issues or PRs: invented examples, counts only.

## Skills (`skills/`)

| skill | one line |
|---|---|
| [`recall`](skills/recall/SKILL.md) | Mode 1: when to search or skip, building the searches, the date-window rule, merging, the cited block |
| [`retain`](skills/retain/SKILL.md) | the chunk-only write path: note date, sources, idempotence, `exact_tsv`, why nothing else is written |
| [`search-and-ask`](skills/search-and-ask/SKILL.md) | typed search, Stage 1 and Stage 2, the fallbacks, citations, `ask` |
| [`tuning-and-testing`](skills/tuning-and-testing/SKILL.md) | changing recall without guessing: invented panels, before and after, read-only replay, counts-only reports |
| [`operating`](skills/operating/SKILL.md) | backups, migrations, restarts, the live-plugin rule, per-bank settings, what never to do |

Spire and rung-memory-prospecta adopt or adapt these skills; lessons found elsewhere come back as PRs to
these skills; every recall PR states the skill it follows or changes.
