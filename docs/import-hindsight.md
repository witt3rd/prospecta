# Importing a Hindsight bank dump

```bash
prospecta --bank mybank import hindsight dump.json [--synthesize] [--json]
```

Exit code 0 = everything accounted for with no failures; 3 = some records failed (listed in the report; the rest were still imported).

## Assumed dump schema

No Hindsight dump format or code was found under `~/src` or `~/Documents`, so the reader targets this minimal schema, modelled on Hindsight's memory-unit/entity/link model. Unknown fields are never dropped (kept in metadata).

A single JSON object, or JSONL with one record per line carrying `"kind": "memory_unit" | "entity" | "link"`:

```json
{
  "bank_id": "demo",
  "entities": [{"id": "e1", "name": "Alice", "aliases": ["Al"]}],
  "memory_units": [{
    "id": "u1", "text": "Alice works at Acme.", "fact_type": "world",
    "entities": ["e1"], "tags": ["work"], "context": "chat",
    "created_at": "2024-03-01T10:00:00Z", "occurred_start": "...", "occurred_end": "...",
    "event_date": "...", "mentioned_at": "...", "document_id": "d1",
    "metadata": {}, "source_unit_ids": ["u0"], "proof_count": 2
  }],
  "links": [{"from_unit_id": "u1", "to_unit_id": "u2", "link_type": "entity", "weight": 1.0}]
}
```

`fact_type` is `world`, `experience` or `observation` (observations carry `source_unit_ids` / `proof_count` provenance). A unit's `entities` may be entity ids or bare names. Only `id` and `text` are required on a unit.

## Mapping

- Each memory unit → one prospecta document, `source = hindsight:<unit id>`, body = `text` (verbatim), `index_text` = the text itself (no LLM, no spend). `--synthesize` uses the configured LLM to author anticipated questions instead (P1).
- Tags: `hindsight`, `fact_type:<type>`, `entity:<slug>`, plus the unit's own tags.
- Document metadata keeps: `hindsight_id`, fact type, context, document id, resolved entity records, outgoing links, all timestamps, original metadata, observation provenance, and any unrecognised fields.
- `created_at` becomes the document/item `created_at` (an unparseable value is a warning; the raw value stays in metadata).
- Links are stored on the source unit's metadata (prospecta has no link table). Entities are stored inside the units that reference them.

## Idempotency

A unit whose `hindsight:<id>` source already exists in the bank is skipped, so re-running the same dump adds nothing. Existing documents are never replaced.

## No-loss report

Links are carried in their source unit, so a link counts as imported only if its source unit was imported; if the source unit was skipped or failed, the link is reported skipped or failed with that reason.

For units, entities and links: `in = imported + skipped_duplicate + failed`. Every skipped/failed record is listed with its reason, e.g. already imported; identical text already stored under another source; empty text or missing id; link endpoint not in the dump; entity with no id/name; entity no imported unit references.
