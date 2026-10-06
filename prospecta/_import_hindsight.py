"""Import a Hindsight bank dump into a prospecta bank.

Idempotent: every memory unit becomes one document with the stable source
``hindsight:<unit id>``; a unit whose source already exists is skipped, so
re-running the same dump adds nothing. Nothing is dropped silently — every
input record lands in exactly one of imported / skipped / failed in the
ImportReport. See docs/import-hindsight.md for the assumed dump schema.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prospecta.memory import Memory

SOURCE_PREFIX = "hindsight:"


@dataclass
class KindCounts:
    total: int = 0
    imported: int = 0
    skipped: int = 0
    failed: int = 0


@dataclass
class ImportReport:
    units: KindCounts = field(default_factory=KindCounts)
    entities: KindCounts = field(default_factory=KindCounts)
    links: KindCounts = field(default_factory=KindCounts)
    skipped: list[dict] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "units": vars(self.units),
            "entities": vars(self.entities),
            "links": vars(self.links),
            "skipped": self.skipped,
            "failed": self.failed,
            "warnings": self.warnings,
        }

    def render(self) -> str:
        lines = []
        for name in ("units", "entities", "links"):
            c = getattr(self, name)
            lines.append(
                f"{name:9} in={c.total} imported={c.imported} "
                f"skipped_duplicate={c.skipped} failed={c.failed}"
            )
        for label, items in (("skipped", self.skipped), ("failed", self.failed),
                             ("warning", self.warnings)):
            for it in items:
                lines.append(f"  {label}: {it['kind']} {it['id']}: {it['reason']}")
        return "\n".join(lines)


def _parse_ts(value) -> datetime | None:
    if value in (None, ""):
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _slug(s: str) -> str:
    return "-".join(s.lower().split())


def load_dump(path: Path) -> dict:
    """Load a dump: one JSON object, or JSONL with a "kind" field per line."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        return data
    if data is not None:
        raise ValueError("dump must be a JSON object or JSONL records")
    data = {"memory_units": [], "entities": [], "links": []}
    plural = {"memory_unit": "memory_units", "unit": "memory_units",
              "entity": "entities", "link": "links"}
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        if not isinstance(rec, dict):
            raise ValueError(f"line {n}: record is not an object")
        key = plural.get(rec.get("kind"))
        if key is None:
            raise ValueError(f"line {n}: unknown or missing kind {rec.get('kind')!r}")
        data[key].append(rec)
    return data


def import_hindsight(
    memory: "Memory", dump: dict, *, synthesize: bool = False
) -> ImportReport:
    rep = ImportReport()
    units = dump.get("memory_units") or []
    entities = dump.get("entities") or []
    links = dump.get("links") or []

    ent_by_id: dict[str, dict] = {}
    for e in entities:
        rep.entities.total += 1
        eid = e.get("id") if isinstance(e, dict) else None
        if eid is None or not (e.get("name") or e.get("canonical_name")):
            rep.entities.failed += 1
            rep.failed.append({"kind": "entity", "id": str(eid),
                               "reason": "entity needs id and name"})
            continue
        ent_by_id[str(eid)] = e

    unit_ids = {str(u["id"]) for u in units if isinstance(u, dict) and u.get("id")}
    out_links: dict[str, list[dict]] = {}
    link_labels: dict[str, list[str]] = {}
    outcome: dict[str, tuple[str, str]] = {}
    for l in links:
        rep.links.total += 1
        lid = f"{l.get('from_unit_id')}->{l.get('to_unit_id')}" if isinstance(l, dict) else "?"
        src, dst = (str(l.get("from_unit_id")), str(l.get("to_unit_id"))) if isinstance(l, dict) else (None, None)
        if src not in unit_ids or dst not in unit_ids:
            rep.links.failed += 1
            rep.failed.append({"kind": "link", "id": lid,
                               "reason": "endpoint not in dump's memory_units"})
            continue
        out_links.setdefault(src, []).append(
            {"to": dst, "type": l.get("link_type"), "weight": l.get("weight"),
             "entity_id": l.get("entity_id")})
        link_labels.setdefault(src, []).append(lid)

    referenced_entities: set[str] = set()
    for u in units:
        rep.units.total += 1
        uid = str(u.get("id")) if isinstance(u, dict) and u.get("id") else None
        label = uid or f"#{rep.units.total}"
        try:
            text = u.get("text") if isinstance(u, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise ValueError("missing or empty text")
            if uid is None:
                raise ValueError("missing id (needed for idempotency)")
            source = SOURCE_PREFIX + uid
            chash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            with memory._pool.cursor() as cur:
                cur.execute(
                    "SELECT content_hash FROM documents "
                    "WHERE bank_id=%s AND source=%s",
                    (memory._default_bank_id, source))
                row = cur.fetchone()
                if row is None:
                    cur.execute(
                        "SELECT source FROM documents "
                        "WHERE bank_id=%s AND content_hash=%s",
                        (memory._default_bank_id, chash))
                    dup = cur.fetchone()
                else:
                    dup = None
            if row is not None:
                reason = ("already imported" if row[0] == chash else
                          "source already imported with different text; kept existing")
                rep.units.skipped += 1
                rep.skipped.append({"kind": "unit", "id": uid, "reason": reason})
                outcome[uid] = ("skipped", reason)
                continue
            if dup is not None:
                rep.units.skipped += 1
                reason = f"identical text already stored as {dup[0]}"
                rep.skipped.append({"kind": "unit", "id": uid, "reason": reason})
                outcome[uid] = ("skipped", reason)
                continue

            ents = []
            for ref in u.get("entities") or []:
                if isinstance(ref, dict):
                    ref = ref.get("id") or ref.get("name")
                rec = ent_by_id.get(str(ref))
                if rec is not None:
                    referenced_entities.add(str(ref))
                    ents.append({k: v for k, v in rec.items()})
                else:
                    ents.append({"name": str(ref)})  # bare name, as Hindsight lists them
            names = [e.get("name") or e.get("canonical_name") for e in ents]
            tags = list(dict.fromkeys(
                ["hindsight"]
                + ([f"fact_type:{u['fact_type']}"] if u.get("fact_type") else [])
                + [f"entity:{_slug(n)}" for n in names if n]
                + [str(t) for t in (u.get("tags") or [])]))
            known = {"id", "text", "context", "fact_type", "entities", "tags",
                     "metadata", "created_at", "event_date", "occurred_start",
                     "occurred_end", "mentioned_at", "document_id", "observations",
                     "proof_count", "source_unit_ids"}
            meta = {
                "hindsight_id": uid,
                "hindsight_fact_type": u.get("fact_type"),
                "hindsight_context": u.get("context"),
                "hindsight_document_id": u.get("document_id"),
                "hindsight_entities": ents,
                "hindsight_links": out_links.get(uid, []),
                "hindsight_timestamps": {k: u.get(k) for k in (
                    "created_at", "event_date", "occurred_start", "occurred_end",
                    "mentioned_at") if u.get(k) is not None},
                "hindsight_metadata": u.get("metadata") or {},
                # observation-only provenance (what consolidated it)
                "hindsight_provenance": {k: u[k] for k in (
                    "proof_count", "source_unit_ids", "observations") if k in u},
                "hindsight_extra": {k: v for k, v in u.items() if k not in known},
            }
            created = None
            try:
                created = _parse_ts(u.get("created_at"))
            except ValueError:
                rep.warnings.append({"kind": "unit", "id": uid,
                                     "reason": "unparseable created_at; kept raw in metadata"})
            doc_id = memory.retain(
                text,
                source=source,
                tags=tags,
                metadata=meta,
                index_text=None if synthesize else [text],
            )
            if created is not None:
                with memory._pool.cursor() as cur:
                    cur.execute("UPDATE documents SET created_at=%s WHERE id=%s",
                                (created, doc_id))
                    cur.execute("UPDATE memory_items SET created_at=%s "
                                "WHERE document_id=%s", (created, doc_id))
            rep.units.imported += 1
            outcome[uid] = ("imported", "")
        except Exception as e:  # reported, never dropped
            rep.units.failed += 1
            rep.failed.append({"kind": "unit", "id": label,
                               "reason": f"{type(e).__name__}: {e}"})
            if uid is not None:
                outcome[uid] = ("failed", f"{type(e).__name__}: {e}")

    for src, labels in link_labels.items():
        status, why = outcome.get(src, ("failed", "source unit not processed"))
        for lid in labels:
            if status == "imported":
                rep.links.imported += 1
            elif status == "skipped":
                rep.links.skipped += 1
                rep.skipped.append({"kind": "link", "id": lid,
                                    "reason": f"source unit skipped: {why}"})
            else:
                rep.links.failed += 1
                rep.failed.append({"kind": "link", "id": lid,
                                   "reason": f"source unit failed: {why}"})

    # Entities are carried inside the units that mention them; ones nothing
    # references have nowhere to live, so report them rather than drop quietly.
    for eid in ent_by_id:
        if eid in referenced_entities:
            rep.entities.imported += 1
        else:
            rep.entities.skipped += 1
            rep.skipped.append({"kind": "entity", "id": eid,
                                "reason": "no imported unit references it (or units were skipped)"})
    return rep
