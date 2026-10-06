"""Hindsight import: synthetic dump, idempotency, no-loss report."""
from __future__ import annotations

import json

from prospecta._import_hindsight import import_hindsight, load_dump


def _dump():
    return {
        "bank_id": "demo",
        "entities": [
            {"id": "e1", "name": "Alice", "aliases": ["Al"]},
            {"id": "e2", "name": "Acme Corp"},
            {"id": "e3", "name": "Orphan"},
            {"name": "no id"},
        ],
        "memory_units": [
            {"id": "u1", "text": "Alice works at Acme Corp.", "fact_type": "world",
             "entities": ["e1", "e2"], "tags": ["work"], "context": "chat",
             "created_at": "2024-03-01T10:00:00Z", "occurred_start": "2023-01-01T00:00:00Z",
             "metadata": {"k": "v"}, "document_id": "d1"},
            {"id": "u2", "text": "Alice prefers tea.", "fact_type": "experience",
             "entities": ["Alice"], "created_at": "2024-03-02T10:00:00Z"},
            {"id": "u3", "text": "Alice likes tea and works at Acme.", "fact_type": "observation",
             "source_unit_ids": ["u1", "u2"], "proof_count": 2},
            {"id": "u4", "text": "Alice prefers tea."},      # same text as u2
            {"id": "u5", "text": "  "},                      # bad
            {"text": "no id"},                               # bad
            {"id": "u7", "text": "Odd date.", "created_at": "yesterday-ish"},
        ],
        "links": [
            {"from_unit_id": "u1", "to_unit_id": "u2", "link_type": "entity", "weight": 1.0},
            {"from_unit_id": "u1", "to_unit_id": "gone", "link_type": "semantic"},
        ],
    }


def test_import_report_and_idempotency(memory_with_bank):
    m = memory_with_bank
    r = import_hindsight(m, _dump())
    assert (r.units.total, r.units.imported, r.units.skipped, r.units.failed) == (7, 4, 1, 2)
    assert (r.links.total, r.links.imported, r.links.failed) == (2, 1, 1)
    assert (r.entities.total, r.entities.imported, r.entities.skipped, r.entities.failed) == (4, 2, 1, 1)
    assert r.units.total == r.units.imported + r.units.skipped + r.units.failed
    assert len(r.warnings) == 1
    docs = m.bank_stats().documents
    assert docs == 4

    r2 = import_hindsight(m, _dump())
    assert r2.units.imported == 0
    assert m.bank_stats().documents == docs
    assert r2.units.skipped == 5  # 4 already imported + u4 duplicate text... still accounted
    assert r2.units.total == r2.units.imported + r2.units.skipped + r2.units.failed

    with m._pool.cursor() as cur:
        cur.execute("SELECT tags, document_metadata, created_at FROM documents "
                    "WHERE source='hindsight:u1'")
        tags, meta, created = cur.fetchone()
    assert "entity:alice" in tags and "fact_type:world" in tags and "work" in tags
    assert meta["hindsight_links"][0]["to"] == "u2"
    assert meta["hindsight_entities"][0]["aliases"] == ["Al"]
    assert created.year == 2024 and created.month == 3


def test_load_jsonl(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in [
        {"kind": "memory_unit", "id": "a", "text": "x"},
        {"kind": "entity", "id": "e", "name": "N"}]))
    d = load_dump(p)
    assert len(d["memory_units"]) == 1 and len(d["entities"]) == 1


def test_cli_import(memory_with_bank, tmp_path, monkeypatch, capsys):
    from prospecta.cli import _common
    from prospecta.cli.__main__ import main
    monkeypatch.setattr(_common, "make_memory", lambda args: memory_with_bank)
    p = tmp_path / "dump.json"
    p.write_text(json.dumps(_dump()))
    assert main(["import", "hindsight", str(p), "--json"]) == 3  # has failures, reported
    out = json.loads(capsys.readouterr().out)
    assert out["units"]["imported"] == 4
    assert main(["import", "hindsight", str(p)]) == 3
