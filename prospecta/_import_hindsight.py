"""Import a Hindsight bank into a prospecta bank.

Source: a Postgres database holding a Hindsight schema (restore a Hindsight
pg_dump / cold copy into a scratch Postgres first). Read-only on the source.
Idempotent: nothing is overwritten and re-runs add nothing. Every source row
of every Hindsight table lands in exactly one of imported / skipped_duplicate /
failed / not_carried (with a reason) in the ImportReport.
See docs/import-hindsight.md for the table-by-table mapping.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import psycopg

if TYPE_CHECKING:
    from prospecta.memory import Memory

SYSTEM = "hindsight"

_DDL = """
CREATE TABLE IF NOT EXISTS imported_records (
    bank_id    TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
    system     TEXT NOT NULL,
    kind       TEXT NOT NULL,
    source_key TEXT NOT NULL,
    data       JSONB NOT NULL,
    PRIMARY KEY (bank_id, system, kind, source_key)
);
CREATE TABLE IF NOT EXISTS imported_edges (
    bank_id TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
    system  TEXT NOT NULL,
    kind    TEXT NOT NULL,
    src     TEXT NOT NULL,
    dst     TEXT NOT NULL,
    subtype TEXT NOT NULL DEFAULT '',
    ref     TEXT NOT NULL DEFAULT '',
    weight  DOUBLE PRECISION,
    count   INTEGER,
    ts      TIMESTAMPTZ,
    PRIMARY KEY (bank_id, system, kind, src, dst, subtype, ref)
);
CREATE INDEX IF NOT EXISTS imported_edges_dst_idx ON imported_edges(bank_id, kind, dst);
"""

# table -> (kind, key columns, WHERE clause with one %s = hindsight bank id)
CARRIED = {
    "banks": ("bank", ["bank_id"], "bank_id = %s"),
    "documents": ("document", ["id"], "bank_id = %s"),
    "chunks": ("chunk", ["chunk_id"], "bank_id = %s"),
    "entities": ("entity", ["id"], "bank_id = %s"),
    "directives": ("directive", ["id"], "bank_id = %s"),
    "mental_models": ("mental_model", ["id"], "bank_id = %s"),
    "mental_model_history": ("mental_model_history", ["id"], "bank_id = %s"),
    "knowledge_pages": ("knowledge_page", ["id"], "bank_id = %s"),
    "observation_history": ("observation_history", ["id"], "bank_id = %s"),
    "invalidated_memory_units": ("invalidated_unit", ["id"], "bank_id = %s"),
    "file_storage": ("file_storage", ["storage_key"],
                     "storage_key IN (SELECT file_storage_key FROM documents WHERE bank_id = %s)"),
    "audit_log": ("audit_log", ["id"], "bank_id = %s"),
    "llm_requests": ("llm_request", ["id"], "bank_id = %s"),
}

# table -> (edge kind, SELECT of src,dst,subtype,ref,weight,count,ts, FROM/WHERE using %s = bank)
EDGES = {
    "memory_links": (
        "memory_link",
        "SELECT from_unit_id::text, to_unit_id::text, link_type, coalesce(entity_id::text,''),"
        " weight, 1, created_at FROM memory_links WHERE bank_id = %s"),
    "unit_entities": (
        "unit_entity",
        "SELECT ue.unit_id::text, ue.entity_id::text, '', '', 1.0, 1, NULL::timestamptz"
        " FROM unit_entities ue JOIN memory_units mu ON mu.id = ue.unit_id WHERE mu.bank_id = %s"),
    "entity_cooccurrences": (
        "cooccurrence",
        "SELECT c.entity_id_1::text, c.entity_id_2::text, '', '', 1.0, c.cooccurrence_count,"
        " c.last_cooccurred FROM entity_cooccurrences c"
        " JOIN entities e ON e.id = c.entity_id_1 WHERE e.bank_id = %s"),
}

# table -> (count SQL with %s = bank or None for whole table, reason)
NOT_CARRIED = {
    "alembic_version": (None, "Hindsight's own migration marker; meaningless in prospecta"),
    "async_operations": ("SELECT count(*) FROM async_operations WHERE bank_id = %s",
                         "transient Hindsight job queue; its work products (units, entities) are imported"),
    "bank_stats_cache": ("SELECT count(*) FROM bank_stats_cache WHERE bank_id = %s",
                         "derived cache; prospecta computes its own stats"),
    "graph_maintenance_queue": ("SELECT count(*) FROM graph_maintenance_queue WHERE bank_id = %s",
                                "transient Hindsight maintenance queue"),
    "webhooks": ("SELECT count(*) FROM webhooks WHERE bank_id = %s",
                 "outbound webhook config holding signing secrets; deliberately not copied"),
}


@dataclass
class Counts:
    total: int = 0
    imported: int = 0
    skipped_duplicate: int = 0
    failed: int = 0
    not_carried: int = 0


@dataclass
class ImportReport:
    hindsight_bank: str = ""
    prospecta_bank: str = ""
    tables: dict = field(default_factory=dict)
    embeddings: dict = field(default_factory=lambda: {"carried": 0, "re_embedded": 0})
    failures: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def counts(self, table: str) -> Counts:
        return self.tables.setdefault(table, Counts())

    @property
    def ok(self) -> bool:
        return not any(c.failed for c in self.tables.values())

    def to_dict(self) -> dict:
        return {
            "hindsight_bank": self.hindsight_bank,
            "prospecta_bank": self.prospecta_bank,
            "tables": {t: vars(c) for t, c in self.tables.items()},
            "embeddings": self.embeddings,
            "failures": self.failures,
            "notes": self.notes,
        }

    def render(self) -> str:
        lines = [f"hindsight bank {self.hindsight_bank!r} -> prospecta bank {self.prospecta_bank!r}"]
        for t, c in self.tables.items():
            if c.not_carried:
                lines.append(f"{t:26} in={c.total} NOT CARRIED")
            else:
                lines.append(f"{t:26} in={c.total} imported={c.imported} "
                             f"skipped_duplicate={c.skipped_duplicate} failed={c.failed}")
        lines.append(f"embeddings: carried={self.embeddings['carried']} "
                     f"re_embedded={self.embeddings['re_embedded']}")
        lines += [f"  note: {n}" for n in self.notes]
        lines += [f"  FAILED {f['table']} {f['key']}: {f['reason']}" for f in self.failures]
        return "\n".join(lines)


def _slug(s: str) -> str:
    return "-".join(str(s).lower().split())


def _batches(cur, size):
    while True:
        rows = cur.fetchmany(size)
        if not rows:
            return
        yield rows


def _table_exists(src, table) -> bool:
    with src.cursor() as cur:
        cur.execute("SELECT to_regclass(%s) IS NOT NULL", (f"public.{table}",))
        return cur.fetchone()[0]


def _ensure_bank(memory: "Memory", src, hbank: str, rep: ImportReport) -> int:
    """Create the target bank if missing (dim from Hindsight's embeddings)."""
    with memory._pool.cursor() as cur:
        cur.execute("SELECT embedding_dim FROM banks WHERE bank_id = %s",
                    (memory._default_bank_id,))
        row = cur.fetchone()
    if row:
        return row[0]
    with src.cursor() as cur:
        cur.execute("SELECT vector_dims(embedding), mission FROM memory_units m "
                    "LEFT JOIN banks b ON b.bank_id = m.bank_id "
                    "WHERE m.bank_id = %s AND embedding IS NOT NULL LIMIT 1", (hbank,))
        r = cur.fetchone()
    if not r:
        raise ValueError(f"bank {memory._default_bank_id!r} does not exist and the source has "
                         "no embeddings to size it; run `prospecta create-bank` first")
    memory.create_bank(memory._default_bank_id, embedding_dim=r[0], mission=r[1])
    rep.notes.append(f"created prospecta bank with embedding_dim={r[0]}")
    return r[0]


def _fail(rep, table, key, reason, n=1):
    rep.counts(table).failed += n
    rep.failures.append({"table": table, "key": str(key), "reason": reason})


def _import_units(memory, src, hbank, dim, rep, batch):
    c = rep.counts("memory_units")
    bank = memory._default_bank_id
    with src.cursor() as cur:
        cur.execute("SELECT count(*) FROM memory_units WHERE bank_id = %s", (hbank,))
        c.total = cur.fetchone()[0]
    sql = """
        SELECT mu.id::text, mu.text, mu.embedding::real[], mu.tags, mu.created_at, mu.updated_at,
               mu.fact_type,
               to_jsonb(mu) - 'embedding' - 'search_vector' - 'text',
               coalesce((SELECT array_agg(e.canonical_name) FROM unit_entities ue
                         JOIN entities e ON e.id = ue.entity_id WHERE ue.unit_id = mu.id), '{}')
        FROM memory_units mu WHERE mu.bank_id = %s ORDER BY mu.created_at, mu.id"""
    with src.cursor(name="hs_units") as cur:
        cur.execute(sql, (hbank,))
        for rows in _batches(cur, batch):
            with memory._pool.cursor() as tcur:
                tcur.execute("SELECT source FROM documents WHERE bank_id = %s AND source = ANY(%s)",
                             (bank, [f"hindsight:{r[0]}" for r in rows]))
                have = {s for (s,) in tcur.fetchall()}
            todo = []
            for r in rows:
                if f"hindsight:{r[0]}" in have:
                    c.skipped_duplicate += 1
                else:
                    todo.append(r)
            if not todo:
                continue
            carried = [r[2] is not None and len(r[2]) == dim for r in todo]
            redo = [r[1] for r, ok in zip(todo, carried) if not ok]
            vecs = {}
            if redo:
                if memory._embed is None:
                    err = RuntimeError("no embedder configured to re-embed units")
                    texts = []
                else:
                    err = None
                    texts = redo
                try:
                    if err:
                        raise err
                    out = list(memory._embed(texts))
                    if len(out) != len(texts):
                        raise RuntimeError("embedder returned wrong number of vectors")
                    vecs = dict(zip(range(len(texts)), out))
                except Exception as e:
                    # one bad text must not sink the batch: embed row by row
                    vecs = {}
                    if memory._embed is not None:
                        for k, t in enumerate(texts):
                            try:
                                vecs[k] = list(memory._embed([t]))[0]
                            except Exception:
                                pass
                    batch_err = e
                ri = 0
                keep = []
                for r, ok in zip(todo, carried):
                    if ok:
                        keep.append((r, r[2], True))
                    else:
                        if ri in vecs:
                            keep.append((r, vecs[ri], False))
                        else:
                            _fail(rep, "memory_units", r[0], f"re-embed failed: {batch_err}")
                        ri += 1
                todo_v = keep
            else:
                todo_v = [(r, r[2], True) for r in todo]
            _write_batch(memory, bank, hbank, todo_v, c, rep)


def _write_batch(memory, bank, hbank, items, c, rep):
    if not items:
        return
    try:
        with memory._pool.connection() as conn:
            for r, v, _ in items:
                _insert_unit(conn, bank, r, v)
            conn.commit()
        c.imported += len(items)
        for _, _, ok in items:
            rep.embeddings["carried" if ok else "re_embedded"] += 1
    except Exception:
        # isolate the bad row(s): retry one by one
        for r, v, ok in items:
            if _write_unit(memory, bank, hbank, r, v, c, rep):
                rep.embeddings["carried" if ok else "re_embedded"] += 1


def _write_unit(memory, bank, hbank, r, vec, c, rep) -> bool:
    try:
        with memory._pool.connection() as conn:
            _insert_unit(conn, bank, r, vec)
            conn.commit()
        c.imported += 1
        return True
    except Exception as e:
        _fail(rep, "memory_units", r[0], f"{type(e).__name__}: {e}")
        return False


def _insert_unit(conn, bank, r, vec):
    uid, text, _emb, tags, created, updated, fact_type, row, ents = r
    row = dict(row)
    tag_list = list(dict.fromkeys(
        [SYSTEM, f"fact_type:{fact_type}"] + [f"entity:{_slug(n)}" for n in ents]
        + [t for t in (tags or [])]))
    chash = hashlib.sha256(f"hindsight:{uid}\n{text}".encode("utf-8")).hexdigest()
    lit = "[" + ",".join(repr(float(x)) for x in vec) + "]"
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO documents (bank_id, source, original_text, content_hash, tags,
                                      document_metadata, created_at, updated_at)
               VALUES (%s,%s,%s,%s,%s,%s::jsonb,%s,%s) RETURNING id""",
            (bank, f"hindsight:{uid}", text, chash, tag_list,
             json.dumps({"hindsight": row, "hindsight_entities": list(ents)}),
             created, updated))
        doc_id = cur.fetchone()[0]
        cur.execute(
            """INSERT INTO memory_items (bank_id, document_id, content, original_chunk, embedding,
                                         metadata, tags, update_mode, llm_generated,
                                         created_at, updated_at)
               VALUES (%s,%s,%s,%s,%s::vector,%s::jsonb,%s,'append',false,%s,%s)""",
            (bank, doc_id, text, text, lit,
             json.dumps({"index_text_caller_supplied": True, "hindsight_id": uid,
                         "fact_type": fact_type}),
             tag_list, created, updated))


def _import_carried(memory, src, hbank, table, rep, batch):
    kind, keys, where = CARRIED[table]
    c = rep.counts(table)
    bank = memory._default_bank_id
    with src.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {table} WHERE {where}", (hbank,))
        c.total = cur.fetchone()[0]
    ins = """INSERT INTO imported_records (bank_id, system, kind, source_key, data)
             SELECT %s, %s, %s, u.k, u.d::jsonb FROM unnest(%s::text[], %s::text[]) AS u(k, d)
             ON CONFLICT DO NOTHING"""
    with src.cursor(name=f"hs_{table}") as cur:
        cur.execute(f"SELECT to_jsonb(t) FROM {table} t WHERE {where}", (hbank,))
        for rows in _batches(cur, batch):
            ks = [":".join(str(r[0][k]) for k in keys) for r in rows]
            ds = [json.dumps(r[0]) for r in rows]
            _insert_batch(memory, ins, (bank, SYSTEM, kind), table, c, rep, ks, [ks, ds])


def _import_edges(memory, src, hbank, table, rep, batch):
    kind, select = EDGES[table]
    c = rep.counts(table)
    bank = memory._default_bank_id
    with src.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM ({select}) q", (hbank,))
        c.total = cur.fetchone()[0]
    ins = """INSERT INTO imported_edges (bank_id, system, kind, src, dst, subtype, ref,
                                         weight, count, ts)
             SELECT %s, %s, %s, a, b, s, r, w, n, t
             FROM unnest(%s::text[], %s::text[], %s::text[], %s::text[],
                         %s::float8[], %s::int[], %s::timestamptz[]) AS u(a,b,s,r,w,n,t)
             ON CONFLICT DO NOTHING"""
    with src.cursor(name=f"hs_{table}") as cur:
        cur.execute(select, (hbank,))
        for rows in _batches(cur, batch):
            cols = list(zip(*rows))
            keys = [f"{r[0]}->{r[1]}/{r[2]}/{r[3]}" for r in rows]
            _insert_batch(memory, ins, (bank, SYSTEM, kind), table, c, rep, keys,
                          [list(x) for x in cols])


def _insert_batch(memory, sql, head, table, c, rep, keys, cols):
    def run(conn, idx):
        args = head + tuple([col[i] for i in idx] for col in cols)
        with conn.cursor() as cur:
            cur.execute(sql, args)
            return cur.rowcount
    try:
        with memory._pool.connection() as conn:
            n = run(conn, range(len(keys)))
            conn.commit()
        c.imported += n
        c.skipped_duplicate += len(keys) - n
    except Exception:
        for i in range(len(keys)):  # isolate the bad row(s)
            try:
                with memory._pool.connection() as conn:
                    n = run(conn, [i])
                    conn.commit()
                c.imported += n
                c.skipped_duplicate += 1 - n
            except Exception as e:
                _fail(rep, table, keys[i], f"{type(e).__name__}: {e}")


def list_banks(src_url: str) -> list[str]:
    with psycopg.connect(src_url) as src, src.cursor() as cur:
        cur.execute("SELECT bank_id FROM banks ORDER BY bank_id")
        return [r[0] for r in cur.fetchall()]


def import_bank(memory: "Memory", src_url: str, hbank: str, *, batch: int = 1000) -> ImportReport:
    """Import Hindsight bank `hbank` from the source DB into memory's bank."""
    rep = ImportReport(hindsight_bank=hbank, prospecta_bank=memory._default_bank_id)
    with psycopg.connect(src_url) as src:
        src.read_only = True
        with src.cursor() as cur:
            cur.execute("SELECT 1 FROM banks WHERE bank_id = %s", (hbank,))
            if cur.fetchone() is None:
                raise ValueError(f"no bank {hbank!r} in source")
        dim = _ensure_bank(memory, src, hbank, rep)
        with memory._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(_DDL)
            conn.commit()
        _import_units(memory, src, hbank, dim, rep, batch)
        for table in CARRIED:
            if _table_exists(src, table):
                _import_carried(memory, src, hbank, table, rep, batch)
            else:
                rep.notes.append(f"source has no table {table}")
        for table in EDGES:
            if _table_exists(src, table):
                _import_edges(memory, src, hbank, table, rep, batch)
            else:
                rep.notes.append(f"source has no table {table}")
        for table, (count_sql, reason) in NOT_CARRIED.items():
            n = 0
            if _table_exists(src, table):
                with src.cursor() as cur:
                    cur.execute(count_sql or f"SELECT count(*) FROM {table}",
                                (hbank,) if count_sql else ())
                    n = cur.fetchone()[0]
            c = rep.counts(table)
            c.total = c.not_carried = n
            rep.notes.append(f"{table}: not carried ({reason})")
    return rep
