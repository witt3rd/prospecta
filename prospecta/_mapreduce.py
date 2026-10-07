"""Map-reduce recall for SET / discovery questions (``what are X's nicknames``).

Top-k retrieval cannot answer a question whose answer is spread over many notes.
This mode fetches EVERY note tied to an entity (documents.person, the entity
and alias tables, an optional metadata filter), MAPs over them in batches (one
small LLM call per batch, each note's relevant excerpt) to extract the asked
facts with a citation per fact, then REDUCEs into one deduplicated cited list.
Completeness over speed. Bounded by an explicit batch size; every batch is
recorded in a progress trace (notes visited, facts found, cost)."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from prospecta._llmutil import llm_call_record, llm_text
from prospecta._template import render_prompt
from prospecta.db.queries import _meta_param

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 8
MAP_ATTEMPTS = 2   # a failed batch is retried once, then surfaced
DEFAULT_EXCERPT_CHARS = 6000   # a note longer than this is sent as its entity-mention paragraphs

_NOTES_SQL = """
SELECT d.id::text, d.source, d.original_text
FROM documents d
WHERE d.bank_id = %(bank)s
  AND (
        lower(btrim(d.person)) = ANY(%(names)s)
     OR d.id IN (
            SELECT m.document_id
            FROM memory_entities e
            JOIN memory_item_entities ie ON ie.entity_id = e.id
            JOIN memory_items m ON m.id = ie.item_id
            WHERE e.bank_id = %(bank)s AND e.norm = ANY(%(names)s))
  )
  AND (%(meta)s::jsonb IS NULL OR EXISTS (
            SELECT 1 FROM memory_items mi
            WHERE mi.document_id = d.id AND mi.metadata @> %(meta)s::jsonb))
ORDER BY d.source, d.id
"""


@dataclass
class MapReduceResult:
    synthesis: str
    items: list[dict]            # [{fact, notes: [name], citations: [{note, document_id, known}]}]
    citations: list[dict]        # [{note, document_id, known}]
    progress: list[dict]         # one record per batch
    calls: list[dict]            # LLM call records (map batches, then reduce)
    notes_visited: int = 0
    facts_found: int = 0
    plan: dict = field(default_factory=dict)


def _norm_name(name: str) -> str:
    n = name.strip().replace("\\", "/").rsplit("/", 1)[-1].casefold()
    return n[:-3] if n.endswith(".md") else n


def _norm_fact(fact: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", fact.casefold())).strip()


def fetch_entity_notes(conn, bank_id: str, names: list[str],
                       metadata_filter: dict | None = None) -> list[tuple[str, str, str]]:
    """(document_id, source, text) of every note tied to the entity: its
    documents.person, or any item mentioning an entity whose norm is one of
    `names` (the entity name and its aliases), within the optional filter."""
    norms = sorted({re.sub(r"\s+", " ", n.strip().lower()) for n in names if n.strip()})
    if not norms:
        raise ValueError("map-reduce recall needs an entity name")
    with conn.cursor() as cur:
        cur.execute(_NOTES_SQL, {"bank": bank_id, "names": norms,
                                 "meta": _meta_param(metadata_filter)})
        return [(r[0], r[1] or r[0], r[2]) for r in cur.fetchall()]


def excerpt(text: str, names: list[str], max_chars: int = DEFAULT_EXCERPT_CHARS) -> str:
    """The note whole when it fits `max_chars`; otherwise the paragraphs that
    mention the entity (plus the first paragraph), in document order."""
    if len(text) <= max_chars:
        return text
    paras = re.split(r"\n\s*\n", text)
    terms = [n.casefold() for n in names if n.strip()]
    keep = [i for i, p in enumerate(paras)
            if i == 0 or any(t in p.casefold() for t in terms)]
    return "\n\n".join(paras[i] for i in keep)


def _render_notes(batch: list[tuple[str, str, str]], names: list[str],
                  max_chars: int) -> str:
    return "\n\n".join(f"### [{src}]\n{excerpt(text, names, max_chars)}"
                       for _, src, text in batch)


def _call(llm, prompt: str, *, purpose: str, calls: list[dict]) -> str:
    t0 = time.monotonic()
    out = llm([{"role": "user", "content": prompt}], json_mode=True)
    rec = llm_call_record(out, purpose=purpose)
    rec.update(duration_ms=int((time.monotonic() - t0) * 1000), prompt_text=prompt,
               response_text=llm_text(out), json_mode=True, messages_count=1, error=None)
    calls.append(rec)
    return llm_text(out)


def _facts_of(raw: str, key: str) -> list[dict]:
    from prospecta.stages import parse_json_object
    v = parse_json_object(raw).get(key)
    return [e for e in v if isinstance(e, dict) and isinstance(e.get("fact"), str)
            and e["fact"].strip()] if isinstance(v, list) else []


def _cost(calls: list[dict]) -> float:
    return float(sum(c["cost_usd"] for c in calls if c.get("cost_usd") is not None))


def run_mapreduce(conn, bank_id: str, question: str, llm, *, entity: str,
                  aliases: list[str] | None = None, metadata_filter: dict | None = None,
                  batch_size: int = DEFAULT_BATCH_SIZE,
                  excerpt_chars: int = DEFAULT_EXCERPT_CHARS,
                  on_progress=None) -> MapReduceResult:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    names = [entity, *(aliases or [])]
    notes = fetch_entity_notes(conn, bank_id, names, metadata_filter)
    by_norm = {_norm_name(src): (doc, src) for doc, src, _ in notes}
    calls: list[dict] = []
    progress: list[dict] = []
    raw_facts: list[dict] = []   # {fact, note}
    visited = 0
    failed_notes: list[str] = []
    n_batches = (len(notes) + batch_size - 1) // batch_size
    for b in range(n_batches):
        batch = notes[b * batch_size:(b + 1) * batch_size]
        prompt = render_prompt("mapreduce-map", {
            "query": question, "context": _render_notes(batch, names, excerpt_chars)})
        found = 0
        err = None
        for attempt in range(1, MAP_ATTEMPTS + 1):   # retry once; never drop a batch silently
            try:
                got = _facts_of(_call(llm, prompt, purpose="mapreduce_map", calls=calls), "facts")
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                logger.warning("map-reduce batch %d/%d attempt %d/%d failed: %s",
                               b + 1, n_batches, attempt, MAP_ATTEMPTS, err)
                continue
            err = None
            for e in got:
                raw_facts.append({"fact": e["fact"].strip(), "note": str(e.get("note", "")).strip()})
            found = len(got)
            break
        if err is not None:
            failed_notes.extend(src for _, src, _ in batch)
            logger.warning("map-reduce batch %d/%d FAILED after %d attempts, %d note(s) NOT read: %s (%s)",
                           b + 1, n_batches, MAP_ATTEMPTS, len(batch),
                           ", ".join(src for _, src, _ in batch), err)
        visited += len(batch)
        rec = {"batch": b + 1, "of": n_batches, "notes": len(batch), "notes_visited": visited,
               "facts_found": found, "facts_total": len(raw_facts),
               "cost_usd": _cost(calls), "error": err}
        progress.append(rec)
        if on_progress:
            on_progress(rec)

    items = _reduce(question, raw_facts, llm, calls)
    citations: list[dict] = []
    seen: set[str] = set()
    for it in items:
        it["citations"] = []
        for n in it["notes"]:
            doc, src = by_norm.get(_norm_name(n), (None, n))
            c = {"note": src, "document_id": doc, "known": doc is not None}
            it["citations"].append(c)
            if _norm_name(src) not in seen:
                seen.add(_norm_name(src))
                citations.append(c)
    text = "\n".join(f"- {it['fact']} " + " ".join(f"[{n}]" for n in
                     dict.fromkeys(c["note"] for c in it["citations"])) for it in items) \
        or "not in memory"
    if failed_notes:
        text += ("\nWARNING: incomplete. These notes could not be read (their batch failed after "
                 f"{MAP_ATTEMPTS} attempts): " + ", ".join(failed_notes))
    return MapReduceResult(
        synthesis=text, items=items, citations=citations, progress=progress, calls=calls,
        notes_visited=visited, facts_found=len(items),
        plan={"mode": "mapreduce", "entity": entity, "aliases": list(aliases or []),
              "batch_size": batch_size, "n_notes": len(notes), "n_batches": n_batches,
              "metadata_filter": metadata_filter, "progress": progress,
              "raw_facts": len(raw_facts), "failed_notes": failed_notes})


def _dedupe(raw_facts: list[dict]) -> list[dict]:
    """Deterministic merge: same normalised fact text -> one item, notes unioned."""
    merged: dict[str, dict] = {}
    for f in raw_facts:
        it = merged.setdefault(_norm_fact(f["fact"]) or f["fact"], {"fact": f["fact"], "notes": []})
        if f["note"] and f["note"] not in it["notes"]:
            it["notes"].append(f["note"])
    return list(merged.values())


def _reduce(question: str, raw_facts: list[dict], llm, calls: list[dict]) -> list[dict]:
    """LLM merge of paraphrased duplicates; on any failure, or when the model
    drops a note-fact pair it was given, fall back to the deterministic merge."""
    base = _dedupe(raw_facts)
    if len(base) < 2:
        return base
    listing = "\n".join(f"- {f['fact']} [{f['note']}]" for f in raw_facts)
    prompt = render_prompt("mapreduce-reduce", {"query": question, "facts": listing})
    try:
        merged = _facts_of(_call(llm, prompt, purpose="mapreduce_reduce", calls=calls), "items")
    except Exception as exc:
        logger.warning("map-reduce reduce failed, using the deterministic merge: %s", exc)
        return base
    out = []
    for e in merged:
        notes = [str(n).strip() for n in (e.get("notes") or []) if str(n).strip()]
        out.append({"fact": e["fact"].strip(), "notes": list(dict.fromkeys(notes))})
    given = {_norm_name(f["note"]) for f in raw_facts if f["note"]}
    kept = {_norm_name(n) for it in out for n in it["notes"]}
    if not out or not given <= kept:   # completeness over tidiness
        logger.warning("map-reduce reduce lost notes; using the deterministic merge")
        return base
    return out
