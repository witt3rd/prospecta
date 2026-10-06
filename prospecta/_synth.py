"""Grounded, cited synthesis over the blended evidence (design 8.8).

Opt-in mode of recall_synth: the evidence is the best chunk of each of the top
notes plus its neighbours (at most 3 chunks per note), or the whole scope set;
the model cites [note name] for every claim and may answer "not in memory";
the citation list comes back with the answer and is stored on the recall event.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from prospecta._template import render_prompt

TOP_NOTES = 6
CHUNKS_PER_NOTE = 3
SCOPE_MAX = 12   # a scope set larger than this is not a "set": fall back to top notes

_CHUNKS_SQL = """
SELECT original_chunk, ordinal
FROM memory_items
WHERE bank_id = %s AND document_id = %s::uuid AND kind = 'chunk'
"""
_SCOPE_SQL = """
SELECT id::text, source FROM documents
WHERE bank_id = %s AND (id::text = ANY(%s) OR source = ANY(%s))
ORDER BY source
"""


@dataclass(frozen=True)
class NoteEvidence:
    name: str
    document_id: str
    chunks: list[str]


@dataclass
class GroundedResult:
    synthesis: str
    prompt: str
    notes: list[NoteEvidence]
    citations: list[dict]
    set_mode: bool
    call: dict = field(default_factory=dict)   # model, tokens, cost, duration_ms


def _norm(name: str) -> str:
    n = name.strip().replace("\\", "/").rsplit("/", 1)[-1].casefold()
    return n[:-3] if n.endswith(".md") else n


def pick_chunks(rows: list[tuple[str, int | None]], best_text: str,
                best_index: int | None, n: int = CHUNKS_PER_NOTE) -> list[str]:
    """The best chunk plus its nearest neighbours by ordinal, at most `n`,
    returned in document order. Without chunk rows, the best text alone."""
    indexed = sorted((i, t) for t, i in rows if i is not None)
    if not indexed:
        return [best_text] if best_text else []
    pos = next((p for p, (i, _) in enumerate(indexed) if i == best_index), None)
    if pos is None:
        pos = next((p for p, (_, t) in enumerate(indexed) if t == best_text), None)
    if pos is None:   # best came from a non-chunk item: show it, then the note's head
        head = [t for _, t in indexed[: n - 1]]
        return ([best_text] if best_text else []) + head
    chosen, lo, hi = [pos], pos - 1, pos + 1
    while len(chosen) < n and (lo >= 0 or hi < len(indexed)):
        if hi < len(indexed):
            chosen.append(hi)
            hi += 1
        if len(chosen) < n and lo >= 0:
            chosen.append(lo)
            lo -= 1
    return [indexed[p][1] for p in sorted(chosen)]


def _chunk_rows(conn, bank_id: str, document_id: str) -> list[tuple[str, int | None]]:
    with conn.cursor() as cur:
        cur.execute(_CHUNKS_SQL, (bank_id, document_id))
        return [(t, int(i) if i is not None else None) for t, i in cur.fetchall()]


def resolve_scope(conn, bank_id: str, scope: list[str]) -> list[tuple[str, str]]:
    """(document_id, source) for each note named by id or source."""
    with conn.cursor() as cur:
        cur.execute(_SCOPE_SQL, (bank_id, list(scope), list(scope)))
        return [(r[0], r[1]) for r in cur.fetchall()]


def gather_evidence(conn, bank_id: str, recalled: list, *, top: int = TOP_NOTES,
                    scope: list[str] | None = None,
                    per_note: int = CHUNKS_PER_NOTE) -> tuple[list[NoteEvidence], bool]:
    """Evidence notes for the synthesizer. `recalled` is the blended recall in
    rank order (RecalledMemory, one per note). Returns (notes, set_mode)."""
    by_doc = {str(r.document_id): r for r in recalled}
    set_mode = False
    order: list[tuple[str, str]] = []
    if scope:
        members = resolve_scope(conn, bank_id, scope)
        if 0 < len(members) <= SCOPE_MAX:
            set_mode = True
            order = members
    if not order:
        seen: set[str] = set()
        for r in recalled:
            d = str(r.document_id)
            if d not in seen:
                seen.add(d)
                order.append((d, r.source))
            if len(order) >= top:
                break
    notes = []
    for doc_id, source in order:
        r = by_doc.get(doc_id)
        best_text = r.original_chunk if r else ""
        chunks = pick_chunks(_chunk_rows(conn, bank_id, doc_id), best_text, None, per_note)
        if chunks:
            notes.append(NoteEvidence(name=source or doc_id, document_id=doc_id, chunks=chunks))
    return notes, set_mode


def render_context(notes: list[NoteEvidence]) -> str:
    if not notes:
        return "(no evidence retrieved)"
    parts = []
    for n in notes:
        body = "\n\n".join(f"(excerpt {i})\n{c}" for i, c in enumerate(n.chunks, start=1))
        parts.append(f"### [{n.name}]\n{body}")
    return "\n\n".join(parts)


def extract_citations(text: str, notes: list[NoteEvidence]) -> list[dict]:
    """Bracketed names in the answer, in order of first citation, matched to the
    evidence notes (case, folder and .md insensitive)."""
    by_norm = {_norm(n.name): n for n in notes}
    out: list[dict] = []
    seen: set[str] = set()
    for m in re.finditer(r"\[([^\[\]\n]+)\]", text):
        for part in m.group(1).split(";"):
            key = _norm(part)
            if not key or key in seen:
                continue
            note = by_norm.get(key)
            if note is None and not ("." in part or " " in part.strip()):
                continue   # a bare word in brackets is not a note name
            seen.add(key)
            out.append({"note": note.name if note else part.strip(),
                        "document_id": note.document_id if note else None,
                        "known": note is not None})
    return out


def synthesize_grounded(query: str, notes: list[NoteEvidence], llm, *,
                        set_mode: bool = False, model: str | None = None) -> GroundedResult:
    prompt = render_prompt("synthesize-grounded", {
        "query": query, "context": render_context(notes), "set_mode": set_mode})
    t0 = time.monotonic()
    out = llm(messages=[{"role": "user", "content": prompt}])
    call = {"purpose": "synthesize_grounded", "model": model, "tokens_in": None,
            "tokens_out": None, "cost_usd": None}
    if not isinstance(out, str):   # an LLMResult: text plus accounting
        call.update(model=getattr(out, "model", None) or model,
                    tokens_in=getattr(out, "tokens_in", None),
                    tokens_out=getattr(out, "tokens_out", None),
                    cost_usd=getattr(out, "cost_usd", None))
        out = out.text
    call["duration_ms"] = int((time.monotonic() - t0) * 1000)
    return GroundedResult(synthesis=out, prompt=prompt, notes=notes,
                          citations=extract_citations(out, notes), set_mode=set_mode,
                          call=call)
