"""Grounded, cited synthesis over the blended evidence (design 8.8).

Opt-in mode of recall_synth: the evidence is the best chunk of each
high-scoring note plus its near-scoring chunks, or the whole scope set;
the model cites [note name] for every claim and may answer "not in memory";
the citation list comes back with the answer and is stored on the recall event.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from prospecta._template import render_prompt

logger = logging.getLogger(__name__)

DEFAULT_MIN_REL_SCORE = 0.5       # evidence_min_rel_score: keep notes/chunks scoring >= this x the best
DEFAULT_CONTEXT_TOKENS = 200_000  # physical: the synthesizer model's context window (config override)
PROMPT_RESERVE_TOKENS = 4_000     # physical headroom for the template, the question and the answer
CHARS_PER_TOKEN = 4               # token estimate

_CHUNKS_SQL = """
SELECT original_chunk, ordinal,
       COALESCE(1 - (embedding <=> %s::vector), 0)::float
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


def pick_chunks(rows: list[tuple[str, int | None, float]], best_text: str,
                min_rel: float = DEFAULT_MIN_REL_SCORE) -> list[tuple[float, int, str]]:
    """The chunks of a note scored at or above `min_rel` x the note's best chunk
    score, plus the recalled best chunk itself. (score, position, text), best first;
    position is the document order."""
    indexed = sorted(((i, t, sc) for t, i, sc in rows if i is not None), key=lambda r: r[0])
    top = max((sc for _, _, sc in indexed), default=0.0)
    out = [(sc, p, t) for p, (_, t, sc) in enumerate(indexed)
           if t == best_text or (top > 0 and sc >= min_rel * top)]
    if best_text and not any(t == best_text for _, _, t in out):
        out.append((top, -1, best_text))
    return sorted(out, key=lambda c: (-c[0], c[1]))


def _chunk_rows(conn, bank_id: str, document_id: str,
                qvec: str) -> list[tuple[str, int | None, float]]:
    with conn.cursor() as cur:
        cur.execute(_CHUNKS_SQL, (qvec, bank_id, document_id))
        return [(t, int(i) if i is not None else None, float(sc))
                for t, i, sc in cur.fetchall()]


def resolve_scope(conn, bank_id: str, scope: list[str]) -> list[tuple[str, str]]:
    """(document_id, source) for each note named by id or source."""
    with conn.cursor() as cur:
        cur.execute(_SCOPE_SQL, (bank_id, list(scope), list(scope)))
        return [(r[0], r[1]) for r in cur.fetchall()]


def gather_evidence(conn, bank_id: str, recalled: list, *, top: int | None = None,
                    qvec: str, scope: list[str] | None = None,
                    min_rel_score: float = DEFAULT_MIN_REL_SCORE,
                    context_tokens: int = DEFAULT_CONTEXT_TOKENS,
                    ) -> tuple[list[NoteEvidence], bool]:
    """Evidence notes for the synthesizer. `recalled` is the blended recall in
    rank order (RecalledMemory, one per note). Notes and chunks are chosen by
    score (>= `min_rel_score` x the best); an explicit `scope` is the whole set;
    `top` is an optional explicit note count. Only the model's context window
    (`context_tokens`) can drop evidence, and then one warning names what was
    dropped. Returns (notes, set_mode)."""
    by_doc: dict[str, object] = {}
    for r in recalled:
        by_doc.setdefault(str(r.document_id), r)
    set_mode = False
    order: list[tuple[str, str]] = []
    if scope:
        members = resolve_scope(conn, bank_id, scope)
        if members:
            set_mode = True
            order = sorted(members, key=lambda m: (-(by_doc[m[0]].score if m[0] in by_doc else 0.0), m[1]))
    if not order:
        best = max((r.score for r in by_doc.values()), default=0.0)
        order = [(d, r.source) for d, r in by_doc.items()
                 if best <= 0 or r.score >= min_rel_score * best]
        if top is not None:
            order = order[:top]
    picked = []
    for doc_id, source in order:
        r = by_doc.get(doc_id)
        chunks = pick_chunks(_chunk_rows(conn, bank_id, doc_id, qvec),
                             r.original_chunk if r else "", min_rel_score)
        if chunks:
            picked.append((doc_id, source or doc_id, chunks))
    budget = max(0, context_tokens - PROMPT_RESERVE_TOKENS) * CHARS_PER_TOKEN
    used, kept, dropped = 0, {}, []
    for doc_id, name, chunks in picked:
        for sc, pos, text in chunks:
            cost = len(text) + len(name) + 32
            if used + cost > budget:
                dropped.append((name, text))
                continue
            used += cost
            kept.setdefault(doc_id, []).append((pos, text))
    if dropped:
        logger.warning(
            "grounded evidence exceeds the model context window (%d tokens); dropped %d "
            "chunk(s), lowest-ranked first: %s", context_tokens, len(dropped),
            "; ".join(f"[{n}] {t}" for n, t in dropped))
    notes = [NoteEvidence(name=name, document_id=doc_id,
                          chunks=[t for _, t in sorted(kept[doc_id])])
             for doc_id, name, _ in picked if doc_id in kept]
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
                        set_mode: bool = False) -> GroundedResult:
    prompt = render_prompt("synthesize-grounded", {
        "query": query, "context": render_context(notes), "set_mode": set_mode})
    t0 = time.monotonic()
    out = llm(messages=[{"role": "user", "content": prompt}])
    call = {"purpose": "synthesize_grounded", "model": None, "tokens_in": None,
            "tokens_out": None, "cost_usd": None}
    if not isinstance(out, str):   # an LLMResult: text plus accounting
        call.update(model=getattr(out, "model", None),
                    tokens_in=getattr(out, "tokens_in", None),
                    tokens_out=getattr(out, "tokens_out", None),
                    cost_usd=getattr(out, "cost_usd", None))
        out = out.text
    call["duration_ms"] = int((time.monotonic() - t0) * 1000)
    return GroundedResult(synthesis=out, prompt=prompt, notes=notes,
                          citations=extract_citations(out, notes), set_mode=set_mode,
                          call=call)
