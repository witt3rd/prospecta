"""Mode 1 of docs/postgres-hybrid.md, the pure parts: automatic recall by query expansion.

A prompt (a chat message, or a task brief) is expanded by the query-writer model into N
structured searches (`SearchSpec`: a `query` for the semantic and stemmed retrievers and the
reranker, `terms` copied verbatim from the prompt for the exact retriever, and a period), the
searches run through Stage 1 and Stage 2 (prospecta/_expand_run.py), and their records merge into
one cited block. This module holds what needs no database: the writer's prompt and its defensive
parsing, the verbatim-terms guard, the stage 0 gate's default (Jev `needs_memory`) and the
normalisation of an injected decision callable, the exact retriever's query, the merge of the
searches' records and the settings' resolution."""
from __future__ import annotations

import dataclasses
import json
import math
import os
import re
from datetime import date, timedelta
from typing import Any, Callable, Mapping, Sequence

from prospecta._template import render_prompt
from prospecta._types import GateDecision, RecalledMemory, SearchSpec, TaskBrief
from prospecta.stages import NEAR_COPY_JACCARD, deduplicate_near_copies

DEFAULT_QUERY_WRITER_MODEL = "openrouter/google/gemini-2.5-flash-lite"
DEFAULT_MAX_SEARCHES = 20          # the hard maximum: a bad model output cannot explode the fan-out
DEFAULT_MAX_CONCURRENCY = 8        # Stage 1 searches at once (and never more than the pool has)
DEFAULT_DEADLINE_S = 7.0           # everything (gate, writer, searches, reranks) inside this
NEEDS_MEMORY_THRESHOLD = 0.5       # the plugin's contract: a noul at or above this is "yes"
FALLBACK_MIN_REL = 0.5             # no stage 2 for a search: its Stage 1 candidates scoring >= this x its best stay
WRITER_SHARE = 0.5                 # the writer's budget, as a share of the deadline
GATE_SHARE = 0.3                   # the gate's budget, as a share of the deadline
SEARCH_RESERVE = 0.25              # the searches always keep this share of the deadline after the writer
MIN_DATE = date(1990, 1, 1)        # a period the writer names must lie between this and a year ahead
TERM_MAX_CHARS = 200

_SPACES = re.compile(r"\s+")


# ------------------------------------------------------------------ settings

def resolve_query_writer_model(call: str | None = None, bank: str | None = None) -> str:
    """The query-writer model: per-call, else the bank's recall_config.models.query_writer_model,
    else PROSPECTA_QUERY_WRITER_MODEL, else the default."""
    return (call or bank or os.environ.get("PROSPECTA_QUERY_WRITER_MODEL")
            or DEFAULT_QUERY_WRITER_MODEL)


def scope_first(items: list, project: str | None) -> list:
    """The notes of the task's own project first, then those without a project, then
    other projects' notes; each group keeps its relevance order."""
    if not project:
        return list(items)

    def group(item) -> int:
        p = (item.metadata or {}).get("project")
        return 0 if p == project else (1 if not p else 2)
    return sorted(items, key=group)


# ------------------------------------------------------------------ the writer's prompt

def brief_text(brief: TaskBrief) -> str:
    """The task brief as the writer reads it: title, project, ask, done criteria."""
    lines = [f"Title: {brief.title}"]
    if brief.project:
        lines.append(f"Project: {brief.project}")
    lines.append(f"Ask: {brief.ask}")
    if brief.done:
        lines.append(f"Done when: {brief.done}")
    return "\n".join(lines)


def count_words(text: str) -> int:
    return len(text.split())


def suggest_searches(kind: str, words: int, max_searches: int) -> int:
    """How many searches the prompt's size calls for (the writer is told this number, and the
    scaling rule): about 3 for a short chat message, more for a longer one, 8 and up for a task
    brief, never more than the hard maximum."""
    if kind == "task":
        target = max(8, min(20, words // 15))
    elif words <= 30:
        target = 3
    elif words <= 80:
        target = 5
    elif words <= 200:
        target = 8
    else:
        target = 12
    return max(1, min(target, max_searches))


def render_writer_prompt(text: str, *, kind: str, agent_last: str | None, max_searches: int,
                         today: date) -> str:
    """The writer's prompt (prompts/expand-searches.md): the rule for what a search is, its
    structured shape, the scaling rule and the maximum, then the prompt itself."""
    words = count_words(text)
    return render_prompt("expand-searches", {
        "kind": kind, "prompt": text, "agent_last": (agent_last or "").strip() or None,
        "max_searches": max_searches, "words": words, "today": today.isoformat(),
        "target": suggest_searches(kind, words, max_searches)})


# ------------------------------------------------------------------ parsing the writer's reply

def _clean_text(v: Any) -> str:
    return _SPACES.sub(" ", v).strip() if isinstance(v, str) else ""


def _parse_day(v: Any, *, end: bool) -> date | None:
    """YYYY-MM-DD, or YYYY-MM / YYYY (the first day, or with `end` the last), else None."""
    if not isinstance(v, str):
        return None
    m = re.fullmatch(r"\s*(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?(?:[T ].*)?\s*", v)
    if not m:
        return None
    y, mo, d = int(m.group(1)), m.group(2), m.group(3)
    try:
        if d:
            return date(y, int(mo), int(d))
        if mo:
            first = date(y, int(mo), 1)
            return (date(y + (first.month == 12), first.month % 12 + 1, 1) - timedelta(days=1)
                    if end else first)
        return date(y, 12, 31) if end else date(y, 1, 1)
    except ValueError:
        return None


def clean_window(date_from: Any, date_to: Any, today: date) -> tuple[str | None, str | None]:
    """A period the writer named, as ISO dates, or (None, None): each date real, between
    MIN_DATE and a year ahead, and from <= to (a window that fails any of that is no window:
    a wrong window would hide the notes, a missing one only widens the search)."""
    lo, hi = _parse_day(date_from, end=False), _parse_day(date_to, end=True)
    limit = today + timedelta(days=366)
    for d in (lo, hi):
        if d is not None and not MIN_DATE <= d <= limit:
            return None, None
    if lo is not None and hi is not None and lo > hi:
        return None, None
    return (lo.isoformat() if lo else None, hi.isoformat() if hi else None)


def _terms_of(v: Any) -> tuple[str, ...]:
    raw = [v] if isinstance(v, str) else (v if isinstance(v, list) else [])
    out: list[str] = []
    for t in raw:
        t = _clean_text(t)
        if t and len(t) <= TERM_MAX_CHARS and t.lower() not in {x.lower() for x in out}:
            out.append(t)
    return tuple(out)


def _entries_of(text: str) -> list | None:
    """The writer's list of searches in its reply: the first JSON value in `text` that is an
    object with a `searches` (or `queries`) list, or a list of objects / strings itself."""
    dec = json.JSONDecoder()
    pos = 0
    while (m := re.search(r"[{\[]", text[pos:])) is not None:
        i = pos + m.start()
        try:
            v, _ = dec.raw_decode(text, i)
        except ValueError:
            pos = i + 1
            continue
        if isinstance(v, dict):
            for key in ("searches", "queries"):
                if isinstance(v.get(key), list):
                    return v[key]
        elif isinstance(v, list) and any(isinstance(x, (dict, str)) for x in v):
            return v
        pos = i + 1
    return None


def parse_searches(raw: str, today: date) -> tuple[list[SearchSpec], str | None]:
    """(the searches, None) from the writer's reply, or ([], why not). Defensive: prose and code
    fences around the JSON are tolerated, a bare array stands for {"searches": [...]}, an entry
    that is a plain string is a query, an entry without a query takes its terms as the query,
    an entry with neither is skipped, dates the writer got wrong are dropped (clean_window),
    duplicates are dropped. The reply is never trusted for its size: the caller applies the maximum."""
    entries = _entries_of(raw) if isinstance(raw, str) else None
    if entries is None:
        has_json = isinstance(raw, str) and re.search(r"[{\[]", raw) is not None
        return [], ("the reply's JSON has no 'searches' list" if has_json else "the reply is not usable JSON")
    specs: list[SearchSpec] = []
    seen: set[tuple] = set()
    for e in entries:
        if isinstance(e, str):
            e = {"query": e}
        if not isinstance(e, dict):
            continue
        terms = _terms_of(e.get("terms"))
        query = _clean_text(e.get("query")) or " ".join(t.strip('"') for t in terms)
        if not query:
            continue
        df, dt = clean_window(e.get("date_from"), e.get("date_to"), today)
        key = (query.lower(), tuple(t.lower() for t in terms), df, dt)
        if key in seen:
            continue
        seen.add(key)
        specs.append(SearchSpec(query=query, terms=terms, date_from=df, date_to=dt))
    return specs, (None if specs else "the reply holds no usable search")


# ------------------------------------------------------------------ the verbatim-terms guard

_QUOTED = re.compile(r'["\u201c]([^"\u201d\n]{1,%d})["\u201d]' % TERM_MAX_CHARS)
_TOKEN_STRIP = ".,;:!?()[]{}<>'\"\u201c\u201d\u2018\u2019`*"
_ORDINAL = re.compile(r"\d+(?:st|nd|rd|th|am|pm|s|k|x|ms|h|d|w|m)", re.IGNORECASE)
_PATHLIKE = re.compile(r"[~.]?/?[\w.\-]+(?:[/\\][\w.\-]+)+/?")
_DOTTED = re.compile(r"[A-Za-z0-9_\-]+(?:\.[A-Za-z0-9_\-]+)+")
_CAMEL = re.compile(r"[A-Za-z]*[a-z][A-Z][A-Za-z0-9]*")
_FLAG = re.compile(r"--[A-Za-z][\w\-]*")
_SLASH_WORDS = {"and/or", "either/or", "yes/no", "he/she", "his/her", "him/her", "w/o", "n/a", "i/o"}
_CAPS_SKIP = {
    "i", "i'm", "i'll", "i've", "i'd", "the", "a", "an", "and", "but", "or", "so", "if", "when", "what",
    "where", "why", "how", "who", "which", "that", "this", "these", "those", "there", "then", "now",
    "do", "does", "did", "can", "could", "would", "should", "will", "please", "also", "is", "are", "was",
    "were", "it", "its", "we", "you", "he", "she", "they", "my", "our", "your", "yes", "no", "ok", "okay",
    "thanks", "thank", "hi", "hello", "hey", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "am", "pm",
}


def _is_identifier(tok: str) -> bool:
    if len(tok) < 2 or _ORDINAL.fullmatch(tok):
        return False
    has_alpha = any(c.isalpha() for c in tok)
    has_digit = any(c.isdigit() for c in tok)
    if has_alpha and has_digit:
        return True                                           # v1.14, 0x80070005, ABC-123, 3f9a1c7e2b
    if _FLAG.fullmatch(tok):
        return True
    if "_" in tok and has_alpha:
        return True                                           # PROSPECTA_DB_POOL_MAX
    if _CAMEL.fullmatch(tok) and has_alpha:
        return True                                           # HybridChunks, jevScore
    if tok.lower() in _SLASH_WORDS:
        return False
    if _PATHLIKE.fullmatch(tok) and has_alpha:
        return any(len(p) >= 2 for p in re.split(r"[/\\]", tok))   # src/main.rs, docs/exact-tsv.md
    if _DOTTED.fullmatch(tok) and has_alpha:
        parts = tok.split(".")
        return len(parts) >= 2 and max(len(p) for p in parts) >= 2 and not all(
            len(p) == 1 for p in parts)                       # config.yaml, main.rs (not e.g. / i.e.)
    return False


def verbatim_terms(text: str) -> list[str]:
    """The names, identifiers and "quoted phrases" of `text`, as written, in this order: every
    "quoted phrase" (kept in its double quotes), then identifiers (versions, error codes, file
    names, flags, snake_case and camelCase words, paths), then capitalised names (a run of
    capitalised words is one name: `Dan Maycock`; one at the start of a sentence counts only as
    part of such a run; months, weekdays and function words never count). What the exact
    retriever must be able to find, whatever the writer did to it."""
    out: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        key = _norm(term)
        if key and key not in seen:
            seen.add(key)
            out.append(term)

    for m in _QUOTED.finditer(text):
        phrase = m.group(1).strip()
        if phrase:
            add(f'"{phrase}"')
    rest = _QUOTED.sub(" ", text)
    toks = [(m.group(), m.start()) for m in re.finditer(r"\S+", rest)]
    cleaned = []
    for raw, start in toks:
        tok = raw.strip(_TOKEN_STRIP)
        before = rest[:start].rstrip()
        sentence_start = not before or before[-1] in ".!?:" or "\n" in rest[len(before):start]
        cleaned.append((tok, sentence_start, raw))
    for tok, _, _ in cleaned:
        if _is_identifier(tok):
            add(tok)
    run: list[str] = []
    run_starts_sentence = False

    def flush() -> None:
        if len(run) >= 2 or (run and not run_starts_sentence):
            add(" ".join(run))
        run.clear()

    for tok, sentence_start, raw in cleaned:
        name = (tok[:1].isupper() and tok.lower() not in _CAPS_SKIP and len(tok) >= 2
                and not _is_identifier(tok) and tok.replace("'", "").replace("-", "").isalpha())
        if name:
            if not run:
                run_starts_sentence = sentence_start
            elif sentence_start:
                flush()
                run_starts_sentence = True
            run.append(tok)
            if raw[-1:] in ".,;:!?)":
                flush()
        else:
            flush()
    flush()
    return out


def _norm(term: str) -> str:
    return _SPACES.sub(" ", term.strip().strip('"').strip()).lower()


def guard_terms(searches: list[SearchSpec], prompt_text: str) -> tuple[list[SearchSpec], list[str]]:
    """The verbatim-terms guard: every "quoted phrase", identifier or capitalised name of the
    prompt that no search carries in its terms is put into the FIRST search's terms (as written
    in the prompt). Returns (the searches, the terms put back). A term counts as carried when
    its text is inside some search's terms, ignoring case, quotes and spacing."""
    if not searches:
        return searches, []
    have = " | ".join(_norm(t) for s in searches for t in s.terms)
    back = [t for t in verbatim_terms(prompt_text) if _norm(t) not in have]
    if not back:
        return searches, []
    first = dataclasses.replace(searches[0], terms=searches[0].terms + tuple(back))
    return [first] + searches[1:], back


def exact_query(terms: Sequence[str]) -> str | None:
    """What the exact retriever reads for a search's terms (websearch syntax over the 'simple'
    configuration): each term as written, a multi-word one as an adjacent phrase, all OR-ed so a
    note holding ANY term is a candidate. None when there is no term (the exact retriever then
    reads the search's query, as it does for any Stage 1 question)."""
    parts: list[str] = []
    for t in terms:
        t = t.strip()
        inner = t[1:-1] if len(t) >= 2 and t[0] == '"' and t[-1] == '"' else t
        inner = _SPACES.sub(" ", inner.replace('"', " ")).strip().lstrip("-").strip()
        if not inner or inner.lower() == "or":
            continue
        parts.append(f'"{inner}"' if " " in inner else inner)
    return " OR ".join(parts) or None


def apply_maximum(searches: list[SearchSpec], max_searches: int) -> tuple[list[SearchSpec], int | None]:
    """(the searches cut to the hard maximum, how many the writer wrote when it was cut)."""
    if len(searches) <= max_searches:
        return searches, None
    return searches[:max_searches], len(searches)


# ------------------------------------------------------------------ the stage 0 gate

NEEDS_MEMORY_QUESTION = {
    "type": "noul",
    "instructions": ("To respond to `new_lines`, does the agent need something that is NOT in "
                     "`agent_last` or `new_lines`: an earlier decision or note, a document, an earlier "
                     "conversation, earlier work, or what is known of the person? An "
                     "acknowledgement, a go-ahead, a greeting, and a question about the live state of "
                     "something that has to be read from a live tool, need nothing from memory."),
    "criteria": {"true": "Something it needs is outside what it has.",
                 "false": "What it has is enough, or nothing is asked, or the answer is live state."},
}


def needs_memory_request(model: str, prompt: str, agent_last: str | None) -> dict:
    """The System One request of the default gate: one `noul` question, `needs_memory`."""
    return {"model": model,
            "state": {"agent": "an assistant of one person",
                      "turn": "a message from the person in a chat with the agent",
                      "agent_last": (agent_last or "").strip()
                      or "(none: this is the first message of the conversation)",
                      "new_lines": [f"The person (to you): {prompt}"],
                      "in_view": "the agent's last answer is in agent_last; no notes, open work or "
                                 "decisions are in view"},
            "questions": {"needs_memory": NEEDS_MEMORY_QUESTION}}


def jev_gate(transport: Callable[[dict, float], dict], model: str, timeout: float):
    """The default stage 0 decision: the decision model (Jev, System One) answers `needs_memory`
    as a noul in 0..1. Returns a callable (prompt, agent_last) -> GateDecision; it raises on a
    provider or reply error (the caller then goes on without the gate)."""
    def decide(prompt: str, agent_last: str | None) -> GateDecision:
        resp = transport(needs_memory_request(model, prompt, agent_last), timeout)
        answers = resp.get("answers") if isinstance(resp, dict) else None
        a = answers.get("needs_memory") if isinstance(answers, dict) else None
        v = a.get("noul") if isinstance(a, dict) else None
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1:
            raise ValueError("the decision model's answer is not a needs_memory noul in 0..1")
        u = resp.get("usage") or {}
        return GateDecision(needs_memory=float(v), model=resp.get("model") or model,
                            tokens_in=u.get("input_tokens"), tokens_out=u.get("output_tokens"),
                            cost_usd=u.get("cost"))
    return decide


def normalize_decision(out: Any) -> GateDecision:
    """An injected decision callable's return value as a GateDecision: a GateDecision, a number
    (needs_memory in 0..1), a bool (needs memory or not), or a mapping with `needs_memory` (and
    optionally skip / reason / model / cost_usd). Anything else is an error (the caller goes on
    without the gate)."""
    if isinstance(out, GateDecision):
        return out
    if isinstance(out, bool):
        return GateDecision(needs_memory=1.0 if out else 0.0)
    if isinstance(out, (int, float)) and math.isfinite(out) and 0 <= out <= 1:
        return GateDecision(needs_memory=float(out))
    if isinstance(out, Mapping) and isinstance(out.get("needs_memory"), (int, float)):
        known = {f.name for f in dataclasses.fields(GateDecision)}
        return GateDecision(**{k: v for k, v in out.items() if k in known})
    raise ValueError(f"a gate decision must be a GateDecision, a number in 0..1, a bool or a mapping "
                     f"with needs_memory, not {type(out).__name__}")


# ------------------------------------------------------------------ the merge

@dataclasses.dataclass
class _Merged:
    item: RecalledMemory
    score: float
    found_by: set[int]


def merge_records(per_search: Mapping[int, Sequence[RecalledMemory]], *,
                  near_copy_jaccard: float = NEAR_COPY_JACCARD) -> list[RecalledMemory]:
    """Merge the searches' records (`per_search`: search number -> its records). One record once:
    a note found by several searches keeps its best score (and that search's chunk and snippet; on a
    tie the copy with an exact-match snippet),
    and `found_by` lists every search that found it. Near-copies (distinct notes whose text is
    over `near_copy_jaccard` alike) collapse into the best-scored one, which also takes the
    searches that found the copies. Ordered by score, best first; nothing is cut by count."""
    best: dict[str, _Merged] = {}
    for n in sorted(per_search):
        for it in per_search[n]:
            score = float(it.relevance_score if it.relevance_score is not None else it.score)
            m = best.get(it.document_id)
            if m is None:
                best[it.document_id] = _Merged(it, score, {n})
                continue
            m.found_by.add(n)
            if score > m.score or (score == m.score and it.exact_snippet and not m.item.exact_snippet):
                m.item, m.score = it, score        # a tie keeps the copy that says why it matched
    ordered = sorted(best.values(), key=lambda m: (-m.score, -len(m.found_by), m.item.source,
                                                   m.item.document_id))
    survivors, collapsed = deduplicate_near_copies([m.item.original_chunk for m in ordered],
                                                   list(range(len(ordered))), near_copy_jaccard)
    for i, j in collapsed.items():
        ordered[j].found_by |= ordered[i].found_by
    return [dataclasses.replace(ordered[i].item, relevance_score=ordered[i].score,
                                found_by=tuple(sorted(ordered[i].found_by))) for i in survivors]
