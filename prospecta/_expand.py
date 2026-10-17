"""Mode 1 of docs/postgres-hybrid.md, the pure parts: automatic recall by query expansion.

A prompt (a chat message, or a task brief) is expanded by the query-writer model into N
structured searches (`SearchSpec`: a `query` for the semantic and stemmed retrievers and the
reranker, `terms` copied verbatim from the prompt for the exact retriever, and a period), the
searches run through Stage 1 and Stage 2 (prospecta/_expand_run.py), and their records merge into
one cited block. This module holds what needs no database: the writer's prompt and its defensive
parsing, the filter that keeps the writer's terms to names and identifiers, the verbatim-terms guard,
the stage 0 gate's default (Jev `needs_memory`) and the normalisation of an injected decision
callable, the exact retriever's query, the merge of the searches' records (near-copies collapse by
wording and by meaning) and the settings' resolution."""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import re
from datetime import date, timedelta
from typing import Any, Callable, Mapping, Sequence

from prospecta._common_words import COMMON_WORDS
from prospecta._template import render_prompt
from prospecta._types import GateDecision, RecalledMemory, SearchSpec, TaskBrief
from prospecta.stages import (
    NEAR_COPY_COSINE, NEAR_COPY_JACCARD, deduplicate_by_meaning, deduplicate_near_copies,
)

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
VECTORS_TIMEOUT_S = 2.0            # reading the survivors' chunk vectors for the meaning collapse (one primary-key read)

logger = logging.getLogger("prospecta.expand")
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
    seen: set[str] = set()
    for t in raw:
        t = _clean_text(t)
        if t and len(t) <= TERM_MAX_CHARS and _norm(t) not in seen:
            seen.add(_norm(t))
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


@dataclasses.dataclass
class ParsedReply:
    """What parse_writer_reply made of the writer's reply: its searches (terms already filtered),
    `why` (None, or why the reply gave no search at all), the terms the filter dropped as common
    words, and how many searches it dropped for holding nothing else (no query, no term left).
    `searches` empty with `why` None means the writer wrote searches and every one was dropped."""

    searches: list[SearchSpec]
    why: str | None
    dropped_terms: list[str] = dataclasses.field(default_factory=list)
    dropped_searches: int = 0


def parse_writer_reply(raw: str, today: date, prompt: "str | TermContext | None" = None) -> ParsedReply:
    """The writer's reply as searches. Defensive: prose and code fences around the JSON are
    tolerated, a bare array stands for {"searches": [...]}, an entry that is a plain string is a
    query, an entry without a query takes its terms as the query, an entry with neither is skipped,
    dates the writer got wrong are dropped (clean_window), duplicates are dropped. The reply is
    never trusted for its size: the caller applies the maximum.

    With `prompt` (the text the writer expanded) the terms are FILTERED as well (`keep_term`): a term
    that is a plain English word, and was not quoted in the prompt, is dropped; a search left with
    neither query nor terms is dropped."""
    entries = _entries_of(raw) if isinstance(raw, str) else None
    if entries is None:
        has_json = isinstance(raw, str) and re.search(r"[{\[]", raw) is not None
        return ParsedReply([], "the reply's JSON has no 'searches' list" if has_json
                           else "the reply is not usable JSON")
    ctx = None if prompt is None else (prompt if isinstance(prompt, TermContext) else term_context(prompt))
    specs: list[SearchSpec] = []
    seen: set[tuple] = set()
    dropped: list[str] = []
    dropped_searches = 0
    for e in entries:
        if isinstance(e, str):
            e = {"query": e}
        if not isinstance(e, dict):
            continue
        asked = _terms_of(e.get("terms"))
        terms = asked if ctx is None else tuple(t for t in asked if keep_term(t, ctx))
        dropped.extend(t for t in asked if t not in terms)
        query = _clean_text(e.get("query")) or " ".join(t.strip('"') for t in terms)
        if not query:
            dropped_searches += 1 if asked and not terms else 0
            continue
        df, dt = clean_window(e.get("date_from"), e.get("date_to"), today)
        key = (query.lower(), tuple(_norm(t) for t in terms), df, dt)
        if key in seen:
            continue
        seen.add(key)
        specs.append(SearchSpec(query=query, terms=terms, date_from=df, date_to=dt))
    why = None if specs or dropped_searches else "the reply holds no usable search"
    return ParsedReply(specs, why, dropped, dropped_searches)


def parse_searches(raw: str, today: date) -> tuple[list[SearchSpec], str | None]:
    """(the searches, None) from the writer's reply, or ([], why not): parse_writer_reply without
    the term filter."""
    parsed = parse_writer_reply(raw, today)
    return parsed.searches, parsed.why


# ------------------------------------------------------------------ plain words

_APOSTROPHE_TAIL = {"s", "t", "ll", "re", "ve", "d", "m"}
_SUFFIXES = (("ies", "y"), ("es", ""), ("s", ""), ("ed", ""), ("ed", "e"), ("d", ""), ("ing", ""),
             ("ing", "e"), ("ly", ""), ("er", ""), ("ers", ""), ("ness", ""), ("ment", ""), ("ments", ""))


def is_plain_word(token: str) -> bool:
    """True for a plain English word (any case): a word of the embedded list of common words
    (prospecta/_common_words.py), or its plural, past, -ing, -ly or agent form, or a contraction of
    one. Names, identifiers, versions, paths, words with digits or underscores, and words the list
    does not hold (rarer words, product names, jargon) are not plain."""
    w = token.lower().replace("\u2019", "'")
    if "'" in w:
        head, *tail = w.split("'")
        if not tail or any(t not in _APOSTROPHE_TAIL for t in tail):
            return False
        w = head[:-1] if head.endswith("n") and tail == ["t"] and head[:-1] in COMMON_WORDS else head
    if not w or not w.isascii() or not w.isalpha():
        return False
    if w in COMMON_WORDS:
        return True
    return any(w.endswith(suf) and len(w) - len(suf) >= 3 and w[:-len(suf)] + rep in COMMON_WORDS
               for suf, rep in _SUFFIXES)


# ------------------------------------------------------------------ the verbatim-terms guard and the term filter

_QUOTED = re.compile(r'["\u201c]([^"\u201d\n]{1,%d})["\u201d]' % TERM_MAX_CHARS)
_TOKEN_STRIP = ".,;:!?()[]{}<>'\"\u201c\u201d\u2018\u2019`*"
_TERM_STRIP = "()[]{}<>'\"\u201c\u201d\u2018\u2019`*,;:!?"          # not the dots and slashes a path starts with
_ORDINAL = re.compile(r"\d+(?:st|nd|rd|th|am|pm|s|k|x|ms|h|d|w|m)", re.IGNORECASE)
_PATHLIKE = re.compile(r"[~.]?/?[\w.\-]+(?:[/\\][\w.\-]+)+/?")
_DOTTED = re.compile(r"[A-Za-z0-9_\-]+(?:\.[A-Za-z0-9_\-]+)+")
_CAMEL = re.compile(r"[A-Za-z]*[a-z][A-Z][A-Za-z0-9]*")
_FLAG = re.compile(r"--[A-Za-z][\w\-]*")
_HYPHENATED = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+")
# what may stand before a word on its line and still leave it at the start of a sentence: a markdown
# heading, bullet, quote or list number
_LINE_START = re.compile(r"\s*(?:#{1,6}|[-*+>\u2022]+|\d{1,3}[.)])?\s*(?:\*\*|__)?")
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


@dataclasses.dataclass(frozen=True)
class _Scan:
    quoted: tuple[str, ...]        # the "quoted phrases", as written
    identifiers: tuple[str, ...]
    names: tuple[str, ...]         # runs of capitalised words


def _scan(text: str) -> _Scan:
    """The quoted phrases, identifiers and capitalised names of `text`, each once, in order of
    appearance. A capitalised word is a name unless it stands at the start of a sentence or a line
    (alone), is a function word, a month or a weekday, or is written in capitals and is a plain
    English word (IMPORTANT, SILENT, DELIVERY: emphasis, not a name)."""
    seen: set[str] = set()

    def once(items: list[str]) -> tuple[str, ...]:
        out = []
        for it in items:
            key = _norm(it)
            if key and key not in seen:
                seen.add(key)
                out.append(it)
        return tuple(out)

    quoted = once([m.group(1).strip() for m in _QUOTED.finditer(text) if m.group(1).strip()])
    rest = _QUOTED.sub(" ", text)
    cleaned = []
    for m in re.finditer(r"\S+", rest):
        raw, start = m.group(), m.start()
        before = rest[:start].rstrip()
        line_prefix = rest[rest.rfind("\n", 0, start) + 1:start]
        sentence_start = (not before or before[-1] in ".!?:" or "\n" in rest[len(before):start]
                          or _LINE_START.fullmatch(line_prefix) is not None)
        cleaned.append((raw.strip(_TOKEN_STRIP), sentence_start, raw))
    identifiers = once([tok for tok, _, _ in cleaned if _is_identifier(tok)])
    names: list[str] = []
    run: list[str] = []
    run_starts_sentence = False

    def flush() -> None:
        if len(run) >= 2 or (run and not run_starts_sentence):
            names.append(" ".join(run))
        run.clear()

    for tok, sentence_start, raw in cleaned:
        name = (tok[:1].isupper() and tok.lower() not in _CAPS_SKIP and len(tok) >= 2
                and not _is_identifier(tok) and tok.replace("'", "").replace("-", "").isalpha()
                and not (tok.isupper() and is_plain_word(tok)))
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
    return _Scan(quoted, identifiers, once(names))


def verbatim_terms(text: str) -> list[str]:
    """The names, identifiers and "quoted phrases" of `text`, as written, in this order: every
    "quoted phrase" (kept in its double quotes), then identifiers (versions, error codes, file
    names, flags, snake_case and camelCase words, paths), then capitalised names (a run of
    capitalised words is one name: `Dan Maycock`; one at the start of a sentence counts only as
    part of such a run; months, weekdays and function words never count, nor does an all-caps
    plain word such as IMPORTANT or SILENT unless it is quoted). What the exact retriever must be
    able to find, whatever the writer did to it."""
    scan = _scan(text)
    return [f'"{q}"' for q in scan.quoted] + list(scan.identifiers) + list(scan.names)


@dataclasses.dataclass(frozen=True)
class TermContext:
    """What the term filter needs to know of the prompt: its "quoted phrases" and the words it
    writes with a capital letter inside a sentence (names)."""

    quoted: frozenset[str]
    name_words: frozenset[str]


def term_context(prompt_text: str) -> TermContext:
    scan = _scan(prompt_text or "")
    words = {w.lower() for run in scan.names for w in run.split()}
    return TermContext(frozenset(_norm(q) for q in scan.quoted),
                       frozenset(words | {re.sub(r"'s$", "", w) for w in words}))


def _token_ok(tok: str, ctx: TermContext) -> bool:
    tok = tok.strip(_TERM_STRIP).rstrip(".")
    if not tok:
        return False
    if any(c.isdigit() for c in tok):                        # versions, error codes, hashes, ports, years
        return not _ORDINAL.fullmatch(tok) and not (tok.isdigit() and len(tok) <= 2)
    if _is_identifier(tok) or _HYPHENATED.fullmatch(tok):    # snake_case, camelCase, flags, paths, files, a-b
        return True
    if "/" in tok or "\\" in tok:                            # a path the identifier test did not take (/a/b/c)
        parts = [p for p in re.split(r"[/\\]", tok) if p]
        return (tok.lower() not in _SLASH_WORDS and len(parts) >= 1
                and (tok[0] in "/~." or any(not is_plain_word(p) for p in parts)))
    if tok.lower() in ctx.name_words:                        # capitalised mid-sentence in the prompt
        return True
    return len(tok) >= 2 and not is_plain_word(tok)          # a word the list does not hold: a name, jargon


def keep_term(term: str, prompt: "str | TermContext") -> bool:
    """Whether a term the writer wrote stays: only proper names, identifiers, versions, paths, error
    codes and "quoted phrases" do. Kept: a phrase the prompt quotes; a token with a digit (not an
    ordinal or a time), an underscore, a dot, a slash, a hyphen inside a word, camelCase or a flag;
    a word the prompt writes with a capital letter inside a sentence; any word that is not a plain
    English word. Dropped: a plain English word (session, report, cron, IMPORTANT), alone or in a
    phrase of plain words, however it is written, unless the prompt quoted it. A phrase stays when
    one of its words qualifies (`the Maycock survey`)."""
    ctx = prompt if isinstance(prompt, TermContext) else term_context(prompt)
    inner = _SPACES.sub(" ", term.strip().strip('"\u201c\u201d').strip())
    if not inner:
        return False
    return _norm(inner) in ctx.quoted or any(_token_ok(tok, ctx) for tok in inner.split())


def _norm(term: str) -> str:
    return _SPACES.sub(" ", term.strip().strip('"').strip()).lower()


def guard_terms(searches: list[SearchSpec], prompt_text: str) -> tuple[list[SearchSpec], list[str]]:
    """The verbatim-terms guard: every "quoted phrase", identifier or capitalised name of the
    prompt that no search carries in its terms is put into the FIRST search's terms (as written
    in the prompt). Returns (the searches, the terms put back). A term counts as carried when
    its text is inside some search's terms, ignoring case, quotes and spacing. Common words never
    come back (verbatim_terms has none) and neither do the all-caps words of instructions."""
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

def resolve_near_copy_cosine(call: float | None = None, bank: float | None = None) -> float:
    """The cosine similarity at or above which two notes collapse into one by meaning: per-call, else
    the bank's recall_config.expansion.near_copy_cosine, else PROSPECTA_NEAR_COPY_COSINE, else the
    default (stages.NEAR_COPY_COSINE). A number in (0, 1]; a bad per-call or bank value is an error,
    a bad environment value is ignored (and logged) so that a typo there never stops a recall."""
    def check(v: Any, name: str) -> float:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 < v <= 1:
            raise ValueError(f"{name} must be a number in (0, 1]")
        return float(v)
    if call is not None:
        return check(call, "near_copy_cosine")
    if bank is not None:
        return check(bank, "recall_config.expansion.near_copy_cosine")
    raw = os.environ.get("PROSPECTA_NEAR_COPY_COSINE", "").strip()
    if raw:
        try:
            return check(float(raw), "PROSPECTA_NEAR_COPY_COSINE")
        except ValueError:
            logger.warning("PROSPECTA_NEAR_COPY_COSINE=%r is not a number in (0, 1]: using %s", raw,
                           NEAR_COPY_COSINE)
    return NEAR_COPY_COSINE


@dataclasses.dataclass
class _Merged:
    item: RecalledMemory
    score: float
    found_by: set[int]
    collapsed: list[tuple[str, str]] = dataclasses.field(default_factory=list)   # (document id, source), best first

    def absorb(self, other: "_Merged") -> None:
        """`other` is a near-copy of this record: it joins it, with the searches that found it and
        the records that had already collapsed into it."""
        self.found_by |= other.found_by
        self.collapsed.append((other.item.document_id, other.item.source))
        self.collapsed.extend(other.collapsed)


def merge_records(per_search: Mapping[int, Sequence[RecalledMemory]], *,
                  near_copy_jaccard: float = NEAR_COPY_JACCARD,
                  vectors: "Callable[[list[str]], Mapping[str, Sequence[float]]] | Mapping[str, Sequence[float]] | None" = None,
                  near_copy_cosine: float | None = None) -> list[RecalledMemory]:
    """Merge the searches' records (`per_search`: search number -> its records). One record once: a
    note found by several searches keeps its best score (and that search's chunk and snippet; on a
    tie the copy with an exact-match snippet), and `found_by` lists every search that found it.
    Near-copies collapse into the best-scored one, which also takes the searches that found the
    copies and lists the records that collapsed into it (`collapsed_ids`, `collapsed_sources`):
    first by wording (distinct notes whose text is over `near_copy_jaccard` alike), then, when
    `vectors` is given, by meaning (a record whose chunk embedding has a cosine similarity of
    `near_copy_cosine` or more, default NEAR_COPY_COSINE, to a record kept before it, which is
    always the better scored; see stages.deduplicate_by_meaning). `vectors` is chunk id
    (RecalledMemory.item_id) -> embedding, or a callable that returns it for the chunk ids it is
    given: those of the records the wording collapse kept, and only when there are two or more. A
    record whose vector is missing stays as it is. Ordered by score, best first; nothing is cut by
    count."""
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
        ordered[j].absorb(ordered[i])
    if vectors is not None and len(survivors) >= 2:
        ids = [ordered[i].item.item_id for i in survivors]
        known = vectors([x for x in ids if x]) if callable(vectors) else vectors
        by_pos = [known.get(x) if x else None for x in ids]
        kept, by_meaning = deduplicate_by_meaning(
            by_pos, list(range(len(survivors))),
            NEAR_COPY_COSINE if near_copy_cosine is None else near_copy_cosine)
        for pos, twin in by_meaning.items():
            ordered[survivors[twin]].absorb(ordered[survivors[pos]])
        survivors = [survivors[p] for p in kept]
    out = []
    for i in survivors:
        m = ordered[i]
        extra = {}
        if m.collapsed:
            extra = {"collapsed_ids": tuple(d for d, _ in m.collapsed),
                     "collapsed_sources": tuple(dict.fromkeys(src for _, src in m.collapsed))}
        out.append(dataclasses.replace(m.item, relevance_score=m.score,
                                       found_by=tuple(sorted(m.found_by)), **extra))
    return out
