"""Filter extraction (design 8.5): one small LLM call returns people[],
date_from, date_to and hard; a regex baseline (month names, ISO dates, years
and the bank's own vocabulary of people) is the fallback when there is no
LLM, the call fails, or its JSON is unusable. The LLM is the supplied
`llm` callable, asked for DEFAULT_EXTRACT_MODEL (Sonnet-5.5) via an optional
`model` keyword when the callable accepts one."""
from __future__ import annotations

import calendar
import inspect
import json
import logging
import re
from datetime import datetime

from prospecta.channels.base import Filters
from prospecta._filters import coerce_date

DEFAULT_EXTRACT_MODEL = "anthropic/claude-sonnet-5.5"

logger = logging.getLogger(__name__)

_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})

_PROMPT = """You extract retrieval filters from a question about a personal notes archive.

Today is {today}.
People known to the archive: {people}

Question: {question}

Return ONLY a JSON object with exactly these keys:
  "people":    list of names from the known people the question is about (exact spelling from the list; [] if none),
  "date_from": "YYYY-MM-DD" or null (start of the period the question is about),
  "date_to":   "YYYY-MM-DD" or null (end of that period, inclusive),
  "hard":      true only if the question explicitly confines itself to those people or that period, else false.
If the question names no person and no period, return empty people, null dates and hard false."""


def _month_range(year: int, month: int) -> tuple[str, str]:
    last = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{last:02d}"


def regex_filters(text: str, people_vocab: list[str], now: datetime) -> Filters:
    """The baseline. Never `hard` (it is too noisy to promote on)."""
    people = [p for p in people_vocab
              if re.search(rf"(?<!\w){re.escape(p)}(?!\w)", text, re.IGNORECASE)]
    date_from = date_to = None
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if m and coerce_date(m.group(0)):
        date_from = date_to = m.group(0)
    if date_from is None:
        names = "|".join(sorted(_MONTHS, key=len, reverse=True))
        m = re.search(rf"\b({names})\.?(?:\s+(\d{{4}}))?\b", text, re.IGNORECASE)
        # a bare "may" is a verb as often as a month: require a year for it
        if m and not (m.group(1).lower() == "may" and not m.group(2)):
            year = int(m.group(2)) if m.group(2) else now.year
            date_from, date_to = _month_range(year, _MONTHS[m.group(1).lower()])
    if date_from is None:
        m = re.search(r"\b(?:in|during|of|from)\s+((?:19|20)\d{2})\b", text)
        if m:
            date_from, date_to = f"{m.group(1)}-01-01", f"{m.group(1)}-12-31"
    return Filters(people=people, date_from=date_from, date_to=date_to, hard=False)


def _accepts_model(llm) -> bool:
    try:
        params = inspect.signature(llm).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "model" and p.kind != p.VAR_POSITIONAL
               or p.kind == p.VAR_KEYWORD for p in params)


def _llm_filters(text: str, people_vocab: list[str], now: datetime, llm,
                 model: str) -> Filters:
    prompt = _PROMPT.format(
        today=now.date().isoformat(), question=text,
        people=", ".join(people_vocab) if people_vocab else "(none)",
    )
    kwargs = {"model": model} if _accepts_model(llm) else {}
    raw = llm(messages=[{"role": "user", "content": prompt}], json_mode=True, **kwargs)
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("extraction JSON is not an object")
    canon = {p.lower(): p for p in people_vocab}
    people: list[str] = []
    for p in data.get("people") or []:
        c = canon.get(str(p).strip().lower())
        if c and c not in people:
            people.append(c)
    d_from = coerce_date(data.get("date_from"))
    d_to = coerce_date(data.get("date_to"))
    if d_from and d_to and d_from > d_to:
        d_from, d_to = d_to, d_from
    has = bool(people or d_from or d_to)
    return Filters(
        people=people,
        date_from=d_from.isoformat() if d_from else None,
        date_to=d_to.isoformat() if d_to else None,
        hard=bool(data.get("hard")) and has,
    )


def extract_filters(text: str, *, llm, people_vocab: list[str], now: datetime,
                    model: str = DEFAULT_EXTRACT_MODEL) -> Filters:
    if llm is not None:
        try:
            return _llm_filters(text, people_vocab, now, llm, model)
        except Exception as exc:  # fall back, never fail the recall
            logger.warning("filter extraction fell back to regex: %s: %s",
                           type(exc).__name__, exc)
    return regex_filters(text, people_vocab, now)
