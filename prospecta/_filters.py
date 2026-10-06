"""Document filter fields (documents.created_on / person / source_kind),
derived from parsed frontmatter and/or retain metadata (migration 0005)."""
from __future__ import annotations

import datetime as _dt
from typing import Any

_DATE_KEYS = ("created_on", "created", "date")
_KIND_KEYS = ("source_kind", "type")


def coerce_date(value: Any) -> _dt.date | None:
    """A date, datetime or 'YYYY-MM-DD[...]' string -> date; else None."""
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    if isinstance(value, str) and len(value.strip()) >= 10:
        try:
            return _dt.date.fromisoformat(value.strip()[:10])
        except ValueError:
            return None
    return None


def _text(value: Any) -> str | None:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None or isinstance(value, (dict, bool)):
        return None
    s = str(value).strip()
    return s or None


def filter_fields(*sources: dict | None) -> dict:
    """First source that carries a key wins. Returns
    {"created_on": date|None, "person": str|None, "source_kind": str|None}."""
    out: dict = {"created_on": None, "person": None, "source_kind": None}
    for src in sources:
        if not src:
            continue
        if out["created_on"] is None:
            for k in _DATE_KEYS:
                d = coerce_date(src.get(k))
                if d is not None:
                    out["created_on"] = d
                    break
        if out["person"] is None:
            out["person"] = _text(src.get("person"))
        if out["source_kind"] is None:
            for k in _KIND_KEYS:
                s = _text(src.get(k))
                if s:
                    out["source_kind"] = s
                    break
    return out
