"""Channel interface (hybrid retrieval design 8.2)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Protocol


@dataclass(frozen=True)
class Candidate:
    document_id: str            # the parent note
    item_id: str | None         # the child chunk or question item that matched
    source: str                 # documents.source (the file name)
    channel: str                # "dense_chunk" | "question" | "bm25" | ...
    rank: int                   # 1-based within the channel (document level)
    score: float                # the channel's own score
    evidence: str | None        # the matched item's original text
    detail: dict = field(default_factory=dict)  # channel specifics


@dataclass(frozen=True)
class Filters:
    people: list[str] = field(default_factory=list)
    date_from: str | None = None
    date_to: str | None = None
    tags: list[str] = field(default_factory=list)
    hard: bool = False


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class QueryPlan:
    text: str
    formulations: list[str] = field(default_factory=list)
    filters: Filters = field(default_factory=Filters)
    now: datetime = field(default_factory=_utcnow)

    def to_json(self) -> dict:
        return {
            "text": self.text,
            "formulations": list(self.formulations),
            "filters": {
                "people": list(self.filters.people),
                "date_from": self.filters.date_from,
                "date_to": self.filters.date_to,
                "tags": list(self.filters.tags),
                "hard": self.filters.hard,
            },
            "now": self.now.isoformat(),
        }


@dataclass
class RecallState:
    """What a channel may use: a connection, the bank, the pool so far
    (so an `expand` channel sees its seeds), and a memoised query embedder."""

    conn: Any
    bank_id: str
    embed: Callable[[list[str]], list[list[float]]] | None = None
    metadata_filter: dict | None = None
    pool: list[Candidate] = field(default_factory=list)
    _embeddings: dict[str, list[float]] = field(default_factory=dict)

    def embed_query(self, text: str) -> list[float]:
        if text not in self._embeddings:
            if self.embed is None:
                raise RuntimeError("this channel needs an `embed` callable")
            self._embeddings[text] = list(self.embed([text])[0])
        return self._embeddings[text]


class Channel(Protocol):
    name: str
    kind: Literal["recall", "filter", "expand"]
    # recall: finds; filter: constrains / promotes; expand: grows the pool from seeds

    def retrieve(self, plan: QueryPlan, state: RecallState, limit: int) -> list[Candidate]: ...
