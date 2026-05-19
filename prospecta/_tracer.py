"""Tracer primitives (M6).

The tracer is the observability fan-out for prospecta. Six named events
fire from canonical call sites:

  retain | recall | formulate_queries | index_single_file | sweep_pass | llm_call

Callers may supply any callable matching the Tracer Protocol (see _types.py).
The default tracer is `PostgresSink` when a database_url is configured (it
dispatches each event to the corresponding append_*_event helper); otherwise
`NoOpTracer`.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class NoOpTracer:
    """Drops every event. Used when no database_url is configured."""

    def __call__(self, event: str, payload: dict[str, Any]) -> None:  # noqa: D401
        return None


class RecordingTracer:
    """Captures events in-memory. For testing.

    Exposes:
      .events: list[tuple[str, dict]] — all events in order received
      .by_name(event) -> list[dict]   — events filtered by name
      .clear()                         — reset
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, event: str, payload: dict[str, Any]) -> None:
        self.events.append((event, dict(payload)))

    def by_name(self, event: str) -> list[dict[str, Any]]:
        return [p for (e, p) in self.events if e == event]

    def clear(self) -> None:
        self.events.clear()


class CompositeTracer:
    """Dispatches each event to multiple tracers in order. Useful for combining
    PostgresSink (canonical persistence) + RecordingTracer (test assertions).
    """

    def __init__(self, *tracers) -> None:
        self._tracers = tuple(tracers)

    def __call__(self, event: str, payload: dict[str, Any]) -> None:
        for t in self._tracers:
            try:
                t(event, payload)
            except Exception:  # pragma: no cover — one bad tracer doesn't poison the rest
                logger.exception("CompositeTracer: sub-tracer raised; continuing")


__all__ = ["NoOpTracer", "RecordingTracer", "CompositeTracer"]
