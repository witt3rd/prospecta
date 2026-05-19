"""PostgresSink — canonical persistence tracer.

Dispatches each of the six prospecta events to the corresponding
append_*_event helper in prospecta.db.queries.

Architecture (per T16 spec, stance (a)):
  - PostgresSink is the default tracer when Memory is constructed with a
    database_url and no explicit tracer is supplied. The direct
    append_*_event calls that used to live in _retain.py / memory.py /
    _sweeper.py have migrated INTO PostgresSink — this is now the canonical
    write path for events. Callers who want to intercept events (for tests,
    extra observability, etc.) supply their own tracer (NoOpTracer,
    RecordingTracer, or CompositeTracer(PostgresSink, ...)).

  - PostgresSink has zero provider imports (P3): just psycopg via the
    injected pool and the queries module.

  - Events:
      'retain'             → append_retain_event
      'recall'             → append_recall_event
      'formulate_queries'  → append_formulate_event
      'sweep_pass'         → append_sweep_pass + upsert_sweeper_state
                              (schema.md §8 co-write contract, one txn)
      'index_single_file'  → no-op for v0.1 (no index_events table)
      'llm_call'           → append_llm_call
"""
from __future__ import annotations

import logging
from typing import Any

from prospecta.db.queries import (
    append_formulate_event,
    append_llm_call,
    append_recall_event,
    append_retain_event,
    append_sweep_pass,
    upsert_sweeper_state,
)

logger = logging.getLogger(__name__)


class PostgresSink:
    """Routes prospecta events into the canonical event tables.

    Constructed by Memory.__init__ when a database_url is configured. May
    also be constructed standalone with a database_url, in which case the
    sink owns its own connection (close() is no-op since psycopg connects
    per-call).
    """

    def __init__(
        self,
        database_url: str | None = None,
        *,
        pool=None,
        bank_id: str | None = None,
    ) -> None:
        if pool is None and database_url is None:
            raise ValueError("PostgresSink requires either a pool or database_url")
        self._pool = pool
        self._database_url = database_url
        self._bank_id = bank_id

    # ------------------------------------------------------------------
    # connection acquisition
    # ------------------------------------------------------------------

    def _connection(self):
        """Yield a psycopg connection (context manager)."""
        if self._pool is not None:
            return self._pool.connection()
        # Standalone fallback: open a connection directly.
        import psycopg
        return psycopg.connect(self._database_url)

    # ------------------------------------------------------------------
    # event dispatch
    # ------------------------------------------------------------------

    def __call__(self, event: str, payload: dict[str, Any]) -> None:
        try:
            handler = _HANDLERS.get(event)
            if handler is None:
                logger.debug("PostgresSink: unknown event %r (dropping)", event)
                return
            handler(self, payload)
        except Exception:  # pragma: no cover — observability never poisons hot path
            logger.exception("PostgresSink: %s handler raised; dropping", event)

    # ------------------------------------------------------------------
    # individual event handlers
    # ------------------------------------------------------------------

    def _handle_retain(self, p: dict[str, Any]) -> None:
        with self._connection() as conn:
            append_retain_event(
                conn,
                bank_id=p["bank_id"],
                document_id=p.get("document_id"),
                items_count=int(p["items_count"]),
                index_text_caller_supplied=bool(p["index_text_caller_supplied"]),
                duration_ms=int(p["duration_ms"]),
                raw_llm_response=p.get("raw_llm_response"),
                error=p.get("error"),
            )
            conn.commit()

    def _handle_recall(self, p: dict[str, Any]) -> None:
        with self._connection() as conn:
            append_recall_event(
                conn,
                bank_id=p["bank_id"],
                queries=list(p.get("queries", []) or []),
                mode=p["mode"],
                n_results=int(p["n_results"]),
                duration_ms=int(p["duration_ms"]),
                trace=p.get("trace"),
            )
            conn.commit()

    def _handle_formulate(self, p: dict[str, Any]) -> None:
        with self._connection() as conn:
            append_formulate_event(
                conn,
                bank_id=p["bank_id"],
                message=p["message"],
                n_queries=int(p["n_queries"]),
                parse_fallback=bool(p["parse_fallback"]),
                raw_response=p.get("raw_response"),
                error_kind=p.get("error_kind"),
                duration_ms=int(p["duration_ms"]),
                json_mode_used=bool(p.get("json_mode_used", True)),
            )
            conn.commit()

    def _handle_sweep_pass(self, p: dict[str, Any]) -> None:
        """Schema.md §8 co-write: sweep_passes + sweeper_state in one txn."""
        error_paths = list(p.get("error_paths", []) or [])
        error_summary = p.get("error")
        if error_summary is None and error_paths:
            first = error_paths[0]
            error_summary = f"{len(error_paths)} file(s) failed; first: {first}"
        pass_metadata = (
            {"error_paths": [{"path": ep[0], "error": ep[1]} for ep in error_paths]}
            if error_paths
            else None
        )
        with self._connection() as conn:
            append_sweep_pass(
                conn,
                bank_id=p["bank_id"],
                corpus_path=p["corpus_path"],
                started_at=p["started_at"],
                ended_at=p["finished_at"],
                files_seen=int(p["files_scanned"]),
                files_indexed=int(p["files_indexed"]),
                files_pruned=int(p.get("files_pruned", 0)),
                errors_count=int(p["errors"]),
                duration_ms=int(p["duration_ms"]),
                error=error_summary,
                pass_metadata=pass_metadata,
            )
            upsert_sweeper_state(
                conn,
                bank_id=p["bank_id"],
                corpus_path=p["corpus_path"],
                last_pass_started_at=p["started_at"],
                last_pass_ended_at=p["finished_at"],
                last_pass_files_seen=int(p["files_scanned"]),
                last_pass_files_indexed=int(p["files_indexed"]),
                last_pass_files_pruned=int(p.get("files_pruned", 0)),
                last_pass_errors=int(p["errors"]),
                last_pass_duration_ms=int(p["duration_ms"]),
                last_error=error_summary,
            )
            conn.commit()

    def _handle_index_single_file(self, p: dict[str, Any]) -> None:
        """v0.1: no index_events table; tracer fires but PostgresSink no-ops."""
        return None

    def _handle_llm_call(self, p: dict[str, Any]) -> None:
        with self._connection() as conn:
            append_llm_call(
                conn,
                bank_id=p.get("bank_id"),
                prompt_name=p["purpose"],
                messages_count=int(p.get("messages_count", 1)),
                json_mode=bool(p.get("json_mode", False)),
                duration_ms=int(p["duration_ms"]),
                error=p.get("error"),
            )
            conn.commit()

    def close(self) -> None:
        """No-op: PostgresSink does not own the pool (Memory does)."""
        return None


_HANDLERS = {
    "retain": PostgresSink._handle_retain,
    "recall": PostgresSink._handle_recall,
    "formulate_queries": PostgresSink._handle_formulate,
    "sweep_pass": PostgresSink._handle_sweep_pass,
    "index_single_file": PostgresSink._handle_index_single_file,
    "llm_call": PostgresSink._handle_llm_call,
}


__all__ = ["PostgresSink"]
