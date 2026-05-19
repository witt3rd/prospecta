"""Background sweeper — filesystem drift safety-net (P14).

The sweeper is the implicit-file-change write path. retain() is the
explicit-content write path. Both ultimately call _index.index_single_file
(P7: single write path).

P14 invariants:
  - Per-file exception isolation: one bad file does NOT abort the pass.
  - Sweeper thread crash does NOT poison Memory hot path.
  - Conservative default cadence (24h).
  - Sweeper is NOT the hot path.

Substrate-opacity (T5 addendum): sweeper is the legitimate home for
filesystem awareness in the library. Downstream code still treats source
strings as opaque.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from prospecta._ignore import should_ignore

if TYPE_CHECKING:
    from prospecta.memory import Memory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config + Result
# ---------------------------------------------------------------------------

@dataclass
class SweeperConfig:
    """Sweeper config. Defaults conservative per P14."""

    corpus_paths: list[Path]
    sweep_interval_seconds: float = 86400.0  # 24h
    file_extensions: tuple[str, ...] = (".md", ".markdown")
    ignore_patterns: list[str] | None = None
    max_error_paths: int = 50  # bound the error_paths list

    def __post_init__(self) -> None:
        self.corpus_paths = [Path(p) for p in self.corpus_paths]


@dataclass(frozen=True)
class SweepResult:
    """Outcome of a single sweep pass over ONE corpus_path."""

    corpus_path: Path
    started_at: datetime
    finished_at: datetime
    duration_ms: int
    files_scanned: int
    files_indexed: int  # new + replaced
    files_skipped: int  # unchanged or non-matching
    errors: int
    error_paths: list[tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Single-pass sweep (synchronous)
# ---------------------------------------------------------------------------

def _iter_corpus_files(
    corpus: Path, config: SweeperConfig
) -> "list[Path]":
    """Walk a corpus root; filter by extension + ignore patterns."""
    if not corpus.exists():
        return []
    exts = tuple(e.lower() for e in config.file_extensions)
    out: list[Path] = []
    if corpus.is_file():
        if corpus.suffix.lower() in exts:
            out.append(corpus)
        return out
    for p in sorted(corpus.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        try:
            if should_ignore(p, corpus):
                continue
        except Exception:  # pragma: no cover — defensive
            pass
        out.append(p)
    return out


def run_one_pass(
    memory: "Memory",
    config: SweeperConfig,
    *,
    stop_event: threading.Event | None = None,
) -> list[SweepResult]:
    """Run one sweep pass across all corpus_paths. Returns one SweepResult
    per corpus_path.

    P7: per-file work delegates to memory.index_single_file(...) — the
    SAME function retain ultimately uses (write-path single source of truth).

    P14: per-file try/except isolates failures; one bad file does not
    abort the pass. Errors are counted, logged, and (up to
    config.max_error_paths) recorded for observability.
    """
    bank_id = memory._default_bank_id
    results: list[SweepResult] = []

    for corpus in config.corpus_paths:
        corpus = Path(corpus)
        started_at = datetime.now(timezone.utc)
        t0 = time.monotonic()

        files = _iter_corpus_files(corpus, config)

        files_scanned = 0
        files_indexed = 0
        files_skipped = 0
        errors = 0
        error_paths: list[tuple[str, str]] = []

        for fp in files:
            if stop_event is not None and stop_event.is_set():
                break
            files_scanned += 1
            file_t0 = time.monotonic()
            try:
                # P7: single write path. Sweeper does NOT reach into _index
                # internals; it calls the same Memory.index_single_file that
                # any caller would use.
                result = memory.index_single_file(
                    fp,
                    source_prefix=None,
                    root=corpus,
                )
            except Exception as e:  # P14: per-file isolation
                errors += 1
                if len(error_paths) < config.max_error_paths:
                    error_paths.append((str(fp), repr(e)))
                logger.warning(
                    "sweeper: index_single_file failed for %s: %s", fp, e
                )
                # T16: fire tracer event for the per-file failure too.
                try:
                    memory._tracer("index_single_file", {
                        "bank_id": bank_id,
                        "source": str(fp),
                        "status": "error",
                        "duration_ms": int((time.monotonic() - file_t0) * 1000),
                        "content_hash": None,
                        "error": repr(e),
                    })
                except Exception:  # pragma: no cover
                    logger.exception("tracer raised on index_single_file; ignoring")
                continue

            if result.status == "error":
                errors += 1
                if len(error_paths) < config.max_error_paths:
                    error_paths.append(
                        (str(fp), result.error or "unknown")
                    )
            elif result.status in ("new", "replaced"):
                files_indexed += 1
            else:  # unchanged | skipped
                files_skipped += 1

            # T16: tracer event for the per-file outcome (PostgresSink no-ops
            # for v0.1 since no index_events table; RecordingTracer captures).
            try:
                memory._tracer("index_single_file", {
                    "bank_id": bank_id,
                    "source": str(fp),
                    "status": result.status,
                    "duration_ms": int((time.monotonic() - file_t0) * 1000),
                    "content_hash": getattr(result, "content_hash", None),
                    "error": getattr(result, "error", None),
                })
            except Exception:  # pragma: no cover
                logger.exception("tracer raised on index_single_file; ignoring")

        finished_at = datetime.now(timezone.utc)
        duration_ms = int((time.monotonic() - t0) * 1000)

        # T16: tracer dispatch (was direct append_sweep_pass +
        # upsert_sweeper_state co-write; default PostgresSink performs both
        # writes atomically per schema.md §8 co-write contract).
        try:
            memory._tracer("sweep_pass", {
                "bank_id": bank_id,
                "corpus_path": str(corpus),
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_ms": duration_ms,
                "files_scanned": files_scanned,
                "files_indexed": files_indexed,
                "files_skipped": files_skipped,
                "files_pruned": 0,
                "errors": errors,
                "error_paths": error_paths,
            })
        except Exception:  # pragma: no cover — observability shouldn't poison
            logger.exception("sweeper: tracer raised on sweep_pass; ignoring")

        results.append(SweepResult(
            corpus_path=corpus,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            files_scanned=files_scanned,
            files_indexed=files_indexed,
            files_skipped=files_skipped,
            errors=errors,
            error_paths=error_paths,
        ))

    return results


# ---------------------------------------------------------------------------
# Background thread
# ---------------------------------------------------------------------------

class SweeperThread(threading.Thread):
    """Daemon thread that runs run_one_pass on a cadence until stopped.

    P14: thread is a safety-net, not a hot path. Crashes are caught at the
    top of the loop body so a single bad pass does not kill the thread
    entirely (logs and proceeds to next interval).
    """

    def __init__(self, memory: "Memory", config: SweeperConfig) -> None:
        super().__init__(daemon=True, name="prospecta-sweeper")
        self._memory = memory
        self._config = config
        self._stop_event = threading.Event()

    def run(self) -> None:  # noqa: D401
        while not self._stop_event.is_set():
            try:
                run_one_pass(
                    self._memory, self._config, stop_event=self._stop_event,
                )
            except Exception:  # pragma: no cover — never let the thread die
                logger.exception("sweeper: run_one_pass raised; will retry next interval")
            # Sleep with early-exit on stop signal so shutdown is prompt.
            interval = max(0.0, float(self._config.sweep_interval_seconds))
            # Use wait() so stop_event short-circuits the sleep.
            if self._stop_event.wait(timeout=interval):
                break

    def stop(self, timeout: float = 5.0) -> bool:
        """Signal stop; join with timeout. Returns True iff thread exited."""
        self._stop_event.set()
        self.join(timeout=timeout)
        return not self.is_alive()
