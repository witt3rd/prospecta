"""prospecta.Memory — the public API.

T4 ships: __init__, create_bank, bank_stats.
Future tasks extend with retain, recall, recall_synth, formulate_queries,
index_directory, etc.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import psycopg

from prospecta._types import EmbedCallable, LLMCallable, Tracer
from prospecta.db.pool import ConnectionPool
from prospecta.db.queries import (
    HNSW_INDEX_TEMPLATE,
    INSERT_BANK,
    SELECT_BANK,
    SELECT_BANK_STATS,
    hnsw_index_name,
    validate_bank_id,
)

logger = logging.getLogger(__name__)


class BankConfigConflict(Exception):
    """Raised when create_bank is called with a different embedding_dim
    than an existing bank with the same bank_id."""


@dataclass(frozen=True)
class BankStats:
    """Bank stats returned by Memory.bank_stats()."""

    bank_id: str
    documents: int
    memory_items: int
    last_retain_at: Any  # datetime or None


class Memory:
    """The public Memory API for prospecta.

    T4 surface: __init__, create_bank, bank_stats, close.
    Future surface (T9+): retain, recall, search, recall_synth,
    formulate_queries, index_directory, ...
    """

    def __init__(
        self,
        *,
        database_url: str,
        llm: LLMCallable | None = None,
        embed: EmbedCallable | None = None,
        bank_id: str = "prospecta",
        tracer: Tracer | None = None,
    ) -> None:
        if not database_url:
            raise ValueError("database_url is required")
        self._database_url = database_url
        self._llm = llm
        self._embed = embed
        self._default_bank_id = bank_id
        self._tracer = tracer
        self._pool = ConnectionPool(database_url=database_url)

    @property
    def database_url(self) -> str:
        return self._database_url

    @property
    def default_bank_id(self) -> str:
        return self._default_bank_id

    def create_bank(
        self,
        bank_id: str,
        *,
        embedding_dim: int,
        embedding_model_id: str | None = None,
        mission: str | None = None,
        retain_mission: str | None = None,
    ) -> None:
        """Create a bank with the given embedding dimensionality.

        Idempotent for identical configs (same bank_id + same embedding_dim
        is a no-op). Raises BankConfigConflict if bank exists with different
        embedding_dim.

        Also creates the per-bank HNSW partial index for embedding columns.
        """
        validate_bank_id(bank_id)
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive int, got {embedding_dim!r}")

        with self._pool.cursor() as cur:
            cur.execute(SELECT_BANK, {"bank_id": bank_id})
            existing = cur.fetchone()

            if existing is not None:
                existing_dim = existing[1]
                if existing_dim != embedding_dim:
                    raise BankConfigConflict(
                        f"Bank {bank_id!r} exists with embedding_dim={existing_dim}, "
                        f"refusing to recreate with embedding_dim={embedding_dim}"
                    )
                logger.debug("Bank %s already exists with matching config", bank_id)
                return

            cur.execute(
                INSERT_BANK,
                {
                    "bank_id": bank_id,
                    "embedding_dim": embedding_dim,
                    "embedding_model_id": embedding_model_id,
                    "mission": mission,
                    "retain_mission": retain_mission,
                },
            )

        # CREATE INDEX CONCURRENTLY cannot run inside a transaction.
        self._create_hnsw_index(bank_id, embedding_dim)

    def _create_hnsw_index(self, bank_id: str, embedding_dim: int) -> None:
        """Create the per-bank HNSW partial index for embeddings.

        Uses CREATE INDEX CONCURRENTLY which CANNOT run in a transaction,
        so opens a dedicated autocommit connection.
        """
        index_name = hnsw_index_name(bank_id)
        # bank_id already validated by hnsw_index_name → validate_bank_id;
        # embedding_dim validated by create_bank.
        sql = HNSW_INDEX_TEMPLATE.format(
            index_name=index_name,
            bank_id_literal=bank_id,
            embedding_dim=int(embedding_dim),
        )
        with psycopg.connect(self._database_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
        logger.info("Created per-bank HNSW index %s", index_name)

    def bank_stats(self, bank_id: str | None = None) -> BankStats:
        """Return basic counts for a bank (defaults to default_bank_id)."""
        bank_id = bank_id or self._default_bank_id
        validate_bank_id(bank_id)
        with self._pool.cursor() as cur:
            cur.execute(SELECT_BANK_STATS, {"bank_id": bank_id})
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"No such bank: {bank_id!r}")
            return BankStats(
                bank_id=row[0],
                documents=row[1],
                memory_items=row[2],
                last_retain_at=row[3],
            )

    def close(self) -> None:
        """Close the connection pool."""
        self._pool.close()

    # ------------------------------------------------------------------
    # T9 — index/search/remove (delegate to _index)
    # ------------------------------------------------------------------

    def index_directory(self, path, **kwargs):
        """Walk a directory, parse + chunk + embed + index each file.

        See prospecta._index.index_directory for full signature.
        """
        from prospecta import _index
        return _index.index_directory(self, path, **kwargs)

    def search(self, text: str, **kwargs):
        """Search across the default bank.

        Modes: 'hybrid' (default), 'semantic', 'lexical'. RecalledMemory.scores
        is always populated with three numeric keys (A6 COALESCE invariant).
        """
        from prospecta import _index
        return _index.search(self, text, **kwargs)

    def remove_documents(self, sources: list[str]) -> int:
        """Explicit removal by caller-supplied source list (P5 substrate-opacity).

        Caller decides which sources are stale. The library does not introspect.
        """
        from prospecta import _index
        return _index.remove_documents(self, sources)

    def retain(self, content: str, **kwargs) -> str:
        """Write-side bilateral spine. Returns document_id (str UUID).

        See prospecta._retain.retain for full signature and semantics:
          - replace-on-source-match (schema.md §6)
          - caller-supplied index_text bypasses LLM (P4)
          - prompt_override bypasses default prompt (P4)
          - DocumentSourceConflictError on hash collision with different source
        """
        from prospecta import _retain
        return _retain.retain(self, content, **kwargs)
