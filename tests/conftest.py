"""Shared pytest fixtures."""
from __future__ import annotations

import time
from urllib.parse import urlparse, urlunparse

import psycopg
import pytest
from testcontainers.postgres import PostgresContainer


def _normalize_url(url: str) -> str:
    """Strip driver hint so the URL is psycopg3-compatible."""
    if "+psycopg2" in url:
        url = url.replace("+psycopg2", "")
    if "+psycopg" in url:
        url = url.replace("+psycopg", "")
    return url


@pytest.fixture(scope="session")
def pg_container():
    """Session-scoped Postgres testcontainer with pgvector pre-enabled."""
    container = PostgresContainer("pgvector/pgvector:pg16")
    container.start()
    try:
        url = _normalize_url(container.get_connection_url())
        with psycopg.connect(url) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
        yield container
    finally:
        container.stop()


@pytest.fixture(scope="session")
def pg_url(pg_container) -> str:
    return _normalize_url(pg_container.get_connection_url())


@pytest.fixture
def fresh_db(pg_container):
    """Per-test fresh database with migrations applied. Returns its URL."""
    base_url = _normalize_url(pg_container.get_connection_url())
    db_name = f"test_{int(time.time() * 1_000_000)}"
    with psycopg.connect(base_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'CREATE DATABASE "{db_name}"')

    parsed = urlparse(base_url)
    new_url = urlunparse(parsed._replace(path=f"/{db_name}"))
    with psycopg.connect(new_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

    from prospecta.db.migrate import run_migrations
    run_migrations(new_url)

    yield new_url

    with psycopg.connect(base_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()",
                (db_name,),
            )
            cur.execute(f'DROP DATABASE "{db_name}"')


@pytest.fixture
def memory_with_bank(fresh_db):
    """Memory wired with the deterministic stub embedder + a created bank.

    32-dim bank named 'test'. Stub LLM attached but T9 paths don't call it.
    """
    from prospecta.memory import Memory
    from tests._stub_embedder import EMBED_DIM, stub_embed, stub_llm

    mem = Memory(
        database_url=fresh_db,
        bank_id="test",
        llm=stub_llm,
        embed=stub_embed,
    )
    mem.create_bank("test", embedding_dim=EMBED_DIM)
    yield mem
    mem.close()


@pytest.fixture
def mock_llm():
    """Deterministic LLM stub for retain/recall tests.

    Conforms to LLMCallable Protocol. Records every call. Default behavior:
    returns three question-shaped strings (newline-separated). Configure
    per-test via mock_llm.set_response(callable) or mock_llm.canned = str.
    """

    class MockLlm:
        def __init__(self) -> None:
            self.calls: list[dict] = []
            self._response_fn = None
            self.canned: str | None = None

        def __call__(self, messages, *, json_mode=False):
            self.calls.append({"messages": messages, "json_mode": json_mode})
            if self._response_fn is not None:
                return self._response_fn(messages, json_mode=json_mode)
            if self.canned is not None:
                return self.canned
            return (
                "What does this say?\n"
                "Why does it matter?\n"
                "When was this stored?"
            )

        def set_response(self, fn) -> None:
            self._response_fn = fn

    return MockLlm()


@pytest.fixture
def memory_with_bank_and_mock_llm(fresh_db, mock_llm):
    """Memory wired with stub embedder + mock_llm + a created bank."""
    from prospecta.memory import Memory
    from tests._stub_embedder import EMBED_DIM, stub_embed

    mem = Memory(
        database_url=fresh_db,
        bank_id="test",
        llm=mock_llm,
        embed=stub_embed,
    )
    mem.create_bank("test", embedding_dim=EMBED_DIM)
    yield mem
    mem.close()


@pytest.fixture
def populated_corpus(memory_with_bank_and_mock_llm):
    """Memory with a small set of retained docs.

    Uses retain() with index_text override so the mock_llm is NOT consumed
    during seeding (P4 caller-wins). Tests can still inspect mock_llm.calls
    cleanly after recall_synth fires.
    """
    mem = memory_with_bank_and_mock_llm
    seeds = [
        (
            "kelly-bio",
            "Kelly was born on March 4th, 1990 in San Diego. She loves the beach.",
            ["When is Kelly's birthday?", "Where was Kelly born?"],
        ),
        (
            "beach-note",
            "The beach at La Jolla has cliffs and tidepools. Kelly visits often.",
            ["What is at La Jolla beach?"],
        ),
        (
            "forge-note",
            "The forge bench stays warm between sessions. Sparks light the dark.",
            ["What does the forge bench feel like?"],
        ),
    ]
    for source, body, idx in seeds:
        mem.retain(body, index_text=idx, source=source)
    return mem


@pytest.fixture
def tmp_corpus(tmp_path):
    """A small markdown corpus with frontmatter `index_text` so indexing
    does NOT require an LLM. T14 sweeper fixture."""
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "doc1.md").write_text(
        "---\nindex_text: 'first doc question'\n---\nBody one."
    )
    (d / "doc2.md").write_text(
        "---\nindex_text: 'second doc question'\n---\nBody two."
    )
    return d


@pytest.fixture
def recording_tracer():
    from prospecta._tracer import RecordingTracer
    return RecordingTracer()


@pytest.fixture
def memory_with_recording_tracer(fresh_db, mock_llm, recording_tracer):
    """Memory wired with RecordingTracer (events captured in-memory,
    NOT persisted to event tables — PostgresSink is bypassed)."""
    from prospecta.memory import Memory
    from tests._stub_embedder import EMBED_DIM, stub_embed

    mem = Memory(
        database_url=fresh_db,
        bank_id="test",
        llm=mock_llm,
        embed=stub_embed,
        tracer=recording_tracer,
    )
    mem.create_bank("test", embedding_dim=EMBED_DIM)
    # expose mock_llm on the memory for convenience in tests
    mem._mock_llm = mock_llm  # type: ignore[attr-defined]
    yield mem
    mem.close()


@pytest.fixture
def populated_corpus_with_recording_tracer(memory_with_recording_tracer):
    """memory_with_recording_tracer + small set of retained docs.

    Uses index_text override so the mock_llm is not consumed during seeding.
    Tests can inspect the tracer's events after recall_synth fires.
    """
    mem = memory_with_recording_tracer
    seeds = [
        ("kelly-bio", "Kelly was born on March 4th, 1990 in San Diego.",
         ["When is Kelly's birthday?", "Where was Kelly born?"]),
        ("beach-note", "La Jolla has cliffs and tidepools.",
         ["What is at La Jolla beach?"]),
    ]
    for source, body, idx in seeds:
        mem.retain(body, index_text=idx, source=source)
    # Clear seeded events so tests assert on the queries-under-test only.
    mem._tracer.clear()
    return mem
