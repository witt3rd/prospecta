"""backup -> restore into a fresh database -> recall, end to end."""
from __future__ import annotations

import time
from urllib.parse import urlparse, urlunparse

import psycopg

from prospecta.cli.__main__ import main
from prospecta.memory import Memory
from tests._stub_embedder import stub_embed, stub_llm


def test_backup_restore_roundtrip(populated_corpus, fresh_db, pg_url, tmp_path):
    tricky = "intro\nSET transaction_timeout = 0;\noutro marker-zebra"
    populated_corpus.retain(tricky, index_text=["zebra marker"], source="tricky")
    dump = tmp_path / "bank.sql"
    assert main(["--database-url", fresh_db, "backup", str(dump)]) == 0
    assert dump.stat().st_size > 0

    name = f"restored_{int(time.time() * 1_000_000)}"
    with psycopg.connect(pg_url, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{name}"')
    new_url = urlunparse(urlparse(pg_url)._replace(path=f"/{name}"))
    try:
        assert main(["--database-url", new_url, "restore", str(dump)]) == 0
        mem = Memory(database_url=new_url, bank_id="test", llm=stub_llm, embed=stub_embed)
        try:
            hits = mem.recall(["When is Kelly's birthday?"])
            assert any("Kelly" in h.content for h in hits)
            hits = mem.recall(["zebra marker"])
            assert any(h.content == tricky for h in hits)
        finally:
            mem.close()
    finally:
        with psycopg.connect(pg_url, autocommit=True) as conn:
            conn.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
