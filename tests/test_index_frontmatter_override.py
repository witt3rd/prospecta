"""T10: frontmatter `index_text:` override pickup.

When a markdown file declares `index_text:` in frontmatter, the library
must use it AS-IS for memory_items.content (no chunking, no LLM) and
preserve the body in documents.original_text. P4 enforced.
"""
from __future__ import annotations

from pathlib import Path

import psycopg


def _count_items_for_source(database_url: str, source: str) -> int:
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM memory_items m
                JOIN documents d ON d.id = m.document_id
                WHERE d.source = %s
                """,
                (source,),
            )
            return cur.fetchone()[0]


def _rows_for_source(database_url: str, source: str) -> list[dict]:
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT m.content, m.metadata, m.original_chunk, d.original_text
                FROM memory_items m
                JOIN documents d ON d.id = m.document_id
                WHERE d.source = %s
                ORDER BY m.content
                """,
                (source,),
            )
            cols = [c.name for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


def test_frontmatter_single_index_text(memory_with_bank, tmp_path):
    body = "Kelly was born on March 4th, 1990. She loves sunshine and dancing."
    f = tmp_path / "kelly.md"
    f.write_text(
        '---\nindex_text: "What is Kelly\'s birthday?"\n---\n'
        f"# Kelly\n\n{body}\n"
    )
    memory_with_bank.index_directory(tmp_path)

    src = str(f)
    assert _count_items_for_source(memory_with_bank.database_url, src) == 1
    rows = _rows_for_source(memory_with_bank.database_url, src)
    assert rows[0]["content"] == "What is Kelly's birthday?"
    assert rows[0]["metadata"].get("index_text_caller_supplied") is True
    # Body preserved in documents.original_text
    assert body in rows[0]["original_text"]


def test_frontmatter_list_index_text(memory_with_bank, tmp_path):
    questions = ["question1", "question2", "question3"]
    f = tmp_path / "multi.md"
    f.write_text(
        "---\nindex_text:\n  - question1\n  - question2\n  - question3\n---\n"
        "# Multi\n\nbody content here for the document.\n"
    )
    memory_with_bank.index_directory(tmp_path)

    src = str(f)
    assert _count_items_for_source(memory_with_bank.database_url, src) == 3
    rows = _rows_for_source(memory_with_bank.database_url, src)
    contents = sorted(r["content"] for r in rows)
    assert contents == sorted(questions)
    for r in rows:
        assert r["metadata"].get("index_text_caller_supplied") is True


def test_no_frontmatter_falls_back_to_chunker(memory_with_bank, tmp_path):
    f = tmp_path / "plain.md"
    f.write_text("# Plain\n\nFirst paragraph here.\n\nSecond paragraph there.\n")
    memory_with_bank.index_directory(tmp_path)

    src = str(f)
    rows = _rows_for_source(memory_with_bank.database_url, src)
    assert len(rows) >= 1
    for r in rows:
        # No override → flag absent or False
        assert not r["metadata"].get("index_text_caller_supplied", False)
        # T9 behavior preserved: content == chunk text
        assert r["content"] == r["original_chunk"]


def test_search_finds_caller_supplied_index_text(memory_with_bank, tmp_path):
    body = "Kelly was born on March 4th, 1990."
    f = tmp_path / "kelly.md"
    f.write_text(
        '---\nindex_text: "When is Kelly\'s birthday?"\n---\n'
        f"# Kelly\n\n{body}\n"
    )
    memory_with_bank.index_directory(tmp_path)

    results = memory_with_bank.search("Kelly's birthday", limit=5)
    assert len(results) > 0
    # The matched row's content is the index_text (the question), NOT the body.
    matches = [r for r in results if r.content == "When is Kelly's birthday?"]
    assert matches, f"expected index_text in results, got: {[r.content for r in results]}"
    m = matches[0]
    assert m.metadata.get("index_text_caller_supplied") is True
