"""Tests for prospecta._chunker."""

from __future__ import annotations

from pathlib import Path

from prospecta._chunker import Chunk, chunk_file, chunk_text


def _chunks(content: str, **kwargs) -> list[Chunk]:
    return list(chunk_text(content, Path("test.md"), **kwargs))


def test_chunk_short_content_returns_single_chunk():
    chunks = _chunks("hello world", chunk_size=1000, overlap=200)
    assert len(chunks) == 1
    assert chunks[0].content == "hello world"
    assert chunks[0].chunk_index == 0
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len("hello world")


def test_chunk_empty_content_returns_no_chunks():
    assert _chunks("") == []
    assert _chunks("   \n\t  ") == []


def test_chunk_long_content_produces_multiple_chunks():
    content = ("word " * 500).strip()  # ~2499 chars
    chunks = _chunks(content, chunk_size=500, overlap=100)
    assert len(chunks) >= 3
    # chunk_index is monotonically increasing from 0
    for i, c in enumerate(chunks):
        assert c.chunk_index == i


def test_chunk_long_content_chunks_overlap():
    # Build content with unique markers to verify overlap
    content = " ".join(f"w{i:04d}" for i in range(400))  # ~2400 chars
    chunks = _chunks(content, chunk_size=400, overlap=100)
    assert len(chunks) >= 2
    # consecutive chunks should overlap on character ranges
    for prev, curr in zip(chunks, chunks[1:]):
        assert curr.start_char < prev.end_char, "chunks must overlap"


def test_chunk_respects_word_boundaries():
    # Force a single internal split; verify the boundary lands on whitespace
    # rather than mid-word when possible.
    content = ("alpha " * 200).strip()  # spaces between words
    chunks = _chunks(content, chunk_size=300, overlap=50)
    assert len(chunks) >= 2
    # First chunk's content should not end mid-word (i.e., should end on
    # a complete 'alpha' token after strip()).
    assert chunks[0].content.endswith("alpha")


def test_chunk_id_and_metadata():
    chunks = _chunks("hi", chunk_size=10, overlap=2)
    c = chunks[0]
    assert "__chunk_0" in c.id
    meta = c.to_metadata()
    assert meta["source_path"] == "test.md"
    assert meta["chunk_index"] == 0


def test_chunk_file_reads_and_chunks(tmp_path: Path):
    p = tmp_path / "note.md"
    p.write_text("hello prospecta", encoding="utf-8")
    chunks = list(chunk_file(p, chunk_size=1000, overlap=200))
    assert len(chunks) == 1
    assert chunks[0].content == "hello prospecta"
    assert chunks[0].source_path == p


def test_chunk_file_missing_returns_empty(tmp_path: Path):
    chunks = list(chunk_file(tmp_path / "nope.md"))
    assert chunks == []
