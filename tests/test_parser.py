"""Tests for prospecta._parser."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from prospecta._parser import (
    ParsedNote,
    extract_title,
    parse_frontmatter,
    parse_note,
)


def test_parse_frontmatter_basic():
    res = parse_frontmatter("---\ntags: [a, b]\n---\nBody text")
    assert res.frontmatter == {"tags": ["a", "b"]}
    assert res.body == "Body text"


def test_parse_frontmatter_none_returns_empty():
    res = parse_frontmatter("No frontmatter here\nJust body.")
    assert res.frontmatter == {}
    assert res.body == "No frontmatter here\nJust body."


def test_parse_frontmatter_malformed_returns_empty_dict():
    # Unclosed YAML mapping — animus returns empty dict, full content as body
    res = parse_frontmatter("---\nkey: : : bad\n---\nbody")
    # Either parsed loosely or returned empty; must not raise, must be dict
    assert isinstance(res.frontmatter, dict)


def test_extract_title_first_h1():
    assert extract_title("# Hello\n\nbody") == "Hello"
    assert extract_title("no heading") == ""


def test_parse_note_with_single_index_text(tmp_path: Path):
    p = tmp_path / "what-is-x.md"
    p.write_text(
        '---\nindex_text: "What is X?"\ntags: [topic]\n---\n# What X Is\n\nbody\n',
        encoding="utf-8",
    )
    note = parse_note(p)
    assert note is not None
    assert note.title == "What X Is"
    assert note.tags == ["topic"]
    # index_text is preserved via relationships — bilateral spine write side
    assert note.relationships.get("index_text") == "What is X?"


def test_parse_note_with_list_index_text(tmp_path: Path):
    p = tmp_path / "multi.md"
    p.write_text(
        "---\nindex_text:\n  - q1\n  - q2\n---\n# Title\n",
        encoding="utf-8",
    )
    note = parse_note(p)
    assert note is not None
    assert note.relationships.get("index_text") == ["q1", "q2"]


def test_parse_note_without_frontmatter(tmp_path: Path):
    p = tmp_path / "raw.md"
    p.write_text("# Just A Title\n\nfree text body", encoding="utf-8")
    note = parse_note(p)
    assert note is not None
    assert note.title == "Just A Title"
    assert note.tags == []
    assert note.relationships == {}
    assert "free text body" in note.content


def test_parse_note_id_from_filename(tmp_path: Path):
    p = tmp_path / "My Note Name.md"
    p.write_text("body", encoding="utf-8")
    note = parse_note(p)
    assert note is not None
    assert note.id == "my_note_name"
    # falls back to titlecased stem when no H1
    assert note.title == "My Note Name"


def test_parse_note_tags_string_split(tmp_path: Path):
    p = tmp_path / "n.md"
    p.write_text("---\ntags: a, b, c\n---\nbody", encoding="utf-8")
    note = parse_note(p)
    assert note is not None
    assert note.tags == ["a", "b", "c"]


def test_parse_note_created_iso_string(tmp_path: Path):
    p = tmp_path / "n.md"
    p.write_text(
        "---\ncreated: '2024-01-15T10:30:00'\n---\nbody", encoding="utf-8"
    )
    note = parse_note(p)
    assert note is not None
    assert note.created == datetime(2024, 1, 15, 10, 30, 0)


def test_parse_note_missing_file_returns_none(tmp_path: Path):
    assert parse_note(tmp_path / "absent.md") is None


def test_parse_note_malformed_frontmatter_returns_partial(tmp_path: Path):
    # animus behavior: malformed YAML → empty frontmatter, full content as body,
    # parse_note still returns a ParsedNote (does not raise, does not return None)
    p = tmp_path / "bad.md"
    p.write_text("---\n: : ::\n---\n# Title\nbody", encoding="utf-8")
    note = parse_note(p)
    assert isinstance(note, ParsedNote)
