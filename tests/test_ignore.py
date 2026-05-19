"""Tests for prospecta._ignore."""

from __future__ import annotations

from pathlib import Path

import pytest

from prospecta._ignore import (
    DEFAULT_IGNORE_DIRS,
    clear_ignore_cache,
    load_ignore_patterns,
    should_ignore,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_ignore_cache()
    yield
    clear_ignore_cache()


def test_default_ignore_dirs_excluded(tmp_path: Path):
    for d in (".git", "__pycache__", ".venv", "node_modules"):
        assert d in DEFAULT_IGNORE_DIRS
        p = tmp_path / d / "file.md"
        assert should_ignore(p, tmp_path) is True


def test_non_ignored_path_passes(tmp_path: Path):
    p = tmp_path / "notes" / "hello.md"
    assert should_ignore(p, tmp_path) is False


def test_default_pyc_pattern_ignored(tmp_path: Path):
    p = tmp_path / "foo.pyc"
    assert should_ignore(p, tmp_path) is True


def test_memoryignore_file_patterns_honored(tmp_path: Path):
    (tmp_path / ".memoryignore").write_text("secret.md\n*.tmp\n", encoding="utf-8")
    assert should_ignore(tmp_path / "secret.md", tmp_path) is True
    assert should_ignore(tmp_path / "scratch.tmp", tmp_path) is True
    assert should_ignore(tmp_path / "keep.md", tmp_path) is False


def test_memoryignore_comments_and_blank_lines(tmp_path: Path):
    (tmp_path / ".memoryignore").write_text(
        "# this is a comment\n\n*.log\n", encoding="utf-8"
    )
    patterns = load_ignore_patterns(tmp_path)
    assert "*.log" in patterns
    assert "# this is a comment" not in patterns


def test_negation_pattern_reincludes(tmp_path: Path):
    (tmp_path / ".memoryignore").write_text(
        "*.md\n!keep.md\n", encoding="utf-8"
    )
    assert should_ignore(tmp_path / "drop.md", tmp_path) is True
    assert should_ignore(tmp_path / "keep.md", tmp_path) is False


def test_directory_anchor_pattern(tmp_path: Path):
    (tmp_path / ".memoryignore").write_text("private/\n", encoding="utf-8")
    assert should_ignore(tmp_path / "private" / "x.md", tmp_path) is True
    assert should_ignore(tmp_path / "public" / "x.md", tmp_path) is False


def test_load_ignore_patterns_no_file(tmp_path: Path):
    patterns = load_ignore_patterns(tmp_path)
    # Just defaults
    assert "*.pyc" in patterns
    assert "*.pyo" in patterns


def test_load_ignore_patterns_cached(tmp_path: Path):
    (tmp_path / ".memoryignore").write_text("a.md\n", encoding="utf-8")
    p1 = load_ignore_patterns(tmp_path)
    # mutate file; cached result should remain
    (tmp_path / ".memoryignore").write_text("b.md\n", encoding="utf-8")
    p2 = load_ignore_patterns(tmp_path)
    assert p1 is p2
    clear_ignore_cache()
    p3 = load_ignore_patterns(tmp_path)
    assert "b.md" in p3
