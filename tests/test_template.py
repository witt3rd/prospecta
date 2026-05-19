"""Tests for prospecta._template Jinja2 loader."""
import pytest

from prospecta._template import render_prompt, render_prompt_text


def test_render_prompt_basic():
    out = render_prompt("generate-index-text", {"content": "hello world"})
    assert "hello world" in out
    assert len(out) > 50


def test_render_prompt_missing_var_raises():
    """StrictUndefined fires on missing variable."""
    with pytest.raises(Exception):  # jinja2.UndefinedError
        render_prompt("generate-index-text", {})


def test_render_prompt_text_inline():
    out = render_prompt_text("Hello {{ name }}", {"name": "world"})
    assert out == "Hello world"


def test_formulate_queries_with_context():
    out = render_prompt(
        "formulate-queries",
        {"message": "did the birthday plan work out?", "context": "DM"},
    )
    assert "did the birthday plan work out?" in out
    assert "Context: DM" in out
    assert "queries" in out


def test_formulate_queries_empty_context():
    out = render_prompt(
        "formulate-queries",
        {"message": "something", "context": ""},
    )
    assert "something" in out
