"""Tests for prospecta._template Jinja2 loader."""
import pytest

from prospecta._template import render_prompt


def test_render_prompt_basic():
    out = render_prompt("ask-cited", {"question": "what is it?", "evidence": "hello world"})
    assert "hello world" in out and "what is it?" in out
    assert len(out) > 50


def test_render_prompt_missing_var_raises():
    """StrictUndefined fires on missing variable."""
    with pytest.raises(Exception):  # jinja2.UndefinedError
        render_prompt("ask-cited", {})
