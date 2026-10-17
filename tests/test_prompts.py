"""Tests verifying prompts are translated, not copied.

These assert prospecta's prompts do NOT contain Cookie/Animus/cookiefam
references. Failure indicates incomplete translation.
"""
from pathlib import Path

import pytest

PROMPTS_DIR = Path(__file__).parent.parent / "prospecta" / "prompts"

PROSPECTA_PROMPTS = [
    "ask-cited.md",
    "expand-searches.md",
    "judge-answer.md",
]

# Forbidden references (case-insensitive check)
FORBIDDEN_TERMS = [
    "cookie",
    "animus",
    "cookiefam",
    "yaya",
    "wizard",
    "donald",
    "kelly",
    "forge",
    "janus",
]

# Forbidden glyphs (exact-match)
FORBIDDEN_GLYPHS = [
    "🍪", "🖤", "🌴", "⚒️", "🧙‍♂️", "⚡", "🔥",
]


@pytest.mark.parametrize("prompt_file", PROSPECTA_PROMPTS)
def test_prompt_exists_and_substantial(prompt_file):
    path = PROMPTS_DIR / prompt_file
    assert path.exists(), f"Missing: {path}"
    assert path.stat().st_size > 100, f"Too short: {path}"


@pytest.mark.parametrize("prompt_file", PROSPECTA_PROMPTS)
@pytest.mark.parametrize("forbidden", FORBIDDEN_TERMS)
def test_prompt_has_no_cookie_references(prompt_file, forbidden):
    text = (PROMPTS_DIR / prompt_file).read_text().lower()
    assert forbidden not in text, (
        f"{prompt_file} contains '{forbidden}' — translation incomplete"
    )


@pytest.mark.parametrize("prompt_file", PROSPECTA_PROMPTS)
@pytest.mark.parametrize("glyph", FORBIDDEN_GLYPHS)
def test_prompt_has_no_forbidden_glyphs(prompt_file, glyph):
    text = (PROMPTS_DIR / prompt_file).read_text()
    assert glyph not in text, (
        f"{prompt_file} contains glyph '{glyph}' — translation incomplete"
    )
