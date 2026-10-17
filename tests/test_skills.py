"""The skills under skills/ keep the house shape; AGENTS.md lists them.

Links and doc paths inside them are checked by tests/test_docs_links.py (skills/ and AGENTS.md are in its
Markdown roots). This file checks the front matter and the index; it needs no database.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SKILLS = ROOT / "skills"
EXPECTED = {"recall", "retain", "search-and-ask", "tuning-and-testing", "operating"}


def _front_matter(path: Path) -> dict:
    text = path.read_text()
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    assert m, f"{path}: no YAML front matter"
    data = yaml.safe_load(m.group(1))
    assert isinstance(data, dict), f"{path}: front matter is not a mapping"
    return data


def test_the_skills_are_exactly_the_documented_five():
    found = {p.parent.name for p in SKILLS.glob("*/SKILL.md")}
    assert found == EXPECTED
    assert not (SKILLS / "SKILL.md").exists(), "skills/<purpose>/SKILL.md, never a monolith skills/SKILL.md"


def test_front_matter_parses_and_the_description_is_short():
    for path in sorted(SKILLS.glob("*/SKILL.md")):
        fm = _front_matter(path)
        assert set(fm) == {"name", "description"}, f"{path}: only name and description"
        assert fm["name"] == path.parent.name, f"{path}: name must equal the directory"
        assert re.fullmatch(r"[a-z0-9-]+", fm["name"])
        desc = fm["description"]
        assert isinstance(desc, str) and 0 < len(desc) <= 300, f"{path}: description is one or two short sentences"
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", desc.strip()) if s]
        assert 1 <= len(sentences) <= 2, f"{path}: {len(sentences)} sentences"
        assert "REQUIRED" not in desc


def test_each_skill_is_a_router_not_a_monolith():
    for path in sorted(SKILLS.glob("*/SKILL.md")):
        assert len(path.read_text().splitlines()) <= 150, f"{path}: push depth into references/"


def test_agents_md_lists_every_skill_and_carries_the_adoption_rule():
    text = (ROOT / "AGENTS.md").read_text()
    for name in EXPECTED:
        assert f"skills/{name}/SKILL.md" in text, f"AGENTS.md does not list {name}"
    assert "adopt or adapt these skills" in text
    assert "come back as PRs to" in text
    assert "every recall PR states the skill it follows or changes" in text
