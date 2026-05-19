"""
Memory Note Parser

Parses markdown notes with YAML frontmatter.

The `index_text` frontmatter key is preserved (not stripped) — it is half
the bilateral spine on the write side: callers stamp `index_text` to steer
what gets embedded for recall, and that key must round-trip through
`parse_note` into the returned mapping.

Ported from animus (animus/memory/parser.py + animus/shared/parsing.py).
Stripped `from animus.shared import parse_frontmatter` — inlined here so
prospecta has no upstream animus dependency.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class FrontmatterParsed:
    """Result of parsing YAML frontmatter from content."""

    frontmatter: dict
    body: str


@dataclass
class ParsedNote:
    """Result of parsing a markdown note file.

    Mirrors the animus `Note` shape for the fields the parser populates.
    The `relationships` field carries all frontmatter keys that aren't
    `tags`/`created`/`title` — including `index_text`, which is load-bearing
    for the bilateral spine.
    """

    id: str
    title: str
    tags: list = field(default_factory=list)
    created: Optional[datetime] = None
    content: str = ""
    relationships: dict = field(default_factory=dict)
    file_path: Optional[Path] = None


def parse_frontmatter(content: str) -> FrontmatterParsed:
    """Extract YAML frontmatter from content.

    Works for markdown (--- delimited) and docstrings.

    Args:
        content: String content potentially containing YAML frontmatter.

    Returns:
        FrontmatterParsed with frontmatter dict and remaining body.
    """
    content = content.strip()

    if not content.startswith("---"):
        return FrontmatterParsed(frontmatter={}, body=content)

    # Find closing ---
    # Look for \n---\n or \n--- followed by end of string
    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        # Try end of content
        end_match = re.search(r"\n---\s*$", content[3:])
        if not end_match:
            return FrontmatterParsed(frontmatter={}, body=content)

    yaml_content = content[3 : end_match.start() + 3]
    remaining = content[end_match.end() + 3 :].lstrip("\n")

    try:
        frontmatter = yaml.safe_load(yaml_content) or {}
    except yaml.YAMLError:
        frontmatter = {}

    # Ensure frontmatter is a dict (could be string if malformed)
    if not isinstance(frontmatter, dict):
        frontmatter = {}

    return FrontmatterParsed(frontmatter=frontmatter, body=remaining)


def extract_title(content: str) -> str:
    """Extract the first H1 heading as the title."""
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return ""


def parse_note(file_path: Path) -> Optional[ParsedNote]:
    """Parse a markdown file into a ParsedNote object."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception:
        return None

    # Extract frontmatter
    parsed = parse_frontmatter(content)

    # Get note ID from filename
    note_id = file_path.stem.lower().replace(" ", "_")

    # Extract title
    title = extract_title(parsed.body)
    if not title:
        title = file_path.stem.replace("_", " ").title()

    # Extract tags
    tags = parsed.frontmatter.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]

    # Parse created date
    created = None
    if "created" in parsed.frontmatter:
        created_val = parsed.frontmatter["created"]
        if isinstance(created_val, datetime):
            created = created_val
        elif isinstance(created_val, str):
            try:
                created = datetime.fromisoformat(created_val)
            except ValueError:
                pass

    # Extract additional frontmatter fields as relationships.
    # These are used for metadata filtering and — load-bearing —
    # carry the `index_text` key on the write-side bilateral spine.
    standard_keys = {"tags", "created", "title"}
    extra_fields = {k: v for k, v in parsed.frontmatter.items() if k not in standard_keys}

    return ParsedNote(
        id=note_id,
        title=title,
        tags=tags,
        created=created,
        content=parsed.body,
        relationships=extra_fields,
        file_path=file_path,
    )
