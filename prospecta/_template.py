"""Jinja2 template loader for prospecta prompts.

Prompts ship with the package at prospecta/prompts/*.md. Callers can
override per-call (M2 fold) by passing prompt text directly.
"""
from __future__ import annotations

from pathlib import Path

import jinja2

_PACKAGE_ROOT = Path(__file__).parent
_PROMPTS_DIR = _PACKAGE_ROOT / "prompts"


def _make_env(prompts_dir: Path | None = None) -> jinja2.Environment:
    search_dir = prompts_dir or _PROMPTS_DIR
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(search_dir)),
        autoescape=False,
        keep_trailing_newline=True,
        undefined=jinja2.StrictUndefined,
    )


def render_prompt(
    name: str,
    variables: dict | None = None,
    *,
    prompts_dir: Path | None = None,
) -> str:
    """Render a named prompt template.

    name — bare name without .md (e.g., 'generate-index-text')
    variables — dict passed to Jinja2
    prompts_dir — override default location (per-call M2 escape hatch)
    """
    env = _make_env(prompts_dir)
    template = env.get_template(f"{name}.md")
    return template.render(**(variables or {}))


def render_prompt_text(text: str, variables: dict | None = None) -> str:
    """Render an inline prompt string (caller-supplied prompt_override)."""
    env = jinja2.Environment(
        autoescape=False,
        keep_trailing_newline=True,
        undefined=jinja2.StrictUndefined,
    )
    template = env.from_string(text)
    return template.render(**(variables or {}))
