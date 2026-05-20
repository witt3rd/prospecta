"""Bilateral-spine write-side helper: LLM → question-shaped index_text strings.

Pure function module. No Memory dependency, no DB. Given content + an
LLMCallable, render the generate-index-text prompt (T7) and parse the
LLM response into a deduplicated list of question-shaped strings that
should retrieve this content when asked.

Used by prospecta._retain (T11). Caller-supplied index_text bypasses this
entirely (P4 caller-wins).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from prospecta._template import render_prompt, render_prompt_text
from prospecta._types import IndexTextGenerationError

if TYPE_CHECKING:
    from prospecta._types import LLMCallable


_DEFAULT_PROMPT_NAME = "generate-index-text"


def generate_index_text(
    content: str,
    llm: "LLMCallable",
    *,
    prompt_override: str | None = None,
    context: dict | None = None,
) -> tuple[list[str], str, str]:
    """Generate question-shaped index_text strings via LLM.

    Returns ``(parsed_questions, rendered_prompt, raw_response)`` — bilateral
    synthesis spine write-side. The prompt + raw_response are returned so the
    caller (``_retain.retain``) can thread them through to the tracer for
    durable trace capture in ``llm_calls`` + ``retain_events``.

    Args:
      content: The body to index (P5: full content, no truncation).
      llm: LLMCallable (Protocol). Library never imports providers (P3).
      prompt_override: Optional Jinja2 template string. When provided,
        renders inline with `{'content': content, **context}` instead of
        loading the default bundled prompt (P4 caller-wins).
      context: Extra Jinja2 variables (tags, source, metadata) for the
        prompt. The `content` key is always set; any `content` in context
        is ignored.

    Raises:
      IndexTextGenerationError: if the LLM produces no usable lines.
    """
    variables: dict = dict(context or {})
    variables["content"] = content

    if prompt_override is not None:
        rendered = render_prompt_text(prompt_override, variables)
    else:
        rendered = render_prompt(_DEFAULT_PROMPT_NAME, variables)

    raw = llm(messages=[{"role": "user", "content": rendered}])
    if not isinstance(raw, str):
        # LLMCallable Protocol says -> str; defend anyway.
        raw = str(raw)

    lines: list[str] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        # Strip common LLM list-noise prefixes (numbering, bullets) defensively.
        # Keep the question text itself intact.
        s = _strip_list_prefix(s)
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        lines.append(s)

    if not lines:
        raise IndexTextGenerationError(raw_response=raw)
    return lines, rendered, raw


def _strip_list_prefix(s: str) -> str:
    """Remove common list-prefix noise like '1.', '- ', '* ', '1) '.

    Keeps the question text intact. Conservative: only strips a single
    leading bullet/number marker followed by whitespace.
    """
    # bullets
    for marker in ("- ", "* ", "• "):
        if s.startswith(marker):
            return s[len(marker) :].strip()
    # numbered: '1.', '12)', etc.
    i = 0
    while i < len(s) and s[i].isdigit():
        i += 1
    if i > 0 and i < len(s) and s[i] in (".", ")"):
        rest = s[i + 1 :].lstrip()
        if rest:
            return rest
    return s
