"""LiteLLM-backed factory implementations for prospecta.defaults.

Only this module imports ``litellm`` (P3: core library has zero provider
imports). The parent ``__init__.py`` lazy-imports this module and re-raises
``ImportError`` with the documented install hint if LiteLLM is missing.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import litellm

if TYPE_CHECKING:
    from prospecta._types import EmbedCallable, LLMCallable

DEFAULT_EMBED_MODEL = "openai/text-embedding-3-small"
DEFAULT_LLM_MODEL = "openai/gpt-4o-mini"


def make_default_embedder(
    model: str | None = None, dimensions: int | None = None
) -> EmbedCallable:
    """Build an EmbedCallable backed by LiteLLM.

    Resolution order for the model id:
      1. explicit ``model`` argument (highest precedence)
      2. ``PROSPECTA_EMBED_MODEL`` env var
      3. ``openai/text-embedding-3-small`` (default, dim 1536)

    Returns a callable conforming to the ``EmbedCallable`` Protocol:
    ``(texts: list[str]) -> list[list[float]]``.

    LiteLLM handles provider routing and authentication via its standard
    env var resolution (e.g., ``OPENAI_API_KEY``); prospecta.defaults
    does not read provider keys directly.
    """
    resolved_model = model or os.environ.get("PROSPECTA_EMBED_MODEL", DEFAULT_EMBED_MODEL)

    def embed(texts: list[str]) -> list[list[float]]:
        kwargs: dict = {"model": resolved_model, "input": texts}
        if dimensions is not None:
            kwargs["dimensions"] = dimensions  # e.g. text-embedding-3-large cut to 1536
        response = litellm.embedding(**kwargs)
        # LiteLLM response.data is a list of dicts (or objects) with "embedding".
        return [_extract_embedding(item) for item in response.data]

    return embed


def make_default_llm(model: str | None = None) -> LLMCallable:
    """Build an LLMCallable backed by LiteLLM.

    Resolution order for the model id:
      1. explicit ``model`` argument (highest precedence)
      2. ``PROSPECTA_LLM_MODEL`` env var
      3. ``openai/gpt-4o-mini`` (default)

    Returns a callable conforming to the ``LLMCallable`` Protocol:
    ``(messages: list[dict], *, json_mode: bool = False,
    model: str | None = None) -> str``; ``model`` overrides the bank model
    for that call.

    When ``json_mode=True``, passes ``response_format={"type": "json_object"}``
    to LiteLLM. Otherwise, no ``response_format`` is set.
    """
    resolved_model = model or os.environ.get("PROSPECTA_LLM_MODEL", DEFAULT_LLM_MODEL)

    def llm(messages: list[dict], *, json_mode: bool = False,
            model: str | None = None) -> str:
        kwargs: dict = {"model": model or resolved_model, "messages": messages}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = litellm.completion(**kwargs)
        return response.choices[0].message.content

    return llm


def make_accounted_llm(model: str = "openrouter/anthropic/claude-sonnet-5.5") -> LLMCallable:
    """Like make_default_llm, but returns a prospecta.stages.LLMResult carrying
    model, tokens and cost, so recall stages can account for every call.
    No max_tokens cap is set."""
    from prospecta.stages import LLMResult

    def llm(messages: list[dict], *, json_mode: bool = False):
        kwargs: dict = {"model": model, "messages": messages}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = litellm.completion(**kwargs)
        usage = getattr(response, "usage", None)
        try:
            cost = float(litellm.completion_cost(completion_response=response))
        except Exception:
            cost = None
        return LLMResult(
            text=response.choices[0].message.content, model=model,
            tokens_in=getattr(usage, "prompt_tokens", None),
            tokens_out=getattr(usage, "completion_tokens", None), cost_usd=cost,
        )

    return llm


def _extract_embedding(item: object) -> list[float]:
    """Extract embedding vector from a LiteLLM response.data entry.

    LiteLLM may return either a dict or an object with ``.embedding``;
    accept both shapes.
    """
    if isinstance(item, dict):
        return item["embedding"]
    return item.embedding  # type: ignore[attr-defined]
