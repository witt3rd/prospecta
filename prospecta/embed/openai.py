"""Embed via direct OpenAI HTTP (no LiteLLM dependency).

Install: ``pip install 'prospecta[embed-openai]'``

Models: any OpenAI embedding model (``text-embedding-3-small``,
``text-embedding-3-large``, etc.).

The ``openai`` package is lazy-imported inside the factory (P3).
"""
from __future__ import annotations

import os

from .._types import EmbedCallable


def openai(
    api_key: str | None = None,
    model: str = "text-embedding-3-small",
) -> EmbedCallable:
    """Return an ``EmbedCallable`` backed by OpenAI's HTTP embedding endpoint.

    Args:
        api_key: OpenAI API key. If ``None``, reads ``OPENAI_API_KEY`` env.
        model: OpenAI embedding model. Default ``text-embedding-3-small``
            (1536-dim).

    Returns:
        Callable taking ``list[str]``, returning ``list[list[float]]``.

    Raises:
        ImportError: if the ``openai`` package is not installed.
        RuntimeError: if ``api_key`` is None and ``OPENAI_API_KEY`` is unset.
    """
    try:
        import openai as openai_pkg
    except ImportError as e:
        raise ImportError(
            "prospecta.embed.openai requires the openai library. "
            "Install with: pip install 'prospecta[embed-openai]'"
        ) from e

    resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_key:
        raise RuntimeError(
            "prospecta.embed.openai requires an API key. "
            "Pass api_key= or set OPENAI_API_KEY."
        )

    client = openai_pkg.OpenAI(api_key=resolved_key)

    def embed(texts: list[str]) -> list[list[float]]:
        response = client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in response.data]

    return embed
