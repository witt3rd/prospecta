"""Embed via an OpenAI-compatible HTTP endpoint.

Use this for infinity-emb, vLLM, TEI, LM Studio, Ollama (OpenAI-compat mode),
or any server that implements the POST /v1/embeddings OpenAI schema.

Install: ``pip install 'prospecta[embed-openai-compatible]'`` (or use
``prospecta[embed-openai]`` — same underlying ``openai`` library).

The ``openai`` package is lazy-imported inside the factory (P3).
"""
from __future__ import annotations

from .._types import EmbedCallable


def openai_compatible(
    base_url: str,
    model: str,
    *,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> EmbedCallable:
    """Return an ``EmbedCallable`` backed by an OpenAI-compatible endpoint.

    Args:
        base_url: server URL (e.g. ``http://localhost:7997/v1`` for infinity-emb).
        model: model id the server expects.
        api_key: optional auth token. Many local servers don't require one;
            defaults to ``"sk-not-needed"`` when None (the openai client
            requires a non-empty string).
        timeout: HTTP timeout in seconds.

    Returns:
        Callable taking ``list[str]``, returning ``list[list[float]]``.

    Raises:
        ImportError: if the ``openai`` package is not installed.
    """
    try:
        import openai as openai_pkg
    except ImportError as e:
        raise ImportError(
            "prospecta.embed.openai_compatible requires the openai library. "
            "Install with: pip install 'prospecta[embed-openai]'"
        ) from e

    client = openai_pkg.OpenAI(
        api_key=api_key or "sk-not-needed",
        base_url=base_url,
        timeout=timeout,
    )

    def embed(texts: list[str]) -> list[list[float]]:
        response = client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in response.data]

    return embed
