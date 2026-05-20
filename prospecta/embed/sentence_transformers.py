"""Embed via the sentence-transformers library (local, no API key).

Install: ``pip install 'prospecta[embed-sentence-transformers]'``

Models: any sentence-transformers model id (e.g. ``all-MiniLM-L6-v2``, 384-dim).

The provider library is lazy-imported inside the factory so importing this
module is free of provider dependencies (P3).
"""
from __future__ import annotations

from .._types import EmbedCallable


def sentence_transformers(model_name: str = "all-MiniLM-L6-v2") -> EmbedCallable:
    """Return an ``EmbedCallable`` backed by sentence-transformers.

    The model is loaded once at factory-call time and reused across embed
    invocations (lazy in the import sense, eager in the model-load sense —
    subsequent calls to the returned callable do not reload the model).

    Args:
        model_name: sentence-transformers model id. Default
            ``all-MiniLM-L6-v2`` (384-dim).

    Returns:
        Callable taking ``list[str]``, returning ``list[list[float]]``.

    Raises:
        ImportError: if sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "prospecta.embed.sentence_transformers requires the sentence-transformers library. "
            "Install with: pip install 'prospecta[embed-sentence-transformers]'"
        ) from e

    model = SentenceTransformer(model_name)

    def embed(texts: list[str]) -> list[list[float]]:
        vectors = model.encode(texts, convert_to_numpy=True)
        return [v.tolist() for v in vectors]

    return embed
