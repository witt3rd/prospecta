"""Lexicon-aware embedder for the bilateral 2×2 matrix integration test.

Purpose-built for T17. Production callers inject real embedders.

Strategy: hash-based bag-of-words across 24 dims (semantic content
signal) + 8 dims that fire on question-shape. L2-normalized.

This means semantic similarity tracks LEXICAL overlap closely (same as
real bag-of-words embeddings on short text). Tokens drive the semantic
signal; the question-shape dims separate question-strings from body
strings.

Design intent:
  - The query "did kelly's birthday work out?" has tokens {did, kelly,
    birthday, work, out} and is question-shaped.
  - The target body kelly_birthday_arc.md uses narrative prose that
    avoids "kelly" and "birthday" — so its semantic dims do NOT light
    up the same way as the query.
  - The red herring body pacific_beach_logistics.md uses both "kelly"
    and "birthday" lexically while being topically about logistics —
    its semantic dims DO light up the same way as the query.
  - When the spine is ON, the target gets indexed by "When is Kelly's
    birthday?" — restoring lexical alignment AND question-shape.
"""
from __future__ import annotations

import hashlib
import math
import re

EMBED_DIM = 32
_SEM_DIMS = 24  # 0..23 for token signal
_SHAPE_DIMS = 8  # 24..31 for question-shape signal

_QUESTION_WORDS = {
    "did", "do", "does", "how", "when", "what", "why", "who", "where",
    "is", "are", "was", "were", "will", "can", "could", "should",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


def bilateral_embedder(texts: list[str]) -> list[list[float]]:
    """Hash-based bag-of-words + question-shape signal, L2-normalized."""
    out: list[list[float]] = []
    for text in texts:
        vec = [0.0] * EMBED_DIM
        tokens = _tokenize(text)
        if not tokens:
            vec[0] = 1.0
            out.append(vec)
            continue

        has_question_mark = "?" in text
        q_word_count = 0
        for tok in tokens:
            if tok in _QUESTION_WORDS:
                q_word_count += 1
            # Each token contributes signed mass at one semantic dim.
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = h[0] % _SEM_DIMS
            sign = 1.0 if (h[1] & 1) else -1.0
            vec[idx] += sign

        # Question-shape signal across dims 24..31. Triggered when a
        # leading question-word is present OR a literal '?' appears.
        # Mass tuned so that the question-shape signal is significant
        # but does NOT dominate the token signal — token alignment must
        # remain the primary driver of similarity.
        is_question = has_question_mark or q_word_count > 0
        if is_question:
            # Scale: enough to materially boost cosine when both query
            # and indexed string are question-shaped, but not so much
            # that body-matches lose entirely.
            mass = 0.6
            for j in range(_SEM_DIMS, EMBED_DIM):
                vec[j] = mass

        # L2-normalize.
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        else:
            vec[0] = 1.0
        out.append(vec)
    return out


bilateral_embedder.dim = EMBED_DIM  # type: ignore[attr-defined]
