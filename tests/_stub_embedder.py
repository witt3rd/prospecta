"""Deterministic stub embedder + LLM for tests.

Hash-based bag-of-words → fixed 32-dim vector. No provider deps. Stable across runs.
"""
from __future__ import annotations

import hashlib
import math
import re


EMBED_DIM = 32


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def stub_embed(texts: list[str]) -> list[list[float]]:
    """Return one 32-dim vector per input text. Same text → same vector."""
    out: list[list[float]] = []
    for text in texts:
        vec = [0.0] * EMBED_DIM
        for tok in _tokenize(text):
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = h[0] % EMBED_DIM
            sign = 1.0 if (h[1] & 1) else -1.0
            vec[idx] += sign
        # L2-normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        else:
            # empty/no-token input: bias to a constant non-zero unit vec
            vec[0] = 1.0
        out.append(vec)
    return out


def stub_llm(messages: list[dict], *, json_mode: bool = False) -> str:
    """Predictable LLM stub. T9 doesn't call it; present for plumbing."""
    if json_mode:
        return '{"queries": []}'
    return "stub-llm-response"
