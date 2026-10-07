"""Score-based candidate pools (captain's rule): what a stage SEES is decided by
a score relative to the best, never by a magic count. The only bound is the
physical context window of the model that reads it; a set that exceeds the
window is split into batches (map-reduce), never truncated."""
from __future__ import annotations

from typing import Callable, Sequence, TypeVar

T = TypeVar("T")

# Claude Sonnet 5.5 context window: 200,000 tokens (same figure and 4 chars/token
# estimate as _synth.DEFAULT_CONTEXT_TOKENS).
SONNET_CONTEXT_TOKENS = 200_000
RESERVED_TOKENS = 4_000        # template, question and answer
CHARS_PER_TOKEN = 4

# Defaults chosen from the v0.2 eval design (docs/limits.md).
# Pool cut = the size of the RERANK input only (quality/cost knob): fused score >= 0.15 x
# best fused (the scout's knee); configurable down to 0.05 (recall_config.rerank.min_rel_score).
POOL_MIN_REL = 0.15
HOP_MIN_REL = 0.4              # new notes of a hop follow-up run: >= 0.4 x the best of that run
READER_MIN_REL = 0.6           # rerank grade (0..3) or fused score >= 0.6 x best
# Retrieval channels do NOT cut: every candidate goes into fusion, bounded only by the
# index fetch completing (semantic._scan_everything). `min_rel` stays an explicit per-bank override.
CHANNEL_MIN_REL = 0.0
FETCH_PAGE = 64                # rows per fetch (throughput; the fetch doubles until the cut is reached)


def rel_cut(items: Sequence[T], score: Callable[[T], float], rel: float) -> list[T]:
    """Items (any order) whose score >= rel x the best score, in the given order.
    A best score <= 0 gives no basis for a relative cut: everything qualifies."""
    if not items:
        return []
    best = max(score(i) for i in items)
    if best <= 0 or rel <= 0:
        return list(items)
    cut = rel * best
    return [i for i in items if score(i) >= cut]


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN + 1


def split_batches(items: Sequence[T], text_of: Callable[[T], str],
                  budget_tokens: int = SONNET_CONTEXT_TOKENS - RESERVED_TOKENS) -> list[list[T]]:
    """Consecutive batches that each fit `budget_tokens`; one item is never cut
    (a lone item over the budget gets a batch of its own)."""
    out: list[list[T]] = []
    cur: list[T] = []
    used = 0
    for it in items:
        n = estimate_tokens(text_of(it))
        if cur and used + n > budget_tokens:
            out.append(cur)
            cur, used = [], 0
        cur.append(it)
        used += n
    if cur:
        out.append(cur)
    return out


def fetch_until_cut(fetch: Callable[[int], list], score: Callable, rel: float,
                    page: int = FETCH_PAGE) -> list:
    """`fetch(n)` returns up to n rows, best first. Grow n (doubling) until the
    last row falls below rel x the best score or the source is exhausted, then
    return the qualifying rows. Nothing qualifying is left behind."""
    n = page
    while True:
        rows = fetch(n)
        if not rows:
            return []
        best = score(rows[0])
        if rel <= 0 or best <= 0:       # no relative basis: the whole source qualifies
            if len(rows) < n:
                return rows
        elif len(rows) < n or score(rows[-1]) < rel * best:
            return [r for r in rows if score(r) >= rel * best]
        n *= 2
