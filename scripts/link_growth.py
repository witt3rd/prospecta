"""Why the link table grows with the square of the bank, counted.

Builds a synthetic bank of N notes (a fixed number of topics, so a topic holds more
notes as the bank grows, as a real bank's subjects do), takes for every note the
neighbour set the Linker uses to pick link candidates (prospecta._linker.neighbours_until_drop,
cosine >= 0.9 x the nearest, floor 0.5, no count limit) and counts the distinct
note pairs, i.e. links per note counted once per direction pair.

    python scripts/link_growth.py [N ...]        # default 1000 4000 16000

`unbounded` is the rule as it was (candidates: every sufficiently similar note);
`capped` is the same rule with the candidate list cut to 2 x max_links_per_note (the
Linker's new reach); the final per-note bound (max_links_per_note, tested in
tests/test_link_cap.py) is applied when the links are written.
"""
from __future__ import annotations

import sys

import numpy as np

from prospecta._linker import NEIGHBOUR_MIN_COS, NEIGHBOUR_MIN_REL, neighbours_until_drop

TOPICS, DIM, SEED, SPAN_DAYS = 6, 64, 7, 365


def bank(n: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    centres = rng.normal(size=(TOPICS, DIM))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    topic = rng.integers(0, TOPICS, size=n)
    x = centres[topic] + 1.0 * rng.normal(size=(n, DIM)) / np.sqrt(DIM)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def links_per_note(x: np.ndarray, limit: int | None) -> tuple[float, int]:
    n = len(x)
    pairs: set[tuple[int, int]] = set()
    for lo in range(0, n, 512):
        sims = x[lo:lo + 512] @ x.T
        for r in range(sims.shape[0]):
            i = lo + r
            row = sims[r].copy()
            row[i] = -1
            order = np.argsort(-row)

            def fetch(k, order=order, row=row):
                return [(int(j), float(row[j])) for j in order[:k]]
            cands, _ = neighbours_until_drop(fetch, lambda c: c[1], NEIGHBOUR_MIN_REL,
                                             floor=NEIGHBOUR_MIN_COS, limit=limit)
            pairs.update((min(i, j), max(i, j)) for j, _ in cands)
    return 2 * len(pairs) / n, len(pairs)


def temporal_per_note(n: int) -> float:
    """TEMPORALLY_CLOSE links per note: notes dated over a fixed span, the Linker's rule
    (within 3 days, closeness 1/(1+days) >= 0.5 x the closest note's), no count limit."""
    rng = np.random.default_rng(SEED)
    day = np.sort(rng.integers(0, SPAN_DAYS, size=n))
    pairs = set()
    for i in range(n):
        lo, hi = np.searchsorted(day, [day[i] - 3, day[i] + 4])
        near = [j for j in range(lo, hi) if j != i]
        if not near:
            continue
        gap = np.abs(day[near] - day[i])
        keep = [j for j, g in zip(near, gap) if 1 / (1 + g) >= 0.5 / (1 + gap.min())]
        pairs.update((min(i, j), max(i, j)) for j in keep)
    return 2 * len(pairs) / n


if __name__ == "__main__":
    sizes = [int(a) for a in sys.argv[1:]] or [1000, 4000, 16000]
    print(f"{'notes':>7} {'semantic':>10} {'semantic, reach cut':>20} {'temporal-close':>15}   (links per note)")
    for n in sizes:
        x = bank(n)
        print(f"{n:>7} {links_per_note(x, None)[0]:>10.1f} {links_per_note(x, 100)[0]:>20.1f} "
              f"{temporal_per_note(n):>15.1f}", flush=True)
