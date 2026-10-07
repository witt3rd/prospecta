"""Weighted reciprocal-rank fusion at document level (design 8.2):
score(d) = sum_c w_c / (k + rank_c(d)); ties broken by source name."""
from __future__ import annotations

from dataclasses import dataclass, field

from prospecta.channels.base import Candidate


@dataclass(frozen=True)
class FusedDoc:
    document_id: str
    source: str
    score: float
    best: Candidate                       # the evidence child
    ranks: dict[str, int] = field(default_factory=dict)     # channel -> rank
    scores: dict[str, float] = field(default_factory=dict)  # channel -> own score


def fuse(
    lists: dict[str, list[Candidate]],
    weights: dict[str, float],
    *,
    k: int = 60,
    pool: int | None = None,
) -> list[FusedDoc]:
    """Fuse per-channel candidate lists. Channels with weight 0 (or absent
    from `weights`) contribute nothing. A document appearing twice in one
    channel counts once, at its best (lowest) rank. `pool` None (default)
    returns every fused document; the score cut is the caller's (stages)."""
    acc: dict[str, dict] = {}
    for channel in sorted(lists):
        w = float(weights.get(channel, 0.0))
        for c in lists[channel]:
            d = acc.setdefault(c.document_id, {
                "source": c.source, "score": 0.0, "ranks": {}, "scores": {},
                "best": None, "best_key": None,
            })
            if channel in d["ranks"] and d["ranks"][channel] <= c.rank:
                continue
            if channel in d["ranks"]:
                d["score"] -= w / (k + d["ranks"][channel])
            d["ranks"][channel] = c.rank
            d["scores"][channel] = c.score
            d["score"] += w / (k + c.rank)
            # evidence: the child from the heaviest channel, then the best rank
            key = (-w, c.rank, channel)
            if w > 0 and (d["best_key"] is None or key < d["best_key"]):
                d["best"], d["best_key"] = c, key
            elif d["best"] is None:
                d["best"] = c
    fused = [
        FusedDoc(document_id=doc, source=d["source"], score=d["score"], best=d["best"],
                 ranks=d["ranks"], scores=d["scores"])
        for doc, d in acc.items() if d["score"] > 0
    ]
    fused.sort(key=lambda f: (-f.score, f.source, f.document_id))
    return fused if pool is None else fused[:pool]
