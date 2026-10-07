"""Run a bank's configured channels, fuse, and describe the whole recall."""
from __future__ import annotations

import time

from prospecta.channels.base import Candidate, QueryPlan, RecallState
from prospecta.channels.fusion import FusedDoc, fuse
from prospecta.channels.registry import build_channels

RRF_K = 60

SELECT_CHANNEL_CONFIG = "SELECT channel_config FROM banks WHERE bank_id = %s"


def read_channel_config(conn, bank_id: str) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(SELECT_CHANNEL_CONFIG, (bank_id,))
        row = cur.fetchone()
    return list(row[0]) if row and row[0] else []


def run_channels(
    state: RecallState, plan: QueryPlan, config: list[dict],
    *, k: int = RRF_K, pool: int | None = None,
) -> tuple[list[FusedDoc], dict]:
    """Returns (fused pool, trace). The trace holds the plan, one entry per
    channel (name, kind, weight, n, latency_ms, cost_usd, error), the fusion
    record, and every channel's candidates. A failing channel is recorded
    with its error and contributes nothing; the others still run."""
    channels = build_channels(config)
    lists: dict[str, list[Candidate]] = {}
    traces: list[dict] = []
    for cc in channels:
        t0 = time.monotonic()
        error = None
        cands: list[Candidate] = []
        state.pool = [c for lst in lists.values() for c in lst]
        try:
            cands = cc.channel.retrieve(plan, state)
        except Exception as exc:  # one channel never sinks the recall
            error = f"{type(exc).__name__}: {exc}"
            state.conn.rollback()
        lists[cc.name] = cands
        traces.append({
            "name": cc.name, "kind": cc.channel.kind, "weight": cc.weight,
            "n": len(cands), "latency_ms": int((time.monotonic() - t0) * 1000),
            "cost_usd": 0.0, "error": error,
        })
    state.channel_lists = lists
    weights = {cc.name: cc.weight for cc in channels}
    fused = fuse(lists, weights, k=k, pool=pool)
    trace = {
        "plan": plan.to_json(),
        "channels": traces,
        "fusion": {
            "method": "weighted_rrf", "k": k, "weights": weights,
            "pool": [f.document_id for f in fused],
        },
        "candidates": [
            {"document_id": c.document_id, "item_id": c.item_id,
             "channel": c.channel, "rank": c.rank, "score": c.score}
            for name in lists for c in lists[name]
        ],
    }
    return fused, trace
