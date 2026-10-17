"""Test helpers shared by the test modules (private; only tests import it)."""
from __future__ import annotations


def no_cut(config: list[dict]) -> list[dict]:
    """A channel config whose recall channels keep every candidate (min_rel 0):
    for tests on tiny stub-embedder banks where cosines are not meaningful."""
    out = []
    for e in config:
        e = dict(e)
        if e["name"] == "dense_chunk":
            e["params"] = {**(e.get("params") or {}), "min_rel": 0}
        out.append(e)
    return out
