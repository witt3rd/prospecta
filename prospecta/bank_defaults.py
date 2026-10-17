"""What a bank created fresh stores: the hybrid architecture out of the box.

`Memory.create_bank` writes these two values into the new row (the INSERT in
db/queries.py), so every path that creates a bank gets them: `prospecta init`,
`prospecta create-bank`, the service, `migrate-bank`, `import hindsight`.

- channel_config: Stage 1, the native Postgres `hybrid_chunk` channel
  (docs/postgres-hybrid.md). It is stored, not implied, so that the row says what the bank runs;
  a bank whose `channel_config` is empty runs the same channel with its defaults
  (channels/registry.py).
- recall_config: Stage 2, the cross-encoder rerank. The model, floor, top_n and
  timeout are not stored: they come from the code defaults and the environment
  (stages.RERANKER_MODEL, PROSPECTA_RERANKER_MODEL), so changing a default reaches
  every bank that does not set its own.

Existing banks are never changed. These constants are used only by that INSERT:
there is no migration that writes them, and the column DEFAULTs of `banks` stay `[]` and `{}`.
Nothing here imports the channel or stage modules (tests validate each value
against them).
"""
from __future__ import annotations

import copy

NEW_BANK_CHANNEL_CONFIG: list[dict] = [
    {"name": "hybrid_chunk", "enabled": True, "weight": 1, "params": {}},
]
NEW_BANK_RECALL_CONFIG: dict = {
    "rerank": {"enabled": True, "stage": "cross_encoder"},
}


def new_bank_configs() -> dict:
    """A fresh copy of the two values, keyed by the `banks` column they fill."""
    return {
        "channel_config": copy.deepcopy(NEW_BANK_CHANNEL_CONFIG),
        "recall_config": copy.deepcopy(NEW_BANK_RECALL_CONFIG),
    }
