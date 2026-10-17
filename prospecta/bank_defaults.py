"""What a bank created fresh stores: the hybrid architecture out of the box.

`Memory.create_bank` writes these three values into the new row (the INSERT in
db/queries.py), so every path that creates a bank gets them: `prospecta init`,
`prospecta create-bank`, the service, `migrate-bank`, `import hindsight`.

- channel_config: Stage 1, the native Postgres `hybrid_chunk` channel
  (docs/postgres-hybrid.md).
- recall_config: Stage 2, the cross-encoder rerank. The model, floor, top_n and
  timeout are not stored: they come from the code defaults and the environment
  (stages.RERANKER_MODEL, PROSPECTA_RERANKER_MODEL), so changing a default reaches
  every bank that does not set its own.
- link_config: linking and entity extraction are off. The linker was removed from this library
  (docs/legacy.md) and nothing here reads the column; it is still written, `{"links": false,
  "entities": false}`, because an older library reading the bank treats a stored `{}` as "entities
  on" and would extract entities on every write.

Existing banks are never changed. These constants are used only by that INSERT:
there is no migration that writes them, and the column DEFAULTs of `banks` stay the
legacy values (`[]`, `{}`, `{}`), so a bank inserted by anything else (an older
library, a restore, raw SQL) is a legacy bank that runs the legacy path unchanged.
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
NEW_BANK_LINK_CONFIG: dict = {"links": False, "entities": False}


def new_bank_configs() -> dict:
    """A fresh copy of the three values, keyed by the `banks` column they fill."""
    return {
        "channel_config": copy.deepcopy(NEW_BANK_CHANNEL_CONFIG),
        "recall_config": copy.deepcopy(NEW_BANK_RECALL_CONFIG),
        "link_config": copy.deepcopy(NEW_BANK_LINK_CONFIG),
    }
