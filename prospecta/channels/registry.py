"""Channel registry: banks.channel_config -> the ordered, weighted channels of a recall.

`hybrid_chunk` (the native Postgres Stage 1, docs/postgres-hybrid.md) is the one channel. A bank's
stored config names it with its weight and its params (the slot counts and the HNSW ef_search), and:

- an empty config, or one that names no channel this library has, means `hybrid_chunk` with its
  defaults (`effective_channel_config`): a bank that never set one runs the same Stage 1 as a bank
  created fresh;
- an entry naming a channel this library no longer has (dense_chunk, meta, bm25, keyword, question,
  graph) is ignored, silently, and listed in the recall trace's `skipped`.

A NEW config (set_channel_config) is stricter: only `hybrid_chunk` may be named. No stored config is
ever rewritten."""
from __future__ import annotations

from dataclasses import dataclass, field

from prospecta.channels.hybrid import HybridChunks

REGISTRY: dict[str, type] = {
    HybridChunks.name: HybridChunks,
}

DEFAULT_CHANNEL_ENTRY: dict = {"name": HybridChunks.name, "enabled": True, "weight": 1, "params": {}}


@dataclass(frozen=True)
class ConfiguredChannel:
    channel: object
    weight: float
    params: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.channel.name  # type: ignore[attr-defined]


def validate_channel_config(config: list[dict]) -> None:
    """For a NEW config (set_channel_config): a list of entries that name `hybrid_chunk` (once), each
    with a weight >= 0. `[]` is valid: it means the defaults."""
    if not isinstance(config, list):
        raise ValueError("channel_config must be a list")
    seen: set[str] = set()
    for e in config:
        name = e.get("name") if isinstance(e, dict) else None
        if name not in REGISTRY:
            raise ValueError(f"unknown channel {name!r}; known: {sorted(REGISTRY)}")
        if name in seen:
            raise ValueError(f"channel {name!r} configured twice")
        seen.add(name)
        w = e.get("weight", 1)
        if isinstance(w, bool) or not isinstance(w, (int, float)) or w < 0:
            raise ValueError(f"channel {name!r}: weight must be a number >= 0")


def effective_channel_config(stored: list[dict]) -> list[dict]:
    """The config a recall runs for a bank's STORED one: the stored entries as they are, with the
    default `hybrid_chunk` entry added when none of them names a channel this library has (an empty
    config, or one of channels that were removed)."""
    stored = [e for e in (stored or []) if isinstance(e, dict)]
    if any(e.get("name") in REGISTRY for e in stored):
        return stored
    return [dict(DEFAULT_CHANNEL_ENTRY, params={}), *stored]


def ignored_channels(config: list[dict]) -> list[str]:
    """The names in a STORED config that no channel of this library answers to."""
    return [e["name"] for e in config
            if isinstance(e, dict) and isinstance(e.get("name"), str) and e["name"] not in REGISTRY]


def build_channels(config: list[dict]) -> list[ConfiguredChannel]:
    """The enabled channels of a STORED config, in config order. An entry naming a channel that was
    removed is skipped, never an error: a recall must not fail for what a bank once stored."""
    out = []
    for e in config:
        if not isinstance(e, dict) or e.get("name") not in REGISTRY or not e.get("enabled", True):
            continue
        params = dict(e.get("params") or {})
        out.append(ConfiguredChannel(
            channel=REGISTRY[e["name"]](**params), weight=float(e.get("weight", 1)), params=params))
    return out
