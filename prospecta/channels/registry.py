"""Channel registry: banks.channel_config -> ordered, weighted channels."""
from __future__ import annotations

from dataclasses import dataclass, field

from prospecta.channels.bm25 import Bm25Chunks
from prospecta.channels.semantic import AnticipatedQuestions, DenseChunks

REGISTRY: dict[str, type] = {
    DenseChunks.name: DenseChunks,
    AnticipatedQuestions.name: AnticipatedQuestions,
    Bm25Chunks.name: Bm25Chunks,
}

# The measured weights (design 8.2) for the channels built so far. `bm25` (weight 1)
# is registered but opt-in: add {"name": "bm25", "weight": 1} to a bank's config.
DEFAULT_CHANNEL_CONFIG: list[dict] = [
    {"name": "dense_chunk", "enabled": True, "weight": 4, "params": {"limit": 50}},
    {"name": "question", "enabled": True, "weight": 1, "params": {"limit": 50}},
]

DEFAULT_LIMIT = 50


@dataclass(frozen=True)
class ConfiguredChannel:
    channel: object
    weight: float
    limit: int
    params: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.channel.name  # type: ignore[attr-defined]


def validate_channel_config(config: list[dict]) -> None:
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


def build_channels(config: list[dict]) -> list[ConfiguredChannel]:
    """Enabled channels in config order, recall first, then filter, then expand."""
    validate_channel_config(config)
    out = []
    for e in config:
        if not e.get("enabled", True):
            continue
        params = dict(e.get("params") or {})
        out.append(ConfiguredChannel(
            channel=REGISTRY[e["name"]](), weight=float(e.get("weight", 1)),
            limit=int(params.get("limit", DEFAULT_LIMIT)), params=params,
        ))
    order = {"recall": 0, "filter": 1, "expand": 2}
    out.sort(key=lambda c: order[c.channel.kind])  # type: ignore[attr-defined]  # stable
    return out
