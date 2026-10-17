"""Channel registry: banks.channel_config -> ordered, weighted channels."""
from __future__ import annotations

from dataclasses import dataclass, field

from prospecta._warn import warn_once
from prospecta.channels.hybrid import HybridChunks
from prospecta.channels.meta import MetadataScope
from prospecta.channels.semantic import DenseChunks

REGISTRY: dict[str, type] = {
    DenseChunks.name: DenseChunks,
    HybridChunks.name: HybridChunks,
    MetadataScope.name: MetadataScope,
}

# Channels that were removed (docs/legacy.md), and why. A NEW config cannot name them
# (validate_channel_config raises); a STORED config that does is still read: the channel is
# skipped with one warning per process (build_channels), never an error that fails a recall.
RETIRED: dict[str, str] = {
    "bm25": "hybrid_chunk's stemmed retriever replaced it",
    "keyword": "hybrid_chunk's exact retriever replaced it",
    "question": "hybrid_chunk reads chunk items, and no channel reads question items any more",
    "graph": "the linker that wrote the links it walked was removed",
}

# The measured weights (design 8.2) of the channels that remain besides `hybrid_chunk`: this is
# the legacy default of a bank that names its channels by hand. `hybrid_chunk`, the native
# Postgres Stage 1 (docs/postgres-hybrid.md), is not in this list because it is what a bank
# created fresh stores (prospecta/bank_defaults.py).
DEFAULT_CHANNEL_CONFIG: list[dict] = [
    {"name": "dense_chunk", "enabled": True, "weight": 4, "params": {}},
    {"name": "meta", "enabled": True, "weight": 3, "params": {}},
]


@dataclass(frozen=True)
class ConfiguredChannel:
    channel: object
    weight: float
    params: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.channel.name  # type: ignore[attr-defined]


def validate_channel_config(config: list[dict]) -> None:
    """For a NEW config (set_channel_config): every channel must exist. A retired one is refused."""
    if not isinstance(config, list):
        raise ValueError("channel_config must be a list")
    seen: set[str] = set()
    for e in config:
        name = e.get("name") if isinstance(e, dict) else None
        if name in RETIRED:
            raise ValueError(f"channel {name!r} was retired ({RETIRED[name]}); see docs/legacy.md")
        if name not in REGISTRY:
            raise ValueError(f"unknown channel {name!r}; known: {sorted(REGISTRY)}")
        if name in seen:
            raise ValueError(f"channel {name!r} configured twice")
        seen.add(name)
        w = e.get("weight", 1)
        if isinstance(w, bool) or not isinstance(w, (int, float)) or w < 0:
            raise ValueError(f"channel {name!r}: weight must be a number >= 0")


def _make(cls, params: dict):
    return cls(**params) if getattr(cls, "takes_params", False) else cls()


def retired_channels(config: list[dict]) -> list[str]:
    """The enabled channels of a STORED config that were retired: skipped, never run."""
    return [e["name"] for e in config
            if isinstance(e, dict) and e.get("name") in RETIRED and e.get("enabled", True)]


def build_channels(config: list[dict]) -> list[ConfiguredChannel]:
    """Enabled channels in config order, recall first, then filter.

    This reads a STORED config: a retired channel it names is skipped with one warning per
    process (the bank keeps what it stored; recall must not fail for it)."""
    for name in retired_channels(config):
        warn_once(f"channel:{name}",
                  "channel_config names the %r channel, which was retired (%s) and is skipped; "
                  "see docs/legacy.md", name, RETIRED[name])
    config = [e for e in config if not (isinstance(e, dict) and e.get("name") in RETIRED)]
    validate_channel_config(config)
    out = []
    for e in config:
        if not e.get("enabled", True):
            continue
        params = dict(e.get("params") or {})
        out.append(ConfiguredChannel(
            channel=_make(REGISTRY[e["name"]], params), weight=float(e.get("weight", 1)),
            params=params,
        ))
    order = {"recall": 0, "filter": 1}
    out.sort(key=lambda c: order[c.channel.kind])  # type: ignore[attr-defined]  # stable
    return out
