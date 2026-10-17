"""Stage 1: the `hybrid_chunk` channel, weighted RRF fusion and the per-bank channel config."""
from prospecta.channels.base import Candidate, Channel, QueryPlan, RecallState
from prospecta.channels.fusion import FusedDoc, fuse
from prospecta.channels.hybrid import HybridChunks
from prospecta.channels.recall import read_channel_config, run_channels
from prospecta.channels.registry import (
    DEFAULT_CHANNEL_ENTRY,
    REGISTRY,
    build_channels,
    effective_channel_config,
    validate_channel_config,
)

__all__ = [
    "Candidate", "Channel", "QueryPlan", "RecallState", "FusedDoc", "fuse",
    "read_channel_config", "run_channels", "DEFAULT_CHANNEL_ENTRY", "REGISTRY",
    "build_channels", "effective_channel_config", "validate_channel_config", "HybridChunks",
]
