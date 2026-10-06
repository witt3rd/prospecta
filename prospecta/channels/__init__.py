"""Recall channels, weighted RRF fusion and the per-bank registry."""
from prospecta.channels.base import Candidate, Channel, Filters, QueryPlan, RecallState
from prospecta.channels.fusion import FusedDoc, fuse
from prospecta.channels.recall import read_channel_config, run_channels
from prospecta.channels.registry import (
    DEFAULT_CHANNEL_CONFIG,
    REGISTRY,
    build_channels,
    validate_channel_config,
)
from prospecta.channels.semantic import AnticipatedQuestions, DenseChunks

__all__ = [
    "Candidate", "Channel", "Filters", "QueryPlan", "RecallState", "FusedDoc", "fuse",
    "read_channel_config", "run_channels", "DEFAULT_CHANNEL_CONFIG", "REGISTRY",
    "build_channels", "validate_channel_config", "AnticipatedQuestions", "DenseChunks",
]
