"""Recall channels, weighted RRF fusion and the per-bank registry."""
from prospecta.channels.base import Candidate, Channel, Filters, QueryPlan, RecallState
from prospecta.channels.bm25 import Bm25Backend, Bm25Chunks, Bm25Index, InProcessBm25
from prospecta.channels.fusion import FusedDoc, fuse
from prospecta.channels.graph import GraphExpand
from prospecta.channels.extract import extract_filters, regex_filters
from prospecta.channels.meta import MetadataScope, promote_scope, scope_members
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
    "build_channels", "extract_filters", "regex_filters", "MetadataScope",
    "promote_scope", "scope_members", "validate_channel_config", "AnticipatedQuestions", "DenseChunks",
    "GraphExpand", "Bm25Backend", "Bm25Chunks", "Bm25Index", "InProcessBm25",
]
