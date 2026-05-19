"""Tests for prospecta._types public API surface."""
import dataclasses

import pytest

from prospecta._types import (
    EmbedCallable,
    LLMCallable,
    Query,
    RAGResult,
    RecalledMemory,
    Tracer,
)


def test_query_constructs_with_defaults():
    q = Query(text="hello")
    assert q.text == "hello"
    assert q.metadata_filter is None
    assert q.tags is None
    assert q.tags_match == "any"


def test_query_is_frozen():
    q = Query(text="hi")
    with pytest.raises(dataclasses.FrozenInstanceError):
        q.text = "changed"  # type: ignore[misc]


def test_recalled_memory_full_construction():
    rm = RecalledMemory(
        content="What is Kelly's birthday?",
        original_chunk="Kelly was born 1971-04-30...",
        source="conversation:2026-05-18",
        score=0.85,
        scores={"semantic": 0.6, "lexical": 0.4, "rrf": 0.85},
        metadata={"tags": ["kelly"]},
        bank_id="forge",
        document_id="abc-123",
    )
    assert rm.content.startswith("What")
    assert rm.scores["rrf"] == 0.85


def test_rag_result_construction():
    q = Query(text="q")
    rm = RecalledMemory(
        content="c", original_chunk="o", source="s", score=0.1,
        scores={"semantic": 0.1, "lexical": 0.0, "rrf": 0.1},
        metadata={}, bank_id="b", document_id="d",
    )
    result = RAGResult(
        synthesis="the synthesis",
        sources=[rm],
        queries=[q],
        queries_to_results={"q": [rm]},
    )
    assert result.synthesis == "the synthesis"
    assert len(result.sources) == 1
    assert result.queries_to_results["q"][0].content == "c"


def test_llm_callable_protocol():
    def my_llm(messages: list[dict], *, json_mode: bool = False) -> str:
        return "response"
    llm: LLMCallable = my_llm
    assert llm([{"role": "user", "content": "hi"}]) == "response"
    assert llm([{"role": "user", "content": "hi"}], json_mode=True) == "response"


def test_embed_callable_protocol():
    def my_embed(texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]
    embed: EmbedCallable = my_embed
    result = embed(["hello", "world"])
    assert len(result) == 2
    assert len(result[0]) == 3


def test_tracer_callable():
    events = []
    def tracer(event: str, payload: dict) -> None:
        events.append((event, payload))
    t: Tracer = tracer
    t("retain", {"source": "test", "duration_ms": 5.0})
    assert events == [("retain", {"source": "test", "duration_ms": 5.0})]
