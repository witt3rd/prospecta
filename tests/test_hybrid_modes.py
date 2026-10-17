"""Stage 1 through Memory.search: the RRF constant, the metadata filter."""
from __future__ import annotations

from pathlib import Path


def _seed(tmp_path: Path) -> None:
    (tmp_path / "alpha.md").write_text(
        "# Alpha\n\nThe forge rings true. Smiths work the metal at the bench.\n"
    )
    (tmp_path / "beta.md").write_text(
        "# Beta\n\nKelly bakes bread on Sunday. The kitchen smells warm.\n"
    )
    (tmp_path / "gamma.md").write_text(
        "# Gamma\n\nForge and Cookie share a bond. Brothers at the workbench.\n"
    )


def test_hybrid_default_rrf_k_60(memory_with_bank, tmp_path):
    _seed(tmp_path)
    memory_with_bank.index_directory(tmp_path)
    results = memory_with_bank.search("forge metal", limit=5)
    assert len(results) > 0
    # hybrid_chunk is the one channel: the fused score is weight / (k + rank) <= 1/61 with k = 60
    for r in results:
        assert 0.0 <= r.scores["rrf"] <= 1.0 / 61.0 + 1e-9
        assert set(r.scores) >= {"hybrid_chunk", "rrf"}


def test_hybrid_rrf_k_override(memory_with_bank, tmp_path):
    _seed(tmp_path)
    memory_with_bank.index_directory(tmp_path)
    r60 = memory_with_bank.search("forge", limit=5, rrf_k=60)
    r1 = memory_with_bank.search("forge", limit=5, rrf_k=1)
    assert r60 and r1
    # k=1 produces much larger reciprocal scores than k=60
    assert max(r.scores["rrf"] for r in r1) > max(r.scores["rrf"] for r in r60)


def test_a_query_with_no_lexical_match_still_returns_the_nearest_chunks(memory_with_bank, tmp_path):
    _seed(tmp_path)
    memory_with_bank.index_directory(tmp_path)
    results = memory_with_bank.search("xyzzy_nonsense_token_qwertyuiop", limit=5)
    assert results                                  # the semantic retriever still fills its slots
    assert all(r.match_kind == "semantic" for r in results)


def test_metadata_filter_json_containment(memory_with_bank, tmp_path):
    """metadata_filter is opaque JSONB containment — caller decides shape."""
    _seed(tmp_path)
    memory_with_bank.index_directory(tmp_path)

    all_results = memory_with_bank.search("forge", limit=20)
    none_results = memory_with_bank.search(
        "forge", limit=20,
        metadata_filter={"nonexistent_key": "nonexistent_value"},
    )
    assert all_results and len(all_results) >= len(none_results)
    assert len(none_results) == 0
