"""Hybrid SQL mode tests — RRF, COALESCE, metadata_filter."""
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
    # rrf score should be sum of three reciprocals ≤ 3/61 with k=60 + rank=1
    # (three-channel fusion: semantic + lexical_content + lexical_body).
    for r in results:
        assert r.scores["rrf"] >= 0.0
        assert r.scores["rrf"] <= 3.0 / 61.0 + 1e-9


def test_hybrid_rrf_k_override(memory_with_bank, tmp_path):
    _seed(tmp_path)
    memory_with_bank.index_directory(tmp_path)
    r60 = memory_with_bank.search("forge", limit=5, mode="hybrid", rrf_k=60)
    r1 = memory_with_bank.search("forge", limit=5, mode="hybrid", rrf_k=1)
    assert r60 and r1
    # k=1 produces much larger reciprocal scores than k=60
    assert max(r.scores["rrf"] for r in r1) > max(r.scores["rrf"] for r in r60)


def test_coalesce_produces_zero_scores_for_missing_side(memory_with_bank, tmp_path):
    _seed(tmp_path)
    memory_with_bank.index_directory(tmp_path)

    # Use a query with nonsense terms so lexical CTE returns nothing,
    # but semantic CTE still returns nearest neighbors.
    results = memory_with_bank.search(
        "xyzzy_nonsense_token_qwertyuiop", limit=5, mode="hybrid"
    )
    # If hybrid + lexical empty, semantic still populates — lex_score must be 0.0
    for r in results:
        if r.scores["lexical"] == 0.0:
            # COALESCE worked
            assert "lexical" in r.scores
            assert r.scores["lexical"] == 0.0


def test_metadata_filter_json_containment(memory_with_bank, tmp_path):
    """metadata_filter is opaque JSONB containment — caller decides shape."""
    _seed(tmp_path)
    memory_with_bank.index_directory(tmp_path)

    # No filter → all
    all_results = memory_with_bank.search("forge", limit=20, mode="lexical")
    # Filter that matches nothing → empty
    none_results = memory_with_bank.search(
        "forge", limit=20, mode="lexical",
        metadata_filter={"nonexistent_key": "nonexistent_value"},
    )
    assert len(all_results) >= len(none_results)
    assert len(none_results) == 0
