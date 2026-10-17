"""Score-based pools: relative cut, window batching, no silent drops."""
from prospecta._scorecut import rel_cut, split_batches


def test_rel_cut_keeps_everything_at_or_above_the_relative_score():
    xs = [10, 9, 5, 4, 3]
    assert rel_cut(xs, float, 0.5) == [10, 9, 5]
    assert rel_cut(xs, float, 0) == xs and rel_cut([], float, 0.5) == []
    assert rel_cut([0, 0], float, 0.5) == [0, 0]      # no basis for a cut


def test_split_batches_never_cuts_or_drops():
    items = ["a" * 400, "b" * 400, "c" * 4000, "d" * 40]
    bs = split_batches(items, str, budget_tokens=250)
    assert [x for b in bs for x in b] == items and len(bs) == 3
    assert split_batches([], str) == []


def test_the_pool_cut_keeps_strict_gold_notes_the_old_cuts_dropped():
    """Synthetic corpus: a gold note scores low against the best (fused 0.2x). The old cuts
    (pool 0.4) dropped it; the rerank pool cut (0.15, deep 0.05) keeps it."""
    from prospecta._scorecut import POOL_MIN_REL
    fused = [("best", 1.0), ("gold", 0.2), ("other", 0.5)]

    def keep(rows, rel):
        return {n for n, _ in rel_cut(rows, lambda r: r[1], rel)}
    assert "gold" not in keep(fused, 0.4)
    assert POOL_MIN_REL == 0.15 and "gold" in keep(fused, POOL_MIN_REL)
    assert "gold" in keep(fused, 0.05)


def test_default_rerank_pool_cut_and_report_line():
    from prospecta._scorecut import POOL_MIN_REL
    from prospecta.evaluation import format_report
    assert POOL_MIN_REL == 0.15
    rep = {"bank": "b", "n_questions": 0,
           "full": {"stages": {}, "summary": {
               "gold": {"hit1": 1.0, "hit10": 1.0, "mrr": 1.0, "cover10": 1.0},
               "gold2": {"n": 0}, "by_class": {}}},
           "recall_config": {}, "depth": "standard", "rerank_min_rel": 0.15,
           "channel_config": [{"name": "hybrid_chunk"}]}
    text = format_report(rep)
    assert "cut: rerank pool >= 0.15 x best fused score (depth: standard)" in text
