"""Score-based pools: relative cut, window batching, paged fetch, no silent drops."""
from prospecta._scorecut import fetch_until_cut, rel_cut, split_batches


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


def test_fetch_until_cut_grows_until_the_cut_is_reached():
    scores = [1.0 - i * 0.01 for i in range(300)]
    asked = []

    def fetch(n):
        asked.append(n)
        return scores[:n]
    got = fetch_until_cut(fetch, float, 0.5, page=16)
    assert got == [s for s in scores if s >= 0.5] and asked == [16, 32, 64]
    assert fetch_until_cut(lambda n: scores[:n], float, 0, page=100) == scores


def test_semantic_channel_reaches_qualifying_items_beyond_hnsw_ef_search(fresh_db):
    import psycopg
    from prospecta.channels.base import QueryPlan, RecallState
    from prospecta.channels.semantic import DenseChunks, _scan_everything
    from prospecta.db.queries import ensure_kind_hnsw_indexes
    from prospecta.memory import Memory
    from tests._stub_embedder import EMBED_DIM, stub_embed, stub_llm

    m = Memory(database_url=fresh_db, bank_id="b", llm=stub_llm, embed=stub_embed)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    total = 150
    with psycopg.connect(fresh_db, autocommit=True) as conn:
        ensure_kind_hnsw_indexes(conn, "b", EMBED_DIM)
        for i in range(total):
            vec = [1.0, 0.001 * i] + [0.0] * (EMBED_DIM - 2)
            lit = "[" + ",".join(map(str, vec)) + "]"
            doc = conn.execute(
                "INSERT INTO documents (bank_id, source, content_hash, original_text) "
                "VALUES ('b', %s, %s, 't') RETURNING id", (f"n{i}.md", f"h{i}")).fetchone()[0]
            conn.execute(
                "INSERT INTO memory_items (bank_id, document_id, content, original_chunk,"
                " embedding, kind, ordinal, char_start, char_end) "
                "VALUES ('b', %s, 'c', 'c', %s::vector, 'chunk', 0, 0, 1)", (doc, lit))
    q = [1.0] + [0.0] * (EMBED_DIM - 1)
    with psycopg.connect(fresh_db) as conn:
        conn.execute("SET enable_seqscan = off")
        state = RecallState(conn=conn, bank_id="b", embed=lambda texts: [q for _ in texts])
        got = DenseChunks(min_rel=0.5).retrieve(QueryPlan(text="x"), state)
    sql = ("SELECT id FROM memory_items WHERE bank_id = 'b' AND kind = 'chunk' "
           "ORDER BY embedding::vector(%d) <=> %%s::vector LIMIT 100" % EMBED_DIM)
    lit = "[" + ",".join(map(str, q)) + "]"
    with psycopg.connect(fresh_db) as conn:
        conn.execute("SET enable_seqscan = off")
        assert len(conn.execute(sql, (lit,)).fetchall()) == 40
        conn.rollback()
        with conn.transaction(), conn.cursor() as cur:
            _scan_everything(cur)
            assert len(cur.execute(sql, (lit,)).fetchall()) == 100
    m.close()
    assert len(got) == total


def test_new_defaults_keep_strict_gold_notes_the_old_ones_drop():
    """Synthetic corpus: gold notes score low against the best (cosine 0.5x, BM25 0.1x,
    fused 0.2x). The old defaults (channels 0.6 / 0.15, pool 0.4) drop them; the new
    ones (no channel cut, pool 0.15) keep them."""
    from prospecta._scorecut import CHANNEL_MIN_REL, POOL_MIN_REL
    from prospecta.channels.semantic import DenseChunks
    cos = [("best", 0.9), ("gold", 0.45), ("other", 0.8)]
    bm = [("best", 20.0), ("gold", 2.0), ("other", 10.0)]
    fused = [("best", 1.0), ("gold", 0.2), ("other", 0.5)]

    def keep(rows, rel):
        return {n for n, _ in rel_cut(rows, lambda r: r[1], rel)}
    assert "gold" not in keep(cos, 0.6) and "gold" not in keep(bm, 0.15)
    assert "gold" not in keep(fused, 0.4)
    assert CHANNEL_MIN_REL == 0 and keep(cos, CHANNEL_MIN_REL) == {"best", "gold", "other"}
    assert keep(bm, CHANNEL_MIN_REL) == {"best", "gold", "other"}
    assert POOL_MIN_REL == 0.15 and "gold" in keep(fused, POOL_MIN_REL)
    assert "gold" in keep(fused, 0.05)
    assert DenseChunks().min_rel == 0
    assert DenseChunks(min_rel=0.6).min_rel == 0.6      # explicit override stays


def test_default_rerank_pool_cut_and_report_line():
    from prospecta._scorecut import POOL_MIN_REL
    from prospecta.evaluation import format_report
    assert POOL_MIN_REL == 0.15
    rep = {"bank": "b", "n_questions": 0, "legacy_bank_scored_with_defaults": False,
           "full": {"stages": {}, "summary": {
               "gold": {"hit1": 1.0, "hit10": 1.0, "mrr": 1.0, "cover10": 1.0},
               "gold2": {"n": 0}, "by_class": {}}},
           "recall_config": {}, "depth": "standard", "rerank_min_rel": 0.15,
           "channel_config": [{"name": "dense_chunk"}]}
    text = format_report(rep)
    assert ("cuts: channels dense_chunk 0.0 (0 = no cut); "
            "rerank pool >= 0.15 x best fused score (depth: standard)") in text
