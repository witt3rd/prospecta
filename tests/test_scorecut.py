"""Score-based pools: relative cut, window batching, paged fetch, no silent drops."""
import json
import logging

import pytest

from prospecta._scorecut import fetch_until_cut, rel_cut, split_batches
from prospecta.channels.base import Candidate
from prospecta.channels.fusion import FusedDoc
from prospecta.stages import Item, SonnetListwise, read_all


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


def _item(i):
    c = Candidate(document_id=f"d{i}", item_id=f"i{i}", source=f"n{i}.md",
                  channel="x", rank=i + 1, score=1.0, evidence="e" * 40)
    return Item(doc=FusedDoc(f"d{i}", f"n{i}.md", 1.0 - i / 100, c), header=f"n{i}.md",
                evidence="e" * 40)


class _Llm:
    def __init__(self, fail_first=0):
        self.fail, self.calls = fail_first, 0

    def __call__(self, messages, *, json_mode=False):
        self.calls += 1
        if self.fail:
            self.fail -= 1
            raise RuntimeError("boom")
        n = messages[0]["content"].count("\n[") + 1
        return json.dumps({"grades": {str(i): (3 if i == n else 0) for i in range(1, n + 1)}})


def test_rerank_over_the_window_is_map_reduced_and_keeps_every_note():
    items = [_item(i) for i in range(6)]
    llm = _Llm()
    out = SonnetListwise(llm, context_tokens=4_000 + 40).rerank("q", items, [])
    assert llm.calls > 1 and out.record["batches"] == llm.calls
    assert sorted(out.order) == list(range(6)) and len(out.grades) == 6


def test_a_failed_batch_is_retried_then_surfaced_with_the_full_notes(caplog):
    items = [_item(i) for i in range(3)]
    assert SonnetListwise(_Llm(fail_first=1)).rerank("q", items, []).order
    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError):
        SonnetListwise(_Llm(fail_first=2)).rerank("the question", items, [])
    msg = caplog.records[-1].getMessage()
    assert "the question" in msg and "d0" in msg and "d2" in msg


def test_reader_reads_every_batch_and_unions_follow_ups():
    class R:
        seen = []

        def read(self, q, items, mfu, calls):
            self.seen.append(len(items))
            return False, [f"fu{len(self.seen)}"]
    r = R()
    ok, fu = read_all(r, "q", [_item(i) for i in range(4)], [], context_tokens=4_000 + 25)
    assert not ok and sum(r.seen) == 4 and len(fu) == len(r.seen) > 1


def test_jev_reader_requests_stay_under_its_byte_limit():
    from prospecta.stages import JEV_INPUT_BYTES, JevReader
    sizes = []

    def transport(req, timeout):
        sizes.append(len(json.dumps(req).encode()))
        return {"answers": {k: {"score": 3} for k in req["questions"]}}
    items = []
    for i in range(20):
        it = _item(i)
        items.append(Item(doc=it.doc, header=it.header, evidence="e" * 8_000))
    ok, _ = read_all(JevReader(transport), "q", items, [])
    assert ok and len(sizes) > 1 and max(sizes) <= JEV_INPUT_BYTES


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
