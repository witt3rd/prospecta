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
