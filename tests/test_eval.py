"""`prospecta eval`: questions.md parsing, metrics, ablations, verdicts,
cost and latency per stage, synthesis + judge. LLM and embedder are stubs."""
from __future__ import annotations

import json
import re
from types import SimpleNamespace

import psycopg
import pytest

from prospecta import evaluation as ev
from prospecta.channels import DEFAULT_CHANNEL_CONFIG
from prospecta.memory import Memory
from prospecta.stages import LLMResult
from tests._stub_embedder import EMBED_DIM, stub_embed

QUESTIONS_MD = """# my questions
ignored preamble

## Q001 | single
What sat on the warm mat?
gold: alpha.md
gold2: notes/alpha

## Q002 | multinote
Which cats sat?
gold: alpha.md, bravo.md, charlie.md
answer: alpha, bravo and charlie
"""

NOTES = {
    "alpha.md": "alpha cat sat on the warm mat",
    "bravo.md": "bravo cat sat near the window",
    "charlie.md": "charlie cat sat under the table",
    "delta.md": "delta dog barked at the moon",
    "echo.md": "echo revenue grew twelve percent",
}


def vec(text):
    return "[" + ",".join(map(str, stub_embed([text])[0])) + "]"


# ------------------------------------------------------------------ parsing

def test_parse_questions_format():
    qs = ev.parse_questions(QUESTIONS_MD)
    assert [q.id for q in qs] == ["Q001", "Q002"]
    assert qs[0].cls == "single" and qs[0].text == "What sat on the warm mat?"
    assert qs[0].gold == ["alpha.md"] and qs[0].gold2 == ["notes/alpha"]
    assert qs[1].gold == ["alpha.md", "bravo.md", "charlie.md"]
    assert qs[1].answer == "alpha, bravo and charlie"


@pytest.mark.parametrize("bad,msg", [
    ("nothing here", "no questions"),
    ("## Q1\nwhat\n", "no `gold:`"),
    ("## Q1\ngold: a.md\n", "no question text"),
    ("## Q1\nx\ngold: a\n## Q1\ny\ngold: b\n", "duplicate"),
])
def test_parse_questions_errors(bad, msg):
    with pytest.raises(ValueError, match=msg):
        ev.parse_questions(bad)


# ------------------------------------------------------------------ metrics

def test_score_ranking_metrics_and_name_normalisation():
    s = ev.score_ranking(["x.md", "Notes/B.md", "a.md"], ["a", "b.md", "zzz.md"])
    assert s["first_rank"] == 2 and s["hit1"] == 0 and s["hit10"] == 1
    assert s["mrr"] == pytest.approx(0.5)
    assert s["cover10"] == pytest.approx(2 / 3)
    miss = ev.score_ranking(["x.md"], ["a.md"])
    assert miss["hit10"] == 0 and miss["mrr"] == 0 and miss["first_rank"] is None
    late = ev.score_ranking([f"n{i}.md" for i in range(10)] + ["a.md"], ["a.md"])
    assert late["hit10"] == 0 and late["mrr"] == pytest.approx(1 / 11)


def _summary(h1, h10, mrr, cov, multi=None, g2=None):
    g = {"hit1": h1, "hit10": h10, "mrr": mrr, "cover10": cov, "n": 10}
    z = {"hit1": 0, "hit10": 0, "mrr": 0, "cover10": 0, "n": 0}
    return {"gold": g, "gold2": g2 or z, "multinote_cover10": multi, "n": 10, "by_class": {}}


def test_verdict_ablation_and_addition_margin():
    full = _summary(0.8, 0.9, 0.85, 0.9, multi=0.6)
    drop = _summary(0.7, 0.9, 0.85, 0.9, multi=0.6)
    v = ev.verdict(full, drop, "channel_off")
    assert v["earns"] and v["because"] == ["gold.hit1"]
    tiny = _summary(0.78, 0.9, 0.84, 0.9, multi=0.6)
    assert not ev.verdict(full, tiny, "channel_off")["earns"]
    cov = _summary(0.8, 0.9, 0.85, 0.9, multi=0.5)
    assert ev.verdict(full, cov, "channel_off")["because"] == ["multinote.cover10"]
    # gold-level cover@10 alone does not count (only the multi-note questions do)
    gcov = _summary(0.8, 0.9, 0.85, 0.7, multi=0.6)
    assert not ev.verdict(full, gcov, "channel_off")["earns"]
    better = _summary(0.8, 0.95, 0.85, 0.9, multi=0.6)
    assert ev.verdict(full, better, "channel_on")["earns"]
    assert not ev.verdict(full, better, "channel_off")["earns"]   # improving is not "earning"


def test_ablation_variants():
    rc = {"rerank": {"enabled": True}, "gate": {"enabled": True}, "reader": {"enabled": False}}
    vs = {v["name"]: v for v in ev.ablation_variants(DEFAULT_CHANNEL_CONFIG, rc)}
    assert set(vs) == {"-dense_chunk", "-question", "+bm25", "-rerank", "-gate"}
    off = [e for e in vs["-question"]["channels"] if e["name"] == "question"][0]
    assert off["enabled"] is False
    assert [e["name"] for e in DEFAULT_CHANNEL_CONFIG if not e.get("enabled", True)] == []
    assert vs["-rerank"]["stages"]["gate"]["enabled"] is False
    assert rc["gate"]["enabled"] is True   # the bank's own config is not mutated
    dark = [dict(e) for e in DEFAULT_CHANNEL_CONFIG]
    dark[1]["weight"] = 0
    assert "-question" not in {v["name"] for v in ev.ablation_variants(dark, {})}
    assert "+question" in {v["name"] for v in ev.ablation_variants(dark, {})}


# ------------------------------------------------------------- integration

class EvalLLM:
    """Rerank: reverse the candidate order. Judge: correct iff answer names alpha.
    Synthesis: cites [alpha.md]."""

    def __init__(self):
        self.calls = []

    def __call__(self, messages, *, json_mode=False):
        text = messages[-1]["content"]
        self.calls.append(text)
        if json_mode and "ranking notes" in text:
            n = len(re.findall(r"^\[(\d+)\] ", text, re.M))
            rev = list(range(n, 0, -1))
            return LLMResult(json.dumps({"grades": {str(i): 3 for i in rev}, "ranking": rev}),
                             model="stub-sonnet", tokens_in=100, tokens_out=10, cost_usd=0.002)
        if json_mode and "grading an answer" in text:
            ans = text.split("## Answer under test")[1]
            return LLMResult(json.dumps({"correct": "alpha" in ans}), cost_usd=0.0005)
        if json_mode:
            return json.dumps({"queries": [{"text": "cat sat"}]})
        return LLMResult("alpha cat sat [alpha.md]", model="stub-sonnet", cost_usd=0.001)


@pytest.fixture
def mem(fresh_db):
    llm = EvalLLM()
    m = Memory(database_url=fresh_db, bank_id="b", llm=llm, embed=stub_embed)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    m.set_channel_config(DEFAULT_CHANNEL_CONFIG)
    for src, text in NOTES.items():
        m.retain(text, source=src, index_text=text)
    with psycopg.connect(fresh_db) as conn:
        for src, text in NOTES.items():
            conn.execute(
                "INSERT INTO memory_items (bank_id, document_id, content, original_chunk,"
                " embedding, kind, ordinal, char_start, char_end) "
                "SELECT 'b', d.id, %s, %s, %s::vector, 'chunk', 0, 0, %s FROM documents d "
                "WHERE d.source = %s", (text, text, vec(text), len(text), src))
        conn.commit()
    yield m
    m.close()


def test_run_eval_full_ablate_stage_costs_and_earning(mem):
    qs = ev.parse_questions(QUESTIONS_MD)
    base, _ = ev.retrieve(mem, qs[0].text, DEFAULT_CHANNEL_CONFIG, {})
    assert base[0] == "alpha.md"
    # the reversing reranker moves the worst fused note to the top: make it the gold
    worst = base[-1]
    qs = [ev.Question("W", qs[0].text, [worst], [], "single", None)]
    mem.set_recall_config({"rerank": {"enabled": True, "stage": "sonnet_listwise"}})
    rep = ev.run_eval(mem, qs, ablate=True)
    assert rep["full"]["summary"]["gold"]["hit1"] == 1.0
    stages = rep["full"]["stages"]
    assert stages["stage:rerank_listwise"]["cost_usd"] == pytest.approx(0.002)
    assert "channel:dense_chunk" in stages and stages["total"]["mean_latency_ms"] >= 0
    abl = {a["name"]: a for a in rep["ablations"]}
    assert set(abl) == {"-dense_chunk", "-question", "+bm25", "-rerank"}
    assert abl["-rerank"]["summary"]["gold"]["hit1"] == 0.0
    assert abl["-rerank"]["earns"] and "gold.hit1" in abl["-rerank"]["because"]
    assert not abl["-question"]["earns"]
    assert abl["+bm25"]["kind"] == "channel_on"
    # the run wrote no config to the bank
    assert mem.search(qs[0].text, limit=3)
    text = ev.format_report(rep)
    assert "-rerank" in text and "EARNS" in text and "stage:rerank_listwise" in text
    json.loads(ev.report_json(rep))


def test_run_eval_legacy_bank_scored_with_defaults(mem):
    mem.set_channel_config([])
    rep = ev.run_eval(mem, ev.parse_questions(QUESTIONS_MD))
    assert rep["legacy_bank_scored_with_defaults"] is True
    assert rep["full"]["summary"]["n"] == 2
    assert "ablations" not in rep and "default channels" in ev.format_report(rep)


def test_run_synthesis_with_judge_accounts_cost(mem):
    qs = ev.parse_questions(QUESTIONS_MD)
    rep = ev.run_eval(mem, qs, synth=True, judge=True)
    s = rep["synthesis"]
    assert s["n"] == 2 and s["judged"] == 1 and s["answer_correct"] == 1
    assert s["answer_correct_rate"] == 1.0
    assert s["synthesis_cost_usd"] == pytest.approx(0.002)
    assert s["judge_cost_usd"] == pytest.approx(0.0005)
    assert s["cited_gold_rate"] == 1.0 and s["not_in_memory"] == 0
    assert "answer-correct 1/1" in ev.format_report(rep)


def test_cli_eval_prints_report_and_json(mem, tmp_path, monkeypatch, capsys):
    from prospecta.cli import _common
    from prospecta.cli.evaluate import cmd_eval
    qf = tmp_path / "questions.md"
    qf.write_text(QUESTIONS_MD)
    monkeypatch.setattr(_common, "make_memory", lambda args: mem)
    args = SimpleNamespace(questions=qf, ablate=True, synth=False, judge=False, json=False)
    assert cmd_eval(args) == 0
    out = capsys.readouterr().out
    assert "full:" in out and "+bm25" in out
    args.json = True
    assert cmd_eval(args) == 0
    assert json.loads(capsys.readouterr().out)["n_questions"] == 2
    qf.write_text("## Q\nx\n")
    assert cmd_eval(args) == 2
