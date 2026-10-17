"""`prospecta eval`: questions.md parsing, metrics, the Stage 2 ablation, verdicts,
cost and latency per stage, `ask` + judge. LLM and embedder are stubs."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from prospecta import evaluation as ev
from prospecta.channels import effective_channel_config
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


HYBRID = effective_channel_config([])


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


def test_verdict_ablation_margin():
    full = _summary(0.8, 0.9, 0.85, 0.9, multi=0.6)
    drop = _summary(0.7, 0.9, 0.85, 0.9, multi=0.6)
    v = ev.verdict(full, drop, "stage_off")
    assert v["earns"] and v["because"] == ["gold.hit1"]
    tiny = _summary(0.78, 0.9, 0.84, 0.9, multi=0.6)
    assert not ev.verdict(full, tiny, "stage_off")["earns"]
    cov = _summary(0.8, 0.9, 0.85, 0.9, multi=0.5)
    assert ev.verdict(full, cov, "stage_off")["because"] == ["multinote.cover10"]
    # gold-level cover@10 alone does not count (only the multi-note questions do)
    gcov = _summary(0.8, 0.9, 0.85, 0.7, multi=0.6)
    assert not ev.verdict(full, gcov, "stage_off")["earns"]
    better = _summary(0.8, 0.95, 0.85, 0.9, multi=0.6)
    assert not ev.verdict(full, better, "stage_off")["earns"]   # improving is not "earning"


def test_ablation_variants_are_stage_2_off_and_nothing_else():
    rc = {"rerank": {"enabled": True, "stage": "cross_encoder"}}
    vs = {v["name"]: v for v in ev.ablation_variants(HYBRID, rc)}
    assert set(vs) == {"-rerank"}
    assert vs["-rerank"]["stages"]["rerank"]["enabled"] is False
    assert rc["rerank"]["enabled"] is True   # the bank's own config is not mutated
    assert ev.ablation_variants(HYBRID, {}) == []                                   # no stage, nothing to ablate
    assert ev.ablation_variants(HYBRID, {"rerank": {"enabled": False}}) == []


# ------------------------------------------------------------- integration

class ReversingRerank:
    """A rerank transport that scores the documents of a request in the reverse of their order."""

    def __call__(self, req, timeout):
        return {"model": req["model"], "usage": {"total_tokens": 100, "cost": 0.002},
                "results": [{"index": i, "relevance_score": 0.2 + 0.1 * i}
                            for i in range(len(req["documents"]))]}


class EvalLLM:
    """Judge: correct iff answer names alpha. `ask`: answers from the first excerpt, [1]."""

    def __init__(self):
        self.calls = []

    def __call__(self, messages, *, json_mode=False):
        text = messages[-1]["content"]
        self.calls.append(text)
        if json_mode and "grading an answer" in text:
            ans = text.split("## Answer under test")[1]
            return LLMResult(json.dumps({"correct": "alpha" in ans}), cost_usd=0.0005)
        if json_mode:
            return json.dumps({"queries": [{"text": "cat sat"}]})
        return LLMResult("alpha cat sat [1]", model="stub-sonnet", cost_usd=0.001)


@pytest.fixture
def mem(fresh_db):
    llm = EvalLLM()
    m = Memory(database_url=fresh_db, bank_id="b", llm=llm, embed=stub_embed, rerank_transport=ReversingRerank())
    m.create_bank("b", embedding_dim=EMBED_DIM)
    m.set_recall_config({})        # Stage 1's order, until a test sets a stage
    for src, text in NOTES.items():
        m.retain(text, source=src)
    yield m
    m.close()


def test_run_eval_full_ablate_stage_costs_and_earning(mem):
    qs = ev.parse_questions(QUESTIONS_MD)
    base, _ = ev.retrieve(mem, qs[0].text, HYBRID, {})
    assert base[0] == "alpha.md"
    # the reversing reranker moves the worst fused note to the top: make it the gold
    worst = base[-1]
    qs = [ev.Question("W", qs[0].text, [worst], [], "single", None)]
    mem.set_recall_config({"rerank": {"enabled": True, "stage": "cross_encoder"}})
    rep = ev.run_eval(mem, qs, ablate=True)
    assert rep["full"]["summary"]["gold"]["hit1"] == 1.0
    stages = rep["full"]["stages"]
    assert stages["stage:cross_encoder_rerank"]["cost_usd"] == pytest.approx(0.002)
    assert "channel:hybrid_chunk" in stages and stages["total"]["mean_latency_ms"] >= 0
    abl = {a["name"]: a for a in rep["ablations"]}
    assert set(abl) == {"-rerank"}
    assert abl["-rerank"]["summary"]["gold"]["hit1"] == 0.0
    assert abl["-rerank"]["earns"] and "gold.hit1" in abl["-rerank"]["because"]
    # the run wrote no config to the bank
    assert mem.search(qs[0].text, limit=3)
    text = ev.format_report(rep)
    assert "-rerank" in text and "EARNS" in text and "stage:cross_encoder_rerank" in text
    json.loads(ev.report_json(rep))


def test_run_eval_scores_a_bank_with_no_channel_config_through_hybrid_chunk(mem):
    mem.set_channel_config([])
    rep = ev.run_eval(mem, ev.parse_questions(QUESTIONS_MD))
    assert [e["name"] for e in rep["channel_config"]] == ["hybrid_chunk"]
    assert rep["full"]["summary"]["n"] == 2 and "ablations" not in rep
    assert "default channels" not in ev.format_report(rep)


def test_run_synthesis_runs_ask_and_the_judge_accounts_cost(mem):
    qs = ev.parse_questions(QUESTIONS_MD)
    rep = ev.run_eval(mem, qs, synth=True, judge=True)
    s = rep["synthesis"]
    assert s["n"] == 2 and s["judged"] == 1 and s["answer_correct"] == 1
    assert s["answer_correct_rate"] == 1.0
    assert s["judge_cost_usd"] == pytest.approx(0.0005)
    assert s["cited_gold_rate"] == 1.0 and s["not_in_memory"] == 0
    assert "answer-correct 1/1" in ev.format_report(rep)


def test_cli_eval_prints_report_and_json(mem, tmp_path, monkeypatch, capsys):
    from prospecta.cli import _common
    from prospecta.cli.evaluate import cmd_eval
    qf = tmp_path / "questions.md"
    qf.write_text(QUESTIONS_MD)
    monkeypatch.setattr(_common, "make_memory", lambda args: mem)
    mem.set_recall_config({"rerank": {"enabled": True, "stage": "cross_encoder"}})
    args = SimpleNamespace(questions=qf, ablate=True, synth=False, judge=False, json=False)
    assert cmd_eval(args) == 0
    out = capsys.readouterr().out
    assert "full:" in out and "-rerank" in out
    args.json = True
    assert cmd_eval(args) == 0
    assert json.loads(capsys.readouterr().out)["n_questions"] == 2
    qf.write_text("## Q\nx\n")
    assert cmd_eval(args) == 2


QMD = """# Questions

### Q001
- class: multinote
- question: What did Kelly say
  about the trip?
- gold: kelly-trip.md, hotel.md

### Q002
class: single
question: Where did we stay?
gold: hotel.md
gold2: a/b/hotel.md
answer: Harbour Inn
"""


def test_parse_questions_md_qblocks():
    qs = ev.parse_questions(QMD)
    assert [q.id for q in qs] == ["Q001", "Q002"]
    assert qs[0].text == "What did Kelly say about the trip?" and qs[0].cls == "multinote"
    assert qs[0].gold == ["kelly-trip.md", "hotel.md"]
    assert qs[1].gold2 == ["a/b/hotel.md"] and qs[1].answer == "Harbour Inn"
    with pytest.raises(ValueError, match="no `gold:`"):
        ev.parse_questions("### Q1\nquestion: x\n")


def test_eval_runs_production_path_and_reports_channel_errors(mem):
    _, st = ev.retrieve(mem, "what did alpha say", HYBRID, {})
    assert "channel:hybrid_chunk" in st
    bad = ev.run_eval(mem, ev.parse_questions(QUESTIONS_MD))
    bad["full"]["stages"]["channel:hybrid_chunk"].update(errors=2, last_error="boom: dim")
    assert "ERROR x2: boom: dim" in ev.format_report(bad)


# ------------------------------------------------ abstention beside false abstention

def test_abstention_counts_pair_false_abstain_with_ground_truth():
    rows = [
        {"id": "a", "abstained": True, "has_gold": True},    # wrong: the note exists
        {"id": "b", "abstained": True, "has_gold": False},   # right: nothing to find
        {"id": "c", "abstained": False, "has_gold": True},
        {"id": "d", "abstained": True, "has_gold": None},    # no ground truth: unjudged
    ]
    c = ev.abstention_counts(rows)
    assert c == {"abstained": 3, "false_abstain": 1, "ground_truth": 3}
    assert ev.abstention_counts([{"abstained": True, "has_gold": None}]) == {
        "abstained": 1, "false_abstain": None, "ground_truth": 0}


def test_format_abstention_always_shows_both_numbers():
    assert ev.format_abstention({"abstained": 3, "false_abstain": 1, "ground_truth": 3}) == \
        "abstained 3, false abstain 1 (of 3 questions with ground truth)"
    assert "false abstain n/a" in ev.format_abstention(
        {"abstained": 2, "false_abstain": None, "ground_truth": 0})


def test_summary_and_synthesis_report_carry_both_numbers(mem):
    qs = ev.parse_questions(QUESTIONS_MD)
    rep = ev.run_eval(mem, qs, synth=True)
    s = rep["synthesis"]
    assert {"not_in_memory", "false_abstain"} <= set(s)
    assert s["false_abstain"] == s["not_in_memory"]       # every question here has a gold note
    assert {"abstained", "false_abstain"} <= set(rep["full"]["summary"]["abstention"])
    text = ev.format_report(rep)
    assert text.count("false abstain") >= 2               # retrieval and synthesis lines
