"""`prospecta eval`: score a bank's recall on a questions file, ablate its
Stage 2 (the rerank), and report cost and latency per stage, so a stage
earns its place (hybrid retrieval design 8.7 / 8.10).

questions.md format (one block per question; text before the first `##` is
ignored). The project's `### Qnnn` blocks with `class:` / `question:` / `gold:`
lines are also accepted (see docs/eval.md):

    ## Q001 | multinote
    What did Kelly say about the trip, and where did we stay?
    gold: kelly-trip.md, hotel.md
    gold2: kelly-trip.md
    answer: She said it was too expensive; we stayed at the Harbour Inn.

`## <id>` is required, ` | <class>` optional. `gold:` lists the note names
(documents.source; folder and `.md` are ignored when matching) that answer the
question; `gold2:` is an optional second, independently made gold. `answer:` is
optional and only used by the answer-correct judge. Every other line is the
question text.

Metrics, per gold: hit@1, hit@10, MRR (over the whole ranked pool), cover@10
(share of the gold notes inside the top 10). The ablation turns Stage 2 off; the
stage "earns" its place when that costs hit@10, hit@1 or MRR 0.03 or more on
either gold, or cover@10 0.03 or more on questions with two or more gold notes.
"""
from __future__ import annotations

import copy
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from prospecta._template import render_prompt
from prospecta.stages import parse_json_object, read_recall_config

MARGIN = 0.03
TOP = 10
METRICS = ("hit1", "hit10", "mrr", "cover10")


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    gold: list[str]
    gold2: list[str] = field(default_factory=list)
    cls: str = ""
    answer: str | None = None


def _norm(name: str) -> str:
    n = name.strip().replace("\\", "/").rsplit("/", 1)[-1].casefold()
    return n[:-3] if n.endswith(".md") else n


def _names(value: str) -> list[str]:
    return [p.strip() for p in re.split(r"[,;]", value) if p.strip()]


def _parse_qblocks(text: str) -> list[Question]:
    """The project's questions.md: `### Qnnn` blocks with `class:` / `question:` /
    `gold:` (and optional `gold2:` / `answer:`) lines; `question:` may continue
    on the following unlabelled lines."""
    out: list[Question] = []
    for block in re.split(r"^###[ \t]+", text, flags=re.M)[1:]:
        head, _, body = block.partition("\n")
        qid = head.strip().split()[0] if head.strip() else ""
        if not qid:
            raise ValueError("a question block has an empty id after '###'")
        fields: dict[str, str] = {}
        last = None
        for line in body.splitlines():
            m = re.match(r"^[-*]?\s*(gold2|gold|answer|class|question):[ \t]*(.*)$", line.strip(), re.I)
            if m:
                last = m.group(1).lower()
                fields[last] = m.group(2).strip()
            elif line.strip() and last == "question":
                fields["question"] += " " + line.strip()
        if not fields.get("question"):
            raise ValueError(f"question {qid}: no `question:` line")
        gold = _names(fields.get("gold", ""))
        if not gold:
            raise ValueError(f"question {qid}: no `gold:` line")
        out.append(Question(
            id=qid, text=fields["question"], gold=gold, gold2=_names(fields.get("gold2", "")),
            cls=fields.get("class", ""), answer=fields.get("answer") or None))
    return out


def parse_questions(text: str) -> list[Question]:
    if re.search(r"^###[ \t]+Q", text, flags=re.M) and not re.search(r"^##[ \t]+[^#]", text, flags=re.M):
        out = _parse_qblocks(text)
        ids = [q.id for q in out]
        dup = {i for i in ids if ids.count(i) > 1}
        if dup:
            raise ValueError(f"duplicate question ids: {sorted(dup)}")
        if out:
            return out
    out: list[Question] = []
    blocks = re.split(r"^##[ \t]+", text, flags=re.M)[1:]
    for block in blocks:
        head, _, body = block.partition("\n")
        qid, _, cls = head.partition("|")
        qid, cls = qid.strip(), cls.strip()
        if not qid:
            raise ValueError("a question block has an empty id after '##'")
        fields: dict[str, str] = {}
        lines: list[str] = []
        for line in body.splitlines():
            m = re.match(r"^(gold2|gold|answer|class):[ \t]*(.*)$", line.strip(), re.I)
            if m:
                fields[m.group(1).lower()] = m.group(2).strip()
            elif line.strip():
                lines.append(line.strip())
        question = " ".join(lines)
        if not question:
            raise ValueError(f"question {qid}: no question text")
        gold = _names(fields.get("gold", ""))
        if not gold:
            raise ValueError(f"question {qid}: no `gold:` line")
        out.append(Question(
            id=qid, text=question, gold=gold, gold2=_names(fields.get("gold2", "")),
            cls=fields.get("class", cls), answer=fields.get("answer") or None))
    if not out:
        raise ValueError("no questions found (expected '## <id>' or '### Qnnn' blocks)")
    ids = [q.id for q in out]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"duplicate question ids: {sorted(dup)}")
    return out


def load_questions(path: str | Path) -> list[Question]:
    return parse_questions(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- scoring

def score_ranking(ranked: list[str], gold: list[str]) -> dict:
    """Metrics for one question and one gold; `ranked` are note names, best first."""
    g = {_norm(x) for x in gold}
    seen: list[str] = []
    for r in ranked:   # note-level: a note counts once, at its best position
        n = _norm(r)
        if n not in seen:
            seen.append(n)
    first = next((i + 1 for i, n in enumerate(seen) if n in g), None)
    return {
        "hit1": 1.0 if first == 1 else 0.0,
        "hit10": 1.0 if first is not None and first <= TOP else 0.0,
        "mrr": 1.0 / first if first else 0.0,
        "cover10": len(g & set(seen[:TOP])) / len(g),
        "first_rank": first,
    }


def abstention_counts(rows: list[dict]) -> dict:
    """How often the library abstained (found nothing / said "not in memory") and how
    often that was wrong. Each row: `abstained` (bool) and `has_gold` (True: a gold
    note exists, so abstaining is a false abstain; False: nothing to find, so it is
    right; None: no ground truth, unjudged). The two numbers always come together;
    `false_abstain` is None only when no row has ground truth; with ground truth
    and no abstention it is 0."""
    abst = [r for r in rows if r.get("abstained")]
    judged = [r for r in rows if r.get("has_gold") is not None]
    return {"abstained": len(abst),
            "false_abstain": sum(bool(r["abstained"]) and bool(r["has_gold"])
                                 for r in judged) if judged else None,
            "ground_truth": len(judged)}


def format_abstention(c: dict) -> str:
    fa = "n/a" if c["false_abstain"] is None else str(c["false_abstain"])
    return f"abstained {c['abstained']}, false abstain {fa} (of {c['ground_truth']} questions with ground truth)"


def _mean(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def summarize(per_question: list[dict]) -> dict:
    """Aggregate per-question score rows ({gold, gold2?, n_gold, cls}) into means."""
    out: dict = {"n": len(per_question),
                 "abstention": abstention_counts(per_question)}
    for which in ("gold", "gold2"):
        rows = [q[which] for q in per_question if q.get(which)]
        out[which] = {m: _mean(rows, m) for m in METRICS} | {"n": len(rows)}
    multi = [q["gold"] for q in per_question if q["n_gold"] >= 2]
    out["multinote_cover10"] = _mean(multi, "cover10") if multi else None
    by_cls: dict[str, list[dict]] = {}
    for q in per_question:
        if q["cls"]:
            by_cls.setdefault(q["cls"], []).append(q["gold"])
    out["by_class"] = {c: {m: _mean(r, m) for m in METRICS} | {"n": len(r)}
                       for c, r in sorted(by_cls.items())}
    return out


# ----------------------------------------------------------------- variants

def ablation_variants(channel_config: list[dict], recall_cfg: dict) -> list[dict]:
    """The runs to compare with the full config: Stage 2 (the rerank) off, when it is enabled.
    (`hybrid_chunk` is the one channel: without it nothing is retrieved.)"""
    variants: list[dict] = []
    if (recall_cfg.get("rerank") or {}).get("enabled"):
        st = copy.deepcopy(recall_cfg)
        st["rerank"]["enabled"] = False
        variants.append({"name": "-rerank", "kind": "stage_off", "target": "rerank",
                         "channels": channel_config, "stages": st})
    return variants


def verdict(full: dict, variant: dict, kind: str = "stage_off") -> dict:
    """Does the ablation cost at least MARGIN?"""
    deltas: dict[str, float] = {}
    for which in ("gold", "gold2"):
        if full[which]["n"] == 0:
            continue
        for m in ("hit10", "hit1", "mrr", "cover10"):
            deltas[f"{which}.{m}"] = variant[which][m] - full[which][m]
    if full.get("multinote_cover10") is not None:
        deltas["multinote.cover10"] = (variant["multinote_cover10"] or 0.0) - full["multinote_cover10"]
    sign = -1.0                                 # ablation: a drop is what counts
    # cover@10 counts only on the multi-note questions (design 8.7)
    hit = [k for k, d in deltas.items()
           if not (k.endswith(".cover10") and not k.startswith("multinote."))
           and sign * d >= MARGIN - 1e-9]
    earns = bool(hit)
    return {"deltas": deltas, "earns": earns, "because": hit}


# ------------------------------------------------------------------ running

def _calls_by_stage(calls: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for c in calls:
        s = out.setdefault(c["purpose"], {"n": 0, "latency_ms": 0, "cost_usd": 0.0,
                                           "tokens_in": 0, "tokens_out": 0})
        s["n"] += 1
        s["latency_ms"] += int(c.get("duration_ms") or 0)
        s["cost_usd"] += float(c.get("cost_usd") or 0.0)
        s["tokens_in"] += int(c.get("tokens_in") or 0)
        s["tokens_out"] += int(c.get("tokens_out") or 0)
    return out


def retrieve(memory, query: str, channel_config: list[dict], recall_cfg: dict,
             *, pool: int | None = None, rrf_k: int = 60, depth: str | None = None) -> tuple[list[str], dict]:
    """One recall under an explicit config (nothing is written to the bank).
    Returns (note names best first, {stage: {n, latency_ms, cost_usd, ...}})."""
    from prospecta._index import _search_channels
    len_all = sys.maxsize   # the whole score-selected pool is scored; a count only if asked
    stages: dict[str, dict] = {}
    t0 = time.monotonic()
    trace: list = []
    # the production path: Stage 1, then the rerank stage
    recalled = _search_channels(
        memory, query, memory.default_bank_id, channel_config, limit=len_all if pool is None else pool,
        metadata_filter=None, rrf_k=rrf_k, trace=trace, recall_cfg=recall_cfg or {},
        depth=depth)
    tr = trace[0]
    for c in tr["channels"]:
        stages[f"channel:{c['name']}"] = {
            "n": c["n"], "latency_ms": c["latency_ms"], "cost_usd": c["cost_usd"],
            "error": c["error"]}
    stages.update({f"stage:{k}": v for k, v in _calls_by_stage(tr.get("calls") or []).items()})
    if tr.get("fallback_reason"):
        stages["stage:fallback"] = {"n": 1, "reason": tr["fallback_reason"]}
    stages["total"] = {"latency_ms": int((time.monotonic() - t0) * 1000)}
    return [r.source for r in recalled], stages


def _add_stages(total: dict, one: dict) -> None:
    for name, s in one.items():
        t = total.setdefault(name, {"n": 0, "latency_ms": 0, "cost_usd": 0.0, "calls": 0})
        t["calls"] += 1
        t["n"] += s.get("n") or 0
        t["latency_ms"] += s.get("latency_ms") or 0
        t["cost_usd"] += s.get("cost_usd") or 0.0
        if s.get("error") or s.get("reason"):
            t["errors"] = t.get("errors", 0) + 1
            t.setdefault("last_error", str(s.get("error") or s.get("reason")))


def run_variant(memory, questions: list[Question], channel_config: list[dict],
                recall_cfg: dict, *, pool: int | None = None,
                depth: str | None = None) -> dict:
    per_q: list[dict] = []
    stages: dict[str, dict] = {}
    for q in questions:
        ranked, st = retrieve(memory, q.text, channel_config, recall_cfg, pool=pool, depth=depth)
        _add_stages(stages, st)
        row = {"id": q.id, "cls": q.cls, "n_gold": len(q.gold),
               "gold": score_ranking(ranked, q.gold), "top": ranked[:TOP],
               "abstained": not ranked, "has_gold": bool(q.gold) if q.gold else None}
        if q.gold2:
            row["gold2"] = score_ranking(ranked, q.gold2)
        per_q.append(row)
    n = max(len(questions), 1)
    for s in stages.values():
        s["mean_latency_ms"] = s["latency_ms"] / n
    return {"summary": summarize(per_q), "stages": stages, "questions": per_q}


def judge_answer(llm, question: str, expected: str, answer: str) -> tuple[bool, dict]:
    prompt = render_prompt("judge-answer", {
        "question": question, "expected": expected, "answer": answer})
    t0 = time.monotonic()
    out = llm([{"role": "user", "content": prompt}], json_mode=True)
    call = {"cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0}
    if not isinstance(out, str):
        call.update(cost_usd=out.cost_usd or 0.0, tokens_in=out.tokens_in or 0,
                    tokens_out=out.tokens_out or 0)
        out = out.text
    call["latency_ms"] = int((time.monotonic() - t0) * 1000)
    return bool(parse_json_object(out).get("correct")), call


def run_synthesis(memory, questions: list[Question], *, judge: bool = False) -> dict:
    """`ask` (Mode 2B) on every question; with `judge`, answer-correct against `answer:`
    (questions without one are not judged)."""
    rows, correct, judged = [], 0, 0
    cost = {"judge_cost_usd": 0.0}
    lat = {"ask_ms": 0, "judge_ms": 0}
    for q in questions:
        t0 = time.monotonic()
        res = memory.ask(q.text)
        ms = int((time.monotonic() - t0) * 1000)
        lat["ask_ms"] += ms
        cited = [c["source"] for c in res.citations if c.get("known")]
        row = {"id": q.id, "synthesis": res.answer, "latency_ms": ms,
               "citations": cited,
               "unknown_citations": [str(c.get("n")) for c in res.citations if not c.get("known")],
               "cited_gold": bool({_norm(c) for c in cited} & {_norm(g) for g in q.gold})}
        if judge and q.answer:
            ok, call = judge_answer(memory._synth_llm or memory._llm, q.text, q.answer,
                                    res.answer)
            row["correct"] = ok
            judged += 1
            correct += ok
            cost["judge_cost_usd"] += call["cost_usd"]
            lat["judge_ms"] += call["latency_ms"]
        row["abstained"] = row["synthesis"].strip().lower().startswith("not in memory")
        row["has_gold"] = True if q.gold else None
        rows.append(row)
    n = max(len(rows), 1)
    return {
        "n": len(rows), "judged": judged, "answer_correct": correct,
        "synthesis_llm": "synth_llm" if memory._synth_llm else "general llm",
        "answer_correct_rate": (correct / judged) if judged else None,
        "cited_gold_rate": sum(r["cited_gold"] for r in rows) / n,
        "not_in_memory": sum(r["synthesis"].strip().lower().startswith("not in memory")
                             for r in rows),
        "false_abstain": abstention_counts(rows)["false_abstain"],
        **cost, **lat, "mean_ask_ms": lat["ask_ms"] / n,
        "questions": rows,
    }


def run_eval(memory, questions: list[Question], *, ablate: bool = False,
             synth: bool = False, judge: bool = False, pool: int | None = None,
             depth: str | None = None) -> dict:
    """Score the bank as configured; optionally ablate Stage 2 and `ask`. `depth` ('standard' |
    'deep'; None = the bank's recall_config.depth, else standard) sets the rerank
    pool cut; the report records the depth used."""
    from prospecta._scorecut import check_depth, pool_min_rel, resolve_depth
    check_depth(depth)
    with memory._pool.connection() as conn:
        from prospecta.channels import read_channel_config
        channel_config = read_channel_config(conn, memory.default_bank_id)
        recall_cfg = read_recall_config(conn, memory.default_bank_id)
    report: dict = {
        "bank": memory.default_bank_id, "n_questions": len(questions),
        "channel_config": channel_config, "recall_config": recall_cfg,
        "margin": MARGIN,
        "depth": resolve_depth(depth, recall_cfg),
        "rerank_min_rel": pool_min_rel(depth, recall_cfg),
    }
    full = run_variant(memory, questions, channel_config, recall_cfg, pool=pool, depth=depth)
    report["full"] = full
    if ablate:
        report["ablations"] = []
        for v in ablation_variants(channel_config, recall_cfg):
            res = run_variant(memory, questions, v["channels"], v["stages"], pool=pool, depth=depth)
            vd = verdict(full["summary"], res["summary"], v["kind"])
            report["ablations"].append({
                "name": v["name"], "kind": v["kind"], "target": v["target"],
                "summary": res["summary"], "stages": res["stages"], **vd})
    if synth:
        report["synthesis"] = run_synthesis(memory, questions, judge=judge)
    return report


# ------------------------------------------------------------------- report

def _fmt(s: dict) -> str:
    g, g2 = s["gold"], s["gold2"]
    out = (f"hit@1 {g['hit1']:.2f}  hit@10 {g['hit10']:.2f}  MRR {g['mrr']:.2f}  "
           f"cover@10 {g['cover10']:.2f}")
    if g2["n"]:
        out += (f"  | gold2 hit@1 {g2['hit1']:.2f}  hit@10 {g2['hit10']:.2f}  "
                f"MRR {g2['mrr']:.2f}")
    return out


def format_report(r: dict) -> str:
    L = [f"bank {r['bank']}: {r['n_questions']} questions"]
    for name, s in r["full"]["stages"].items():
        if name.startswith("channel:") and s.get("errors"):
            L.append(f"CHANNEL ERROR {name[8:]} x{s['errors']}: {s['last_error']}")
    L.append(f"cut: rerank pool >= {r.get('rerank_min_rel')} x best fused score "
             f"(depth: {r.get('depth', 'standard')})")
    L.append(f"full: {_fmt(r['full']['summary'])}")
    if r["full"]["summary"].get("abstention"):
        L.append("retrieval: " + format_abstention(r["full"]["summary"]["abstention"]))
    for c, s in r["full"]["summary"]["by_class"].items():
        L.append(f"  {c} (n={s['n']}): hit@1 {s['hit1']:.2f}  hit@10 {s['hit10']:.2f}  "
                 f"cover@10 {s['cover10']:.2f}")
    L.append("stages (mean per question):")
    for name, s in r["full"]["stages"].items():
        L.append(f"  {name:<28} {s['mean_latency_ms']:8.1f} ms  ${s['cost_usd'] / max(r['n_questions'], 1):.5f}"
                 + (f"  ERROR x{s['errors']}: {s['last_error']}" if s.get("errors") else ""))
    if r.get("ablations"):
        L.append(f"ablations (a stage earns its place when removing it costs >= {r['margin']}):")
        for a in r["ablations"]:
            tag = "EARNS" if a["earns"] else "does not earn"
            L.append(f"  {a['name']:<18} {_fmt(a['summary'])}  -> {tag}"
                     + (f" ({', '.join(a['because'])})" if a["because"] else ""))
    if r.get("synthesis"):
        s = r["synthesis"]
        L.append(f"ask: {s['n']} answered, cited a gold note {s['cited_gold_rate']:.2f}, "
                 f"'not in memory' {s['not_in_memory']} (false abstain "
                 f"{'n/a' if s['false_abstain'] is None else s['false_abstain']}), mean {s['mean_ask_ms']:.0f} ms "
                 f"({s['synthesis_llm']})")
        if s["judged"]:
            L.append(f"  answer-correct {s['answer_correct']}/{s['judged']} "
                     f"({s['answer_correct_rate']:.2f}), judge ${s['judge_cost_usd']:.4f}")
    return "\n".join(L)


def report_json(r: dict) -> str:
    return json.dumps(r, indent=2, default=str)
