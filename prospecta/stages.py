"""Recall stages after fusion (hybrid retrieval design 8.7 / 8.8): the
Reranker (SonnetListwise, JevScore, and the Jev gate over Sonnet), the reader
and the agentic second hop. Every stage is off unless banks.recall_config
turns it on; a failing stage never sinks the recall (the fused order stands
and the record says why)."""
from __future__ import annotations

import dataclasses
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from prospecta._template import render_prompt
from prospecta.channels.base import QueryPlan, RecallState
from prospecta.channels.fusion import FusedDoc
from prospecta.channels.recall import run_channels

EVIDENCE_CHARS = 1000          # best chunk per note shown to the reranker
RERANK_POOL = 30
JEV_BATCH = 15                 # candidates per Jev request (Spire's RANK_POOL is 16)
JEV_INPUT_BYTES = 60_000
JEV_PASSAGE_CHARS = 2_400
JEV_TIMEOUT_S = 10.0
JEV_GATE_THRESHOLD = 2.95
JEV_LEGEND = [
    "Unrelated",
    "On the same subject, but does not help answer the query",
    "Partly answers",
    "Holds what is needed",
]
CHEAP_CHANNELS = ("dense_chunk", "bm25", "question")
SONNET_MODEL = "anthropic/claude-sonnet-5.5"

STAGES = ("sonnet_listwise", "jev_score")

DEFAULT_RECALL_CONFIG: dict = {
    "rerank": {"enabled": True, "stage": "sonnet_listwise", "pool": RERANK_POOL},
    "gate": {"enabled": False, "threshold": JEV_GATE_THRESHOLD},
    "reader": {"enabled": False, "top": 8, "join_top": 15,
               "max_follow_ups": 2, "max_new": 10},
}


# --------------------------------------------------------------------- config

def validate_recall_config(cfg: dict) -> None:
    if not isinstance(cfg, dict):
        raise ValueError("recall_config must be an object")
    for key in cfg:
        if key not in ("rerank", "gate", "reader"):
            raise ValueError(f"unknown recall_config key {key!r}")
    rr, gate, rd = (cfg.get(k) or {} for k in ("rerank", "gate", "reader"))
    if rr.get("stage", "sonnet_listwise") not in STAGES:
        raise ValueError(f"rerank.stage must be one of {STAGES}")
    if gate.get("enabled"):
        if not rr.get("enabled"):
            raise ValueError("gate needs rerank enabled (the stage it hands over to)")
        if rr.get("stage", "sonnet_listwise") != "sonnet_listwise":
            raise ValueError("gate needs rerank.stage = sonnet_listwise")
    for section, name, lo in (
        (rr, "pool", 1), (rd, "top", 1), (rd, "join_top", 1),
        (rd, "max_follow_ups", 1), (rd, "max_new", 1),
    ):
        v = section.get(name)
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < lo):
            raise ValueError(f"{name} must be an integer >= {lo}")
    t = gate.get("threshold")
    if t is not None and (isinstance(t, bool) or not isinstance(t, (int, float))):
        raise ValueError("gate.threshold must be a number")


def read_recall_config(conn, bank_id: str) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT recall_config FROM banks WHERE bank_id = %s", (bank_id,))
        row = cur.fetchone()
    return dict(row[0]) if row and row[0] else {}


# ---------------------------------------------------------------- LLM plumbing

@dataclass(frozen=True)
class LLMResult:
    """What an LLM callable may return instead of a bare string, so a recall
    can account for tokens and cost."""
    text: str
    model: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None


def call_llm(llm, messages: list[dict], *, purpose: str, calls: list[dict],
             model: str | None = None) -> str:
    """Call `llm` in JSON mode; append one call record (success or failure)."""
    t0 = time.monotonic()
    prompt = "\n\n".join(str(m.get("content", "")) for m in messages)
    rec: dict = {"purpose": purpose, "model": model, "tokens_in": None,
                 "tokens_out": None, "cost_usd": None, "json_mode": True,
                 "messages_count": len(messages), "prompt_text": prompt,
                 "response_text": None, "error": None}
    try:
        out = llm(messages, json_mode=True)
        if isinstance(out, LLMResult):
            rec.update(model=out.model or model, tokens_in=out.tokens_in,
                       tokens_out=out.tokens_out, cost_usd=out.cost_usd)
            out = out.text
        rec["response_text"] = out
        return out
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        rec["duration_ms"] = int((time.monotonic() - t0) * 1000)
        calls.append(rec)


def totals(calls: list[dict]) -> dict:
    def s(key):
        vals = [c[key] for c in calls if c.get(key) is not None]
        return sum(vals) if vals else None
    return {"n_llm_calls": len(calls), "tokens_in": s("tokens_in"),
            "tokens_out": s("tokens_out"), "cost_usd": s("cost_usd")}


def parse_json_object(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        v = json.loads(t)
    except ValueError:
        i, j = t.find("{"), t.rfind("}")
        if i < 0 or j <= i:
            raise ValueError("reply holds no JSON object")
        v = json.loads(t[i:j + 1])
    if not isinstance(v, dict):
        raise ValueError("reply JSON is not an object")
    return v


# ------------------------------------------------------------------- reranking

@dataclass(frozen=True)
class Item:
    """One note to rank: its fused doc, a header and its best chunk."""
    doc: FusedDoc
    header: str
    evidence: str


@dataclass
class Outcome:
    order: list[int]                                    # indexes into items, best first
    grades: dict[str, float] = field(default_factory=dict)   # document_id -> grade
    jev: dict[str, float] = field(default_factory=dict)      # document_id -> Jev score
    record: dict = field(default_factory=dict)


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, items: list[Item], calls: list[dict]) -> Outcome: ...
    # Raises on failure; run_reranker turns that into the fused-order fallback.


def build_items(conn, docs: list[FusedDoc]) -> list[Item]:
    """Header = note name, date, person; evidence = the best chunk (<= 1,000 chars)."""
    ids = [d.document_id for d in docs]
    meta: dict[str, tuple] = {}
    if ids:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id::text, created_at, document_metadata->>'person' "
                "FROM documents WHERE id = ANY(%s::uuid[])", (ids,))
            meta = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    items = []
    for d in docs:
        created, person = meta.get(d.document_id, (None, None))
        parts = [d.source or d.document_id]
        if created is not None:
            parts.append(f"date: {created.date().isoformat()}")
        if person:
            parts.append(f"person: {person}")
        ev = (d.best.evidence if d.best else None) or ""
        items.append(Item(doc=d, header=" | ".join(parts), evidence=ev[:EVIDENCE_CHARS]))
    return items


def _order_by(items: list[Item], primary: list[int], grades: dict[str, float]) -> list[int]:
    """ranking first, then grade (desc), then the fused order."""
    seen = set(primary)
    rest = sorted((i for i in range(len(items)) if i not in seen),
                  key=lambda i: (-grades.get(items[i].doc.document_id, 0.0), i))
    return list(primary) + rest


class SonnetListwise:
    name = "sonnet_listwise"

    def __init__(self, llm, model: str | None = SONNET_MODEL):
        self.llm, self.model = llm, model

    def rerank(self, query: str, items: list[Item], calls: list[dict]) -> Outcome:
        cands = "\n\n".join(f"[{i}] {it.header}\n{it.evidence}"
                            for i, it in enumerate(items, start=1))
        prompt = render_prompt("rerank-listwise", {"query": query, "candidates": cands})
        raw = call_llm(self.llm, [{"role": "user", "content": prompt}],
                       purpose="rerank_listwise", calls=calls, model=self.model)
        obj = parse_json_object(raw)
        n = len(items)
        ranking: list[int] = []
        for v in obj.get("ranking") or []:
            if isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= n \
                    and (v - 1) not in ranking:
                ranking.append(v - 1)
        grades: dict[str, float] = {}
        g = obj.get("grades") or {}
        pairs = g.items() if isinstance(g, dict) else enumerate(g, start=1)
        for k, v in pairs:
            try:
                idx = int(k) - 1
                if 0 <= idx < n and not isinstance(v, bool):
                    grades[items[idx].doc.document_id] = float(v)
            except (TypeError, ValueError):
                continue
        if not ranking and not grades:
            raise ValueError("reply carries neither a usable ranking nor grades")
        order = _order_by(items, ranking, grades)
        return Outcome(order=order, grades=grades,
                       record={"stage": self.name, "model": self.model,
                               "n_ranked": len(ranking), "n_graded": len(grades)})


# ------------------------------------------------------------------------ Jev

JevTransport = Callable[[dict, float], dict]


def openrouter_jev_transport(api_key: str | None = None,
                             url: str = "https://openrouter.ai/api/v1/systemone"):
    """POST to OpenRouter's System One door. The key comes from the argument or
    OPENROUTER_API_KEY and is never stored."""
    def transport(request: dict, timeout: float) -> dict:
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("no OpenRouter key for Jev")
        req = urllib.request.Request(
            url, data=json.dumps(request).encode(),
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    return transport


class JevScore:
    """Spire's call shape: state {query}, one `score` question per candidate,
    criteria 0 to 3, the candidate quoted as material with its title; at most
    15 per request (two requests for a pool of 30), 60 KB per request, 10 s
    timeout. Jev only reorders: it never adds a candidate or returns text."""
    name = "jev_score"

    def __init__(self, transport: JevTransport, model: str = "jev",
                 timeout: float = JEV_TIMEOUT_S):
        self.transport, self.model, self.timeout = transport, model, timeout

    def _request(self, query: str, batch: list[tuple[int, Item]]) -> dict:
        questions = []
        for i, it in batch:
            questions.append({
                "id": f"c{i}", "kind": "score", "question": "How well does the "
                "material answer the query?", "criteria": JEV_LEGEND,
                "material": {"title": it.header,
                             "text": it.evidence[:JEV_PASSAGE_CHARS]},
            })
        return {"model": self.model, "state": {"query": query}, "questions": questions}

    def _batches(self, query: str, items: list[Item]) -> list[list[tuple[int, Item]]]:
        batches: list[list[tuple[int, Item]]] = []
        cur: list[tuple[int, Item]] = []
        for pair in enumerate(items):
            trial = cur + [pair]
            too_big = len(json.dumps(self._request(query, trial)).encode()) > JEV_INPUT_BYTES
            if cur and (len(trial) > JEV_BATCH or too_big):
                batches.append(cur)
                cur = [pair]
            else:
                cur = trial
        if cur:
            batches.append(cur)
        return batches

    def score(self, query: str, items: list[Item], calls: list[dict]) -> list[float]:
        def one(batch):
            t0 = time.monotonic()
            req = self._request(query, batch)
            rec = {"purpose": "jev_score", "model": self.model, "tokens_in": None,
                   "tokens_out": None, "cost_usd": None, "json_mode": False,
                   "messages_count": len(batch), "prompt_text": json.dumps(req),
                   "response_text": None, "error": None}
            try:
                resp = self.transport(req, self.timeout)
                rec["response_text"] = json.dumps(resp)
                u = resp.get("usage") or {}
                rec.update(tokens_in=u.get("input_tokens", u.get("prompt_tokens")),
                           tokens_out=u.get("output_tokens", u.get("completion_tokens")),
                           cost_usd=u.get("cost"))
                got = {a.get("id"): a.get("score") for a in resp.get("answers") or []}
                out = {}
                for i, _ in batch:
                    v = got.get(f"c{i}")
                    if isinstance(v, bool) or not isinstance(v, (int, float)):
                        raise ValueError("answer does not score every candidate")
                    out[i] = float(v)
                return out
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                rec["duration_ms"] = int((time.monotonic() - t0) * 1000)
                calls.append(rec)

        batches = self._batches(query, items)
        with ThreadPoolExecutor(max_workers=max(1, len(batches))) as ex:
            parts = list(ex.map(one, batches))  # raises on any failed batch
        scores: dict[int, float] = {}
        for p in parts:
            scores.update(p)
        return [scores[i] for i in range(len(items))]

    def rerank(self, query: str, items: list[Item], calls: list[dict]) -> Outcome:
        s = self.score(query, items, calls)
        order = sorted(range(len(items)), key=lambda i: (-s[i], i))
        return Outcome(order=order,
                       jev={items[i].doc.document_id: s[i] for i in range(len(items))},
                       record={"stage": self.name, "model": self.model})


class JevGate:
    """Jev scores the pool; if its top score reaches the threshold the Jev
    order stands, otherwise `fallback` (Sonnet) reranks."""
    name = "jev_gate"

    def __init__(self, jev: JevScore, fallback: Reranker,
                 threshold: float = JEV_GATE_THRESHOLD):
        self.jev, self.fallback, self.threshold = jev, fallback, threshold

    def rerank(self, query: str, items: list[Item], calls: list[dict]) -> Outcome:
        try:
            out = self.jev.rerank(query, items, calls)
        except Exception as exc:
            res = self.fallback.rerank(query, items, calls)
            res.record = {**res.record, "gate": {
                "decision": "jev_failed", "threshold": self.threshold,
                "reason": f"Jev could not rank: {type(exc).__name__}: {exc}"}}
            return res
        top = max(out.jev.values(), default=0.0)
        if top >= self.threshold:
            out.record = {**out.record, "gate": {
                "decision": "jev", "threshold": self.threshold, "top_score": top}}
            return out
        res = self.fallback.rerank(query, items, calls)
        res.jev = out.jev  # keep Jev's scores next to Sonnet's grades
        res.record = {**res.record, "gate": {
            "decision": "sonnet", "threshold": self.threshold, "top_score": top}}
        return res


def build_reranker(cfg: dict, *, llm, jev: JevScore | None, model: str | None = None):
    """The configured Reranker, or None when reranking is off."""
    rr = cfg.get("rerank") or {}
    if not rr.get("enabled"):
        return None
    stage = rr.get("stage", "sonnet_listwise")
    if stage == "jev_score":
        if jev is None:
            raise RuntimeError("rerank.stage = jev_score needs a Jev transport")
        return jev
    if llm is None:
        raise RuntimeError("rerank needs an llm callable")
    sonnet = SonnetListwise(llm, model or SONNET_MODEL)
    gate = cfg.get("gate") or {}
    if gate.get("enabled"):
        if jev is None:
            raise RuntimeError("gate needs a Jev transport")
        return JevGate(jev, sonnet, float(gate.get("threshold", JEV_GATE_THRESHOLD)))
    return sonnet


def run_reranker(reranker: Reranker, query: str, items: list[Item],
                 calls: list[dict]) -> tuple[list[Item], Outcome | None, dict]:
    """Apply a reranker. On any failure the fused order stands, and the record
    says why. Returns (ordered items, outcome or None, record)."""
    t0 = time.monotonic()
    n0 = len(calls)
    try:
        out = reranker.rerank(query, items, calls)
        rec = {**out.record, "fallback_reason": None}
        ordered = [items[i] for i in out.order]
        rec["order"] = [it.doc.document_id for it in ordered]
        rec["grades"] = out.grades
        rec["jev"] = out.jev
    except Exception as exc:
        out = None
        rec = {"stage": getattr(reranker, "name", "?"),
               "fallback_reason": f"fused order stands: {type(exc).__name__}: {exc}",
               "order": [it.doc.document_id for it in items], "grades": {}, "jev": {}}
        ordered = list(items)
    rec["n_candidates"] = len(items)
    rec["latency_ms"] = int((time.monotonic() - t0) * 1000)
    rec.update(totals(calls[n0:]))
    return ordered, out, rec


# --------------------------------------------------------------- reader + hop

class SonnetReader:
    name = "sonnet_reader"

    def __init__(self, llm, model: str | None = SONNET_MODEL):
        self.llm, self.model = llm, model

    def read(self, query: str, items: list[Item], max_follow_ups: int,
             calls: list[dict]) -> tuple[bool, list[str]]:
        ex = "\n\n".join(f"[{i}] {it.header}\n{it.evidence}"
                         for i, it in enumerate(items, start=1))
        prompt = render_prompt("read-sufficiency", {
            "query": query, "excerpts": ex, "max_follow_ups": max_follow_ups})
        raw = call_llm(self.llm, [{"role": "user", "content": prompt}],
                       purpose="reader_sufficiency", calls=calls, model=self.model)
        obj = parse_json_object(raw)
        fu = [q.strip() for q in (obj.get("follow_ups") or [])
              if isinstance(q, str) and q.strip()][:max_follow_ups]
        sufficient = bool(obj.get("sufficient", not fu))
        if sufficient:
            fu = []
        return sufficient, fu


@dataclass
class StageDeps:
    llm: Any = None
    jev: JevScore | None = None
    model: str | None = None


def run_stages(
    state: RecallState, query: str, fused: list[FusedDoc], channel_config: list[dict],
    recall_cfg: dict, deps: StageDeps, *, k: int = 60,
) -> tuple[list[FusedDoc], dict]:
    """Rerank, then (optionally) read and hop. Returns the final document
    order (all pool docs, best first) and {rerank, hops, calls, **totals}."""
    calls: list[dict] = []
    trace: dict = {"rerank": None, "hops": None, "calls": calls, **totals(calls)}
    try:
        return _run_stages(state, query, fused, channel_config, recall_cfg, deps, k,
                           calls, trace)
    except Exception as exc:
        trace.update(rerank=None, hops=None, **totals(calls),
                     fallback_reason=f"{type(exc).__name__}: {exc}")
        return fused, trace


def _run_stages(state, query, fused, channel_config, recall_cfg, deps, k, calls, trace):
    reranker = build_reranker(recall_cfg, llm=deps.llm, jev=deps.jev, model=deps.model)
    reader_cfg = recall_cfg.get("reader") or {}
    if reranker is None and not reader_cfg.get("enabled"):
        trace.update(totals(calls))
        return fused, trace

    pool_n = int((recall_cfg.get("rerank") or {}).get("pool", RERANK_POOL))
    pool, tail = fused[:pool_n], fused[pool_n:]
    items = build_items(state.conn, pool)
    rerank_rec = None
    if reranker is not None:
        items, out, rerank_rec = run_reranker(reranker, query, items, calls)
        if out is not None:
            items = [dataclasses.replace(it, doc=_with_scores(it.doc, out)) for it in items]
        trace["rerank"] = rerank_rec

    if reader_cfg.get("enabled"):
        if deps.llm is None:
            raise RuntimeError("reader needs an llm callable")
        items, hops = _hop(state, query, items, channel_config, reader_cfg, reranker,
                           SonnetReader(deps.llm, deps.model or SONNET_MODEL), calls, k)
        trace["hops"] = hops

    trace.update(totals(calls))
    return [it.doc for it in items] + tail, trace


def _with_scores(doc: FusedDoc, out: Outcome) -> FusedDoc:
    extra = {}
    if doc.document_id in out.grades:
        extra["rerank"] = out.grades[doc.document_id]
    if doc.document_id in out.jev:
        extra["jev"] = out.jev[doc.document_id]
    return dataclasses.replace(doc, scores={**doc.scores, **extra}) if extra else doc


def _hop(state, query, items, channel_config, rcfg, reranker, reader, calls, k):
    top_n = int(rcfg.get("top", 8))
    join_top = int(rcfg.get("join_top", 15))
    max_fu = int(rcfg.get("max_follow_ups", 2))
    max_new = int(rcfg.get("max_new", 10))
    hops: dict = {"reader": reader.name, "verdict": None, "follow_ups": [],
                  "new_candidates": [], "rerank": None, "error": None}
    try:
        sufficient, follow_ups = reader.read(query, items[:top_n], max_fu, calls)
    except Exception as exc:
        hops["verdict"] = "error"
        hops["error"] = f"{type(exc).__name__}: {exc}"
        return items, hops
    hops["verdict"] = "sufficient" if sufficient else "follow_up"
    hops["follow_ups"] = follow_ups
    if sufficient or not follow_ups:
        return items, hops

    cheap = [e for e in channel_config if e.get("name") in CHEAP_CHANNELS]
    lists: list[list[FusedDoc]] = []
    for fq in follow_ups:   # cheap channels only; one failed follow-up skips only itself
        try:
            fused, tr = run_channels(state, QueryPlan(text=fq), cheap, k=k, pool=max_new + join_top)
            lists.append(fused)
            for c in tr["channels"]:
                if c["error"]:
                    hops["error"] = f"{c['name']}: {c['error']}"
        except Exception as exc:
            hops["error"] = f"{type(exc).__name__}: {exc}"
    keep = {it.doc.document_id for it in items[:join_top]}
    new: list[FusedDoc] = []
    for rank in range(max((len(l) for l in lists), default=0)):   # round-robin by rank
        for l in lists:
            if rank < len(l) and len(new) < max_new:
                d = l[rank]
                if d.document_id not in keep and all(d.document_id != n.document_id for n in new):
                    new.append(d)
    hops["new_candidates"] = [d.document_id for d in new]
    if not new:
        return items, hops

    head = items[:join_top]
    joined = head + build_items(state.conn, new)
    joined_ids = {it.doc.document_id for it in joined}
    rest = [it for it in items[join_top:] if it.doc.document_id not in joined_ids]
    if reranker is not None:
        joined, out, rec = run_reranker(reranker, query, joined, calls)
        if out is not None:
            joined = [dataclasses.replace(it, doc=_with_scores(it.doc, out)) for it in joined]
        hops["rerank"] = rec
    return joined + rest, hops
