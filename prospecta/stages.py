"""Recall stages after fusion (docs/postgres-hybrid.md): Stage 2, the cross-encoder rerank
(CrossEncoderReranker, OpenRouter's /rerank endpoint), and the decision model's System One
door (JevScore: the holder of its transport and model; Stage 0's `needs_memory` gate and Stage 3's
`evidence_sufficient` question go through it). A stage is off unless banks.recall_config turns
it on; a failing stage never sinks the recall (the fused order stands and the record says why).

The Sonnet listwise rerank, the Jev rerank, the Jev gate over Sonnet, the reader and its
second hop, and the rerank blend were removed (docs/legacy.md). A bank that stored a
recall_config naming them keeps it: recall skips them with one warning per process, and
`set_recall_config` no longer accepts them."""
from __future__ import annotations

import copy
import dataclasses
import logging
import json
import math
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from prospecta._filters import coerce_date
from prospecta._scorecut import check_depth, estimate_tokens, pool_min_rel, rel_cut, split_batches
from prospecta._warn import warn_once
from prospecta.channels.base import RecallState
from prospecta.channels.fusion import FusedDoc
from prospecta.channels.fusion import METHODS as FUSION_METHODS

JEV_TIMEOUT_S = 10.0
STAGE3_THRESHOLD = 0.70        # stage 3: evidence_sufficient / 3 under this runs one broadened retry
# The decision model (Jev): PROSPECTA_DECISION_MODEL, else its older alias
# PROSPECTA_JEV_MODEL, else the default; read once at import. A bank's
# recall_config.models.decision_model and StageDeps.decision_model override it.
JEV_MODEL = (os.environ.get("PROSPECTA_DECISION_MODEL") or os.environ.get("PROSPECTA_JEV_MODEL")
             or "typesafe/jev-1.14")
JEV_URL = "https://openrouter.ai/api/v1/systemone"
logger = logging.getLogger(__name__)

STAGES = ("cross_encoder",)
# What a stored recall_config may still name and recall now skips (docs/legacy.md): the rerank
# stages, and the keys of the Jev gate, the reader / hop and the linker.
RETIRED_STAGES = {
    "sonnet_listwise": "the Sonnet listwise rerank was removed; use the cross_encoder stage",
    "jev_score": "the Jev rerank was removed; use the cross_encoder stage",
}
RETIRED_RECALL_KEYS = {
    "gate": "the Jev gate over the Sonnet rerank was removed",
    "reader": "the reader and its second hop were removed",
    "link_completeness": "the linker was removed",
}

# Cross-encoder (Stage 2) defaults; the model and every number is configurable
# (recall_config.rerank.*, see docs/recall-stages.md).
RERANKER_MODEL = "qwen/qwen3-reranker-8b"
RERANK_URL = "https://openrouter.ai/api/v1/rerank"
RERANK_INSTRUCTION = ("Instruct: Retrieve memory records that directly answer or provide factual "
                      "background for the query, honoring exact names, dates, and quoted phrases.\n"
                      "Query: ")
RERANK_TOP_N = 10
RERANK_MIN_SCORE = 0.15        # absolute floor on the calibrated relevance score (0..1)
RERANK_TIMEOUT_S = 2.5
NEAR_COPY_JACCARD = 0.88       # token Jaccard above which two notes are near-copies
NEAR_COPY_COSINE = 0.90        # chunk-embedding cosine at or above which two notes say the same thing (query expansion's merge; measured, docs/postgres-hybrid.md)
# qwen3-reranker-8b reads 41K tokens per request. A request carries at most 32K
# estimated tokens (4 chars/token): the rest is margin for tokenizer variance,
# the instruction and the query. A larger pool is split into several requests.
RERANK_BATCH_TOKENS = 32_000


def resolve_reranker_model(call: str | None = None, bank: str | None = None) -> str:
    """The reranker model: per-call, else the bank's recall_config.rerank.model,
    else PROSPECTA_RERANKER_MODEL, else the default."""
    return call or bank or os.environ.get("PROSPECTA_RERANKER_MODEL") or RERANKER_MODEL


# --------------------------------------------------------------------- config

def validate_recall_config(cfg: dict) -> None:
    """For a NEW config (set_recall_config). A retired key or stage is refused; a STORED config
    that names one is still read (see warn_retired_recall_config)."""
    if not isinstance(cfg, dict):
        raise ValueError("recall_config must be an object")
    for key in cfg:
        if key in RETIRED_RECALL_KEYS:
            raise ValueError(f"recall_config key {key!r} was retired ({RETIRED_RECALL_KEYS[key]}); "
                             "see docs/legacy.md")
        if key not in ("rerank", "evidence", "depth", "fusion", "models", "stage3_sufficiency",
                       "stage0_gate", "expansion"):
            raise ValueError(f"unknown recall_config key {key!r}")
    check_depth(cfg.get("depth"))
    fu = cfg.get("fusion", {})
    if not isinstance(fu, dict) or set(fu) - {"method", "floor"}:
        raise ValueError("fusion must be an object of method/floor")
    if fu.get("method", "rrf") not in FUSION_METHODS:
        raise ValueError(f"fusion.method must be one of {FUSION_METHODS}")
    if "floor" in fu and (isinstance(fu["floor"], bool) or not isinstance(fu["floor"], int)
                          or fu["floor"] < 1):
        raise ValueError("fusion.floor must be an integer >= 1")
    models = cfg.get("models", {})
    if not isinstance(models, dict) or set(models) - {"decision_model", "query_writer_model"}:
        raise ValueError("models must be an object of decision_model/query_writer_model")
    s3 = cfg.get("stage3_sufficiency", {})
    if not isinstance(s3, dict) or set(s3) - {"enabled", "threshold"}:
        raise ValueError("stage3_sufficiency must be an object of enabled/threshold")
    if not isinstance(s3.get("enabled", False), bool):
        raise ValueError("stage3_sufficiency.enabled must be true or false")
    v = s3.get("threshold")
    if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1):
        raise ValueError("stage3_sufficiency.threshold must be a number in [0, 1]")
    g0 = cfg.get("stage0_gate", {})
    if not isinstance(g0, dict) or set(g0) - {"enabled", "needs_memory_threshold"}:
        raise ValueError("stage0_gate must be an object of enabled/needs_memory_threshold")
    if not isinstance(g0.get("enabled", True), bool):
        raise ValueError("stage0_gate.enabled must be true or false")
    v = g0.get("needs_memory_threshold")
    if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1):
        raise ValueError("stage0_gate.needs_memory_threshold must be a number in [0, 1]")
    ex = cfg.get("expansion", {})
    if not isinstance(ex, dict) or set(ex) - {"max_searches", "max_concurrency", "deadline_s",
                                              "max_tokens", "fallback_min_rel", "near_copy_cosine"}:
        raise ValueError("expansion must be an object of max_searches/max_concurrency/"
                         "deadline_s/max_tokens/fallback_min_rel/near_copy_cosine")
    for name, floor in (("max_searches", 1), ("max_concurrency", 1), ("max_tokens", 0)):
        v = ex.get(name)
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < floor):
            raise ValueError(f"expansion.{name} must be an integer >= {floor}")
    v = ex.get("deadline_s")
    if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0):
        raise ValueError("expansion.deadline_s must be a number > 0")
    v = ex.get("fallback_min_rel")
    if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1):
        raise ValueError("expansion.fallback_min_rel must be a number in [0, 1]")
    v = ex.get("near_copy_cosine")
    if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v <= 1):
        raise ValueError("expansion.near_copy_cosine must be a number in (0, 1]")
    rr = cfg.get("rerank") or {}
    for section, name in ((models, "decision_model"), (models, "query_writer_model"), (rr, "model")):
        v = section.get(name)
        if v is not None and (not isinstance(v, str) or not v):
            raise ValueError(f"{name} must be a non-empty string")
    v = rr.get("top_n")
    if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 1):
        raise ValueError("rerank.top_n must be an integer >= 1")
    v = rr.get("timeout_s")
    if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0):
        raise ValueError("rerank.timeout_s must be a number > 0")
    if not isinstance(rr.get("deduplicate_near_copies", True), bool):
        raise ValueError("rerank.deduplicate_near_copies must be true or false")
    stage = rr.get("stage")
    if stage in RETIRED_STAGES:
        raise ValueError(f"rerank.stage {stage!r} was retired ({RETIRED_STAGES[stage]}); "
                         "see docs/legacy.md")
    if stage is not None and stage not in STAGES:
        raise ValueError(f"rerank.stage must be one of {STAGES}")
    if rr.get("enabled") and stage is None:
        raise ValueError(f"rerank.stage is required when rerank is enabled: one of {STAGES} "
                         "(its old default, sonnet_listwise, was retired; see docs/legacy.md)")
    if "blend" in rr:
        raise ValueError("rerank.blend was retired (it regressed hit@1 and was off everywhere); "
                         "see docs/legacy.md")
    for name in ("min_rel_score", "min_score"):
        v = rr.get(name)
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))
                              or not 0 <= v <= 1):
            raise ValueError(f"{name} must be a number in [0, 1]")
    for gone in ("pool", "top", "join_top", "max_new", "max_follow_ups"):
        if gone in rr:
            raise ValueError(f"{gone!r} was a count and is gone: pools are score-based "
                             "(rerank min_rel_score, see docs/limits.md)")
    ev = cfg.get("evidence") or {}
    m = ev.get("min_rel_score")
    if m is not None and (isinstance(m, bool) or not isinstance(m, (int, float)) or not 0 <= m <= 1):
        raise ValueError("evidence.min_rel_score must be a number in [0, 1]")
    c = ev.get("context_tokens")
    if c is not None and (isinstance(c, bool) or not isinstance(c, int) or c < 1):
        raise ValueError("evidence.context_tokens must be an integer >= 1")


def read_recall_config(conn, bank_id: str) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT recall_config FROM banks WHERE bank_id = %s", (bank_id,))
        row = cur.fetchone()
    return dict(row[0]) if row and row[0] else {}


def warn_retired_recall_config(cfg: dict) -> None:
    """A STORED recall_config may name what was removed: recall skips it, saying so once per process."""
    for key, why in RETIRED_RECALL_KEYS.items():
        if key in cfg:
            warn_once(f"recall_config:{key}",
                      "recall_config names %r, which was retired (%s) and is skipped; see docs/legacy.md",
                      key, why)
    rr = cfg.get("rerank") or {}
    if rr.get("enabled"):
        stage = rr.get("stage")
        if stage in RETIRED_STAGES:
            warn_once(f"recall_config:{stage}",
                      "recall_config.rerank.stage is %r, which was retired (%s): the Stage 1 order stands; "
                      "see docs/legacy.md", stage, RETIRED_STAGES[stage])
        elif stage is None:
            warn_once("recall_config:default_stage",
                      "recall_config.rerank names no stage, and its old default 'sonnet_listwise' was "
                      "retired: the Stage 1 order stands; set rerank.stage to 'cross_encoder' (docs/legacy.md)")
    if "blend" in rr:
        warn_once("recall_config:blend",
                  "recall_config.rerank.blend was retired and is skipped; see docs/legacy.md")


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


def totals(calls: list[dict]) -> dict:
    def s(key):
        vals = [c[key] for c in calls if c.get(key) is not None]
        return sum(vals) if vals else None
    return {"n_llm_calls": len(calls), "tokens_in": s("tokens_in"),
            "tokens_out": s("tokens_out"), "cost_usd": s("cost_usd")}


def parse_json_object(text: str) -> dict:
    """The FIRST valid JSON object in a reply; tolerates prose, code fences
    and repeated copies of the object before and after it."""
    dec = json.JSONDecoder()
    pos, saw_other = 0, False
    while (i := text.find("{", pos)) >= 0:
        try:
            v, _ = dec.raw_decode(text, i)
        except ValueError:
            pos = i + 1
            continue
        if isinstance(v, dict):
            return v
        saw_other = True
        pos = i + 1
    try:
        v = json.loads(text.strip())
    except ValueError:
        v = None
    if v is not None or saw_other:
        raise ValueError("reply JSON is not an object")
    raise ValueError("reply holds no JSON object")


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
    record: dict = field(default_factory=dict)


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, items: list[Item], calls: list[dict]) -> Outcome: ...
    # Raises on failure; run_reranker turns that into the fused-order fallback.


_FILENAME_DATE = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")


def _note_date(created_on, source, dmeta, created_at):
    """The note's own date: created_on, then metadata/filename date; the
    import time (created_at) only as the last fallback."""
    if created_on is not None:
        return created_on
    for k in ("created_on", "created", "date"):
        d = coerce_date(dmeta.get(k))
        if d is not None:
            return d
    m = _FILENAME_DATE.search(source or "")
    if m:
        d = coerce_date(m.group(1))
        if d is not None:
            return d
    return created_at.date() if created_at is not None else None


def build_items(conn, docs: list[FusedDoc]) -> list[Item]:
    """Header = note name, date, person; evidence = the best chunk, in full."""
    ids = [d.document_id for d in docs]
    meta: dict[str, tuple] = {}
    if ids:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id::text, created_on, created_at, source, document_metadata, person "
                "FROM documents WHERE id = ANY(%s::uuid[])", (ids,))
            meta = {r[0]: r[1:] for r in cur.fetchall()}
    items = []
    for d in docs:
        created_on, created_at, source, dmeta, person = meta.get(
            d.document_id, (None, None, None, None, None))
        dmeta = dmeta or {}
        person = person or dmeta.get("person")
        note_date = _note_date(created_on, source or d.source, dmeta, created_at)
        parts = [d.source or d.document_id]
        if note_date is not None:
            parts.append(f"date: {note_date.isoformat()}")
        if person:
            parts.append(f"person: {person}")
        ev = (d.best.evidence if d.best else None) or ""
        items.append(Item(doc=d, header=" | ".join(parts), evidence=ev))
    return items


# ------------------------------------------------------------------------ Jev

JevTransport = Callable[[dict, float], dict]


def _openrouter_transport(url: str, api_key: str | None, what: str):
    def transport(request: dict, timeout: float) -> dict:
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(f"no OpenRouter key for {what}")
        req = urllib.request.Request(
            url, data=json.dumps(request).encode(),
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    return transport


def openrouter_jev_transport(api_key: str | None = None,
                             url: str = JEV_URL):
    """POST to OpenRouter's System One door. The key comes from the argument or
    OPENROUTER_API_KEY and is never stored."""
    return _openrouter_transport(url, api_key, "Jev")


class JevScore:
    """The decision model (typesafe/jev-1.14, System One): the holder of its transport (a callable
    `(request, timeout) -> response`, e.g. openrouter_jev_transport), its model and its timeout.
    Stage 0 (the `needs_memory` gate of Memory.expand_and_recall) and Stage 3 (the
    `evidence_sufficient` question of Memory.ask, JevSufficiency) call it through
    `Memory(jev=...)`. (Its listwise `score` rerank was removed: docs/legacy.md.)"""
    name = "jev"

    def __init__(self, transport: JevTransport, model: str = JEV_MODEL,
                 timeout: float = JEV_TIMEOUT_S):
        self.transport, self.model, self.timeout = transport, model, timeout


# ---------------------------------------------------------------- cross-encoder

def openrouter_rerank_transport(api_key: str | None = None, url: str = RERANK_URL):
    """POST to OpenRouter's rerank endpoint. The key comes from the argument or
    OPENROUTER_API_KEY and is never stored."""
    return _openrouter_transport(url, api_key, "rerank")


def _words(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"\w+", text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def deduplicate_near_copies(texts: Sequence[str], order: Sequence[int],
                            threshold: float = NEAR_COPY_JACCARD) -> tuple[list[int], dict[int, int]]:
    """Collapse near-copies. `order` lists indexes into `texts`, best first. A text whose token
    Jaccard with an already-kept text is over `threshold` is a near-copy of the first such
    kept text and collapses into it (the best-scored copy survives). Returns (the surviving
    indexes, in `order`; {collapsed index: the surviving index it collapsed into}).

    Exactly the greedy comparison of every text with every survivor, without making it: a pair
    over the threshold must share a token among the rarest `n - ceil(threshold * n) + 1` tokens
    of each (prefix filtering over the tokens' document frequency), so only texts that share
    such a token are compared. A merge of hundreds of records (20 searches) stays cheap."""
    sets = {i: _words(texts[i]) for i in order}
    freq: dict[str, int] = {}
    for toks in sets.values():
        for t in toks:
            freq[t] = freq.get(t, 0) + 1
    index: dict[str, list[int]] = {}     # token -> positions (in `survivors`) whose prefix holds it
    survivors: list[int] = []
    collapsed: dict[int, int] = {}
    for i in order:
        toks = sets[i]
        if not toks:                      # no words: nothing to be a copy of
            survivors.append(i)
            continue
        ranked = sorted(toks, key=lambda t: (freq[t], t))
        prefix = ranked[:max(1, len(ranked) - math.ceil(threshold * len(ranked) - 1e-9) + 1)]
        candidates = sorted({pos for t in prefix for pos in index.get(t, ())})
        twin = next((survivors[pos] for pos in candidates
                     if _jaccard(toks, sets[survivors[pos]]) > threshold), None)
        if twin is not None:
            collapsed[i] = twin
            continue
        survivors.append(i)
        for t in prefix:
            index.setdefault(t, []).append(len(survivors) - 1)
    return survivors, collapsed


def _numpy():
    """numpy when it is installed (pgvector needs it, so it is), else None: the pure-Python path
    below gives the same answer, only slower."""
    try:
        import numpy
        return numpy
    except ImportError:         # pragma: no cover
        return None


def deduplicate_by_meaning(vectors: Sequence[Sequence[float] | None], order: Sequence[int],
                           threshold: float = NEAR_COPY_COSINE) -> tuple[list[int], dict[int, int]]:
    """Collapse near-copies BY MEANING: `vectors` are embeddings indexed like the records (None where
    a record has none), `order` lists indexes best first. A record whose cosine similarity to an
    already kept record is at or above `threshold` collapses into the first (best) such kept record;
    a record is compared with the kept ones only, never with one that collapsed. A record without a
    usable vector (None, zero, not finite, or not of the usual length) is kept as it is and never
    compared. Returns (the surviving indexes, in `order`; {collapsed index: the survivor it
    collapsed into}), like deduplicate_near_copies."""
    sizes: dict[int, int] = {}
    for i in order:
        v = vectors[i]
        if v is not None:
            sizes[len(v)] = sizes.get(len(v), 0) + 1
    dim = max(sizes, key=lambda d: (sizes[d], d), default=0)
    np = _numpy()
    unit: dict[int, Any] = {}
    for i in order:
        v = vectors[i]
        if v is None or len(v) != dim or dim == 0:
            continue
        if np is not None:
            a = np.asarray(v, dtype=np.float64)
            norm = float(np.linalg.norm(a)) if np.isfinite(a).all() else 0.0
            if norm > 0:
                unit[i] = a / norm
        else:
            vals = [float(x) for x in v]
            norm = math.sqrt(sum(x * x for x in vals)) if all(math.isfinite(x) for x in vals) else 0.0
            if norm > 0:
                unit[i] = [x / norm for x in vals]
    survivors: list[int] = []
    collapsed: dict[int, int] = {}
    kept: list[int] = []                    # the survivors that have a vector, best first
    matrix = np.zeros((len(unit), dim)) if np is not None else None
    for i in order:
        u = unit.get(i)
        if u is None:
            survivors.append(i)
            continue
        twin = None
        if kept:
            if np is not None:
                hits = np.flatnonzero(matrix[:len(kept)] @ u >= threshold - 1e-12)
                twin = kept[int(hits[0])] if len(hits) else None
            else:
                twin = next((k for k in kept
                             if sum(x * y for x, y in zip(unit[k], u)) >= threshold - 1e-12), None)
        if twin is not None:
            collapsed[i] = twin
            continue
        survivors.append(i)
        if matrix is not None:
            matrix[len(kept)] = u
        kept.append(i)
    return survivors, collapsed


class CrossEncoderReranker:
    """Stage 2 of docs/postgres-hybrid.md: a cross-encoder scores every
    (query, note) pair through OpenRouter's rerank endpoint (verified shape:
    POST /api/v1/rerank, body {model, query, documents, top_n}, reply
    {results:[{index, relevance_score}], usage:{total_tokens, cost}}).

    Every candidate is scored (a pool over RERANK_BATCH_TOKENS goes out as
    several parallel requests; scores are calibrated, so they merge). Then, in
    this order: notes under `min_score` are cut; near-copies collapse into the
    highest-scored one (token Jaccard over the evidence text above
    NEAR_COPY_JACCARD: no embedding is at hand here and it needs no extra
    call); the best `top_n` stay. The stage only removes candidates by those
    cuts and never adds one. Any provider error, timeout, 429/503 or malformed
    reply raises, and run_reranker keeps the fused order with the reason."""
    name = "cross_encoder"

    def __init__(self, transport: JevTransport, model: str = RERANKER_MODEL,
                 top_n: int = RERANK_TOP_N, min_score: float = RERANK_MIN_SCORE,
                 timeout: float = RERANK_TIMEOUT_S, deduplicate: bool = True):
        self.transport, self.model, self.top_n = transport, model, top_n
        self.min_score, self.timeout, self.deduplicate = min_score, timeout, deduplicate

    def _score_batch(self, query: str, batch: list[tuple[int, str]],
                     calls: list[dict]) -> dict[int, float]:
        t0 = time.monotonic()
        # top_n = every document sent: the cuts below need all the scores, and
        # the provider bills the input tokens either way.
        req = {"model": self.model, "query": query,
               "documents": [text for _, text in batch], "top_n": len(batch)}
        rec = {"purpose": "cross_encoder_rerank", "model": self.model, "tokens_in": None,
               "tokens_out": None, "cost_usd": None, "json_mode": False,
               "messages_count": len(batch), "prompt_text": json.dumps(req),
               "response_text": None, "error": None}
        try:
            resp = self.transport(req, self.timeout)
            rec["response_text"] = json.dumps(resp)
            if not isinstance(resp, dict):
                raise ValueError("response is not an object")
            u = resp.get("usage") or {}
            rec.update(model=resp.get("model") or self.model,
                       tokens_in=u.get("total_tokens"), cost_usd=u.get("cost"))
            results = resp.get("results")
            if not isinstance(results, list):
                raise ValueError("response has no results")
            out: dict[int, float] = {}
            for r in results:
                n = r.get("index") if isinstance(r, dict) else None
                v = r.get("relevance_score") if isinstance(r, dict) else None
                if (isinstance(n, bool) or not isinstance(n, int) or not 0 <= n < len(batch)
                        or n in out or isinstance(v, bool) or not isinstance(v, (int, float))
                        or not math.isfinite(v)):
                    raise ValueError("results hold an invalid index or relevance_score")
                out[n] = float(v)
            if len(out) != len(batch):
                raise ValueError("results do not score every candidate")
            return {batch[n][0]: v for n, v in out.items()}
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            rec["duration_ms"] = int((time.monotonic() - t0) * 1000)
            calls.append(rec)

    def rerank(self, query: str, items: list[Item], calls: list[dict]) -> Outcome:
        question = RERANK_INSTRUCTION + query
        texts = [(i, f"{it.header}\n{it.evidence}") for i, it in enumerate(items)]
        batches = split_batches(texts, lambda p: p[1],
                                RERANK_BATCH_TOKENS - estimate_tokens(question))
        with ThreadPoolExecutor(max_workers=max(1, len(batches))) as ex:
            parts = list(ex.map(lambda b: self._score_batch(question, b, calls), batches))
        scores: dict[int, float] = {}
        for p in parts:
            scores.update(p)
        ranked = sorted(scores, key=lambda i: (-scores[i], i))
        kept = [i for i in ranked if scores[i] >= self.min_score]
        n_below = len(ranked) - len(kept)
        collapsed: dict[str, str] = {}
        if self.deduplicate:
            kept, twins = deduplicate_near_copies([it.evidence for it in items], kept)
            collapsed = {items[i].doc.document_id: items[j].doc.document_id for i, j in twins.items()}
        order = kept[:self.top_n]
        return Outcome(order=order,
                       grades={items[i].doc.document_id: scores[i] for i in order},
                       record={"stage": self.name, "model": self.model, "top_n": self.top_n,
                               "min_score": self.min_score, "n_below_min_score": n_below,
                               "collapsed": collapsed, "n_over_top_n": len(kept) - len(order),
                               "requests": len(batches)})


def build_reranker(cfg: dict, *, rerank_transport: JevTransport | None = None,
                   reranker_model: str | None = None):
    """The configured Reranker, or None when reranking is off (or names a retired stage: the
    Stage 1 order stands, warn_retired_recall_config says so)."""
    rr = cfg.get("rerank") or {}
    if not rr.get("enabled") or rr.get("stage") != "cross_encoder":
        return None
    if rerank_transport is None:
        raise RuntimeError("rerank.stage = cross_encoder needs a rerank transport")
    return CrossEncoderReranker(
        rerank_transport, resolve_reranker_model(reranker_model, rr.get("model")),
        top_n=int(rr.get("top_n", RERANK_TOP_N)),
        min_score=float(rr.get("min_score", RERANK_MIN_SCORE)),
        timeout=float(rr.get("timeout_s", RERANK_TIMEOUT_S)),
        deduplicate=bool(rr.get("deduplicate_near_copies", True)))


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
    except Exception as exc:
        out = None
        rec = {"stage": getattr(reranker, "name", "?"),
               "model": getattr(reranker, "model", None),
               "fallback_reason": f"fused order stands: {type(exc).__name__}: {exc}",
               "order": [it.doc.document_id for it in items], "grades": {}}
        ordered = list(items)
    rec["n_candidates"] = len(items)
    rec["latency_ms"] = int((time.monotonic() - t0) * 1000)
    rec.update(totals(calls[n0:]))
    return ordered, out, rec


# ------------------------------------------------------------------- Stage 3

_STOP_QUESTIONS = {
    "evidence_sufficient": "Do the excerpts together hold what is needed to answer state.query?",
}


def jev_stop_scores(transport, model: str, timeout: float, query: str, text: str,
                    ids: list[str], purpose: str, calls: list[dict]) -> dict[str, float]:
    """One System One request: each of the stopping questions `ids` (see _STOP_QUESTIONS)
    asked as a `score` question (0..3) over the excerpts `text`. Returns {id: score}; any
    malformed reply raises. The call is recorded in `calls`."""
    req = {"model": model, "state": {"query": query},
           "questions": {f"p{n}": {
               "type": "score",
               "instructions": {"task": _STOP_QUESTIONS[k], "kind": "note",
                                "title": k, "text": text},
               "criteria": ["No", "Barely", "Mostly", "Yes"]}
               for n, k in enumerate(ids)}}
    t0 = time.monotonic()
    rec: dict = {"purpose": purpose, "model": model, "tokens_in": None,
                 "tokens_out": None, "cost_usd": None, "json_mode": False,
                 "messages_count": len(ids), "prompt_text": json.dumps(req),
                 "response_text": None, "error": None}
    try:
        resp = transport(req, timeout)
        rec["response_text"] = json.dumps(resp)
        u = resp.get("usage") or {}
        rec.update(model=resp.get("model") or model, tokens_in=u.get("input_tokens"),
                   tokens_out=u.get("output_tokens"), cost_usd=u.get("cost"))
        answers = resp.get("answers")
        if not isinstance(answers, dict) or set(answers) != {f"p{n}" for n in range(len(ids))}:
            raise ValueError("answers are not exactly the asked ids")
        val: dict[str, float] = {}
        for n, k in enumerate(ids):
            a = answers[f"p{n}"]
            v = a.get("score") if isinstance(a, dict) else None
            if isinstance(v, bool) or not isinstance(v, (int, float)) \
                    or not math.isfinite(v) or not 0 <= v <= 3:
                raise ValueError("answer does not score every question in 0..3")
            val[k] = float(v)
        return val
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        rec["duration_ms"] = int((time.monotonic() - t0) * 1000)
        calls.append(rec)


class JevSufficiency:
    """Stage 3 of docs/postgres-hybrid.md: does the evidence found suffice to answer
    the question? The decision model answers the stopping question `evidence_sufficient`
    (System One `score`, 0..3) over the excerpts; the verdict is that score / 3 (0..1).
    `Memory.ask` compares it with `recall_config.stage3_sufficiency.threshold` (0.70)
    and runs one broadened retrieval when it falls short. Raises on any provider or
    reply error; the caller records it and answers anyway."""
    name = "stage3_sufficiency"

    def __init__(self, transport: JevTransport, model: str = JEV_MODEL,
                 timeout: float = JEV_TIMEOUT_S):
        self.transport, self.model, self.timeout = transport, model, timeout

    def score(self, query: str, excerpts: str, calls: list[dict]) -> float:
        return jev_stop_scores(self.transport, self.model, self.timeout, query, excerpts,
                               ["evidence_sufficient"], self.name, calls)["evidence_sufficient"] / 3


@dataclass
class StageDeps:
    jev: JevScore | None = None
    rerank_transport: JevTransport | None = None
    reranker_model: str | None = None      # per-call, beats recall_config.rerank.model
    decision_model: str | None = None      # per-call, beats recall_config.models.decision_model


def _with_decision_model(jev: JevScore | None, call: str | None, bank: str | None):
    """The Jev to use: a copy on the per-call or bank decision model when one
    is set (the shared Jev is never mutated), else the Jev as built."""
    model = call or bank
    if jev is None or not model:
        return jev
    jev = copy.copy(jev)
    jev.model = model
    return jev


def run_stages(
    state: RecallState, query: str, fused: list[FusedDoc], recall_cfg: dict, deps: StageDeps,
    *, depth: str | None = None,
) -> tuple[list[FusedDoc], dict]:
    """Rerank (Stage 2). Returns the final document order (all pool docs, best first) and
    {rerank, calls, **totals}. Any failure keeps the fused order and says why."""
    calls: list[dict] = []
    trace: dict = {"rerank": None, "calls": calls, **totals(calls)}
    warn_retired_recall_config(recall_cfg)
    try:
        return _run_stages(state, query, fused, recall_cfg, deps, calls, trace, depth)
    except Exception as exc:
        trace.update(rerank=None, **totals(calls),
                     fallback_reason=f"{type(exc).__name__}: {exc}")
        return fused, trace


def _run_stages(state, query, fused, recall_cfg, deps, calls, trace, depth=None):
    reranker = build_reranker(recall_cfg, rerank_transport=deps.rerank_transport,
                              reranker_model=deps.reranker_model)
    if reranker is None:
        trace.update(totals(calls))
        return fused, trace

    pool = rel_cut(fused, lambda d: d.score, pool_min_rel(depth, recall_cfg))
    in_pool = {d.document_id for d in pool}
    tail = [d for d in fused if d.document_id not in in_pool]
    items = build_items(state.conn, pool)
    items, out, rerank_rec = run_reranker(reranker, query, items, calls)
    if out is not None:
        items = [dataclasses.replace(it, doc=_with_scores(it.doc, out)) for it in items]
    trace["rerank"] = rerank_rec

    trace.update(totals(calls))
    return [it.doc for it in items] + tail, trace


def _with_scores(doc: FusedDoc, out: Outcome) -> FusedDoc:
    if doc.document_id in out.grades:
        return dataclasses.replace(doc, scores={**doc.scores, "rerank": out.grades[doc.document_id]})
    return doc
