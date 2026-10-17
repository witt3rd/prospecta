"""Fixture banks that mirror the four live banks, for the tests that pin what removing the retired
recall code must not change (tests/test_retrieval_golden.py, tests/test_retired_code.py).

The four banks as they were stored: `augur` and `forge` are old (channel_config `[]`, recall_config `{}`,
link_config `{}`: the column defaults) and hold question items beside their chunks and the rows of the linker
that once ran (entities, aliases, links, link state); `augur-v2` and `forge-v2` hold chunk items only, with
`hybrid_chunk` and a cross_encoder rerank set by hand and link_config `{}`. The code that wrote the old rows is
gone, so the fixture writes them raw. Invented text, the stub embedder, a fake reranker and a scripted
model: nothing here is a real note."""
from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import date

import psycopg

from prospecta.memory import Memory
from tests._stub_embedder import EMBED_DIM

TODAY = date(2026, 6, 1)

HYBRID = [{"name": "hybrid_chunk", "enabled": True, "weight": 1, "params": {}}]
HAND_SET_RECALL = {"rerank": {"enabled": True, "stage": "cross_encoder", "model": "qwen/qwen3-reranker-8b",
                              "top_n": 10, "min_score": 0.15, "timeout_s": 2.5,
                              "deduplicate_near_copies": True}}

# (source, created_on, kind, person, text)
CORPUS = {
    "augur": [
        ("harbour/lamps-decision.md", "2026-03-05", "decision", "Dan Maycock",
         "We decided the harbour lamps get a warm filter. The decision stands until the winter storm review."),
        ("harbour/lamps-incident.md", "2026-04-01", "note", "Dan Maycock",
         "The bridge lamps flickered for two nights; the night crew traced it to the old relay in cabinet 7."),
        ("harbour/electric-dreams.md", "2026-04-02", "note", "Mira Okafor",
         "The night crew called the bridge lamps electric dreams. Nobody remembers who started it."),
        ("harbour/pontoon-survey.md", "2026-02-10", "note", "Dan Maycock",
         "Dan Maycock said the pontoon survey is due in May. The survey covers pontoons 3 to 9."),
        ("harbour/pool-limit.md", "2025-11-11", "note", None,
         "PROSPECTA_DB_POOL_MAX caps the database connections of the gateway. Raise it only with the DBA."),
        ("harbour/rollout-v114.md", "2026-05-20", "report", "Mira Okafor",
         "The updater stopped with error 0x80070005 after the v1.14 rollout. Rolling back to v1.13 fixed it."),
        ("harbour/rollout-v115.md", "2026-05-27", "report", "Mira Okafor",
         "Release v1.15 notes mention error 0x80070006 only. The rollout is scheduled for June."),
        ("harbour/storm-review.md", "2026-01-15", "decision", "Dan Maycock",
         "The winter storm review moved the lamp inspection to October and added a spare relay to the stores."),
        ("harbour/orchids.md", "2025-09-02", "note", None,
         "Orchids need humid air and a shaded window. The greenhouse fan runs on a timer."),
        ("harbour/tree-view.md", "2025-12-09", "note", "Ines Varga",
         "The library tree view loads folders lazily. Large folders page in, two hundred entries at a time."),
        ("harbour/lamps-decision-copy.md", "2026-03-06", "note", "Dan Maycock",
         "We decided the harbour lamps get a warm filter. The decision stands until the winter storm review, as the crew agreed."),
        ("harbour/budget.md", "2026-02-28", "decision", "Ines Varga",
         "The budget for the pontoon survey is capped at four thousand; Ines Varga signs any overrun."),
        ("harbour/crew-rota.md", "2026-05-02", "note", "Mira Okafor",
         "The night crew rota alternates weekly. Mira Okafor swaps shifts on request; no more than two swaps a month."),
        ("harbour/relay-spec.md", "2026-04-15", "note", None,
         "Relay model R-220 is rated for 24 V. The cabinet 7 relay was a R-180 and ran hot."),
    ],
    "forge": [
        ("shop/kiln-schedule.md", "2026-02-03", "note", "Tomas Reyes",
         "The kiln runs the glaze firing on Thursdays. Cone 6 holds for twelve minutes."),
        ("shop/glaze-recipe.md", "2026-02-11", "decision", "Tomas Reyes",
         "We decided the celadon glaze uses 4 percent iron. The recipe is frozen until the autumn batch."),
        ("shop/kiln-fault.md", "2026-03-18", "report", "Lena Sorokina",
         "Kiln 2 tripped error E-41 mid firing. The thermocouple was replaced and the schedule rerun."),
        ("shop/order-4412.md", "2026-04-04", "note", "Lena Sorokina",
         "Order 4412 is forty celadon mugs for the harbour cafe, due before the lamp festival."),
        ("shop/clay-supplier.md", "2025-10-12", "note", None,
         "The clay supplier delivers on the first Monday. Stoneware body 5 is out of stock until November."),
        ("shop/electric-dreams.md", "2026-04-20", "note", "Tomas Reyes",
         "The mug line called electric dreams has a turquoise rim. Lena Sorokina painted the first batch."),
        ("shop/wheel-service.md", "2026-01-09", "note", "Tomas Reyes",
         "The wheel motor was serviced in January; the next service is due in July."),
        ("shop/glaze-recipe-copy.md", "2026-02-12", "note", "Tomas Reyes",
         "We decided the celadon glaze uses 4 percent iron. The recipe is frozen until the autumn batch, as the potters agreed."),
        ("shop/safety.md", "2025-12-01", "decision", None,
         "Respirators are mandatory when mixing dry glaze. The extractor must run before the lid comes off."),
        ("shop/market-day.md", "2026-05-09", "note", "Lena Sorokina",
         "Market day sold thirty mugs and six bowls; the electric dreams rim sold out first."),
        ("shop/kiln-furniture.md", "2026-03-02", "note", "Tomas Reyes",
         "Kiln shelves are cordierite, not silicon carbide. Never stack above the third post."),
    ],
}

QUERIES = {
    "augur": ["what did we decide about the harbour lamps", "how does the updater error 0x80070005 get fixed",
              '"electric dreams"', "who approves the budget overrun", "Dan Maycock survey",
              "relay in cabinet 7"],
    "forge": ["what did we decide about the celadon glaze", "kiln tripped error E-41",
              '"electric dreams"', "order 4412 mugs", "Tomas Reyes kiln", "supplier delivery stoneware"],
}


def embed(texts: list[str]) -> list[list[float]]:
    """A deterministic embedder with no ties: every word is a fixed pseudo-random unit-ish vector, a text
    is their normalised sum. (The stub embedder's bag of buckets scores many unrelated notes the same,
    and the HNSW index orders equal scores arbitrarily: a golden file cannot pin that.)"""
    out = []
    for text in texts:
        vec = [0.0] * EMBED_DIM
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            raw = hashlib.sha256(tok.encode()).digest()
            for i in range(EMBED_DIM):
                vec[i] += (raw[i] - 127.5) / 127.5
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        out.append([v / norm for v in vec])
    return out


class FakeRerank:
    """A rerank transport: 0.2 + 0.2 per word the document shares with the query, at most 0.99."""

    def __init__(self):
        self.lock = threading.Lock()
        self.requests = 0

    def __call__(self, req, timeout):
        with self.lock:
            self.requests += 1
        query = req["query"].split("Query:", 1)[-1].strip()
        asked = set(re.findall(r"\w{3,}", query.lower()))
        results = [{"index": i, "relevance_score": min(0.99, 0.2 + 0.2 * len(
            asked & set(re.findall(r"\w{3,}", d.lower()))))} for i, d in enumerate(req["documents"])]
        return {"model": req["model"], "results": results, "usage": {"total_tokens": 10, "cost": 0.0}}


class ScriptedLLM:
    """One model for every role: it tells the roles apart by the first words of the prompt."""

    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, messages, *, json_mode=False, model=None):
        prompt = messages[-1]["content"]
        if prompt.startswith("You are formulating retrieval queries"):
            self.calls.append("formulate")
            words = re.findall(r"[A-Za-z]{5,}", prompt.split("## Input", 1)[-1])[:2] or ["memory"]
            return json.dumps({"queries": [{"text": f"what do we know about {w.lower()}"} for w in words]})
        if prompt.startswith("You write searches of a team's memory"):
            self.calls.append("expand")
            return json.dumps({"searches": [
                {"query": "the decision about the lamps or the glaze", "terms": [], "date_from": None, "date_to": None},
                {"query": "the error and the rollout", "terms": ['"electric dreams"', "0x80070005"],
                 "date_from": "2026-03-01", "date_to": "2026-06-01"}]})
        if prompt.startswith("You are answering a question from numbered evidence"):
            self.calls.append("ask")
            return "The decision is recorded in the first excerpt [1]."
        self.calls.append("synthesize")
        return "A synthesis from the chunks."


def connect(url):
    return psycopg.connect(url, autocommit=True)


def set_stored_config(url: str, bank: str, channel, recall, link) -> None:
    """A bank as an older install stored it: its three config columns written raw."""
    with connect(url) as c:
        c.execute("UPDATE banks SET channel_config = %s::jsonb, recall_config = %s::jsonb, "
                  "link_config = %s::jsonb WHERE bank_id = %s",
                  (json.dumps(channel), json.dumps(recall), json.dumps(link), bank))


def stored_config(url: str, bank: str) -> tuple:
    with connect(url) as c:
        return c.execute("SELECT channel_config, recall_config, link_config FROM banks WHERE bank_id = %s",
                         (bank,)).fetchone()


def make_memory(url: str, bank: str, **kw) -> Memory:
    return Memory(database_url=url, bank_id=bank, embed=embed, **kw)


def _question(source: str, text: str) -> list[str]:
    """The anticipated question a model once wrote for a note of an old bank. One per note, naming its note."""
    stem = re.sub(r"[^a-z0-9]+", " ", source.rsplit("/", 1)[-1].lower())
    words = re.findall(r"[A-Za-z]{6,}", text)[:3]
    return [f"What do we know about {' '.join(w.lower() for w in words)} in {stem}?"]


def seed_question_items(url: str, bank: str) -> None:
    """The `kind = 'question'` items an old bank holds: written raw, as the removed write path wrote them."""
    with connect(url) as c:
        docs = c.execute("SELECT id, source, original_text FROM documents WHERE bank_id = %s ORDER BY source",
                         (bank,)).fetchall()
        for doc_id, source, text in docs:
            for question in _question(source, text):
                vec = embed([question])[0]
                lit = "[" + ",".join(repr(float(x)) for x in vec) + "]"
                c.execute(
                    "INSERT INTO memory_items (bank_id, document_id, content, original_chunk, embedding, metadata, "
                    "tags, update_mode, llm_generated, kind) VALUES (%s, %s, %s, %s, %s::vector, "
                    "'{\"index_text_caller_supplied\": false}'::jsonb, '{}', 'append', true, 'question')",
                    (bank, doc_id, question, text, lit))


def build_bank(url: str, bank: str, corpus: str, *, legacy: bool, **memory_kw) -> Memory:
    """One bank as stored. Old (`legacy=True`): chunks, question items, config `[]`/`{}`/`{}` and the rows of a
    linker that once ran. Current: chunk items, hybrid_chunk + cross_encoder set by hand."""
    mem = make_memory(url, bank, **memory_kw)
    mem.create_bank(bank, embedding_dim=EMBED_DIM)
    for source, created_on, kind, person, text in CORPUS[corpus]:
        meta = {"created_on": created_on, "source_kind": kind, "project": "atlas"}
        if person:
            meta["person"] = person
        mem.retain(text, source=source, metadata=meta)
    if legacy:
        seed_question_items(url, bank)
        set_stored_config(url, bank, [], {}, {})
        seed_retired_rows(url, bank, "Mira Okafor" if corpus == "augur" else "Lena Sorokina")
    else:
        set_stored_config(url, bank, HYBRID, HAND_SET_RECALL, {})
    return mem


def seed_retired_rows(url: str, bank: str, person: str = "Mira Okafor") -> None:
    """The rows a bank that once ran the linker holds: entities, aliases, which item mentions which,
    links between items, link state, the pair cache. Written raw (the code that wrote them is gone)."""
    with connect(url) as c:
        docs = c.execute("SELECT id, source, original_text FROM documents WHERE bank_id = %s ORDER BY source",
                         (bank,)).fetchall()
        ent = c.execute("INSERT INTO memory_entities (bank_id, name, norm, etype) VALUES (%s, %s, %s, 'person') "
                        "RETURNING id", (bank, person, person.lower())).fetchone()[0]
        c.execute("INSERT INTO memory_entity_aliases (bank_id, norm, alias, entity_id) "
                  "VALUES (%s, 'mo', 'MO', %s)", (bank, ent))
        prev = None
        for doc_id, source, text in docs:
            items = [r[0] for r in c.execute(
                "SELECT id FROM memory_items WHERE document_id = %s ORDER BY created_at, id",
                (doc_id,)).fetchall()]
            for item in items:
                if person.split()[0] in text:
                    c.execute("INSERT INTO memory_item_entities (item_id, entity_id) VALUES (%s, %s)", (item, ent))
                if prev is not None and prev != item:
                    c.execute("INSERT INTO memory_links (bank_id, src, dst, link_type, subtype) "
                              "VALUES (%s, %s, %s, 'TEMPORAL', 'NEXT')", (bank, prev, item))
                    c.execute("INSERT INTO memory_link_pairs (pair_hash, bank_id, src, dst) VALUES (%s, %s, %s, %s)",
                              (hashlib.sha256(f"{prev}{item}".encode()).hexdigest(), bank, prev, item))
                prev = item
            c.execute("INSERT INTO memory_link_state (document_id, bank_id, status) VALUES (%s, %s, 'linked')",
                      (doc_id, bank))
            c.execute("INSERT INTO memory_alias_state (document_id, bank_id, status) VALUES (%s, %s, 'done')",
                      (doc_id, bank))


def retired_row_counts(url: str, bank: str) -> dict:
    out = {}
    with connect(url) as c:
        for table in ("memory_entities", "memory_entity_aliases", "memory_links", "memory_link_state",
                      "memory_link_pairs", "memory_alias_state"):
            out[table] = c.execute(f"SELECT count(*) FROM {table} WHERE bank_id = %s", (bank,)).fetchone()[0]
        out["memory_item_entities"] = c.execute(
            "SELECT count(*) FROM memory_item_entities e JOIN memory_items m ON m.id = e.item_id "
            "WHERE m.bank_id = %s", (bank,)).fetchone()[0]
    return out
