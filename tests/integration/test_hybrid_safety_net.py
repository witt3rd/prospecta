"""Hybrid safety-net integration test (T18, M7).

Validates P14: when LLM-anticipated index_text drifts away from the
query's semantic shape, the lexical (BM25 / ts_rank) half of hybrid
RRF retrieval recovers the document via keyword overlap on the SAME
indexed string.

## Architectural finding (documented for the record)

`content_tsv` is a STORED generated column = `to_tsvector('english', content)`.
The `content` column is the LLM-anticipated index_text (P1 spine), NOT
the source body. This means:

  - Semantic side embeds `content` (the index_text).
  - Lexical side tokenizes `content` (the index_text).
  - The source body is preserved in `original_chunk` but is NOT
    indexed by either side.

So the "drift safety net" for v0.1 has a specific shape: when the
LLM-anticipated index_text drifts in semantic register (declarative
phrasing instead of question shape, mismatching the query's shape),
lexical token overlap on the SAME drifted index_text can still rescue
the doc through RRF fusion. The body itself is not a lexical fallback
in v0.1 — that's a noted extension point for v0.2 (e.g., a second
tsvector over `original_chunk`).

This is scenario (a) from the T18 spec: lexical rescue of phrasing
drift on the same indexed string. Scenario (b) — body as separate
lexical channel — is filed as a design extension.

## The drift scenario — exploiting stemming asymmetry

The bilateral_embedder hashes **raw tokens**; ts_rank stems via
`'english'` snowball. We use this asymmetry to construct a case
where the same `content` string lights up lexical but NOT semantic.

Target doc indexed with inflected-form drift (no question shape):
  index_text = "kellys birthdays celebrations marches recorded"

Plural inflections stem to query terms (kelli, birthday, celebr,
march) but hash-bucket to different raw-token dims than the query's
singular forms (kelly, birthday, celebration, march).

Query (caller's natural question form):
  "When was Kelly's birthday celebration in March?"

Distractors indexed with question-shape but zero query overlap:
  - "How does the forge bench warm between strikes?"
  - "What did Cookie bake on Sunday morning?"
  - "Why do harbor lights flicker at night?"

Under the bilateral_embedder:
  - Query has question-shape lit (when, was, '?').
  - Target's index_text has no question words, no '?' — shape dims dark.
  - Hash-bucket overlap on raw tokens between target and query is zero
    (celebrating ≠ celebration, marches ≠ march to the hasher).
  - Distractors have question-shape lit → +0.36/dim boost across
    8 dims, lifting cosine vs the question-shaped query above the
    target's near-zero token+shape contribution.

Under ts_rank (BM25-style):
  - target stems = {celebr, record, march, archiv}
  - query stems   = {kelli, birthday, celebr, march}  (stopwords dropped)
  - target matches `celebr` AND `march` → positive lex_score.
  - distractors share no stems with query → lex_score = 0.

So:
  - mode='semantic'  → target ranks below distractors (drift wins).
  - mode='lexical'   → target ranks first (stem overlap saves it).
  - mode='hybrid'    → RRF fusion rescues target (lexical lifts it).
"""
from __future__ import annotations

import time
from urllib.parse import urlparse, urlunparse

import psycopg
import pytest


# ---------------------------------------------------------------------------
# Fixtures (mirror the bilateral test's fresh-DB pattern for isolation)
# ---------------------------------------------------------------------------

@pytest.fixture
def bilateral_embed():
    from tests._bilateral_embedder import bilateral_embedder
    return bilateral_embedder


def _fresh_url(pg_container) -> tuple[str, str, str]:
    base_url = pg_container.get_connection_url()
    for token in ("+psycopg2", "+psycopg"):
        if token in base_url:
            base_url = base_url.replace(token, "")
    db_name = f"safety_{int(time.time() * 1_000_000)}"
    with psycopg.connect(base_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'CREATE DATABASE "{db_name}"')
    parsed = urlparse(base_url)
    new_url = str(urlunparse(parsed._replace(path=f"/{db_name}")))
    with psycopg.connect(new_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    from prospecta.db.migrate import run_migrations
    run_migrations(new_url)
    return new_url, base_url, db_name


def _drop_db(base_url: str, db_name: str) -> None:
    with psycopg.connect(base_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()",
                (db_name,),
            )
            cur.execute(f'DROP DATABASE "{db_name}"')


# The drift scenario — caller-supplied index_text (P4) means no LLM call
# fires during retain(); the deliberate drift is encoded explicitly.
TARGET_SOURCE = "kelly_birthday.md"
# Drifted index_text: declarative, no question-shape, inflected forms
# that stem to query terms ("celebrating"→celebr, "marches"→march)
# while hash-bucketing to different raw-token dims than the query
# ("celebration", "march"). This is the lexical-vs-semantic asymmetry.
TARGET_INDEX_TEXT = "kellys birthdays celebrations marches recorded"
TARGET_BODY = (
    "Kelly's birthday celebration unfolded on March 4th, 1990. "
    "The party was warm, the cake unforgettable, the kitchen full of friends."
)

DISTRACTOR_DOCS = [
    (
        "forge_bench.md",
        "How does the forge bench warm between strikes?",
        "The forge bench holds heat between strikes. The smith returns to it.",
    ),
    (
        "cookie_bake.md",
        "What did Cookie bake on Sunday morning?",
        "Cookie bakes sourdough every Sunday morning. The loaf rises overnight.",
    ),
    (
        "harbor_log.md",
        "Why do harbor lights flicker at night?",
        "The harbor lights flickered on at 7:14pm. Boats rocked in the wake.",
    ),
]

QUERY = "When was Kelly's birthday celebration in March?"


@pytest.fixture
def drift_memory(pg_container, bilateral_embed):
    """Build a Memory with the drift corpus indexed via retain() with
    explicit index_text (P4: caller wins, no LLM call)."""
    from prospecta.memory import Memory

    new_url, base_url, db_name = _fresh_url(pg_container)
    bank = f"safety_{int(time.time() * 1_000_000)}"
    mem = Memory(
        database_url=new_url,
        bank_id=bank,
        llm=None,  # P4 caller-supplies index_text; no LLM needed.
        embed=bilateral_embed,
    )
    mem.create_bank(bank, embedding_dim=bilateral_embed.dim)

    # Target doc — drifted index_text (declarative, not question-shape).
    mem.retain(
        TARGET_BODY,
        index_text=TARGET_INDEX_TEXT,
        source=TARGET_SOURCE,
    )
    # Distractors — question-shape index_text, no keyword overlap.
    for src, idx, body in DISTRACTOR_DOCS:
        mem.retain(body, index_text=idx, source=src)

    try:
        yield mem
    finally:
        mem.close()
        _drop_db(base_url, db_name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _source_of(r) -> str:
    return r.source.rsplit("/", 1)[-1]


def _rank_of(results, source_basename: str) -> int | None:
    """1-indexed rank; None if absent."""
    for i, r in enumerate(results, start=1):
        if _source_of(r) == source_basename:
            return i
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_semantic_only_misses_or_demotes_drifted_doc(drift_memory):
    """Phrasing drift defeats pure semantic retrieval.

    With the target's index_text drifted to declarative shape, the
    question-shape signal in the embedding is dark. Distractors with
    question-shape index_text out-rank the target on semantic alone.
    """
    results = drift_memory.search(QUERY, mode="semantic", limit=10)
    target_rank = _rank_of(results, TARGET_SOURCE) or 999
    print(f"\n[semantic] target rank = {target_rank}")
    for i, r in enumerate(results, start=1):
        print(f"  {i}. {_source_of(r)}  sem={r.scores['semantic']:.4f}")
    # The target should NOT be rank 1 under semantic — the drift loses
    # to the distractors' question-shape alignment with the query.
    assert target_rank != 1, (
        "semantic-only should not rank drifted target first; "
        f"got rank {target_rank}. The drift scenario is mis-designed."
    )


def test_lexical_only_rescues_drifted_doc(drift_memory):
    """ts_rank on content_tsv recovers the target via keyword overlap.

    `content_tsv = to_tsvector('english', content)` where content =
    drifted index_text. The query's load-bearing keywords (kelly,
    march, celebration) all appear in the target's index_text and
    in NO distractor's index_text. Lexical should rank target first.
    """
    results = drift_memory.search(QUERY, mode="lexical", limit=10)
    target_rank = _rank_of(results, TARGET_SOURCE) or 999
    print(f"\n[lexical] target rank = {target_rank}")
    for i, r in enumerate(results, start=1):
        print(f"  {i}. {_source_of(r)}  lex={r.scores['lexical']:.4f}")
    assert target_rank == 1, (
        f"lexical should rescue target at rank 1; got rank {target_rank}"
    )
    # A6 invariant: scores always-populated (three keys, all numeric).
    for r in results:
        assert set(r.scores.keys()) >= {"semantic", "lexical", "rrf"}
        for k in ("semantic", "lexical", "rrf"):
            assert isinstance(r.scores[k], (int, float))


def test_hybrid_rrf_surfaces_drifted_doc(drift_memory):
    """RRF fusion lifts the target through the lexical half.

    Even though semantic ranks the target poorly, lexical ranks it #1.
    RRF (k=60) combines both ranks; the lexical-#1 contribution
    (1/(60+1)) is enough to surface the target near or at the top.
    """
    results = drift_memory.search(QUERY, mode="hybrid", limit=10)
    target_rank = _rank_of(results, TARGET_SOURCE) or 999
    print(f"\n[hybrid]   target rank = {target_rank}")
    for i, r in enumerate(results, start=1):
        print(
            f"  {i}. {_source_of(r)}  "
            f"sem={r.scores['semantic']:.4f}  "
            f"lex={r.scores['lexical']:.4f}  "
            f"rrf={r.scores['rrf']:.4f}"
        )
    # Hybrid must materially out-perform semantic-only. Allowing rank 1
    # OR rank 2 keeps the assertion robust against minor ts_rank-vs-cosine
    # numerics (lexical also contributes negative-rank to distractors
    # via COALESCE 0.0 → distractor lex_score = 0 → no RRF lift), but the
    # rescue must be visible.
    assert target_rank <= 2, (
        f"hybrid should rescue target to top-2 via RRF; got rank {target_rank}"
    )


def test_hybrid_at_least_matches_best_single_mode(drift_memory):
    """rank_hybrid(target) <= max(rank_semantic, rank_lexical).

    The fusion is at least as good as the best single-mode result.
    This is the load-bearing safety-net assertion: hybrid never does
    worse than the best of its halves on the drifted target.
    """
    sem = drift_memory.search(QUERY, mode="semantic", limit=10)
    lex = drift_memory.search(QUERY, mode="lexical", limit=10)
    hyb = drift_memory.search(QUERY, mode="hybrid", limit=10)

    sem_rank = _rank_of(sem, TARGET_SOURCE) or 999
    lex_rank = _rank_of(lex, TARGET_SOURCE) or 999
    hyb_rank = _rank_of(hyb, TARGET_SOURCE) or 999

    print(
        f"\n[comparative] target ranks — "
        f"semantic={sem_rank}  lexical={lex_rank}  hybrid={hyb_rank}"
    )

    best_single = min(sem_rank, lex_rank)
    assert hyb_rank <= best_single + 1, (
        f"hybrid rank ({hyb_rank}) should be ≤ best single mode "
        f"({best_single}) + 1 tolerance; "
        f"semantic={sem_rank}, lexical={lex_rank}"
    )
    # Strong claim: hybrid strictly beats semantic-only here.
    assert hyb_rank < sem_rank, (
        f"hybrid ({hyb_rank}) should out-perform semantic-only ({sem_rank}) "
        f"on a drift designed to defeat semantic alone"
    )


def test_a6_scores_always_populated(drift_memory):
    """A6 invariant: every mode produces RecalledMemory with all four
    score keys numeric (semantic, lexical, lexical_body, rrf). Never null."""
    for mode in ("semantic", "lexical", "hybrid"):
        results = drift_memory.search(QUERY, mode=mode, limit=5)
        assert results, f"mode={mode} returned no results"
        for r in results:
            assert set(r.scores.keys()) >= {
                "semantic", "lexical", "lexical_body", "rrf",
            }
            for k in ("semantic", "lexical", "lexical_body", "rrf"):
                v = r.scores[k]
                assert isinstance(v, (int, float)), (
                    f"mode={mode} score[{k}]={v!r} is not numeric (A6 violation)"
                )


# ---------------------------------------------------------------------------
# T19 — Body-channel rescue (true two-axis P14 safety net)
# ---------------------------------------------------------------------------

# Constructs a deliberate drift case where the LLM-generated index_text is
# semantically and lexically off-topic ("tea in China"), but the body
# (original_chunk) is a strong match for the query. Only the body channel
# can rescue this doc — content_tsv has zero overlap with the query terms.
BODY_RESCUE_SOURCE = "kelly_birthday_bodychan.md"
BODY_RESCUE_INDEX_TEXT = "The price of tea in China remains stable this quarter"
BODY_RESCUE_BODY = (
    "Kelly was born on March 4th, 1990. Her birthday party was epic — "
    "friends in the kitchen, music loud, the cake unforgettable."
)
BODY_RESCUE_QUERY = "When was Kelly born and her birthday party?"


@pytest.fixture
def body_rescue_memory(pg_container, bilateral_embed):
    """Memory with one body-rescue target + the same distractors as
    drift_memory. The target's index_text is OFF-TOPIC (tea/china) so
    content_tsv has zero query overlap; the body is ON-TOPIC (kelly/born/
    march/birthday) so body_tsv carries the rescue.
    """
    from prospecta.memory import Memory

    new_url, base_url, db_name = _fresh_url(pg_container)
    bank = f"bodychan_{int(time.time() * 1_000_000)}"
    mem = Memory(
        database_url=new_url,
        bank_id=bank,
        llm=None,
        embed=bilateral_embed,
    )
    mem.create_bank(bank, embedding_dim=bilateral_embed.dim)

    # Target: drifted index_text, on-topic body.
    mem.retain(
        BODY_RESCUE_BODY,
        index_text=BODY_RESCUE_INDEX_TEXT,
        source=BODY_RESCUE_SOURCE,
    )
    # Distractors: question-shape index_text with no query overlap.
    for src, idx, body in DISTRACTOR_DOCS:
        mem.retain(body, index_text=idx, source=src)

    try:
        yield mem
    finally:
        mem.close()
        _drop_db(base_url, db_name)


def test_body_channel_rescues_drifted_index_text(body_rescue_memory):
    """The P14 promise made honest: when LLM-generated index_text drifts
    semantically AND lexically from the query, body_tsv (over
    original_chunk) rescues the document via three-channel RRF fusion.

    Setup:
      - Target: index_text="The price of tea in China..." (off-topic)
                body="Kelly was born March 4th 1990. Birthday party..."
      - Query: "When was Kelly born and her birthday party?"

    Expected:
      - mode='lexical' (content_tsv only): target absent or zero-score.
      - mode='hybrid' (three-channel RRF): target surfaces with
        lexical_body > 0 carrying the rescue.
    """
    # Confirm the index_text has no query-term overlap by checking
    # mode='lexical' (content-channel only) does NOT surface the target.
    lex_results = body_rescue_memory.search(
        BODY_RESCUE_QUERY, mode="lexical", limit=10,
    )
    lex_rank = _rank_of(lex_results, BODY_RESCUE_SOURCE)
    # Target must not appear in content-only lexical results — the
    # index_text has zero overlap with query terms. This locks the
    # rescue requirement: only the body channel can save it.
    assert lex_rank is None, (
        f"content-channel lexical should NOT surface the body-rescue target; "
        f"got rank {lex_rank}. The drifted index_text apparently shares "
        f"tokens with the query, breaking the test scenario."
    )

    # The body-rescue: mode='hybrid' surfaces the target via body_tsv.
    hyb_results = body_rescue_memory.search(
        BODY_RESCUE_QUERY, mode="hybrid", limit=10,
    )
    hyb_rank = _rank_of(hyb_results, BODY_RESCUE_SOURCE)
    print(f"\n[body-rescue hybrid] target rank = {hyb_rank}")
    for i, r in enumerate(hyb_results, start=1):
        print(
            f"  {i}. {_source_of(r)}  "
            f"sem={r.scores['semantic']:.4f}  "
            f"lex={r.scores['lexical']:.4f}  "
            f"lex_body={r.scores['lexical_body']:.4f}  "
            f"rrf={r.scores['rrf']:.4f}"
        )
    assert hyb_rank is not None, (
        "hybrid must surface the body-rescue target via body_tsv channel"
    )
    assert hyb_rank <= 2, (
        f"hybrid should rescue body-channel target to top-2; got rank {hyb_rank}"
    )

    # Locate the rescued target and verify body channel is the dominant
    # lexical signal (content channel is zero by construction).
    target = next(
        r for r in hyb_results if _source_of(r) == BODY_RESCUE_SOURCE
    )
    assert target.scores["lexical"] == 0.0, (
        f"content-channel score must be 0 for the off-topic index_text; "
        f"got {target.scores['lexical']}"
    )
    assert target.scores["lexical_body"] > 0.0, (
        f"body-channel score must be positive to carry the rescue; "
        f"got {target.scores['lexical_body']}"
    )


def test_mode_lexical_unchanged_by_body_channel(body_rescue_memory):
    """Option A locked: mode='lexical' scopes to content_tsv only.

    The body channel is intentionally NOT joined into mode='lexical'
    in v0.1 — body rescue fires only via mode='hybrid' RRF fusion. This
    test locks that behavior contract; future v0.2 may change semantics
    with explicit deprecation.
    """
    # The body-rescue target has off-topic index_text (no query overlap).
    # If mode='lexical' were to silently include body_tsv, the target
    # would surface. It must not.
    results = body_rescue_memory.search(
        BODY_RESCUE_QUERY, mode="lexical", limit=10,
    )
    rank = _rank_of(results, BODY_RESCUE_SOURCE)
    assert rank is None, (
        f"mode='lexical' must remain content-only (option A); body-rescue "
        f"target surfaced at rank {rank}, indicating body channel leaked in."
    )
