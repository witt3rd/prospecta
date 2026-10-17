# PostgreSQL Hybrid Search & RAG Architecture for Prospecta

Prospecta's hybrid search and RAG engine serves two primary operating modes on top of a single PostgreSQL + `typesafe/jev-1.14` + `qwen/qwen3-reranker-8b` substrate:

1. **Automatic Recall for Agent Operations** (Augur, Forge, Spire):
   - **Conversational turns (1:1 or group chats):** Proactive pre-turn memory gathering triggered by a new message in an ongoing thread.
   - **Non-interactive task-based prompts (commissions, background jobs, autonomous wakes):** Proactive pre-task context gathering from a task brief, goal, and acceptance criteria before the worker begins.
2. **Interactive Search** (Human UI search bars, `#` Library / artifact pickers, Find, and explicit agent search tools):
   - **Direct lookup & navigation:** Keyword, boolean (`OR`, `-exclude`), and `"exact phrase"` search with highlighted exact-match snippets (`ts_headline`).
   - **Interactive RAG / Question-Answering:** Natural-language questions answered with grounded synthesis, inline citations (`[source_id, date]`), and verbatim supporting quotes.

Across **all** modes, every retrieved or synthesized answer is **RAG with citations**, accompanied by **exact-match snippets** whenever lexical or phrase tokens match. We index **chunks and notes directly** (embeddings + stemmed/exact lexical vectors + metadata) and do not generate or carry model-written question tables at write time.

---

## 1. Use Cases & Best Practices: What the Engine Must Serve

Designing the SQL schema, stages, and output format starts from how each caller consumes the result:

### Mode 1A: Automatic Recall in Conversational Turns (Augur, Forge, Spire Chats)

In a live chat, the conversation history already holds the recent exchange. Automatic recall exists to surface **what is not already in the active window**.

* **Gate aggressively (`jev-1.14` Stage 0):** Skip retrieval (`0ms` DB/rerank cost) on acknowledgements (`"ok"`, `"ship it"`), greetings, self-contained instructions, and questions about live system state that must be read from live tools rather than memory.
* **De-contextualize follow-ups using only the immediate exchange:** A user message like *"do you remember who I saw it with?"* cannot be searched raw. Resolve pronouns (`"it"`) against the **last assistant turn** (not the last 10 turns, which drowns the query in topic drift) to produce 1–3 standalone search queries, **preserving any `"quoted strings"` and proper nouns verbatim**.
* **Exclude the active context window (Anti-Echo Invariant):** Never return notes or turn records that came from the last 2 turns of the current session—the agent already has them in front of it. Older turns from the same long-lived session remain eligible.
* **Deduplicate near-identical notes before packing:** Repeated summary notes about the same event collapse to the single highest-scored / most complete note, citing its date range.
* **Inject as a cited, non-instructional evidence block:** Every recalled item carries its **citation header** (`[source_id | kind | YYYY-MM-DD | score]`), the **full relevant chunk/note**, and, when an exact/phrase match fired, the **verbatim highlighted snippet**.

### Mode 1B: Automatic Recall in Non-Interactive Task Prompts (Commissions, Wakes, Background Jobs)

When a Spire specialist, SWE, Researcher, or background wake starts a task, there is no prior chat turn—instead, there is a **structured task brief** (`title`, `ask` / `goal`, `done` criteria, and `project_id`).

* **No `needs_memory` skip:** A non-interactive task almost always benefits from prior project decisions, architecture notes, and recent related work.
* **Faceted multi-query decomposition (2–4 queries in one batch):** Decompose the task brief into distinct retrieval facets executed in a single batched Stage 1 + Stage 2 pass:
  1. *Domain / subject query:* core entities, subsystem names, or `"exact identifiers"` named in the ask.
  2. *Prior decisions & conventions query:* standing rulings or design notes governing that subsystem.
  3. *Recent related work query:* prior commissions, reports, or incidents touching the same area.
* **Scope-first with labeled cross-project recall:** Prioritize the task's own `project_id` / `bank_id`, while allowing high-relevance cross-project notes (clearly labeled with their origin project) to surface.
* **Cited RAG brief at task start:** Deliver a structured `## Recalled Project Context` block where every claim or excerpt has a clickable/fetchable citation (`[note_id / doc_path, YYYY-MM-DD]`) and exact-match snippets for any code symbols, error strings, or quoted terms in the brief.

### Mode 2: Interactive Search (Human Search Bars, `#` Pickers, Explicit Agent Search)

Interactive search serves two user intents that share the same Stage 1 + Stage 2 pipeline:

| Intent | Trigger / Input Shape | Pipeline Behavior & Output Contract |
| --- | --- | --- |
| **2A. Direct Lookup & Filter** (`electric dreams`, `"error 0x80070005"`, `augur OR forge -shadow`) | Short keyword queries, quoted phrases, identifiers, `#` picker typeahead, or explicit filter operators. | **Skip Stage 0 Jev** (user explicitly pressed search). Run Stage 1 (Postgres hybrid) + Stage 2 (`qwen3-reranker-8b`, or fast Stage 1 RRF for `<100ms` typeahead). Return ranked records with **citation metadata** (`title`, `source_id`, `created_at`, `category`) and **exact-match snippets** (`ts_headline` showing the matched tokens/phrase in context). |
| **2B. Interactive Question-Answering (RAG)** (`"what did we decide about the library tree view and why?"`) | Natural-language question in a search bar, "Ask the Library", or an agent's explicit `recall` / `search` tool call. | Run Stage 1 ($\rightarrow$ 50 candidates) + Stage 2 (`qwen3-reranker-8b` $\rightarrow$ top 5–10 chunks) + **Stage 3 (`jev-1.14` sufficiency gate, if multi-hop)**. Synthesize a grounded answer where **every factual statement carries an inline citation** (`[1]`, `[2]`) linked to the source record, its date, and its **verbatim supporting snippet** (with exact-match highlights when lexical terms matched). |

---

## 2. Pipeline Architecture: Four Stages, Clear Roles

Each stage does one job that the other stages cannot do well:

| Stage | Component | Used In | Role | Typical Latency | Cost |
| --- | --- | --- | --- | --- | --- |
| **Stage 0: Gate, Filter & Query Formation** | `typesafe/jev-1.14` (`/api/alpha/decisions`) + fast query de-contextualizer | Mode 1A (Chat), Mode 1B (Task decomposition) | Decide `needs_memory` (Mode 1A), extract structured SQL filters (date window, category), and form 1–4 standalone search queries while keeping `"quoted phrases"` and identifiers intact. Skipped in Mode 2A. | ~150–300ms | $0.042 / 1M input ($0 output) |
| **Stage 1: Hybrid Candidate Generation + Snippets** | PostgreSQL (`pgvector` HNSW + dual `tsvector` GIN + `ts_headline`) | All modes | Narrow tens of thousands of notes/chunks down to **~50 candidates** per query using **Reserved-Slot Union + RRF**, and extract **exact-match snippets** (`ts_headline`) whenever `fts_exact` or `fts_en` matches. | ~5–25ms | $0 |
| **Stage 2: Cross-Encoder Precision Reranking** | `qwen/qwen3-reranker-8b` (`/api/v1/rerank`) | All modes (optional fast-path skip for `<100ms` keystroke typeahead) | Score all ~50 `(query, candidate)` pairs with full cross-attention across a 41K context window; return calibrated `relevance_score`s in `[0, 1]`, deduplicate near-copies, and keep the top chunks above floor. Automatic fallback to Stage 1 RRF order on provider error. | ~230–450ms | $0.20 / 1M tokens (~$0.001–$0.003/call) |
| **Stage 3: Sufficiency Gate & Cited RAG** | `typesafe/jev-1.14` (`noul`) + cited answer/brief formatter | Mode 2B (Interactive QA), Agentic multi-hop search | Check whether top reranked chunks suffice (`noul < 0.70` triggers a broadened retry), then format the cited RAG response or evidence block with `[source_id, date]` citations and exact-match snippets. | ~120ms (gate) | $0.042 / 1M input |

### Core Design Invariants

1. **Cross-encoding makes RRF weight-tuning redundant:** When Stage 2 passes all ~50 candidates to `qwen3-reranker-8b` and re-scores every `(query, chunk)` pair from scratch, the internal order inside the top 50 before Stage 2 does not matter—only **membership in the top 50** matters.
2. **Reserved quotas beat pure RRF for candidate admission:** In pure RRF (`1 / (60 + rank)`), a document that ranks #1 in exact lexical search (`1/61 = 0.0164`) but misses semantic search loses to a mediocre chunk that ranks #15 in both (`2/75 = 0.0267`). Guaranteeing reserved candidate slots per retriever (top 25 semantic, top 20 stemmed lexical, top 20 exact/phrase lexical, deduped to ~50) ensures exact hits always reach the cross-encoder.
3. **Every output carries citations and exact-match snippets:** Neither interactive search nor automatic recall ever returns unattributed prose. Every chunk/note is bound to its `source_id`, `created_at` date, `category`, `relevance_score`, and—whenever lexical or quoted-phrase matching fired—a verbatim `exact_snippet` showing the matched words in context.

---

## 3. Lexical Engine, Query Syntax & Exact-Match Snippets

We use **Dual-Config Native PostgreSQL FTS (`english` + `simple`)** with GIN indexes—zero third-party C/Rust extensions required, portable across local Docker and managed cloud Postgres:

| Column | Configuration | What It Catches |
| --- | --- | --- |
| `fts_en` | `to_tsvector('english', content)` | Stemmed keywords (`running` $\leftrightarrow$ `run`), natural-language terms, boolean operators. |
| `fts_exact` | `to_tsvector('simple', content)` | Unstemmed exact tokens, proper names (`dan maycock`), IDs, flags, version strings (`v1.14`), and quoted exact phrases without stemmer distortion. |

**Implementation names.** In Prospecta's schema `fts_en` is `memory_items.body_tsv` (migration 0002) and `fts_exact` is `memory_items.exact_tsv`, a STORED generated column (migration 0020) holding the keyword word rule over `original_chunk` (the `simple` configuration plus the pieces of a path split at slashes; defined once as `indexed_expression()` in `prospecta/db/keyword_index.py`), with its own GIN index. Adding the column rewrites the table: `docs/exact-tsv.md` is the operator runbook.

### Supported User & Agent Query Syntax (`websearch_to_tsquery`)

Never pass raw strings into `to_tsquery()`. Always use `websearch_to_tsquery()`, which never throws syntax errors on punctuation and natively supports standard search operators:

* **Keyword search:** `electric dreams` (implicit `AND` in `fts_exact`, unstemmed; see the plain-question rule below for `fts_en`)
* **Exact phrase match (case-insensitive):** `"electric dreams"` (enforces token adjacency via `<->`; evaluated against `fts_exact` so words inside quotes are never stemmed)
* **Boolean `OR`:** `augur OR forge`
* **Exclusion (`-`):** `memory -shadow`

**Plain questions are an OR in the stemmed retriever.** `websearch_to_tsquery` ANDs every word, so a natural-language question ("which harbour lamps did the crew repaint before the winter storm") matched `fts_en` only in a chunk that held every one of its words. When the query has **no operator** (no straight double quote, no standalone upper-case `OR`, no word led by `-`), the stemmed retriever matches the **OR** of the query's stemmed words (`english`; stop words drop; at most the 8 longest words, since cover-density ranking costs more with every query word) and ranks by `ts_rank_cd`, so coverage orders the candidates and Stage 2 decides. The words are taken from the parser, never spliced from the query text, so no text can add an operator. The exact retriever always keeps websearch semantics (words ANDed as written, quoted phrases adjacent), and a query **with** an operator keeps websearch semantics on both retrievers: the search-box syntax above is exactly as documented.

**Date-only questions.** When the query has no searchable words (its `english` reading is empty: blank, punctuation, only stop words) and a date window is given, there is nothing to match: Stage 1 returns the chunks inside the window, newest note first (`COALESCE(created_on, created_at::date)`, then the note created last), up to `fused_limit` rows with `match_kind` `date`, and Stage 2 reranks them. Every other pushed-down filter still applies. No words and no window is empty.

### Exact-Match Snippet Extraction (`ts_headline`)

Whenever a candidate matches `fts_exact` or `fts_en`, PostgreSQL extracts the exact matching passage window at query time using `ts_headline()` on the **winning top candidates only** (after `LIMIT 50`, so headline generation never runs over the full table):

```sql
-- Prefer the unstemmed 'simple' match snippet when fts_exact matched;
-- fall back to the stemmed 'english' snippet when only fts_en matched.
CASE
    WHEN f.exact_rank IS NOT NULL THEN
        ts_headline(
            'simple',
            c.content,
            websearch_to_tsquery('simple', $2),
            'StartSel=<<, StopSel=>>, MaxWords=35, MinWords=15, MaxFragments=2, FragmentDelimiter=" … "'
        )
    WHEN f.stem_rank IS NOT NULL THEN
        ts_headline(
            'english',
            c.content,
            websearch_to_tsquery('english', $2),
            'StartSel=<<, StopSel=>>, MaxWords=35, MinWords=15, MaxFragments=2, FragmentDelimiter=" … "'
        )
    ELSE NULL
END AS exact_snippet
```

---

## 4. Schema & Indexing Architecture

At our corpus scale (5K–100K notes/chunks per bank), standard `vector(1536)` HNSW indexes fit comfortably in RAM (~450 MB for 73K vectors). Neither `ParadeDB` nor table partitioning is needed; composite GIN indexes and `pgvector 0.8+` iterative scans keep Stage 1 under 25ms.

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS btree_gin;

CREATE TABLE rag_chunks (
    id BIGSERIAL PRIMARY KEY,
    bank_id TEXT NOT NULL,
    source_id TEXT NOT NULL,          -- Durable citation target (note_id, turn_id, doc_path)
    session_id TEXT,                  -- Used by Mode 1A anti-echo filter
    title TEXT,
    category TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,  -- Mandatory for citation dating and time_scope filtering
    content TEXT NOT NULL,
    -- 1. Stemmed lexical vector
    fts_en TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    -- 2. Unstemmed exact lexical vector (preserves identifiers, versions, exact phrases)
    fts_exact TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
    -- 3. Dense bi-encoder embedding
    embedding VECTOR(1536) NOT NULL
);

-- Vector Index (HNSW)
CREATE INDEX idx_rag_embedding ON rag_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- GIN Indexes for Stemmed and Exact FTS
CREATE INDEX idx_rag_fts_en ON rag_chunks USING gin (bank_id, fts_en);
CREATE INDEX idx_rag_fts_exact ON rag_chunks USING gin (bank_id, fts_exact);

-- B-Tree Index for Bank + Date Range Pre-Filtering
CREATE INDEX idx_rag_bank_date ON rag_chunks (bank_id, created_at DESC);
```

> **Scale note (>500K chunks):** If a single table grows past ~500K chunks and HNSW memory pressure appears, switch `VECTOR(1536)` to `HALFVEC(1536)` (`halfvec_cosine_ops`) to cut HNSW RAM usage by 50% with <1% recall loss, and `LIST` partition by `bank_id`.

---

## 5. Solving the Vector Pre-Filtering Trap (`pgvector 0.8+`)

Combining `WHERE bank_id = $1 AND created_at >= $2` with an HNSW index historically caused **post-filtering under-recall** (Postgres fetched the global top-$K$ vectors and then filtered out rows from other banks or dates).

Always enable **Iterative Index Scans** inside the transaction before running filtered vector queries:

```sql
SET LOCAL hnsw.iterative_scan = relaxed_order;
SET LOCAL hnsw.ef_search = 100;
SET LOCAL hnsw.max_scan_tuples = 20000;
```

Push all structured `WHERE` filters (`bank_id`, `created_at` bounds, `category`, and the Mode 1A `excluded_source_ids` anti-echo list) into **every** CTE (semantic, stemmed lexical, exact lexical) before combining candidates.

---

## 6. Production SQL Pattern: Reserved-Slot Hybrid Query with Citations & Snippets

This single query serves all modes (Mode 1A chat recall, Mode 1B task recall, Mode 2A/2B interactive search). It fetches up to ~50 deduplicated candidates across semantic, stemmed lexical, and exact/phrase lexical retrievers, attaches citation metadata and `exact_snippet` highlights, and computes an `rrf_score` fallback:

```sql
BEGIN;

SET LOCAL hnsw.iterative_scan = relaxed_order;
SET LOCAL hnsw.ef_search = 100;

WITH
-- 1. Semantic Retriever (Top 25 by cosine distance)
semantic_hits AS (
    SELECT
        id,
        ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS rank_ix
    FROM rag_chunks
    WHERE bank_id = $3
      AND ($4::timestamptz IS NULL OR created_at >= $4)
      AND ($5::timestamptz IS NULL OR created_at <= $5)
      AND ($6::text[] IS NULL OR NOT (source_id = ANY($6::text[]))) -- Anti-echo exclusion
    ORDER BY embedding <=> $1::vector
    LIMIT 25
),
-- 2. Stemmed Lexical Retriever (Top 20 via 'english' websearch_to_tsquery)
stemmed_hits AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            ORDER BY ts_rank_cd(fts_en, websearch_to_tsquery('english', $2)) DESC,
                     created_at DESC
        ) AS rank_ix
    FROM rag_chunks
    WHERE bank_id = $3
      AND ($4::timestamptz IS NULL OR created_at >= $4)
      AND ($5::timestamptz IS NULL OR created_at <= $5)
      AND ($6::text[] IS NULL OR NOT (source_id = ANY($6::text[])))
      AND fts_en @@ websearch_to_tsquery('english', $2)
    ORDER BY ts_rank_cd(fts_en, websearch_to_tsquery('english', $2)) DESC,
             created_at DESC
    LIMIT 20
),
-- 3. Unstemmed Exact / Quoted-Phrase Retriever (Top 20 via 'simple' websearch_to_tsquery)
exact_hits AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            ORDER BY ts_rank_cd(fts_exact, websearch_to_tsquery('simple', $2), 4) DESC,
                     created_at DESC
        ) AS rank_ix
    FROM rag_chunks
    WHERE bank_id = $3
      AND ($4::timestamptz IS NULL OR created_at >= $4)
      AND ($5::timestamptz IS NULL OR created_at <= $5)
      AND ($6::text[] IS NULL OR NOT (source_id = ANY($6::text[])))
      AND fts_exact @@ websearch_to_tsquery('simple', $2)
    ORDER BY ts_rank_cd(fts_exact, websearch_to_tsquery('simple', $2), 4) DESC,
             created_at DESC
    LIMIT 20
),
-- 4. Union all reserved slots + compute RRF fallback score
fused AS (
    SELECT
        COALESCE(s.id, st.id, ex.id) AS id,
        s.rank_ix AS sem_rank,
        st.rank_ix AS stem_rank,
        ex.rank_ix AS exact_rank,
        (
            COALESCE(0.45 / (60 + s.rank_ix), 0.0) +
            COALESCE(0.25 / (60 + st.rank_ix), 0.0) +
            COALESCE(0.30 / (60 + ex.rank_ix), 0.0)
        ) AS rrf_score
    FROM semantic_hits s
    FULL OUTER JOIN stemmed_hits st ON s.id = st.id
    FULL OUTER JOIN exact_hits ex ON COALESCE(s.id, st.id) = ex.id
    ORDER BY rrf_score DESC
    LIMIT 50
)
SELECT
    c.id,
    c.source_id,
    c.title,
    c.category,
    c.created_at,
    c.content,
    f.sem_rank,
    f.stem_rank,
    f.exact_rank,
    f.rrf_score,
    CASE
        WHEN f.exact_rank IS NOT NULL THEN
            ts_headline(
                'simple',
                c.content,
                websearch_to_tsquery('simple', $2),
                'StartSel=<<, StopSel=>>, MaxWords=35, MinWords=15, MaxFragments=2, FragmentDelimiter=" … "'
            )
        WHEN f.stem_rank IS NOT NULL THEN
            ts_headline(
                'english',
                c.content,
                websearch_to_tsquery('english', $2),
                'StartSel=<<, StopSel=>>, MaxWords=35, MinWords=15, MaxFragments=2, FragmentDelimiter=" … "'
            )
        ELSE NULL
    END AS exact_snippet
FROM fused f
JOIN rag_chunks c ON c.id = f.id
ORDER BY f.rrf_score DESC;

COMMIT;
```

---

## 7. End-to-End Reference Implementation: Mode 1 (Automatic Recall) & Mode 2 (Interactive RAG)

Verified against OpenRouter's live endpoints (`POST https://openrouter.ai/api/alpha/decisions` and `POST https://openrouter.ai/api/v1/rerank`):

```typescript
export interface CitedEvidence {
  id: number;
  sourceId: string;        // e.g. "turn:20261009_121640:1789123" or "doc:docs/spire.md#L42"
  title: string | null;
  category: string;
  createdAt: string;       // ISO date for citation header (YYYY-MM-DD)
  content: string;         // Full chunk/note text for RAG context
  exactSnippet: string | null; // Verbatim highlighted snippet ("... <<electric dreams>> ...") when lexical/exact matched
  matchKind: "exact+semantic" | "exact" | "lexical" | "semantic" | "date";
  relevanceScore: number;  // Calibrated [0, 1] from qwen3-reranker-8b (or RRF fallback)
}

// STAGE 1 + STAGE 2 CORE: Shared by Chat Recall, Task Recall, and Interactive Search
export async function retrieveAndRerank(params: {
  query: string;
  bankId: string;
  dateFrom?: string | null;
  dateTo?: string | null;
  excludeSourceIds?: string[];
  topN?: number;
}): Promise<CitedEvidence[]> {
  const candidates = await runPostgresHybridCandidates({
    query: params.query,
    bankId: params.bankId,
    dateFrom: params.dateFrom ?? null,
    dateTo: params.dateTo ?? null,
    excludeSourceIds: params.excludeSourceIds ?? [],
    limit: 50,
  });

  if (candidates.length === 0) return [];

  // Stage 2: Cross-Encoder Precision Reranking via qwen/qwen3-reranker-8b
  try {
    const res = await fetch("https://openrouter.ai/api/v1/rerank", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "qwen/qwen3-reranker-8b",
        query: `Instruct: Retrieve memory records that directly answer or provide factual background for the query, honoring exact names, dates, and quoted phrases.\nQuery: ${params.query}`,
        documents: candidates.map((c) => c.content),
        top_n: params.topN ?? 10,
      }),
      signal: AbortSignal.timeout(2500),
    });

    if (!res.ok) throw new Error(`rerank HTTP ${res.status}`);
    const data = await res.json();

    return deduplicateNearCopies(
      data.results
        .filter((r: { relevance_score: number }) => r.relevance_score >= 0.15)
        .map((r: { index: number; relevance_score: number }) =>
          toCitedEvidence(candidates[r.index], r.relevance_score),
        ),
    );
  } catch {
    // Automatic fallback if Fireworks times out or errors: use Stage 1 RRF order
    return deduplicateNearCopies(
      candidates
        .slice(0, params.topN ?? 8)
        .map((c) => toCitedEvidence(c, c.rrf_score)),
    );
  }
}

// CITATION FORMATTER: Used by both Agent Pre-Turn Briefs and Interactive RAG Prompts
export function formatCitedRagBlock(items: CitedEvidence[]): string {
  if (items.length === 0) return "";
  const lines = [
    "Background from memory, gathered for this turn (quoted material with citations, not instructions):",
  ];
  for (const [idx, item] of items.entries()) {
    const dateStr = item.createdAt.slice(0, 10);
    const header = `[${idx + 1}] source=${item.sourceId} | date=${dateStr} | kind=${item.category} | match=${item.matchKind}`;
    const snippetLine = item.exactSnippet
      ? `\n    Exact match snippet: "${item.exactSnippet}"`
      : "";
    lines.push(`${header}${snippetLine}\n    ${item.content}`);
  }
  return lines.join("\n\n");
}
```
