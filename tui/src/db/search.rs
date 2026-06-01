//! Search axis — cross-table free-text over the two lexical channels.
//!
//! prospecta indexes two tsvector channels on memory_items (schema §2,
//! migration 0002):
//!   - content_tsv : the LLM-anticipated question form (index_text). This is
//!                   the spine — what the writer thought the reader would ask.
//!   - body_tsv    : the source body (original_chunk). The P14 honest safety
//!                   net — when the anticipated question drifts from query
//!                   language, the body still surfaces the doc.
//!
//! This view searches both with one `websearch_to_tsquery` and reports which
//! channel(s) matched plus a per-channel rank. It mirrors the lexical legs of
//! the library's hybrid retrieval, minus the semantic channel (which needs an
//! embedding the TUI doesn't compute — that path stays in the Python library
//! and is reachable via manual recall shell-out).

use chrono::{DateTime, Utc};
use sqlx::PgPool;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub item_id: Uuid,
    pub document_id: Uuid,
    pub source: Option<String>,
    /// The matched index_text (question form).
    pub content: String,
    /// The source chunk this index_text indexes.
    pub original_chunk: String,
    pub llm_generated: bool,
    /// ts_rank against content_tsv (the question channel). 0.0 if no hit.
    pub content_rank: f32,
    /// ts_rank against body_tsv (the body channel). 0.0 if no hit.
    pub body_rank: f32,
    /// Did the query match the content (question) channel?
    pub content_hit: bool,
    /// Did the query match the body channel?
    pub body_hit: bool,
    pub created_at: DateTime<Utc>,
}

impl SearchHit {
    /// Compact label for which channel(s) surfaced this row.
    pub fn channel_label(&self) -> &'static str {
        match (self.content_hit, self.body_hit) {
            (true, true) => "both",
            (true, false) => "question",
            (false, true) => "body",
            // Shouldn't happen — the WHERE clause requires at least one hit.
            (false, false) => "—",
        }
    }

    /// The stronger of the two channel ranks, for display.
    pub fn best_rank(&self) -> f32 {
        self.content_rank.max(self.body_rank)
    }
}

/// Free-text search across both lexical channels in one bank.
///
/// Uses `websearch_to_tsquery` — the same parser the library's lexical legs
/// use — so operator queries behave consistently with agent-path retrieval
/// (quoted phrases, OR, -negation all work). Empty / whitespace-only queries
/// short-circuit to no results rather than erroring.
pub async fn search(
    pool: &PgPool,
    bank_id: &str,
    query: &str,
    limit: i64,
) -> sqlx::Result<Vec<SearchHit>> {
    if query.trim().is_empty() {
        return Ok(Vec::new());
    }

    let rows = sqlx::query!(
        r#"
        SELECT
            mi.id            AS item_id,
            mi.document_id   AS document_id,
            d.source         AS source,
            mi.content       AS content,
            mi.original_chunk AS original_chunk,
            mi.llm_generated AS llm_generated,
            mi.created_at    AS created_at,
            ts_rank(mi.content_tsv, q) AS content_rank,
            ts_rank(mi.body_tsv, q)    AS body_rank,
            (mi.content_tsv @@ q)      AS content_hit,
            (mi.body_tsv @@ q)         AS body_hit
        FROM memory_items mi
        JOIN documents d ON d.id = mi.document_id
        CROSS JOIN websearch_to_tsquery('english', $2) AS q
        WHERE mi.bank_id = $1
          AND (mi.content_tsv @@ q OR mi.body_tsv @@ q)
        ORDER BY GREATEST(ts_rank(mi.content_tsv, q), ts_rank(mi.body_tsv, q)) DESC,
                 mi.created_at DESC
        LIMIT $3
        "#,
        bank_id,
        query,
        limit
    )
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|r| SearchHit {
            item_id: r.item_id,
            document_id: r.document_id,
            source: r.source,
            content: r.content,
            original_chunk: r.original_chunk,
            llm_generated: r.llm_generated,
            // ts_rank returns real (f32); the @@ booleans are non-null because
            // they're computed expressions, but sqlx types them Option.
            content_rank: r.content_rank.unwrap_or(0.0),
            body_rank: r.body_rank.unwrap_or(0.0),
            content_hit: r.content_hit.unwrap_or(false),
            body_hit: r.body_hit.unwrap_or(false),
            created_at: r.created_at,
        })
        .collect())
}
