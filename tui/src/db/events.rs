//! Event-stream queries — pulls from the five append-only event tables,
//! normalizes to a common shape, and merges in Rust by created_at DESC.
//!
//! Each table gets its own `sqlx::query!` so compile-time schema checking
//! stays intact (UNION ALL across heterogeneous tables defeats type inference).

use chrono::{DateTime, Utc};
use sqlx::PgPool;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    Retain,
    Recall,
    Formulate,
    LlmCall,
    SweepPass,
}

impl EventKind {
    pub fn tag(&self) -> &'static str {
        match self {
            EventKind::Retain => "retain",
            EventKind::Recall => "recall",
            EventKind::Formulate => "formulate",
            EventKind::LlmCall => "llm_call",
            EventKind::SweepPass => "sweep",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Event {
    pub kind: EventKind,
    pub id: i64,
    /// `bank_id` is Optional only for `llm_calls` (FK is ON DELETE SET NULL).
    pub bank_id: Option<String>,
    pub created_at: DateTime<Utc>,
    pub duration_ms: Option<i32>,
    pub has_error: bool,
    pub summary: String,
}

/// Fetch the most recent `limit_per_kind` rows from each event table,
/// merge by created_at DESC, and return the top `total_limit`.
///
/// The five `query!` calls run concurrently — total wall-clock is dominated
/// by the slowest, not the sum. For our typical event volume (low hundreds)
/// this is faster than a UNION and keeps compile-time checking intact.
pub async fn recent(
    pool: &PgPool,
    bank_filter: Option<&str>,
    limit_per_kind: i64,
    total_limit: usize,
) -> sqlx::Result<Vec<Event>> {
    let (retains, recalls, formulates, llms, sweeps) = tokio::try_join!(
        fetch_retains(pool, bank_filter, limit_per_kind),
        fetch_recalls(pool, bank_filter, limit_per_kind),
        fetch_formulates(pool, bank_filter, limit_per_kind),
        fetch_llm_calls(pool, bank_filter, limit_per_kind),
        fetch_sweeps(pool, bank_filter, limit_per_kind),
    )?;

    let mut all: Vec<Event> = retains
        .into_iter()
        .chain(recalls)
        .chain(formulates)
        .chain(llms)
        .chain(sweeps)
        .collect();

    // Stable sort by created_at DESC.
    all.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    all.truncate(total_limit);
    Ok(all)
}

async fn fetch_retains(pool: &PgPool, bank: Option<&str>, limit: i64) -> sqlx::Result<Vec<Event>> {
    let rows = sqlx::query!(
        r#"
        SELECT id, bank_id, created_at, duration_ms, items_count,
               index_text_caller_supplied, error
        FROM retain_events
        WHERE ($1::text IS NULL OR bank_id = $1)
        ORDER BY created_at DESC
        LIMIT $2
        "#,
        bank,
        limit
    )
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|r| Event {
            kind: EventKind::Retain,
            id: r.id,
            bank_id: Some(r.bank_id),
            created_at: r.created_at,
            duration_ms: Some(r.duration_ms),
            has_error: r.error.is_some(),
            summary: format!(
                "items={} caller_supplied={}",
                r.items_count, r.index_text_caller_supplied
            ),
        })
        .collect())
}

async fn fetch_recalls(pool: &PgPool, bank: Option<&str>, limit: i64) -> sqlx::Result<Vec<Event>> {
    let rows = sqlx::query!(
        r#"
        SELECT id, bank_id, created_at, duration_ms, mode, n_results,
               synthesis IS NOT NULL AS has_synth
        FROM recall_events
        WHERE ($1::text IS NULL OR bank_id = $1)
        ORDER BY created_at DESC
        LIMIT $2
        "#,
        bank,
        limit
    )
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|r| Event {
            kind: EventKind::Recall,
            id: r.id,
            bank_id: Some(r.bank_id),
            created_at: r.created_at,
            duration_ms: Some(r.duration_ms),
            has_error: false,
            summary: format!(
                "mode={} n={}{}",
                r.mode,
                r.n_results,
                if r.has_synth.unwrap_or(false) {
                    " +synth"
                } else {
                    ""
                }
            ),
        })
        .collect())
}

async fn fetch_formulates(
    pool: &PgPool,
    bank: Option<&str>,
    limit: i64,
) -> sqlx::Result<Vec<Event>> {
    let rows = sqlx::query!(
        r#"
        SELECT id, bank_id, created_at, duration_ms, n_queries_out,
               parse_fallback, error_kind
        FROM formulate_events
        WHERE ($1::text IS NULL OR bank_id = $1)
        ORDER BY created_at DESC
        LIMIT $2
        "#,
        bank,
        limit
    )
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|r| {
            let summary = if r.parse_fallback {
                format!(
                    "n_out={} FALLBACK={}",
                    r.n_queries_out,
                    r.error_kind.as_deref().unwrap_or("?")
                )
            } else {
                format!("n_out={}", r.n_queries_out)
            };
            Event {
                kind: EventKind::Formulate,
                id: r.id,
                bank_id: Some(r.bank_id),
                created_at: r.created_at,
                duration_ms: Some(r.duration_ms),
                has_error: r.parse_fallback,
                summary,
            }
        })
        .collect())
}

async fn fetch_llm_calls(
    pool: &PgPool,
    bank: Option<&str>,
    limit: i64,
) -> sqlx::Result<Vec<Event>> {
    let rows = sqlx::query!(
        r#"
        SELECT id, bank_id, created_at, duration_ms, prompt_name, error
        FROM llm_calls
        WHERE ($1::text IS NULL OR bank_id = $1)
        ORDER BY created_at DESC
        LIMIT $2
        "#,
        bank,
        limit
    )
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|r| Event {
            kind: EventKind::LlmCall,
            id: r.id,
            bank_id: r.bank_id,
            created_at: r.created_at,
            duration_ms: Some(r.duration_ms),
            has_error: r.error.is_some(),
            summary: match &r.error {
                Some(e) => format!("{} ERR: {}", r.prompt_name, e),
                None => r.prompt_name,
            },
        })
        .collect())
}

async fn fetch_sweeps(pool: &PgPool, bank: Option<&str>, limit: i64) -> sqlx::Result<Vec<Event>> {
    let rows = sqlx::query!(
        r#"
        SELECT id, bank_id, started_at, duration_ms,
               files_seen, files_indexed, files_pruned, errors_count, corpus_path
        FROM sweep_passes
        WHERE ($1::text IS NULL OR bank_id = $1)
        ORDER BY started_at DESC
        LIMIT $2
        "#,
        bank,
        limit
    )
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|r| Event {
            kind: EventKind::SweepPass,
            id: r.id,
            bank_id: Some(r.bank_id),
            created_at: r.started_at,
            duration_ms: r.duration_ms,
            has_error: r.errors_count > 0,
            summary: format!(
                "{} seen={} idx={} pruned={} err={}",
                r.corpus_path, r.files_seen, r.files_indexed, r.files_pruned, r.errors_count
            ),
        })
        .collect())
}
