//! Recall-thread query — end-to-end inspection of one recall_event.
//!
//! Joins the recall_event with its preceding formulate_event (most recent
//! in the same bank ≤ recall.created_at) and the llm_calls window around
//! the recall (±10s, same bank). Mirrors the cheat-sheet recipe in
//! docs/observability.md "Reading a recall event end-to-end".

use chrono::{DateTime, Utc};
use serde_json::Value;
use sqlx::PgPool;

#[derive(Debug, Clone)]
pub struct RecallRow {
    pub id: i64,
    pub bank_id: String,
    pub created_at: DateTime<Utc>,
    pub queries: Value,
    pub mode: String,
    pub n_results: i32,
    pub duration_ms: i32,
    pub synthesis: Option<String>,
    pub results: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct FormulateRow {
    pub id: i64,
    pub created_at: DateTime<Utc>,
    pub message: String,
    pub n_queries_out: i32,
    pub json_mode_used: bool,
    pub parse_fallback: bool,
    pub raw_response: String,
    pub duration_ms: i32,
    pub error_kind: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LlmCallRow {
    pub id: i64,
    pub created_at: DateTime<Utc>,
    pub prompt_name: String,
    pub duration_ms: i32,
    pub json_mode: bool,
    pub error: Option<String>,
    pub prompt_text: Option<String>,
    pub response_text: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RecallThread {
    pub recall: RecallRow,
    /// None if no formulate_event preceded this recall (the LATERAL join
    /// found nothing within the time-and-bank constraint).
    pub formulate: Option<FormulateRow>,
    pub llm_calls: Vec<LlmCallRow>,
}

/// Fetch the full thread for one recall_event. Three queries in parallel.
pub async fn fetch(pool: &PgPool, recall_id: i64) -> sqlx::Result<RecallThread> {
    let recall = fetch_recall(pool, recall_id).await?;

    // Window around the recall for llm_calls: 10s either side, same bank,
    // matches the cheat-sheet intuition ("the formulate_call shows you the
    // JSON-mode prompt; the synthesize_call shows you the chunk-stuffed prompt").
    let bank = recall.bank_id.clone();
    let ts = recall.created_at;
    let (formulate, llm_calls) = tokio::try_join!(
        fetch_formulate(pool, &bank, ts),
        fetch_llm_calls(pool, &bank, ts),
    )?;

    Ok(RecallThread {
        recall,
        formulate,
        llm_calls,
    })
}

async fn fetch_recall(pool: &PgPool, id: i64) -> sqlx::Result<RecallRow> {
    let r = sqlx::query!(
        r#"
        SELECT id, bank_id, created_at, queries, mode, n_results,
               duration_ms, synthesis, results
        FROM recall_events
        WHERE id = $1
        "#,
        id
    )
    .fetch_one(pool)
    .await?;

    Ok(RecallRow {
        id: r.id,
        bank_id: r.bank_id,
        created_at: r.created_at,
        queries: r.queries,
        mode: r.mode,
        n_results: r.n_results,
        duration_ms: r.duration_ms,
        synthesis: r.synthesis,
        results: r.results,
    })
}

async fn fetch_formulate(
    pool: &PgPool,
    bank: &str,
    recall_created_at: DateTime<Utc>,
) -> sqlx::Result<Option<FormulateRow>> {
    let r = sqlx::query!(
        r#"
        SELECT id, created_at, message, n_queries_out, json_mode_used,
               parse_fallback, raw_response, duration_ms, error_kind
        FROM formulate_events
        WHERE bank_id = $1 AND created_at <= $2
        ORDER BY created_at DESC
        LIMIT 1
        "#,
        bank,
        recall_created_at
    )
    .fetch_optional(pool)
    .await?;

    Ok(r.map(|r| FormulateRow {
        id: r.id,
        created_at: r.created_at,
        message: r.message,
        n_queries_out: r.n_queries_out,
        json_mode_used: r.json_mode_used,
        parse_fallback: r.parse_fallback,
        raw_response: r.raw_response,
        duration_ms: r.duration_ms,
        error_kind: r.error_kind,
    }))
}

async fn fetch_llm_calls(
    pool: &PgPool,
    bank: &str,
    recall_created_at: DateTime<Utc>,
) -> sqlx::Result<Vec<LlmCallRow>> {
    // ±10s window. llm_calls.bank_id is nullable (ON DELETE SET NULL on the FK),
    // but the typical case is non-null and same-bank.
    let rows = sqlx::query!(
        r#"
        SELECT id, created_at, prompt_name, duration_ms, json_mode, error,
               prompt_text, response_text
        FROM llm_calls
        WHERE bank_id = $1
          AND created_at BETWEEN ($2::timestamptz - INTERVAL '10 seconds')
                             AND ($2::timestamptz + INTERVAL '10 seconds')
        ORDER BY created_at ASC, id ASC
        "#,
        bank,
        recall_created_at
    )
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|r| LlmCallRow {
            id: r.id,
            created_at: r.created_at,
            prompt_name: r.prompt_name,
            duration_ms: r.duration_ms,
            json_mode: r.json_mode,
            error: r.error,
            prompt_text: r.prompt_text,
            response_text: r.response_text,
        })
        .collect())
}
