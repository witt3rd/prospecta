//! `banks` table queries.

use chrono::{DateTime, Utc};
use sqlx::PgPool;

#[derive(Debug, Clone)]
#[allow(dead_code)] // embedding_model_id + created_at land in the detail strip next.
pub struct Bank {
    pub bank_id: String,
    pub embedding_dim: i32,
    pub mission: Option<String>,
    pub embedding_model_id: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// Bank with rollup counts joined in. The five count columns mirror the
/// observability cheat sheet's "bank stats one-liner".
#[derive(Debug, Clone)]
pub struct BankSummary {
    pub bank: Bank,
    pub documents: i64,
    pub memory_items: i64,
    pub retain_events: i64,
    pub recall_events: i64,
}

/// List all banks with their summary counts. Ordered by bank_id for stable display.
pub async fn list_with_counts(pool: &PgPool) -> sqlx::Result<Vec<BankSummary>> {
    // Single round-trip with correlated subqueries — same shape as the
    // canonical SQL in docs/observability.md.
    let rows = sqlx::query!(
        r#"
        SELECT
            b.bank_id,
            b.embedding_dim,
            b.mission,
            b.embedding_model_id,
            b.created_at,
            (SELECT count(*) FROM documents     WHERE bank_id = b.bank_id) AS documents,
            (SELECT count(*) FROM memory_items  WHERE bank_id = b.bank_id) AS memory_items,
            (SELECT count(*) FROM retain_events WHERE bank_id = b.bank_id) AS retain_events,
            (SELECT count(*) FROM recall_events WHERE bank_id = b.bank_id) AS recall_events
        FROM banks b
        ORDER BY b.bank_id
        "#
    )
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|r| BankSummary {
            bank: Bank {
                bank_id: r.bank_id,
                embedding_dim: r.embedding_dim,
                mission: r.mission,
                embedding_model_id: r.embedding_model_id,
                created_at: r.created_at,
            },
            documents: r.documents.unwrap_or(0),
            memory_items: r.memory_items.unwrap_or(0),
            retain_events: r.retain_events.unwrap_or(0),
            recall_events: r.recall_events.unwrap_or(0),
        })
        .collect())
}
