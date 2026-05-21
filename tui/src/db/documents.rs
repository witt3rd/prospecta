//! `documents` and `memory_items` queries — the manage-axis drill-down.

use chrono::{DateTime, Utc};
use sqlx::PgPool;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct Document {
    pub id: Uuid,
    pub bank_id: String,
    pub source: Option<String>,
    pub content_hash: String,
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
    /// Rolled up from memory_items at query time.
    pub item_count: i64,
}

/// Documents in a bank, ordered by created_at DESC (newest first).
/// Paginated via offset/limit; v0.1 keeps it simple — no cursor.
pub async fn list_for_bank(
    pool: &PgPool,
    bank_id: &str,
    limit: i64,
    offset: i64,
) -> sqlx::Result<Vec<Document>> {
    let rows = sqlx::query!(
        r#"
        SELECT
            d.id,
            d.bank_id,
            d.source,
            d.content_hash,
            d.tags,
            d.created_at,
            (SELECT count(*) FROM memory_items mi WHERE mi.document_id = d.id) AS item_count
        FROM documents d
        WHERE d.bank_id = $1
        ORDER BY d.created_at DESC, d.id
        LIMIT $2 OFFSET $3
        "#,
        bank_id,
        limit,
        offset
    )
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|r| Document {
            id: r.id,
            bank_id: r.bank_id,
            source: r.source,
            content_hash: r.content_hash,
            tags: r.tags,
            created_at: r.created_at,
            item_count: r.item_count.unwrap_or(0),
        })
        .collect())
}

#[derive(Debug, Clone)]
pub struct MemoryItem {
    pub id: Uuid,
    pub bank_id: String,
    pub document_id: Uuid,
    /// LLM-anticipated question form (the spine column).
    pub content: String,
    /// Source chunk this index_text indexes (P5 no-truncation full audit).
    pub original_chunk: String,
    pub llm_generated: bool,
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
}

/// Memory items for one document, ordered by created_at then id for stable display.
pub async fn list_for_document(pool: &PgPool, document_id: Uuid) -> sqlx::Result<Vec<MemoryItem>> {
    let rows = sqlx::query!(
        r#"
        SELECT
            id, bank_id, document_id,
            content, original_chunk,
            llm_generated, tags, created_at
        FROM memory_items
        WHERE document_id = $1
        ORDER BY created_at, id
        "#,
        document_id
    )
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|r| MemoryItem {
            id: r.id,
            bank_id: r.bank_id,
            document_id: r.document_id,
            content: r.content,
            original_chunk: r.original_chunk,
            llm_generated: r.llm_generated,
            tags: r.tags,
            created_at: r.created_at,
        })
        .collect())
}
