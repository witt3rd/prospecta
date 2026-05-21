//! Retain-thread query — end-to-end inspection of one retain_event.
//!
//! Joins the retain_event with its document, the memory_items it produced,
//! and the paired index_text llm_call (the call that generated the
//! question-shaped index_text strings, if LLM-authored).
//!
//! Mirrors recall_thread.rs in shape. Schema-checked via sqlx::query!.

use chrono::{DateTime, Utc};
use sqlx::PgPool;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct RetainRow {
    pub id: i64,
    pub bank_id: String,
    pub created_at: DateTime<Utc>,
    pub document_id: Option<Uuid>,
    pub items_count: i32,
    pub index_text_caller_supplied: bool,
    pub index_text_generated: Option<Vec<String>>,
    pub duration_ms: i32,
    pub raw_llm_response: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DocumentRow {
    pub id: Uuid,
    pub source: Option<String>,
    pub content_hash: String,
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub original_text: String,
}

#[derive(Debug, Clone)]
pub struct ItemRow {
    pub id: Uuid,
    pub content: String,
    pub llm_generated: bool,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct IndexTextCall {
    pub id: i64,
    pub created_at: DateTime<Utc>,
    pub duration_ms: i32,
    pub json_mode: bool,
    pub error: Option<String>,
    pub prompt_text: Option<String>,
    pub response_text: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RetainThread {
    pub retain: RetainRow,
    /// None when retain_event predated documents linkage or document was deleted.
    pub document: Option<DocumentRow>,
    pub items: Vec<ItemRow>,
    /// The most recent `prompt_name='index_text'` llm_call within ±10s on the
    /// same bank. Optional because caller-supplied retains skip the LLM call.
    pub index_text_call: Option<IndexTextCall>,
}

pub async fn fetch(pool: &PgPool, retain_id: i64) -> sqlx::Result<RetainThread> {
    let retain = fetch_retain(pool, retain_id).await?;
    let bank = retain.bank_id.clone();
    let ts = retain.created_at;
    let doc_id = retain.document_id;

    let (document, items, index_text_call) = tokio::try_join!(
        fetch_document(pool, doc_id),
        fetch_items(pool, doc_id),
        fetch_index_text_call(pool, &bank, ts),
    )?;

    Ok(RetainThread {
        retain,
        document,
        items,
        index_text_call,
    })
}

async fn fetch_retain(pool: &PgPool, id: i64) -> sqlx::Result<RetainRow> {
    let r = sqlx::query!(
        r#"
        SELECT id, bank_id, created_at, document_id, items_count,
               index_text_caller_supplied, index_text_generated,
               duration_ms, raw_llm_response, error
        FROM retain_events
        WHERE id = $1
        "#,
        id
    )
    .fetch_one(pool)
    .await?;

    Ok(RetainRow {
        id: r.id,
        bank_id: r.bank_id,
        created_at: r.created_at,
        document_id: r.document_id,
        items_count: r.items_count,
        index_text_caller_supplied: r.index_text_caller_supplied,
        index_text_generated: r.index_text_generated,
        duration_ms: r.duration_ms,
        raw_llm_response: r.raw_llm_response,
        error: r.error,
    })
}

async fn fetch_document(
    pool: &PgPool,
    document_id: Option<Uuid>,
) -> sqlx::Result<Option<DocumentRow>> {
    let Some(id) = document_id else {
        return Ok(None);
    };
    let r = sqlx::query!(
        r#"
        SELECT id, source, content_hash, tags, created_at, original_text
        FROM documents
        WHERE id = $1
        "#,
        id
    )
    .fetch_optional(pool)
    .await?;

    Ok(r.map(|r| DocumentRow {
        id: r.id,
        source: r.source,
        content_hash: r.content_hash,
        tags: r.tags,
        created_at: r.created_at,
        original_text: r.original_text,
    }))
}

async fn fetch_items(pool: &PgPool, document_id: Option<Uuid>) -> sqlx::Result<Vec<ItemRow>> {
    let Some(id) = document_id else {
        return Ok(Vec::new());
    };
    let rows = sqlx::query!(
        r#"
        SELECT id, content, llm_generated, tags
        FROM memory_items
        WHERE document_id = $1
        ORDER BY created_at, id
        "#,
        id
    )
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|r| ItemRow {
            id: r.id,
            content: r.content,
            llm_generated: r.llm_generated,
            tags: r.tags,
        })
        .collect())
}

async fn fetch_index_text_call(
    pool: &PgPool,
    bank: &str,
    retain_created_at: DateTime<Utc>,
) -> sqlx::Result<Option<IndexTextCall>> {
    let r = sqlx::query!(
        r#"
        SELECT id, created_at, duration_ms, json_mode, error,
               prompt_text, response_text
        FROM llm_calls
        WHERE bank_id = $1
          AND prompt_name = 'index_text'
          AND created_at BETWEEN ($2::timestamptz - INTERVAL '10 seconds')
                             AND ($2::timestamptz + INTERVAL '10 seconds')
        ORDER BY ABS(EXTRACT(EPOCH FROM (created_at - $2::timestamptz))) ASC
        LIMIT 1
        "#,
        bank,
        retain_created_at
    )
    .fetch_optional(pool)
    .await?;

    Ok(r.map(|r| IndexTextCall {
        id: r.id,
        created_at: r.created_at,
        duration_ms: r.duration_ms,
        json_mode: r.json_mode,
        error: r.error,
        prompt_text: r.prompt_text,
        response_text: r.response_text,
    }))
}
