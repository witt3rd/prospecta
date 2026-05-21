//! Per-bank dashboard — health signals derived from substrate state.
//!
//! Five concurrent queries via tokio::try_join!. Mirrors the cheat-sheet
//! "Bank stats one-liner" and "Volume + timing distribution" shapes from
//! docs/observability.md.

use sqlx::PgPool;

#[derive(Debug, Clone)]
pub struct DashboardStats {
    pub bank_id: String,
    pub embedding_dim: i32,
    pub documents: i64,
    pub memory_items: i64,
    /// retain_events in the last 24h.
    pub retains_24h: i64,
    /// recall_events in the last 24h.
    pub recalls_24h: i64,
    /// Mean recall duration over the last 24h (None if no recalls).
    pub mean_recall_ms_24h: Option<f64>,
    /// Mean retain duration over the last 24h (None if no retains).
    pub mean_retain_ms_24h: Option<f64>,
    /// Formulate parse_fallback rate over the last 24h (None if no formulates).
    /// 0.0..=1.0; P18 safety-net health signal.
    pub formulate_fallback_rate_24h: Option<f64>,
    pub formulates_24h: i64,
    /// Per-prompt LLM call summary (last 24h).
    pub llm_calls: Vec<LlmCallStat>,
    /// Sum of all LLM-call durations in the last 24h. Cost-estimate proxy.
    pub total_llm_ms_24h: i64,
}

#[derive(Debug, Clone)]
pub struct LlmCallStat {
    pub prompt_name: String,
    pub calls: i64,
    pub avg_ms: f64,
    pub max_ms: i32,
    pub errors: i64,
}

pub async fn fetch(pool: &PgPool, bank_id: &str) -> sqlx::Result<DashboardStats> {
    let (bank_meta, doc_counts, retain_recall, formulate, llm) = tokio::try_join!(
        fetch_bank_meta(pool, bank_id),
        fetch_doc_counts(pool, bank_id),
        fetch_retain_recall(pool, bank_id),
        fetch_formulate(pool, bank_id),
        fetch_llm(pool, bank_id),
    )?;

    Ok(DashboardStats {
        bank_id: bank_meta.0,
        embedding_dim: bank_meta.1,
        documents: doc_counts.0,
        memory_items: doc_counts.1,
        retains_24h: retain_recall.retains,
        recalls_24h: retain_recall.recalls,
        mean_recall_ms_24h: retain_recall.mean_recall_ms,
        mean_retain_ms_24h: retain_recall.mean_retain_ms,
        formulate_fallback_rate_24h: formulate.fallback_rate,
        formulates_24h: formulate.count,
        llm_calls: llm.0,
        total_llm_ms_24h: llm.1,
    })
}

async fn fetch_bank_meta(pool: &PgPool, bank_id: &str) -> sqlx::Result<(String, i32)> {
    let r = sqlx::query!(
        "SELECT bank_id, embedding_dim FROM banks WHERE bank_id = $1",
        bank_id
    )
    .fetch_one(pool)
    .await?;
    Ok((r.bank_id, r.embedding_dim))
}

async fn fetch_doc_counts(pool: &PgPool, bank_id: &str) -> sqlx::Result<(i64, i64)> {
    let r = sqlx::query!(
        r#"
        SELECT
            (SELECT count(*) FROM documents    WHERE bank_id = $1) AS documents,
            (SELECT count(*) FROM memory_items WHERE bank_id = $1) AS memory_items
        "#,
        bank_id
    )
    .fetch_one(pool)
    .await?;
    Ok((r.documents.unwrap_or(0), r.memory_items.unwrap_or(0)))
}

struct RetainRecall {
    retains: i64,
    recalls: i64,
    mean_recall_ms: Option<f64>,
    mean_retain_ms: Option<f64>,
}

async fn fetch_retain_recall(pool: &PgPool, bank_id: &str) -> sqlx::Result<RetainRecall> {
    // Single round trip. NULLIF avoids divide-by-zero AVG semantics; AVG already
    // returns NULL on empty groups, so we just take the Option.
    let r = sqlx::query!(
        r#"
        SELECT
            (SELECT count(*) FROM retain_events
                WHERE bank_id = $1
                  AND created_at >= now() - INTERVAL '24 hours') AS retains,
            (SELECT count(*) FROM recall_events
                WHERE bank_id = $1
                  AND created_at >= now() - INTERVAL '24 hours') AS recalls,
            (SELECT AVG(duration_ms)::float8 FROM recall_events
                WHERE bank_id = $1
                  AND created_at >= now() - INTERVAL '24 hours') AS mean_recall_ms,
            (SELECT AVG(duration_ms)::float8 FROM retain_events
                WHERE bank_id = $1
                  AND created_at >= now() - INTERVAL '24 hours') AS mean_retain_ms
        "#,
        bank_id
    )
    .fetch_one(pool)
    .await?;
    Ok(RetainRecall {
        retains: r.retains.unwrap_or(0),
        recalls: r.recalls.unwrap_or(0),
        mean_recall_ms: r.mean_recall_ms,
        mean_retain_ms: r.mean_retain_ms,
    })
}

struct Formulate {
    count: i64,
    fallback_rate: Option<f64>,
}

async fn fetch_formulate(pool: &PgPool, bank_id: &str) -> sqlx::Result<Formulate> {
    let r = sqlx::query!(
        r#"
        SELECT
            count(*) AS total,
            count(*) FILTER (WHERE parse_fallback) AS fallbacks
        FROM formulate_events
        WHERE bank_id = $1
          AND created_at >= now() - INTERVAL '24 hours'
        "#,
        bank_id
    )
    .fetch_one(pool)
    .await?;

    let total = r.total.unwrap_or(0);
    let fallbacks = r.fallbacks.unwrap_or(0);
    let rate = if total > 0 {
        Some(fallbacks as f64 / total as f64)
    } else {
        None
    };
    Ok(Formulate {
        count: total,
        fallback_rate: rate,
    })
}

async fn fetch_llm(pool: &PgPool, bank_id: &str) -> sqlx::Result<(Vec<LlmCallStat>, i64)> {
    // Per-prompt breakdown over the last 24h.
    let rows = sqlx::query!(
        r#"
        SELECT
            prompt_name,
            count(*) AS calls,
            AVG(duration_ms)::float8 AS avg_ms,
            MAX(duration_ms) AS max_ms,
            count(*) FILTER (WHERE error IS NOT NULL) AS errors
        FROM llm_calls
        WHERE bank_id = $1
          AND created_at >= now() - INTERVAL '24 hours'
        GROUP BY prompt_name
        ORDER BY calls DESC
        "#,
        bank_id
    )
    .fetch_all(pool)
    .await?;

    let total_ms = sqlx::query!(
        r#"
        SELECT COALESCE(SUM(duration_ms), 0)::int8 AS total_ms
        FROM llm_calls
        WHERE bank_id = $1
          AND created_at >= now() - INTERVAL '24 hours'
        "#,
        bank_id
    )
    .fetch_one(pool)
    .await?;

    let stats = rows
        .into_iter()
        .map(|r| LlmCallStat {
            prompt_name: r.prompt_name,
            calls: r.calls.unwrap_or(0),
            avg_ms: r.avg_ms.unwrap_or(0.0),
            max_ms: r.max_ms.unwrap_or(0),
            errors: r.errors.unwrap_or(0),
        })
        .collect();

    Ok((stats, total_ms.total_ms.unwrap_or(0)))
}
