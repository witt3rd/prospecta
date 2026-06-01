//! prospecta-tui — interactive inspection surface for the prospecta memory substrate.
//!
//! v0.1: bank-list view + help overlay. Future commits add event-stream view,
//! document drill-down, recall/retain thread views, search, and manual ops.

use std::io;

use clap::Parser;
use color_eyre::eyre::{Result, WrapErr};
use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use prospecta_tui::{app, config};
use ratatui::{backend::CrosstermBackend, Terminal};
use sqlx::postgres::PgPoolOptions;

#[derive(Parser, Debug)]
#[command(
    name = "prospecta-tui",
    version,
    about = "Interactive TUI for prospecta — browse, search, and review the memory substrate."
)]
struct Cli {
    /// Postgres connection string. Falls back to the DATABASE_URL env var (or .env file).
    #[arg(long, env = "DATABASE_URL")]
    database_url: String,

    /// Smoke-test the connection: print bank count and exit. No TUI launched.
    #[arg(long)]
    smoke: bool,

    /// Print the most recent events across all five event tables and exit.
    /// Useful for sanity-checking the substrate without launching the TUI.
    #[arg(long)]
    dump_events: bool,

    /// Dump documents (and optionally memory_items) for a bank and exit.
    /// Pass --dump-docs default to list documents; add --doc-id <uuid>
    /// to descend into memory_items for one document.
    #[arg(long, value_name = "BANK")]
    dump_docs: Option<String>,

    /// When set with --dump-docs, descend into this document's memory_items
    /// instead of listing the bank's documents.
    #[arg(long, value_name = "UUID")]
    doc_id: Option<uuid::Uuid>,

    /// Dump the full recall thread (preceding formulate + recall body +
    /// paired llm_calls) for a given recall_event id and exit.
    #[arg(long, value_name = "RECALL_ID")]
    dump_recall_thread: Option<i64>,

    /// Dump the full retain thread (retain_event + document + memory_items +
    /// paired index_text llm_call) for a given retain_event id and exit.
    #[arg(long, value_name = "RETAIN_ID")]
    dump_retain_thread: Option<i64>,

    /// Dump per-bank dashboard stats (24h health signals) and exit.
    #[arg(long, value_name = "BANK")]
    dump_dashboard: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;
    let _ = dotenvy::dotenv(); // opt-in for local dev; ignore not-found.

    // Treat broken-pipe (head, less, etc.) as a clean exit rather than a panic.
    // Only matters for the --dump-* paths; harmless for the TUI.
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }

    let cli = Cli::parse();
    config::validate(&cli.database_url)?;
    let redacted = config::redact_database_url(&cli.database_url);

    let pool = PgPoolOptions::new()
        .max_connections(4)
        .connect(&cli.database_url)
        .await
        .wrap_err("connecting to prospecta substrate")?;

    if cli.smoke {
        let row = sqlx::query!("SELECT count(*) AS n FROM banks")
            .fetch_one(&pool)
            .await
            .wrap_err("querying banks")?;
        println!(
            "prospecta-tui smoke: connected to {}, {} banks visible",
            redacted,
            row.n.unwrap_or(0)
        );
        return Ok(());
    }

    if cli.dump_events {
        let events = prospecta_tui::db::events::recent(&pool, None, 50, 50)
            .await
            .wrap_err("loading events")?;
        println!(
            "prospecta-tui dump_events: {} events from {}",
            events.len(),
            redacted
        );
        for e in events {
            let bank = e.bank_id.as_deref().unwrap_or("—");
            let dur = match e.duration_ms {
                Some(n) if n >= 1000 => format!("{:.1}s", n as f64 / 1000.0),
                Some(n) => format!("{}ms", n),
                None => "—".to_string(),
            };
            let marker = if e.has_error { "!" } else { " " };
            println!(
                "{}  {:<9} bank={:<12} id={:<5} dur={:<7} {} {}",
                e.created_at.format("%Y-%m-%d %H:%M:%S"),
                e.kind.tag(),
                bank,
                e.id,
                dur,
                marker,
                e.summary,
            );
        }
        return Ok(());
    }

    if let Some(bank) = cli.dump_docs.as_deref() {
        match cli.doc_id {
            None => {
                let docs_ = prospecta_tui::db::documents::list_for_bank(&pool, bank, 100, 0)
                    .await
                    .wrap_err("loading documents")?;
                println!(
                    "prospecta-tui dump_docs: bank={} count={}",
                    bank,
                    docs_.len()
                );
                for d in docs_ {
                    println!(
                        "  {}  items={:<3} source={}  tags=[{}]  created={}",
                        d.id,
                        d.item_count,
                        d.source.unwrap_or_else(|| "—".into()),
                        d.tags.join(","),
                        d.created_at.format("%Y-%m-%d %H:%M:%S"),
                    );
                }
            }
            Some(doc_id) => {
                let items = prospecta_tui::db::documents::list_for_document(&pool, doc_id)
                    .await
                    .wrap_err("loading memory_items")?;
                println!(
                    "prospecta-tui dump_docs: doc={} items={}",
                    doc_id,
                    items.len()
                );
                for it in items {
                    let src = if it.llm_generated { "llm" } else { "caller" };
                    println!("  [{}] {}", src, it.content);
                }
            }
        }
        return Ok(());
    }

    if let Some(recall_id) = cli.dump_recall_thread {
        let thread = prospecta_tui::db::recall_thread::fetch(&pool, recall_id)
            .await
            .wrap_err("loading recall thread")?;
        println!(
            "prospecta-tui dump_recall_thread: id={} bank={}",
            thread.recall.id, thread.recall.bank_id
        );
        println!(
            "  recall:    mode={} n={} dur={}ms",
            thread.recall.mode, thread.recall.n_results, thread.recall.duration_ms
        );
        match &thread.formulate {
            Some(f) => println!(
                "  formulate: id={} parse_fallback={} msg={:?}",
                f.id, f.parse_fallback, f.message
            ),
            None => println!("  formulate: (none in window)"),
        }
        println!("  llm_calls: {} in ±10s window", thread.llm_calls.len());
        for c in &thread.llm_calls {
            println!(
                "    [{}] {:<10} dur={}ms {}",
                c.id,
                c.prompt_name,
                c.duration_ms,
                if c.error.is_some() { "ERR" } else { "" }
            );
        }
        if let Some(s) = &thread.recall.synthesis {
            println!("  synthesis: {}", s);
        }
        return Ok(());
    }

    if let Some(retain_id) = cli.dump_retain_thread {
        let t = prospecta_tui::db::retain_thread::fetch(&pool, retain_id)
            .await
            .wrap_err("loading retain thread")?;
        println!(
            "prospecta-tui dump_retain_thread: id={} bank={}",
            t.retain.id, t.retain.bank_id
        );
        println!(
            "  retain:    items={} caller_supplied={} dur={}ms",
            t.retain.items_count, t.retain.index_text_caller_supplied, t.retain.duration_ms
        );
        match &t.document {
            Some(d) => println!(
                "  document:  id={} source={}",
                d.id,
                d.source.clone().unwrap_or_else(|| "—".into())
            ),
            None => println!("  document:  (none linked)"),
        }
        println!("  items:     {} memory_items", t.items.len());
        for (i, it) in t.items.iter().enumerate() {
            let src = if it.llm_generated { "llm" } else { "caller" };
            println!("    {:>2}. [{}] {}", i + 1, src, it.content);
        }
        match &t.index_text_call {
            Some(c) => println!("  index_text llm_call: id={} dur={}ms", c.id, c.duration_ms),
            None if t.retain.index_text_caller_supplied => {
                println!("  index_text llm_call: (skipped — caller-supplied)")
            }
            None => println!("  index_text llm_call: (none in ±10s window)"),
        }
        return Ok(());
    }

    if let Some(bank) = cli.dump_dashboard.as_deref() {
        let s = prospecta_tui::db::dashboard::fetch(&pool, bank)
            .await
            .wrap_err("loading dashboard")?;
        println!(
            "prospecta-tui dump_dashboard: bank={} dim={}",
            s.bank_id, s.embedding_dim
        );
        println!(
            "  substrate: documents={} memory_items={}",
            s.documents, s.memory_items
        );
        println!(
            "  activity (24h): retains={} recalls={} formulates={}",
            s.retains_24h, s.recalls_24h, s.formulates_24h
        );
        let mr = s
            .mean_recall_ms_24h
            .map(|v| format!("{:.0}ms", v))
            .unwrap_or_else(|| "—".into());
        let mre = s
            .mean_retain_ms_24h
            .map(|v| format!("{:.0}ms", v))
            .unwrap_or_else(|| "—".into());
        println!(
            "  latency (24h): mean recall={} mean retain={} Σ llm={}ms",
            mr, mre, s.total_llm_ms_24h
        );
        let fb = s
            .formulate_fallback_rate_24h
            .map(|v| format!("{:.1}%", v * 100.0))
            .unwrap_or_else(|| "—".into());
        println!("  health (24h): parse_fallback={}", fb);
        if s.llm_calls.is_empty() {
            println!("  llm_calls (24h): none");
        } else {
            println!("  llm_calls (24h):");
            for c in &s.llm_calls {
                println!(
                    "    {:<12} calls={:<4} avg={:.0}ms max={}ms err={}",
                    c.prompt_name, c.calls, c.avg_ms, c.max_ms, c.errors
                );
            }
        }
        if s.sweeps.is_empty() {
            println!("  sweeps: none (bank never swept)");
        } else {
            println!("  sweeps (latest pass per corpus):");
            for sw in &s.sweeps {
                let status = if sw.ended_at.is_none() {
                    "running"
                } else if sw.error.is_some() || sw.errors_count > 0 {
                    "error"
                } else {
                    "ok"
                };
                println!(
                    "    {:<24} {:<8} idx/seen={}/{} pruned={} errs={}",
                    sw.corpus_path,
                    status,
                    sw.files_indexed,
                    sw.files_seen,
                    sw.files_pruned,
                    sw.errors_count
                );
            }
        }
        return Ok(());
    }

    let result = run_tui(pool, redacted).await;
    // Always restore the terminal on the way out, even on error.
    let _ = restore_terminal();
    result
}

async fn run_tui(pool: sqlx::PgPool, redacted_url: String) -> Result<()> {
    enable_raw_mode().wrap_err("enable raw mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).wrap_err("enter alt screen")?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).wrap_err("init ratatui terminal")?;

    let mut app = app::App::new(pool, redacted_url).await;
    let res = app.run(&mut terminal).await;

    // Best-effort terminal restoration; surface app error if present.
    let _ = restore_terminal();
    res
}

fn restore_terminal() -> Result<()> {
    disable_raw_mode().wrap_err("disable raw mode")?;
    execute!(io::stdout(), LeaveAlternateScreen).wrap_err("leave alt screen")?;
    Ok(())
}
