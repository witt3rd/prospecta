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
