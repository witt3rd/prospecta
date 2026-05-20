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
}

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;
    let _ = dotenvy::dotenv(); // opt-in for local dev; ignore not-found.

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
