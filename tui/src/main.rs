//! prospecta-tui — interactive inspection surface for the prospecta memory substrate.
//!
//! v0.1 scaffold: connect to DATABASE_URL, verify schema is reachable, list banks.
//! Ratatui event loop and the four feature axes (manage/use/search/observability)
//! land in subsequent commits — see HANDOFF note in repo root for the full plan.

use clap::Parser;
use color_eyre::eyre::{Result, WrapErr};
use sqlx::postgres::PgPoolOptions;

#[derive(Parser, Debug)]
#[command(name = "prospecta-tui", version, about = "Interactive TUI for prospecta", long_about = None)]
struct Cli {
    /// Postgres connection string. Falls back to the DATABASE_URL env var (or .env file).
    #[arg(long, env = "DATABASE_URL")]
    database_url: String,

    /// Smoke-test the connection: print a count of banks and exit. No TUI.
    #[arg(long)]
    smoke: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;
    // .env is opt-in for local dev; ignore the "not found" case.
    let _ = dotenvy::dotenv();

    let cli = Cli::parse();

    let pool = PgPoolOptions::new()
        .max_connections(4)
        .connect(&cli.database_url)
        .await
        .wrap_err("connecting to prospecta substrate")?;

    // Compile-time-checked query against the live schema.
    // If this won't compile, your DATABASE_URL at build time isn't pointing at a
    // migrated prospecta DB (need at least 0001 applied). See tui/README.md.
    let row = sqlx::query!("SELECT count(*) AS n FROM banks")
        .fetch_one(&pool)
        .await
        .wrap_err("querying banks")?;

    let n_banks = row.n.unwrap_or(0);

    if cli.smoke {
        println!("prospecta-tui smoke: connected, {} banks visible", n_banks);
        return Ok(());
    }

    // TUI proper not implemented yet. Until it is, --smoke is the contract.
    eprintln!(
        "prospecta-tui v0.1 scaffold (TUI not yet wired). {} banks visible.\n\
         Run with --smoke for an explicit health check.\n\
         Next: Ratatui event loop + bank-list view (see HANDOFF doc).",
        n_banks
    );
    Ok(())
}
