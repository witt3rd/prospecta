//! Shell-out to the Python `prospecta` CLI for write paths.
//!
//! Read paths go direct via sqlx. Write paths (retain) shell out to the
//! library CLI so the LLM-in-the-loop work and the spine stay Python-owned —
//! the schema is the contract, the CLI is the writer (handoff decision 3).
//!
//! Invocation is configurable so the same binary works in dev (co-located
//! repo, `uv run prospecta` from the library root) and in a deployed setup
//! (`prospecta` on PATH):
//!   - `PROSPECTA_CLI`     — whitespace-split command prefix; default
//!                           `prospecta`. e.g. `PROSPECTA_CLI="uv run prospecta"`.
//!   - `PROSPECTA_CLI_CWD` — working dir for the child (lets `uv run` resolve
//!                           the project); default: inherit the TUI's cwd.
//!
//! Connection + embedder env (DATABASE_URL, PROSPECTA_EMBEDDER,
//! PROSPECTA_EMBED_MODEL) inherit from the TUI's own environment, so the CLI
//! retains into the same substrate the TUI is inspecting. Only PROSPECTA_BANK
//! is overridden per-call to pin the form's selected bank.

use tokio::process::Command;

/// Inputs for a single retain shell-out.
#[derive(Debug, Clone)]
pub struct RetainArgs {
    pub bank_id: String,
    pub content: String,
    pub source: Option<String>,
    /// Comma-separated tag string (passed through to `--tags`).
    pub tags: Option<String>,
    /// `--index-text` overrides (repeatable). When non-empty, the CLI skips
    /// LLM index_text generation (P4) — this is the fully-offline retain path.
    pub index_text: Vec<String>,
}

/// Result of a retain shell-out — the honest outcome contract: we report what
/// the child process actually returned, never a fabricated success.
#[derive(Debug, Clone)]
pub struct RetainOutcome {
    pub ok: bool,
    /// Parsed document UUID from stdout on success.
    pub document_id: Option<String>,
    pub stdout: String,
    pub stderr: String,
    pub code: Option<i32>,
}

/// Resolve the CLI command prefix from `PROSPECTA_CLI` (whitespace-split),
/// defaulting to a bare `prospecta` on PATH.
fn cli_prefix() -> Vec<String> {
    std::env::var("PROSPECTA_CLI")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.split_whitespace().map(String::from).collect::<Vec<_>>())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec!["prospecta".to_string()])
}

/// Shell out to `prospecta retain`. Returns the real child outcome — caller
/// renders success (document_id) or the verbatim stderr on failure.
pub async fn retain(args: &RetainArgs) -> std::io::Result<RetainOutcome> {
    let prefix = cli_prefix();
    let (program, base) = prefix
        .split_first()
        .expect("cli_prefix never returns empty");

    let mut cmd = Command::new(program);
    cmd.args(base);
    cmd.arg("retain").arg(&args.content);

    if let Some(s) = &args.source {
        if !s.trim().is_empty() {
            cmd.arg("--source").arg(s);
        }
    }
    if let Some(t) = &args.tags {
        if !t.trim().is_empty() {
            cmd.arg("--tags").arg(t);
        }
    }
    for it in &args.index_text {
        if !it.trim().is_empty() {
            cmd.arg("--index-text").arg(it);
        }
    }

    // Pin the form's bank; connection + embedder env inherit from the TUI.
    cmd.env("PROSPECTA_BANK", &args.bank_id);

    if let Ok(dir) = std::env::var("PROSPECTA_CLI_CWD") {
        if !dir.trim().is_empty() {
            cmd.current_dir(dir);
        }
    }

    let out = cmd.output().await?;
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let ok = out.status.success();

    // On success the CLI prints the document UUID as its stdout. Take the last
    // non-empty stdout line (progress bars / warnings go to stderr).
    let document_id = if ok {
        stdout
            .lines()
            .rev()
            .map(str::trim)
            .find(|l| !l.is_empty())
            .map(String::from)
    } else {
        None
    };

    Ok(RetainOutcome {
        ok,
        document_id,
        stdout,
        stderr,
        code: out.status.code(),
    })
}
