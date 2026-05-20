# prospecta-tui

Interactive terminal UI for managing, using, searching, and reviewing a [prospecta](../README.md) memory substrate.

prospecta is the bilateral-prospective-synthesis memory library; everything load-bearing happens through six event tables in Postgres. This TUI is the inspection surface for that durable trace, *and* the operator surface for substrate management (bank list/create, document drill-down, manual retain/recall/search).

## Status

**v0.1 — scaffold.** Connects to a prospecta DB, validates the schema is reachable, runs a smoke query. Ratatui event loop + the four feature axes (manage / use / search / observability) land in subsequent commits on the `tui-v0.1` branch.

## Build

Requires a running prospecta DB at compile time — `sqlx::query!` macros verify SQL against the live schema. The cheapest path:

```bash
# From repo root, in a sibling shell:
docker compose up -d
prospecta migrate
prospecta create-bank --id default --embedding-dim 32

# In tui/:
cp .env.example .env   # then set DATABASE_URL
cargo build
./target/debug/prospecta-tui --smoke
```

A successful smoke prints `prospecta-tui smoke: connected, N banks visible`.

## Architecture (planned)

Read paths via `sqlx` direct against the substrate. Write paths (retain, recall_synth) via shell-out to the Python `prospecta` CLI — keeps the LLM-in-the-loop work owned by the library. The schema is the contract; the TUI does not introduce schema changes.

See the [v0.1 handoff](../../../../forge/notes/2026-05-20_HANDOFF_prospecta-tui-rust.md) for the full scope, locked decisions, and required-reading order.

## License

MIT.
