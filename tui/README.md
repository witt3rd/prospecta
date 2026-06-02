# prospecta-tui

Interactive terminal UI for managing, using, searching, and reviewing a
[prospecta](../README.md) memory substrate.

prospecta is the bilateral-prospective-synthesis memory library; everything
load-bearing happens through six event tables in Postgres. This TUI is the
inspection surface for that durable trace (principle P19), *and* the operator
surface for substrate management — bank browsing, document drill-down, manual
retain, and cross-channel search.

## Status

**v0.1.** Five tabs, all four feature axes live (manage · use · search ·
observability). Read paths go direct via `sqlx` (compile-time-checked against
the schema); the retain write path shells out to the Python `prospecta` CLI so
the LLM-in-the-loop work and the spine stay library-owned. The schema is the
contract; the TUI never mutates it.

## Build & run

The TUI connects to a running prospecta Postgres. The cheapest path from a
clean checkout:

```bash
# From the repo root, in a sibling shell — stand up the substrate:
docker compose up -d
prospecta migrate
prospecta create-bank --id default --embedding-dim 32

# In tui/:
cp .env.example .env          # then set DATABASE_URL
cargo run                     # launches the TUI
```

`DATABASE_URL` resolves from the `--database-url` flag, the `DATABASE_URL` env
var, or a `.env` file (in that order).

### Compile-time note (offline by default)

The `sqlx::query!` macros verify SQL against the schema at compile time. A
committed `.sqlx/` query cache lets the crate build **with no database
present** — `cargo build` and `cargo install` work offline out of the box.

If you change a query, regenerate the cache against a live DB:

```bash
cargo install sqlx-cli --no-default-features --features postgres,rustls
DATABASE_URL=postgres://… cargo sqlx prepare   # rewrites .sqlx/, commit it
```

## Install

```bash
cargo install --path tui      # from the repo root
prospecta-tui --database-url postgres://…
```

Produces a single `prospecta-tui` binary on your `$PATH`. No system deps beyond
what `cargo build` needs; builds offline via the committed `.sqlx/` cache.

## Tabs

| # | Tab | What it shows |
|---|-----|---------------|
| 1 | **banks** | All banks with per-bank counts; `Enter` drills into documents |
| 2 | **docs** | Documents in a bank → `Enter` descends into a document's `memory_items` (the question-shaped index_text) |
| 3 | **events** | Live tail across all five event tables; `Enter` on a recall/retain row opens the full **thread** (formulate → recall → synthesis → llm_calls, or retain → document → items → index_text call) |
| 4 | **dashboard** | Per-bank 24h health: substrate counts, retain/recall rates, latency, the P18 parse_fallback safety signal, per-prompt llm_calls, and the **sweep status** panel (latest pass per corpus, ok/error/running + age) |
| 5 | **search** | Free-text over both lexical channels — `content_tsv` (the anticipated questions) and `body_tsv` (the source body). Each hit is labelled `both` / `question` / `body` so the P14 honest-safety-net is visible. `R` opens the **retain** form |

## Keybindings

| Key | Action |
|-----|--------|
| `Tab` / `⇧Tab` | cycle tabs (banks → docs → events → dashboard → search) |
| `1`–`5` | jump to a tab directly |
| `Enter` / `→` / `l` | drill down (bank → docs → items; recall/retain → thread) |
| `Esc` / `←` / `h` | ascend one level |
| `↑↓` / `j` `k` | select row |
| `g` / `G` (Home/End) | first / last row |
| `f` | toggle auto-scroll (events tab) |
| `r` | reload the current view from the substrate |
| **search tab** — type + `Enter` | run search; `Esc` toggles edit / browse |
| **search tab** — `R` | open the retain form |
| **retain form** — `Tab`/`↑↓` | move between fields |
| **retain form** — `Ctrl-S` | submit (shell-out to `prospecta retain`) |
| **retain form** — `Esc` | cancel |
| `?` | toggle the keybinding help overlay |
| `q` / `Ctrl-C` | quit |

Color output honors `NO_COLOR`.

## Manual retain (the write path)

The retain form shells out to the Python `prospecta retain` CLI. Connection +
embedder env inherit from the TUI; only the bank is pinned per-form. Two
variables configure the invocation so the same binary works in dev and
deployed:

- `PROSPECTA_CLI` — command prefix (default `prospecta`; e.g.
  `PROSPECTA_CLI="uv run prospecta"` when running against a co-located checkout)
- `PROSPECTA_CLI_CWD` — working directory for the child (lets `uv run` resolve
  the project)

For a fully **offline, keyless** retain, set
`PROSPECTA_EMBEDDER=sentence-transformers` (see the library README's
"Offline / keyless retain" section) and provide an `index_text` override in the
form — that skips the LLM index_text generation.

The form reports the honest CLI outcome: the document UUID on success, or the
verbatim CLI error on failure. It never fabricates success.

## Headless verifiers

Every interactive surface has a `--dump-*` flag that exercises the same query
path and exits — useful for sanity-checking the substrate or scripting without
launching the TUI:

```bash
prospecta-tui --smoke                              # connect + count banks
prospecta-tui --dump-events                        # recent events, all tables
prospecta-tui --dump-docs default                  # documents in a bank
prospecta-tui --dump-dashboard default             # 24h health signals
prospecta-tui --dump-search "default:bilateral"    # cross-channel search
prospecta-tui --dump-retain "default:some content" # retain shell-out
```

## Architecture

Read paths via `sqlx` direct against Postgres, checked against the live schema
at compile time. Write paths (retain) shell out to the Python `prospecta` CLI —
keeps the LLM-in-the-loop work owned by the library. The schema is the
contract; the TUI does not introduce schema changes. If a query needs schema it
doesn't have, that's a coordination issue against the library, not a TUI-side
migration.

## License

MIT.
