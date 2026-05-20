"""Entry point for the `prospecta` CLI.

T15 — full surface: migrate, create-bank, index, search, retain, stats,
config, sweep. Each subcommand is its own module; this dispatcher only
wires argparse and routes args.command to the right handler.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prospecta",
        description="Bilateral LLM-mediated memory library — CLI",
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="Postgres connection URL (default: DATABASE_URL env)",
    )
    parser.add_argument(
        "--bank",
        default=os.environ.get("PROSPECTA_BANK", "prospecta"),
        help="Bank id (default: 'prospecta' or PROSPECTA_BANK env)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- init (T22, P16) -------------------------------------------------
    init_p = sub.add_parser(
        "init",
        help="One-command bootstrap: substrate up + migrate + default bank",
    )
    init_p.add_argument(
        "--no-substrate",
        action="store_true",
        help="Skip docker-compose substrate detection (use existing DATABASE_URL)",
    )
    init_p.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Embedding dim for default bank (default: PROSPECTA_EMBEDDING_DIM or 1536)",
    )
    init_p.add_argument(
        "--substrate-timeout",
        type=float,
        default=60.0,
        help="Max seconds to wait for docker compose healthcheck (default: 60)",
    )

    # ---- migrate (T3) ---------------------------------------------------
    sub.add_parser("migrate", help="Apply pending database migrations")

    # ---- create-bank (T4) -----------------------------------------------
    cb = sub.add_parser(
        "create-bank", help="Create a bank with given embedding dimensionality"
    )
    cb.add_argument("--id", required=True, help="Bank id")
    cb.add_argument("--embedding-dim", type=int, required=True)
    cb.add_argument("--embedding-model-id", default=None)
    cb.add_argument("--mission", default=None)
    cb.add_argument("--retain-mission", default=None)

    # ---- index (T15) ----------------------------------------------------
    ix = sub.add_parser("index", help="Index a corpus directory")
    ix.add_argument("path", type=Path)
    ix.add_argument("--rebuild", action="store_true")
    ix.add_argument("--source-prefix", default=None)

    # ---- search (T15) ---------------------------------------------------
    sr = sub.add_parser("search", help="Hybrid retrieval")
    sr.add_argument("query")
    sr.add_argument(
        "--mode",
        choices=["hybrid", "semantic", "lexical"],
        default="hybrid",
    )
    sr.add_argument("--limit", type=int, default=10)
    sr.add_argument("--rrf-k", type=int, default=60)
    sr.add_argument(
        "--json", action="store_true", help="Emit JSON (else pretty output)"
    )

    # ---- retain (T15) ---------------------------------------------------
    rt = sub.add_parser("retain", help="Write a document via the spine")
    rt.add_argument("content", help="Body content, or @path/to/file")
    rt.add_argument("--source", default=None)
    rt.add_argument("--path", default=None, type=Path)
    rt.add_argument("--tags", default=None, help="Comma-separated tags")
    rt.add_argument(
        "--index-text",
        action="append",
        default=None,
        help="Override LLM index_text generation (repeat for multi)",
    )

    # ---- stats (T15) ----------------------------------------------------
    st = sub.add_parser("stats", help="Show counters across substrate")
    st.add_argument(
        "--bank-only",
        action="store_true",
        help="Stats for active bank only",
    )

    # ---- config (T15) ---------------------------------------------------
    sub.add_parser("config", help="Print loaded config (redacted)")

    # ---- sweep (T15) ----------------------------------------------------
    sw = sub.add_parser("sweep", help="Run sweeper")
    sw.add_argument("path", type=Path, nargs="+", help="Corpus path(s)")
    sw.add_argument(
        "--once",
        action="store_true",
        help="Run one pass synchronously, exit",
    )
    sw.add_argument(
        "--watch",
        action="store_true",
        help="Start daemon; block until SIGINT",
    )
    sw.add_argument("--interval", type=float, default=86400.0)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "init":
        from prospecta.cli.init import cmd_init
        return cmd_init(args)
    if args.command == "migrate":
        from prospecta.cli import migrate as migrate_cmd
        return migrate_cmd.run(args)
    if args.command == "create-bank":
        from prospecta.cli.bank import run_create_bank
        return run_create_bank(args)
    if args.command == "index":
        from prospecta.cli.index import cmd_index
        return cmd_index(args)
    if args.command == "search":
        from prospecta.cli.search import cmd_search
        return cmd_search(args)
    if args.command == "retain":
        from prospecta.cli.retain import cmd_retain
        return cmd_retain(args)
    if args.command == "stats":
        from prospecta.cli.stats import cmd_stats
        return cmd_stats(args)
    if args.command == "config":
        from prospecta.cli.config import cmd_config
        return cmd_config(args)
    if args.command == "sweep":
        from prospecta.cli.sweep import cmd_sweep
        return cmd_sweep(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
