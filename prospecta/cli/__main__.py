"""Entry point for `prospecta` CLI."""
from __future__ import annotations

import argparse
import logging
import os
import sys

from prospecta.cli import migrate as migrate_cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prospecta",
        description="Bilateral LLM-mediated memory library",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # `prospecta migrate`
    migrate_parser = subparsers.add_parser("migrate", help="Apply pending database migrations")
    migrate_parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="Postgres connection URL (default: DATABASE_URL env var)",
    )

    # `prospecta create-bank`
    cb = subparsers.add_parser(
        "create-bank",
        help="Create a bank with the given embedding dimensionality",
    )
    cb.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    cb.add_argument("--id", required=True, help="Bank ID (alphanumeric + - + _, max 63 chars)")
    cb.add_argument("--embedding-dim", type=int, required=True)
    cb.add_argument("--embedding-model-id", default=None)
    cb.add_argument("--mission", default=None)
    cb.add_argument("--retain-mission", default=None)

    # `prospecta stats`
    st = subparsers.add_parser("stats", help="Show bank stats (counts, last retain)")
    st.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    st.add_argument("--bank", default=None, help="Bank ID (default: prospecta)")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "migrate":
        return migrate_cmd.run(args)
    elif args.command == "create-bank":
        from prospecta.cli.bank import run_create_bank
        return run_create_bank(args)
    elif args.command == "stats":
        from prospecta.cli.bank import run_stats
        return run_stats(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
