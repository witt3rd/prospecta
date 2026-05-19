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

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "migrate":
        return migrate_cmd.run(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
