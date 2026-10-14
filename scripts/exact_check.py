"""Read-only check of the exact-identifier gate against a live bank.

    python scripts/exact_check.py --url "$DATABASE_URL" --schema prospecta TOKEN [TOKEN ...]
    python scripts/exact_check.py --url "$DATABASE_URL" --schema prospecta --file tokens.txt

For each token it prints how many documents the gate finds (and of which source
kinds) next to how many the full-text legs of the legacy recall find
(`body_tsv` / `content_tsv` with `websearch_to_tsquery`), and the seconds the
gate took. Zero gate documents means the token is in no record: nothing for
any leg to find. The session is read-only (`default_transaction_read_only=on`),
nothing is written, and no record text is printed: only the tokens you pass,
counts and source kinds.
"""
from __future__ import annotations

import argparse
import collections
import re
import sys
import time

import psycopg

from prospecta import _exact


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tokens", nargs="*")
    ap.add_argument("--url", required=True)
    ap.add_argument("--schema", default=None, help="the PROSPECTA_SCHEMA of a shared-database bank")
    ap.add_argument("--bank", default="prospecta")
    ap.add_argument("--file", help="one token per line")
    args = ap.parse_args(argv)
    tokens = list(args.tokens)
    if args.file:
        tokens += [t.strip() for t in open(args.file) if t.strip()]
    if not tokens:
        ap.error("give tokens or --file")
    schema = args.schema
    if schema is not None and not re.fullmatch(r"[a-z_][a-z0-9_]*", schema):
        ap.error("--schema must be a lower-case identifier")
    path = f'-c search_path="{schema}",public' if schema else ""
    conn = psycopg.connect(args.url, options=f"-c default_transaction_read_only=on {path}".strip())
    if conn.execute("SHOW default_transaction_read_only").fetchone()[0] != "on":
        print("refusing: the session is not read-only", file=sys.stderr)
        return 2
    for tok in tokens:
        t0 = time.time()
        hits = _exact.find_exact(conn, args.bank, tok)
        secs = time.time() - t0
        kinds = collections.Counter(h.source.split(":", 1)[0] for h in hits)
        fts = {}
        for col in ("body_tsv", "content_tsv"):
            fts[col] = conn.execute(
                f"SELECT count(DISTINCT document_id) FROM memory_items WHERE bank_id = %s "
                f"AND {col} @@ websearch_to_tsquery('english', %s)", (args.bank, tok)).fetchone()[0]
        conn.rollback()
        print(f"{tok!r}: gate {len(hits)} {dict(kinds)}; full text body={fts['body_tsv']} "
              f"content={fts['content_tsv']}; {secs:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
