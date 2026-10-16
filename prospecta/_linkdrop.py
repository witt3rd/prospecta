"""drop-links: remove a bank's link rows without a table lock."""
from __future__ import annotations


def link_rows_and_bytes(conn, bank_id: str) -> tuple[int, int]:
    """Rows of the bank in memory_links, and their estimated size (average row of
    the table plus its indexes, from the planner's statistics)."""
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM memory_links WHERE bank_id = %s", (bank_id,))
        rows = cur.fetchone()[0]
        cur.execute("SELECT pg_total_relation_size('memory_links'), "
                    "GREATEST(reltuples, 0)::bigint FROM pg_class "
                    "WHERE oid = 'memory_links'::regclass")
        total, est = cur.fetchone()
    conn.commit()
    if rows == 0:
        return 0, 0
    per_row = total / max(est, rows)
    return rows, int(per_row * rows)


def drop_links(conn, bank_id: str, *, yes: bool = False, batch: int = 10000) -> dict:
    if not yes:
        raise PermissionError("drop-links removes link rows and needs yes=True (--yes)")
    if batch < 1:
        raise ValueError("batch must be >= 1")
    rows, size = link_rows_and_bytes(conn, bank_id)
    removed = 0
    while True:   # one short transaction per batch: row locks only
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM memory_links WHERE ctid = ANY(ARRAY(SELECT ctid FROM memory_links "
                "WHERE bank_id = %s LIMIT %s FOR UPDATE SKIP LOCKED))", (bank_id, batch))
            n = cur.rowcount
        conn.commit()
        removed += n
        if n == 0:
            break
    # rows counted before the run; report what was actually removed
    return {"rows": removed, "bytes": int(size * removed / rows) if rows else 0}
