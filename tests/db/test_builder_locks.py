"""The post-commit index builders (finish_0004 +1, finish_0005 +2, the keyword index +3, the
exact_tsv index +4: advisory keys MIGRATE_LOCK_KEY + n) serialize on those keys so that no two
CREATE INDEX CONCURRENTLY run on the table at once (they can deadlock each other).

The first version of that rule had each builder take its OWN key and then poll the OTHER
builders' keys. Two builders running at once in different processes (`prospecta keyword-index
--rebuild` during a migration) then each held their own key and polled the other's forever: a
livelock. The rule now: every builder takes ALL the builder keys, its own included, in ONE fixed
ascending order (polling, never blocking), and releases them all."""
from __future__ import annotations

import multiprocessing as mp
import threading
import time

import psycopg
import pytest

from prospecta.db.exact_tsv import ensure_exact_tsv_index
from prospecta.db.keyword_index import ensure_keyword_index
from prospecta.db.migrate import MIGRATE_LOCK_KEY, finish_0004, finish_0005, run_migrations

BUILDER_KEYS = [MIGRATE_LOCK_KEY + n for n in (1, 2, 3, 4)]

BUILDERS = {
    1: lambda url: finish_0004(url),
    2: lambda url: finish_0005(url),
    3: lambda url: ensure_keyword_index(url, rebuild=True),
    4: lambda url: ensure_exact_tsv_index(url, rebuild=True),
}


def test_the_keys_are_the_migrate_keys_in_ascending_order():
    from prospecta.db.builder_locks import BASE_KEY, BUILDER_KEYS as KEYS
    from prospecta.db.exact_tsv import LOCK_KEY
    assert BASE_KEY == MIGRATE_LOCK_KEY
    assert list(KEYS) == BUILDER_KEYS == sorted(BUILDER_KEYS)
    assert LOCK_KEY == MIGRATE_LOCK_KEY + 4


def _held_elsewhere(watcher: psycopg.Connection, key: int) -> bool:
    """True when another session holds the advisory key."""
    got = watcher.execute("SELECT pg_try_advisory_lock(%s)", (key,)).fetchone()[0]
    if got:
        watcher.execute("SELECT pg_advisory_unlock(%s)", (key,))
    return not got


@pytest.mark.parametrize("builder", [1, 2, 3, 4])
@pytest.mark.parametrize("peer", [1, 2, 3, 4])
def test_a_builder_takes_every_key_in_ascending_order_and_releases_them_all(fresh_db, builder, peer):
    """While a peer holds its key the builder waits (polling: no backend waits on an advisory
    lock) having taken exactly the keys BELOW the peer's, own key included once it is below,
    and none above: one fixed ascending order is what rules out a cycle. Once the peer lets
    go it builds and every key is free again."""
    with psycopg.connect(fresh_db, autocommit=True) as holder, \
            psycopg.connect(fresh_db, autocommit=True) as watcher:
        holder.execute("SELECT pg_advisory_lock(%s)", (MIGRATE_LOCK_KEY + peer,))
        done = []
        t = threading.Thread(target=lambda: done.append(BUILDERS[builder](fresh_db)))
        t.start()
        time.sleep(0.6)
        assert t.is_alive()                                        # waiting its turn ...
        assert watcher.execute("SELECT count(*) FROM pg_stat_activity WHERE wait_event_type = 'Lock' "
                               "AND wait_event = 'advisory'").fetchone()[0] == 0   # ... by polling
        for n in (1, 2, 3, 4):
            expected = n <= peer                                    # the peer's own key is the holder's
            assert _held_elsewhere(watcher, MIGRATE_LOCK_KEY + n) is expected, (builder, peer, n)
        holder.execute("SELECT pg_advisory_unlock(%s)", (MIGRATE_LOCK_KEY + peer,))
        t.join(timeout=60)
        assert not t.is_alive() and done
        for key in BUILDER_KEYS:
            assert not _held_elsewhere(watcher, key), key


def _worker(url: str, builder: int, rounds: int, barrier, out) -> None:
    try:
        for _ in range(rounds):
            barrier.wait(timeout=60)                               # start the two builds together
            BUILDERS[builder](url)
        out.put((builder, "ok"))
    except Exception as exc:                                       # pragma: no cover - failure path
        out.put((builder, f"error: {type(exc).__name__}: {exc}"))


@pytest.mark.parametrize("pair", [(3, 4), (1, 4), (2, 3)])
def test_two_builders_running_at_once_in_two_processes_finish(fresh_db, pair):
    """The livelock: builder A holds its own key and polls B's, B holds its own and polls A's,
    both forever. Two processes run two different builders together, round after round, and
    both must finish inside the timeout."""
    run_migrations(fresh_db)
    ctx = mp.get_context("spawn")
    barrier, out = ctx.Barrier(2), ctx.Queue()
    rounds = 12
    procs = [ctx.Process(target=_worker, args=(fresh_db, b, rounds, barrier, out)) for b in pair]
    for p in procs:
        p.start()
    deadline = time.monotonic() + 60
    for p in procs:
        p.join(timeout=max(1.0, deadline - time.monotonic()))
    stuck = [p for p in procs if p.is_alive()]
    for p in stuck:
        p.terminate()
    assert not stuck, f"builders {pair} livelocked: still running after the timeout"
    results = [out.get(timeout=5) for _ in procs]
    assert all(r[1] == "ok" for r in results), results
