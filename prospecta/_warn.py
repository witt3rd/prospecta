"""One warning per process for each stored setting that names retired code.

A bank keeps the config it stored (the library never rewrites it), so a bank created before the retired
recall code was removed may still name a channel, a rerank stage or a reader that no longer exists
(docs/legacy.md). Recall skips what it cannot run and says so here, once per process per setting: a
recall that runs on every turn must not fill a log, and must never raise for it."""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger("prospecta.retired")

_seen: set[str] = set()
_lock = threading.Lock()


def warn_once(key: str, message: str, *args) -> bool:
    """Log `message % args` at WARNING the first time `key` is seen in this process; True if it logged."""
    with _lock:
        if key in _seen:
            return False
        _seen.add(key)
    logger.warning(message, *args)
    return True
