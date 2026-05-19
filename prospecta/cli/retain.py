"""`prospecta retain CONTENT` — write a document via the spine."""
from __future__ import annotations

import sys
from pathlib import Path


def cmd_retain(args) -> int:
    from prospecta.cli import _common

    # Resolve content — @path indirection reads from disk.
    content = args.content
    if content.startswith("@"):
        body_path = Path(content[1:])
        try:
            content = body_path.read_text(encoding="utf-8")
        except OSError as e:
            print(f"error: failed to read {body_path}: {e}", file=sys.stderr)
            return 1

    # Parse tags
    tags = None
    if args.tags:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    index_text = args.index_text  # already a list[str] | None from action="append"

    try:
        memory = _common.make_memory(args)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1

    try:
        document_id = memory.retain(
            content,
            index_text=index_text,
            source=args.source,
            path=args.path,
            tags=tags,
        )
    except Exception as e:
        print(f"error: retain failed: {e}", file=sys.stderr)
        return 2

    print(document_id)
    return 0
