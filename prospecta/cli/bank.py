"""`prospecta create-bank` and `prospecta stats` CLI subcommands."""
from __future__ import annotations

import json
import sys

from prospecta.memory import BankConfigConflict, Memory


def run_create_bank(args) -> int:
    if not args.database_url:
        print("error: --database-url or DATABASE_URL env var required", file=sys.stderr)
        return 2
    mem = Memory(database_url=args.database_url, bank_id=args.id)
    try:
        mem.create_bank(
            args.id,
            embedding_dim=args.embedding_dim,
            embedding_model_id=args.embedding_model_id,
            mission=args.mission,
            retain_mission=args.retain_mission,
        )
    except BankConfigConflict as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 4
    finally:
        mem.close()
    print(f"Bank {args.id!r} ready (embedding_dim={args.embedding_dim})")
    return 0


def run_stats(args) -> int:
    if not args.database_url:
        print("error: --database-url or DATABASE_URL env var required", file=sys.stderr)
        return 2
    bank_id = args.bank or "prospecta"
    mem = Memory(database_url=args.database_url, bank_id=bank_id)
    try:
        stats = mem.bank_stats(bank_id)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    finally:
        mem.close()
    output = {
        "bank_id": stats.bank_id,
        "documents": stats.documents,
        "memory_items": stats.memory_items,
        "last_retain_at": stats.last_retain_at.isoformat() if stats.last_retain_at else None,
    }
    print(json.dumps(output, indent=2))
    return 0


def run_migrate_bank(args) -> int:
    """`prospecta migrate-bank`: see docs/embedding-migration.md (dump first)."""
    from prospecta._embed_migrate import migrate_bank

    if not args.database_url:
        print("error: --database-url or DATABASE_URL env var required", file=sys.stderr)
        return 2
    try:
        from prospecta import defaults
        embed = defaults.make_default_embedder(
            args.embed_model, dimensions=args.embedding_dim
        )
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    mem = Memory(database_url=args.database_url, bank_id=args.source_bank)
    try:
        report = migrate_bank(
            mem, args.source_bank, target_bank=args.target_bank, embed=embed,
            embedding_dim=args.embedding_dim,
            embedding_model_id=args.embedding_model_id,
            batch_size=args.batch_size, max_documents=args.max_documents,
        )
    except BankConfigConflict as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"error: {e} (re-run the same command to resume)", file=sys.stderr)
        return 4
    finally:
        mem.close()
    print(json.dumps(report.__dict__, indent=2))
    return 0 if report.remaining == 0 else 5
