"""prospecta.cli._common — shared CLI helpers.

`make_memory` constructs a Memory from args+env. For v0.1 this requires the
caller's environment to provide an LLM/embed (via prospecta.defaults, T23).
If T23 hasn't shipped, surfaces an actionable error.

Tests monkeypatch make_memory to inject pre-built Memory fixtures.
"""
from __future__ import annotations

import json
import os
import sys
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

if TYPE_CHECKING:
    from prospecta.memory import Memory


# ---------------------------------------------------------------------------
# Memory construction
# ---------------------------------------------------------------------------

def make_memory(args) -> "Memory":
    """Construct a Memory from CLI args + env.

    For v0.1, this attempts to load defaults from `prospecta.defaults` (T23,
    not yet shipped). If that import fails, raise a clear actionable error
    pointing at the missing extras / T23 work.

    Tests monkeypatch this function to inject fixture-built Memory.
    """
    from prospecta.memory import Memory

    if not args.database_url:
        print(
            "error: --database-url or DATABASE_URL env var required",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # Resolve embed and llm independently so an offline embedder
    # (PROSPECTA_EMBEDDER=sentence-transformers) works even when LiteLLM is
    # not installed. Combined with `retain --index-text "..."` (which skips
    # LLM index_text generation), this gives a fully offline retain path.
    embed = None
    try:
        embed = _resolve_embedder()
    except ImportError:
        # Only the default/litellm path reaches here — _resolve_embedder
        # converts an EXPLICIT offline selection's import failure into a
        # loud SystemExit. So this swallow is the litellm-missing case:
        # embed stays None; write subcommands surface a RuntimeError at use
        # time, while read-only paths (stats, config) still work.
        pass

    llm = None
    try:
        from prospecta import defaults  # type: ignore[import-not-found]
        llm = defaults.make_default_llm()  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        # LiteLLM not installed / T23 defaults absent. Read-only paths
        # (stats, config) and offline retain (--index-text) work without llm.
        pass

    return Memory(
        database_url=args.database_url,
        bank_id=args.bank,
        llm=llm,
        embed=embed,
    )


def _resolve_embedder():
    """Resolve an EmbedCallable from PROSPECTA_EMBEDDER (offline-selectable).

    Selection (case-insensitive):
      - unset / "litellm" / "default" → prospecta.defaults.make_default_embedder()
        (LiteLLM, multi-provider; honors PROSPECTA_EMBED_MODEL).
      - "sentence-transformers" / "st" → prospecta.embed.sentence_transformers(...)
        (offline, no API key; model from PROSPECTA_EMBED_MODEL, default
        all-MiniLM-L6-v2 / 384-dim).

    Unknown values raise SystemExit(1) with the valid set, rather than
    silently falling back — wrong-embedder mismatches corrupt a bank's
    vector space, so failing loud is the honest default.

    Provider-import behavior is asymmetric by design:
      - The default/litellm path soft-fails: if LiteLLM is missing the
        ImportError propagates to make_memory, which builds a Memory without
        embed so read-only commands (stats, config) still work.
      - An EXPLICIT offline selection that can't import its provider fails
        loud here with the actionable install hint, rather than devolving
        into a confusing downstream RuntimeError about Memory(embed=...).
    """
    kind = os.environ.get("PROSPECTA_EMBEDDER", "default").strip().lower()

    if kind in ("", "default", "litellm"):
        from prospecta import defaults  # type: ignore[import-not-found]
        return defaults.make_default_embedder()  # type: ignore[attr-defined]

    if kind in ("sentence-transformers", "sentence_transformers", "st"):
        from prospecta.embed import sentence_transformers
        model = os.environ.get("PROSPECTA_EMBED_MODEL", "all-MiniLM-L6-v2")
        try:
            return sentence_transformers(model)
        except ImportError as e:
            # User explicitly asked for the offline embedder but the extra
            # isn't installed. Surface the factory's actionable install hint
            # instead of letting make_memory swallow it.
            print(f"error: {e}", file=sys.stderr)
            raise SystemExit(1) from e

    print(
        f"error: unknown PROSPECTA_EMBEDDER={kind!r}; "
        "valid: default | litellm | sentence-transformers (alias: st)",
        file=sys.stderr,
    )
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

def redact_database_url(url: str | None) -> str:
    """Redact password in a postgres URL.

    postgres://user:password@host/db → postgres://user:****@host/db
    """
    if not url:
        return "(unset)"
    try:
        parsed = urlparse(url)
        if parsed.password:
            # Reconstruct netloc with redacted password
            new_netloc = parsed.username or ""
            new_netloc += ":****"
            new_netloc += f"@{parsed.hostname}"
            if parsed.port:
                new_netloc += f":{parsed.port}"
            return urlunparse(parsed._replace(netloc=new_netloc))
        return url
    except Exception:
        return url


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------

def pretty_print_recalled(results) -> None:
    """Pretty-print a list[RecalledMemory] for `prospecta search` output.

    P5: full content, no truncation.
    """
    if not results:
        print("(no results)")
        return
    for i, r in enumerate(results, start=1):
        scores = r.scores or {}
        score_parts = []
        for k in ("rrf", "semantic", "lexical"):
            v = scores.get(k)
            if v is not None:
                score_parts.append(f"{k}={v:.4f}")
        score_str = " ".join(score_parts) or f"score={r.score:.4f}"
        print(f"--- [{i}] {score_str}  source={r.source}")
        print(r.content)
        print()


def recalled_to_json(results) -> str:
    """Serialize list[RecalledMemory] to JSON."""
    payload = []
    for r in results:
        payload.append(
            {
                "content": r.content,
                "original_chunk": r.original_chunk,
                "source": r.source,
                "score": r.score,
                "scores": dict(r.scores or {}),
                "metadata": dict(r.metadata or {}),
                "bank_id": r.bank_id,
                "document_id": r.document_id,
            }
        )
    return json.dumps(payload, indent=2, default=str)


def pretty_print_index_stats(stats) -> None:
    """Pretty-print an IndexStats dataclass."""
    print(f"Files scanned:       {stats.files_scanned}")
    print(f"Files unchanged:     {stats.files_unchanged}")
    print(f"Files skipped:       {stats.files_skipped}")
    print(f"Documents added:     {stats.documents_added}")
    print(f"Documents replaced:  {stats.documents_replaced}")
    print(f"Memory items added:  {stats.memory_items_added}")
    print(f"Errors:              {stats.errors}")


def pretty_print_sweep_result(result) -> None:
    """Pretty-print a single SweepResult."""
    print(f"--- corpus: {result.corpus_path}")
    print(f"  files_scanned: {result.files_scanned}")
    print(f"  files_indexed: {result.files_indexed}")
    print(f"  files_skipped: {result.files_skipped}")
    print(f"  errors:        {result.errors}")
    print(f"  duration_ms:   {result.duration_ms}")
    if result.error_paths:
        print(f"  error_paths:")
        for path, err in result.error_paths:
            print(f"    {path}: {err}")


def pretty_print_stats_table(stats: dict) -> None:
    """Pretty-print a stats dict (key=value, one per line)."""
    width = max((len(k) for k in stats), default=0)
    for k, v in stats.items():
        print(f"  {k.ljust(width)}  {v}")
