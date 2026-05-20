"""Test helpers for the bilateral 2×2 matrix integration test (T17).

Builds a Memory selectively wired with write-side and/or read-side spine
disabled via monkey-patching, with all four cells running against the
SAME canonical corpus. This is the cleanest way to demonstrate the
bilateral effect without diverging fixtures across cells.

NOTE: Lives in prospecta/ rather than tests/ because importing from
tests/ across packages is brittle; the module is private (_-prefixed)
and only the bilateral test imports it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prospecta._types import EmbedCallable, LLMCallable, Tracer
    from prospecta.memory import Memory


def build_memory_with_spine_config(
    write_on: bool,
    read_on: bool,
    *,
    database_url: str,
    bank_id: str,
    embed: "EmbedCallable",
    llm: "LLMCallable",
    tracer: "Tracer | None" = None,
) -> "Memory":
    """Construct a Memory with the spine selectively enabled.

    write_on=False: monkeypatch generate_index_text to echo raw content
                    as a single index_text string (classical RAG: chunk
                    body == memory_items.content).
    write_on=True:  normal spine path; the supplied llm produces
                    question-shaped index_text.

    read_on=False:  monkeypatch Memory.formulate_queries (instance-level)
                    to return [Query(text=message)] (identity).
    read_on=True:   normal spine path; the supplied llm produces multi-
                    query expansion.
    """
    from prospecta._types import Query
    from prospecta.memory import Memory

    mem = Memory(
        database_url=database_url,
        bank_id=bank_id,
        llm=llm,
        embed=embed,
        tracer=tracer,
    )

    if not write_on:
        # Monkeypatch the module-level reference that _retain.py uses.
        # _retain.py imports `generate_index_text` at module top, so we
        # patch _retain's binding (not _index_text's).
        from prospecta import _retain as _retain_mod

        def _identity_index_text(content, llm, *, prompt_override=None, context=None):
            # Classical RAG: index_text == content (single chunk).
            # Match generate_index_text's tuple shape (parsed, prompt, raw).
            return [content], "", ""

        mem._test_orig_generate_index_text = _retain_mod.generate_index_text  # type: ignore[attr-defined]
        _retain_mod.generate_index_text = _identity_index_text  # type: ignore[assignment]

    if not read_on:
        # Instance-level monkeypatch: skip LLM, return identity.
        def _identity_formulate(self_, message, *, context=None, prompt_override=None):
            return [Query(text=message)]

        import types
        mem.formulate_queries = types.MethodType(_identity_formulate, mem)  # type: ignore[method-assign]

    return mem


def restore_module_patches(mem: "Memory") -> None:
    """Undo any module-level monkeypatches applied by build_memory_with_spine_config.

    Module-level patches (generate_index_text) leak across tests if not
    restored. Call this in test teardown or after the cell's recall_synth.
    """
    orig = getattr(mem, "_test_orig_generate_index_text", None)
    if orig is not None:
        from prospecta import _retain as _retain_mod
        _retain_mod.generate_index_text = orig
        try:
            delattr(mem, "_test_orig_generate_index_text")
        except AttributeError:
            pass
