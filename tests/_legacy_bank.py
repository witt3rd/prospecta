"""A legacy bank, for the tests of the legacy path.

`Memory.create_bank` gives a new bank the hybrid architecture (channel_config = [hybrid_chunk],
a cross_encoder rerank, linking off). Banks created before that stored the column defaults
(`[]`, `{}`, `{}`) and run the legacy three-channel recall, which the library still supports; the
tests of that path build such a bank here. The new defaults are tested in
tests/test_new_bank_defaults.py."""
from __future__ import annotations


def create_legacy_bank(mem, bank_id: str, *, embedding_dim: int, **kw) -> None:
    mem.create_bank(bank_id, embedding_dim=embedding_dim, **kw)
    mem.set_channel_config([], bank_id)
    mem.set_recall_config({}, bank_id)
    mem.set_link_config({}, bank_id)
