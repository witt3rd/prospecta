"""Capable defaults for prospecta — LiteLLM-backed factories (P17).

Gated behind ``pip install prospecta[defaults]``. The core library keeps
zero provider imports (P3); only this subpackage imports LiteLLM.

Public surface::

    from prospecta.defaults import make_default_embedder, make_default_llm

    embed = make_default_embedder()  # PROSPECTA_EMBED_MODEL or default
    llm   = make_default_llm()       # PROSPECTA_LLM_MODEL or default

If LiteLLM is not installed, importing this module raises ``ImportError``
with a clear install hint.
"""
from __future__ import annotations

try:
    from ._litellm import make_default_embedder, make_default_llm
except ImportError as exc:  # pragma: no cover - exercised by gate test when extra missing
    raise ImportError(
        "prospecta.defaults requires LiteLLM. "
        "Install with: pip install 'prospecta[defaults]'"
    ) from exc

__all__ = ["make_default_embedder", "make_default_llm"]
