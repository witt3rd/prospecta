"""Single-provider embedding helpers (P3, P17).

Each helper returns an ``EmbedCallable`` conforming to
``(list[str]) -> list[list[float]]``. The underlying provider library is
**lazy-imported** — it is only imported when the factory is called, not at
module-level. This keeps `import prospecta.embed` free of provider deps.

For multi-provider routing via LiteLLM, see ``prospecta.defaults`` instead.
These helpers are the minimal-deps alternative: one provider per helper,
no LiteLLM dependency.

Install hints::

    pip install 'prospecta[embed-sentence-transformers]'
    pip install 'prospecta[embed-openai]'
    pip install 'prospecta[embed-openai-compatible]'
"""
from __future__ import annotations

from .openai import openai
from .openai_compatible import openai_compatible
from .sentence_transformers import sentence_transformers

__all__ = ["sentence_transformers", "openai", "openai_compatible"]
