"""Accounted-LLM helper: every LLM user accepts a bare str or an LLMResult
(prospecta.stages.LLMResult: text plus model, tokens and cost)."""
from __future__ import annotations

import inspect


def llm_text(out) -> str:
    """The reply text of an LLM callable's return value (str or LLMResult)."""
    if isinstance(out, str):
        return out
    text = getattr(out, "text", None)
    return text if isinstance(text, str) else str(out)


def llm_call_record(out, *, purpose: str, model: str | None = None) -> dict:
    """A call record (the shape the stages and the query writer append) for a return value."""
    return {
        "purpose": purpose,
        "model": getattr(out, "model", None) or model,
        "tokens_in": getattr(out, "tokens_in", None),
        "tokens_out": getattr(out, "tokens_out", None),
        "cost_usd": getattr(out, "cost_usd", None),
    }


def accepts_model(llm) -> bool:
    """Does this LLM callable take a `model` keyword (a hint for the model to use)?"""
    try:
        params = inspect.signature(llm).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "model" and p.kind != p.VAR_POSITIONAL
               or p.kind == p.VAR_KEYWORD for p in params)
