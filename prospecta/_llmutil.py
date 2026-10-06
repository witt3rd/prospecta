"""Accounted-LLM helper: every LLM user accepts a bare str or an LLMResult
(prospecta.stages.LLMResult: text plus model, tokens and cost)."""
from __future__ import annotations


def llm_text(out) -> str:
    """The reply text of an LLM callable's return value (str or LLMResult)."""
    if isinstance(out, str):
        return out
    text = getattr(out, "text", None)
    return text if isinstance(text, str) else str(out)


def llm_call_record(out, *, purpose: str, model: str | None = None) -> dict:
    """A call record (the shape stages.call_llm appends) for a return value."""
    return {
        "purpose": purpose,
        "model": getattr(out, "model", None) or model,
        "tokens_in": getattr(out, "tokens_in", None),
        "tokens_out": getattr(out, "tokens_out", None),
        "cost_usd": getattr(out, "cost_usd", None),
    }
