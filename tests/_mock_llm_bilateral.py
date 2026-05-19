"""Deterministic mock LLM for the bilateral 2×2 matrix test.

Dispatches on prompt content + json_mode to produce shaped responses:

  - Write-side (generate-index-text, json_mode=False): returns
    question-shaped lines anticipated for the known corpus bodies.
  - Read-side (formulate-queries, json_mode=True): returns a JSON
    expansion of the canonical QUERY into 2 sub-questions.
  - Synthesis (rag-synthesize, json_mode=False, not write-side): returns
    a canned synthesis string.

Records every call on self.calls for tests to inspect.
"""
from __future__ import annotations

import json


# The canonical query used by the bilateral test.
QUERY = "did kelly's birthday work out?"

# Mappings keyed on substring(s) in the prompt's `content` variable.
# Each value is a newline-separated string of question-shaped index_text
# lines that the corpus body should be retrievable by.
_INDEX_TEXT_BY_BODY: list[tuple[tuple[str, ...], str]] = [
    # kelly_birthday_arc.md — target. Body uses narrative prose only;
    # the spine's job is to surface question-form retrieval keys that
    # include "kelly" and "birthday" lexically.
    (
        ("March came around", "Candles were lit"),
        "When is Kelly's birthday?\n"
        "What happened on Kelly's birthday?\n"
        "Did Kelly's birthday celebration go well?\n"
        "How did the family celebrate Kelly's birthday?",
    ),
    # pacific_beach_logistics.md — red herring. Body already uses kelly
    # and birthday lexically; the spine indexes the document by its
    # actual topic (beach event logistics), not Kelly's birthday.
    (
        ("Beach event logistics", "Kelly Point Beach"),
        "What permits are needed for beach events?\n"
        "How are catering arrangements made for ocean venues?\n"
        "How long in advance should a beach venue be booked?",
    ),
    (
        ("compiler", "abstract syntax tree"),
        "How does the compiler emit bytecode?\n"
        "When are module imports resolved?",
    ),
    (
        ("garden", "tulips bloomed"),
        "What flowers bloom in the garden first?\n"
        "How is the compost pile managed?",
    ),
    (
        ("software module", "runtime symbol table"),
        "How are function definitions loaded at runtime?\n"
        "When do module imports resolve at startup?",
    ),
    (
        ("seed catalog",),
        "What is in the seed catalog?\n"
        "How should soil be amended before planting?",
    ),
    (
        ("Tide tables", "sand erosion"),
        "What are the tide conditions on the pacific coastline?\n"
        "Why are coastal construction permits suspended?",
    ),
    (
        ("catering vendor review", "approved suppliers"),
        "How are catering vendors approved for events?\n"
        "What does the logistics planning committee review?",
    ),
]


# Formulate-queries response for the canonical QUERY. Two sub-questions
# both shaped to land in the kelly_birthday cluster + carry the
# question-shape signal.
_FORMULATE_RESPONSE = json.dumps({
    "queries": [
        {"text": "When is Kelly's birthday?"},
        {"text": "How did Kelly's birthday celebration go?"},
    ]
})


class BilateralMockLLM:
    """LLM mock that produces deterministic, semantically shaped output.

    Conforms to LLMCallable Protocol: __call__(messages, *, json_mode=False) -> str.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, messages: list[dict], *, json_mode: bool = False) -> str:
        self.calls.append({"messages": messages, "json_mode": json_mode})

        # Extract the rendered prompt text.
        rendered = ""
        if messages:
            content = messages[-1].get("content", "")
            if isinstance(content, str):
                rendered = content

        if json_mode:
            # Read-side formulate-queries.
            return _FORMULATE_RESPONSE

        # Write-side generate-index-text vs synthesis: distinguish by
        # looking for the index-text prompt's signature phrase.
        if "question-form index entries" in rendered or "Questions:" in rendered:
            for substrings, response in _INDEX_TEXT_BY_BODY:
                if all(s in rendered for s in substrings):
                    return response
            # Default for unknown body: emit something generic.
            return "What is this document about?"

        # Synthesis or anything else.
        return "Synthesis based on retrieved chunks."
