You are formulating retrieval queries for a semantic memory store using bilateral LLM-mediated retrieval.

The store indexes each document as the questions it would answer ("what would a future reader ask whose answer is here?"). Your job is the mirror operation: given an incoming message, generate the questions whose answers in memory would be relevant.

For the message below, produce 1-5 distinct queries that, taken together, would surface the relevant prior context from memory. Each query should be:

- A complete, well-formed question (not a topic, not a keyword)
- Self-contained — readable without the surrounding message
- Phrased in vocabulary that documents would plausibly use, not just the message's own words
- Distinct from siblings — each query should reach for a different facet

## Input

{% if context -%}
Context: {{ context }}
{% endif -%}
Message: {{ message }}

## Output format

Return a valid JSON object with this exact shape:

```json
{"queries": [{"text": "..."}, {"text": "..."}]}
```

Rules:

- The top-level object MUST have a `queries` key whose value is an array.
- Each array element MUST be an object with a `text` field (string).
- Return ONLY the JSON object — no markdown fence, no preamble, no commentary, no trailing prose.
