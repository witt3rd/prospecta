You are generating question-form index entries for a document being filed in a semantic memory store.

The store uses bilateral LLM-mediated retrieval: when a future query arrives, an LLM will formulate the questions someone might ask to recall this content. Your job now is the inverse — anticipate the questions this document would answer.

For the content below, produce 1-5 questions a future reader might ask whose answer is in this document. Each question should be:

- A complete, well-formed question (not a topic, not a keyword)
- Specific enough that the document content meaningfully answers it
- Self-contained (no pronouns or context that requires the original source)
- Phrased in vocabulary a searcher would plausibly use, not just the document's own words

Format: one question per line, no numbering, no preamble, no commentary.

---

{{ content }}

---

Questions:
