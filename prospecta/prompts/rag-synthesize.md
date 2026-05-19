You are synthesizing recalled chunks from a semantic memory store into a focused response to a query.

The chunks below were retrieved as the most relevant material for the query. Use them to compose a concise, grounded answer.

## Rules

- Answer the query using ONLY the provided chunks.
- Ground every claim in the retrieved material; do not fabricate.
- If the chunks are insufficient to answer the query, say so plainly rather than guessing.
- When chunks complement each other, synthesize across them rather than quoting each in isolation.
- When chunks contradict each other, name the disagreement rather than silently picking one.
- Be concise. Prefer specificity over hedged generality.

## Query

{{ query }}

## Retrieved chunks

{{ context }}

## Response
