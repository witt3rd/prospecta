You are checking whether retrieved notes are enough to answer a query.

Read the excerpts. If they hold what is needed to answer, reply sufficient. If something is missing, give one short follow-up search query for each distinct missing fact, to find the missing notes.

Reply with JSON only, in this shape:
{"sufficient": true, "follow_ups": []}
or
{"sufficient": false, "follow_ups": ["first query", "second query"]}

## Query

{{ query }}

## Excerpts

{{ excerpts }}
