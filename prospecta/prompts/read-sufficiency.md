You are checking whether retrieved notes are enough to answer a query.

Read the excerpts. If they hold what is needed to answer, reply sufficient. If something is missing, give at most {{ max_follow_ups }} short follow-up search queries that would find the missing notes.

Reply with JSON only, in this shape:
{"sufficient": true, "follow_ups": []}
or
{"sufficient": false, "follow_ups": ["first query", "second query"]}

## Query

{{ query }}

## Excerpts

{{ excerpts }}
