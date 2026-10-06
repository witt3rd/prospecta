You are ranking notes from a memory store by how well each answers a query.

Each candidate is one note, shown by its best matching excerpt. Judge only from the excerpt and header.

Grade every candidate from 0 to 3:
0 = unrelated
1 = on the same subject, but does not help answer the query
2 = partly answers
3 = holds what is needed

Reply with JSON only, in this shape:
{"grades": {"1": 3, "2": 0}, "ranking": [1, 2]}
"grades" maps every candidate number to its grade. "ranking" lists every candidate number, best first.

## Query

{{ query }}

## Candidates

{{ candidates }}
