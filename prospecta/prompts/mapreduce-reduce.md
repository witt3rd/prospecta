You are merging facts that were extracted from many notes, to answer a question that asks for every item of a set.

## Rules

- Facts are numbered. Group the numbers of entries that state the very same fact (the same name or thing said in different words) into one item.
- Two different names, nicknames, pen names or aliases are DIFFERENT facts even when they belong to the same person: never merge them. When unsure, keep them apart.
- Every number must appear in exactly one item, including facts that stand alone (a fact in one note only). Never invent a fact.
- The item's "fact" repeats the fact in its clearest wording.

Reply with JSON only, in this shape:
{"items": [{"fact": "<the fact>", "ids": [1, 4]}]}

## Question

{{ query }}

## Extracted facts

{{ facts }}
