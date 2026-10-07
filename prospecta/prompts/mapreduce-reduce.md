You are merging facts that were extracted from many notes, to answer a question that asks for every item of a set.

## Rules

- Merge entries that state the same fact (same name, same thing said in different words) into one item.
- Keep distinct facts distinct. Never drop a fact that is not a duplicate, and never invent one.
- Each item keeps the union of the notes of the entries merged into it. Use the exact note names given.

Reply with JSON only, in this shape:
{"items": [{"fact": "<the fact>", "notes": ["<exact note name>"]}]}

## Question

{{ query }}

## Extracted facts

{{ facts }}
