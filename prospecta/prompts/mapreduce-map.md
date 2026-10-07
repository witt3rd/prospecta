You are extracting facts from a batch of notes, to answer a question that asks for every item of a set.

## Rules

- Use ONLY the notes below. Do not use outside knowledge.
- Extract every fact in these notes that bears on the question. Do not summarise and do not skip a fact because it looks minor or repeated elsewhere.
- When the question asks for names of a person (nicknames, pen names, aliases, handles, stage names, "also known as"), extract EVERY name the person goes by in the notes, of any kind: nicknames, pen names, pseudonyms, handles, titles used as a name, roles used as a name. Do not limit yourself to the word the question happened to use, and do not judge which kind a name is: list each one.
- One entry per fact. Each entry names the note it comes from, using the exact note name shown in the header.
- A note that holds nothing relevant contributes no entry.
- If no note holds anything relevant, return an empty list.

Reply with JSON only, in this shape:
{"facts": [{"fact": "<the fact, in a short phrase>", "note": "<exact note name>"}]}

## Question

{{ query }}

## Notes

{{ context }}
