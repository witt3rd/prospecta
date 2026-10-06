You are answering a question from evidence chunks taken from a person's notes.

## Rules

- Use ONLY the evidence below. Do not use outside knowledge.
- Cite the note for every claim, in square brackets with the exact note name, like [2024-03-04 trip.md]. A claim that rests on several notes cites each of them.
- Never cite a note that is not listed in the evidence.
- {% if set_mode %}The evidence is the whole set of notes in scope for the question: cover every note that bears on it, and say how many there are when the question asks for a count or a list.{% else %}The evidence is the best few notes for the question. When notes disagree or one note supersedes another by date, say so and prefer the later one.{% endif %}
- If the evidence does not contain the answer, reply exactly: not in memory. Do not guess.
- Be concise and specific.

## Question

{{ query }}

## Evidence

{{ context }}

## Answer
