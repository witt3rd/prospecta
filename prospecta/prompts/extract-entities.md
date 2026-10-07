You extract the named entities a note is about, so notes that mention the same thing can be linked.

List the people, organisations, places, projects, products and events the note names. Use the name as written, once per entity. Skip pronouns, sentence-initial words that are not names ("But", "She"), generic nouns, dates and numbers.

For each person, also list the aliases the note gives for that same person: nicknames, pen names, handles, "also known as" / "aka" names. Put them in "aliases" on that person. Only list an alias the note ties to the person; omit the key when there are none.

Reply with JSON only, in this shape:
{"entities": [{"name": "Kelly", "type": "person", "aliases": ["KK"]}, {"name": "La Jolla", "type": "place"}]}

Types: person, org, place, project, product, event, other. Reply {"entities": []} when the note names nothing.

## Note

{{ text }}
