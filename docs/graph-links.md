# Links, entities and the graph channel (migration 0006)

The Jev-Mem techniques, Postgres-native (hybrid retrieval design 8.7).

**Tables** (additive; `memory_entities`, `memory_item_entities`, `memory_links`,
`memory_link_state`; links and entity rows cascade on item delete).

**Linker** (`prospecta._linker.Linker`, `JevRelationJudge`; `Memory(linker=Linker(llm=..., judge=...))`) runs after
`retain` / `index_single_file` on one worker thread (`asynchronous=False` runs inline;
`Memory.wait_for_links()` drains; `Memory.link_document(id)` and `Memory.link_pending()`
are the on-demand and safety-net paths). Per document:

- `NEXT` between consecutive chunks, and `PRECEDES` / `SUCCEEDS` / `TEMPORALLY_CLOSE`
  (within 3 days, at most 5) to the chronological neighbours of the same `person`, in SQL
  from `prospecta_doc_date(document_metadata, created_at, created_on)` (created_on first); no model call.
- Entities by the injected `llm` (Sonnet; prompt `extract-entities`), then
  `ENTITY/SHARED_ENTITY` links by a join, capped at 10 items per entity.
- Semantic links: the top 10 pgvector neighbours per anchor item. With a
  `JevRelationJudge(transport)` Jev answers `semantic`, `causes`, `caused_by` as System
  One `score` questions (score/3 >= 0.6 makes a link, `origin='jev'`); without one, or if
  Jev fails, neighbours with cosine >= 0.75 become `RELATED_TO` (`origin='pgvector'`).

**GraphExpand** (`graph`, kind `expand`) seeds from the best 10 documents of the pool,
walks one hop by join and a second by `WITH RECURSIVE` in both directions, multiplying by
decay 0.5, the link type weight (SEMANTIC 1.0, CAUSAL 0.8, TEMPORAL 0.5, ENTITY 0.5) and
the link confidence; at most 60 neighbour items, reported per document. It is **enabled
in `DEFAULT_CHANNEL_CONFIG` at weight 1** (captain's decision; the design had it dark);
set `weight` (0 silences it) and `params` (`seeds`, `max_hops`, `decay`, `node_cap`,
`type_weights`) per bank. Its contribution is to be measured on real questions.

**JevReader** (`recall_config.reader.type = "jev"`, needs `Memory(jev=...)`): Jev's
`evidence_sufficient`, `continue_useful`, `missing_evidence` as one request decide
sufficiency; the follow-up probe is the best excerpt itself.
