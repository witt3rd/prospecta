# lung-memory: Prospecta as a network memory provider

The lung agent uses Prospecta over HTTP. The service runs **on roger** as a systemd user unit, against the **existing shared
Prospecta Postgres** (container `prospecta-roger-postgres`, `127.0.0.1:54321`). There is no second Postgres and no compose
stack. Banks are `lung-class-<profile>` and `lung-domain-<project>`, created on first retain.

| piece | how |
|---|---|
| unit | `prospecta-lung-memory.service` (template; `python -m prospecta.service` from the pinned venv) |
| env | `~/.config/prospecta-lung-memory/env`, written by fleet-ops from Doppler; names in `prospecta-lung-memory.env.example` |
| exposure | bound to `LUNG_MEMORY_BIND` (the roger tailnet address). No default: the unit refuses to start if it is unset |
| database | `DATABASE_URL` points at the existing shared DB (shared-database mode, own schema as that DB is configured) |

The unit is declared in fleet-ops later; nothing here installs or starts it.

**Light on roger:** embeddings, rerank and the query writer are OpenRouter calls; the service itself holds only a small
DB connection pool (`PROSPECTA_DB_POOL_MAX`, default 10).

**Later move to the cloud (Supabase):** change `DATABASE_URL` and `LUNG_MEMORY_BIND`; nothing else.

## Secrets and variables

Secrets come from **Doppler**; this repo holds names only. fleet-ops renders them to the env file; the unit does not call `doppler run`.

- `LUNG_MEMORY_BIND`: the tailnet IP to listen on. Required.
- `DATABASE_URL`: the shared Prospecta Postgres.
- `PROSPECTA_SERVICE_TOKEN`: the bearer token callers send; required (the service refuses a non-loopback bind without it).
- `PROSPECTA_SERVICE_PORT` (default `8080`).
- `OPENROUTER_API_KEY`: the key the models are called with (see ZDR below).
- `PROSPECTA_ZDR_ATTESTED`: set to exactly `1` on the `work` profile (the operator asserts the key above is zero-data-retention); lung's wire then says `"attested"`. Unset or anything else: `"unattested"`. It enforces nothing.
- `PROSPECTA_QUERY_WRITER_MODEL`, `PROSPECTA_EMBED_MODEL`, `PROSPECTA_EMBEDDING_DIM`, `PROSPECTA_RERANKER_MODEL`, `PROSPECTA_DECISION_MODEL`: models (see profiles).

## ZDR: a property of the key, not of Prospecta

Zero data retention is a property of the **OpenRouter key / workspace** you choose for a deployment. **Prospecta does not
enforce it**: it sends whatever model it is configured with to whatever key it is given. Choosing a ZDR-guardrailed key and
only ZDR-eligible models is the deployer's job. Two profiles:

### `work`: ZDR-guardrailed OpenRouter key

Use a key from a workspace whose guardrail enforces ZDR, and only ZDR-eligible models:

```
PROSPECTA_EMBED_MODEL=openai/text-embedding-3-large     # 1536 dims
PROSPECTA_EMBEDDING_DIM=1536
PROSPECTA_RERANKER_MODEL=qwen/qwen3-reranker-8b
PROSPECTA_DECISION_MODEL=typesafe/jev-1.14
PROSPECTA_QUERY_WRITER_MODEL=deepseek/deepseek-v4.1-flash-1017
```

(Set these in the Doppler config for the work deployment.)

Set `PROSPECTA_ZDR_ATTESTED=1` only for this profile and only when that is true: lung treats the word as the provider's claim.

### `personal`: non-ZDR key

A non-ZDR OpenRouter key; any model may be used. Set the same variables to whatever models you want. Do not point
work data at this profile.

The embedding dimension is fixed per bank when it is created; do not change `PROSPECTA_EMBED_MODEL`/`PROSPECTA_EMBEDDING_DIM` on a
bank that has data.

## Check (after fleet-ops declares and starts it)

```
curl http://$LUNG_MEMORY_BIND:8080/healthz          # needs no token
systemctl --user status prospecta-lung-memory
```

Verify the template without starting it: `systemd-analyze --user verify prospecta-lung-memory.service`.
Stopping the unit keeps all data (it lives in the shared Postgres); removing the lung banks is a deliberate bank deletion, not a deploy step.

## API

All requests except `/healthz` carry `Authorization: Bearer <PROSPECTA_SERVICE_TOKEN>`. JSON in, JSON out.
Lung calls one endpoint, `POST /lung-memory/1`; `/retain` and `/lookup` are the service's own shape for other clients.

- `GET /healthz`: liveness.
- `POST /retain` `{"bank": "...", "content": "...", "source": "...", "metadata": {...}?, "create": true?}`: store a verbatim note as chunks in `bank`; `create` creates the bank if missing.
- `POST /lookup` `{"task": {"title": "...", "ask": "...", "done": "..."?, "project": "..."?}, "bank": "...", "excluded_sources": [...]?}`: the task brief's recall; returns the cited block and results.

```
curl -sS -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"bank":"lung","content":"a note","source":"lung/session-1","create":true}' \
  http://$LUNG_MEMORY_BIND:8080/retain
```

### `POST /lung-memory/1`: what lung calls

The wire `lung-memory/1` (lung's `spec/LungSpec/MemoryWire.lean`, PART B). Same header as above:
`Authorization: Bearer <PROSPECTA_SERVICE_TOKEN>`. HTTP 200 for every well-formed request, a failure included; 400 only for a body
that is none of the three ops. Every reply carries `"retention": "attested"` (only with `PROSPECTA_ZDR_ATTESTED=1`) or `"unattested"`.
Full reference: `docs/source-contract.md`, "The `lung-memory/1` adapter".

```
# retain (source is the idempotency key; the bank is created on first retain)
curl -sS -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"op":"retain","bank":"lung-class-work","text":"Deploys go out on Tuesdays.","source":"lung/session-7"}' \
  http://$LUNG_MEMORY_BIND:8080/lung-memory/1
{"status":"retained","id":"6e664361-a6f0-40ad-b032-681928cfc6f4","retention":"unattested"}

# search (questions recall) over named banks
  -d '{"op":"search","query":"when do deploys go out","banks":["lung-class-work"]}'
{"status":"ok","records":[{"bank":"lung-class-work","id":"6e664361-a6f0-40ad-b032-681928cfc6f4","text":"Deploys go out on Tuesdays.","source":"lung/session-7","date":"2026-11-23"}],"retention":"unattested"}

# recall (the task brief, Mode 1) over named banks; nothing found
  -d '{"op":"recall","task":"plan this week'"'"'s release","banks":["lung-class-work","lung-domain-shop"]}'
{"status":"ok","records":[],"retention":"unattested"}

# a failure
{"status":"unavailable","reason":"provider","retention":"unattested"}
```

A record is always `{bank, id, text, source, date}` with `bank` one of the request's banks; a hit with no source or date is left out, not defaulted
(only a document written outside `/retain` can lack a source).
