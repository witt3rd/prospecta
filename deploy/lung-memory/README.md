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
The per-request `bank` and `POST /retain` are added by the service change in the sibling PR (`prosp-service-bank-retain`);
this section describes that shape.

- `GET /healthz`: liveness.
- `POST /retain` `{"bank": "...", "content": "...", "source": "...", "metadata": {...}?, "create": true?}`: store a verbatim note as chunks in `bank`; `create` creates the bank if missing.
- `POST /lookup` `{"task": {"title": "...", "ask": "...", "done": "..."?, "project": "..."?}, "bank": "...", "excluded_sources": [...]?}`: the task brief's recall; returns the cited block and results.

```
curl -sS -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"bank":"lung","content":"a note","source":"lung/session-1","create":true}' \
  http://$LUNG_MEMORY_BIND:8080/retain
```
