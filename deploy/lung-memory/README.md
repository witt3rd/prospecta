# lung-memory: Prospecta as a network memory provider

The lung agent uses Prospecta over HTTP. The service runs on the fleet host **mcfeely** (docker), with its own
Postgres (`pgvector/pgvector:pg16`, pgvector 0.8.x), bound to the tailnet address only. Nothing here runs on roger.

| piece | how |
|---|---|
| service | built from this repo's `Dockerfile` (`python -m prospecta.service`) |
| Postgres | `pgvector/pgvector:pg16`; **no published port**; data in the named volume `pgdata` |
| exposure | the service port is published only on `LUNG_MEMORY_BIND` (the host's tailnet IP). There is no default; compose refuses to start if it is unset |
| labels | every container, volume, network and the built image carry `org.prospecta.project=lung-memory` (override the value with `LUNG_MEMORY_LABEL`, together with `COMPOSE_PROJECT_NAME`, for a scratch stack) |

## Secrets and variables

Secrets come from **Doppler**; this repo holds names only (`lung-memory.env.example`). Variables:

- `POSTGRES_PASSWORD`: the database password (compose builds `DATABASE_URL` from it; the service inside the stack talks to the stack's own Postgres).
- `DATABASE_URL`: set by compose; set it yourself only if you run the service image against another database.
- `OPENROUTER_API_KEY`: the key the models are called with (see ZDR below). Without it the service starts, but recall has no reranker, decision model or query writer, and embedding fails.
- `PROSPECTA_SERVICE_TOKEN`: the bearer token callers send; required.
- `PROSPECTA_SERVICE_HOST` (default `0.0.0.0` inside the container) and `PROSPECTA_SERVICE_PORT` (default `8080`).
- `PROSPECTA_QUERY_WRITER_MODEL`: the query-writer model (see profiles).
- `LUNG_MEMORY_BIND`: the tailnet IP to publish on (`tailscale ip -4`). Required.
- Optional, defaulted in compose: `PROSPECTA_EMBED_MODEL`, `PROSPECTA_EMBEDDING_DIM`, `PROSPECTA_RERANKER_MODEL`, `PROSPECTA_DECISION_MODEL`.

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

(Set these in the Doppler config for the work deployment; the first four are also the compose defaults.)

### `personal`: non-ZDR key

A non-ZDR OpenRouter key; any model may be used. Set the same variables to whatever models you want. Do not point
work data at this profile.

The embedding dimension is fixed per bank when it is created; do not change `PROSPECTA_EMBED_MODEL`/`PROSPECTA_EMBEDDING_DIM` on a
bank that has data.

## Run it (on mcfeely)

```
ssh mcfeely
cd <checkout of this repo>/deploy/lung-memory
export LUNG_MEMORY_BIND=$(tailscale ip -4 | head -1)
doppler run -- docker compose up -d --build        # `up -d` after the first build
```

Health (needs no token):

```
curl http://$LUNG_MEMORY_BIND:8080/healthz
docker compose exec postgres psql -U prospecta -d prospecta -c "select extversion from pg_extension where extname='vector'"   # >= 0.8
```

Tear down, including the data volume (destructive: the memory is gone):

```
doppler run -- docker compose down -v
docker ps -a --filter label=org.prospecta.project=lung-memory
docker volume ls --filter label=org.prospecta.project=lung-memory
docker network ls --filter label=org.prospecta.project=lung-memory     # all three must list nothing
```

Use `docker compose down` (no `-v`) to stop but keep the data.

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
