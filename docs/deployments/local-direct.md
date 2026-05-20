# Local Postgres (direct, no docker-compose)

Step-by-step for running prospecta against a Postgres instance you already
manage on your machine — no Docker, no compose file. Good fit if you already
have Postgres running for other projects or want minimal moving parts.

If you want zero-setup, prefer the bundled `docker-compose.yml` at the repo
root. This guide is for the case where you want the database off-Docker.

---

## Prerequisites

- **Postgres 14+** with the `pgvector` extension available.
  - macOS: `brew install postgresql@16 pgvector`
  - Ubuntu/Debian: `apt install postgresql-16 postgresql-16-pgvector`
    (note: pgvector via apt on older Ubuntu can lag the upstream release —
    check `apt show postgresql-NN-pgvector` and fall back to building from
    source if you need a recent version)
  - From source: requires Postgres dev headers
    (`apt install postgresql-server-dev-16` or equivalent), then
    `git clone https://github.com/pgvector/pgvector && cd pgvector && make && sudo make install`
- **Python 3.11+** and `uv` (or `pip`) for installing prospecta.
- A user with `CREATE DATABASE` and `CREATE EXTENSION` privileges on the
  target Postgres instance (often the local superuser during dev).

---

## Provisioning

```bash
# 1. Create the database
createdb prospecta

# 2. Enable pgvector against that database
psql -d prospecta -c 'CREATE EXTENSION vector;'

# Verify the extension landed
psql -d prospecta -c '\dx vector'
```

If `CREATE EXTENSION vector` errors with "could not open extension control
file", the pgvector library is not installed where this Postgres can find it.
Re-check your install path (`pg_config --sharedir`) and re-run the install
step for pgvector.

---

## Configure prospecta

```bash
# Install prospecta with the embedder of your choice
pip install 'prospecta[embed-sentence-transformers]'

# Point at the local database
export DATABASE_URL='postgres://localhost:5432/prospecta'

# Bootstrap the schema (skip docker-compose detection)
prospecta init --no-substrate --embedding-dim 384
```

The `--embedding-dim 384` matches sentence-transformers `all-MiniLM-L6-v2`.
If you plan to use OpenAI `text-embedding-3-small`, pass `--embedding-dim 1536`.

---

## Smoke test

```bash
# Retain a memory
prospecta retain "Kelly was born March 4th 1990" \
  --source kelly.md \
  --index-text "When was Kelly born?"

# Recall it
prospecta search "Kelly birthday"
```

Expected: the recall returns one hit referencing the kelly.md source with
the original chunk text intact.

---

## Caveats

- **pgvector index build cost.** First HNSW index creation on a populated bank
  can take a few seconds to minutes depending on row count. This is fine in
  dev; production wants the index built once after initial load.
- **No connection pooling.** Direct Postgres + `psycopg` is fine for local
  use; introduce PgBouncer if you go multi-process.
- **No TLS by default.** Local connections over the unix socket or localhost
  are fine; if you expose Postgres on a network, add `?sslmode=require` to
  the `DATABASE_URL` and configure server certs.

---

## See also

- `docs/deployments/neon.md` — managed serverless Postgres
- `docs/deployments/azure.md` — Azure Database for PostgreSQL Flexible Server
- Repo root `docker-compose.yml` — zero-setup local alternative
