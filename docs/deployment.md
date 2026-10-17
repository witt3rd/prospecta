# Deployment

Prospecta needs one thing from its environment: a Postgres with the `pgvector` extension.
Everything else (the embedder, the models, the service) is configuration. This page covers
what every deployment shares, then the four ways to get the database: Docker, a local
Postgres, Neon and Azure Database for PostgreSQL.

## What every deployment needs

- **Postgres with pgvector 0.8 or newer.** Stage 1 (`hybrid_chunk`) sets `hnsw.iterative_scan`,
  which arrived in pgvector 0.8. The tests and the bundled `docker-compose.yml` run Postgres 16
  (`pgvector/pgvector:pg16`); other versions are expected to work and are not tested here.
  Check what the server has:

  ```sql
  SELECT extversion FROM pg_extension WHERE extname = 'vector';   -- 0.8.x or newer
  SELECT * FROM pg_available_extensions WHERE name = 'vector';    -- what could be installed
  ```

  A provider that ships an older pgvector can run legacy banks (`docs/legacy.md`) but not a
  bank on `hybrid_chunk`; `ALTER EXTENSION vector UPDATE;` after the provider offers 0.8.
- **Python 3.11 or newer**, and an embedder: `pip install 'prospecta[defaults]'` (LiteLLM, any
  provider), `'prospecta[embedder]'` (local sentence-transformers, no API key),
  `'prospecta[embed-openai]'` or `'prospecta[embed-openai-compatible]'`.
- **`DATABASE_URL`**, and the role it names may `CREATE EXTENSION` (or the extension is already
  installed: in shared-database mode, `PROSPECTA_SCHEMA`, Prospecta never creates extensions,
  `docs/source-contract.md`).

### Choose the embedding width once

A bank's `embedding_dim` is fixed when it is created and the embedder must produce exactly
that width (`prospecta health` prints `PROSPECTA DOWN` if it does not).

| embedder | model | width |
|---|---|---|
| LiteLLM default (`prospecta[defaults]`) | `openai/text-embedding-3-large` cut to 1536 | 1536 |
| `prospecta[embed-openai]` | `text-embedding-3-small` | 1536 |
| `PROSPECTA_EMBEDDER=sentence-transformers` | `all-MiniLM-L6-v2` | 384 |

`prospecta init` defaults to 1536 (`--embedding-dim` or `PROSPECTA_EMBEDDING_DIM` changes it).

### Configure, migrate, create the bank

```bash
export DATABASE_URL='postgres://USER:PASSWORD@HOST:5432/prospecta'
prospecta init --no-substrate --embedding-dim 1536     # migrate + create the default bank
# or, step by step:
prospecta migrate
prospecta create-bank --id mybank --embedding-dim 1536
```

`prospecta init` without `--no-substrate` first brings up `docker-compose.yml` if it finds one
(below).
A bank created this way **runs the full architecture with no further configuration**: Stage 1
`hybrid_chunk`, Stage 2 `cross_encoder`, no linker, no entity extraction
(`prospecta/bank_defaults.py`; `docs/schema.md` for what is stored).

What the models need on top of the database:

| piece | needs | without it |
|---|---|---|
| write (`retain`, `index`) | the embedder only; no model call | n/a |
| Stage 2 cross-encoder | `OPENROUTER_API_KEY` (the CLI builds the rerank transport from it) or `Memory(rerank_transport=...)` | the Stage 1 order stands; the recall records why |
| Mode 1 writer and gate, Mode 2B `ask` | `Memory(llm=...)` / the CLI's LiteLLM LLM; the decision model needs `OPENROUTER_API_KEY` (`Memory(jev=...)`) | the writer falls back to one search with the prompt itself; no gate runs |

### Smoke test

```bash
prospecta retain "Kelly was born March 4th 1990" --source kelly.md
prospecta search "Kelly birthday"
```

Expected: one hit, `source` `kelly.md`, the note's chunk text unchanged. (A note retained
with `--index-text` is a question item, which a new bank does not read.)

```sql
SELECT bank_id, channel_config, recall_config, link_config FROM banks;
```

shows the stored config; `channel_config` names `hybrid_chunk`.

### Caveats common to every substrate

- **Per-bank HNSW indexes.** `create_bank` builds three partial indexes
  (`docs/schema.md`). On a populated table the first build takes seconds to minutes; build the
  bank once, load, then query. For a bulk load scale the database up first.
- **Migration `0020` rewrites `memory_items`.** Read `docs/exact-tsv.md` before upgrading a
  database that already holds a bank.
- **Connections.** One process shares one pool: `PROSPECTA_DB_POOL_MAX` (default 10),
  `PROSPECTA_DB_POOL_IDLE_SECONDS`, `PROSPECTA_REQUEST_TIMEOUT` (`docs/source-contract.md`).
  Put PgBouncer in front only for many processes.
- **TLS and secrets.** `DATABASE_URL` holds the password: treat it as a secret. Add
  `?sslmode=require` for any network connection.
- **Backups.** `prospecta backup` / `restore` (`docs/backup-restore.md`), and your provider's.

## Docker (the default)

The repository's `docker-compose.yml` runs `pgvector/pgvector:pg16` with database, user
`prospecta` and the password from `POSTGRES_PASSWORD` (`.env.example`), on port 5432, with a
named volume.

```bash
cp .env.example .env            # set POSTGRES_PASSWORD and the matching DATABASE_URL
export DATABASE_URL=postgresql://prospecta:<POSTGRES_PASSWORD>@localhost:5432/prospecta
prospecta init                  # compose up, wait until it runs, migrate, create the default bank
```

`prospecta init` looks for `docker-compose.yml` in `$PROSPECTA_HOME`, else the current
directory, and still needs `DATABASE_URL` to connect (it does not guess it). Without Docker or
with `--no-substrate` it uses the database `DATABASE_URL` names. It is idempotent.

## Local Postgres (no Docker)

1. Install Postgres and pgvector: macOS `brew install postgresql@16 pgvector`; Debian / Ubuntu
   `apt install postgresql-16 postgresql-16-pgvector` (an older distribution may ship an old
   pgvector: check `apt show postgresql-16-pgvector`, or build from source with the server dev
   headers: `git clone https://github.com/pgvector/pgvector && cd pgvector && make && sudo make install`).
2. Create the database and the extension, as a role that may:

   ```bash
   createdb prospecta
   psql -d prospecta -c 'CREATE EXTENSION vector;' -c '\dx vector'
   ```

   "could not open extension control file" means pgvector is not installed where this server
   looks: check `pg_config --sharedir`.
3. `export DATABASE_URL='postgres://localhost:5432/prospecta'`, then
   `prospecta init --no-substrate --embedding-dim 384` (the sentence-transformers width).

A unix socket or localhost needs no TLS; if you expose the server, add `sslmode=require` and
server certificates.

## Neon

1. Create a project in the Neon console, in the region nearest the application (it cannot be
   changed without a migration).
2. Copy the connection string; keep its `?sslmode=require`:
   `postgres://USER:PASSWORD@ep-<id>.<region>.aws.neon.tech/neondb?sslmode=require`.
3. Enable pgvector: console, Project, Extensions, `vector`; or in the SQL editor
   `CREATE EXTENSION vector;`. Check `extversion` as above.
4. `export DATABASE_URL=...` and `prospecta init --no-substrate --embedding-dim 1536`.
   Prospecta creates its tables inside the database Neon provisioned (`neondb`); create another
   database in the console first if you want a separate one.

Neon suspends an idle project: the first query after a pause waits 5 to 15 seconds for the
compute (paid tiers reduce this). Index builds on a small compute are slow: scale up before a
bulk load. Network distance to the region is paid on every query: co-locate.

## Azure Database for PostgreSQL, Flexible Server

```bash
az login && az account set --subscription <SUBSCRIPTION_ID>

az postgres flexible-server create --name prospecta-pg --resource-group <RG> --location <REGION> \
  --tier Burstable --sku-name Standard_B1ms --version 16 --storage-size 32 \
  --admin-user prospecta --admin-password '<STRONG_PASSWORD>' --public-access 0.0.0.0
```

`--public-access 0.0.0.0` opens the firewall to the internet: for anything real restrict it
(`az postgres flexible-server firewall-rule create --resource-group <RG> --name prospecta-pg
--rule-name allow-app --start-ip-address <IP> --end-ip-address <IP>`) or use VNet
integration. Creation takes 5 to 10 minutes.

Allow the extension (the parameter change needs a restart), then create the database and the
extension:

```bash
az postgres flexible-server parameter set --resource-group <RG> --server-name prospecta-pg \
  --name azure.extensions --value VECTOR
az postgres flexible-server restart --resource-group <RG> --name prospecta-pg

psql "host=prospecta-pg.postgres.database.azure.com port=5432 dbname=postgres user=prospecta sslmode=require" \
  -c 'CREATE DATABASE prospecta'
psql "host=prospecta-pg.postgres.database.azure.com port=5432 dbname=prospecta user=prospecta sslmode=require" \
  -c 'CREATE EXTENSION vector'
```

Then `export DATABASE_URL='postgres://prospecta:<STRONG_PASSWORD>@prospecta-pg.postgres.database.azure.com:5432/prospecta?sslmode=require'`
and `prospecta init --no-substrate --embedding-dim 1536`.

Check the pgvector version your region and server version offer (`pg_available_extensions`,
above) before committing: it varies, and `hybrid_chunk` needs 0.8. TLS is enforced, so
`sslmode=require` is mandatory. Burstable `Standard_B1ms` is for development; a large bank's
HNSW build wants a memory-optimised SKU for the build, which you can scale down afterwards.
Backups are automatic; high availability is chosen at creation or with
`az postgres flexible-server update`.

None of the provisioning commands above are exercised by the test suite; they are the
provider's documented steps as last written.
