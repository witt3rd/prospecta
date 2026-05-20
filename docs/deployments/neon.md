# Neon (serverless Postgres)

Walk-through for running prospecta against [Neon](https://neon.tech), a
serverless Postgres provider with a generous free tier and one-click
pgvector. Good fit for hosted dev/staging without provisioning anything
yourself.

---

## Prerequisites

- A Neon account (sign up at https://neon.tech — free tier exists).
- Python 3.11+ and `uv` (or `pip`).
- An embedding provider (sentence-transformers for local, or OpenAI for hosted).

---

## Provisioning

1. **Create a Neon project.** In the Neon console, create a new project.
   Pick the region nearest your application; this region cannot be changed
   later without a project migration.

2. **Copy the connection string.** Neon shows it in the project dashboard.
   The string already includes `?sslmode=require` — keep that suffix.

   ```
   postgres://USER:PASSWORD@ep-<id>.us-east-2.aws.neon.tech/neondb?sslmode=require
   ```

3. **Enable pgvector.** Two paths:
   - **Console:** Project → Extensions → enable `vector` (one click).
   - **SQL editor:** open the SQL editor and run `CREATE EXTENSION vector;`.

   Verify with `SELECT * FROM pg_extension WHERE extname = 'vector';`.

---

## Configure prospecta

```bash
pip install 'prospecta[embed-sentence-transformers]'

export DATABASE_URL='postgres://USER:PASSWORD@ep-<id>.us-east-2.aws.neon.tech/neondb?sslmode=require'

prospecta init --no-substrate --embedding-dim 384
```

Note: prospecta creates the schema inside the database Neon already
provisioned (`neondb` by default). If you want a separate database, create
it from the Neon console first and use that name in the URL.

---

## Smoke test

```bash
prospecta retain "Kelly was born March 4th 1990" \
  --source kelly.md \
  --index-text "When was Kelly born?"

prospecta search "Kelly birthday"
```

If the first call hangs for 5–15s before returning, that's Neon's compute
cold-starting (see caveats).

---

## Caveats

- **Cold-start delay (free tier).** Neon suspends idle projects to save
  compute. The first query after suspension wakes the compute, which adds
  5–15 seconds of latency. Subsequent queries are normal-speed. Paid tiers
  reduce or eliminate this.
- **HNSW index build on first populated bank.** Initial pgvector index
  creation against a non-empty table can be slow on Neon's smaller compute
  sizes. Scale compute up before bulk-loading, or let the index build
  asynchronously.
- **Connection string secrets.** The URL contains the password — treat it
  as a secret. Use Neon's role/permission system if you need separate dev
  and prod credentials.
- **Region matters for latency.** If your application is far from the Neon
  region, every query pays round-trip cost. Co-locate where possible.

---

## See also

- `docs/deployments/local-direct.md` — local Postgres without Docker
- `docs/deployments/azure.md` — Azure Database for PostgreSQL Flexible Server
