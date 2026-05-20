# Azure Database for PostgreSQL — Flexible Server

Walk-through for running prospecta against Azure Database for PostgreSQL
Flexible Server with pgvector. Good fit for Azure-resident deployments
where you want a managed Postgres co-located with your compute.

---

## Prerequisites

- Azure subscription with permission to create database resources.
- `az` CLI installed and logged in:

  ```bash
  az login
  az account set --subscription <SUBSCRIPTION_ID>
  ```

- A resource group (`<RG>`) and a region (`<REGION>`) where pgvector is
  available — most public regions are supported; verify with the parameter
  listing step below.
- Python 3.11+ and `uv` (or `pip`).

---

## Provisioning

### 1. Create the Flexible Server

```bash
az postgres flexible-server create \
  --name prospecta-pg \
  --resource-group <RG> \
  --location <REGION> \
  --tier Burstable \
  --sku-name Standard_B1ms \
  --version 16 \
  --storage-size 32 \
  --admin-user prospecta \
  --admin-password '<STRONG_PASSWORD>' \
  --public-access 0.0.0.0
```

`--public-access 0.0.0.0` opens the server to the public internet with the
firewall rule "AllowAll". For real deployments, restrict this to your
application's egress IPs or use a VNet integration instead.

Provisioning takes 5–10 minutes. The command returns the fully qualified
DNS name (e.g. `prospecta-pg.postgres.database.azure.com`).

### 2. Allow the `vector` extension

Flexible Server gates extensions per server. Enable `vector` on the
allow-list:

```bash
az postgres flexible-server parameter set \
  --resource-group <RG> \
  --server-name prospecta-pg \
  --name azure.extensions \
  --value VECTOR
```

This parameter change requires a restart:

```bash
az postgres flexible-server restart \
  --resource-group <RG> \
  --name prospecta-pg
```

### 3. Create the database and the extension

```bash
# Connect (psql will prompt for the admin password)
psql "host=prospecta-pg.postgres.database.azure.com \
      port=5432 \
      dbname=postgres \
      user=prospecta \
      sslmode=require"

# Inside psql:
CREATE DATABASE prospecta;
\c prospecta
CREATE EXTENSION vector;
\q
```

Verify pgvector is available in this region/version before provisioning by
listing supported extensions:

```bash
az postgres flexible-server list-skus --location <REGION> --query "[].name"
# and check extension availability via:
az postgres flexible-server parameter show \
  --resource-group <RG> \
  --server-name prospecta-pg \
  --name azure.extensions
```

---

## Configure prospecta

```bash
pip install 'prospecta[embed-openai]'

export DATABASE_URL='postgres://prospecta:<STRONG_PASSWORD>@prospecta-pg.postgres.database.azure.com:5432/prospecta?sslmode=require'

prospecta init --no-substrate --embedding-dim 1536
```

The `--embedding-dim 1536` matches OpenAI `text-embedding-3-small`. Adjust
to match your chosen embedder.

---

## Smoke test

```bash
prospecta retain "Kelly was born March 4th 1990" \
  --source kelly.md \
  --index-text "When was Kelly born?"

prospecta search "Kelly birthday"
```

---

## Caveats

- **Provisioning latency.** Flexible Server creation takes 5–10 minutes;
  parameter changes that need restart add another 1–2 minutes.
- **pgvector availability varies by region.** Most public regions support
  it on Postgres 14+; verify before committing to a region. The
  `parameter show` command above will list `VECTOR` if available.
- **Burstable tier is dev-only.** `Standard_B1ms` (Burstable) is fine for
  v0.1 dev work but production HNSW index builds on large banks want
  Memory-Optimized tier (`Standard_E*` SKUs) for the build phase, then
  can scale back down. Watch index build IO.
- **Firewall rules.** `--public-access 0.0.0.0` opens the world. For real
  deployments use a VNet integration or restrict to known egress IPs:

  ```bash
  az postgres flexible-server firewall-rule create \
    --resource-group <RG> --name prospecta-pg \
    --rule-name allow-app --start-ip-address <IP> --end-ip-address <IP>
  ```

- **SSL is required.** Flexible Server enforces TLS; the `?sslmode=require`
  in `DATABASE_URL` is mandatory.
- **Backups and HA.** Flexible Server has automated backups and optional
  HA (zone-redundant or same-zone). Configure at creation time or via
  `az postgres flexible-server update`.

---

## See also

- `docs/deployments/local-direct.md` — local Postgres without Docker
- `docs/deployments/neon.md` — managed serverless alternative
