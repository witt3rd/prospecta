# Testing without Docker

The suite normally starts a `pgvector/pgvector:pg16` testcontainer. Where Docker
is unavailable, point it at any throwaway Postgres with pgvector instead:

```
uv venv -p 3.12 /tmp/pgv && uv pip install --python /tmp/pgv/bin/python pgserver
/tmp/pgv/bin/python -c "import pgserver; print(pgserver.get_server('/tmp/pgdata', cleanup_mode=None).get_uri())"
PROSPECTA_TEST_PG_URL='postgresql://postgres:@/postgres?host=/tmp/pgdata' \
    uv run --with pytest python -m pytest -q
```

`pgserver` ships Postgres and pgvector in a wheel (use Python 3.12; there is no
3.13 wheel). `PROSPECTA_TEST_PG_URL` makes `tests/conftest.py` skip the
container; each test still gets its own fresh database. Stop the server when
done (`get_server(..., cleanup_mode='stop').cleanup()`). Never point it at a
live bank's database.
