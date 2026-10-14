"""Shared-database mode: keep everything Prospecta owns inside one schema.

Prospecta never assumes it owns the whole database. When a schema name is
configured (``schema=`` / ``PROSPECTA_SCHEMA``) every connection gets
``search_path = "<schema>", public`` so all tables, indexes, functions and the
migration bookkeeping table are created in, and read from, that schema only.
Nothing is created elsewhere; the ``vector`` extension (and ``pg_search`` when
wanted) must already be installed by the database owner.

With no schema configured nothing changes: the standalone behaviour is kept.
"""
from __future__ import annotations

import os
import re
from urllib.parse import parse_qsl, quote, urlsplit, urlunsplit

import psycopg

SCHEMA_ENV = "PROSPECTA_SCHEMA"
_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


class SchemaError(RuntimeError):
    """The configured schema cannot be used (bad name, missing, not writable)."""


def validate_schema_name(name: str) -> str:
    """Lower-case SQL identifier, <= 63 bytes, not a system schema."""
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise SchemaError(
            f"invalid schema name {name!r}: use lower-case letters, digits, underscore")
    if name.startswith("pg_") or name in ("information_schema", "public"):
        raise SchemaError(f"schema name {name!r} is reserved; choose a dedicated schema")
    return name


def resolve_schema(schema: str | None = None) -> str | None:
    """Explicit argument, else $PROSPECTA_SCHEMA, else None (standalone)."""
    s = schema if schema is not None else os.environ.get(SCHEMA_ENV)
    s = (s or "").strip()
    return validate_schema_name(s) if s else None


def scoped_url(database_url: str, schema: str | None) -> str:
    """The URL with a libpq ``options`` setting that pins the search_path.

    A no-op when schema is None. Applying it twice is harmless.
    """
    if schema is None:
        return database_url
    validate_schema_name(schema)
    opts = f'-c search_path="{schema}",public -c prospecta.shared=on'
    if "://" in database_url:
        parts = urlsplit(database_url)
        q = [f"{quote(k, safe='')}={quote(v, safe='')}"
             for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != "options"]
        q.append(f"options={quote(opts, safe='')}")   # %20, never '+': libpq does not decode it
        return urlunsplit(parts._replace(query="&".join(q)))
    esc = opts.replace("\\", "\\\\").replace("'", "\\'")
    return f"{database_url} options='{esc}'"


def schema_exists(conn: psycopg.Connection, schema: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM pg_namespace WHERE nspname = %s", (schema,)).fetchone() is not None


def ensure_schema(database_url: str, schema: str) -> bool:
    """Create the schema if missing (needs CREATE on the database). Returns
    True when it was created. An existing schema is left alone."""
    validate_schema_name(schema)
    with psycopg.connect(database_url, autocommit=True) as conn:
        if schema_exists(conn, schema):
            return False
        try:
            conn.execute(f'CREATE SCHEMA "{schema}"')
        except psycopg.errors.InsufficientPrivilege as e:
            raise SchemaError(
                f"schema {schema!r} does not exist and this role may not create it; "
                f"ask the database owner to create it and grant USAGE, CREATE: {e}") from e
    return True


def check_schema_writable(database_url: str, schema: str, *, need_vector: bool = True) -> None:
    """Refuse (SchemaError) unless the role can create objects in `schema`.

    Probes with a real CREATE TABLE inside a transaction that is rolled back,
    so nothing remains. Also requires the vector extension to be present
    (it is never created here).
    """
    validate_schema_name(schema)
    try:
        conn = psycopg.connect(database_url, autocommit=False, connect_timeout=10)
    except psycopg.Error as e:
        raise SchemaError(f"cannot connect to the database: {e}") from e
    with conn:
        if not schema_exists(conn, schema):
            raise SchemaError(f"schema {schema!r} does not exist")
        usage, create = conn.execute(
            "SELECT has_schema_privilege(%s, 'USAGE'), has_schema_privilege(%s, 'CREATE')",
            (schema, schema)).fetchone()
        if not (usage and create):
            raise SchemaError(
                f"schema {schema!r} is not writable by this role (USAGE={usage}, CREATE={create})")
        try:
            conn.execute(f'CREATE TABLE "{schema}"._prospecta_write_probe (x int)')
            conn.execute(f'INSERT INTO "{schema}"._prospecta_write_probe VALUES (1)')
        except psycopg.Error as e:
            raise SchemaError(f"schema {schema!r} is not writable: {e}") from e
        finally:
            conn.rollback()
        if need_vector and conn.execute(
                "SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone() is None:
            raise SchemaError(
                "the vector extension is not installed in this database; in shared "
                "mode Prospecta does not create extensions (ask the database owner: "
                "CREATE EXTENSION vector)")
