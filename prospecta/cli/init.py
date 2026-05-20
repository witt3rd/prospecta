"""`prospecta init` — one-command bootstrap (T22, P16).

Composes existing primitives:
  1. Substrate up (docker compose) — optional, gated by detection
  2. Migrations (prospecta.db.migrate.run_migrations)
  3. Default bank (Memory.create_bank)

Exit codes:
  0  success
  1  missing docker AND DATABASE_URL (and not --no-substrate)
  2  docker compose substrate-up healthcheck timeout
  3  migration failure
  4  default bank creation failure (non-already-exists)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psycopg

DEFAULT_DIM = 1536  # OpenAI text-embedding-3-small shape
DEFAULT_DIM_NOTE = (
    "(defaulting to OpenAI text-embedding-3-small shape; "
    "set --embedding-dim or PROSPECTA_EMBEDDING_DIM to override)"
)


def _find_compose_file() -> Path | None:
    """Return path to docker-compose.yml in PROSPECTA_HOME or CWD, else None."""
    home = os.environ.get("PROSPECTA_HOME")
    if home:
        candidate = Path(home) / "docker-compose.yml"
        if candidate.is_file():
            return candidate
    cwd_candidate = Path.cwd() / "docker-compose.yml"
    if cwd_candidate.is_file():
        return cwd_candidate
    return None


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _compose_already_up(compose_file: Path) -> bool:
    """Check if `docker compose ps` shows running services."""
    try:
        result = subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "ps", "--status=running", "-q"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _compose_up(compose_file: Path, timeout: float) -> tuple[bool, str]:
    """Run docker compose up -d, wait for healthcheck up to `timeout` seconds.

    Returns (ok, message). On timeout, message contains last 20 lines of logs.
    """
    try:
        subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "up", "-d"],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as e:
        return False, f"docker compose up failed: {e.stderr}"
    except subprocess.TimeoutExpired:
        return False, "docker compose up timed out"

    # Poll for running state up to remaining time. Simple: wait for ps to show
    # services running. We don't have a way to introspect healthcheck status
    # generically here; rely on `compose up -d` returning OK as readiness signal.
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _compose_already_up(compose_file):
            return True, "ok (brought up)"
        time.sleep(1.0)

    # Timed out — fetch last 20 lines of logs.
    try:
        logs = subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "logs", "--tail=20"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        log_tail = logs.stdout + logs.stderr
    except Exception:
        log_tail = "(could not fetch logs)"
    return False, f"substrate up healthcheck timeout\n{log_tail}"


def _verify_connectivity(database_url: str) -> bool:
    try:
        with psycopg.connect(database_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception:
        return False


def _bank_exists(database_url: str, bank_id: str) -> bool:
    try:
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM information_schema.tables WHERE table_name = 'banks'"
                )
                if cur.fetchone() is None:
                    return False
                cur.execute("SELECT 1 FROM banks WHERE bank_id = %s", (bank_id,))
                return cur.fetchone() is not None
    except Exception:
        return False


def _resolve_embedding_dim(args) -> tuple[int, bool]:
    """Resolve embedding dim. Returns (dim, used_default)."""
    if getattr(args, "embedding_dim", None) is not None:
        return int(args.embedding_dim), False
    env_val = os.environ.get("PROSPECTA_EMBEDDING_DIM")
    if env_val:
        try:
            return int(env_val), False
        except ValueError:
            pass
    return DEFAULT_DIM, True


def cmd_init(args) -> int:
    # ------------------------------------------------------------------
    # 1. Substrate
    # ------------------------------------------------------------------
    no_substrate = getattr(args, "no_substrate", False)
    substrate_timeout = float(getattr(args, "substrate_timeout", 60.0))
    database_url = args.database_url or os.environ.get("DATABASE_URL")

    if no_substrate:
        if not database_url:
            print(
                "error: --no-substrate requires DATABASE_URL or --database-url",
                file=sys.stderr,
            )
            return 1
        if not _verify_connectivity(database_url):
            print(
                f"error: cannot connect to DATABASE_URL (--no-substrate)",
                file=sys.stderr,
            )
            return 1
        print("substrate: ok (external, --no-substrate)")
    else:
        compose_file = _find_compose_file()
        have_docker = _docker_available()

        if compose_file and have_docker:
            if _compose_already_up(compose_file):
                print("substrate: ok (already up)")
            else:
                ok, msg = _compose_up(compose_file, substrate_timeout)
                if not ok:
                    print(f"substrate: {msg}", file=sys.stderr)
                    return 2
                print(f"substrate: {msg}")
            # After compose up, prefer container's URL only if user set DATABASE_URL
            # otherwise we can't connect (no conventional default here).
            if not database_url:
                print(
                    "error: substrate up but no DATABASE_URL set; "
                    "set DATABASE_URL to point at the substrate",
                    file=sys.stderr,
                )
                return 1
        elif database_url:
            # No compose file and/or no docker, but DATABASE_URL is set — use it.
            if not _verify_connectivity(database_url):
                print(
                    "error: cannot connect to DATABASE_URL",
                    file=sys.stderr,
                )
                return 1
            print("substrate: ok (external via DATABASE_URL)")
        else:
            print(
                "error: no substrate available. "
                "Install docker (https://docs.docker.com/engine/install/) "
                "with a docker-compose.yml, or export DATABASE_URL=postgres://...",
                file=sys.stderr,
            )
            return 1

    # ------------------------------------------------------------------
    # 2. Migrations
    # ------------------------------------------------------------------
    from prospecta.db.migrate import get_schema_version, run_migrations

    pre_version = get_schema_version(database_url)
    try:
        result = run_migrations(database_url)
    except Exception as e:
        print(f"error: migration failed: {e}", file=sys.stderr)
        return 3
    post_version = get_schema_version(database_url)

    applied = result.get("applied", [])
    if not applied:
        print(f"migrations: ok (already at version {post_version})")
    else:
        print(f"migrations: applied {len(applied)} (now at version {post_version})")

    # ------------------------------------------------------------------
    # 3. Default bank
    # ------------------------------------------------------------------
    embedding_dim, used_default = _resolve_embedding_dim(args)
    if used_default:
        print(
            f"note: embedding_dim defaulting to {DEFAULT_DIM} {DEFAULT_DIM_NOTE}"
        )

    bank_id = "default"
    already_existed = _bank_exists(database_url, bank_id)

    from prospecta.memory import BankConfigConflict, Memory

    mem = Memory(database_url=database_url, bank_id=bank_id)
    try:
        try:
            mem.create_bank(bank_id, embedding_dim=embedding_dim)
        except BankConfigConflict as e:
            print(f"error: default bank: {e}", file=sys.stderr)
            return 4
        except Exception as e:
            print(f"error: default bank creation failed: {e}", file=sys.stderr)
            return 4
    finally:
        mem.close()

    if already_existed:
        print(f"default bank: ok (already exists, id={bank_id})")
    else:
        print(f"default bank: created (id={bank_id}, embedding_dim={embedding_dim})")

    return 0
