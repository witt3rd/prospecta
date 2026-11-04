#!/usr/bin/env bash
# Tier 1: the fast gate every landing must pass locally (target < 60 s): the lander's own self-test and the
# Python suite without the Postgres-backed tests (conftest.py skips them when PROSPECTA_TIER1=1).
# Tier 2 (scheduled in CI, never gates landing): the full suite with Postgres testcontainers.
export LC_ALL=C
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/test_land_direct.sh
PROSPECTA_TIER1=1 uv run --extra dev python -m pytest -q -p no:cacheprovider
