"""P17 extras-import gate: validate the public surface and P3 invariant."""
from __future__ import annotations

import subprocess


def test_litellm_present_import_succeeds():
    """When litellm is installed (CI with extras=defaults), import works."""
    from prospecta.defaults import make_default_embedder, make_default_llm

    assert callable(make_default_embedder)
    assert callable(make_default_llm)


def test_p3_audit_core_has_no_litellm_imports():
    """P3 enforcement: only prospecta/defaults/ may import litellm."""
    result = subprocess.run(
        ["grep", "-rn", "-l", r"\bimport litellm\|from litellm", "prospecta/"],
        capture_output=True,
        text=True,
        cwd="/home/dt/src/witt3rd/prospecta",
    )
    matched_files = [line for line in result.stdout.strip().split("\n") if line]
    for f in matched_files:
        assert f.startswith("prospecta/defaults/"), (
            f"P3 violation: litellm imported outside prospecta/defaults/: {f}"
        )
