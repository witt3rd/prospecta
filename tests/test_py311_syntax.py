"""Every module in the package must compile on Python 3.11 (live gateways run it)."""
import pathlib
import pytest

import prospecta

ROOT = pathlib.Path(prospecta.__file__).parent
FILES = sorted(ROOT.rglob("*.py"))


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_compiles_under_py311(path):
    compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True)
