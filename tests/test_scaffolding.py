"""Scaffolding acceptance tests for prospecta v0.1.

These tests verify the bootstrap layout per plan-v2.md §3.1 + §3.2.
"""


def test_pyproject_toml_parses():
    import tomllib
    from pathlib import Path
    data = tomllib.loads(Path("pyproject.toml").read_text())
    assert data["project"]["name"] == "prospecta"
    assert data["project"]["version"] == "0.1.0.dev0"
    deps = data["project"]["dependencies"]
    assert any("psycopg" in d for d in deps)
    assert any("pgvector" in d for d in deps)
    # NOTE: spec contained a typo (`deps.lower()` on a list); fixed to
    # check each dep string. Original intent: jinja2 + pyyaml present.
    assert any("jinja2" in d.lower() for d in deps)
    assert any("pyyaml" in d.lower() for d in deps)


def test_docker_compose_parses():
    import yaml
    from pathlib import Path
    data = yaml.safe_load(Path("docker-compose.yml").read_text())
    assert "postgres" in data["services"]
    assert "pgvector/pgvector" in data["services"]["postgres"]["image"]


def test_import_prospecta():
    import prospecta  # bare import works  # noqa: F401


def test_gitignore_contents():
    from pathlib import Path
    text = Path(".gitignore").read_text()
    for pattern in [".env", "__pycache__", "*.egg-info", ".pytest_cache"]:
        assert pattern in text


def test_readme_mentions_spine():
    from pathlib import Path
    text = Path("README.md").read_text().lower()
    assert any(
        phrase in text
        for phrase in ["bilateral", "both sides", "question-form", "llm-mediated"]
    )


def test_env_example_has_database_url():
    from pathlib import Path
    text = Path(".env.example").read_text()
    assert "DATABASE_URL" in text
