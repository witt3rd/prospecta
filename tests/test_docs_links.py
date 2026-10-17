"""The documentation links and the doc paths the code cites must resolve.

* every relative link `[text](path#anchor)` of a Markdown file points at a file that exists
  (and, for a `.md` target with an anchor, at a heading that exists);
* every `docs/<name>.md` path named in a Markdown file, a source file, a test, a script, a
  Rust comment or the packaging files exists, unless the line points at git history
  ("git log", "git checkout", "git history");
* the build files that name the README (`pyproject.toml`, `Dockerfile`) name a file that exists.

Run it as a script for the report:  python tests/test_docs_links.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", ".venv", "node_modules", "target", "__pycache__", ".pytest_cache", "fixtures"}
MARKDOWN_ROOTS = ["README.md", "PRINCIPLES.md", "AGENTS.md", "docs", "skills", "tui/README.md"]
CODE_GLOBS = ["prospecta/**/*.py", "prospecta/**/*.sql", "tests/**/*.py", "scripts/*.py",
              "tui/src/**/*.rs", ".env.example", "Dockerfile", "pyproject.toml",
              "docker-compose.yml", ".github/workflows/*.yml"]
LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
DOC_PATH = re.compile(r"(?<![\w/.-])(docs/[A-Za-z0-9_./-]*[A-Za-z0-9_]\.md)")
HISTORY = re.compile(r"git (log|checkout|history|show)|in the git history|git keeps")


def _skipped(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts)


def markdown_files() -> list[Path]:
    out: list[Path] = []
    for spec in MARKDOWN_ROOTS:
        p = ROOT / spec
        out.extend([p] if p.is_file() else sorted(p.rglob("*.md")))
    return [p for p in out if not _skipped(p)]


# An applied migration is never edited, so its comments are history: a doc it cites may be gone since.
MIGRATIONS = ROOT / "prospecta" / "db" / "migrations"


def code_files() -> list[Path]:
    out: list[Path] = []
    for pattern in CODE_GLOBS:
        out.extend(p for p in ROOT.glob(pattern)
                   if p.is_file() and not _skipped(p) and MIGRATIONS not in p.parents)
    return sorted(set(out) - {Path(__file__).resolve()})


def _slug(heading: str) -> str:
    """GitHub's anchor of a heading: lower case, punctuation dropped, spaces to hyphens."""
    text = re.sub(r"`", "", heading.strip().lower())
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    out: set[str] = set()
    fenced = False
    for line in path.read_text().splitlines():
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced and line.startswith("#"):
            out.add(_slug(line.lstrip("#")))
    return out


def check() -> tuple[list[str], int, int]:
    """(problems, files checked, references checked)."""
    problems: list[str] = []
    refs = 0
    md = markdown_files()
    for path in md:
        rel = path.relative_to(ROOT)
        fenced = False
        for n, line in enumerate(path.read_text().splitlines(), start=1):
            if line.startswith("```"):
                fenced = not fenced
            if not fenced:
                for target in LINK.findall(line):
                    if re.match(r"[a-z][a-z0-9+.-]*:", target):         # http:, https:, mailto:
                        continue
                    refs += 1
                    name, _, anchor = target.partition("#")
                    dest = path if not name else (path.parent / name).resolve()
                    if not dest.exists():
                        problems.append(f"{rel}:{n}: link to {target!r}: {dest.relative_to(ROOT) if dest.is_relative_to(ROOT) else dest} does not exist")
                    elif anchor and dest.suffix == ".md" and anchor not in anchors(dest):
                        problems.append(f"{rel}:{n}: link to {target!r}: no heading #{anchor} in {dest.relative_to(ROOT)}")
            if not HISTORY.search(line):
                for doc in DOC_PATH.findall(line):
                    refs += 1
                    if not (ROOT / doc).is_file():
                        problems.append(f"{rel}:{n}: names {doc}, which does not exist")
    code = code_files()
    for path in code:
        rel = path.relative_to(ROOT)
        for n, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
            if HISTORY.search(line):
                continue
            for doc in DOC_PATH.findall(line):
                refs += 1
                if not (ROOT / doc).is_file():
                    problems.append(f"{rel}:{n}: names {doc}, which does not exist")
    for build, pattern in (("pyproject.toml", r'readme\s*=\s*"([^"]+)"'), ("Dockerfile", r"COPY\s+(.*)")):
        text = (ROOT / build).read_text()
        for m in re.finditer(pattern, text):
            for name in m.group(1).split():
                if name.endswith(".md") or name == "LICENSE":
                    refs += 1
                    if not (ROOT / name).is_file():
                        problems.append(f"{build}: names {name}, which does not exist")
    return problems, len(md) + len(code), refs


def test_documentation_links_and_cited_doc_paths_resolve():
    problems, files, refs = check()
    assert refs > 50, "the checker found almost nothing to check: its patterns are broken"
    assert problems == [], "\n".join(problems)


def test_every_doc_is_reachable_from_the_readme():
    """No orphans: each Markdown file under docs/ is linked or named from the README or from
    another doc that is (the doc map is the README's)."""
    names = {p.relative_to(ROOT).as_posix() for p in markdown_files() if p.parent.name != "tests"}
    text = {n: (ROOT / n).read_text() for n in names}
    reached = {"README.md"}
    frontier = ["README.md"]
    while frontier:
        cur = frontier.pop()
        linked = set()          # the files a relative link of `cur` resolves to (skills link `references/x.md`)
        for target in LINK.findall(text[cur]):
            name = target.partition("#")[0]
            if name and not re.match(r"[a-z][a-z0-9+.-]*:", name):
                try:
                    linked.add((ROOT / cur).parent.joinpath(name).resolve().relative_to(ROOT).as_posix())
                except ValueError:
                    pass
        for other in names:
            base = Path(other).name
            if other not in reached and (other in linked or other in text[cur] or f"]({base}" in text[cur]
                                         or f"](./{base}" in text[cur]):
                reached.add(other)
                frontier.append(other)
    orphans = sorted(n for n in names if n not in reached)
    assert orphans == [], f"documents nothing leads to: {orphans}"


if __name__ == "__main__":
    problems, files, refs = check()
    print(f"{files} files, {refs} references checked, {len(problems)} broken")
    for p in problems:
        print("  ", p)
    sys.exit(1 if problems else 0)
