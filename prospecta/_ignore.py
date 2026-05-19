"""
Memory Ignore Patterns

Controls which paths are excluded from indexing.

No external dependencies - can be imported standalone.

Ported from animus (animus/memory/ignore.py). Behavior preserved.
"""

from __future__ import annotations

import fnmatch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Always excluded - never part of the index
DEFAULT_IGNORE_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".claude",
    "site-packages",
}

DEFAULT_IGNORE_PATTERNS = [
    "*.pyc",
    "*.pyo",
    "*.egg-info",
]

_ignore_patterns_cache: dict[Path, list[str]] = {}


def load_ignore_patterns(root: Path) -> list[str]:
    """Load ignore patterns from .memoryignore file.

    Args:
        root: Directory to look for .memoryignore

    Returns:
        List of gitignore-style patterns
    """
    if root in _ignore_patterns_cache:
        return _ignore_patterns_cache[root]

    patterns = list(DEFAULT_IGNORE_PATTERNS)
    ignore_file = root / ".memoryignore"

    if ignore_file.exists():
        try:
            content = ignore_file.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                patterns.append(line)
        except Exception as e:
            logger.debug(f"[memory] failed to read .memoryignore at {ignore_file}: {e}")

    _ignore_patterns_cache[root] = patterns
    return patterns


def should_ignore(path: Path, root: Path) -> bool:
    """Check if a path should be ignored based on .memoryignore patterns.

    Args:
        path: Path to check
        root: Root directory for pattern matching

    Returns:
        True if path should be ignored
    """
    # Check default ignore directories
    for part in path.parts:
        if part in DEFAULT_IGNORE_DIRS:
            return True

    # Get relative path for pattern matching
    try:
        rel_path = path.relative_to(root)
    except ValueError:
        rel_path = path

    rel_str = str(rel_path).replace("\\", "/")
    patterns = load_ignore_patterns(root)

    ignored = False
    for pattern in patterns:
        # Handle negation
        if pattern.startswith("!"):
            if fnmatch.fnmatch(rel_str, pattern[1:]):
                ignored = False
            elif fnmatch.fnmatch(path.name, pattern[1:]):
                ignored = False
        else:
            # Directory patterns (ending with /)
            if pattern.endswith("/"):
                dir_pattern = pattern.rstrip("/")
                if rel_str.startswith(dir_pattern + "/") or rel_str == dir_pattern:
                    ignored = True
                elif any(part == dir_pattern for part in rel_path.parts[:-1]):
                    ignored = True
            # File patterns
            elif fnmatch.fnmatch(rel_str, pattern):
                ignored = True
            elif fnmatch.fnmatch(path.name, pattern):
                ignored = True

    return ignored


def clear_ignore_cache():
    """Clear the ignore patterns cache (useful for testing)."""
    _ignore_patterns_cache.clear()
