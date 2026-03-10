# ===========================================================================
# Master Thesis - Root Pytest Collection Guard
#   - André Filipe Gomes Silvestre, 20240502
#
# Restricts pytest collection to this repository's own test suites. This avoids
# accidental collection of unrelated `tests/` folders elsewhere in OneDrive when
# pytest is invoked with absolute paths containing `[` or `]`, which pytest may
# interpret as glob-style patterns on Windows.
# ===========================================================================

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
_ALLOWED_TEST_ROOTS = (
    PROJECT_ROOT / "tests",
    PROJECT_ROOT / "eval" / "tests",
)


def _is_same_or_relative_to(path: Path, candidate_root: Path) -> bool:
    """Return True when ``path`` is the same as or nested below ``candidate_root``."""
    try:
        path.relative_to(candidate_root)
        return True
    except ValueError:
        return path == candidate_root


def _is_same_or_ancestor_of(path: Path, candidate_root: Path) -> bool:
    """Return True when ``path`` is the same as or an ancestor of ``candidate_root``."""
    try:
        candidate_root.relative_to(path)
        return True
    except ValueError:
        return path == candidate_root


def pytest_ignore_collect(collection_path, config) -> bool:  # type: ignore[no-untyped-def]
    """Ignore paths outside the repository test suites during pytest collection."""
    path = Path(str(collection_path)).resolve()

    if not _is_same_or_relative_to(path, PROJECT_ROOT) and not _is_same_or_ancestor_of(path, PROJECT_ROOT):
        return True

    if path == PROJECT_ROOT:
        return False

    return not any(
        _is_same_or_relative_to(path, allowed_root)
        or _is_same_or_ancestor_of(path, allowed_root)
        for allowed_root in _ALLOWED_TEST_ROOTS
    )
