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

import os
from pathlib import Path

import pytest

_TRUTHY_VALUES = {"1", "true", "yes", "on"}
_ENABLE_TEST_LANGSMITH = (
    os.getenv("LISBOA_ENABLE_TEST_LANGSMITH", "").strip().lower()
    in _TRUTHY_VALUES
)

if not _ENABLE_TEST_LANGSMITH:
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

PROJECT_ROOT = Path(__file__).resolve().parent
_ALLOWED_TEST_ROOTS = (
    PROJECT_ROOT / "tests",
    PROJECT_ROOT / "eval" / "tests",
)


def pytest_addoption(parser) -> None:  # type: ignore[no-untyped-def]
    """Register opt-in flags for expensive integration suites."""
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run tests marked as live and allowed to hit external services.",
    )


def pytest_configure(config) -> None:  # type: ignore[no-untyped-def]
    """Register project-specific markers used by integration and coverage tests."""
    config.addinivalue_line(
        "markers",
        "coverage: tests that validate tool and prompt coverage manifests",
    )
    config.addinivalue_line(
        "markers",
        "live: tests that may call live external APIs or local runtime resources",
    )


@pytest.fixture(scope="session", autouse=True)
def _disable_langsmith_tracing_for_pytest() -> None:  # type: ignore[no-untyped-def]
    """Disable LangSmith tracing across pytest unless explicitly re-enabled."""
    if _ENABLE_TEST_LANGSMITH:
        yield    # type: ignore[unreachable]
        return

    from agent.utils.langsmith_tracing import tracing_context

    with tracing_context(enabled=False):
        yield    # type: ignore[unreachable]


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


def pytest_collection_modifyitems(config, items) -> None:  # type: ignore[no-untyped-def]
    """Skip live tests unless the user explicitly opts in."""
    if config.getoption("--run-live"):
        return

    skip_live = pytest.mark.skip(reason="Live test skipped. Re-run with --run-live to exercise external services.")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
