# ===========================================================================
# Master Thesis - Lean Pytest Configuration
#   - Andre Filipe Gomes Silvestre, 20240502
#
# Tests are lightweight safety nets only. User-facing quality must be checked
# with real LISBOA prompt runs and visual/app validation where relevant.
# ===========================================================================

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add the opt-in flag used for live API smoke tests."""
    try:
        parser.addoption(
            "--run-live",
            action="store_true",
            default=False,
            help="Run live API smoke tests marked with @pytest.mark.live.",
        )
    except ValueError as exc:
        if "--run-live" not in str(exc):
            raise


def pytest_configure(config: pytest.Config) -> None:
    """Register local markers so lean runs stay warning-free."""
    config.addinivalue_line(
        "markers",
        "live: tests that call live or on-demand external providers; skipped unless --run-live is set",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip live API smoke tests unless explicitly requested."""
    if config.getoption("--run-live"):
        return

    skip_live = pytest.mark.skip(reason="use --run-live to run live API smoke tests")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the repository root."""
    return PROJECT_ROOT
