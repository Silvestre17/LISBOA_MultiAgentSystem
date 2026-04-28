# ==========================================================================
# Master Thesis - VisitLisboa Scraper Safety Tests
#   - André Filipe Gomes Silvestre, 20240502
#
# Offline regressions for the VisitLisboa webscrapers. These tests verify that
# large harvest drops fail closed before overwriting checked-in JSON snapshots.
# ===========================================================================

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


class _FakeSession:
    """Minimal requests.Session replacement for scraper safety tests."""

    def __enter__(self) -> "_FakeSession":
        """Return the fake context manager session."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Close the fake context manager session."""
        return None


def _load_script_module(script_name: str):
    """Load a webscraping script by file path without requiring package imports."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "data_collection" / "webscraping" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_existing_snapshot(path: Path, prefix: str, count: int = 10) -> None:
    """Write a deterministic existing VisitLisboa snapshot."""
    payload = [{"url": f"https://example.test/{prefix}/{index}", "name": str(index)} for index in range(count)]
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_events_scraper_aborts_on_large_harvest_drop(tmp_path: Path, monkeypatch) -> None:
    """Events scraper should not overwrite JSON when harvested URLs drop by more than half."""
    module = _load_script_module("visitlisbon_events.py")
    module.__file__ = str(tmp_path / "visitlisbon_events.py")
    snapshot_path = tmp_path / "events.json"
    _write_existing_snapshot(snapshot_path, "events")

    monkeypatch.setattr(module.requests, "Session", lambda: _FakeSession())
    monkeypatch.setattr(module, "get_total_pages", lambda session, base_url: 1)
    monkeypatch.setattr(
        module,
        "get_event_urls_from_page",
        lambda session, page, base_url: {f"https://example.test/events/{index}" for index in range(4)},
    )

    with pytest.raises(SystemExit) as exc_info:
        module.main()

    assert exc_info.value.code == 1
    assert len(json.loads(snapshot_path.read_text(encoding="utf-8"))) == 10


def test_places_scraper_aborts_on_large_harvest_drop(tmp_path: Path, monkeypatch) -> None:
    """Places scraper should not overwrite JSON when harvested URLs drop by more than half."""
    module = _load_script_module("visitlisbon_places.py")
    module.__file__ = str(tmp_path / "visitlisbon_places.py")
    snapshot_path = tmp_path / "places.json"
    _write_existing_snapshot(snapshot_path, "places")

    monkeypatch.setattr(module.requests, "Session", lambda: _FakeSession())
    monkeypatch.setattr(module, "get_total_pages", lambda session, base_url: 1)
    monkeypatch.setattr(
        module,
        "get_place_urls_from_page",
        lambda session, page, base_url: {f"https://example.test/places/{index}" for index in range(4)},
    )

    with pytest.raises(SystemExit) as exc_info:
        module.main()

    assert exc_info.value.code == 1
    assert len(json.loads(snapshot_path.read_text(encoding="utf-8"))) == 10
