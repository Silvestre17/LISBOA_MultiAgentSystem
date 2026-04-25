# ===========================================================================
# Master Thesis - Deployment Freshness Tests
#   - André Filipe Gomes Silvestre, 20240502
#
#   Focused regressions for Streamlit deployment fingerprinting and markers.
# ===========================================================================

from __future__ import annotations

from unittest.mock import patch

from agent.utils.deployment_freshness import (
    compute_deployment_fingerprint,
    fingerprint_changed,
)


class FakeSessionState(dict):
    """Small dict with Streamlit session-state attribute access."""

    def __getattr__(self, name: str):
        """Return a stored value via attribute access."""
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value) -> None:
        """Store a value via attribute access."""
        self[name] = value


def test_deployment_fingerprint_changes_when_watched_file_changes(tmp_path) -> None:
    """A changed watched file must invalidate the runtime fingerprint."""
    app_file = tmp_path / "app.py"
    app_file.write_text("print('old')\n", encoding="utf-8")

    first = compute_deployment_fingerprint(tmp_path, watched_relative_paths=("app.py",))

    app_file.write_text("print('new deployment')\n", encoding="utf-8")
    second = compute_deployment_fingerprint(tmp_path, watched_relative_paths=("app.py",))

    assert first["fingerprint"] != second["fingerprint"]


def test_fingerprint_changed_compares_payloads() -> None:
    """Fingerprint comparison should not rely on any disk marker."""
    stored = {"fingerprint": "abc123"}

    assert not fingerprint_changed({"fingerprint": "abc123"}, stored)
    assert fingerprint_changed({"fingerprint": "def456"}, stored)
    assert not fingerprint_changed({"fingerprint": ""}, stored)


def test_app_first_boot_records_fingerprint_without_cache_clear(tmp_path) -> None:
    """Fresh Streamlit sessions must not clear caches or rerun on first boot."""
    import app

    (tmp_path / "app.py").write_text("print('boot')\n", encoding="utf-8")
    session_state = FakeSessionState()

    with patch("app.st.session_state", session_state), patch(
        "app.clear_known_runtime_caches"
    ) as clear_caches, patch("app.purge_lisboa_import_cache") as purge_imports, patch(
        "app.st.rerun"
    ) as rerun:
        refreshed = app.ensure_fresh_runtime_after_deploy(
            root_dir=tmp_path,
            rerun_on_refresh=True,
        )

    assert refreshed is False
    assert session_state.get("deployment_fingerprint")
    clear_caches.assert_not_called()
    purge_imports.assert_not_called()
    rerun.assert_not_called()


def test_app_loaded_session_refreshes_when_fingerprint_changes(tmp_path) -> None:
    """Loaded sessions should clear stale runtime state after code/data changes."""
    import app

    watched_file = tmp_path / "app.py"
    watched_file.write_text("print('old')\n", encoding="utf-8")
    previous = compute_deployment_fingerprint(tmp_path, watched_relative_paths=("app.py",))[
        "fingerprint"
    ]
    watched_file.write_text("print('new')\n", encoding="utf-8")
    session_state = FakeSessionState(
        {
            "deployment_fingerprint": previous,
            "assistant": object(),
            "initialized": True,
        }
    )

    with patch("app.st.session_state", session_state), patch(
        "app.clear_known_runtime_caches",
        return_value=["streamlit.cache_data"],
    ) as clear_caches, patch("app.purge_lisboa_import_cache") as purge_imports, patch(
        "app.st.rerun"
    ) as rerun:
        refreshed = app.ensure_fresh_runtime_after_deploy(
            root_dir=tmp_path,
            rerun_on_refresh=False,
        )

    assert refreshed is True
    assert session_state.get("deployment_fingerprint") != previous
    assert session_state.get("initialized") is False
    assert session_state.get("deployment_refresh_details", {}).get("cleared") == [
        "streamlit.cache_data"
    ]
    clear_caches.assert_called_once()
    purge_imports.assert_not_called()
    rerun.assert_not_called()
