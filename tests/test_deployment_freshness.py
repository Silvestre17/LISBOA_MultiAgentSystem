# ===========================================================================
# Master Thesis - Deployment Freshness Tests
#   - André Filipe Gomes Silvestre, 20240502
#
#   Focused regressions for Streamlit deployment fingerprinting and markers.
# ===========================================================================

from __future__ import annotations

from agent.utils.deployment_freshness import (
    compute_deployment_fingerprint,
    fingerprint_changed,
    read_runtime_marker,
    runtime_marker_path,
    write_runtime_marker,
)


def test_deployment_fingerprint_changes_when_watched_file_changes(tmp_path) -> None:
    """A changed watched file must invalidate the runtime fingerprint."""
    app_file = tmp_path / "app.py"
    app_file.write_text("print('old')\n", encoding="utf-8")

    first = compute_deployment_fingerprint(tmp_path, watched_relative_paths=("app.py",))

    app_file.write_text("print('new deployment')\n", encoding="utf-8")
    second = compute_deployment_fingerprint(tmp_path, watched_relative_paths=("app.py",))

    assert first["fingerprint"] != second["fingerprint"]


def test_runtime_marker_round_trip_and_change_detection(tmp_path) -> None:
    """The marker file should persist and compare the active fingerprint."""
    marker = runtime_marker_path(tmp_path)
    current = {"fingerprint": "abc123", "git_commit": "commit-a"}

    assert fingerprint_changed(current, read_runtime_marker(marker))
    assert write_runtime_marker(marker, current)

    stored = read_runtime_marker(marker)
    assert stored["fingerprint"] == "abc123"
    assert not fingerprint_changed(current, stored)
    assert fingerprint_changed({"fingerprint": "def456"}, stored)
