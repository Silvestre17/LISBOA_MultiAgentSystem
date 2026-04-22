# ===========================================================================
# Master Thesis - Startup Gate Regressions
#   - André Filipe Gomes Silvestre, 20240502
#
#   Focused regressions for the production startup gate.
#   Features:
#     - Verify the app blocks requests until preload succeeds
#     - Verify startup messaging surfaces failed readiness checks
#     - Verify shared preload status fails closed on transport or KB errors
#   Usage:
#     > python -m pytest tests/test_startup_gate.py -q
# ===========================================================================

from app import build_startup_gate_message, select_new_request, startup_gate_allows_requests
from agent.utils import startup_resources


def test_startup_gate_allows_requests_requires_transport_and_kb() -> None:
    """Requests must stay blocked until both transport and KB preload succeed."""
    assert not startup_gate_allows_requests(
        False,
        {"transport_ok": True, "kb_ok": True, "ok": False},
        use_multi_agent=True,
    )
    assert not startup_gate_allows_requests(
        True,
        {"transport_ok": False, "kb_ok": True, "ok": False},
        use_multi_agent=True,
    )
    assert not startup_gate_allows_requests(
        True,
        {"transport_ok": True, "kb_ok": False, "ok": False},
        use_multi_agent=True,
    )
    assert startup_gate_allows_requests(
        True,
        {"transport_ok": True, "kb_ok": True, "ok": True},
        use_multi_agent=True,
    )


def test_build_startup_gate_message_lists_failed_readiness_checks() -> None:
    """Gate messaging must surface the concrete startup checks that are still failing."""
    message = build_startup_gate_message(
        {
            "transport_ok": False,
            "transport_status": "⚠️ Transport layer incomplete: Metro [fallback], CP [0 stops], Carris Urban [0 stops], Carris Met. [reachable]",
            "kb_ok": False,
            "kb_status": "Could not load the knowledge base.",
        },
        language="en",
        use_multi_agent=True,
    )

    assert "Startup checks are incomplete" in message
    assert "Transport layer incomplete" in message
    assert "Could not load the knowledge base." in message


def test_select_new_request_blocks_capture_when_startup_gate_is_closed() -> None:
    """No new prompt should be queued while the startup gate is closed."""
    assert select_new_request(
        sidebar_request="weather",
        welcome_request="events",
        chat_request="typed prompt",
        pending_request=None,
        allow_requests=False,
    ) is None


def test_run_startup_preload_fails_closed_when_transport_layer_is_incomplete(monkeypatch) -> None:
    """Shared preload should fail closed when transport readiness is incomplete."""
    monkeypatch.setattr(
        startup_resources,
        "pre_warm_transport_networks",
        lambda: {
            "ok": False,
            "statuses": {"metro": "Metro line status is unavailable"},
            "details": {"metro": {"ok": False, "mode": "unreachable", "stations": 0}},
            "summary": "⚠️ Transport layer incomplete: Metro [unreachable], CP [0 stops], Carris Urban [0 stops], Carris Met. [reachable]",
        },
    )
    monkeypatch.setattr(startup_resources, "pre_warm_vector_store", lambda: True)

    preload_status = startup_resources.run_startup_preload(language="en", use_multi_agent=True)

    assert not preload_status["transport_ok"]
    assert not preload_status["ok"]
    assert "Transport layer incomplete" in preload_status["transport_status"]


def test_format_transport_layer_summary_reports_modes_and_counts() -> None:
    """Transport readiness summaries must expose the operator mode/count contract used by startup logs."""
    summary = startup_resources.format_transport_layer_summary(
        {
            "metro": {"ok": True, "mode": "fallback"},
            "cp": {"ok": True, "stops": 81},
            "carris": {"ok": True, "stops": 5123},
            "carris_metropolitana": {"ok": False},
        },
        overall_ok=False,
    )

    assert summary == (
        "⚠️ Transport layer incomplete: Metro [fallback], CP [81 stops], "
        "Carris Urban [5123 stops], Carris Met. [unreachable]"
    )
