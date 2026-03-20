# ===========================================================================
# Master Thesis - LangSmith Tracing Guard Tests
#   - André Filipe Gomes Silvestre, 20240502
#
# Regression tests for the optional LangSmith tracing bootstrap. These tests
# ensure tracing only activates when explicitly requested and preflight
# validation succeeds, and that it fails closed for placeholder or forbidden
# credentials without leaking runtime errors.
#
# Run from the repository root with a relative path:
#   python -m pytest tests/test_langsmith_tracing.py -q
# Useful parameters:
#   -vv             verbose mode
#   -k preflight    run only the preflight-related checks
#   -x              stop on first failure
#   --tb=short      shorter tracebacks
# Notes:
#   - Prefer relative paths in this workspace. Absolute pytest paths may be
#     treated as glob patterns on Windows because the folder name includes
#     `[` and `]`.
# ===========================================================================

# Required libraries:
# pip install pytest

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict

import agent.utils.langsmith_tracing as langsmith_tracing
from agent.utils.langsmith_tracing import (
    annotate_current_run,
    get_langsmith_display_state,
    get_langsmith_project_name,
    get_langsmith_request_tracking_status,
    get_langsmith_scoped_project_name,
    is_langsmith_tracing_opted_in,
    resolve_langsmith_tracing_status,
    tracing_disabled_unless_opted_in,
    tracing_project_override,
)


def _fake_symbols() -> Dict[str, Any]:
    """Return a minimal injected LangSmith symbol table for tests."""

    class FakeExecutor:
        """Dummy executor used to avoid importing LangSmith in tests."""

    class FakeRunTree:
        """Dummy run tree type used for enabled-path assertions."""

    def fake_traceable(*args, **kwargs):
        """Return a decorator that preserves the wrapped function."""

        def decorator(func):
            return func

        return decorator

    def fake_get_current_run_tree() -> None:
        """Return None to mimic the no-active-run state."""
        return None

    class FakeClient:
        """Placeholder client type passed into the injected preflight probe."""

    return {
        "Client": FakeClient,
        "traceable": fake_traceable,
        "get_current_run_tree": fake_get_current_run_tree,
        "RunTree": FakeRunTree,
        "ContextThreadPoolExecutor": FakeExecutor,
    }


def test_tracing_stays_disabled_when_flag_is_off() -> None:
    """Tracing should remain disabled when the env flag is false."""
    env = {
        "LANGCHAIN_TRACING_V2": "false",
        "LANGCHAIN_API_KEY": "real-looking-key",
    }

    status = resolve_langsmith_tracing_status(env=env)

    assert status["enabled"] is False
    assert status["requested"] is False
    assert env["LANGCHAIN_TRACING_V2"] == "false"
    assert env["LANGSMITH_TRACING"] == "false"


def test_tracing_disables_placeholder_api_keys() -> None:
    """Tracing should fail closed when the API key still looks templated."""
    env = {
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_API_KEY": "your_langsmith_api_key_here",
        "LANGCHAIN_ENDPOINT": "https://eu.api.smith.langchain.com",
    }

    status = resolve_langsmith_tracing_status(
        env=env,
        imported_symbols=_fake_symbols(),
    )

    assert status["enabled"] is False
    assert status["requested"] is True
    assert "placeholder" in status["reason"].lower()
    assert env["LANGCHAIN_TRACING_V2"] == "false"
    assert env["LANGSMITH_TRACING"] == "false"


def test_tracing_disables_forbidden_credentials() -> None:
    """Tracing should auto-disable when LangSmith rejects the credentials."""
    env = {
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_API_KEY": "lsv2-real-looking-key",
        "LANGCHAIN_ENDPOINT": "https://eu.api.smith.langchain.com",
    }

    status = resolve_langsmith_tracing_status(
        env=env,
        imported_symbols=_fake_symbols(),
        probe=lambda client_cls, endpoint, api_key: (
            False,
            "LangSmith tracing disabled: API key is forbidden for this endpoint",
        ),
    )

    assert status["enabled"] is False
    assert status["requested"] is True
    assert "forbidden" in status["reason"].lower()
    assert env["LANGCHAIN_TRACING_V2"] == "false"
    assert env["LANGSMITH_TRACING"] == "false"


def test_tracing_enables_after_successful_preflight() -> None:
    """Tracing should stay enabled only when the preflight check succeeds."""
    symbols = _fake_symbols()
    env = {
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_API_KEY": "lsv2-real-looking-key",
        "LANGCHAIN_ENDPOINT": "https://eu.api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "legacy-project",
    }

    status = resolve_langsmith_tracing_status(
        env=env,
        imported_symbols=symbols,
        probe=lambda client_cls, endpoint, api_key: (True, "LangSmith tracing enabled"),
    )

    assert status["enabled"] is True
    assert status["requested"] is True
    assert status["project_name"] == "legacy-project"
    assert status["traceable"] is symbols["traceable"]
    assert status["ContextThreadPoolExecutor"] is symbols["ContextThreadPoolExecutor"]
    assert env["LANGCHAIN_TRACING_V2"] == "true"
    assert env["LANGSMITH_TRACING"] == "true"
    assert env["LANGSMITH_API_KEY"] == "lsv2-real-looking-key"
    assert env["LANGSMITH_ENDPOINT"] == "https://eu.api.smith.langchain.com"
    assert env["LANGSMITH_PROJECT"] == "legacy-project"


def test_tracing_passes_workspace_id_to_preflight() -> None:
    """Workspace ids should be forwarded to the LangSmith preflight probe."""
    captured: Dict[str, Any] = {}

    def probe(client_cls, endpoint, api_key, workspace_id):
        captured["client_cls"] = client_cls
        captured["endpoint"] = endpoint
        captured["api_key"] = api_key
        captured["workspace_id"] = workspace_id
        return True, "LangSmith tracing enabled"

    status = resolve_langsmith_tracing_status(
        env={
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_API_KEY": "lsv2-real-looking-key",
            "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com",
            "LANGSMITH_WORKSPACE_ID": "ws_12345",
        },
        imported_symbols=_fake_symbols(),
        probe=probe,
    )

    assert status["enabled"] is True
    assert status["workspace_id"] == "ws_12345"
    assert captured["workspace_id"] == "ws_12345"


def test_display_state_classifies_active_tracing() -> None:
    """UI display state should report active tracing when enabled."""
    display = get_langsmith_display_state(
        {
            "enabled": True,
            "requested": True,
            "reason": "LangSmith tracing enabled",
        }
    )

    assert display == {
        "state": "active",
        "reason": "LangSmith tracing enabled",
    }


def test_display_state_classifies_invalid_credentials() -> None:
    """Forbidden or invalid API keys should map to the credential warning state."""
    display = get_langsmith_display_state(
        {
            "enabled": False,
            "requested": True,
            "reason": "LangSmith tracing disabled: API key is forbidden for this endpoint",
        }
    )

    assert display["state"] == "auto_disabled_invalid_credentials"
    assert "forbidden" in display["reason"].lower()


def test_display_state_classifies_invalid_configuration() -> None:
    """Broken endpoints or generic preflight failures should map to config warnings."""
    display = get_langsmith_display_state(
        {
            "enabled": False,
            "requested": True,
            "reason": "LangSmith tracing disabled: endpoint is missing or still a placeholder",
        }
    )

    assert display["state"] == "auto_disabled_invalid_configuration"


def test_display_state_classifies_env_disabled() -> None:
    """Explicitly disabled tracing should remain a plain disabled state."""
    display = get_langsmith_display_state(
        {
            "enabled": False,
            "requested": False,
            "reason": "LangSmith tracing is disabled by environment",
        }
    )

    assert display["state"] == "disabled"


def test_get_langsmith_project_name_prefers_status_then_default(monkeypatch) -> None:
    """Project helper should prefer resolved status and otherwise fall back to default."""
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)

    assert get_langsmith_project_name({"project_name": "thesis-traces"}) == "thesis-traces"
    assert get_langsmith_project_name({"project_name": None}) == "default"


def test_request_tracking_status_marks_active_request_as_save_attempt() -> None:
    """An attached run context should count as a save attempt for the current request."""

    class FakeRunTree:
        id = "run_123"

    tracking = get_langsmith_request_tracking_status(
        status={
            "enabled": True,
            "requested": True,
            "reason": "LangSmith tracing enabled",
            "project_name": "LISBOA Chat",
        },
        run_tree=FakeRunTree(),
    )

    assert tracking["tracking_state"] == "tracking_request"
    assert tracking["status_label"] == "enabled"
    assert tracking["save_attempted"] is True
    assert tracking["current_run_attached"] is True
    assert tracking["project_name"] == "LISBOA Chat"
    assert tracking["run_id"] == "run_123"
    assert "asynchronous" in tracking["note"].lower()


def test_request_tracking_status_reports_auto_disabled_request() -> None:
    """Auto-disabled tracing should surface that this request is not being saved."""
    tracking = get_langsmith_request_tracking_status(
        status={
            "enabled": False,
            "requested": True,
            "reason": "LangSmith tracing disabled: API key is forbidden for this endpoint",
            "project_name": "LISBOA Chat",
        },
        run_tree=None,
    )

    assert tracking["tracking_state"] == "auto_disabled"
    assert tracking["status_label"] == "auto-disabled"
    assert tracking["save_attempted"] is False
    assert tracking["current_run_attached"] is False
    assert tracking["project_name"] == "LISBOA Chat"
    assert "forbidden" in tracking["note"].lower()


def test_is_langsmith_tracing_opted_in_uses_truthy_custom_env_flag() -> None:
    """Non-interactive tracing should require an explicit truthy opt-in flag."""
    assert is_langsmith_tracing_opted_in(
        "LISBOA_ENABLE_CLI_LANGSMITH",
        env={"LISBOA_ENABLE_CLI_LANGSMITH": "true"},
    ) is True
    assert is_langsmith_tracing_opted_in(
        "LISBOA_ENABLE_CLI_LANGSMITH",
        env={"LISBOA_ENABLE_CLI_LANGSMITH": "false"},
    ) is False


def test_get_langsmith_scoped_project_name_derives_benchmark_project_from_base_env() -> None:
    """Offline analysis runs should get a project distinct from the base chat project."""
    project_name = get_langsmith_scoped_project_name(
        "Benchmark",
        env={"LANGSMITH_PROJECT": "LISBOA Chat"},
    )

    assert project_name == "LISBOA Chat - Benchmark"


def test_get_langsmith_scoped_project_name_prefers_explicit_override() -> None:
    """Runner-specific project overrides should win over the derived suffix."""
    project_name = get_langsmith_scoped_project_name(
        "Ablation",
        env_name="LISBOA_LANGSMITH_ABLATION_PROJECT",
        env={
            "LANGSMITH_PROJECT": "LISBOA Chat",
            "LISBOA_LANGSMITH_ABLATION_PROJECT": "LISBOA Ablation Study",
        },
    )

    assert project_name == "LISBOA Ablation Study"


def test_tracing_disabled_unless_opted_in_uses_disabled_context(monkeypatch) -> None:
    """Offline workloads should enter a disabled tracing context unless explicitly opted in."""
    calls = []

    @contextmanager
    def fake_tracing_context(**kwargs):
        calls.append(kwargs)
        yield

    monkeypatch.setattr(langsmith_tracing, "tracing_context", fake_tracing_context)

    with tracing_disabled_unless_opted_in(env={}):
        pass

    assert calls == [{"enabled": False}]


def test_tracing_disabled_unless_opted_in_skips_override_when_enabled() -> None:
    """Explicit opt-in flags should leave the tracing context untouched."""
    calls = []

    @contextmanager
    def fake_tracing_context(**kwargs):
        calls.append(kwargs)
        yield

    original_tracing_context = langsmith_tracing.tracing_context
    langsmith_tracing.tracing_context = fake_tracing_context
    try:
        with tracing_disabled_unless_opted_in(
            "LISBOA_ENABLE_CLI_LANGSMITH",
            env={"LISBOA_ENABLE_CLI_LANGSMITH": "1"},
        ):
            pass
    finally:
        langsmith_tracing.tracing_context = original_tracing_context

    assert calls == []


def test_tracing_project_override_sets_and_restores_project_env(monkeypatch) -> None:
    """Dedicated offline runs should override the LangSmith project only inside the wrapped scope."""
    calls = []

    @contextmanager
    def fake_tracing_context(**kwargs):
        calls.append(kwargs)
        yield

    env = {
        "LANGSMITH_PROJECT": "LISBOA Chat",
        "LANGCHAIN_PROJECT": "LISBOA Chat",
    }
    monkeypatch.setattr(langsmith_tracing, "tracing_context", fake_tracing_context)

    with tracing_project_override("LISBOA Chat - Benchmark", env=env):
        assert env["LANGSMITH_PROJECT"] == "LISBOA Chat - Benchmark"
        assert env["LANGCHAIN_PROJECT"] == "LISBOA Chat - Benchmark"

    assert env["LANGSMITH_PROJECT"] == "LISBOA Chat"
    assert env["LANGCHAIN_PROJECT"] == "LISBOA Chat"
    assert calls == [{"project_name": "LISBOA Chat - Benchmark"}]


def test_annotate_current_run_updates_metadata_and_tags(monkeypatch) -> None:
    """Metadata and tags should be attached through the RunTree API, not extra payloads."""

    class FakeRunTree:
        def __init__(self) -> None:
            self.metadata = {"existing": "value"}
            self.tags = ["baseline"]

    fake_run_tree = FakeRunTree()
    monkeypatch.setattr(langsmith_tracing, "get_current_run_tree", lambda: fake_run_tree)

    updated = annotate_current_run(
        metadata={"assistant_mode": "multi-agent", "language": "pt"},
        tags=["weather", "baseline", "transport"],
    )

    assert updated is True
    assert fake_run_tree.metadata == {
        "existing": "value",
        "assistant_mode": "multi-agent",
        "language": "pt",
    }
    assert fake_run_tree.tags == ["baseline", "weather", "transport"]
