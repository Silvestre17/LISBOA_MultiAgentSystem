# ===========================================================================
# Master Thesis - LangSmith Tracing Guard Tests
#   - André Filipe Gomes Silvestre, 20240502
#
# Regression tests for the optional LangSmith tracing bootstrap. These tests
# ensure tracing only activates when explicitly requested and preflight
# validation succeeds, and that it fails closed for placeholder or forbidden
# credentials without leaking runtime errors.
# ===========================================================================

# Required libraries:
# pip install pytest

from __future__ import annotations

from typing import Any, Dict

import agent.utils.langsmith_tracing as langsmith_tracing
from agent.utils.langsmith_tracing import (
    annotate_current_run,
    get_langsmith_display_state,
    get_langsmith_project_name,
    resolve_langsmith_tracing_status,
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
