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

from agent.utils.langsmith_tracing import (
    get_langsmith_display_state,
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
    }

    status = resolve_langsmith_tracing_status(
        env=env,
        imported_symbols=symbols,
        probe=lambda client_cls, endpoint, api_key: (True, "LangSmith tracing enabled"),
    )

    assert status["enabled"] is True
    assert status["requested"] is True
    assert status["traceable"] is symbols["traceable"]
    assert status["ContextThreadPoolExecutor"] is symbols["ContextThreadPoolExecutor"]
    assert env["LANGCHAIN_TRACING_V2"] == "true"


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
