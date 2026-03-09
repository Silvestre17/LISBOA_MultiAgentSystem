# ===========================================================================
# Master Thesis - Raw Model Connection Probe Tests
#   - André Filipe Gomes Silvestre, 20240502
#
# Regression tests for the raw HTTP model connection checks used by the
# Streamlit configuration flows. These helpers must never rely on LangChain
# model invocation paths, otherwise the UI health checks would consume
# unnecessary LangSmith traces.
# ===========================================================================

# Required libraries:
# pip install pytest

from __future__ import annotations

from typing import Any, Dict

import agent.utils.model_connection_probe as model_connection_probe
from agent.utils.model_connection_probe import (
    build_connection_probe_request,
    perform_raw_model_connection_probe,
)


class _SecretValue:
    """Minimal SecretStr-like object used in tests."""

    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        return self._value


class _FakeResponse:
    """Small fake HTTP response for requests.post monkeypatching."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Dict[str, Any]:
        return self._payload


def test_build_connection_probe_request_for_reasoning_model() -> None:
    """Reasoning models should use max_completion_tokens in the raw payload."""

    class FakeLLM:
        openai_api_base = "https://api.example.com/v1/"
        openai_api_key = _SecretValue("sk-test-secret")
        model_name = "gpt-5-mini"

        def invoke(self, *_args, **_kwargs):
            raise AssertionError("Raw probe must not call invoke().")

    request_data = build_connection_probe_request(FakeLLM(), "gpt-5-mini")

    assert request_data["endpoint"] == "https://api.example.com/v1/chat/completions"
    assert request_data["headers"]["Authorization"] == "Bearer sk-test-secret"
    assert request_data["payload"]["model"] == "gpt-5-mini"
    assert request_data["payload"]["max_completion_tokens"] == 100
    assert "temperature" not in request_data["payload"]
    assert "max_tokens" not in request_data["payload"]


def test_build_connection_probe_request_for_standard_model() -> None:
    """Standard chat models should use a 1-token minimal completion payload."""

    class FakeLLM:
        base_url = "http://localhost:1234/v1"
        api_key = "lm-studio"
        model = "gpt-4.1-mini"

        def invoke(self, *_args, **_kwargs):
            raise AssertionError("Raw probe must not call invoke().")

    request_data = build_connection_probe_request(FakeLLM(), "gpt-4.1-mini")

    assert request_data["endpoint"] == "http://localhost:1234/v1/chat/completions"
    assert request_data["payload"]["max_tokens"] == 1
    assert request_data["payload"]["temperature"] == 0
    assert "max_completion_tokens" not in request_data["payload"]


def test_perform_raw_model_connection_probe_uses_requests_post(monkeypatch) -> None:
    """The Streamlit health check should stay on raw HTTP requests and never invoke the model."""
    captured: Dict[str, Any] = {}

    class FakeLLM:
        base_url = "http://localhost:1234/v1"
        api_key = "lm-studio"
        model = "gpt-4.1-mini"

        def invoke(self, *_args, **_kwargs):
            raise AssertionError("Health checks must not call invoke().")

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(model_connection_probe.requests, "post", fake_post)

    result = perform_raw_model_connection_probe(
        test_llm=FakeLLM(),
        provider="lmstudio",
        model_display="gpt-4.1-mini",
    )

    assert captured["url"] == "http://localhost:1234/v1/chat/completions"
    assert captured["timeout"] == 15
    assert captured["json"]["messages"] == [{"role": "user", "content": "hi"}]
    assert result["model_id"] == "gpt-4.1-mini"
