# ==========================================================================
# Master Thesis
#   - André Filipe Gomes Silvestre, 20240502
#
# Raw model connection probe helpers.
#
# These helpers validate provider/model readiness using direct HTTP requests
# instead of LangChain/LangGraph invocation paths. This is intentional so the
# UI connection checks do not create LangSmith traces or consume the tracing
# quota reserved for real user requests.
# ==========================================================================

# Required libraries:
# pip install requests

from __future__ import annotations

from typing import Any, Dict, Optional

import requests

from agent.llm_factory import LLMFactory


def build_connection_probe_request(test_llm: Any, model_display: str) -> Dict[str, Any]:
    """Build the raw HTTP request payload used by the model readiness check.

    Args:
        test_llm: LangChain chat model instance used by the app.
        model_display: Human-readable model label used as a fallback.

    Returns:
        Dict[str, Any]: Endpoint, headers, payload, and resolved model id.
    """
    raw_base = getattr(
        test_llm,
        "openai_api_base",
        getattr(test_llm, "base_url", "https://api.openai.com/v1/"),
    )
    base_url = str(raw_base) if not isinstance(raw_base, str) else raw_base

    api_key_obj = getattr(
        test_llm,
        "openai_api_key",
        getattr(test_llm, "api_key", getattr(test_llm, "_api_key", "")),
    )
    api_key = (
        api_key_obj.get_secret_value()
        if hasattr(api_key_obj, "get_secret_value")
        else str(api_key_obj)
    )
    model_id = getattr(test_llm, "model_name", getattr(test_llm, "model", model_display))

    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "None":
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "hi"}],
    }
    if LLMFactory._is_reasoning_model(model_id):
        payload["max_completion_tokens"] = 100
    else:
        payload["max_tokens"] = 1
        payload["temperature"] = 0

    return {
        "endpoint": f"{base_url.rstrip('/')}/chat/completions",
        "headers": headers,
        "payload": payload,
        "model_id": model_id,
    }


def perform_raw_model_connection_probe(
    test_llm: Any,
    provider: str,
    model_display: str,
    timeout_by_provider: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Validate model readiness using a raw HTTP request outside tracing paths.

    Args:
        test_llm: LangChain chat model instance used by the app.
        provider: Provider identifier (for timeout selection).
        model_display: Human-readable model label used as a fallback.
        timeout_by_provider: Optional timeout overrides by provider.

    Returns:
        Dict[str, Any]: Probe metadata, including the resolved model id.

    Raises:
        requests.RequestException: If the HTTP request fails.
        RuntimeError: If the provider responds without a completion payload.
    """
    request_data = build_connection_probe_request(test_llm, model_display)
    timeout_lookup = timeout_by_provider or {}
    timeout = timeout_lookup.get(provider, 15 if provider == "lmstudio" else 10)

    response = requests.post(
        request_data["endpoint"],
        headers=request_data["headers"],
        json=request_data["payload"],
        timeout=timeout,
    )
    response.raise_for_status()

    response_json = response.json()
    if not response_json.get("choices"):
        raise RuntimeError("The server responded without a completion.")

    return {
        "endpoint": request_data["endpoint"],
        "model_id": request_data["model_id"],
        "response": response_json,
    }
