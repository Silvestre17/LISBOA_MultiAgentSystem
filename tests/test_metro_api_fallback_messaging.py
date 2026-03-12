# ========================================================================== 
# Master Thesis
#   - André Filipe Gomes Silvestre, 20240502
# 
# Regression tests for user-facing Metro API fallback and outage messaging.
#
# Run from the repository root with a relative path:
#   python -m pytest tests/test_metro_api_fallback_messaging.py -q
# Useful parameters:
#   -vv         verbose mode
#   -x          stop on first failure
#   --tb=short  shorter tracebacks
# Notes:
#   - Prefer relative paths in this workspace. Absolute pytest paths may be
#     treated as glob patterns on Windows because the folder name includes
#     `[` and `]`.
# ==========================================================================

# Required libraries:
# pip install pytest

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools import metrolisboa_api  # noqa: E402


def test_get_metro_status_fallback_mentions_official_api_unavailable() -> None:
    """Fallback line status should explicitly say the official Metro API is unavailable."""
    fallback_payload = {
        "resposta": {
            "amarela": " Ok",
            "azul": " Ok",
            "verde": " Ok",
            "vermelha": " Ok",
        }
    }

    with patch.object(metrolisboa_api, "METRO_CONSUMER_KEY", "dummy-key"), patch.object(
        metrolisboa_api,
        "METRO_CONSUMER_SECRET",
        "dummy-secret",
    ), patch.dict(
        metrolisboa_api._metro_runtime_state,
        {
            "token_status": "timeout",
            "token_error": "Read timed out",
            "request_status": "token_unavailable",
            "request_error": "Read timed out",
        },
        clear=False,
    ), patch.object(
        metrolisboa_api,
        "_metro_api_request",
        return_value=None,
    ), patch.object(
        metrolisboa_api,
        "fetch_json_with_retry",
        return_value=fallback_payload,
    ):
        result = metrolisboa_api.get_metro_status.invoke({})

    assert "official metro de lisboa real-time api is currently unavailable or timing out" in result.lower()
    assert "public fallback endpoint" in result.lower()
    assert "all lines operating normally" in result.lower()


def test_get_metro_wait_time_timeout_message_is_explicit() -> None:
    """Realtime wait-time failures should explain that the official Metro API is not responding."""
    with patch.object(metrolisboa_api, "METRO_CONSUMER_KEY", "dummy-key"), patch.object(
        metrolisboa_api,
        "METRO_CONSUMER_SECRET",
        "dummy-secret",
    ), patch.dict(
        metrolisboa_api._metro_runtime_state,
        {
            "token_status": "timeout",
            "token_error": "Read timed out",
            "request_status": "token_unavailable",
            "request_error": "Read timed out",
        },
        clear=False,
    ), patch.object(
        metrolisboa_api,
        "get_station_id",
        return_value="CG",
    ), patch.object(
        metrolisboa_api,
        "_metro_api_request",
        return_value=None,
    ):
        result = metrolisboa_api.get_metro_wait_time.invoke({"station": "Campo Grande"})

    assert "official metro de lisboa api is not responding right now" in result.lower()
    assert "fallback endpoint still provides line status" in result.lower()
    assert "campo grande" in result.lower()


def test_get_metro_access_token_retries_with_dynamic_ca_bundle_after_certificate_failure() -> None:
    """Metro token acquisition should prefer a dynamic CA bundle before any insecure fallback."""
    success_response = MagicMock()
    success_response.status_code = 200
    success_response.json.return_value = {
        "access_token": "token-123",
        "expires_in": 3600,
    }

    verify_calls = []
    dynamic_bundle = r"C:\temp\metro_dynamic_bundle.pem"

    def fake_request(method, url, verify=None, **kwargs):
        verify_calls.append(verify)
        if len(verify_calls) == 1:
            raise requests.exceptions.SSLError("certificate verify failed")
        return success_response

    with patch.object(metrolisboa_api, "METRO_CONSUMER_KEY", "dummy-key"), patch.object(
        metrolisboa_api,
        "METRO_CONSUMER_SECRET",
        "dummy-secret",
    ), patch.object(metrolisboa_api, "_metro_access_token", None), patch.object(
        metrolisboa_api,
        "_metro_token_expiry",
        None,
    ), patch.object(
        metrolisboa_api,
        "METRO_SSL_VERIFY",
        True,
    ), patch.object(
        metrolisboa_api,
        "_build_runtime_metro_ca_bundle",
        return_value=dynamic_bundle,
    ), patch.object(
        metrolisboa_api,
        "_METRO_SSL_ALLOW_INSECURE_FALLBACK",
        False,
    ), patch.object(metrolisboa_api.requests, "request", side_effect=fake_request):
        token = metrolisboa_api._get_metro_access_token(force_refresh=True)

    assert token == "token-123"
    assert verify_calls == [True, dynamic_bundle]
    assert metrolisboa_api._metro_runtime_state["token_status"] == "ok"


def test_metro_request_only_retries_insecurely_when_explicitly_allowed() -> None:
    """Insecure Metro retries should happen only when explicitly enabled by configuration."""
    success_response = MagicMock()
    success_response.status_code = 200
    success_response.json.return_value = {"ok": True}

    verify_calls = []

    def fake_request(method, url, verify=None, **kwargs):
        verify_calls.append(verify)
        if len(verify_calls) == 1:
            raise requests.exceptions.SSLError("certificate verify failed")
        return success_response

    with patch.object(metrolisboa_api, "METRO_SSL_VERIFY", True), patch.object(
        metrolisboa_api,
        "_build_runtime_metro_ca_bundle",
        return_value=None,
    ), patch.object(
        metrolisboa_api,
        "_METRO_SSL_ALLOW_INSECURE_FALLBACK",
        True,
    ), patch.object(metrolisboa_api.requests, "request", side_effect=fake_request):
        response = metrolisboa_api._metro_request(
            "get",
            "https://api.metrolisboa.pt:8243/estadoServicoML/1.0.1/test",
        )

    assert response is success_response
    assert verify_calls == [True, False]
