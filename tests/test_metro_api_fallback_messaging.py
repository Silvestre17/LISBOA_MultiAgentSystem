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
from unittest.mock import patch

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
