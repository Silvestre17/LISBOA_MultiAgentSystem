# ==========================================================================
# Master Thesis
#   - André Filipe Gomes Silvestre, 20240502
#
# Lean live smoke tests for Lisbon transport and weather integrations.
#
# Run from the repository root with a relative path:
#   python -m pytest tests/test_lisbon_transport.py -q --run-live -m live
#
# These tests intentionally stay short and provider-focused. They are meant to
# confirm that the main external integrations still respond sensibly, not to
# exhaust every edge case of every transport API.
# ==========================================================================

# Required libraries:
# pip install pytest langchain-core

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.carris_api import carris_get_stops
from tools.carrismetropolitana_api import search_carris_metropolitana_lines
from tools.cp_api import search_cp_stations
from tools.ipma_api import get_weather_forecast
from tools.metrolisboa_api import get_metro_status
from tools.transport_api import get_route_between_stations

pytestmark = pytest.mark.live


def _contains_any(text: str, markers: list[str]) -> bool:
    """Return whether at least one marker appears in the response text."""
    lowered = (text or "").lower()
    return any(marker.lower() in lowered for marker in markers)


def test_metro_status_live_smoke() -> None:
    """Metro status should return a recognizable user-facing service summary."""
    response = str(get_metro_status.invoke({}))

    assert response.strip()
    assert _contains_any(
        response,
        ["Metro de Lisboa", "Yellow Line", "Blue Line", "normal service", "serviço normal"],
    )


def test_carris_urban_stop_lookup_live_smoke() -> None:
    """Carris Urban should still resolve a central Lisbon stop query."""
    response = str(carris_get_stops.invoke({"query": "Rossio", "limit": 5}))

    assert response.strip()
    assert _contains_any(response, ["Rossio", "stop", "paragem"])


def test_carris_metropolitana_line_search_live_smoke() -> None:
    """Carris Metropolitana should still return suburban/intermunicipal lines for Sintra."""
    response = str(search_carris_metropolitana_lines.invoke({"query": "Sintra"}))

    assert response.strip()
    assert _contains_any(response, ["Sintra", "linha", "line"])


def test_cp_station_search_live_smoke() -> None:
    """CP station search should still find Lisboa-Oriente cleanly."""
    response = str(search_cp_stations.invoke({"query": "Oriente"}))

    assert response.strip()
    assert _contains_any(response, ["Oriente", "Lisboa", "station", "estação"])


def test_ipma_forecast_live_smoke() -> None:
    """IPMA forecast should still return a Lisbon forecast summary."""
    response = str(get_weather_forecast.invoke({"days": 3}))

    assert response.strip()
    assert _contains_any(response, ["Forecast", "forecast", "Lisbon", "Lisboa"])


def test_integrated_route_live_smoke() -> None:
    """Integrated routing should still produce a usable route between two common points."""
    response = str(
        get_route_between_stations.invoke(
            {"origin": "Saldanha", "destination": "Oriente"}
        )
    )

    assert response.strip()
    assert "Saldanha" in response
    assert "Oriente" in response
    assert _contains_any(response, ["Route", "Rota", "Metro", "Linha", "line"])
