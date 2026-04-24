# ==========================================================================
# Master Thesis - Natural Query Routing Regressions
#   - André Filipe Gomes Silvestre, 20240502
#
#   Focused regressions for natural user phrasing introduced in the shared
#   evaluation corpus. These tests only cover deterministic routing helpers.
# ==========================================================================

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.agents.researcher_agent import ResearcherAgent
from agent.agents.transport_agent import (
    _build_carris_metropolitana_tool_spec,
    _build_carris_urban_tool_spec,
    _build_cp_tool_spec,
)


def test_carris_urban_stop_search_accepts_natural_wording() -> None:
    """Urban stop lookup should recognize a plain user phrasing near a landmark."""
    spec = _build_carris_urban_tool_spec("What bus stops are near Rato?")

    assert spec == {
        "name": "carris_get_stops",
        "args": {"query": "Rato"},
    }


def test_carris_metropolitana_line_search_accepts_generic_bus_wording() -> None:
    """Suburban line lookup should not require the operator name in the query."""
    spec = _build_carris_metropolitana_tool_spec("Which bus lines serve Cacilhas?")

    assert spec == {
        "name": "search_carris_metropolitana_lines",
        "args": {"query": "Cacilhas"},
    }


def test_carris_metropolitana_nearby_buses_accept_current_location_wording() -> None:
    """Nearby-bus queries should accept 'I'm in ... right now' phrasing."""
    spec = _build_carris_metropolitana_tool_spec(
        "I'm in Almada right now. Are there any buses nearby?"
    )

    assert spec == {
        "name": "get_real_time_bus_positions",
        "args": {"location": "Almada", "radius_km": 1.0},
    }


def test_carris_metropolitana_realtime_line_accepts_bare_line_number() -> None:
    """Realtime suburban-line queries should work even without the literal word 'line'."""
    spec = _build_carris_metropolitana_tool_spec(
        "I'm waiting for the 1507 in Almada. Where is it right now?"
    )

    assert spec == {
        "name": "get_bus_realtime_locations",
        "args": {"line_id": "1507"},
    }


def test_cp_station_search_accepts_closest_station_wording() -> None:
    """Train-station lookup should support 'closest to' phrasing."""
    spec = _build_cp_tool_spec("Which train stations are closest to Rossio?")

    assert spec == {
        "name": "search_cp_stations",
        "args": {"query": "Rossio"},
    }


def test_researcher_service_categories_accept_generic_help_query() -> None:
    """Service-category routing should not depend on Lisboa Aberta wording."""
    tool_call = ResearcherAgent._build_deterministic_subgraph_tool_call(
        "I need to sort out some paperwork in Lisbon. What kinds of public services can you help me find?"
    )

    assert tool_call is not None
    assert tool_call.tool_calls[0]["name"] == "list_service_categories"


def test_researcher_event_categories_accept_generic_browsing_query() -> None:
    """Event-category routing should accept natural browsing language."""
    tool_call = ResearcherAgent._build_deterministic_subgraph_tool_call(
        "I'm not sure what to do this week. What kinds of events can I look for in Lisbon?"
    )

    assert tool_call is not None
    assert tool_call.tool_calls[0]["name"] == "get_event_categories"


def test_researcher_place_categories_accept_generic_exploration_query() -> None:
    """Place-category routing should accept natural itinerary-planning language."""
    tool_call = ResearcherAgent._build_deterministic_subgraph_tool_call(
        "I'm planning my itinerary and want ideas. What kinds of places can I explore in Lisbon?"
    )

    assert tool_call is not None
    assert tool_call.tool_calls[0]["name"] == "get_place_categories"
