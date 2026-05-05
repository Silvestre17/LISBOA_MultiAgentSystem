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
from agent.agents.supervisor import SupervisorAgent
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


def test_carris_metropolitana_alert_queries_preserve_area_filter() -> None:
    """Area-scoped service-alert questions should pass the municipality into the alerts tool."""
    spec = _build_carris_metropolitana_tool_spec(
        "Any bus service alerts around Almada today?"
    )

    assert spec == {
        "name": "get_carris_metropolitana_alerts",
        "args": {"area": "Almada"},
    }


def test_cp_station_search_accepts_closest_station_wording() -> None:
    """Train-station lookup should support 'closest to' phrasing."""
    spec = _build_cp_tool_spec("Which train stations are closest to Rossio?")

    assert spec == {
        "name": "search_cp_stations",
        "args": {"query": "Rossio"},
    }


def test_cp_trip_from_sete_rios_to_oriente_uses_cp_train_planner() -> None:
    """Explicit train trips between Lisbon rail stations should use CP, not Metro routing."""
    spec = _build_cp_tool_spec("How do I get by train from Sete Rios to Oriente?")

    assert spec == {
        "name": "plan_train_trip",
        "args": {"origin": "Sete Rios", "destination": "Oriente"},
    }


def test_supervisor_geographic_out_of_scope_route_stays_direct() -> None:
    """Non-Lisbon intercity route requests should not be sent to transport workers."""
    supervisor = SupervisorAgent.__new__(SupervisorAgent)
    decision = SupervisorAgent._direct_routing_override(
        supervisor,
        "Como posso ir de Madrid a Barcelona?",
        "pt",
    )

    assert decision is not None
    assert decision["agents"] == []
    assert decision["direct_response"]


def test_supervisor_lisbon_joke_is_not_obvious_out_of_scope() -> None:
    """A light Lisbon-themed joke is Lisbon-context UX, not a generic trivia failure."""
    assert SupervisorAgent._is_obvious_out_of_scope("Tell me a joke about Lisbon.") is False


def test_supervisor_booking_request_gets_direct_unsupported_capability_reply() -> None:
    """The supervisor should refuse real booking actions without routing to researcher."""
    supervisor = SupervisorAgent.__new__(SupervisorAgent)
    decision = SupervisorAgent._direct_routing_override(
        supervisor,
        "Can you book a table at Ramiro tonight?",
        "en",
    )

    assert decision is not None
    assert decision["agents"] == []
    assert "can't make bookings" in decision["direct_response"]


def test_supervisor_weather_aware_event_query_does_not_route_transport() -> None:
    """Weather-dependent event discovery should use weather and researcher without transport noise."""
    decision = SupervisorAgent._single_domain_override(
        "What outdoor events are happening tomorrow in Lisbon? Should I bring an umbrella?"
    )

    assert decision is not None
    assert decision["agents"] == ["weather", "researcher"]


def test_supervisor_sailing_safety_query_stays_weather_only() -> None:
    """Weather-dependent sailing safety prompts should not trigger event search or planning."""
    decision = SupervisorAgent._single_domain_override(
        "Is it safe to go sailing in Lisbon tomorrow?"
    )

    assert decision is not None
    assert decision["agents"] == ["weather"]


def test_supervisor_current_wind_query_stays_weather_only() -> None:
    """Wind-only weather wording should not be routed to local-knowledge retrieval."""
    decision = SupervisorAgent._single_domain_override("Como está o vento hoje?")

    assert decision is not None
    assert decision["agents"] == ["weather"]


def test_supervisor_metro_wait_query_stays_transport_only() -> None:
    """Metro wait-time wording with Cais do Sodré should not be fuzzily treated as local places."""
    decision = SupervisorAgent._single_domain_override(
        "How long do I have to wait for the green line at Cais do Sodre?"
    )

    assert decision is not None
    assert decision["agents"] == ["transport"]


def test_supervisor_portuguese_booking_request_gets_direct_unsupported_capability_reply() -> None:
    """Portuguese reservation verbs should trigger the same unsupported-action guard."""
    supervisor = SupervisorAgent.__new__(SupervisorAgent)
    decision = SupervisorAgent._direct_routing_override(
        supervisor,
        "Consegues reservar mesa no Ramiro hoje?",
        "pt",
    )

    assert decision is not None
    assert decision["agents"] == []
    assert "Não consigo fazer reservas" in decision["direct_response"]


def test_researcher_service_categories_accept_generic_help_query() -> None:
    """Service-category routing should not depend on Lisboa Aberta wording."""
    tool_call = ResearcherAgent._build_deterministic_subgraph_tool_call(
        "I need to sort out some paperwork in Lisbon. What kinds of public services can you help me find?"
    )

    assert tool_call is not None
    assert tool_call.tool_calls[0]["name"] == "list_service_categories"


def test_researcher_service_queries_use_lisboa_aberta_in_subgraph() -> None:
    """Deterministic researcher subgraph routing must not send services to VisitLisboa."""
    tool_call = ResearcherAgent._build_deterministic_subgraph_tool_call(
        "Hospital mais próximo do Marquês de Pombal"
    )

    assert tool_call is not None
    assert tool_call.tool_calls[0]["name"] == "find_nearby_services"
    assert tool_call.tool_calls[0]["args"]["service_type"] == "hospitais"
    assert tool_call.tool_calls[0]["args"]["near_location_name"] == "Marquês de Pombal"


def test_researcher_service_queries_extract_feminine_nearest_reference() -> None:
    """Portuguese feminine proximity phrasing should preserve the reference landmark."""
    tool_call = ResearcherAgent._build_deterministic_subgraph_tool_call(
        "Onde fica a farmácia mais próxima do Rossio?"
    )

    assert tool_call is not None
    assert tool_call.tool_calls[0]["name"] == "find_nearby_services"
    assert tool_call.tool_calls[0]["args"]["service_type"] == "farmácias"
    assert tool_call.tool_calls[0]["args"]["near_location_name"] == "Rossio"


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


def test_researcher_history_query_uses_history_or_knowledge_tool() -> None:
    """Named Lisbon history queries should not fall through to events or generic attraction search."""
    tool_call = ResearcherAgent._build_deterministic_subgraph_tool_call(
        "Tell me about the history of Castelo de SÃ£o Jorge."
    )

    assert tool_call is not None
    assert tool_call.tool_calls[0]["name"] == "search_history_culture"
    assert "Castelo de SÃ£o Jorge" in tool_call.tool_calls[0]["args"]["query"]


def test_researcher_history_query_is_not_event_lookup() -> None:
    """History phrasing with named-place wording should not be treated as an event lookup."""
    assert ResearcherAgent._is_direct_event_lookup_query(
        "Tell me about the history of Castelo de São Jorge."
    ) is False


def test_researcher_lisboa_card_named_attraction_uses_knowledge_base() -> None:
    """Lisboa Card inclusion queries should search curated knowledge, not event/place categories."""
    tool_call = ResearcherAgent._build_deterministic_subgraph_tool_call(
        "Is the OceanÃ¡rio included in the Lisboa Card?"
    )

    assert tool_call is not None
    assert tool_call.tool_calls[0]["name"] == "search_lisbon_knowledge"
