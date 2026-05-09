# ==========================================================================
# Master Thesis
#   - Andre Filipe Gomes Silvestre, 20240502
#
# Regression tests for deterministic routes that must stay aligned with the
# evaluation ground-truth tool expectations.
# ==========================================================================

from agent.agents.researcher_agent import ResearcherAgent
from agent.agents.weather_agent import WeatherAgent


def _tool_names(message: object) -> list[str]:
    """Return tool-call names from a deterministic agent message."""
    tool_calls = getattr(message, "tool_calls", []) or []
    return [str(tool_call["name"]) for tool_call in tool_calls]


def test_weather_sailing_safety_subgraph_uses_forecast_and_warnings() -> None:
    """Sailing safety needs both tomorrow's forecast and active warning context."""
    message = WeatherAgent._build_deterministic_subgraph_tool_call(
        "Is it safe to go sailing in Lisbon tomorrow?"
    )

    assert _tool_names(message) == ["get_weather_warnings", "get_weather_forecast"]


def test_researcher_subgraph_routes_event_category_queries_before_discovery() -> None:
    """Event category browsing should use the category tool, not generic place search."""
    message = ResearcherAgent._build_deterministic_subgraph_tool_call(
        "I'm not sure what to do this week. What kinds of events can I look for in Lisbon?"
    )

    assert _tool_names(message) == ["get_event_categories"]


def test_researcher_subgraph_routes_place_category_queries_before_discovery() -> None:
    """Place category browsing should use the category tool, not generic place search."""
    message = ResearcherAgent._build_deterministic_subgraph_tool_call(
        "I'm planning my itinerary and want ideas. What kinds of places can I explore in Lisbon?"
    )

    assert _tool_names(message) == ["get_place_categories"]
