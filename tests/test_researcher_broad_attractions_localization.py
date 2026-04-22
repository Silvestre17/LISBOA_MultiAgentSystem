# ===========================================================================
# Master Thesis - Researcher Broad Attractions Localization Tests
#   - André Filipe Gomes Silvestre, 20240502
#
# Focused regressions for the deterministic broad-attractions lookup path.
# Ensures Portuguese attraction-list queries rewrite grounded English-only
# VisitLisboa place data before the final source footer is appended.
# ===========================================================================

from __future__ import annotations

from agent.agents.researcher_agent import ResearcherAgent


RAW_BROAD_ATTRACTIONS_RESULT = """1. 🏛️ **Fronteira Palace**
   📂 Category: Monuments
   Discover one of the great Lisbon palaces and its significant art collection.
"""


def _build_minimal_researcher_agent() -> ResearcherAgent:
    """Build a minimal researcher-agent instance for deterministic lookup tests."""
    agent = ResearcherAgent.__new__(ResearcherAgent)
    agent._get_tool_by_name = lambda _name: object()
    agent._invoke_tool = lambda _tool, _args, tool_name=None: RAW_BROAD_ATTRACTIONS_RESULT
    agent._extract_service_types = lambda _user_message: []
    agent._extract_near_location_name = lambda _user_message: None
    agent._remember_search_context = lambda **_kwargs: None
    agent._build_places_source_line = lambda _result, language: f"SOURCE-{language}"
    agent._count_ranked_results = lambda _result: 1
    agent._has_specific_lookup_fallback_intro = lambda _result: False
    agent._infer_place_category_hint = lambda _user_message: None
    return agent


def test_broad_attractions_direct_lookup_rewrites_portuguese_output() -> None:
    """Portuguese broad-attractions queries should rewrite English-only tool output before returning it."""
    agent = _build_minimal_researcher_agent()
    rewrite_calls: list[tuple[str, str, str]] = []

    def fake_rewrite(raw_result: str, user_message: str, language: str) -> str:
        rewrite_calls.append((raw_result, user_message, language))
        return "### Atrações imperdíveis em Lisboa\n\n- Conteúdo final em PT-PT"

    agent._rewrite_broad_attractions_result = fake_rewrite

    response = agent._run_direct_place_lookup(
        "Lista as atrações imperdíveis para quem visita Lisboa pela primeira vez.",
        "pt",
    )

    assert response.startswith("### Atrações imperdíveis em Lisboa")
    assert "Conteúdo final em PT-PT" in response
    assert response.endswith("SOURCE-pt")
    assert rewrite_calls == [
        (
            RAW_BROAD_ATTRACTIONS_RESULT.strip(),
            "Lista as atrações imperdíveis para quem visita Lisboa pela primeira vez.",
            "pt",
        )
    ]


def test_broad_attractions_direct_lookup_keeps_english_output_without_rewrite() -> None:
    """English broad-attractions queries should keep the raw deterministic output path."""
    agent = _build_minimal_researcher_agent()
    rewrite_calls: list[tuple[str, str, str]] = []

    def fake_rewrite(raw_result: str, user_message: str, language: str) -> str:
        rewrite_calls.append((raw_result, user_message, language))
        return "SHOULD NOT BE USED"

    agent._rewrite_broad_attractions_result = fake_rewrite

    response = agent._run_direct_place_lookup(
        "Show me the best attractions for a first time visit to Lisbon.",
        "en",
    )

    assert "Discover one of the great Lisbon palaces" in response
    assert response.endswith("SOURCE-en")
    assert rewrite_calls == []
