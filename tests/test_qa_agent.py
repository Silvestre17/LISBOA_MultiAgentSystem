# ==========================================================================
# Master Thesis - QA Agent Unit Tests
#   - André Filipe Gomes Silvestre, 20240502
#
#   Unit tests for QualityAssuranceAgent covering:
#     - Deterministic fact-checking (_verify_facts)
#     - Prompt generation (get_qa_prompt)
#     - JSON parsing resilience
#     - User context and conversation history injection
#
#   Run from the repository root with a relative path:
#     python -m pytest tests/test_qa_agent.py -q
#   Useful parameters:
#     -vv         verbose mode
#     -k prompt   focus on prompt-generation checks
#     -x          stop on first failure
#     --tb=short  shorter tracebacks
#   Notes:
#     - Prefer relative paths in this workspace. Absolute pytest paths may be
#       treated as glob patterns on Windows because the folder name includes
#       `[` and `]`.
# ==========================================================================

import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------
# 1. Deterministic fact-check tests (no LLM needed)
# ---------------------------------------------------------------

class TestVerifyFacts:
    """Tests for QualityAssuranceAgent._verify_facts (deterministic)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Import module-level constants and create an uninitialised QA agent."""
        from agent.agents.qa_agent import (
            _AML_BOUNDS,
            _IPMA_FORECAST_DAYS,
            _METRO_CANONICAL_STATIONS,
            _VALID_DOMAINS,
            QualityAssuranceAgent,
        )
        self.QA = QualityAssuranceAgent
        self.STATIONS = _METRO_CANONICAL_STATIONS
        self.BOUNDS = _AML_BOUNDS
        self.DOMAINS = _VALID_DOMAINS
        self.FORECAST_DAYS = _IPMA_FORECAST_DAYS

        # Create an agent instance without calling __init__ (no LLM needed)
        self.agent = object.__new__(self.QA)

    # --- Metro station checks ---

    def test_valid_metro_station_no_warning(self):
        """Known station names should not trigger warnings."""
        text = "Take metro to estação de Alameda then transfer at Saldanha."
        result = self.agent._verify_facts(text, "test query", None)
        metro_warns = [w for w in result["disclaimers"] if "station" in w.lower() or "verified" in w.lower()]
        assert len(metro_warns) == 0, f"Unexpected metro warnings: {metro_warns}"

    def test_invalid_metro_station_flagged(self):
        """Invented station name should generate a disclaimer."""
        text = "Go to estação de Benfica on the metro Azul line."
        result = self.agent._verify_facts(text, "test query", None)
        assert any("Benfica" in w or "benfica" in w for w in result["disclaimers"]), (
            f"Expected Benfica warning in {result['disclaimers']}"
        )

    def test_valid_metro_line_no_warning(self):
        """Known line colours should not trigger warnings."""
        text = "Linha Verde is running normally."
        result = self.agent._verify_facts(text, "test query", None)
        line_warns = [w for w in result["disclaimers"] if "line" in w.lower() or "linha" in w.lower()]
        assert len(line_warns) == 0

    # --- AML coordinate checks ---

    def test_valid_coordinates_no_warning(self):
        """Coordinates inside AML bounding box should not warn."""
        text = "Located at 38.72, -9.14 in central Lisbon"
        result = self.agent._verify_facts(text, "test query", None)
        coord_warns = [w for w in result["disclaimers"] if "outside" in w.lower()]
        assert len(coord_warns) == 0

    def test_coordinates_outside_aml_flagged(self):
        """Coordinates far outside AML should be flagged."""
        text = "Located at 41.15, -8.61 near Porto"  # Porto
        result = self.agent._verify_facts(text, "test query", None)
        assert any("outside" in w.lower() for w in result["disclaimers"]), (
            f"Expected AML warning in {result['disclaimers']}"
        )

    def test_coordinates_latlon_text_format(self):
        """Coordinates in 'latitude X, longitude Y' format should also be checked."""
        text = "The location is at latitude 41.15, longitude -8.61"
        result = self.agent._verify_facts(text, "test query", None)
        assert any("outside" in w.lower() for w in result["disclaimers"]), (
            f"Expected AML warning for lat/lon text format in {result['disclaimers']}"
        )

    # --- Date sanity checks ---

    def test_forecast_within_range_no_warning(self):
        """Dates within IPMA's 5-day range should not warn."""
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        text = f"Weather forecast for {tomorrow}: sunny, 25C."
        result = self.agent._verify_facts(text, "test query", None)
        date_warns = [w for w in result["disclaimers"] if "beyond" in w.lower() or "forecast range" in w.lower()]
        assert len(date_warns) == 0

    def test_forecast_beyond_range_flagged(self):
        """Dates beyond 5-day IPMA range should be flagged."""
        far_date = (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")
        text = f"Weather forecast for {far_date}: rain expected."
        result = self.agent._verify_facts(text, "test query", None)
        assert any("forecast range" in w.lower() or "beyond" in w.lower() for w in result["disclaimers"]), (
            f"Expected IPMA range warning in {result['disclaimers']}"
        )

    # --- URL validation ---

    def test_valid_url_no_warning(self):
        """URLs from known domains should not warn."""
        text = "More info at https://www.visitlisboa.com/events/some-event"
        result = self.agent._verify_facts(text, "test query", None)
        url_warns = [w for w in result["disclaimers"] if "unverified" in w.lower()]
        assert len(url_warns) == 0

    def test_unknown_url_flagged(self):
        """URLs from unknown domains should be flagged."""
        text = "Book at https://www.fake-lisbon-tours.com/book"
        result = self.agent._verify_facts(text, "test query", None)
        assert any("unverified" in w.lower() or "fake-lisbon" in w.lower() for w in result["disclaimers"]), (
            f"Expected domain warning in {result['disclaimers']}"
        )

    # --- User preferences check ---

    def test_mobility_wheelchair_disclaimer(self):
        """When user has wheelchair mobility and transport info has no accessibility, flag it."""
        text = "Take transport from Alameda to Oriente via metro line vermelha"
        result = self.agent._verify_facts(
            text, "plan accessible route",
            user_context={"mobility": "wheelchair"},
        )
        access_warns = [w for w in result["disclaimers"] if "accessibility" in w.lower()]
        assert len(access_warns) > 0, f"Expected accessibility disclaimer in {result['disclaimers']}"

    def test_all_checks_performed(self):
        """All 9 deterministic checks should run for any input."""
        text = "Some generic text about Lisbon."
        result = self.agent._verify_facts(text, "test query", None)
        core_checks = {
            "metro_stations", "metro_line_station_pairs", "cp_lines",
            "aml_coordinates", "date_sanity", "url_validation",
            "user_preferences", "temperature_sanity", "dynamic_data_disclaimers",
        }
        performed = set(result["checks_performed"])
        assert core_checks.issubset(performed), (
            f"Missing checks: {core_checks - performed}"
        )

    def test_result_structure(self):
        """Verify _verify_facts returns the correct dict structure."""
        result = self.agent._verify_facts("test text", "test query", None)
        assert "valid" in result
        assert "disclaimers" in result
        assert "critical_issues" in result
        assert "checks_performed" in result
        assert isinstance(result["valid"], bool)
        assert isinstance(result["disclaimers"], list)


# ---------------------------------------------------------------
# 2. Prompt generation tests (no LLM needed)
# ---------------------------------------------------------------

class TestQAPrompt:
    """Tests for get_qa_prompt() function."""

    def test_en_prompt_has_required_sections(self):
        """EN prompt should contain all new enhanced sections."""
        from agent.prompts.qa import get_qa_prompt
        prompt = get_qa_prompt("en")
        for section in [
            "USER CONTEXT VALIDATION",
            "FOLLOW-UP COHERENCE",
            "ANTI-HALLUCINATION CHECK",
            "Fabricated URLs",
            "Accessibility concern",
        ]:
            assert section in prompt, f"EN prompt missing section: {section}"

    def test_pt_prompt_has_required_sections(self):
        """PT prompt should contain all new enhanced sections."""
        from agent.prompts.qa import get_qa_prompt
        prompt = get_qa_prompt("pt")
        for section in [
            "VALIDAÇÃO DO CONTEXTO",
            "COERÊNCIA COM FOLLOW-UPS",
            "URLs fabricados",
            "acessibilidade",
        ]:
            assert section in prompt, f"PT prompt missing section: {section}"

    def test_user_context_injection(self):
        """User context dict should be rendered into the prompt."""
        from agent.prompts.qa import get_qa_prompt
        prompt = get_qa_prompt(
            "en",
            user_context={"mobility": "wheelchair", "available_time": 120},
        )
        assert "wheelchair" in prompt
        assert "120" in prompt

    def test_conversation_history_injection(self):
        """Conversation history should appear in the prompt."""
        from agent.prompts.qa import get_qa_prompt
        prompt = get_qa_prompt(
            "en",
            conversation_history=["User: What's the weather?", "Assistant: Sunny, 24C."],
        )
        assert "What's the weather?" in prompt
        assert "Sunny" in prompt

    def test_prompt_without_context_still_works(self):
        """Prompt should render cleanly even without optional context."""
        from agent.prompts.qa import get_qa_prompt
        prompt = get_qa_prompt("en")
        # Should not contain raw format placeholders
        assert "{user_context_section}" not in prompt
        assert "{conversation_history_section}" not in prompt

    def test_en_prompt_has_date_and_time(self):
        """Prompt should contain formatted date and time."""
        from agent.prompts.qa import get_qa_prompt
        prompt = get_qa_prompt("en")
        today_str = datetime.now().strftime("%B %d, %Y")
        assert today_str in prompt, f"Expected date '{today_str}' in prompt"

    def test_pt_prompt_sufficient_length(self):
        """PT prompt should have substantial content."""
        from agent.prompts.qa import get_qa_prompt
        prompt = get_qa_prompt("pt")
        assert len(prompt) > 2000, f"PT prompt too short: {len(prompt)} chars"


# ---------------------------------------------------------------
# 3. Integration: planner_agent.synthesize with _qa_disclaimers
# ---------------------------------------------------------------

class TestPlannerDisclaimers:
    """Tests that PlannerAgent.synthesize extracts and passes QA disclaimers."""

    def test_synthesize_extracts_qa_disclaimers(self):
        """synthesize() should extract _qa_disclaimers from agent_outputs."""
        from agent.agents.planner_agent import PlannerAgent

        # Patch __init__ to avoid LLM initialisation
        with patch.object(PlannerAgent, "__init__", return_value=None):
            agent = PlannerAgent.__new__(PlannerAgent)
            agent.system_prompt = "test"
            agent.tools = []
            agent.model_info = "test"

            # Mock invoke to capture the qa_disclaimers arg
            captured = {}

            def fake_invoke(**kwargs):
                captured.update(kwargs)
                return "fake response"

            agent.invoke = lambda **kw: fake_invoke(**kw)
            agent._safe_llm_invoke = MagicMock()

            outputs = {
                "weather": "Sunny 24C",
                "researcher": "Belem Tower",
                "_qa_disclaimers": ["Opening hours not verified"],
            }
            agent.synthesize("Plan my day", outputs)

            assert "qa_disclaimers" in captured
            assert captured["qa_disclaimers"] == ["Opening hours not verified"]

    def test_synthesize_handles_no_disclaimers(self):
        """synthesize() should work fine without _qa_disclaimers."""
        from agent.agents.planner_agent import PlannerAgent

        with patch.object(PlannerAgent, "__init__", return_value=None):
            agent = PlannerAgent.__new__(PlannerAgent)
            agent.system_prompt = "test"
            agent.tools = []
            agent.model_info = "test"

            captured = {}

            def fake_invoke(**kwargs):
                captured.update(kwargs)
                return "fake response"

            agent.invoke = lambda **kw: fake_invoke(**kw)

            outputs = {"weather": "Sunny", "researcher": "Alfama"}
            agent.synthesize("Plan my day", outputs)

            assert captured.get("qa_disclaimers") is None


# ---------------------------------------------------------------
# 4. Static data integrity
# ---------------------------------------------------------------

class TestStaticData:
    """Verify static data in qa_agent.py is valid."""

    def test_metro_stations_not_empty(self):
        from agent.agents.qa_agent import _METRO_CANONICAL_STATIONS
        assert len(_METRO_CANONICAL_STATIONS) >= 50, (
            f"Expected 50+ stations, got {len(_METRO_CANONICAL_STATIONS)}"
        )

    def test_metro_stations_all_lowercase(self):
        from agent.agents.qa_agent import _METRO_CANONICAL_STATIONS
        for s in _METRO_CANONICAL_STATIONS:
            assert s == s.lower(), f"Station '{s}' is not lowercase"

    def test_aml_bounds_valid(self):
        from agent.agents.qa_agent import _AML_BOUNDS
        assert _AML_BOUNDS["lat_min"] < _AML_BOUNDS["lat_max"]
        assert _AML_BOUNDS["lon_min"] < _AML_BOUNDS["lon_max"]
        # Lisbon should be within these bounds
        assert _AML_BOUNDS["lat_min"] <= 38.72 <= _AML_BOUNDS["lat_max"]
        assert _AML_BOUNDS["lon_min"] <= -9.14 <= _AML_BOUNDS["lon_max"]

    def test_known_domains_include_essentials(self):
        from agent.agents.qa_agent import _VALID_DOMAINS
        essentials = {"visitlisboa.com", "metrolisboa.pt", "cp.pt", "ipma.pt"}
        assert essentials.issubset(_VALID_DOMAINS), (
            f"Missing essential domains: {essentials - _VALID_DOMAINS}"
        )

    def test_forecast_days_is_5(self):
        from agent.agents.qa_agent import _IPMA_FORECAST_DAYS
        assert _IPMA_FORECAST_DAYS == 5

    def test_truncation_limit_reasonable(self):
        from agent.agents.qa_agent import _TRUNCATION_LIMIT
        assert 4000 <= _TRUNCATION_LIMIT <= 10000, (
            f"Truncation limit {_TRUNCATION_LIMIT} seems unreasonable"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
