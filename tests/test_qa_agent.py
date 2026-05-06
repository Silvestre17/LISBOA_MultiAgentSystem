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

import json
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _make_llm_response(qa_json: dict) -> MagicMock:
    """Create a mock LLM response carrying a serialized QA payload."""
    mock_response = MagicMock()
    mock_response.content = json.dumps(qa_json)
    return mock_response


def _default_validation_payload(**overrides) -> dict:
    """Return a compact baseline payload for validate() integration tests."""
    payload = {
        "complete": True,
        "missing_data": [],
        "required_agents": [],
        "reasoning": "All data present.",
        "disclaimers": [],
    }
    payload.update(overrides)
    return payload


def test_repair_final_response_uses_runtime_output_language_in_prompt() -> None:
    """The QA repair pass must obey the runtime-resolved PT/EN output language, not the raw user-message language."""
    from agent.agents.qa_agent import QualityAssuranceAgent

    agent = object.__new__(QualityAssuranceAgent)
    agent.llm = MagicMock()

    captured: dict[str, str] = {}

    def _fake_safe_invoke(_llm, messages):
        captured["system"] = messages[0].content
        captured["human"] = messages[1].content
        response = MagicMock()
        response.content = "Repaired answer in English."
        return response

    agent._safe_llm_invoke = _fake_safe_invoke

    repaired = agent.repair_final_response(
        user_query="Quel temps fait-il à Lisbonne aujourd'hui?",
        draft_response="Brouillon en français.",
        agent_outputs={"weather": "Weather output.", "transport": "Transport output."},
        qa_result={
            "missing_data": ["transport details"],
            "critical_issues": ["final answer is in the wrong language"],
            "disclaimers": ["Language must follow runtime output language"],
        },
        language="en",
    )

    assert repaired == "Repaired answer in English."
    assert "required runtime output language" in captured["system"].lower()
    assert "preserve the user's language" not in captured["system"].lower()
    assert "**Required output language:** English" in captured["human"]


def test_repair_final_response_filters_internal_disclaimers_before_prompt() -> None:
    """Internal QA warnings must guide repair silently, never become user-facing caveats."""
    from agent.agents.qa_agent import QualityAssuranceAgent

    agent = object.__new__(QualityAssuranceAgent)
    agent.llm = MagicMock()

    captured: dict[str, str] = {}

    def _fake_safe_invoke(_llm, messages):
        captured["human"] = messages[1].content
        response = MagicMock()
        response.content = "### Resposta\n\nRota revista.\n\n📌 **Fonte:** [*Carris*](https://www.carris.pt)"
        return response

    agent._safe_llm_invoke = _fake_safe_invoke

    repaired = agent.repair_final_response(
        user_query="Como vou de autocarro para Belém?",
        draft_response="Rota inicial.",
        agent_outputs={"transport": "Linha 15E para Algés."},
        qa_result={
            "missing_data": [],
            "critical_issues": ["Technical transport identifiers leaked into user-facing output."],
            "disclaimers": [
                "Quality validation could not produce a valid structured result after retry",
                "Source footer is missing or malformed.",
                "Carris bus route numbers and schedules should be verified at carris.pt, as GTFS data may not reflect the most recent changes.",
            ],
        },
        language="pt",
    )

    assert "Quality validation" not in captured["human"]
    assert "Source footer" not in captured["human"]
    assert "Os números das linhas e os horários da Carris devem ser confirmados" in captured["human"]
    assert "Quality validation" not in repaired
    assert "Source footer" not in repaired


def test_guard_final_response_strips_residual_qa_sections_and_keeps_source_last() -> None:
    """The deterministic final QA guard should clean post-synthesis QA leakage."""
    from agent.agents.qa_agent import QualityAssuranceAgent

    agent = object.__new__(QualityAssuranceAgent)

    guarded = agent.guard_final_response(
        (
            "### Resultado\n\n"
            "Rota confirmada.\n\n"
            "📌 **Fonte:** [*Metro de Lisboa*](https://www.metrolisboa.pt)\n\n"
            "### QA Findings\n\n"
            "- Missing data: source footer is malformed\n"
            "- Reasoning: internal validation note"
        ),
        language="pt",
    )

    assert "QA Findings" not in guarded
    assert "Missing data" not in guarded
    assert "Reasoning" not in guarded
    assert guarded.rstrip().endswith("[*Metro de Lisboa*](https://www.metrolisboa.pt)")


def test_validate_marks_double_json_parse_failure_as_incomplete() -> None:
    """If QA cannot parse valid JSON twice, it should fail closed and trigger repair."""
    from agent.agents.qa_agent import QualityAssuranceAgent

    agent = object.__new__(QualityAssuranceAgent)
    agent.llm = MagicMock()
    agent._safe_llm_invoke = MagicMock(
        side_effect=[
            MagicMock(content="not json"),
            MagicMock(content="still not json"),
        ]
    )

    result = agent.validate(
        user_query="How do I get from Rossio to Belém?",
        agent_outputs={"transport": "Transport output."},
        agents_called=["transport"],
        language="en",
    )

    assert result["complete"] is False
    assert result["needs_repair"] is True
    assert any("qa validation structure could not be confirmed" in item.lower() for item in result["missing_data"])
    assert any("quality validation could not produce a valid structured result" in item.lower() for item in result["disclaimers"])


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

    def test_departure_time_lists_do_not_trigger_false_coordinate_warning(self):
        """Comma-separated departure times must not be misread as out-of-scope coordinates."""
        text = (
            "Next departures: 11:33, 11:59, 12:14\n"
            "The bus feed is active and the stop is Rossio."
        )
        result = self.agent._verify_facts(text, "test query", None)
        coord_warns = [w for w in result["disclaimers"] if "outside" in w.lower()]
        assert coord_warns == []

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

    def test_en_prompt_mentions_place_card_completeness_and_malformed_ticket_links(self):
        """EN QA prompt should explicitly guard against collapsed place cards and malformed ticket links."""
        from agent.prompts.qa import get_qa_prompt

        prompt = get_qa_prompt("en")

        assert "Collapsed place cards" in prompt
        assert "Malformed markdown links" in prompt
        assert "[Bilhetes](Não disponível)" in prompt

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

    def test_researcher_prompt_omits_ticket_field_for_non_urls(self):
        """Researcher prompt should omit ticket placeholders when no real URL exists."""
        from agent.prompts.researcher import get_researcher_prompt

        prompt = get_researcher_prompt()

        assert "only render a markdown link when the value is a real URL" in prompt
        assert "otherwise omit the field" in prompt

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


class TestValidatePipeline:
    """Focused integration tests for QualityAssuranceAgent.validate()."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from agent.agents.qa_agent import QualityAssuranceAgent

        with patch.object(QualityAssuranceAgent, "__init__", lambda self: None):
            self.agent = QualityAssuranceAgent()
            self.agent.agent_name = "qa"
            self.agent.tools = []
            self.agent.llm = MagicMock()
            self.agent.llm_with_tools = None

    def test_validate_complete_weather_response(self):
        """A clean weather answer should pass the full QA pipeline."""
        self.agent._safe_llm_invoke = MagicMock(
            return_value=_make_llm_response(_default_validation_payload())
        )

        result = self.agent.validate(
            user_query="What's the weather today?",
            agent_outputs={"weather": "Lisbon today: 25C, sunny. No warnings active."},
            agents_called=["weather"],
            language="en",
        )

        assert result["complete"] is True
        assert result["required_agents"] == []
        assert "fact_check" in result

    def test_validate_requests_worker_retry_when_data_is_missing(self):
        """QA should preserve a legitimate missing-data retry request from the LLM layer."""
        self.agent._safe_llm_invoke = MagicMock(
            return_value=_make_llm_response(
                _default_validation_payload(
                    complete=False,
                    missing_data=["temperature", "precipitation"],
                    required_agents=["weather"],
                    reasoning="Weather output lacks temperature and precipitation data.",
                )
            )
        )

        result = self.agent.validate(
            user_query="Will it rain tomorrow?",
            agent_outputs={"weather": "Tomorrow: some clouds expected."},
            agents_called=["weather"],
            language="en",
        )

        assert result["complete"] is False
        assert result["required_agents"] == ["weather"]
        assert "temperature" in result["missing_data"]

    def test_validate_event_only_query_stays_research_only(self):
        """Pure event discovery should not be expanded into weather or transport retries by QA."""
        self.agent._safe_llm_invoke = MagicMock(
            return_value=_make_llm_response(
                _default_validation_payload(
                    complete=False,
                    missing_data=["weather forecast for this week", "transport routes between venues"],
                    required_agents=["weather", "transport"],
                    reasoning="Useful weekly event answer should also include weather and transport context.",
                    disclaimers=[
                        "Dados de meteorologia não fornecidos na saída do researcher.",
                        "Sem rota ou ligação entre os eventos sugeridos.",
                    ],
                )
            )
        )

        result = self.agent.validate(
            user_query="Quero explorar a cultura local. Que grandes eventos temos esta semana?",
            agent_outputs={"researcher": "1. Concerto A\n- Data: 12 Mar\n- Local: Lisboa"},
            agents_called=["researcher"],
            language="pt",
        )

        assert result["complete"] is True
        assert result["required_agents"] == []
        assert result["missing_data"] == []
        assert not any(
            marker in disclaimer.lower()
            for disclaimer in result["disclaimers"]
            for marker in ("meteorolog", "transport", "rota")
        )

    def test_validate_merges_deterministic_fact_check_disclaimer(self):
        """Deterministic fact checks should still warn even if the LLM says the answer is complete."""
        self.agent._safe_llm_invoke = MagicMock(
            return_value=_make_llm_response(_default_validation_payload())
        )

        result = self.agent.validate(
            user_query="How do I get to Benfica by metro?",
            agent_outputs={"transport": "Take metro to estação de Benfica on the metro Azul line."},
            agents_called=["transport"],
            language="en",
        )

        assert result["complete"] is True
        assert any("benfica" in warning.lower() for warning in result["disclaimers"])

    def test_validate_marks_repairable_output_hygiene_issue(self):
        """Technical transport metadata leaks should mark the worker output for repair."""
        self.agent._safe_llm_invoke = MagicMock(
            return_value=_make_llm_response(_default_validation_payload())
        )

        result = self.agent.validate(
            user_query="Show me the live buses right now.",
            agent_outputs={
                "transport": (
                    "🚌 Live buses\n"
                    "- 📍 GPS: 38.72410, -9.14820\n"
                    "- 🚏 Next stop ID: 060001\n"
                    "- **Plate**: 12-AB-34"
                )
            },
            agents_called=["transport"],
            language="en",
        )

        assert result["complete"] is True
        assert result["needs_repair"] is True
        assert "transport" in result["repairable_agents"]
        assert result["fact_check"]["per_agent"]["transport"]["critical_issues"]

    def test_validate_marks_mixed_labels_and_bad_links_for_repair(self):
        """Mixed-language labels, non-URL markdown links, and post-source warnings must trigger repair."""
        self.agent._safe_llm_invoke = MagicMock(
            return_value=_make_llm_response(_default_validation_payload())
        )

        result = self.agent.validate(
            user_query="Show me the event details in English.",
            agent_outputs={
                "researcher": (
                    "### 🎭 Event\n\n"
                    "**Categoria:** Música\n"
                    "🎟️ **Bilhetes:** [Bilhetes](Não disponível)\n\n"
                    "📌 **Source:** [*VisitLisboa*](https://www.visitlisboa.com) | **Updated:** 14:00\n"
                    "- ⚠️ Confirm before you go."
                )
            },
            agents_called=["researcher"],
            language="en",
        )

        assert result["complete"] is True
        assert result["needs_repair"] is True
        assert "researcher" in result["repairable_agents"]
        issues = " ".join(result["fact_check"]["per_agent"]["researcher"]["critical_issues"]).lower()
        assert "portuguese field labels" in issues
        assert "markdown links" in issues
        assert "source footer" in issues

    def test_validate_marks_collapsed_place_cards_for_repair(self):
        """Place-only answers without canonical fields should be flagged for repair."""
        self.agent._safe_llm_invoke = MagicMock(
            return_value=_make_llm_response(_default_validation_payload())
        )

        result = self.agent.validate(
            user_query="Lista as atrações imperdíveis para quem visita Lisboa pela primeira vez.",
            agent_outputs={
                "researcher": (
                    "### 🏛️ Monument to the Discoveries\n\n"
                    "Um dos monumentos mais emblemáticos de Lisboa."
                )
            },
            agents_called=["researcher"],
            language="pt",
        )

        assert result["complete"] is True
        assert result["needs_repair"] is True
        assert "researcher" in result["repairable_agents"]
        assert any(
            "place cards" in issue.lower()
            for issue in result["fact_check"]["per_agent"]["researcher"]["critical_issues"]
        )

    def test_validate_retries_after_invalid_json(self):
        """One malformed LLM reply should trigger a retry instead of an immediate fallback."""
        bad_response = MagicMock()
        bad_response.content = "This is not JSON at all."
        good_response = _make_llm_response(
            _default_validation_payload(reasoning="Retry successful.")
        )
        self.agent._safe_llm_invoke = MagicMock(side_effect=[bad_response, good_response])

        result = self.agent.validate(
            user_query="Test query",
            agent_outputs={"weather": "Sunny, 22C."},
            agents_called=["weather"],
            language="en",
        )

        assert result["complete"] is True
        assert "Retry successful" in result["reasoning"]
        assert self.agent._safe_llm_invoke.call_count == 2

    def test_validate_falls_back_after_double_json_failure(self):
        """Two malformed replies should fail closed with a conservative fallback result."""
        bad_response = MagicMock()
        bad_response.content = "Not JSON"
        self.agent._safe_llm_invoke = MagicMock(return_value=bad_response)

        result = self.agent.validate(
            user_query="Test query",
            agent_outputs={"weather": "Sunny, 22C."},
            agents_called=["weather"],
            language="en",
        )

        assert result["complete"] is False
        assert "could not be confirmed" in result["missing_data"][0].lower()
        assert any("structured result after retry" in disclaimer.lower() for disclaimer in result["disclaimers"])
        assert "fact_check" in result

    def test_validate_includes_context_and_skips_internal_keys(self):
        """User context and history should reach the validation prompt, but internal keys should stay hidden."""
        captured_messages = []

        def capture_invoke(_llm, messages, **_kwargs):
            captured_messages.extend(messages)
            return _make_llm_response(_default_validation_payload())

        self.agent._safe_llm_invoke = capture_invoke

        self.agent.validate(
            user_query="And what about the trains?",
            agent_outputs={
                "transport": "CP trains run hourly to Sintra.",
                "_qa_disclaimers": ["This should stay hidden"],
            },
            agents_called=["transport"],
            language="en",
            user_context={
                "preferences": ["history", "architecture"],
                "available_time": 6,
            },
            conversation_history=[
                "What's the weather today?",
                "Plan a day in Sintra",
                "And what about the trains?",
            ],
        )

        human_messages = [
            message for message in captured_messages
            if hasattr(message, "content") and "VALIDATION TASK" in message.content
        ]
        assert human_messages
        content = human_messages[0].content
        assert "history" in content
        assert "architecture" in content
        assert "Sintra" in content
        assert "_qa_disclaimers" not in content
