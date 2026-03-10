# ==========================================================================
# Master Thesis - QA Agent Integration Tests
#   - André Filipe Gomes Silvestre, 20240502
#
#   Integration tests for QualityAssuranceAgent.validate() pipeline.
#   Mocks the LLM to test the full two-phase validation flow without
#   requiring network access or API keys.
#
#   Run from the repository root with a relative path:
#     python -m pytest tests/test_qa_integration.py -q
#   Useful parameters:
#     -vv                            verbose mode
#     -k complete or -k incomplete   focus on one validation branch
#     -x                             stop on first failure
#     --tb=short                     shorter tracebacks
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.agents.qa_agent import QualityAssuranceAgent

# ==========================================================================
# Helpers
# ==========================================================================


def _make_llm_response(qa_json: dict) -> MagicMock:
    """Creates a mock LLM response containing a JSON validation result."""
    mock_resp = MagicMock()
    mock_resp.content = json.dumps(qa_json)
    return mock_resp


def _default_qa_json(**overrides) -> dict:
    """Returns a baseline QA JSON with optional overrides."""
    base = {
        "complete": True,
        "missing_data": [],
        "required_agents": [],
        "reasoning": "All data present.",
        "disclaimers": [],
    }
    base.update(overrides)
    return base


# ==========================================================================
# Integration Tests: Full validate() Pipeline
# ==========================================================================


class TestValidateComplete:
    """Test validate() returns correct results when LLM says data is complete."""

    @pytest.fixture(autouse=True)
    def setup(self):
        with patch.object(QualityAssuranceAgent, "__init__", lambda self: None):
            self.agent = QualityAssuranceAgent()
            self.agent.agent_name = "qa"
            self.agent.tools = []
            self.agent.llm = MagicMock()
            self.agent.llm_with_tools = None

    def test_complete_weather_response(self):
        """Weather data marked complete, no fact-check issues."""
        qa_json = _default_qa_json()
        self.agent._safe_llm_invoke = MagicMock(return_value=_make_llm_response(qa_json))

        result = self.agent.validate(
            user_query="What's the weather today?",
            agent_outputs={"weather": "Lisbon today: 25C, sunny. No warnings active."},
            agents_called=["weather"],
            language="en",
        )

        assert result["complete"] is True
        assert result["missing_data"] == []
        assert result["required_agents"] == []
        assert "fact_check" in result
        assert isinstance(result["disclaimers"], list)

    def test_complete_transport_with_valid_metro(self):
        """Transport with valid metro stations passes both phases."""
        qa_json = _default_qa_json()
        self.agent._safe_llm_invoke = MagicMock(return_value=_make_llm_response(qa_json))

        result = self.agent.validate(
            user_query="How to get to Alameda?",
            agent_outputs={
                "transport": (
                    "Take metro verde line from Rossio to Alameda. "
                    "Wait time: 4 minutes. Service status: normal."
                )
            },
            agents_called=["transport"],
            language="en",
        )

        assert result["complete"] is True
        # Phase 2 should not flag known stations
        metro_warns = [w for w in result["disclaimers"] if "metro" in w.lower() and "verified" in w.lower()]
        assert len(metro_warns) == 0


class TestValidateIncomplete:
    """Test validate() handles LLM marking data as incomplete."""

    @pytest.fixture(autouse=True)
    def setup(self):
        with patch.object(QualityAssuranceAgent, "__init__", lambda self: None):
            self.agent = QualityAssuranceAgent()
            self.agent.agent_name = "qa"
            self.agent.tools = []
            self.agent.llm = MagicMock()
            self.agent.llm_with_tools = None

    def test_missing_weather_data(self):
        """LLM flags missing weather data, requires weather agent retry."""
        qa_json = _default_qa_json(
            complete=False,
            missing_data=["temperature", "precipitation"],
            required_agents=["weather"],
            reasoning="Weather output lacks temperature and precipitation data.",
        )
        self.agent._safe_llm_invoke = MagicMock(return_value=_make_llm_response(qa_json))

        result = self.agent.validate(
            user_query="Will it rain tomorrow?",
            agent_outputs={"weather": "Tomorrow: some clouds expected."},
            agents_called=["weather"],
            language="en",
        )

        assert result["complete"] is False
        assert "temperature" in result["missing_data"]
        assert "weather" in result["required_agents"]

    def test_missing_transport_requests_retry(self):
        """LLM requests transport agent retry for missing schedule data."""
        qa_json = _default_qa_json(
            complete=False,
            missing_data=["schedule", "wait_time"],
            required_agents=["transport"],
            reasoning="Transport data lacks schedule information.",
        )
        self.agent._safe_llm_invoke = MagicMock(return_value=_make_llm_response(qa_json))

        result = self.agent.validate(
            user_query="When is the next metro to Oriente?",
            agent_outputs={"transport": "Metro verde line serves Oriente station."},
            agents_called=["transport"],
            language="en",
        )

        assert result["complete"] is False
        assert "transport" in result["required_agents"]

    def test_invalid_agent_names_filtered(self):
        """Required agents list filters out invalid agent names."""
        qa_json = _default_qa_json(
            complete=False,
            required_agents=["weather", "planner", "nonexistent_agent"],
        )
        self.agent._safe_llm_invoke = MagicMock(return_value=_make_llm_response(qa_json))

        result = self.agent.validate(
            user_query="Plan my day",
            agent_outputs={"researcher": "Some data about Lisbon."},
            agents_called=["researcher"],
            language="en",
        )

        assert "weather" in result["required_agents"]
        assert "planner" not in result["required_agents"]
        assert "nonexistent_agent" not in result["required_agents"]

    def test_event_only_query_does_not_escalate_to_weather_or_transport(self):
        """Events-only discovery queries must not be expanded into weather/transport retries by QA."""
        qa_json = _default_qa_json(
            complete=False,
            missing_data=[
                "weather forecast for this week",
                "transport routes between venues",
            ],
            required_agents=["weather", "transport"],
            reasoning="Useful weekly event answer should also include weather and transport context.",
            disclaimers=[
                "Dados de meteorologia não fornecidos na saída do researcher; necessidade de previsão para a semana para tornar o planeamento útil.",
                "Sem rota/ligação entre os eventos sugeridos; sem indicar meios e transferências entre locais.",
            ],
        )
        self.agent._safe_llm_invoke = MagicMock(return_value=_make_llm_response(qa_json))

        result = self.agent.validate(
            user_query="Quero explorar a cultura local. Que grandes eventos temos esta semana?",
            agent_outputs={
                "researcher": "1. Concerto A\n- Data: 12 Mar\n- Local: Lisboa"
            },
            agents_called=["researcher"],
            language="pt",
        )

        assert result["complete"] is True
        assert result["required_agents"] == []
        assert result["missing_data"] == []
        assert not any("meteorolog" in item.lower() for item in result["disclaimers"])
        assert not any("rota" in item.lower() or "transport" in item.lower() for item in result["disclaimers"])


class TestValidateWithFactCheck:
    """Test that Phase 2 deterministic checks merge into Phase 1 results."""

    @pytest.fixture(autouse=True)
    def setup(self):
        with patch.object(QualityAssuranceAgent, "__init__", lambda self: None):
            self.agent = QualityAssuranceAgent()
            self.agent.agent_name = "qa"
            self.agent.tools = []
            self.agent.llm = MagicMock()
            self.agent.llm_with_tools = None

    def test_invalid_metro_merges_disclaimer(self):
        """Phase 2 metro check adds disclaimer even when LLM says complete."""
        qa_json = _default_qa_json()
        self.agent._safe_llm_invoke = MagicMock(return_value=_make_llm_response(qa_json))

        result = self.agent.validate(
            user_query="How to get to Benfica by metro?",
            agent_outputs={
                "transport": "Take metro to estação de Benfica on the metro Azul line."
            },
            agents_called=["transport"],
            language="en",
        )

        assert result["complete"] is True  # LLM said complete
        # But Phase 2 should have added a metro disclaimer
        metro_warns = [w for w in result["disclaimers"] if "benfica" in w.lower()]
        assert len(metro_warns) > 0, f"Expected Benfica warning, got {result['disclaimers']}"

    def test_suspicious_url_merges_disclaimer(self):
        """Phase 2 URL check flags unknown domains."""
        qa_json = _default_qa_json()
        self.agent._safe_llm_invoke = MagicMock(return_value=_make_llm_response(qa_json))

        result = self.agent.validate(
            user_query="Where can I book a tour?",
            agent_outputs={
                "researcher": "Book at https://www.fake-lisbon-tours.com/book for best prices."
            },
            agents_called=["researcher"],
            language="en",
        )

        url_warns = [w for w in result["disclaimers"] if "unverified" in w.lower()]
        assert len(url_warns) > 0

    def test_out_of_bounds_coordinates_merges(self):
        """Phase 2 coordinate check flags non-AML coordinates."""
        qa_json = _default_qa_json()
        self.agent._safe_llm_invoke = MagicMock(return_value=_make_llm_response(qa_json))

        result = self.agent.validate(
            user_query="Find restaurants near me",
            agent_outputs={
                "researcher": "Found restaurants near 41.15, -8.61 in Porto area."
            },
            agents_called=["researcher"],
            language="en",
        )

        coord_warns = [w for w in result["disclaimers"] if "outside" in w.lower()]
        assert len(coord_warns) > 0

    def test_forecast_beyond_range_merges(self):
        """Phase 2 date check flags forecasts beyond IPMA range."""
        qa_json = _default_qa_json()
        self.agent._safe_llm_invoke = MagicMock(return_value=_make_llm_response(qa_json))

        far_date = (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")
        result = self.agent.validate(
            user_query="What's the weather next week?",
            agent_outputs={
                "weather": f"Weather forecast for {far_date}: expect rain and 18C."
            },
            agents_called=["weather"],
            language="en",
        )

        date_warns = [w for w in result["disclaimers"] if "forecast range" in w.lower()]
        assert len(date_warns) > 0

    def test_wheelchair_accessibility_merges(self):
        """Phase 2 flags missing accessibility info for wheelchair users."""
        qa_json = _default_qa_json()
        self.agent._safe_llm_invoke = MagicMock(return_value=_make_llm_response(qa_json))

        result = self.agent.validate(
            user_query="How to get to Belem?",
            agent_outputs={
                "transport": "Take public transport tram 15E from Praça do Comércio to Belém. Departs every 12 min."
            },
            agents_called=["transport"],
            language="en",
            user_context={"mobility": "wheelchair"},
        )

        access_warns = [
            w for w in result["disclaimers"]
            if "accessibility" in w.lower() or "wheelchair" in w.lower()
        ]
        assert len(access_warns) > 0, f"Expected accessibility warning, got {result['disclaimers']}"


class TestValidateJSONFallback:
    """Test fallback behavior when LLM returns invalid JSON."""

    @pytest.fixture(autouse=True)
    def setup(self):
        with patch.object(QualityAssuranceAgent, "__init__", lambda self: None):
            self.agent = QualityAssuranceAgent()
            self.agent.agent_name = "qa"
            self.agent.tools = []
            self.agent.llm = MagicMock()
            self.agent.llm_with_tools = None

    def test_invalid_json_triggers_retry(self):
        """First call returns garbage, retry returns valid JSON."""
        bad_resp = MagicMock()
        bad_resp.content = "This is not JSON at all, just text."

        good_resp = _make_llm_response(_default_qa_json(
            reasoning="Retry successful."
        ))

        self.agent._safe_llm_invoke = MagicMock(side_effect=[bad_resp, good_resp])

        result = self.agent.validate(
            user_query="Test query",
            agent_outputs={"weather": "Sunny, 22C."},
            agents_called=["weather"],
            language="en",
        )

        assert result["complete"] is True
        assert "Retry successful" in result["reasoning"]
        assert self.agent._safe_llm_invoke.call_count == 2

    def test_double_failure_falls_back(self):
        """Both LLM calls return invalid JSON, falls back to safe default."""
        bad_resp = MagicMock()
        bad_resp.content = "Not JSON"

        self.agent._safe_llm_invoke = MagicMock(return_value=bad_resp)

        result = self.agent.validate(
            user_query="Test query",
            agent_outputs={"weather": "Sunny, 22C."},
            agents_called=["weather"],
            language="en",
        )

        # Should fall back to complete=True with disclaimer
        assert result["complete"] is True
        assert any("limited" in d.lower() for d in result["disclaimers"])
        assert "fact_check" in result


class TestValidateWithUserContext:
    """Test that user context and conversation history flow through correctly."""

    @pytest.fixture(autouse=True)
    def setup(self):
        with patch.object(QualityAssuranceAgent, "__init__", lambda self: None):
            self.agent = QualityAssuranceAgent()
            self.agent.agent_name = "qa"
            self.agent.tools = []
            self.agent.llm = MagicMock()
            self.agent.llm_with_tools = None

    def test_user_context_appears_in_llm_prompt(self):
        """User context is included in the context sent to LLM."""
        qa_json = _default_qa_json()
        captured_messages = []

        def capture_invoke(llm, messages, **kwargs):
            captured_messages.extend(messages)
            return _make_llm_response(qa_json)

        self.agent._safe_llm_invoke = capture_invoke

        self.agent.validate(
            user_query="Plan my day in Lisbon",
            agent_outputs={"researcher": "Visit Belem Tower and Jerónimos."},
            agents_called=["researcher"],
            language="en",
            user_context={
                "preferences": ["history", "architecture"],
                "mobility": "normal",
                "available_time": 6,
                "latitude": 38.7223,
                "longitude": -9.1393,
            },
        )

        # The HumanMessage should contain user context info
        human_msg = [m for m in captured_messages if hasattr(m, "content") and "VALIDATION TASK" in m.content]
        assert len(human_msg) > 0
        content = human_msg[0].content
        assert "history" in content
        assert "architecture" in content
        assert "6" in content  # available_time
        assert "38.7223" in content  # latitude

    def test_conversation_history_in_prompt(self):
        """Conversation history is included in the context sent to LLM."""
        qa_json = _default_qa_json()
        captured_messages = []

        def capture_invoke(llm, messages, **kwargs):
            captured_messages.extend(messages)
            return _make_llm_response(qa_json)

        self.agent._safe_llm_invoke = capture_invoke

        self.agent.validate(
            user_query="And what about the trains?",
            agent_outputs={"transport": "CP trains run hourly to Sintra."},
            agents_called=["transport"],
            language="en",
            conversation_history=[
                "What's the weather today?",
                "Plan a day in Sintra",
                "And what about the trains?",
            ],
        )

        human_msg = [m for m in captured_messages if hasattr(m, "content") and "VALIDATION TASK" in m.content]
        assert len(human_msg) > 0
        content = human_msg[0].content
        assert "Sintra" in content
        assert "Recent conversation" in content

    def test_internal_keys_skipped(self):
        """Keys starting with _ are not included in agent outputs context."""
        qa_json = _default_qa_json()
        captured_messages = []

        def capture_invoke(llm, messages, **kwargs):
            captured_messages.extend(messages)
            return _make_llm_response(qa_json)

        self.agent._safe_llm_invoke = capture_invoke

        self.agent.validate(
            user_query="Test",
            agent_outputs={
                "weather": "Sunny, 22C.",
                "_qa_disclaimers": ["This should not appear"],
                "_internal": "hidden",
            },
            agents_called=["weather"],
            language="en",
        )

        human_msg = [m for m in captured_messages if hasattr(m, "content") and "VALIDATION TASK" in m.content]
        content = human_msg[0].content
        assert "WEATHER Agent Output" in content
        assert "_qa_disclaimers" not in content
        assert "hidden" not in content


class TestValidatePortuguese:
    """Test validate() works correctly with Portuguese language."""

    @pytest.fixture(autouse=True)
    def setup(self):
        with patch.object(QualityAssuranceAgent, "__init__", lambda self: None):
            self.agent = QualityAssuranceAgent()
            self.agent.agent_name = "qa"
            self.agent.tools = []
            self.agent.llm = MagicMock()
            self.agent.llm_with_tools = None

    def test_portuguese_prompt_used(self):
        """Portuguese language triggers PT prompt template."""
        qa_json = _default_qa_json()
        captured_messages = []

        def capture_invoke(llm, messages, **kwargs):
            captured_messages.extend(messages)
            return _make_llm_response(qa_json)

        self.agent._safe_llm_invoke = capture_invoke

        self.agent.validate(
            user_query="Como está o tempo hoje?",
            agent_outputs={"weather": "Lisboa: 25C, sol, sem avisos."},
            agents_called=["weather"],
            language="pt",
        )

        system_msg = [m for m in captured_messages if hasattr(m, "content") and len(m.content) > 500]
        assert len(system_msg) > 0
        # PT prompt should contain Portuguese keywords
        sys_content = system_msg[0].content
        assert any(kw in sys_content for kw in ["Controlo de Qualidade", "VALIDAÇÃO", "completo"]), (
            "Expected Portuguese prompt content"
        )

    def test_portuguese_with_user_context(self):
        """Portuguese + user context correctly assembled."""
        qa_json = _default_qa_json()
        self.agent._safe_llm_invoke = MagicMock(return_value=_make_llm_response(qa_json))

        result = self.agent.validate(
            user_query="Planeia o meu dia em Lisboa",
            agent_outputs={
                "researcher": "Visite a Torre de Belém e o Mosteiro dos Jerónimos.",
                "transport": "Apanhe o Metro verde na estação de Alameda até ao Oriente.",
            },
            agents_called=["researcher", "transport"],
            language="pt",
            user_context={"preferences": ["história", "cultura"], "mobility": "normal"},
        )

        assert result["complete"] is True
        assert "fact_check" in result


class TestTruncation:
    """Test that long outputs are truncated properly."""

    @pytest.fixture(autouse=True)
    def setup(self):
        with patch.object(QualityAssuranceAgent, "__init__", lambda self: None):
            self.agent = QualityAssuranceAgent()
            self.agent.agent_name = "qa"
            self.agent.tools = []
            self.agent.llm = MagicMock()
            self.agent.llm_with_tools = None

    def test_long_output_truncated_in_context(self):
        """Agent outputs over 6000 chars are truncated before LLM call."""
        qa_json = _default_qa_json()
        captured_messages = []

        def capture_invoke(llm, messages, **kwargs):
            captured_messages.extend(messages)
            return _make_llm_response(qa_json)

        self.agent._safe_llm_invoke = capture_invoke

        long_text = "z" * 10000  # 10K chars, using 'z' to avoid counting header chars
        self.agent.validate(
            user_query="Test",
            agent_outputs={"weather": long_text},
            agents_called=["weather"],
            language="en",
        )

        human_msg = [m for m in captured_messages if hasattr(m, "content") and "VALIDATION TASK" in m.content]
        content = human_msg[0].content
        # The output in the context should be truncated to 6000 chars
        # (total content will be larger due to headers, but the AAAA block should be capped)
        assert content.count("z") <= 6001  # z only appears in the agent output, not headers
