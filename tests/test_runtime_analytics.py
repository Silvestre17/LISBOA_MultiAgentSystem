# ==========================================================================
# Master Thesis - Runtime Analytics and Planning Guardrails Tests
#   - André Filipe Gomes Silvestre, 20240502
#
#   Focused regression tests for:
#     - runtime execution summary and cost accounting
#     - assistant history persistence
#     - lightweight weather fact-checking
#     - future-planning transport responses
#     - multi-day planner quality mode
# ==========================================================================

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from agent.agents.planner_agent import PlannerAgent, enforce_multi_day_quality_mode
from agent.agents.researcher_agent import ResearcherAgent
from agent.agents.transport_agent import (
    TransportAgent,
    _build_deterministic_metro_route_response,
)
from agent.graph import MultiAgentAssistant
from agent.utils.response_formatter import finalize_worker_response


def _make_usage_summary(
    *,
    model_id: str = "Unknown",
    input_tokens: int = 0,
    output_tokens: int = 0,
    call_count: int = 0,
    agent_name: str | None = None,
) -> dict:
    """Build a stable mocked usage summary for runtime tests."""
    breakdown = []
    total_tokens = input_tokens + output_tokens
    if call_count > 0:
        provider, model = model_id.split("::", 1) if "::" in model_id else ("unknown", model_id)
        breakdown.append(
            {
                "call_index": 1,
                "agent_name": agent_name,
                "provider": provider,
                "model": model,
                "model_id": model_id,
                "tokens": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                },
                "usage_available": True,
            }
        )

    return {
        "call_count": call_count,
        "usage_available": bool(call_count),
        "model_id": model_id,
        "tokens": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        },
        "llm_usage_breakdown": breakdown,
    }


def _make_worker_mock(
    *,
    output: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    call_count: int = 0,
    model_id: str = "azure::gpt-5-mini",
    tool_log: list[dict] | None = None,
) -> MagicMock:
    """Create a worker-like mock with the methods used by ``MultiAgentAssistant``."""
    worker = MagicMock()
    worker.invoke = MagicMock(return_value=output)
    worker.reset_llm_usage_tracking = MagicMock()
    worker.get_llm_usage_summary = MagicMock(
        return_value=_make_usage_summary(
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            call_count=call_count,
        )
    )
    worker.get_tool_calls_log = MagicMock(return_value=list(tool_log or []))
    worker.llm_provider = "azure"
    return worker


def test_multiagent_direct_response_records_history_and_prints_execution_summary(capsys) -> None:
    """Direct supervisor responses should still be recorded in history and emit the runtime summary."""
    assistant = MultiAgentAssistant.__new__(MultiAgentAssistant)
    assistant.state = {"messages": [], "user_context": None}

    assistant.supervisor = MagicMock()
    assistant.supervisor.reset_llm_usage_tracking = MagicMock()
    assistant.supervisor.route = MagicMock(
        return_value={"agents": [], "direct_response": "Olá direto", "reasoning": "Direct reply"}
    )
    assistant.supervisor.get_llm_usage_summary = MagicMock(
        return_value=_make_usage_summary(
            model_id="azure::gpt-5-mini",
            input_tokens=4000,
            output_tokens=800,
            call_count=1,
            agent_name="supervisor",
        )
    )

    assistant.qa_agent = MagicMock()
    assistant.qa_agent.reset_llm_usage_tracking = MagicMock()
    assistant.qa_agent.get_llm_usage_summary = MagicMock(return_value=_make_usage_summary(agent_name="qa"))

    assistant.agents = {
        "weather": _make_worker_mock(),
        "transport": _make_worker_mock(),
        "researcher": _make_worker_mock(),
        "planner": _make_worker_mock(),
    }

    with patch("agent.graph.LANGSMITH_AVAILABLE", False), patch.object(
        __import__("agent.graph", fromlist=["Config"]).Config,
        "SHOW_MARKDOWN_RESPONSE_IN_TERMINAL",
        False,
    ), patch(
        "agent.graph.get_langsmith_request_tracking_status",
        return_value={
            "tracking_state": "disabled",
            "status_label": "disabled",
            "save_attempted": False,
            "persistence_state": "disabled",
            "current_run_attached": False,
            "project_name": None,
            "run_id": None,
            "reason": "LangSmith tracing is disabled by environment",
            "note": "LangSmith tracing is disabled by environment",
        },
    ):
        output = assistant.chat("Olá", language="pt", verbose=False)

    captured = capsys.readouterr().out
    assert output == "Olá direto"
    assert isinstance(assistant.state["messages"][-1], AIMessage)
    assert assistant.state["messages"][-1].content == "Olá direto"
    assert "EXECUTION SUMMARY" in captured
    assert "User request: Olá" in captured
    assert "Routed agents: direct response" in captured
    assert "Execution: direct" in captured
    assert re.search(r"Total Cost: \(0\.\d{3,6}\$\)", captured)
    assert "Pricing Snapshot: 2026-04-17" in captured
    assert "LangSmith: disabled | Run context: not-attached | Persistence: disabled" in captured


def test_execution_summary_prints_active_langsmith_save_attempt(capsys) -> None:
    """The terminal summary should expose whether the current request is being traced."""
    assistant = MultiAgentAssistant.__new__(MultiAgentAssistant)

    assistant._print_execution_summary(
        {
            "elapsed_time": 1.23,
            "execution_type": "hybrid",
            "worker_mode": "parallel",
            "qa_path": "validated",
            "langsmith": {
                "tracking_state": "tracking_request",
                "status_label": "enabled",
                "save_attempted": True,
                "persistence_state": "unconfirmed",
                "current_run_attached": True,
                "project_name": "LISBOA Chat",
                "run_id": "run_123",
                "reason": "LangSmith tracing enabled",
                "note": "Run context is attached locally. LangSmith persistence remains unconfirmed and may fail asynchronously, for example because of remote quota or ingestion limits.",
            },
            "usage": {
                "call_count": 2,
                "tokens": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                },
            },
            "pricing_metadata": {"pricing_snapshot_date": "2026-03-19"},
            "total_cost": {"total_cost_usd": 0.0015, "missing_pricing_models": []},
            "models_used": ["azure::gpt-5-mini"],
            "relevant_agents": [],
            "agent_usage": {},
            "agent_costs": {},
            "agent_tool_logs": {},
            "total_tool_invocations": 0,
            "retry_agents_used": [],
        }
    )

    captured = capsys.readouterr().out
    assert "LangSmith: enabled | Run context: attached | Persistence: unconfirmed | Project: LISBOA Chat" in captured
    assert "Run ID: run_123" in captured
    assert "quota or ingestion limits" in captured.lower()


def test_execution_summary_marks_tool_only_agents_and_prints_tool_args(capsys) -> None:
    """Tool-only workers should be labeled clearly and print their logged arguments."""
    assistant = MultiAgentAssistant.__new__(MultiAgentAssistant)

    assistant._print_execution_summary(
        {
            "elapsed_time": 2.01,
            "user_request": "Give me the next 5 events that match",
            "routing_reasoning": "Follow-up domain override from previous user query (researcher)",
            "selected_agents": ["researcher"],
            "execution_type": "single-worker",
            "worker_mode": "sequential",
            "qa_path": "validated",
            "langsmith": {
                "tracking_state": "disabled",
                "status_label": "disabled",
                "save_attempted": False,
                "persistence_state": "disabled",
                "current_run_attached": False,
                "project_name": None,
                "run_id": None,
                "reason": "LangSmith tracing is disabled by environment",
                "note": "LangSmith tracing is disabled by environment",
            },
            "usage": {
                "call_count": 0,
                "tokens": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
            },
            "pricing_metadata": {"pricing_snapshot_date": "2026-03-19"},
            "total_cost": {"total_cost_usd": 0.0, "missing_pricing_models": []},
            "models_used": [],
            "relevant_agents": ["researcher"],
            "agent_usage": {"researcher": _make_usage_summary(agent_name="researcher")},
            "agent_costs": {"researcher": {"total_cost_usd": 0.0, "missing_pricing_models": []}},
            "agent_tool_logs": {
                "researcher": [
                    {
                        "tool_name": "search_cultural_events",
                        "args": {"date_filter": "this week", "max_results": 5, "offset": 5},
                    }
                ]
            },
            "total_tool_invocations": 1,
            "retry_agents_used": [],
        }
    )

    captured = capsys.readouterr().out
    assert "User request: Give me the next 5 events that match" in captured
    assert "Routed agents: researcher" in captured
    assert "Researcher [tool-only]" in captured
    assert "search_cultural_events" in captured
    assert '"offset": 5' in captured


def test_multiagent_simple_weather_runs_lightweight_fact_check_but_skips_validate() -> None:
    """Simple weather queries should keep low latency while still running deterministic fact-checking."""
    assistant = MultiAgentAssistant.__new__(MultiAgentAssistant)
    assistant.state = {"messages": [], "user_context": None}

    assistant.supervisor = MagicMock()
    assistant.supervisor.reset_llm_usage_tracking = MagicMock()
    assistant.supervisor.route = MagicMock(
        return_value={"agents": ["weather"], "direct_response": None, "reasoning": "weather only"}
    )
    assistant.supervisor.get_llm_usage_summary = MagicMock(return_value=_make_usage_summary(agent_name="supervisor"))

    weather_agent = _make_worker_mock(output="🌤️ Forecast body")
    weather_agent._is_current_weather_query = MagicMock(return_value=False)
    weather_agent._is_simple_forecast_query = MagicMock(return_value=True)

    assistant.agents = {
        "weather": weather_agent,
        "transport": _make_worker_mock(),
        "researcher": _make_worker_mock(),
        "planner": _make_worker_mock(),
    }

    assistant.qa_agent = MagicMock()
    assistant.qa_agent.reset_llm_usage_tracking = MagicMock()
    assistant.qa_agent.get_llm_usage_summary = MagicMock(return_value=_make_usage_summary(agent_name="qa"))
    assistant.qa_agent._verify_facts = MagicMock(
        return_value={
            "valid": True,
            "disclaimers": [],
            "critical_issues": [],
            "checks_performed": ["output_hygiene"],
            "repairable_agents": [],
            "per_agent": {},
        }
    )
    assistant.qa_agent.validate = MagicMock(side_effect=AssertionError("Full QA should stay skipped"))

    with patch("agent.graph.LANGSMITH_AVAILABLE", False), patch.object(
        __import__("agent.graph", fromlist=["Config"]).Config,
        "SHOW_MARKDOWN_RESPONSE_IN_TERMINAL",
        False,
    ), patch("agent.graph.clean_response", side_effect=lambda text: text), patch(
        "agent.graph.format_response", side_effect=lambda text: text
    ), patch("agent.graph.generate_response_title", return_value=None), patch(
        "agent.graph.ensure_response_title", side_effect=lambda text, _title: text
    ):
        output = assistant.chat(
            "Qual é a previsão do tempo para os próximos 3 dias?",
            language="pt",
            verbose=False,
        )

    weather_agent.invoke.assert_called_once()
    assistant.qa_agent._verify_facts.assert_called_once()
    assistant.qa_agent.validate.assert_not_called()
    assert output.startswith("🌤️ Forecast body")


def test_multiagent_simple_weather_surfaces_lightweight_qa_disclaimers() -> None:
    """Single-worker weather answers should keep lightweight QA caveats in the final render."""
    assistant = MultiAgentAssistant.__new__(MultiAgentAssistant)
    assistant.state = {"messages": [], "user_context": None}

    assistant.supervisor = MagicMock()
    assistant.supervisor.reset_llm_usage_tracking = MagicMock()
    assistant.supervisor.route = MagicMock(
        return_value={"agents": ["weather"], "direct_response": None, "reasoning": "weather only"}
    )
    assistant.supervisor.get_llm_usage_summary = MagicMock(return_value=_make_usage_summary(agent_name="supervisor"))

    weather_agent = _make_worker_mock(
        output="### 🌤️ Previsão\n\nSol todo o dia.\n\n📌 **Fonte:** [*IPMA*](https://www.ipma.pt) | **Atualizado:** 10:00"
    )
    weather_agent._is_current_weather_query = MagicMock(return_value=False)
    weather_agent._is_simple_forecast_query = MagicMock(return_value=True)

    assistant.agents = {
        "weather": weather_agent,
        "transport": _make_worker_mock(),
        "researcher": _make_worker_mock(),
        "planner": _make_worker_mock(),
    }

    assistant.qa_agent = MagicMock()
    assistant.qa_agent.reset_llm_usage_tracking = MagicMock()
    assistant.qa_agent.get_llm_usage_summary = MagicMock(return_value=_make_usage_summary(agent_name="qa"))
    assistant.qa_agent.validate = MagicMock(side_effect=AssertionError("Full QA should stay skipped"))
    assistant._run_lightweight_weather_fact_check = MagicMock(
        return_value={
            "performed": True,
            "requires_full_qa": False,
            "disclaimers": ["Live precipitation confidence should be rechecked before departure."],
            "fact_check": {
                "valid": True,
                "disclaimers": ["Live precipitation confidence should be rechecked before departure."],
                "critical_issues": [],
                "checks_performed": ["output_hygiene"],
                "repairable_agents": [],
                "per_agent": {},
            },
        }
    )

    with patch("agent.graph.LANGSMITH_AVAILABLE", False), patch.object(
        __import__("agent.graph", fromlist=["Config"]).Config,
        "SHOW_MARKDOWN_RESPONSE_IN_TERMINAL",
        False,
    ):
        output = assistant.chat(
            "Qual é a previsão do tempo para os próximos 3 dias?",
            language="pt",
            verbose=False,
        )

    assert "- ⚠️ Live precipitation confidence should be" in output
    assert "Live precipitation confidence should be rechecked before departure." in output
    assistant.qa_agent.validate.assert_not_called()


def test_multiagent_blocks_planner_synthesis_when_qa_is_still_incomplete() -> None:
    """Planner synthesis should be skipped, but QA may still repair the grounded fallback draft."""
    assistant = MultiAgentAssistant.__new__(MultiAgentAssistant)
    assistant.state = {"messages": [], "user_context": None}

    assistant.supervisor = MagicMock()
    assistant.supervisor.reset_llm_usage_tracking = MagicMock()
    assistant.supervisor.route = MagicMock(
        return_value={
            "agents": ["weather", "transport", "planner"],
            "direct_response": None,
            "reasoning": "planner request",
        }
    )
    assistant.supervisor.get_llm_usage_summary = MagicMock(
        return_value=_make_usage_summary(agent_name="supervisor")
    )

    assistant.agents = {
        "weather": _make_worker_mock(
            output="### 🌤️ Weather\n\nSunny all afternoon.\n\n📌 **Source:** [*IPMA*](https://www.ipma.pt) | **Updated:** 10:00"
        ),
        "transport": _make_worker_mock(
            output="### 🚇 Transport\n\nTake tram 15E.\n\n📌 **Source:** [*Carris*](https://www.carris.pt) | **Updated:** 10:00"
        ),
        "researcher": _make_worker_mock(),
        "planner": MagicMock(),
    }
    assistant.agents["planner"].reset_llm_usage_tracking = MagicMock()
    assistant.agents["planner"].get_llm_usage_summary = MagicMock(
        return_value=_make_usage_summary(agent_name="planner")
    )
    assistant.agents["planner"].synthesize = MagicMock(return_value="PLANNER OUTPUT")

    assistant.qa_agent = MagicMock()
    assistant.qa_agent.reset_llm_usage_tracking = MagicMock()
    assistant.qa_agent.get_llm_usage_summary = MagicMock(
        return_value=_make_usage_summary(agent_name="qa")
    )
    assistant.qa_agent.validate = MagicMock(
        return_value={
            "complete": False,
            "missing_data": ["museum opening hours"],
            "required_agents": [],
            "reasoning": "Still incomplete.",
            "disclaimers": ["Museum opening hours are not confirmed."],
            "critical_issues": [],
            "repairable_agents": [],
            "needs_repair": True,
            "fact_check": {
                "critical_issues": [],
                "disclaimers": [],
                "checks_performed": [],
                "repairable_agents": [],
                "per_agent": {},
            },
        }
    )
    assistant.qa_agent.repair_final_response = MagicMock(return_value="REPAIRED SYNTHESIS")

    assistant._append_assistant_message = MagicMock()
    assistant._collect_execution_summary = MagicMock(return_value={})
    assistant._print_execution_summary = MagicMock()

    with patch("agent.graph.LANGSMITH_AVAILABLE", False), patch.object(
        __import__("agent.graph", fromlist=["Config"]).Config,
        "SHOW_MARKDOWN_RESPONSE_IN_TERMINAL",
        False,
    ):
        output = assistant.chat(
            "Plan a museum afternoon in Lisbon with transport.",
            language="en",
            verbose=False,
        )

    assistant.agents["planner"].synthesize.assert_not_called()
    assistant.qa_agent.repair_final_response.assert_called_once()
    assert "PLANNER OUTPUT" not in output
    assert "REPAIRED SYNTHESIS" in output


def test_transport_future_metro_route_omits_realtime_waits() -> None:
    """Future transport planning should not show current Metro waits or line-status blocks."""
    wait_lines = MagicMock(return_value=["- ⏱️ 2 min"])

    with patch(
        "tools.transport_api.get_route_between_stations",
        new=MagicMock(invoke=MagicMock(return_value="METRO ROUTE")),
    ), patch(
        "agent.agents.transport_agent._parse_route_details",
        return_value={
            "board_station": "Rossio",
            "final_station": "Aeroporto",
            "transfer_station": None,
            "directions": ["Aeroporto"],
            "estimated_time": "~25 min",
            "walk_target": None,
        },
    ), patch(
        "agent.agents.transport_agent._get_line_id_between",
        return_value="verde",
    ), patch(
        "agent.agents.transport_agent._build_route_state_lines",
        return_value=["- 🟢 **Linha Verde**: circulação normal"],
    ), patch(
        "agent.agents.transport_agent._build_metro_wait_lines",
        wait_lines,
    ), patch(
        "agent.agents.transport_agent._build_practical_tip",
        return_value="",
    ), patch(
        "agent.agents.transport_agent._build_additional_route_options",
        return_value=[],
    ), patch(
        "agent.agents.transport_agent._get_transport_display_name",
        side_effect=lambda value, detailed=False: value,
    ), patch(
        "tools.metrolisboa_api.get_landmark_info",
        return_value=None,
    ), patch(
        "tools.metrolisboa_api.METRO_LINES",
        {"verde": {"emoji": "🟢"}},
    ):
        output = _build_deterministic_metro_route_response(
            "Como vou amanhã do Rossio ao Aeroporto de metro?",
            context="User language: pt",
        )

    wait_lines.assert_not_called()
    assert output is not None
    assert "Planeamento futuro" in output
    assert "Próximos Metros" not in output
    assert "circulação normal" not in output


def test_planner_multi_day_request_injects_quality_mode_instruction() -> None:
    """Multi-day planner queries should explicitly activate the day-by-day quality mode."""
    with patch.object(PlannerAgent, "__init__", lambda self: None), patch(
        "agent.agents.planner_agent.finalize_worker_response",
        side_effect=lambda response, **_kwargs: response,
    ):
        agent = PlannerAgent()
        agent.system_prompt = "PLANNER PROMPT"
        agent.llm = object()
        agent._safe_llm_invoke = MagicMock(return_value=SimpleNamespace(content="### 📅 Day 1\n- Grounded content"))

        output = agent.invoke(
            user_message="Plan 3 days in Lisbon for me.",
            weather_data="🌤️ Dry weather",
            transport_data="🚇 Metro available",
            places_data="- **Jerónimos Monastery**",
            events_data="",
        )

    sent_messages = agent._safe_llm_invoke.call_args.args[1]
    system_messages = [message.content for message in sent_messages if hasattr(message, "content")]
    assert any("MULTI-DAY QUALITY MODE" in content for content in system_messages)
    assert output.startswith("### 📅 Day 1")


def test_enforce_multi_day_quality_mode_trims_later_days() -> None:
    """The deterministic clamp should remove Day 2+ content from explicit multi-day responses."""
    draft = (
        "### 📅 Plano de 3 dias em Lisboa\n"
        "- 📅 Dia 1\n"
        "- Museu Nacional do Azulejo\n"
        "- 📅 Dia 2\n"
        "- MAAT\n"
        "- 📅 Dia 3\n"
        "- Museu do Chiado"
    )

    output = enforce_multi_day_quality_mode(
        response=draft,
        user_message="Planeia 3 dias em Lisboa.",
        language="pt",
    )

    assert output.startswith("### 📅 Dia 1 · Itinerário Sugerido")
    assert "\n- 📅 Dia 2" not in output
    assert "\n- 📅 Dia 3" not in output
    assert "Museu Nacional do Azulejo" in output
    assert "este primeiro bloco cobre apenas o Dia 1" in output


def test_enforce_multi_day_quality_mode_trims_bold_day_two_headers() -> None:
    """The multi-day clamp should catch bold markdown variants for Day 2+ sections."""
    draft = (
        "### 📅 3-day Lisbon plan\n"
        "- **Day 1:** Belém\n"
        "- Jerónimos Monastery\n"
        "- **Day 2:** Alfama\n"
        "- São Jorge Castle"
    )

    output = enforce_multi_day_quality_mode(
        response=draft,
        user_message="Plan 3 days in Lisbon for me.",
        language="en",
    )

    assert output.startswith("### 📅 Day 1 · Suggested Itinerary")
    assert "**Day 2:**" not in output
    assert "Jerónimos Monastery" in output
    assert "ask me next for Day 2" in output


def test_enforce_multi_day_quality_mode_trims_shorthand_day_markers() -> None:
    """The multi-day clamp should also catch compact D2/D3 section markers."""
    draft = (
        "### 📅 Lisbon plan\n"
        "- Day 1 · Baixa\n"
        "- Rossio and Chiado\n"
        "- D2: Sintra\n"
        "- Pena Palace"
    )

    output = enforce_multi_day_quality_mode(
        response=draft,
        user_message="Plan 2 days in Lisbon for me.",
        language="en",
    )

    assert "D2:" not in output
    assert "Rossio and Chiado" in output


def test_planner_formatter_keeps_schedule_lines_out_of_timed_cards() -> None:
    """Planner finalization must not turn schedule metadata into fake 01:00/03:00 activities."""
    draft = (
        "### 📅 Recomendação para domingo, entre as 19:00 e as 20:00\n"
        "- 🕐 Próximas saídas do Rossio\n"
        "- 🕒 Domingo: 10:00–19:00 ou 10:00–18:00\n"
        "- 🕒 Domingo: 10:00–17:00\n"
        "- 🏛️ Monument to the Discoveries\n"
        "📌 **Fonte:** [*VisitLisboa*](https://www.visitlisboa.com) | **Atualizado:** 10:25"
    )

    output = finalize_worker_response(
        draft,
        agent_name="planner",
        user_query="Qual museu ou monumento recomendas ir neste domingo sendo que apenas tenho das 19 às 20h para visitar?",
        language="pt",
    )

    assert "01:00 · Próximas saídas" not in output
    assert "03:00 · Domingo:" not in output
    assert "Monument to the Discoveries" in output


def test_transport_deterministic_tool_call_records_tool_usage() -> None:
    """Deterministic transport fast paths should still populate the tool-call log."""
    with patch.object(TransportAgent, "__init__", lambda self: None), patch(
        "agent.agents.transport_agent.finalize_worker_response",
        side_effect=lambda response, **_kwargs: response,
    ):
        agent = TransportAgent()
        agent._tool_calls_log = []
        tool = MagicMock()
        tool.invoke = MagicMock(return_value="ok")
        agent._get_tool_by_name = MagicMock(return_value=tool)

        output = agent._invoke_deterministic_tool_call(
            "What are the direct Carris Metropolitana buses from Oeiras to Amadora?",
            language="en",
        )

    assert output is not None
    assert "ok" in output
    tool.invoke.assert_called_once()
    assert agent.get_tool_calls_log()[0]["tool_name"] == "find_direct_bus_lines"


def test_transport_formats_direct_carris_metropolitana_output() -> None:
    """Direct Carris Metropolitana fast paths should strip raw wrapper/scope lines."""
    raw_result = (
        "🚌 **Buses: Oeiras → Amadora**\n\n"
        "✅ **19 direct line(s) found:**\n\n"
        "**1. 🚍 Linha 1501**\n"
        " 📍 **Terminals**: Reboleira (Estação) | Circular via Alfragide\n"
        "💡 **How to use it:**\n"
        " - Look for the line number\n"
        "⚠️ **Scope**: raw wrapper that should disappear"
    )

    with patch.object(TransportAgent, "__init__", lambda self: None):
        agent = TransportAgent()
        formatted = agent._format_deterministic_tool_result(
            tool_name="find_direct_bus_lines",
            tool_args={"origin": "Oeiras", "destination": "Amadora"},
            result=raw_result,
            language="en",
        )

    assert formatted.startswith("### 🚌 Direct Carris Metropolitana lines for Oeiras → Amadora")
    assert "How to use it" in formatted
    assert "Scope" not in formatted
    assert "Carris Metropolitana" in formatted


def test_transport_formats_no_direct_carris_metropolitana_output_concisely() -> None:
    """No-direct Carris Metropolitana results should collapse into a short grounded summary."""
    raw_result = (
        "🚌 **BUS ROUTE FINDER**\n"
        "==================================================\n"
        "📍 From: Rossio\n"
        "📍 To: Museu Nacional de Arte Antiga\n"
        "❌ **No direct bus routes found**\n\n"
        "📊 **Lines available near your locations:**\n\n"
        "   At Rossio: 1002, 1013, 1510\n"
        "   At Museu Nacional de Arte Antiga: 3708\n\n"
        "⚠️ **IMPORTANT: Carris Metropolitana Scope Note**\n"
        "For Lisbon city-only trips, always cross-check Carris Urban when relevant\n"
    )

    with patch.object(TransportAgent, "__init__", lambda self: None):
        agent = TransportAgent()
        formatted = agent._format_deterministic_tool_result(
            tool_name="find_direct_bus_lines",
            tool_args={"origin": "Rossio", "destination": "Museu Nacional de Arte Antiga"},
            result=raw_result,
            language="pt",
        )

    assert formatted.startswith(
        "### 🚌 Linhas diretas da Carris Metropolitana para Rossio → Museu Nacional de Arte Antiga"
    )
    assert "Sem linha suburbana direta confirmada" in formatted
    assert "Linhas disponíveis perto da origem" in formatted
    assert "Linhas disponíveis perto do destino" in formatted
    assert "Carris Urban" in formatted
    assert "BUS ROUTE FINDER" not in formatted
    assert "Suggestions" not in formatted


def test_researcher_follow_up_paginates_the_next_event_batch() -> None:
    """Researcher event follow-ups should advance the offset instead of repeating the first batch."""

    class DummyEventsTool:
        name = "search_cultural_events"

        def __init__(self) -> None:
            self.calls: list[dict] = []

        def invoke(self, args: dict) -> str:
            self.calls.append(dict(args))
            offset = int(args.get("offset", 0) or 0)
            if offset == 0:
                return "1. 📅 **Event A**\n2. 📅 **Event B**"
            if offset == 2:
                return "1. 📅 **Event C**\n2. 📅 **Event D**"
            return "❌ There are no more events to show for this filter window."

    with patch.object(ResearcherAgent, "__init__", lambda self: None):
        agent = ResearcherAgent()
        agent.tools = [DummyEventsTool()]
        agent._last_search_context = None

        first_batch = agent._run_direct_event_lookup(
            "What major events are happening this week?",
            "en",
        )
        second_batch = agent._maybe_continue_previous_search(
            "Give me the next 2 events that match",
            "en",
        )

    tool = agent.tools[0]
    assert "Event A" in first_batch and "Event B" in first_batch
    assert second_batch is not None
    assert "Event C" in second_batch and "Event D" in second_batch
    assert "Event A" not in second_batch
    assert tool.calls[0]["offset"] == 0
    assert tool.calls[1]["offset"] == 2


def test_researcher_follow_up_paginates_specific_place_fallback_with_more_options() -> None:
    """Specific place fallback batches should continue from the next concise window when the user asks for more options."""

    class DummyPlacesTool:
        name = "search_places_attractions"

        def __init__(self) -> None:
            self.calls: list[dict] = []

        def invoke(self, args: dict) -> str:
            self.calls.append(dict(args))
            offset = int(args.get("offset", 0) or 0)
            if offset == 0:
                return (
                    "❌ Não encontrei um local específico com o nome **museu do livro** na base de dados disponível. "
                    "Como alternativa, deixo abaixo locais do mesmo tipo, estilo ou afinidade temática.\n\n"
                    "🏛️ **Found 2 Places/Attractions in Lisbon:**\n"
                    "🧭 **Janela de resultados:** 1-2 de 4.\n\n"
                    "1. 🏛️ **Words Factory**\n"
                    "   📂 Category: Museums & Monuments\n"
                    "   📍 Lisboa\n"
                    "   🔗 https://example.com/words-factory\n\n"
                    "2. 🏛️ **João de Deus Museum**\n"
                    "   📂 Category: Museums & Monuments\n"
                    "   📍 Lisboa\n"
                    "   🔗 https://example.com/joao-de-deus-museum\n"
                )
            if offset == 2:
                return (
                    "🏛️ **Found 2 Places/Attractions in Lisbon:**\n"
                    "🧭 **Janela de resultados:** 3-4 de 4.\n\n"
                    "1. 🏛️ **Museum of Illusions**\n"
                    "   📂 Category: Museums & Monuments\n"
                    "   📍 Lisboa\n"
                    "   🔗 https://example.com/museum-of-illusions\n\n"
                    "2. 🏛️ **National Museum of Sport**\n"
                    "   📂 Category: Museums & Monuments\n"
                    "   📍 Lisboa\n"
                    "   🔗 https://example.com/national-museum-of-sport\n"
                )
            return "❌ Já não há mais locais para mostrar com este filtro."

    with patch.object(ResearcherAgent, "__init__", lambda self: None):
        agent = ResearcherAgent()
        agent.tools = [DummyPlacesTool()]
        agent._last_search_context = None

        first_batch = agent._run_direct_place_lookup("Onde fica o Museu do Livro?", "pt")
        second_batch = agent._maybe_continue_previous_search(
            "Não me interessei por nenhum, quero mais opções que possa ir de outros museus",
            "pt",
        )

    tool = agent.tools[0]
    assert "Words Factory" in first_batch and "João de Deus Museum" in first_batch
    assert second_batch is not None
    assert "Museum of Illusions" in second_batch and "National Museum of Sport" in second_batch
    assert "Words Factory" not in second_batch
    assert tool.calls[0]["offset"] == 0
    assert tool.calls[1]["offset"] == 2
    assert tool.calls[1]["max_results"] == 2


def test_researcher_same_follow_up_retry_reuses_the_same_paginated_batch_once() -> None:
    """If QA retries the same paginated follow-up in the same turn, the researcher should replay the same batch once instead of skipping ahead."""

    class DummyPlacesTool:
        name = "search_places_attractions"

        def __init__(self) -> None:
            self.calls: list[dict] = []

        def invoke(self, args: dict) -> str:
            self.calls.append(dict(args))
            offset = int(args.get("offset", 0) or 0)
            if offset == 0:
                return (
                    "❌ Não encontrei um local específico com o nome **museu do livro** na base de dados disponível.\n\n"
                    "1. 🏛️ **Words Factory**\n"
                    "2. 🏛️ **João de Deus Museum**"
                )
            if offset == 2:
                return (
                    "🏛️ **Found 2 Places/Attractions in Lisbon:**\n"
                    "🧭 **Janela de resultados:** 3-4 de 6.\n\n"
                    "1. 🏛️ **Museum of Illusions**\n"
                    "2. 🏛️ **Museu da Marioneta**"
                )
            if offset == 4:
                return (
                    "🏛️ **Found 2 Places/Attractions in Lisbon:**\n"
                    "🧭 **Janela de resultados:** 5-6 de 6.\n\n"
                    "1. 🏛️ **National Museum of Sport**\n"
                    "2. 🏛️ **Portuguese Museum of Freemasonry**"
                )
            return "❌ Já não há mais locais para mostrar com este filtro."

    with patch.object(ResearcherAgent, "__init__", lambda self: None):
        agent = ResearcherAgent()
        agent.tools = [DummyPlacesTool()]
        agent._last_search_context = None
        agent._pending_pagination_replay = None

        agent._run_direct_place_lookup("Onde fica o Museu do Livro?", "pt")
        first_follow_up = agent._maybe_continue_previous_search("Quero mais opções de museus", "pt")
        replayed_follow_up = agent._maybe_continue_previous_search("Quero mais opções de museus", "pt")

    tool = agent.tools[0]
    assert first_follow_up == replayed_follow_up
    assert "Museum of Illusions" in replayed_follow_up
    assert "National Museum of Sport" not in replayed_follow_up
    assert [call["offset"] for call in tool.calls] == [0, 2]


def test_researcher_same_named_place_retry_reuses_cached_deterministic_response_once() -> None:
    """A same-message retry of a deterministic place lookup should replay the cached response once."""
    with patch.object(ResearcherAgent, "__init__", lambda self: None):
        agent = ResearcherAgent()
        agent.system_prompt = "RESEARCHER PROMPT"
        agent._last_search_context = None
        agent._pending_pagination_replay = None
        agent._pending_deterministic_replay = None

        places_tool = MagicMock()
        places_tool.name = "search_places_attractions"
        places_tool.invoke = MagicMock(
            return_value=(
                "❌ Não encontrei um local específico com o nome **museu do livro** na base de dados disponível.\n\n"
                "1. 🏛️ **Words Factory**\n"
                "   📂 Category: Museums & Monuments\n"
                "   📍 Lisboa\n"
                "   🔗 https://example.com/words-factory\n"
            )
        )
        agent.tools = [places_tool]
        agent.execute_react_loop = MagicMock(side_effect=AssertionError("LLM flow should be skipped"))

        first_response = agent.invoke("Onde fica o Museu do Livro?", context="User language: pt", verbose=False)
        replayed_response = agent.invoke("Onde fica o Museu do Livro?", context="User language: pt", verbose=False)

    assert first_response == replayed_response
    assert places_tool.invoke.call_count == 1


def test_transport_follow_up_reuses_previous_route_for_mode_switch() -> None:
    """Short transport follow-ups like 'And by metro?' should inherit the last route endpoints."""
    with patch.object(TransportAgent, "__init__", lambda self: None):
        agent = TransportAgent()
        agent._last_transport_context = {
            "origin": "Rossio",
            "destination": "Oriente",
            "last_user_query": "How do I get from Rossio to Oriente?",
            "mode": None,
        }

        rewritten = agent._rewrite_follow_up_transport_query("And by metro?", "en")

    assert rewritten == "How do I get from Rossio to Oriente by metro?"


def test_multiagent_message_history_is_pruned_to_recent_window() -> None:
    """Conversation state should keep a bounded recent window instead of growing forever."""
    assistant = MultiAgentAssistant.__new__(MultiAgentAssistant)
    assistant.state = {
        "messages": [HumanMessage(content=f"msg {index}") for index in range(59)],
        "user_context": None,
    }

    assistant._append_user_message("latest user")
    assistant._append_assistant_message("latest assistant")

    assert len(assistant.state["messages"]) == 60
    assert assistant.state["messages"][-2].content == "latest user"
    assert assistant.state["messages"][-1].content == "latest assistant"
    assert assistant.state["messages"][0].content == "msg 1"
