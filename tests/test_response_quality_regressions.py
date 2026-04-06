import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.agents.qa_agent import QualityAssuranceAgent
from agent.agents.researcher_agent import ResearcherAgent
from agent.agents.transport_agent import TransportAgent
from agent.graph import MultiAgentAssistant
from agent.utils.langsmith_tracing import get_langsmith_request_tracking_status
from agent.utils.response_formatter import finalize_worker_response, infer_response_language


def test_infer_response_language_prefers_english_query_even_with_pt_default() -> None:
    """The effective reply language should follow the user's English query, not the PT UI default."""
    assert infer_response_language(
        user_query="Tell me about Book Fair 2026",
        default="pt",
    ) == "en"


def test_multiagent_chat_routes_using_effective_query_language() -> None:
    """Multi-agent orchestration should route and finalize using the detected query language."""
    assistant = MultiAgentAssistant.__new__(MultiAgentAssistant)
    assistant.state = {"messages": [], "user_context": None}
    assistant.supervisor = MagicMock()
    assistant.supervisor.reset_llm_usage_tracking = MagicMock()
    assistant.supervisor.route = MagicMock(
        return_value={"agents": [], "direct_response": "Book Fair response", "reasoning": "direct"}
    )
    assistant.qa_agent = MagicMock()
    assistant.qa_agent.reset_llm_usage_tracking = MagicMock()
    assistant.agents = {}
    assistant._finalize_chat_response = MagicMock(return_value="ok")

    with patch("agent.graph.LANGSMITH_AVAILABLE", False):
        result = assistant.chat("Tell me about Book Fair 2026", language="pt")

    assert result == "ok"
    assert assistant.supervisor.route.call_args.kwargs["language"] == "en"
    assert assistant._finalize_chat_response.call_args.kwargs["language"] == "en"
    user_context = assistant.state["user_context"]
    assert user_context is not None
    assert user_context.get("language") == "en"
    assert user_context.get("ui_language") == "pt"


def test_researcher_worker_formats_nearby_services_as_structured_cards() -> None:
    """Raw nearby-service dumps should be normalized into the same structured researcher style."""
    raw = (
        "\U0001F4CD Found 2 results from 'Farm\u00e1cias e Parafarm\u00e1cias (near Saldanha)':\n\n"
        "1. Farm\u00e1cia Dalva\n"
        "   \U0001F4CD Avenida Duque d'\u00c1vila, 125\n"
        "   \U0001F4CF 0.07 km away\n"
        "   \U0001F5FA\uFE0F (38.735010, -9.145924)\n\n"
        "2. Farm\u00e1cia Duque de \u00c1vila\n"
        "   \U0001F4CD Avenida Duque d'\u00c1vila 32C-D\n"
        "   \U0001F4CF 0.08 km away\n"
        "   \U0001F5FA\uFE0F (38.735301, -9.144639)\n"
    )

    output = finalize_worker_response(
        raw,
        agent_name="researcher",
        user_query="Qual a farm\u00e1cia mais perto do Saldanha?",
        language="pt",
    )

    assert "####" in output
    assert "**Farm\u00e1cia Dalva**" in output
    assert "**Morada:**" in output
    assert "0.07 km away" in output
    assert "Lisboa Aberta" in output


def test_researcher_direct_place_lookup_covers_multiple_requested_services() -> None:
    """Direct nearby-service lookups should answer every requested service component."""
    with patch.object(ResearcherAgent, "__init__", lambda self: None):
        agent = ResearcherAgent()

        nearby_tool = MagicMock()
        nearby_tool.name = "find_nearby_services"
        nearby_tool.invoke = MagicMock(
            side_effect=[
                "\U0001F4CD Found 1 results from 'Hospitais (near Saldanha)':\n\n1. Hospital Curry Cabral\n   \U0001F4CD Rua Benefic\u00eancia\n",
                "\U0001F4CD Found 1 results from 'Farm\u00e1cias e Parafarm\u00e1cias (near Saldanha)':\n\n1. Farm\u00e1cia Dalva\n   \U0001F4CD Avenida Duque d'\u00c1vila, 125\n",
            ]
        )
        agent.tools = [nearby_tool]

        result = agent._run_direct_place_lookup(
            "Qual o hospital e a farm\u00e1cia mais perto do Saldanha?",
            "pt",
        )

        assert nearby_tool.invoke.call_count == 2
        assert "Hospital Curry Cabral" in result
        assert "Farm\u00e1cia Dalva" in result
        assert "Lisboa Aberta" in result


def test_transport_agent_compares_metro_and_train_and_states_fare_limitation() -> None:
    """Mode-comparison queries should answer fastest/cheapest explicitly instead of returning only one mode."""
    with patch.object(TransportAgent, "__init__", lambda self: None):
        agent = TransportAgent()
        agent.system_prompt = "TRANSPORT PROMPT"
        agent.execute_react_loop = MagicMock(side_effect=AssertionError("LLM path should be skipped"))

        trip_tool = MagicMock()
        trip_tool.name = "plan_train_trip"
        trip_tool.invoke = MagicMock(
            return_value=(
                "\U0001F686 **Comboio: Entrecampos \u2192 Sete Rios**\n"
                "\U0001F4CA **RESUMO DA VIAGEM**\n"
                "   \U0001F686 Linhas: **Linha de Sintra, IC**\n"
                "   \u23F1\uFE0F Dura\u00e7\u00e3o: **3 minutos**\n"
                "\U0001F4CB **Pr\u00f3ximas 3 Partidas:**\n"
                "   \U0001F550 **16:17** \u2192 16:20 (3min)\n"
                "   \U0001F550 **16:47** \u2192 16:50 (3min)\n"
            )
        )
        agent.tools = [trip_tool]

        with patch(
            "agent.agents.transport_agent._build_deterministic_metro_route_response",
            return_value=(
                "\U0001F687 **Entrecampos** \u2192 **Sete Rios**\n"
                "\u23F3 **Tempo total estimado:** 8 min\n"
            ),
        ):
            result = agent.invoke(
                "Quero ir de metro ou comboio entre Entrecampos e Sete Rios? Qual o mais r\u00e1pido e o mais barato?"
            )

        assert "Mais r\u00e1pido" in result
        assert "Comboio" in result
        assert "Mais barato" in result
        assert "n\u00e3o foi poss\u00edvel confirmar" in result.lower()
        assert "Metro de Lisboa" in result
        assert "CP" in result


def test_qa_augments_missing_components_for_mode_comparison_queries() -> None:
    """QA should flag incomplete metro-vs-train comparisons instead of silently approving them."""
    result = QualityAssuranceAgent._augment_query_specific_validation(
        user_query="Quero ir de metro ou comboio entre Entrecampos e Sete Rios? Qual o mais r\u00e1pido e o mais barato?",
        agent_outputs={"transport": "\U0001F686 **Comboio: Entrecampos \u2192 Sete Rios**\n\u23F1\uFE0F Dura\u00e7\u00e3o: **3 minutos**"},
        llm_result={
            "complete": True,
            "missing_data": [],
            "required_agents": [],
            "reasoning": "",
            "disclaimers": [],
        },
        language="pt",
    )

    assert result["complete"] is False
    assert "transport" in result["required_agents"]
    assert any("metro" in item.lower() for item in result["missing_data"])
    assert any("mais barata" in item.lower() or "tarifa" in item.lower() for item in result["missing_data"])


def test_langsmith_request_tracking_surfaces_runtime_failure_message() -> None:
    """Per-request tracking should expose the exact runtime persistence failure when one was captured."""
    with patch("agent.utils.langsmith_tracing.get_last_langsmith_runtime_failure") as mocked_failure:
        mocked_failure.return_value = {
            "persistence_state": "failed_remote_quota",
            "message": "LangSmith API error: monthly credits exhausted",
        }

        class FakeRunTree:
            id = "run_123"

        tracking = get_langsmith_request_tracking_status(
            status={
                "enabled": True,
                "requested": True,
                "reason": "LangSmith tracing enabled",
                "project_name": "LISBOA Chat",
            },
            run_tree=FakeRunTree(),
        )

    assert tracking["tracking_state"] == "tracking_request_failed_remote"
    assert tracking["persistence_state"] == "failed_remote_quota"
    assert "credits exhausted" in tracking["note"].lower()


def test_execution_summary_prints_langsmith_runtime_failure_note(capsys) -> None:
    """Execution summaries should surface the exact LangSmith persistence failure when known."""
    assistant = MultiAgentAssistant.__new__(MultiAgentAssistant)

    assistant._print_execution_summary(
        {
            "elapsed_time": 1.23,
            "execution_type": "single-worker",
            "worker_mode": "sequential",
            "qa_path": "validated",
            "langsmith": {
                "tracking_state": "tracking_request_failed_remote",
                "status_label": "enabled",
                "save_attempted": True,
                "persistence_state": "failed_remote_quota",
                "current_run_attached": True,
                "project_name": "LISBOA Chat",
                "run_id": "run_123",
                "reason": "LangSmith tracing enabled",
                "note": "LangSmith API error: monthly credits exhausted",
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
            "relevant_agents": [],
            "agent_usage": {},
            "agent_costs": {},
            "agent_tool_logs": {},
            "total_tool_invocations": 0,
            "retry_agents_used": [],
        }
    )

    captured = capsys.readouterr().out
    assert "Persistence:" in captured
    assert "credits exhausted" in captured.lower()
