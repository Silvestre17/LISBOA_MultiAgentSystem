# ===========================================================================
# Master Thesis
#   - André Filipe Gomes Silvestre, 20240502
#
# Regression tests for transport-agent deterministic fast paths, invoke/build-
# subgraph parity helpers, and structured multi-agent rendering.
#
# Run from the repository root with a relative path:
#   python -m pytest tests/test_transport_parity_and_rendering.py -q
# Useful parameters:
#   -vv         verbose mode
#   -k cp or -k carris   focus on one transport family
#   -x          stop on first failure
#   --tb=short  shorter tracebacks
# Notes:
#   - Prefer relative paths in this workspace. Absolute pytest paths may be
#     treated as glob patterns on Windows because the folder name includes
#     `[` and `]`.
# ===========================================================================

# Required libraries:
# pip install pytest

import os
import sys
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.agents import transport_agent as transport_agent_module
from agent.agents.transport_agent import (
    TransportAgent,
    _build_deterministic_metro_route_response,
    _build_deterministic_transport_tool_call,
    _build_metro_wait_lines,
    _extract_route_endpoints,
    _parse_metro_wait_request,
)
from agent.graph import MultiAgentAssistant
from agent.state import create_initial_state
from tools import carris_api, cp_api, metrolisboa_api, transport_api


def _extract_tool_call_spec(message) -> dict:
    """Extract the first tool-call spec from an AIMessage helper result."""
    assert message is not None
    assert getattr(message, "tool_calls", None)
    return message.tool_calls[0]


def test_transport_tool_call_builder_maps_cp_schedule_query() -> None:
    """Natural CP schedule questions should map to the train-schedule tool with the right station."""
    spec = _extract_tool_call_spec(
        _build_deterministic_transport_tool_call("When are the next trains from Entrecampos?")
    )

    assert spec["name"] == "get_train_schedule"
    assert spec["args"] == {"station_name": "Entrecampos"}


def test_transport_tool_call_builder_maps_cp_trip_query() -> None:
    """Natural CP trip questions should map to plan_train_trip with origin and destination."""
    spec = _extract_tool_call_spec(
        _build_deterministic_transport_tool_call("How do I get from Rossio to Sintra by train?")
    )

    assert spec["name"] == "plan_train_trip"
    assert spec["args"] == {"origin": "Rossio", "destination": "Sintra"}


def test_extract_route_endpoints_handles_pt_ate_a_metro_phrase() -> None:
    """PT-PT metro phrasings with `até à` should parse clean endpoints."""
    endpoints = _extract_route_endpoints("Quero ir de metro de Entrecampos até à NOVA IMS")

    assert endpoints == ("Entrecampos", "NOVA IMS")


def test_extract_route_endpoints_handles_colloquial_pt_origin_then_destination_phrase() -> None:
    """Colloquial PT route questions should keep the real origin instead of drifting to another landmark."""
    endpoints = _extract_route_endpoints(
        "Tou no Rossio e preciso de ir ao Estádio da Luz sem complicações, qual é o melhor caminho?"
    )

    assert endpoints == ("Rossio", "Estádio da Luz")


def test_extract_route_endpoints_handles_reverse_a_partir_do_phrase() -> None:
    """Reverse PT phrasings with `a partir do` should recover origin and destination cleanly."""
    endpoints = _extract_route_endpoints(
        "Quero ir para o MEO Arena a partir do Saldanha, o metro serve bem?"
    )

    assert endpoints == ("Saldanha", "MEO Arena")


def test_transport_tool_call_builder_maps_cm_direct_bus_query() -> None:
    """Suburban direct-bus requests should map to Carris Metropolitana direct-line lookup."""
    spec = _extract_tool_call_spec(
        _build_deterministic_transport_tool_call(
            "What are the direct Carris Metropolitana buses from Oeiras to Amadora?"
        )
    )

    assert spec["name"] == "find_direct_bus_lines"
    assert spec["args"] == {"origin": "Oeiras", "destination": "Amadora"}


def test_transport_tool_call_builder_maps_cm_live_positions_query() -> None:
    """Carris Metropolitana proximity queries should map to real-time position lookup."""
    spec = _extract_tool_call_spec(
        _build_deterministic_transport_tool_call(
            "Show real-time Carris Metropolitana buses near Almada"
        )
    )

    assert spec["name"] == "get_real_time_bus_positions"
    assert spec["args"] == {"location": "Almada", "radius_km": 1.0}


def test_transport_tool_call_builder_maps_metro_status_paraphrase() -> None:
    """Metro status routing should work for paraphrases, not only fixed manifest strings."""
    spec = _extract_tool_call_spec(
        _build_deterministic_transport_tool_call(
            "What's the current status of Lisbon metro lines right now?"
        )
    )

    assert spec["name"] == "get_metro_status"
    assert spec["args"] == {}


def test_transport_tool_call_builder_maps_nearest_metro_coordinates_query() -> None:
    """Nearest-metro routing should extract coordinates from natural GPS phrasings."""
    spec = _extract_tool_call_spec(
        _build_deterministic_transport_tool_call(
            "Which metro station is nearest to GPS coordinates 38.725, -9.149?"
        )
    )

    assert spec["name"] == "find_nearest_metro"
    assert spec["args"] == {"latitude": 38.725, "longitude": -9.149}


def test_transport_tool_call_builder_maps_carris_route_info_paraphrase() -> None:
    """Carris urban route info should tolerate paraphrases and spaced tram codes."""
    spec = _extract_tool_call_spec(
        _build_deterministic_transport_tool_call(
            "Can you show route details for tram 28 E?"
        )
    )

    assert spec["name"] == "carris_get_routes"
    assert spec["args"] == {"route_id": "28E"}


def test_parse_metro_wait_request_fuzzy_resolves_station_typos() -> None:
    """Metro wait parsing should resolve small typos in station and direction names."""
    request = _parse_metro_wait_request(
        "When is the next metro at Saldahna towards Odivela?"
    )

    assert request == {
        "station": "Saldanha",
        "direction": "Odivelas",
        "status_requested": False,
    }


def test_deterministic_metro_route_response_uses_landmark_aware_tip_and_two_wait_targets() -> None:
    """Metro route fast paths should keep factual structure while sounding aware of landmark destinations like NOVA IMS."""
    route_result = (
        "🗺️ **Route: Entrecampos → Nova Ims**\n\n"
        "📍 **LOCATION INFORMATION**\n"
        "**NOVA IMS - Information Management School**\n"
        "   🚇 Nearest Metro: **São Sebastião** (🔵 Azul/Vermelha Line)\n"
        "   ℹ️ Information Management School at Universidade NOVA de Lisboa's Campolide campus\n\n"
        "🚇 **METRO ROUTE**\n"
        "🔄 **Transfer Required**\n\n"
        "   💡 **Transfer at**: Saldanha (🟡 ↔ 🔴)\n"
        "   ⏱️ Estimated travel time: **~11 min** (3 stations + 1 transfer)\n\n"
        "   **Full Route**:\n"
        "   1. 🟡 Board at **Entrecampos** → direção **Rato**\n"
        "   2. Exit at **Saldanha**\n"
        "   3. 🔴 Transfer to **Red Line (S. Sebastião ↔ Aeroporto)** → direção **São Sebastião**\n"
        "   4. Exit at **São Sebastião**\n"
        "   5. Walk to Nova Ims\n\n"
        "📌 **Fonte:** [*Metro de Lisboa*](https://www.metrolisboa.pt) **| Atualizado:** 19:54\n"
    )

    def fake_wait_invoke(args: dict) -> str:
        if args == {"station": "Entrecampos", "direction": "Rato"}:
            return (
                "🚇 Metro Wait Times at Entre Campos\n"
                "==================================================\n\n"
                "🟡 Direction: Rato\n"
                "   ⏱️ Next train: 2 min\n"
                "   ⏳ Following: 5 min, 8 min\n\n"
                "📍 Updated: 19:54:10"
            )
        if args == {"station": "Saldanha", "direction": "São Sebastião"}:
            return (
                "🚇 Metro Wait Times at Saldanha\n"
                "==================================================\n\n"
                "🔴 Direction: São Sebastião\n"
                "   ⏱️ Next train: 3 min\n"
                "   ⏳ Following: 7 min, 10 min\n\n"
                "📍 Updated: 19:54:10"
            )
        raise AssertionError(f"Unexpected wait query: {args}")

    route_tool = MagicMock()
    route_tool.invoke = MagicMock(return_value=route_result)

    wait_tool = MagicMock()
    wait_tool.invoke = MagicMock(side_effect=fake_wait_invoke)

    with patch.object(transport_api, "get_route_between_stations", route_tool), patch.object(
        transport_api,
        "_get_line_status",
        return_value="Ok",
    ), patch.object(metrolisboa_api, "get_metro_wait_time", wait_tool):
        response = _build_deterministic_metro_route_response(
            "Quero ir de metro de Entrecampos até à NOVA IMS",
            "",
        )

    assert response is not None
    assert "🚇 **Entrecampos** → **NOVA IMS**" in response
    assert "Siga a pé para NOVA IMS" in response
    assert "NOVA IMS - Information Management School" in response
    assert "cerca de 6 min a pé" in response
    assert "**Estação Entrecampos:** Direção Rato" in response
    assert "**Estação Saldanha:** Direção São Sebastião" in response
    assert "Outras opções" not in response


def test_route_tool_recognizes_common_hospital_and_stadium_landmarks() -> None:
    """Generic landmark routing should understand common hospitals and stadiums, not just one custom campus."""
    with patch.object(transport_api, "_get_line_status", return_value="Ok"):
        hospital_route = transport_api.get_route_between_stations.invoke(
            {"origin": "Saldanha", "destination": "Hospital Santa Maria"}
        )
        stadium_route = transport_api.get_route_between_stations.invoke(
            {"origin": "Rossio", "destination": "Estádio da Luz"}
        )

    assert "Hospital de Santa Maria" in hospital_route
    assert "Cidade Universitária" in hospital_route
    assert "Estádio da Luz" in stadium_route
    assert "Colégio Militar/Luz" in stadium_route


def test_transport_tool_selects_shortest_transfer_hub_for_entrecampos_to_nova_ims() -> None:
    """The multimodal route tool should prefer the shorter Saldanha transfer over the first matching interchange."""
    with patch.object(transport_api, "_get_line_status", return_value="Ok"):
        result = transport_api.get_route_between_stations.invoke(
            {"origin": "Entrecampos", "destination": "NOVA IMS"}
        )

    assert "**Transfer at**: Saldanha" in result
    assert "**~11 min**" in result


def test_deterministic_generic_route_response_mentions_train_and_bus_alternatives_when_sensible() -> None:
    """Open-ended route questions should keep the best metro path while surfacing factual train and bus alternatives."""
    route_result = (
        "🗺️ **Route: Entrecampos → Nova Ims**\n\n"
        "📍 **LOCATION INFORMATION**\n"
        "**NOVA IMS - Information Management School**\n"
        "   🚇 Nearest Metro: **São Sebastião** (🔵 Azul/Vermelha Line)\n"
        "   ℹ️ Information Management School at Universidade NOVA de Lisboa's Campolide campus\n\n"
        "🚇 **METRO ROUTE**\n"
        "🔄 **Transfer Required**\n\n"
        "   💡 **Transfer at**: Saldanha (🟡 ↔ 🔴)\n"
        "   ⏱️ Estimated travel time: **~11 min** (3 stations + 1 transfer)\n\n"
        "   **Full Route**:\n"
        "   1. 🟡 Board at **Entrecampos** → direção **Rato**\n"
        "   2. Exit at **Saldanha**\n"
        "   3. 🔴 Transfer to **Red Line (S. Sebastião ↔ Aeroporto)** → direção **São Sebastião**\n"
        "   4. Exit at **São Sebastião**\n"
        "   5. Walk to Nova Ims\n\n"
        "📌 **Fonte:** [*Metro de Lisboa*](https://www.metrolisboa.pt) **| Atualizado:** 19:54\n"
    )

    def fake_wait_invoke(args: dict) -> str:
        if args == {"station": "Entrecampos", "direction": "Rato"}:
            return (
                "🚇 Metro Wait Times at Entrecampos\n"
                "==================================================\n\n"
                "🟡 Direction: Rato\n"
                "   ⏱️ Next train: 2 min\n"
                "   ⏳ Following: 5 min, 8 min\n\n"
                "📍 Updated: 19:54:10"
            )
        if args == {"station": "Saldanha", "direction": "São Sebastião"}:
            return (
                "🚇 Metro Wait Times at Saldanha\n"
                "==================================================\n\n"
                "🔴 Direction: São Sebastião\n"
                "   ⏱️ Next train: 3 min\n"
                "   ⏳ Following: 7 min, 10 min\n\n"
                "📍 Updated: 19:54:10"
            )
        raise AssertionError(f"Unexpected wait query: {args}")

    route_tool = MagicMock()
    route_tool.invoke = MagicMock(return_value=route_result)

    wait_tool = MagicMock()
    wait_tool.invoke = MagicMock(side_effect=fake_wait_invoke)

    train_tool = MagicMock()
    train_tool.invoke = MagicMock(
        return_value=(
            "🚆 **Comboio: Entrecampos → Campolide**\n"
            "📊 **RESUMO DA VIAGEM**\n"
            "   🚆 Linha: **Linha da Azambuja**\n"
            "   ⏱️ Duração: **5 minutos**\n"
            "📋 **Próximas 3 Partidas:**\n\n"
            "   🕐 **20:52** → 20:57 (5min)\n"
        )
    )

    bus_tool = MagicMock()
    bus_tool.invoke = MagicMock(
        return_value=(
            "Routes: Entrecampos -> NOVA IMS\n"
            "=======================================================\n\n"
            "BUSES\n"
            "----------------------------------------\n"
            "   701: para Campo Ourique\n"
            "     Next: 20:30 (Live), 20:48, 21:06 (stop Av. Forças Armadas)\n"
            "     ~19min travel\n\n"
        )
    )

    with patch.object(transport_api, "get_route_between_stations", route_tool), patch.object(
        transport_api,
        "_get_line_status",
        return_value="Ok",
    ), patch.object(metrolisboa_api, "get_metro_wait_time", wait_tool), patch.object(
        cp_api,
        "plan_train_trip",
        train_tool,
    ), patch.object(carris_api, "carris_find_routes_between", bus_tool):
        response = _build_deterministic_metro_route_response(
            "Como ir de Entrecampos à NOVA IMS?",
            "",
        )

    assert response is not None
    assert "**Transferência em Saldanha**" in response
    assert "🔁 **Outras opções que também fazem sentido:**" in response
    assert "**Autocarro 701**" in response
    assert "Av. Forças Armadas" in response
    assert "**Comboio via Campolide**" in response
    assert "~9 min a pé até NOVA IMS" in response


def test_transport_agent_invoke_preserves_train_wording_inside_multimodal_route_answer() -> None:
    """Final transport formatting should not rewrite a legitimate train alternative as metro inside a mixed-mode route answer."""
    route_result = (
        "🗺️ **Route: Entrecampos → Nova Ims**\n\n"
        "📍 **LOCATION INFORMATION**\n"
        "**NOVA IMS - Information Management School**\n"
        "   🚇 Nearest Metro: **São Sebastião** (🔵 Azul/Vermelha Line)\n"
        "   ℹ️ Information Management School at Universidade NOVA de Lisboa's Campolide campus\n\n"
        "🚇 **METRO ROUTE**\n"
        "🔄 **Transfer Required**\n\n"
        "   💡 **Transfer at**: Saldanha (🟡 ↔ 🔴)\n"
        "   ⏱️ Estimated travel time: **~11 min** (3 stations + 1 transfer)\n\n"
        "   **Full Route**:\n"
        "   1. 🟡 Board at **Entrecampos** → direção **Rato**\n"
        "   2. Exit at **Saldanha**\n"
        "   3. 🔴 Transfer to **Red Line (S. Sebastião ↔ Aeroporto)** → direção **São Sebastião**\n"
        "   4. Exit at **São Sebastião**\n"
        "   5. Walk to Nova Ims\n\n"
        "📌 **Fonte:** [*Metro de Lisboa*](https://www.metrolisboa.pt) **| Atualizado:** 19:54\n"
    )

    route_tool = MagicMock()
    route_tool.invoke = MagicMock(return_value=route_result)

    wait_tool = MagicMock()
    wait_tool.invoke = MagicMock(
        side_effect=[
            (
                "🚇 Metro Wait Times at Entrecampos\n"
                "==================================================\n\n"
                "🟡 Direction: Rato\n"
                "   ⏱️ Next train: 2 min\n"
                "   ⏳ Following: 5 min, 8 min\n\n"
                "📍 Updated: 19:54:10"
            ),
            (
                "🚇 Metro Wait Times at Saldanha\n"
                "==================================================\n\n"
                "🔴 Direction: São Sebastião\n"
                "   ⏱️ Next train: 3 min\n"
                "   ⏳ Following: 7 min, 10 min\n\n"
                "📍 Updated: 19:54:10"
            ),
        ]
    )

    train_tool = MagicMock()
    train_tool.invoke = MagicMock(
        return_value=(
            "🚆 **Comboio: Entrecampos → Campolide**\n"
            "📊 **RESUMO DA VIAGEM**\n"
            "   🚆 Linha: **Linha da Azambuja**\n"
            "   ⏱️ Duração: **5 minutos**\n"
            "📋 **Próximas 3 Partidas:**\n\n"
            "   🕐 **20:52** → 20:57 (5min)\n"
        )
    )

    bus_tool = MagicMock()
    bus_tool.invoke = MagicMock(
        return_value=(
            "Routes: Entrecampos -> NOVA IMS\n"
            "=======================================================\n\n"
            "BUSES\n"
            "----------------------------------------\n"
            "   701: para Campo Ourique\n"
            "     Next: 20:30 (Live), 20:48, 21:06 (stop Av. Forças Armadas)\n"
            "     ~19min travel\n\n"
        )
    )

    with patch.object(TransportAgent, "__init__", lambda self: None):
        agent = TransportAgent()
        agent.system_prompt = "TRANSPORT PROMPT"
        agent.execute_react_loop = MagicMock(side_effect=AssertionError("LLM path should be skipped"))
        agent.tools = []

        with patch.object(transport_api, "get_route_between_stations", route_tool), patch.object(
            transport_api,
            "_get_line_status",
            return_value="Ok",
        ), patch.object(metrolisboa_api, "get_metro_wait_time", wait_tool), patch.object(
            cp_api,
            "plan_train_trip",
            train_tool,
        ), patch.object(carris_api, "carris_find_routes_between", bus_tool):
            result = agent.invoke("Como ir de Entrecampos à NOVA IMS?")

    assert "**Comboio via Campolide**" in result
    assert "**Metro via Campolide**" not in result


def test_build_metro_wait_lines_explains_official_api_outage_instead_of_generic_no_data() -> None:
    """Route summaries should not collapse official Metro API outages into a misleading generic 'no real-time data'."""
    outage_message = (
        "❌ Metro wait times are temporarily unavailable because the official Metro de Lisboa API is not responding right now.\n"
        "The public fallback endpoint still provides line status, but not live wait-time or frequency data."
    )

    wait_tool = MagicMock()
    wait_tool.invoke = MagicMock(return_value=outage_message)

    with patch.object(metrolisboa_api, "get_metro_wait_time", wait_tool):
        lines = _build_metro_wait_lines([("Entrecampos", "Rato")], language="pt")

    combined = "\n".join(lines)
    assert "Dados oficiais do Metro em tempo real estão temporariamente indisponíveis" in combined
    assert "Sem dados em tempo real" not in combined


def test_transport_tool_call_builder_fuzzy_resolves_cp_station_typos() -> None:
    """CP deterministic routing should correct small station typos before invoking tools."""
    spec = _extract_tool_call_spec(
        _build_deterministic_transport_tool_call(
            "When are the next trains from Entrecamposs?"
        )
    )

    assert spec["name"] == "get_train_schedule"
    assert spec["args"] == {"station_name": "Entrecampos"}


def test_transport_agent_invoke_uses_cp_schedule_tool_from_natural_query() -> None:
    """Invoke should bypass the LLM and call the CP schedule tool directly for train schedule questions."""
    with patch.object(TransportAgent, "__init__", lambda self: None):
        agent = TransportAgent()
        agent.system_prompt = "TRANSPORT PROMPT"
        agent.execute_react_loop = MagicMock(side_effect=AssertionError("LLM path should be skipped"))

        schedule_tool = MagicMock()
        schedule_tool.name = "get_train_schedule"
        schedule_tool.invoke = MagicMock(return_value="🚆 **Departures from Entrecampos**\n🕐 **20:30** → Sintra")

        agent.tools = [schedule_tool]

        result = agent.invoke("When are the next trains from Entrecampos?")

        schedule_tool.invoke.assert_called_once_with({"station_name": "Entrecampos"})
        assert "Departures from Entrecampos" in result


def test_transport_agent_invoke_uses_cp_trip_tool_from_natural_query() -> None:
    """Invoke should call the CP trip planner directly for natural train route questions."""
    with patch.object(TransportAgent, "__init__", lambda self: None):
        agent = TransportAgent()
        agent.system_prompt = "TRANSPORT PROMPT"
        agent.execute_react_loop = MagicMock(side_effect=AssertionError("LLM path should be skipped"))

        trip_tool = MagicMock()
        trip_tool.name = "plan_train_trip"
        trip_tool.invoke = MagicMock(return_value="🚆 **Comboio: Rossio → Sintra**\n⏱️ Duração: **39 minutos**")

        agent.tools = [trip_tool]

        result = agent.invoke("How do I get from Rossio to Sintra by train?")

        trip_tool.invoke.assert_called_once_with({"origin": "Rossio", "destination": "Sintra"})
        assert "Rossio → Sintra" in result


def test_transport_agent_invoke_uses_cm_direct_bus_tool_from_natural_query() -> None:
    """Invoke should use the Carris Metropolitana direct-line tool for suburban direct bus questions."""
    with patch.object(TransportAgent, "__init__", lambda self: None):
        agent = TransportAgent()
        agent.system_prompt = "TRANSPORT PROMPT"
        agent.execute_react_loop = MagicMock(side_effect=AssertionError("LLM path should be skipped"))

        direct_tool = MagicMock()
        direct_tool.name = "find_direct_bus_lines"
        direct_tool.invoke = MagicMock(return_value="🚌 **Autocarros: Oeiras → Amadora**\n✅ **2 linha(s) direta(s) encontrada(s):**")

        agent.tools = [direct_tool]

        result = agent.invoke("What are the direct Carris Metropolitana buses from Oeiras to Amadora?")

        direct_tool.invoke.assert_called_once_with({"origin": "Oeiras", "destination": "Amadora"})
        assert "Oeiras" in result and "Amadora" in result


def test_transport_agent_invoke_uses_cm_realtime_positions_tool_from_natural_query() -> None:
    """Invoke should route CM proximity queries to the real-time bus-position tool with the expected radius."""
    with patch.object(TransportAgent, "__init__", lambda self: None):
        agent = TransportAgent()
        agent.system_prompt = "TRANSPORT PROMPT"
        agent.execute_react_loop = MagicMock(side_effect=AssertionError("LLM path should be skipped"))

        positions_tool = MagicMock()
        positions_tool.name = "get_real_time_bus_positions"
        positions_tool.invoke = MagicMock(return_value="🚌 **Carris Metropolitana near Almada**\n📍 3 active buses")

        agent.tools = [positions_tool]

        result = agent.invoke("Show real-time Carris Metropolitana buses near Almada")

        positions_tool.invoke.assert_called_once_with({"location": "Almada", "radius_km": 1.0})
        assert "Almada" in result


def test_transport_subgraph_uses_finalized_cp_tool_fast_path_without_llm() -> None:
    """The transport subgraph should reuse the finalized deterministic fast path instead of summarizing via the LLM."""
    with patch.object(TransportAgent, "__init__", lambda self: None):
        agent = TransportAgent()
        agent.system_prompt = "TRANSPORT PROMPT"
        agent._safe_llm_invoke = MagicMock(side_effect=AssertionError("LLM path should be skipped"))
        agent.llm_with_tools = MagicMock()

        recorded_calls: list[str] = []

        @tool("get_train_schedule")
        def schedule_tool(station_name: str) -> str:
            """Return a fixed schedule snippet for deterministic subgraph testing."""
            recorded_calls.append(station_name)
            return "🚆 **Departures from Entrecampos**\n🕐 **20:30** → Sintra"

        agent.tools = [schedule_tool]

        state = create_initial_state()
        state["messages"].append(HumanMessage(content="When are the next trains from Entrecampos?"))
        graph = agent.build_subgraph()

        result = graph.invoke(state)
        last_message = result["messages"][-1]

        assert recorded_calls == ["Entrecampos"]
        assert "Departures from Entrecampos" in last_message.content


def test_multiagent_combine_outputs_uses_structured_renderer_without_llm() -> None:
    """Hybrid multi-agent answers should use the structured renderer and skip extra LLM synthesis."""
    assistant = MultiAgentAssistant.__new__(MultiAgentAssistant)
    assistant.supervisor = MagicMock()
    assistant.supervisor._safe_llm_invoke = MagicMock(side_effect=AssertionError("LLM synthesis should not run"))

    output = assistant._combine_outputs(
        {
            "weather": (
                "🌤️ **Lisbon Weather Summary**\n"
                "- ☀️ **Today**: Dry and mild\n\n"
                "📌 **Source:** [*IPMA*](https://www.ipma.pt/en/) | **Updated:** 09:31"
            ),
            "transport": (
                "🚇 **Saldanha** → **Odivelas**\n"
                "- 🟡 **Yellow Line**: normal service\n\n"
                "📌 **Source:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | **Updated:** 09:33"
            ),
            "_qa_disclaimers": ["Opening hours may vary, check the official website."],
        },
        language="en",
    )

    assert "### 🌤️ Weather Snapshot" in output
    assert "### 🚇 Mobility and Connections" in output
    assert "### ⚠️ Helpful Notes" in output
    assert output.count("📌 **Source:**") == 1
    assert "[*IPMA*](https://www.ipma.pt/en/)" in output
    assert "[*Metro de Lisboa*](https://www.metrolisboa.pt)" in output


def test_multiagent_combine_outputs_uses_pt_labels_and_single_footer() -> None:
    """Structured hybrid rendering should localize section labels and consolidate source lines in PT."""
    assistant = MultiAgentAssistant.__new__(MultiAgentAssistant)

    output = assistant._combine_outputs(
        {
            "researcher": (
                "🏛️ **Museus em Belém**\n"
                "- **Museu Nacional dos Coches**\n\n"
                "📌 **Fonte:** [*VisitLisboa Places*](https://www.visitlisboa.com/pt-pt/locais) | **Atualizado:** 10:11"
            ),
            "transport": (
                "🚌 **Next Departures from Rossio**\n"
                "📍 **[732] To Caselas**\n\n"
                "📌 **Fonte:** [*Carris*](https://www.carris.pt) | **Atualizado:** 10:12"
            ),
        },
        language="pt",
    )

    assert "### 📍 Destaques Locais" in output
    assert "### 🚇 Mobilidade e Ligações" in output
    assert output.count("📌 **Fonte:**") == 1
    assert "[*VisitLisboa Places*](https://www.visitlisboa.com/pt-pt/locais)" in output
    assert "[*Carris*](https://www.carris.pt)" in output


def test_transport_agent_finalizes_cp_trip_output_cleanly_in_english() -> None:
    """English CP fast-path answers should not leak PT labels after finalization."""
    with patch.object(TransportAgent, "__init__", lambda self: None):
        agent = TransportAgent()
        agent.system_prompt = "TRANSPORT PROMPT"
        agent.execute_react_loop = MagicMock(side_effect=AssertionError("LLM path should be skipped"))

        trip_tool = MagicMock()
        trip_tool.name = "plan_train_trip"
        trip_tool.invoke = MagicMock(
            return_value=(
                "🚆 **Comboio: Lisboa Rossio → Sintra**\n"
                "📊 **RESUMO DA VIAGEM**\n"
                "   🚆 Linha: **Linha de Sintra**\n"
                "   ⏱️ Duração: **40 minutos**\n"
                "   📍 Estado: ⚠️ Alguns comboios com +9min atraso\n"
                "   📊 Partidas restantes hoje: **5**\n"
                "📋 **Próximas 5 Partidas:**\n"
                "💡 **Horários**: cp.pt | Bilhetes: app CP ou estação"
            )
        )
        agent.tools = [trip_tool]

        result = agent.invoke("How do I get from Rossio to Sintra by train?")

        assert "**Train: Lisboa Rossio → Sintra**" in result
        assert "**TRIP SUMMARY**" in result
        assert "Line: **Linha de Sintra**" in result
        assert "Duration: **40 minutos**" in result
        assert "Some trains are delayed by 9 min" in result
        assert "Remaining departures today" in result
        assert "**Next 5 Departures:**" in result
        assert "**Schedules**" in result
        assert "Tickets:" in result
        assert "comboios" not in result.lower()


def test_transport_agent_finalizes_cm_direct_bus_output_cleanly_in_english() -> None:
    """English Carris Metropolitana direct-line answers should translate the remaining guidance labels."""
    with patch.object(TransportAgent, "__init__", lambda self: None):
        agent = TransportAgent()
        agent.system_prompt = "TRANSPORT PROMPT"
        agent.execute_react_loop = MagicMock(side_effect=AssertionError("LLM path should be skipped"))

        direct_tool = MagicMock()
        direct_tool.name = "find_direct_bus_lines"
        direct_tool.invoke = MagicMock(
            return_value=(
                "🚌 **Autocarros: Oeiras → Amadora**\n"
                "✅ **19 linha(s) direta(s) encontrada(s):**\n"
                "   📍 Terminais: Algés (Estação) ↔ Amadora (Estação Sul)\n"
                "📋 Outras linhas: 1503, 1504\n"
                "💡 **Como usar:**\n"
                "   • Procure pelo número da linha (ex: **1502**) na paragem\n"
                "   • Verifique a direção do autocarro (Oeiras → Amadora)\n"
                "   • Horários e paragens: carrismetropolitana.pt\n"
            )
        )
        agent.tools = [direct_tool]

        result = agent.invoke("What are the direct Carris Metropolitana buses from Oeiras to Amadora?")

        assert "**Buses: Oeiras → Amadora**" in result
        assert "direct line(s) found" in result
        assert "Terminals" in result
        assert "**How to use it:**" in result
        assert "Look for the line number" in result
        assert "Check the bus direction" in result
        assert "Schedules and stops" in result
        assert "Other lines" in result


def test_transport_tool_call_builder_skips_single_tool_fast_path_for_mode_restricted_routes() -> None:
    """Mode-restricted route queries should bypass the single-tool shortcut so the agent can combine/filter operators."""
    message = _build_deterministic_transport_tool_call(
        "How do I get from Belém to Marquês de Pombal by bus only?"
    )

    assert message is None


def test_transport_agent_invoke_combines_bus_only_options_across_operators() -> None:
    """Bus-only route requests should combine Carris Urban bus results with Carris Metropolitana options and exclude trams."""
    with patch.object(TransportAgent, "__init__", lambda self: None):
        agent = TransportAgent()
        agent.system_prompt = "TRANSPORT PROMPT"
        agent.execute_react_loop = MagicMock(side_effect=AssertionError("LLM path should be skipped"))

        urban_tool = MagicMock()
        urban_tool.name = "carris_find_routes_between"
        urban_tool.invoke = MagicMock(
            return_value=(
                "Routes: Belém -> Marquês de Pombal\n"
                "=======================================================\n\n"
                "TRAMS\n----------------------------------------\n"
                "   15E: para Praça da Figueira\n"
                "     Next: 12:05 (Live)\n\n"
                "BUSES\n----------------------------------------\n"
                "   714: para Outurela\n"
                "     Next: 12:07 (Live)\n\n"
            )
        )

        metropolitan_tool = MagicMock()
        metropolitan_tool.name = "find_direct_bus_lines"
        metropolitan_tool.invoke = MagicMock(
            return_value=(
                "🚌 **Autocarros: Belém → Marquês De Pombal**\n"
                "✅ **2 linha(s) direta(s) encontrada(s):**\n"
                "**1. 🚍 Linha 1717**\n"
                "   📍 Terminais: Belém ↔ Marquês de Pombal\n"
            )
        )

        agent.tools = [urban_tool, metropolitan_tool]

        result = agent.invoke("How do I get from Belém to Marquês de Pombal by bus only?")

        urban_tool.invoke.assert_called_once_with(
            {"origin": "Belém", "destination": "Marquês de Pombal"}
        )
        metropolitan_tool.invoke.assert_called_once_with(
            {"origin": "Belém", "destination": "Marquês de Pombal"}
        )
        assert "Carris Urban" in result
        assert "Carris Metropolitana" in result
        assert "714" in result
        assert "1717" in result
        assert "TRAMS" not in result
        assert "15E" not in result
        assert "### 🚌" in result
        assert "**🚌 Carris Urban**" in result
        assert "BUSES" not in result


def test_transport_agent_invoke_uses_non_bus_route_when_buses_are_excluded() -> None:
    """When the user excludes buses, the deterministic route response should prefer a non-bus route and skip bus tools."""
    with patch.object(TransportAgent, "__init__", lambda self: None):
        agent = TransportAgent()
        agent.system_prompt = "TRANSPORT PROMPT"
        agent.execute_react_loop = MagicMock(side_effect=AssertionError("LLM path should be skipped"))

        urban_tool = MagicMock()
        urban_tool.name = "carris_find_routes_between"
        urban_tool.invoke = MagicMock(side_effect=AssertionError("Carris should be skipped when metro already satisfies the constraint"))

        agent.tools = [urban_tool]

        with patch.object(
            transport_agent_module,
            "_build_deterministic_metro_route_response",
            return_value=(
                "🚇 **Saldanha** → **Odivelas**\n\n"
                "⚠️ **Line Status:**\n"
                "- 🟡 **Yellow Line**: normal service\n"
            ),
        ) as metro_builder:
            result = agent.invoke("How do I get from Saldanha to Odivelas without taking a bus?")

        metro_builder.assert_called_once()
        urban_tool.invoke.assert_not_called()
        assert "Saldanha" in result
        assert "Odivelas" in result
        assert "Yellow Line" in result or "Metro" in result


def test_transport_agent_invoke_returns_gentle_message_when_bus_only_matches_do_not_exist() -> None:
    """If neither operator can confirm a bus-only route, the response should say so clearly and politely."""
    with patch.object(TransportAgent, "__init__", lambda self: None):
        agent = TransportAgent()
        agent.system_prompt = "TRANSPORT PROMPT"
        agent.execute_react_loop = MagicMock(side_effect=AssertionError("LLM path should be skipped"))

        urban_tool = MagicMock()
        urban_tool.name = "carris_find_routes_between"
        urban_tool.invoke = MagicMock(
            return_value=(
                "Routes: Belém -> Ajuda\n"
                "=======================================================\n\n"
                "TRAMS\n----------------------------------------\n"
                "   15E: para Algés\n"
            )
        )

        metropolitan_tool = MagicMock()
        metropolitan_tool.name = "find_direct_bus_lines"
        metropolitan_tool.invoke = MagicMock(
            return_value="❌ **Sem linhas diretas entre 'Belém' e 'Ajuda'**"
        )

        agent.tools = [urban_tool, metropolitan_tool]

        result = agent.invoke("Mostra-me uma rota de Belém para Ajuda só de autocarro.")

        assert "autocarro" in result.lower()
        assert "não consegui confirmar" in result.lower() or "nao consegui confirmar" in result.lower()
        assert "**ℹ️ Notas de Cobertura**" in result


def test_transport_agent_invoke_formats_surface_only_routes_when_metro_is_excluded() -> None:
    """`Não quero metro` requests should return structured surface options instead of raw tool dumps."""
    with patch.object(TransportAgent, "__init__", lambda self: None):
        agent = TransportAgent()
        agent.system_prompt = "TRANSPORT PROMPT"
        agent.execute_react_loop = MagicMock(side_effect=AssertionError("LLM path should be skipped"))

        urban_tool = MagicMock()
        urban_tool.name = "carris_find_routes_between"
        urban_tool.invoke = MagicMock(
            return_value=(
                "Routes: Belém -> Marquês de Pombal\n"
                "=======================================================\n\n"
                "TRAMS\n----------------------------------------\n"
                "   15E: para Praça da Figueira\n"
                "     Next: 12:05 (Live)\n\n"
                "BUSES\n----------------------------------------\n"
                "   714: para Outurela\n"
                "     Next: 12:07 (Live)\n\n"
            )
        )

        metropolitan_tool = MagicMock()
        metropolitan_tool.name = "find_direct_bus_lines"
        metropolitan_tool.invoke = MagicMock(
            return_value=(
                "🚌 **Autocarros: Belém → Marquês de Pombal**\n"
                "✅ **1 linha(s) direta(s) encontrada(s):**\n"
                "**1. 🚍 Linha 1717**\n"
                "   📍 Terminais: Belém ↔ Marquês de Pombal\n"
            )
        )

        agent.tools = [urban_tool, metropolitan_tool]

        result = agent.invoke("Não quero metro: como vou de Belém para o Marquês de Pombal?")

        assert "Opções de superfície sem metro" in result
        assert "Carris Urban" in result
        assert "Carris Metropolitana" in result
        assert "714" in result
        assert "15E" in result
        assert "1717" in result
        assert "Routes:" not in result
        assert "BUSES" not in result
        assert "TRAMS" not in result


def test_transport_agent_bus_only_route_enriches_missing_schedule_placeholder() -> None:
    """Bus-only route summaries should replace bare 'Check schedule' placeholders with actionable schedule guidance."""
    with patch.object(TransportAgent, "__init__", lambda self: None):
        agent = TransportAgent()
        agent.system_prompt = "TRANSPORT PROMPT"
        agent.execute_react_loop = MagicMock(side_effect=AssertionError("LLM path should be skipped"))

        urban_tool = MagicMock()
        urban_tool.name = "carris_find_routes_between"
        urban_tool.invoke = MagicMock(
            return_value=(
                "Routes: Belém -> Ajuda\n"
                "=======================================================\n\n"
                "BUSES\n----------------------------------------\n"
                "   727: Estação Roma-Areeiro - Restelo\n"
                "     Check schedule\n\n"
            )
        )

        metropolitan_tool = MagicMock()
        metropolitan_tool.name = "find_direct_bus_lines"
        metropolitan_tool.invoke = MagicMock(
            return_value="❌ **Sem linhas diretas entre 'Belém' e 'Ajuda'**"
        )

        frequency_tool = MagicMock()
        frequency_tool.name = "carris_get_service_frequency"
        frequency_tool.invoke = MagicMock(
            return_value="No scheduled trips found for route '727' today."
        )

        agent.tools = [urban_tool, metropolitan_tool, frequency_tool]

        result = agent.invoke("Mostra-me uma rota de Belém para Ajuda só de autocarro.")

        assert "Consultar horários" not in result
        assert "não foram encontradas partidas" in result.lower()
        assert "paragem específica" in result.lower()
