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
    _build_destination_only_transport_overview_response,
    _build_deterministic_metro_route_response,
    _build_deterministic_transport_tool_call,
    _extract_destination_only_target,
    _build_metro_wait_lines,
    _extract_route_endpoints,
    _parse_route_mode_preferences,
    _parse_metro_wait_request,
    _query_has_status_intent,
)
from agent.graph import MultiAgentAssistant
from agent.state import create_initial_state
from agent.utils.response_formatter import (
    normalize_transport_notes_block,
    operators_from_tool_names,
    rebuild_transport_source_line,
)
from tools.transport_api import _build_ambiguity_preamble
from tools.carrismetropolitana_api import find_bus_routes, find_direct_bus_lines, resolve_location
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


def test_transport_tool_call_builder_maps_combios_cascais_to_cascais_line() -> None:
    """The common 'combios cascais' typo should still map to the default Cascais-line CP trip."""
    spec = _extract_tool_call_spec(
        _build_deterministic_transport_tool_call("combios cascais")
    )

    assert spec["name"] == "plan_train_trip"
    assert spec["args"] == {"origin": "Cais do Sodré", "destination": "Cascais"}


def test_extract_route_endpoints_handles_pt_ate_a_metro_phrase() -> None:
    """PT-PT metro phrasings with `até à` should parse clean endpoints."""
    endpoints = _extract_route_endpoints("Quero ir de metro de Entrecampos até à NOVA IMS")

    assert endpoints == ("Entrecampos", "NOVA IMS")


def test_extract_route_endpoints_handles_metro_shorthand_station_pair() -> None:
    """Short Metro shorthand should recover the intended station pair without LLM fallback."""
    endpoints = _extract_route_endpoints("ML azul baixa chiado rato")

    assert endpoints == ("Baixa-Chiado", "Rato")


def test_parse_route_mode_preferences_recognizes_ml_prefix_as_metro_only() -> None:
    """ML + line-colour shorthand should keep Metro route queries on the metro-only path."""
    preferences = _parse_route_mode_preferences("ML azul baixa chiado rato")

    assert preferences["metro_only"] is True


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


def test_extract_route_endpoints_strips_future_trip_follow_up_clause() -> None:
    """Trailing PT follow-up clauses about future-trip differences must not leak into the destination."""
    endpoints = _extract_route_endpoints(
        "Como vou amanhã do Rossio ao Aeroporto de metro e o que muda por ser uma viagem futura?"
    )

    assert endpoints == ("Rossio", "Aeroporto")


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


def test_query_has_status_intent_handles_pt_transport_overview_wording() -> None:
    """PT overview phrasings should trigger the deterministic transport-summary path."""
    assert _query_has_status_intent(
        "Dá-me o ponto de situação do Metro, autocarros e comboios em Lisboa."
    ) is True


def test_transport_agent_direct_status_query_uses_summary_tool_for_pt_overview() -> None:
    """Broad PT status overviews should skip the LLM path and use the summary tool directly."""
    with patch.object(TransportAgent, "__init__", lambda self: None):
        agent = TransportAgent()
        summary_tool = MagicMock()
        summary_tool.name = "get_transport_summary"
        agent.tools = [summary_tool]
        agent._invoke_tool = MagicMock(return_value="Resumo PT dos transportes")

        result = agent._run_direct_tool_fallback(
            "Dá-me o ponto de situação do Metro, autocarros e comboios em Lisboa."
        )

    agent._invoke_tool.assert_called_once_with(
        summary_tool,
        {},
        tool_name="get_transport_summary",
    )
    assert result == "Resumo PT dos transportes"


def test_transport_agent_direct_status_query_uses_metro_status_tool_for_pt_disruptions() -> None:
    """PT metro-disruption questions should use the deterministic metro-status tool path."""
    with patch.object(TransportAgent, "__init__", lambda self: None):
        agent = TransportAgent()
        metro_tool = MagicMock()
        metro_tool.name = "get_metro_status"
        agent.tools = [metro_tool]
        agent._invoke_tool = MagicMock(return_value="Estado PT do metro")

        result = agent._run_direct_tool_fallback(
            "Existem perturbações nas linhas do metro de Lisboa?"
        )

    agent._invoke_tool.assert_called_once_with(
        metro_tool,
        {},
        tool_name="get_metro_status",
    )
    assert result == "Estado PT do metro"


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


def test_route_tool_resolves_sete_rios_to_jardim_zoologico_metro() -> None:
    """Sete Rios should resolve to Jardim Zoológico for Metro route calculations."""
    with patch.object(transport_api, "_get_line_status", return_value="Ok"):
        result = transport_api.get_route_between_stations.invoke(
            {"origin": "Entrecampos", "destination": "Sete Rios"}
        )

    assert "🚇 **METRO ROUTE**" in result
    assert "**Transfer at**: Marquês de Pombal" in result
    assert "Board at **Entrecampos**" in result
    assert "Exit at **Jardim Zoológico**" in result
    assert "Destination 'Sete Rios' not on Metro" not in result


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


def test_multiagent_finalize_chat_response_restores_missing_footer_from_agent_outputs() -> None:
    """Final response formatting should restore a consolidated footer if repair removed it."""
    assistant = MultiAgentAssistant.__new__(MultiAgentAssistant)
    assistant.state = {"messages": [], "user_context": {}}
    assistant._append_assistant_message = MagicMock()
    assistant._collect_execution_summary = MagicMock(return_value={})
    assistant._print_execution_summary = MagicMock()

    agent_outputs = {
        "weather": (
            "### 🌤️ Weather Snapshot\n\n"
            "- Dry and mild\n\n"
            "📌 **Source:** [*IPMA*](https://www.ipma.pt/en/) | **Updated:** 09:31"
        ),
        "transport": (
            "### 🚇 Mobility and Connections\n\n"
            "- Tram 15E\n\n"
            "📌 **Source:** [*Carris*](https://www.carris.pt) | **Updated:** 09:33"
        ),
    }

    with patch("agent.graph.clean_response", side_effect=lambda text: text), patch(
        "agent.graph.format_response",
        side_effect=lambda text: text,
    ), patch("agent.graph.generate_response_title", return_value=None), patch(
        "agent.graph.ensure_response_title",
        side_effect=lambda text, _title: text,
    ), patch(
        "agent.graph.Config.SHOW_MARKDOWN_RESPONSE_IN_TERMINAL",
        False,
    ):
        output = assistant._finalize_chat_response(
            response="### 🌤️ Weather Snapshot\n\n- Dry and mild",
            message="Plan a city afternoon.",
            language="en",
            agents_to_call=["weather", "transport"],
            routing_reasoning="hybrid",
            agent_outputs=agent_outputs,
            direct_response_used=False,
            start_time=0.0,
            workers=["weather", "transport"],
            run_workers_in_parallel=False,
            qa_result=None,
            retry_agents_used=[],
            final_repair_ran=True,
            simple_weather_fact_check=None,
        )

    assert output.count("📌 **Source:**") == 1
    assert "[*IPMA*](https://www.ipma.pt/en/)" in output
    assert "[*Carris*](https://www.carris.pt)" in output


def test_multiagent_finalize_chat_response_localizes_live_transport_summary_to_pt() -> None:
    """Final chat formatting should localize plain-text transport summaries and QA notes in PT."""
    assistant = MultiAgentAssistant.__new__(MultiAgentAssistant)
    assistant.state = {"messages": [], "user_context": {}}
    assistant._append_assistant_message = MagicMock()
    assistant._collect_execution_summary = MagicMock(return_value={})
    assistant._print_execution_summary = MagicMock()

    response = (
        "🚇 🚌 🚆 Lisbon Transport Status — Updated: 19:19\n\n"
        "🚇 Metro de Lisboa\n\n"
        "🟢 Status: Normal service on all lines\n\n"
        "🚌 Carris (Urban buses)\n\n"
        "🟢 Vehicles in service: 249 vehicles\n\n"
        "🚌 Carris Metropolitana (Suburban buses)\n\n"
        "⚠️ Active alerts: 93 alerts\n"
        "The available data does not specify which routes are affected or the exact disruption details, so this should be verified.\n\n"
        "🚆 CP trains (AML)\n\n"
        "📊 Trains running in AML: 30 trains\n"
        "⚠️ Trains with delays over 1 minute: 24 trains\n"
        "The available data does not specify the affected lines, directions, or transfer points, so this should be verified.\n\n"
        "---\n\n"
        "- ⚠️ The source list is incomplete for the full transport picture; only Metro de Lisboa is cited explicitly.\n"
        "- ⚠️ Carris bus route numbers and schedules should be confirmed at carris.pt, because GGTFS data may miss very recent changes.\n"
        "- ⚠️ The Carris Metropolitana alert count and CP delay counts are not enough to describe the actual disruption status without affected lines/routes or service details.\n\n"
        "📌 Source: Metro de Lisboa | Updated: 19:20"
    )

    with patch("agent.graph.clean_response", side_effect=lambda text: text), patch(
        "agent.graph.format_response",
        side_effect=lambda text: text,
    ), patch("agent.graph.generate_response_title", return_value=None), patch(
        "agent.graph.ensure_response_title",
        side_effect=lambda text, _title: text,
    ), patch(
        "agent.graph.Config.SHOW_MARKDOWN_RESPONSE_IN_TERMINAL",
        False,
    ):
        output = assistant._finalize_chat_response(
            response=response,
            message="Dá-me o ponto de situação do Metro, autocarros e comboios em Lisboa.",
            language="pt",
            agents_to_call=["transport", "researcher"],
            routing_reasoning="hybrid-like transport response",
            agent_outputs={"transport": response},
            direct_response_used=False,
            start_time=0.0,
            workers=["transport"],
            run_workers_in_parallel=False,
            qa_result=None,
            retry_agents_used=[],
            final_repair_ran=False,
            simple_weather_fact_check=None,
        )

    assert "Situação dos Transportes de Lisboa" in output
    assert "Carris (Urbano)" in output
    assert "Carris Metropolitana (Suburbano)" in output
    assert "CP Comboios (AML)" in output
    assert "Estado: Circulação normal em todas as linhas" in output
    assert "Veículos em serviço: 249 veículos" in output
    assert "Alertas ativos: 93 alertas" in output
    assert "Comboios a circular na AML: 30 comboios" in output
    assert "Comboios com atrasos superiores a 1 minuto: 24 comboios" in output
    assert "Os dados disponíveis não especificam quais as rotas afetadas" in output
    assert "Os dados disponíveis não especificam as linhas, direções ou pontos de transbordo afetados" in output
    assert "A lista de fontes está incompleta" not in output
    assert "Os números das linhas e os horários da Carris devem ser confirmados" not in output
    assert "A contagem de alertas da Carris Metropolitana e os atrasos da CP" not in output
    assert "GGTFS" not in output
    assert "Lisbon Transport Status" not in output
    assert "Urban buses" not in output
    assert "Suburban buses" not in output
    assert "Helpful Notes" not in output


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
        assert "Duration: **40 min**" in result
        assert "Some trains are delayed by 9 min" in result
        assert "Remaining departures today" in result
        assert "**Next 5 Departures:**" in result
        assert "**Schedules**" in result
        assert "Tickets:" in result
        assert "comboios" not in result.lower()


def test_transport_agent_rewrites_carris_only_footer_on_deterministic_early_return() -> None:
    """Deterministic fast paths should rebuild the footer before invoke() returns."""
    with patch.object(TransportAgent, "__init__", lambda self: None):
        agent = TransportAgent()
        agent.system_prompt = "TRANSPORT PROMPT"
        agent._tool_calls_log = []
        agent._rewrite_follow_up_transport_query = lambda message, language: message
        agent._remember_transport_context = MagicMock()
        agent.execute_react_loop = MagicMock(side_effect=AssertionError("LLM path should be skipped"))
        agent._resolve_deterministic_response = MagicMock(
            side_effect=lambda user_message, context="", language=None: (
                agent._record_tool_call("carris_find_routes_between", {"origin": "Rossio", "destination": "Belém"})
                or agent._finalize_transport_response(
                    "🚌 **Carris: Rossio → Belém**\n\n"
                    "📌 **Fonte:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | "
                    "[*Carris*](https://www.carris.pt) | [*CP*](https://www.cp.pt) | **Atualizado:** 16:45",
                    user_message=user_message,
                    language=language or "pt",
                )
            )
        )
        agent._invoke_deterministic_tool_call = MagicMock(return_value=None)

        result = agent.invoke("Quais os próximos autocarros da Carris no Rossio para seguir para Belém agora?")

        assert "[*Carris*](https://www.carris.pt)" in result
        assert "[*CP*](https://www.cp.pt)" not in result
        assert "[*Metro de Lisboa*](https://www.metrolisboa.pt)" not in result


def test_rebuild_transport_source_line_collapses_duplicate_transport_footers() -> None:
    """Transport footer rebuilding should replace all duplicate transport source lines with one canonical footer."""
    raw = (
        "Route body.\n\n"
        "📌 **Source:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | **Updated:** 10:49\n\n"
        "📌 **Source:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | **Updated:** 10:49"
    )

    rebuilt = rebuild_transport_source_line(raw, ["metro"], language="en")

    assert rebuilt.count("📌 **Source:**") == 1
    assert rebuilt.count("[*Metro de Lisboa*](https://www.metrolisboa.pt)") == 1


def test_transport_finalize_metro_wait_response_keeps_single_metro_footer() -> None:
    """Metro wait answers should end with exactly one Metro-only footer."""
    agent = TransportAgent()
    agent._tool_calls_log = [{"tool_name": "get_metro_wait_time", "args": {"station": "Rato"}}]

    raw = (
        "🚇 **Next metro at Rato**\n\n"
        "🗓️ **Next Metro:**\n"
        "- **Station Rato:** Direction **Odivelas** — **⏱️ Next metro in:** 3 min 19s | 7 min 24s"
    )

    result = agent._finalize_transport_response(
        raw,
        user_message="proximo metro rato",
        language="en",
    )

    assert result.count("📌 **Source:**") == 1
    assert "[*Metro de Lisboa*](https://www.metrolisboa.pt)" in result
    assert "[*Carris*](https://www.carris.pt)" not in result
    assert "[*Carris Metropolitana*](https://www.carrismetropolitana.pt)" not in result
    assert "[*CP*](https://www.cp.pt)" not in result


def test_transport_finalize_preserves_structured_metro_route_layout() -> None:
    """A fully structured Metro route should not collapse into an arrivals-style block during finalization."""
    agent = TransportAgent()
    agent._record_tool_call("get_metro_status", {})
    agent._record_tool_call("get_metro_wait_time", {"station": "Baixa-Chiado"})
    structured_route = (
        "🚇 **Baixa-Chiado** → **Rato**\n\n"
        "⚠️ **Estado das Linhas:**\n"
        "- 🔵 **Linha Azul**: circulação normal\n"
        "- 🟡 **Linha Amarela**: circulação normal\n\n"
        "⏳ **Tempo total estimado:** ~13 min\n\n"
        "🗺️ **O seu Trajeto de Metro:**\n"
        "- 📍 **Embarque na estação Baixa-Chiado**\n"
        "- 🔵 **Linha Azul** - direção **Reboleira**\n"
        "- 🔄 **Transferência em Marquês de Pombal**\n"
        "- 🟡 **Linha Amarela** - direção **Rato**\n"
        "- 🎯 **Saia na estação Rato**\n\n"
        "🗓️ **Próximos Metros** (tempo real):\n"
        "- **Estação Baixa-Chiado:** Direção Reboleira — **⏱️ Próximo Metro em:** 3 min 10s\n\n"
        "💡 **Dica rápida:** Em Marquês de Pombal, siga a sinalização para a Linha Amarela."
    )

    result = agent._finalize_transport_response(
        structured_route,
        user_message="ML azul baixa chiado rato",
        language="pt",
    )

    assert "### 🚌 Próximas Chegadas" not in result
    assert "🗺️ **O seu Trajeto de Metro:**" in result
    assert result.count("📌 **Fonte:**") == 1


def test_transport_agent_invoke_clears_stale_operator_log_before_new_query() -> None:
    """A fresh transport query must not inherit source operators from the previous one."""
    agent = TransportAgent()
    agent._tool_calls_log = [{"tool_name": "get_transport_summary", "args": {}}]

    def _deterministic(*, user_message: str, context: str = "", language: str | None = None) -> str:
        agent._record_tool_call("get_metro_wait_time", {"station": "Rato"})
        return agent._finalize_transport_response(
            "🚇 **Next metro at Rato**\n\n🗓️ **Next Metro:**\n- **Station Rato:** Direction **Odivelas** — **⏱️ Next metro in:** 3 min 19s | 7 min 24s",
            user_message=user_message,
            language=language or "en",
        )

    with patch.object(agent, "_resolve_deterministic_response", side_effect=_deterministic), patch.object(
        agent,
        "_remember_transport_context",
        lambda *_args, **_kwargs: None,
    ):
        result = agent.invoke("proximo metro rato", context="User language: en")

    assert result.count("📌 **Source:**") == 1
    assert "[*Metro de Lisboa*](https://www.metrolisboa.pt)" in result
    assert "[*Carris*](https://www.carris.pt)" not in result
    assert "[*Carris Metropolitana*](https://www.carrismetropolitana.pt)" not in result
    assert "[*CP*](https://www.cp.pt)" not in result


def test_parse_metro_wait_request_handles_short_station_forms() -> None:
    """Short metro wait queries should stay on the deterministic wait-time path."""
    short_wait = _parse_metro_wait_request("proximo metro rato")
    compact_station = _parse_metro_wait_request("metro oriente")
    route_query = _parse_metro_wait_request("metro verde rossio rato")

    assert short_wait == {"station": "Rato", "direction": None, "status_requested": False}
    assert compact_station == {"station": "Oriente", "direction": None, "status_requested": False}
    assert route_query is None


def test_build_ambiguity_preamble_handles_new_bare_place_tokens() -> None:
    """Only genuinely ambiguous bare endpoints should surface a clarification preamble."""
    oriente_note = _build_ambiguity_preamble("Rossio", "Oriente")
    madeira_note = _build_ambiguity_preamble("Cais do Sodré", "Madeira")
    explicit_street = _build_ambiguity_preamble("Cais do Sodré", "Rua Humberto Madeira")

    assert oriente_note == ""
    assert "Ilha da Madeira" in madeira_note
    assert explicit_street == ""


def test_extract_destination_only_target_handles_nearby_option_queries() -> None:
    """Single-destination transport questions should be separated from real origin→destination routes."""
    assert _extract_destination_only_target("Transport to Museu Calouste Gulbenkian") == "Museu Calouste Gulbenkian"
    assert _extract_destination_only_target("Como vou para o MNAC?") == "MNAC"
    assert _extract_destination_only_target("transportes disponiveis alfama") == "alfama"
    assert _extract_destination_only_target("How do I get to Museu Nacional do Azulejo by bus?") is None


def test_build_destination_only_transport_overview_response_uses_nearby_stops() -> None:
    """Destination-only transport prompts should surface nearby Metro, CP, and Carris options instead of a zero-distance route."""
    with patch("tools.transport_api.find_nearest_stops_for_place", return_value={
        "display_name": "Museu Nacional de Arte Contemporânea (MNAC)",
        "metro": "baixa/chiado",
        "metro_line": "azul/verde",
        "metro_walk_minutes": 3,
        "train_station": "Lisboa Rossio",
        "train_walk_minutes": 6,
        "carris_stops": [
            {"stop_name": "Rua do Alecrim", "distance_km": 0.18},
            {"stop_name": "Chiado", "distance_km": 0.32},
        ],
    }):
        response = _build_destination_only_transport_overview_response("Como vou para o MNAC?", "")

    assert response is not None
    assert "MNAC" in response
    assert "Nearest Metro" not in response
    assert "Metro mais próximo" in response
    assert "Rua do Alecrim" in response
    assert "Lisboa Rossio" in response
    assert "Já está no destino" not in response


def test_operators_from_tool_names_expands_transport_summary_to_all_used_networks() -> None:
    """The broad transport summary tool is a true four-network overview and must cite every network it uses."""
    operators = operators_from_tool_names(["get_transport_summary"])

    assert operators == ["metro", "carris", "carris_metropolitana", "cp"]


def test_normalize_transport_notes_block_detaches_footer_from_warning_items() -> None:
    """Transport warning blocks should render as plain warning paragraphs so the footer stays outside the last item."""
    raw = (
        "### ⚠️ Notas Úteis\n\n"
        "- ⚠️ Nota 1\n"
        "- ⚠️ Nota 2\n\n"
        "📌 **Fonte:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | **Atualizado:** 23:35"
    )

    normalized = normalize_transport_notes_block(raw)

    assert "- ⚠️ Nota 1" not in normalized
    assert "- ⚠️ Nota 2" not in normalized
    assert "⚠️ Nota 1" in normalized
    assert "⚠️ Nota 2" in normalized
    assert normalized.rstrip().endswith(
        "📌 **Fonte:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | **Atualizado:** 23:35"
    )


def test_find_direct_bus_lines_falls_back_to_route_finder_for_poi_destinations() -> None:
    """POI-like endpoints should trigger the stop-based route finder instead of a raw no-direct-lines miss."""
    fake_lines = [
        {
            "id": "line_1",
            "short_name": "1718",
            "long_name": "Oeiras - Amadora",
            "localities": ["Oeiras", "Amadora"],
            "municipalities": ["Oeiras", "Amadora"],
        }
    ]

    fallback_tool = MagicMock()
    fallback_tool.invoke = MagicMock(
        return_value="🚌 **BUS ROUTE FINDER**\nResolved via nearby stops."
    )

    with patch("tools.carrismetropolitana_api.fetch_json_with_retry", return_value=fake_lines), patch(
        "tools.carrismetropolitana_api.find_bus_routes",
        fallback_tool,
    ):
        result = find_direct_bus_lines.invoke(
            {"origin": "Rossio", "destination": "Museu Nacional de Arte Antiga"}
        )

    fallback_tool.invoke.assert_called_once_with(
        {
            "origin": "Rossio",
            "destination": "Museu Nacional de Arte Antiga",
            "search_radius_km": 0.5,
        }
    )
    assert "BUS ROUTE FINDER" in result


def test_resolve_location_lisbon_city_poi_without_cm_stops_avoids_warning() -> None:
    """Geocoded Lisbon POIs without nearby CM stops should stay graceful, not log hard resolution failures."""
    geocoded = {
        "name": "Museu Nacional do Azulejo",
        "lat": 38.7247242,
        "lon": -9.113914,
    }

    with patch("tools.carrismetropolitana_api.find_stops_by_name", return_value=[]), patch(
        "tools.carrismetropolitana_api.geocode_location",
        return_value=geocoded,
    ), patch(
        "tools.carrismetropolitana_api.find_stops_near_coordinates",
        side_effect=[[], []],
    ), patch("tools.carrismetropolitana_api.logger.warning") as warn_mock, patch(
        "tools.carrismetropolitana_api.logger.info"
    ) as info_mock:
        result = resolve_location("Museu Nacional do Azulejo")

    assert result["success"] is False
    assert result["location"] == geocoded
    warn_mock.assert_not_called()
    info_mock.assert_called_once()


def test_find_bus_routes_lisbon_city_poi_without_cm_stops_uses_scope_note() -> None:
    """Lisbon-city POIs without nearby CM stops should produce a scope note, not an unresolved-location error."""
    with patch(
        "tools.carrismetropolitana_api.resolve_location",
        side_effect=[
            {
                "success": True,
                "method": "name_match",
                "stops": [{"id": "1001", "name": "Rossio"}],
                "location": None,
            },
            {
                "success": False,
                "method": None,
                "stops": [],
                "location": {
                    "name": "Museu Nacional do Azulejo",
                    "lat": 38.7247242,
                    "lon": -9.113914,
                },
            },
        ],
    ):
        result = find_bus_routes.invoke(
            {"origin": "Rossio", "destination": "Museu Nacional do Azulejo"}
        )

    assert "Could not resolve 'Museu Nacional do Azulejo'" not in result
    assert "better served by **Carris Urbana**" in result


def test_extract_route_endpoints_handles_carris_wait_query_with_destination() -> None:
    """Carris wait queries should still recover clean origin/destination endpoints."""
    endpoints = _extract_route_endpoints(
        "Quais os próximos autocarros da Carris no Rossio para seguir para Belém agora?"
    )

    assert endpoints == ("Rossio", "Belém")


def test_transport_agent_skips_generic_route_fast_path_for_carris_wait_query() -> None:
    """Carris wait/departure queries with a destination should still reach Carris-specific tools."""
    with patch.object(TransportAgent, "__init__", lambda self: None):
        agent = TransportAgent()
        agent.system_prompt = "TRANSPORT PROMPT"
        agent._tool_calls_log = []
        agent._remember_transport_context = MagicMock()
        agent._rewrite_follow_up_transport_query = lambda message, language: message
        agent.execute_react_loop = MagicMock(side_effect=AssertionError("LLM path should be skipped"))

        route_tool = MagicMock()
        route_tool.name = "carris_find_routes_between"
        route_tool.invoke = MagicMock(
            return_value=(
                "**Route:** Rossio -> Belém\n\n"
                "📊 **Direct connections found:** 1\n"
                "#### 🚌 Buses\n\n"
                "1. 🚌 **Line 15E** — Algés\n"
                "   🕐 **Next departures:** 12:05, 12:12\n"
                "   ℹ️ **Real time:** live active\n\n"
            )
        )
        agent.tools = [route_tool]

        result = agent.invoke("Quais os próximos autocarros da Carris no Rossio para seguir para Belém agora?")

        route_tool.invoke.assert_called_once_with({"origin": "Rossio", "destination": "Belém"})
        assert "15E" in result
        assert "**Ligações diretas encontradas:** 1" in result
        assert "**🚌 Autocarros**" in result
        assert "**Linha 15E**" in result
        assert "**Tempo real:** tempo real ativo" in result
        assert "[*Carris*](https://www.carris.pt)" in result
        assert "[*CP*](https://www.cp.pt)" not in result
        assert "Direct connections found" not in result
        assert "(Live)" not in result
        assert "**Line 15E**" not in result
        assert "**🚌 Buses**" not in result


def test_transport_agent_skips_metro_fast_path_when_destination_is_ambiguous() -> None:
    """Bare ambiguous destinations like Madeira should fall through to the route tool output with its ambiguity preamble."""
    with patch.object(TransportAgent, "__init__", lambda self: None), patch(
        "agent.agents.transport_agent._build_deterministic_metro_route_response",
        return_value="WRONG deterministic metro path",
    ), patch(
        "agent.agents.transport_agent._build_deterministic_route_tool_response",
        return_value=(
            "⚠️ **Ambiguidade no destino**: **Madeira** pode referir-se à **Ilha da Madeira** ou a **Rua Humberto Madeira**, em Lisboa.\n"
            "- Assumo a interpretação urbana abaixo.\n\n"
            "🗺️ **Route: Metro Santos → Rua Humberto Madeira**\n\n"
            "📍 **LOCATION INFORMATION**\n"
            "**Metro Santos**\n"
            "   🚇 Nearest Metro: **Cais Do Sodré** (🟢 Verde Line)\n"
            "   ℹ️ Resolved dynamically via OpenStreetMap/Nominatim\n\n"
            "**Rua Humberto Madeira**\n"
            "   🚇 Nearest Metro: **Encarnação** (🔴 Vermelha Line)\n"
            "   ℹ️ Resolved dynamically via OpenStreetMap/Nominatim\n\n"
            "🚇 **METRO ROUTE**\n"
            "🔄 **Transfer Required**\n\n"
            "   💡 **Transfer at**: Alameda (🟢 ↔ 🔴)\n"
            "   ⏱️ Estimated travel time: **~35 min** (15 stations + 1 transfer)\n\n"
            "   **Full Route**:\n"
            "   1. Walk from Metro Santos to **Cais Do Sodré**\n"
            "   2. 🟢 Board at **Cais Do Sodré** → direção **Telheiras**\n"
            "   3. Exit at **Alameda**\n"
            "   4. 🔴 Transfer to **Red Line** → direção **Aeroporto**\n"
            "   5. Exit at **Encarnação**\n"
            "   6. Walk to Rua Humberto Madeira\n\n"
            "📌 **Fonte:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | **Atualizado:** 12:00"
        ),
    ):
        agent = TransportAgent()
        agent.system_prompt = "TRANSPORT PROMPT"
        agent._tool_calls_log = []
        agent._build_mode_comparison_response = MagicMock(return_value=None)
        agent._build_mode_constrained_route_response = MagicMock(return_value=None)

        result = agent._resolve_deterministic_response(
            "Quero ir de metro para a Madeira.",
            language="pt",
        )

    assert result is not None
    assert result.lstrip().startswith("⚠️ **Ambiguidade")
    assert "**Trajeto:** Metro Santos → Rua Humberto Madeira" in result
    assert "Ambiguidade no destino" in result
    assert "Ilha da Madeira" in result
    assert "WRONG deterministic metro path" not in result
    assert "**Route:" not in result
    assert "**Informação de localização**" in result
    assert "Metro mais próximo" in result
    assert "**Percurso de metro**" in result
    assert "**É necessária transferência**" in result
    assert "**Percurso completo**" in result
    assert "Caminha desde Metro Santos" in result
    assert "Apanha em **Cais do Sodré**" in result
    assert "Transferência em" in result or "Transfere para" in result
    assert "Sai em **Alameda**" in result
    assert agent.get_tool_calls_log() == [
        {"tool_name": "get_route_between_stations", "args": {"origin": "metro", "destination": "Madeira"}}
    ]


def test_transport_agent_formats_28e_live_snapshot_without_fake_punctuality_claims() -> None:
    """28E live-vehicle prompts should summarize active vehicles without pretending to know punctuality."""
    with patch.object(TransportAgent, "__init__", lambda self: None):
        agent = TransportAgent()
        agent.system_prompt = "TRANSPORT PROMPT"
        agent._tool_calls_log = []
        agent._remember_transport_context = MagicMock()
        agent._rewrite_follow_up_transport_query = lambda message, language: message
        agent.execute_react_loop = MagicMock(side_effect=AssertionError("LLM path should be skipped"))

        realtime_tool = MagicMock()
        realtime_tool.name = "carris_get_realtime_vehicles"
        realtime_tool.invoke = MagicMock(
            return_value=(
                "Veículos Carris em Tempo Real\n"
                "=======================================================\n"
                "Dados de: 20:50:30\n\n"
                "📡 Carris GTFS-RT: live vehicle feed active.\n\n"
                "ELÉTRICOS\n"
                "----------------------------------------\n"
                "28E -> Martim Moniz [Em trânsito]\n"
                "   GPS: 38.71155, -9.13060 | Próxima paragem: Miradouro Sta. Luzia\n"
                "28E -> Campo Ourique (Prazeres) [Em trânsito]\n"
                "   GPS: 38.70810, -9.13990 | Próxima paragem: R. Conceição\n\n"
                "Total: 2 elétricos + 0 autocarros = 2 veículos\n"
            )
        )
        agent.tools = [realtime_tool]

        result = agent.invoke("Is the 28E tram running on time right now, and if not what fallback should I take?")

        realtime_tool.invoke.assert_called_once_with({"route_short_name": "28E"})
        assert "### 🚋 28E live snapshot" in result
        assert "**Active vehicles:** 2 28E tram(s) currently in service." in result
        assert "**Martim Moniz**: 1 vehicle(s)" in result
        assert "**Campo Ourique (Prazeres)**: 1 vehicle(s)" in result
        assert "does not confirm whether the line is on time, delayed, or disrupted" in result
        assert "If you tell me your origin and destination on the 28E, I can suggest a grounded fallback" in result
        assert "[*Carris*](https://www.carris.pt)" in result
        assert "I can confirm there are" not in result


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
