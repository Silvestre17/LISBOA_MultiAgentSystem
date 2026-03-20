# ===========================================================================
# Master Thesis - Response Guardrails Tests
#   - André Filipe Gomes Silvestre, 20240502
#
#   Regression tests for worker-level response finalization, content-filter
#   fallback, and supervisor direct-routing guardrails.
#
#   Run from the repository root with a relative path:
#     python -m pytest tests/test_response_guardrails.py -q
#   Useful parameters:
#     -vv                           verbose mode
#     -k weather or -k supervisor   focus on one guardrail family
#     -x                            stop on first failure
#     --tb=short                    shorter tracebacks
#   Notes:
#     - Prefer relative paths in this workspace. Absolute pytest paths may be
#       treated as glob patterns on Windows because the folder name includes
#       `[` and `]`.
# ===========================================================================

import os
import re
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.agents.base import BaseAgent, clean_response
from agent.agents.planner_agent import PlannerAgent
from agent.agents.qa_agent import QualityAssuranceAgent
from agent.agents.researcher_agent import ResearcherAgent
from agent.agents.supervisor import SupervisorAgent
from agent.agents.transport_agent import (
    TransportAgent,
    _clean_query_fragment,
    _extract_route_endpoints,
)
from agent.agents.weather_agent import WeatherAgent
from agent.graph import MultiAgentAssistant
from agent.utils.response_formatter import (
    finalize_worker_response,
    strip_unsupported_closing_offers,
)
from tools import ipma_api, visitlisboa_api

# ===========================================================================
# Worker finalization tests
# ===========================================================================


def test_weather_worker_finalization_canonicalizes_ipma_source() -> None:
    """Weather worker output should end with a canonical IPMA source line."""
    raw = """⚠️ **Avisos Meteorológicos:**
- 🟡 **Agitação Marítima** - Ondas fortes.

💡 **Dicas Práticas:**
- ☔ Leve guarda-chuva.

📌 **Fonte:** Dados do IPMA. Para informação oficial, consulta ipma.pt
Observação: Se quiser, posso enviar lembretes."""

    output = finalize_worker_response(
        raw,
        agent_name="weather",
        user_query="What is the weather in Lisbon?",
    )

    assert "[*IPMA*](https://www.ipma.pt/en/)" in output
    assert re.search(r"\|\s\*\*Updated:\*\*\s\d{2}:\d{2}", output)
    assert "Observação:" not in output
    assert "lembrete" not in output.lower()


def test_weather_worker_finalization_preserves_tool_update_time() -> None:
    """Weather source lines should prefer the tool timestamp over the current wall clock."""
    raw = """🌤️ Lisbon Weather Summary

📅 Updated: 2026-03-09T09:31:02
📅 Today (2026-03-09):
- 🌡️ Temperature: 8.1°C to 13.1°C
- 🌤️ Conditions: Light rain"""

    output = finalize_worker_response(
        raw,
        agent_name="weather",
        user_query="What is the weather in Lisbon today?",
    )

    assert "**Updated:** 09:31" in output


def test_weather_worker_finalization_keeps_forecast_after_warning_source_line() -> None:
    """Combined warning + forecast weather outputs should not be truncated after the first source line."""
    raw = """⚠️ Active Weather Warnings (LSB):
🟡 Rough sea
📌 Fonte: [IPMA](https://www.ipma.pt) - Instituto Português do Mar e da Atmosfera

---

🌤️ Weather Forecast for Lisbon
📅 Updated: 2026-03-09T15:31:03
🌧️ Tuesday, Mar 10
   🌡️ 8.0°C to 14.9°C"""

    output = finalize_worker_response(
        raw,
        agent_name="weather",
        user_query="Qual é a previsão do tempo para os próximos 3 dias?",
        language="pt",
    )

    assert "Terça-feira, Mar 10" in output
    assert output.count("📌 **Fonte:**") == 1
    assert output.rstrip().endswith("**Atualizado:** 15:31")


def test_weather_worker_finalization_localizes_fast_path_output_for_pt() -> None:
    """Portuguese weather responses should localize common tool-output labels on the fast path."""
    raw = """⚠️ Active Weather Warnings (LSB):
Level: Be aware

---

🌤️ Weather Forecast for Lisbon
📅 Updated: 2026-03-09T15:31:03
🌧️ Monday, Mar 09
   🌡️ Temperature: 8.1°C to 13.1°C
   🌤️ Conditions: Light rain
   💧 Rain: Very likely (94.0%) | Intensity: Weak
   💨 Wind: Northwest (Moderate)"""

    output = finalize_worker_response(
        raw,
        agent_name="weather",
        user_query="Qual é a previsão do tempo para os próximos 3 dias?",
        language="pt",
    )

    assert "Avisos Meteorológicos" in output
    assert "Previsão do Tempo para Lisboa" in output
    assert "Segunda-feira" in output
    assert "Temperatura" in output
    assert "Condições" in output
    assert "Vento" in output


def test_weather_worker_finalization_localizes_summary_labels_for_pt() -> None:
    """Portuguese weather summaries should not leak common English IPMA labels in the UI."""
    raw = """🌤️ Lisbon Weather Summary

📅 Updated: 2026-03-10T14:31:02
📅 Today (2026-03-10):
   🌡️ Temperature: 8.4°C to 15.5°C
   🌤️ Conditions: Sunny intervals
   💧 Rain probability: 60.0%
   💨 Wind: North (Moderate)

✅ No active weather warnings."""

    output = finalize_worker_response(
        raw,
        agent_name="weather",
        user_query="Qual é a previsão detalhada para Lisboa hoje? Existem avisos ativos?",
        language="pt",
    )

    assert "Resumo Meteorológico de Lisboa" in output
    assert "- **📅 Hoje (2026-03-10)**" in output
    assert "  - 🌡️ **Temperatura**: 8.4°C a 15.5°C" in output
    assert "Períodos de céu limpo" in output
    assert "Probabilidade de chuva" in output
    assert "- ✅ Sem avisos meteorológicos ativos." in output
    assert "Sunny intervals" not in output
    assert "Rain probability" not in output
    assert "No active weather warnings" not in output


def test_weather_worker_finalization_localizes_follow_up_warning_summary_for_pt() -> None:
    """Portuguese weather follow-ups should fully localize the warnings summary lines."""
    raw = """✅ No active weather warnings for area 'LSB'.

🌤️ Weather conditions are normal.

---

🌤️ Weather Forecast for Lisbon
📅 Updated: 2026-03-10T14:31:02
☀️ Wednesday, Mar 11
   🌡️ 9.0°C to 19.7°C
   🌤️ Partly cloudy
   💧 Rain: No rain expected (0.0%)
   💨 Wind: North (Moderate)"""

    output = finalize_worker_response(
        raw,
        agent_name="weather",
        user_query="E amanhã?",
        language="pt",
    )

    assert "Sem avisos meteorológicos ativos para Lisboa." in output
    assert "As condições meteorológicas são normais" in output
    assert "- **☀️ Quarta-feira, Mar 11**" in output
    assert "Weather conditions are normal" not in output
    assert "for area 'LSB'" not in output
    assert "LSB" not in output


def test_get_weather_warnings_tool_hides_internal_lisbon_area_code_when_clear() -> None:
    """The weather warnings tool should never expose the internal Lisbon IPMA code in user-facing clear-weather text."""
    with patch.object(
        ipma_api,
        "fetch_json",
        return_value=[{"idAreaAviso": "LSB", "awarenessLevelID": "green"}],
    ):
        output = ipma_api.get_weather_warnings.invoke({"area": "LSB"})

    assert "LSB" not in output
    assert "No active weather warnings for Lisbon." in output


def test_get_weather_warnings_tool_hides_internal_lisbon_area_code_when_active() -> None:
    """Active warning headers should mention Lisbon, never the raw LSB code."""
    mock_warning = {
        "idAreaAviso": "LSB",
        "awarenessLevelID": "yellow",
        "awarenessTypeName": "Vento",
        "text": "Rajadas fortes no litoral.",
        "startTime": "2026-03-11T09:00:00",
        "endTime": "2026-03-11T18:00:00",
    }

    with patch.object(ipma_api, "fetch_json", return_value=[mock_warning]):
        output = ipma_api.get_weather_warnings.invoke({"area": "LSB"})

    assert "LSB" not in output
    assert "Active Weather Warnings for Lisbon:" in output


def test_supervisor_frequency_query_does_not_trigger_weather() -> None:
    """Generic PT frequency queries with the word `tempo` should stay transport-only."""
    with patch.object(SupervisorAgent, "__init__", lambda self: None):
        agent = SupervisorAgent()
        decision = agent._fallback_routing(
            "De quanto em quanto tempo passa o 28E?",
            "",
            language="pt",
        )

        assert "transport" in decision["agents"]
        assert "weather" not in decision["agents"]


def test_supervisor_events_today_query_does_not_force_planner_weather() -> None:
    """A simple events-today query should not be promoted into planning logic."""
    with patch.object(SupervisorAgent, "__init__", lambda self: None):
        agent = SupervisorAgent()
        agent.llm = object()
        agent._safe_llm_invoke = MagicMock(
            return_value=MagicMock(
                content='{"reasoning": "Events today query", "agents": ["researcher"], "direct_response": null}'
            )
        )

        decision = agent.route("Are there any events today?", language="en")

        assert decision["agents"] == ["researcher"]


def test_supervisor_events_this_week_query_stays_research_only_in_pt() -> None:
    """Portuguese event-discovery queries for this week should stay in the researcher domain."""
    with patch.object(SupervisorAgent, "__init__", lambda self: None):
        agent = SupervisorAgent()
        agent.llm = object()
        agent._safe_llm_invoke = MagicMock(
            return_value=MagicMock(
                content='{"reasoning": "Consulta de eventos culturais na AML", "agents": ["researcher"], "direct_response": null}'
            )
        )

        decision = agent.route(
            "Quero explorar a cultura local. Que grandes eventos temos esta semana?",
            language="pt",
        )

        assert decision["agents"] == ["researcher"]


def test_supervisor_standalone_attractions_query_ignores_previous_topic_history() -> None:
    """A fresh attractions query should not inherit weather/transport agents from previous turns."""
    with patch.object(SupervisorAgent, "__init__", lambda self: None):
        agent = SupervisorAgent()

        decision = agent.route(
            "Lista as atrações imperdíveis para quem visita Lisboa pela primeira vez.",
            language="pt",
            conversation_history=[
                HumanMessage(content="Que grandes eventos temos esta semana?"),
                AIMessage(content="Event list"),
                HumanMessage(content="E amanhã chove?"),
            ],
        )

        assert decision["agents"] == ["researcher"]


def test_supervisor_lmstudio_route_uses_same_llm_path_as_other_providers() -> None:
    """LM Studio supervisor routing should use the same LLM path as cloud providers."""
    with patch.object(SupervisorAgent, "__init__", lambda self: None):
        agent = SupervisorAgent()
        agent.llm_provider = "lmstudio"
        agent.llm = object()
        agent._safe_llm_invoke = MagicMock(
            return_value=MagicMock(
                content='{"reasoning": "Route with LLM", "agents": ["weather", "transport"], "direct_response": null}'
            )
        )

        decision = agent.route(
            "Tell me the weather today and how I get from Rossio to Belém.",
            language="en",
        )

        agent._safe_llm_invoke.assert_called_once()
        assert decision["agents"] == ["weather", "transport"]
        assert decision["direct_response"] is None


def test_supervisor_fallback_mixed_weather_and_route_query_detects_both_domains() -> None:
    """Fallback routing should capture both weather and transport in natural PT route phrasing."""
    with patch.object(SupervisorAgent, "__init__", lambda self: None):
        agent = SupervisorAgent()

        decision = agent._fallback_routing(
            "Diz-me o tempo hoje em Lisboa e como vou do Rossio para Belém.",
            llm_response="",
            language="pt",
        )

        assert "weather" in decision["agents"]
        assert "transport" in decision["agents"]
        assert "researcher" not in decision["agents"]


def test_supervisor_planning_query_forces_research_transport_and_weather_when_needed() -> None:
    """Planning queries should deterministically include the grounding workers needed for a robust itinerary."""
    with patch.object(SupervisorAgent, "__init__", lambda self: None):
        agent = SupervisorAgent()
        agent.llm = object()
        agent._safe_llm_invoke = MagicMock(
            return_value=MagicMock(
                content='{"reasoning": "Planner-only draft", "agents": ["planner"], "direct_response": null}'
            )
        )

        decision = agent.route(
            "Planeia a minha tarde em Belém, diz-me como lá chegar a partir do Rossio e considera o tempo.",
            language="pt",
        )

        assert decision["agents"] == ["planner", "researcher", "transport", "weather"]



def test_researcher_worker_finalization_uses_places_source_label_en() -> None:
    """Place queries should use the VisitLisboa Places source label in English."""
    raw = """**1.** 🏛️ **Museu Nacional do Azulejo**
- 📍 **Morada**: Rua da Madre de Deus 4, Lisboa
- 🌐 **[Site Oficial](https://www.museudoazulejo.pt/)**

📌 **Source:** [*VisitLisboa*](https://www.visitlisboa.com/en/places)
Observation: If you want, I can fetch updated prices and opening hours."""

    output = finalize_worker_response(
        raw,
        agent_name="researcher",
        user_query="Museums in Lisbon",
    )

    assert "[*VisitLisboa Places*](https://www.visitlisboa.com/en/places)" in output
    assert "Observation:" not in output
    assert "updated prices" not in output.lower()



def test_researcher_worker_finalization_uses_events_source_label_pt() -> None:
    """Event queries should use the VisitLisboa Eventos source label in PT-PT."""
    raw = """**1.** 🎭 **Concerto em Lisboa**
- 📅 **Data/Hora**: Hoje, 21:00
- 📍 **Morada**: Lisboa

📌 **Fonte:** [*VisitLisboa*](https://www.visitlisboa.com/pt-pt/eventos)"""

    output = finalize_worker_response(
        raw,
        agent_name="researcher",
        user_query="Eventos hoje em Lisboa",
    )

    assert "[*VisitLisboa Eventos*](https://www.visitlisboa.com/pt-pt/eventos)" in output


def test_researcher_worker_finalization_localizes_common_labels_for_en() -> None:
    """English researcher answers should not leak common PT-PT field labels."""
    raw = """**1.** 🏛️ **Museu Nacional de Arte Antiga**
- 📍 **Morada**: Rua das Janelas Verdes, Lisboa
- 🕒 **Horário**: Hoje: 10:00 - 18:00
- 💡 **Dica**: Verifique descontos Lisboa Card.

📌 **Fonte:** [*VisitLisboa*](https://www.visitlisboa.com/en/places)"""

    output = finalize_worker_response(
        raw,
        agent_name="researcher",
        user_query="Best museums in Lisbon",
    )

    assert "**Address**" in output
    assert "**Opening hours**" in output
    assert "**Tip**" in output
    assert "Morada" not in output
    assert "Horário" not in output


def test_researcher_worker_finalization_strips_raw_tool_artifacts() -> None:
    """Researcher post-processing should remove raw tool summary scaffolding."""
    raw = """🏛️ **Found 2 Places/Attractions in Lisbon:**

1. 🏛️ **National Ancient Art Museum**
   📂 Category: Museums & Monuments
   Name: National Ancient Art Museum
Url: https://www.visitlisboa.com/en/places/national-ancient-art-museum
Category: Museums & Monuments
Short Description: Major public museum in Lisbon.
   📍 Lisbon
   🔗 https://www.visitlisboa.com/en/places/national-ancient-art-museum

📊 **Sources:** 2 from VisitLisboa, 0 from Lisboa Aberta
💡 Try more specific queries for better results.

📌 **Source:** [*VisitLisboa*](https://www.visitlisboa.com/en/places)"""

    output = finalize_worker_response(
        raw,
        agent_name="researcher",
        user_query="Museums in Lisbon",
    )

    assert "Found 2 Places/Attractions" not in output
    assert "**Name**:" not in output
    assert "**Url**:" not in output
    assert "**Short Description**:" not in output
    assert "Try more specific queries" not in output
    assert "[*VisitLisboa Places*](https://www.visitlisboa.com/en/places)" in output


def test_researcher_worker_finalization_structures_ranked_results_into_nested_lists() -> None:
    """Numbered researcher outputs should render as nested markdown lists, not dense paragraphs."""
    raw = """1. 📅 **Artur Pizarro Prokofiev 2**
   🗓️ **When:** 14 Mar at 19:00
   📂 **Category**: Music
   📍 Avenida da Liberdade 182-188 Lisboa

2. 🏛️ **Mosteiro dos Jerónimos**
   📂 **Category**: Museums & Monuments
   📍 Praça do Império, Lisboa

📌 **Fonte:** [*VisitLisboa Eventos*](https://www.visitlisboa.com/pt-pt/eventos)"""

    output = finalize_worker_response(
        raw,
        agent_name="researcher",
        user_query="Que grandes eventos temos esta semana?",
        language="pt",
    )

    assert "- 📅 **Artur Pizarro Prokofiev 2**" in output
    assert "  - 🗓️ **Quando:** 14 Mar às 19:00" in output
    assert "  - 📂 **Categoria**: Música" in output
    assert "- 🏛️ **Mosteiro dos Jerónimos**" in output


def test_researcher_worker_finalization_localizes_common_metadata_values_for_pt() -> None:
    """PT-PT researcher answers should localize common English metadata values, not only the field labels."""
    raw = """1. 📅 **Artur Pizarro Prokofiev 2**
   🗓️ **When:** 14 Mar at 19:00
   ⏱️ **Duration:** 🎯 Single day
   📂 **Category**: Music
   📍 Lisbon
   💰 **Price**: From €15 to €35
   ⭐ **TripAdvisor**: 4.5/5 (197 reviews)

📌 **Fonte:** [*VisitLisboa Eventos*](https://www.visitlisboa.com/pt-pt/eventos)"""

    output = finalize_worker_response(
        raw,
        agent_name="researcher",
        user_query="Que grandes eventos temos esta semana?",
        language="pt",
    )

    assert "**Quando:** 14 Mar às 19:00" in output
    assert "**Duração:** 🎯 Um só dia" in output
    assert "**Categoria**: Música" in output
    assert "📍 Lisboa" in output
    assert "**Preço**: de €15 a €35" in output
    assert "(197 avaliações)" in output
    assert "Single day" not in output
    assert " reviews" not in output


def test_researcher_worker_finalization_localizes_event_summary_notes_for_pt() -> None:
    """PT-PT event summaries should explain filters cleanly and never leak unnamed placeholders."""
    raw = """1. 📅 **Artur Pizarro Prokofiev 2**
   🗓️ **When:** 14 Mar at 19:00

🧭 **Filter used:** this week (2026-03-11 to 2026-03-16), all categories, broad event discovery.
📊 **Result count:** 26 confirmed-date event(s) match this filter.
✨ **Highlights shown:** 5 most relevant result(s).
⚠️ **Source completeness note:** 30 additional matching record(s) were excluded because the source does not confirm their dates yet.

📌 **Fonte:** [*VisitLisboa Eventos*](https://www.visitlisboa.com/pt-pt/eventos)"""

    output = finalize_worker_response(
        raw,
        agent_name="researcher",
        user_query="Que grandes eventos temos esta semana?",
        language="pt",
    )

    assert "**Filtro aplicado:** esta semana (2026-03-11 a 2026-03-16), todas as categorias, pesquisa geral de eventos." in output
    assert "**Resultado do filtro:** 26 evento(s) com data confirmada correspondem a este filtro." in output
    assert "**Destaques mostrados:** 5 resultado(s) mais relevantes." in output
    assert "**Nota sobre a completude da fonte:** 30 registo(s) adicional(is) compatíveis foram excluídos" in output
    assert "Evento sem nome" not in output
    assert "Unknown event" not in output


def test_researcher_worker_finalization_sanitizes_slug_suffixes_and_adds_description_label() -> None:
    """Event cards should drop slug-like numeric suffixes and label concise descriptions explicitly."""
    raw = """1. 📅 **Michael Lives Forever 0326**
   🗓️ **When:** 13 Mar
   📂 **Category**: Music
   📝 **Description:** The show includes classic songs such as \"Billie Jean\", \"Thriller\" and \"Smooth Criminal\".

📌 **Fonte:** [*VisitLisboa Eventos*](https://www.visitlisboa.com/pt-pt/eventos)"""

    output = finalize_worker_response(
        raw,
        agent_name="researcher",
        user_query="Que grandes eventos temos esta semana?",
        language="pt",
    )

    assert "Michael Lives Forever 0326" not in output
    assert "Michael Lives Forever" in output
    assert "**Descrição:**" in output


def test_transport_worker_finalization_strips_gps_and_internal_ids() -> None:
    """Final transport responses must not expose GPS coordinates, stop IDs, or plate metadata."""
    raw = """🚌 **Real-Time Bus Locations - Line 1001**

- 📍 GPS: 38.72410, -9.14820
- 🚏 Next stop ID: 060001
- **Plate**: 12-AB-34
- **Status**: On time

📌 **Fonte:** [*Carris Metropolitana*](https://www.carrismetropolitana.pt) | **Atualizado:** 20:10"""

    output = finalize_worker_response(
        raw,
        agent_name="transport",
        user_query="Mostra os autocarros em tempo real",
        language="pt",
    )

    assert "GPS:" not in output
    assert "060001" not in output
    assert "Plate" not in output
    assert "Matrícula" not in output
    assert "Status" in output or "Estado" in output


def test_researcher_direct_event_lookup_bypasses_llm_for_week_query() -> None:
    """Broad date-based event discovery queries should skip the free-form LLM path."""
    with patch.object(ResearcherAgent, "__init__", lambda self: None):
        agent = ResearcherAgent()
        agent.system_prompt = "PRIMARY PROMPT"

        events_tool = MagicMock()
        events_tool.name = "search_cultural_events"
        events_tool.invoke = MagicMock(
            return_value=(
                "1. 📅 **Artur Pizarro Prokofiev 2**\n"
                "   🗓️ **When:** 14 Mar at 19:00"
            )
        )
        agent.tools = [events_tool]
        agent.execute_react_loop = MagicMock(side_effect=AssertionError("LLM flow should be skipped"))

        output = agent.invoke("Que grandes eventos temos esta semana?", context="", verbose=False)

        events_tool.invoke.assert_called_once_with({"max_results": 5, "language": "pt", "date_filter": "this week"})
        assert "Artur Pizarro Prokofiev 2" in output
        assert "[*VisitLisboa Eventos*](https://www.visitlisboa.com/pt-pt/eventos)" in output


def test_researcher_accessibility_query_strips_unconfirmed_accessibility_claims() -> None:
    """Accessibility queries should not surface unconfirmed wheelchair-access claims from researcher drafts."""
    raw = """**1.** 🏛️ **Museum of Illusions**
- ♿ Wheelchair accessible with lifts and adapted toilets

📌 **Source:** [*VisitLisboa*](https://www.visitlisboa.com/en/places)"""

    output = finalize_worker_response(
        raw,
        agent_name="researcher",
        user_query="I use a wheelchair. Which museums in Belem are accessible?",
    )

    assert "wheelchair accessible" not in output.lower()
    assert "adapted toilets" not in output.lower()
    assert "not confirmed" in output.lower()


def test_transport_worker_finalization_localizes_summary_counts_for_en() -> None:
    """English transport summaries should not keep PT-only count nouns."""
    raw = """🚇 🚌 🚆 **Situação dos Transportes de Lisboa** — Atualizado: 20:52

🚇 **Metro de Lisboa**
- 🟢 **Estado**: Circulação normal em todas as linhas

🚋 **Carris (Urbano)**
- 🟢 **Veículos em serviço**: 175 veículos

🚌 **Carris Metropolitana (Suburbano)**
- ⚠️ **Alertas ativos**: 86 alertas

🚆 **CP Comboios (AML)**
- 📊 **Comboios a circular na AML**: 33 comboios"""

    output = finalize_worker_response(
        raw,
        agent_name="transport",
        user_query="Is the metro working?",
    )

    assert "Lisbon Transport Status" in output
    assert "175 vehicles" in output
    assert "86 alerts" in output
    assert "33 trains" in output
    assert "veículos" not in output
    assert "comboios" not in output


def test_transport_worker_finalization_strips_embedded_stop_and_vehicle_ids_from_arrivals() -> None:
    """Transport finalization should remove embedded stop IDs and vehicle IDs from Carris arrival summaries."""
    raw = """🚌 **Rossio → Próximas chegadas (paragem ID 908)**

- 🚌 **732** - **Destino:** Hosp. Egas Moniz / Restauradores
    - 🕒 **10:27** (vehicle 6045, 2 paragens restantes) — **Em tempo real**

- 🚌 **711** - **Destino:** Sul e Sueste / Alto Damaia
    - 🕒 **10:32** — (em trânsito, 2 paragens restantes; viatura **2784**, matrícula **AI-09-BT**)

- 🚋 **51E** - **Destino:** Glória / Restauradores (Elétrico)
    - 🕒 **10:33** — **Horario**

📌 **Fonte:** [*Carris*](https://www.carris.pt) | **Atualizado:** 10:22"""

    output = finalize_worker_response(
        raw,
        agent_name="transport",
        user_query="Quais os próximos autocarros da Carris no Rossio?",
        language="pt",
    )

    assert "paragem ID 908" not in output
    assert "vehicle 6045" not in output.lower()
    assert "viatura **2784**" not in output.lower()
    assert "matrícula **ai-09-bt**" not in output.lower()
    assert "2 paragens restantes" in output
    assert "Horário" in output


def test_planner_worker_finalization_rebuilds_timed_sections_cleanly() -> None:
    """Planner finalization should convert malformed pseudo-headers into clean timed activity sections."""
    raw = """📅 Itinerário sugerido para uma tarde em Belém

### - 🔹 **Antes de sair**: verifique a previsão meteorológica atualizada.
🏛️ 14:00 — Chegada a Belém (Praça do Império)
- Comece pela praça para contextualizar os monumentos.

### - Observe a arquitetura manuelina do exterior.
🏛️ 14:15 — Mosteiro dos Jerónimos

Dicas práticas e notas importantes
- 🚇 **Transporte**: confirme tempos e rotas antes de sair.

📌 **Fonte:** [*VisitLisboa*](https://www.visitlisboa.com) | [*Metro de Lisboa*](https://www.metrolisboa.pt) | **Atualizado:** 10:22"""

    output = finalize_worker_response(
        raw,
        agent_name="planner",
        user_query="Sugere um plano para uma tarde em Belém com detalhes históricos e onde comer um pastel.",
        language="pt",
    )

    assert "### - " not in output
    assert "**⛅ Antes de sair**" in output
    assert "### 🏛️ 14:00 · Chegada a Belém (Praça do Império)" in output
    assert "### 🏛️ 14:15 · Mosteiro dos Jerónimos" in output
    assert "- Observe a arquitetura manuelina do exterior." in output
    assert "**✨ Dicas práticas e notas importantes**" in output


def test_planner_lmstudio_uses_same_prompt_structure_as_other_providers() -> None:
    """LM Studio planner invocations should reuse the same prompt structure as cloud models."""
    with patch.object(PlannerAgent, "__init__", lambda self: None), patch(
        "agent.agents.planner_agent.finalize_worker_response",
        side_effect=lambda response, **_kwargs: response,
    ):
        agent = PlannerAgent()
        agent.system_prompt = "PLANNER PROMPT"
        agent.llm_provider = "lmstudio"
        agent.llm = object()
        agent._safe_llm_invoke = MagicMock(return_value=SimpleNamespace(content="Draft itinerary"))

        output = agent.invoke(
            user_message="Plan my afternoon in Belém.",
            weather_data="Sunny.",
            transport_data="Metro available.",
            places_data="1. **Jerónimos Monastery**",
        )

        messages = agent._safe_llm_invoke.call_args.args[1]
        system_messages = [message.content for message in messages if isinstance(message, SystemMessage)]
        assert system_messages[0] == "PLANNER PROMPT"
        assert any("Respond ENTIRELY in English." in content for content in system_messages)
        assert any("GROUNDING RULES:" in content for content in system_messages)
        assert any("OUTPUT BUDGET:" in content for content in system_messages)
        assert any("# Data from Specialized Agents" in content for content in system_messages)
        assert output == "Draft itinerary"


def test_planner_falls_back_to_deterministic_template_when_llm_fails() -> None:
    """Planner should return a compact deterministic itinerary if the planner LLM fails."""
    with patch.object(PlannerAgent, "__init__", lambda self: None):
        agent = PlannerAgent()
        agent.system_prompt = "PLANNER PROMPT"
        agent.llm_provider = "lmstudio"
        agent.llm = object()
        agent._safe_llm_invoke = MagicMock(side_effect=TimeoutError("planner timeout"))

        output = agent.invoke(
            user_message="Planeia a minha tarde em Belém e considera o tempo.",
            weather_data="- Chuva forte provável\n📌 **Fonte:** [*IPMA*](https://www.ipma.pt)",
            transport_data="- Confirma o trajeto em carris.pt\n📌 **Fonte:** [*Carris*](https://www.carris.pt)",
            places_data="- **MAAT**\n- **Museu Nacional dos Coches**",
            qa_disclaimers=["Verifique horários antes de sair."],
        )

        assert "### 📅" in output
        assert "**⛅ Condições e segurança**" in output
        assert "**🚇 Como chegar e deslocação**" in output
        assert "**📍 Sugestões para a visita**" in output
        assert "**✨ Notas práticas**" in output
        assert "MAAT" in output


def test_planner_falls_back_when_cleaned_response_is_generic_processing_error() -> None:
    """Planner should also switch to the deterministic template when clean_response collapses the draft into a generic error."""
    with patch.object(PlannerAgent, "__init__", lambda self: None):
        agent = PlannerAgent()
        agent.system_prompt = "PLANNER PROMPT"
        agent.llm_provider = "lmstudio"
        agent.llm = object()
        agent._safe_llm_invoke = MagicMock(
            return_value=SimpleNamespace(
                content="How do I get there?\n\nWe are in Portuguese.\n\nStep-by-step:"
            )
        )

        output = agent.invoke(
            user_message="Planeia a minha tarde em Belém e considera o tempo.",
            weather_data="- Chuva forte provável",
            transport_data="- Confirma o trajeto em carris.pt",
            places_data="- **MAAT**",
            qa_disclaimers=["Verifique horários antes de sair."],
        )

        assert "Itinerário sugerido" in output
        assert "MAAT" in output
        assert "dificuldades em processar" not in output.lower()


def test_transport_worker_finalization_groups_live_and_scheduled_arrivals() -> None:
    """Carris arrival summaries should group real-time and scheduled items instead of repeating the schedule label per line."""
    raw = """🚌 **Rossio → Próximas chegadas (paragem ID 908)**

- 🚌 **732** - **Destino:** Hosp. Egas Moniz / Restauradores
    - 🕒 **10:27** (vehicle 6045, 2 paragens restantes) — **Em tempo real**

- 🚌 **711** - **Destino:** Sul e Sueste / Alto Damaia
    - 🕒 **10:32** (vehicle 2784, 5 paragens restantes) — **Em tempo real**

- 🚋 **51E** - **Destino:** Glória / Restauradores (Elétrico)
    - 🕒 **10:33** — **Horario**

- 🚌 **759** - **Destino:** Restauradores / Estação Oriente
    - 🕒 **10:34** — **Horario**

💡 **Dica rápida:** Os tempos em “Em tempo real” usam GPS — aparecem veículos identificados.

📌 **Fonte:** [*Carris*](https://www.carris.pt) | **Atualizado:** 10:22"""

    output = finalize_worker_response(
        raw,
        agent_name="transport",
        user_query="Quais os próximos autocarros da Carris no Rossio?",
        language="pt",
    )

    assert "### 🚌 Rossio · Próximas chegadas" in output
    assert "**Em tempo real**" in output
    assert "**Horários programados**" in output
    assert "vehicle 6045" not in output.lower()
    assert "veículos identificados" not in output.lower()
    assert "- 🚌 **732** → Hosp. Egas Moniz / Restauradores · **10:27** · 2 paragens restantes" in output
    assert "- 🚋 **51E** → Glória / Restauradores (Elétrico) · **10:33**" in output


def test_transport_worker_finalization_strips_inline_gps_vehicle_and_plate_metadata_in_pt() -> None:
    """PT transport answers should hide inline GPS coordinates, vehicle IDs, and license plates."""
    raw = """### 🚇 Informação de Transportes

- 🚌 **Paragem: Rossio** — paragem ID: **908** — GPS: **38.71331, -9.13962** (dados atualizados às **10:33**)
- 🕒 **Próxima chegada:** **11:03** — **em tempo real: +5 min (atraso)** | veículo: **2685** (matrícula **93-XA-46**)
- 🕒 **Seguinte:** **11:23** — **Veículo 22685 (matrícula 99-XE-555)**

📌 **Fonte:** [*Carris*](https://www.carris.pt) | **Atualizado:** 10:33"""

    output = finalize_worker_response(
        raw,
        agent_name="transport",
        user_query="Quais os próximos autocarros da Carris no Rossio?",
        language="pt",
    )

    assert "908" not in output
    assert "38.71331" not in output
    assert "2685" not in output
    assert "93-XA-46" not in output
    assert "22685" not in output
    assert "99-XE-555" not in output
    assert "dados atualizados" in output.lower()


def test_transport_worker_finalization_strips_weather_disclaimer_block() -> None:
    """Transport answers should not keep cross-domain weather disclaimers when weather is handled elsewhere."""
    raw = """Aqui está o ponto de situação para o teu trajeto:

⛈️ **Tempo em Lisboa**
- Infelizmente não tenho acesso a dados meteorológicos em tempo real. Recomendo consultar [IPMA](https://www.ipma.pt) ou o [In-Weather](https://in-weather.com) para previsões detalhadas.

🚇🚌 **Como ir do Rossio para Belém:**

**Opção 1: Autocarro (Carris Urbano)**
- 📍 **Embarque**: Rossio
- 🚌 **Linha 732** - para Caselas — ~43 min

📌 **Fonte:** [*Carris*](https://www.carris.pt) | **Atualizado:** 11:03"""

    output = finalize_worker_response(
        raw,
        agent_name="transport",
        user_query="Diz-me o tempo hoje em Lisboa e como vou do Rossio para Belém.",
        language="pt",
    )

    assert "não tenho acesso a dados meteorológicos" not in output.lower()
    assert "in-weather" not in output.lower()
    assert "Como ir do Rossio para Belém" in output


def test_transport_worker_finalization_strips_inline_weather_side_note_variant() -> None:
    """Transport answers should remove inline weather-side notes such as 'Sobre o tempo em Lisboa'."""
    raw = """### 🚌 Sobre o tempo em Lisboa: 🌤️ Não tenho acesso a dados meteorológicos no momento. Recomendo verificar o [IPMA](https://www.ipma.pt) ou o [Google Weather](https://weather.google.com) para a previsão atualizada!

**Horários programados**
- 🚌 **732** → para Caselas · **11:25** · Próximo: (tempo real)

📌 **Fonte:** [*Carris*](https://www.carris.pt) | **Atualizado:** 11:10"""

    output = finalize_worker_response(
        raw,
        agent_name="transport",
        user_query="Diz-me o tempo hoje em Lisboa e como vou do Rossio para Belém.",
        language="pt",
    )

    assert "Sobre o tempo em Lisboa" not in output
    assert "google weather" not in output.lower()
    assert "Horários programados" in output


def test_strip_unsupported_closing_offers_removes_inline_offer_clause() -> None:
    """Inline follow-up offers should be removed, not just standalone offer lines."""
    raw = (
        "That topic is outside scope for this assistant. "
        "Se quiser, posso verificar o tempo em Lisboa por si."
    )

    output = strip_unsupported_closing_offers(raw)

    assert output == "That topic is outside scope for this assistant."


# ===========================================================================
# Researcher content-filter fallback tests
# ===========================================================================


def test_researcher_content_filter_fallback_uses_safe_prompt() -> None:
    """Researcher should retry with the safe prompt variant after content filter."""
    with patch.object(ResearcherAgent, "__init__", lambda self: None):
        agent = ResearcherAgent()
        agent.system_prompt = "PRIMARY PROMPT"

        prompt_calls: list[str] = []

        def fake_execute_react_loop(*, messages, verbose, max_iterations, tool_enforcement_msg):
            prompt_calls.append(messages[0].content)
            if len(prompt_calls) == 1:
                raise RuntimeError(
                    "400 content_filter ResponsibleAIPolicyViolation jailbreak"
                )
            return (
                "**1.** 🏛️ **Museu Calouste Gulbenkian**\n"
                "- 📍 **Morada**: Avenida de Berna 45A, Lisboa\n\n"
                "📌 **Source:** [*VisitLisboa*](https://www.visitlisboa.com/en/places)"
            )

        agent.execute_react_loop = fake_execute_react_loop

        output = agent.invoke("Museums in Lisbon")

        assert len(prompt_calls) == 2
        assert prompt_calls[0] == "PRIMARY PROMPT"
        assert prompt_calls[1] != prompt_calls[0]
        assert "Lisbon Places and Events Researcher" in prompt_calls[1]
        assert "[*VisitLisboa Places*](https://www.visitlisboa.com/en/places)" in output


def test_researcher_double_content_filter_falls_back_to_direct_tool() -> None:
    """Researcher should fall back to direct tool invocation after two blocks."""
    with patch.object(ResearcherAgent, "__init__", lambda self: None):
        agent = ResearcherAgent()
        agent.system_prompt = "PRIMARY PROMPT"

        dummy_places_tool = MagicMock()
        dummy_places_tool.name = "search_places_attractions"
        dummy_places_tool.invoke = MagicMock(
            return_value=(
                "**1.** 🏛️ **Museu Nacional de Arte Antiga**\n"
                "- 📍 **Address**: Rua das Janelas Verdes, Lisbon"
            )
        )
        agent.tools = [dummy_places_tool]

        call_counter = {"count": 0}

        def fake_execute_react_loop(*, messages, verbose, max_iterations, tool_enforcement_msg):
            call_counter["count"] += 1
            raise RuntimeError("400 content_filter ResponsibleAIPolicyViolation jailbreak")

        agent.execute_react_loop = fake_execute_react_loop

        output = agent.invoke("Museums in Lisbon")

        assert call_counter["count"] == 2
        dummy_places_tool.invoke.assert_called_once_with(
            {"query": "Museums in Lisbon", "max_results": 5, "category": "Museums & Monuments"}
        )
        assert "[*VisitLisboa Places*](https://www.visitlisboa.com/en/places)" in output


def test_researcher_accessibility_place_queries_skip_freeform_llm() -> None:
    """Accessibility-focused place queries should go straight to the place-search tool."""
    with patch.object(ResearcherAgent, "__init__", lambda self: None):
        agent = ResearcherAgent()
        agent.system_prompt = "PRIMARY PROMPT"

        dummy_places_tool = MagicMock()
        dummy_places_tool.name = "search_places_attractions"
        dummy_places_tool.invoke = MagicMock(
            return_value=(
                "**1.** 🏛️ **Jerónimos Monastery**\n"
                "- 📍 **Address**: Praça do Império"
            )
        )
        agent.tools = [dummy_places_tool]
        agent.execute_react_loop = MagicMock(side_effect=AssertionError("LLM flow should be skipped"))

        output = agent.invoke("Belem museums wheelchair accessible")

        dummy_places_tool.invoke.assert_called_once_with(
            {"query": "Belem museums wheelchair accessible", "max_results": 5, "category": "Museums & Monuments"}
        )
        assert "Jerónimos Monastery" in output
        assert "[*VisitLisboa Places*](https://www.visitlisboa.com/en/places)" in output


def test_researcher_direct_place_lookup_applies_restaurant_category_for_dining_queries() -> None:
    """Dining queries should narrow deterministic place lookups to the restaurant category."""
    with patch.object(ResearcherAgent, "__init__", lambda self: None):
        agent = ResearcherAgent()
        agent.system_prompt = "PRIMARY PROMPT"

        dummy_places_tool = MagicMock()
        dummy_places_tool.name = "search_places_attractions"
        dummy_places_tool.invoke = MagicMock(
            return_value=(
                "**1.** 🏛️ **5 Oceanos**\n"
                "- 📍 **Address**: Doca do Bom Sucesso, Lisbon"
            )
        )
        agent.tools = [dummy_places_tool]
        agent.execute_react_loop = MagicMock(side_effect=AssertionError("LLM flow should be skipped"))

        output = agent.invoke("Best seafood restaurants near the Tagus river.")

        dummy_places_tool.invoke.assert_called_once_with(
            {"query": "Best seafood restaurants near the Tagus river.", "max_results": 5, "category": "Restaurants"}
        )
        assert "5 Oceanos" in output
        assert "[*VisitLisboa Places*](https://www.visitlisboa.com/en/places)" in output


def test_weather_double_content_filter_falls_back_to_direct_tools() -> None:
    """Weather should fall back to direct tool invocation after two prompt blocks."""
    with patch.object(WeatherAgent, "__init__", lambda self: None):
        agent = WeatherAgent()
        agent.system_prompt = "PRIMARY WEATHER PROMPT"

        warnings_tool = MagicMock()
        warnings_tool.name = "get_weather_warnings"
        warnings_tool.invoke = MagicMock(return_value="✅ No active weather warnings for area 'LSB'.")

        forecast_tool = MagicMock()
        forecast_tool.name = "get_weather_forecast"
        forecast_tool.invoke = MagicMock(return_value="🌤️ Weather Forecast for Lisbon\n\n☀️ Sunday\n   🌡️ 9°C to 16°C")

        agent.tools = [warnings_tool, forecast_tool]

        call_counter = {"count": 0}

        def fake_execute_react_loop(*, messages, verbose, max_iterations, tool_enforcement_msg):
            call_counter["count"] += 1
            raise RuntimeError("400 content_filter ResponsibleAIPolicyViolation jailbreak")

        agent.execute_react_loop = fake_execute_react_loop

        output = agent.invoke("Plan the safest outdoor activities in Lisbon this week based on the weather")

        assert call_counter["count"] == 2
        warnings_tool.invoke.assert_called_once_with({"area": "LSB"})
        forecast_tool.invoke.assert_called_once_with({"days": 5})
        assert "[*IPMA*](https://www.ipma.pt/en/)" in output


def test_weather_english_language_drift_uses_direct_tool_fallback() -> None:
    """English weather queries should fall back to deterministic tool output if the model drifts into PT."""
    with patch.object(WeatherAgent, "__init__", lambda self: None):
        agent = WeatherAgent()
        agent.system_prompt = "PRIMARY WEATHER PROMPT"

        current_tool = MagicMock()
        current_tool.name = "get_current_weather_summary"
        current_tool.invoke = MagicMock(
            return_value=(
                "🌤️ Lisbon Weather Summary\n"
                "========================================\n\n"
                "📅 Today (2026-03-09):\n"
                "   🌡️ Temperature: 8.1°C to 13.1°C\n"
                "   🌤️ Conditions: Light rain\n"
                "   💧 Rain probability: 94.0% (Weak)\n"
                "   💨 Wind: Northwest (Moderate)"
            )
        )
        agent.tools = [current_tool]

        agent.execute_react_loop = MagicMock(
            return_value=(
                "**📅 Segunda-feira, 9 de Março**\n"
                "- 🌡️ **Temperatura**: 8.1°C a 13.1°C\n"
                "- ☁️ **Condições**: Chuva leve\n"
                "- 💡 **Dicas Práticas**:\n"
                "- 🧥 Vista casaco"
            )
        )

        output = agent.invoke("What is the weather in Lisbon?")

        current_tool.invoke.assert_called_once_with({})
        assert "Segunda-feira" not in output
        assert "Vista casaco" not in output
        assert "[*IPMA*](https://www.ipma.pt/en/)" in output


def test_weather_current_query_uses_direct_current_summary_tool() -> None:
    """Simple current-weather queries should bypass the free-form LLM loop."""
    with patch.object(WeatherAgent, "__init__", lambda self: None):
        agent = WeatherAgent()
        agent.system_prompt = "PRIMARY WEATHER PROMPT"

        current_tool = MagicMock()
        current_tool.name = "get_current_weather_summary"
        current_tool.invoke = MagicMock(
            return_value=(
                "🌤️ Lisbon Weather Summary\n"
                "📅 Updated: 2026-03-09T11:31:02\n"
                "📅 Today (2026-03-09):\n"
                "   🌡️ Temperature: 8.1°C to 13.1°C"
            )
        )
        agent.tools = [current_tool]
        agent.execute_react_loop = MagicMock(side_effect=AssertionError("LLM path should be skipped"))

        output = agent.invoke("What's the weather like in Lisbon today?")

        current_tool.invoke.assert_called_once_with({})
        assert "**Updated:** 11:31" in output


def test_weather_lmstudio_multiagent_context_uses_same_react_loop_path() -> None:
    """LM Studio weather multi-agent runs should use the same LLM path unless a generic deterministic fast path already applies."""
    with patch.object(WeatherAgent, "__init__", lambda self: None):
        agent = WeatherAgent()
        agent.system_prompt = "PRIMARY WEATHER PROMPT"
        agent.llm_provider = "lmstudio"
        agent.execute_react_loop = MagicMock(return_value="🌤️ Weather Forecast for Lisbon")
        agent._run_direct_tool_fallback = MagicMock(side_effect=AssertionError("Local-only fallback should be skipped"))

        output = agent.invoke(
            "Planeia a minha tarde em Belém, diz-me como lá chegar a partir do Rossio e considera o tempo.",
            context="User language: pt",
            verbose=False,
        )

        agent.execute_react_loop.assert_called_once()
        agent._run_direct_tool_fallback.assert_not_called()
        assert output.strip()


def test_weather_forecast_query_uses_direct_tool_path_with_requested_days() -> None:
    """Simple forecast queries should bypass the LLM and use the requested day window."""
    with patch.object(WeatherAgent, "__init__", lambda self: None):
        agent = WeatherAgent()
        agent.system_prompt = "PRIMARY WEATHER PROMPT"

        warnings_tool = MagicMock()
        warnings_tool.name = "get_weather_warnings"
        warnings_tool.invoke = MagicMock(return_value="⚠️ Active Weather Warnings (LSB):\n🟡 Rough sea")

        forecast_tool = MagicMock()
        forecast_tool.name = "get_weather_forecast"
        forecast_tool.invoke = MagicMock(
            return_value=(
                "🌤️ Weather Forecast for Lisbon\n"
                "📅 Updated: 2026-03-09T15:22:00\n"
                "📅 Today (2026-03-09):\n"
                "   🌡️ Temperature: 8.1°C to 13.1°C"
            )
        )

        agent.tools = [warnings_tool, forecast_tool]
        agent.execute_react_loop = MagicMock(side_effect=AssertionError("LLM path should be skipped"))

        output = agent.invoke("Qual é a previsão do tempo para os próximos 3 dias?")

        warnings_tool.invoke.assert_called_once_with({"area": "LSB"})
        forecast_tool.invoke.assert_called_once_with({"days": 3})
        assert "**Atualizado:** 15:22" in output


def test_weather_beyond_horizon_query_returns_limit_message_without_tool_calls_in_en() -> None:
    """Queries that clearly exceed IPMA's forecast horizon should return a limit message instead of a fake 7-day forecast."""
    with patch.object(WeatherAgent, "__init__", lambda self: None):
        agent = WeatherAgent()
        agent.system_prompt = "PRIMARY WEATHER PROMPT"

        warnings_tool = MagicMock()
        warnings_tool.name = "get_weather_warnings"
        warnings_tool.invoke = MagicMock(side_effect=AssertionError("Warnings tool should be skipped"))

        forecast_tool = MagicMock()
        forecast_tool.name = "get_weather_forecast"
        forecast_tool.invoke = MagicMock(side_effect=AssertionError("Forecast tool should be skipped"))

        current_tool = MagicMock()
        current_tool.name = "get_current_weather_summary"
        current_tool.invoke = MagicMock(side_effect=AssertionError("Current summary tool should be skipped"))

        agent.tools = [warnings_tool, forecast_tool, current_tool]
        agent.execute_react_loop = MagicMock(side_effect=AssertionError("LLM path should be skipped"))

        output = agent.invoke("What's the weather in Lisbon in a week?")

        warnings_tool.invoke.assert_not_called()
        forecast_tool.invoke.assert_not_called()
        current_tool.invoke.assert_not_called()
        assert "next 5 days" in output.lower()
        assert "can't confirm" in output.lower() or "cannot confirm" in output.lower()
        assert "IPMA" in output


def test_weather_beyond_horizon_query_returns_limit_message_without_tool_calls_in_pt() -> None:
    """Portuguese follow-ups beyond the 5-day window should say so clearly instead of stretching the forecast."""
    with patch.object(WeatherAgent, "__init__", lambda self: None):
        agent = WeatherAgent()
        agent.system_prompt = "PRIMARY WEATHER PROMPT"

        forecast_tool = MagicMock()
        forecast_tool.name = "get_weather_forecast"
        forecast_tool.invoke = MagicMock(side_effect=AssertionError("Forecast tool should be skipped"))

        agent.tools = [forecast_tool]
        agent.execute_react_loop = MagicMock(side_effect=AssertionError("LLM path should be skipped"))

        output = agent.invoke("E daqui a uma semana?")

        forecast_tool.invoke.assert_not_called()
        assert "5 dias" in output.lower()
        assert "não consigo confirmar" in output.lower() or "nao consigo confirmar" in output.lower()
        assert "IPMA" in output


def test_planner_retries_when_draft_mentions_unsupported_venue() -> None:
    """Planner should self-correct if the draft introduces venues not present in the provided data."""
    with patch.object(PlannerAgent, "__init__", lambda self: None):
        agent = PlannerAgent()
        agent.system_prompt = "PLANNER PROMPT"
        agent.llm = object()

        llm_calls = []
        draft_response = MagicMock(
            content=(
                "📅 **Itinerary**\n\n"
                "🕐 **09:30** - **Calouste Gulbenkian Museum**\n"
                "📍 Lisbon"
            )
        )
        corrected_response = MagicMock(
            content=(
                "📅 **Itinerary**\n\n"
                "🕐 **09:30** - **National Museum of Natural History and Science**\n"
                "📍 Lisbon"
            )
        )

        def fake_safe_llm_invoke(llm, messages):
            llm_calls.append(messages)
            return draft_response if len(llm_calls) == 1 else corrected_response

        agent._safe_llm_invoke = fake_safe_llm_invoke

        output = agent.invoke(
            user_message="Plan my morning around museums in Lisbon.",
            weather_data="Tomorrow: light showers.",
            places_data=(
                "1. 🏛️ **National Museum of Natural History and Science**\n"
                "2. 🏛️ **Museum of the Lisbon Geographical Society**"
            ),
        )

        assert len(llm_calls) == 2
        retry_prompt_text = "\n".join(str(message.content) for message in llm_calls[1])
        assert "Unsupported venue mentioned" in retry_prompt_text
        assert "Calouste Gulbenkian Museum" not in output
        assert "National Museum of Natural History and Science" in output


def test_planner_retries_when_accessibility_is_not_confirmed() -> None:
    """Planner should remove unsupported accessibility claims when none were provided in the data."""
    with patch.object(PlannerAgent, "__init__", lambda self: None):
        agent = PlannerAgent()
        agent.system_prompt = "PLANNER PROMPT"
        agent.llm = object()

        llm_calls = []
        draft_response = MagicMock(
            content=(
                "📅 **Itinerary**\n\n"
                "🕐 **09:30** - **National Museum of Natural History and Science**\n"
                "💡 Fully wheelchair-accessible with elevators and adapted toilets."
            )
        )
        corrected_response = MagicMock(
            content=(
                "📅 **Itinerary**\n\n"
                "🕐 **09:30** - **National Museum of Natural History and Science**\n"
                "💡 Accessibility details are not confirmed in the provided data, so check the official venue page before going."
            )
        )

        def fake_safe_llm_invoke(llm, messages):
            llm_calls.append(messages)
            return draft_response if len(llm_calls) == 1 else corrected_response

        agent._safe_llm_invoke = fake_safe_llm_invoke

        output = agent.invoke(
            user_message="I use a wheelchair. Plan my museum morning in Lisbon.",
            weather_data="Tomorrow: light showers.",
            places_data="1. 🏛️ **National Museum of Natural History and Science**",
        )

        assert len(llm_calls) >= 2
        assert "wheelchair-accessible" not in output.lower()
        assert "not confirmed" in output.lower()


def test_researcher_history_response_keeps_non_visitlisboa_source_space() -> None:
    """History/web answers should not receive a fabricated VisitLisboa source line."""
    raw = (
        "📚 **Wikipédia: Castelo de São Jorge**\n"
        "🔗 URL: https://pt.wikipedia.org/wiki/Castelo_de_S%C3%A3o_Jorge\n\n"
        "If you’d like, I can provide more details."
    )

    output = finalize_worker_response(
        raw,
        agent_name="researcher",
        user_query="Tell me about the history of Castelo de São Jorge",
    )

    assert "VisitLisboa" not in output
    assert "If you’d like" not in output


def test_transport_direct_status_fallback_prefers_metro_tool() -> None:
    """Metro-specific status queries should use the metro status tool before broad summary."""
    with patch.object(TransportAgent, "__init__", lambda self: None):
        agent = TransportAgent()

        metro_tool = MagicMock()
        metro_tool.name = "get_metro_status"
        metro_tool.invoke = MagicMock(return_value="metro ok")

        summary_tool = MagicMock()
        summary_tool.name = "get_transport_summary"
        summary_tool.invoke = MagicMock(return_value="summary")

        agent.tools = [summary_tool, metro_tool]

        output = agent._run_direct_tool_fallback("Is the metro working?")

        assert output == "metro ok"
        metro_tool.invoke.assert_called_once_with({})
        summary_tool.invoke.assert_not_called()


def test_base_agent_safe_llm_invoke_collapses_multiple_system_messages_for_lmstudio() -> None:
    """LM Studio invocations should merge multiple system messages into one compatible payload."""
    agent = BaseAgent.__new__(BaseAgent)
    agent.llm_provider = "lmstudio"
    agent._record_llm_usage = lambda _llm, _response: None

    captured = {}

    class FakeLLM:
        def invoke(self, messages):
            captured["messages"] = messages
            return SimpleNamespace(content="ok")

    response = agent._safe_llm_invoke(
        FakeLLM(),
        [
            SystemMessage(content="System A"),
            SystemMessage(content="System B"),
            HumanMessage(content="Olá"),
        ],
        retries=0,
    )

    assert response.content == "ok"
    system_messages = [message for message in captured["messages"] if isinstance(message, SystemMessage)]
    assert len(system_messages) == 1
    assert "System A" in system_messages[0].content
    assert "System B" in system_messages[0].content


def test_base_agent_safe_llm_invoke_preserves_multiple_system_messages_for_non_lmstudio() -> None:
    """Azure/OpenAI paths should keep the original multi-system payload unchanged."""
    agent = BaseAgent.__new__(BaseAgent)
    agent.llm_provider = "azure"
    agent._record_llm_usage = lambda _llm, _response: None

    captured = {}

    class FakeLLM:
        def invoke(self, messages):
            captured["messages"] = messages
            return SimpleNamespace(content="ok")

    response = agent._safe_llm_invoke(
        FakeLLM(),
        [
            SystemMessage(content="System A"),
            SystemMessage(content="System B"),
            HumanMessage(content="Hello"),
        ],
        retries=0,
    )

    assert response.content == "ok"
    system_messages = [message for message in captured["messages"] if isinstance(message, SystemMessage)]
    assert len(system_messages) == 2


def test_clean_response_removes_dangling_think_block() -> None:
    """Dangling LM Studio think blocks should be stripped even when the tag is never closed."""
    raw = "### 🚇 Informação de Transportes\n\n<think>Now I have all the information."

    output = clean_response(raw)

    assert "<think>" not in output
    assert output.strip() == "### 🚇 Informação de Transportes"


def test_base_agent_execute_react_loop_keeps_parallel_tool_execution_for_lmstudio() -> None:
    """LM Studio tool batches should still run in parallel because tools do not spawn extra LLMs."""
    agent = BaseAgent.__new__(BaseAgent)
    agent.llm_provider = "lmstudio"
    agent.llm_with_tools = object()
    agent._record_llm_usage = lambda _llm, _response: None
    agent._record_tool_call = lambda _tool_name, _args: None

    tool_a = MagicMock()
    tool_a.name = "tool_a"

    tool_b = MagicMock()
    tool_b.name = "tool_b"

    agent.tools = [tool_a, tool_b]
    agent.execute_tools_parallel = MagicMock(
        return_value={"call_a": "result a", "call_b": "result b"}
    )

    responses = iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "tool_a", "args": {"origin": "Rossio"}, "id": "call_a"},
                    {"name": "tool_b", "args": {"destination": "Belém"}, "id": "call_b"},
                ],
            ),
            SimpleNamespace(content="done", tool_calls=[]),
        ]
    )
    agent._safe_llm_invoke = MagicMock(side_effect=lambda _llm, _messages, verbose=False: next(responses))

    output = agent.execute_react_loop(messages=[HumanMessage(content="Route test")], verbose=False)

    agent.execute_tools_parallel.assert_called_once()
    tool_calls = agent.execute_tools_parallel.call_args.args[0]
    assert [tool_call["name"] for tool_call in tool_calls] == ["tool_a", "tool_b"]
    assert output == "done"


def test_base_agent_execute_react_loop_falls_back_to_tool_results_for_incomplete_think_reply() -> None:
    """If the final LM Studio reply is just a header plus dangling think text, return the real tool result instead."""
    agent = BaseAgent.__new__(BaseAgent)
    agent.llm_provider = "lmstudio"
    agent.llm_with_tools = object()
    agent._record_llm_usage = lambda _llm, _response: None
    agent._record_tool_call = lambda _tool_name, _args: None

    route_tool = MagicMock()
    route_tool.name = "route_tool"
    route_tool.invoke = MagicMock(return_value="🗺️ **Route: Rossio → Belém**")
    agent.tools = [route_tool]

    responses = iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "route_tool", "args": {"origin": "Rossio", "destination": "Belém"}, "id": "call_route"},
                ],
            ),
            SimpleNamespace(content="### 🚇 Informação de Transportes\n\n<think>Now I have all the information.", tool_calls=[]),
        ]
    )
    agent._safe_llm_invoke = MagicMock(side_effect=lambda _llm, _messages, verbose=False: next(responses))

    output = agent.execute_react_loop(messages=[HumanMessage(content="Route test")], verbose=False)

    assert "<think>" not in output
    assert "Rossio" in output


def test_qa_lmstudio_validate_uses_same_llm_validation_path() -> None:
    """LM Studio QA should use the same prompt-driven validation path as cloud providers."""
    with patch.object(QualityAssuranceAgent, "__init__", lambda self: None):
        agent = QualityAssuranceAgent()
        agent.llm_provider = "lmstudio"
        agent.llm = object()
        agent._safe_llm_invoke = MagicMock(
            return_value=MagicMock(
                content='{"complete": true, "missing_data": [], "required_agents": [], "reasoning": "All data present.", "disclaimers": []}'
            )
        )

        fact_check = {
            "valid": True,
            "disclaimers": [],
            "critical_issues": [],
            "checks_performed": ["output_hygiene"],
            "repairable_agents": [],
            "per_agent": {},
        }
        agent._verify_facts = MagicMock(return_value=fact_check)
        agent._merge_fact_check_results = MagicMock(return_value=fact_check)

        result = agent.validate(
            user_query="Planeia a minha tarde em Belém.",
            agent_outputs={"weather": "ok", "transport": "ok", "researcher": "ok"},
            agents_called=["weather", "transport", "researcher"],
            language="pt",
        )

        agent._safe_llm_invoke.assert_called_once()
        assert result["complete"] is True
        assert result["required_agents"] == []
        assert result["needs_repair"] is False


def test_qa_lmstudio_repair_final_response_uses_same_llm_repair_path() -> None:
    """LM Studio final QA repair should use the same repair path as cloud models."""
    with patch.object(QualityAssuranceAgent, "__init__", lambda self: None):
        agent = QualityAssuranceAgent()
        agent.llm_provider = "lmstudio"
        agent.llm = object()
        agent._safe_llm_invoke = MagicMock(return_value=MagicMock(content="### 📅 Repaired\n- Conteúdo"))

        draft = "### 📅 Rascunho\n- Conteúdo"
        result = agent.repair_final_response(
            user_query="Planeia a minha tarde em Belém.",
            draft_response=draft,
            agent_outputs={"weather": "ok", "transport": "ok", "researcher": "ok"},
            qa_result={"disclaimers": ["Verificar horários"], "fact_check": {"disclaimers": []}},
            language="pt",
        )

        agent._safe_llm_invoke.assert_called_once()
        assert result.startswith("### 📅 Repaired")


def test_qa_repair_final_response_keeps_draft_when_repair_collapses_to_generic_error() -> None:
    """QA repair must not replace a usable draft with the generic clean_response error placeholder."""
    with patch.object(QualityAssuranceAgent, "__init__", lambda self: None):
        agent = QualityAssuranceAgent()
        agent.llm_provider = "lmstudio"
        agent.llm = object()
        agent._safe_llm_invoke = MagicMock(
            return_value=MagicMock(
                content="How do I get there?\n\nWe are in Portuguese.\n\nStep-by-step:"
            )
        )

        draft = "### 📅 Draft\n- Conteúdo grounded"
        result = agent.repair_final_response(
            user_query="Planeia a minha tarde em Belém.",
            draft_response=draft,
            agent_outputs={"weather": "ok"},
            qa_result={"disclaimers": ["Verificar horários"], "fact_check": {"disclaimers": []}},
            language="pt",
        )

        assert result == draft


def test_transport_clean_query_fragment_strips_using_the_metro_suffix() -> None:
    """Route endpoint cleanup should strip English mode suffixes from location fragments."""
    assert _clean_query_fragment("Rossio using the metro") == "Rossio"


def test_transport_extract_route_endpoints_from_planner_style_phrase() -> None:
    """Planner-style PT route phrasing should still yield clean origin/destination endpoints."""
    assert _extract_route_endpoints(
        "Planeia a minha tarde em Belém, diz-me como lá chegar a partir do Rossio e considera o tempo."
    ) == ("Rossio", "Belém")


def test_transport_extract_route_endpoints_from_do_para_phrase() -> None:
    """Simple PT phrasing with 'do X para Y' should resolve route endpoints deterministically."""
    assert _extract_route_endpoints("Como vou do Rossio para Belém?") == ("Rossio", "Belém")


def test_transport_lmstudio_multiagent_context_uses_same_react_loop_path() -> None:
    """LM Studio transport multi-agent runs should use the same LLM path as cloud providers."""
    with patch.object(TransportAgent, "__init__", lambda self: None):
        agent = TransportAgent()
        agent.system_prompt = "TRANSPORT PROMPT"
        agent.llm_provider = "lmstudio"
        agent.execute_react_loop = MagicMock(return_value="🗺️ **Route: Rossio → Belém**")
        agent.tools = []
        agent._resolve_deterministic_response = MagicMock(return_value=None)
        agent._invoke_deterministic_tool_call = MagicMock(return_value=None)

        with patch(
            "agent.agents.transport_agent.finalize_worker_response",
            side_effect=lambda response, **_kwargs: response,
        ):
            output = agent.invoke(
                "Planeia a minha tarde em Belém, diz-me como lá chegar a partir do Rossio e considera o tempo.",
                context="User language: pt",
                verbose=False,
            )

        agent.execute_react_loop.assert_called_once()
        assert "Rossio" in output


def test_transport_stop_name_arrivals_query_uses_deterministic_carris_tool() -> None:
    """Carris stop-name arrival queries should bypass the free-form LLM path."""
    with patch.object(TransportAgent, "__init__", lambda self: None):
        agent = TransportAgent()
        agent.system_prompt = "TRANSPORT PROMPT"
        agent.execute_react_loop = MagicMock(side_effect=AssertionError("LLM path should be skipped"))
        agent.tools = []
        arrivals_tool = MagicMock()
        arrivals_tool.invoke = MagicMock(
            return_value=(
                "Próximas Chegadas: Rossio\n"
                "ID: 908 | Atualizado: 10:44\n"
                "[REAL-TIME] Autocarro 711 -> Alto Damaia\n"
                "Hora: 10:58 (6 min late)\n"
                "Vehicle: 2685 | Plate: 93-XA-46"
            )
        )

        with patch(
            "agent.agents.transport_agent._resolve_carris_stop",
            return_value=("908", "Rossio"),
        ), patch(
            "tools.carris_api.carris_get_arrivals",
            new=arrivals_tool,
        ):
            output = agent.invoke("Quais os próximos autocarros da Carris no Rossio?", context="", verbose=False)

        arrivals_tool.invoke.assert_called_once_with({"stop_id": "908", "limit": 8})
        assert "908" not in output
        assert "2685" not in output
        assert "93-XA-46" not in output
        assert "Rossio" in output


def test_transport_agent_returns_honest_limitation_for_ferry_queries() -> None:
    """Ferry queries should avoid the LLM path and explain the unsupported runtime scope clearly."""
    with patch.object(TransportAgent, "__init__", lambda self: None):
        agent = TransportAgent()
        agent.system_prompt = "TRANSPORT PROMPT"
        agent.execute_react_loop = MagicMock(side_effect=AssertionError("LLM path should be skipped"))
        agent.tools = []

        output = agent.invoke("Ferry to Cacilhas right now?", context="", verbose=False)

    assert "runtime" in output.lower()
    assert "ferry" in output.lower() or "transtejo" in output.lower()
    assert "Metro de Lisboa" in output
    assert "Carris Metropolitana" in output
    assert "CP" in output


def test_transport_agent_returns_honest_limitation_for_fertagus_queries() -> None:
    """Fertagus-specific queries should be answered with an explicit limitation note, not invented schedules."""
    with patch.object(TransportAgent, "__init__", lambda self: None):
        agent = TransportAgent()
        agent.system_prompt = "TRANSPORT PROMPT"
        agent.execute_react_loop = MagicMock(side_effect=AssertionError("LLM path should be skipped"))
        agent.tools = []

        output = agent.invoke("What is the next Fertagus train to Setúbal?", context="", verbose=False)

    assert "Fertagus" in output
    assert "can't directly verify" in output.lower() or "not yet confirmed" in output.lower()


def test_multiagent_skips_qa_for_simple_weather_queries() -> None:
    """Simple deterministic weather queries should not pay the extra QA latency."""
    assistant = MultiAgentAssistant.__new__(MultiAgentAssistant)
    assistant.state = {"messages": [], "user_context": None}

    assistant.supervisor = MagicMock()
    assistant.supervisor.route = MagicMock(
        return_value={"agents": ["weather"], "direct_response": None, "reasoning": "weather only"}
    )

    weather_agent = MagicMock()
    weather_agent._is_current_weather_query = MagicMock(return_value=False)
    weather_agent._is_simple_forecast_query = MagicMock(return_value=True)
    weather_agent.invoke = MagicMock(return_value="🌤️ Forecast body")

    assistant.qa_agent = MagicMock()
    assistant.agents = {"weather": weather_agent}

    with patch("agent.graph.LANGSMITH_AVAILABLE", False), patch(
        "agent.graph.clean_response", side_effect=lambda text: text
    ), patch("agent.graph.format_response", side_effect=lambda text: text), patch(
        "agent.graph.generate_response_title", return_value=None
    ), patch("agent.graph.ensure_response_title", side_effect=lambda text, title: text):
        output = assistant.chat(
            "Qual é a previsão do tempo para os próximos 3 dias?",
            language="pt",
            verbose=False,
        )

    weather_agent.invoke.assert_called_once()
    assistant.qa_agent.validate.assert_not_called()
    assert output == "🌤️ Forecast body"


def test_multiagent_local_worker_batches_run_sequentially_without_threadpool() -> None:
    """Local LM Studio worker batches should not use the parallel executor."""
    assistant = MultiAgentAssistant.__new__(MultiAgentAssistant)
    assistant.state = {"messages": [], "user_context": None}

    assistant.supervisor = MagicMock()
    assistant.supervisor.route = MagicMock(
        return_value={"agents": ["weather", "transport"], "direct_response": None, "reasoning": "local worker batch"}
    )

    call_order: list[str] = []

    weather_agent = MagicMock()
    weather_agent.llm_provider = "lmstudio"
    weather_agent._is_current_weather_query = MagicMock(return_value=False)
    weather_agent._is_simple_forecast_query = MagicMock(return_value=False)
    weather_agent.invoke = MagicMock(side_effect=lambda *_args, **_kwargs: call_order.append("weather") or "🌤️ Weather ok")

    transport_agent = MagicMock()
    transport_agent.llm_provider = "lmstudio"
    transport_agent.invoke = MagicMock(side_effect=lambda *_args, **_kwargs: call_order.append("transport") or "🚇 Transport ok")

    assistant.agents = {"weather": weather_agent, "transport": transport_agent}
    assistant.qa_agent = MagicMock()
    assistant.qa_agent.validate = MagicMock(
        return_value={
            "complete": True,
            "missing_data": [],
            "required_agents": [],
            "reasoning": "All data present.",
            "disclaimers": [],
            "critical_issues": [],
            "repairable_agents": [],
            "needs_repair": False,
            "fact_check": {
                "disclaimers": [],
                "critical_issues": [],
                "repairable_agents": [],
                "per_agent": {},
            },
        }
    )
    assistant._combine_outputs = MagicMock(return_value="combined")

    with patch("agent.graph.LANGSMITH_AVAILABLE", False), patch(
        "agent.graph.clean_response", side_effect=lambda text: text
    ), patch("agent.graph.format_response", side_effect=lambda text: text), patch(
        "agent.graph.generate_response_title", return_value=None
    ), patch("agent.graph.ensure_response_title", side_effect=lambda text, title: text), patch(
        "agent.graph.ContextThreadPoolExecutor",
        side_effect=AssertionError("Parallel worker executor should be skipped"),
    ):
        output = assistant.chat(
            "Planeia a minha tarde em Belém, diz-me como lá chegar a partir do Rossio e considera o tempo.",
            language="pt",
            verbose=False,
        )

    assert call_order == ["weather", "transport"]
    assistant._combine_outputs.assert_called_once_with(
        {"weather": "🌤️ Weather ok", "transport": "🚇 Transport ok"},
        language="pt",
    )
    assert output == "combined"


def test_multiagent_retries_worker_when_qa_flags_repairable_critical_issue() -> None:
    """QA critical issues should trigger a focused retry of the offending worker even when completeness is otherwise fine."""
    assistant = MultiAgentAssistant.__new__(MultiAgentAssistant)
    assistant.state = {"messages": [], "user_context": None}

    assistant.supervisor = MagicMock()
    assistant.supervisor.route = MagicMock(
        return_value={"agents": ["transport"], "direct_response": None, "reasoning": "transport only"}
    )

    transport_agent = MagicMock()
    transport_agent.invoke = MagicMock(
        side_effect=[
            "🚌 Live buses\n- 📍 GPS: 38.72410, -9.14820\n- 🚏 Next stop ID: 060001",
            "🚇 Clean transport answer",
        ]
    )

    assistant.agents = {"transport": transport_agent}
    assistant.qa_agent = MagicMock()
    assistant.qa_agent.validate = MagicMock(
        side_effect=[
            {
                "complete": True,
                "missing_data": [],
                "required_agents": [],
                "reasoning": "Transport output leaked technical transport metadata.",
                "disclaimers": [],
                "critical_issues": ["Raw GPS coordinates leaked into user-facing output."],
                "repairable_agents": ["transport"],
                "needs_repair": True,
                "fact_check": {
                    "disclaimers": [],
                    "critical_issues": ["Raw GPS coordinates leaked into user-facing output."],
                    "repairable_agents": ["transport"],
                    "per_agent": {
                        "transport": {
                            "valid": False,
                            "disclaimers": [],
                            "critical_issues": ["Raw GPS coordinates leaked into user-facing output."],
                            "checks_performed": ["output_hygiene"],
                        }
                    },
                },
            },
            {
                "complete": True,
                "missing_data": [],
                "required_agents": [],
                "reasoning": "Clean after retry.",
                "disclaimers": [],
                "critical_issues": [],
                "repairable_agents": [],
                "needs_repair": False,
                "fact_check": {
                    "disclaimers": [],
                    "critical_issues": [],
                    "repairable_agents": [],
                    "per_agent": {
                        "transport": {
                            "valid": True,
                            "disclaimers": [],
                            "critical_issues": [],
                            "checks_performed": ["output_hygiene"],
                        }
                    },
                },
            },
        ]
    )
    assistant.qa_agent.repair_final_response = MagicMock(side_effect=AssertionError("Final QA repair should not run after a clean worker retry"))

    with patch("agent.graph.LANGSMITH_AVAILABLE", False), patch(
        "agent.graph.clean_response", side_effect=lambda text: text
    ), patch("agent.graph.format_response", side_effect=lambda text: text), patch(
        "agent.graph.generate_response_title", return_value=None
    ), patch("agent.graph.ensure_response_title", side_effect=lambda text, title: text):
        output = assistant.chat(
            "Mostra os autocarros em tempo real.",
            language="pt",
            verbose=False,
        )

    assert output == "🚇 Clean transport answer"
    assert transport_agent.invoke.call_count == 2
    retry_context = transport_agent.invoke.call_args_list[1].args[1]
    assert "Raw GPS coordinates leaked into user-facing output." in retry_context
    assert "do not mention qa" in retry_context.lower()


def test_multiagent_runs_final_qa_repair_for_planner_responses() -> None:
    """Planner responses should receive a final QA repair pass before the answer reaches the user."""
    assistant = MultiAgentAssistant.__new__(MultiAgentAssistant)
    assistant.state = {"messages": [], "user_context": None}

    assistant.supervisor = MagicMock()
    assistant.supervisor.route = MagicMock(
        return_value={"agents": ["researcher", "planner"], "direct_response": None, "reasoning": "research + planner"}
    )

    researcher_agent = MagicMock()
    researcher_agent.invoke = MagicMock(return_value="📍 Researcher notes")

    planner_agent = MagicMock()
    planner_agent.synthesize = MagicMock(return_value="🗓️ Draft itinerary")

    assistant.agents = {"researcher": researcher_agent, "planner": planner_agent}
    assistant.qa_agent = MagicMock()
    assistant.qa_agent.validate = MagicMock(
        return_value={
            "complete": True,
            "missing_data": [],
            "required_agents": [],
            "reasoning": "All data present.",
            "disclaimers": ["Opening hours should still be confirmed."],
            "critical_issues": [],
            "repairable_agents": [],
            "needs_repair": False,
            "fact_check": {
                "disclaimers": ["Opening hours should still be confirmed."],
                "critical_issues": [],
                "repairable_agents": [],
                "per_agent": {},
            },
        }
    )
    assistant.qa_agent.repair_final_response = MagicMock(return_value="🗓️ Repaired itinerary")

    with patch("agent.graph.LANGSMITH_AVAILABLE", False), patch(
        "agent.graph.clean_response", side_effect=lambda text: text
    ), patch("agent.graph.format_response", side_effect=lambda text: text), patch(
        "agent.graph.generate_response_title", return_value=None
    ), patch("agent.graph.ensure_response_title", side_effect=lambda text, title: text):
        output = assistant.chat(
            "Plan my day around Belém.",
            language="en",
            verbose=False,
        )

    planner_agent.synthesize.assert_called_once()
    assistant.qa_agent.repair_final_response.assert_called_once()
    assert output.startswith("### 📅")
    assert output.endswith("Repaired itinerary")


def test_multiagent_structured_response_filters_internal_qa_warnings_and_localizes_public_notes() -> None:
    """Hybrid responses should hide internal QA chatter while keeping localized user-facing caveats."""
    assistant = MultiAgentAssistant.__new__(MultiAgentAssistant)

    output = assistant._render_structured_hybrid_response(
        {
            "weather": "Tempo estável.",
            "transport": "Rota confirmada.",
            "_qa_disclaimers": [
                "O Agente de Transporte mencionou não ter acesso a dados meteorológicos, o que contradiz o Agente de Meteorologia; esta informação deve ser ignorada na resposta final.",
                "Some URLs reference unverified domains: in-weather.com. Please verify links before visiting.",
                "Carris bus route numbers and schedules should be verified at carris.pt, as GTFS data may not reflect the most recent changes.",
                "Dados de transporte em tempo real podem sofrer alterações.",
                "Some metro station names could not be verified: metro",
            ],
        },
        language="pt",
    )

    assert "Agente de Transporte" not in output
    assert "deve ser ignorada na resposta final" not in output
    assert "domínios não verificados (in-weather" in output
    assert "os horários da Carris devem ser confirmados em carris.pt" in output
    assert "Dados de transporte em tempo real podem sofrer alterações." in output
    assert "could not be verified" not in output


def test_search_places_attractions_respects_category_and_excludes_service_like_results() -> None:
    """Museum-focused searches should not return hotels or tourist offices ahead of museums."""

    class DummyKB:
        def search_with_scores(self, query, k, collections):
            return [
                (
                    Document(
                        page_content="Name: Hotel Jerónimos 8\nCategory: Hotel\nShort Description: Stay near Belém.",
                        metadata={
                            "title": "Hotel Jerónimos 8",
                            "category": "Hotel",
                            "url": "https://www.visitlisboa.com/en/places/hotel-jeronimos-8",
                        },
                    ),
                    0.32,
                ),
                (
                    Document(
                        page_content="Name: Ask Me Lisboa | Belém - Jerónimos Monastery\nCategory: Tourist Offices\nShort Description: Tourist office.",
                        metadata={
                            "title": "Ask Me Lisboa | Belém - Jerónimos Monastery",
                            "category": "Tourist Offices",
                            "url": "https://www.visitlisboa.com/en/places/ask-me-lisboa-belem-jeronimos-monastery",
                        },
                    ),
                    0.35,
                ),
                (
                    Document(
                        page_content="Name: National Coach Museum\nCategory: Museums & Monuments\nShort Description: Museum in Belém.",
                        metadata={
                            "title": "National Coach Museum",
                            "category": "Museums & Monuments",
                            "url": "https://www.visitlisboa.com/en/places/national-coach-museum",
                        },
                    ),
                    0.40,
                ),
            ]

    with patch.object(visitlisboa_api, "_get_vector_store", return_value=DummyKB()), patch.object(
        visitlisboa_api,
        "_should_search_dados_abertos",
        return_value=False,
    ), patch.object(visitlisboa_api, "_get_place_by_url", return_value=None):
        output = visitlisboa_api.search_places_attractions.invoke(
            {"query": "best museums in lisbon", "category": "Museums & Monuments", "max_results": 5}
        )

    assert "National Coach Museum" in output
    assert "Hotel Jerónimos 8" not in output
    assert "Ask Me Lisboa | Belém - Jerónimos Monastery" not in output


def test_search_places_attractions_museum_query_filters_pure_monuments() -> None:
    """Museum-only queries should filter out pure monuments that do not mention museums."""

    class DummyKB:
        def search_with_scores(self, query, k, collections):
            return [
                (
                    Document(
                        page_content="Name: Monument to the Discoveries\nCategory: Monuments\nShort Description: Riverfront monument.",
                        metadata={
                            "title": "Monument to the Discoveries",
                            "category": "Monuments",
                            "url": "https://www.visitlisboa.com/en/places/monument-to-the-discoveries",
                        },
                    ),
                    0.35,
                ),
                (
                    Document(
                        page_content="Name: National Museum of Sport\nCategory: Museums & Monuments\nShort Description: Sports museum.",
                        metadata={
                            "title": "National Museum of Sport",
                            "category": "Museums & Monuments",
                            "url": "https://www.visitlisboa.com/en/places/national-museum-of-sport",
                        },
                    ),
                    0.40,
                ),
            ]

    with patch.object(visitlisboa_api, "_get_vector_store", return_value=DummyKB()), patch.object(
        visitlisboa_api,
        "_should_search_dados_abertos",
        return_value=False,
    ), patch.object(visitlisboa_api, "_get_place_by_url", return_value=None):
        output = visitlisboa_api.search_places_attractions.invoke(
            {"query": "best museums in lisbon", "category": "Museums & Monuments", "max_results": 5}
        )

    assert "National Museum of Sport" in output
    assert "Monument to the Discoveries" not in output


def test_search_places_attractions_museum_query_excludes_composite_non_museum_titles() -> None:
    """Museum-only queries should exclude monument-like titles even inside the composite category."""

    class DummyKB:
        def search_with_scores(self, query, k, collections):
            return [
                (
                    Document(
                        page_content="Name: Monument to the Discoveries\nCategory: Museums & Monuments\nShort Description: Riverfront monument.",
                        metadata={
                            "title": "Monument to the Discoveries",
                            "category": "Museums & Monuments",
                            "url": "https://www.visitlisboa.com/en/places/monument-to-the-discoveries",
                        },
                    ),
                    0.30,
                ),
                (
                    Document(
                        page_content="Name: National Music Museum\nCategory: Museums & Monuments\nShort Description: Music museum.",
                        metadata={
                            "title": "National Music Museum",
                            "category": "Museums & Monuments",
                            "url": "https://www.visitlisboa.com/en/places/national-music-museum",
                        },
                    ),
                    0.40,
                ),
            ]

    with patch.object(visitlisboa_api, "_get_vector_store", return_value=DummyKB()), patch.object(
        visitlisboa_api,
        "_should_search_dados_abertos",
        return_value=False,
    ), patch.object(visitlisboa_api, "_get_place_by_url", return_value=None):
        output = visitlisboa_api.search_places_attractions.invoke(
            {"query": "best museums in lisbon", "category": "Museums & Monuments", "max_results": 5}
        )

    assert "National Music Museum" in output
    assert "Monument to the Discoveries" not in output


def test_search_places_attractions_best_query_prefers_stronger_social_proof() -> None:
    """Broad 'best museums' queries should favor stronger rating/review evidence when relevance is close."""

    class DummyKB:
        def search_with_scores(self, query, k, collections):
            return [
                (
                    Document(
                        page_content="Name: Museum of the Lisbon Geographical Society\nCategory: Museums\nShort Description: Geography museum.",
                        metadata={
                            "title": "Museum of the Lisbon Geographical Society",
                            "category": "Museums",
                            "url": "https://www.visitlisboa.com/en/places/museum-of-the-lisbon-geographical-society",
                            "rating": 4.6,
                            "reviews": 18,
                        },
                    ),
                    0.30,
                ),
                (
                    Document(
                        page_content="Name: Money Museum\nCategory: Museums\nShort Description: Museum about money.",
                        metadata={
                            "title": "Money Museum",
                            "category": "Museums",
                            "url": "https://www.visitlisboa.com/en/places/money-museum",
                            "rating": 4.5,
                            "reviews": 197,
                        },
                    ),
                    0.45,
                ),
            ]

    with patch.object(visitlisboa_api, "_get_vector_store", return_value=DummyKB()), patch.object(
        visitlisboa_api,
        "_should_search_dados_abertos",
        return_value=False,
    ), patch.object(visitlisboa_api, "_get_place_by_url", return_value=None):
        output = visitlisboa_api.search_places_attractions.invoke(
            {"query": "best museums in lisbon", "category": "Museums & Monuments", "max_results": 5}
        )

    assert output.index("Money Museum") < output.index("Museum of the Lisbon Geographical Society")


def test_search_places_attractions_first_time_query_excludes_tour_companies() -> None:
    """First-time attraction lists should prioritize attractions, not guided-tour providers."""

    class DummyKB:
        def search_with_scores(self, query, k, collections):
            return [
                (
                    Document(
                        page_content="Name: Take Lisboa Free Tours\nCategory: Tours\nShort Description: Guided walks across Lisbon.",
                        metadata={
                            "title": "Take Lisboa Free Tours",
                            "category": "Tours",
                            "url": "https://www.visitlisboa.com/en/places/take-lisboa-free-tours",
                            "rating": 5.0,
                            "reviews": 45979,
                        },
                    ),
                    0.28,
                ),
                (
                    Document(
                        page_content="Name: Jerónimos Monastery\nCategory: Museums & Monuments\nShort Description: UNESCO monastery in Belém.",
                        metadata={
                            "title": "Jerónimos Monastery",
                            "category": "Museums & Monuments",
                            "url": "https://www.visitlisboa.com/en/places/jeronimos-monastery",
                            "rating": 4.8,
                            "reviews": 18000,
                        },
                    ),
                    0.34,
                ),
            ]

    with patch.object(visitlisboa_api, "_get_vector_store", return_value=DummyKB()), patch.object(
        visitlisboa_api,
        "_should_search_dados_abertos",
        return_value=False,
    ), patch.object(visitlisboa_api, "_get_place_by_url", return_value=None):
        output = visitlisboa_api.search_places_attractions.invoke(
            {"query": "Lista as atrações imperdíveis para quem visita Lisboa pela primeira vez.", "max_results": 5}
        )

    assert "Jerónimos Monastery" in output
    assert "Take Lisboa Free Tours" not in output


def test_search_places_attractions_location_hint_filters_unrelated_results() -> None:
    """Neighborhood-specific queries should prefer candidates that mention that location."""

    class DummyKB:
        def search_with_scores(self, query, k, collections):
            return [
                (
                    Document(
                        page_content="Name: Museum of Illusions\nCategory: Museums\nShort Description: Central Lisbon interactive museum.",
                        metadata={
                            "title": "Museum of Illusions",
                            "category": "Museums",
                            "url": "https://www.visitlisboa.com/en/places/museum-of-illusions",
                        },
                    ),
                    0.25,
                ),
                (
                    Document(
                        page_content="Name: National Coach Museum\nCategory: Museums & Monuments\nShort Description: Museum in Belém.",
                        metadata={
                            "title": "National Coach Museum",
                            "category": "Museums & Monuments",
                            "url": "https://www.visitlisboa.com/en/places/national-coach-museum-belem",
                        },
                    ),
                    0.40,
                ),
            ]

    with patch.object(visitlisboa_api, "_get_vector_store", return_value=DummyKB()), patch.object(
        visitlisboa_api,
        "_should_search_dados_abertos",
        return_value=False,
    ), patch.object(visitlisboa_api, "_get_place_by_url", return_value=None):
        output = visitlisboa_api.search_places_attractions.invoke(
            {"query": "Belem museums", "category": "Museums & Monuments", "max_results": 5}
        )

    assert "National Coach Museum" in output
    assert "Museum of Illusions" not in output


def test_qa_station_check_ignores_non_station_help_text() -> None:
    """QA should not hallucinate bogus station names from generic help text."""
    agent = QualityAssuranceAgent.__new__(QualityAssuranceAgent)

    result = agent._verify_facts(
        "Use station names like: Campo Grande, Aeroporto, Baixa-Chiado, Rossio. Nenhuma estação de metro próxima reconhecida pelo Metropolitano de Lisboa.",
        "transport help",
        None,
    )

    assert not any("could not be verified" in disclaimer.lower() for disclaimer in result["disclaimers"])


def test_qa_place_only_query_does_not_require_weather_or_transport_retries() -> None:
    """Standalone attraction queries should stay in the researcher domain during QA normalization."""
    normalized = QualityAssuranceAgent._normalize_place_query_validation(
        user_query="Lista as atrações imperdíveis para quem visita Lisboa pela primeira vez.",
        agents_called=["researcher"],
        llm_result={
            "complete": False,
            "missing_data": ["weather forecast missing", "transport links missing"],
            "required_agents": ["weather", "transport"],
            "reasoning": "LLM QA over-requested cross-domain context.",
            "disclaimers": [
                "Weather context would improve the answer.",
                "Transport options should be included.",
            ],
        },
    )

    assert normalized["complete"] is True
    assert normalized["required_agents"] == []
    assert normalized["missing_data"] == []
    assert normalized["disclaimers"] == []


# ===========================================================================
# Supervisor direct-routing guardrail tests
# ===========================================================================


def test_supervisor_route_greeting_bypasses_llm() -> None:
    """Greeting-only queries should return directly without invoking the LLM."""
    with patch.object(SupervisorAgent, "__init__", lambda self: None):
        agent = SupervisorAgent()
        agent._safe_llm_invoke = MagicMock(side_effect=AssertionError("LLM should not be called"))

        decision = agent.route("Bom dia!", language="pt")

        assert decision["agents"] == []
        assert decision["direct_response"]
        assert "Assistente Urbano de Lisboa" in decision["direct_response"]



def test_supervisor_route_trivia_bypasses_llm() -> None:
    """Obvious trivia/out-of-scope queries should return directly without the LLM."""
    with patch.object(SupervisorAgent, "__init__", lambda self: None):
        agent = SupervisorAgent()
        agent._safe_llm_invoke = MagicMock(side_effect=AssertionError("LLM should not be called"))

        decision = agent.route("What is the capital of Japan?", language="en")

        assert decision["agents"] == []
        assert decision["direct_response"]
        assert "Lisbon Metropolitan Area" in decision["direct_response"]
        assert "Japan" not in decision["agents"]


@pytest.mark.parametrize(
    ("query", "language"),
    [
        ("Como se diz obrigado em mandarim?", "pt"),
        ("2+2?", "en"),
    ],
)
def test_supervisor_fallback_handles_obvious_oos(query: str, language: str) -> None:
    """Fallback routing should also keep obvious out-of-scope queries direct."""
    with patch.object(SupervisorAgent, "__init__", lambda self: None):
        agent = SupervisorAgent()
        decision = agent._fallback_routing(query, "", language=language)

        assert decision["agents"] == []
        assert decision["direct_response"]
