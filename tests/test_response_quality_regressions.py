import os
import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.agents.qa_agent import QualityAssuranceAgent
from agent.agents.researcher_agent import ResearcherAgent
from agent.agents.transport_agent import TransportAgent
from agent.agents.weather_agent import WeatherAgent
from agent.graph import MultiAgentAssistant
from agent.utils.langsmith_tracing import get_langsmith_request_tracking_status
from agent.utils.response_formatter import (
    build_bilingual_note,
    canonicalize_local_information_terms,
    final_visual_pass,
    finalize_worker_response,
    infer_response_language,
    localize_local_information_values,
    reconcile_researcher_event_response,
    resolve_output_language,
    structure_weather_markdown,
)


def test_infer_response_language_prefers_english_query_even_with_pt_default() -> None:
    """The effective reply language should follow the user's English query, not the PT UI default."""
    assert infer_response_language(
        user_query="Tell me about Book Fair 2026",
        default="pt",
    ) == "en"


def test_infer_response_language_keeps_short_pt_transport_overview_query_in_pt() -> None:
    """Short PT transport-overview questions should not fall back to English just because they mention metro/bus/train nouns."""
    assert infer_response_language(
        user_query="Dá-me o ponto de situação do Metro, autocarros e comboios em Lisboa.",
        default="en",
    ) == "pt"


def test_lisboa_aberta_service_categories_follow_requested_language() -> None:
    """Public-service category browsing should not mix PT headings into English answers."""
    from tools.dados_abertos import list_service_categories

    english = list_service_categories.invoke({"language": "en"})
    portuguese = list_service_categories.invoke({"language": "pt"})

    assert "Available Public-Service Categories" in english
    assert "**Health**" in english
    assert "Categorias de Serviços" not in english
    assert "Categorias de Serviços Disponíveis" in portuguese
    assert "**Saúde**" in portuguese


def test_visitlisboa_categories_are_compact_language_aware_lists() -> None:
    """Event/place category tools should return category lists, not pseudo cards or placeholders."""
    from tools.visitlisboa_api import get_event_categories, get_place_categories

    events = get_event_categories.invoke({"language": "en"})
    places = get_place_categories.invoke({"language": "en"})

    assert events.startswith("### 🎭 **Event Categories in Lisbon**")
    assert "event" in events
    assert "**Theater" in events or "**Theatre" in events
    assert "1 events" not in events
    assert places.startswith("### 🏛️ **Available Place Categories**")
    assert "**Museums & Monuments:**" in places
    assert "Guided tours" in places
    assert "Uncategorized" not in places
    assert "DMCS" not in places
    assert "Verify on the official website" not in places
    assert "google.com/maps" not in places


def test_hybrid_renderer_keeps_weather_transport_and_local_sections_separate() -> None:
    """Hybrid output should not let one worker repeat another worker's domain."""
    assistant = MultiAgentAssistant.__new__(MultiAgentAssistant)

    output = assistant._combine_outputs(
        {
            "weather": (
                "Amanhã o tempo está agradável.\n"
                "- 🌡️ **Temperatura**: 11°C a 21°C\n"
                "- Para ir do Rossio a Belém de transportes públicos, usa autocarro.\n\n"
                "📌 **Fonte:** [*IPMA*](https://www.ipma.pt) | **Atualizado:** 10:00"
            ),
            "transport": (
                "**Trajeto:** Rossio → Belém\n"
                "- 🚌 **Linha 732** — para Caselas\n\n"
                "📌 **Fonte:** [*Carris*](https://www.carris.pt) | **Atualizado:** 10:01"
            ),
            "researcher": (
                "### 🏛️ Mosteiro dos Jerónimos\n"
                "- 📝 **Descrição:** Monumento em Belém.\n"
                "- O tempo amanhã estará agradável.\n\n"
                "📌 **Fonte:** [*VisitLisboa Locais*](https://www.visitlisboa.com/pt-pt/locais) | **Atualizado:** 10:02"
            ),
        },
        language="pt",
    )

    assert "Para ir do Rossio" not in output
    assert "O tempo amanhã" not in output
    assert "Linha 732" in output
    assert "Mosteiro dos Jerónimos" in output


def test_final_visual_pass_repairs_truncated_scope_intro() -> None:
    """Out-of-scope templates must not leak the old truncated 'Here's what' line."""
    output = final_visual_pass("Oops.\n\nHere's what\n\n- 🌤️ Weather")

    assert "Here's what I can help you with:" in output
    assert "\nHere's what\n" not in output


def test_final_visual_pass_normalizes_carris_realtime_feed_phrase() -> None:
    """PT transport answers should not show mixed-language cached snapshot phrases."""
    raw = "📡 **Tempo real:** 📡 Carris GTFS-RT: cached — em tempo real snapshot in use (0s old)."

    output = final_visual_pass(raw)

    assert output == "📡 **Tempo real:** Carris GTFS-RT com snapshot em cache (0s)."


def test_final_visual_pass_hides_coordinates_when_address_exists() -> None:
    """Service cards should not show raw coordinates when an address is already available."""
    raw = """- 💊 **Farmácia Azevedo**
    - 📍 **Morada:** [Praça D. Pedro IV, 31](https://maps.example)
    - 📏 **Distância:** 0.08 km
    - 🗺️ **Coordenadas:** [38.713326, -9.139880](https://maps.example)

- 🏛️ **Torre de Belém**
    - 📍 **Morada:** [Avenida de Brasília](https://maps.example)
    - 🗺️ **GPS**: [38.69159, -9.21593](https://maps.example)

- 💊 **Sem morada**
    - 🗺️ **Coordenadas:** [38.700000, -9.100000](https://maps.example)
"""

    output = final_visual_pass(raw)

    assert "Farmácia Azevedo" in output
    assert "Praça D. Pedro IV" in output
    assert "38.713326" not in output
    assert "38.69159" not in output
    assert "38.700000" in output


class _FakeResearcherTool:
    """Tool stub for deterministic ResearcherAgent tests."""

    def __init__(self, name: str, response: str = "ok") -> None:
        self.name = name
        self.response = response
        self.calls: list[dict] = []

    def invoke(self, args: dict) -> str:
        """Record the call and return the configured response."""
        self.calls.append(args)
        return self.response


def test_researcher_lisboa_card_lookup_uses_knowledge_not_events() -> None:
    """Lisboa Card attraction questions should never be answered as event alternatives."""
    knowledge = _FakeResearcherTool("search_lisbon_knowledge", "knowledge")
    events = _FakeResearcherTool("search_cultural_events", "events")
    agent = ResearcherAgent.__new__(ResearcherAgent)
    agent.tools = [knowledge, events]
    agent._tool_calls_log = []

    response = agent._run_lisboa_card_lookup("Is the Oceanário included in the Lisboa Card?", "en")

    assert knowledge.calls == [{"query": "Is the Oceanário included in the Lisboa Card?", "max_results": 5}]
    assert events.calls == []
    assert "Oceanário de Lisboa" in response
    assert "discount" in response.lower()
    assert "not free entry" in response.lower()


def test_weather_wind_today_query_uses_current_summary_path() -> None:
    """Wind-only current weather prompts should use grounded current IPMA data, not free-form synthesis."""
    assert WeatherAgent._is_current_weather_query("Como está o vento hoje?") is True
    assert WeatherAgent._is_current_weather_query("How is the wind right now?") is True


class _FakeWeatherTool:
    """Small tool stub for deterministic WeatherAgent routing tests."""

    def __init__(self, name: str, response: str) -> None:
        self.name = name
        self.response = response
        self.calls: list[dict] = []

    def invoke(self, args: dict) -> str:
        """Record the call and return the configured response."""
        self.calls.append(args)
        return self.response


def _build_weather_agent_with_tools(*tools: _FakeWeatherTool) -> WeatherAgent:
    """Build a WeatherAgent instance without initializing an LLM provider."""
    agent = WeatherAgent.__new__(WeatherAgent)
    agent.tools = list(tools)
    agent._tool_calls_log = []
    return agent


def test_weather_warning_query_uses_warning_tool_even_when_user_says_right_now() -> None:
    """Warning status prompts should not be swallowed by the current-summary route."""
    warnings = _FakeWeatherTool(
        "get_weather_warnings",
        "✅ No active weather warnings for Lisbon.\n\n🌤️ Weather conditions are normal.",
    )
    current = _FakeWeatherTool("get_current_weather_summary", "SHOULD NOT BE USED")
    agent = _build_weather_agent_with_tools(warnings, current)

    output = agent._run_direct_tool_fallback("Are there any weather warnings active right now?")

    assert warnings.calls == [{"area": "LSB"}]
    assert current.calls == []
    assert output.startswith("### 🌤️ **Weather Warnings**")
    assert "No, there are **no active weather warnings**" in output
    assert "Weather conditions are normal" not in output
    assert output.count("No active weather warnings") + output.count("no active weather warnings") == 1


def test_weather_warning_finalization_removes_semantic_clear_status_duplicates() -> None:
    """Final weather output should not repeat equivalent clear-warning statements."""
    raw = """### 🌤️ **Weather Warnings**

✅ No, there are **no active weather warnings** for Lisbon right now.

---

✅ No active weather warnings for Lisbon.

🌤️ Weather conditions are normal.
"""

    output = finalize_worker_response(
        raw,
        agent_name="weather",
        user_query="Are there any weather warnings active right now?",
        language="en",
    )

    assert output.count("weather warnings") == 1
    assert "Weather conditions are normal" not in output
    assert "---\n\n📌 **Source:**" not in output
    assert output.endswith("**Updated:** " + output.rsplit("**Updated:** ", 1)[1])


def test_weather_warning_finalization_removes_pt_clear_status_variants() -> None:
    """PT warning answers must treat 'não há' and 'sem avisos' as the same status."""
    raw = """### 🌤️ **Avisos Meteorológicos**

✅ Não, não há **avisos meteorológicos ativos** para Lisboa neste momento.

---

✅ Sem avisos meteorológicos ativos para Lisboa.
"""

    output = finalize_worker_response(
        raw,
        agent_name="weather",
        user_query="Há avisos meteorológicos ativos agora?",
        language="pt",
    )

    assert "não há **avisos meteorológicos ativos**" in output
    assert "Sem avisos meteorológicos ativos" not in output
    assert "---\n\n📌 **Fonte:**" not in output


def test_weather_alertas_query_uses_warning_tool_and_horizon_note_for_named_day() -> None:
    """PT alertas wording should route as warnings and disclose named days outside horizon."""
    warnings = _FakeWeatherTool(
        "get_weather_warnings",
        "✅ No active weather warnings for Lisbon.\n\n🌤️ Weather conditions are normal.",
    )
    agent = _build_weather_agent_with_tools(warnings)

    with patch("agent.agents.weather_agent.datetime") as fake_datetime:
        fake_datetime.now.return_value = datetime(2026, 5, 5, 10, 0)
        fake_datetime.strptime.side_effect = datetime.strptime
        output = agent._run_direct_tool_fallback("Há alertas meteorológicos para domingo?")

    assert warnings.calls == [{"area": "LSB"}]
    assert "Avisos Meteorológicos" in output
    assert "fora do horizonte IPMA de 5 dias" in output


def test_weather_tonight_query_uses_forecast_tool_and_answers_temperature_first() -> None:
    """Tonight/cold prompts should use forecast data and put the low temperature first."""
    forecast = _FakeWeatherTool(
        "get_weather_forecast",
        "🌤️ Weather Forecast for Lisbon\n📅 Updated: 2026-05-05T21:00:00\n\n"
        "☀️ Tuesday, May 05\n   🌡️ 11.1°C to 20.2°C\n   🌤️ Partly cloudy\n"
        "   💧 Rain: Very unlikely (2.0%)\n   💨 Wind: Northwest (Moderate)",
    )
    agent = _build_weather_agent_with_tools(forecast)

    output = agent._run_direct_tool_fallback("How cold will it get tonight?")

    assert forecast.calls == [{"days": 1, "day_offset": 0}]
    assert "Tonight should get down to about **11.1°C**" in output
    assert output.index("Tonight should") < output.index("Weather Forecast for Lisbon")


def test_weather_tomorrow_rain_query_uses_only_tomorrow_forecast_window() -> None:
    """Tomorrow rain prompts should not include today's forecast block."""
    forecast = _FakeWeatherTool(
        "get_weather_forecast",
        "🌤️ Weather Forecast for Lisbon\n📅 Updated: 2026-05-05T21:00:00\n\n"
        "☀️ Wednesday, May 06\n   🌡️ 11.3°C to 20.5°C\n   🌤️ Sunny intervals\n"
        "   💧 Rain: Very unlikely (14.0%)\n   💨 Wind: Northwest (Moderate)",
    )
    agent = _build_weather_agent_with_tools(forecast)

    output = agent._run_direct_tool_fallback("Vai chover amanhã em Lisboa?")

    assert forecast.calls == [{"days": 1, "day_offset": 1}]
    assert "Não deverá chover" in output
    assert "14%" in output


def test_weather_unsupported_uv_logs_current_context_without_dumping_forecast() -> None:
    """Unsupported weather fields should stay concise while preserving expected current-summary coverage."""
    current = _FakeWeatherTool(
        "get_current_weather_summary",
        "🌤️ Lisbon Weather Summary\n📅 Updated: 2026-05-05T21:00:00\n📅 Today: 11°C to 20°C",
    )
    agent = _build_weather_agent_with_tools(current)

    output = agent._run_direct_tool_fallback("Tell me the UV index for today.")

    assert current.calls == [{}]
    assert "can't confirm that indicator" in output
    assert "11°C to 20°C" not in output


def test_weather_portugal_overview_uses_country_title_not_today_summary_title() -> None:
    """Portugal-wide overview prompts should not inherit the Lisbon current-weather title."""
    overview = _FakeWeatherTool(
        "get_portugal_weather_overview",
        "🇵🇹 Portugal Weather Overview - Today\n📅 Forecast date: 2026-05-05\n📊 Locations: 27",
    )
    agent = _build_weather_agent_with_tools(overview)

    output = agent._run_direct_tool_fallback("Give me a Portugal-wide weather overview for today.")

    assert overview.calls == [{"day": 0}]
    assert output.startswith("### 🌤️ **Portugal Weather Overview**")
    assert "Lisbon Weather Summary" not in output


def test_weather_finalization_normalizes_precipitation_intensity_label_in_pt() -> None:
    """PT weather intensity should be a bold sibling label, not a lowercase inline fragment."""
    raw = (
        "### 🌤️ **Avisos Meteorológicos**\n\n"
        "**🌤️ Previsão do Tempo para Lisboa**\n\n"
        "- **🌧️ Sábado, Mai 09**\n"
        "- 💧 **Chuva**: Muito provável (100.0%) | intensidade: moderado\n"
    )

    output = finalize_worker_response(
        raw,
        agent_name="weather",
        user_query="Há alertas meteorológicos para o fim de semana?",
        language="pt",
    )

    assert "| **Intensidade:** moderada" in output
    assert "| intensidade:" not in output


def test_final_visual_pass_keeps_route_fields_on_separate_streamlit_lines() -> None:
    """Standalone route fields need blank separation so Streamlit does not join them inline."""
    raw = """### 🚇 **Baixa-Chiado → Aeroporto**

⏳ **Estimated total time:** ~35–40 min
🗺️ **Recommended route:**

- 📍 **Board at** Baixa-Chiado

📌 **Source:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | **Updated:** 17:00
"""

    output = final_visual_pass(raw)

    assert "~35–40 min\n\n🗺️ **Recommended route:**" in output


def test_final_visual_pass_nests_planner_transport_flow_children() -> None:
    """Planner transport-flow child bullets should be indented under their parent section."""
    raw = """### 📅 **Suggested Itinerary**

- 🚌 **Transport**:
- From Rossio, take tram 15E toward Belém.
- Exact times should be checked on the travel day.

⚠️ 💡 Tip: Keep the museum order compact.

📌 **Source:** [*Carris*](https://www.carris.pt) | **Updated:** 17:00
"""

    output = final_visual_pass(raw)

    assert "- 🚌 **Transport**:\n    - From Rossio" in output
    assert "\n    - Exact times should be checked" in output
    assert "⚠️ 💡 Tip" not in output
    assert "💡 **Tip:** Keep the museum order compact." in output


def test_planner_tourism_cards_keep_visitlisboa_source_footer() -> None:
    """Tourism itineraries that contain place cards should retain VisitLisboa attribution."""
    raw = """### 📅 **Itinerary for Tuesday, May 05, 2026**

### 🏛️ **Carris Museum**
- 📍 **Address:** Rua 1.º de Maio, Lisboa
- 🌐 **Website:** [Official website](https://www.visitlisboa.com/en/places/carris-museum)

- 🚌 **Transport**:
- From Rossio, use Carris toward Belém.

📌 **Source:** [*IPMA*](https://www.ipma.pt) | [*Carris*](https://www.carris.pt) | **Updated:** 17:00
"""

    output = finalize_worker_response(
        raw,
        agent_name="planner",
        user_query="Plan a full museum day in Lisbon for tomorrow, starting in Rossio and using public transport.",
        language="en",
    )

    assert "[*VisitLisboa" in output
    assert "[*Carris*](https://www.carris.pt)" in output


def test_final_visual_pass_does_not_prefix_parking_heading_with_museum_icon() -> None:
    """Service headings that already have a domain icon should not receive a generic place icon."""
    raw = """**🅿️ Praça da Figueira**
- 📍 **Address:** Praça da Figueira

📌 **Source:** [*Lisboa Aberta*](https://dados.cm-lisboa.pt/) | **Updated:** 17:00
"""

    output = final_visual_pass(raw)

    assert "**🅿️ Praça da Figueira**" in output
    assert "🏛️ 🅿️" not in output


def test_final_visual_pass_structures_inline_parking_service_results() -> None:
    """QA compact parking bullets should become aligned cards with address and distance fields."""
    raw = """You can park in central Lisbon at these nearby car parks:

- **Praça da Figueira** — **0.17 km** from the queried central Lisbon area
**Address:** [Praça da Figueira](https://www.google.com/maps/search/?api=1&query=Pra%C3%A7a+da+Figueira)

📌 **Source:** [*Lisboa Aberta*](https://dados.cm-lisboa.pt/) | **Updated:** 17:00
"""

    output = final_visual_pass(raw)

    assert "**🅿️ Praça da Figueira**" in output
    assert "- 📏 **Distance:** 0.17 km from the queried central Lisbon area" in output
    assert "- 📍 **Address:** [Praça da Figueira]" in output
    assert "\n**Address:**" not in output


def test_researcher_parking_service_extraction_does_not_return_gardens() -> None:
    """Parking/car-park wording should stay in the parking service lane, not parks/gardens."""
    services = ResearcherAgent._extract_service_types(
        "Where can I park my car in central Lisbon? Are there municipal car parks?"
    )

    assert services == ["parking"]
    assert ResearcherAgent._service_category_for_type("parking") == "transportes"
    assert ResearcherAgent._extract_near_location_name(
        "Where can I park my car in central Lisbon? Are there municipal car parks?"
    ) == "central Lisbon"


def test_final_visual_pass_removes_open_data_place_noise() -> None:
    """Open-data place cards should not expose internal source prefixes or placeholder descriptions."""
    raw = """- 📊 **Torre de Belém**
    - 📂 **Categoria**: 📊 Open Data: Monumentos Nacionais
    - Descrição disponível na página oficial do local.
    - 📍 **Morada:** Avenida de Brasília

📌 **Fonte:** [*Lisboa Aberta*](https://dados.cm-lisboa.pt/) | **Atualizado:** 17:00
"""

    output = final_visual_pass(raw)

    assert "Open Data:" not in output
    assert "Descrição disponível" not in output
    assert "- 📂 **Categoria:** Monumentos Nacionais" in output


def test_final_visual_pass_removes_split_source_heading_block() -> None:
    """QA should not leave a separate source heading plus a canonical source footer."""
    raw = """### 🌤️ Resumo Meteorológico

- ✅ Sem avisos.

### 📌 Fonte
[*IPMA*](https://www.ipma.pt) | **Atualizado:** 17:00

📌 **Fonte:** [*IPMA*](https://www.ipma.pt) | **Atualizado:** 17:00
"""

    output = final_visual_pass(raw)

    assert "### 📌 Fonte" not in output
    assert output.count("📌 **Fonte:**") == 1


def test_weather_far_future_month_name_date_is_outside_forecast_horizon() -> None:
    """Natural month-name dates such as December 25th 2030 must not reach forecast tools."""
    assert WeatherAgent._is_beyond_forecast_horizon_query(
        "What will the weather be like in Lisbon on December 25th 2030?"
    ) is True


def test_final_visual_pass_removes_duplicate_forecast_horizon_notes() -> None:
    """QA-added helpful notes must not repeat the direct horizon-limit answer."""
    raw = """### 🌤️ **Weather Forecast**

⚠️ I only have reliable IPMA weather forecast data for Lisbon for the next 5 days, so I can't confirm the weather for December 25th, 2030.

---

⚠️ Helpful Notes

⚠️ IPMA weather forecasts are only reliable for up to 5 days ahead, so a forecast for December 25th, 2030 is not available.

📌 **Source:** [*IPMA*](https://www.ipma.pt/en/) | **Updated:** 20:29"""

    output = final_visual_pass(raw)

    assert "Helpful Notes" not in output
    assert output.count("5 days") == 1
    assert "December 25th, 2030" in output


def test_weather_forecast_tool_supports_focused_day_offset_window() -> None:
    """The IPMA forecast tool should support focused windows instead of only day-0 dumps."""
    from tools import ipma_api

    fake_payload = {
        "dataUpdate": "2026-05-05T21:00:00",
        "data": [
            {"forecastDate": "2026-05-05", "tMin": "10", "tMax": "20", "precipitaProb": "80", "predWindDir": "NW", "idWeatherType": 7, "classWindSpeed": 2, "classPrecInt": 1},
            {"forecastDate": "2026-05-06", "tMin": "11", "tMax": "21", "precipitaProb": "5", "predWindDir": "N", "idWeatherType": 2, "classWindSpeed": 1, "classPrecInt": 0},
        ],
    }

    with patch.object(ipma_api, "fetch_json", return_value=fake_payload):
        output = ipma_api.get_weather_forecast.invoke({"days": 1, "day_offset": 1})

    assert "Wednesday, May 06" in output
    assert "Tuesday, May 05" not in output


def test_nearest_service_query_is_not_treated_as_pagination() -> None:
    """Comparative service queries with 'mais perto' must not replay the previous search page."""
    assert ResearcherAgent._extract_pagination_request(
        "Qual o hospital e a farmácia mais perto do Saldanha?"
    ) is None
    assert ResearcherAgent._extract_pagination_request("Give me the next 2 events that match") == {"count": 2}


def test_weather_warnings_are_separated_from_forecast_day() -> None:
    """Weather warnings should not visually run into the first forecast day in Streamlit."""
    raw = """### 🌤️ Previsão Meteorológica

⚠️ Avisos meteorológicos ativos para Lisboa:

- 🟡 🌧️ PRECIPITAÇÃO — Nível: Atenção — Até 29 Abr, 10:00 — Aguaceiros.
- 🟡 ⛈️ TROVOADA — Nível: Atenção — Até 29 Abr, 10:00 — Trovoadas.
- **📅 Quinta-feira, 30 de Abril**
    - 🌡️ Temperatura: 12,9°C a 20,5°C"""

    output = finalize_worker_response(
        raw,
        agent_name="weather",
        user_query="Vai chover amahna em Lisboa?",
        language="pt",
    )

    assert "⚠️ Avisos meteorológicos ativos para Lisboa:" in output
    assert "\n\n---\n\n- **📅 Quinta-feira, 30 de Abril**" in output
    assert "- ⚠️ Avisos meteorológicos ativos" not in output


def test_transport_one_space_child_bullets_are_nested_for_streamlit() -> None:
    """One-space child bullets from tools should become valid nested Markdown bullets."""
    raw = """- 🚌 **Linha 732** — para Caselas
 - 🕐 **Próximas partidas:** 10:53, 11:06
 - ⏱️ **Tempo estimado de viagem:** ~43 min"""

    output = final_visual_pass(raw)

    assert "\n    - 🕐 **Próximas partidas:**" in output
    assert "\n    - ⏱️ **Tempo estimado de viagem:**" in output


def test_metro_route_steps_do_not_keep_google_maps_links() -> None:
    """Metro station names in route steps should render uniformly without Maps links."""
    raw = """🗺️ **Your Metro Route:**

📍 **Board at** [Aeroporto](https://www.google.com/maps/search/?api=1&query=Aeroporto)
🔁 **Transfer at** [Alameda](https://www.google.com/maps/search/?api=1&query=Alameda)"""

    output = final_visual_pass(raw)

    assert "google.com/maps" not in output
    assert "Board at** Aeroporto" in output
    assert "Transfer at** Alameda" in output


def test_malformed_service_heading_bullets_are_repaired() -> None:
    """QA repair must not leave service list items as visible heading bullets."""
    raw = """### 💊 Farmácia mais perto de Saldanha
---

### - 💊 **Farmácia Dalva**
---

### ### - 📍 **Morada:** [Avenida Duque d'Ávila, 125](https://www.google.com/maps/search/?api=1&query=Avenida)
---

### ### ### - 📏 **Distância:** 0.07 km"""

    output = final_visual_pass(raw)

    assert "### -" not in output
    assert "### ###" not in output
    assert "\n- 💊 **Farmácia Dalva**\n    - 📍 **Morada:**" in output
    assert "\n    - 📏 **Distância:** 0.07 km" in output


def test_event_card_date_and_duration_stay_aligned_with_address() -> None:
    """Event date/duration fields must not be nested below the address field."""
    raw = (
        "### 🛍️ Feira do Livro 2026\n\n"
        "- 📍 **Morada:** [Parque Eduardo VII, Lisboa](https://www.google.com/maps/search/?api=1&query=Parque)\n"
        "  - 📅 **Data/Hora:** 27 de maio a 14 de junho de 2026\n"
        "  - ⏱️ **Duração:** Cerca de 1 mês\n"
        "- 📂 **Categoria:** Feiras"
    )

    output = final_visual_pass(raw)

    assert "\n- 📅 **Data/Hora:** 27 de maio" in output
    assert "\n- ⏱️ **Duração:** Cerca de 1 mês" in output
    assert "\n  - 📅 **Data/Hora:**" not in output
    assert "\n  - ⏱️ **Duração:**" not in output


def test_place_focus_extraction_uses_actual_pt_subject() -> None:
    """PT contractions such as 'Fala-me do X' should extract X, not the whole question."""
    assert ResearcherAgent._extract_place_focus_query(
        "Fala-me do Mosteiro dos Jerónimos"
    ) == "Mosteiro dos Jerónimos"


def test_history_compaction_filters_unrelated_web_results() -> None:
    """Historical context should not leak unrelated broad-history snippets into place cards."""
    unrelated = (
        "📚 **Wikipédia: História da humanidade**\n"
        "História da humanidade é a história dos seres humanos como determinada pelos estudos arqueológicos."
    )
    related = (
        "📚 **Wikipédia: Mosteiro dos Jerónimos**\n"
        "O Mosteiro dos Jerónimos é um mosteiro manuelino situado em Belém, Lisboa. "
        "Foi mandado construir por D. Manuel I e está associado aos Descobrimentos portugueses."
    )

    unrelated_output = ResearcherAgent._compact_history_result(
        unrelated,
        "pt",
        "Mosteiro dos Jerónimos",
    )
    related_output = ResearcherAgent._compact_history_result(
        related,
        "pt",
        "Mosteiro dos Jerónimos",
    )

    assert "História da humanidade" not in unrelated_output
    assert "seres humanos" not in unrelated_output
    assert "Mosteiro dos Jerónimos" in related_output
    assert "Descobrimentos portugueses" in related_output


def test_transport_comparison_note_and_conclusion_layout_is_clean() -> None:
    """Metro/CP comparison notes and conclusions should not render as broken headings."""
    raw = (
        "### 🚇 Mobilidade em Lisboa\n\n"
        "**Comparação:** Entrecampos → Sete Rios\n\n"
        "**🚇 Metro de Lisboa**\n\n"
        "🧭 **Trajeto Metro:**\n"
        "- 📍 **Embarque na estação Entrecampos**\n"
        "- 🔵 **Linha Azul** - direção **Reboleira**\n\n"
        "- 🎯 **Saia na estação Jardim Zoológico**\n"
        "---\n\n"
        "### ****ℹ️ **Sete Rios no Metro:** a Estação Que Serve Sete Rios Chama-Se **Jardim Zoológico**.****\n"
        "---\n"
        "**🚆 Comboio**\n"
        "- ⏱️ **Tempo estimado:** 2 min\n"
        "- 🕐 **Próximas saídas mostradas:** 14:52, 15:07, 15:22\n"
        "**✅ Conclusão**\n"
        "- **Mais rápido:** Comboio"
    )

    output = final_visual_pass(raw)

    assert "### ****" not in output
    assert "**ℹ️ Sete Rios no Metro:" in output
    assert "\n---\n\n**🚆 Comboio**" in output
    assert "\n---\n\n**✅ Conclusão**" in output


def test_practical_tips_and_sentence_headings_do_not_become_hero_headers() -> None:
    """Planner/weather practical-tip prose should stay as bullets, not sentence headings between rules."""
    raw = (
        "💡 **Practical Tips** \n\n"
        "---\n\n"
        "### Expect a wet afternoon, so choose indoor or mixed indoor-outdoor stops in Belém and carry an umbrella.\n"
        "---\n\n"
        "### 🚇 Mobility and Connections"
    )

    output = final_visual_pass(raw)

    assert "### Expect" not in output
    assert "\n- Expect a wet afternoon" in output
    assert "💡 **Practical Tips**" in output


def test_cp_train_duplicate_heading_markers_are_removed() -> None:
    """CP train answers must not display literal duplicated heading markers in Streamlit."""
    raw = "### ### 🚆 **Train: Cais do Sodré → Cascais**\n**📊 Trip Summary**"

    output = final_visual_pass(raw)

    assert output.startswith("### 🚆 **Train: Cais do Sodré → Cascais**")
    assert "### ###" not in output


def test_missing_address_placeholder_link_is_removed() -> None:
    """Missing address placeholders must not survive as Google Maps links."""
    raw = "- 📍 **Morada:** [Não disponível in data](https://www.google.com/maps/search/?api=1&query=N%C3%A3o)\n- 💰 **Preço:** Gratuito"

    output = final_visual_pass(raw)

    assert "Não disponível in data" not in output
    assert "google.com/maps" not in output
    assert "- 💰 **Preço:** Gratuito" in output


def test_generic_maps_search_address_line_is_removed() -> None:
    """Generic Maps search labels are not grounded addresses and must be omitted."""
    raw = (
        "**🏛️ Museu Exemplo**\n"
        "- 📍 **Morada:** [Pesquisar no Maps](https://www.google.com/maps/search/?api=1&query=Museu+Exemplo+Lisboa)\n"
        "- 🌐 **Website:** [Página do local](https://example.com)"
    )

    output = final_visual_pass(raw)

    assert "Pesquisar no Maps" not in output
    assert "google.com/maps" not in output
    assert "Página do local" in output


def test_address_verification_placeholder_link_is_removed() -> None:
    """Address verification placeholders must not survive as Maps links."""
    raw = (
        "**🏛️ Museu Exemplo**\n"
        "- 📍 **Morada:** [Deve ser verificada.](https://www.google.com/maps/search/?api=1&query=Deve+ser+verificada)\n"
        "- 🌐 **Website:** [Página oficial](https://example.com)"
    )

    output = final_visual_pass(raw)

    assert "Deve ser verificada" not in output
    assert "google.com/maps" not in output
    assert "Página oficial" in output


def test_lisboa_card_ticket_field_moves_to_price() -> None:
    """Lisboa Card benefits must not be rendered as ticket fields without URLs."""
    raw = (
        "**🏛️ Museu da Marioneta**\n"
        "- 🎟️ **Bilhetes:** Gratuito com Lisboa Card\n"
        "- 🌐 **Website:** [Página oficial](https://example.com)"
    )

    output = final_visual_pass(raw)

    assert "🎟️" not in output
    assert "**Bilhetes:**" not in output
    assert "💶 **Preço:** Gratuito com Lisboa Card" in output
    assert "Página oficial" in output


def test_researcher_bare_item_headings_gain_place_emoji_and_pt_link_label() -> None:
    """Researcher place cards should keep emoji item headers and PT link text."""
    raw = (
        "**Museu Exemplo**\n"
        "- 📝 **Descrição:** Museu em Lisboa.\n"
        "- 🌐 **Website:** [Official website](https://example.com)"
    )

    output = final_visual_pass(canonicalize_local_information_terms(raw, language="pt"))

    assert "**🏛️ Museu Exemplo**" in output
    assert "[Página oficial](https://example.com)" in output
    assert "Official website" not in output


def test_madeira_ambiguity_route_card_fields_are_separate_bullets() -> None:
    """Madeira ambiguity follow-up fields should not collapse into one oversized paragraph."""
    raw = (
        "⚠️ **Ambiguidade em 'Madeira':** posso estar a interpretar uma destas opções:\n\n"
        "A) 🏝️ **Ilha da Madeira** — requer avião.\n"
        "B) 🚇 **Rua Humberto Madeira / Av. Ilha da Madeira, em Lisboa** — sigo abaixo.\n\n"
        "### 🚇 Mobilidade em Lisboa\n\n"
        "🚇 **Opção urbana em Lisboa:** Rua Humberto Madeira / Av. Ilha da Madeira\n\n"
        "📍 **Destino Provável:** Rua Humberto Madeira, Lisboa\n\n"
        "🚇 **Metro Mais Próximo:** Encarnação (🔴 Linha Vermelha) 🎯 **Como Usar o Metro:** Segue pela Linha Vermelha."
    )

    output = final_visual_pass(raw)

    assert "\n- 🚇 **Opção urbana em Lisboa:**" in output
    assert "\n- 📍 **Destino Provável:**" in output
    assert "\n- 🚇 **Metro Mais Próximo:**" in output
    assert "\n- 🎯 **Como Usar o Metro:**" in output


def test_pt_metro_oriente_query_is_not_detected_as_spanish() -> None:
    """Short PT transport phrases containing 'metro' should remain PT, not Spanish fallback."""
    assert resolve_output_language("Quero ir de metro para Oriente.", "en") == ("pt", False, "pt")


def test_generic_public_transport_route_composes_bus_and_metro_options() -> None:
    """Open-ended public-transport route questions should surface bus and Metro options when both exist."""
    from agent.agents.transport_agent import _build_deterministic_route_tool_response

    metro_result = "🗺️ **Route: ISCTE → Zara do Rossio**\n\n🚇 **METRO ROUTE**\n   1. Board at **Entrecampos**\n   2. Exit at **Rossio**\n\n📌 **Source:** [*Metro de Lisboa*](https://www.metrolisboa.pt)"
    carris_result = "🚌 **Autocarros**\n\nBUSES\n----\n732: para Caselas\nNext: 15:00 (stop Faculdade Farmácia)\n~37min travel\n\n📌 **Fonte:** [*Carris*](https://www.carris.pt)"

    with patch("tools.transport_api.get_route_between_stations.func", side_effect=lambda **_: metro_result), patch(
        "tools.carris_api.carris_find_routes_between.func",
        side_effect=lambda **_: carris_result,
    ):
        output = _build_deterministic_route_tool_response(
            "Quero ir de transportes públicos entre o ISCTE e a Zara do Rossio"
        )

    assert output is not None
    assert "Opções de transporte público" in output
    assert "**🚌 Autocarros**" in output
    assert "**🚇 Metro**" in output
    assert "[*Carris*]" in output
    assert "[*Metro de Lisboa*]" in output


def test_portuguese_public_transport_endpoint_drops_mode_suffix() -> None:
    """Destination names should not absorb 'de transportes públicos' into the place name."""
    from agent.agents.transport_agent import _build_deterministic_route_tool_response

    metro_result = "No Metro route for Belém."
    carris_result = "🚌 **Autocarros**\n\n- 🚌 **15E** — Praça da Figueira → Belém\n    - 🚏 **Embarque:** Praça da Figueira\n    - 🚏 **Saída:** Belém"

    with patch("tools.transport_api.get_route_between_stations.func", side_effect=lambda **_: metro_result) as metro, patch(
        "tools.carris_api.carris_find_routes_between.func",
        side_effect=lambda **_: carris_result,
    ) as carris:
        output = _build_deterministic_route_tool_response(
            "Como vou do Rossio para Belém de transportes públicos?"
        )

    assert output is not None
    metro.assert_called_once_with(origin="Rossio", destination="Belém")
    carris.assert_called_once_with(origin="Rossio", destination="Belém")
    assert "15E" in output


def test_after_hours_culture_recommendation_avoids_closed_indoor_museums() -> None:
    """Late museum/monument requests should not recommend venues closed before the requested window."""
    response = ResearcherAgent._maybe_answer_after_hours_culture_query(
        "Qual museu ou monumento recomendas ir neste domingo sendo que apenas tenho das 19 às 20h para visitar?",
        "pt",
    )

    assert response is not None
    assert "evitaria recomendar **museus interiores**" in response
    assert "### 🏛️ Recomendação Para 19:00-20:00" in response
    assert "📍 **Morada:**" in response
    assert "09:30" not in response
    assert "17:30" not in response


def test_after_hours_culture_recommendation_bypasses_planner_rewrite() -> None:
    """Grounded single-answer researcher shortcuts should not be rewritten as itineraries."""
    response = ResearcherAgent._maybe_answer_after_hours_culture_query(
        "Qual museu ou monumento recomendas ir neste domingo sendo que apenas tenho das 19 às 20h para visitar?",
        "pt",
    )

    assert response is not None
    assert MultiAgentAssistant._should_preserve_direct_researcher_answer(
        {"researcher": response}
    ) is True
    assert MultiAgentAssistant._should_preserve_direct_researcher_answer(
        {"researcher": response, "transport": "Metro route"}
    ) is False


def test_final_visual_pass_removes_empty_tip_labels_and_pt_missing_values() -> None:
    """Final rendering should omit empty tips and PT missing-value field rows."""
    raw = (
        "### 🏛️ Recomendação\n\n"
        "- Nota prática:\n"
        "- 🕒 **Horário de funcionamento:** Deve ser verificado no website oficial\n"
        "- 💰 **Preço:** Não indicado na base de dados\n"
        "📌 **Fonte:** [*VisitLisboa Locais*](https://www.visitlisboa.com/pt-pt/locais)"
    )

    output = final_visual_pass(raw)

    assert "Nota prática" not in output
    assert "Deve ser verificado" not in output
    assert "VisitLisboa Locais" in output


def test_final_visual_pass_repairs_detached_transport_delayed_metric() -> None:
    """QA-repaired CP status rows should not leave warning metrics as standalone paragraphs."""
    raw = """### 🚆 **CP Trains - Lisbon Metropolitan Area (AML)**

**Current situation**
- 📊 **Tracked suburban trains:** 32
- ✅ **Shown without delay:** 11

⚠️ **Delayed:** 21

📌 **Source:** [*CP*](https://www.cp.pt) | **Updated:** 20:55"""

    output = final_visual_pass(raw)

    assert "\n- ⚠️ **Delayed:** 21" in output
    assert "\n\n⚠️ **Delayed:**" not in output


def test_pt_local_information_values_localizes_event_date_fragments() -> None:
    """PT event metadata should not keep English month abbreviations or date-count text."""
    raw = "- 📅 **Data/Hora:** 09 May | 16 May | (+49 more dates)"

    output = localize_local_information_values(raw, language="pt")

    assert "09 Mai" in output
    assert "16 Mai" in output
    assert "+49 datas adicionais" in output
    assert "May" not in output
    assert "more dates" not in output


def test_final_visual_pass_strips_internal_qa_annotations_and_pt_technical_terms() -> None:
    """Final rendering must remove QA leakage and localize recurring English technical words in PT responses."""
    raw = (
        "### Resultado\n"
        "- ⚠️ Os horários de funcionamento não foram fornecidos pelo QA.\n"
        "- O runtime não conseguiu confirmar este detalhe no backend.\n"
        "📌 **Fonte:** [*VisitLisboa*](https://www.visitlisboa.com)"
    )

    output = final_visual_pass(raw)

    assert "QA" not in output
    assert "horários de funcionamento não foram fornecidos" not in output.lower()
    assert "runtime" not in output.lower()
    assert "backend" not in output.lower()
    assert "sistema" in output.lower()


def test_final_visual_pass_removes_generic_city_only_address_lines() -> None:
    """City-only address stubs should not survive final card rendering."""
    raw = "### Local\n- 📍 **Morada:** Lisboa\n- 🌐 **Website:** [site](https://example.com)"

    output = final_visual_pass(raw)

    assert "**Morada:** Lisboa" not in output
    assert "Website" in output


def test_final_visual_pass_splits_adjacent_warning_blocks() -> None:
    """Adjacent warning blocks should render as separate Streamlit paragraphs."""
    raw = "⚠️ **Aviso:** primeiro aviso ⚠️ **Nota:** segundo aviso\n📌 **Fonte:** [*Metro de Lisboa*](https://www.metrolisboa.pt)"

    output = final_visual_pass(raw)

    assert "primeiro aviso\n\n⚠️ **Nota:** segundo aviso" in output
    assert "segundo aviso\n\n📌 **Fonte:**" in output


def test_final_visual_pass_removes_invalid_links_and_keeps_single_footer_last() -> None:
    """Broken placeholder links and trailing QA notes must not survive final rendering."""
    raw = (
        "### Evento\n\n"
        "- 🎟️ [Bilhetes](Não disponível)\n\n"
        "📌 **Fonte:** [*VisitLisboa Eventos*](https://www.visitlisboa.com/pt-pt/eventos)\n\n"
        "- ⚠️ QA validation structure could not be confirmed after retry\n"
        "📌 **Fonte:** [*VisitLisboa*](https://www.visitlisboa.com)"
    )

    output = final_visual_pass(raw)

    assert "[Bilhetes](Não disponível)" not in output
    assert "**Bilhetes:** Não disponível" not in output
    assert "QA validation" not in output
    assert output.count("📌 **Fonte:**") == 1
    assert output.rstrip().endswith("[*VisitLisboa*](https://www.visitlisboa.com)")


def test_pt_label_localizer_does_not_corrupt_url_paths() -> None:
    """PT label localization must not translate URL path segments such as tickets/location."""
    raw = (
        "🌐 **Website:** [visitlisboa.com](https://www.visitlisboa.com/en/events/foo#tickets)\n"
        "📍 **Location:** Rua Augusta, Lisboa"
    )

    output = localize_local_information_values(raw, language="pt")

    assert "https://www.visitlisboa.com/en/events/foo#tickets" in output
    assert "#bilhetes" not in output.lower()
    assert "📍 **Localização:** Rua Augusta, Lisboa" in output


def test_researcher_prompt_requires_restaurant_criteria_extraction() -> None:
    """Restaurant queries should preserve criteria such as views, price, and touristiness."""
    from agent.prompts.researcher import get_researcher_prompt

    prompt = get_researcher_prompt(language="en")

    assert "Criteria Extraction" in prompt
    assert "river or Tagus view" in prompt
    assert "touristiness" in prompt
    assert "curated from available data" in prompt


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


@patch("agent.graph.LANGSMITH_AVAILABLE", False)
def test_multiagent_chat_prepends_bilingual_note_for_french_query() -> None:
    """Non-PT/EN queries should get the English fallback note at the top of the final answer."""
    assistant = MultiAgentAssistant.__new__(MultiAgentAssistant)
    assistant.state = {"messages": [], "user_context": None}
    assistant.supervisor = MagicMock()
    assistant.supervisor.reset_llm_usage_tracking = MagicMock()
    assistant.supervisor.route = MagicMock(
        return_value={"agents": [], "direct_response": "Sunny in Lisbon.", "reasoning": "direct"}
    )
    assistant.qa_agent = MagicMock()
    assistant.qa_agent.reset_llm_usage_tracking = MagicMock()
    assistant.agents = {}
    assistant._dedupe_preserve_order = lambda items: items
    assistant._append_assistant_message = MagicMock()
    assistant._collect_execution_summary = MagicMock(return_value={})
    assistant._print_execution_summary = MagicMock()
    result = assistant.chat("Quel temps fait-il a Lisbonne aujourd'hui?", language="pt")

    expected_note = build_bilingual_note("fr")
    user_context = assistant.state["user_context"] or {}
    assert result.startswith(expected_note)
    assert "Sunny in Lisbon." in result
    assert user_context.get("requires_bilingual_note") is True
    assert user_context.get("detected_language") == "fr"


@pytest.mark.parametrize(
    ("query", "ui_default", "expected_language", "expected_requires_note", "expected_detected"),
    [
        ("Quel temps fait-il a Lisbonne aujourd'hui?", "pt", "en", True, "fr"),
        ("Wie komme ich vom Rossio nach Belem?", "pt", "en", True, "de"),
        ("里斯本今天的天气怎么样？", "pt", "en", True, "zh-cn"),
        ("リスボンの今日の天気は？", "pt", "en", True, "ja"),
        ("¿Como llegar a Belem desde Rossio?", "pt", "en", True, "es"),
        ("Como vou do Rossio para Belem?", "en", "pt", False, "pt"),
        ("Dá-me o ponto de situação do Metro, autocarros e comboios em Lisboa.", "pt", "pt", False, "pt"),
        ("How do I get from Rossio to Belem?", "pt", "en", False, "en"),
        ("ola quero ir ao rossio amanha", "en", "pt", False, "pt"),
    ],
)
def test_resolve_output_language_handles_supported_and_fallback_languages(
    query: str,
    ui_default: str,
    expected_language: str,
    expected_requires_note: bool,
    expected_detected: str,
) -> None:
    """Language routing must keep PT/EN, and push every other language to English with a note."""
    language, requires_note, detected = resolve_output_language(query, ui_default)

    assert language == expected_language
    assert requires_note is expected_requires_note
    assert detected == expected_detected


def test_build_bilingual_note_matches_requested_markdown_shape() -> None:
    """The bilingual fallback note should follow the exact three-line quoted structure."""
    assert build_bilingual_note("fr") == (
        "> ℹ️ **This assistant speaks Portuguese and English.**\n"
        "> Your message was detected as **French** — answering in English below.\n"
        "> *Português · English · Type in either language anytime.*"
    )


def test_sanitize_qa_warning_filters_internal_transport_caveats_in_pt() -> None:
    """Internal QA-only transport caveats should be dropped or rewritten before reaching the user."""
    assert MultiAgentAssistant._sanitize_single_qa_warning(
        "Se a resposta final mencionar títulos/categorias, deve manter rótulos consistentes em PT-PT.",
        "pt",
    ) is None

    assert MultiAgentAssistant._sanitize_single_qa_warning(
        "A fonte indicada é apenas do Metro de Lisboa; os restantes operadores não têm fonte explícita no output.",
        "pt",
    ) == (
        "A fonte indicada é apenas a do Metro de Lisboa; confirme Carris, "
        "Carris Metropolitana e CP nas respetivas fontes oficiais."
    )


def test_final_visual_pass_repairs_known_live_typos() -> None:
    """The last formatting pass should scrub the narrow repeated-letter glitches seen in live QA repairs."""
    raw = (
        "The line iis on time. "
        "The route tool used this orrigin input. "
        "Carris Metropolitanaa may miss veryy recent changes. "
        "Visit the TTour de Belém. "
        "The GTFS feed became GGTFS and the stop was afffected. "
        "A resposta ficou ppara veriiificar e o estadoo ficou com espa\u00e7os  a mais. "
        "Conv\u00e9m confirmar com foontes oficiais, estado opeeracional, nota expl\u00edcitta e fontes oficiaiis."
    )

    output = final_visual_pass(raw)

    assert "iis" not in output
    assert "orrigin" not in output
    assert "Metropolitanaa" not in output
    assert "veryy" not in output
    assert "TTour" not in output
    assert "GGTFS" not in output
    assert "afffected" not in output
    assert "ppara" not in output
    assert "veriiificar" not in output
    assert "estadoo" not in output
    assert "foontes" not in output
    assert "opeeracional" not in output
    assert "expl\u00edcitta" not in output
    assert "oficiaiis" not in output
    assert "  " not in output
    assert "The line is on time." in output
    assert "origin input" in output
    assert "Carris Metropolitana may miss very recent changes." in output
    assert "Tour de Belém" in output
    assert "GTFS" in output
    assert "affected" in output
    assert "fontes oficiais" in output
    assert "estado operacional" in output
    assert "nota explícita" in output


def test_structure_weather_markdown_nests_days_even_after_generic_bullet_normalization() -> None:
    """Weather day rows must remain parent items after generic formatting has already added bullets."""
    raw = (
        "**Previsão do Tempo para Lisboa**\n\n"
        "- **☀️ Sábado, Abr 18**\n"
        "- 🌡️ 13.1°C a 28.2°C\n"
        "- 🌤️ Parcialmente nublado\n"
        "- 💧 **Chuva**: sem precipitação (0.0%)\n"
        "- 💨 **Vento**: Norte (fraca)\n"
    )

    structured = structure_weather_markdown(raw)

    assert "- **☀️ Sábado, Abr 18**" in structured
    assert "    - 🌡️ 13.1°C a 28.2°C" in structured
    assert "    - 🌤️ Parcialmente nublado" in structured
    assert "    - 💧 **Chuva**: sem precipitação (0.0%)" in structured
    assert "    - 💨 **Vento**: Norte (fraca)" in structured


def test_final_visual_pass_normalizes_quick_action_weather_summary_layout() -> None:
    """Quick-action weather snippets should stay nested without duplicate clear-status lines."""
    raw = (
        "### 🌤️ Resumo Meteorológico\n\n"
        "- ✅ Sem avisos meteorológicos ativos para Lisboa.\n\n"
        "- 🌤️ As condições meteorológicas são normais.\n"
        "**🌤️ Previsão do Tempo para Lisboa**\n\n"
        "- **⛈️ Quarta-feira, Abr 29**\n"
        " - 🌡️ 13.7°C a 18.7°C\n"
        " - 🌤️ *Showers and thunderstorms*\n"
        " - 💧 **Chuva**: muito provável (100.0%) | intensidade: forte\n\n"
        "- 💨 **Vento**: Sudoeste (moderado)\n"
        "---\n"
    )

    output = final_visual_pass(raw)

    assert "- ✅ Sem avisos meteorológicos ativos para Lisboa." in output
    assert "As condições meteorológicas são normais" not in output
    assert "Sem avisos meteorológicos ativos para Lisboa.\n\n**🌤️ Previsão do Tempo para Lisboa**" in output
    assert "- **⛈️ Quarta-feira, Abr 29**" in output
    assert "\n    - 🌡️ 13.7°C a 18.7°C" in output
    assert "\n    - 🌤️ *Showers and thunderstorms*" in output
    assert "\n    - 💧 **Chuva**: muito provável (100.0%) | intensidade: forte" in output
    assert "\n    - 💨 **Vento**: Sudoeste (moderado)" in output
    assert not output.endswith("---")
    assert "\n - 🌡️" not in output


def test_multiagent_finalize_chat_response_reapplies_weather_structure_at_the_end() -> None:
    """The final chat wrapper must preserve nested weather bullets after its own footer and polish passes."""
    assistant = MultiAgentAssistant.__new__(MultiAgentAssistant)
    assistant.state = {"messages": [], "user_context": {}}
    assistant._append_assistant_message = MagicMock()
    assistant._collect_execution_summary = MagicMock(return_value={})
    assistant._print_execution_summary = MagicMock()

    weather_response = (
        "### 🌤️ Previsão Meteorológica\n\n"
        "✅ Sem avisos meteorológicos ativos para Lisboa.\n\n"
        "- 🌤️ As condições meteorológicas são normais.\n\n"
        "---\n\n"
        "**🌤️ Previsão do Tempo para Lisboa**\n\n"
        "- **☀️ Sábado, Abr 18**\n"
        "- 🌡️ 13.1°C a 28.2°C\n"
        "- 🌤️ Parcialmente nublado\n"
        "- 💧 **Chuva**: sem precipitação (0.0%)\n"
        "- 💨 **Vento**: Norte (fraca)\n\n"
        "📌 **Fonte:** [*IPMA*](https://www.ipma.pt) | **Atualizado:** 23:13"
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
            response=weather_response,
            message="Qual é a previsão do tempo para os próximos 3 dias?",
            language="pt",
            agents_to_call=["weather"],
            routing_reasoning="single weather",
            agent_outputs={"weather": weather_response},
            direct_response_used=False,
            start_time=0.0,
            workers=["weather"],
            run_workers_in_parallel=False,
            qa_result=None,
            retry_agents_used=[],
            final_repair_ran=False,
            simple_weather_fact_check={"performed": True},
        )

    assert "- **☀️ Sábado, Abr 18**" in output
    assert "    - 🌡️ 13.1°C a 28.2°C" in output
    assert "    - 🌤️ Parcialmente nublado" in output
    assert "📌 **Fonte:** [*IPMA*](https://www.ipma.pt) | **Atualizado:**" in output


def test_multiagent_finalize_chat_response_rehydrates_event_metadata_from_deterministic_lookup() -> None:
    """Final publication should restore grounded event metadata when the synthesized researcher answer drops fields."""
    assistant = MultiAgentAssistant.__new__(MultiAgentAssistant)
    assistant.state = {"messages": [], "user_context": {}}
    assistant._append_assistant_message = MagicMock()
    assistant._collect_execution_summary = MagicMock(return_value={})
    assistant._print_execution_summary = MagicMock()
    researcher_agent = MagicMock()
    researcher_agent._run_direct_event_lookup.return_value = (
        "1. 📅 **Lxtriathlon Grand Prix**\n"
        "   🗓️ **Quando:** 19 Abr\n"
        "   ⏱️ **Duração:** 🎯 Um só dia\n"
        "   📂 **Categoria:** Desporto\n"
        "   📝 **Descrição:** Grande evento de atletismo organizado pelo Lxtriathlon.\n"
        "   📍 Complexo Desportivo Moniz Pereira, Lisboa\n"
        "   💰 **Preço:** Não indicado\n"
        "   🔗 https://www.visitlisboa.com/en/events/lxtriathlon-grand-prix\n"
        "   🎟️ **Comprar bilhetes:** https://www.visitlisboa.com/en/events/lxtriathlon-grand-prix#tickets\n\n"
        "📌 **Fonte:** [*VisitLisboa Eventos*](https://www.visitlisboa.com/pt-pt/eventos)"
    )
    assistant.agents = {"researcher": researcher_agent}

    final_text = (
        "### 🏅 Lxtriathlon Grand Prix\n\n"
        "- 📝 **Descrição:** Grande evento de atletismo organizado pelo Lxtriathlon.\n"
        "- 📍 **Morada:** [Complexo Desportivo Moniz Pereira, Lisboa](https://www.google.com/maps/search/?api=1&query=Complexo+Desportivo+Moniz+Pereira%2C+Lisboa)\n"
        "- 📅 **Data/Hora:** 19 Abr\n"
        "- 💰 **Preço:** Não indicado\n\n"
        "📌 **Fonte:** [*VisitLisboa Eventos*](https://www.visitlisboa.com/pt-pt/eventos)"
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
            response=final_text,
            message="Quero explorar a cultura local. Que grandes eventos temos esta semana?",
            language="pt",
            agents_to_call=["researcher"],
            routing_reasoning="single researcher",
            agent_outputs={"researcher": final_text},
            direct_response_used=False,
            start_time=0.0,
            workers=["researcher"],
            run_workers_in_parallel=False,
            qa_result=None,
            retry_agents_used=[],
            final_repair_ran=False,
            simple_weather_fact_check=None,
        )

    assert "- ⏱️ **Duração:** 🎯 Um só dia" in output
    assert "- 📂 **Categoria:** Desporto" in output
    assert "- 🌐 [Mais detalhes](https://www.visitlisboa.com/en/events/lxtriathlon-grand-prix)" in output
    assert "- 🎟️ [Bilhetes](https://www.visitlisboa.com/en/events/lxtriathlon-grand-prix#tickets)" in output
    assert "Aqui tens uma seleção de eventos culturais e de grande visibilidade" in output
    researcher_agent._run_direct_event_lookup.assert_called_once()


def test_sanitize_qa_warning_drops_benign_visitlisboa_domain_meta_notes_in_pt() -> None:
    """QA meta-notes about visitlisboa.com or bare www markers must not leak into PT user output."""
    assert MultiAgentAssistant._sanitize_single_qa_warning(
        "Os links apresentados usam o domínio visitlisboa.com, que é aceitável.",
        "pt",
    ) is None

    assert MultiAgentAssistant._sanitize_single_qa_warning(
        "Some URLs reference unverified domains: www. Please verify them before visiting.",
        "pt",
    ) is None

    assert MultiAgentAssistant._sanitize_single_qa_warning(
        "Event details (dates, times, ticket prices) should be confirmed at visitlisboa.com, as this data is synced daily and may have changed.",
        "pt",
    ) is None


def test_sanitize_qa_warning_drops_transport_meta_domain_and_detail_notes_in_pt() -> None:
    """Transport QA notes about domain trust or missing operator detail must never leak to PT output."""
    assert MultiAgentAssistant._sanitize_single_qa_warning(
        "As ligações/URLs citadas pertencem a domínios conhecidos e não levantam suspeita de fabricação.",
        "pt",
    ) is None

    assert MultiAgentAssistant._sanitize_single_qa_warning(
        "Os indicadores apresentados devem ser interpretados conforme a fonte em tempo real; não foram fornecidos detalhes de cada alerta ou linha afetada.",
        "pt",
    ) is None

    assert MultiAgentAssistant._sanitize_single_qa_warning(
        "Os dados de autocarros e comboios apresentados são parciais e não especificam perturbações concretas por linha ou serviço.",
        "pt",
    ) is None


def test_researcher_worker_formats_event_cards_with_maps_links_in_pt() -> None:
    """PT event answers should render stable cards, Google Maps links, and only the events source label."""
    raw = (
        "📊 **Filtro aplicado:** esta semana\n"
        "🎭 **Eventos Culturais**\n\n"
        "1. 📅 **Lxtriathlon Grand Prix**\n"
        "   🗓️ **Quando:** 19 abr.\n"
        "   ⏱️ **Duração:** 1 dia\n"
        "   📂 **Categoria:** Desporto\n"
        "   📝 **Descrição:** Evento desportivo de grande dimensão em Lisboa.\n"
        "   📍 Complexo Desportivo Moniz Pereira, R. João Amaral, Lisboa\n"
        "   💰 **Preço:** Não indicado\n"
        "   🔗 https://www.visitlisboa.com/pt-pt/eventos/lxtriathlon-grand-prix\n\n"
        "📌 **Fonte:** [*VisitLisboa*](https://www.visitlisboa.com/pt-pt/eventos)"
    )

    output = finalize_worker_response(
        raw,
        agent_name="researcher",
        user_query="Quero explorar a cultura local. Que grandes eventos temos esta semana?",
        language="pt",
    )

    assert "### 🏅 Lxtriathlon Grand Prix" in output
    assert "- 📝 **Descrição:** Evento desportivo de grande dimensão em Lisboa." in output
    assert "- 📅 **Data/Hora:** 19 abr." in output
    assert "- 📍 **Morada:** [Complexo Desportivo Moniz Pereira, R. João Amaral, Lisboa](https://www.google.com/maps/search/?api=1&query=Complexo+Desportivo+Moniz+Pereira%2C+R.+Jo%C3%A3o+Amaral%2C+Lisboa)" in output
    assert "- 🌐 [Mais detalhes](https://www.visitlisboa.com/pt-pt/eventos/lxtriathlon-grand-prix)" in output
    assert "[*VisitLisboa Eventos*](https://www.visitlisboa.com/pt-pt/eventos)" in output
    assert "Resumo da pesquisa" not in output
    assert "domínios conhecidos" not in output


def test_researcher_worker_keeps_structured_event_cards_stable_on_second_pass() -> None:
    """A second researcher finalization pass must not turn event metadata lines into fake headings."""
    first_pass = finalize_worker_response(
        (
            "1. 📅 **Lxtriathlon Grand Prix**\n"
            "   🗓️ **Quando:** 19 abr.\n"
            "   ⏱️ **Duração:** 1 dia\n"
            "   📂 **Categoria:** Desporto\n"
            "   📝 **Descrição:** Evento desportivo de grande dimensão em Lisboa.\n"
            "   📍 Complexo Desportivo Moniz Pereira, R. João Amaral, Lisboa\n"
            "   💰 **Preço:** Não indicado\n"
            "   🔗 https://www.visitlisboa.com/pt-pt/eventos/lxtriathlon-grand-prix\n\n"
            "- ⚠️ Alguns eventos não indicam preço. Os URLs apresentados parecem usar domínios conhecidos (visitlisboa.com) e não levantam alerta.\n\n"
            "📌 **Fonte:** [*VisitLisboa*](https://www.visitlisboa.com/pt-pt/eventos)"
        ),
        agent_name="researcher",
        user_query="Quero explorar a cultura local. Que grandes eventos temos esta semana?",
        language="pt",
    )

    second_pass = finalize_worker_response(
        first_pass,
        agent_name="researcher",
        user_query="Quero explorar a cultura local. Que grandes eventos temos esta semana?",
        language="pt",
    )

    assert "### 🏅 Lxtriathlon Grand Prix" in second_pass
    assert "- 📅 **Data/Hora:** 19 abr." in second_pass
    assert "### 📅 Data/Hora" not in second_pass
    assert "### 💶 Preço" not in second_pass
    assert "domínios conhecidos" not in second_pass


def test_researcher_worker_normalizes_qa_repaired_event_heading_fields() -> None:
    """QA final-repair event cards with `### 📅 Data/Hora` headings must collapse back into one indented event card."""
    raw = (
        "### 🎭 Eventos Culturais\n\n"
        "Claro — esta semana, há vários eventos interessantes.\n\n"
        "---\n\n"
        "### 🎵 Lxtriathlon Grand Prix\n\n"
        "📝 **Descrição:** Grande evento atlético.\n\n"
        "📍 **Morada:** [Complexo Desportivo Moniz Pereira, R. João Amaral, Lisboa](https://www.google.com/maps/search/?api=1&query=Complexo+Desportivo+Moniz+Pereira%2C+R.+Jo%C3%A3o+Amaral%2C+Lisboa)\n\n"
        "---\n\n"
        "### 📅 Data/Hora: **19 Abr**\n\n"
        "💰 **Preço:** Não indicado\n\n"
        "---\n\n"
        "### 🌐 [Mais Detalhes](https://www.visitlisboa.com/pt-pt/events/lxtriathlon-grand-prix)\n\n"
        "---\n\n"
        "- ⚠️ Alguns eventos repetem-se em várias datas; acima estão listados apenas os que aparecem nesta semana.\n"
        "- ⚠️ A disponibilidade, datas, horários e preços devem ser confirmados no VisitLisboa antes de ires.\n"
        "- ⚠️ Os links podem variar entre versões e devem ser verificados antes de abrir.\n\n"
        "📌 **Fonte:** [*VisitLisboa Eventos*](https://www.visitlisboa.com/pt-pt/eventos)"
    )

    output = finalize_worker_response(
        raw,
        agent_name="researcher",
        user_query="Quero explorar a cultura local. Que grandes eventos temos esta semana?",
        language="pt",
    )

    assert "### 🏅 Lxtriathlon Grand Prix" in output
    assert "- 📅 **Data/Hora:** 19 Abr" in output
    assert "- 💰 **Preço:** Não indicado" in output
    assert "- 🌐 [Mais detalhes](https://www.visitlisboa.com/pt-pt/events/lxtriathlon-grand-prix)" in output
    assert "Alguns eventos repetem-se" not in output
    assert "### 📅 Data/Hora" not in output
    assert "### 🌐" not in output


def test_researcher_worker_drops_remaining_live_event_qa_warning_variants() -> None:
    """Live QA caveats about missing exact times/prices or mixed link language must not leak into event cards."""
    raw = (
        "### 🎵 Lxtriathlon Grand Prix\n\n"
        "📝 **Descrição:** Grande evento atlético.\n\n"
        "📍 **Morada:** [Complexo Desportivo Moniz Pereira, R. João Amaral, Lisboa](https://www.google.com/maps/search/?api=1&query=Complexo+Desportivo+Moniz+Pereira%2C+R.+Jo%C3%A3o+Amaral%2C+Lisboa)\n"
        "🗓️ **Data/Hora:** 19 Abr\n"
        "💰 **Preço:** Não indicado\n\n"
        "---\n\n"
        "### 🌐 [Mais Detalhes](https://www.visitlisboa.com/en/events/lxtriathlon-grand-prix)\n\n"
        "---\n\n"
        "- ⚠️ Alguns eventos não apresentam hora exata e/ou preço indicado na fonte.\n"
        "- ⚠️ Há mistura de idioma nos links/URLs (pt e en) dentro da fonte, mas os campos principais estão em português.\n\n"
        "📌 **Fonte:** [*VisitLisboa Eventos*](https://www.visitlisboa.com/pt-pt/eventos)"
    )

    output = finalize_worker_response(
        raw,
        agent_name="researcher",
        user_query="Quero explorar a cultura local. Que grandes eventos temos esta semana?",
        language="pt",
    )

    assert "- 📅 **Data/Hora:** 19 Abr" in output
    assert "- 🌐 [Mais detalhes](https://www.visitlisboa.com/en/events/lxtriathlon-grand-prix)" in output
    assert "hora exata" not in output
    assert "mistura de idioma" not in output


def test_researcher_worker_normalizes_malformed_live_final_repair_event_card() -> None:
    """Malformed QA final-repair event cards with broken price markdown and a notes heading must collapse into one clean card."""
    raw = (
        "### 🎭 Festival de Cinema Italiano em Lisboa\n\n"
        "- 📝 **Descrição:** O festival regressa a Lisboa de **9 a 19 de abril**, com exibições em vários espaços da cidade, incluindo o Cinema São Jorge e a Cinemateca Portuguesa.\n"
        "- 📍 **Morada:** [Cinema São Jorge, Avenida da Liberdade, 175, 1250-141 Lisboa](https://www.google.com/maps/search/?api=1&query=Cinema+S%C3%A3o+Jorge%2C+Avenida+da+Liberdade%2C+175%2C+1250-141%2C+Lisboa)\n"
        "- 📅 **Data/Hora:** 9 Abr a 19 Abr\n"
        "- 🌐 [Mais detalhes](https://www.visitlisboa.com/en/events/italian-film-festival-1)\n"
        "- 💶 Preço:** **€4 a €5\n\n"
        "### 📝 Descrição:** ⚠️ **Notas úteis\n\n"
        "- 📝 **Descrição:** ⚠️ As datas e preços acima devem ser confirmados no VisitLisboa, porque podem ter sido atualizados.\n"
        "- 🌐 [Mais detalhes](https://www.visitlisboa.com/en/events/international-day-for-monuments-and-sites-1)\n"
        "- ⚠️ Em alguns eventos, o preço não está disponível nos dados e deve ser verificado.\n"
        "- ⚠️ Alguns eventos usam datas amplas ou múltiplas ocorrências;\n\n"
        "📌 **Fonte:** [*VisitLisboa Eventos*](https://www.visitlisboa.com/pt-pt/eventos)"
    )

    output = finalize_worker_response(
        raw,
        agent_name="researcher",
        user_query="Quero explorar a cultura local. Que grandes eventos temos esta semana?",
        language="pt",
    )

    assert "### 🎬 Festival de Cinema Italiano em Lisboa" in output
    assert "- 📅 **Data/Hora:** 9 Abr a 19 Abr" in output
    assert "- 💰 **Preço:** €4 a €5" in output
    assert "- 🌐 [Mais detalhes](https://www.visitlisboa.com/en/events/italian-film-festival-1)" in output
    assert "Notas úteis" not in output
    assert "As datas e preços acima" not in output
    assert "### 📝 Descrição:" not in output


def test_researcher_worker_drops_separator_between_intro_and_first_event_card() -> None:
    """Event responses must not keep `---` between the intro paragraph and the first event card."""
    raw = (
        "### 🎭 Eventos Culturais\n"
        "Aqui tens uma seleção de eventos culturais e de grande visibilidade **esta semana em Lisboa**:\n"
        "---\n"
        "### 🎵 Lxtriathlon Grand Prix\n\n"
        "- 📝 **Descrição:** Grande evento de atletismo.\n"
        "- 📍 **Morada:** [Complexo Desportivo Moniz Pereira, Lisboa](https://www.google.com/maps/search/?api=1&query=Complexo+Desportivo+Moniz+Pereira%2C+Lisboa)\n"
        "- 📅 **Data/Hora:** 19 Abr\n"
    )

    output = finalize_worker_response(
        raw,
        agent_name="researcher",
        user_query="Quero explorar a cultura local. Que grandes eventos temos esta semana?",
        language="pt",
    )

    assert "### 🎭 Eventos Culturais" in output
    assert "Aqui tens uma seleção" in output
    assert "esta semana em Lisboa**:\n---\n### 🎵" not in output
    assert "Aqui tens uma seleção de eventos culturais e de grande visibilidade **esta semana em Lisboa**:\n\n### 🏅 Lxtriathlon Grand Prix" in output


def test_reconcile_researcher_event_response_restores_missing_event_fields_from_worker_output() -> None:
    """The final researcher response should recover grounded metadata from the raw worker output after QA trims it."""
    final_text = (
        "### 🎭 Eventos Culturais\n"
        "Aqui tens uma seleção de eventos culturais e de grande visibilidade **esta semana em Lisboa**:\n"
        "---\n"
        "### 🎵 Lxtriathlon Grand Prix\n\n"
        "- 📝 **Descrição:** Grande evento de atletismo organizado pelo Lxtriathlon.\n"
        "- 📍 **Morada:** [Complexo Desportivo Moniz Pereira, Lisboa](https://www.google.com/maps/search/?api=1&query=Complexo+Desportivo+Moniz+Pereira%2C+Lisboa)\n"
        "- 📅 **Data/Hora:** 19 Abr\n"
        "- 💰 **Preço:** Não indicado nos dados\n"
        "- 🌐 [Mais detalhes](https://www.visitlisboa.com/en/events/lxtriathlon-grand-prix)\n\n"
        "### 🎭 Sacramento Handicrafts Market\n\n"
        "- 📝 **Descrição:** 🔎 **Nota:** alguns destes eventos são recorrentes, por isso convém verificar a página oficial antes de ires.\n"
        "- 📍 **Morada:** [Largo do Carmo, 1200-092 Lisboa](https://www.google.com/maps/search/?api=1&query=Largo+do+Carmo%2C+1200-092+Lisboa)\n"
        "- 📅 **Data/Hora:** 19 Abr\n"
        "- 💰 **Preço:** Entrada gratuita\n"
        "- 🌐 [Mais detalhes](https://www.visitlisboa.com/en/events/sacramento-handicrafts-market)\n\n"
        "📌 **Fonte:** [*VisitLisboa Eventos*](https://www.visitlisboa.com/pt-pt/eventos)"
    )
    worker_text = (
        "1. 📅 **Lxtriathlon Grand Prix**\n"
        "   🗓️ **Quando:** 19 Abr\n"
        "   ⏱️ **Duração:** 🎯 Um só dia\n"
        "   📂 **Categoria:** Desporto\n"
        "   📝 **Descrição:** Grande evento de atletismo organizado pelo Lxtriathlon.\n"
        "   📍 Complexo Desportivo Moniz Pereira, Lisboa\n"
        "   💰 **Preço:** Não indicado\n"
        "   🔗 https://www.visitlisboa.com/en/events/lxtriathlon-grand-prix\n"
        "   🎟️ **Comprar bilhetes:** https://www.visitlisboa.com/en/events/lxtriathlon-grand-prix#tickets\n\n"
        "2. 📅 **Sacramento Handicrafts Market**\n"
        "   🗓️ **Quando:** 19 Abr\n"
        "   ⏱️ **Duração:** 🎯 Um só dia\n"
        "   📂 **Categoria:** Feiras\n"
        "   📝 **Descrição:** Mercado de artesanato no Largo do Carmo.\n"
        "   📍 Largo do Carmo, 1200-092 Lisboa\n"
        "   💰 **Preço:** Entrada gratuita\n"
        "   🔗 https://www.visitlisboa.com/en/events/sacramento-handicrafts-market\n\n"
        "📌 **Fonte:** [*VisitLisboa Eventos*](https://www.visitlisboa.com/pt-pt/eventos)"
    )

    output = reconcile_researcher_event_response(
        final_text,
        worker_text,
        language="pt",
        user_query="Quero explorar a cultura local. Que grandes eventos temos esta semana?",
    )

    assert "Aqui tens uma seleção de eventos culturais e de grande visibilidade **esta semana em Lisboa**:\n\n### 🏅 Lxtriathlon Grand Prix" in output
    assert "- ⏱️ **Duração:** 🎯 Um só dia" in output
    assert "- 📂 **Categoria:** Desporto" in output
    assert "- 🎟️ [Bilhetes](https://www.visitlisboa.com/en/events/lxtriathlon-grand-prix#tickets)" in output
    assert "### 🛍️ Sacramento Handicrafts Market" in output
    assert "- 📝 **Descrição:** Mercado de artesanato no Largo do Carmo." in output
    assert "convém verificar a página oficial" not in output
    assert "\n---\n### 🏅 Lxtriathlon Grand Prix" not in output
    assert "Aqui tens uma seleção de eventos culturais e de grande visibilidade" in output


def test_reconcile_researcher_event_response_prefers_explicit_exact_not_found_intro() -> None:
    """If grounded worker output says the exact event was not found, the final render should preserve that explicit intro."""
    final_text = (
        "### 🎭 Eventos Culturais\n"
        "Aqui tens os principais eventos culturais que encontrei em Lisboa:\n\n"
        "### 🎉 Lisbon Innovation Forum\n\n"
        "- 📝 **Descrição:** Major innovation conference in Lisbon.\n"
        "- 📍 **Morada:** [Lisboa](https://www.google.com/maps/search/?api=1&query=Lisboa)\n"
        "- 📅 **Data/Hora:** 15 Nov\n"
        "- 🌐 [Mais detalhes](https://www.visitlisboa.com/en/events/lisbon-innovation-forum)"
    )
    worker_text = (
        "❌ Não encontrei um evento específico com o nome **web summit** na base de dados disponível. "
        "Como alternativa, deixo abaixo eventos do mesmo tipo, estilo ou afinidade temática.\n\n"
        "- 🧾 **Resumo da pesquisa**\n"
        "    - 🧭 **Filtro aplicado:** all available dates, Main Events, theme focus: Web Summit.\n"
        "1. 📅 **Lisbon Innovation Forum**\n"
        "   🗓️ **Quando:** 15 Nov\n"
        "   📂 **Categoria:** Main Events\n"
        "   📝 **Descrição:** Major innovation conference in Lisbon.\n"
        "   📍 Lisboa\n"
        "   🔗 https://www.visitlisboa.com/en/events/lisbon-innovation-forum\n"
    )

    output = reconcile_researcher_event_response(
        final_text,
        worker_text,
        language="pt",
        user_query="Fala-me do Web Summit",
    )

    assert "Não encontrei um evento específico com o nome **web summit**" in output
    assert "Como alternativa" in output
    assert "Aqui tens os principais eventos culturais que encontrei em Lisboa" not in output


def test_final_visual_pass_linkifies_localizacao_and_splits_event_metadata_lines() -> None:
    """QA-repaired PT event cards should still get Maps links and visible metadata line breaks."""
    raw = (
        "### 🎉 Lxtriathlon Grand Prix\n\n"
        "📝 **Descrição:** Grande evento de atletismo em Lisboa.\n"
        "📍 **Localização:** Complexo Desportivo Moniz Pereira, R. João Amaral, Lisboa\n"
        "📅 **Data/Hora:** 19 de Abril\n"
        "💰 **Preço:** Não disponível nos dados\n"
        "🌐 [Mais detalhes](https://www.visitlisboa.com/en/events/lxtriathlon-grand-prix)"
    )

    output = final_visual_pass(raw)

    assert "📍 **Localização:** [Complexo Desportivo Moniz Pereira, R. João Amaral, Lisboa](https://www.google.com/maps/search/?api=1&query=Complexo+Desportivo+Moniz+Pereira%2C+R.+Jo%C3%A3o+Amaral%2C+Lisboa)" in output
    assert "\n\n📅 **Data/Hora:** 19 de Abril" in output
    assert "\n\n💰 **Preço:** Não disponível nos dados" in output


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

    assert "### 💊 Farmácias perto de Saldanha" in output
    assert "- 💊 **Farmácia Dalva**" in output
    assert "**Farm\u00e1cia Dalva**" in output
    assert "**Morada:**" in output
    # The formatter normalizes the raw "0.07 km away" into the cleaner PT form
    # "0.07 km" under a **Distância:** label. Both the numeric distance and the
    # label must survive the structured-card rewrite.
    assert "0.07 km" in output
    assert "**Dist\u00e2ncia:**" in output
    assert "Lisboa Aberta" in output


def test_researcher_worker_formats_place_cards_with_links_and_english_labels() -> None:
    """Q9-style place lookups should become canonical English cards with Maps and tel links."""
    raw = (
        "🏛️ **Found 1 Places/Attractions in Lisbon:**\n\n"
        "1. 🏛️ **D'Bacalhau | Restaurant**\n"
        "   📂 **Category**: Restaurant\n"
        "   Waterfront seafood dining by the Tagus.\n"
        "   📍 Rua da Pimenta 81, Lisboa\n"
        "   🕐 **Today**: 12:00 - 23:00\n"
        "   ⭐ **TripAdvisor**: 4.5/5 (3860 reviews)\n"
        "   📞 351967353664\n"
        "   🔗 https://www.visitlisboa.com/en/places/restaurante-d-bacalhau\n\n"
        "📌 **Source:** [*VisitLisboa Places*](https://www.visitlisboa.com/en/places) | **Updated:** 14:00"
    )

    output = finalize_worker_response(
        raw,
        agent_name="researcher",
        user_query="Best seafood restaurants near the Tagus river with a nice view and not overly touristy.",
        language="en",
    )

    assert "Here are dining spots in Lisbon that match your request:" in output
    assert "### 🏛️ D'Bacalhau" in output
    assert "📝 **Description:** Waterfront seafood dining by the Tagus." in output
    assert "📂 **Category:** Restaurant" in output
    assert "📍 **Address:** [Rua da Pimenta 81, Lisboa](https://www.google.com/maps/search/?api=1&query=Rua+da+Pimenta+81%2C+Lisboa)" in output
    assert "📞 **Phone:** [+351 967 353 664](tel:+351967353664)" in output
    assert "⭐ **Rating:** 4.5/5 (3860 reviews)" in output
    assert "🕐 **Today:** 12:00 - 23:00" in output
    assert "🌐 **Website:** [visitlisboa.com](https://www.visitlisboa.com/en/places/restaurante-d-bacalhau)" in output
    assert "**Morada:**" not in output


def test_researcher_worker_keeps_lisboa_card_and_price_fields_out_of_place_description() -> None:
    """Place cards should keep Lisboa Card benefits and ticket-offer pricing out of the description field."""
    raw = (
        "🏛️ **Found 2 Places/Attractions in Lisbon:**\n\n"
        "1. 🏛️ **Museum of Lisbon – Pimenta Palace**\n"
        "   📂 Category: Museums\n"
        "   🎫 Free with Lisboa Card\n"
        "   Discover the various lives of the city of Lisbon, from the Roman age onwards.\n"
        "   📍 Lisboa\n"
        "   🕐 **Today**: Closed\n"
        "   💰 Children Free until (age): 12 Adult: 3 € + info\n"
        "   📞 +351 217 513 200\n"
        "   🔗 https://www.visitlisboa.com/en/places/museum-of-lisbon-pimenta-palace\n\n"
        "2. 🏛️ **Arco da Rua Augusta**\n"
        "   📂 Category: Monuments\n"
        "   Climb up one of Lisbon’s iconic buildings for a unique view of the city.\n"
        "   📍 Rua Augusta, 2, 1100-053, Lisboa\n"
        "   🕐 **Today**: 10:00 - 19:00\n"
        "   🔗 https://www.visitlisboa.com/en/places/arco-da-rua-augusta\n\n"
        "📌 **Source:** [*VisitLisboa Places*](https://www.visitlisboa.com/en/places) | **Updated:** 14:00"
    )

    output = finalize_worker_response(
        raw,
        agent_name="researcher",
        user_query="Lista as atrações imperdíveis para quem visita Lisboa pela primeira vez.",
        language="pt",
    )

    assert output.startswith("### 🏛️ Atrações Imperdíveis")
    assert "### 🏛️ Museum of Lisbon – Pimenta Palace" in output
    assert "- 🎫 **Lisboa Card:**" not in output
    assert "- 📝 **Descrição:** Discover the various lives of the city of Lisbon, from the Roman age onwards." in output
    assert "Children Free until (age)" not in output
    assert "- 💰 **Preço:** Children free until age 12; Adult: 3 €; Gratuito com Lisboa Card" in output
    assert "- 📝 **Descrição:** Free with Lisboa Card" not in output
    assert "+ info" not in output


def test_researcher_worker_drops_plain_buy_ticket_placeholder_from_generic_place_lists() -> None:
    """Generic attraction lists should not leak raw `BUY:` ticket placeholders into the final UI."""
    raw = (
        "🏛️ **Found 2 Places/Attractions in Lisbon:**\n\n"
        "1. 🏛️ **Palace of Belém**\n"
        "   📂 Category: Monuments\n"
        "   Pretend you’re on a state visit as you tour the Palace of Belém.\n"
        "   📍 Praça Afonso de Albuquerque, 1349-022, Lisboa\n"
        "   🕐 **Today**: Closed\n"
        "   🎟️ BUY: https://www.visitlisboa.com/en/places/palace-of-belem#tickets\n"
        "   🔗 https://www.visitlisboa.com/en/places/palace-of-belem\n\n"
        "2. 🏛️ **Arco da Rua Augusta**\n"
        "   📂 Category: Monuments\n"
        "   Climb up one of Lisbon’s iconic buildings for a unique view of the city.\n"
        "   📍 Rua Augusta, 2, 1100-053, Lisboa\n"
        "   🕐 **Today**: 10:00 - 19:00\n"
        "   🔗 https://www.visitlisboa.com/en/places/arco-da-rua-augusta\n\n"
        "📌 **Source:** [*VisitLisboa Places*](https://www.visitlisboa.com/en/places) | **Updated:** 14:00"
    )

    output = finalize_worker_response(
        raw,
        agent_name="researcher",
        user_query="Lista as atrações imperdíveis para quem visita Lisboa pela primeira vez.",
        language="pt",
    )

    assert "BUY" not in output
    assert "#tickets" not in output
    assert "**Preço:** BUY" not in output
    assert "**Website:** [visitlisboa.com](https://www.visitlisboa.com/en/places/palace-of-belem#tickets)" not in output


def test_researcher_worker_replaces_invalid_ticket_placeholder_with_source_note() -> None:
    """Malformed or non-URL ticket placeholders must never survive as nested markdown links."""
    raw = (
        "### 🎬 Jazz at the Cinema São Jorge\n\n"
        "- 📝 **Descrição:** Ciclo de quatro concertos com propostas de jazz originais e variadas.\n"
        "- 📍 **Morada:** [Cinema São Jorge, Avenida da Liberdade, 175, 1250-141, Lisboa](https://www.google.com/maps/search/?api=1&query=Cinema+S%C3%A3o+Jorge%2C+Avenida+da+Liberdade%2C+175%2C+1250-141%2C+Lisboa)\n"
        "- 📅 **Data/Hora:** 23 a 24 de abril\n"
        "- 💰 **Preço:** Gratuito\n"
        "- 🌐 [Mais detalhes](https://www.visitlisboa.com/en/events/jazz-at-the-cinema-sao-jorge)\n"
        "- 🎟️ [Bilhetes]([Bilhetes](Bilhetes: indisponíveis))\n\n"
        "📌 **Fonte:** [*VisitLisboa Eventos*](https://www.visitlisboa.com/pt-pt/eventos)"
    )

    output = finalize_worker_response(
        raw,
        agent_name="researcher",
        user_query="Quero explorar a cultura local. Que grandes eventos temos esta semana?",
        language="pt",
    )

    assert "Bilhetes" not in output
    assert "[Bilhetes]([Bilhetes](Bilhetes: indisponíveis))" not in output
    assert "[Bilhetes](Não disponível)" not in output


def test_researcher_mixed_museum_and_event_query_skips_event_only_shortcut() -> None:
    """Mixed place+event requests should not be swallowed by the deterministic event-only shortcut."""
    with patch.object(ResearcherAgent, "__init__", lambda self: None):
        agent = ResearcherAgent()
        agent.system_prompt = "RESEARCHER PROMPT"
        agent.tools = []
        agent.llm = MagicMock()
        agent._last_search_context = None
        agent.execute_react_loop = MagicMock(return_value="Combined museums and events response.")

        with patch.object(
            ResearcherAgent,
            "_run_direct_event_lookup",
            side_effect=AssertionError("direct event shortcut should be skipped"),
        ), patch.object(
            ResearcherAgent,
            "_run_direct_place_lookup",
            side_effect=AssertionError("direct place shortcut should be skipped"),
        ):
            result = agent.invoke(
                "Quais são os museus gratuitos este fim de semana em Lisboa e algum evento interessante igualmente gratuito?"
            )

    agent.execute_react_loop.assert_called_once()
    assert "Combined museums and events response." in result


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


def test_researcher_named_service_lookup_uses_open_data_services_not_visitlisboa() -> None:
    """Named service queries should stay on Lisboa Aberta instead of VisitLisboa attractions."""
    with patch.object(ResearcherAgent, "__init__", lambda self: None):
        agent = ResearcherAgent()

        places_tool = MagicMock()
        places_tool.name = "search_places_attractions"
        places_tool.invoke = MagicMock(side_effect=AssertionError("VisitLisboa should not handle service queries"))
        nearby_tool = MagicMock()
        nearby_tool.name = "find_nearby_services"
        nearby_tool.invoke = MagicMock(
            return_value=(
                "\U0001F4CD Found 1 results from 'Hospitais':\n\n"
                "1. Hospital Santa Maria\n   \U0001F4CD Avenida Professor Egas Moniz, Lisboa"
            )
        )
        agent.tools = [places_tool, nearby_tool]

        result = agent._run_direct_place_lookup("Onde fica o Hospital Santa Maria?", "pt")

        places_tool.invoke.assert_not_called()
        nearby_tool.invoke.assert_called_once()
        assert "Hospital Santa Maria" in result
        assert "Lisboa Aberta" in result


def test_researcher_pharmacy_nearby_query_uses_lisboa_aberta_only() -> None:
    """Pharmacy-near-place queries must not return VisitLisboa attractions or restaurants."""
    with patch.object(ResearcherAgent, "__init__", lambda self: None):
        agent = ResearcherAgent()

        places_tool = MagicMock()
        places_tool.name = "search_places_attractions"
        places_tool.invoke = MagicMock(side_effect=AssertionError("VisitLisboa should not be called for pharmacy queries"))
        nearby_tool = MagicMock()
        nearby_tool.name = "find_nearby_services"
        nearby_tool.invoke = MagicMock(
            return_value=(
                "\U0001F4CD Found 1 results from 'Farmácias e Parafarmácias (near Parque das Nações)':\n\n"
                "1. Farmácia Expo\n   \U0001F4CD Alameda dos Oceanos"
            )
        )
        agent.tools = [places_tool, nearby_tool]

        result = agent._run_direct_place_lookup("Farmácia perto do Parque das Nações", "pt")

        places_tool.invoke.assert_not_called()
        nearby_tool.invoke.assert_called_once()
        assert "Farmácia Expo" in result
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
