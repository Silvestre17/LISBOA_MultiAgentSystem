# ==========================================================================
# Master Thesis
#   - Andre Filipe Gomes Silvestre, 20240502
#
# Tests for planner output quality guards.
# ==========================================================================

from types import SimpleNamespace

from agent.agents.planner_agent import (
    PlannerAgent,
    _build_deterministic_planner_fallback,
    _build_structured_plan_fallback,
    _planner_response_matches_schema,
    _build_next_day_historic_food_transport_fallback,
    _build_public_transport_synthesis_instruction,
    _build_resident_service_plan_fallback,
    _build_short_coffee_culture_fallback,
    _build_card_based_itinerary_fallback,
    _extract_visitlisboa_place_cards,
    _extract_weather_safety_bullets,
    enforce_multi_day_quality_mode,
    _is_historic_gastronomy_day_request,
    _is_next_day_planning_follow_up,
    _planner_response_has_markdown_contract_defects,
    _planner_response_has_incomplete_museum_day_blocks,
    _planner_response_has_transport_quality_defects,
)
from agent.agents.qa_agent import QualityAssuranceAgent
from agent.agents.researcher_agent import ResearcherAgent
from agent.agents.supervisor import SupervisorAgent
from agent.agents.transport_agent import _build_unsupported_transport_scope_response
from agent.agents.transport_agent import TransportAgent
from agent.graph import MultiAgentAssistant
from agent.planning.evidence import build_evidence_bundle
from agent.planning.models import PlanBlock, PlanDraft, SourceRef
from agent.planning.renderer import render_plan_markdown
from agent.utils.response_formatter import (
    canonicalize_planner_source_line,
    final_visual_pass,
    final_post_qa_guard,
    build_bounded_planning_framework,
    is_overcomplex_planning_request,
    strip_internal_repository_source_links,
)


def test_public_transport_instruction_added_when_route_context_exists() -> None:
    """Public-transport itinerary requests should receive a strict synthesis contract."""
    instruction = _build_public_transport_synthesis_instruction(
        "Plan a museum day using public transport.",
        "### Route\n- Carris line 732 from Rossio to Belem.",
    )

    assert "PUBLIC TRANSPORT SYNTHESIS CONTRACT" in instruction
    assert "board/alight" in instruction


def test_planner_qa_retry_skips_already_usable_worker_evidence() -> None:
    """Planner synthesis should not trigger a second broad worker pass."""
    qa_result = {
        "missing_data": ["exact opening hours"],
        "repairable_agents": ["researcher", "transport"],
    }

    retry_agents = MultiAgentAssistant._filter_planner_qa_retry_agents(
        ["researcher", "transport"],
        user_message="Plan a relaxed evening around Príncipe Real from Saldanha.",
        agents_to_call=["researcher", "transport", "planner"],
        workers=["researcher", "transport"],
        agent_outputs={
            "researcher": "### Cultural options\n- Museu Nacional de História Natural",
            "transport": "### Route\n- Metro Yellow Line from Saldanha to Rato",
        },
        qa_result=qa_result,
    )

    assert retry_agents == []
    assert qa_result["_skipped_planner_retry_agents"] == ["researcher", "transport"]


def test_planner_qa_retry_still_fetches_missing_domains() -> None:
    """QA may still request domains that the supervisor did not gather."""
    retry_agents = MultiAgentAssistant._filter_planner_qa_retry_agents(
        ["transport"],
        user_message="Plan a relaxed evening around Príncipe Real from Saldanha.",
        agents_to_call=["researcher", "planner"],
        workers=["researcher"],
        agent_outputs={"researcher": "### Cultural options\n- Local evidence"},
        qa_result={"missing_data": ["public transport route"]},
    )

    assert retry_agents == ["transport"]


def test_planner_qa_retry_does_not_add_weather_for_generic_evening() -> None:
    """A generic evening plan should not become weather-aware after QA."""
    retry_agents = MultiAgentAssistant._filter_planner_qa_retry_agents(
        ["weather"],
        user_message="Plan a relaxed evening around Príncipe Real starting from Saldanha.",
        agents_to_call=["transport", "researcher", "planner"],
        workers=["transport", "researcher"],
        agent_outputs={"transport": "metro route", "researcher": "local cultural stop"},
        qa_result={"missing_data": ["weather"]},
    )

    assert retry_agents == []


def test_qa_relaxes_final_markdown_contract_for_planner_worker_evidence() -> None:
    """Intermediate worker evidence should be fact-checked without UI footer retries."""
    qa_agent = QualityAssuranceAgent.__new__(QualityAssuranceAgent)

    direct_result = qa_agent._verify_facts(
        "### Route\n- **Address:** Saldanha\n- Metro Yellow Line from Saldanha to Rato",
        "Plan an evening around Principe Real using public transport.",
        language="en",
        intermediate_output=False,
    )
    intermediate_result = qa_agent._verify_facts(
        "### Route\n- **Address:** Saldanha\n- Metro Yellow Line from Saldanha to Rato",
        "Plan an evening around Principe Real using public transport.",
        language="en",
        intermediate_output=True,
    )

    assert "Source footer is missing or malformed." in direct_result["critical_issues"]
    assert "Source footer is missing or malformed." not in intermediate_result["critical_issues"]


def test_structured_planner_renderer_omits_absent_weather_section() -> None:
    """Generic plans should not show weather caveats when weather was not used."""
    draft = PlanDraft(
        title="Relaxed evening",
        direct_answer="Use the metro and keep the final walk short.",
        constraints_used=["public transport from Saldanha"],
        blocks=[
            PlanBlock(
                title="Saldanha to Príncipe Real",
                kind="transport",
                movement=["Metro from Saldanha to Avenida, then walk."],
                source_ids=["metro"],
            )
        ],
        movement_logic=["Metro first, then short walk."],
        weather_strategy=["Weather was not confirmed in the provided data."],
        source_ids=["metro", "ipma"],
    )
    rendered = render_plan_markdown(
        draft,
        sources={
            "metro": SourceRef("metro", "Metro de Lisboa", "Metro de Lisboa", "https://www.metrolisboa.pt"),
            "ipma": SourceRef("ipma", "IPMA", "IPMA", "https://www.ipma.pt"),
        },
        language="en",
    )

    assert "Weather adaptation" not in rendered
    assert "IPMA" not in rendered
    assert "Metro de Lisboa" in rendered


def test_structured_planner_renderer_drops_unused_carris_source() -> None:
    """Source footers should cite the operator materially used in the answer."""
    draft = PlanDraft(
        title="Metro evening",
        direct_answer="Use the metro from Saldanha to Avenida.",
        blocks=[
            PlanBlock(
                title="Metro leg",
                kind="transport",
                movement=["Yellow line from Saldanha, transfer to the Blue line, alight at Avenida."],
                limitations=["Carris line numbers and schedules should be confirmed only if using bus alternatives."],
                source_ids=["metro", "carris"],
            )
        ],
        movement_logic=["Metro is the clearest supported route."],
    )
    rendered = render_plan_markdown(
        draft,
        sources={
            "metro": SourceRef("metro", "Metro de Lisboa", "Metro de Lisboa", "https://www.metrolisboa.pt"),
            "carris": SourceRef("carris", "Carris", "Carris", "https://www.carris.pt"),
        },
        language="en",
    )

    assert "Metro de Lisboa" in rendered
    assert "Carris" not in rendered.split("📌 **Source:**", 1)[-1]


def test_structured_planner_accepts_route_blocks_and_renders_place_fields() -> None:
    """Planner JSON aliases should render rich fields instead of falling back."""
    draft = PlanDraft.from_dict(
        {
            "title": "Relaxed evening around Príncipe Real",
            "direct_answer": "Use metro first, then keep the cultural stop local.",
            "route_blocks": [
                {
                    "title": "Water Museum - Patriarchal Reservoir",
                    "kind": "museum",
                    "purpose": "Compact cultural stop in Príncipe Real.",
                    "details": [
                        "Description: Underground reservoir completed in 1864.",
                        "Address: [Praça do Príncipe Real](https://www.google.com/maps/search/?api=1&query=Pra%C3%A7a+do+Pr%C3%ADncipe+Real)",
                        "Website: [Official website](https://www.visitlisboa.com/en/places/water-museum-patriarchal-reservoir)",
                    ],
                    "movement": ["Metro from Saldanha to Avenida, then short walk."],
                    "source_ids": ["visitlisboa_places", "metro"],
                }
            ],
            "movement": ["Yellow line from Saldanha, transfer at Marquês de Pombal, exit at Avenida."],
        }
    )
    rendered = render_plan_markdown(
        draft,
        sources={
            "visitlisboa_places": SourceRef("visitlisboa_places", "VisitLisboa Places", "VisitLisboa Locais", "https://www.visitlisboa.com/en/places"),
            "metro": SourceRef("metro", "Metro de Lisboa", "Metro de Lisboa", "https://www.metrolisboa.pt"),
        },
        language="en",
    )

    assert "📍 **Address:** [Praça do Príncipe Real]" in rendered
    assert "🌐 **Website:** [Official website]" in rendered
    assert "### 🚇 **How to move**" in rendered


def test_planner_evidence_preserves_plain_visitlisboa_fields() -> None:
    """Research cards should carry worker field data into planner evidence."""
    bundle = build_evidence_bundle(
        places_data=(
            "### 🏛️ **Water Museum - Patriarchal Reservoir**\n"
            "    - 📝 **Description:** Historic reservoir near Príncipe Real.\n"
            "    - 📍 **Address:** Praça do Príncipe Real, Lisboa\n"
            "    - 🕐 Today: 10:00-17:30\n"
            "    - 💰 Price: €4\n"
            "    - 🔗 https://www.visitlisboa.com/en/places/water-museum-patriarchal-reservoir\n"
            "\n"
            "📌 **Source:** [*VisitLisboa Places*](https://www.visitlisboa.com/en/places)\n"
        )
    )

    assert bundle.cards
    fields = bundle.cards[0].fields
    assert fields["Description"] == "Historic reservoir near Príncipe Real."
    assert fields["Address"] == "Praça do Príncipe Real, Lisboa"
    assert fields["Hours"] == "10:00-17:30"
    assert fields["Price"] == "€4"
    assert fields["Website"].startswith("https://www.visitlisboa.com")


def test_card_based_planner_fallback_uses_renderer_contract_for_evening_plan() -> None:
    """Card fallback should not publish orphan headings or morning blocks."""
    response = _build_card_based_itinerary_fallback(
        user_message=(
            "Plan a relaxed evening around Príncipe Real starting from Saldanha, "
            "with one cultural stop and realistic public transport."
        ),
        language="en",
        weather_data="",
        transport_data=(
            "### 🚇 Main move: Metro\n"
            "- 🔹 **Best Metro Option**: Saldanha → Marquês de Pombal → Avenida\n"
            "- 🔹 **Estimated Total Travel Time**: ~11 min\n"
            "- Carris line numbers and schedules should be confirmed at carris.pt.\n"
            "📌 **Source:** [*Metro de Lisboa*](https://www.metrolisboa.pt)\n"
        ),
        places_data=(
            "### 🏛️ **Water Museum - Patriarchal Reservoir**\n"
            "    - 📝 **Description:** Underground reservoir hidden beneath the garden of Príncipe Real.\n"
            "    - 📂 Category: Museums & Monuments\n"
            "    - 📍 **Address:** Praça do Príncipe Real, Lisboa\n"
            "    - 🕐 Today: 10:00-17:30\n"
            "    - 💰 Price: €4\n"
            "    - 🔗 https://www.visitlisboa.com/en/places/water-museum-patriarchal-reservoir\n"
            "\n"
            "### 🏛️ **Galeria Subterrânea do Loreto**\n"
            "    - 📝 **Description:** Historic underground water gallery.\n"
            "    - 📂 Category: Museums & Monuments\n"
            "    - 📍 **Address:** Lisboa\n"
            "📌 **Source:** [*VisitLisboa Places*](https://www.visitlisboa.com/en/places)\n"
        ),
        events_data="",
        qa_disclaimers=None,
    )

    assert "✅ **Direct answer:**" in response
    assert "09h30" not in response
    assert response.count("Water Museum - Patriarchal Reservoir") == 1
    assert "Galeria Subterrânea do Loreto" not in response
    assert "\n### 🚇 **How to move**" in response
    assert "\n### 🚇 Main move" not in response
    assert "🏷️ **Category:** Museums & Monuments" in response
    assert "📍 **Address:** [Praça do Príncipe Real, Lisboa]" in response
    assert "💶 **Price:** €4" in response
    assert "🌐 **Website:** [VisitLisboa]" in response


def _build_planner_with_fake_llm(response_text: str) -> PlannerAgent:
    """Build a PlannerAgent shell that returns a deterministic fake LLM response."""
    planner = PlannerAgent.__new__(PlannerAgent)
    planner.system_prompt = "TEST PLANNER PROMPT"
    planner._system_prompt_dynamic = False
    planner.llm = object()
    planner._safe_llm_invoke = lambda _llm, _messages: SimpleNamespace(content=response_text)
    return planner


def test_planner_invoke_uses_llm_before_multiday_fallback() -> None:
    """Normal multi-day requests should not bypass dynamic synthesis with templates."""
    planner = _build_planner_with_fake_llm(
        """### 📅 Custom Dynamic 3-Day Plan

### ✅ Direct Answer
Use a grounded 3-day sequence.

### 🧭 Constraints Used
- Public transport and rain backups.

### 📍 Plan Blocks
- **Day 1 · Alfama:** Use the grounded first stop.
- **Day 2 · Belém:** Use the grounded second stop.

### 🚇 Movement Logic
Use Metro de Lisboa and Carris where grounded.

### ⛅ Weather Strategy
Keep indoor backups.

### ⚠️ Limitations
Exact prices and opening hours are not confirmed."""
    )

    response = planner.invoke(
        "Plan 3 days in Lisbon with public transport and rain backups.",
        weather_data="No active weather warnings for Lisbon.",
        transport_data="Metro de Lisboa and Carris data available.",
        places_data="""### 🏛️ Grounded stop
- Belém Tower
- Alfama viewpoint""",
    )

    assert "Custom Dynamic 3-Day Plan" in response
    assert "5-Day Lisbon Itinerary" not in response
    assert "planning framework" not in response.lower()


def test_planner_invoke_uses_llm_for_conversation_follow_up() -> None:
    """Follow-up planning should preserve dynamic synthesis and conversation context."""
    planner = _build_planner_with_fake_llm(
        """### 📅 Dynamic Follow-Up Plan

### ✅ Direct Answer
Use Estrela and Campo de Ourique to avoid repeating the previous route.

### 🧭 Constraints Used
- Previous route exclusions and transport.

### 📍 Plan Blocks
- **Estrela:** Keep the plan compact and rain-safe.
- **Campo de Ourique:** Add a second nearby anchor.

### 🚇 Movement Logic
Use Metro de Lisboa and Carris where grounded.

### ⛅ Weather Strategy
Prioritise indoor backup due light rain.

### ⚠️ Limitations
Exact live departures are not confirmed."""
    )

    response = planner.invoke(
        "PLANEIA O DIA SEGUINTE E DIZ-ME TRANSPORTES",
        weather_data="Chuva fraca amanhã.",
        transport_data="Metro de Lisboa e Carris disponíveis.",
        places_data="""### 🏛️ Estrela
- Jardim da Estrela
- Basílica da Estrela""",
        conversation_context="Plano anterior: Baixa, Sé, Carmo e Belém.",
    )

    assert "Dynamic Follow-Up Plan" in response
    assert "Estrela" in response


def test_planner_rejects_vague_transport_when_context_exists() -> None:
    """Planner drafts must not hide available transport evidence behind vague prose."""
    response = (
        "### Suggested Itinerary\n"
        "- Use public transport from the Gulbenkian area back toward central Lisbon.\n"
        "- Verify locally for the most direct bus or metro connection."
    )

    assert _planner_response_has_transport_quality_defects(
        response,
        "Plan a full museum day using public transport.",
        "Carris line 732 and Metro Blue Line route details are available.",
    )


def test_planner_allows_specific_transport_details() -> None:
    """Specific grounded route details should not be rejected by the transport guard."""
    response = (
        "### Suggested Itinerary\n"
        "- Metro: Rossio to Baixa-Chiado, transfer to the Blue Line toward Sao Sebastiao.\n"
        "- Carris line 732 toward Caselas from Rossio when moving west."
    )

    assert not _planner_response_has_transport_quality_defects(
        response,
        "Plan a full museum day using public transport.",
        "Metro and Carris route details are available.",
    )


def test_planner_rejects_broken_https_field() -> None:
    """Broken map-link fields should be blocked before Streamlit rendering."""
    response = "- Location: **https**: //www.google.com/maps/search/?api=1&query=Lisbon"

    assert _planner_response_has_markdown_contract_defects(response)


def test_oriente_evening_food_culture_fallback_is_actionable() -> None:
    """Dinner+culture fallbacks around Oriente should not become generic non-answers."""
    response = _build_deterministic_planner_fallback(
        user_message=(
            "I arrive at Oriente around 18:00 and want dinner plus one "
            "cultural stop, avoiding rain if needed."
        ),
        language="en",
        weather_data="✅ No active weather warnings for Lisbon.\n💧 Rain: Very unlikely (12.0%)",
        transport_data="",
        places_data="VisitLisboa result: Centro Vasco da Gama and Parque das Nações restaurants.",
        events_data="",
        qa_disclaimers=None,
    )

    assert "Rain-Safe Evening From Oriente" in response
    assert "Oriente Station" in response
    assert "Centro Vasco da Gama" in response
    assert "gathered data did not confirm" not in response


def test_generic_evening_food_culture_fallback_keeps_tonight_limits_visible() -> None:
    """Dinner+culture plans should not imply tonight availability without evidence."""
    response = _build_deterministic_planner_fallback(
        user_message="Plan dinner and one cultural stop near Santos tonight.",
        language="en",
        weather_data=(
            "Yes — tonight near Santos should be comfortable for dinner and a cultural stop.\n"
            "Tonight’s weather in Lisbon\n"
            "Temperature: 11.7°C to 20.8°C\n"
            "Conditions: Sunny intervals"
        ),
        transport_data="Could not resolve cultural stop near Santos for routing.",
        places_data="VisitLisboa Places: Doca de Santo. Museu Nacional de Arte Antiga.",
        events_data="",
        qa_disclaimers=None,
    )

    assert "Suggested Evening Plan" in response
    assert "Doca de Santo" in response
    assert "Museu Nacional de Arte Antiga" in response
    assert "Confirm tonight's opening" in response
    assert "Santos tonight" not in response
    assert "Santos / Cais do Sodré / Santos" not in response
    assert "weather context" not in response.lower()
    assert "weather in lisbon" not in response.lower()
    assert "should be comfortable" not in response.lower()
    assert "[*VisitLisboa Places*]" in response
    assert "[*Carris*]" not in response
    assert "[*CP*]" not in response


def test_generic_evening_food_culture_pt_filters_weather_intro_lines() -> None:
    """Portuguese evening plans should keep weather facts, not WeatherAgent prose."""
    response = _build_deterministic_planner_fallback(
        user_message="Planeia jantar e uma paragem cultural perto de Santos esta noite.",
        language="pt",
        weather_data=(
            "Esta noite em Santos dá para fazer um jantar confortável.\n"
            "Estado do tempo para esta noite em Lisboa\n"
            "Temperatura: 11.7°C a 20.8°C\n"
            "Condições: Intervalos de sol"
        ),
        transport_data="",
        places_data="VisitLisboa Locais: Doca de Santo. Museu Nacional de Arte Antiga.",
        events_data="",
        qa_disclaimers=None,
    )

    assert "dá para" not in response.lower()
    assert "estado do tempo" not in response.lower()
    assert "**Temperatura:** 11.7°C a 20.8°C" in response
    assert "[*VisitLisboa Locais*]" in response


def test_planner_scope_source_rebuild_keeps_ipma_for_weather_advice() -> None:
    """Planner fallback source rebuild must not drop IPMA when weather advice remains."""
    response = (
        "### 📅 One-Day History And Traditional Food Itinerary\n\n"
        "### ⛅ Weather and pacing\n"
        "- With sunny intervals and moderate wind, use a light layer.\n\n"
        "📌 **Source:** [*VisitLisboa Places*](https://www.visitlisboa.com/en/places) | **Updated:** 09:00"
    )

    rebuilt = MultiAgentAssistant._rebuild_planner_scope_fallback_source_line(
        response,
        language="en",
        effective_agents=["weather", "researcher", "planner"],
    )

    assert "[*IPMA*]" in rebuilt
    assert "[*VisitLisboa Places*]" in rebuilt


def test_planner_scope_source_rebuild_keeps_lisboa_aberta_for_market_anchor() -> None:
    """Planner source consolidation should cite Lisboa Aberta for municipal market anchors."""
    response = (
        "### 📅 Short Coffee And Culture Plan\n\n"
        "### ☕ Coffee\n"
        "- **Candidate from gathered data:** Mercado de Campo de Ourique\n\n"
        "### 🏛️ Cultural stop\n"
        "- **Candidate from gathered data:** Casa Fernando Pessoa\n\n"
        "📌 **Source:** [*VisitLisboa Places*](https://www.visitlisboa.com/en/places) | **Updated:** 09:00"
    )

    rebuilt = MultiAgentAssistant._rebuild_planner_scope_fallback_source_line(
        response,
        language="en",
        effective_agents=["weather", "researcher", "planner"],
    )

    assert "[*Lisboa Aberta*]" in rebuilt
    assert "[*VisitLisboa Places*]" in rebuilt


def test_short_coffee_culture_fallback_does_not_publish_closed_stop() -> None:
    """Short coffee+culture fallbacks should flag closed venues instead of recommending them."""
    response = _build_deterministic_planner_fallback(
        user_message="Plan a quiet 90-minute morning in Campo de Ourique with coffee and one cultural stop.",
        language="en",
        weather_data="🌤️ Weather Forecast for Lisbon\n💧 Rain: 55% - Weak",
        transport_data="",
        places_data="**Casa Fernando Pessoa**\n🕐 Today: Closed\n**Clueless Wines**",
        events_data="",
        qa_disclaimers=None,
    )

    assert "Short Coffee And Culture Plan" in response
    assert "marked closed today" in response
    assert "do not use it as the main stop" in response


def test_weather_safety_bullets_preserve_ipma_facts() -> None:
    """Planner fallbacks should not replace available IPMA facts with generic advice."""
    bullets = _extract_weather_safety_bullets(
        "📅 Tomorrow\n"
        "   🌡️ 10.8°C to 19.6°C\n"
        "   🌤️ Light showers/rain\n"
        "   💧 Rain: Very likely (100.0%) | Intensity: Weak\n"
        "   💨 Wind: South (Moderate)",
        "en",
    )

    joined = "\n".join(bullets)

    assert "Temperature" in joined
    assert "10.8°C to 19.6°C" in joined
    assert "Check the latest IPMA forecast" not in joined


def test_short_coffee_culture_fallback_preserves_place_cards() -> None:
    """Short planner fallback should keep concrete place candidates gathered by Researcher."""
    places_data = """
1. 🏛️ **Clueless Wines**
   📂 Category: Restaurants & Cafes
   At Clueless Wines, a welcoming wine tasting studio nestled in Campo de Ourique.
   📍 **Address:** [Rua Tenente Ferreira Durão, 62 B, 1350-318, Lisboa](https://www.google.com/maps/search/?api=1&query=Rua+Tenente)
   🔗 https://www.visitlisboa.com/en/places/clueless-wines

2. 🏛️ **Casa Fernando Pessoa**
   📂 Category: Museums
   Discover the home and work of Fernando Pessoa.
   📍 **Address:** [Rua Coelho da Rocha, 16/18, Campo de Ourique, 1250-088, Lisboa](https://www.google.com/maps/search/?api=1&query=Rua+Coelho)
   🕐 Today: Closed
   🔗 https://www.visitlisboa.com/en/places/casa-fernando-pessoa
"""
    response = _build_short_coffee_culture_fallback(
        user_message="Plan a quiet 90-minute morning in Campo de Ourique with coffee and one cultural stop.",
        language="en",
        weather_data="🌡️ 10.8°C to 19.6°C\n💧 Rain: Very likely (100.0%) | Intensity: Weak",
        places_data=places_data,
        events_data="",
    )

    assert "Clueless Wines" in response
    assert "Casa Fernando Pessoa" in response
    assert "Rua Coelho da Rocha" in response
    assert "marked closed today" in response
    assert "Check the latest IPMA forecast" not in response
    assert "[*VisitLisboa Places*]" in response


def test_short_coffee_culture_fallback_uses_known_local_anchor_when_retrieval_is_thin() -> None:
    """Common neighborhood plans should not collapse into vague no-venue prose."""
    response = _build_short_coffee_culture_fallback(
        user_message="Plan a quiet 90-minute morning in Campo de Ourique with coffee and one cultural stop.",
        language="en",
        weather_data="🌡️ 11.5°C to 20.3°C\n☁️ **Conditions**: Sunny intervals",
        places_data="VisitLisboa Places: gathered search returned no exact local card.",
        events_data="",
    )

    assert "Mercado de Campo de Ourique" in response
    assert "Casa Fernando Pessoa" in response
    assert "Rua Coelho da Rocha" in response
    assert "No specific cultural venue" not in response
    assert "[*VisitLisboa Places*]" in response
    assert "[*Lisboa Aberta*]" in response


def test_short_coffee_culture_visual_pass_removes_leading_separator() -> None:
    """Short planner cleanup must not infer extra sources from body keywords."""
    response = final_visual_pass(
        "### 📅 Short Coffee And Culture Plan\n\n"
        "---\n\n"
        "### ⛅ Conditions\n"
        "- 🌡️ **Temperature:** 11.5°C to 20.3°C.\n\n"
        "---\n\n"
        "### ☕ Coffee\n"
        "- **Candidate from gathered data:** Mercado de Campo de Ourique\n\n"
        "---\n\n"
        "### 🏛️ Cultural stop\n"
        "- **Candidate from gathered data:** Casa Fernando Pessoa\n\n"
        "📌 **Source:** [*VisitLisboa Places*](https://www.visitlisboa.com/en/places) | **Updated:** 09:00"
    )

    assert "Short Coffee And Culture Plan\n\n---\n\n### ⛅" not in response
    assert "[*Lisboa Aberta*]" not in response


def test_multi_day_fallback_builds_bounded_itinerary() -> None:
    """Multi-day requests should return a visitable itinerary, not Day-1-only sprawl."""
    response = _build_deterministic_planner_fallback(
        user_message="Plan 5 days in Lisbon with public transport and indoor backups.",
        language="en",
        weather_data="✅ No active weather warnings for Lisbon.",
        transport_data="",
        places_data="",
        events_data="",
        qa_disclaimers=None,
    )

    assert "5-Day Lisbon Itinerary" in response
    assert "Day 5" in response
    assert "Planning Framework" not in response
    assert "Morning" in response
    assert "Afternoon" in response
    assert "Rain backup" in response
    assert "future real-time departures" in response


def test_multi_day_fallback_limits_above_five_days() -> None:
    """Very long trips should be capped to five days with an explicit limitation."""
    response = _build_deterministic_planner_fallback(
        user_message="Plan 7 days in Lisbon with public transport and indoor backups.",
        language="en",
        weather_data="",
        transport_data="",
        places_data="",
        events_data="",
        qa_disclaimers=None,
    )

    assert "First 5 Days in Lisbon" in response
    assert "request covers **7 days**" in response
    assert "Day 5" in response
    assert "Day 6" not in response


def test_overcomplex_regional_fallback_refuses_unsupported_live_details() -> None:
    """Regional booking/ferry/live-price requests should fail bounded, not hallucinate."""
    response = _build_deterministic_planner_fallback(
        user_message=(
            "Can you plan Lisbon, Sintra, Cascais and Setúbal next Saturday "
            "with live transport times, ferry options, ticket prices, and restaurant bookings?"
        ),
        language="en",
        weather_data="",
        transport_data="",
        places_data="",
        events_data="",
        qa_disclaimers=None,
    )

    assert "Request Too Broad" in response
    assert "What I will not invent" in response
    assert "Transtejo/Soflusa ferry times" in response


def test_low_walk_day_fallback_avoids_raw_place_card_dump() -> None:
    """Low-walk rain-backup day plans should stay compact and visually coherent."""
    response = _build_deterministic_planner_fallback(
        user_message=(
            "Plan one relaxed day in Belem tomorrow starting from Rossio, "
            "avoiding long walks and keeping an indoor backup if it rains."
        ),
        language="en",
        weather_data="No active weather warnings for Lisbon.",
        transport_data="Carris direct bus option from Rossio to Belem.",
        places_data="National Coach Museum; MAAT; Address: Av. Brasilia; Phone: +351",
        events_data="",
        qa_disclaimers=None,
    )

    assert "Low-Walk Day Plan" in response
    assert "National Coach Museum" in response
    assert "MAAT" in response
    assert "**Address**" not in response
    assert "**Phone**" not in response


def test_single_day_museum_garden_fallback_avoids_service_dump() -> None:
    """Museum+garden day plans should not publish unrelated nearby-service cards."""
    response = _build_deterministic_planner_fallback(
        user_message=(
            "Plan a single relaxed day in Lisbon from Saldanha with one museum, "
            "one garden, realistic public transport, and a rain backup."
        ),
        language="en",
        weather_data="Rain: 55% - Weak",
        transport_data="Metro from Saldanha to Sao Sebastiao.",
        places_data="Jardins - Parques Urbanos: Jardim do Japão; Jardim Vasco da Gama",
        events_data="",
        qa_disclaimers=None,
    )

    assert "Relaxed One-Day Plan" in response
    assert "Museu Calouste Gulbenkian" in response
    assert "Gulbenkian Garden" in response
    assert "Dataset:" not in response
    assert "Results:" not in response


def test_historic_gastronomy_day_fallback_keeps_weather_detail_and_minimal_sources() -> None:
    """History+food day plans need rich structure without inherited broad transport footers."""
    response = _build_deterministic_planner_fallback(
        user_message="Cria um roteiro otimizado de 1 dia com monumentos históricos e gastronomia tradicional.",
        language="pt",
        weather_data=(
            "Claro — aqui tens um roteiro de 1 dia com base nas condições meteorológicas de hoje.\n"
            "Previsão para Lisboa\n"
            "Temperatura: 11.3°C a 20.3°C\n"
            "Chuva: 55% - fraca\n"
            "Vento: norte moderado"
        ),
        transport_data="Carris and CP public-transport options toward Belem are available for same-day confirmation.",
        places_data="VisitLisboa Locais: monumentos historicos e restaurantes tradicionais.",
        events_data="",
        qa_disclaimers=None,
    )

    assert "Roteiro histórico e gastronómico" in response
    assert "**Temperatura:** 11.3°C a 20.3°C" in response
    assert "**Chuva:** 55% - fraca" in response
    assert "Claro" not in response
    assert "Granja Velha" in response
    assert "Como chegar e deslocação" in response
    assert "Mosteiro dos Jerónimos" not in response.split("### ⛅ Condições meteorológicas", 1)[1].split("---", 1)[0]
    assert "não foram fornecidos detalhes climáticos" not in response.lower()
    assert "[*IPMA*]" in response
    assert "[*VisitLisboa Locais*]" in response
    assert "[*Carris*]" in response
    assert "[*CP*]" in response
    assert "[*Metro de Lisboa*]" not in response


def test_weather_extraction_does_not_treat_generic_tempo_as_forecast() -> None:
    """Planner weather blocks must not absorb itinerary rows containing generic 'tempo'."""
    response = _build_deterministic_planner_fallback(
        user_message="Cria um roteiro otimizado de 1 dia com monumentos históricos e gastronomia tradicional.",
        language="pt",
        weather_data=(
            "10:45 — Mosteiro dos Jerónimos: reserva algum tempo para visitar com calma.\n"
            "Temperatura: 11.3°C a 20.3°C\n"
            "Chuva: 55% - fraca"
        ),
        transport_data="Carris option toward Belém.",
        places_data="VisitLisboa Locais: monumentos historicos e restaurantes tradicionais.",
        events_data="",
        qa_disclaimers=None,
    )

    weather_block = response.split("### ⛅ Condições meteorológicas", 1)[1].split("---", 1)[0]

    assert "Mosteiro dos Jerónimos" not in weather_block
    assert "**Temperatura:** 11.3°C a 20.3°C" in weather_block
    assert "**Chuva:** 55% - fraca" in weather_block


def test_next_day_planning_follow_up_keeps_preferences_but_changes_route() -> None:
    """A 'next day' continuation should become a new plan, not a generic transport dump."""
    context = (
        "Previous planning request: Cria um roteiro otimizado de 1 dia com monumentos históricos "
        "e gastronomia tradicional.\n"
        "Previous final plan excerpt: Baixa, Sé, Carmo, Belém, Mosteiro dos Jerónimos, "
        "Padrão dos Descobrimentos, Torre de Belém."
    )

    assert _is_next_day_planning_follow_up("PLANEIA O DIA SEGUINTE E DIZ-ME TRANSPORTES", context)

    response = _build_next_day_historic_food_transport_fallback(
        language="pt",
        weather_data="Chuva: 100% - fraca\nTemperatura: 10.8°C a 19.6°C",
        transport_data="Metro circulation normal. Carris urban network active.",
        conversation_context=context,
    )

    assert "Dia Seguinte" in response
    assert "Campo de Ourique" in response
    assert "Ajuda" in response
    assert "Baixa" in response
    assert "evitando repetir" in response.lower()
    assert "Transportes públicos\n- Metro\n-" not in response
    assert "[*Metro de Lisboa*]" in response
    assert "[*Carris*]" in response


def test_historic_gastronomy_day_detector_requires_explicit_day_scope() -> None:
    """The history+food fallback must not steal ordinary dinner/culture planning."""
    assert _is_historic_gastronomy_day_request(
        "cria um roteiro otimizado de 1 dia com monumentos historicos e gastronomia tradicional"
    )
    assert not _is_historic_gastronomy_day_request(
        "plan dinner and one cultural stop near santos"
    )


def test_historic_gastronomy_day_fallback_omits_transport_operators_without_transport_context() -> None:
    """Operator guidance and footers should match whether transport evidence exists."""
    response = _build_deterministic_planner_fallback(
        user_message="Cria um roteiro otimizado de 1 dia com monumentos históricos e gastronomia tradicional.",
        language="pt",
        weather_data="Temperatura: 12°C a 20°C\nChuva: 20% - fraca",
        transport_data="",
        places_data="VisitLisboa Locais: monumentos historicos e restaurantes tradicionais.",
        events_data="",
        qa_disclaimers=None,
    )

    assert "Carris" not in response
    assert "CP" not in response
    assert "[*Carris*]" not in response
    assert "[*CP*]" not in response
    assert "[*VisitLisboa Locais*]" in response


def test_single_day_plans_do_not_use_direct_weather_transport_override() -> None:
    """Single-day planner prompts with weather and transport terms still need planner routing."""
    assert not SupervisorAgent._is_direct_weather_transport_query(
        "Plan a single relaxed day in Lisbon from Saldanha with one museum, one garden, "
        "realistic public transport, and a rain backup."
    )
    assert SupervisorAgent._is_direct_weather_transport_query(
        "Will it rain and how do I get from Baixa to Belem by public transport?"
    )


def test_researcher_extracts_start_location_for_resident_service_plans() -> None:
    """Resident multi-service plans should search municipal services near the origin."""
    query = (
        "Tomorrow I need a resident-oriented plan: start in Areeiro, find a recycling point, "
        "a pharmacy if needed late, and get to a quiet dinner in Alvalade if it rains."
    )
    variant = (
        "Tomorrow from Roma-Areeiro I need to drop recycling, keep a late pharmacy backup, "
        "and finish with a calm indoor dinner near Alvalade if the weather is bad."
    )

    assert ResearcherAgent._extract_near_location_name(query) == "Areeiro"
    assert ResearcherAgent._extract_near_location_name(variant) == "Roma-Areeiro"


def test_researcher_routes_local_culture_event_discovery_as_events_not_history() -> None:
    """Event discovery phrased as local culture should not fall into web history fallback."""
    query = "Quero explorar a cultura local. Que grandes eventos temos esta semana?"

    assert ResearcherAgent._is_direct_event_lookup_query(query)
    assert not ResearcherAgent._is_history_culture_query(query)
    assert not ResearcherAgent._is_mixed_event_place_query(query)


def test_resident_service_fallback_uses_concrete_service_cards() -> None:
    """Planner fallback should answer with the actual municipal service cards when available."""
    places_data = """
### 📍 **Nearest services**

- ♻️ **Service:** Recycling point Areeiro (0.18 km)
- 💊 **Pharmacy:** Farmácia Avenida de Roma (0.42 km)

---

### ♻️ **Recycling points near Areeiro**

- ✅ **Nearest:** Recycling point Areeiro (0.18 km from Areeiro)

- ♻️ **Recycling point Areeiro**
    - 📍 **Address:** [Rua João Villaret, Lisboa](https://www.google.com/maps/search/?api=1&query=Rua+Joao+Villaret)
    - 📏 **Distance:** 0.18 km

### 💊 **Pharmacies near Areeiro**

- ✅ **Nearest:** Farmácia Avenida de Roma (0.42 km from Areeiro)

- 💊 **Farmácia Avenida de Roma**
    - 📍 **Address:** [Avenida de Roma, Lisboa](https://www.google.com/maps/search/?api=1&query=Avenida+de+Roma)
    - 📏 **Distance:** 0.42 km
"""
    response = _build_resident_service_plan_fallback(
        "Tomorrow I need a resident-oriented plan: start in Areeiro, find a recycling point, "
        "a pharmacy if needed late, and get to a quiet dinner in Alvalade if it rains.",
        "en",
        "No active weather warnings for Lisbon.",
        places_data,
        "Metro de Lisboa Green Line available between Areeiro and Alvalade.",
    )

    assert "Resident Plan: Areeiro → Alvalade" in response
    assert "Recycling point Areeiro" in response
    assert "Farmácia Avenida de Roma" in response
    assert "Rua João Villaret" in response
    assert "Avenida de Roma" in response
    assert "Metro Green Line" in response
    assert "duty-pharmacy status" in response
    assert "github.com/Silvestre17" not in response
    assert "[*LISBOA*]" not in response


def test_resident_service_fallback_remains_honest_when_datasets_fail() -> None:
    """Unavailable municipal datasets should be visible without collapsing into a vague non-answer."""
    response = _build_resident_service_plan_fallback(
        "Tomorrow I need a resident-oriented plan: start in Areeiro, find a recycling point, "
        "a pharmacy if needed late, and get to a quiet dinner in Alvalade if it rains.",
        "en",
        "Rain probability: 100% - Weak.",
        "❌ Could not load usable data for 'ecopontos'.\n\n❌ Could not load usable data for 'farmácias'.",
        "Metro de Lisboa Green Line available between Areeiro and Alvalade.",
    )

    assert "Resident Plan: Areeiro → Alvalade" in response
    assert "Recycling near Areeiro" in response
    assert "Pharmacy near Areeiro" in response
    assert "Could not load usable data" in response
    assert "use the nearest municipal recycling point only after confirming" in response
    assert "Metro Green Line" in response


def test_resident_service_fallback_explains_generic_recycling_names() -> None:
    """Generic municipal recycling labels should be made user-facing and honest."""
    places_data = """
### ♻️ **Recycling points near Areeiro**
- ✅ **Nearest:** Recycling point 1 (0.97 km from Areeiro)
- ♻️ **Recycling point 1**
    - 📏 **Distance:** 0.97 km

### 💊 **Pharmacies near Areeiro**
- ✅ **Nearest:** Farmácia Garantia (0.09 km from Areeiro)
- 💊 **Farmácia Garantia**
    - 📍 **Address:** [Avenida Almirante Reis, Lisboa](https://www.google.com/maps/search/?api=1&query=Avenida+Almirante+Reis)
    - 📏 **Distance:** 0.09 km
"""
    response = _build_resident_service_plan_fallback(
        "Tomorrow I need a resident-oriented plan: start in Areeiro, find a recycling point, "
        "a pharmacy if needed late, and get to a quiet dinner in Alvalade if it rains.",
        "en",
        "Light showers/rain.",
        places_data,
        "Metro de Lisboa Green Line available between Areeiro and Alvalade.",
    )

    assert "nearest municipal recycling point returned by Lisboa Aberta" in response
    assert "Location detail" in response
    assert "0.97 km" in response
    assert "Farmácia Garantia" in response


def test_resident_plan_footer_keeps_all_material_sources() -> None:
    """Final cleanup must preserve explicit sources without adding inferred ones."""
    response = final_visual_pass(
        "### 📅 Resident Plan: Areeiro → Alvalade\n\n"
        "### ⛅ Tomorrow Conditions\n"
        "- No active weather warnings for Lisbon.\n\n"
        "### ♻️ Recycling near Areeiro\n"
        "- **Recommended stop:** Recycling point 1\n\n"
        "### 🚇 Movement and Dinner\n"
        "- **Areeiro → Alvalade:** use the **Metro Green Line** between Areeiro/Roma and Alvalade.\n\n"
        "📌 **Source:** [*Lisboa Aberta*](https://dados.cm-lisboa.pt/) | **Updated:** 12:00"
    )

    assert "[*IPMA*]" not in response
    assert "[*Lisboa Aberta*]" in response
    assert "[*Metro de Lisboa*]" not in response
    assert "github.com/Silvestre17" not in response


def test_internal_repository_source_link_is_removed_from_mixed_footer() -> None:
    """Final source footers must not cite the implementation repository."""
    response = (
        "Answer first.\n\n"
        "📌 **Source:** [*IPMA*](https://www.ipma.pt) | "
        "[*LISBOA*](https://github.com/Silvestre17/LISBOA_MultiAgentSystem) | "
        "**Updated:** 18:20"
    )

    cleaned = strip_internal_repository_source_links(response)

    assert "github.com/Silvestre17" not in cleaned
    assert "[*LISBOA*]" not in cleaned
    assert "[*IPMA*](https://www.ipma.pt)" in cleaned


def test_internal_repository_only_source_footer_is_removed() -> None:
    """A footer with only internal implementation context should be dropped."""
    response = (
        "Answer first.\n\n"
        "📌 **Source:** [*LISBOA*](https://github.com/Silvestre17/LISBOA_MultiAgentSystem) | "
        "**Updated:** 18:20"
    )

    cleaned = strip_internal_repository_source_links(response)

    assert cleaned == "Answer first."


def test_final_visual_pass_does_not_inject_cp_source_from_text_cues() -> None:
    """Source footers must come from tool provenance, not route words in prose."""
    response = (
        "Take the train from Cais do Sodré to Belém if you confirm the timetable.\n\n"
        "📌 **Source:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | **Updated:** 12:00"
    )

    cleaned = final_visual_pass(response)

    assert "[*CP*]" not in cleaned
    assert "[*Metro de Lisboa*](https://www.metrolisboa.pt)" in cleaned


def test_final_visual_pass_keeps_public_service_source_municipal_only() -> None:
    """Municipal services such as recycling must not gain tourism source footers."""
    response = (
        "### ♻️ Ecopontos perto de Avenida da Igreja\n\n"
        "- Ponto de reciclagem confirmado no dataset municipal.\n\n"
        "📌 **Fonte:** [*Lisboa Aberta*](https://dados.cm-lisboa.pt/) | **Atualizado:** 12:00"
    )

    cleaned = final_visual_pass(response)

    assert "[*Lisboa Aberta*](https://dados.cm-lisboa.pt/)" in cleaned
    assert "VisitLisboa" not in cleaned


def test_planner_source_canonicalizer_does_not_infer_from_body_keywords() -> None:
    """Planner source cleanup should preserve existing sources, not infer from final prose."""
    response = (
        "Use the train only if it is confirmed later; keep weather backups in mind.\n\n"
        "📌 **Source:** [*VisitLisboa Places*](https://www.visitlisboa.com/en/places) | **Updated:** 12:00"
    )

    cleaned = canonicalize_planner_source_line(response, language="en")

    assert "[*VisitLisboa Places*](https://www.visitlisboa.com/en/places)" in cleaned
    assert "[*CP*]" not in cleaned
    assert "[*IPMA*]" not in cleaned


def test_transport_finalizer_strips_operator_sources_without_tool_calls() -> None:
    """Transport limitation text must not cite operators that were not checked."""
    agent = TransportAgent.__new__(TransportAgent)
    agent._tool_calls_log = []
    response = (
        "Não consegui confirmar aqui uma ligação direta entre Cacilhas e Cristo Rei.\n\n"
        "📌 **Fonte:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | [*Carris*](https://www.carris.pt) | **Atualizado:** 12:00"
    )

    cleaned = agent._finalize_transport_response(
        response,
        user_message="Estou em Cacilhas e quero ir ao Cristo Rei. Há ligação direta?",
        language="pt",
    )

    assert "Fonte:" not in cleaned
    assert "Metro de Lisboa" not in cleaned
    assert "Carris" not in cleaned


def test_final_visual_pass_removes_unlinked_source_claims() -> None:
    """Unlinked source notes are not acceptable final provenance footers."""
    response = (
        "Não foi confirmada aqui uma ligação direta específica.\n\n"
        "**Fonte:** dados de transporte apresentados na resposta anterior."
    )

    cleaned = final_visual_pass(response)

    assert "Fonte:" not in cleaned
    assert "dados de transporte" not in cleaned


def test_final_visual_pass_restores_missing_nearest_metro_line_field() -> None:
    """Nearest-Metro cards must keep line metadata even after QA repair."""
    response = final_visual_pass(
        "🚇 Nearest Metro Stations\n\n"
        "- 🔵 **Restauradores**\n"
        "    - 📏 **Distance:** 3.1km (~36 min walk)\n\n"
        "📌 **Source:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | **Updated:** 22:55"
    )

    assert "- 🚇 **Lines:** Azul" in response


def test_final_visual_pass_links_explicit_carris_snapshot_source() -> None:
    """Explicit Carris GTFS-RT tool-source lines should become linked footers."""
    response = final_visual_pass(
        "### 🚇 Lisbon Mobility\n\n"
        "- 🚋 **Line 15E** — To Algés\n\n"
        "**Source:** Carris GTFS-RT cached snapshot (0s old)."
    )

    assert "[*Carris*](https://www.carris.pt)" in response
    assert "Carris GTFS-RT cached snapshot" not in response


def test_ride_hailing_queries_route_to_transport_limitations() -> None:
    """Uber/Bolt questions are mobility scope limits, not researcher web lookups."""
    decision = SupervisorAgent._single_domain_override(
        "Preciso de saber se Uber ou Bolt é melhor do Cais do Sodré para Alcântara agora."
    )

    assert decision is not None
    assert decision["agents"] == ["transport"]


def test_ride_hailing_limitation_uses_plain_user_language() -> None:
    """Ride-hailing limitations should not expose transport shorthand in final text."""
    response_pt = _build_unsupported_transport_scope_response(
        "Preciso de saber se Uber ou Bolt é melhor do Cais do Sodré para Alcântara agora.",
        "pt",
    )
    response_en = _build_unsupported_transport_scope_response(
        "Is Uber or Bolt better from Cais do Sodré to Alcântara right now?",
        "en",
    )

    assert response_pt is not None
    assert response_en is not None
    assert "ETA" not in response_pt
    assert "ETA" not in response_en
    assert "tempo estimado de recolha" in response_pt
    assert "estimated pickup time" in response_en


def test_final_visual_pass_replaces_eta_shorthand() -> None:
    """Final Markdown should not expose transport shorthand that normal users may not know."""
    response = (
        "### 🚌 736 at Marquês Pombal\n\n"
        "- ℹ️ **Live ETA:** no active vehicle is currently matched to this stop.\n"
        "- Compare the ETA before boarding."
    )

    cleaned = final_visual_pass(response)

    assert "ETA" not in cleaned
    assert "Live arrival estimate" in cleaned
    assert "estimated arrival" in cleaned


def test_final_visual_pass_keeps_carris_delay_language_consistent() -> None:
    """English transport answers should not leak Portuguese delay fragments."""
    response = (
        "### 🚇 Lisbon Mobility\n\n"
        "- 🚌 **Line 732** — To Caselas\n"
        "    - 🕐 **Next departures:** 14:36 *(atraso de 13 min)*, 14:47\n\n"
        "📌 **Source:** [*Carris*](https://www.carris.pt) | **Updated:** 14:08"
    )

    cleaned = final_visual_pass(response)

    assert "atraso" not in cleaned.lower()
    assert "13 min late" in cleaned


def test_transport_summary_tool_uses_aligned_sections(monkeypatch) -> None:
    """The aggregate status tool should emit clean section cards before LLM synthesis."""
    from tools import carris_api, transport_api

    def fake_fetch_json(url: str, *args: object, **kwargs: object) -> object:
        if url == transport_api.METRO_STATUS_URL:
            return {"resposta": {key: "OK" for key in transport_api.METRO_LINES}}
        return [{"id": "alert-1"}, {"id": "alert-2"}]

    monkeypatch.setattr(transport_api, "fetch_json_with_retry", fake_fetch_json)
    monkeypatch.setattr(carris_api, "fetch_gtfs_rt_vehicles", lambda: [{"id": "v1"}, {"id": "v2"}])
    monkeypatch.setattr(
        transport_api,
        "get_cp_aml_trains",
        lambda: [{"delay": 0}, {"delay": 90}, {"delay": 120}],
    )

    response = transport_api.get_transport_summary.func()

    assert "### 🔵 **Ponto de situação dos transportes em Lisboa**" in response
    assert "✅ **Resposta direta:**" in response
    assert "- **🚇 Metro de Lisboa**" in response
    assert "    - 🟡 **Amarela:** Ok" in response
    assert "    - 🔵 **Azul:** Ok" in response
    assert "    - 🟢 **Verde:** Ok" in response
    assert "    - 🔴 **Vermelha:** Ok" in response
    assert "    - ✅ **Estado geral:** Circulação normal em todas as linhas" in response
    assert "- 🟢 **Estado:** Circulação normal em todas as linhas" not in response
    assert "- **🚌 Carris Urban**" in response
    assert "    - ✅ **Veículos em serviço:** 2 veículos" in response
    assert "- **🚌 Carris Metropolitana**" in response
    assert "    - ⚠️ **Alertas ativos:** 2 alertas" in response
    assert "    - ⚠️ **Atrasos superiores a 1 min:** 2 comboios" in response
    assert ", [*Carris*]" not in response


def test_transport_summary_tool_supports_english_and_disruptions(monkeypatch) -> None:
    """The aggregate status tool should preserve structure in English and abnormal states."""
    from tools import carris_api, transport_api

    def fake_fetch_json(url: str, *args: object, **kwargs: object) -> object:
        if url == transport_api.METRO_STATUS_URL:
            statuses = {key: "OK" for key in transport_api.METRO_LINES}
            statuses["verde"] = "Service interrupted"
            return {"resposta": statuses}
        return []

    monkeypatch.setattr(transport_api, "fetch_json_with_retry", fake_fetch_json)
    monkeypatch.setattr(carris_api, "fetch_gtfs_rt_vehicles", lambda: [{"id": "v1"}])
    monkeypatch.setattr(transport_api, "get_cp_aml_trains", lambda: [{"delay": 0}])

    response = transport_api.get_transport_summary.func(language="en")

    assert "### 🔵 **Transport Status in Lisbon**" in response
    assert "✅ **Direct answer:**" in response
    assert "- **🚇 Metro de Lisboa**" in response
    assert "    - 🟡 **Yellow:** Ok" in response
    assert "    - 🟢 **Green:** Service interrupted" in response
    assert "    - ⚠️ **Overall status:** Disruptions are reported" in response
    assert "Normal service on all lines" not in response
    assert "- **🚌 Carris Urban**" in response
    assert "    - ✅ **Vehicles in service:** 1 vehicle" in response
    assert "- **🚆 CP Suburban Trains in Lisbon/AML**" in response
    assert "📌 **Source:**" in response


def test_transport_summary_routing_wins_over_cp_status_for_multi_operator_english_query() -> None:
    """A Metro+bus+train status query should not collapse to CP-only status."""
    from agent.agents.transport_agent import _build_deterministic_transport_tool_call

    message = _build_deterministic_transport_tool_call(
        "Give me the current status of the Metro, buses, and trains in Lisbon."
    )

    assert message is not None
    assert message.tool_calls[0]["name"] == "get_transport_summary"
    assert message.tool_calls[0]["args"] == {"language": "en"}


def test_final_visual_pass_keeps_transport_summary_operator_metrics_nested() -> None:
    """QA repair must not promote transport-summary operators into repeated H3 sections."""
    response = (
        "### 🚇 **Situação dos Transportes em Lisboa**\n\n"
        "### 🚇 **Metro de Lisboa**\n"
        "- 🟡 **Amarela:** Ok\n"
        "- 🟢 **Estado:** Circulação normal em todas as linhas\n\n"
        "**🚌 Carris Metropolitana**\n\n"
        "- ⚠️ **Alertas ativos:** 2 alertas\n\n"
        "### 🚆 **Comboios suburbanos CP em Lisboa/AML**\n"
        "- 📊 **Comboios a circular na AML:** 3 comboios\n"
        "- ⚠️ **Atrasos superiores a 1 min:** 2 comboios\n\n"
        "📌 **Fonte:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | [*Carris Metropolitana*](https://www.carrismetropolitana.pt) | [*CP*](https://www.cp.pt) | **Atualizado:** 09:00"
    )

    cleaned = final_visual_pass(response)

    assert "### 🚇 **Metro de Lisboa**" not in cleaned
    assert "### 🚆 **Comboios suburbanos CP em Lisboa/AML**" not in cleaned
    assert "- **🚇 Metro de Lisboa**" in cleaned
    assert "    - 🟡 **Amarela:** Ok" in cleaned
    assert "    - ✅ **Estado:** Circulação normal em todas as linhas" in cleaned
    assert "- 🟢 **Estado:** Circulação normal em todas as linhas" not in cleaned
    assert "    - ⚠️ **Alertas ativos:** 2 alertas" in cleaned
    assert "    - ⚠️ **Atrasos superiores a 1 min:** 2 comboios" in cleaned


def test_full_museum_day_fallback_is_complete_and_transport_grounded() -> None:
    """Museum-day fallbacks need a real route structure, not a vague list."""
    response = _build_deterministic_planner_fallback(
        user_message="Plan a full museum day in Lisbon for tomorrow, starting in Rossio and using public transport.",
        language="en",
        weather_data="✅ No active weather warnings for Lisbon.\n💧 Rain: 100% - Weak",
        transport_data="Metro de Lisboa and Carris data available.",
        places_data="VisitLisboa museums in Lisbon.",
        events_data="",
        qa_disclaimers=None,
    )

    assert "Full Museum Day From Rossio" in response
    assert "**09:30 · Chiado / São Roque**" in response
    assert "**14:15 · Gulbenkian / São Sebastião**" in response
    assert "**16:45 · Belém, if still realistic**" in response
    assert "### 🚇 **How to move**" in response
    assert "Recommended order" not in response
    assert "Opening hours, tickets, and crowding were not confirmed live" in response


def test_planner_fallback_uses_grounded_cards_before_static_templates() -> None:
    """Fallback should synthesize worker cards instead of freezing museum-day templates."""
    places_data = """
### 🏛️ National Tile Museum
- 📂 **Category:** Museums
- 📍 **Address:** Rua da Madre de Deus, 4, Lisboa
- 🌐 **Website:** [VisitLisboa](https://www.visitlisboa.com/en/places/national-tile-museum)

### 🏛️ Carris Museum
- 📂 **Category:** Museums
- 📍 **Address:** Rua 1º de Maio, Lisboa
- 🌐 **Website:** [VisitLisboa](https://www.visitlisboa.com/en/places/carris-museum)

### 🏛️ Museum of Lisbon
- 📂 **Category:** Museums
- 📍 **Address:** Campo Grande, Lisboa
- 🌐 **Website:** [VisitLisboa](https://www.visitlisboa.com/en/places/museum-of-lisbon)
"""

    response = _build_deterministic_planner_fallback(
        user_message="Plan a full museum day in Lisbon for tomorrow, starting in Rossio and using public transport.",
        language="en",
        weather_data="✅ No active weather warnings for Lisbon.",
        transport_data="Carris route details available from Rossio.",
        places_data=places_data,
        events_data="",
        qa_disclaimers=None,
    )

    assert "Lisbon museum day" in response
    assert "National Tile Museum" in response
    assert "Carris Museum" in response
    assert "Museum of Lisbon" in response
    assert "Chiado / São Roque" not in response
    assert "Gulbenkian / São Sebastião" not in response


def test_planner_card_extraction_rejects_field_labels_as_stops() -> None:
    """Fallback card extraction must not turn labels into itinerary stops."""
    raw = """
### 🏛️ 09:30 · Carmo Archaeological Museum
- 📂 **Category:** Museums
- 📍 **Address:** Largo do Carmo, Lisboa
- 📞 **Phone:** +351 000 000
- 🌐 **Website:** [VisitLisboa](https://www.visitlisboa.com/en/places/carmo)

### 🏛️ Category
- Not a real place heading.

### 🏛️ Phone
- Not a real place heading.
"""

    cards = _extract_visitlisboa_place_cards(raw)
    names = [card["name"] for card in cards]

    assert "Carmo Archaeological Museum" in names
    assert "Category" not in names
    assert "Phone" not in names


def test_full_museum_day_fallback_respects_requested_start_area() -> None:
    """Museum-day fallback should not force Rossio when another origin is requested."""
    response = _build_deterministic_planner_fallback(
        user_message="Plan a rainy museum day tomorrow from Baixa using public transport, with two indoor backup options.",
        language="en",
        weather_data="✅ No active weather warnings for Lisbon.",
        transport_data="Metro de Lisboa and Carris data available.",
        places_data="VisitLisboa museums in Lisbon.",
        events_data="",
        qa_disclaimers=None,
    )

    assert "Full Museum Day From Baixa" in response
    assert "From Baixa" in response
    assert "From Rossio" not in response
    assert "Rossio/Baixa-Chiado" not in response


def test_planner_rejects_empty_museum_day_blocks() -> None:
    """Full-day museum plans should not publish placeholder blocks."""
    response = """### 📅 Structured 1-Day Lisbon Plan

### ✅ Direct Answer
Here is a short ordered plan.

### 📍 Plan Blocks
- **Block 1 · Carris Museum**
- **Block 2 · short food/coffee break without assuming a booking.**
- **Block 3 · indoor backup or return leg.**

### 🚇 Movement Logic
- Use public transport.

### ⚠️ Limitations
- Opening hours are not confirmed."""

    assert _planner_response_has_incomplete_museum_day_blocks(
        "Plan a full museum day in Lisbon for tomorrow, starting in Rossio and using public transport.",
        response,
    )


def test_planner_accepts_concrete_museum_day_blocks() -> None:
    """Concrete museum-day drafts should remain eligible for normal grounding checks."""
    response = """### 📅 Full Museum Day From Rossio

### ✅ Direct Answer
Use a central morning and a Belém afternoon.

### 📍 Recommended Itinerary
- **09:30 · Chiado museum area**
- **11:30 · National Museum of Ancient Art**
- **15:00 · Belém museum cluster**

### 🚇 Movement Logic
- Use Carris 15E toward Belém where available.

### ⚠️ Limitations
- Opening hours are not confirmed."""

    assert not _planner_response_has_incomplete_museum_day_blocks(
        "Plan a full museum day in Lisbon for tomorrow, starting in Rossio and using public transport.",
        response,
    )


def test_final_visual_pass_repairs_full_museum_day_headings() -> None:
    """Formatter should not leave planner section headers as top-level bullets."""
    response = (
        "- 📅 **Full Museum Day From Rossio**\n"
        "- The strongest route is to start centrally, move to São Sebastião/Gulbenkian by Metro.\n"
        "- ⛅ Conditions and Rain Strategy\n"
        "- No active weather warnings for Lisbon.\n\n"
        "### 📅 Recommended Itinerary\n\n---\n\n"
        "### 🏛️ 09:30 · Chiado / São Roque\n"
        "- 🚇 Movement Logic\n"
        "- Use Metro for Rossio/Baixa-Chiado → São Sebastião.\n"
    )

    cleaned = final_visual_pass(response)

    assert cleaned.startswith("### 📅 Full Museum Day From Rossio")
    assert "- 📅 **Full Museum Day" not in cleaned
    assert "### ⛅ Conditions and Rain Strategy" in cleaned
    assert "### 🚇 Movement Logic" in cleaned
    assert "### 📅 Recommended Itinerary" in cleaned
    assert "### 🏛️ 09:30 · Chiado / São Roque" in cleaned
    assert "### 📅 Recommended Itinerary\n\n---\n\n### 🏛️" not in cleaned


def test_multi_day_planner_fallback_is_a_tourist_itinerary_not_framework() -> None:
    """Multi-day fallback should provide visitable days, not a thin framework."""
    response = _build_deterministic_planner_fallback(
        user_message=(
            "Plan 5 days in Lisbon for two adults and a 7-year-old, staying near Saldanha, "
            "using public transport, avoiding long walks, with indoor backups if it rains "
            "and one low-cost meal idea per day."
        ),
        language="en",
        weather_data="✅ No active weather warnings for Lisbon.",
        transport_data="Metro de Lisboa and Carris data available.",
        places_data="VisitLisboa places include Jerónimos, MAAT, Gulbenkian, Oceanário, and Chiado museums.",
        events_data="",
        qa_disclaimers=None,
    )

    assert "5-Day Lisbon Itinerary" in response
    assert "planning framework" not in response.lower()
    assert "Day 1 · Baixa, Chiado and Old Lisbon" in response
    assert "Day 2 · Belém History" in response
    assert "Jerónimos Monastery" in response
    assert "Oceanário de Lisboa" in response
    assert "Low-cost meal" in response
    assert "Rain backup" in response
    assert "How to move" in response
    assert "github.com/Silvestre17" not in final_visual_pass(response)


def test_overlong_multi_day_request_is_bounded_but_still_detailed() -> None:
    """Seven-day requests can be bounded to five days without becoming vague."""
    response = _build_deterministic_planner_fallback(
        user_message=(
            "Plan 7 days in Lisbon from Avenida da Liberdade with museums, viewpoints, "
            "easy public transport, and rain backups, but do not invent opening hours or ticket prices."
        ),
        language="en",
        weather_data="",
        transport_data="Metro de Lisboa and Carris data available.",
        places_data="VisitLisboa museums and viewpoints.",
        events_data="",
        qa_disclaimers=None,
    )

    assert "First 5 Days in Lisbon" in response
    assert "thin 7-day list" in response
    assert "Day 5 · Alcântara or Príncipe Real" in response
    assert "opening hours, tickets" in response
    assert "### 📍 Day 1" in response
    assert response.count("### 📍 Day") == 5


def test_pt_multi_day_base_extraction_is_clean() -> None:
    """Portuguese base extraction should not include the generic city phrase."""
    response = _build_deterministic_planner_fallback(
        user_message=(
            "Planeia 4 dias em Lisboa a partir do Marquês de Pombal, com monumentos, "
            "museus, transportes simples, pouca caminhada e alternativas interiores se chover."
        ),
        language="pt",
        weather_data="✅ Sem avisos meteorológicos ativos para Lisboa.",
        transport_data="Metro de Lisboa e Carris disponíveis.",
        places_data="VisitLisboa locais.",
        events_data="",
        qa_disclaimers=None,
    )

    assert "Base:** Marquês de Pombal" in response
    assert "Lisboa a partir do Marquês" not in response
    assert "Linha Vermelha entre **Marquês de Pombal**/Saldanha" not in response
    assert "visita principal;" not in response


def test_multi_day_transport_phrasing_avoids_duplicate_station_slash() -> None:
    """Transport guidance should not emit awkward Saldanha/Saldanha shortcuts."""
    response = _build_deterministic_planner_fallback(
        user_message="Plan 5 days in Lisbon staying near Saldanha using public transport.",
        language="en",
        weather_data="",
        transport_data="Metro de Lisboa and Carris data available.",
        places_data="VisitLisboa places.",
        events_data="",
        qa_disclaimers=None,
    )

    assert "Saldanha/Saldanha" not in response
    assert "Use Metro toward Oriente" in response


def test_multi_day_quality_mode_preserves_later_days() -> None:
    """The quality guard must not collapse a useful multi-day answer into Day 1."""
    response = (
        "### 📅 5-Day Lisbon Itinerary\n\n"
        "### 📍 Day 1 · Baixa\n- Morning stop\n\n"
        "### 📍 Day 2 · Belém\n- Morning stop\n\n"
        "### 📍 Day 3 · Oriente\n- Morning stop"
    )

    cleaned = enforce_multi_day_quality_mode(
        response=response,
        user_message="Plan 5 days in Lisbon with public transport.",
        language="en",
    )

    assert "Day 2 · Belém" in cleaned
    assert "Day 3 · Oriente" in cleaned
    assert "Day 1 · Suggested Itinerary" not in cleaned


def test_non_evidence_source_line_is_removed_from_scope_limitations() -> None:
    """A limitation statement should not be converted into a fake source footer."""
    response_pt = (
        "### 🚕 Mobilidade fora do âmbito confirmado\n\n"
        "Não consigo confirmar preços em tempo real.\n\n"
        "---\n\n"
        "**Fonte:** informação não confirmada em tempo real."
    )
    response_pt_variant = (
        "### 🚕 Mobilidade fora do âmbito confirmado\n\n"
        "Não consigo confirmar preços em tempo real.\n\n"
        "*Fonte: informação de mobilidade não confirmada em tempo real.*"
    )
    response_en = (
        "### 🚕 Mobility outside confirmed scope\n\n"
        "I cannot verify real-time ride-hailing prices.\n\n"
        "**Source:** Transport output only."
    )
    response_en_variant = (
        "### 🚕 Mobility outside confirmed scope\n\n"
        "I cannot verify real-time ride-hailing prices.\n\n"
        "*Source: Real-time ride-hailing data not available in this system.*"
    )
    response_en_second_variant = (
        "### 🚕 Cais do Sodré → Alcântara\n\n"
        "I do not have live ride-hailing prices.\n\n"
        "---\n\n"
        "**Source:** Live Uber/Bolt pricing and availability were not available in the provided data."
    )
    response_en_third_variant = (
        "### 🚕 Mobility outside confirmed scope\n\n"
        "I cannot verify live ride-hailing prices.\n\n"
        "**Source:** User request; no real-time ride-hailing data available."
    )

    cleaned_pt = final_visual_pass(response_pt)
    cleaned_pt_variant = final_visual_pass(response_pt_variant)
    cleaned_en = final_visual_pass(response_en)
    cleaned_en_variant = final_visual_pass(response_en_variant)
    cleaned_en_second_variant = final_visual_pass(response_en_second_variant)
    cleaned_en_third_variant = final_visual_pass(response_en_third_variant)

    assert "Fonte:" not in cleaned_pt
    assert "informação não confirmada" not in cleaned_pt
    assert "Fonte:" not in cleaned_pt_variant
    assert "informação de mobilidade não confirmada" not in cleaned_pt_variant
    assert "Source:" not in cleaned_en
    assert "Transport output only" not in cleaned_en
    assert "Source:" not in cleaned_en_variant
    assert "not available in this system" not in cleaned_en_variant
    assert "Source:" not in cleaned_en_second_variant
    assert "provided data" not in cleaned_en_second_variant
    assert "Source:" not in cleaned_en_third_variant
    assert "User request" not in cleaned_en_third_variant


def test_standalone_planner_tip_lines_keep_bullet_marker() -> None:
    """Tip rows should render as list items instead of orphan text in Streamlit."""
    response = (
        "### 🏛️ Plano otimizado\n"
        "**09:00 · Paragem histórica**\n"
        "- 📍 **Localização:** Lisboa\n\n"
        "💡 **Dica:** mantém o início compacto."
    )

    cleaned = final_visual_pass(response)

    assert "- 💡 **Dica:** mantém o início compacto." in cleaned


def test_planner_card_field_bullets_do_not_keep_orphan_indentation() -> None:
    """Card field rows accidentally indented by repair passes should align visually."""
    response = (
        "### 🏛️ Plano alternativo\n"
        "**11:30 · Campo de Ourique**\n"
        "    - 📍 **Zona:** Campo de Ourique\n"
        "- 🏷️ **Tema:** gastronomia tradicional"
    )

    cleaned = final_visual_pass(response)

    assert "\n    - 📍" not in cleaned
    assert "- 📍 **Zona:** Campo de Ourique" in cleaned


def test_planner_tip_bullets_stay_attached_to_previous_card_field() -> None:
    """Tip bullets should not be visually detached from their card metadata."""
    response = (
        "### 🏛️ Plano alternativo\n"
        "**11:30 · Campo de Ourique**\n"
        "- 🏷️ **Tema:** gastronomia tradicional\n\n"
        "- 💡 **Dica:** escolhe aqui o almoço."
    )

    cleaned = final_visual_pass(response)

    assert "- 🏷️ **Tema:** gastronomia tradicional\n- 💡 **Dica:** escolhe aqui o almoço." in cleaned


def test_history_pastry_route_is_planning_not_direct_weather_transport() -> None:
    """Rich route requests with history/food/pacing need planner synthesis."""
    query = (
        "I am in Chiado this afternoon and want a useful Belém route with history, "
        "transport, weather-aware pacing, and a custard tart stop."
    )

    assert SupervisorAgent._is_planning_query(query)
    assert not SupervisorAgent._is_direct_weather_transport_query(query)
    assert SupervisorAgent._single_domain_override(query) is None
    assert SupervisorAgent()._requires_weather_for_planning(query)


def test_belem_history_pastry_fallback_is_rich_but_not_live_departure_dump() -> None:
    """Belém history+pastry plans should integrate weather/transport without live-time misuse."""
    response = _build_deterministic_planner_fallback(
        user_message=(
            "Plan a full afternoon in Belém starting from Chiado, include historical context, "
            "realistic transport, and one pastry stop."
        ),
        language="en",
        weather_data="Weather Today in Lisbon: 11.3°C to 20.3°C\nRain: 55% - weak",
        transport_data="Carris route 15E and bus 728. Cais do Sodre CP Cascais line to Belem.",
        places_data="VisitLisboa Belém historical places and Pastéis de Belém.",
        events_data="",
        qa_disclaimers=None,
    )

    assert "Belém Afternoon From Chiado" in response
    assert "Weather and pacing" in response
    assert "Recommended transport" in response
    assert "Jerónimos Monastery" in response
    assert "Pastéis de Belém" in response
    assert "Current status" not in response
    assert "departures from CP" not in response
    assert "taxi/rideshare" not in response.lower()
    assert "Helpful Notes" not in response
    assert "[*VisitLisboa" in response


def test_planner_rejects_live_departures_in_non_live_itinerary() -> None:
    """Current departures captured now should not be reused as an afternoon itinerary schedule."""
    response = (
        "### 📅 Belém Afternoon\n"
        "- **Transport:** CP Belém departures: **23:02**, **23:08**, **23:32**."
    )

    assert _planner_response_has_transport_quality_defects(
        response,
        "I am in Chiado this afternoon and want a useful Belém route with history and a pastry stop.",
        "CP Cascais line details are available.",
    )


def test_final_post_qa_guard_removes_bad_qa_footer_after_clean_transport_answer() -> None:
    """The final deterministic guard must clean corruption added after QA repair."""
    response = (
        "### 🚇 Nearest Metro Stations\n\n"
        "- 🟡 **Rato**\n"
        "    - 📏 **Distance:** 2.4 km\n"
        "    - 🚇 **Lines:** Yellow line\n\n"
        "📌 **Source:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | **Updated:** 12:00\n\n"
        "Fonte: dados de transporte apresentados na resposta anterior\n"
        "- 📏 **Distance:** not provided\n"
        "- 🚇 **Lines:** not provided\n"
        "###    \n"
        "⚠️ ⚠️ Check operator updates."
    )

    guarded = final_post_qa_guard(response, language="en")

    assert "dados de transporte apresentados" not in guarded
    assert "not provided" not in guarded
    assert guarded.count("https://www.metrolisboa.pt") == 1
    assert "###    " not in guarded
    assert "⚠️ ⚠️" not in guarded


def test_structured_planner_fallback_is_schema_valid_and_limits_overlong_requests() -> None:
    """Planner safety fallback is a reusable schema, not a prompt-specific answer."""
    response = _build_structured_plan_fallback(
        user_message=(
            "Plan 7 days in Lisbon with exact routes, restaurants, tickets, prices, "
            "weather, beaches, museums, nightlife, and no repeated neighbourhoods."
        ),
        language="en",
        weather_data="",
        transport_data="",
        places_data="",
        events_data="",
        qa_disclaimers=None,
    )

    assert _planner_response_matches_schema(response)
    assert "first 5 days" in response.lower()
    assert "not confirmed" in response.lower() or "did not confirm" in response.lower()
    assert "prices" in response.lower()
    assert "tickets" in response.lower()
    assert "Restaurant:**" not in response
    assert "Museum:**" not in response


def test_final_guard_renders_raw_planner_schema_visual_contract() -> None:
    """Raw planner schema labels are stable formatter defects, not valid final UI."""
    raw = """### Title
5-Day Lisbon Plan

### Direct Answer
A high-level plan is possible, but exact prices are not confirmed.

### Constraints Used
- history and viewpoints
- public transport
- rainy backups

### Plan Blocks
- Day 1 — Baixa and Chiado
Why This Day: central orientation with short walks.
Transport Note: use nearby metro anchors.

### Movement Logic
Use transport principles, not live departures.

### Weather Strategy
Keep indoor backups.

### Limitations
Opening hours and tickets were not confirmed.

📌 **Source:** [*VisitLisboa*](https://www.visitlisboa.com)
"""

    guarded = final_post_qa_guard(raw, language="en")

    assert "### Title" not in guarded
    assert "### Direct Answer" not in guarded
    assert "### Plan Blocks" not in guarded
    assert "Why This Day:" not in guarded
    assert "Transport Note:" not in guarded
    assert "### 📅 **" in guarded
    assert "✅ **Direct answer:**" in guarded
    assert "### 🧭 **Plan basis**" in guarded
    assert "### 🚇 **How to move**" in guarded
    assert "### ⚠️ **Final notes**" in guarded
    assert "### 📍 **Day 1" in guarded
    assert "    - " in guarded
    assert guarded.count("📌 **Source:**") == 1


def test_overcomplex_planning_guard_builds_bounded_visual_answer() -> None:
    """Long exact-detail itinerary requests should be bounded before worker synthesis."""
    prompt = (
        "Plan 7 days in Lisbon with exact routes, restaurants, tickets, prices, "
        "weather, beaches, museums, nightlife, and no repeated neighbourhoods."
    )

    assert is_overcomplex_planning_request(prompt)
    response = build_bounded_planning_framework("en")

    assert "### 📅 **" in response
    assert "### Title" not in response
    assert "### Direct Answer" not in response
    assert "5-day high-level framework" in response
    assert "exact live routes and schedules" in response.lower()
    assert "📌 **Source:**" not in response
    assert "    - " in response


def test_conversation_anchor_extraction_rejects_schema_labels() -> None:
    """Follow-up destination anchors must be real place-like labels, not schema headings."""
    assistant = MultiAgentAssistant.__new__(MultiAgentAssistant)
    response = """### 📅 **Structured 1-Day Lisbon Plan**

✅ **Direct answer:** Use a compact plan.

### 📍 **Block 1 · Museu de Lisboa - Santo António**
    - 🎯 **Purpose:** Indoor stop.

### 🚇 **Movement logic**
    - 🚇 Use Rossio as origin.
"""

    anchors = assistant._extract_destination_candidates_from_plan(response)

    assert "Structured 1-Day Lisbon Plan" not in anchors
    assert "Direct Answer" not in anchors
    assert "Movement logic" not in anchors
    assert "Museu de Lisboa - Santo António" in anchors


def test_oriente_nearby_fallback_uses_parque_das_nacoes_not_museu_do_oriente() -> None:
    """Oriente station locality should not be conflated with Museu do Oriente."""
    raw = _build_structured_plan_fallback(
        user_message="I arrive at Oriente at 18:30 and want dinner plus a rain-safe cultural stop nearby.",
        language="en",
        weather_data="Rain possible this evening.",
        transport_data="Metro Red line serves Oriente.",
        places_data="### Museu do Oriente\nCategory: museum\nDescription: Cultural museum west of the station area.",
        events_data="",
        qa_disclaimers=None,
    )
    guarded = final_post_qa_guard(raw, language="en")

    assert "Museu do Oriente" not in guarded
    assert "Parque das Nações" in guarded or "Oriente station" in guarded
    assert "Centro Vasco da Gama" in guarded
    assert "opening hours" in guarded.lower() or "availability" in guarded.lower()
