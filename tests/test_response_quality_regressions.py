import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.agents.qa_agent import QualityAssuranceAgent
from agent.agents.researcher_agent import ResearcherAgent
from agent.agents.transport_agent import TransportAgent
from agent.graph import MultiAgentAssistant
from agent.utils.langsmith_tracing import get_langsmith_request_tracking_status
from agent.utils.response_formatter import (
    build_bilingual_note,
    final_visual_pass,
    finalize_worker_response,
    infer_response_language,
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
    assert "📌 **Fonte:** Dados do [*IPMA*](https://www.ipma.pt) | **Atualizado:**" in output


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

    assert "####" in output
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


def test_researcher_named_service_lookup_prefers_place_search_over_generic_nearby_results() -> None:
    """Named service queries should search by the facility name instead of returning generic nearby-service lists."""
    with patch.object(ResearcherAgent, "__init__", lambda self: None):
        agent = ResearcherAgent()

        places_tool = MagicMock()
        places_tool.name = "search_places_attractions"
        places_tool.invoke = MagicMock(
            return_value=(
                "### \U0001F4CD Hospital Santa Maria\n\n"
                "- \U0001F4CD **Morada:** Avenida Professor Egas Moniz, Lisboa"
            )
        )
        nearby_tool = MagicMock()
        nearby_tool.name = "find_nearby_services"
        nearby_tool.invoke = MagicMock(side_effect=AssertionError("generic nearby lookup should be skipped"))
        agent.tools = [places_tool, nearby_tool]

        result = agent._run_direct_place_lookup("Onde fica o Hospital Santa Maria?", "pt")

        places_tool.invoke.assert_called_once_with(
            {
                "query": "Hospital Santa Maria",
                "max_results": 5,
                "offset": 0,
                "language": "pt",
                "specific_lookup": True,
            }
        )
        assert "Hospital Santa Maria" in result


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
