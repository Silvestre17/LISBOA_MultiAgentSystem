# ==========================================================================
# Master Thesis - VisitLisboa API Tests
#   - André Filipe Gomes Silvestre, 20240502
#
#   Focused regressions for VisitLisboa runtime helpers.
# ==========================================================================

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from unittest.mock import patch

from agent.agents.researcher_agent import ResearcherAgent
from agent.utils.response_formatter import (
    canonicalize_visitlisboa_source_line,
    final_post_qa_guard,
    final_visual_pass,
    format_researcher_event_cards,
    format_researcher_card,
    reconcile_researcher_place_response,
    researcher_place_response_missing_requested_fields,
)
import tools.visitlisboa_api as visitlisboa_api


class _DummyLLMResponse:
    """Minimal response object for ResearcherAgent LLM-localization tests."""

    def __init__(self, content: str) -> None:
        self.content = content


def test_get_vector_store_initializes_once_under_parallel_calls() -> None:
    """Parallel callers should share one lazy KnowledgeBase initialization."""
    sentinel = object()
    original_store = visitlisboa_api._vector_store

    try:
        visitlisboa_api._vector_store = None

        with patch("tools.vector_store.KnowledgeBase", return_value=sentinel) as kb_cls:
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(lambda _idx: visitlisboa_api._get_vector_store(), range(4)))

        assert results == [sentinel, sentinel, sentinel, sentinel]
        kb_cls.assert_called_once_with(use_gpu=False)
    finally:
        visitlisboa_api._vector_store = original_store


def test_search_cultural_events_filters_free_event_queries() -> None:
    """Free-event queries should keep only free-admission events from the VisitLisboa event pool."""
    event_day = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    sample_events = [
        {
            "title": "Free Jazz Night",
            "category": "Music",
            "description": "Free entry jazz showcase.",
            "price": "Free Entry",
            "url": "https://example.com/free-jazz",
            "dates": [{"date": {"datetime_iso": event_day}}],
        },
        {
            "title": "Paid Club Night",
            "category": "Music",
            "description": "Ticketed electronic music event.",
            "price": "desde €25",
            "url": "https://example.com/paid-club",
            "dates": [{"date": {"datetime_iso": event_day}}],
        },
    ]

    with patch.object(visitlisboa_api, "_load_events_json", return_value=sample_events):
        result = str(
            visitlisboa_api.search_cultural_events.invoke(
                {"query": "eventos gratuitos em Lisboa", "language": "pt", "max_results": 5}
            )
        )

    assert "Free Jazz Night" in result
    assert "Paid Club Night" not in result


def test_search_cultural_events_filters_confirmed_outdoor_events() -> None:
    """Outdoor-event queries should not keep indoor events just because their address has a street."""
    event_day = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    sample_events = [
        {
            "title": "Picnic Concert",
            "category": "Music",
            "full_description": "Blankets on the grass with live music.",
            "venue_name": "Parque Eduardo VII",
            "location": "Parque Eduardo VII, Lisboa",
            "url": "https://example.com/picnic-concert",
            "dates": [{"date": {"datetime_iso": event_day}}],
        },
        {
            "title": "Indoor Theatre",
            "category": "Theater Opera & Dance",
            "full_description": "A theatre performance indoors.",
            "venue_name": "Teatro Aberto",
            "location": "Teatro Aberto, Rua Armando Cortez, Lisboa",
            "url": "https://example.com/indoor-theatre",
            "dates": [{"date": {"datetime_iso": event_day}}],
        },
    ]

    with patch.object(visitlisboa_api, "_load_events_json", return_value=sample_events):
        result = str(
            visitlisboa_api.search_cultural_events.invoke(
                {"query": "outdoor events", "date_filter": "tomorrow", "language": "en", "max_results": 5}
            )
        )

    assert "Picnic Concert" in result
    assert "Indoor Theatre" not in result


def test_known_place_aliases_cover_diacritics_typos_and_abbreviations() -> None:
    """VisitLisboa place lookups should normalize common PT/EN aliases and typos."""
    assert visitlisboa_api._apply_known_place_lookup_alias("Mosteiro dos Jerónimos") == "Jerónimos Monastery"
    assert visitlisboa_api._apply_known_place_lookup_alias("Jeronimos") == "Jerónimos Monastery"
    assert visitlisboa_api._apply_known_place_lookup_alias("Gulbenkiam") == "Gulbenkian Museum"
    assert visitlisboa_api._apply_known_place_lookup_alias("MAAT") == "Museu de Arte, Arquitetura e Tecnologia"
    assert visitlisboa_api._apply_known_place_lookup_alias("CCB") == "Centro Cultural de Belém"


def test_pt_visitlisboa_description_and_value_helpers_do_not_leak_raw_english() -> None:
    """PT tool output should not expose raw English scraped descriptions or values."""
    description = "The global world of innovation converges here with visitors from many countries."

    assert visitlisboa_api._localize_visitlisboa_description(description, "pt", content_kind="event") == (
        "Descrição disponível na página oficial do evento."
    )
    assert visitlisboa_api._localize_place_value_text("Free with Lisboa Card", "pt") == "Gratuito com Lisboa Card"
    assert visitlisboa_api._localize_place_value_text("Price: Free", "pt") == "Preço: Gratuito"
    assert "with Lisboa Card" not in visitlisboa_api._localize_place_value_text("20% with Lisboa Card", "pt")
    assert visitlisboa_api._localize_place_category("Attractions", "pt") == "Atrações"
    assert visitlisboa_api._localize_place_title("Castle Museum", "pt") == "Castelo de São Jorge"
    assert visitlisboa_api._localize_place_title("National Palace and Gardens of Queluz", "pt") == "Palácio Nacional e Jardins de Queluz"
    assert visitlisboa_api._localize_place_title("Pena National Palace", "pt") == "Palácio Nacional da Pena"
    assert visitlisboa_api._localize_place_title("Carmo Archaeological Museum", "pt") == "Museu Arqueológico do Carmo"


def test_place_ticket_price_compaction_removes_scraper_scaffolding() -> None:
    """VisitLisboa place prices should be compacted before truncation or rendering."""
    raw = "link Children Free until (age): 3 Children (4-12): 4 € Adult: 8 € Family: 21 € Senior: 5 € Student: 5 €"

    result = visitlisboa_api._compact_place_ticket_price_text(raw, language="en")

    assert result.startswith("Children free until age 3")
    assert "link Children" not in result
    assert "; Adult: 8 €" in result
    assert "; ;" not in result
    assert "S..." not in result
    assert visitlisboa_api._compact_place_ticket_price_text("Price: Gratis", language="en") == "Free"
    assert visitlisboa_api._compact_place_ticket_price_text("Price: Gratis", language="pt") == "Gratuito"


def test_generic_visitlisboa_location_is_omitted_instead_of_maps_search() -> None:
    """Generic Lisbon-only locations should not become ungrounded Maps searches."""
    line = visitlisboa_api._format_visitlisboa_location_line("Lisbon", "Gulbenkian Museum", language="pt")

    assert line == ""


def test_specific_visitlisboa_location_is_labelled_and_linked() -> None:
    """Specific VisitLisboa locations should render as address fields with map links."""
    line = visitlisboa_api._format_visitlisboa_location_line(
        "Rua Ivens, 62, 1200-227, Lisboa",
        "Museum of Illusions",
        language="pt",
    )

    assert "📍 **Morada:**" in line
    assert "Rua Ivens, 62, 1200-227, Lisboa" in line
    assert "https://www.google.com/maps/search/?api=1&query=Rua+Ivens" in line


def test_search_places_attractions_renders_enriched_place_cards() -> None:
    """VisitLisboa place results should expose enriched fields as aligned user-facing cards."""
    sample_place = {
        "url": "https://www.visitlisboa.com/en/places/national-tile-museum",
        "title": "National Tile Museum",
        "category": "Museums & Monuments",
        "short_description": "A museum dedicated to the history of Portuguese tilework.",
        "location": "Rua da Madre de Deus, 4, 1900-312 Lisboa",
        "features": ["Museum", "Wi-Fi"],
        "contact_info": {
            "phone": "351218100340",
            "email": "info@museudoazulejo.pt",
            "website": "https://www.museudoazulejo.gov.pt",
        },
        "schedules": [
            {
                "name": "Schedule",
                "hours": {"Tuesday": "10:00 - 18:00"},
                "today": "Today: 10:00 - 18:00",
            }
        ],
        "tickets_offers": {
            "description": "Adult: 8 € Student: 4 €",
            "links": [{"text": "BUY", "url": "https://tickets.example.com/azulejo"}],
        },
        "tripadvisor": {"rating": "4.5", "reviews_count": "1900"},
        "lisboa_card_benefit": "Free with Lisboa Card",
    }

    with (
        patch.object(visitlisboa_api, "_get_vector_store", return_value=None),
        patch.object(visitlisboa_api, "_load_places_json", return_value=[sample_place]),
        patch.object(visitlisboa_api, "_get_place_by_url", return_value=sample_place),
    ):
        result = str(
            visitlisboa_api.search_places_attractions.invoke(
                {"query": "National Tile Museum", "language": "en", "max_results": 5, "specific_lookup": True}
            )
        )

    assert "**🏛️ National Tile Museum**" in result
    assert "    - 📝 **Description:** A museum dedicated to the history of Portuguese tilework." in result
    assert "    - 📂 **Category:** Museums & Monuments" in result
    assert "    - 📍 **Address:** [Rua da Madre de Deus" in result
    assert "    - 🕒 **Hours:** Today: 10:00 - 18:00" in result
    assert "    - 💶 **Price:** Free with Lisboa Card; Adult: 8 €; Student: 4 €" in result
    assert "    - ⭐ **Rating:** TripAdvisor 4.5/5 (1900 reviews)" in result
    assert "    - 📞 **Phone:** [+351 218 100 340](tel:+351218100340)" in result
    assert "    - ✉️ **Email:** [info@museudoazulejo.pt](mailto:info@museudoazulejo.pt)" in result
    assert "    - 🌐 **Website:** [Official website](https://www.museudoazulejo.gov.pt)" in result
    assert "    - 🎟️ **Tickets:** [Buy tickets](https://tickets.example.com/azulejo)" in result
    assert "    - 🔗 **More details:** [VisitLisboa](https://www.visitlisboa.com/en/places/national-tile-museum)" in result
    assert not re.search(r"(?m)^\s*\d+\.\s+", result)
    assert "📊 **Source mix:**" not in result


def test_search_cultural_events_renders_enriched_event_cards() -> None:
    """VisitLisboa event results should use descriptions, venue, prices, links, tickets, and highlights."""
    event_day = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    sample_event = {
        "url": "https://www.visitlisboa.com/en/events/free-jazz-night",
        "title": "Free Jazz Night",
        "category": "Music",
        "short_description": "An outdoor jazz concert in a Lisbon garden.",
        "full_description": "Long fallback text.",
        "price": "Free Entry",
        "information_links": {"official.example.com": "https://official.example.com/free-jazz"},
        "buy_tickets_url": "https://tickets.example.com/free-jazz",
        "venue_locations": [
            {"venue_name": "Jardim da Estrela", "location": "Jardim da Estrela, Lisboa"}
        ],
        "schedule_notes": ["Saturday, 21:00"],
        "highlight_links": [{"title": "Programme", "url": "https://official.example.com/programme"}],
        "dates": [{"type": "single", "date": {"datetime_iso": event_day, "display_text": event_day, "time": "21:00"}}],
    }

    with patch.object(visitlisboa_api, "_load_events_json", return_value=[sample_event]):
        result = str(
            visitlisboa_api.search_cultural_events.invoke(
                {"query": "jazz", "date_filter": "upcoming", "language": "en", "max_results": 5}
            )
        )

    assert "**🎵 Free Jazz Night**" in result
    assert "    - 📝 **Description:** An outdoor jazz concert in a Lisbon garden." in result
    assert "    - 📍 **Venue:** [Jardim da Estrela, Lisboa]" in result
    assert "    - 💶 **Price:** Free Entry" in result
    assert "    - 🌐 **Website:** [Official website](https://official.example.com/free-jazz)" in result
    assert "    - 🎟️ **Tickets:** [Buy tickets](https://tickets.example.com/free-jazz)" in result
    assert "    - 🕒 **Schedule:** Saturday, 21:00" in result
    assert "    - ✨ **Highlights:** [Programme](https://official.example.com/programme)" in result
    assert "    - 🔗 **More details:** [VisitLisboa](https://www.visitlisboa.com/en/events/free-jazz-night)" in result
    assert not re.search(r"(?m)^\s*\d+\.\s+", result)
    assert "Result count" not in result
    assert "Highlights shown" not in result


def test_pt_visitlisboa_event_cards_do_not_localize_detail_urls_or_emit_loose_more_marker() -> None:
    """Portuguese event cards should keep canonical /en/ VisitLisboa URLs and complete cards only."""
    event_day = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    sample_events = [
        {
            "url": "https://www.visitlisboa.com/en/events/free-jazz-night",
            "title": "Free Jazz Night",
            "category": "Music",
            "short_description": "Descrição disponível na página oficial do evento.",
            "price": "Free Entry",
            "location": "Jardim da Estrela, Lisboa",
            "dates": [{"date": {"datetime_iso": event_day}}],
        },
        {
            "url": "https://www.visitlisboa.com/en/events/book-market",
            "title": "Book Market",
            "category": "Other",
            "location": "Lisboa",
            "dates": [{"date": {"datetime_iso": event_day}}],
        },
    ]

    with patch.object(visitlisboa_api, "_load_events_json", return_value=sample_events):
        result = str(
            visitlisboa_api.search_cultural_events.invoke(
                {"query": "eventos esta semana", "date_filter": "this week", "language": "pt", "max_results": 1}
            )
        )

    assert "https://www.visitlisboa.com/en/events/free-jazz-night" in result
    assert "pt-pt" not in result
    assert "... e 1 mais" not in result
    assert "Destaques mostrados" not in result
    assert "Resultado do filtro" not in result
    assert "Descrição disponível na página oficial do evento" not in result


def test_event_formatter_recovers_worker_bold_event_cards_without_collapsing_titles() -> None:
    """QA guard should not merge later event titles into the first event card."""
    worker_output = """### 🔵 **Eventos encontrados**
✅ **Resposta direta:** Encontrei 2 eventos com data confirmada.

**🎵 Gipsy Kings**
    - 🗓️ **Quando:** 10 de maio
    - ⏱️ **Duração:** 🎯 Um só dia
    - 📂 **Categoria:** Música
    - 📍 **Local:** [Coliseu de Lisboa](https://www.google.com/maps/search/?api=1&query=Coliseu+de+Lisboa)
    - 🔗 **Mais detalhes:** [VisitLisboa](https://www.visitlisboa.com/en/events/gipsy-kings)

**📅 Morabeza LX**
    - 🗓️ **Quando:** 09 de maio
    - ⏱️ **Duração:** 🎯 Um só dia
    - 📂 **Categoria:** Festivais
    - 🔗 **Mais detalhes:** [VisitLisboa](https://www.visitlisboa.com/en/events/morabeza-lx)

📌 **Fonte:** [*VisitLisboa Eventos*](https://www.visitlisboa.com/pt-pt/eventos) | **Atualizado:** 09:05"""

    cleaned = final_post_qa_guard(
        format_researcher_event_cards(worker_output, language="pt", user_query="eventos esta semana"),
        language="pt",
    )

    assert "**🎵 Gipsy Kings**" in cleaned
    assert "**🎭 Morabeza LX**" in cleaned
    assert "- **📅 Morabeza LX**" not in cleaned
    assert "    - 📍 **Morada:** [Coliseu de Lisboa]" in cleaned
    assert "    - ⏱️ **Duração:** 🎯 Um só dia" in cleaned
    assert cleaned.count("📌 **Fonte:**") == 1
    assert "https://www.visitlisboa.com/en/events/gipsy-kings" in cleaned
    assert "[*VisitLisboa Eventos*](https://www.visitlisboa.com/pt-pt/eventos)" in cleaned


def test_final_guard_keeps_event_card_fields_at_one_streamlit_safe_level() -> None:
    """Event cards should use a real list parent so Streamlit does not stair-step fields."""
    raw = """### 🎭 **Eventos Culturais**

**🎵 Gipsy Kings by Diego Baliardo | "A Tu Vera” Tour**
    - 📍 **Morada:** [Coliseu de Lisboa, Lisboa](https://www.google.com/maps/search/?api=1&query=Coliseu)
    - 📅 **Data/Hora:** 10 de maio
    - ⏱️ **Duração:** 🎯 Um só dia
    - 📂 **Categoria:** Música
    - 💰 **Preço:** de €25 a €50
    - 🌐 [Mais detalhes](https://www.visitlisboa.com/en/events/gipsy-kings)
    - 🎟️ [Bilhetes](https://tickets.example.com)

📌 **Fonte:** [*VisitLisboa Eventos*](https://www.visitlisboa.com/pt-pt/eventos) | **Atualizado:** 09:05"""

    cleaned = final_post_qa_guard(raw, language="pt")
    lines = cleaned.splitlines()
    start = lines.index('- **🎵 Gipsy Kings by Diego Baliardo | "A Tu Vera” Tour**')
    field_lines = [line for line in lines[start + 1:] if line.strip().startswith("- ")]

    assert field_lines[:7] == [
        "    - 📍 **Morada:** [Coliseu de Lisboa, Lisboa](https://www.google.com/maps/search/?api=1&query=Coliseu)",
        "    - 📅 **Data/Hora:** 10 de maio",
        "    - ⏱️ **Duração:** 🎯 Um só dia",
        "    - 📂 **Categoria:** Música",
        "    - 💰 **Preço:** de €25 a €50",
        "    - 🌐 [Mais detalhes](https://www.visitlisboa.com/en/events/gipsy-kings)",
        "    - 🎟️ [Bilhetes](https://tickets.example.com)",
    ]
    assert not any(line.startswith("        - ") for line in field_lines)


def test_final_guard_removes_event_source_completeness_note_from_description_field() -> None:
    """QA repair must not publish source-completeness notes as event descriptions."""
    raw = """### 🎭 **Eventos Culturais**

### 🎭 **Billy's Tour**
    - 📝 **Descrição:** 29 registo(s) adicional(is) compatíveis foram excluídos porque a fonte ainda não confirma a respetiva data.
    - 📍 **Morada:** [Poço do Borratém, Lisboa](https://www.google.com/maps/search/?api=1&query=Poco)
    - 📅 **Data/Hora:** 08 de maio a 10 de maio

📌 **Fonte:** [*VisitLisboa Eventos*](https://www.visitlisboa.com/pt-pt/eventos) | **Atualizado:** 10:36"""

    cleaned = final_post_qa_guard(raw, language="pt")

    assert "registo(s) adicional(is)" not in cleaned
    assert "    - 📍 **Morada:**" in cleaned


def test_final_guard_localizes_visitlisboa_source_footer_but_keeps_detail_url_canonical() -> None:
    """PT source footer may use PT VisitLisboa pages while card detail links keep canonical /en/ URLs."""
    raw = """### 🎭 **Eventos Culturais**

### 🎵 **Evento**
    - 🔗 **Mais detalhes:** [VisitLisboa](https://www.visitlisboa.com/en/events/example)

📌 **Fonte:** [*VisitLisboa Events*](https://www.visitlisboa.com/en/events) | **Atualizado:** 10:00"""

    cleaned = final_post_qa_guard(raw, language="pt")

    assert "https://www.visitlisboa.com/en/events/example" in cleaned
    assert "[*VisitLisboa Eventos*](https://www.visitlisboa.com/pt-pt/eventos)" in cleaned


def test_researcher_formatter_removes_event_intro_from_place_answers() -> None:
    """Place-only answers should not keep event headings or VisitLisboa Events source links."""
    raw = """### 🎭 **Cultural Events**
Here are the main cultural events I found in Lisbon:

**🏛️ National Tile Museum**
- 📝 **Description:** A museum dedicated to Portuguese tilework.
- 📂 **Category:** Museums
- 📍 **Address:** Rua da Madre de Deus, 4, Lisboa
- ✉️ **Email**: info@museudoazulejo.pt
- 🌐 **Website:** [Official website](https://www.museudoazulejo.gov.pt)

📌 **Source:** [*VisitLisboa Places*](https://www.visitlisboa.com/en/places) | [*VisitLisboa Events*](https://www.visitlisboa.com/en/events) | **Updated:** 18:45"""

    formatted = format_researcher_card(raw, language="en", user_query="Show me museums in Lisbon")
    finalized = canonicalize_visitlisboa_source_line(
        formatted,
        user_query="Show me museums in Lisbon",
        language="en",
    )

    assert "Cultural Events" not in finalized
    assert "Here are the main cultural events" not in finalized
    assert "### 🏛️ Recommended Places" in finalized
    assert "- ✉️ **Email:** [info@museudoazulejo.pt](mailto:info@museudoazulejo.pt)" in finalized
    assert "[*VisitLisboa Places*](https://www.visitlisboa.com/en/places)" in finalized
    assert "VisitLisboa Events" not in finalized


def test_researcher_reconciliation_restores_place_fields_lost_by_qa() -> None:
    """QA repair should not be allowed to drop enriched VisitLisboa place fields."""
    qa_collapsed = """### 🔵 **Places and Attractions**

### 🏛️ Maritime Museum

- 📝 **Description:** Maritime museum description.
- 📍 **Address:** [Praça do Império, Lisboa](https://www.google.com/maps/search/?api=1&query=Pra%C3%A7a+do+Imp%C3%A9rio)
- 🕒 **Hours:** Today: 10:00 - 18:00
- 🎟️ **Tickets:** [Buy tickets](https://tickets.example.com/maritime)

### 🏛️ Pavilion of Knowledge

- 📝 **Description:** Interactive science museum.
- 📍 **Address:** [Largo José Mariano Gago, Lisboa](https://www.google.com/maps/search/?api=1&query=Largo+Jos%C3%A9+Mariano+Gago)
- 🕒 **Hours:** Today: Closed
- ✉️ **Email:** [info@cienciaviva.pt](mailto:info@cienciaviva.pt)

📌 **Source:** [*VisitLisboa Places*](https://www.visitlisboa.com/en/places) | **Updated:** 19:32"""

    worker_output = """### 🔵 **Places and Attractions**

**🏛️ Maritime Museum**
    - 📝 **Description:** Maritime museum description.
    - 📍 **Address:** [Praça do Império, Lisboa](https://www.google.com/maps/search/?api=1&query=Pra%C3%A7a+do+Imp%C3%A9rio)
    - 🕒 **Hours:** Today: 10:00 - 18:00
    - 🎟️ **Tickets:** [Buy tickets](https://tickets.example.com/maritime)
    - 🌐 **Website:** [Official website](https://example.com/maritime)
    - 🔗 **More details:** [VisitLisboa](https://www.visitlisboa.com/en/places/maritime-museum)

**🏛️ Pavilion of Knowledge**
    - 📝 **Description:** Interactive science museum.
    - 📂 **Category:** Museums
    - 📍 **Address:** [Largo José Mariano Gago, Lisboa](https://www.google.com/maps/search/?api=1&query=Largo+Jos%C3%A9+Mariano+Gago)
    - 🕒 **Hours:** Today: Closed
    - 🎟️ **Tickets:** [Buy tickets](https://tickets.example.com/pavilion)
    - ✉️ **Email:** [info@cienciaviva.pt](mailto:info@cienciaviva.pt)
    - 🌐 **Website:** [Official website](https://www.cienciaviva.pt)
    - 🔗 **More details:** [VisitLisboa](https://www.visitlisboa.com/en/places/pavilion-of-knowledge)

📌 **Source:** [*VisitLisboa Places*](https://www.visitlisboa.com/en/places) | **Updated:** 19:32"""

    reconciled = reconcile_researcher_place_response(
        qa_collapsed,
        worker_output,
        language="en",
        user_query="Show me two museums in Lisbon with opening hours and ticket links.",
    )

    assert "### 🏛️ Pavilion of Knowledge" in reconciled
    assert "- 🎟️ **Tickets:** [Buy tickets](https://tickets.example.com/pavilion)" in reconciled
    assert "- 🌐 **Website:** [Official website](https://www.cienciaviva.pt)" in reconciled
    assert "- 🔗 **More details:** [VisitLisboa](https://www.visitlisboa.com/en/places/pavilion-of-knowledge)" in reconciled


def test_final_visual_pass_keeps_repeated_place_card_fields() -> None:
    """Repeated labels in separate place cards are not duplicate bullets."""
    raw = """### 🔵 **Places and Attractions**

### 🏛️ Maritime Museum

- 📂 **Category:** Museums
- 🌐 **Website:** [Official website](https://example.com/maritime)
- 🎟️ **Tickets:** [Buy tickets](https://tickets.example.com/maritime)
- 🔗 **More details:** [VisitLisboa](https://www.visitlisboa.com/en/places/maritime-museum)

### 🏛️ Pavilion of Knowledge

- 📂 **Category:** Museums
- 🌐 **Website:** [Official website](https://example.com/pavilion)
- 🎟️ **Tickets:** [Buy tickets](https://tickets.example.com/pavilion)
- 🔗 **More details:** [VisitLisboa](https://www.visitlisboa.com/en/places/pavilion-of-knowledge)"""

    cleaned = final_visual_pass(raw)

    assert cleaned.count("**Category:** Museums") == 2
    assert cleaned.count("**Website:** [Official website]") == 2
    assert cleaned.count("**Tickets:** [Buy tickets]") == 2
    assert cleaned.count("**More details:** [VisitLisboa]") == 2


def test_final_guard_keeps_place_card_fields_at_one_streamlit_safe_level() -> None:
    """Place cards should use a list parent so Streamlit does not stair-step fields."""
    raw = """### 🔵 **Locais e atrações**

### 🏛️ Castle Museum
    - 📂 **Categoria:** Museus
    - 📍 **Morada:** [Castelo de São Jorge, Lisboa](https://www.google.com/maps/search/?api=1&query=Castelo)
    - 💶 **Preço:** Gratuito com Lisboa Card
    - ⭐ **Avaliação:** TripAdvisor 4.2/5
    - 📞 **Telefone:** [+351 218 800 620](tel:+351218800620)
    - ✉️ **Email:** [info@example.pt](mailto:info@example.pt)
    - 🌐 **Website:** [Website oficial](https://example.pt)
    - 🔗 **Mais detalhes:** [VisitLisboa](https://www.visitlisboa.com/en/places/castle-museum)

📌 **Fonte:** [*VisitLisboa Locais*](https://www.visitlisboa.com/pt-pt/locais) | **Atualizado:** 10:46"""

    cleaned = final_visual_pass(raw)
    lines = cleaned.splitlines()
    start = lines.index("- **🏛️ Castle Museum**")
    field_lines = [line for line in lines[start + 1:] if line.strip().startswith("- ")]

    assert field_lines[:8] == [
        "    - 📂 **Categoria:** Museus",
        "    - 📍 **Morada:** [Castelo de São Jorge, Lisboa](https://www.google.com/maps/search/?api=1&query=Castelo)",
        "    - 💶 **Preço:** Gratuito com Lisboa Card",
        "    - ⭐ **Avaliação:** TripAdvisor 4.2/5",
        "    - 📞 **Telefone:** [+351 218 800 620](tel:+351218800620)",
        "    - ✉️ **Email:** [info@example.pt](mailto:info@example.pt)",
        "    - 🌐 **Website:** [Website oficial](https://example.pt)",
        "    - 🔗 **Mais detalhes:** [VisitLisboa](https://www.visitlisboa.com/en/places/castle-museum)",
    ]
    assert not any(line.startswith("        - ") for line in field_lines)


def test_researcher_localizes_unknown_english_place_titles_with_llm_cache() -> None:
    """ResearcherAgent should localize new English VisitLisboa place titles in PT cards."""
    agent = object.__new__(ResearcherAgent)
    agent.llm = object()
    agent._place_title_localization_cache = {}
    calls = []

    def fake_safe_llm_invoke(llm, messages, retries=2, verbose=False):  # noqa: ANN001, ANN202
        calls.append(messages)
        return _DummyLLMResponse("Museu de Marinha")

    agent._safe_llm_invoke = fake_safe_llm_invoke
    text = """### 🔵 **Locais e atrações**

- **🏛️ Maritime Museum**
    - 📂 **Categoria:** Museus
    - 🔗 **Mais detalhes:** [VisitLisboa](https://www.visitlisboa.com/en/places/maritime-museum)"""

    first = agent._localize_place_card_titles_with_llm(text, "pt")
    second = agent._localize_place_card_titles_with_llm(text, "pt")

    assert "- **🏛️ Museu de Marinha**" in first
    assert "Maritime Museum" not in first
    assert "    - 📂 **Categoria:** Museus" in first
    assert second == first
    assert len(calls) == 1


def test_researcher_keeps_place_title_when_llm_is_not_confident() -> None:
    """Uncertain or invalid LLM localizations must not corrupt official place titles."""
    agent = object.__new__(ResearcherAgent)
    agent.llm = object()
    agent._place_title_localization_cache = {}

    def fake_safe_llm_invoke(llm, messages, retries=2, verbose=False):  # noqa: ANN001, ANN202
        return _DummyLLMResponse("SAME")

    agent._safe_llm_invoke = fake_safe_llm_invoke
    text = """### 🔵 **Locais e atrações**

- **🏛️ Design Museum**
    - 📂 **Categoria:** Museus
    - 🔗 **Mais detalhes:** [VisitLisboa](https://www.visitlisboa.com/en/places/design-museum)"""

    localized = agent._localize_place_card_titles_with_llm(text, "pt")

    assert "- **🏛️ Design Museum**" in localized


def test_researcher_does_not_localize_place_titles_in_english_answers() -> None:
    """English answers should keep English place titles and avoid localizer LLM calls."""
    agent = object.__new__(ResearcherAgent)
    agent.llm = object()
    agent._place_title_localization_cache = {}
    calls = []

    def fake_safe_llm_invoke(llm, messages, retries=2, verbose=False):  # noqa: ANN001, ANN202
        calls.append(messages)
        return _DummyLLMResponse("Museu de Marinha")

    agent._safe_llm_invoke = fake_safe_llm_invoke
    text = """### 🔵 **Places and Attractions**

- **🏛️ Maritime Museum**
    - 📂 **Category:** Museums
    - 🔗 **More details:** [VisitLisboa](https://www.visitlisboa.com/en/places/maritime-museum)"""

    localized = agent._localize_place_card_titles_with_llm(text, "en")

    assert localized == text
    assert calls == []


def test_place_response_missing_requested_fields_detects_partial_ticket_loss() -> None:
    """A multi-card place answer must keep ticket fields on every card when requested."""
    partial = """### 🔵 **Places and Attractions**

### 🏛️ Maritime Museum

- 🕐 **Opening hours:** Today: 10:00 - 18:00
- 🎟️ **Tickets:** [Buy tickets](https://tickets.example.com/maritime)

### 🏛️ Pavilion of Knowledge

- 🕐 **Opening hours:** Today: Closed
- 🌐 **Website:** [Official website](https://example.com/pavilion)"""

    assert researcher_place_response_missing_requested_fields(
        partial,
        user_query="Show me two museums with opening hours and ticket links",
        language="en",
    )


def test_lisbon_scoped_place_geography_checks_title_url_and_location() -> None:
    """Lisbon-only place queries should filter obvious AML candidates even with generic locations."""
    sintra_text = "Sintra Myths and Legends Interactive Centre https://www.visitlisboa.com/en/places/sintra-myths Lisbon"
    lisbon_text = "Miradouro de Santa Luzia https://www.visitlisboa.com/en/places/miradouro-de-santa-luzia Lisbon"

    assert visitlisboa_api._place_within_requested_geography(lisbon_text, "atrações imperdíveis em Lisboa")
    assert not visitlisboa_api._place_within_requested_geography(sintra_text, "atrações imperdíveis em Lisboa")
    assert visitlisboa_api._place_within_requested_geography(sintra_text, "atrações em Sintra")


def test_plain_torre_does_not_force_belem_alias() -> None:
    """Only explicit Belém tower aliases should map to Torre de Belém."""
    assert visitlisboa_api._apply_known_place_lookup_alias("Torre Vasco da Gama") == "Torre Vasco da Gama"
    assert visitlisboa_api._apply_known_place_lookup_alias("Torre do Tombo") == "Torre do Tombo"
    assert visitlisboa_api._apply_known_place_lookup_alias("Tour de Belém") == "Torre de Belém"


def test_broad_museum_queries_are_not_treated_as_specific_place_misses() -> None:
    """Broad museum discovery should not produce an exact-place miss for fragments such as 'museums in'."""
    assert visitlisboa_api._extract_specific_place_lookup_phrase(
        "Show me museums in Lisbon with opening hours and ticket links"
    ) is None
    assert visitlisboa_api._extract_specific_place_lookup_phrase(
        "Show me two museums in Lisbon with opening hours and ticket links"
    ) is None
