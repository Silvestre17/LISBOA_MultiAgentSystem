# ==========================================================================
# Master Thesis - VisitLisboa API Tests
#   - André Filipe Gomes Silvestre, 20240502
#
#   Focused regressions for VisitLisboa runtime helpers.
# ==========================================================================

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from unittest.mock import patch

import tools.visitlisboa_api as visitlisboa_api


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
    assert "with Lisboa Card" not in visitlisboa_api._localize_place_value_text("20% with Lisboa Card", "pt")


def test_generic_visitlisboa_location_becomes_maps_search_link() -> None:
    """Generic Lisbon-only locations should become Maps searches for the specific place."""
    line = visitlisboa_api._format_visitlisboa_location_line("Lisbon", "Gulbenkian Museum", language="pt")

    assert "Lisbon" not in line
    assert "Pesquisar no Maps" in line
    assert "Gulbenkian+Museum+Lisboa" in line
