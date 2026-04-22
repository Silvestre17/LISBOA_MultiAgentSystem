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
