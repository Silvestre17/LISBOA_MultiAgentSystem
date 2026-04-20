# ==========================================================================
# Master Thesis - VisitLisboa API Tests
#   - André Filipe Gomes Silvestre, 20240502
#
#   Focused regressions for VisitLisboa runtime helpers.
# ==========================================================================

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
