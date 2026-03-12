# ==========================================================================
# Master Thesis - Config & Audit Fix Tests
#   - André Filipe Gomes Silvestre, 20240502
#
# Tests that validate audit fixes are correctly applied:
#   - A4: MARDKOWN typo corrected to MARKDOWN
#   - A6: No extra LLM instance in LisbonAssistant
#   - A7: No import json inside loops
#   - Config loads without errors
#
# Run from the repository root with a relative path:
#   python -m pytest tests/test_audit_fixes.py -q
# Useful parameters:
#   -vv         verbose test names
#   -k <expr>   run a subset, e.g. -k metro_ssl
#   -x          stop on first failure
#   --tb=short  shorter tracebacks
# Notes:
#   - Prefer relative paths in this workspace. Absolute pytest paths may be
#     treated as glob patterns on Windows because the folder name includes
#     `[` and `]`.
# ==========================================================================

import inspect
import os
import re
import sqlite3
import sys
import time
from typing import Generator
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ==========================================================================
# Test Result Tracking (matches test_lisbon_transport.py pattern)
# ==========================================================================

class TestResults:
    """Tracks test results with pass/fail counts."""

    __test__ = False
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.start_time = time.time()
    
    def add_pass(self, test_name):
        self.passed += 1
        print(f"  \033[1;32m✅ PASS\033[0m: {test_name}")
    
    def add_fail(self, test_name, reason):
        self.failed += 1
        self.errors.append(f"{test_name}: {reason}")
        print(f"  \033[1;31m❌ FAIL\033[0m: {test_name} - {reason}")
    
    def summary(self):
        total = self.passed + self.failed
        elapsed = time.time() - self.start_time
        print("\n" + "=" * 70)
        print("\033[1m📊 TEST SUMMARY\033[0m")
        print("=" * 70)
        print(f"\033[1;32m✅ Passed: {self.passed}/{total}\033[0m")
        print(f"\033[1;31m❌ Failed: {self.failed}/{total}\033[0m")
        print(f"⏱️ Duration: {elapsed:.2f}s")
        
        if self.errors:
            print("\n\033[1;31mFailed tests:\033[0m")
            for err in self.errors:
                print(f"  • {err}")
        
        if self.failed == 0:
            print("\n\033[1;32m🎉 ALL TESTS PASSED!\033[0m")
        
        print("=" * 70 + "\n")
        return self.failed


@pytest.fixture
def results() -> Generator[TestResults, None, None]:
    """Provide a per-test result tracker that fails the pytest test when needed."""
    tracker = TestResults()
    yield tracker
    if tracker.failed:
        pytest.fail(
            "Audit validation checks failed:\n- " + "\n- ".join(tracker.errors)
        )


# ==========================================================================
# A4: Config typo fix validation
# ==========================================================================

def test_config_typo_fix(results: TestResults):
    """Tests that the MARDKOWN typo is corrected."""
    print("\n\033[1m📋 A4: Config Typo Fix (MARDKOWN → MARKDOWN)\033[0m")
    print("-" * 50)
    
    try:
        from config import Config

        # Check that the corrected attribute exists
        if hasattr(Config, 'SHOW_MARKDOWN_RESPONSE_IN_TERMINAL'):
            results.add_pass("Config.SHOW_MARKDOWN_RESPONSE_IN_TERMINAL exists")
        else:
            results.add_fail("SHOW_MARKDOWN_RESPONSE_IN_TERMINAL", "Attribute not found in Config")
        
        # Check that the OLD typo does NOT exist
        if hasattr(Config, 'SHOW_MARDKOWN_RESPONSE_IN_TERMINAL'):
            results.add_fail("SHOW_MARDKOWN_RESPONSE_IN_TERMINAL", "Old typo still exists!")
        else:
            results.add_pass("Old typo SHOW_MARDKOWN_RESPONSE_IN_TERMINAL removed")
        
        # Check the value is boolean
        val = Config.SHOW_MARKDOWN_RESPONSE_IN_TERMINAL
        if isinstance(val, bool):
            results.add_pass(f"SHOW_MARKDOWN_RESPONSE_IN_TERMINAL is bool ({val})")
        else:
            results.add_fail("SHOW_MARKDOWN_RESPONSE_IN_TERMINAL type", f"Expected bool, got {type(val)}")
            
    except Exception as e:
        results.add_fail("Config import", str(e))


# ==========================================================================
# A4: BaseAgent reference fix validation
# ==========================================================================

def test_base_agent_typo_fix(results: TestResults):
    """Tests that BaseAgent references the corrected config name."""
    print("\n\033[1m📋 A4: BaseAgent Config Reference Fix\033[0m")
    print("-" * 50)
    
    try:
        # Read the source file and check for old typo
        base_path = os.path.join(os.path.dirname(__file__), '..', 'agent', 'agents', 'base.py')
        with open(base_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'SHOW_MARDKOWN_RESPONSE_IN_TERMINAL' in content:
            results.add_fail("BaseAgent source", "Still contains MARDKOWN typo")
        else:
            results.add_pass("BaseAgent source has no MARDKOWN typo")
        
        if 'SHOW_MARKDOWN_RESPONSE_IN_TERMINAL' in content:
            results.add_pass("BaseAgent uses corrected SHOW_MARKDOWN_RESPONSE_IN_TERMINAL")
        else:
            results.add_fail("BaseAgent source", "Missing MARKDOWN reference")
            
    except Exception as e:
        results.add_fail("BaseAgent source check", str(e))


# ==========================================================================
# A7: import json not inside loops
# ==========================================================================

def test_no_import_json_in_loops(results: TestResults):
    """Tests that any json import stays at top level and never inside loops."""
    print("\n\033[1m📋 A7: No import json inside loops\033[0m")
    print("-" * 50)
    
    files_to_check = [
        ('agent/agents/researcher_agent.py', 'ResearcherAgent'),
        ('agent/agents/transport_agent.py', 'TransportAgent'),
    ]
    
    for filepath, name in files_to_check:
        try:
            full_path = os.path.join(os.path.dirname(__file__), '..', filepath)
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            json_import_lines = [
                i
                for i, line in enumerate(lines, start=1)
                if line.strip() == 'import json'
            ]

            if not json_import_lines:
                results.add_pass(f"{name}: json import not needed")
            else:
                top_level_json = any(
                    line.strip() == 'import json' and not line.startswith((' ', '\t'))
                    for line in lines[:40]
                )

                if top_level_json:
                    results.add_pass(f"{name}: import json at top level")
                else:
                    results.add_fail(
                        f"{name}: import json",
                        "Found import json, but not at file top level",
                    )
            
            # Check no 'import json' inside indented blocks (loop bodies)
            inside_loop_import = False
            for i, line in enumerate(lines[20:], start=21):
                stripped = line.strip()
                if stripped == 'import json' and line.startswith((' ', '\t')):
                    inside_loop_import = True
                    results.add_fail(f"{name}: import json", f"Found inside block at line {i}")
                    break
            
            if not inside_loop_import:
                results.add_pass(f"{name}: no import json inside loops")
                
        except Exception as e:
            results.add_fail(f"{name} check", str(e))


# ==========================================================================
# A3: No duplicate OpenAI elif block
# ==========================================================================

def test_no_duplicate_openai_block(results: TestResults):
    """Tests that app.py has no duplicate OpenAI provider block."""
    print("\n\033[1m📋 A3: No Duplicate OpenAI Block in app.py\033[0m")
    print("-" * 50)
    
    try:
        app_path = os.path.join(os.path.dirname(__file__), '..', 'app.py')
        with open(app_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count occurrences of the OpenAI elif pattern
        count = content.count('elif selected_provider == "openai":')
        
        if count == 1:
            results.add_pass(f"app.py: exactly 1 OpenAI elif block (found {count})")
        elif count == 0:
            results.add_fail("app.py OpenAI block", "No OpenAI elif block found (expected 1)")
        else:
            results.add_fail("app.py OpenAI block", f"Found {count} OpenAI elif blocks (expected 1)")
            
    except Exception as e:
        results.add_fail("app.py check", str(e))


# ==========================================================================
# A2: ResearcherAgent uses full tool set
# ==========================================================================

def test_researcher_tools(results: TestResults):
    """Tests that ResearcherAgent gets all tools from base class."""
    print("\n\033[1m📋 A2: ResearcherAgent Tool Loading\033[0m")
    print("-" * 50)
    
    try:
        from agent.agents.base import get_agent_tools
        from agent.agents.researcher_agent import ResearcherAgent

        # Check that we get tools from get_agent_tools
        expected_tools = get_agent_tools("researcher")
        expected_count = len(expected_tools)
        
        if expected_count >= 10:
            results.add_pass(f"get_agent_tools('researcher') returns {expected_count} tools (>= 10)")
        else:
            results.add_fail("Tool count", f"Expected >= 10 tools, got {expected_count}")
        
        # Verify no hardcoded tool list in __init__
        source_path = os.path.join(os.path.dirname(__file__), '..', 'agent', 'agents', 'researcher_agent.py')
        with open(source_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Check that __init__ doesn't have a hardcoded self.tools = [...] list
        init_match = re.search(r'def __init__\(self.*?\n(.*?)(?=\n    def )', source, re.DOTALL)
        if init_match:
            init_body = init_match.group(1)
            if 'self.tools = [' in init_body:
                results.add_fail("ResearcherAgent.__init__", "Still has hardcoded self.tools = [...]")
            else:
                results.add_pass("ResearcherAgent.__init__ has no hardcoded tool list")
        else:
            results.add_pass("ResearcherAgent.__init__ check (no override pattern found)")
            
    except Exception as e:
        results.add_fail("ResearcherAgent tools", str(e))


# ==========================================================================
# A6: No extra LLM instance in LisbonAssistant
# ==========================================================================

def test_no_extra_llm_instance(results: TestResults):
    """Tests that LisbonAssistant.__init__ doesn't create unnecessary LLM."""
    print("\n\033[1m📋 A6: No Extra LLM in LisbonAssistant\033[0m")
    print("-" * 50)
    
    try:
        graph_path = os.path.join(os.path.dirname(__file__), '..', 'agent', 'graph.py')
        with open(graph_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check that LisbonAssistant.__init__ doesn't call LLMFactory.get_llm
        init_match = re.search(
            r'class LisbonAssistant.*?def __init__\(self.*?\n(.*?)(?=\n    def |\nclass )',
            content, re.DOTALL
        )
        
        if init_match:
            init_body = init_match.group(1)
            if 'LLMFactory.get_llm' in init_body:
                results.add_fail("LisbonAssistant.__init__", "Still calls LLMFactory.get_llm()")
            else:
                results.add_pass("LisbonAssistant.__init__ does NOT call LLMFactory.get_llm()")
            
            if 'Config' in init_body or 'Config.' in content:
                results.add_pass("Uses Config directly for model info")
        else:
            results.add_fail("LisbonAssistant", "Could not find __init__ in source")
            
    except Exception as e:
        results.add_fail("LisbonAssistant check", str(e))


# ==========================================================================
# Transport improvements: B1 & B4
# ==========================================================================

def test_transport_improvements(results: TestResults):
    """Tests travel time estimation and metro status integration."""
    print("\n\033[1m📋 B1/B4: Transport Improvements\033[0m")
    print("-" * 50)
    
    try:
        from tools.transport_api import (
            _count_metro_stations,
            _estimate_metro_time,
            _get_line_status,
            get_route_between_stations,
        )

        # B4: Station counting
        count = _count_metro_stations("amarela", "rato", "odivelas")
        if count == 12:
            results.add_pass(f"_count_metro_stations: Rato->Odivelas = {count}")
        else:
            results.add_fail("Station count", f"Expected 12, got {count}")
        
        # B4: Time estimation
        time_est = _estimate_metro_time(5, transfers=0)
        if "12" in time_est:
            results.add_pass(f"_estimate_metro_time(5, 0) = {time_est}")
        else:
            results.add_fail("Time estimate", f"Expected ~12 min, got {time_est}")
        
        time_est = _estimate_metro_time(8, transfers=1)
        if "21" in time_est:
            results.add_pass(f"_estimate_metro_time(8, 1) = {time_est}")
        else:
            results.add_fail("Time estimate with transfer", f"Expected ~21 min, got {time_est}")
        
        # B1: Line status (just verify it returns something)
        status = _get_line_status("verde")
        if status:
            results.add_pass(f"_get_line_status('verde') = '{status}'")
        else:
            results.add_fail("Line status", "Empty result")
        
        # Verify route output includes time estimate
        route = get_route_between_stations.invoke({
            "origin": "Aeroporto", "destination": "Saldanha"
        })
        if "Estimated travel time" in route:
            results.add_pass("Route output includes travel time estimate")
        else:
            results.add_fail("Route output", "Missing 'Estimated travel time'")
            
    except Exception as e:
        results.add_fail("Transport improvements", str(e))


# ==========================================================================
# B5: Transport prompt includes bus examples
# ==========================================================================

def test_transport_prompt(results: TestResults):
    """Tests that transport prompt includes bus/tram routing examples."""
    print("\n\033[1m📋 B5: Transport Prompt Bus/Tram Examples\033[0m")
    print("-" * 50)
    
    try:
        from agent.prompts.transport import get_transport_prompt
        
        prompt = get_transport_prompt()
        
        checks = [
            (("carris_find_routes_between",), "Carris Urbana tool"),
            (("find_direct_bus_lines",), "Carris Metropolitana tool"),
            (("find_bus_routes",), "GPS-based bus tool"),
            (("get_transport_summary",), "Transport summary tool"),
            (("Tempo total estimado", "Tempo estimado"), "Travel time in template"),
            (("temporarily unavailable",), "Missing-data guardrail"),
            (("Carris Urban", "Carris Metropolitana (Suburban)"), "Operator labeling guidance"),
            (("Carris Urban-only",), "Carris Metropolitana scope nuance"),
        ]
        
        for terms, description in checks:
            if any(term in prompt for term in terms):
                results.add_pass(f"Prompt includes {description}")
            else:
                results.add_fail(
                    "Prompt missing",
                    f"one of {terms} ({description})",
                )
                
    except Exception as e:
        results.add_fail("Transport prompt", str(e))


def test_search_history_culture_prefers_wikipedia_for_background_queries() -> None:
    """Non-live history queries should prefer encyclopedia sources before broad web snippets."""
    from tools import web_knowledge

    with patch.object(web_knowledge, "_search_wikipedia_with_fallback", return_value="wiki result") as wiki_mock, patch.object(
        web_knowledge,
        "_search_tavily",
        side_effect=AssertionError("Tavily should not run before Wikipedia for background queries"),
    ), patch.object(
        web_knowledge,
        "_search_duckduckgo",
        side_effect=AssertionError("DuckDuckGo should not run before Wikipedia for background queries"),
    ):
        result = web_knowledge.search_history_culture.invoke(
            {"query": "History of Castelo de São Jorge", "language": "en"}
        )

    wiki_mock.assert_called_once()
    assert result == "wiki result"


def test_search_history_culture_english_wikipedia_fallback_note() -> None:
    """Wikipedia fallback should explicitly disclose when it had to switch to English."""
    from tools import web_knowledge

    with patch.object(web_knowledge, "_search_wikipedia", side_effect=[None, "english wiki result"]):
        result = web_knowledge._search_wikipedia_with_fallback("Castelo de São Jorge", "pt")

    assert result is not None
    assert "english wiki result" in result
    assert "fallback" in result.lower() or "fallback da fonte" in result.lower()


def test_search_history_culture_live_query_adds_verification_note() -> None:
    """Live web queries should carry an explicit temporal verification reminder."""
    from tools import web_knowledge

    with patch.object(web_knowledge, "_search_tavily", return_value="official live result"):
        result = web_knowledge.search_history_culture.invoke(
            {"query": "Metro strike in Lisbon today", "language": "en"}
        )

    assert "official live result" in result
    assert "confirm the date and time" in result.lower()


def test_search_history_culture_live_query_filters_stale_web_results() -> None:
    """Live-search formatting should suppress stale generic web results when fresher authoritative ones exist."""
    from tools import web_knowledge

    fake_results = [
        {
            "url": "https://www.metrolisboa.pt/institucional/2026/03/09/greve-hoje/",
            "content": "Mar 9, 2026 Metro service notice for today.",
        },
        {
            "url": "https://www.bbc.com/news/articles/old-strike",
            "content": "May 13, 2025 historic strike article.",
        },
    ]

    with patch.object(web_knowledge, "TavilySearchResults") as tavily_tool:
        tavily_tool.return_value.invoke.return_value = fake_results
        result = web_knowledge.search_history_culture.invoke(
            {"query": "Metro strike in Lisbon today", "language": "en"}
        )

    assert "2026-03-09" in result
    assert "bbc.com" not in result


def test_search_cultural_events_reports_matching_undated_candidates() -> None:
    """Date-filtered event queries should disclose undated matches without leaking placeholder event rows."""
    from datetime import datetime

    from tools import visitlisboa_api

    today_iso = datetime.now().strftime("%Y-%m-%d")
    fake_events = [
        {
            "title": "Jazz by the River",
            "category": "Music",
            "dates": [
                {
                    "type": "single",
                    "date": {
                        "datetime_iso": today_iso,
                        "display_text": "Today",
                        "time": "21:00",
                    },
                }
            ],
            "short_description": "Live jazz performance.",
        },
        {
            "title": "Fado Mystery Session",
            "category": "Music",
            "dates": [],
            "short_description": "Live fado with date still to be confirmed.",
            "url": "https://example.com/fado-mystery-session",
        },
    ]

    with patch.object(visitlisboa_api, "_load_events_json", return_value=fake_events):
        result = visitlisboa_api.search_cultural_events.invoke(
            {"query": "fado", "date_filter": "today", "max_results": 5}
        )

    assert "Fado Mystery Session" not in result
    assert "source completeness note" in result.lower()
    assert "excluded because the source does not confirm their dates yet" in result.lower()


def test_search_cultural_events_sanitizes_slug_like_title_suffixes() -> None:
    """Slug-derived titles should not leak numeric suffixes like 0326 into the final tool output."""
    from datetime import datetime

    from tools import visitlisboa_api

    today_iso = datetime.now().strftime("%Y-%m-%d")
    fake_events = [
        {
            "url": "https://www.visitlisboa.com/en/events/michael-lives-forever-0326",
            "category": "Music",
            "dates": [
                {
                    "type": "single",
                    "date": {
                        "datetime_iso": today_iso,
                        "display_text": "Today",
                        "time": "21:00",
                    },
                }
            ],
            "full_description": "The show includes classic songs such as Billie Jean and Thriller.",
            "location": "Campo Pequeno, Lisboa",
        }
    ]

    with patch.object(visitlisboa_api, "_load_events_json", return_value=fake_events):
        result = visitlisboa_api.search_cultural_events.invoke(
            {"date_filter": "today", "max_results": 5, "language": "pt"}
        )

    assert "Michael Lives Forever 0326" not in result
    assert "Michael Lives Forever" in result
    assert "**Descrição:**" in result


def test_search_places_attractions_uses_json_fallback_for_location_queries() -> None:
    """Location-specific museum queries should fall back to JSON data when vector retrieval under-recovers."""
    from tools import visitlisboa_api

    class EmptyKB:
        def search_with_scores(self, query, k, collections):
            return []

    fake_places = [
        {
            "title": "National Coach Museum",
            "category": "Museums & Monuments",
            "location": "Belém, Lisbon",
            "short_description": "Museum in Belém with royal coaches.",
            "url": "https://www.visitlisboa.com/en/places/national-coach-museum",
        }
    ]

    with patch.object(visitlisboa_api, "_get_vector_store", return_value=EmptyKB()), patch.object(
        visitlisboa_api,
        "_load_places_json",
        return_value=fake_places,
    ), patch.object(
        visitlisboa_api,
        "_should_search_dados_abertos",
        return_value=False,
    ), patch.object(visitlisboa_api, "_get_place_by_url", return_value=None):
        result = visitlisboa_api.search_places_attractions.invoke(
            {"query": "belem museums", "category": "Museums & Monuments", "max_results": 5}
        )

    assert "National Coach Museum" in result
    assert "No places found" not in result


def test_search_stop_rows_matches_accentless_queries() -> None:
    """Carris stop lookup should work even when the user omits Portuguese accents."""
    from tools import carris_api

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE stops (stop_id TEXT, stop_name TEXT, stop_lat REAL, stop_lon REAL, stop_code TEXT)"
    )
    conn.execute(
        "INSERT INTO stops VALUES (?, ?, ?, ?, ?)",
        ("1", "Praça da Figueira", 38.713, -9.137, "001"),
    )
    conn.commit()

    rows = carris_api._search_stop_rows(conn, "Praca da Figueira", limit=5)

    assert rows
    assert rows[0]["stop_name"] == "Praça da Figueira"
    conn.close()


def test_resolve_carris_headsign_uses_direction_id_when_trip_headsign_missing() -> None:
    """Carris route labels should collapse to the correct terminal when only route_long_name is available."""
    from tools import carris_api

    assert carris_api._resolve_carris_headsign(None, "Hosp. Sta. Maria - Caselas", 0) == "Caselas"
    assert carris_api._resolve_carris_headsign(None, "Hosp. Sta. Maria - Caselas", 1) == "Hosp. Sta. Maria"


def test_search_places_attractions_service_query_filters_irrelevant_hotel_matches() -> None:
    """Service-heavy queries should not keep hotel results just because they share a generic location token."""
    from langchain_core.documents import Document

    from tools import visitlisboa_api

    class DummyKB:
        def search_with_scores(self, query, k, collections):
            return [
                (
                    Document(
                        page_content="Name: Hospital da Luz Lisboa\nCategory: General\nShort Description: Large private hospital in Lisbon.",
                        metadata={
                            "title": "Hospital da Luz Lisboa",
                            "category": "General",
                            "url": "https://www.visitlisboa.com/en/places/hospital-da-luz-lisboa",
                            "location": "Lisbon",
                        },
                    ),
                    0.35,
                ),
                (
                    Document(
                        page_content="Name: VIP Executive Santa Iria Hotel\nCategory: Hotel\nShort Description: Hotel in Santa Iria.",
                        metadata={
                            "title": "VIP Executive Santa Iria Hotel",
                            "category": "Hotel",
                            "url": "https://www.visitlisboa.com/en/places/vip-executive-santa-iria-hotel",
                            "location": "Lisbon",
                        },
                    ),
                    0.38,
                ),
            ]

    with patch.object(visitlisboa_api, "_get_vector_store", return_value=DummyKB()), patch.object(
        visitlisboa_api,
        "_should_search_dados_abertos",
        return_value=False,
    ), patch.object(visitlisboa_api, "_load_places_json", return_value=[]), patch.object(
        visitlisboa_api,
        "_get_place_by_url",
        return_value=None,
    ):
        result = visitlisboa_api.search_places_attractions.invoke(
            {"query": "hospital santa maria", "max_results": 5}
        )

    assert "Hospital da Luz Lisboa" in result
    assert "VIP Executive Santa Iria Hotel" not in result


def test_search_places_attractions_strips_raw_metadata_lines_from_tool_output() -> None:
    """Direct place-search tool output should not leak raw Name/Url/Category scaffolding."""
    from langchain_core.documents import Document

    from tools import visitlisboa_api

    class DummyKB:
        def search_with_scores(self, query, k, collections):
            return [
                (
                    Document(
                        page_content=(
                            "Name: Museum of the Lisbon Geographical Society\n"
                            "Url: https://www.visitlisboa.com/en/places/museum-of-the-lisbon-geographical-society\n"
                            "Category: Museums\n"
                            "Short Description: Be a 21st-century explorer."
                        ),
                        metadata={
                            "title": "Museum of the Lisbon Geographical Society",
                            "category": "Museums",
                            "url": "https://www.visitlisboa.com/en/places/museum-of-the-lisbon-geographical-society",
                            "location": "Lisbon",
                        },
                    ),
                    0.25,
                )
            ]

    with patch.object(visitlisboa_api, "_get_vector_store", return_value=DummyKB()), patch.object(
        visitlisboa_api,
        "_should_search_dados_abertos",
        return_value=False,
    ), patch.object(visitlisboa_api, "_load_places_json", return_value=[]), patch.object(
        visitlisboa_api,
        "_get_place_by_url",
        return_value=None,
    ):
        result = visitlisboa_api.search_places_attractions.invoke(
            {"query": "best museums in lisbon", "category": "Museums & Monuments", "max_results": 5}
        )

    assert "Name:" not in result
    assert "Url:" not in result
    assert "Short Description:" not in result
    assert "Be a 21st-century explorer." in result


def test_open_data_place_scoring_prefers_exact_named_facilities() -> None:
    """Named open-data facilities should outrank generic partial matches."""
    from tools import visitlisboa_api

    exact_score = visitlisboa_api._score_open_data_place_match(
        "hospital santa maria",
        "Hospital de Santa Maria",
        "Avenida Prof. Egas Moniz, Lisboa",
    )
    generic_score = visitlisboa_api._score_open_data_place_match(
        "hospital santa maria",
        "Hospital Egas Moniz",
        "Rua da Junqueira, Lisboa",
    )

    assert exact_score > generic_score


def test_extract_wait_times_for_direction_accepts_platform_note_line() -> None:
    """TransportAgent should still parse wait times when the Metro tool explains a platform-label fallback."""
    from agent.agents.transport_agent import _extract_wait_times_for_direction

    wait_result = (
        "🟡 Direction: Odivelas\n"
        "   ℹ️ Platform indicator currently shows Campo Grande.\n"
        "   ⏱️ Next train: 4 min 10s\n"
        "   ⏳ Following: 9 min 00s, 14 min 20s\n"
    )

    extracted = _extract_wait_times_for_direction(wait_result, "Odivelas")

    assert extracted == "4 min 10s | 9 min 00s"


def test_carris_metropolitana_realtime_response_includes_scope_and_freshness() -> None:
    """Realtime suburban bus answers should expose freshness and scope instead of looking city-wide by default."""
    from tools import carrismetropolitana_api

    carrismetropolitana_api._vehicle_feed_meta.update(
        {
            "source": "live",
            "generated_at": None,
            "data_age_seconds": 0,
            "last_error": None,
            "missing_coordinates": 2,
            "vehicle_count": 1,
        }
    )

    fake_vehicle = {
        "id": "veh-1",
        "line_id": "1001",
        "lat": 38.75,
        "lon": -9.18,
        "bearing": 90,
        "speed": 24,
        "timestamp": int(time.time() * 1000),
        "current_status": "IN_TRANSIT_TO",
        "license_plate": "AA-00-BB",
        "vehicle_model": "Mercedes",
    }
    fake_lines = [{"id": "1001", "short_name": "1001", "long_name": "Sintra - Lisboa"}]

    with patch.object(carrismetropolitana_api, "load_carris_metropolitana_vehicles", return_value=[fake_vehicle]), patch.object(
        carrismetropolitana_api,
        "load_carris_metropolitana_lines",
        return_value=fake_lines,
    ):
        result = carrismetropolitana_api.get_real_time_bus_positions.invoke({"line_id": "1001"})

    assert "Data freshness" in result
    assert "missing_coordinates" not in result
    assert "Scope: Carris Metropolitana covers AML metropolitan / intermunicipal buses" in result
    assert "entering Lisbon municipality" in result
    assert "Carris Urban" in result


def test_search_carris_metropolitana_lines_hides_numeric_municipality_codes() -> None:
    """Carris Metropolitana line search should not expose raw municipality codes when no human-readable names are available."""
    from tools import carrismetropolitana_api

    fake_lines = [
        {
            "short_name": "1201",
            "long_name": "Terrugem (Escola) | Circular",
            "id": "1201",
            "municipalities": ["1111", "1111"],
            "localities": ["Sintra", "Sintraa", "Terrugem", "Sintra"],
        }
    ]

    with patch.object(carrismetropolitana_api, "fetch_json_with_retry", return_value=fake_lines):
        result = carrismetropolitana_api.search_carris_metropolitana_lines.invoke({"query": "Sintra"})

    assert "1111" not in result
    assert "Sintraa" not in result
    assert "Localities: Sintra, Terrugem" in result or "Localities: Terrugem, Sintra" in result


def test_get_metro_wait_time_direction_fallback_resolves_same_side_platform_destination() -> None:
    """Metro wait-time filtering should map Saldanha→Odivelas to the Campo Grande platform when needed."""
    from tools import metrolisboa_api

    fake_wait_data = {
        "codigo": "200",
        "resposta": [
            {"destino": "48", "tempoChegada1": "240", "tempoChegada2": "600", "tempoChegada3": "900"},
            {"destino": "45", "tempoChegada1": "180", "tempoChegada2": "420", "tempoChegada3": "780"},
        ],
    }

    with patch.object(metrolisboa_api, "_is_metro_api_available", return_value=True), patch.object(
        metrolisboa_api,
        "_metro_api_request",
        return_value=fake_wait_data,
    ):
        result = metrolisboa_api.get_metro_wait_time.invoke(
            {"station": "Saldanha", "direction": "Odivelas"}
        )

    assert "Direction: Odivelas" in result
    assert "Platform indicator currently shows Campo Grande" in result
    assert "Direction: Rato" not in result


def test_carris_metropolitana_timestamp_parser_accepts_seconds_and_milliseconds() -> None:
    """Carris Metropolitana timestamps should parse correctly whether the feed uses seconds or milliseconds."""
    from tools import carrismetropolitana_api

    dt_seconds = carrismetropolitana_api._parse_unix_timestamp(1741543200)
    dt_millis = carrismetropolitana_api._parse_unix_timestamp(1741543200000)

    assert dt_seconds is not None
    assert dt_millis is not None
    assert dt_seconds.year == 2025
    assert dt_millis.year == 2025


def test_carris_metropolitana_realtime_response_flags_stale_vehicle_timestamps() -> None:
    """Vehicle-level ages should be omitted when the upstream API reports implausibly stale timestamps."""
    from tools import carrismetropolitana_api

    carrismetropolitana_api._vehicle_feed_meta.update(
        {
            "source": "live",
            "generated_at": None,
            "data_age_seconds": 0,
            "last_error": None,
            "missing_coordinates": 0,
            "vehicle_count": 1,
        }
    )

    stale_vehicle = {
        "id": "veh-1",
        "line_id": "1001",
        "lat": 38.75,
        "lon": -9.18,
        "bearing": 90,
        "speed": 24,
        "timestamp": 1764099960,
        "current_status": "IN_TRANSIT_TO",
        "license_plate": "AA-00-BB",
        "vehicle_model": "Mercedes",
    }
    fake_lines = [{"id": "1001", "short_name": "1001", "long_name": "Sintra - Lisboa"}]

    with patch.object(carrismetropolitana_api, "load_carris_metropolitana_vehicles", return_value=[stale_vehicle]), patch.object(
        carrismetropolitana_api,
        "load_carris_metropolitana_lines",
        return_value=fake_lines,
    ):
        result = carrismetropolitana_api.get_real_time_bus_positions.invoke({"line_id": "1001"})

    assert "Vehicle-level timestamp note" in result
    assert "149610m ago" not in result


def test_get_route_between_stations_cp_direct_route_keeps_cp_source_line() -> None:
    """Direct CP routes should end with CP source attribution instead of returning early without provenance."""
    from tools import transport_api

    with patch.object(transport_api, "get_landmark_info", return_value=None), patch.object(
        transport_api,
        "get_station_lines",
        side_effect=lambda name: ["verde"] if name.lower() == "rossio" else [],
    ), patch.object(
        transport_api,
        "get_cp_station_info",
        side_effect=[{"lines": ["sintra"]}, {"lines": ["sintra"]}],
    ):
        result = transport_api.get_route_between_stations.invoke({"origin": "Rossio", "destination": "Sintra"})

    assert "[*CP*](https://www.cp.pt)" in result


def test_get_route_between_stations_same_location_short_circuits() -> None:
    """Same-origin routes should return a short no-travel-needed answer instead of a fabricated itinerary."""
    from tools import transport_api

    result = transport_api.get_route_between_stations.invoke({"origin": "Saldanha", "destination": "Saldanha"})

    assert "already at the destination" in result.lower()


def test_transport_agent_same_location_query_keeps_short_circuit_response() -> None:
    """TransportAgent should preserve the no-travel-needed route response instead of falling through to bus routing."""
    from agent.agents.transport_agent import TransportAgent

    agent = TransportAgent()
    result = agent.invoke("How do I get from Saldanha to Saldanha?")

    assert "already at the destination" in result.lower()
    assert "direct routes" not in result.lower()


def test_researcher_agent_place_query_uses_direct_lookup_path() -> None:
    """ResearcherAgent should bypass free-form synthesis for straightforward place lookups."""
    from agent.agents.researcher_agent import ResearcherAgent

    agent = ResearcherAgent()

    with patch.object(agent, "_run_direct_place_lookup", return_value="DIRECT PLACE RESULT") as direct_mock, patch.object(
        agent,
        "execute_react_loop",
        side_effect=AssertionError("execute_react_loop should not run for deterministic place lookups"),
    ):
        result = agent.invoke("Best museums in Belém")

    direct_mock.assert_called_once()
    assert "DIRECT PLACE RESULT" in result


def test_researcher_agent_hybrid_place_source_line_mentions_lisboa_aberta() -> None:
    """Hybrid place outputs should credit Lisboa Aberta in the final researcher source line."""
    from agent.agents.researcher_agent import ResearcherAgent

    source_line = ResearcherAgent._build_places_source_line(
        "📊 **Sources:** 1 from VisitLisboa, 4 from Lisboa Aberta",
        "en",
    )

    assert "Lisboa Aberta" in source_line


def test_finalize_worker_response_preserves_lisboa_aberta_for_hybrid_places() -> None:
    """Final researcher formatting should keep Lisboa Aberta in hybrid place source lines."""
    from agent.utils.response_formatter import finalize_worker_response

    raw_text = (
        "1. 📊 **Hospital Santa Maria**\n"
        "   📂 Category: 📊 Open Data: Hospitais Públicos\n"
        "   📍 Avenida Professor Egas Moniz\n"
        "📌 **Source:** [*VisitLisboa Places*](https://www.visitlisboa.com/en/places) and [*Lisboa Aberta*](https://dados.cm-lisboa.pt/)"
    )

    result = finalize_worker_response(
        raw_text,
        agent_name="researcher",
        user_query="Where is Hospital Santa Maria in Lisbon?",
        language="en",
    )

    assert "Lisboa Aberta" in result


def test_researcher_prompt_mentions_safe_web_fallback() -> None:
    """Researcher prompt should explain when web search is allowed and how to treat it."""
    from agent.prompts.researcher import get_researcher_prompt

    prompt = get_researcher_prompt()

    assert "search_history_culture" in prompt
    assert "knowledge base is insufficient" in prompt
    assert "keep any source/caution context" in prompt


# ==========================================================================
# TLS: Metro secure CA bundle default
# ==========================================================================

def _check_metro_ssl_secure_default(results: TestResults, monkeypatch: pytest.MonkeyPatch):
    """Shared regression check for Metro TLS secure defaults."""
    print("\n\033[1m📋 TLS: Metro Secure CA Bundle Default\033[0m")
    print("-" * 50)

    try:
        import importlib
        import tempfile
        from unittest.mock import patch

        import requests

        monkeypatch.delenv("METRO_CA_BUNDLE", raising=False)
        monkeypatch.delenv("METRO_SSL_VERIFY", raising=False)

        module_name = "tools.metrolisboa_api"
        if module_name in sys.modules:
            metro_api = importlib.reload(sys.modules[module_name])
        else:
            metro_api = importlib.import_module(module_name)

        verify_value = metro_api.METRO_SSL_VERIFY
        if verify_value is True:
            results.add_pass("METRO_SSL_VERIFY defaults to standard secure verification")
        else:
            results.add_fail("Metro SSL default", f"Expected True, got {verify_value!r}")
            return

        dynamic_bundle_path = os.path.join(
            tempfile.gettempdir(),
            "metro_audit_dynamic_bundle.pem",
        )
        with open(dynamic_bundle_path, "w", encoding="utf-8") as file:
            file.write("dummy bundle")

        captured = {"verify_calls": []}

        def fake_request(method, url, verify=None, **kwargs):
            captured["method"] = method
            captured["url"] = url
            captured["verify_calls"].append(verify)
            if len(captured["verify_calls"]) == 1:
                raise requests.exceptions.SSLError("certificate verify failed")
            response = requests.Response()
            response.status_code = 200
            response._content = b"{}"
            return response

        with patch.object(
            metro_api,
            "_build_runtime_metro_ca_bundle",
            return_value=dynamic_bundle_path,
        ), patch.object(
            metro_api,
            "_METRO_SSL_ALLOW_INSECURE_FALLBACK",
            False,
        ), patch.object(metro_api.requests, "request", side_effect=fake_request):
            metro_api._metro_request(
                "get",
                "https://api.metrolisboa.pt:8243/estadoServicoML/1.0.1/test",
            )

        if captured.get("verify_calls") == [True, dynamic_bundle_path]:
            results.add_pass("_metro_request upgrades to a dynamic CA bundle after SSL failure")
        else:
            results.add_fail(
                "_metro_request verify sequence",
                f"Expected [True, {dynamic_bundle_path!r}], got {captured.get('verify_calls')!r}",
            )

    except Exception as e:
        results.add_fail("Metro SSL secure default", str(e))


def test_metro_ssl_secure_default(results: TestResults, monkeypatch: pytest.MonkeyPatch):
    """Tests that the Metro API defaults to a secure CA bundle rather than verify=False."""
    _check_metro_ssl_secure_default(results, monkeypatch)
