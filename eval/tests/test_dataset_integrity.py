# ==========================================================================
# Master Thesis - Dataset Integrity Tests
#   - Andre Filipe Gomes Silvestre, 20240502
#
#   Validates that dataset.json is well-formed with correct tool names,
#   unique IDs, valid domains, and complete expected_facts.
#
#   Run: python -m pytest eval/tests/test_dataset_integrity.py -v
# ==========================================================================

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset.json")

# All valid tool names from tools/__init__.py
VALID_TOOL_NAMES = {
    # Weather (IPMA)
    "get_weather_warnings",
    "get_weather_forecast",
    "get_current_weather_summary",
    "get_portugal_weather_overview",
    # Transport - Metro
    "get_metro_status",
    "get_metro_wait_time",
    "get_metro_line_wait_times",
    "find_nearest_metro",
    "get_metro_frequency",
    "get_all_metro_stations",
    # Transport - Bus (Carris Metropolitana)
    "get_real_time_bus_positions",
    "get_carris_metropolitana_alerts",
    "get_carris_metropolitana_stop_info",
    "search_carris_metropolitana_lines",
    "find_bus_routes",
    "get_bus_realtime_locations",
    "get_bus_next_departures",
    "find_direct_bus_lines",
    # Transport - Train (CP)
    "get_train_status",
    "search_cp_stations",
    "get_train_schedule",
    "get_cp_routes",
    "plan_train_trip",
    "get_train_frequency",
    # Transport - Multi-modal
    "get_transport_summary",
    "get_route_between_stations",
    # Open Data (Lisboa Aberta)
    "find_nearby_services",
    "list_available_datasets",
    "get_dataset_details",
    "find_place_in_datasets",
    "list_service_categories",
    # VisitLisboa (Events & Places)
    "search_cultural_events",
    "search_places_attractions",
    "get_event_categories",
    "get_place_categories",
    "search_lisbon_knowledge",
    # Transport - Carris Urban (Buses & Trams)
    "carris_get_stops",
    "carris_get_routes",
    "carris_get_next_departures",
    "carris_find_routes_between",
    "carris_get_realtime_vehicles",
    "carris_get_arrivals",
    "carris_vehicle_eta",
    "carris_get_service_frequency",
    # Web Knowledge
    "search_history_culture",
}

VALID_DOMAINS = {"weather", "transport", "researcher"}
VALID_LANGUAGES = {"en", "pt", "fr", "de", "mixed"}


@pytest.fixture
def dataset():
    """Load dataset.json."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class TestDatasetIntegrity:
    """Validates structure and correctness of dataset.json."""

    def test_dataset_loads_and_is_list(self, dataset):
        """Dataset should load as a non-empty list."""
        assert isinstance(dataset, list)
        assert len(dataset) > 0

    def test_all_entries_have_required_fields(self, dataset):
        """Every entry must have id, query, domain, expected_tools, expected_facts."""
        required_fields = {"id", "query", "domain", "expected_tools", "expected_facts", "language"}
        for item in dataset:
            missing = required_fields - set(item.keys())
            assert not missing, f"Entry {item.get('id', '?')} missing fields: {missing}"

    def test_ids_are_unique(self, dataset):
        """All IDs must be unique."""
        ids = [item["id"] for item in dataset]
        duplicates = [x for x in ids if ids.count(x) > 1]
        assert len(set(duplicates)) == 0, f"Duplicate IDs found: {set(duplicates)}"

    def test_domains_are_valid(self, dataset):
        """All domains must be one of: weather, transport, researcher."""
        for item in dataset:
            assert item["domain"] in VALID_DOMAINS, (
                f"Entry {item['id']} has invalid domain '{item['domain']}'"
            )

    def test_expected_tools_reference_real_tools(self, dataset):
        """All expected_tools must reference tools from tools/__init__.py."""
        invalid_refs = []
        for item in dataset:
            for tool in item.get("expected_tools", []):
                if tool not in VALID_TOOL_NAMES:
                    invalid_refs.append(f"{item['id']}: {tool}")
        assert not invalid_refs, (
            "Invalid tool references found:\n" + "\n".join(invalid_refs)
        )

    def test_edge_cases_have_edge_type(self, dataset):
        """Entries marked as edge_case=true should have an edge_type field."""
        missing_type = []
        for item in dataset:
            if item.get("edge_case", False) and "edge_type" not in item:
                missing_type.append(item["id"])
        assert not missing_type, f"Edge cases without edge_type: {missing_type}"

    def test_minimum_queries_per_domain(self, dataset):
        """Each domain should have at least 8 queries for meaningful evaluation."""
        from collections import Counter
        domain_counts = Counter(item["domain"] for item in dataset)
        for domain in VALID_DOMAINS:
            count = domain_counts.get(domain, 0)
            assert count >= 8, (
                f"Domain '{domain}' only has {count} queries (need >= 8)"
            )

    def test_languages_are_valid(self, dataset):
        """All language codes should be recognized."""
        for item in dataset:
            assert item["language"] in VALID_LANGUAGES, (
                f"Entry {item['id']} has invalid language '{item['language']}'"
            )

    def test_expected_facts_non_empty_for_normal_cases(self, dataset):
        """Non-edge queries should have at least one expected fact."""
        empty_facts = []
        for item in dataset:
            if not item.get("edge_case", False) and not item.get("expected_facts"):
                empty_facts.append(item["id"])
        assert not empty_facts, f"Non-edge entries with empty expected_facts: {empty_facts}"

    def test_total_dataset_size(self, dataset):
        """Dataset should have a reasonable number of queries (>= 30)."""
        assert len(dataset) >= 30, f"Dataset only has {len(dataset)} queries, expected >= 30"
