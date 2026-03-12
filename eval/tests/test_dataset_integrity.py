# ==========================================================================
# Master Thesis - Dataset Integrity Tests
#   - André Filipe Gomes Silvestre, 20240502
#
#   Validates that evaluation_groundtruth_queries.json is well-formed with correct tool names,
#   unique IDs, valid domains, and complete expected_facts.
#
#   Run from the repository root with a relative path:
#     python -m pytest eval/tests/test_dataset_integrity.py -q
#   Useful parameters:
#     -vv                    verbose mode
#     -k tool or -k domain   focus on one integrity slice
#     -x                     stop on first failure
#     --tb=short             shorter tracebacks
#   Notes:
#     - Prefer relative paths in this workspace. Absolute pytest paths may be
#       treated as glob patterns on Windows because the folder name includes
#       `[` and `]`.
# ==========================================================================

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest

from agent.agents.base import get_agent_tools
from agent.graph import get_all_tools
from tools import __all__ as EXPORTED_TOOL_NAMES

GROUNDTRUTH_QUERIES_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "evaluation_groundtruth_queries.json",
)
COVERAGE_MANIFEST_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "tests",
    "fixtures",
    "tool_coverage_manifest.json",
)

VALID_TOOL_NAMES = set(EXPORTED_TOOL_NAMES)
EXPECTED_TOOL_COUNT = 45

VALID_DOMAINS = {"weather", "transport", "researcher", "multi_agent", "greeting", "out_of_scope"}
VALID_COVERAGE_DOMAINS = {"weather", "transport", "researcher"}
VALID_LANGUAGES = {"en", "pt", "fr", "de", "mixed"}
VALID_EDGE_TYPES = {
    "temporal_out_of_bounds",
    "geographic_out_of_bounds",
    "missing_data_field",
    "implicit_constraint",
    "climatology_vs_meteorology",
    "hallucinated_locations",
    "invalid_entity",
    "gps_coordinates",
    "unsupported_provider",
    "cross_lingual_query",
    "adversarial_hallucination_request",
    "unsupported_action",
    "out_of_scope_topic",
    "out_of_scope_task",
}


@pytest.fixture
def dataset():
    """Load the shared evaluation ground-truth query corpus."""
    with open(GROUNDTRUTH_QUERIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def coverage_manifest():
    """Load the strict live coverage manifest used for worker-tool validation."""
    with open(COVERAGE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _agent_tool_union() -> set[str]:
    """Return the union of worker-agent tool registries."""
    worker_domains = ("weather", "transport", "researcher")
    return {
        tool.name
        for domain in worker_domains
        for tool in get_agent_tools(domain)
    }


def _graph_tool_names() -> set[str]:
    """Return the full graph-level tool registry."""
    return {tool.name for tool in get_all_tools()}


class TestDatasetIntegrity:
    """Validates structure and correctness of evaluation_groundtruth_queries.json."""

    def test_dataset_loads_and_is_list(self, dataset):
        """Dataset should load as a non-empty list."""
        assert isinstance(dataset, list)
        assert len(dataset) > 0

    def test_exported_tool_registry_size(self):
        """The authoritative exported tool registry must remain stable."""
        assert len(VALID_TOOL_NAMES) == EXPECTED_TOOL_COUNT, (
            f"Expected {EXPECTED_TOOL_COUNT} exported tools, found {len(VALID_TOOL_NAMES)}"
        )

    def test_tool_registries_are_in_sync(self):
        """tools.__all__, graph registry, and worker registries must match."""
        graph_tool_names = _graph_tool_names()
        agent_tool_names = _agent_tool_union()
        assert VALID_TOOL_NAMES == graph_tool_names == agent_tool_names, (
            "Tool registries are out of sync across tools.__all__, agent.graph.get_all_tools(), "
            "and agent.agents.base.get_agent_tools()"
        )

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

    def test_edge_types_are_valid(self, dataset):
        """All edge_type values must be from the recognized set."""
        invalid = []
        for item in dataset:
            etype = item.get("edge_type")
            if etype is not None and etype not in VALID_EDGE_TYPES:
                invalid.append(f"{item['id']}: {etype}")
        assert not invalid, (
            "Invalid edge_type values found:\n" + "\n".join(invalid)
        )

    def test_edge_cases_have_expected_behavior(self, dataset):
        """Every edge case entry must have an expected_behavior string."""
        for item in dataset:
            if item.get("edge_case", False):
                assert "expected_behavior" in item, (
                    f"{item['id']}: edge_case=true but missing expected_behavior field"
                )
                assert isinstance(item["expected_behavior"], str), (
                    f"{item['id']}: expected_behavior must be a string"
                )
                assert len(item["expected_behavior"]) > 10, (
                    f"{item['id']}: expected_behavior too short"
                )

    def test_minimum_queries_per_domain(self, dataset):
        """Core domains need >= 8 queries; auxiliary domains need >= 1."""
        core_domains = {"weather", "transport", "researcher"}
        domain_counts = Counter(item["domain"] for item in dataset)
        for domain in VALID_DOMAINS:
            count = domain_counts.get(domain, 0)
            minimum = 8 if domain in core_domains else 1
            assert count >= minimum, (
                f"Domain '{domain}' only has {count} queries (need >= {minimum})"
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

    def test_dataset_covers_all_exported_tools(self, dataset):
        """The benchmark dataset must reference every exported tool at least once."""
        covered_tools = {
            tool_name
            for item in dataset
            for tool_name in item.get("expected_tools", [])
        }
        missing_tools = sorted(VALID_TOOL_NAMES - covered_tools)
        assert not missing_tools, (
            "Dataset does not cover all exported tools:\n" + "\n".join(missing_tools)
        )

    def test_total_dataset_size(self, dataset):
        """Dataset should have a reasonable number of queries (>= 70)."""
        assert len(dataset) >= 70, f"Dataset only has {len(dataset)} queries, expected >= 70"


class TestCoverageManifestIntegrity:
    """Validate the live prompt coverage manifest alongside the evaluation corpus."""

    REQUIRED_FIELDS = {"id", "domain", "language", "query", "expected_tools"}

    def test_manifest_has_required_fields(self, coverage_manifest):
        """Every coverage-manifest entry should expose the core routing fields."""
        for item in coverage_manifest:
            missing = self.REQUIRED_FIELDS - set(item.keys())
            assert not missing, f"{item.get('id', '?')} missing fields: {sorted(missing)}"

    def test_manifest_domains_are_valid(self, coverage_manifest):
        """Coverage prompts should stay within the worker-agent domains."""
        invalid = [
            item["id"]
            for item in coverage_manifest
            if item["domain"] not in VALID_COVERAGE_DOMAINS
        ]
        assert not invalid, f"Invalid manifest domains: {invalid}"

    def test_manifest_expected_tools_exist(self, coverage_manifest):
        """Expected coverage tools must point at real exported tools."""
        invalid = []
        for item in coverage_manifest:
            for tool_name in item.get("expected_tools", []):
                if tool_name not in VALID_TOOL_NAMES:
                    invalid.append(f"{item['id']}: {tool_name}")
        assert not invalid, "Invalid tool references in manifest:\n" + "\n".join(invalid)

    def test_manifest_covers_all_exported_tools(self, coverage_manifest):
        """The strict live manifest should exercise every exported tool at least once."""
        covered = {
            tool_name
            for item in coverage_manifest
            for tool_name in item.get("expected_tools", [])
        }
        missing = sorted(VALID_TOOL_NAMES - covered)
        assert not missing, "Manifest does not cover all exported tools:\n" + "\n".join(missing)

    def test_manifest_ids_are_unique(self, coverage_manifest):
        """Coverage prompt IDs should be unique so failures are traceable."""
        ids = [item["id"] for item in coverage_manifest]
        assert len(ids) == len(set(ids)), "Coverage manifest contains duplicate IDs"
