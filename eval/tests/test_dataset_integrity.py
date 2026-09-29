# ==========================================================================
# Master Thesis - Dataset Integrity Tests
#   - André Filipe Gomes Silvestre, 20240502
#
#   Validates that evaluation_groundtruth_queries.json and the paper-evaluation
#   corpus are structurally usable and reference real tool names, and that the
#   paper-evaluation corpus keeps the original queries unchanged and has
#   end-to-end annotations. It intentionally does not enforce strict
#   prompt-coverage manifests; live/system quality belongs in prompt smoke runs.
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
# RINENG revision: the 72 original queries plus new itinerary requests, and their annotations.
PAPER_EVAL_QUERIES_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "evaluation_groundtruth_queries_paper_eval.json",
)
PAPER_EVAL_ANNOTATIONS_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "paper_eval_annotations.json",
)
ABLATION_DOMAINS = {"weather", "transport", "researcher", "multi_agent"}
ANNOTATION_AGENTS = {"weather", "transport", "researcher", "planner"}
ANNOTATION_QUERY_TYPES = {"single_domain", "cross_domain", "itinerary"}
MIN_ITINERARY_CONSTRAINTS = 3
VALID_TOOL_NAMES = set(EXPORTED_TOOL_NAMES)

VALID_DOMAINS = {"weather", "transport", "researcher", "multi_agent", "greeting", "out_of_scope"}
VALID_LANGUAGES = {"en", "pt", "fr", "de", "mixed"}
VALID_TOOL_EXPECTATIONS = {"strict", "flexible"}
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


def _load_json(path: str):
    """Load one JSON file from the evaluation folder."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(
    params=[GROUNDTRUTH_QUERIES_PATH, PAPER_EVAL_QUERIES_PATH],
    ids=["shared_corpus", "paper_eval_corpus"],
)
def dataset(request):
    """Load each evaluation ground-truth query corpus in turn."""
    return _load_json(request.param)


@pytest.fixture
def original_dataset():
    """Load the shared corpus used by the May 2026 benchmark and ablation."""
    return _load_json(GROUNDTRUTH_QUERIES_PATH)


@pytest.fixture
def paper_eval_dataset():
    """Load the paper-evaluation corpus."""
    return _load_json(PAPER_EVAL_QUERIES_PATH)


@pytest.fixture
def paper_eval_annotations():
    """Load the end-to-end annotations of the paper-evaluation corpus."""
    return _load_json(PAPER_EVAL_ANNOTATIONS_PATH)


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

    def test_acceptable_tool_sets_reference_real_tools(self, dataset):
        """Alternative tool sets must point at real exported tools."""
        invalid_refs = []
        for item in dataset:
            acceptable_tool_sets = item.get("acceptable_tool_sets", [])
            assert isinstance(acceptable_tool_sets, list), (
                f"{item['id']}: acceptable_tool_sets must be a list"
            )
            for tool_set in acceptable_tool_sets:
                assert isinstance(tool_set, list), (
                    f"{item['id']}: each acceptable tool set must be a list"
                )
                for tool_name in tool_set:
                    if tool_name not in VALID_TOOL_NAMES:
                        invalid_refs.append(f"{item['id']}: {tool_name}")
        assert not invalid_refs, (
            "Invalid acceptable tool references found:\n" + "\n".join(invalid_refs)
        )

    def test_tool_expectations_are_valid(self, dataset):
        """Optional tool expectation metadata must use a recognized policy."""
        invalid = []
        for item in dataset:
            expectation = item.get("tool_expectation", "strict")
            if expectation not in VALID_TOOL_EXPECTATIONS:
                invalid.append(f"{item['id']}: {expectation}")
        assert not invalid, "Invalid tool_expectation values found:\n" + "\n".join(invalid)

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

    def test_total_dataset_size(self, dataset):
        """Dataset should have at least 72 queries (current corpus baseline)."""
        assert len(dataset) >= 72, f"Dataset only has {len(dataset)} queries, expected >= 72"


class TestPaperEvalCorpus:
    """Validates the paper-evaluation corpus against the original one, and its annotations."""

    def test_original_queries_are_unchanged(self, original_dataset, paper_eval_dataset):
        """Every original query must appear unchanged, so May and revision results stay comparable."""
        paper_by_id = {item["id"]: item for item in paper_eval_dataset}
        changed = [item["id"] for item in original_dataset if paper_by_id.get(item["id"]) != item]
        assert not changed, f"Original queries missing or changed in the paper-evaluation corpus: {changed}"

    def test_new_queries_are_multi_agent(self, original_dataset, paper_eval_dataset):
        """The revision only adds multi-agent requests; the single-domain corpus is not touched."""
        original_ids = {item["id"] for item in original_dataset}
        new_items = [item for item in paper_eval_dataset if item["id"] not in original_ids]
        assert new_items, "The paper-evaluation corpus adds no queries."
        wrong_domain = [item["id"] for item in new_items if item["domain"] != "multi_agent"]
        assert not wrong_domain, f"New queries outside the multi_agent domain: {wrong_domain}"

    def test_every_ablation_query_is_annotated(self, paper_eval_dataset, paper_eval_annotations):
        """Each query in the ablation domains needs an annotation, and annotations need a query."""
        annotated = paper_eval_annotations["queries"]
        in_scope = {item["id"] for item in paper_eval_dataset if item["domain"] in ABLATION_DOMAINS}
        missing = sorted(in_scope - set(annotated))
        orphaned = sorted(set(annotated) - in_scope)
        assert not missing, f"Ablation queries without annotations: {missing}"
        assert not orphaned, f"Annotations for queries outside the ablation scope: {orphaned}"

    def test_annotations_use_known_agents_and_types(self, paper_eval_annotations):
        """Expected agents and query types must come from the documented sets."""
        invalid = []
        for query_id, annotation in paper_eval_annotations["queries"].items():
            if annotation.get("query_type") not in ANNOTATION_QUERY_TYPES:
                invalid.append(f"{query_id}: query_type {annotation.get('query_type')}")
            agents = annotation.get("expected_agents") or []
            if not agents or not set(agents) <= ANNOTATION_AGENTS:
                invalid.append(f"{query_id}: expected_agents {agents}")
        assert not invalid, "Invalid annotations:\n" + "\n".join(invalid)

    def test_annotation_types_match_domains(self, paper_eval_dataset, paper_eval_annotations):
        """Single-domain queries expect their own worker; multi-agent queries expect several agents."""
        annotated = paper_eval_annotations["queries"]
        mismatched = []
        for item in paper_eval_dataset:
            annotation = annotated.get(item["id"])
            if annotation is None:
                continue
            if item["domain"] == "multi_agent":
                if annotation["query_type"] == "single_domain" or len(annotation["expected_agents"]) < 2:
                    mismatched.append(item["id"])
            elif annotation != {"query_type": "single_domain", "expected_agents": [item["domain"]]}:
                mismatched.append(item["id"])
        assert not mismatched, f"Annotations inconsistent with the query domain: {mismatched}"

    def test_itinerary_requests_have_constraints_and_planner(self, paper_eval_annotations):
        """Itinerary requests need checkable constraints and must expect the planner."""
        problems = []
        for query_id, annotation in paper_eval_annotations["queries"].items():
            constraints = annotation.get("constraints")
            if annotation.get("query_type") == "itinerary":
                if "planner" not in annotation["expected_agents"]:
                    problems.append(f"{query_id}: itinerary without the planner")
                if not isinstance(constraints, list) or len(constraints) < MIN_ITINERARY_CONSTRAINTS:
                    problems.append(f"{query_id}: fewer than {MIN_ITINERARY_CONSTRAINTS} constraints")
                elif len(set(constraints)) != len(constraints) or not all(str(c).strip() for c in constraints):
                    problems.append(f"{query_id}: empty or repeated constraints")
            elif constraints:
                problems.append(f"{query_id}: constraints on a non-itinerary query")
        assert not problems, "Itinerary annotation problems:\n" + "\n".join(problems)
