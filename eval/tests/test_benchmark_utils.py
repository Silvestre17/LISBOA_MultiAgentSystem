# ==========================================================================
# Master Thesis - Benchmark Utility Tests
#   - André Filipe Gomes Silvestre, 20240502
#
#   Unit tests for deterministic utilities in run_benchmark.py:
#   compute_tool_metrics() and SLA_THRESHOLDS configuration.
#   No LLM, network, or agent calls required.
#
#   Run from the repository root with a relative path:
#     python -m pytest eval/tests/test_benchmark_utils.py -q
#   Useful parameters:
#     -vv         verbose mode
#     -k summary  focus on summary-building checks
#     -x          stop on first failure
#     --tb=short  shorter tracebacks
#   Notes:
#     - Prefer relative paths in this workspace. Absolute pytest paths may be
#       treated as glob patterns on Windows because the folder name includes
#       `[` and `]`.
# ==========================================================================

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest

import eval.run_ablation as ablation_module
import eval.run_benchmark as benchmark_module
from config import Config
from eval.run_benchmark import (
    SLA_THRESHOLDS,
    _build_summary,
    compute_tool_metrics,
    parse_response_model_spec,
    resolve_judge_models,
    resolve_response_models,
)
from eval.runtime_utils import aggregate_judge_runs

# ==========================================================================
# Tests for compute_tool_metrics()
# ==========================================================================


class TestComputeToolMetrics:
    """Tests for Precision / Recall / F1 computation on tool usage."""

    def test_perfect_match(self):
        """Same expected and actual tools: all metrics = 1.0."""
        result = compute_tool_metrics(
            expected=["get_metro_status", "get_metro_wait_time"],
            actual=["get_metro_status", "get_metro_wait_time"],
        )
        assert result["tool_precision"] == 1.0
        assert result["tool_recall"] == 1.0
        assert result["tool_f1"] == 1.0

    def test_both_empty(self):
        """No tools expected and none called: perfect score (greeting case)."""
        result = compute_tool_metrics(expected=[], actual=[])
        assert result["tool_precision"] == 1.0
        assert result["tool_recall"] == 1.0
        assert result["tool_f1"] == 1.0

    def test_over_tooling(self):
        """Calling more tools than expected: recall=1.0 but precision < 1.0."""
        result = compute_tool_metrics(
            expected=["get_metro_status"],
            actual=["get_metro_status", "get_metro_wait_time", "get_all_metro_stations"],
        )
        assert result["tool_recall"] == 1.0
        assert result["tool_precision"] < 1.0
        assert result["tool_f1"] < 1.0

    def test_under_tooling(self):
        """Missing expected tools: recall < 1.0 but precision = 1.0."""
        result = compute_tool_metrics(
            expected=["get_metro_status", "get_metro_wait_time"],
            actual=["get_metro_status"],
        )
        assert result["tool_precision"] == 1.0
        assert result["tool_recall"] == pytest.approx(0.5)
        assert 0 < result["tool_f1"] < 1.0

    def test_no_tools_called_but_expected(self):
        """Recall = 0.0 when no tools called but tools were expected."""
        result = compute_tool_metrics(
            expected=["get_weather_forecast"],
            actual=[],
        )
        assert result["tool_recall"] == 0.0
        assert result["tool_f1"] == 0.0

    def test_tools_called_but_none_expected(self):
        """Calling tools when none expected: precision = 0.0."""
        result = compute_tool_metrics(
            expected=[],
            actual=["get_metro_status"],
        )
        assert result["tool_precision"] == 0.0
        assert result["tool_f1"] == 0.0

    def test_completely_wrong_tools(self):
        """All called tools are different from expected: all metrics = 0.0."""
        result = compute_tool_metrics(
            expected=["get_weather_forecast"],
            actual=["get_metro_status"],
        )
        assert result["tool_precision"] == 0.0
        assert result["tool_recall"] == 0.0
        assert result["tool_f1"] == 0.0

    def test_partial_overlap(self):
        """1 of 2 expected tools called: precision=1.0, recall=0.5, f1=0.667."""
        result = compute_tool_metrics(
            expected=["get_weather_forecast", "get_current_weather_summary"],
            actual=["get_weather_forecast"],
        )
        assert result["tool_precision"] == 1.0
        assert result["tool_recall"] == pytest.approx(0.5)
        assert result["tool_f1"] == pytest.approx(2 / 3, rel=1e-3)

    def test_duplicates_treated_as_sets(self):
        """Duplicate entries in both lists should be de-duplicated (set semantics)."""
        result = compute_tool_metrics(
            expected=["get_metro_status", "get_metro_status"],
            actual=["get_metro_status"],
        )
        assert result["tool_precision"] == 1.0
        assert result["tool_recall"] == 1.0
        assert result["tool_f1"] == 1.0

    def test_output_keys_always_present(self):
        """Return dict must always contain all three keys."""
        for expected, actual in [
            ([], []),
            (["a"], []),
            ([], ["a"]),
            (["a"], ["a"]),
            (["a", "b"], ["b", "c"]),
        ]:
            result = compute_tool_metrics(expected=expected, actual=actual)
            assert "tool_precision" in result
            assert "tool_recall" in result
            assert "tool_f1" in result

    def test_scores_are_rounded_to_3_decimals(self):
        """All scores should have at most 3 decimal places."""
        result = compute_tool_metrics(
            expected=["a", "b", "c"],
            actual=["a", "b"],
        )
        for key in ("tool_precision", "tool_recall", "tool_f1"):
            val = result[key]
            assert round(val, 3) == val, f"{key} not rounded to 3 decimals: {val}"

    def test_flexible_tool_expectations_are_unscored(self):
        """Flexible rows should carry metadata but not produce numeric tool F1."""
        result = compute_tool_metrics(
            expected=["search_places_attractions"],
            actual=["search_cultural_events"],
            tool_expectation="flexible",
            acceptable_tool_sets=[["search_cultural_events"]],
        )

        assert result["tool_f1"] is None
        assert result["tool_metric_scored"] is False
        assert result["tool_expectation"] == "flexible"


def test_benchmark_summary_skips_unscored_flexible_tool_rows() -> None:
    """Summary averages should ignore flexible rows where deterministic tool F1 is None."""
    base_record = {
        "domain": "researcher",
        "response_model": "azure::gpt-5.4-mini",
        "error": None,
        "latency_s": 1.0,
        "scores": {"composite_score": 4.0, "factual_accuracy": 4.0},
        "heuristics": {"overall_pass": True},
        "sla_met": True,
        "response_usage": {},
        "evaluation_usage": {},
        "combined_usage": {},
        "response_cost_usd": {},
        "evaluation_cost_usd": {},
        "combined_cost_usd": {},
    }
    strict_record = {
        **base_record,
        "tool_metrics": {"tool_f1": 1.0, "tool_metric_scored": True},
    }
    flexible_record = {
        **base_record,
        "tool_metrics": {"tool_f1": None, "tool_metric_scored": False},
    }

    summary = _build_summary([strict_record, flexible_record])

    assert summary["overall"]["avg_tool_f1"] == 1.0
    assert summary["per_domain"]["researcher"]["avg_tool_f1"] == 1.0


def test_benchmark_judge_receives_flexible_tool_metadata() -> None:
    """Benchmark judge calls should receive flexible-tool metadata from the dataset row."""
    captured = {}

    class FakeJudge:
        def evaluate(self, **kwargs):
            captured.update(kwargs)
            return {
                "factual_accuracy": 4,
                "tool_usage": 4,
                "completeness": 4,
                "relevance": 4,
                "response_quality": 4,
                "composite_score": 4.0,
                "reasoning": "ok",
                "evaluation_usage": {},
                "evaluation_cost_usd": {},
            }

    benchmark_module._evaluate_with_judges(
        judges=[FakeJudge()],
        judge_model_manifests=[{"model_id": "test::judge"}],
        query="q",
        expected_facts=[],
        expected_tools=["a"],
        actual_tools=["b"],
        retrieved_context="ctx",
        reference_context="Expected facts:\n- fact",
        response="resp",
        response_error=None,
        pricing_by_model=None,
        tool_expectation="flexible",
        acceptable_tool_sets=[["b"]],
    )

    assert captured["tool_expectation"] == "flexible"
    assert captured["acceptable_tool_sets"] == [["b"]]
    assert captured["reference_context"] == "Expected facts:\n- fact"


def test_ablation_judge_receives_flexible_tool_metadata() -> None:
    """Ablation judge calls should receive flexible-tool metadata for both arms."""
    captured = {}

    class FakeJudge:
        def evaluate(self, **kwargs):
            captured.update(kwargs)
            return {
                "factual_accuracy": 4,
                "tool_usage": 4,
                "completeness": 4,
                "relevance": 4,
                "response_quality": 4,
                "composite_score": 4.0,
                "reasoning": "ok",
                "evaluation_usage": {},
                "evaluation_cost_usd": {},
            }

    ablation_module._evaluate_with_judges(
        judges=[FakeJudge()],
        judge_model_manifests=[{"model_id": "test::judge"}],
        query="q",
        expected_facts=[],
        expected_tools=["a"],
        actual_tools=["b"],
        retrieved_context="ctx",
        reference_context="Expected facts:\n- fact",
        response="resp",
        response_error=None,
        pricing_by_model=None,
        tool_expectation="flexible",
        acceptable_tool_sets=[["b"]],
    )

    assert captured["tool_expectation"] == "flexible"
    assert captured["acceptable_tool_sets"] == [["b"]]
    assert captured["reference_context"] == "Expected facts:\n- fact"


def test_ablation_quality_score_excludes_tool_usage() -> None:
    """Primary ablation score should average equal non-tool dimensions for both arms."""
    scores = ablation_module._with_ablation_primary_score(
        {
            "factual_accuracy": 5,
            "tool_usage": 1,
            "completeness": 4,
            "relevance": 4,
            "response_quality": 3,
            "composite_score": 3.4,
            "reasoning": "Tool use should not drive ablation quality.",
        }
    )

    assert scores["ablation_quality_score"] == pytest.approx(4.0)
    assert scores["composite_score"] == pytest.approx(4.0)
    assert scores["tool_inclusive_composite_score"] == pytest.approx(3.4)
    assert scores["ablation_score_dimensions"] == [
        "factual_accuracy",
        "completeness",
        "relevance",
        "response_quality",
    ]


# ==========================================================================
# Tests for SLA_THRESHOLDS
# ==========================================================================


class TestSLAThresholds:
    """Validates that SLA_THRESHOLDS covers all dataset domains with sensible values."""

    EXPECTED_DOMAINS = {"weather", "transport", "researcher", "multi_agent", "greeting", "out_of_scope"}

    def test_all_domains_have_thresholds(self):
        """Every dataset domain must have an SLA threshold defined."""
        missing = self.EXPECTED_DOMAINS - set(SLA_THRESHOLDS.keys())
        assert not missing, f"Missing SLA threshold for domains: {missing}"

    def test_thresholds_are_positive_floats(self):
        """All threshold values must be positive numbers."""
        for domain, threshold in SLA_THRESHOLDS.items():
            assert isinstance(threshold, (int, float)), \
                f"{domain}: threshold must be numeric, got {type(threshold)}"
            assert threshold > 0, f"{domain}: threshold must be > 0, got {threshold}"

    def test_multi_agent_is_slowest(self):
        """Multi-agent queries should have the highest (most lenient) SLA."""
        assert SLA_THRESHOLDS["multi_agent"] == max(SLA_THRESHOLDS.values()), \
            "multi_agent should have the longest SLA threshold"

    def test_simple_domains_are_fastest(self):
        """Greeting and out_of_scope should have the tightest SLAs."""
        assert SLA_THRESHOLDS["greeting"] <= SLA_THRESHOLDS["transport"]
        assert SLA_THRESHOLDS["out_of_scope"] <= SLA_THRESHOLDS["transport"]

    def test_transport_sla_greater_than_weather(self):
        """Transport (28 tools) inherently takes longer than weather (4 tools)."""
        assert SLA_THRESHOLDS["transport"] >= SLA_THRESHOLDS["weather"]


class TestBenchmarkModelSelection:
    """Validate CLI-friendly benchmark model selection helpers."""

    def test_parse_response_model_spec_accepts_double_colon(self):
        """The preferred provider::model format should parse cleanly."""
        parsed = parse_response_model_spec("azure::gpt-5-mini", temperature=0.25)

        assert parsed == {
            "provider": "azure",
            "model": "gpt-5-mini",
            "temperature": 0.25,
        }

    def test_parse_response_model_spec_accepts_single_colon(self):
        """A single-colon shorthand should remain supported for convenience."""
        parsed = parse_response_model_spec("openai:gpt-5-mini")

        assert parsed == {
            "provider": "openai",
            "model": "gpt-5-mini",
            "temperature": 0.0,
        }

    def test_parse_response_model_spec_rejects_invalid_format(self):
        """Malformed model specs should fail loudly instead of silently guessing."""
        with pytest.raises(ValueError, match="provider::model|provider:model"):
            parse_response_model_spec("gpt-5-mini")

    def test_resolve_response_models_returns_deepcopied_defaults(self):
        """Default benchmark matrix should be copied so CLI overrides do not mutate module constants."""
        resolved = resolve_response_models()

        assert resolved == benchmark_module.MODELS_TO_TEST
        assert resolved is not benchmark_module.MODELS_TO_TEST

    def test_resolve_response_models_deduplicates_custom_specs(self):
        """Repeated model specs should collapse to one effective benchmark entry."""
        resolved = resolve_response_models(
            ["azure::gpt-5-mini", "azure::gpt-5-mini", "openai:gpt-5-mini"],
            temperature=0.1,
        )

        assert resolved == [
            {"provider": "azure", "model": "gpt-5-mini", "temperature": 0.1},
            {"provider": "openai", "model": "gpt-5-mini", "temperature": 0.1},
        ]

    def test_resolve_judge_models_defaults_to_closed_and_open_pair(self):
        """The benchmark should default to the closed/open judge matrix."""
        resolved = resolve_judge_models()

        assert resolved == benchmark_module.DEFAULT_JUDGE_MODELS
        assert resolved is not benchmark_module.DEFAULT_JUDGE_MODELS

    def test_resolve_judge_models_accepts_repeatable_specs(self):
        """Judge model specs should support explicit closed/open overrides."""
        resolved = resolve_judge_models(
            ["azure::gpt-5-mini", "azure::Kimi-K2.5"],
        )

        assert resolved == [
            {"provider": "azure", "model": "gpt-5-mini", "temperature": 0.0},
            {"provider": "azure", "model": "Kimi-K2.5", "temperature": 0.0},
        ]


class TestAblationProviderOverrides:
    """Validate temporary provider overrides used by the ablation runner."""

    def test_normalize_model_provider_accepts_supported_values(self):
        """Supported providers should normalize to lowercase strings."""
        assert ablation_module.normalize_model_provider("Azure") == "azure"
        assert ablation_module.normalize_model_provider("openai") == "openai"
        assert ablation_module.normalize_model_provider(None) is None

    def test_normalize_model_provider_rejects_unknown_values(self):
        """Unknown providers should fail fast."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            ablation_module.normalize_model_provider("anthropic")

    def test_temporary_lisboa_provider_restores_original_config(self):
        """The ablation runner must not leave global provider selection mutated after the run."""
        original_provider = Config.MODEL_PROVIDER
        Config.MODEL_PROVIDER = "azure"

        try:
            with ablation_module.temporary_lisboa_provider("openai") as active_provider:
                assert active_provider == "openai"
                assert Config.MODEL_PROVIDER == "openai"

            assert Config.MODEL_PROVIDER == "azure"
        finally:
            Config.MODEL_PROVIDER = original_provider

    def test_temporary_lisboa_provider_can_override_model_name(self):
        """Dual-profile ablation should be able to swap LISBOA onto another model within the same provider."""
        original_provider = Config.MODEL_PROVIDER
        original_azure_models = [config.copy() for config in Config.AGENT_MODELS_AZURE.values()]
        Config.MODEL_PROVIDER = "azure"

        try:
            with ablation_module.temporary_lisboa_provider("azure", model_name="Kimi-K2.5") as active_provider:
                assert active_provider == "azure"
                assert Config.MODEL_PROVIDER == "azure"
                assert {config["model"] for config in Config.AGENT_MODELS_AZURE.values()} == {"Kimi-K2.5"}

            assert Config.MODEL_PROVIDER == "azure"
            restored_models = [config["model"] for config in Config.AGENT_MODELS_AZURE.values()]
            assert restored_models == [config["model"] for config in original_azure_models]
        finally:
            Config.MODEL_PROVIDER = original_provider

    def test_ablation_groundtruth_defaults_to_core_domains(self):
        """Default ablation runs should focus on the grounded task domains under study."""
        queries = ablation_module.load_groundtruth_queries(limit=None)
        assert queries, "Default ablation corpus should not be empty"
        assert {item["domain"] for item in queries} == set(ablation_module.DEFAULT_ABLATION_DOMAINS)

    def test_ablation_groundtruth_accepts_explicit_domain_override(self):
        """Researchers should still be able to target non-default domains intentionally."""
        queries = ablation_module.load_groundtruth_queries(
            limit=None,
            include_domains=("greeting",),
        )
        assert queries, "Explicit domain overrides should still return matching rows"
        assert {item["domain"] for item in queries} == {"greeting"}


class TestBenchmarkSummaryCostAccounting:
    """Validates that benchmark summary aggregation preserves cost blocks."""

    def test_summary_includes_usage_and_cost_blocks(self):
        """Summary should aggregate response/evaluation/combined usage and cost."""
        sample_results = [
            {
                "domain": "weather",
                "response_model": "azure::gpt-5-mini",
                "scores": {
                    "composite_score": 4.5,
                    "factual_accuracy": 5,
                },
                "error": None,
                "error_type": None,
                "latency_s": 1.25,
                "heuristics": {"overall_pass": True},
                "sla_met": True,
                "response_usage": {
                    "call_count": 2,
                    "usage_available": True,
                    "tokens": {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
                },
                "evaluation_usage": {
                    "call_count": 1,
                    "usage_available": True,
                    "tokens": {"input_tokens": 60, "output_tokens": 15, "total_tokens": 75},
                },
                "combined_usage": {
                    "call_count": 3,
                    "usage_available": True,
                    "tokens": {"input_tokens": 180, "output_tokens": 45, "total_tokens": 225},
                },
                "response_cost_usd": {
                    "model_id": "azure::gpt-5-mini",
                    "pricing_lookup_key": "azure::gpt-5-mini",
                    "pricing_found": True,
                    "pricing_complete": True,
                    "tokens": {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
                    "input_per_million_usd": 0.25,
                    "output_per_million_usd": 2.0,
                    "cached_input_per_million_usd": 0.03,
                    "input_cost_usd": 0.00003,
                    "output_cost_usd": 0.00006,
                    "total_cost_usd": 0.00009,
                    "missing_pricing_models": [],
                },
                "evaluation_cost_usd": {
                    "pricing_found": True,
                    "pricing_complete": True,
                    "tokens": {"input_tokens": 60, "output_tokens": 15, "total_tokens": 75},
                    "input_cost_usd": 0.000015,
                    "output_cost_usd": 0.00003,
                    "total_cost_usd": 0.000045,
                    "missing_pricing_models": [],
                },
                "combined_cost_usd": {
                    "pricing_found": True,
                    "pricing_complete": True,
                    "tokens": {"input_tokens": 180, "output_tokens": 45, "total_tokens": 225},
                    "input_cost_usd": 0.000045,
                    "output_cost_usd": 0.00009,
                    "total_cost_usd": 0.000135,
                    "missing_pricing_models": [],
                },
            }
        ]

        summary = _build_summary(sample_results)
        assert summary["overall"]["response_usage"]["tokens"]["input_tokens"] == 120
        assert summary["overall"]["successful_responses"] == 1
        assert summary["overall"]["errored_responses"] == 0
        assert summary["overall"]["scored_responses"] == 1
        assert summary["overall"]["evaluation_usage"]["tokens"]["output_tokens"] == 15
        assert summary["overall"]["combined_usage"]["tokens"]["total_tokens"] == 225
        assert summary["overall"]["response_cost_usd"]["pricing_lookup_key"] == "azure::gpt-5-mini"
        assert summary["overall"]["response_cost_usd"]["output_per_million_usd"] == pytest.approx(2.0)
        assert summary["overall"]["response_cost_usd"]["total_cost_usd"] == pytest.approx(0.00009)
        assert summary["overall"]["evaluation_cost_usd"]["total_cost_usd"] == pytest.approx(0.000045)
        assert summary["overall"]["combined_cost_usd"]["total_cost_usd"] == pytest.approx(0.000135)
        assert summary["overall"]["heuristics_pass_count"] == 1
        assert summary["overall"]["sla_met_count"] == 1
        assert summary["per_domain"]["weather"]["combined_cost_usd"]["total_cost_usd"] == pytest.approx(0.000135)
        assert summary["per_domain"]["weather"]["successful_responses"] == 1
        assert summary["per_domain"]["weather"]["heuristics_pass_count"] == 1
        assert summary["per_response_model"]["azure::gpt-5-mini"]["response_cost_usd"]["model_id"] == "azure::gpt-5-mini"
        assert summary["per_response_model"]["azure::gpt-5-mini"]["successful_responses"] == 1
        assert summary["per_response_model"]["azure::gpt-5-mini"]["combined_usage"]["tokens"]["total_tokens"] == 225

    def test_aggregate_judge_runs_excludes_failed_judges_from_average(self):
        """A failed judge should remain stored but should not drag the compatibility average to zero."""
        aggregated = aggregate_judge_runs(
            [
                {
                    "judge_model": "azure::gpt-5-mini",
                    "scores": {
                        "factual_accuracy": 5,
                        "tool_usage": 4,
                        "completeness": 5,
                        "relevance": 4,
                        "response_quality": 5,
                        "composite_score": 4.6,
                        "reasoning": "Closed judge succeeded.",
                    },
                    "evaluation_usage": {"tokens": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}, "call_count": 1, "usage_available": True},
                    "evaluation_cost_usd": {"tokens": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}, "pricing_found": True, "pricing_complete": True, "input_cost_usd": 0.1, "output_cost_usd": 0.2, "total_cost_usd": 0.3, "missing_pricing_models": []},
                    "error": None,
                },
                {
                    "judge_model": "azure::Kimi-K2.5",
                    "scores": {
                        "factual_accuracy": None,
                        "tool_usage": None,
                        "completeness": None,
                        "relevance": None,
                        "response_quality": None,
                        "composite_score": None,
                        "reasoning": "Judge Failed: empty content.",
                    },
                    "evaluation_usage": {"tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, "call_count": 1, "usage_available": False},
                    "evaluation_cost_usd": {"tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, "pricing_found": True, "pricing_complete": True, "input_cost_usd": 0.0, "output_cost_usd": 0.0, "total_cost_usd": 0.0, "missing_pricing_models": []},
                    "error": "Judge Failed: empty content.",
                },
            ]
        )

        assert aggregated["scores"]["composite_score"] == pytest.approx(4.6)
        assert aggregated["judge_summary"]["successful_judges"] == 1
        assert aggregated["judge_summary"]["failed_judges"] == 1


class TestRunIsolatedAgent:
    """Ensures the benchmark runner measures the real worker invoke path."""

    def test_run_isolated_agent_tracks_tool_calls_from_invoke(self, monkeypatch):
        """Tool usage should be captured even when the worker calls tools directly in invoke()."""

        class DummyTool:
            def __init__(self, name: str):
                self.name = name

            def invoke(self, payload):
                return f"tool-output:{payload['query']}"

        class DummyTransportAgent:
            def __init__(self):
                self.tools = [DummyTool("get_metro_status")]

            def init_llm(self, provider, model, temperature):
                self.llm_config = {
                    "provider": provider,
                    "model": model,
                    "temperature": temperature,
                }

            def reset_llm_usage_tracking(self):
                return None

            def get_llm_usage_summary(self):
                return {}

            def invoke(self, query: str):
                tool_result = self.tools[0].invoke({"query": query})
                return f"final-response:{tool_result}"

        monkeypatch.setattr(benchmark_module, "TransportAgent", DummyTransportAgent)

        response, tools, retrieved_context, latency, error, usage = benchmark_module.run_isolated_agent(
            domain="transport",
            query="Is the metro working correctly right now?",
            config={"provider": "azure", "model": "gpt-5-mini", "temperature": 0.0},
        )

        assert error is None
        assert response == "final-response:tool-output:Is the metro working correctly right now?"
        assert tools == ["get_metro_status"]
        assert "[get_metro_status] returned:\ntool-output:Is the metro working correctly right now?" in retrieved_context
        assert latency >= 0
        assert usage["model_id"] == "azure::gpt-5-mini"


class TestResponseTelemetryDescriptions:
    """Validate the persisted response telemetry annotations."""

    def test_benchmark_marks_deterministic_tool_responses_as_no_llm(self):
        """Zero response usage with tool output should be marked as deterministic, not missing."""
        telemetry = benchmark_module._describe_response_telemetry(
            response_usage={
                "call_count": 0,
                "usage_available": False,
                "tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            },
            tools_used=["get_route_between_stations"],
            error=None,
        )

        assert telemetry["response_generation_mode"] == "deterministic_tool"
        assert telemetry["response_usage_status"] == "not_applicable_no_llm"
        assert "zero by design" in str(telemetry["response_usage_note"])

    def test_ablation_marks_llm_responses_as_captured(self):
        """Non-zero response usage should be marked as captured."""
        telemetry = ablation_module._describe_response_telemetry(
            response_usage={
                "call_count": 1,
                "usage_available": True,
                "tokens": {"input_tokens": 100, "output_tokens": 25, "total_tokens": 125},
            },
            tools_used=[],
            error=None,
        )

        assert telemetry == {
            "response_generation_mode": "llm",
            "response_usage_status": "captured",
            "response_usage_note": None,
        }


class TestAblationZeroShotRetries:
    """Validate transient retry behavior in the ablation zero-shot path."""

    def test_run_zero_shot_retries_transient_rate_limit_once(self, monkeypatch):
        """A transient 429 should be retried before the zero-shot arm fails."""
        attempts = {"count": 0}

        class FakeLLM:
            def invoke(self, _messages):
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise RuntimeError(
                        '{"object": "error", "message": "Server at maximum concurrent capacity. Try again later.", '
                        '"type": "429", "param": null, "code": 429}'
                    )
                return SimpleNamespace(content="Recovered")

        monkeypatch.setattr(ablation_module.LLMFactory, "get_llm", lambda **_kwargs: FakeLLM())
        monkeypatch.setattr(
            ablation_module.LLMFactory,
            "extract_usage_metadata",
            lambda _response: {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )

        with patch("eval.run_ablation.time.sleep") as mocked_sleep:
            response, tools, retrieved_context, latency, error, usage = ablation_module.run_zero_shot(
                "Plan a full day in Lisbon",
                provider="azure",
                model_name="Kimi-K2.5",
            )

        assert response == "Recovered"
        assert tools == []
        assert retrieved_context == ""
        assert error is None
        assert latency >= 0
        assert usage["call_count"] == 1
        assert usage["tokens"]["total_tokens"] == 15
        assert attempts["count"] == 2
        mocked_sleep.assert_called_once_with(2.0)

    def test_run_zero_shot_uses_lisboa_no_tool_system_prompt(self, monkeypatch):
        """Zero-shot should be instruction-matched to LISBOA scope while remaining tool-free."""
        captured_messages = []

        class FakeLLM:
            def invoke(self, messages):
                captured_messages.extend(messages)
                return SimpleNamespace(content="Scoped baseline response")

        monkeypatch.setattr(ablation_module.LLMFactory, "get_llm", lambda **_kwargs: FakeLLM())
        monkeypatch.setattr(ablation_module.LLMFactory, "extract_usage_metadata", lambda _response: {})

        response, tools, retrieved_context, _latency, error, _usage = ablation_module.run_zero_shot(
            "Tell me about Lisbon transport",
        )

        assert response == "Scoped baseline response"
        assert tools == []
        assert retrieved_context == ""
        assert error is None
        assert len(captured_messages) == 2
        assert "no-tool baseline" in captured_messages[0].content
        assert captured_messages[1].content == "Tell me about Lisbon transport"
