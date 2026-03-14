# ==========================================================================
# Master Thesis - Cost Accounting Tests
#   - André Filipe Gomes Silvestre, 20240502
#
#   Deterministic unit tests for token usage normalization, dynamic pricing
#   lookup, and cost aggregation utilities used by benchmark/ablation runs.
#
#   Run from the repository root with a relative path:
#     python -m pytest eval/tests/test_cost_accounting.py -q
#   Useful parameters:
#     -vv         verbose mode
#     -k pricing  focus on pricing checks
#     -x          stop on first failure
#     --tb=short  shorter tracebacks
#   Notes:
#     - Prefer relative paths in this workspace. Absolute pytest paths may be
#       treated as glob patterns on Windows because the folder name includes
#       `[` and `]`.
# ==========================================================================

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest

from eval.runtime_utils import (
    build_cost_payload,
    build_usage_payload,
    combine_cost_payloads,
    combine_usage_payloads,
    get_pricing_metadata,
    split_pricing_config,
)


class TestUsagePayloads:
    """Tests for token usage normalization and aggregation."""

    def test_build_usage_payload_normalizes_aliases(self):
        """Prompt/completion aliases should normalize to input/output tokens."""
        payload = build_usage_payload(
            {
                "prompt_tokens": 100,
                "completion_tokens": 25,
            },
            model_id="azure::gpt-5-mini",
            call_count=1,
        )

        assert payload["tokens"]["input_tokens"] == 100
        assert payload["tokens"]["output_tokens"] == 25
        assert payload["tokens"]["total_tokens"] == 125
        assert payload["usage_available"]

    def test_combine_usage_payloads_sums_tokens_and_calls(self):
        """Combined usage should sum token totals and call counts."""
        first = build_usage_payload(
            {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            model_id="azure::gpt-5-mini",
            call_count=1,
        )
        second = build_usage_payload(
            {"input_tokens": 50, "output_tokens": 10, "total_tokens": 60},
            model_id="azure::gpt-5-mini",
            call_count=2,
        )

        combined = combine_usage_payloads([first, second])
        assert combined["call_count"] == 3
        assert combined["tokens"]["input_tokens"] == 150
        assert combined["tokens"]["output_tokens"] == 30
        assert combined["tokens"]["total_tokens"] == 180


class TestPricingLookup:
    """Tests for flexible pricing config parsing and metadata extraction."""

    def test_split_pricing_config_supports_wrapped_catalog(self):
        """Wrapped pricing metadata with a models dict should parse correctly."""
        catalog, metadata = split_pricing_config(
            {
                "pricing_source": "https://www.llm-prices.com/current-v1.json",
                "pricing_updated_at": "2026-03-05",
                "models": {
                    "azure::gpt-5-mini": {"input": 0.25, "output": 2.0},
                },
            }
        )

        assert "azure::gpt-5-mini" in catalog
        assert catalog["azure::gpt-5-mini"]["input"] == 0.25
        assert metadata["pricing_source"] == "https://www.llm-prices.com/current-v1.json"

    def test_get_pricing_metadata_normalizes_aliases(self):
        """Alias keys should be normalized in persisted artefact metadata."""
        metadata = get_pricing_metadata(
            {
                "source": "https://www.llm-prices.com/current-v1.json",
                "updated_at": "2026-03-05",
                "models": {"gpt-5-mini": {"input": 0.25, "output": 2.0}},
            }
        )

        assert metadata == {
            "pricing_source": "https://www.llm-prices.com/current-v1.json",
            "pricing_updated_at": "2026-03-05",
        }


class TestCostPayloads:
    """Tests for single-model and multi-model cost computation."""

    def test_build_cost_payload_for_single_model(self):
        """Single-model costs should use input and output prices separately."""
        usage = build_usage_payload(
            {"input_tokens": 1200, "output_tokens": 300, "total_tokens": 1500},
            model_id="azure::gpt-5-mini",
            call_count=1,
        )
        cost = build_cost_payload(
            usage,
            {
                "azure::gpt-5-mini": {
                    "input": 0.25,
                    "output": 2.0,
                }
            },
        )

        assert cost["pricing_complete"]
        assert cost["input_cost_usd"] == pytest.approx(0.0003)
        assert cost["output_cost_usd"] == pytest.approx(0.0006)
        assert cost["total_cost_usd"] == pytest.approx(0.0009)

    def test_build_cost_payload_for_multi_model_breakdown(self):
        """Multi-agent usage breakdown should price each call with its own model."""
        usage = build_usage_payload(
            {
                "tokens": {"input_tokens": 300, "output_tokens": 75, "total_tokens": 375},
                "call_count": 2,
                "usage_available": True,
                "llm_usage_breakdown": [
                    {
                        "call_index": 1,
                        "agent_name": "supervisor",
                        "model_id": "azure::gpt-5-mini",
                        "tokens": {"input_tokens": 100, "output_tokens": 25, "total_tokens": 125},
                        "usage_available": True,
                    },
                    {
                        "call_index": 2,
                        "agent_name": "weather",
                        "model_id": "azure::gpt-5-mini",
                        "tokens": {"input_tokens": 200, "output_tokens": 50, "total_tokens": 250},
                        "usage_available": True,
                    },
                ],
            }
        )
        pricing = {
            "azure::gpt-5-mini": {"input": 0.25, "output": 2.0},
            "azure::gpt-5-mini": {"input": 0.05, "output": 0.4},
        }

        cost = build_cost_payload(usage, pricing)
        assert cost["pricing_complete"]
        assert len(cost["llm_cost_breakdown"]) == 2
        assert cost["input_cost_usd"] == pytest.approx(0.000035)
        assert cost["output_cost_usd"] == pytest.approx(0.00007)
        assert cost["total_cost_usd"] == pytest.approx(0.000105)

    def test_combine_cost_payloads_preserves_totals(self):
        """Combining response and evaluation cost payloads should sum totals."""
        response_cost = {
            "pricing_found": True,
            "pricing_complete": True,
            "tokens": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            "input_cost_usd": 0.001,
            "output_cost_usd": 0.002,
            "total_cost_usd": 0.003,
            "missing_pricing_models": [],
        }
        evaluation_cost = {
            "pricing_found": True,
            "pricing_complete": True,
            "tokens": {"input_tokens": 50, "output_tokens": 10, "total_tokens": 60},
            "input_cost_usd": 0.0005,
            "output_cost_usd": 0.001,
            "total_cost_usd": 0.0015,
            "missing_pricing_models": [],
        }

        combined = combine_cost_payloads([response_cost, evaluation_cost])
        assert combined["tokens"]["input_tokens"] == 150
        assert combined["tokens"]["output_tokens"] == 30
        assert combined["tokens"]["total_tokens"] == 180
        assert combined["total_cost_usd"] == pytest.approx(0.0045)
