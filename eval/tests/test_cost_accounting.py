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
    load_pricing_catalog,
    resolve_model_pricing,
    split_pricing_config,
    write_json_artifact,
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

    def test_build_usage_payload_extracts_cached_input_tokens(self):
        """Provider cache-hit token fields should normalize to cached_input_tokens."""
        payload = build_usage_payload(
            {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 400},
            },
            model_id="azure::gpt-5.4-mini",
            call_count=1,
        )

        assert payload["tokens"]["input_tokens"] == 1000
        assert payload["tokens"]["cached_input_tokens"] == 400

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

    def test_repository_catalog_includes_selected_azure_foundry_models(self):
        """The checked-in pricing catalog should cover selected Azure non-OpenAI models."""
        catalog = load_pricing_catalog()

        expected_prices = {
            "azure::deepseek-r1": (1.35, 5.4),
            "azure::phi-4-reasoning-plus": (0.125, 0.5),
            "azure::grok-4": (3.0, 15.0),
            "azure::llama-3.3-70b": (0.71, 0.71),
            "azure::kimi-k2.5": (0.6, 3.0),
            "azure::claude-haiku-4.5": (1.0, 5.0),
            "azure::claude-sonnet-4.5": (3.0, 15.0),
            "azure::claude-opus-4.1": (15.0, 75.0),
        }

        for model_id, (expected_input, expected_output) in expected_prices.items():
            pricing = resolve_model_pricing(catalog, model_id)
            assert pricing is not None, model_id
            assert pricing["input"] == pytest.approx(expected_input)
            assert pricing["output"] == pytest.approx(expected_output)

    def test_repository_catalog_includes_selected_2026_04_model_expansion_entries(self):
        """The 2026-04 pricing refresh should cover the newly added GPT-5.4 and Foundry entries."""
        catalog = load_pricing_catalog()

        expected_prices = {
            "azure::gpt-5.4-mini": (0.75, 4.5),
            "azure::gpt-5.4": (2.5, 15.0),
            "azure::gpt-5.4-pro": (30.0, 180.0),
            "openai::gpt-5.4-nano": (0.2, 1.25),
            "anthropic::claude-opus-4.7": (5.0, 25.0),
            "google::gemini-3.1-pro-preview": (2.0, 12.0),
        }

        for model_id, (expected_input, expected_output) in expected_prices.items():
            pricing = resolve_model_pricing(catalog, model_id)
            assert pricing is not None, model_id
            assert pricing["input"] == pytest.approx(expected_input)
            assert pricing["output"] == pytest.approx(expected_output)

    def test_repository_catalog_resolves_kimi_deployment_alias(self):
        """Kimi deployment labels should map to the catalog entry used for cost accounting."""
        catalog = load_pricing_catalog()

        pricing = resolve_model_pricing(catalog, "azure::Kimi-K2.5")

        assert pricing is not None
        assert pricing["pricing_lookup_key"] == "azure::kimi-k2.5"
        assert pricing["input"] == pytest.approx(0.6)
        assert pricing["output"] == pytest.approx(3.0)

    def test_repository_catalog_resolves_kimi_k25_punctuation_aliases_only(self):
        """Only punctuation variants of the same Kimi K2.5 SKU should resolve to the K2.5 pricing entry."""
        catalog = load_pricing_catalog()

        for model_id in ("azure::kimi-k2-5", "azure::kimi-k2_5"):
            pricing = resolve_model_pricing(catalog, model_id)
            assert pricing is not None, model_id
            assert pricing["pricing_lookup_key"] == "azure::kimi-k2.5"
            assert pricing["input"] == pytest.approx(0.6)
            assert pricing["output"] == pytest.approx(3.0)

    def test_repository_catalog_does_not_conflate_distinct_kimi_skus(self):
        """Kimi K2 and thinking-labelled variants must not silently inherit K2.5 pricing."""
        catalog = load_pricing_catalog()

        for model_id in (
            "azure::kimi-k2",
            "azure::kimi-k2-thinking",
            "azure::kimi-k2.5-thinking",
            "kimi-k2.5-thinking",
        ):
            assert resolve_model_pricing(catalog, model_id) is None, model_id


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

    def test_build_cost_payload_charges_cached_tokens_at_cached_rate(self):
        """Cached prompt tokens should use cached-input pricing when reported."""
        usage = build_usage_payload(
            {
                "input_tokens": 1000,
                "output_tokens": 100,
                "total_tokens": 1100,
                "cached_input_tokens": 400,
            },
            model_id="azure::gpt-5.4-mini",
            call_count=1,
        )

        cost = build_cost_payload(
            usage,
            {
                "azure::gpt-5.4-mini": {
                    "input": 1.0,
                    "input_cached": 0.1,
                    "output": 2.0,
                }
            },
        )

        assert cost["tokens"]["cached_input_tokens"] == 400
        assert cost["input_cost_usd"] == pytest.approx(0.00064)
        assert cost["output_cost_usd"] == pytest.approx(0.0002)
        assert cost["total_cost_usd"] == pytest.approx(0.00084)

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
                ],
            }
        )
        pricing = {
            "azure::gpt-5-mini": {"input": 0.25, "output": 2.0},
        }

        cost = build_cost_payload(usage, pricing)
        assert cost["pricing_complete"]
        assert len(cost["llm_cost_breakdown"]) == 2
        assert cost["llm_cost_breakdown"][1]["agent_name"] == "unattributed"
        assert cost["input_cost_usd"] == pytest.approx(0.000075)
        assert cost["output_cost_usd"] == pytest.approx(0.00015)
        assert cost["total_cost_usd"] == pytest.approx(0.000225)

    def test_combine_cost_payloads_preserves_totals(self):
        """Combining response and evaluation cost payloads should sum totals."""
        response_cost = {
            "model_id": "azure::gpt-5-mini",
            "pricing_lookup_key": "azure::gpt-5-mini",
            "pricing_found": True,
            "pricing_complete": True,
            "tokens": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            "input_per_million_usd": 0.25,
            "output_per_million_usd": 2.0,
            "cached_input_per_million_usd": 0.03,
            "input_cost_usd": 0.001,
            "output_cost_usd": 0.002,
            "total_cost_usd": 0.003,
            "missing_pricing_models": [],
        }
        response_cost["tokens"]["cached_input_tokens"] = 40
        evaluation_cost = {
            "model_id": "azure::gpt-5-mini",
            "pricing_lookup_key": "azure::gpt-5-mini",
            "pricing_found": True,
            "pricing_complete": True,
            "tokens": {"input_tokens": 50, "output_tokens": 10, "total_tokens": 60},
            "input_per_million_usd": 0.25,
            "output_per_million_usd": 2.0,
            "cached_input_per_million_usd": 0.03,
            "input_cost_usd": 0.0005,
            "output_cost_usd": 0.001,
            "total_cost_usd": 0.0015,
            "missing_pricing_models": [],
        }
        evaluation_cost["tokens"]["cached_input_tokens"] = 10

        combined = combine_cost_payloads([response_cost, evaluation_cost])
        assert combined["tokens"]["input_tokens"] == 150
        assert combined["tokens"]["output_tokens"] == 30
        assert combined["tokens"]["total_tokens"] == 180
        assert combined["tokens"]["cached_input_tokens"] == 50
        assert combined["model_id"] == "azure::gpt-5-mini"
        assert combined["pricing_lookup_key"] == "azure::gpt-5-mini"
        assert combined["output_per_million_usd"] == pytest.approx(2.0)
        assert combined["total_cost_usd"] == pytest.approx(0.0045)

    def test_combine_cost_payloads_does_not_mislabel_mixed_model_totals(self):
        """Mixed totals should not inherit a misleading single-model label."""
        response_cost = {
            "model_id": "azure::gpt-5.4-mini",
            "pricing_lookup_key": "azure::gpt-5.4-mini",
            "pricing_found": True,
            "pricing_complete": True,
            "tokens": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            "input_per_million_usd": 0.75,
            "output_per_million_usd": 4.5,
            "cached_input_per_million_usd": 0.075,
            "input_cost_usd": 0.001,
            "output_cost_usd": 0.002,
            "total_cost_usd": 0.003,
            "missing_pricing_models": [],
        }
        evaluation_cost = {
            "model_id": None,
            "pricing_lookup_key": None,
            "pricing_found": True,
            "pricing_complete": True,
            "tokens": {"input_tokens": 50, "output_tokens": 10, "total_tokens": 60},
            "input_per_million_usd": None,
            "output_per_million_usd": None,
            "cached_input_per_million_usd": None,
            "input_cost_usd": 0.0005,
            "output_cost_usd": 0.001,
            "total_cost_usd": 0.0015,
            "missing_pricing_models": [],
        }

        combined = combine_cost_payloads([response_cost, evaluation_cost])

        assert combined["tokens"]["total_tokens"] == 180
        assert combined["model_id"] is None
        assert combined["pricing_lookup_key"] is None
        assert combined["input_per_million_usd"] is None
        assert combined["contributing_model_ids"] == ["azure::gpt-5.4-mini"]
        assert combined["total_cost_usd"] == pytest.approx(0.0045)

    def test_write_json_artifact_formats_money_fields_with_minimum_decimals(self, tmp_path):
        """Persisted evaluation artefacts should keep USD fields readable at small magnitudes."""
        output_path = tmp_path / "artifact.json"

        write_json_artifact(
            {
                "input_cost_usd": 0.01,
                "output_cost_usd": 3.0,
                "total_cost_usd": 0.0007,
                "input_per_million_usd": 0.6,
            },
            output_path,
        )

        content = output_path.read_text(encoding="utf-8")
        assert '"input_cost_usd": 0.01000' in content
        assert '"output_cost_usd": 3.00000' in content
        assert '"total_cost_usd": 0.00070' in content
        assert '"input_per_million_usd": 0.60000' in content

    def test_build_cost_payload_treats_local_models_as_zero_cost(self):
        """Local providers such as LM Studio should default to zero-cost pricing."""
        usage = build_usage_payload(
            {"input_tokens": 1200, "output_tokens": 300, "total_tokens": 1500},
            model_id="lmstudio::qwen/qwen3.5-9b",
            call_count=1,
        )

        cost = build_cost_payload(usage, {})

        assert cost["pricing_found"] is True
        assert cost["pricing_complete"] is True
        assert cost["total_cost_usd"] == pytest.approx(0.0)
        assert cost["missing_pricing_models"] == []
