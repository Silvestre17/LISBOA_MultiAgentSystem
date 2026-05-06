# ==========================================================================
# Master Thesis - Statistical Analysis Tests
#   - André Filipe Gomes Silvestre, 20240502
#
#   Unit tests for paired statistical analysis helpers used by the thesis
#   benchmark and ablation analysis pipeline.
# ==========================================================================

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest

from eval.statistical_analysis import (
    ablation_arm_tests,
    benchmark_model_tests,
    flatten_ablation_scores,
)


def test_flatten_ablation_scores_derives_quality_without_tool_usage() -> None:
    """Ablation quality must use the four non-tool judge dimensions."""
    payload = {
        "ablation_results": [
            {
                "id": "Q1",
                "domain": "weather",
                "comparisons": {
                    "closed_source": {
                        "profile": {"zero_shot_model_config": {"model_id": "azure::gpt-5.4-mini"}},
                        "metrics": {
                            "zero_shot": {
                                "scores": {
                                    "factual_accuracy": 5,
                                    "tool_usage": 1,
                                    "completeness": 4,
                                    "relevance": 4,
                                    "response_quality": 3,
                                }
                            },
                            "lisboa": {
                                "scores": {
                                    "factual_accuracy": 5,
                                    "tool_usage": 5,
                                    "completeness": 5,
                                    "relevance": 5,
                                    "response_quality": 5,
                                }
                            },
                        },
                    }
                },
            }
        ]
    }

    rows = flatten_ablation_scores(payload)
    quality_rows = [row for row in rows if row["dimension"] == "ablation_quality_score"]

    assert len(quality_rows) == 2
    assert {row["arm"]: row["score"] for row in quality_rows} == {
        "zero_shot": pytest.approx(4.0),
        "lisboa": pytest.approx(5.0),
    }


def test_ablation_arm_tests_compare_lisboa_to_zero_shot() -> None:
    """Ablation tests should produce paired LISBOA-minus-zero-shot effects."""
    payload = {
        "ablation_results": [
            {
                "id": "Q1",
                "domain": "weather",
                "comparisons": {
                    "closed_source": {
                        "profile": {"zero_shot_model_config": {"model_id": "azure::gpt-5.4-mini"}},
                        "metrics": {
                            "zero_shot": {"scores": {"ablation_quality_score": 3, "factual_accuracy": 3, "completeness": 3, "relevance": 3, "response_quality": 3, "tool_usage": 1}},
                            "lisboa": {"scores": {"ablation_quality_score": 5, "factual_accuracy": 5, "completeness": 5, "relevance": 5, "response_quality": 5, "tool_usage": 5}},
                        },
                    }
                },
            },
            {
                "id": "Q2",
                "domain": "weather",
                "comparisons": {
                    "closed_source": {
                        "profile": {"zero_shot_model_config": {"model_id": "azure::gpt-5.4-mini"}},
                        "metrics": {
                            "zero_shot": {"scores": {"ablation_quality_score": 4, "factual_accuracy": 4, "completeness": 4, "relevance": 4, "response_quality": 4, "tool_usage": 1}},
                            "lisboa": {"scores": {"ablation_quality_score": 5, "factual_accuracy": 5, "completeness": 5, "relevance": 5, "response_quality": 5, "tool_usage": 5}},
                        },
                    }
                },
            },
        ]
    }

    results = ablation_arm_tests(payload, bootstrap_iterations=100, seed=7)
    quality = next(
        result for result in results
        if result["domain"] == "all" and result["dimension"] == "ablation_quality_score"
    )

    assert quality["label_a"] == "Zero-shot"
    assert quality["label_b"] == "LISBOA"
    assert quality["n_pairs"] == 2
    assert quality["mean_diff_b_minus_a"] == pytest.approx(1.5)
    assert quality["rank_biserial_correlation"] == pytest.approx(1.0)


def test_benchmark_model_tests_pair_models_per_domain() -> None:
    """Benchmark tests should pair response models within each isolated agent domain."""
    payload = {
        "benchmark_results": [
            {"id": "W1", "domain": "weather", "response_model": "azure::gpt-5.4-mini", "scores": {"composite_score": 4, "factual_accuracy": 4, "tool_usage": 4, "completeness": 4, "relevance": 4, "response_quality": 4}},
            {"id": "W1", "domain": "weather", "response_model": "azure::Kimi-K2.5", "scores": {"composite_score": 5, "factual_accuracy": 5, "tool_usage": 5, "completeness": 5, "relevance": 5, "response_quality": 5}},
            {"id": "W2", "domain": "weather", "response_model": "azure::gpt-5.4-mini", "scores": {"composite_score": 3, "factual_accuracy": 3, "tool_usage": 3, "completeness": 3, "relevance": 3, "response_quality": 3}},
            {"id": "W2", "domain": "weather", "response_model": "azure::Kimi-K2.5", "scores": {"composite_score": 4, "factual_accuracy": 4, "tool_usage": 4, "completeness": 4, "relevance": 4, "response_quality": 4}},
        ]
    }

    results = benchmark_model_tests(payload, bootstrap_iterations=100, seed=7)
    composite = next(result for result in results if result["dimension"] == "composite_score")

    assert composite["domain"] == "weather"
    assert composite["n_pairs"] == 2
    assert composite["mean_diff_b_minus_a"] == pytest.approx(1.0)
