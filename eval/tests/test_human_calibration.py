# ==========================================================================
# Master Thesis - Human Calibration Reader Tests
#   - André Filipe Gomes Silvestre, 20240502
#
#   Deterministic tests for calibration readers that consume benchmark artefacts
#   with averaged multi-judge scores and judge-specific runs.
# ==========================================================================

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from eval.human_calibration.run_calibration import load_judge_scores, run_calibration


def test_load_judge_scores_uses_averaged_scores_by_default(tmp_path: Path) -> None:
    """Calibration should keep working with the compatibility average score block."""
    artifact_path = tmp_path / "benchmark.json"
    artifact_path.write_text(
        json.dumps(
            {
                "benchmark_results": [
                    {
                        "id": "W01",
                        "scores": {
                            "factual_accuracy": 4.5,
                            "tool_usage": 5.0,
                            "completeness": 4.5,
                            "relevance": 4.0,
                            "response_quality": 4.5,
                            "composite_score": 4.5,
                            "reasoning": "Average of two judges.",
                        },
                        "judge_runs": [
                            {
                                "judge_model": "azure::gpt-5-mini",
                                "scores": {
                                    "factual_accuracy": 5,
                                    "tool_usage": 5,
                                    "completeness": 5,
                                    "relevance": 4,
                                    "response_quality": 5,
                                    "composite_score": 4.8,
                                    "reasoning": "Closed judge",
                                },
                            },
                            {
                                "judge_model": "lmstudio::qwen/qwen3.5-9b",
                                "scores": {
                                    "factual_accuracy": 4,
                                    "tool_usage": 5,
                                    "completeness": 4,
                                    "relevance": 4,
                                    "response_quality": 4,
                                    "composite_score": 4.2,
                                    "reasoning": "Open judge",
                                },
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    scores = load_judge_scores(str(artifact_path))

    assert scores["W01"] == {
        "factual_accuracy": 4.5,
        "tool_usage": 5.0,
        "completeness": 4.5,
        "relevance": 4.0,
        "response_quality": 4.5,
    }


def test_load_judge_scores_can_select_a_specific_judge(tmp_path: Path) -> None:
    """Calibration readers should allow selecting one judge run by model id."""
    artifact_path = tmp_path / "benchmark.json"
    artifact_path.write_text(
        json.dumps(
            {
                "benchmark_results": [
                    {
                        "id": "T01",
                        "scores": {
                            "factual_accuracy": 4.5,
                            "tool_usage": 4.5,
                            "completeness": 4.0,
                            "relevance": 4.0,
                            "response_quality": 4.5,
                            "composite_score": 4.3,
                            "reasoning": "Average of two judges.",
                        },
                        "judge_runs": [
                            {
                                "judge_model": "azure::gpt-5-mini",
                                "scores": {
                                    "factual_accuracy": 5,
                                    "tool_usage": 5,
                                    "completeness": 4,
                                    "relevance": 4,
                                    "response_quality": 5,
                                    "composite_score": 4.6,
                                    "reasoning": "Closed judge",
                                },
                            },
                            {
                                "judge_model": "lmstudio::qwen/qwen3.5-9b",
                                "scores": {
                                    "factual_accuracy": 4,
                                    "tool_usage": 4,
                                    "completeness": 4,
                                    "relevance": 4,
                                    "response_quality": 4,
                                    "composite_score": 4.0,
                                    "reasoning": "Open judge",
                                },
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    scores = load_judge_scores(
        str(artifact_path),
        judge_source="lmstudio::qwen/qwen3.5-9b",
    )

    assert scores["T01"] == {
        "factual_accuracy": 4,
        "tool_usage": 4,
        "completeness": 4,
        "relevance": 4,
        "response_quality": 4,
    }


def test_load_judge_scores_skips_none_and_failed_judge_blocks(tmp_path: Path) -> None:
    """Calibration input should exclude unusable average and failed judge-specific score blocks."""
    artifact_path = tmp_path / "benchmark.json"
    artifact_path.write_text(
        json.dumps(
            {
                "benchmark_results": [
                    {
                        "id": "BAD",
                        "scores": {
                            "factual_accuracy": None,
                            "tool_usage": None,
                            "completeness": None,
                            "relevance": None,
                            "response_quality": None,
                        },
                        "judge_runs": [
                            {
                                "judge_model": "azure::gpt-5-mini",
                                "error": "Judge failed",
                                "scores": {
                                    "factual_accuracy": 5,
                                    "tool_usage": 5,
                                    "completeness": 5,
                                    "relevance": 5,
                                    "response_quality": 5,
                                },
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_judge_scores(str(artifact_path)) == {}
    assert load_judge_scores(str(artifact_path), judge_source="azure::gpt-5-mini") == {}


def test_run_calibration_preserves_average_float_scores(tmp_path: Path) -> None:
    """Averaged judge scores such as 4.5 should not be truncated before MAE/Pearson."""
    human_path = tmp_path / "human.json"
    benchmark_path = tmp_path / "benchmark.json"
    human_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "source_id": "W01",
                        "human_scores": {
                            "factual_accuracy": 4,
                            "tool_usage": 4,
                            "completeness": 4,
                            "relevance": 4,
                            "response_quality": 4,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    benchmark_path.write_text(
        json.dumps(
            {
                "benchmark_results": [
                    {
                        "id": "W01",
                        "scores": {
                            "factual_accuracy": 4.5,
                            "tool_usage": 4.5,
                            "completeness": 4.5,
                            "relevance": 4.5,
                            "response_quality": 4.5,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = run_calibration(str(human_path), str(benchmark_path))

    assert result["per_dimension"]["factual_accuracy"]["judge_mean"] == 4.5
    assert result["per_dimension"]["factual_accuracy"]["mae"] == 0.5
