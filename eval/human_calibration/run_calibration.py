# ==========================================================================
# Master Thesis - Human-Judge Calibration Analysis
#   - André Filipe Gomes Silvestre, 20240502
#
# Computes inter-rater agreement (Cohen's Kappa, Pearson correlation)
# between human scores and LLM-as-a-Judge scores for calibration.
#
# Usage:
#   > python -m eval.human_calibration.run_calibration \
#       --human eval/human_calibration/calibration_filled.json \
#       --benchmark eval/results/benchmark/benchmark_results_YYYYMMDD_HHMMSS.json
#       Use averaged benchmark judge scores for the calibration summary.
#   > python -m eval.human_calibration.run_calibration \
#       --human eval/human_calibration/calibration_filled.json \
#       --benchmark eval/results/benchmark/benchmark_results_YYYYMMDD_HHMMSS.json \
#       --judge-source openai::gpt-5.4-mini \
#       --output eval/results/calibration/calibration_summary.json
#       Use a specific judge entry from `judge_runs` and persist the output to a fixed JSON path.
# ==========================================================================

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from eval.runtime_utils import build_results_output_path

DIMENSIONS = [
    "factual_accuracy",
    "tool_usage",
    "completeness",
    "relevance",
    "response_quality",
]


def _has_complete_numeric_scores(scores: dict[str, Any]) -> bool:
    """Return whether all calibration dimensions contain usable numeric scores."""
    for dimension in DIMENSIONS:
        value = scores.get(dimension)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
    return True


def _rounded_ordinal_scores(values: list[float]) -> list[int]:
    """Round continuous judge averages to the nearest 1-5 ordinal score for Kappa."""
    return [min(5, max(1, int(round(value)))) for value in values]


def load_human_scores(filepath: str) -> dict[str, dict[str, int]]:
    """Load human calibration scores from filled template.

    Args:
        filepath: Path to the filled calibration JSON.

    Returns:
        Dict mapping source_id to human scores dict.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    scores = {}
    for entry in data.get("entries", []):
        source_id = entry["source_id"]
        human = entry.get("human_scores", {})
        # Only include if all dimensions are scored
        if all(human.get(d) is not None for d in DIMENSIONS):
            scores[source_id] = {d: human[d] for d in DIMENSIONS}

    return scores


def load_judge_scores(
    filepath: str,
    *,
    judge_source: str = "average",
) -> dict[str, dict[str, float]]:
    """Load LLM judge scores from benchmark results.

    Args:
        filepath: Path to the benchmark results JSON.
        judge_source: ``average`` for the compatibility score block, or a
            specific judge model id to select one judge run from ``judge_runs``.

    Returns:
        Dict mapping entry id to judge scores dict.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    scores = {}
    for entry in data.get("benchmark_results", []):
        entry_id = entry["id"]
        judge = entry.get("scores", {})
        if judge_source != "average":
            selected_run = next(
                (
                    judge_run
                    for judge_run in entry.get("judge_runs", [])
                    if judge_run.get("judge_model") == judge_source
                    and not judge_run.get("error")
                ),
                None,
            )
            judge = selected_run.get("scores", {}) if isinstance(selected_run, dict) else {}
        if _has_complete_numeric_scores(judge):
            scores[entry_id] = {d: float(judge[d]) for d in DIMENSIONS}

    return scores


def cohens_kappa(human: list[int], judge: list[int], k: int = 5) -> float:
    """Compute Cohen's Kappa for ordinal agreement.

    Args:
        human: List of human scores (1-5).
        judge: List of judge scores (1-5).
        k: Number of categories.

    Returns:
        Cohen's Kappa coefficient (-1 to 1).
    """
    n = len(human)
    if n == 0:
        return 0.0

    # Build confusion matrix
    matrix = np.zeros((k, k), dtype=int)
    for h, j in zip(human, judge):
        matrix[h - 1][j - 1] += 1

    # Observed agreement
    po = np.trace(matrix) / n

    # Expected agreement
    row_sums = matrix.sum(axis=1) / n
    col_sums = matrix.sum(axis=0) / n
    pe = np.sum(row_sums * col_sums)

    if pe == 1.0:
        return 1.0

    kappa = (po - pe) / (1 - pe)
    return round(float(kappa), 4)


def pearson_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient.

    Args:
        x: First variable values.
        y: Second variable values.

    Returns:
        Pearson r (-1 to 1).
    """
    if len(x) < 2 or len(y) < 2:
        return 0.0
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return round(float(np.corrcoef(x, y)[0, 1]), 4)


def mean_absolute_error(human: list[float], judge: list[float]) -> float:
    """Compute Mean Absolute Error between human and judge scores.

    Args:
        human: Human scores.
        judge: Judge scores.

    Returns:
        MAE (0 to 4 for 1-5 scale).
    """
    if not human:
        return 0.0
    return round(float(np.mean(np.abs(np.array(human) - np.array(judge)))), 4)


def run_calibration(
    human_file: str,
    benchmark_file: str,
    *,
    judge_source: str = "average",
) -> dict[str, Any]:
    """Run full calibration analysis.

    Args:
        human_file: Path to filled human calibration JSON.
        benchmark_file: Path to benchmark results JSON.

    Returns:
        Dict with per-dimension and overall agreement metrics.
    """
    human_scores = load_human_scores(human_file)
    judge_scores = load_judge_scores(benchmark_file, judge_source=judge_source)

    if not human_scores:
        print("\033[1;31mERROR:\033[0m No valid human scores found. "
              "Make sure all dimensions are filled in the calibration template.")
        return {}

    # Find matching entries
    matched_ids = set(human_scores.keys()) & set(judge_scores.keys())
    if not matched_ids:
        print("\033[1;31mERROR:\033[0m No matching entries between human and judge scores.")
        print(f"  Human IDs: {sorted(human_scores.keys())}")
        print(f"  Judge IDs: {sorted(list(judge_scores.keys())[:10])}...")
        return {}

    print("\n\033[1mCalibration Analysis\033[0m")
    print(f"  Matched entries: {len(matched_ids)}")
    print(f"  Human entries: {len(human_scores)}")
    print(f"  Judge entries: {len(judge_scores)}")
    print()

    results = {"matched_count": len(matched_ids), "per_dimension": {}}

    all_human = []
    all_judge = []

    for dim in DIMENSIONS:
        h_vals = [human_scores[eid][dim] for eid in sorted(matched_ids)]
        j_vals = [float(judge_scores[eid][dim]) for eid in sorted(matched_ids)]

        kappa = cohens_kappa(h_vals, _rounded_ordinal_scores(j_vals))
        pearson = pearson_correlation(h_vals, j_vals)
        mae = mean_absolute_error(h_vals, j_vals)

        results["per_dimension"][dim] = {
            "cohens_kappa": kappa,
            "pearson_r": pearson,
            "mae": mae,
            "human_mean": round(float(np.mean(h_vals)), 2),
            "judge_mean": round(float(np.mean(j_vals)), 2),
        }

        # Kappa interpretation
        if kappa > 0.8:
            interp = "Almost Perfect"
        elif kappa > 0.6:
            interp = "Substantial"
        elif kappa > 0.4:
            interp = "Moderate"
        elif kappa > 0.2:
            interp = "Fair"
        else:
            interp = "Slight/Poor"

        print(f"  \033[1m{dim:25s}\033[0m  "
              f"Kappa={kappa:+.3f} ({interp:16s})  "
              f"Pearson={pearson:+.3f}  "
              f"MAE={mae:.3f}  "
              f"H_mean={np.mean(h_vals):.1f}  J_mean={np.mean(j_vals):.1f}")

        all_human.extend(h_vals)
        all_judge.extend(j_vals)

    # Overall
    overall_kappa = cohens_kappa(all_human, _rounded_ordinal_scores(all_judge))
    overall_pearson = pearson_correlation(all_human, all_judge)
    overall_mae = mean_absolute_error(all_human, all_judge)

    results["overall"] = {
        "cohens_kappa": overall_kappa,
        "pearson_r": overall_pearson,
        "mae": overall_mae,
    }

    print(f"\n  \033[1m{'OVERALL':25s}\033[0m  "
          f"Kappa={overall_kappa:+.3f}  "
          f"Pearson={overall_pearson:+.3f}  "
          f"MAE={overall_mae:.3f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human-Judge Calibration Analysis")
    parser.add_argument("--human", required=True, help="Path to filled calibration JSON")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark results JSON")
    parser.add_argument(
        "--judge-source",
        default="average",
        help="Use averaged benchmark scores (default) or provide a specific judge model id from judge_runs.",
    )
    parser.add_argument("--output", default=None, help="Output file for results (optional, defaults to eval/results/calibration/)")
    args = parser.parse_args()

    results = run_calibration(args.human, args.benchmark, judge_source=args.judge_source)

    if results:
        output_path = (
            Path(args.output)
            if args.output
            else build_results_output_path(
                "calibration",
                "calibration_summary",
                datetime.now().strftime("%Y%m%d_%H%M%S"),
            )
        )

        with open(args.benchmark, "r", encoding="utf-8") as f:
            benchmark_artifact = json.load(f)

        payload = {
            "calibration_metadata": {
                "human_scores_path": args.human,
                "benchmark_results_path": args.benchmark,
                "judge_source": args.judge_source,
                "benchmark_response_models": benchmark_artifact.get("benchmark_metadata", {}).get("response_models"),
                "benchmark_evaluation_model": benchmark_artifact.get("benchmark_metadata", {}).get("evaluation_model"),
                "benchmark_evaluation_models": benchmark_artifact.get("benchmark_metadata", {}).get("evaluation_models"),
                "timestamp": datetime.now().isoformat(),
                "output_directory": str(output_path.parent),
            },
            "summary": results,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\n\033[1;32mResults saved to {output_path}\033[0m")
