# ==========================================================================
# Master Thesis
#   - André Filipe Gomes Silvestre, 20240502
#
# Statistical analysis for LISBOA benchmark and ablation evaluation artefacts.
# Computes paired Wilcoxon signed-rank tests, paired bootstrap confidence
# intervals, and rank-biserial effect sizes for thesis result interpretation.
# ==========================================================================

# Required libraries:
# pip install pandas

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import datetime
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Sequence

from eval.runtime_utils import build_results_output_path, write_json_artifact

BENCHMARK_DIMENSIONS = (
    "composite_score",
    "factual_accuracy",
    "tool_usage",
    "completeness",
    "relevance",
    "response_quality",
)
ABLATION_PRIMARY_DIMENSIONS = (
    "ablation_quality_score",
    "factual_accuracy",
    "completeness",
    "relevance",
    "response_quality",
)
ABLATION_SUPPORTING_DIMENSIONS = ("tool_usage",)
DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_RANDOM_SEED = 20240502


def _load_json(path: str | Path) -> dict[str, Any]:
    """Load one JSON artefact."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _score_value(scores: dict[str, Any], dimension: str) -> float | None:
    """Return a numeric score, deriving ablation quality when necessary."""
    if dimension == "ablation_quality_score":
        direct_value = scores.get("ablation_quality_score")
        if direct_value is not None:
            return float(direct_value)
        values = [scores.get(field) for field in ABLATION_PRIMARY_DIMENSIONS[1:]]
        if any(value is None for value in values):
            return None
        return float(sum(float(value) for value in values) / len(values))

    value = scores.get(dimension)
    return None if value is None else float(value)


def _model_label(model_id: str | None) -> str:
    """Return a compact model label for statistical tables."""
    label = str(model_id or "unknown")
    return label.split("::", 1)[1] if "::" in label else label


def _model_sort_key(model_label: str) -> tuple[int, str]:
    """Sort common thesis model pairs in the intended closed/open order."""
    lowered = model_label.lower()
    if lowered.startswith("gpt"):
        return (0, lowered)
    if lowered.startswith("kimi"):
        return (1, lowered)
    return (2, lowered)


def _normal_cdf(value: float) -> float:
    """Return the standard normal cumulative distribution value."""
    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def _rank_absolute_differences(differences: Sequence[float]) -> list[float]:
    """Rank absolute paired differences with average ranks for ties."""
    indexed = sorted(enumerate(abs(value) for value in differences), key=lambda item: item[1])
    ranks = [0.0] * len(indexed)
    position = 0
    while position < len(indexed):
        tie_end = position + 1
        while tie_end < len(indexed) and indexed[tie_end][1] == indexed[position][1]:
            tie_end += 1
        average_rank = (position + 1 + tie_end) / 2.0
        for tied_position in range(position, tie_end):
            original_index = indexed[tied_position][0]
            ranks[original_index] = average_rank
        position = tie_end
    return ranks


def _wilcoxon_signed_rank(differences: Sequence[float]) -> dict[str, Any]:
    """Compute a paired Wilcoxon signed-rank test with SciPy fallback support."""
    nonzero = [float(value) for value in differences if abs(float(value)) > 1e-12]
    if not nonzero:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "method": "all_zero_differences",
            "n_nonzero": 0,
        }

    try:
        from scipy.stats import wilcoxon  # type: ignore

        result = wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
        return {
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "method": "scipy_wilcoxon_two_sided",
            "n_nonzero": len(nonzero),
        }
    except Exception:
        ranks = _rank_absolute_differences(nonzero)
        positive_rank_sum = sum(rank for rank, diff in zip(ranks, nonzero, strict=False) if diff > 0)
        negative_rank_sum = sum(rank for rank, diff in zip(ranks, nonzero, strict=False) if diff < 0)
        statistic = min(positive_rank_sum, negative_rank_sum)
        n = len(nonzero)
        expected = n * (n + 1) / 4.0
        variance = n * (n + 1) * (2 * n + 1) / 24.0
        if variance <= 0:
            p_value = 1.0
        else:
            z_value = (statistic - expected + 0.5) / math.sqrt(variance)
            p_value = min(1.0, 2.0 * _normal_cdf(z_value))
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "method": "normal_approximation_no_scipy",
            "n_nonzero": n,
        }


def _rank_biserial_correlation(differences: Sequence[float]) -> float | None:
    """Return paired rank-biserial correlation for signed differences."""
    nonzero = [float(value) for value in differences if abs(float(value)) > 1e-12]
    if not nonzero:
        return 0.0
    ranks = _rank_absolute_differences(nonzero)
    positive_rank_sum = sum(rank for rank, diff in zip(ranks, nonzero, strict=False) if diff > 0)
    negative_rank_sum = sum(rank for rank, diff in zip(ranks, nonzero, strict=False) if diff < 0)
    total_rank_sum = sum(ranks)
    if total_rank_sum == 0:
        return None
    return (positive_rank_sum - negative_rank_sum) / total_rank_sum


def _bootstrap_mean_difference_ci(
    differences: Sequence[float],
    *,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, float | int]:
    """Compute a paired bootstrap confidence interval for the mean difference."""
    clean = [float(value) for value in differences]
    if not clean:
        return {"low": math.nan, "high": math.nan, "iterations": 0, "confidence_level": confidence_level}

    generator = random.Random(seed)
    boot_means = []
    for _ in range(iterations):
        sample = [clean[generator.randrange(len(clean))] for _ in clean]
        boot_means.append(mean(sample))
    boot_means.sort()

    alpha = 1.0 - confidence_level
    low_index = max(0, int((alpha / 2.0) * iterations))
    high_index = min(iterations - 1, int((1.0 - alpha / 2.0) * iterations) - 1)
    return {
        "low": float(boot_means[low_index]),
        "high": float(boot_means[high_index]),
        "iterations": iterations,
        "confidence_level": confidence_level,
    }


def _paired_records(
    rows: Iterable[dict[str, Any]],
    *,
    left_key: str,
    right_key: str,
    key_field: str,
    value_field: str,
) -> list[tuple[float, float]]:
    """Return paired numeric values aligned by query id."""
    by_key: dict[str, dict[str, float]] = {}
    for row in rows:
        query_id = str(row.get("id"))
        side = str(row.get(key_field))
        value = row.get(value_field)
        if value is None:
            continue
        by_key.setdefault(query_id, {})[side] = float(value)
    return [
        (values[left_key], values[right_key])
        for values in by_key.values()
        if left_key in values and right_key in values
    ]


def _comparison_result(
    *,
    comparison_type: str,
    dimension: str,
    label_a: str,
    label_b: str,
    pairs: Sequence[tuple[float, float]],
    group: dict[str, str],
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Summarize one paired statistical comparison."""
    differences = [right - left for left, right in pairs]
    wilcoxon = _wilcoxon_signed_rank(differences)
    ci = _bootstrap_mean_difference_ci(differences, iterations=bootstrap_iterations, seed=seed)
    values_a = [left for left, _right in pairs]
    values_b = [right for _left, right in pairs]
    return {
        "comparison_type": comparison_type,
        **group,
        "dimension": dimension,
        "label_a": label_a,
        "label_b": label_b,
        "n_pairs": len(pairs),
        "mean_a": round(mean(values_a), 4) if values_a else None,
        "mean_b": round(mean(values_b), 4) if values_b else None,
        "mean_diff_b_minus_a": round(mean(differences), 4) if differences else None,
        "median_diff_b_minus_a": round(median(differences), 4) if differences else None,
        "wilcoxon_statistic": wilcoxon["statistic"],
        "wilcoxon_p_value": wilcoxon["p_value"],
        "wilcoxon_method": wilcoxon["method"],
        "rank_biserial_correlation": (
            None
            if _rank_biserial_correlation(differences) is None
            else round(float(_rank_biserial_correlation(differences)), 4)
        ),
        "bootstrap_ci_low": round(float(ci["low"]), 4) if not math.isnan(float(ci["low"])) else None,
        "bootstrap_ci_high": round(float(ci["high"]), 4) if not math.isnan(float(ci["high"])) else None,
        "bootstrap_iterations": ci["iterations"],
        "confidence_level": ci["confidence_level"],
    }


def flatten_benchmark_scores(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten benchmark scores for paired model comparisons."""
    rows = []
    for record in payload.get("benchmark_results", []):
        scores = record.get("scores") or {}
        for dimension in BENCHMARK_DIMENSIONS:
            value = _score_value(scores, dimension)
            if value is None:
                continue
            rows.append(
                {
                    "id": record.get("id"),
                    "domain": record.get("domain"),
                    "response_model": record.get("response_model"),
                    "response_model_label": _model_label(record.get("response_model")),
                    "dimension": dimension,
                    "score": value,
                }
            )
    return rows


def flatten_ablation_scores(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten ablation scores for paired zero-shot versus LISBOA comparisons."""
    rows = []
    primary_profile = (
        payload.get("summary", {}).get("primary_comparison_profile")
        or payload.get("ablation_metadata", {}).get("primary_comparison_profile")
        or "primary"
    )
    dimensions = ABLATION_PRIMARY_DIMENSIONS + ABLATION_SUPPORTING_DIMENSIONS

    for record in payload.get("ablation_results", []):
        comparisons_root = record.get("comparisons") or {primary_profile: {"metrics": record.get("metrics") or {}}}
        for profile_name, comparison in comparisons_root.items():
            profile = (comparison or {}).get("profile") or {}
            profile_label = _model_label(
                ((profile.get("zero_shot_model_config") or {}).get("model_id"))
                or profile.get("profile_id")
                or profile_name
            )
            metrics_root = (comparison or {}).get("metrics") or {}
            for arm in ("zero_shot", "lisboa"):
                scores = (metrics_root.get(arm) or {}).get("scores") or {}
                for dimension in dimensions:
                    value = _score_value(scores, dimension)
                    if value is None:
                        continue
                    rows.append(
                        {
                            "id": record.get("id"),
                            "domain": record.get("domain"),
                            "profile_name": profile_name,
                            "profile_label": profile_label,
                            "arm": arm,
                            "dimension": dimension,
                            "score": value,
                        }
                    )
    return rows


def benchmark_model_tests(
    payload: dict[str, Any],
    *,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Compute paired model-vs-model tests per benchmark agent/domain."""
    rows = flatten_benchmark_scores(payload)
    results = []
    domains = sorted({str(row["domain"]) for row in rows})
    for domain in domains:
        domain_rows = [row for row in rows if row["domain"] == domain]
        models = sorted({str(row["response_model_label"]) for row in domain_rows}, key=_model_sort_key)
        for model_a, model_b in combinations(models, 2):
            for dimension in BENCHMARK_DIMENSIONS:
                dimension_rows = [row for row in domain_rows if row["dimension"] == dimension]
                pairs = _paired_records(
                    dimension_rows,
                    left_key=model_a,
                    right_key=model_b,
                    key_field="response_model_label",
                    value_field="score",
                )
                if pairs:
                    results.append(
                        _comparison_result(
                            comparison_type="benchmark_model_pair",
                            dimension=dimension,
                            label_a=model_a,
                            label_b=model_b,
                            pairs=pairs,
                            group={"domain": domain},
                            bootstrap_iterations=bootstrap_iterations,
                            seed=seed,
                        )
                    )
    return results


def ablation_arm_tests(
    payload: dict[str, Any],
    *,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Compute paired LISBOA-vs-zero-shot tests per ablation profile/model."""
    rows = flatten_ablation_scores(payload)
    results = []
    profiles = sorted({str(row["profile_label"]) for row in rows})
    dimensions = ABLATION_PRIMARY_DIMENSIONS + ABLATION_SUPPORTING_DIMENSIONS
    for profile_label in profiles:
        profile_rows = [row for row in rows if row["profile_label"] == profile_label]
        domains = sorted({str(row["domain"]) for row in profile_rows})
        for domain in ["all", *domains]:
            domain_rows = profile_rows if domain == "all" else [row for row in profile_rows if row["domain"] == domain]
            for dimension in dimensions:
                dimension_rows = [row for row in domain_rows if row["dimension"] == dimension]
                pairs = _paired_records(
                    dimension_rows,
                    left_key="zero_shot",
                    right_key="lisboa",
                    key_field="arm",
                    value_field="score",
                )
                if pairs:
                    results.append(
                        _comparison_result(
                            comparison_type="ablation_lisboa_vs_zero_shot",
                            dimension=dimension,
                            label_a="Zero-shot",
                            label_b="LISBOA",
                            pairs=pairs,
                            group={"profile": profile_label, "domain": domain},
                            bootstrap_iterations=bootstrap_iterations,
                            seed=seed,
                        )
                    )
    return results


def run_statistical_analysis(
    *,
    benchmark_path: str | Path | None = None,
    ablation_path: str | Path | None = None,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    """Run all available statistical comparisons and return a JSON payload."""
    benchmark_payload = _load_json(benchmark_path) if benchmark_path else None
    ablation_payload = _load_json(ablation_path) if ablation_path else None
    benchmark_results = (
        benchmark_model_tests(benchmark_payload, bootstrap_iterations=bootstrap_iterations, seed=seed)
        if benchmark_payload
        else []
    )
    ablation_results = (
        ablation_arm_tests(ablation_payload, bootstrap_iterations=bootstrap_iterations, seed=seed)
        if ablation_payload
        else []
    )
    return {
        "statistical_metadata": {
            "benchmark_path": str(benchmark_path) if benchmark_path else None,
            "ablation_path": str(ablation_path) if ablation_path else None,
            "bootstrap_iterations": bootstrap_iterations,
            "random_seed": seed,
            "benchmark_dimensions": list(BENCHMARK_DIMENSIONS),
            "ablation_primary_dimensions": list(ABLATION_PRIMARY_DIMENSIONS),
            "ablation_supporting_dimensions": list(ABLATION_SUPPORTING_DIMENSIONS),
            "paired_difference_direction": "label_b_minus_label_a",
        },
        "benchmark_model_tests": benchmark_results,
        "ablation_arm_tests": ablation_results,
    }


def write_csv(rows: Sequence[dict[str, Any]], output_path: str | Path) -> None:
    """Write statistical rows to CSV for appendix tables."""
    if not rows:
        Path(output_path).write_text("", encoding="utf-8")
        return
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """CLI entrypoint for statistical analysis artefact generation."""
    parser = argparse.ArgumentParser(description="Run LISBOA paired statistical analyses.")
    parser.add_argument("--benchmark", type=Path, help="Path to benchmark_results_*.json")
    parser.add_argument("--ablation", type=Path, help="Path to ablation_results_*.json")
    parser.add_argument("--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--output-prefix", default="statistical_analysis")
    args = parser.parse_args()

    payload = run_statistical_analysis(
        benchmark_path=args.benchmark,
        ablation_path=args.ablation,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = build_results_output_path("statistics", args.output_prefix, timestamp)
    write_json_artifact(payload, output_path)

    benchmark_csv = output_path.with_name(output_path.stem + "_benchmark.csv")
    ablation_csv = output_path.with_name(output_path.stem + "_ablation.csv")
    write_csv(payload["benchmark_model_tests"], benchmark_csv)
    write_csv(payload["ablation_arm_tests"], ablation_csv)
    print(f"Statistical analysis saved to {output_path}")
    print(f"Benchmark CSV saved to {benchmark_csv}")
    print(f"Ablation CSV saved to {ablation_csv}")


if __name__ == "__main__":
    main()
