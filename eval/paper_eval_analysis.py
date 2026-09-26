# ==========================================================================
# Master Thesis - Paper Evaluation Analysis (RINENG revision)
#   - André Filipe Gomes Silvestre, 20240502
#
#   Turns one ablation run into every number the revised paper reports:
#     - provenance: evaluated commit, sessions, models, data, protocol checks,
#       for the ablation and the benchmark, and whether both ran on one system;
#     - quality: paired Wilcoxon, rank-biserial r, and bootstrap CI per model
#       and domain (eval/statistical_analysis.py), plus itinerary and
#       cross-domain subsets from the annotations;
#     - operation: latency and cost per model and condition, with the cost
#       recomputed from tokens and the price table;
#     - end to end: supervisor routing, coverage of the expected agents,
#       planner use, execution types, and QA paths (LISBOA condition);
#     - judges: QWK, ICC(2,1), ICC(2,2), exact agreement, and the gain
#       measured by each judge alone (same or other model family);
#     - constraints: the itinerary checklist of eval/constraint_judge.py.
#   Nothing here calls a model.
#
# Usage:
#   > python -m eval.paper_eval_analysis --ablation eval/results/ablation/ablation_final_<timestamp>.json
#   > python -m eval.paper_eval_analysis --ablation <file> --benchmark <file> --constraints <file>
# ==========================================================================

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Sequence

import numpy as np

from eval.runtime_utils import build_results_output_path, describe_file, load_pricing_catalog, write_json_artifact
from eval.statistical_analysis import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_RANDOM_SEED,
    _comparison_result,
    _wilcoxon_signed_rank,
    ablation_arm_tests,
    benchmark_model_tests,
)

DEFAULT_ANNOTATIONS_PATH = Path(__file__).with_name("paper_eval_annotations.json")
ARMS = ("zero_shot", "lisboa")
QUALITY_DIMENSIONS = ("factual_accuracy", "completeness", "relevance", "response_quality")
BENCHMARK_DIMENSIONS = ("factual_accuracy", "tool_usage", "completeness", "relevance", "response_quality")
WORKER_DOMAINS = ("weather", "transport", "researcher")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else round(float(value), digits)


def _describe(values: Sequence[float]) -> dict[str, Any]:
    """Mean, SD, median, P90, and max of a list of numbers."""
    values = sorted(float(value) for value in values)
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": _round(mean(values)),
        "sd": _round(stdev(values)) if len(values) > 1 else 0.0,
        "median": _round(median(values)),
        "p90": _round(values[int(0.9 * (len(values) - 1))]),
        "max": _round(values[-1]),
    }


def _slowest_decile_share(values: Sequence[float]) -> float | None:
    """Share of the total time spent in the slowest tenth of the responses."""
    ordered = sorted((float(value) for value in values), reverse=True)
    total = sum(ordered)
    if not ordered or total <= 0:
        return None
    return _round(sum(ordered[: max(1, math.ceil(len(ordered) / 10))]) / total)


def _query_types(annotations: dict | None) -> dict[str, str]:
    return {query_id: str(annotation.get("query_type")) for query_id, annotation in ((annotations or {}).get("queries") or {}).items()}


def _arm_block(record: dict, profile: str, arm: str) -> dict | None:
    return (((record.get("comparisons") or {}).get(profile) or {}).get("metrics") or {}).get(arm)


def _quality_score(block: dict | None) -> float | None:
    scores = (block or {}).get("scores") or {}
    value = scores.get("ablation_quality_score", scores.get("composite_score"))
    return None if value is None else float(value)


def _profiles(payload: dict) -> list[str]:
    summary = payload.get("summary") or {}
    order = summary.get("comparison_profile_order") or list((summary.get("comparison_profiles") or {}).keys())
    return [str(profile) for profile in order]


def _profile_model_id(payload: dict, profile: str) -> str:
    profiles = (payload.get("ablation_metadata") or {}).get("comparison_profiles") or {}
    return str(((profiles.get(profile) or {}).get("zero_shot_model_config") or {}).get("model_id") or profile)


def _judge_ids(payload: dict) -> list[str]:
    configs = (payload.get("ablation_metadata") or {}).get("judge_model_configs") or []
    return [str(config["model_id"]) for config in configs]


def qwk(a: Sequence[int], b: Sequence[int], low: int = 1, high: int = 5) -> float:
    """Quadratic-weighted Cohen's kappa for two raters on an ordinal scale."""
    a_array, b_array, size = np.asarray(a, int), np.asarray(b, int), high - low + 1
    observed = np.zeros((size, size))
    for x, y in zip(a_array, b_array):
        observed[x - low, y - low] += 1
    weights = np.array([[(i - j) ** 2 / (size - 1) ** 2 for j in range(size)] for i in range(size)])
    expected = np.outer(observed.sum(1), observed.sum(0)) / observed.sum()
    return float(1 - (weights * observed).sum() / (weights * expected).sum())


def icc2(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """ICC(2,1) for one judge and ICC(2,2) for the mean of two (two-way random, absolute agreement)."""
    matrix = np.column_stack([a, b]).astype(float)
    n, k = matrix.shape
    grand_mean = matrix.mean()
    ms_rows = k * ((matrix.mean(1) - grand_mean) ** 2).sum() / (n - 1)
    ms_cols = n * ((matrix.mean(0) - grand_mean) ** 2).sum() / (k - 1)
    ms_error = (
        (matrix - matrix.mean(1, keepdims=True) - matrix.mean(0, keepdims=True) + grand_mean) ** 2
    ).sum() / ((n - 1) * (k - 1))
    single = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n)
    average = (ms_rows - ms_error) / (ms_rows + (ms_cols - ms_error) / n)
    return float(single), float(average)


def _agreement_row(a: Sequence[int], b: Sequence[int]) -> dict[str, Any]:
    single, average = icc2(a, b)
    differences = np.abs(np.asarray(a) - np.asarray(b))
    return {
        "n": len(a),
        "qwk": _round(qwk(a, b)),
        "icc_2_1": _round(single),
        "icc_2_2": _round(average),
        "exact_agreement": _round(float(np.mean(differences == 0))),
        "within_one_point": _round(float(np.mean(differences <= 1))),
        "mean_absolute_difference": _round(float(np.mean(differences))),
    }


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _first_session_start(metadata: dict) -> str | None:
    """Start of the first session of a run that was resumed; the run's own start otherwise."""
    starts = [str(s.get("started_at")) for s in metadata.get("run_sessions") or [] if s.get("type") == "session" and s.get("started_at")]
    return min(starts) if starts else metadata.get("run_started_at")


def provenance_section(payload: dict) -> dict[str, Any]:
    """Protocol and provenance checks for the run."""
    metadata = payload.get("ablation_metadata") or {}
    provenance = metadata.get("provenance") or {}
    git = provenance.get("git") or {}
    sessions = metadata.get("run_sessions") or []
    session_commits = sorted({s.get("git_commit") for s in sessions if s.get("type") == "session" and s.get("git_commit")})
    records = payload.get("ablation_results", [])
    lisboa_blocks = [
        _arm_block(record, profile, "lisboa")
        for record in records
        for profile in (record.get("comparisons") or {})
    ]
    lisboa_blocks = [block for block in lisboa_blocks if block is not None]
    # Merged artefacts (eval/merge_ablation.py) hold runs with different protocols:
    # check fresh sessions only where the run asked for them.
    merged_runs = metadata.get("merged_runs") or []
    if merged_runs:
        fresh_runs = {str(run.get("source_run")) for run in merged_runs if run.get("fresh_session")}
        expected_fresh = [
            _arm_block(record, profile, "lisboa")
            for record in records
            if str(record.get("source_run")) in fresh_runs
            for profile in (record.get("comparisons") or {})
        ]
    else:
        fresh_requested = bool((metadata.get("run_options") or {}).get("fresh_session"))
        expected_fresh = lisboa_blocks if fresh_requested else []
    expected_fresh = [block for block in expected_fresh if block is not None]
    per_run = {}
    for record in records:
        run_name = str(record.get("source_run") or "single_run")
        entry = per_run.setdefault(run_name, {"queries": 0, "lisboa_responses_fresh": 0})
        entry["queries"] += 1
        entry["lisboa_responses_fresh"] += sum(
            1
            for profile in (record.get("comparisons") or {})
            if (_arm_block(record, profile, "lisboa") or {}).get("fresh_session") is True
        )
    errors = Counter()
    judge_errors = Counter()
    for record in records:
        for profile in record.get("comparisons") or {}:
            for arm in ARMS:
                block = _arm_block(record, profile, arm) or {}
                if block.get("error") is not None:
                    errors[f"{profile}::{arm}::{block.get('error_type') or 'error'}"] += 1
                for judge_run in block.get("judge_runs") or []:
                    if judge_run.get("error") and block.get("error") is None:
                        judge_errors[f"{profile}::{arm}::{judge_run.get('judge_model')}"] += 1
    # Every model call inside a LISBOA response must use the model of its profile.
    lisboa_calls = 0
    wrong_model_calls = Counter()
    for record in records:
        for profile in record.get("comparisons") or {}:
            expected_model = _profile_model_id(payload, profile).lower()
            block = _arm_block(record, profile, "lisboa") or {}
            for call in (block.get("response_usage") or {}).get("llm_usage_breakdown") or []:
                lisboa_calls += 1
                if str(call.get("model_id") or "").lower() != expected_model:
                    wrong_model_calls[f"{profile}::{call.get('model_id')}"] += 1
    return {
        "run_started_at": _first_session_start(metadata),
        "run_finished_at": metadata.get("run_finished_at"),
        "sessions": len([s for s in sessions if s.get("type") == "session"]) or 1,
        "commit": git.get("commit"),
        "branch": git.get("branch"),
        "has_uncommitted_changes": git.get("has_uncommitted_changes"),
        "session_commits": session_commits,
        "system_code_last_commit": git.get("system_code_last_commit"),
        "dataset": metadata.get("groundtruth_queries_path"),
        "dataset_fingerprint": metadata.get("groundtruth_queries_fingerprint"),
        "queries": metadata.get("groundtruth_queries_count"),
        "tool_registry": {"count": metadata.get("tool_registry_count"), "fingerprint": metadata.get("tool_registry_fingerprint")},
        "api_model_names": metadata.get("api_model_names"),
        "judges": _judge_ids(payload),
        "pricing_snapshot_date": metadata.get("pricing_snapshot_date"),
        "runtime_data_at_start": provenance.get("runtime_data"),
        "runtime_data_at_end": metadata.get("runtime_data_at_end"),
        "fresh_session": {
            "requested": (metadata.get("run_options") or {}).get("fresh_session"),
            "lisboa_responses": len(lisboa_blocks),
            "with_fresh_session": sum(1 for block in lisboa_blocks if block.get("fresh_session") is True),
            "with_execution_summary": sum(1 for block in lisboa_blocks if block.get("execution_summary")),
            "per_run": per_run,
        },
        "merged_runs": [
            {key: run.get(key) for key in ("source_run", "run_started_at", "commit", "fresh_session")}
            for run in merged_runs
        ],
        "response_errors": dict(sorted(errors.items())),
        "judge_errors": dict(sorted(judge_errors.items())),
        "lisboa_model_calls": {"total": lisboa_calls, "not_profile_model": dict(sorted(wrong_model_calls.items()))},
        # True or False when the run recorded the information; None (unknown) for older runs.
        "checks": {
            "single_commit": len(session_commits) <= 1 if session_commits else None,
            "clean_working_tree": None if git.get("has_uncommitted_changes") is None else not git["has_uncommitted_changes"],
            "fresh_session_where_requested": (
                None if not expected_fresh else all(block.get("fresh_session") is True for block in expected_fresh)
            ),
            "no_response_errors": not errors,
            "no_judge_errors": not judge_errors,
            "lisboa_calls_use_profile_model": None if not lisboa_calls else not wrong_model_calls,
        },
    }


def benchmark_provenance_section(benchmark: dict, ablation: dict | None = None) -> dict[str, Any]:
    """Protocol checks for the benchmark run, and whether it evaluated the ablation's system.

    Args:
        benchmark: Benchmark results payload.
        ablation: Optional ablation payload of the same revision.

    Returns:
        Run dates, sessions, commit, served model names, error counts, calls to a
        model other than the row's, and checks, including one system-code
        fingerprint and one corpus file for both runs.
    """
    metadata = benchmark.get("benchmark_metadata") or {}
    git = (metadata.get("provenance") or {}).get("git") or {}
    records = benchmark.get("benchmark_results", [])
    sessions = metadata.get("run_sessions") or []
    session_codes = sorted({s.get("system_code_sha256") for s in sessions if s.get("type") == "session" and s.get("system_code_sha256")})
    errors = Counter(str(r.get("response_model")) for r in records if r.get("error") is not None)
    judge_errors = Counter(
        str(r.get("response_model"))
        for r in records
        if r.get("error") is None and any(j.get("error") for j in r.get("judge_runs") or [])
    )
    mismatches = Counter(str(r.get("response_model")) for r in records if r.get("model_call_mismatches"))
    expected = len(metadata.get("groundtruth_ids") or []) * len(metadata.get("response_models") or [])
    output = {
        "run_started_at": _first_session_start(metadata),
        "run_finished_at": metadata.get("run_finished_at"),
        "sessions": len([s for s in sessions if s.get("type") == "session"]) or 1,
        "commit": git.get("commit"),
        "has_uncommitted_changes": git.get("has_uncommitted_changes"),
        "responses": len(records),
        "expected_responses": expected or None,
        "api_model_names": metadata.get("api_model_names"),
        "response_errors": dict(errors),
        "judge_errors": dict(judge_errors),
        "responses_with_calls_to_another_model": dict(mismatches),
        "checks": {
            "complete": None if not expected else len(records) == expected,
            "single_system_code": None if not session_codes else len(session_codes) == 1,
            "clean_working_tree": None if git.get("has_uncommitted_changes") is None else not git["has_uncommitted_changes"],
            "no_response_errors": not errors,
            "no_judge_errors": not judge_errors,
            "calls_use_row_model": None if "model_call_mismatches" not in (records[0] if records else {}) else not mismatches,
        },
    }
    if ablation is not None:
        ablation_metadata = ablation.get("ablation_metadata") or {}
        ablation_provenance = ablation_metadata.get("provenance") or {}
        ablation_code = (ablation_provenance.get("git") or {}).get("system_code_sha256")
        benchmark_code = git.get("system_code_sha256")
        ablation_dataset = ((ablation_provenance.get("input_files") or {}).get("dataset") or {}).get("sha256")
        benchmark_dataset = (((metadata.get("provenance") or {}).get("input_files") or {}).get("dataset") or {}).get("sha256")
        output["checks"]["same_system_code_as_ablation"] = (
            None if not (ablation_code and benchmark_code) else ablation_code == benchmark_code
        )
        output["checks"]["same_corpus_file_as_ablation"] = (
            None if not (ablation_dataset and benchmark_dataset) else ablation_dataset == benchmark_dataset
        )
    return output


def quality_section(
    payload: dict,
    annotations: dict | None,
    *,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Paired LISBOA versus zero-shot tests, per model, domain, and query type."""
    domain_tests = [
        row
        for row in ablation_arm_tests(payload, bootstrap_iterations=bootstrap_iterations, seed=seed)
        if row["dimension"] in ("ablation_quality_score", *QUALITY_DIMENSIONS)
    ]
    subset_tests = []
    query_types = (annotations or {}).get("queries") or {}
    records = payload.get("ablation_results", [])
    for profile in _profiles(payload):
        label = _profile_model_id(payload, profile).split("::")[-1]
        for query_type in ("itinerary", "cross_domain"):
            ids = {query_id for query_id, annotation in query_types.items() if annotation.get("query_type") == query_type}
            pairs = []
            for record in records:
                if record["id"] not in ids:
                    continue
                zero_shot = _quality_score(_arm_block(record, profile, "zero_shot"))
                lisboa = _quality_score(_arm_block(record, profile, "lisboa"))
                if zero_shot is not None and lisboa is not None:
                    pairs.append((zero_shot, lisboa))
            if pairs:
                subset_tests.append(
                    _comparison_result(
                        comparison_type="ablation_lisboa_vs_zero_shot_subset",
                        dimension="ablation_quality_score",
                        label_a="Zero-shot",
                        label_b="LISBOA",
                        pairs=pairs,
                        group={"profile": label, "domain": query_type},
                        bootstrap_iterations=bootstrap_iterations,
                        seed=seed,
                    )
                )
    # How many queries LISBOA wins, ties, or loses, per model and subset.
    types = _query_types(annotations)
    paired_outcomes = []
    for profile in _profiles(payload):
        label = _profile_model_id(payload, profile).split("::")[-1]
        subsets: dict[str, list[tuple[float, float]]] = {}
        for record in records:
            zero_shot = _quality_score(_arm_block(record, profile, "zero_shot"))
            lisboa = _quality_score(_arm_block(record, profile, "lisboa"))
            if zero_shot is None or lisboa is None:
                continue
            for subset in {"all", str(record.get("domain")), types.get(record["id"], "")} - {""}:
                subsets.setdefault(subset, []).append((zero_shot, lisboa))
        for subset, pairs in sorted(subsets.items()):
            paired_outcomes.append(
                {
                    "profile": label,
                    "subset": subset,
                    "n": len(pairs),
                    "lisboa_higher": sum(1 for zero_shot, lisboa in pairs if lisboa > zero_shot),
                    "ties": sum(1 for zero_shot, lisboa in pairs if lisboa == zero_shot),
                    "zero_shot_higher": sum(1 for zero_shot, lisboa in pairs if lisboa < zero_shot),
                }
            )
    return {"by_domain": domain_tests, "by_query_type": subset_tests, "paired_outcomes": paired_outcomes}


def operational_section(payload: dict, pricing: dict | None, annotations: dict | None = None) -> dict[str, Any]:
    """Latency and response cost per model and condition; judges excluded from cost."""
    models = (pricing or {}).get("models") or {}
    types = _query_types(annotations)
    rows = []
    worst_difference = 0.0
    recomputed = 0
    for profile in _profiles(payload):
        model_id = _profile_model_id(payload, profile)
        price = models.get(model_id.lower()) or models.get(model_id)
        for arm in ARMS:
            blocks = [_arm_block(record, profile, arm) for record in payload.get("ablation_results", [])]
            blocks = [block for block in blocks if block is not None]
            llm_blocks = [block for block in blocks if (block.get("response_usage") or {}).get("usage_available")]
            costs = [float((block.get("response_cost_usd") or {}).get("total_cost_usd") or 0.0) for block in llm_blocks]
            tokens_in = [int(((block.get("response_cost_usd") or {}).get("tokens") or {}).get("input_tokens", 0) or 0) for block in llm_blocks]
            tokens_out = [int(((block.get("response_cost_usd") or {}).get("tokens") or {}).get("output_tokens", 0) or 0) for block in llm_blocks]
            if price:
                for block in llm_blocks:
                    tokens = (block.get("response_cost_usd") or {}).get("tokens") or {}
                    cached = int(tokens.get("cached_input_tokens", 0) or 0)
                    fresh = int(tokens.get("input_tokens", 0) or 0) - cached
                    independent = (
                        fresh * price["input"]
                        + cached * price.get("input_cached", price["input"])
                        + int(tokens.get("output_tokens", 0) or 0) * price["output"]
                    ) / 1e6
                    worst_difference = max(
                        worst_difference,
                        abs(independent - float((block.get("response_cost_usd") or {}).get("total_cost_usd") or 0.0)),
                    )
                    recomputed += 1
            evaluation_cost = sum(float((block.get("evaluation_cost_usd") or {}).get("total_cost_usd") or 0.0) for block in blocks)
            row = {
                "profile": profile,
                "model": model_id,
                "arm": arm,
                "responses": len(blocks),
                "llm_responses": len(llm_blocks),
                "deterministic_responses": len(blocks) - len(llm_blocks),
                "latency_s_all": _describe([block.get("latency_s", block.get("latency", 0.0)) for block in blocks]),
                "latency_s_llm_only": _describe([block.get("latency_s", block.get("latency", 0.0)) for block in llm_blocks]),
                "slowest_decile_time_share": _slowest_decile_share([block.get("latency_s", block.get("latency", 0.0)) for block in blocks]),
                "cost_usd_per_llm_response": {
                    **_describe(costs),
                    "mean": _round(mean(costs), 5) if costs else None,
                    "sd": _round(stdev(costs), 5) if len(costs) > 1 else None,
                    "total": _round(sum(costs), 4),
                },
                "tokens_mean": {"input": _round(mean(tokens_in), 0) if tokens_in else None, "output": _round(mean(tokens_out), 0) if tokens_out else None},
                # Every query, with answers that called no model counted at zero:
                # what a condition costs per question asked.
                "cost_usd_per_query": _round(sum(costs) / len(blocks), 5) if blocks else None,
                "judge_cost_usd_total": _round(evaluation_cost, 4),
            }
            if arm == "lisboa":
                by_domain = {}
                for domain in sorted({record.get("domain") for record in payload.get("ablation_results", [])}):
                    domain_blocks = [
                        _arm_block(record, profile, arm)
                        for record in payload.get("ablation_results", [])
                        if record.get("domain") == domain and _arm_block(record, profile, arm) is not None
                    ]
                    by_domain[domain] = _describe([block.get("latency_s", 0.0) for block in domain_blocks])
                row["latency_s_by_domain"] = by_domain
                by_type: dict[str, list[float]] = {}
                for record in payload.get("ablation_results", []):
                    block = _arm_block(record, profile, arm)
                    if block is not None and types.get(record["id"]):
                        by_type.setdefault(types[record["id"]], []).append(block.get("latency_s", 0.0))
                row["latency_s_by_query_type"] = {query_type: _describe(values) for query_type, values in sorted(by_type.items())}
            rows.append(row)
    total_response = sum(row["cost_usd_per_llm_response"]["total"] or 0.0 for row in rows)
    total_judges = sum(row["judge_cost_usd_total"] or 0.0 for row in rows)
    return {
        "rows": rows,
        "run_cost_usd": {"responses": _round(total_response, 2), "judges": _round(total_judges, 2), "total": _round(total_response + total_judges, 2)},
        "cost_recomputation": {"responses_recomputed": recomputed, "max_abs_difference_usd": worst_difference},
        "note": "Cost is list price x tokens for the model calls of each response; judge calls are an evaluation cost and are reported apart.",
    }


def benchmark_operational_section(benchmark: dict) -> dict[str, Any]:
    """Latency and response cost of the worker benchmark, per response model.

    The benchmark runs each worker alone with each candidate model, so these
    figures compare the models inside LISBOA's workers; judge calls are an
    evaluation cost and are reported apart.

    Args:
        benchmark: Benchmark results payload (``benchmark_results`` records).

    Returns:
        Rows per response model, with latency over all responses and over the
        responses that called the model, cost per model-calling response and
        per query, token means, the share of deterministic answers, and the
        median latency per domain.
    """
    records = benchmark.get("benchmark_results", []) if benchmark else []
    rows = []
    for model in sorted({str(record.get("response_model") or "") for record in records if record.get("response_model")}):
        model_records = [record for record in records if record.get("response_model") == model]
        llm_records = [record for record in model_records if (record.get("response_usage") or {}).get("usage_available")]
        costs = [float((record.get("response_cost_usd") or {}).get("total_cost_usd") or 0.0) for record in llm_records]
        tokens = [(record.get("response_cost_usd") or {}).get("tokens") or {} for record in llm_records]
        rows.append(
            {
                "model": model,
                "responses": len(model_records),
                "llm_responses": len(llm_records),
                "deterministic_responses": len(model_records) - len(llm_records),
                "errors": sum(1 for record in model_records if record.get("error")),
                "latency_s_all": _describe([record.get("latency_s", 0.0) for record in model_records]),
                "latency_s_llm_only": _describe([record.get("latency_s", 0.0) for record in llm_records]),
                "cost_usd_per_llm_response": {
                    "mean": _round(mean(costs), 5) if costs else None,
                    "sd": _round(stdev(costs), 5) if len(costs) > 1 else None,
                    "total": _round(sum(costs), 4),
                },
                "cost_usd_per_query": _round(sum(costs) / len(model_records), 5) if model_records else None,
                "tokens_mean": {
                    "input": _round(mean(int(t.get("input_tokens", 0) or 0) for t in tokens), 0) if tokens else None,
                    "output": _round(mean(int(t.get("output_tokens", 0) or 0) for t in tokens), 0) if tokens else None,
                },
                "latency_s_median_by_domain": {
                    domain: _describe([record.get("latency_s", 0.0) for record in model_records if record.get("domain") == domain]).get("median")
                    for domain in sorted({str(record.get("domain")) for record in model_records})
                },
                "judge_cost_usd_total": _round(
                    sum(float((record.get("evaluation_cost_usd") or {}).get("total_cost_usd") or 0.0) for record in model_records), 4
                ),
            }
        )
    return {
        "rows": rows,
        "run_cost_usd": {
            "responses": _round(sum(row["cost_usd_per_llm_response"]["total"] or 0.0 for row in rows), 2),
            "judges": _round(sum(row["judge_cost_usd_total"] or 0.0 for row in rows), 2),
        },
        "note": "Worker-level latency (one worker with each model); response cost is list price x tokens; judges excluded.",
    }


def _selected_agents(block: dict) -> tuple[list[str], str]:
    """Agents chosen for the query, from the execution summary or, for older runs, from usage."""
    summary = block.get("execution_summary")
    if isinstance(summary, dict) and summary.get("selected_agents") is not None:
        return [str(agent) for agent in summary["selected_agents"]], "execution_summary"
    workers = [agent for agent in block.get("agents_used") or [] if agent not in ("supervisor", "qa")]
    return workers, "agents_used"


def end_to_end_section(payload: dict, annotations: dict | None) -> dict[str, Any]:
    """Routing, coordination, planner use, and QA paths in the LISBOA condition."""
    annotated = (annotations or {}).get("queries") or {}
    output = {}
    for profile in _profiles(payload):
        routing = {
            "n": 0,
            "expected_selected": 0,
            "boundary_declined_without_worker": 0,
            "exact": 0,
            "misses": [],
            "with_extra_agents": [],
            "boundary_declined_ids": [],
        }
        coverage_rows = []
        execution_types = Counter()
        qa_paths = Counter()
        qa_scores: dict[str, list[float]] = {}
        # "final-repair" also marks edits of the deterministic final guard; these made no QA model call.
        final_repair_without_qa_call = 0
        sources = Counter()
        for record in payload.get("ablation_results", []):
            block = _arm_block(record, profile, "lisboa")
            if block is None:
                continue
            selected, source = _selected_agents(block)
            sources[source] += 1
            summary = block.get("execution_summary") or {}
            execution_types[summary.get("execution_type") or "not_recorded"] += 1
            qa_path = summary.get("qa_path") or "not_recorded"
            qa_paths[qa_path] += 1
            qa_calls = int(((block.get("agent_usage") or {}).get("qa") or {}).get("call_count", 0) or 0)
            if "final-repair" in qa_path and qa_calls == 0:
                final_repair_without_qa_call += 1
            score = _quality_score(block)
            if score is not None:
                qa_scores.setdefault(qa_path, []).append(score)

            annotation = annotated.get(record["id"])
            expected = list((annotation or {}).get("expected_agents") or ([record["domain"]] if record["domain"] in WORKER_DOMAINS else []))
            query_type = (annotation or {}).get("query_type") or ("single_domain" if record["domain"] in WORKER_DOMAINS else record["domain"])
            if query_type == "single_domain":
                routing["n"] += 1
                workers_selected = [agent for agent in selected if agent in WORKER_DOMAINS or agent == "planner"]
                # A boundary query (no tool expected: unsupported area, field, action, or entity)
                # answered directly, before any worker, is the intended behaviour.
                boundary = not record.get("expected_tools")
                if set(expected) <= set(selected):
                    routing["expected_selected"] += 1
                elif boundary and not workers_selected:
                    routing["boundary_declined_without_worker"] += 1
                    routing["boundary_declined_ids"].append(record["id"])
                else:
                    routing["misses"].append(record["id"])
                if set(selected) == set(expected):
                    routing["exact"] += 1
                elif set(expected) <= set(selected):
                    routing["with_extra_agents"].append(record["id"])
            elif expected:
                covered = [agent for agent in expected if agent in selected]
                coverage_rows.append(
                    {
                        "id": record["id"],
                        "query_type": query_type,
                        "expected_agents": expected,
                        "selected_agents": selected,
                        "coverage": _round(len(covered) / len(expected)),
                        "missing_agents": [agent for agent in expected if agent not in selected],
                        "planner_selected": "planner" in selected,
                        "execution_type": summary.get("execution_type"),
                        "quality_score": score,
                    }
                )
        itinerary_rows = [row for row in coverage_rows if row["query_type"] == "itinerary"]
        cross_rows = [row for row in coverage_rows if row["query_type"] != "itinerary"]
        output[profile] = {
            "agent_source": dict(sources),
            "routing_single_domain": {
                **routing,
                "correct": routing["expected_selected"] + routing["boundary_declined_without_worker"],
                "expected_selected_rate": _round(routing["expected_selected"] / routing["n"]) if routing["n"] else None,
                "exact_rate": _round(routing["exact"] / routing["n"]) if routing["n"] else None,
            },
            "coverage_multi_agent": {
                "n": len(coverage_rows),
                "mean_coverage": _round(mean(row["coverage"] for row in coverage_rows)) if coverage_rows else None,
                "full_coverage": sum(1 for row in coverage_rows if row["coverage"] == 1.0),
                "itinerary_n": len(itinerary_rows),
                "itinerary_planner_selected": sum(1 for row in itinerary_rows if row["planner_selected"]),
                "cross_domain_n": len(cross_rows),
                "cross_domain_planner_selected": sum(1 for row in cross_rows if row["planner_selected"]),
                "rows": coverage_rows,
            },
            "execution_types": dict(execution_types.most_common()),
            "qa_paths": {
                path: {"n": count, "mean_quality_score": _round(mean(qa_scores[path])) if qa_scores.get(path) else None}
                for path, count in qa_paths.most_common()
            },
            "qa_intervened": (
                None
                if set(qa_paths) == {"not_recorded"}
                else sum(count for path, count in qa_paths.items() if "retry" in path or "final-repair" in path)
            ),
            "qa_worker_retry": None if set(qa_paths) == {"not_recorded"} else sum(count for path, count in qa_paths.items() if "retry" in path),
            "qa_final_repair": None if set(qa_paths) == {"not_recorded"} else sum(count for path, count in qa_paths.items() if "final-repair" in path),
            "qa_final_repair_deterministic_only": None if set(qa_paths) == {"not_recorded"} else final_repair_without_qa_call,
        }
    return output


def judges_section(payload: dict, benchmark: dict | None) -> dict[str, Any]:
    """Inter-judge reliability and the gain measured by each judge alone."""
    judges = _judge_ids(payload)
    output: dict[str, Any] = {"judges": judges, "ablation_agreement": {}, "gain_by_judge": [], "benchmark_agreement": {}}
    # Pooled over the five judged dimensions, as in the submitted version of Section 5.3.
    if len(judges) != 2:
        return output
    first, second = judges
    records = payload.get("ablation_results", [])
    for dimension in QUALITY_DIMENSIONS:
        a, b = [], []
        for record in records:
            for profile in record.get("comparisons") or {}:
                for arm in ARMS:
                    by_judge = (_arm_block(record, profile, arm) or {}).get("scores_by_judge") or {}
                    if by_judge.get(first, {}).get(dimension) is not None and by_judge.get(second, {}).get(dimension) is not None:
                        a.append(int(by_judge[first][dimension]))
                        b.append(int(by_judge[second][dimension]))
        if a:
            output["ablation_agreement"][dimension] = _agreement_row(a, b)
    pooled_a, pooled_b = [], []
    for record in records:
        for profile in record.get("comparisons") or {}:
            for arm in ARMS:
                by_judge = (_arm_block(record, profile, arm) or {}).get("scores_by_judge") or {}
                for dimension in BENCHMARK_DIMENSIONS:
                    if by_judge.get(first, {}).get(dimension) is not None and by_judge.get(second, {}).get(dimension) is not None:
                        pooled_a.append(int(by_judge[first][dimension]))
                        pooled_b.append(int(by_judge[second][dimension]))
    if pooled_a:
        output["ablation_pooled"] = _agreement_row(pooled_a, pooled_b)

    for profile in _profiles(payload):
        generator = _profile_model_id(payload, profile)
        for label, subset in (("both judges", (first, second)), (first, (first,)), (second, (second,))):
            differences, zero_shot_means, lisboa_means = [], [], []
            for record in records:
                arms = {arm: (_arm_block(record, profile, arm) or {}).get("scores_by_judge") or {} for arm in ARMS}
                try:
                    values = {
                        arm: mean(mean(float(arms[arm][judge][d]) for d in QUALITY_DIMENSIONS) for judge in subset)
                        for arm in ARMS
                    }
                except (KeyError, TypeError):
                    continue
                zero_shot_means.append(values["zero_shot"])
                lisboa_means.append(values["lisboa"])
                differences.append(values["lisboa"] - values["zero_shot"])
            if not differences:
                continue
            family = "" if len(subset) == 2 else ("same family" if subset[0].lower() == generator.lower() else "other family")
            output["gain_by_judge"].append(
                {
                    "profile": profile,
                    "generator": generator,
                    "judges": label,
                    "family": family,
                    "n": len(differences),
                    "zero_shot_mean": _round(mean(zero_shot_means)),
                    "lisboa_mean": _round(mean(lisboa_means)),
                    "gain": _round(mean(differences)),
                    "wilcoxon_p": _wilcoxon_signed_rank(differences)["p_value"],
                }
            )

    if benchmark:
        results = benchmark.get("benchmark_results", [])
        for dimension in BENCHMARK_DIMENSIONS:
            pairs = [
                (int(r["scores_by_judge"][first][dimension]), int(r["scores_by_judge"][second][dimension]))
                for r in results
                if (r.get("scores_by_judge") or {}).get(first, {}).get(dimension) is not None
                and (r.get("scores_by_judge") or {}).get(second, {}).get(dimension) is not None
            ]
            if pairs:
                output["benchmark_agreement"][dimension] = _agreement_row([p[0] for p in pairs], [p[1] for p in pairs])
        pooled = [
            (int(r["scores_by_judge"][first][dimension]), int(r["scores_by_judge"][second][dimension]))
            for r in results
            for dimension in BENCHMARK_DIMENSIONS
            if (r.get("scores_by_judge") or {}).get(first, {}).get(dimension) is not None
            and (r.get("scores_by_judge") or {}).get(second, {}).get(dimension) is not None
        ]
        if pooled:
            output["benchmark_pooled"] = _agreement_row([p[0] for p in pooled], [p[1] for p in pooled])
        cells: dict[tuple[str, str], list[float]] = {}
        for r in results:
            for judge in (first, second):
                scores = (r.get("scores_by_judge") or {}).get(judge) or {}
                if all(scores.get(d) is not None for d in BENCHMARK_DIMENSIONS):
                    cells.setdefault((judge, str(r.get("response_model"))), []).append(mean(float(scores[d]) for d in BENCHMARK_DIMENSIONS))
        means = {key: mean(values) for key, values in cells.items()}
        if all((judge, model) in means for judge in (first, second) for model in (first, second)):
            output["benchmark_own_family"] = {
                "cell_means": {f"{judge} judges {model}": _round(value) for (judge, model), value in means.items()},
                # How much higher each judge rates the first family's answers than the second's.
                "first_minus_second_model_by_judge": {
                    judge: _round(means[(judge, first)] - means[(judge, second)]) for judge in (first, second)
                },
                "own_family_interaction": _round(
                    (means[(first, first)] - means[(first, second)]) - (means[(second, first)] - means[(second, second)])
                ),
            }
    return output


def constraints_section(constraints: dict | None) -> dict[str, Any] | None:
    """Itinerary constraint checklist, as summarized by eval/constraint_judge.py."""
    if not constraints:
        return None
    metadata = constraints.get("constraint_metadata") or {}
    return {
        "file_judges": metadata.get("judges"),
        "errors": metadata.get("errors"),
        "cost_usd": (metadata.get("cost_usd") or {}).get("total_cost_usd"),
        "cells": (constraints.get("summary") or {}).get("cells"),
        "inter_judge_agreement": (constraints.get("summary") or {}).get("inter_judge_agreement"),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    """Render the paper-facing tables as Markdown."""
    lines = ["# Paper evaluation analysis", "", f"Generated {report['generated_at']} from `{report['inputs']['ablation']['path']}`.", ""]
    provenance = report["provenance"]
    lines += [
        "## Provenance",
        "",
        f"- Commit `{_fmt(provenance['commit'])}` on `{_fmt(provenance['branch'])}`; uncommitted changes: {_fmt(provenance['has_uncommitted_changes'])}",
        f"- Last system-code commit: {_fmt((provenance.get('system_code_last_commit') or {}).get('date'))} ({_fmt((provenance.get('system_code_last_commit') or {}).get('subject'))})",
        f"- Run: {_fmt(provenance['run_started_at'])} to {_fmt(provenance['run_finished_at'])}; sessions: {provenance['sessions']}",
        f"- Queries: {_fmt(provenance['queries'])}; tools: {_fmt(provenance['tool_registry']['count'])}; API model names: {_fmt(provenance['api_model_names'])}",
        f"- Fresh session: {provenance['fresh_session']['with_fresh_session']} of {provenance['fresh_session']['lisboa_responses']} LISBOA responses; execution summaries: {provenance['fresh_session']['with_execution_summary']}",
        "- Checks: " + ", ".join(
            f"{name} {'unknown' if ok is None else ('OK' if ok else 'FAILED')}" for name, ok in provenance["checks"].items()
        ),
        f"- Response errors: {_fmt(provenance['response_errors'] or 'none')}; judge errors: {_fmt(provenance['judge_errors'] or 'none')}",
    ]
    benchmark_provenance = report.get("benchmark_provenance")
    if benchmark_provenance:
        lines += [
            f"- Benchmark: {benchmark_provenance['responses']} of {_fmt(benchmark_provenance['expected_responses'])} responses; "
            f"run {_fmt(benchmark_provenance['run_started_at'])} to {_fmt(benchmark_provenance['run_finished_at'])}; "
            f"API model names: {_fmt(benchmark_provenance['api_model_names'])}",
            "- Benchmark checks: " + ", ".join(
                f"{name} {'unknown' if ok is None else ('OK' if ok else 'FAILED')}" for name, ok in benchmark_provenance["checks"].items()
            ),
        ]
    lines += [
        "",
        "## Quality (ablation quality score; LISBOA minus zero-shot)",
        "",
        "| Model | Subset | n | Zero-shot | LISBOA | Gain | 95% CI | p | r_rb |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    quality_rows = [row for row in report["quality"]["by_domain"] if row["dimension"] == "ablation_quality_score"]
    quality_rows += report["quality"]["by_query_type"]
    for row in quality_rows:
        lines.append(
            f"| {row['profile']} | {row['domain']} | {row['n_pairs']} | {_fmt(row['mean_a'])} | {_fmt(row['mean_b'])} | "
            f"{_fmt(row['mean_diff_b_minus_a'])} | [{_fmt(row['bootstrap_ci_low'])}, {_fmt(row['bootstrap_ci_high'])}] | "
            f"{row['wilcoxon_p_value']:.2e} | {_fmt(row['rank_biserial_correlation'])} |"
        )
    lines += ["", "LISBOA ahead / tied / behind, per query:", ""]
    for row in report["quality"].get("paired_outcomes", []):
        lines.append(f"- {row['profile']}, {row['subset']}: {row['lisboa_higher']} / {row['ties']} / {row['zero_shot_higher']} of {row['n']}")
    lines += ["", "## Latency and cost per response", "", "| Model | Condition | Latency mean ± SD (s) | Median | P90 | LLM responses | Cost mean ± SD (USD) | Tokens in / out |", "|---|---|---|---:|---:|---:|---|---|"]
    for row in report["operational"]["rows"]:
        latency = row["latency_s_all"]
        cost = row["cost_usd_per_llm_response"]
        lines.append(
            f"| {row['model']} | {row['arm']} | {_fmt(latency.get('mean'), 2)} ± {_fmt(latency.get('sd'), 2)} | {_fmt(latency.get('median'), 2)} | "
            f"{_fmt(latency.get('p90'), 2)} | {row['llm_responses']}/{row['responses']} | {_fmt(cost.get('mean'), 4)} ± {_fmt(cost.get('sd'), 4)} | "
            f"{_fmt(row['tokens_mean']['input'], 0)} / {_fmt(row['tokens_mean']['output'], 0)} |"
        )
    run_cost = report["operational"]["run_cost_usd"]
    for row in report["operational"]["rows"]:
        extra = f"slowest tenth holds {_fmt(row.get('slowest_decile_time_share'))} of the time"
        if row.get("latency_s_by_domain"):
            extra += "; mean by domain " + ", ".join(f"{d} {_fmt(v.get('mean'), 1)}" for d, v in row["latency_s_by_domain"].items())
        if row.get("latency_s_by_query_type"):
            extra += "; mean by query type " + ", ".join(f"{t} {_fmt(v.get('mean'), 1)}" for t, v in row["latency_s_by_query_type"].items())
        lines.append(f"- {row['model']} {row['arm']}: {extra}")
    lines += ["", f"Run cost: responses USD {_fmt(run_cost['responses'], 2)}, judges USD {_fmt(run_cost['judges'], 2)}, total USD {_fmt(run_cost['total'], 2)}. "
              f"Cost recomputed from tokens for {report['operational']['cost_recomputation']['responses_recomputed']} responses; largest difference USD {report['operational']['cost_recomputation']['max_abs_difference_usd']:.2e}.", ""]
    lines += ["Zero-shot vs LISBOA per query (answers without a model call count as zero cost):", "", "| Model | Zero-shot latency median (s) | LISBOA latency median (s) | Zero-shot cost per query (USD) | LISBOA cost per query (USD) | Cost ratio |", "|---|---:|---:|---:|---:|---:|"]
    by_model: dict[str, dict[str, Any]] = {}
    for row in report["operational"]["rows"]:
        by_model.setdefault(row["model"], {})[row["arm"]] = row
    for model, arms in by_model.items():
        zero, lisboa = arms.get("zero_shot"), arms.get("lisboa")
        if not (zero and lisboa):
            continue
        ratio = (lisboa["cost_usd_per_query"] / zero["cost_usd_per_query"]) if zero.get("cost_usd_per_query") else None
        lines.append(
            f"| {model} | {_fmt(zero['latency_s_all'].get('median'), 2)} | {_fmt(lisboa['latency_s_all'].get('median'), 2)} | "
            f"{_fmt(zero.get('cost_usd_per_query'), 5)} | {_fmt(lisboa.get('cost_usd_per_query'), 5)} | {_fmt(ratio, 2)} |"
        )
    benchmark_ops = report.get("benchmark_operational")
    if benchmark_ops and benchmark_ops.get("rows"):
        lines += ["", "## Worker benchmark: latency and cost per response model", "", "| Model | Latency median (s) | P90 (s) | Model-calling responses | Cost per model-calling response (USD) | Cost per query (USD) | Tokens in / out | Errors |", "|---|---:|---:|---:|---:|---:|---|---:|"]
        for row in benchmark_ops["rows"]:
            latency = row["latency_s_all"]
            lines.append(
                f"| {row['model']} | {_fmt(latency.get('median'), 2)} | {_fmt(latency.get('p90'), 2)} | {row['llm_responses']}/{row['responses']} | "
                f"{_fmt(row['cost_usd_per_llm_response'].get('mean'), 5)} | {_fmt(row['cost_usd_per_query'], 5)} | "
                f"{_fmt(row['tokens_mean']['input'], 0)} / {_fmt(row['tokens_mean']['output'], 0)} | {row['errors']} |"
            )
        lines += ["", "Median latency by domain (s): " + "; ".join(
            f"{row['model']}: " + ", ".join(f"{domain} {_fmt(value, 2)}" for domain, value in row["latency_s_median_by_domain"].items())
            for row in benchmark_ops["rows"]
        ), f"\nBenchmark run cost: responses USD {_fmt(benchmark_ops['run_cost_usd']['responses'], 2)}, judges USD {_fmt(benchmark_ops['run_cost_usd']['judges'], 2)}.", ""]
    lines += ["## End to end (LISBOA condition)", "", "| Model | Routing: expected worker selected | Exact routing | Multi-agent coverage | Itinerary: planner | Cross-domain: planner | QA intervened | Agent source |", "|---|---|---|---|---|---|---:|---|"]
    for profile, section in report["end_to_end"].items():
        routing = section["routing_single_domain"]
        coverage = section["coverage_multi_agent"]
        lines.append(
            f"| {profile} | {routing['expected_selected']}/{routing['n']} | {routing['exact']}/{routing['n']} | "
            f"{_fmt(coverage['mean_coverage'])} ({coverage['full_coverage']}/{coverage['n']} full) | "
            f"{coverage['itinerary_planner_selected']}/{coverage['itinerary_n']} | {coverage['cross_domain_planner_selected']}/{coverage['cross_domain_n']} | "
            f"{_fmt(section['qa_intervened'])} | {_fmt(section['agent_source'])} |"
        )
    for profile, section in report["end_to_end"].items():
        lines.append(f"\nQA paths, {profile}: " + "; ".join(f"{path} n={value['n']} (QS {_fmt(value['mean_quality_score'])})" for path, value in section["qa_paths"].items()))
        routing = section["routing_single_domain"]
        lines.append(
            f"Routing, {profile}: correct {routing['correct']}/{routing['n']} (expected worker {routing['expected_selected']}, "
            f"boundary queries declined before any worker {routing['boundary_declined_without_worker']} {routing['boundary_declined_ids']}); "
            f"misses {routing['misses'] or 'none'}; QA worker retry {_fmt(section.get('qa_worker_retry'))}, final repair {_fmt(section.get('qa_final_repair'))} ({_fmt(section.get('qa_final_repair_deterministic_only'))} by the deterministic guard alone)"
        )
    lines += ["", "## Judges", "", "| Set | Dimension | n | QWK | ICC(2,1) | ICC(2,2) | Exact | Within one | Mean abs. diff. |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for set_name, key in (("ablation", "ablation_agreement"), ("benchmark", "benchmark_agreement")):
        for dimension, row in report["judges"].get(key, {}).items():
            lines.append(
                f"| {set_name} | {dimension} | {row['n']} | {_fmt(row['qwk'])} | {_fmt(row['icc_2_1'])} | {_fmt(row['icc_2_2'])} | "
                f"{_fmt(row['exact_agreement'])} | {_fmt(row.get('within_one_point'))} | {_fmt(row.get('mean_absolute_difference'))} |"
            )
    lines += ["", "| Generator | Judges | Family | Zero-shot | LISBOA | Gain | p |", "|---|---|---|---:|---:|---:|---:|"]
    for row in report["judges"].get("gain_by_judge", []):
        lines.append(f"| {row['generator']} | {row['judges']} | {row['family'] or 'both'} | {_fmt(row['zero_shot_mean'])} | {_fmt(row['lisboa_mean'])} | {_fmt(row['gain'])} | {row['wilcoxon_p']:.1e} |")
    for key, label in (("benchmark_pooled", "Benchmark"), ("ablation_pooled", "Ablation")):
        pooled = report["judges"].get(key)
        if pooled:
            lines.append(
                f"\n{label}, all five dimensions pooled: exact {_fmt(pooled['exact_agreement'])}, within one {_fmt(pooled['within_one_point'])}, "
                f"mean abs. diff. {_fmt(pooled['mean_absolute_difference'])} (n={pooled['n']})"
            )
    if report["judges"].get("benchmark_own_family"):
        own = report["judges"]["benchmark_own_family"]
        lines.append(
            f"\nBenchmark own-family interaction: {_fmt(own['own_family_interaction'])}; first minus second model, by judge: "
            f"{_fmt(own.get('first_minus_second_model_by_judge'))}"
        )
    constraints = report.get("constraints")
    if constraints:
        lines += ["", "## Itinerary constraints (judge consensus)", "", "| Cell | Queries | Constraints | Met | Not met | Cannot assess | Disagreements | Met / all constraints | Met / unanimous assessable |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for cell, value in (constraints.get("cells") or {}).items():
            consensus = value["consensus"]
            all_constraints = consensus.get("all_constraints", consensus["total"] + consensus["judge_disagreements"])
            met_all = consensus.get("met_rate_all_constraints")
            if met_all is None and all_constraints:
                met_all = round(consensus["met"] / all_constraints, 4)
            lines.append(f"| {cell} | {value['queries']} | {all_constraints} | {consensus['met']} | {consensus['not_met']} | {consensus['cannot_assess']} | {consensus['judge_disagreements']} | {_fmt(met_all)} | {_fmt(consensus['met_rate_assessable'])} |")
        lines += ["", "Per judge:", "", "| Cell | Judge | Constraints | Met | Not met | Cannot assess | Met / all | Met / assessable |", "|---|---|---:|---:|---:|---:|---:|---:|"]
        for cell, value in (constraints.get("cells") or {}).items():
            for judge, rates in (value.get("per_judge") or {}).items():
                lines.append(f"| {cell} | {judge} | {rates['total']} | {rates['met']} | {rates['not_met']} | {rates['cannot_assess']} | {_fmt(rates['met_rate_all'])} | {_fmt(rates['met_rate_assessable'])} |")
        lines.append(f"\nInter-judge agreement on verdicts: {_fmt(constraints.get('inter_judge_agreement'))}")
    return "\n".join(lines) + "\n"


def run_paper_eval_analysis(
    ablation_path: str | Path,
    *,
    annotations_path: str | Path | None = DEFAULT_ANNOTATIONS_PATH,
    benchmark_path: str | Path | None = None,
    constraints_path: str | Path | None = None,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_RANDOM_SEED,
    output_prefix: str = "paper_eval_analysis",
) -> tuple[Path, Path]:
    """Build the JSON and Markdown analysis of one ablation run.

    Args:
        ablation_path: Ablation results JSON.
        annotations_path: End-to-end annotations; skipped when missing.
        benchmark_path: Optional benchmark results JSON for judge reliability.
        constraints_path: Optional output of ``eval.constraint_judge``.
        bootstrap_iterations: Bootstrap resamples for the confidence intervals.
        seed: Random seed shared with ``eval.statistical_analysis``.
        output_prefix: File prefix inside ``eval/results/statistics/``.

    Returns:
        tuple[Path, Path]: The JSON report and its Markdown rendering.
    """
    payload = _load_json(ablation_path)
    annotations = _load_json(annotations_path) if annotations_path and Path(annotations_path).is_file() else None
    benchmark = _load_json(benchmark_path) if benchmark_path else None
    constraints = _load_json(constraints_path) if constraints_path else None
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "ablation": describe_file(ablation_path),
            "annotations": describe_file(annotations_path) if annotations else None,
            "benchmark": describe_file(benchmark_path) if benchmark_path else None,
            "constraints": describe_file(constraints_path) if constraints_path else None,
            "bootstrap_iterations": bootstrap_iterations,
            "seed": seed,
        },
        "provenance": provenance_section(payload),
        "quality": quality_section(payload, annotations, bootstrap_iterations=bootstrap_iterations, seed=seed),
        "operational": operational_section(payload, load_pricing_catalog(), annotations),
        "end_to_end": end_to_end_section(payload, annotations),
        "judges": judges_section(payload, benchmark),
        "constraints": constraints_section(constraints),
    }
    if benchmark:
        report["benchmark_provenance"] = benchmark_provenance_section(benchmark, payload)
        report["benchmark_model_tests"] = benchmark_model_tests(benchmark, bootstrap_iterations=bootstrap_iterations, seed=seed)
        report["benchmark_operational"] = benchmark_operational_section(benchmark)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = build_results_output_path("statistics", output_prefix, timestamp)
    markdown_path = json_path.with_suffix(".md")
    write_json_artifact(report, json_path)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Paper evaluation analysis saved to {json_path}")
    print(f"Markdown report saved to {markdown_path}")
    return json_path, markdown_path


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Compute every number the revised paper reports from one ablation run.")
    parser.add_argument("--ablation", required=True, help="Ablation results JSON.")
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS_PATH), help="End-to-end annotations.")
    parser.add_argument("--benchmark", default=None, help="Optional benchmark results JSON (judge agreement, model tests).")
    parser.add_argument("--constraints", default=None, help="Optional constraint checklist JSON from eval/constraint_judge.py.")
    parser.add_argument("--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--output-prefix", default="paper_eval_analysis")
    args = parser.parse_args()
    run_paper_eval_analysis(
        args.ablation,
        annotations_path=args.annotations,
        benchmark_path=args.benchmark,
        constraints_path=args.constraints,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    main()
