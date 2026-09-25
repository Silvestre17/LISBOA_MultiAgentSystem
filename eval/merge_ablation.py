# ==========================================================================
# Master Thesis - Ablation Results Merge
#   - André Filipe Gomes Silvestre, 20240502
#
#   Joins an earlier ablation artefact with a later run on additional queries
#   (for example, the itinerary requests added in the RINENG revision) into
#   one artefact with recomputed summaries, so the analysis notebook and
#   eval/statistical_analysis.py keep reading a single file.
#
#   The merge refuses overlapping query ids and runs with different model
#   profiles or judges. It also refuses runs with a different session
#   protocol (`--fresh-session`) or different system code, because those
#   runs evaluate different conditions and cannot form one controlled result;
#   `--allow-mixed-protocol` merges them anyway for archival use and marks the
#   artefact as mixed. Every merged record keeps the run it came from
#   (`source_run`), and the metadata lists each merged run with its dates,
#   commit, and protocol.
#
# Usage:
#   > python -m eval.merge_ablation --base eval/results/ablation/ablation_final_20260515_154508.json --add eval/results/ablation/ablation_new_queries_<timestamp>.json
#       Write eval/results/ablation/ablation_final_<timestamp>.json, which the notebook picks up as the latest artefact.
# ==========================================================================

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from eval.run_ablation import _build_profile_summary
from eval.runtime_utils import build_results_output_path, describe_file, write_json_artifact


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _profile_models(payload: dict) -> dict[str, str | None]:
    """Model of each comparison profile, as recorded by the run."""
    profiles = (payload.get("ablation_metadata") or {}).get("comparison_profiles") or {}
    return {
        str(key): ((meta or {}).get("zero_shot_model_config") or {}).get("model_id")
        for key, meta in profiles.items()
    }


def _judge_models(payload: dict) -> list[str]:
    configs = (payload.get("ablation_metadata") or {}).get("judge_model_configs") or []
    return [str(config.get("model_id")) for config in configs]


def _describe_run(path: str | Path, payload: dict) -> dict[str, Any]:
    """Summarize one merged run for the metadata."""
    metadata = payload.get("ablation_metadata") or {}
    git = (metadata.get("provenance") or {}).get("git") or {}
    return {
        "source_run": Path(path).stem,
        "file": describe_file(path),
        "run_started_at": metadata.get("run_started_at"),
        "run_finished_at": metadata.get("run_finished_at"),
        "total_runtime_s": metadata.get("total_runtime_s"),
        "queries": [record.get("id") for record in payload.get("ablation_results", [])],
        "groundtruth_queries_path": metadata.get("groundtruth_queries_path"),
        "groundtruth_queries_fingerprint": metadata.get("groundtruth_queries_fingerprint"),
        "commit": git.get("commit"),
        "system_code_last_commit": git.get("system_code_last_commit"),
        "system_code_sha256": git.get("system_code_sha256"),
        "fresh_session": (metadata.get("run_options") or {}).get("fresh_session", False),
        "api_model_names": metadata.get("api_model_names"),
    }


def merge_ablation_payloads(
    base_path: str | Path,
    added_paths: Sequence[str | Path],
    *,
    allow_mixed_protocol: bool = False,
) -> dict[str, Any]:
    """Merge ablation artefacts that share profiles and judges but cover different queries.

    Args:
        base_path: Earlier ablation artefact; its metadata is the starting point.
        added_paths: Later artefacts with additional, non-overlapping queries.
        allow_mixed_protocol: Merge runs whose session protocol or system code
            differ, marking the result as mixed instead of refusing.

    Returns:
        dict[str, Any]: One ablation artefact with all records and recomputed summaries.

    Raises:
        ValueError: When query ids overlap, when profiles or judges differ, or
            when the session protocol or system code differ and
            ``allow_mixed_protocol`` is false.
    """
    base = _load_json(base_path)
    runs = [(Path(base_path), base)] + [(Path(path), _load_json(path)) for path in added_paths]
    reference_profiles = _profile_models(base)
    reference_judges = _judge_models(base)
    seen_ids: dict[str, str] = {}
    for path, payload in runs:
        if _profile_models(payload) != reference_profiles:
            raise ValueError(f"{path.name}: comparison profiles differ from the base run: {_profile_models(payload)}")
        if _judge_models(payload) != reference_judges:
            raise ValueError(f"{path.name}: judges differ from the base run: {_judge_models(payload)}")
        for record in payload.get("ablation_results", []):
            query_id = str(record.get("id"))
            if query_id in seen_ids:
                raise ValueError(f"Query {query_id} appears in both {seen_ids[query_id]} and {path.name}.")
            seen_ids[query_id] = path.name

    base_summary = base.get("summary") or {}
    profile_order = [str(key) for key in (base_summary.get("comparison_profile_order") or reference_profiles)]
    primary_profile_key = str(base_summary.get("primary_comparison_profile") or profile_order[0])

    results = []
    for path, payload in runs:
        for record in payload.get("ablation_results", []):
            merged_record = deepcopy(record)
            merged_record["source_run"] = path.stem
            results.append(merged_record)

    profile_summaries = {key: _build_profile_summary(results, key) for key in profile_order}
    summary = deepcopy(profile_summaries.get(primary_profile_key, {}))
    summary["primary_score"] = deepcopy(base_summary.get("primary_score"))
    summary["comparison_profiles"] = profile_summaries
    summary["primary_comparison_profile"] = primary_profile_key
    summary["comparison_profile_order"] = profile_order

    run_descriptions = [_describe_run(path, payload) for path, payload in runs]
    protocols = {bool(run["fresh_session"]) for run in run_descriptions}
    code_versions = {run["system_code_sha256"] or run["commit"] for run in run_descriptions}
    mixed_protocol = len(protocols) > 1 or len(code_versions) > 1
    if mixed_protocol and not allow_mixed_protocol:
        raise ValueError(
            "The runs differ in session protocol or system code "
            f"(fresh_session: {sorted(protocols)}; code versions: {len(code_versions)}). "
            "They are not one controlled run; pass --allow-mixed-protocol only for an archival merge."
        )
    metadata = deepcopy(base.get("ablation_metadata") or {})
    metadata.update(
        {
            "groundtruth_queries_path": run_descriptions[-1]["groundtruth_queries_path"],
            "groundtruth_queries_fingerprint": None,
            "groundtruth_queries_count": len(results),
            "total_queries": len(results),
            "ablation_domains": sorted({str(record.get("domain")) for record in results}),
            "run_started_at": min(str(run["run_started_at"]) for run in run_descriptions if run["run_started_at"]),
            "run_finished_at": max(str(run["run_finished_at"]) for run in run_descriptions if run["run_finished_at"]),
            "total_runtime_s": round(sum(float(run["total_runtime_s"] or 0.0) for run in run_descriptions), 3),
            "merged_runs": run_descriptions,
            "mixed_protocol": mixed_protocol,
            "merged_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    return {"ablation_metadata": metadata, "summary": summary, "ablation_results": results}


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Merge ablation artefacts that cover different queries.")
    parser.add_argument("--base", required=True, help="Earlier ablation artefact.")
    parser.add_argument("--add", required=True, action="append", dest="added", help="Repeatable. Later artefact to add.")
    parser.add_argument("--output-prefix", default="ablation_final", help="Prefix inside eval/results/ablation/.")
    parser.add_argument("--output-file", default=None, help="Explicit output path, instead of eval/results/ablation/.")
    parser.add_argument(
        "--allow-mixed-protocol",
        action="store_true",
        help="Merge runs with a different session protocol or system code (archival use; marked as mixed).",
    )
    args = parser.parse_args()

    merged = merge_ablation_payloads(args.base, args.added, allow_mixed_protocol=args.allow_mixed_protocol)
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = build_results_output_path("ablation", args.output_prefix, datetime.now().strftime("%Y%m%d_%H%M%S"))
    merged["ablation_metadata"]["output_file"] = str(output_path)
    merged["ablation_metadata"]["output_directory"] = str(output_path.parent)
    write_json_artifact(merged, output_path)
    runs = merged["ablation_metadata"]["merged_runs"]
    print(f"Merged {len(merged['ablation_results'])} queries from {len(runs)} runs: "
          + ", ".join(f"{run['source_run']} ({len(run['queries'])})" for run in runs))
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
