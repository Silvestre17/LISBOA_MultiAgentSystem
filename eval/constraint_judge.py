# ==========================================================================
# Master Thesis - Itinerary Constraint Checklist
#   - André Filipe Gomes Silvestre, 20240502
#
#   Runs after an ablation, on the stored responses. For every itinerary
#   request in eval/paper_eval_annotations.json, each judge marks each
#   explicit user constraint as met, not met, or not assessable, in the
#   zero-shot and LISBOA responses of every model profile.
#
#   The judge sees only the request, the constraints, and the response: no
#   tools, retrieved context, reference facts, or condition label. The check
#   is therefore blind to the condition by construction. It measures whether
#   a plan does what was asked, not whether the plan is feasible in practice.
#
# Usage:
#   > python -m eval.constraint_judge --ablation eval/results/ablation/ablation_final_<timestamp>.json
#       Check every itinerary response with the judges recorded in the ablation file.
#   > python -m eval.constraint_judge --ablation <file> --limit 2 --output-prefix constraint_checklist_test
#       Check only the first two itinerary requests (smoke test).
# ==========================================================================

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.llm_factory import LLMFactory
from eval.llm_judge import LLMJudge
from eval.runtime_utils import (
    build_cost_payload,
    build_model_id,
    build_results_output_path,
    build_run_provenance,
    build_usage_payload,
    combine_cost_payloads,
    combine_usage_payloads,
    describe_file,
    load_pricing_catalog,
    parse_model_spec,
    write_json_artifact,
)

DEFAULT_ANNOTATIONS_PATH = Path(__file__).with_name("paper_eval_annotations.json")
VERDICTS = ("met", "not_met", "cannot_assess")
ARMS = ("zero_shot", "lisboa")
MAX_ATTEMPTS = 2
CONSTRAINT_PROMPT_TEMPLATE = """You check whether a response to a trip-planning request satisfies each explicit constraint of that request.

REQUEST:
{query}

CONSTRAINTS:
{constraints}

RESPONSE:
{response}

For each constraint, choose one verdict:
- "met": the response explicitly does what the constraint asks.
- "not_met": the response does not address the constraint, or contradicts it.
- "cannot_assess": the response addresses the constraint, but too ambiguously to decide.

Judge only what the response states; do not infer what it leaves unsaid. Do not use outside knowledge to check facts, travel times, or opening hours: this check is about whether the plan does what was asked, not whether its details are correct. When a constraint accepts a stated limitation (for example, "or says when a connection cannot be confirmed"), the constraint is met if the response states that limitation explicitly.

Return ONLY a JSON object, without markdown fences, in this form:
{{"verdicts": [{{"constraint": 1, "verdict": "met", "justification": "One short sentence."}}]}}
Give exactly one verdict for each of the {count} constraints, in order."""


def _load_json(path: str | Path) -> Any:
    """Load one JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_judges(ablation_payload: dict, judge_model_specs: list[str] | None) -> list[dict]:
    """Use the judges recorded in the ablation file unless others are given."""
    if judge_model_specs:
        return [parse_model_spec(spec, temperature=0.0) for spec in judge_model_specs]
    recorded = (ablation_payload.get("ablation_metadata") or {}).get("judge_model_configs") or []
    if not recorded:
        raise ValueError("The ablation file records no judge models; pass --judge-model-spec.")
    return [
        {"provider": str(config["provider"]), "model": str(config["model"]), "temperature": 0.0}
        for config in recorded
    ]


def collect_checklist_items(ablation_payload: dict, annotations: dict, limit: int | None = None) -> list[dict]:
    """List every (itinerary request, profile, arm) response to check.

    Args:
        ablation_payload: Ablation results as written by ``eval.run_ablation``.
        annotations: Annotation file content with ``query_type`` and ``constraints``.
        limit: Optional number of itinerary requests to keep, in corpus order.

    Returns:
        list[dict]: One item per stored response, with the request, its
        constraints, the response text, and the response error, if any.
    """
    itinerary = {
        query_id: annotation["constraints"]
        for query_id, annotation in (annotations.get("queries") or {}).items()
        if annotation.get("query_type") == "itinerary" and annotation.get("constraints")
    }
    records = [record for record in ablation_payload.get("ablation_results", []) if record.get("id") in itinerary]
    if limit is not None:
        records = records[:limit]

    items = []
    for record in records:
        for profile_key, comparison in sorted((record.get("comparisons") or {}).items()):
            for arm in ARMS:
                metrics = ((comparison or {}).get("metrics") or {}).get(arm) or {}
                items.append(
                    {
                        "id": record["id"],
                        "query": record["query"],
                        "language": record.get("language"),
                        "profile": profile_key,
                        "arm": arm,
                        "constraints": list(itinerary[record["id"]]),
                        "response": str(metrics.get("response") or ""),
                        "response_error": metrics.get("error"),
                    }
                )
    return items


def _parse_verdicts(text: str, constraint_count: int) -> list[dict]:
    """Parse and validate the judge's JSON verdict list."""
    payload = LLMJudge._extract_json_candidate(text)
    verdicts = payload.get("verdicts")
    if not isinstance(verdicts, list) or len(verdicts) != constraint_count:
        raise ValueError(f"Expected {constraint_count} verdicts, got {len(verdicts) if isinstance(verdicts, list) else 'none'}.")
    parsed = []
    for position, entry in enumerate(verdicts, start=1):
        verdict = str((entry or {}).get("verdict") or "").strip().lower().replace(" ", "_").replace("-", "_")
        if verdict not in VERDICTS:
            raise ValueError(f"Constraint {position}: unknown verdict '{verdict}'.")
        parsed.append({"verdict": verdict, "justification": str((entry or {}).get("justification") or "").strip()})
    return parsed


def check_item(item: dict, judge: dict, llm: Any, pricing_by_model: dict | None) -> dict:
    """Ask one judge for the constraint verdicts of one stored response.

    Args:
        item: One response from ``collect_checklist_items``.
        judge: Judge config with ``provider`` and ``model``.
        llm: Chat model for that judge, at temperature 0.
        pricing_by_model: Price catalog used to cost the judge calls.

    Returns:
        dict: Verdicts with one-line justifications, usage, cost, latency, and
        the error when the response was missing or the judge output was invalid.
    """
    model_id = build_model_id(judge["provider"], judge["model"])
    result = {
        "id": item["id"],
        "profile": item["profile"],
        "arm": item["arm"],
        "judge_model": model_id,
        "verdicts": [],
        "error": None,
    }
    if item["response_error"] is not None or not item["response"].strip():
        result["error"] = f"Skipped: no response to check ({item['response_error'] or 'empty response'})."
        result["usage"] = build_usage_payload({}, model_id=model_id, call_count=0)
        result["cost_usd"] = build_cost_payload(result["usage"], pricing_by_model, model_id=model_id)
        return result

    prompt = CONSTRAINT_PROMPT_TEMPLATE.format(
        query=item["query"],
        constraints="\n".join(f"{index}. {text}" for index, text in enumerate(item["constraints"], start=1)),
        response=item["response"],
        count=len(item["constraints"]),
    )
    usages = []
    started = time.perf_counter()
    last_error = None
    for _attempt in range(MAX_ATTEMPTS):
        try:
            raw = llm.invoke(prompt)
            usages.append(build_usage_payload(LLMFactory.extract_usage_metadata(raw), model_id=model_id, call_count=1))
            verdicts = _parse_verdicts(LLMJudge._extract_response_text(raw), len(item["constraints"]))
            result["verdicts"] = [
                {"constraint": constraint, **verdict}
                for constraint, verdict in zip(item["constraints"], verdicts, strict=True)
            ]
            last_error = None
            break
        except Exception as exc:  # parse or API failure: one retry, then record the error
            last_error = f"{type(exc).__name__}: {exc}"
    result["error"] = last_error
    result["latency_s"] = round(time.perf_counter() - started, 3)
    result["usage"] = combine_usage_payloads(usages) if usages else build_usage_payload({}, model_id=model_id, call_count=0)
    result["cost_usd"] = build_cost_payload(result["usage"], pricing_by_model, model_id=model_id)
    return result


def _rates(counts: dict[str, int]) -> dict[str, Any]:
    """Return verdict counts with the met rate over assessable and over all constraints."""
    total = sum(counts.values())
    assessable = counts["met"] + counts["not_met"]
    return {
        **counts,
        "total": total,
        "met_rate_assessable": round(counts["met"] / assessable, 4) if assessable else None,
        "met_rate_all": round(counts["met"] / total, 4) if total else None,
    }


def summarize(results: list[dict]) -> dict[str, Any]:
    """Summarize verdicts per profile and arm, per judge and on judge consensus."""
    by_cell: dict[str, Any] = {}
    cells = sorted({(result["profile"], result["arm"]) for result in results})
    judges = sorted({result["judge_model"] for result in results})
    for profile, arm in cells:
        cell_results = [r for r in results if r["profile"] == profile and r["arm"] == arm and not r["error"]]
        per_judge = {}
        for judge in judges:
            counts = dict.fromkeys(VERDICTS, 0)
            for result in cell_results:
                if result["judge_model"] == judge:
                    for verdict in result["verdicts"]:
                        counts[verdict["verdict"]] += 1
            per_judge[judge] = _rates(counts)

        # Consensus: a constraint counts only when every judge gave the same verdict.
        consensus = dict.fromkeys(VERDICTS, 0)
        disagreements = 0
        grouped: dict[str, list[dict]] = {}
        for result in cell_results:
            grouped.setdefault(result["id"], []).append(result)
        for query_results in grouped.values():
            if len(query_results) != len(judges):
                continue
            for position in range(len(query_results[0]["verdicts"])):
                labels = {result["verdicts"][position]["verdict"] for result in query_results}
                if len(labels) == 1:
                    consensus[labels.pop()] += 1
                else:
                    disagreements += 1
        # Rates over unanimous verdicts alone would hide the disputed
        # constraints, so the consensus block also reports every constraint.
        unanimous = _rates(consensus)
        all_constraints = unanimous["total"] + disagreements
        by_cell[f"{profile}::{arm}"] = {
            "profile": profile,
            "arm": arm,
            "queries": len(grouped),
            "per_judge": per_judge,
            "consensus": {
                **unanimous,
                "judge_disagreements": disagreements,
                "all_constraints": all_constraints,
                "met_rate_all_constraints": round(consensus["met"] / all_constraints, 4) if all_constraints else None,
            },
        }

    # Inter-judge agreement over all paired constraint verdicts.
    agreement = None
    if len(judges) == 2:
        paired: dict[tuple, dict[str, str]] = {}
        for result in results:
            if result["error"]:
                continue
            for position, verdict in enumerate(result["verdicts"]):
                key = (result["id"], result["profile"], result["arm"], position)
                paired.setdefault(key, {})[result["judge_model"]] = verdict["verdict"]
        pairs = [(labels[judges[0]], labels[judges[1]]) for labels in paired.values() if len(labels) == 2]
        if pairs:
            observed = sum(a == b for a, b in pairs) / len(pairs)
            expected = sum(
                (sum(a == label for a, _ in pairs) / len(pairs)) * (sum(b == label for _, b in pairs) / len(pairs))
                for label in VERDICTS
            )
            agreement = {
                "judges": judges,
                "paired_verdicts": len(pairs),
                "exact_agreement": round(observed, 4),
                "cohen_kappa": round((observed - expected) / (1 - expected), 4) if expected < 1 else None,
            }
    return {"cells": by_cell, "inter_judge_agreement": agreement}


def run_constraint_checklist(
    ablation_path: str | Path,
    *,
    annotations_path: str | Path = DEFAULT_ANNOTATIONS_PATH,
    judge_model_specs: list[str] | None = None,
    limit: int | None = None,
    workers: int = 4,
    output_prefix: str = "constraint_checklist",
) -> Path:
    """Check every itinerary response in an ablation file and save the verdicts.

    Args:
        ablation_path: Ablation results JSON.
        annotations_path: Annotation file with the itinerary constraints.
        judge_model_specs: Optional ``provider::model`` judges; defaults to the
            judges recorded in the ablation file.
        limit: Optional number of itinerary requests to check.
        workers: Parallel judge calls.
        output_prefix: File prefix inside ``eval/results/constraints/``.

    Returns:
        Path: The JSON file with metadata, summary, and every verdict.
    """
    started_at = datetime.now()
    ablation_payload = _load_json(ablation_path)
    annotations = _load_json(annotations_path)
    pricing_by_model = load_pricing_catalog()
    judges = resolve_judges(ablation_payload, judge_model_specs)
    llms = {
        build_model_id(judge["provider"], judge["model"]): LLMFactory.get_llm(
            provider=judge["provider"], model=judge["model"], temperature=0.0
        )
        for judge in judges
    }
    items = collect_checklist_items(ablation_payload, annotations, limit=limit)
    tasks = [(item, judge) for item in items for judge in judges]
    print(f"[Constraints] {len(items)} responses x {len(judges)} judges = {len(tasks)} checks ({workers} in parallel).")

    def _run(task: tuple[dict, dict]) -> dict:
        item, judge = task
        return check_item(item, judge, llms[build_model_id(judge["provider"], judge["model"])], pricing_by_model)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        results = list(executor.map(_run, tasks))
    for result in results:
        status = "ERROR " + str(result["error"])[:80] if result["error"] else " ".join(
            verdict["verdict"] for verdict in result["verdicts"]
        )
        print(f"  {result['id']} {result['profile']:<13} {result['arm']:<9} {result['judge_model']:<22} {status}")

    usage = combine_usage_payloads([result["usage"] for result in results])
    cost = combine_cost_payloads([result["cost_usd"] for result in results])
    output_path = build_results_output_path("constraints", output_prefix, started_at.strftime("%Y%m%d_%H%M%S"))
    write_json_artifact(
        {
            "constraint_metadata": {
                "started_at": started_at.isoformat(),
                "finished_at": datetime.now().isoformat(),
                "ablation_file": describe_file(ablation_path),
                "ablation_run_started_at": (ablation_payload.get("ablation_metadata") or {}).get("run_started_at"),
                "judges": judges,
                "verdicts": list(VERDICTS),
                "prompt_template": CONSTRAINT_PROMPT_TEMPLATE,
                "blind_to_condition": True,
                "limit": limit,
                "usage": usage,
                "cost_usd": cost,
                "errors": sum(1 for result in results if result["error"]),
                "provenance": build_run_provenance(dataset_path=ablation_path, annotations_path=annotations_path),
            },
            "summary": summarize(results),
            "results": results,
        },
        output_path,
    )
    print(f"[Constraints] Saved to {output_path} | cost USD {cost.get('total_cost_usd', 0):.4f}")
    return output_path


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Check itinerary constraints in stored ablation responses.")
    parser.add_argument("--ablation", required=True, help="Path to the ablation results JSON.")
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS_PATH), help="Annotation file with the constraints.")
    parser.add_argument(
        "--judge-model-spec",
        action="append",
        dest="judge_model_specs",
        help="Repeatable provider::model judge. Defaults to the judges recorded in the ablation file.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Check only the first N itinerary requests.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel judge calls.")
    parser.add_argument("--output-prefix", default="constraint_checklist", help="Output prefix in eval/results/constraints/.")
    args = parser.parse_args()
    run_constraint_checklist(
        args.ablation,
        annotations_path=args.annotations,
        judge_model_specs=args.judge_model_specs,
        limit=args.limit,
        workers=args.workers,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    main()
