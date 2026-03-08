# ==========================================================================
# Master Thesis - Ablation Runner
#   - André Filipe Gomes Silvestre, 20240502
#
#   Compares a zero-shot baseline against the full LISBOA multi-agent system
#   and writes the JSON artefacts into:
#   eval/results/ablation/
#
# Usage:
#   python eval/run_ablation.py --mode run_test                    # Quick ablation with 5 dataset entries
#   python eval/run_ablation.py --limit 20                         # Run the first 20 dataset entries
#   python eval/run_ablation.py --zero-shot-model gpt-5-mini       # Override the zero-shot baseline model
# ==========================================================================

import json
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage

from agent.graph import MultiAgentAssistant  # For the LISBOA full pipeline comparison
from agent.llm_factory import LLMFactory
from config import Config
from eval.llm_judge import LLMJudge
from eval.runtime_utils import (
    build_cost_payload,
    build_model_id,
    build_model_manifest,
    build_results_output_path,
    build_run_metadata,
    build_usage_payload,
    categorize_error,
    combine_cost_payloads,
    combine_usage_payloads,
    compute_tool_metrics,
    get_pricing_metadata,
    select_balanced_subset,
    split_pricing_config,
)
from eval.validators.response_heuristics import run_all_heuristics

# TEST: This shared ground-truth corpus feeds both benchmark and ablation.
GROUNDTRUTH_QUERIES_PATH = Path(__file__).with_name("evaluation_groundtruth_queries.json")
# TEST: Zero-shot baseline model lives here by default, or can be overridden via CLI.
DEFAULT_ZERO_SHOT_PROVIDER = "azure"
DEFAULT_ZERO_SHOT_MODEL = "gpt-5-mini"


def load_groundtruth_queries(filepath: str | Path = GROUNDTRUTH_QUERIES_PATH, limit=20):
    """Load a prefix of the shared evaluation ground-truth corpus for ablation runs."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return select_balanced_subset(data, limit, group_key="domain")


def run_zero_shot(
    query: str,
    provider: str = DEFAULT_ZERO_SHOT_PROVIDER,
    model_name: str = DEFAULT_ZERO_SHOT_MODEL,
):
    """Run the raw zero-shot baseline without LISBOA tool grounding."""
    llm = LLMFactory.get_llm(provider=provider, model=model_name, temperature=0.0)
    model_id = build_model_id(provider, model_name)
    
    start = time.time()
    try:
        response = llm.invoke([HumanMessage(content=query)])
        latency = time.time() - start
        response_usage = build_usage_payload(
            LLMFactory.extract_usage_metadata(response),
            model_id=model_id,
            call_count=1,
        )
        return response.content, [], "", latency, None, response_usage
    except Exception as e:
        return (
            f"Error: {e}",
            [],
            "",
            time.time() - start,
            str(e),
            build_usage_payload({}, model_id=model_id, call_count=0),
        )


def run_lisboa(query: str, system: MultiAgentAssistant):
    """Run the full LISBOA system while instrumenting tool calls for evaluation."""
    start = time.time()
    tools_called = []
    retrieved_context_blocks = []
    patched_invocations = []
    system.reset_llm_usage_tracking()

    def _wrap_tool(tool_name, original_invoke):
        def _instrumented_invoke(tool_args):
            result = original_invoke(tool_args)
            tools_called.append(tool_name)
            retrieved_context_blocks.append(f"[{tool_name}] returned:\n{result}")
            return result

        return _instrumented_invoke

    try:
        for agent in system.agents.values():
            for tool in getattr(agent, "tools", []):
                original_invoke = tool.invoke
                object.__setattr__(tool, "invoke", _wrap_tool(tool.name, original_invoke))
                patched_invocations.append((tool, original_invoke))

        response = system.chat(query)
        latency = time.time() - start
        
        retrieved_context_str = "\n---\n".join(retrieved_context_blocks)
        response_usage = build_usage_payload(system.get_llm_usage_summary())
        
        return response, tools_called, retrieved_context_str, latency, None, response_usage
    except Exception as e:
        return (
            f"Error: {e}",
            [],
            "",
            time.time() - start,
            str(e),
            build_usage_payload(system.get_llm_usage_summary()),
        )
    finally:
        for tool, original_invoke in patched_invocations:
            object.__setattr__(tool, "invoke", original_invoke)


def _build_lisboa_response_model_manifest(system: MultiAgentAssistant) -> dict:
    """Return a compact, explicit summary of the models used by LISBOA."""
    configured_agent_models = Config.get_agent_models()
    compact_agent_models = {}
    for agent_name, model_info in getattr(system, "model_info", {}).items():
        configured_model = configured_agent_models.get(agent_name, {})
        extra = {}
        if isinstance(model_info, dict):
            extra["type"] = model_info.get("type", "Unknown")
        else:
            extra["type"] = "Unknown"

        compact_agent_models[agent_name] = build_model_manifest(
            configured_model.get("provider", "unknown"),
            configured_model.get(
                "model",
                str(model_info) if not isinstance(model_info, dict) else model_info.get("model", "Unknown"),
            ),
            configured_model.get("temperature"),
            extra=extra,
        )

    return {
        "kind": "multi_agent",
        "display_model": system.model_name,
        "agent_models": compact_agent_models,
    }


def run_ablation(
    limit: int = None,
    zero_shot_provider: str = DEFAULT_ZERO_SHOT_PROVIDER,
    zero_shot_model: str = DEFAULT_ZERO_SHOT_MODEL,
    pricing_by_model: dict | None = None,
):
    """
    Execute the ablation study and save the results JSON.

    Args:
        limit: Maximum number of shared ground-truth queries to compare.
        zero_shot_provider: Provider used by the baseline arm.
        zero_shot_model: Model used by the baseline arm.
        pricing_by_model: Optional pricing catalog keyed by model name or
            ``provider::model`` with ``input`` and ``output`` prices in USD per
            million tokens. When provided, both zero-shot and LISBOA arms store
            organized response/evaluation/combined token counts and costs.
    """
    print("=" * 60)
    print(f"STARTING ABLATION STUDY (Zero-Shot vs LISBOA Framework) (LIMIT={limit})")
    print("=" * 60)
    
    groundtruth_queries = load_groundtruth_queries(
        GROUNDTRUTH_QUERIES_PATH,
        limit=limit if limit else 20,
    )
    try:
        judge = LLMJudge()
    except ValueError as e:
        print(f"FAILED TO INIT JUDGE: {e}")
        return

    lisboa_system = MultiAgentAssistant()
    zero_shot_model_manifest = build_model_manifest(
        zero_shot_provider,
        zero_shot_model,
        0.0,
    )
    zero_shot_model_id = zero_shot_model_manifest["model_id"]
    # TEST: The evaluator model is selected in eval/llm_judge.py via
    # EVAL_JUDGE_PROVIDER / EVAL_JUDGE_MODEL_NAME, not in the zero-shot defaults above.
    evaluation_model_manifest = build_model_manifest(
        judge.provider,
        judge.model_name,
        0.0,
    )
    evaluation_model_id = evaluation_model_manifest["model_id"]
    lisboa_response_model_config = _build_lisboa_response_model_manifest(lisboa_system)
    lisboa_response_model = lisboa_response_model_config["display_model"]
    pricing_catalog, _ = split_pricing_config(pricing_by_model)
    
    results = []
    consecutive_errors = 0

    for idx, item in enumerate(groundtruth_queries):
        print(f"\n[{idx+1}/{len(groundtruth_queries)}] ABLATING: {item['query']}")
        
        # 1. Zero-Shot
        zs_resp, zs_tools, zs_ctx, zs_lat, zs_err, zs_response_usage = run_zero_shot(
            item["query"],
            provider=zero_shot_provider,
            model_name=zero_shot_model,
        )
        zs_response_cost = build_cost_payload(
            zs_response_usage,
            pricing_by_model,
            model_id=zero_shot_model_id,
        )
        zs_heuristics = None
        zs_tool_metrics = compute_tool_metrics(
            expected=item.get("expected_tools", []),
            actual=zs_tools,
        )
        empty_eval_usage = build_usage_payload({}, model_id=evaluation_model_id, call_count=0)
        empty_eval_cost = build_cost_payload(
            empty_eval_usage,
            pricing_by_model,
            model_id=evaluation_model_id,
        )
        if zs_err is not None:
            zs_judge_result = {
                "composite_score": None,
                "reasoning": f"Model failed. Skipped judge. Error: {zs_err}",
                "factual_accuracy": None,
                "tool_usage": None,
                "completeness": None,
                "relevance": None,
                "response_quality": None,
                "evaluation_usage": empty_eval_usage,
                "evaluation_cost_usd": empty_eval_cost,
            }
            print(f"  [Zero-Shot] Error: {zs_err}")
        else:
            try:
                zs_judge_result = judge.evaluate(
                    query=item['query'],
                    expected_facts=item.get('expected_facts', []),
                    expected_tools=item.get('expected_tools', []),
                    actual_tools=zs_tools,
                    retrieved_context=zs_ctx,
                    response=zs_resp,
                    pricing_by_model=pricing_by_model,
                )
            except Exception as e:
                zs_judge_result = {
                    "composite_score": None,
                    "reasoning": f"Judge error: {e}",
                    "factual_accuracy": None,
                    "tool_usage": None,
                    "completeness": None,
                    "relevance": None,
                    "response_quality": None,
                    "evaluation_usage": empty_eval_usage,
                    "evaluation_cost_usd": empty_eval_cost,
                }
            zs_heuristics = run_all_heuristics(
                response=zs_resp,
                expected_language=item.get("language", "en"),
            )

        zs_evaluation_usage = zs_judge_result.get("evaluation_usage", empty_eval_usage)
        zs_evaluation_cost = zs_judge_result.get("evaluation_cost_usd", empty_eval_cost)
        zs_score = {
            key: value
            for key, value in zs_judge_result.items()
            if key not in {"evaluation_usage", "evaluation_cost_usd"}
        }
        zs_combined_usage = combine_usage_payloads([zs_response_usage, zs_evaluation_usage])
        zs_combined_cost = combine_cost_payloads([zs_response_cost, zs_evaluation_cost])
        
        score_disp_zs = f"{zs_score['composite_score']:.2f}/5.0" if zs_score['composite_score'] is not None else "N/A"
        print(f"  [Zero-Shot] Score: {score_disp_zs} | Lat: {zs_lat:.2f}s")
        
        # 2. LISBOA (Tool Grounded)
        ls_resp, ls_tools, ls_ctx, ls_lat, ls_err, ls_response_usage = run_lisboa(item['query'], lisboa_system)
        ls_response_cost = build_cost_payload(
            ls_response_usage,
            pricing_by_model,
        )
        ls_tool_metrics = compute_tool_metrics(
            expected=item.get("expected_tools", []),
            actual=ls_tools,
        )
        if ls_err is not None:
            ls_judge_result = {
                "composite_score": None,
                "reasoning": f"Model failed. Skipped judge. Error: {ls_err}",
                "factual_accuracy": None,
                "tool_usage": None,
                "completeness": None,
                "relevance": None,
                "response_quality": None,
                "evaluation_usage": empty_eval_usage,
                "evaluation_cost_usd": empty_eval_cost,
            }
            print(f"  [LISBOA] Error: {ls_err}")
        else:
            try:
                ls_judge_result = judge.evaluate(
                    query=item['query'],
                    expected_facts=item.get('expected_facts', []),
                    expected_tools=item.get('expected_tools', []),
                    actual_tools=ls_tools,
                    retrieved_context=ls_ctx,
                    response=ls_resp,
                    pricing_by_model=pricing_by_model,
                )
            except Exception as e:
                ls_judge_result = {
                    "composite_score": None,
                    "reasoning": f"Judge error: {e}",
                    "factual_accuracy": None,
                    "tool_usage": None,
                    "completeness": None,
                    "relevance": None,
                    "response_quality": None,
                    "evaluation_usage": empty_eval_usage,
                    "evaluation_cost_usd": empty_eval_cost,
                }
        ls_heuristics = None if ls_err is not None else run_all_heuristics(
            response=ls_resp,
            expected_language=item.get("language", "en"),
        )
        ls_evaluation_usage = ls_judge_result.get("evaluation_usage", empty_eval_usage)
        ls_evaluation_cost = ls_judge_result.get("evaluation_cost_usd", empty_eval_cost)
        ls_score = {
            key: value
            for key, value in ls_judge_result.items()
            if key not in {"evaluation_usage", "evaluation_cost_usd"}
        }
        ls_combined_usage = combine_usage_payloads([ls_response_usage, ls_evaluation_usage])
        ls_combined_cost = combine_cost_payloads([ls_response_cost, ls_evaluation_cost])
        
        score_disp_ls = f"{ls_score['composite_score']:.2f}/5.0" if ls_score['composite_score'] is not None else "N/A"
        print(f"  [LISBOA]    Score: {score_disp_ls} | Lat: {ls_lat:.2f}s | Tools: {len(ls_tools)}")

        if zs_err is not None or ls_err is not None:
            consecutive_errors += 1
        else:
            consecutive_errors = 0

        comparison_usage = combine_usage_payloads([zs_combined_usage, ls_combined_usage])
        comparison_cost = combine_cost_payloads([zs_combined_cost, ls_combined_cost])

        results.append({
            "id": item["id"],
            "query": item["query"],
            "domain": item["domain"],
            "language": item.get("language", "en"),
            "edge_case": item.get("edge_case", False),
            "edge_type": item.get("edge_type", None),
            "expected_behavior": item.get("expected_behavior"),
            "expected_facts": item.get("expected_facts", []),
            "expected_tools": item.get("expected_tools", []),
            "comparison_usage": comparison_usage,
            "comparison_cost_usd": comparison_cost,
            "metrics": {
                "zero_shot": {
                    "response_model": zero_shot_model_id,
                    "response_model_config": deepcopy(zero_shot_model_manifest),
                    "evaluation_model": evaluation_model_id,
                    "evaluation_model_config": deepcopy(evaluation_model_manifest),
                    "scores": zs_score,
                    "response": zs_resp,
                    "response_usage": zs_response_usage,
                    "response_cost_usd": zs_response_cost,
                    "evaluation_usage": zs_evaluation_usage,
                    "evaluation_cost_usd": zs_evaluation_cost,
                    "combined_usage": zs_combined_usage,
                    "combined_cost_usd": zs_combined_cost,
                    "latency": round(zs_lat, 3),
                    "error": zs_err,
                    "error_type": categorize_error(zs_err),
                    "tool_metrics": zs_tool_metrics,
                    "heuristics": zs_heuristics,
                },
                "lisboa": {
                    "response_model": lisboa_response_model,
                    "response_model_config": deepcopy(lisboa_response_model_config),
                    "evaluation_model": evaluation_model_id,
                    "evaluation_model_config": deepcopy(evaluation_model_manifest),
                    "scores": ls_score,
                    "response": ls_resp,
                    "tools_used": ls_tools,
                    "retrieved_context": ls_ctx,
                    "response_usage": ls_response_usage,
                    "response_cost_usd": ls_response_cost,
                    "evaluation_usage": ls_evaluation_usage,
                    "evaluation_cost_usd": ls_evaluation_cost,
                    "combined_usage": ls_combined_usage,
                    "combined_cost_usd": ls_combined_cost,
                    "llm_usage_breakdown": ls_response_usage.get("llm_usage_breakdown", []),
                    "llm_usage_by_agent": ls_response_usage.get("by_agent", {}),
                    "latency": round(ls_lat, 3),
                    "error": ls_err,
                    "error_type": categorize_error(ls_err),
                    "tool_metrics": ls_tool_metrics,
                    "heuristics": ls_heuristics,
                }
            }
        })
        
        if consecutive_errors >= 2:
            print("\nABORTING: API or Models are failing continuously. Saving costs.")
            break

    # Build aggregate summary
    zs_scores = [r["metrics"]["zero_shot"]["scores"]["composite_score"] for r in results
                 if r["metrics"]["zero_shot"]["scores"]["composite_score"] is not None]
    ls_scores = [r["metrics"]["lisboa"]["scores"]["composite_score"] for r in results
                 if r["metrics"]["lisboa"]["scores"]["composite_score"] is not None]
    
    summary = {
        "total_queries": len(results),
        "zero_shot_avg": round(sum(zs_scores) / len(zs_scores), 3) if zs_scores else 0,
        "lisboa_avg": round(sum(ls_scores) / len(ls_scores), 3) if ls_scores else 0,
        "zero_shot_avg_tool_f1": round(
            sum(r["metrics"]["zero_shot"]["tool_metrics"]["tool_f1"] for r in results) / len(results),
            3,
        ) if results else 0,
        "lisboa_avg_tool_f1": round(
            sum(r["metrics"]["lisboa"]["tool_metrics"]["tool_f1"] for r in results) / len(results),
            3,
        ) if results else 0,
        "lisboa_improvement": round(
            (sum(ls_scores) / len(ls_scores)) - (sum(zs_scores) / len(zs_scores)), 3
        ) if zs_scores and ls_scores else 0,
        "zero_shot_heuristics_pass_rate": round(
            sum(1 for r in results if (r["metrics"]["zero_shot"].get("heuristics") or {}).get("overall_pass", False)) / len(results),
            3,
        ) if results else 0,
        "lisboa_heuristics_pass_rate": round(
            sum(1 for r in results if (r["metrics"]["lisboa"].get("heuristics") or {}).get("overall_pass", False)) / len(results),
            3,
        ) if results else 0,
        "per_domain": {},
        "error_categories": {
            "zero_shot": {},
            "lisboa": {},
        },
    }

    def _compact_usage(payload: dict) -> dict:
        return {
            "call_count": int(payload.get("call_count", 0) or 0),
            "usage_available": bool(payload.get("usage_available", False)),
            "tokens": payload.get(
                "tokens",
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            ),
        }

    def _compact_cost(payload: dict) -> dict:
        return {
            "pricing_found": bool(payload.get("pricing_found", False)),
            "pricing_complete": bool(payload.get("pricing_complete", False)),
            "tokens": payload.get(
                "tokens",
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            ),
            "input_cost_usd": float(payload.get("input_cost_usd", 0.0) or 0.0),
            "output_cost_usd": float(payload.get("output_cost_usd", 0.0) or 0.0),
            "total_cost_usd": float(payload.get("total_cost_usd", 0.0) or 0.0),
            "missing_pricing_models": payload.get("missing_pricing_models", []),
        }

    summary["zero_shot_usage"] = {
        "response": _compact_usage(combine_usage_payloads([r["metrics"]["zero_shot"].get("response_usage", {}) for r in results])),
        "evaluation": _compact_usage(combine_usage_payloads([r["metrics"]["zero_shot"].get("evaluation_usage", {}) for r in results])),
        "combined": _compact_usage(combine_usage_payloads([r["metrics"]["zero_shot"].get("combined_usage", {}) for r in results])),
    }
    summary["zero_shot_cost_usd"] = {
        "response": _compact_cost(combine_cost_payloads([r["metrics"]["zero_shot"].get("response_cost_usd", {}) for r in results])),
        "evaluation": _compact_cost(combine_cost_payloads([r["metrics"]["zero_shot"].get("evaluation_cost_usd", {}) for r in results])),
        "combined": _compact_cost(combine_cost_payloads([r["metrics"]["zero_shot"].get("combined_cost_usd", {}) for r in results])),
    }
    summary["lisboa_usage"] = {
        "response": _compact_usage(combine_usage_payloads([r["metrics"]["lisboa"].get("response_usage", {}) for r in results])),
        "evaluation": _compact_usage(combine_usage_payloads([r["metrics"]["lisboa"].get("evaluation_usage", {}) for r in results])),
        "combined": _compact_usage(combine_usage_payloads([r["metrics"]["lisboa"].get("combined_usage", {}) for r in results])),
    }
    summary["lisboa_cost_usd"] = {
        "response": _compact_cost(combine_cost_payloads([r["metrics"]["lisboa"].get("response_cost_usd", {}) for r in results])),
        "evaluation": _compact_cost(combine_cost_payloads([r["metrics"]["lisboa"].get("evaluation_cost_usd", {}) for r in results])),
        "combined": _compact_cost(combine_cost_payloads([r["metrics"]["lisboa"].get("combined_cost_usd", {}) for r in results])),
    }
    summary["comparison_usage"] = _compact_usage(combine_usage_payloads([r.get("comparison_usage", {}) for r in results]))
    summary["comparison_cost_usd"] = _compact_cost(combine_cost_payloads([r.get("comparison_cost_usd", {}) for r in results]))
    
    # Per-domain comparison
    domains = set(r["domain"] for r in results)
    for domain in sorted(domains):
        dr = [r for r in results if r["domain"] == domain]
        d_zs = [r["metrics"]["zero_shot"]["scores"]["composite_score"] for r in dr
                if r["metrics"]["zero_shot"]["scores"]["composite_score"] is not None]
        d_ls = [r["metrics"]["lisboa"]["scores"]["composite_score"] for r in dr
                if r["metrics"]["lisboa"]["scores"]["composite_score"] is not None]
        summary["per_domain"][domain] = {
            "count": len(dr),
            "zero_shot_avg": round(sum(d_zs) / len(d_zs), 3) if d_zs else 0,
            "lisboa_avg": round(sum(d_ls) / len(d_ls), 3) if d_ls else 0,
            "zero_shot_avg_tool_f1": round(
                sum(r["metrics"]["zero_shot"]["tool_metrics"]["tool_f1"] for r in dr) / len(dr),
                3,
            ) if dr else 0,
            "lisboa_avg_tool_f1": round(
                sum(r["metrics"]["lisboa"]["tool_metrics"]["tool_f1"] for r in dr) / len(dr),
                3,
            ) if dr else 0,
            "zero_shot_usage": {
                "response": _compact_usage(combine_usage_payloads([r["metrics"]["zero_shot"].get("response_usage", {}) for r in dr])),
                "evaluation": _compact_usage(combine_usage_payloads([r["metrics"]["zero_shot"].get("evaluation_usage", {}) for r in dr])),
                "combined": _compact_usage(combine_usage_payloads([r["metrics"]["zero_shot"].get("combined_usage", {}) for r in dr])),
            },
            "zero_shot_cost_usd": {
                "response": _compact_cost(combine_cost_payloads([r["metrics"]["zero_shot"].get("response_cost_usd", {}) for r in dr])),
                "evaluation": _compact_cost(combine_cost_payloads([r["metrics"]["zero_shot"].get("evaluation_cost_usd", {}) for r in dr])),
                "combined": _compact_cost(combine_cost_payloads([r["metrics"]["zero_shot"].get("combined_cost_usd", {}) for r in dr])),
            },
            "lisboa_usage": {
                "response": _compact_usage(combine_usage_payloads([r["metrics"]["lisboa"].get("response_usage", {}) for r in dr])),
                "evaluation": _compact_usage(combine_usage_payloads([r["metrics"]["lisboa"].get("evaluation_usage", {}) for r in dr])),
                "combined": _compact_usage(combine_usage_payloads([r["metrics"]["lisboa"].get("combined_usage", {}) for r in dr])),
            },
            "lisboa_cost_usd": {
                "response": _compact_cost(combine_cost_payloads([r["metrics"]["lisboa"].get("response_cost_usd", {}) for r in dr])),
                "evaluation": _compact_cost(combine_cost_payloads([r["metrics"]["lisboa"].get("evaluation_cost_usd", {}) for r in dr])),
                "combined": _compact_cost(combine_cost_payloads([r["metrics"]["lisboa"].get("combined_cost_usd", {}) for r in dr])),
            },
        }

    zero_shot_error_categories = {}
    lisboa_error_categories = {}
    for record in results:
        zs_type = record["metrics"]["zero_shot"].get("error_type")
        ls_type = record["metrics"]["lisboa"].get("error_type")
        if zs_type:
            zero_shot_error_categories[zs_type] = zero_shot_error_categories.get(zs_type, 0) + 1
        if ls_type:
            lisboa_error_categories[ls_type] = lisboa_error_categories.get(ls_type, 0) + 1
    summary["error_categories"] = {
        "zero_shot": dict(sorted(zero_shot_error_categories.items())),
        "lisboa": dict(sorted(lisboa_error_categories.items())),
    }

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = build_results_output_path("ablation", "ablation_results", timestamp)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "ablation_metadata": build_run_metadata(
                GROUNDTRUTH_QUERIES_PATH,
                groundtruth_queries,
                response_models={
                    "zero_shot": zero_shot_model_id,
                    "lisboa": lisboa_response_model,
                },
                evaluation_model=evaluation_model_id,
                extra={
                    "response_model_configs": {
                        "zero_shot": zero_shot_model_manifest,
                        "lisboa": lisboa_response_model_config,
                    },
                    "evaluation_model_config": evaluation_model_manifest,
                    "timestamp": datetime.now().isoformat(),
                    "comparison": "zero_shot_vs_lisboa",
                    "real_services": True,
                    "pricing_model_count": len(pricing_catalog),
                    "output_directory": str(output_path.parent),
                    **get_pricing_metadata(pricing_by_model),
                },
            ),
            "summary": summary,
            "ablation_results": results,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nAblation Study complete. Results saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run the LISBOA ablation study (zero-shot baseline vs full system)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max queries to run per model")
    parser.add_argument("--mode", type=str, choices=["run_test", "full"], default="full", help="Mode: run_test (limit=5) or full (all dataset)")
    parser.add_argument(
        "--zero-shot-provider",
        type=str,
        default=DEFAULT_ZERO_SHOT_PROVIDER,
        help="LLM provider used for the zero-shot baseline",
    )
    parser.add_argument(
        "--zero-shot-model",
        type=str,
        default=DEFAULT_ZERO_SHOT_MODEL,
        help="Model used for the zero-shot baseline",
    )
    args = parser.parse_args()
    
    limit = 5 if args.mode == "run_test" else args.limit
    
    run_ablation(
        limit=limit,
        zero_shot_provider=args.zero_shot_provider,
        zero_shot_model=args.zero_shot_model,
    )
