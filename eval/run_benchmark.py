# ==========================================================================
# Master Thesis - Benchmark Runner
#   - André Filipe Gomes Silvestre, 20240502
#
#   Runs the academic LISBOA benchmark over isolated worker agents
#   (weather, transport, researcher) and writes the JSON artefacts into:
#   eval/results/benchmark/
#
# Usage:
#   python eval/run_benchmark.py --mode run_test        # Quick benchmark with 5 dataset entries per response model
#   python eval/run_benchmark.py --limit 20             # Benchmark the first 20 dataset entries per response model
#   python eval/run_benchmark.py --mode full            # Benchmark the full dataset with all configured response models
# ==========================================================================

import json
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from agent.agents.researcher_agent import ResearcherAgent
from agent.agents.transport_agent import TransportAgent
from agent.agents.weather_agent import WeatherAgent
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
    summarize_error_categories,
)
from eval.validators.response_heuristics import run_all_heuristics

GROUNDTRUTH_QUERIES_PATH = Path(__file__).with_name("evaluation_groundtruth_queries.json")
BENCHMARK_DOMAINS = ("weather", "transport", "researcher")


# TEST: Define the per-agent response-model benchmark matrix here.
# These defaults avoid placeholder models and run in the current Azure-based
# environment. Uncomment and edit the LM Studio entries when you want the
# 2 open-source + 2 proprietary matrix described in Section 3.6.1.
MODELS_TO_TEST = [
    # TEST: open-source model 1
    # {"provider": "lmstudio", "model": "qwen/qwen3-4b-2507", "temperature": 0.0},
    # TEST: open-source model 2
    # {"provider": "lmstudio", "model": "qwen/qwen3-8b", "temperature": 0.0},
    # TEST: proprietary model 1
    {"provider": "azure", "model": "gpt-5-mini", "temperature": 0.0},
    # TEST: proprietary model 2
    {"provider": "azure", "model": "gpt-5-nano", "temperature": 0.0},
]

# Latency SLA thresholds (seconds) per domain
SLA_THRESHOLDS = {
    "weather": 10.0,
    "transport": 15.0,
    "researcher": 20.0,
    "multi_agent": 30.0,
    "greeting": 5.0,
    "out_of_scope": 5.0,
}


def load_groundtruth_queries(
    filepath: str | Path = GROUNDTRUTH_QUERIES_PATH,
    *,
    domains: tuple[str, ...] = BENCHMARK_DOMAINS,
):
    """Load the shared evaluation ground-truth corpus for worker-agent benchmark runs."""
    with open(filepath, "r", encoding="utf-8") as f:
        records = json.load(f)
    return [item for item in records if item["domain"] in domains]


def run_isolated_agent(domain: str, query: str, config: dict):
    """Run a real worker agent in isolation and capture its live tool usage."""
    agent = None
    response_model_id = build_model_id(config["provider"], config["model"])
    if domain == "weather":
        agent = WeatherAgent()
    elif domain == "transport":
        agent = TransportAgent()
    elif domain == "researcher":
        agent = ResearcherAgent()
    else:
        # Domains without a dedicated agent (greeting, out_of_scope, multi_agent)
        # Return empty response - these are evaluated differently
        return (
            "",
            [],
            "",
            0.0,
            f"No isolated agent for domain: {domain}",
            build_usage_payload({}, model_id=response_model_id, call_count=0),
        )

    # Override the agent's LLM config dynamically
    try:
        agent.init_llm(
            provider=config["provider"],
            model=config["model"],
            temperature=config["temperature"]
        )
    except Exception as e:
        return (
            f"LLM Setup Error: {str(e)}",
            [],
            "",
            0.0,
            f"Setup Error: {str(e)}",
            build_usage_payload({}, model_id=response_model_id, call_count=0),
        )
    
    # Track execution
    start_time = time.time()
    tools_called = []
    retrieved_context_blocks = []
    final_response = ""
    error = None
    agent.reset_llm_usage_tracking()

    original_tool_invokes = []
    try:
        for tool in getattr(agent, "tools", []):
            original_invoke = getattr(tool, "invoke", None)
            if not callable(original_invoke):
                continue

            tool_name = getattr(tool, "name", "unknown_tool")

            def _make_invoke_wrapper(name, invoke_fn):
                def _wrapped(tool_input):
                    result = invoke_fn(tool_input)
                    tools_called.append(name)
                    retrieved_context_blocks.append(f"[{name}] returned:\n{result}")
                    return result

                return _wrapped

            original_tool_invokes.append((tool, original_invoke))
            object.__setattr__(tool, "invoke", _make_invoke_wrapper(tool_name, original_invoke))

        final_response = str(agent.invoke(query))
    except Exception as e:
        error = str(e)
        final_response = f"Execution Error: {error}"
    finally:
        for tool, original_invoke in original_tool_invokes:
            object.__setattr__(tool, "invoke", original_invoke)

    latency = time.time() - start_time
    retrieved_context_str = "\n---\n".join(retrieved_context_blocks)
    response_usage = build_usage_payload(
        agent.get_llm_usage_summary(),
        model_id=response_model_id,
    )
    return final_response, tools_called, retrieved_context_str, latency, error, response_usage


def run_benchmark(
    limit: int = None,
    models: list = MODELS_TO_TEST,
    pricing_by_model: dict | None = None,
):
    """
    Execute the academic benchmark and save the results JSON.

    Args:
        limit: Maximum number of shared ground-truth queries per response model.
        models: Response-model matrix to evaluate.
        pricing_by_model: Optional pricing catalog keyed by model name or
            ``provider::model`` with ``input`` and ``output`` prices in USD per
            million tokens. When provided, each record stores organized
            response/evaluation/combined token counts and costs.
    """
    print("=" * 60)
    print(f"STARTING ACADEMIC LISBOA BENCHMARK (LIMIT={limit})")
    print("=" * 60)
    
    groundtruth_queries = load_groundtruth_queries(GROUNDTRUTH_QUERIES_PATH)
    if limit:
        groundtruth_queries = select_balanced_subset(
            groundtruth_queries,
            limit,
            group_key="domain",
        )

    # Initialize the judge
    try:
        judge = LLMJudge()
    except ValueError as e:
        print(f"FAILED TO INIT JUDGE: {e}")
        return

    # TEST: The evaluator model is selected in eval/llm_judge.py via
    # EVAL_JUDGE_PROVIDER / EVAL_JUDGE_MODEL_NAME, not in the response-model matrix.
    evaluation_model_manifest = build_model_manifest(
        judge.provider,
        judge.model_name,
        0.0,
    )
    evaluation_model_id = evaluation_model_manifest["model_id"]

    response_model_manifests = [
        build_model_manifest(
            model_config["provider"],
            model_config["model"],
            model_config.get("temperature"),
        )
        for model_config in models
    ]

    results = []
    pricing_catalog, _ = split_pricing_config(pricing_by_model)

    for model_config in models:
        response_model_manifest = build_model_manifest(
            model_config["provider"],
            model_config["model"],
            model_config.get("temperature"),
        )
        response_model_id = response_model_manifest["model_id"]
        print(f"\nEvaluating Model: {response_model_id}")
        
        model_consecutive_errors = 0
        
        for idx, item in enumerate(groundtruth_queries):
            print(f"  [{idx+1}/{len(groundtruth_queries)}] [{item['domain'].upper()}] {item['query'][:50]}...")
            
            response, tools, retrieved_context, latency, error, response_usage = run_isolated_agent(
                domain=item['domain'], 
                query=item['query'], 
                config=model_config
            )
            response_cost = build_cost_payload(
                response_usage,
                pricing_by_model,
                model_id=response_model_id,
            )
            empty_evaluation_usage = build_usage_payload(
                {},
                model_id=evaluation_model_id,
                call_count=0,
            )
            empty_evaluation_cost = build_cost_payload(
                empty_evaluation_usage,
                pricing_by_model,
                model_id=evaluation_model_id,
            )

            # Skip judge if model failed to save tokens/API costs
            if error is not None:
                model_consecutive_errors += 1
                judge_result = {
                    "composite_score": None,
                    "reasoning": f"Model failed to generate response. Skipped judge. Error: {error}",
                    "factual_accuracy": None,
                    "tool_usage": None,
                    "completeness": None,
                    "relevance": None,
                    "response_quality": None,
                    "evaluation_usage": empty_evaluation_usage,
                    "evaluation_cost_usd": empty_evaluation_cost,
                }
            else:
                model_consecutive_errors = 0
                try:
                    # Judge response
                    judge_result = judge.evaluate(
                        query=item['query'],
                        expected_facts=item.get('expected_facts', []),
                        expected_tools=item.get('expected_tools', []),
                        actual_tools=tools,
                        retrieved_context=retrieved_context,
                        response=response,
                        pricing_by_model=pricing_by_model,
                    )
                except Exception as e:
                    print(f"          → [JUDGE ERROR] {e}")
                    judge_result = {
                        "composite_score": None,
                        "reasoning": f"Judge API error: {e}",
                        "factual_accuracy": None,
                        "tool_usage": None,
                        "completeness": None,
                        "relevance": None,
                        "response_quality": None,
                        "evaluation_usage": empty_evaluation_usage,
                        "evaluation_cost_usd": empty_evaluation_cost,
                    }

            evaluation_usage = judge_result.get("evaluation_usage", empty_evaluation_usage)
            evaluation_cost = judge_result.get("evaluation_cost_usd", empty_evaluation_cost)
            judge_scores = {
                key: value
                for key, value in judge_result.items()
                if key not in {"evaluation_usage", "evaluation_cost_usd"}
            }
            combined_usage = combine_usage_payloads([response_usage, evaluation_usage])
            combined_cost = combine_cost_payloads([response_cost, evaluation_cost])

            # Deterministic tool metrics
            tool_metrics = compute_tool_metrics(
                expected=item.get("expected_tools", []),
                actual=tools,
            )
            heuristics = None if error is not None else run_all_heuristics(
                response=response,
                expected_language=item.get("language", "en"),
            )
            error_type = categorize_error(error)

            record = {
                "id": item["id"],
                "domain": item["domain"],
                "query": item["query"],
                "language": item.get("language", "en"),
                "edge_case": item.get("edge_case", False),
                "edge_type": item.get("edge_type", None),
                "expected_behavior": item.get("expected_behavior"),
                "response_model": response_model_id,
                "response_model_config": deepcopy(response_model_manifest),
                "evaluation_model": evaluation_model_id,
                "evaluation_model_config": deepcopy(evaluation_model_manifest),
                "latency_s": round(latency, 2),
                "error": error,
                "error_type": error_type,
                "response": response,
                "tools_used": tools,
                "expected_tools": item.get("expected_tools", []),
                "expected_facts": item.get("expected_facts", []),
                "retrieved_context": retrieved_context,
                "scores": judge_scores,
                "response_usage": response_usage,
                "response_cost_usd": response_cost,
                "evaluation_usage": evaluation_usage,
                "evaluation_cost_usd": evaluation_cost,
                "combined_usage": combined_usage,
                "combined_cost_usd": combined_cost,
                "tool_metrics": tool_metrics,
                "heuristics": heuristics,
                "sla_met": latency <= SLA_THRESHOLDS.get(item["domain"], 15.0),
            }
            results.append(record)

            score_display = f"{judge_scores['composite_score']:.2f}/5.0" if judge_scores['composite_score'] is not None else "N/A"
            print(f"          -> Score: {score_display} | Latency: {latency:.2f}s | Reason: {judge_scores['reasoning']}")
            
            if error is not None and ("Setup Error" in error or model_consecutive_errors >= 2):
                print(f"          -> ABORTING {response_model_id}: Model is failing continuously. Saving costs.")
                break

    # Build aggregate summary
    summary = _build_summary(results)

    # Save Results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = build_results_output_path("benchmark", "benchmark_results", timestamp)
    with open(output_path, "w", encoding="utf-8") as f:
        benchmark_metadata = build_run_metadata(
            GROUNDTRUTH_QUERIES_PATH,
            groundtruth_queries,
            response_models=[manifest["model_id"] for manifest in response_model_manifests],
            evaluation_model=evaluation_model_id,
            extra={
                "response_model_configs": response_model_manifests,
                "evaluation_model_config": evaluation_model_manifest,
                "benchmark_domains": list(BENCHMARK_DOMAINS),
                "timestamp": datetime.now().isoformat(),
                "real_services": True,
                "pricing_model_count": len(pricing_catalog),
                "output_directory": str(output_path.parent),
                **get_pricing_metadata(pricing_by_model),
            },
        )
        json.dump({
            "benchmark_metadata": benchmark_metadata,
            "summary": summary,
            "benchmark_results": results,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nBenchmark complete. Results saved to {output_path}")


def _build_summary(results: list) -> dict:
    """Builds aggregate summary statistics from benchmark results."""
    if not results:
        return {}

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

    overall_response_usage = combine_usage_payloads([r.get("response_usage", {}) for r in results])
    overall_evaluation_usage = combine_usage_payloads([r.get("evaluation_usage", {}) for r in results])
    overall_combined_usage = combine_usage_payloads([r.get("combined_usage", {}) for r in results])
    overall_response_cost = combine_cost_payloads([r.get("response_cost_usd", {}) for r in results])
    overall_evaluation_cost = combine_cost_payloads([r.get("evaluation_cost_usd", {}) for r in results])
    overall_combined_cost = combine_cost_payloads([r.get("combined_cost_usd", {}) for r in results])
    
    # Overall averages
    scores = [r["scores"]["composite_score"] for r in results if r["scores"]["composite_score"] is not None]
    latencies = [r["latency_s"] for r in results if r["error"] is None]
    errors = [r for r in results if r["error"] is not None]
    
    summary = {
        "overall": {
            "total_evaluated": len(results),
            "total_errors": len(errors),
            "error_categories": summarize_error_categories(results),
            "avg_composite_score": round(sum(scores) / len(scores), 3) if scores else 0,
            "avg_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else 0,
            "avg_tool_f1": round(
                sum(r.get("tool_metrics", {}).get("tool_f1", 0) for r in results) / len(results), 3
            ) if results else 0,
            "heuristics_pass_rate": round(
                sum(1 for r in results if (r.get("heuristics") or {}).get("overall_pass", False)) / len(results), 3
            ) if results else 0,
            "sla_compliance": round(
                sum(1 for r in results if r.get("sla_met", False)) / len(results), 3
            ) if results else 0,
            "response_usage": _compact_usage(overall_response_usage),
            "evaluation_usage": _compact_usage(overall_evaluation_usage),
            "combined_usage": _compact_usage(overall_combined_usage),
            "response_cost_usd": _compact_cost(overall_response_cost),
            "evaluation_cost_usd": _compact_cost(overall_evaluation_cost),
            "combined_cost_usd": _compact_cost(overall_combined_cost),
        },
        "per_domain": {},
        "per_response_model": {},
    }
    
    # Per-domain breakdown
    domains = set(r["domain"] for r in results)
    for domain in sorted(domains):
        domain_results = [r for r in results if r["domain"] == domain]
        domain_scores = [r["scores"]["composite_score"] for r in domain_results if r["scores"]["composite_score"] is not None]
        factual_scores = [r["scores"]["factual_accuracy"] for r in domain_results if r["scores"].get("factual_accuracy") is not None]
        domain_response_usage = combine_usage_payloads([r.get("response_usage", {}) for r in domain_results])
        domain_evaluation_usage = combine_usage_payloads([r.get("evaluation_usage", {}) for r in domain_results])
        domain_combined_usage = combine_usage_payloads([r.get("combined_usage", {}) for r in domain_results])
        domain_response_cost = combine_cost_payloads([r.get("response_cost_usd", {}) for r in domain_results])
        domain_evaluation_cost = combine_cost_payloads([r.get("evaluation_cost_usd", {}) for r in domain_results])
        domain_combined_cost = combine_cost_payloads([r.get("combined_cost_usd", {}) for r in domain_results])
        
        summary["per_domain"][domain] = {
            "count": len(domain_results),
            "errors": sum(1 for r in domain_results if r["error"] is not None),
            "avg_composite_score": round(sum(domain_scores) / len(domain_scores), 3) if domain_scores else 0,
            "avg_factual_accuracy": round(sum(factual_scores) / len(factual_scores), 3) if factual_scores else 0,
            "avg_tool_f1": round(
                sum(r.get("tool_metrics", {}).get("tool_f1", 0) for r in domain_results) / max(1, len(domain_results)), 3
            ),
            "heuristics_pass_rate": round(
                sum(1 for r in domain_results if (r.get("heuristics") or {}).get("overall_pass", False)) / max(1, len(domain_results)),
                3,
            ),
            "response_usage": _compact_usage(domain_response_usage),
            "evaluation_usage": _compact_usage(domain_evaluation_usage),
            "combined_usage": _compact_usage(domain_combined_usage),
            "response_cost_usd": _compact_cost(domain_response_cost),
            "evaluation_cost_usd": _compact_cost(domain_evaluation_cost),
            "combined_cost_usd": _compact_cost(domain_combined_cost),
        }
    
    # Per-model breakdown
    response_models = set(r["response_model"] for r in results)
    for model in sorted(response_models):
        model_results = [r for r in results if r["response_model"] == model]
        model_scores = [r["scores"]["composite_score"] for r in model_results if r["scores"]["composite_score"] is not None]
        model_response_usage = combine_usage_payloads([r.get("response_usage", {}) for r in model_results])
        model_evaluation_usage = combine_usage_payloads([r.get("evaluation_usage", {}) for r in model_results])
        model_combined_usage = combine_usage_payloads([r.get("combined_usage", {}) for r in model_results])
        model_response_cost = combine_cost_payloads([r.get("response_cost_usd", {}) for r in model_results])
        model_evaluation_cost = combine_cost_payloads([r.get("evaluation_cost_usd", {}) for r in model_results])
        model_combined_cost = combine_cost_payloads([r.get("combined_cost_usd", {}) for r in model_results])
        summary["per_response_model"][model] = {
            "count": len(model_results),
            "avg_composite_score": round(sum(model_scores) / len(model_scores), 3) if model_scores else 0,
            "errors": sum(1 for r in model_results if r["error"] is not None),
            "error_categories": summarize_error_categories(model_results),
            "heuristics_pass_rate": round(
                sum(1 for r in model_results if (r.get("heuristics") or {}).get("overall_pass", False)) / max(1, len(model_results)),
                3,
            ),
            "response_usage": _compact_usage(model_response_usage),
            "evaluation_usage": _compact_usage(model_evaluation_usage),
            "combined_usage": _compact_usage(model_combined_usage),
            "response_cost_usd": _compact_cost(model_response_cost),
            "evaluation_cost_usd": _compact_cost(model_evaluation_cost),
            "combined_cost_usd": _compact_cost(model_combined_cost),
        }
    
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run the academic LISBOA benchmark over isolated worker agents",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max queries to run per model")
    parser.add_argument("--mode", type=str, choices=["run_test", "full"], default="full", help="Mode: run_test (limit=5) or full (all dataset)")
    args = parser.parse_args()
    
    limit = 5 if args.mode == "run_test" else args.limit
    
    run_benchmark(limit=limit)
