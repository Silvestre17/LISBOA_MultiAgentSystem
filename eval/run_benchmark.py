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
import os
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from agent.agents.researcher_agent import ResearcherAgent
from agent.agents.transport_agent import TransportAgent
from agent.agents.weather_agent import WeatherAgent
from agent.utils.langsmith_tracing import (
    LANGSMITH_AVAILABLE,
    get_langsmith_scoped_project_name,
    get_langsmith_tracing_status,
    get_last_langsmith_runtime_failure,
    tracing_project_override,
)
from eval.llm_judge import LLMJudge
from eval.runtime_utils import (
    aggregate_judge_runs,
    build_cost_payload,
    build_model_id,
    build_model_manifest,
    build_multi_judge_manifest,
    build_results_output_path,
    build_run_metadata,
    build_usage_payload,
    categorize_error,
    combine_cost_payloads,
    combine_usage_payloads,
    compute_tool_metrics,
    get_pricing_metadata,
    load_pricing_catalog,
    parse_model_spec,
    resolve_model_specs,
    select_balanced_subset,
    split_pricing_config,
    summarize_error_categories,
    write_json_artifact,
)
from eval.validators.response_heuristics import run_all_heuristics

GROUNDTRUTH_QUERIES_PATH = Path(__file__).with_name("evaluation_groundtruth_queries.json")
BENCHMARK_DOMAINS = ("weather", "transport", "researcher")
BENCHMARK_LANGSMITH_PROJECT_ENV = "LISBOA_LANGSMITH_BENCHMARK_PROJECT"
BENCHMARK_LANGSMITH_SCOPE_LABEL = "Benchmark"
SUPPORTED_MODEL_PROVIDERS = {"azure", "openai", "lmstudio"}


# TEST: Define the per-agent response-model benchmark matrix here.
# These defaults mirror the active evaluation setup on this machine:
# Azure GPT-5-mini for the closed model and Azure Kimi-K2.5 for the
# open-model comparison profile.
MODELS_TO_TEST = [
    # TEST: proprietary model 1
    {"provider": "azure", "model": "gpt-5-mini", "temperature": 0.0},
    # TEST: open-model profile served through Azure
    {"provider": "azure", "model": "Kimi-K2.5", "temperature": 0.0},
]
DEFAULT_JUDGE_MODELS = deepcopy(MODELS_TO_TEST)

# Latency SLA thresholds (seconds) per domain
SLA_THRESHOLDS = {
    "weather": 10.0,
    "transport": 15.0,
    "researcher": 20.0,
    "multi_agent": 30.0,
    "greeting": 5.0,
    "out_of_scope": 5.0,
}


def resolve_groundtruth_path(dataset_path: str | Path | None = None) -> Path:
    """Resolve an optional ground-truth dataset path relative to the repository root."""
    if dataset_path is None:
        return GROUNDTRUTH_QUERIES_PATH

    candidate = Path(dataset_path)
    if candidate.is_absolute():
        return candidate

    repo_root = Path(__file__).resolve().parent.parent
    repo_relative = repo_root / candidate
    if repo_relative.exists():
        return repo_relative

    return (Path.cwd() / candidate).resolve()


def resolve_pricing_catalog(pricing_by_model: dict | None = None) -> dict | None:
    """Return the active pricing catalog, defaulting to the checked-in repository snapshot."""
    if pricing_by_model is not None:
        return pricing_by_model
    return load_pricing_catalog()


def parse_response_model_spec(
    model_spec: str,
    *,
    temperature: float | None = None,
) -> dict[str, str | float]:
    """Parse a CLI response-model spec into the benchmark matrix format."""
    return parse_model_spec(
        model_spec,
        temperature=temperature,
        supported_providers=SUPPORTED_MODEL_PROVIDERS,
    )


def resolve_response_models(
    model_specs: list[str] | None = None,
    *,
    temperature: float | None = None,
) -> list[dict[str, str | float]]:
    """Return the benchmark response-model matrix after optional CLI overrides."""
    return resolve_model_specs(
        MODELS_TO_TEST,
        model_specs,
        temperature=temperature,
        supported_providers=SUPPORTED_MODEL_PROVIDERS,
    )


def resolve_judge_models(
    judge_model_specs: list[str] | None = None,
    *,
    provider: str | None = None,
    model_name: str | None = None,
) -> list[dict[str, str | float]]:
    """Resolve the benchmark judge matrix after CLI and environment overrides."""
    env_judge_specs = [
        spec.strip()
        for spec in str(os.getenv("EVAL_JUDGE_MODEL_SPECS", "") or "").split(",")
        if spec.strip()
    ]

    if judge_model_specs:
        return resolve_model_specs(
            DEFAULT_JUDGE_MODELS,
            judge_model_specs,
            temperature=0.0,
            supported_providers=SUPPORTED_MODEL_PROVIDERS,
        )
    if env_judge_specs:
        return resolve_model_specs(
            DEFAULT_JUDGE_MODELS,
            env_judge_specs,
            temperature=0.0,
            supported_providers=SUPPORTED_MODEL_PROVIDERS,
        )

    if provider or model_name:
        resolved_provider = str(provider or os.getenv("EVAL_JUDGE_PROVIDER", DEFAULT_JUDGE_MODELS[0]["provider"]))
        resolved_model_name = str(model_name or os.getenv("EVAL_JUDGE_MODEL_NAME", DEFAULT_JUDGE_MODELS[0]["model"]))
        return [
            parse_model_spec(
                f"{resolved_provider}::{resolved_model_name}",
                temperature=0.0,
                supported_providers=SUPPORTED_MODEL_PROVIDERS,
            )
        ]

    return deepcopy(DEFAULT_JUDGE_MODELS)


def _build_empty_judge_payload(
    judge_model_manifest: dict[str, object],
    pricing_by_model: dict | None,
    *,
    call_count: int = 0,
) -> tuple[dict, dict]:
    """Return empty-but-shaped usage and cost payloads for a judge model."""
    model_id = str(judge_model_manifest.get("model_id") or "")
    empty_usage = build_usage_payload({}, model_id=model_id, call_count=call_count)
    empty_cost = build_cost_payload(
        empty_usage,
        pricing_by_model,
        model_id=model_id,
    )
    return empty_usage, empty_cost


def _evaluate_with_judges(
    *,
    judges: list[LLMJudge],
    judge_model_manifests: list[dict[str, object]],
    query: str,
    expected_facts: list[str],
    expected_tools: list[str],
    actual_tools: list[str],
    retrieved_context: str,
    response: str,
    response_error: str | None,
    pricing_by_model: dict | None,
) -> tuple[list[dict], dict[str, object]]:
    """Evaluate one response with every configured judge and average the scores."""
    judge_runs: list[dict] = []

    for judge, judge_model_manifest in zip(judges, judge_model_manifests):
        empty_eval_usage, empty_eval_cost = _build_empty_judge_payload(
            judge_model_manifest,
            pricing_by_model,
            call_count=0,
        )

        judge_scores = {
            "composite_score": None,
            "reasoning": "",
            "factual_accuracy": None,
            "tool_usage": None,
            "completeness": None,
            "relevance": None,
            "response_quality": None,
        }
        evaluation_usage = empty_eval_usage
        evaluation_cost = empty_eval_cost
        judge_error = None

        if response_error is not None:
            judge_error = f"Generator failed before judgment. Error: {response_error}"
            judge_scores["reasoning"] = f"Judge skipped because the generator failed: {response_error}"
        else:
            try:
                judge_result = judge.evaluate(
                    query=query,
                    expected_facts=expected_facts,
                    expected_tools=expected_tools,
                    actual_tools=actual_tools,
                    retrieved_context=retrieved_context,
                    response=response,
                    pricing_by_model=pricing_by_model,
                )
                evaluation_usage = judge_result.get("evaluation_usage", empty_eval_usage)
                evaluation_cost = judge_result.get("evaluation_cost_usd", empty_eval_cost)
                judge_scores = {
                    key: value
                    for key, value in judge_result.items()
                    if key not in {"evaluation_usage", "evaluation_cost_usd"}
                }
                if str(judge_scores.get("reasoning") or "").startswith("Judge Failed"):
                    judge_error = str(judge_scores.get("reasoning"))
            except Exception as exc:
                judge_error = str(exc)
                judge_scores["reasoning"] = f"Judge API error: {exc}"

        judge_runs.append(
            {
                "judge_model": str(judge_model_manifest["model_id"]),
                "judge_model_config": deepcopy(judge_model_manifest),
                "scores": judge_scores,
                "evaluation_usage": evaluation_usage,
                "evaluation_cost_usd": evaluation_cost,
                "error": judge_error,
                "error_type": categorize_error(judge_error),
            }
        )

    aggregated = aggregate_judge_runs(judge_runs)
    return judge_runs, aggregated


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
    judge_model_specs: list[str] | None = None,
    judge_provider: str | None = None,
    judge_model: str | None = None,
    groundtruth_path: str | Path | None = None,
    output_prefix: str = "benchmark_results",
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
        judge_model_specs: Optional repeatable list of judge model specs.
        judge_provider: Optional provider override for a single evaluation judge.
        judge_model: Optional model override for a single evaluation judge.
    """
    benchmark_langsmith_project = get_langsmith_scoped_project_name(
        BENCHMARK_LANGSMITH_SCOPE_LABEL,
        env_name=BENCHMARK_LANGSMITH_PROJECT_ENV,
    )
    langsmith_status = get_langsmith_tracing_status()
    if LANGSMITH_AVAILABLE:
        print(
            f"[LangSmith] Benchmark traces will be saved to project: {benchmark_langsmith_project}"
        )
    else:
        print(
            "[LangSmith] Benchmark tracing is inactive. "
            f"{langsmith_status.get('reason', 'LangSmith tracing is disabled')} "
            "Set LANGSMITH_TRACING=true with valid credentials to save these runs to project: "
            f"{benchmark_langsmith_project}"
        )

    with tracing_project_override(benchmark_langsmith_project):
        print("=" * 60)
        print(f"STARTING ACADEMIC LISBOA BENCHMARK (LIMIT={limit})")
        print("=" * 60)

        resolved_groundtruth_path = resolve_groundtruth_path(groundtruth_path)
        pricing_by_model = resolve_pricing_catalog(pricing_by_model)
        groundtruth_queries = load_groundtruth_queries(resolved_groundtruth_path)
        if limit:
            groundtruth_queries = select_balanced_subset(
                groundtruth_queries,
                limit,
                group_key="domain",
            )

        judge_configs = resolve_judge_models(
            judge_model_specs,
            provider=judge_provider,
            model_name=judge_model,
        )

        # Initialize the judges
        try:
            judges = [
                LLMJudge(
                    provider=str(judge_config["provider"]),
                    model_name=str(judge_config["model"]),
                )
                for judge_config in judge_configs
            ]
        except ValueError as e:
            print(f"FAILED TO INIT JUDGE: {e}")
            return

        judge_model_manifests = [
            build_model_manifest(
                str(judge_config["provider"]),
                str(judge_config["model"]),
                float(judge_config.get("temperature", 0.0) or 0.0),
            )
            for judge_config in judge_configs
        ]
        evaluation_model_manifest = build_multi_judge_manifest(judge_model_manifests)
        evaluation_model_id = str(evaluation_model_manifest["model_id"])

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
                if error is not None:
                    model_consecutive_errors += 1
                else:
                    model_consecutive_errors = 0

                judge_runs, aggregated_judges = _evaluate_with_judges(
                    judges=judges,
                    judge_model_manifests=judge_model_manifests,
                    query=item['query'],
                    expected_facts=item.get('expected_facts', []),
                    expected_tools=item.get('expected_tools', []),
                    actual_tools=tools,
                    retrieved_context=retrieved_context,
                    response=response,
                    response_error=error,
                    pricing_by_model=pricing_by_model,
                )

                evaluation_usage = aggregated_judges.get("evaluation_usage", build_usage_payload({}, model_id=evaluation_model_id, call_count=0))
                evaluation_cost = aggregated_judges.get("evaluation_cost_usd", build_cost_payload(build_usage_payload({}, model_id=evaluation_model_id, call_count=0), pricing_by_model, model_id=evaluation_model_id))
                judge_scores = dict(aggregated_judges.get("scores", {}))
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
                    "agents_used": [item["domain"]],
                    "response_model": response_model_id,
                    "response_model_config": deepcopy(response_model_manifest),
                    "evaluation_model": evaluation_model_id,
                    "evaluation_model_config": deepcopy(evaluation_model_manifest),
                    "evaluation_models": list(evaluation_model_manifest.get("judge_models", [])),
                    "latency_s": round(latency, 2),
                    "error": error,
                    "error_type": error_type,
                    "response": response,
                    "tools_used": tools,
                    "expected_tools": item.get("expected_tools", []),
                    "expected_facts": item.get("expected_facts", []),
                    "retrieved_context": retrieved_context,
                    "scores": judge_scores,
                    "judge_runs": judge_runs,
                    "scores_by_judge": aggregated_judges.get("scores_by_judge", {}),
                    "judge_summary": aggregated_judges.get("judge_summary", {}),
                    "response_usage": response_usage,
                    "response_cost_usd": response_cost,
                    "agent_usage": {item["domain"]: deepcopy(response_usage)},
                    "agent_costs": {item["domain"]: deepcopy(response_cost)},
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
        output_path = build_results_output_path("benchmark", output_prefix, timestamp)
        benchmark_metadata = build_run_metadata(
            resolved_groundtruth_path,
            groundtruth_queries,
            response_models=[manifest["model_id"] for manifest in response_model_manifests],
            evaluation_model=evaluation_model_id,
            extra={
                "response_model_configs": response_model_manifests,
                "evaluation_model_config": evaluation_model_manifest,
                "benchmark_domains": list(BENCHMARK_DOMAINS),
                "langsmith_enabled": LANGSMITH_AVAILABLE,
                "langsmith_project": benchmark_langsmith_project,
                "timestamp": datetime.now().isoformat(),
                "real_services": True,
                "evaluation_models": list(evaluation_model_manifest.get("judge_models", [])),
                "judge_model_configs": judge_model_manifests,
                "pricing_model_count": len(pricing_catalog),
                "output_directory": str(output_path.parent),
                **get_pricing_metadata(pricing_by_model),
            },
        )
        write_json_artifact(
            {
                "benchmark_metadata": benchmark_metadata,
                "summary": summary,
                "benchmark_results": results,
            },
            output_path,
        )

        print(f"\nBenchmark complete. Results saved to {output_path}")
        runtime_failure = get_last_langsmith_runtime_failure()
        if runtime_failure:
            print(
                "[LangSmith] Latest persistence status: "
                f"{runtime_failure.get('persistence_state', 'failed_remote')} - "
                f"{runtime_failure.get('message', '')}"
            )


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
            "model_id": payload.get("model_id"),
            "pricing_lookup_key": payload.get("pricing_lookup_key"),
            "pricing_found": bool(payload.get("pricing_found", False)),
            "pricing_complete": bool(payload.get("pricing_complete", False)),
            "tokens": payload.get(
                "tokens",
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            ),
            "input_per_million_usd": None if payload.get("input_per_million_usd") is None else float(payload.get("input_per_million_usd") or 0.0),
            "output_per_million_usd": None if payload.get("output_per_million_usd") is None else float(payload.get("output_per_million_usd") or 0.0),
            "cached_input_per_million_usd": None if payload.get("cached_input_per_million_usd") is None else float(payload.get("cached_input_per_million_usd") or 0.0),
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
    parser.add_argument(
        "--response-model",
        action="append",
        dest="response_models",
        help="Repeatable response-model override in provider::model format, for example azure::gpt-5-mini.",
    )
    parser.add_argument(
        "--response-temperature",
        type=float,
        default=None,
        help="Optional temperature applied to the selected benchmark response models.",
    )
    parser.add_argument(
        "--judge-model-spec",
        action="append",
        dest="judge_model_specs",
        help="Repeatable evaluation-judge model spec in provider::model format, for example lmstudio::qwen/qwen3.5-9b.",
    )
    parser.add_argument(
        "--judge-provider",
        type=str,
        default=None,
        help="Optional provider override for a single evaluation judge when --judge-model-spec is not used.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Optional model override for a single evaluation judge when --judge-model-spec is not used.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset path, for example eval/evaluation_groundtruth_queries_demo.json.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="benchmark_results",
        help="Output filename prefix inside eval/results/benchmark/.",
    )
    args = parser.parse_args()

    limit = 5 if args.mode == "run_test" else args.limit
    selected_models = resolve_response_models(
        args.response_models,
        temperature=args.response_temperature,
    )

    run_benchmark(
        limit=limit,
        models=selected_models,
        judge_model_specs=args.judge_model_specs,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        groundtruth_path=args.dataset,
        output_prefix=args.output_prefix,
    )
