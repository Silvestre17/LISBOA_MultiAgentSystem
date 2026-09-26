# ==========================================================================
# Master Thesis - Benchmark Runner
#   - André Filipe Gomes Silvestre, 20240502
#
#   Runs the academic LISBOA benchmark over isolated worker agents
#   (weather, transport, researcher) and writes the JSON artefacts into:
#   eval/results/benchmark/
#
# Usage:
#   > python -m eval.run_benchmark --mode run_test
#       Quick benchmark with 5 dataset entries per response model.
#   > python -m eval.run_benchmark --limit 20
#       Benchmark the first 20 dataset entries per response model.
#   > python -m eval.run_benchmark --mode full
#       Benchmark the full dataset with all configured response models.
#   > python -m eval.run_benchmark --dataset eval/evaluation_groundtruth_queries_demo.json --output-prefix benchmark_results_demo
#       Run the benchmark against an alternate dataset and keep the artefacts separate from the main notebook inputs.
#   > python -m eval.run_benchmark --response-model azure::gpt-5.4-mini --response-model lmstudio::qwen/qwen3.5-9b
#       Override the benchmark response-model set with one or more explicit provider::model identifiers.
#   > python -m eval.run_benchmark --judge-model-spec openai::gpt-5.4-mini
#       Override the evaluation judge with one or more explicit provider::model identifiers.
#   > python -m eval.run_benchmark --dataset eval/evaluation_groundtruth_queries_paper_eval.json --output-prefix benchmark_final
#       Paper evaluation run; each finished response is appended to a .partial.jsonl checkpoint.
#   > python -m eval.run_benchmark --dataset <same dataset> --resume eval/results/benchmark/<prefix>_<timestamp>.partial.jsonl
#       Resume an interrupted run from its checkpoint (add --retry-errors to rerun failed responses).
# ==========================================================================

import json
import os
import sys
import time
from statistics import median
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
    append_checkpoint_line,
    build_reference_context,
    build_cost_payload,
    build_model_id,
    build_model_manifest,
    build_multi_judge_manifest,
    build_results_output_path,
    build_run_metadata,
    build_run_provenance,
    build_usage_payload,
    categorize_error,
    collect_runtime_data_provenance,
    combine_cost_payloads,
    combine_usage_payloads,
    compute_dataset_fingerprint,
    compute_tool_metrics,
    get_pricing_metadata,
    load_pricing_catalog,
    parse_model_spec,
    resolve_model_specs,
    select_balanced_subset,
    split_pricing_config,
    summarize_error_categories,
    warm_up_runtime_resources,
    write_json_artifact,
)
from eval.validators.response_heuristics import run_all_heuristics

GROUNDTRUTH_QUERIES_PATH = Path(__file__).with_name("evaluation_groundtruth_queries.json")
BENCHMARK_DOMAINS = ("weather", "transport", "researcher")
BENCHMARK_LANGSMITH_PROJECT_ENV = "LISBOA_LANGSMITH_BENCHMARK_PROJECT"
BENCHMARK_LANGSMITH_SCOPE_LABEL = "Benchmark"
SUPPORTED_MODEL_PROVIDERS = {"azure", "openai", "lmstudio"}
DEFAULT_OUTPUT_PREFIX = "benchmark_results"
CHECKPOINT_SUFFIX = ".partial.jsonl"


def _average_tool_f1(records: list[dict]) -> float:
    """Return the mean deterministic tool F1 over rows where tool scoring applies."""
    values = []
    for record in records:
        metrics = record.get("tool_metrics") or {}
        value = metrics.get("tool_f1")
        if value is not None and metrics.get("tool_metric_scored", True):
            values.append(float(value))
    return round(sum(values) / len(values), 3) if values else 0


# TEST: Define the per-agent response-model benchmark matrix here.
# These defaults mirror the active evaluation setup on this machine:
# Azure GPT-5.4-mini for the closed model and Azure Kimi-K2.5 for the
# open-model comparison profile.
MODELS_TO_TEST = [
    # TEST: proprietary model 1
    {"provider": "azure", "model": "gpt-5.4-mini", "temperature": 0.0},
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


def _describe_response_telemetry(
    *,
    response_usage: dict,
    tools_used: list[str],
    error: str | None,
) -> dict[str, str | None]:
    """Describe whether response-side usage metrics were captured or not applicable."""
    call_count = int(response_usage.get("call_count", 0) or 0)
    tokens = response_usage.get("tokens", {}) if isinstance(response_usage, dict) else {}
    total_tokens = int(tokens.get("total_tokens", 0) or 0)

    if error:
        return {
            "response_generation_mode": "failed",
            "response_usage_status": "unavailable_due_to_error",
            "response_usage_note": "Response generation failed before response-side usage telemetry could be finalized.",
        }
    if call_count > 0 or total_tokens > 0:
        return {
            "response_generation_mode": "llm",
            "response_usage_status": "captured",
            "response_usage_note": None,
        }
    if tools_used:
        return {
            "response_generation_mode": "deterministic_tool",
            "response_usage_status": "not_applicable_no_llm",
            "response_usage_note": "The worker answered through a deterministic tool or rule path without an LLM generation call, so response-side tokens and cost are zero by design.",
        }
    return {
        "response_generation_mode": "deterministic_rule",
        "response_usage_status": "not_applicable_no_llm",
        "response_usage_note": "The worker answered through a deterministic non-LLM path, so response-side tokens and cost are zero by design.",
    }


def _record_has_error(record: dict) -> bool:
    """Return whether a stored benchmark response failed or lacks a judge score."""
    if record.get("error") is not None:
        return True
    return any(judge_run.get("error") for judge_run in record.get("judge_runs") or [])


def _prefix_from_checkpoint(checkpoint_path: Path) -> str:
    """Return the output prefix of a ``<prefix>_<YYYYmmdd>_<HHMMSS>.partial.jsonl`` checkpoint."""
    stem = checkpoint_path.name[: -len(CHECKPOINT_SUFFIX)] if checkpoint_path.name.endswith(CHECKPOINT_SUFFIX) else checkpoint_path.stem
    parts = stem.rsplit("_", 2)
    return parts[0] if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit() else DEFAULT_OUTPUT_PREFIX


def _load_checkpoint(checkpoint_path: Path, *, retry_errors: bool = False) -> dict:
    """Read a benchmark checkpoint: sessions and finished responses keyed by (query id, model).

    Args:
        checkpoint_path: ``.partial.jsonl`` file written by an earlier run.
        retry_errors: Drop responses whose generation or judge call failed, so they run again.

    Returns:
        dict: Sessions, finished records, the number dropped for a retry, and the
        query fingerprint, judge list, and system-code fingerprint of the first session.
    """
    sessions: list[dict] = []
    completed: dict[tuple[str, str], dict] = {}
    raw_text = checkpoint_path.read_text(encoding="utf-8")
    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            # An interrupted write can truncate the last line; every earlier line is intact.
            print(f"[Checkpoint] Ignoring unreadable line {line_number} in {checkpoint_path.name}.")
            continue
        if entry.get("type") in {"session", "session_end"}:
            session = {key: value for key, value in entry.items() if key != "provenance"}
            if entry.get("type") == "session":
                git_provenance = (entry.get("provenance") or {}).get("git") or {}
                session["git_commit"] = git_provenance.get("commit")
                session["has_uncommitted_changes"] = git_provenance.get("has_uncommitted_changes")
                session["system_code_sha256"] = git_provenance.get("system_code_sha256")
            sessions.append(session)
        elif entry.get("type") == "result":
            # Later lines win, so a retried response replaces the failed one.
            completed[(str(entry["id"]), str(entry["response_model"]))] = entry["record"]
    if raw_text and not raw_text.endswith("\n"):
        with checkpoint_path.open("a", encoding="utf-8") as handle:
            handle.write("\n")
    retried = 0
    if retry_errors:
        retried = sum(1 for record in completed.values() if _record_has_error(record))
        completed = {key: record for key, record in completed.items() if not _record_has_error(record)}
    first_session = next((session for session in sessions if session.get("type") == "session"), {})
    return {
        "sessions": sessions,
        "completed": completed,
        "retried_errors": retried,
        "groundtruth_fingerprint": first_session.get("groundtruth_fingerprint"),
        "judge_model_ids": first_session.get("judge_model_ids"),
        "system_code_sha256": first_session.get("system_code_sha256"),
    }


def _format_duration(seconds: float) -> str:
    """Render seconds as ``1h05m`` or ``12m30s``."""
    seconds = int(max(seconds, 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"


def _format_record_log(record: dict) -> str:
    """One console line per response: score per judge, latency, cost, tools, and failures."""
    scores = record.get("scores") or {}
    quality = scores.get("composite_score")
    by_judge = [
        f"{judge_id.split('::')[-1]} {float(judge_scores['composite_score']):.2f}"
        for judge_id, judge_scores in (record.get("scores_by_judge") or {}).items()
        if judge_scores.get("composite_score") is not None
    ]
    usage = record.get("response_usage") or {}
    cost = float((record.get("response_cost_usd") or {}).get("total_cost_usd") or 0.0)
    tool_f1 = (record.get("tool_metrics") or {}).get("tool_f1")
    line = (f"          -> QS {quality:.2f}" if quality is not None else "          -> QS n/a") + (
        f" ({' / '.join(by_judge)})" if by_judge else ""
    )
    line += f" | {float(record.get('latency_s') or 0.0):.1f} s"
    line += f" | USD {cost:.4f}" if usage.get("call_count") else " | no model call"
    line += f" | tools {','.join(record.get('tools_used') or []) or 'none'}"
    line += f" | tool F1 {tool_f1:.2f}" if tool_f1 is not None else ""
    if record.get("model_call_mismatches"):
        line += f"\n          MODEL MISMATCH: calls served by {record['model_call_mismatches']}"
    if record.get("error") is not None:
        line += f"\n          RESPONSE ERROR ({record.get('error_type')}): {str(record['error'])[:240]}"
    for judge_run in record.get("judge_runs") or []:
        if judge_run.get("error") and record.get("error") is None:
            line += f"\n          JUDGE ERROR {judge_run.get('judge_model')}: {str(judge_run['error'])[:240]}"
    return line


def _print_run_summary(results: list[dict], model_ids: list[str], query_count: int, checkpoint_path: Path) -> None:
    """Print, per response model, what the paper reports: quality, latency, cost, and failures."""
    print("\n" + "=" * 60 + "\nRUN SUMMARY (response cost only; judge calls excluded)\n" + "=" * 60)
    incomplete = False
    for model_id in model_ids:
        records = [record for record in results if record["response_model"] == model_id]
        if len(records) < query_count:
            incomplete = True
        qualities = [r["scores"]["composite_score"] for r in records if (r.get("scores") or {}).get("composite_score") is not None]
        latencies = sorted(float(r.get("latency_s") or 0.0) for r in records)
        model_calls = [r for r in records if (r.get("response_usage") or {}).get("call_count")]
        costs = [float((r.get("response_cost_usd") or {}).get("total_cost_usd") or 0.0) for r in model_calls]
        print(
            f"\n{model_id}: {len(records)} of {query_count} queries"
            + (f" ({query_count - len(records)} STILL TO RUN)" if len(records) < query_count else "")
        )
        if qualities:
            print(f"  QS {sum(qualities) / len(qualities):.3f} (n={len(qualities)})", end="")
            for domain in BENCHMARK_DOMAINS:
                domain_scores = [r["scores"]["composite_score"] for r in records if r["domain"] == domain and r["scores"].get("composite_score") is not None]
                if domain_scores:
                    print(f" | {domain} {sum(domain_scores) / len(domain_scores):.3f}", end="")
            print()
        if latencies:
            print(f"  latency median {median(latencies):.1f} s, P90 {latencies[int(0.9 * (len(latencies) - 1))]:.1f} s")
        print(
            f"  model-calling responses {len(model_calls)}/{len(records)}"
            + (f" | cost USD {sum(costs):.3f} total, {sum(costs) / len(costs):.4f} per model-calling response" if costs else "")
        )
        print(
            f"  response errors {sum(1 for r in records if r.get('error') is not None)}"
            f" | judge errors {sum(1 for r in records if r.get('error') is None and any(j.get('error') for j in r.get('judge_runs') or []))}"
            f" | responses with calls to another model {sum(1 for r in records if r.get('model_call_mismatches'))}"
        )
    if incomplete:
        print(
            f"\nThis file is INCOMPLETE. Finish it with: --resume \"{checkpoint_path}\" and the same --dataset "
            "(add --retry-errors to rerun failed responses)."
        )


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
    reference_context: str,
    response: str,
    response_error: str | None,
    pricing_by_model: dict | None,
    tool_expectation: str = "strict",
    acceptable_tool_sets: list[list[str]] | None = None,
) -> tuple[list[dict], dict[str, object]]:
    """Evaluate one response with every configured judge and average the scores."""
    judge_runs: list[dict] = []

    for judge, judge_model_manifest in zip(judges, judge_model_manifests, strict=False):
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
                    reference_context=reference_context,
                    response=response,
                    pricing_by_model=pricing_by_model,
                    tool_expectation=tool_expectation,
                    acceptable_tool_sets=acceptable_tool_sets,
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

    # Record the model name the API reports for every call, next to the
    # configured one, so the artefact shows which model actually answered.
    served_model_names: list[str] = []
    original_record_usage = agent._record_llm_usage

    def _record_usage_with_served_model(llm, response):
        raw = response.get("raw") if isinstance(response, dict) else response
        served = (getattr(raw, "response_metadata", None) or {}).get("model_name")
        if served:
            served_model_names.append(str(served))
        return original_record_usage(llm, response)

    object.__setattr__(agent, "_record_llm_usage", _record_usage_with_served_model)

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
    response_usage["api_model_names"] = sorted(set(served_model_names))
    return final_response, tools_called, retrieved_context_str, latency, error, response_usage


def run_benchmark(
    limit: int = None,
    models: list = MODELS_TO_TEST,
    pricing_by_model: dict | None = None,
    judge_model_specs: list[str] | None = None,
    judge_provider: str | None = None,
    judge_model: str | None = None,
    groundtruth_path: str | Path | None = None,
    output_prefix: str | None = None,
    resume_path: str | Path | None = None,
    retry_errors: bool = False,
    query_ids: list[str] | None = None,
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
        groundtruth_path: Optional dataset path; the worker domains are kept.
        output_prefix: File prefix inside ``eval/results/benchmark/``. With
            ``resume_path`` and no prefix, the checkpoint's prefix is kept.
        resume_path: Optional ``.partial.jsonl`` checkpoint of an earlier run with the
            same queries, judges, and system code. Finished responses are reused.
        retry_errors: With ``resume_path``, run again the responses whose generation
            or judge call failed.
        query_ids: Optional query ids to run, in corpus order.
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
        run_started_at = datetime.now()
        run_started_perf = time.perf_counter()
        print("=" * 60)
        print(f"STARTING ACADEMIC LISBOA BENCHMARK (LIMIT={limit})")
        print("=" * 60)

        resolved_groundtruth_path = resolve_groundtruth_path(groundtruth_path)
        pricing_by_model = resolve_pricing_catalog(pricing_by_model)
        groundtruth_queries = load_groundtruth_queries(resolved_groundtruth_path)
        # Evaluated commit, environment, and data snapshots, taken before any query runs.
        run_provenance = build_run_provenance(dataset_path=resolved_groundtruth_path)
        if run_provenance["git"].get("has_uncommitted_changes"):
            print(
                "[Provenance] Warning: code or evaluation files have uncommitted changes, so the stored "
                "commit does not fully describe the evaluated code."
            )
        if query_ids:
            wanted = set(query_ids)
            unknown = sorted(wanted - {item["id"] for item in groundtruth_queries})
            if unknown:
                print(f"[Benchmark] Unknown or non-worker query ids: {unknown}. Aborting.")
                return
            groundtruth_queries = [item for item in groundtruth_queries if item["id"] in wanted]
        if limit:
            groundtruth_queries = select_balanced_subset(
                groundtruth_queries,
                limit,
                group_key="domain",
            )
        groundtruth_fingerprint = compute_dataset_fingerprint(groundtruth_queries)
        print(f"[Benchmark] {len(groundtruth_queries)} worker queries x {len(models)} response models.")

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
        judge_model_ids = [str(manifest["model_id"]) for manifest in judge_model_manifests]

        if resume_path is not None:
            checkpoint_path = Path(resume_path)
            if not checkpoint_path.is_file():
                print(f"[Checkpoint] Resume file not found: {checkpoint_path}")
                return
            checkpoint = _load_checkpoint(checkpoint_path, retry_errors=retry_errors)
            if checkpoint["groundtruth_fingerprint"] not in (None, groundtruth_fingerprint):
                print("[Checkpoint] The query selection differs from the checkpoint. Aborting so that two corpora are never mixed.")
                return
            stored_code = checkpoint.get("system_code_sha256")
            current_code = run_provenance["git"].get("system_code_sha256")
            if stored_code and current_code and stored_code != current_code:
                print("[Checkpoint] The system code differs from the code of the checkpoint. Aborting so that one run evaluates one version.")
                return
            if checkpoint.get("judge_model_ids") not in (None, judge_model_ids):
                print(f"[Checkpoint] The judges must match the checkpoint ({checkpoint['judge_model_ids']}). Aborting.")
                return
            output_prefix = output_prefix or _prefix_from_checkpoint(checkpoint_path)
            print(
                f"[Checkpoint] Resuming {checkpoint_path.name}: {len(checkpoint['completed'])} responses reused"
                + (f", {checkpoint['retried_errors']} failed ones to run again" if retry_errors else "")
                + "."
            )
        else:
            output_prefix = output_prefix or DEFAULT_OUTPUT_PREFIX
            checkpoint_path = build_results_output_path(
                "benchmark", output_prefix, run_started_at.strftime("%Y%m%d_%H%M%S"), suffix=CHECKPOINT_SUFFIX
            )
            checkpoint = {"sessions": [], "completed": {}, "retried_errors": 0}

        run_options = {
            "limit": limit,
            "query_ids": list(query_ids) if query_ids else None,
            "response_models": [build_model_id(config["provider"], config["model"]) for config in models],
            "judge_model_ids": judge_model_ids,
            "resumed_from_checkpoint": resume_path is not None,
            "retry_errors": bool(retry_errors),
            "checkpoint_path": str(checkpoint_path),
        }
        append_checkpoint_line(
            checkpoint_path,
            {
                "type": "session",
                "started_at": run_started_at.isoformat(),
                "argv": sys.argv[1:],
                "options": run_options,
                "groundtruth_path": str(resolved_groundtruth_path),
                "groundtruth_fingerprint": groundtruth_fingerprint,
                "judge_model_ids": judge_model_ids,
                "provenance": run_provenance,
            },
        )
        print(f"[Checkpoint] Each finished response is appended to {checkpoint_path}")
        print(f"[Checkpoint] If the run stops, resume with: --resume \"{checkpoint_path}\" and the same --dataset")
        responses_to_run = sum(
            1
            for config in models
            for item in groundtruth_queries
            if (item["id"], build_model_id(config["provider"], config["model"])) not in checkpoint["completed"]
        )
        responses_this_session = 0
        warm_up = warm_up_runtime_resources()
        print(
            f"[Warm-up] Vector store and transport data loaded in {warm_up['seconds']} s "
            f"(vector store {'OK' if warm_up['kb_ok'] else 'FAILED'}, transport {'OK' if warm_up['transport_ok'] else 'FAILED'}); "
            "as in the app, this one-time start-up stays out of the timed answers."
        )
        loop_started_perf = time.perf_counter()

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
                stored_record = checkpoint["completed"].get((item["id"], response_model_id))
                if stored_record is not None:
                    results.append(stored_record)
                    continue
                print(f"  [{idx + 1}/{len(groundtruth_queries)}] [{item['id']}] [{item['domain'].upper()}] {item['query'][:70]}")

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
                    reference_context=build_reference_context(
                        expected_facts=item.get('expected_facts', []),
                        expected_behavior=item.get('expected_behavior'),
                    ),
                    response=response,
                    response_error=error,
                    pricing_by_model=pricing_by_model,
                    tool_expectation=item.get("tool_expectation", "strict"),
                    acceptable_tool_sets=item.get("acceptable_tool_sets"),
                )

                evaluation_usage = aggregated_judges.get("evaluation_usage", build_usage_payload({}, model_id=evaluation_model_id, call_count=0))
                evaluation_cost = aggregated_judges.get("evaluation_cost_usd", build_cost_payload(build_usage_payload({}, model_id=evaluation_model_id, call_count=0), pricing_by_model, model_id=evaluation_model_id))
                judge_scores = dict(aggregated_judges.get("scores", {}))
                combined_usage = combine_usage_payloads([response_usage, evaluation_usage])
                combined_cost = combine_cost_payloads([response_cost, evaluation_cost])
                response_telemetry = _describe_response_telemetry(
                    response_usage=response_usage,
                    tools_used=tools,
                    error=error,
                )

                # Deterministic tool metrics
                tool_metrics = compute_tool_metrics(
                    expected=item.get("expected_tools", []),
                    actual=tools,
                    acceptable_tool_sets=item.get("acceptable_tool_sets"),
                    tool_expectation=item.get("tool_expectation", "strict"),
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
                    "reference_context": build_reference_context(
                        expected_facts=item.get('expected_facts', []),
                        expected_behavior=item.get('expected_behavior'),
                    ),
                    "retrieved_context": retrieved_context,
                    "scores": judge_scores,
                    "judge_runs": judge_runs,
                    "scores_by_judge": aggregated_judges.get("scores_by_judge", {}),
                    "judge_summary": aggregated_judges.get("judge_summary", {}),
                    "response_generation_mode": response_telemetry["response_generation_mode"],
                    "response_usage_status": response_telemetry["response_usage_status"],
                    "response_usage_note": response_telemetry["response_usage_note"],
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
                    "api_model_names": response_usage.get("api_model_names", []),
                    # Calls whose configured model is not the response model of this row.
                    "model_call_mismatches": sorted(
                        {
                            str(call.get("model_id"))
                            for call in response_usage.get("llm_usage_breakdown") or []
                            if str(call.get("model_id") or "").lower() != str(response_model_id).lower()
                        }
                    ),
                }
                results.append(record)
                append_checkpoint_line(
                    checkpoint_path,
                    {"type": "result", "id": item["id"], "response_model": response_model_id, "record": record},
                )
                responses_this_session += 1
                print(_format_record_log(record))
                elapsed = time.perf_counter() - loop_started_perf
                print(
                    f"          [Progress] {responses_this_session}/{responses_to_run} responses this session | "
                    f"elapsed {_format_duration(elapsed)} | ETA {_format_duration(elapsed / responses_this_session * (responses_to_run - responses_this_session))}"
                )

                if error is not None and ("Setup Error" in error or model_consecutive_errors >= 2):
                    print(f"          -> ABORTING {response_model_id}: Model is failing continuously. Saving costs.")
                    break

        # Build aggregate summary
        summary = _build_summary(results)

        # Save Results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = build_results_output_path("benchmark", output_prefix, timestamp)
        run_finished_at = datetime.now()
        total_runtime_s = round(time.perf_counter() - run_started_perf, 3)
        session_end_entry = {
            "type": "session_end",
            "started_at": run_started_at.isoformat(),
            "finished_at": run_finished_at.isoformat(),
            "runtime_s": total_runtime_s,
            "responses_completed": responses_this_session,
            "git_commit": run_provenance["git"].get("commit"),
            "output_file": str(output_path),
        }
        append_checkpoint_line(checkpoint_path, session_end_entry)
        run_sessions = [
            *checkpoint["sessions"],
            {
                "type": "session",
                "started_at": run_started_at.isoformat(),
                "options": run_options,
                "git_commit": run_provenance["git"].get("commit"),
                "has_uncommitted_changes": run_provenance["git"].get("has_uncommitted_changes"),
                "system_code_sha256": run_provenance["git"].get("system_code_sha256"),
            },
            session_end_entry,
        ]
        api_model_names = {
            manifest["model_id"]: sorted(
                {name for r in results if r["response_model"] == manifest["model_id"] for name in r.get("api_model_names") or []}
            )
            for manifest in response_model_manifests
        }
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
                "timestamp": run_finished_at.isoformat(),
                "run_started_at": run_started_at.isoformat(),
                "run_finished_at": run_finished_at.isoformat(),
                "total_runtime_s": total_runtime_s,
                "real_services": True,
                "evaluation_models": list(evaluation_model_manifest.get("judge_models", [])),
                "judge_model_configs": judge_model_manifests,
                "pricing_model_count": len(pricing_catalog),
                "output_directory": str(output_path.parent),
                "output_file": str(output_path),
                "provenance": run_provenance,
                "runtime_data_at_end": collect_runtime_data_provenance(),
                "run_options": run_options,
                "warm_up": warm_up,
                "run_sessions": run_sessions,
                "api_model_names": api_model_names,
                "groundtruth_ids": [item["id"] for item in groundtruth_queries],
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

        _print_run_summary(
            results,
            [manifest["model_id"] for manifest in response_model_manifests],
            len(groundtruth_queries),
            checkpoint_path,
        )
        print(f"\nBenchmark complete. Results saved to {output_path}")
        print(f"Checkpoint kept at {checkpoint_path}")
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
    successful_results = [r for r in results if r["error"] is None]
    scored_results = [r for r in results if r.get("scores", {}).get("composite_score") is not None]
    heuristics_pass_count = sum(
        1 for r in results if (r.get("heuristics") or {}).get("overall_pass", False)
    )
    sla_met_count = sum(1 for r in results if r.get("sla_met", False))

    summary = {
        "overall": {
            "total_evaluated": len(results),
            "successful_responses": len(successful_results),
            "errored_responses": len(errors),
            "scored_responses": len(scored_results),
            "total_errors": len(errors),
            "error_categories": summarize_error_categories(results),
            "avg_composite_score": round(sum(scores) / len(scores), 3) if scores else 0,
            "avg_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else 0,
            "avg_tool_f1": _average_tool_f1(results),
            "heuristics_pass_count": heuristics_pass_count,
            "heuristics_pass_rate": round(
                heuristics_pass_count / len(results), 3
            ) if results else 0,
            "sla_met_count": sla_met_count,
            "sla_compliance": round(
                sla_met_count / len(results), 3
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
            "successful_responses": sum(1 for r in domain_results if r["error"] is None),
            "errors": sum(1 for r in domain_results if r["error"] is not None),
            "scored_responses": sum(1 for r in domain_results if r.get("scores", {}).get("composite_score") is not None),
            "avg_composite_score": round(sum(domain_scores) / len(domain_scores), 3) if domain_scores else 0,
            "avg_factual_accuracy": round(sum(factual_scores) / len(factual_scores), 3) if factual_scores else 0,
            "avg_tool_f1": _average_tool_f1(domain_results),
            "heuristics_pass_count": sum(
                1 for r in domain_results if (r.get("heuristics") or {}).get("overall_pass", False)
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
            "successful_responses": sum(1 for r in model_results if r["error"] is None),
            "avg_composite_score": round(sum(model_scores) / len(model_scores), 3) if model_scores else 0,
            "errors": sum(1 for r in model_results if r["error"] is not None),
            "scored_responses": sum(1 for r in model_results if r.get("scores", {}).get("composite_score") is not None),
            "error_categories": summarize_error_categories(model_results),
            "heuristics_pass_count": sum(
                1 for r in model_results if (r.get("heuristics") or {}).get("overall_pass", False)
            ),
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
        help="Repeatable response-model override in provider::model format, for example azure::gpt-5.4-mini.",
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
        default=None,
        help="Output filename prefix inside eval/results/benchmark/ (default benchmark_results; with --resume, the checkpoint's prefix).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a .partial.jsonl checkpoint. Use the same dataset as the original run.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="With --resume, run again the responses whose generation or judge call failed.",
    )
    parser.add_argument(
        "--query-id",
        action="append",
        dest="query_ids",
        help="Repeatable or comma-separated. Run only these worker query ids, for example W01,T05.",
    )
    args = parser.parse_args()
    if args.retry_errors and not args.resume:
        parser.error("--retry-errors requires --resume.")
    query_ids = [
        query_id.strip()
        for value in (args.query_ids or [])
        for query_id in value.split(",")
        if query_id.strip()
    ]

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
        resume_path=args.resume,
        retry_errors=args.retry_errors,
        query_ids=query_ids or None,
    )
