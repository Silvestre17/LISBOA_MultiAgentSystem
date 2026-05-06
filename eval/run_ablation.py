# ==========================================================================
# Master Thesis - Ablation Runner
#   - André Filipe Gomes Silvestre, 20240502
#
#   Compares a zero-shot baseline against the full LISBOA multi-agent system
#   and writes the JSON artefacts into:
#   eval/results/ablation/
#
# Usage:
#   > python -m eval.run_ablation --mode run_test
#       Quick ablation with 5 dataset entries.
#   > python -m eval.run_ablation --limit 20
#       Run the first 20 dataset entries.
#   > python -m eval.run_ablation --zero-shot-model gpt-5.4-mini
#       Override the zero-shot baseline model.
#   > python -m eval.run_ablation --dataset eval/evaluation_groundtruth_queries_demo.json --output-prefix ablation_results_demo
#       Run ablation on an alternate dataset and keep the artefacts separate from the main notebook inputs.
#   > python -m eval.run_ablation --include-domain weather --include-domain transport
#       Restrict the ablation run to a repeated set of specific domains.
#   > python -m eval.run_ablation --open-model-spec azure::Kimi-K2.5 --judge-model-spec openai::gpt-5.4-mini
#       Add an explicit open-model comparison profile and override the evaluation judge.
# ==========================================================================

import io
import json
import os
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph import MultiAgentAssistant  # For the LISBOA full pipeline comparison
from agent.llm_factory import LLMFactory
from agent.utils.langsmith_tracing import (
    LANGSMITH_AVAILABLE,
    get_langsmith_scoped_project_name,
    get_langsmith_tracing_status,
    get_last_langsmith_runtime_failure,
    tracing_project_override,
)
from config import Config
from eval.llm_judge import LLMJudge
from eval.runtime_utils import (
    aggregate_judge_runs,
    build_reference_context,
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
    write_json_artifact,
)
from eval.validators.response_heuristics import run_all_heuristics

# TEST: This shared ground-truth corpus feeds both benchmark and ablation.
GROUNDTRUTH_QUERIES_PATH = Path(__file__).with_name("evaluation_groundtruth_queries.json")
ABLATION_LANGSMITH_PROJECT_ENV = "LISBOA_LANGSMITH_ABLATION_PROJECT"
ABLATION_LANGSMITH_SCOPE_LABEL = "Ablation"
SUPPORTED_MODEL_PROVIDERS = {"azure", "openai", "lmstudio"}
# TEST: Zero-shot baseline model lives here by default, or can be overridden via CLI.
DEFAULT_ZERO_SHOT_PROVIDER = "azure"
DEFAULT_ZERO_SHOT_MODEL = "gpt-5.4-mini"
DEFAULT_OPEN_PROVIDER = "azure"
DEFAULT_OPEN_MODEL = "Kimi-K2.5"
DEFAULT_ABLATION_DOMAINS = ("weather", "transport", "researcher", "multi_agent")
ABLATION_PRIMARY_SCORE_FIELDS = (
    "factual_accuracy",
    "completeness",
    "relevance",
    "response_quality",
)
ZERO_SHOT_BASELINE_SYSTEM_PROMPT = """You are LISBOA's no-tool baseline for an academic ablation study.

Answer as the LISBOA assistant for Lisbon and the Lisbon Metropolitan Area, but do not call or pretend to call tools, APIs, retrieval, live transport feeds, weather services, booking systems, or web search. Keep the same user-facing scope as LISBOA: Lisbon tourism, local services, weather, and urban mobility. If the user asks for live, current, unsupported, or out-of-scope information, state the limitation explicitly instead of inventing data. Match the user's language and keep the response concise, practical, and grounded in general knowledge only."""


def _average_tool_f1(blocks: list[dict]) -> float:
    """Return the mean deterministic tool F1 over blocks where tool scoring applies."""
    values = []
    for block in blocks:
        metrics = block.get("tool_metrics") or {}
        value = metrics.get("tool_f1")
        if value is not None and metrics.get("tool_metric_scored", True):
            values.append(float(value))
    return round(sum(values) / len(values), 3) if values else 0


def _compute_ablation_quality_score(scores: dict[str, object]) -> float | None:
    """Return the ablation primary score, excluding tool-usage by design."""
    values = []
    for field in ABLATION_PRIMARY_SCORE_FIELDS:
        value = scores.get(field)
        if value is None:
            return None
        values.append(float(value))
    return round(sum(values) / len(values), 4)


def _with_ablation_primary_score(scores: dict[str, object]) -> dict[str, object]:
    """Attach the four-dimension ablation score while preserving raw judge composite."""
    updated = dict(scores)
    tool_inclusive = updated.get("composite_score")
    if tool_inclusive is not None:
        updated["tool_inclusive_composite_score"] = tool_inclusive
    ablation_quality_score = _compute_ablation_quality_score(updated)
    updated["ablation_quality_score"] = ablation_quality_score
    updated["ablation_score_dimensions"] = list(ABLATION_PRIMARY_SCORE_FIELDS)
    updated["composite_score"] = ablation_quality_score
    return updated


DEFAULT_JUDGE_MODELS = [
    {"provider": DEFAULT_ZERO_SHOT_PROVIDER, "model": DEFAULT_ZERO_SHOT_MODEL, "temperature": 0.0},
    {"provider": DEFAULT_OPEN_PROVIDER, "model": DEFAULT_OPEN_MODEL, "temperature": 0.0},
]


def _is_transient_rate_limit_error(error: Exception | str | None) -> bool:
    """Return whether the error matches a transient capacity or rate-limit failure."""
    lowered = str(error or "").lower()
    return any(
        token in lowered
        for token in (
            "429",
            "rate limit",
            "too many requests",
            "maximum concurrent capacity",
            "concurrent capacity",
        )
    )


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
            "response_usage_note": "The response was produced through a deterministic tool or rule path without an LLM generation call, so response-side tokens and cost are zero by design.",
        }
    return {
        "response_generation_mode": "deterministic_rule",
        "response_usage_status": "not_applicable_no_llm",
        "response_usage_note": "The response was produced through a deterministic non-LLM path, so response-side tokens and cost are zero by design.",
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


def normalize_model_provider(provider: str | None) -> str | None:
    """Normalize and validate a model provider override."""
    if provider is None:
        return None

    normalized_provider = str(provider).strip().lower()
    if not normalized_provider:
        return None
    if normalized_provider not in SUPPORTED_MODEL_PROVIDERS:
        raise ValueError(
            f"Unsupported provider '{provider}'. Expected one of: {sorted(SUPPORTED_MODEL_PROVIDERS)}"
        )
    return normalized_provider


@contextmanager
def temporary_lisboa_provider(
    provider: str | None,
    *,
    model_name: str | None = None,
    temperature: float | None = None,
):
    """Temporarily override the provider family and optional model profile used by LISBOA agents."""
    original_provider = Config.MODEL_PROVIDER
    original_agent_maps = {
        "azure": deepcopy(Config.AGENT_MODELS_AZURE),
        "openai": deepcopy(Config.AGENT_MODELS_OPENAI),
        "lmstudio": deepcopy(Config.AGENT_MODELS_LMSTUDIO),
    }
    normalized_provider = normalize_model_provider(provider)
    if normalized_provider is not None:
        Config.MODEL_PROVIDER = normalized_provider
        if model_name is not None or temperature is not None:
            agent_map_attr = f"AGENT_MODELS_{normalized_provider.upper()}"
            current_agent_map = deepcopy(getattr(Config, agent_map_attr))
            for agent_config in current_agent_map.values():
                agent_config["provider"] = normalized_provider
                if model_name is not None:
                    agent_config["model"] = str(model_name)
                if temperature is not None:
                    agent_config["temperature"] = float(temperature)
            setattr(Config, agent_map_attr, current_agent_map)

    try:
        yield Config.MODEL_PROVIDER
    finally:
        Config.MODEL_PROVIDER = original_provider
        Config.AGENT_MODELS_AZURE = deepcopy(original_agent_maps["azure"])
        Config.AGENT_MODELS_OPENAI = deepcopy(original_agent_maps["openai"])
        Config.AGENT_MODELS_LMSTUDIO = deepcopy(original_agent_maps["lmstudio"])


def resolve_ablation_profile(
    *,
    profile_name: str,
    paradigm: str,
    default_provider: str,
    default_model: str,
    model_spec: str | None = None,
    temperature: float | None = None,
) -> dict[str, str | float]:
    """Resolve one ablation comparison profile."""
    if model_spec:
        resolved = parse_model_spec(
            model_spec,
            temperature=temperature,
            supported_providers=SUPPORTED_MODEL_PROVIDERS,
        )
    else:
        resolved = {
            "provider": default_provider,
            "model": default_model,
            "temperature": 0.0 if temperature is None else float(temperature),
        }

    return {
        **resolved,
        "profile_name": profile_name,
        "profile_id": f"{profile_name}::{resolved['provider']}::{resolved['model']}",
        "paradigm": paradigm,
    }


def resolve_judge_models(
    judge_model_specs: list[str] | None = None,
    *,
    provider: str | None = None,
    model_name: str | None = None,
) -> list[dict[str, str | float]]:
    """Resolve the configured ablation judge matrix."""
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
    limit: int | None = None,
    include_domains: Sequence[str] | None = DEFAULT_ABLATION_DOMAINS,
):
    """Load the ablation corpus, optionally filtering domains and selecting a balanced subset."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if include_domains is not None:
        allowed_domains = {str(domain) for domain in include_domains}
        data = [item for item in data if item.get("domain") in allowed_domains]
    if limit is None:
        return data
    return select_balanced_subset(data, limit, group_key="domain")


def run_zero_shot(
    query: str,
    provider: str = DEFAULT_ZERO_SHOT_PROVIDER,
    model_name: str = DEFAULT_ZERO_SHOT_MODEL,
):
    """Run the no-tool LISBOA-instructed zero-shot baseline."""
    llm = LLMFactory.get_llm(provider=provider, model=model_name, temperature=0.0)
    model_id = build_model_id(provider, model_name)

    start = time.time()
    last_error = None
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response = llm.invoke([
                SystemMessage(content=ZERO_SHOT_BASELINE_SYSTEM_PROMPT),
                HumanMessage(content=query),
            ])
            latency = time.time() - start
            response_usage = build_usage_payload(
                LLMFactory.extract_usage_metadata(response),
                model_id=model_id,
                call_count=1,
            )
            return response.content, [], "", latency, None, response_usage
        except Exception as e:
            last_error = e
            if _is_transient_rate_limit_error(e) and attempt < max_attempts - 1:
                wait_s = 2.0 * (attempt + 1)
                print(
                    f"      [Zero-Shot Retry] Transient model capacity/rate-limit failure "
                    f"(attempt {attempt + 1}/{max_attempts}). Retrying in {wait_s}s..."
                )
                time.sleep(wait_s)
                continue
            break

    return (
        f"Error: {last_error}",
        [],
        "",
        time.time() - start,
        str(last_error),
        build_usage_payload({}, model_id=model_id, call_count=0),
    )


def run_lisboa(query: str, system: MultiAgentAssistant, *, language: str = "en"):
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

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            response = system.chat(query, language=language, verbose=False)
        latency = time.time() - start

        retrieved_context_str = "\n---\n".join(retrieved_context_blocks)
        response_usage = build_usage_payload(system.get_llm_usage_summary())
        agent_usage_snapshot = {
            agent_name: build_usage_payload(summary)
            for agent_name, summary in system.get_llm_usage_snapshot().items()
        }
        agent_tool_logs = {
            "supervisor": system.supervisor.get_tool_calls_log(),
            "qa": system.qa_agent.get_tool_calls_log(),
            **{
                agent_name: agent.get_tool_calls_log()
                for agent_name, agent in system.agents.items()
            },
        }
        agents_used = [
            agent_name
            for agent_name, usage_payload in agent_usage_snapshot.items()
            if int(usage_payload.get("call_count", 0) or 0) > 0 or agent_tool_logs.get(agent_name)
        ]

        return (
            response,
            tools_called,
            retrieved_context_str,
            latency,
            None,
            response_usage,
            {
                "agent_usage": agent_usage_snapshot,
                "agent_tool_logs": agent_tool_logs,
                "agents_used": agents_used,
            },
        )
    except Exception as e:
        return (
            f"Error: {e}",
            [],
            "",
            time.time() - start,
            str(e),
            build_usage_payload(system.get_llm_usage_summary()),
            {
                "agent_usage": {
                    agent_name: build_usage_payload(summary)
                    for agent_name, summary in system.get_llm_usage_snapshot().items()
                },
                "agent_tool_logs": {
                    "supervisor": system.supervisor.get_tool_calls_log(),
                    "qa": system.qa_agent.get_tool_calls_log(),
                    **{
                        agent_name: agent.get_tool_calls_log()
                        for agent_name, agent in system.agents.items()
                    },
                },
                "agents_used": [],
            },
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


def _compact_usage(payload: dict) -> dict:
    """Return a compact usage payload for persisted ablation summaries."""
    return {
        "call_count": int(payload.get("call_count", 0) or 0),
        "usage_available": bool(payload.get("usage_available", False)),
        "tokens": payload.get(
            "tokens",
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        ),
    }


def _compact_cost(payload: dict) -> dict:
    """Return a compact cost payload for persisted ablation summaries."""
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


def _determine_ablation_winner(
    zero_shot_score: float | None,
    lisboa_score: float | None,
) -> str | None:
    """Return the winning arm label for a comparison when a score is available."""
    if zero_shot_score is None and lisboa_score is None:
        return None
    if zero_shot_score is None:
        return "lisboa"
    if lisboa_score is None:
        return "zero_shot"
    if abs(float(lisboa_score) - float(zero_shot_score)) < 1e-12:
        return "tie"
    return "lisboa" if float(lisboa_score) > float(zero_shot_score) else "zero_shot"


def _build_profile_summary(results: list[dict], profile_key: str) -> dict[str, object]:
    """Build one ablation summary block for a specific comparison profile."""
    profile_records = [
        (record, record["comparisons"][profile_key])
        for record in results
        if profile_key in record.get("comparisons", {})
    ]
    if not profile_records:
        return {
            "total_queries": 0,
            "total_comparisons": 0,
            "zero_shot_avg": 0,
            "lisboa_avg": 0,
            "zero_shot_avg_tool_f1": 0,
            "lisboa_avg_tool_f1": 0,
            "lisboa_improvement": 0,
            "zero_shot_counts": {
                "total": 0,
                "successful_responses": 0,
                "errored_responses": 0,
                "scored_responses": 0,
                "heuristics_pass_count": 0,
            },
            "lisboa_counts": {
                "total": 0,
                "successful_responses": 0,
                "errored_responses": 0,
                "scored_responses": 0,
                "heuristics_pass_count": 0,
            },
            "winner_counts": {"zero_shot": 0, "lisboa": 0, "tie": 0, "unresolved": 0},
            "zero_shot_heuristics_pass_rate": 0,
            "lisboa_heuristics_pass_rate": 0,
            "per_domain": {},
            "error_categories": {"zero_shot": {}, "lisboa": {}},
            "zero_shot_usage": {"response": _compact_usage({}), "evaluation": _compact_usage({}), "combined": _compact_usage({})},
            "zero_shot_cost_usd": {"response": _compact_cost({}), "evaluation": _compact_cost({}), "combined": _compact_cost({})},
            "lisboa_usage": {"response": _compact_usage({}), "evaluation": _compact_usage({}), "combined": _compact_usage({})},
            "lisboa_cost_usd": {"response": _compact_cost({}), "evaluation": _compact_cost({}), "combined": _compact_cost({})},
            "comparison_usage": _compact_usage({}),
            "comparison_cost_usd": _compact_cost({}),
        }

    comparison_blocks = [comparison for _, comparison in profile_records]
    zero_shot_blocks = [comparison["metrics"]["zero_shot"] for comparison in comparison_blocks]
    lisboa_blocks = [comparison["metrics"]["lisboa"] for comparison in comparison_blocks]
    zs_scores = [block["scores"]["composite_score"] for block in zero_shot_blocks if block["scores"].get("composite_score") is not None]
    ls_scores = [block["scores"]["composite_score"] for block in lisboa_blocks if block["scores"].get("composite_score") is not None]
    zero_shot_heuristics_pass_count = sum(
        1 for block in zero_shot_blocks if (block.get("heuristics") or {}).get("overall_pass", False)
    )
    lisboa_heuristics_pass_count = sum(
        1 for block in lisboa_blocks if (block.get("heuristics") or {}).get("overall_pass", False)
    )
    winner_counts = {"zero_shot": 0, "lisboa": 0, "tie": 0, "unresolved": 0}
    for comparison in comparison_blocks:
        winner = comparison.get("comparison_summary", {}).get("winner_by_avg_composite")
        if winner in winner_counts:
            winner_counts[str(winner)] += 1
        else:
            winner_counts["unresolved"] += 1

    per_domain_summary: dict[str, object] = {}
    summary: dict[str, object] = {
        "total_queries": len(profile_records),
        "total_comparisons": len(comparison_blocks),
        "zero_shot_avg": round(sum(zs_scores) / len(zs_scores), 3) if zs_scores else 0,
        "lisboa_avg": round(sum(ls_scores) / len(ls_scores), 3) if ls_scores else 0,
        "zero_shot_avg_tool_f1": _average_tool_f1(zero_shot_blocks),
        "lisboa_avg_tool_f1": _average_tool_f1(lisboa_blocks),
        "lisboa_improvement": round(
            (sum(ls_scores) / len(ls_scores)) - (sum(zs_scores) / len(zs_scores)), 3
        ) if zs_scores and ls_scores else 0,
        "zero_shot_counts": {
            "total": len(zero_shot_blocks),
            "successful_responses": sum(1 for block in zero_shot_blocks if block.get("error") is None),
            "errored_responses": sum(1 for block in zero_shot_blocks if block.get("error") is not None),
            "scored_responses": len(zs_scores),
            "heuristics_pass_count": zero_shot_heuristics_pass_count,
        },
        "lisboa_counts": {
            "total": len(lisboa_blocks),
            "successful_responses": sum(1 for block in lisboa_blocks if block.get("error") is None),
            "errored_responses": sum(1 for block in lisboa_blocks if block.get("error") is not None),
            "scored_responses": len(ls_scores),
            "heuristics_pass_count": lisboa_heuristics_pass_count,
        },
        "winner_counts": winner_counts,
        "zero_shot_heuristics_pass_rate": round(
            zero_shot_heuristics_pass_count / len(zero_shot_blocks),
            3,
        ) if zero_shot_blocks else 0,
        "lisboa_heuristics_pass_rate": round(
            lisboa_heuristics_pass_count / len(lisboa_blocks),
            3,
        ) if lisboa_blocks else 0,
        "per_domain": per_domain_summary,
        "error_categories": {"zero_shot": {}, "lisboa": {}},
    }

    summary["zero_shot_usage"] = {
        "response": _compact_usage(combine_usage_payloads([block.get("response_usage", {}) for block in zero_shot_blocks])),
        "evaluation": _compact_usage(combine_usage_payloads([block.get("evaluation_usage", {}) for block in zero_shot_blocks])),
        "combined": _compact_usage(combine_usage_payloads([block.get("combined_usage", {}) for block in zero_shot_blocks])),
    }
    summary["zero_shot_cost_usd"] = {
        "response": _compact_cost(combine_cost_payloads([block.get("response_cost_usd", {}) for block in zero_shot_blocks])),
        "evaluation": _compact_cost(combine_cost_payloads([block.get("evaluation_cost_usd", {}) for block in zero_shot_blocks])),
        "combined": _compact_cost(combine_cost_payloads([block.get("combined_cost_usd", {}) for block in zero_shot_blocks])),
    }
    summary["lisboa_usage"] = {
        "response": _compact_usage(combine_usage_payloads([block.get("response_usage", {}) for block in lisboa_blocks])),
        "evaluation": _compact_usage(combine_usage_payloads([block.get("evaluation_usage", {}) for block in lisboa_blocks])),
        "combined": _compact_usage(combine_usage_payloads([block.get("combined_usage", {}) for block in lisboa_blocks])),
    }
    summary["lisboa_cost_usd"] = {
        "response": _compact_cost(combine_cost_payloads([block.get("response_cost_usd", {}) for block in lisboa_blocks])),
        "evaluation": _compact_cost(combine_cost_payloads([block.get("evaluation_cost_usd", {}) for block in lisboa_blocks])),
        "combined": _compact_cost(combine_cost_payloads([block.get("combined_cost_usd", {}) for block in lisboa_blocks])),
    }
    summary["comparison_usage"] = _compact_usage(
        combine_usage_payloads([comparison.get("comparison_usage", {}) for comparison in comparison_blocks])
    )
    summary["comparison_cost_usd"] = _compact_cost(
        combine_cost_payloads([comparison.get("comparison_cost_usd", {}) for comparison in comparison_blocks])
    )

    domains = sorted({record["domain"] for record, _ in profile_records})
    for domain in domains:
        domain_blocks = [
            comparison
            for record, comparison in profile_records
            if record["domain"] == domain
        ]
        domain_zero_shot = [comparison["metrics"]["zero_shot"] for comparison in domain_blocks]
        domain_lisboa = [comparison["metrics"]["lisboa"] for comparison in domain_blocks]
        domain_zs_scores = [block["scores"]["composite_score"] for block in domain_zero_shot if block["scores"].get("composite_score") is not None]
        domain_ls_scores = [block["scores"]["composite_score"] for block in domain_lisboa if block["scores"].get("composite_score") is not None]
        domain_zero_shot_heuristics_pass_count = sum(
            1 for block in domain_zero_shot if (block.get("heuristics") or {}).get("overall_pass", False)
        )
        domain_lisboa_heuristics_pass_count = sum(
            1 for block in domain_lisboa if (block.get("heuristics") or {}).get("overall_pass", False)
        )
        per_domain_summary[domain] = {
            "count": len(domain_blocks),
            "zero_shot_avg": round(sum(domain_zs_scores) / len(domain_zs_scores), 3) if domain_zs_scores else 0,
            "lisboa_avg": round(sum(domain_ls_scores) / len(domain_ls_scores), 3) if domain_ls_scores else 0,
            "zero_shot_avg_tool_f1": _average_tool_f1(domain_zero_shot),
            "lisboa_avg_tool_f1": _average_tool_f1(domain_lisboa),
            "zero_shot_counts": {
                "total": len(domain_zero_shot),
                "successful_responses": sum(1 for block in domain_zero_shot if block.get("error") is None),
                "errored_responses": sum(1 for block in domain_zero_shot if block.get("error") is not None),
                "scored_responses": len(domain_zs_scores),
                "heuristics_pass_count": domain_zero_shot_heuristics_pass_count,
            },
            "lisboa_counts": {
                "total": len(domain_lisboa),
                "successful_responses": sum(1 for block in domain_lisboa if block.get("error") is None),
                "errored_responses": sum(1 for block in domain_lisboa if block.get("error") is not None),
                "scored_responses": len(domain_ls_scores),
                "heuristics_pass_count": domain_lisboa_heuristics_pass_count,
            },
            "zero_shot_heuristics_pass_rate": round(
                domain_zero_shot_heuristics_pass_count / len(domain_zero_shot),
                3,
            ) if domain_zero_shot else 0,
            "lisboa_heuristics_pass_rate": round(
                domain_lisboa_heuristics_pass_count / len(domain_lisboa),
                3,
            ) if domain_lisboa else 0,
            "zero_shot_usage": {
                "response": _compact_usage(combine_usage_payloads([block.get("response_usage", {}) for block in domain_zero_shot])),
                "evaluation": _compact_usage(combine_usage_payloads([block.get("evaluation_usage", {}) for block in domain_zero_shot])),
                "combined": _compact_usage(combine_usage_payloads([block.get("combined_usage", {}) for block in domain_zero_shot])),
            },
            "zero_shot_cost_usd": {
                "response": _compact_cost(combine_cost_payloads([block.get("response_cost_usd", {}) for block in domain_zero_shot])),
                "evaluation": _compact_cost(combine_cost_payloads([block.get("evaluation_cost_usd", {}) for block in domain_zero_shot])),
                "combined": _compact_cost(combine_cost_payloads([block.get("combined_cost_usd", {}) for block in domain_zero_shot])),
            },
            "lisboa_usage": {
                "response": _compact_usage(combine_usage_payloads([block.get("response_usage", {}) for block in domain_lisboa])),
                "evaluation": _compact_usage(combine_usage_payloads([block.get("evaluation_usage", {}) for block in domain_lisboa])),
                "combined": _compact_usage(combine_usage_payloads([block.get("combined_usage", {}) for block in domain_lisboa])),
            },
            "lisboa_cost_usd": {
                "response": _compact_cost(combine_cost_payloads([block.get("response_cost_usd", {}) for block in domain_lisboa])),
                "evaluation": _compact_cost(combine_cost_payloads([block.get("evaluation_cost_usd", {}) for block in domain_lisboa])),
                "combined": _compact_cost(combine_cost_payloads([block.get("combined_cost_usd", {}) for block in domain_lisboa])),
            },
        }

    zero_shot_error_categories: dict[str, int] = {}
    lisboa_error_categories: dict[str, int] = {}
    for block in zero_shot_blocks:
        error_type = block.get("error_type")
        if error_type:
            zero_shot_error_categories[str(error_type)] = zero_shot_error_categories.get(str(error_type), 0) + 1
    for block in lisboa_blocks:
        error_type = block.get("error_type")
        if error_type:
            lisboa_error_categories[str(error_type)] = lisboa_error_categories.get(str(error_type), 0) + 1
    summary["error_categories"] = {
        "zero_shot": dict(sorted(zero_shot_error_categories.items())),
        "lisboa": dict(sorted(lisboa_error_categories.items())),
    }

    return summary


def run_ablation(
    limit: int = None,
    zero_shot_provider: str = DEFAULT_ZERO_SHOT_PROVIDER,
    zero_shot_model: str = DEFAULT_ZERO_SHOT_MODEL,
    pricing_by_model: dict | None = None,
    open_model_spec: str | None = None,
    judge_model_specs: list[str] | None = None,
    lisboa_provider: str | None = None,
    judge_provider: str | None = None,
    judge_model: str | None = None,
    groundtruth_path: str | Path | None = None,
    include_domains: Sequence[str] | None = DEFAULT_ABLATION_DOMAINS,
    output_prefix: str = "ablation_results",
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
        open_model_spec: Optional open-model response profile in provider::model format.
        judge_model_specs: Optional repeatable list of judge model specs.
        lisboa_provider: Legacy compatibility override. Dual-profile mode now keeps
            zero-shot and LISBOA within the same provider/model family per profile.
        judge_provider: Optional provider override for a single evaluation judge.
        judge_model: Optional model override for a single evaluation judge.
        include_domains: Optional domain filter for the shared corpus. By default
            the ablation excludes ``greeting`` and ``out_of_scope`` because LISBOA
            answers those through hard-coded supervisor shortcuts rather than the
            grounded pipeline under study.
    """
    ablation_langsmith_project = get_langsmith_scoped_project_name(
        ABLATION_LANGSMITH_SCOPE_LABEL,
        env_name=ABLATION_LANGSMITH_PROJECT_ENV,
    )
    langsmith_status = get_langsmith_tracing_status()
    if LANGSMITH_AVAILABLE:
        print(
            f"[LangSmith] Ablation traces will be saved to project: {ablation_langsmith_project}"
        )
    else:
        print(
            "[LangSmith] Ablation tracing is inactive. "
            f"{langsmith_status.get('reason', 'LangSmith tracing is disabled')} "
            "Set LANGSMITH_TRACING=true with valid credentials to save these runs to project: "
            f"{ablation_langsmith_project}"
        )

    with tracing_project_override(ablation_langsmith_project):
        run_started_at = datetime.now()
        run_started_perf = time.perf_counter()
        print("=" * 60)
        print(f"STARTING ABLATION STUDY (Zero-Shot vs LISBOA Framework) (LIMIT={limit})")
        print("=" * 60)

        resolved_groundtruth_path = resolve_groundtruth_path(groundtruth_path)
        pricing_by_model = resolve_pricing_catalog(pricing_by_model)
        groundtruth_queries = load_groundtruth_queries(
            resolved_groundtruth_path,
            limit=limit,
            include_domains=include_domains,
        )
        active_domains = sorted({item["domain"] for item in groundtruth_queries})
        print(f"[Ablation] Domains in scope: {active_domains}")

        if lisboa_provider is not None:
            print(
                "[Ablation] Note: --lisboa-provider is ignored in dual-profile mode so each pair remains fair."
            )

        comparison_profiles = [
            resolve_ablation_profile(
                profile_name="closed_source",
                paradigm="closed_source",
                default_provider=zero_shot_provider,
                default_model=zero_shot_model,
                model_spec=f"{zero_shot_provider}::{zero_shot_model}",
            ),
            resolve_ablation_profile(
                profile_name="open_source",
                paradigm="open_source",
                default_provider=DEFAULT_OPEN_PROVIDER,
                default_model=DEFAULT_OPEN_MODEL,
                model_spec=open_model_spec,
            ),
        ]
        primary_profile_key = str(comparison_profiles[0]["profile_name"])

        judge_configs = resolve_judge_models(
            judge_model_specs,
            provider=judge_provider,
            model_name=judge_model,
        )
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
        pricing_catalog, _ = split_pricing_config(pricing_by_model)

        results_by_id = {
            item["id"]: {
                "id": item["id"],
                "query": item["query"],
                "domain": item["domain"],
                "language": item.get("language", "en"),
                "edge_case": item.get("edge_case", False),
                "edge_type": item.get("edge_type", None),
                "expected_behavior": item.get("expected_behavior"),
                "expected_facts": item.get("expected_facts", []),
                "expected_tools": item.get("expected_tools", []),
                "reference_context": build_reference_context(
                    expected_facts=item.get("expected_facts", []),
                    expected_behavior=item.get("expected_behavior"),
                ),
                "primary_comparison_profile": primary_profile_key,
                "comparisons": {},
            }
            for item in groundtruth_queries
        }
        profile_metadata: dict[str, dict[str, object]] = {}

        for profile in comparison_profiles:
            profile_key = str(profile["profile_name"])
            print(
                f"\n{'=' * 60}\nPROFILE: {profile_key} -> {profile['provider']}::{profile['model']}\n{'=' * 60}"
            )

            profile_consecutive_errors = 0
            with temporary_lisboa_provider(
                str(profile["provider"]),
                model_name=str(profile["model"]),
                temperature=float(profile.get("temperature", 0.0) or 0.0),
            ) as active_lisboa_provider:
                lisboa_system = MultiAgentAssistant()
                zero_shot_model_manifest = build_model_manifest(
                    str(profile["provider"]),
                    str(profile["model"]),
                    float(profile.get("temperature", 0.0) or 0.0),
                    extra={
                        "profile_name": profile_key,
                        "profile_id": str(profile["profile_id"]),
                        "paradigm": str(profile["paradigm"]),
                    },
                )
                zero_shot_model_id = str(zero_shot_model_manifest["model_id"])
                lisboa_response_model_config = _build_lisboa_response_model_manifest(lisboa_system)
                lisboa_response_model = lisboa_response_model_config["display_model"]
                profile_metadata[profile_key] = {
                    "profile_name": profile_key,
                    "profile_id": str(profile["profile_id"]),
                    "paradigm": str(profile["paradigm"]),
                    "zero_shot_model_config": deepcopy(zero_shot_model_manifest),
                    "lisboa_response_model_config": deepcopy(lisboa_response_model_config),
                    "lisboa_provider": active_lisboa_provider,
                }

                for idx, item in enumerate(groundtruth_queries):
                    print(f"\n[{idx + 1}/{len(groundtruth_queries)}] [{profile_key}] ABLATING: {item['query']}")

                    zs_resp, zs_tools, zs_ctx, zs_lat, zs_err, zs_response_usage = run_zero_shot(
                        item["query"],
                        provider=str(profile["provider"]),
                        model_name=str(profile["model"]),
                    )
                    zs_response_cost = build_cost_payload(
                        zs_response_usage,
                        pricing_by_model,
                        model_id=zero_shot_model_id,
                    )
                    zs_tool_metrics = compute_tool_metrics(
                        expected=item.get("expected_tools", []),
                        actual=zs_tools,
                        acceptable_tool_sets=item.get("acceptable_tool_sets"),
                        tool_expectation=item.get("tool_expectation", "strict"),
                    )
                    zs_heuristics = None if zs_err is not None else run_all_heuristics(
                        response=zs_resp,
                        expected_language=item.get("language", "en"),
                    )
                    zs_judge_runs, zs_aggregated_judges = _evaluate_with_judges(
                        judges=judges,
                        judge_model_manifests=judge_model_manifests,
                        query=item["query"],
                        expected_facts=item.get("expected_facts", []),
                        expected_tools=item.get("expected_tools", []),
                        actual_tools=zs_tools,
                        retrieved_context=zs_ctx,
                        reference_context=build_reference_context(
                            expected_facts=item.get("expected_facts", []),
                            expected_behavior=item.get("expected_behavior"),
                        ),
                        response=zs_resp,
                        response_error=zs_err,
                        pricing_by_model=pricing_by_model,
                        tool_expectation=item.get("tool_expectation", "strict"),
                        acceptable_tool_sets=item.get("acceptable_tool_sets"),
                    )
                    zs_score = _with_ablation_primary_score(dict(zs_aggregated_judges.get("scores", {})))
                    zs_evaluation_usage = zs_aggregated_judges.get(
                        "evaluation_usage",
                        build_usage_payload({}, model_id=evaluation_model_id, call_count=0),
                    )
                    zs_evaluation_cost = zs_aggregated_judges.get(
                        "evaluation_cost_usd",
                        build_cost_payload(
                            build_usage_payload({}, model_id=evaluation_model_id, call_count=0),
                            pricing_by_model,
                            model_id=evaluation_model_id,
                        ),
                    )
                    zs_combined_usage = combine_usage_payloads([zs_response_usage, zs_evaluation_usage])
                    zs_combined_cost = combine_cost_payloads([zs_response_cost, zs_evaluation_cost])
                    zs_response_telemetry = _describe_response_telemetry(
                        response_usage=zs_response_usage,
                        tools_used=zs_tools,
                        error=zs_err,
                    )

                    score_disp_zs = f"{zs_score['composite_score']:.2f}/5.0" if zs_score.get("composite_score") is not None else "N/A"
                    print(f"  [Zero-Shot] Score: {score_disp_zs} | Lat: {zs_lat:.2f}s")

                    ls_resp, ls_tools, ls_ctx, ls_lat, ls_err, ls_response_usage, ls_runtime = run_lisboa(
                        item["query"],
                        lisboa_system,
                        language=item.get("language", "en"),
                    )
                    ls_response_cost = build_cost_payload(
                        ls_response_usage,
                        pricing_by_model,
                    )
                    ls_tool_metrics = compute_tool_metrics(
                        expected=item.get("expected_tools", []),
                        actual=ls_tools,
                        acceptable_tool_sets=item.get("acceptable_tool_sets"),
                        tool_expectation=item.get("tool_expectation", "strict"),
                    )
                    ls_heuristics = None if ls_err is not None else run_all_heuristics(
                        response=ls_resp,
                        expected_language=item.get("language", "en"),
                    )
                    ls_judge_runs, ls_aggregated_judges = _evaluate_with_judges(
                        judges=judges,
                        judge_model_manifests=judge_model_manifests,
                        query=item["query"],
                        expected_facts=item.get("expected_facts", []),
                        expected_tools=item.get("expected_tools", []),
                        actual_tools=ls_tools,
                        retrieved_context=ls_ctx,
                        reference_context=build_reference_context(
                            expected_facts=item.get("expected_facts", []),
                            expected_behavior=item.get("expected_behavior"),
                        ),
                        response=ls_resp,
                        response_error=ls_err,
                        pricing_by_model=pricing_by_model,
                        tool_expectation=item.get("tool_expectation", "strict"),
                        acceptable_tool_sets=item.get("acceptable_tool_sets"),
                    )
                    ls_score = _with_ablation_primary_score(dict(ls_aggregated_judges.get("scores", {})))
                    ls_evaluation_usage = ls_aggregated_judges.get(
                        "evaluation_usage",
                        build_usage_payload({}, model_id=evaluation_model_id, call_count=0),
                    )
                    ls_evaluation_cost = ls_aggregated_judges.get(
                        "evaluation_cost_usd",
                        build_cost_payload(
                            build_usage_payload({}, model_id=evaluation_model_id, call_count=0),
                            pricing_by_model,
                            model_id=evaluation_model_id,
                        ),
                    )
                    ls_combined_usage = combine_usage_payloads([ls_response_usage, ls_evaluation_usage])
                    ls_combined_cost = combine_cost_payloads([ls_response_cost, ls_evaluation_cost])
                    ls_response_telemetry = _describe_response_telemetry(
                        response_usage=ls_response_usage,
                        tools_used=ls_tools,
                        error=ls_err,
                    )

                    score_disp_ls = f"{ls_score['composite_score']:.2f}/5.0" if ls_score.get("composite_score") is not None else "N/A"
                    print(f"  [LISBOA]    Score: {score_disp_ls} | Lat: {ls_lat:.2f}s | Tools: {len(ls_tools)}")

                    if zs_err is not None or ls_err is not None:
                        profile_consecutive_errors += 1
                    else:
                        profile_consecutive_errors = 0

                    comparison_usage = combine_usage_payloads([zs_combined_usage, ls_combined_usage])
                    comparison_cost = combine_cost_payloads([zs_combined_cost, ls_combined_cost])

                    ls_agent_usage = deepcopy(ls_runtime.get("agent_usage", {})) if isinstance(ls_runtime, dict) else {}
                    ls_agent_tool_logs = deepcopy(ls_runtime.get("agent_tool_logs", {})) if isinstance(ls_runtime, dict) else {}
                    ls_agent_costs = {
                        agent_name: build_cost_payload(
                            usage_payload,
                            pricing_by_model,
                            model_id=usage_payload.get("model_id"),
                        )
                        for agent_name, usage_payload in ls_agent_usage.items()
                    }
                    ls_agents_used = list(ls_runtime.get("agents_used", [])) if isinstance(ls_runtime, dict) else []

                    comparison_block = {
                        "profile": deepcopy(profile_metadata[profile_key]),
                        "profile_id": str(profile["profile_id"]),
                        "paradigm": str(profile["paradigm"]),
                        "comparison_usage": comparison_usage,
                        "comparison_cost_usd": comparison_cost,
                        "comparison_summary": {
                            "winner_by_avg_composite": _determine_ablation_winner(
                                zs_score.get("composite_score"),
                                ls_score.get("composite_score"),
                            ),
                            "lisboa_minus_zero_shot": round(
                                float(ls_score.get("composite_score")) - float(zs_score.get("composite_score")),
                                4,
                            ) if zs_score.get("composite_score") is not None and ls_score.get("composite_score") is not None else None,
                        },
                        "metrics": {
                            "zero_shot": {
                                "agents_used": ["zero_shot"],
                                "response_model": zero_shot_model_id,
                                "response_model_config": deepcopy(zero_shot_model_manifest),
                                "evaluation_model": evaluation_model_id,
                                "evaluation_model_config": deepcopy(evaluation_model_manifest),
                                "evaluation_models": list(evaluation_model_manifest.get("judge_models", [])),
                                "response_generation_mode": zs_response_telemetry["response_generation_mode"],
                                "response_usage_status": zs_response_telemetry["response_usage_status"],
                                "response_usage_note": zs_response_telemetry["response_usage_note"],
                                "scores": zs_score,
                                "judge_runs": zs_judge_runs,
                                "scores_by_judge": zs_aggregated_judges.get("scores_by_judge", {}),
                                "judge_summary": zs_aggregated_judges.get("judge_summary", {}),
                                "response": zs_resp,
                                "tools_used": zs_tools,
                                "retrieved_context": zs_ctx,
                                "reference_context": build_reference_context(
                                    expected_facts=item.get("expected_facts", []),
                                    expected_behavior=item.get("expected_behavior"),
                                ),
                                "response_usage": zs_response_usage,
                                "response_cost_usd": zs_response_cost,
                                "agent_usage": {"zero_shot": deepcopy(zs_response_usage)},
                                "agent_costs": {"zero_shot": deepcopy(zs_response_cost)},
                                "evaluation_usage": zs_evaluation_usage,
                                "evaluation_cost_usd": zs_evaluation_cost,
                                "combined_usage": zs_combined_usage,
                                "combined_cost_usd": zs_combined_cost,
                                "latency": round(zs_lat, 3),
                                "latency_s": round(zs_lat, 3),
                                "error": zs_err,
                                "error_type": categorize_error(zs_err),
                                "tool_metrics": zs_tool_metrics,
                                "heuristics": zs_heuristics,
                            },
                            "lisboa": {
                                "agents_used": ls_agents_used,
                                "response_model": lisboa_response_model,
                                "response_model_config": deepcopy(lisboa_response_model_config),
                                "evaluation_model": evaluation_model_id,
                                "evaluation_model_config": deepcopy(evaluation_model_manifest),
                                "evaluation_models": list(evaluation_model_manifest.get("judge_models", [])),
                                "response_generation_mode": ls_response_telemetry["response_generation_mode"],
                                "response_usage_status": ls_response_telemetry["response_usage_status"],
                                "response_usage_note": ls_response_telemetry["response_usage_note"],
                                "scores": ls_score,
                                "judge_runs": ls_judge_runs,
                                "scores_by_judge": ls_aggregated_judges.get("scores_by_judge", {}),
                                "judge_summary": ls_aggregated_judges.get("judge_summary", {}),
                                "response": ls_resp,
                                "tools_used": ls_tools,
                                "retrieved_context": ls_ctx,
                                "reference_context": build_reference_context(
                                    expected_facts=item.get("expected_facts", []),
                                    expected_behavior=item.get("expected_behavior"),
                                ),
                                "response_usage": ls_response_usage,
                                "response_cost_usd": ls_response_cost,
                                "agent_usage": ls_agent_usage,
                                "agent_costs": ls_agent_costs,
                                "agent_tool_logs": ls_agent_tool_logs,
                                "evaluation_usage": ls_evaluation_usage,
                                "evaluation_cost_usd": ls_evaluation_cost,
                                "combined_usage": ls_combined_usage,
                                "combined_cost_usd": ls_combined_cost,
                                "llm_usage_breakdown": ls_response_usage.get("llm_usage_breakdown", []),
                                "llm_usage_by_agent": ls_agent_usage,
                                "latency": round(ls_lat, 3),
                                "latency_s": round(ls_lat, 3),
                                "error": ls_err,
                                "error_type": categorize_error(ls_err),
                                "tool_metrics": ls_tool_metrics,
                                "heuristics": ls_heuristics,
                            },
                        },
                    }

                    result_record = results_by_id[item["id"]]
                    result_record["comparisons"][profile_key] = comparison_block
                    result_record["comparison_profiles"] = sorted(result_record["comparisons"].keys())
                    if profile_key == primary_profile_key:
                        result_record["comparison_usage"] = comparison_usage
                        result_record["comparison_cost_usd"] = comparison_cost
                        result_record["metrics"] = deepcopy(comparison_block["metrics"])

                    if profile_consecutive_errors >= 2:
                        print(f"\nABORTING PROFILE {profile_key}: API or models are failing continuously.")
                        break

        results = list(results_by_id.values())
        profile_summaries = {
            str(profile["profile_name"]): _build_profile_summary(results, str(profile["profile_name"]))
            for profile in comparison_profiles
        }
        summary = deepcopy(profile_summaries.get(primary_profile_key, {}))
        summary["primary_score"] = {
            "name": "ablation_quality_score",
            "dimensions": list(ABLATION_PRIMARY_SCORE_FIELDS),
            "excluded_dimensions": ["tool_usage"],
            "note": "Ablation primary means exclude Tool Usage so zero-shot and LISBOA are averaged over identical non-tool quality dimensions. Tool metrics are reported separately for LISBOA.",
        }
        summary["comparison_profiles"] = profile_summaries
        summary["primary_comparison_profile"] = primary_profile_key
        summary["comparison_profile_order"] = [str(profile["profile_name"]) for profile in comparison_profiles]

        primary_profile_metadata = profile_metadata.get(primary_profile_key, {})
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = build_results_output_path("ablation", output_prefix, timestamp)
        run_finished_at = datetime.now()
        total_runtime_s = round(time.perf_counter() - run_started_perf, 3)
        write_json_artifact(
            {
                "ablation_metadata": build_run_metadata(
                    resolved_groundtruth_path,
                    groundtruth_queries,
                    response_models={
                        "zero_shot": (primary_profile_metadata.get("zero_shot_model_config", {}) or {}).get("model_id"),
                        "lisboa": (primary_profile_metadata.get("lisboa_response_model_config", {}) or {}).get("display_model"),
                    },
                    evaluation_model=evaluation_model_id,
                    extra={
                        "response_model_configs": {
                            "zero_shot": primary_profile_metadata.get("zero_shot_model_config"),
                            "lisboa": primary_profile_metadata.get("lisboa_response_model_config"),
                        },
                        "comparison_profiles": profile_metadata,
                        "comparison_profile_order": [str(profile["profile_name"]) for profile in comparison_profiles],
                        "primary_comparison_profile": primary_profile_key,
                        "evaluation_models": list(evaluation_model_manifest.get("judge_models", [])),
                        "judge_model_configs": judge_model_manifests,
                        "evaluation_model_config": evaluation_model_manifest,
                        "langsmith_enabled": LANGSMITH_AVAILABLE,
                        "langsmith_project": ablation_langsmith_project,
                        "timestamp": run_finished_at.isoformat(),
                        "run_started_at": run_started_at.isoformat(),
                        "run_finished_at": run_finished_at.isoformat(),
                        "total_runtime_s": total_runtime_s,
                        "comparison": "zero_shot_vs_lisboa_dual_paradigm",
                        "primary_score": {
                            "name": "ablation_quality_score",
                            "dimensions": list(ABLATION_PRIMARY_SCORE_FIELDS),
                            "excluded_dimensions": ["tool_usage"],
                        },
                        "zero_shot_baseline": "lisboa_instruction_no_tools",
                        "real_services": True,
                        "ablation_domains": active_domains,
                        "pricing_model_count": len(pricing_catalog),
                        "output_directory": str(output_path.parent),
                        "output_file": str(output_path),
                        **get_pricing_metadata(pricing_by_model),
                    },
                ),
                "summary": summary,
                "ablation_results": results,
            },
            output_path,
        )

        print(f"\nAblation Study complete. Results saved to {output_path}")
        runtime_failure = get_last_langsmith_runtime_failure()
        if runtime_failure:
            print(
                "[LangSmith] Latest persistence status: "
                f"{runtime_failure.get('persistence_state', 'failed_remote')} - "
                f"{runtime_failure.get('message', '')}"
            )


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
    parser.add_argument(
        "--open-model-spec",
        type=str,
        default=None,
        help="Optional open-model comparison profile in provider::model format, for example azure::Kimi-K2.5.",
    )
    parser.add_argument(
        "--lisboa-provider",
        type=str,
        default=None,
        help="Legacy compatibility flag. Dual-profile ablation now keeps zero-shot and LISBOA within the same provider/model family per profile.",
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
        "--include-domain",
        action="append",
        dest="include_domains",
        help=(
            "Repeatable domain filter for ablation runs. Defaults to weather, transport, "
            "researcher, and multi_agent."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="ablation_results",
        help="Output filename prefix inside eval/results/ablation/.",
    )
    args = parser.parse_args()

    limit = 5 if args.mode == "run_test" else args.limit

    run_ablation(
        limit=limit,
        zero_shot_provider=args.zero_shot_provider,
        zero_shot_model=args.zero_shot_model,
        open_model_spec=args.open_model_spec,
        judge_model_specs=args.judge_model_specs,
        lisboa_provider=args.lisboa_provider,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        groundtruth_path=args.dataset,
        include_domains=args.include_domains or DEFAULT_ABLATION_DOMAINS,
        output_prefix=args.output_prefix,
    )
