# ===========================================================================
# Master Thesis - Evaluation Runtime Utilities
#   - André Filipe Gomes Silvestre, 20240502
#
# Lightweight helpers shared by benchmark and ablation scripts.
# These utilities stay inside eval/ and do NOT affect application runtime.
# ===========================================================================

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import ast
from collections import Counter
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

RESULTS_ROOT = Path(__file__).with_name("results")
SUPPORTED_MODEL_PROVIDERS = frozenset({"azure", "openai", "lmstudio"})
JUDGE_NUMERIC_SCORE_FIELDS = (
    "factual_accuracy",
    "tool_usage",
    "completeness",
    "relevance",
    "response_quality",
    "composite_score",
)
JUDGE_SCORE_FIELDS = JUDGE_NUMERIC_SCORE_FIELDS + ("reasoning",)
PRICING_METADATA_ALIASES = {
    "source": "pricing_source",
    "pricing_source": "pricing_source",
    "updated_at": "pricing_updated_at",
    "pricing_updated_at": "pricing_updated_at",
    "snapshot_date": "pricing_snapshot_date",
    "pricing_snapshot_date": "pricing_snapshot_date",
}
JSON_MONEY_FIELD_NAMES = (
    "input_cost_usd",
    "output_cost_usd",
    "total_cost_usd",
    "input_per_million_usd",
    "output_per_million_usd",
    "cached_input_per_million_usd",
)
MIN_JSON_MONEY_DECIMALS = 5
_JSON_MONEY_FIELD_PATTERN = re.compile(
    rf'(?P<prefix>\s*"(?:{"|".join(JSON_MONEY_FIELD_NAMES)})"\s*:\s*)(?P<value>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?P<suffix>\s*,?\s*)$'
)


def _load_exported_tool_names() -> tuple[str, ...]:
    """Read ``tools.__all__`` without importing LangChain-backed tool modules."""
    tools_init_path = Path(__file__).resolve().parents[1] / "tools" / "__init__.py"
    try:
        module_ast = ast.parse(tools_init_path.read_text(encoding="utf-8"))
    except OSError:
        return ()

    for node in module_ast.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
            continue
        try:
            exported_names = ast.literal_eval(node.value)
        except (SyntaxError, ValueError):
            return ()
        return tuple(str(name) for name in exported_names)

    return ()


EXPORTED_TOOL_NAMES = _load_exported_tool_names()


def _load_shared_usage_costs_module() -> Any:
    """Load shared usage-cost helpers without importing the full agent package.

    Importing ``agent.utils`` executes ``agent/__init__.py``, which imports the
    interactive LangChain runtime. The evaluation notebooks only need the
    standalone usage-cost helpers, so loading this file directly keeps the
    analysis path independent from optional application runtime dependencies.
    """
    module_path = Path(__file__).resolve().parents[1] / "agent" / "utils" / "usage_costs.py"
    spec = importlib.util.spec_from_file_location("_lisboa_shared_usage_costs", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load shared usage-cost helpers from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_shared_usage_costs = _load_shared_usage_costs_module()


def _coerce_int(value: Any) -> int:
    """Safely coerces numeric values to integers."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _coerce_float(value: Any) -> float | None:
    """Safely coerces numeric values to floats."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_money(value: float) -> float:
    """Rounds small USD values while preserving useful precision."""
    return round(float(value), 10)


def _round_score(value: float) -> float:
    """Round averaged judge scores while preserving useful report precision."""
    return round(float(value), 4)


def format_json_money_number(
    value: Any,
    *,
    min_decimal_places: int = MIN_JSON_MONEY_DECIMALS,
) -> str:
    """Format a USD numeric literal with a minimum number of decimal places."""
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return str(value)

    literal = format(decimal_value, "f")
    if "." not in literal:
        return f"{literal}.{'0' * min_decimal_places}"

    integer_part, fractional_part = literal.split(".", 1)
    trimmed_fractional = fractional_part.rstrip("0")
    required_length = max(len(trimmed_fractional), min_decimal_places)
    normalized_fractional = trimmed_fractional.ljust(required_length, "0")
    return f"{integer_part}.{normalized_fractional}"


def serialize_json_artifact(payload: Any) -> str:
    """Serialize an evaluation artefact with readable USD field precision."""
    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    formatted_lines: list[str] = []
    for line in serialized.splitlines():
        match = _JSON_MONEY_FIELD_PATTERN.match(line)
        if not match:
            formatted_lines.append(line)
            continue
        formatted_lines.append(
            f"{match.group('prefix')}{format_json_money_number(match.group('value'))}{match.group('suffix')}"
        )
    return "\n".join(formatted_lines) + "\n"


def write_json_artifact(payload: Any, output_path: str | Path) -> None:
    """Persist a JSON artefact with stable minimum-decimal USD formatting."""
    Path(output_path).write_text(
        serialize_json_artifact(payload),
        encoding="utf-8",
    )


def normalize_model_lookup_key(model_id: str | None) -> str:
    """Normalizes a provider/model lookup key for case-insensitive matching."""
    return str(model_id or "").strip().lower()


def normalize_token_usage(usage: Any) -> dict[str, int]:
    """
    Normalizes token usage payloads to input/output/total integers.

    Args:
        usage: Raw usage payload, nested usage dict, or already-normalized tokens.

    Returns:
        Dict[str, int]: Normalized token counts.
    """
    candidates: list[dict[str, Any]] = []
    if isinstance(usage, dict):
        candidates.append(usage)
        for key in ("tokens", "usage", "usage_metadata", "token_usage"):
            nested = usage.get(key)
            if isinstance(nested, dict):
                candidates.append(nested)

    input_tokens = 0
    output_tokens = 0
    total_tokens = 0

    for candidate in candidates:
        input_tokens = max(
            input_tokens,
            _coerce_int(
                candidate.get("input_tokens", candidate.get(
                    "prompt_tokens", candidate.get("input_token_count")))
            ),
        )
        output_tokens = max(
            output_tokens,
            _coerce_int(
                candidate.get(
                    "output_tokens",
                    candidate.get("completion_tokens", candidate.get("output_token_count")),
                )
            ),
        )
        total_tokens = max(
            total_tokens,
            _coerce_int(candidate.get("total_tokens", candidate.get("total_token_count"))),
        )

    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def build_usage_payload(
    usage: Any,
    *,
    model_id: str | None = None,
    call_count: int = 0,
    usage_available: bool | None = None,
    llm_usage_breakdown: Optional[list[dict[str, Any]]] = None,
    by_agent: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Builds a stable usage payload for persisted evaluation artefacts.

    Args:
        usage: Raw usage payload or normalized usage dict.
        model_id: Optional model identifier.
        call_count: Number of LLM calls represented in the payload.
        usage_available: Whether the provider exposed token usage metadata.
        llm_usage_breakdown: Optional per-call breakdown list.
        by_agent: Optional per-agent usage mapping.

    Returns:
        Dict[str, Any]: Stable usage payload with token counts.
    """
    tokens = normalize_token_usage(usage)
    usage_dict = usage if isinstance(usage, dict) else {}
    breakdown = llm_usage_breakdown
    if breakdown is None and isinstance(usage_dict.get("llm_usage_breakdown"), list):
        breakdown = deepcopy(usage_dict["llm_usage_breakdown"])

    if call_count == 0:
        call_count = _coerce_int(usage_dict.get("call_count", len(breakdown or [])))

    if usage_available is None:
        usage_available = bool(
            usage_dict.get("usage_available", False)
            or tokens["total_tokens"] > 0
            or breakdown
        )

    payload = {
        "model_id": model_id or usage_dict.get("model_id"),
        "call_count": call_count,
        "usage_available": bool(usage_available),
        "tokens": tokens,
    }
    if breakdown is not None:
        payload["llm_usage_breakdown"] = breakdown
    if by_agent is not None:
        payload["by_agent"] = deepcopy(by_agent)
    elif isinstance(usage_dict.get("by_agent"), dict):
        payload["by_agent"] = deepcopy(usage_dict["by_agent"])
    return payload


def combine_usage_payloads(
    payloads: Iterable[dict[str, Any]],
    *,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Combines multiple usage payloads into a single aggregate record."""
    combined_tokens = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    call_count = 0
    usage_available = False
    breakdown: list[dict[str, Any]] = []

    for payload in payloads:
        tokens = normalize_token_usage(payload.get("tokens", payload))
        combined_tokens["input_tokens"] += tokens["input_tokens"]
        combined_tokens["output_tokens"] += tokens["output_tokens"]
        combined_tokens["total_tokens"] += tokens["total_tokens"]
        call_count += _coerce_int(payload.get("call_count", 0))
        usage_available = usage_available or bool(payload.get("usage_available", False))
        if isinstance(payload.get("llm_usage_breakdown"), list):
            breakdown.extend(deepcopy(payload["llm_usage_breakdown"]))

    return build_usage_payload(
        combined_tokens,
        model_id=model_id,
        call_count=call_count,
        usage_available=usage_available,
        llm_usage_breakdown=breakdown or None,
    )


def split_pricing_config(
    pricing_by_model: Optional[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """
    Splits pricing config into a normalized model catalog and metadata.

    Supports either a flat mapping of ``model_id -> price dict`` or a wrapped
    structure with a top-level ``models`` key plus metadata.
    """
    if not pricing_by_model:
        return {}, {}

    raw_catalog: dict[str, Any]
    metadata: dict[str, Any] = {}
    if isinstance(pricing_by_model.get("models"), dict):
        raw_catalog = pricing_by_model["models"]
        metadata = {k: v for k, v in pricing_by_model.items() if k != "models"}
    else:
        raw_catalog = {}
        for key, value in pricing_by_model.items():
            if isinstance(value, dict) and any(
                field in value
                for field in (
                    "input",
                    "output",
                    "input_cached",
                    "input_per_million_usd",
                    "output_per_million_usd",
                    "cached_input_per_million_usd",
                )
            ):
                raw_catalog[key] = value
            else:
                metadata[key] = value

    catalog: dict[str, dict[str, Any]] = {}
    for key, value in raw_catalog.items():
        normalized_key = normalize_model_lookup_key(key)
        catalog[normalized_key] = {
            "input": _coerce_float(value.get("input", value.get("input_per_million_usd"))),
            "output": _coerce_float(value.get("output", value.get("output_per_million_usd"))),
            "input_cached": _coerce_float(
                value.get("input_cached", value.get("cached_input_per_million_usd"))
            ),
            "name": value.get("name", key),
        }

    return catalog, metadata


def get_pricing_metadata(pricing_by_model: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Returns normalized pricing metadata ready for persisted artefacts."""
    _, metadata = split_pricing_config(pricing_by_model)
    normalized: dict[str, Any] = {}
    for key, value in metadata.items():
        canonical = PRICING_METADATA_ALIASES.get(key)
        if canonical and value is not None:
            normalized[canonical] = value
    return normalized


def resolve_model_pricing(
    pricing_by_model: Optional[dict[str, Any]],
    model_id: str | None,
) -> Optional[dict[str, Any]]:
    """Resolves model pricing using exact and model-only lookup fallbacks."""
    if not pricing_by_model or not model_id:
        return None

    catalog, _ = split_pricing_config(pricing_by_model)
    normalized_model_id = normalize_model_lookup_key(model_id)
    lookup_candidates = [normalized_model_id]
    if "::" in normalized_model_id:
        lookup_candidates.append(normalized_model_id.split("::", 1)[1])

    for lookup_key in lookup_candidates:
        if lookup_key in catalog:
            pricing = deepcopy(catalog[lookup_key])
            pricing["pricing_lookup_key"] = lookup_key
            return pricing

    return None


def resolve_usage_model_id(
    usage_entry: dict[str, Any],
    default_model_id: str | None = None,
) -> str | None:
    """Resolves the most specific model identifier available for a usage entry."""
    model_id = usage_entry.get("model_id")
    if model_id:
        return str(model_id)

    provider = usage_entry.get("provider")
    model = usage_entry.get("model")
    if provider and model:
        return build_model_id(str(provider), str(model))

    return default_model_id


def _build_single_model_cost_payload(
    tokens: dict[str, int],
    pricing_by_model: Optional[dict[str, Any]],
    model_id: str | None,
) -> dict[str, Any]:
    """Builds the cost payload for a single-model usage payload."""
    pricing = resolve_model_pricing(pricing_by_model, model_id)
    input_price = pricing.get("input") if pricing else None
    output_price = pricing.get("output") if pricing else None
    cached_price = pricing.get("input_cached") if pricing else None

    input_cost = (tokens["input_tokens"] / 1_000_000) * \
        input_price if input_price is not None else 0.0
    output_cost = (tokens["output_tokens"] / 1_000_000) * \
        output_price if output_price is not None else 0.0
    total_tokens = tokens["total_tokens"]
    pricing_complete = input_price is not None and output_price is not None
    missing_models = []
    if total_tokens > 0 and not pricing_complete:
        missing_models = [model_id] if model_id else ["unknown_model"]

    return {
        "model_id": model_id,
        "pricing_lookup_key": pricing.get("pricing_lookup_key") if pricing else None,
        "pricing_found": pricing is not None,
        "pricing_complete": pricing_complete,
        "tokens": deepcopy(tokens),
        "input_per_million_usd": input_price,
        "output_per_million_usd": output_price,
        "cached_input_per_million_usd": cached_price,
        "input_cost_usd": _round_money(input_cost),
        "output_cost_usd": _round_money(output_cost),
        "total_cost_usd": _round_money(input_cost + output_cost),
        "missing_pricing_models": missing_models,
    }


def build_cost_payload(
    usage_payload: dict[str, Any],
    pricing_by_model: Optional[dict[str, Any]],
    *,
    model_id: str | None = None,
) -> dict[str, Any]:
    """
    Builds a stable cost payload from a usage payload and pricing config.

    Supports both single-model usage payloads and multi-model payloads with a
    per-call ``llm_usage_breakdown``.
    """
    usage_payload = build_usage_payload(usage_payload, model_id=model_id)
    breakdown = usage_payload.get("llm_usage_breakdown")
    if isinstance(breakdown, list) and breakdown:
        total_input_cost = 0.0
        total_output_cost = 0.0
        missing_pricing_models: list[str] = []
        cost_breakdown = []
        pricing_found = True
        pricing_complete = True

        for entry in breakdown:
            entry_tokens = normalize_token_usage(entry.get("tokens", entry))
            entry_model_id = resolve_usage_model_id(
                entry,
                usage_payload.get("model_id"),
            )
            event_cost = _build_single_model_cost_payload(
                entry_tokens,
                pricing_by_model,
                entry_model_id,
            )
            total_input_cost += event_cost["input_cost_usd"]
            total_output_cost += event_cost["output_cost_usd"]
            missing_pricing_models.extend(event_cost.get("missing_pricing_models", []))
            pricing_found = pricing_found and bool(event_cost.get(
                "pricing_found", False) or entry_tokens["total_tokens"] == 0)
            pricing_complete = pricing_complete and bool(event_cost.get(
                "pricing_complete", False) or entry_tokens["total_tokens"] == 0)
            cost_breakdown.append(
                {
                    "call_index": entry.get("call_index"),
                    "agent_name": entry.get("agent_name"),
                    "provider": entry.get("provider"),
                    "model": entry.get("model"),
                    "model_id": entry_model_id,
                    "tokens": deepcopy(event_cost["tokens"]),
                    "pricing_lookup_key": event_cost.get("pricing_lookup_key"),
                    "pricing_found": event_cost.get("pricing_found", False),
                    "pricing_complete": event_cost.get("pricing_complete", False),
                    "input_cost_usd": event_cost.get("input_cost_usd", 0.0),
                    "output_cost_usd": event_cost.get("output_cost_usd", 0.0),
                    "total_cost_usd": event_cost.get("total_cost_usd", 0.0),
                }
            )

        return {
            "model_id": model_id or usage_payload.get("model_id"),
            "pricing_lookup_key": None,
            "pricing_found": pricing_found,
            "pricing_complete": pricing_complete,
            "tokens": deepcopy(usage_payload.get("tokens", {})),
            "input_per_million_usd": None,
            "output_per_million_usd": None,
            "cached_input_per_million_usd": None,
            "input_cost_usd": _round_money(total_input_cost),
            "output_cost_usd": _round_money(total_output_cost),
            "total_cost_usd": _round_money(total_input_cost + total_output_cost),
            "missing_pricing_models": sorted({model for model in missing_pricing_models if model}),
            "llm_cost_breakdown": cost_breakdown,
        }

    return _build_single_model_cost_payload(
        normalize_token_usage(usage_payload.get("tokens", usage_payload)),
        pricing_by_model,
        model_id or usage_payload.get("model_id"),
    )


def combine_cost_payloads(payloads: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Combines multiple cost payloads into a single aggregate record."""
    combined_tokens = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    total_input_cost = 0.0
    total_output_cost = 0.0
    pricing_found = True
    pricing_complete = True
    missing_pricing_models: list[str] = []
    cost_breakdown: list[dict[str, Any]] = []

    for payload in payloads:
        tokens = normalize_token_usage(payload.get("tokens", payload))
        combined_tokens["input_tokens"] += tokens["input_tokens"]
        combined_tokens["output_tokens"] += tokens["output_tokens"]
        combined_tokens["total_tokens"] += tokens["total_tokens"]
        total_input_cost += float(payload.get("input_cost_usd", 0.0) or 0.0)
        total_output_cost += float(payload.get("output_cost_usd", 0.0) or 0.0)

        payload_tokens = tokens["total_tokens"]
        pricing_found = pricing_found and bool(payload.get(
            "pricing_found", False) or payload_tokens == 0)
        pricing_complete = pricing_complete and bool(
            payload.get("pricing_complete", False) or payload_tokens == 0)
        missing_pricing_models.extend(payload.get("missing_pricing_models", []))
        if isinstance(payload.get("llm_cost_breakdown"), list):
            cost_breakdown.extend(deepcopy(payload["llm_cost_breakdown"]))

    result = {
        "model_id": None,
        "pricing_lookup_key": None,
        "pricing_found": pricing_found,
        "pricing_complete": pricing_complete,
        "tokens": combined_tokens,
        "input_per_million_usd": None,
        "output_per_million_usd": None,
        "cached_input_per_million_usd": None,
        "input_cost_usd": _round_money(total_input_cost),
        "output_cost_usd": _round_money(total_output_cost),
        "total_cost_usd": _round_money(total_input_cost + total_output_cost),
        "missing_pricing_models": sorted({model for model in missing_pricing_models if model}),
    }
    if cost_breakdown:
        result["llm_cost_breakdown"] = cost_breakdown
    return result


def fingerprint_payload(payload: Any) -> str:
    """Return a stable SHA-256 fingerprint for a JSON-serializable payload."""
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_model_id(provider: str, model_name: str) -> str:
    """Return a stable provider::model identifier used in result artefacts."""
    return f"{provider}::{model_name}"


def parse_model_spec(
    model_spec: str,
    *,
    temperature: float | None = None,
    supported_providers: Optional[Sequence[str]] = None,
) -> dict[str, str | float]:
    """Parse a provider-qualified model spec into a stable config dict.

    Args:
        model_spec: Model spec in ``provider::model`` or ``provider:model`` format.
        temperature: Optional temperature override.
        supported_providers: Optional allowed provider family list.

    Returns:
        Dict containing ``provider``, ``model``, and ``temperature``.

    Raises:
        ValueError: If the spec is empty, malformed, or uses an unsupported provider.
    """
    normalized_spec = str(model_spec or "").strip()
    if not normalized_spec:
        raise ValueError("Model spec cannot be empty.")

    separator = "::" if "::" in normalized_spec else ":" if ":" in normalized_spec else None
    if separator is None:
        raise ValueError(
            "Model spec must use 'provider::model' or 'provider:model' format."
        )

    provider, model_name = [part.strip() for part in normalized_spec.split(separator, 1)]
    normalized_provider = provider.lower()
    allowed = set(supported_providers or SUPPORTED_MODEL_PROVIDERS)
    if normalized_provider not in allowed:
        raise ValueError(
            f"Unsupported provider '{provider}'. Expected one of: {sorted(allowed)}"
        )
    if not model_name:
        raise ValueError("Model spec is missing the model name.")

    return {
        "provider": normalized_provider,
        "model": model_name,
        "temperature": 0.0 if temperature is None else float(temperature),
    }


def resolve_model_specs(
    default_models: Sequence[dict[str, Any]],
    model_specs: list[str] | None = None,
    *,
    temperature: float | None = None,
    supported_providers: Optional[Sequence[str]] = None,
) -> list[dict[str, str | float]]:
    """Resolve a repeatable list of CLI model specs against default configs."""
    if not model_specs:
        resolved_defaults = deepcopy(list(default_models))
        if temperature is not None:
            for model_config in resolved_defaults:
                model_config["temperature"] = float(temperature)
        return resolved_defaults

    resolved_models: list[dict[str, str | float]] = []
    seen_configs: set[tuple[str, str, float]] = set()
    for model_spec in model_specs:
        parsed = parse_model_spec(
            model_spec,
            temperature=temperature,
            supported_providers=supported_providers,
        )
        dedupe_key = (
            str(parsed["provider"]),
            str(parsed["model"]),
            float(parsed["temperature"]),
        )
        if dedupe_key in seen_configs:
            continue
        seen_configs.add(dedupe_key)
        resolved_models.append(parsed)

    return resolved_models


def build_model_manifest(
    provider: str,
    model_name: str,
    temperature: float | None = None,
    *,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Return a compact, explicit model manifest for persisted artefacts."""
    manifest: dict[str, Any] = {
        "model_id": build_model_id(provider, model_name),
        "provider": provider,
        "model": model_name,
    }
    if temperature is not None:
        manifest["temperature"] = temperature
    if extra:
        manifest.update(extra)
    return manifest


def build_multi_judge_manifest(
    judge_model_manifests: Sequence[dict[str, Any]],
    *,
    aggregation: str = "mean",
) -> dict[str, Any]:
    """Build a compact synthetic manifest describing a multi-judge average."""
    judge_models = [manifest.get("model_id")
                    for manifest in judge_model_manifests if manifest.get("model_id")]
    return {
        "kind": "judge_average",
        "model_id": f"judge_average::{len(judge_models)}",
        "aggregation": aggregation,
        "judge_models": judge_models,
        "judge_model_configs": deepcopy(list(judge_model_manifests)),
    }


def aggregate_judge_runs(judge_runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate judge-level outputs into compatibility-friendly average fields."""
    normalized_runs = [deepcopy(run) for run in judge_runs]
    successful_runs = []
    for run in normalized_runs:
        if run.get("error"):
            continue
        scores = run.get("scores", {})
        if not isinstance(scores, dict):
            continue
        if any(scores.get(field) is not None for field in JUDGE_NUMERIC_SCORE_FIELDS):
            successful_runs.append(run)

    averaged_scores: dict[str, Any] = {field: None for field in JUDGE_SCORE_FIELDS}
    if successful_runs:
        for field in JUDGE_NUMERIC_SCORE_FIELDS:
            values = [
                float(run.get("scores", {}).get(field))
                for run in successful_runs
                if run.get("scores", {}).get(field) is not None
            ]
            averaged_scores[field] = _round_score(sum(values) / len(values)) if values else None

        reasoning_parts = []
        for run in successful_runs:
            judge_model = str(run.get("judge_model") or "unknown_judge")
            reasoning = str(run.get("scores", {}).get("reasoning") or "").strip()
            if reasoning:
                reasoning_parts.append(f"[{judge_model}] {reasoning}")
        averaged_scores["reasoning"] = (
            "\n\n".join(reasoning_parts)
            if reasoning_parts
            else "No successful judge reasoning available."
        )
    else:
        averaged_scores["reasoning"] = "No successful judge evaluations were available."

    scores_by_judge: dict[str, Any] = {}
    for index, run in enumerate(normalized_runs, start=1):
        judge_model = str(run.get("judge_model") or f"judge_{index}")
        unique_key = judge_model if judge_model not in scores_by_judge else f"{judge_model}#{index}"
        scores_by_judge[unique_key] = deepcopy(run.get("scores", {}))

    return {
        "scores": averaged_scores,
        "scores_by_judge": scores_by_judge,
        "evaluation_usage": combine_usage_payloads(
            [run.get("evaluation_usage", {}) for run in normalized_runs]
        ),
        "evaluation_cost_usd": combine_cost_payloads(
            [run.get("evaluation_cost_usd", {}) for run in normalized_runs]
        ),
        "judge_summary": {
            "judge_count": len(normalized_runs),
            "successful_judges": len(successful_runs),
            "failed_judges": len(normalized_runs) - len(successful_runs),
            "all_judges_succeeded": bool(normalized_runs) and len(successful_runs) == len(normalized_runs),
            "averaging_method": "mean",
            "judge_models": [
                run.get("judge_model")
                for run in normalized_runs
                if run.get("judge_model")
            ],
        },
    }


def ensure_results_dir(result_type: str) -> Path:
    """Create and return the output directory for a result category."""
    target_dir = RESULTS_ROOT / result_type
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def build_results_output_path(
    result_type: str,
    prefix: str,
    timestamp: str,
    suffix: str = ".json",
) -> Path:
    """Build an output path inside eval/results/<result_type>/."""
    return ensure_results_dir(result_type) / f"{prefix}_{timestamp}{suffix}"


def compute_dataset_fingerprint(dataset: list[dict[str, Any]]) -> str:
    """Return a fingerprint for the benchmark dataset content."""
    return fingerprint_payload(dataset)


def compute_tool_registry_fingerprint(tool_names: Optional[Iterable[str]] = None) -> str:
    """Return a fingerprint for the authoritative tool registry."""
    registry = sorted(tool_names or EXPORTED_TOOL_NAMES)
    return fingerprint_payload(registry)


def _compute_single_tool_match(expected: Sequence[str], actual: Sequence[str]) -> tuple[float, float, float]:
    """Return Precision, Recall, F1 for one expected-tool set."""
    expected_set = set(expected)
    actual_set = set(actual)

    if not expected_set and not actual_set:
        return 1.0, 1.0, 1.0
    if not expected_set:
        return 0.0, 1.0, 0.0
    if not actual_set:
        return 1.0, 0.0, 0.0

    true_positives = len(expected_set & actual_set)
    precision = true_positives / len(actual_set) if actual_set else 0.0
    recall = true_positives / len(expected_set) if expected_set else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def _normalize_tool_sets(tool_sets: Optional[Sequence[Sequence[str]]]) -> list[list[str]]:
    """Return unique tool sets with deterministic ordering."""
    normalized: list[list[str]] = []
    for tool_set in tool_sets or []:
        candidate = sorted(set(tool_set))
        if candidate not in normalized:
            normalized.append(candidate)
    return normalized


def compute_tool_metrics(
    expected: list[str],
    actual: list[str],
    *,
    acceptable_tool_sets: Optional[Sequence[Sequence[str]]] = None,
    tool_expectation: str = "strict",
) -> dict[str, Any]:
    """Compute deterministic tool metrics for strict expectations only.

    In ``strict`` mode, the best match across ``expected`` and any
    ``acceptable_tool_sets`` is scored. In ``flexible`` mode, the dataset only
    records reference tools and deterministic tool scoring is skipped.
    """
    normalized_expected = sorted(set(expected))
    normalized_actual = sorted(set(actual))
    normalized_alternatives = _normalize_tool_sets(acceptable_tool_sets)

    if tool_expectation == "flexible":
        return {
            "tool_precision": None,
            "tool_recall": None,
            "tool_f1": None,
            "tool_metric_scored": False,
            "tool_expectation": tool_expectation,
            "matched_expected_tools": normalized_expected,
        }

    candidate_sets = _normalize_tool_sets(
        [normalized_expected] if (normalized_expected or not normalized_alternatives) else []
    )
    for candidate in normalized_alternatives:
        if candidate not in candidate_sets:
            candidate_sets.append(candidate)

    if not candidate_sets:
        candidate_sets = [[]]

    best_expected = candidate_sets[0]
    best_scores = _compute_single_tool_match(best_expected, normalized_actual)
    best_key = (best_scores[2], best_scores[1], best_scores[0], -len(best_expected))

    for candidate in candidate_sets[1:]:
        candidate_scores = _compute_single_tool_match(candidate, normalized_actual)
        candidate_key = (
            candidate_scores[2],
            candidate_scores[1],
            candidate_scores[0],
            -len(candidate),
        )
        if candidate_key > best_key:
            best_expected = candidate
            best_scores = candidate_scores
            best_key = candidate_key

    precision, recall, f1 = best_scores
    return {
        "tool_precision": round(precision, 3),
        "tool_recall": round(recall, 3),
        "tool_f1": round(f1, 3),
        "tool_metric_scored": True,
        "tool_expectation": tool_expectation,
        "matched_expected_tools": best_expected,
    }


def build_reference_context(
    *,
    expected_facts: Sequence[str] | None = None,
    expected_behavior: str | None = None,
) -> str:
    """Build a symmetric judge reference from dataset facts and behavior notes.

    This reference is passed to the judge for both zero-shot and grounded
    LISBOA responses. Retrieved tool context can add evidence, but absence of
    retrieved context should not leave zero-shot factuality without a benchmark
    reference.
    """
    sections: list[str] = []
    facts = [str(fact).strip() for fact in expected_facts or [] if str(fact).strip()]
    if facts:
        sections.append("Expected facts:\n" + "\n".join(f"- {fact}" for fact in facts))

    behavior = str(expected_behavior or "").strip()
    if behavior:
        sections.append(f"Expected behavior or limitation:\n- {behavior}")

    return "\n\n".join(sections) if sections else "No explicit reference facts or behavior were provided."


def select_balanced_subset(
    records: list[dict[str, Any]],
    limit: int | None,
    *,
    group_key: str,
) -> list[dict[str, Any]]:
    """Return a stable round-robin subset balanced across a grouping key."""
    if limit is None or limit >= len(records):
        return records

    grouped_records: dict[str, list[dict[str, Any]]] = {}
    group_order: list[str] = []
    for record in records:
        group_value = str(record.get(group_key, "ungrouped"))
        if group_value not in grouped_records:
            grouped_records[group_value] = []
            group_order.append(group_value)
        grouped_records[group_value].append(record)

    subset: list[dict[str, Any]] = []
    while len(subset) < limit and any(grouped_records[group] for group in group_order):
        for group in group_order:
            if grouped_records[group]:
                subset.append(grouped_records[group].pop(0))
                if len(subset) == limit:
                    break
    return subset


def categorize_error(error: str | None) -> str | None:
    """Map raw runtime errors to stable evaluation categories."""
    if not error:
        return None

    text = error.lower()

    if "setup error" in text or "not found in .env" in text or "not configured" in text:
        return "setup_error"
    if "401" in text or "403" in text or "unauthorized" in text or "forbidden" in text:
        return "auth_error"
    if "api key" in text or "credentials" in text:
        return "auth_error"
    if "429" in text or "rate limit" in text:
        return "rate_limit"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "connection" in text or "network" in text or "dns" in text:
        return "network_error"
    if "no isolated agent for domain" in text:
        return "unsupported_domain"
    if "judge" in text:
        return "judge_error"
    return "execution_error"


def summarize_error_categories(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Count error categories across benchmark/ablation records."""
    counter: Counter[str] = Counter()
    for record in records:
        category = record.get("error_type") or categorize_error(record.get("error"))
        if category:
            counter[category] += 1
    return dict(sorted(counter.items()))


def build_run_metadata(
    groundtruth_queries_path: str | Path,
    groundtruth_queries: list[dict[str, Any]],
    *,
    response_models: Optional[Any] = None,
    evaluation_model: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build lightweight reproducibility metadata for evaluation outputs."""
    metadata: dict[str, Any] = {
        "groundtruth_queries_path": str(groundtruth_queries_path),
        "groundtruth_queries_count": len(groundtruth_queries),
        "groundtruth_queries_fingerprint": compute_dataset_fingerprint(groundtruth_queries),
        "total_queries": len(groundtruth_queries),
        "tool_registry_count": len(EXPORTED_TOOL_NAMES),
        "tool_registry_fingerprint": compute_tool_registry_fingerprint(),
    }
    if response_models is not None:
        metadata["response_models"] = response_models
    if evaluation_model is not None:
        metadata["evaluation_model"] = evaluation_model
    if extra:
        metadata.update(extra)
    return metadata


# Keep evaluation cost/usage helpers aligned with the runtime-safe shared module.

globals().update(
    {
        "PRICING_METADATA_ALIASES": _shared_usage_costs.PRICING_METADATA_ALIASES,
        "normalize_model_lookup_key": _shared_usage_costs.normalize_model_lookup_key,
        "normalize_token_usage": _shared_usage_costs.normalize_token_usage,
        "build_usage_payload": _shared_usage_costs.build_usage_payload,
        "combine_usage_payloads": _shared_usage_costs.combine_usage_payloads,
        "split_pricing_config": _shared_usage_costs.split_pricing_config,
        "get_pricing_metadata": _shared_usage_costs.get_pricing_metadata,
        "resolve_model_pricing": _shared_usage_costs.resolve_model_pricing,
        "resolve_usage_model_id": _shared_usage_costs.resolve_usage_model_id,
        "build_cost_payload": _shared_usage_costs.build_cost_payload,
        "combine_cost_payloads": _shared_usage_costs.combine_cost_payloads,
        "build_model_id": _shared_usage_costs.build_model_id,
        "load_pricing_catalog": _shared_usage_costs.load_pricing_catalog,
    }
)

# ===========================================================================
# Test Block
# ===========================================================================
if __name__ == "__main__":
    print("\033[1m" + "=" * 68 + "\033[0m")
    print("\033[1m🧪 Evaluation Runtime Utilities Smoke Test\033[0m")
    print("\033[1m" + "=" * 68 + "\033[0m")

    counters = {"passed": 0, "failed": 0}

    def _check(condition: bool, label: str) -> None:
        if condition:
            counters["passed"] += 1
            print(f"\033[1;32m✅ PASS\033[0m: {label}")
        else:
            counters["failed"] += 1
            print(f"\033[1;31m❌ FAIL\033[0m: {label}")

    usage = build_usage_payload(
        {"prompt_tokens": 120, "completion_tokens": 30},
        model_id="azure::gpt-5-mini",
        call_count=1,
    )
    pricing = {
        "models": {
            "azure::gpt-5-mini": {"input": 0.25, "output": 2.0},
            "azure::claude-haiku-4.5": {"input": 1.0, "output": 5.0},
        },
        "pricing_snapshot_date": "2026-03-19",
    }
    cost = build_cost_payload(usage, pricing)
    metrics = compute_tool_metrics(
        ["get_weather_forecast", "get_metro_status"], ["get_metro_status"])
    metadata = build_run_metadata(
        groundtruth_queries_path="eval/evaluation_groundtruth_queries.json",
        groundtruth_queries=[{"domain": "weather", "query": "test"}],
        response_models={"weather": "azure::gpt-5-mini"},
    )

    _check(usage["tokens"]["total_tokens"] == 150,
           "Usage normalization handles prompt/completion aliases")
    _check(cost["pricing_complete"] is True and cost["total_cost_usd"]
           > 0, "Cost payload computes a positive total")
    _check(metrics["tool_precision"] == 1.0 and metrics["tool_recall"]
           == 0.5, "Tool metrics remain deterministic")
    _check(bool(metadata.get("tool_registry_fingerprint")),
           "Run metadata includes tool registry fingerprint")
    _check(fingerprint_payload({"a": 1}) == fingerprint_payload(
        {"a": 1}), "Fingerprinting is stable")

    print("\n\033[1mSummary:\033[0m")
    print(f"   Passed: {counters['passed']}")
    print(f"   Failed: {counters['failed']}")
    if counters["failed"]:
        raise SystemExit(1)
    print("\n\033[1;32m✅ Evaluation runtime utilities smoke test passed!\033[0m")
