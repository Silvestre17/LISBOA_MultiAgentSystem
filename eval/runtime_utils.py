# ===========================================================================
# Master Thesis - Evaluation Runtime Utilities
#   - André Filipe Gomes Silvestre, 20240502
#
# Lightweight helpers shared by benchmark and ablation scripts.
# These utilities stay inside eval/ and do NOT affect application runtime.
# ===========================================================================

from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Optional

from tools import __all__ as EXPORTED_TOOL_NAMES

RESULTS_ROOT = Path(__file__).with_name("results")
PRICING_METADATA_ALIASES = {
    "source": "pricing_source",
    "pricing_source": "pricing_source",
    "updated_at": "pricing_updated_at",
    "pricing_updated_at": "pricing_updated_at",
    "snapshot_date": "pricing_snapshot_date",
    "pricing_snapshot_date": "pricing_snapshot_date",
}


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
                candidate.get("input_tokens", candidate.get("prompt_tokens", candidate.get("input_token_count")))
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

    input_cost = (tokens["input_tokens"] / 1_000_000) * input_price if input_price is not None else 0.0
    output_cost = (tokens["output_tokens"] / 1_000_000) * output_price if output_price is not None else 0.0
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
            entry_model_id = entry.get("model_id") or usage_payload.get("model_id")
            event_cost = _build_single_model_cost_payload(
                entry_tokens,
                pricing_by_model,
                entry_model_id,
            )
            total_input_cost += event_cost["input_cost_usd"]
            total_output_cost += event_cost["output_cost_usd"]
            missing_pricing_models.extend(event_cost.get("missing_pricing_models", []))
            pricing_found = pricing_found and bool(event_cost.get("pricing_found", False) or entry_tokens["total_tokens"] == 0)
            pricing_complete = pricing_complete and bool(event_cost.get("pricing_complete", False) or entry_tokens["total_tokens"] == 0)
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
        pricing_found = pricing_found and bool(payload.get("pricing_found", False) or payload_tokens == 0)
        pricing_complete = pricing_complete and bool(payload.get("pricing_complete", False) or payload_tokens == 0)
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


def compute_tool_metrics(expected: list[str], actual: list[str]) -> dict[str, float]:
    """Compute deterministic Precision, Recall, F1 for tool usage."""
    if not expected and not actual:
        return {"tool_precision": 1.0, "tool_recall": 1.0, "tool_f1": 1.0}
    if not expected:
        return {"tool_precision": 0.0, "tool_recall": 1.0, "tool_f1": 0.0}
    if not actual:
        return {"tool_precision": 1.0, "tool_recall": 0.0, "tool_f1": 0.0}

    expected_set = set(expected)
    actual_set = set(actual)
    tp = len(expected_set & actual_set)
    precision = tp / len(actual_set) if actual_set else 0.0
    recall = tp / len(expected_set) if expected_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "tool_precision": round(precision, 3),
        "tool_recall": round(recall, 3),
        "tool_f1": round(f1, 3),
    }


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
