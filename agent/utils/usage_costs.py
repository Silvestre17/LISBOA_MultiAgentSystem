# ==========================================================================
# Master Thesis
#   - André Filipe Gomes Silvestre, 20240502
#
# Shared LLM usage and cost utilities.
#
# These helpers are runtime-safe and can be reused by both the interactive
# multi-agent runtime and the evaluation scripts. They keep token normalization,
# pricing resolution, and cost aggregation in one place.
# ==========================================================================

from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

PRICING_METADATA_ALIASES = {
    "source": "pricing_source",
    "pricing_source": "pricing_source",
    "updated_at": "pricing_updated_at",
    "pricing_updated_at": "pricing_updated_at",
    "snapshot_date": "pricing_snapshot_date",
    "pricing_snapshot_date": "pricing_snapshot_date",
}
_LOCAL_PROVIDER_PREFIXES = ("lmstudio::", "local::", "ollama::")
# NOTE:
# These aliases are only for lexical normalization, such as punctuation or
# separator variants of the *same* catalog model identifier. They must not be
# used to collapse different SKUs or different serving modes into one pricing
# record. The JSON pricing catalog remains the source of truth.
_MODEL_PRICING_ALIASES = {
    "azure::kimi-k2-5": ("azure::kimi-k2.5", "kimi-k2.5"),
    "azure::kimi-k2_5": ("azure::kimi-k2.5", "kimi-k2.5"),
    "kimi-k2-5": ("kimi-k2.5",),
    "kimi-k2_5": ("kimi-k2.5",),
}


def _coerce_int(value: Any) -> int:
    """Safely coerce numeric values to integers.

    Args:
        value: Raw numeric-like value.

    Returns:
        int: Parsed integer or zero when coercion fails.
    """
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _coerce_float(value: Any) -> float | None:
    """Safely coerce numeric values to floats.

    Args:
        value: Raw numeric-like value.

    Returns:
        Optional[float]: Parsed float or None when coercion fails.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_money(value: float) -> float:
    """Round USD values while preserving useful precision.

    Args:
        value: Raw floating-point USD value.

    Returns:
        float: Rounded USD value.
    """
    return round(float(value), 10)


def build_model_id(provider: str, model_name: str) -> str:
    """Return a stable provider::model identifier.

    Args:
        provider: Model provider name.
        model_name: Provider model name.

    Returns:
        str: Stable provider-qualified identifier.
    """
    return f"{provider}::{model_name}"


def normalize_model_lookup_key(model_id: str | None) -> str:
    """Normalize a provider/model lookup key for case-insensitive matching.

    Args:
        model_id: Raw model identifier.

    Returns:
        str: Normalized lookup key.
    """
    return str(model_id or "").strip().lower()


def _get_model_lookup_candidates(model_id: str | None) -> list[str]:
    """Return pricing lookup candidates including normalized aliases."""
    normalized_model_id = normalize_model_lookup_key(model_id)
    if not normalized_model_id:
        return []

    candidates = [normalized_model_id]
    if "::" in normalized_model_id:
        candidates.append(normalized_model_id.split("::", 1)[1])

    for alias in _MODEL_PRICING_ALIASES.get(normalized_model_id, ()): 
        if alias not in candidates:
            candidates.append(alias)

    return candidates


def _single_non_null_value(values: Iterable[Any]) -> Any | None:
    """Return the unique non-null value when all observed values agree.

    Args:
        values: Candidate values gathered during an aggregate computation.

    Returns:
        Any | None: The shared value when exactly one unique non-null value is
        present, otherwise None.
    """
    unique_values: list[Any] = []
    for value in values:
        if value is None:
            continue
        if value not in unique_values:
            unique_values.append(value)
        if len(unique_values) > 1:
            return None
    return unique_values[0] if unique_values else None


def normalize_token_usage(usage: Any) -> dict[str, int]:
    """Normalize token usage payloads to input/output/total integers.

    Args:
        usage: Raw usage payload, nested usage dict, or normalized tokens.

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
                candidate.get(
                    "input_tokens",
                    candidate.get("prompt_tokens", candidate.get("input_token_count")),
                )
            ),
        )
        output_tokens = max(
            output_tokens,
            _coerce_int(
                candidate.get(
                    "output_tokens",
                    candidate.get(
                        "completion_tokens",
                        candidate.get("output_token_count"),
                    ),
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
    """Build a stable usage payload for runtime or evaluation artefacts.

    Args:
        usage: Raw usage payload or normalized usage dict.
        model_id: Optional model identifier.
        call_count: Number of represented LLM calls.
        usage_available: Whether token metadata was exposed by the provider.
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
    """Combine multiple usage payloads into a single aggregate record.

    Args:
        payloads: Usage payloads to combine.
        model_id: Optional aggregate model identifier.

    Returns:
        Dict[str, Any]: Aggregate usage payload.
    """
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
    """Split pricing config into a normalized catalog and metadata.

    Args:
        pricing_by_model: Flat or wrapped pricing configuration.

    Returns:
        Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]: Normalized catalog and metadata.
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
    """Return normalized pricing metadata ready for persisted artefacts.

    Args:
        pricing_by_model: Pricing catalog or wrapped pricing config.

    Returns:
        Dict[str, Any]: Normalized metadata.
    """
    _, metadata = split_pricing_config(pricing_by_model)
    normalized: dict[str, Any] = {}
    for key, value in metadata.items():
        canonical = PRICING_METADATA_ALIASES.get(key)
        if canonical and value is not None:
            normalized[canonical] = value
    return normalized


def _build_zero_cost_local_pricing(model_id: str) -> dict[str, Any]:
    """Build an explicit zero-cost pricing record for local providers.

    Args:
        model_id: Provider-qualified local model identifier.

    Returns:
        Dict[str, Any]: Zero-cost pricing record.
    """
    model_name = model_id.split("::", 1)[1] if "::" in model_id else model_id
    return {
        "input": 0.0,
        "output": 0.0,
        "input_cached": 0.0,
        "name": model_name or "local-model",
        "pricing_lookup_key": normalize_model_lookup_key(model_id),
    }


def resolve_model_pricing(
    pricing_by_model: Optional[dict[str, Any]],
    model_id: str | None,
) -> Optional[dict[str, Any]]:
    """Resolve model pricing using exact and model-only lookup fallbacks.

    Local providers such as LM Studio default to zero-cost pricing when the
    model is not explicitly present in the catalog.

    Args:
        pricing_by_model: Pricing catalog or wrapped pricing config.
        model_id: Provider-qualified model identifier.

    Returns:
        Optional[Dict[str, Any]]: Resolved pricing record, if any.
    """
    if not model_id:
        return None

    catalog, _ = split_pricing_config(pricing_by_model)
    normalized_model_id = normalize_model_lookup_key(model_id)
    lookup_candidates = _get_model_lookup_candidates(model_id)

    for lookup_key in lookup_candidates:
        if lookup_key in catalog:
            pricing = deepcopy(catalog[lookup_key])
            pricing["pricing_lookup_key"] = lookup_key
            return pricing

    if normalized_model_id.startswith(_LOCAL_PROVIDER_PREFIXES):
        return _build_zero_cost_local_pricing(normalized_model_id)

    return None


def resolve_usage_model_id(
    usage_entry: dict[str, Any],
    default_model_id: str | None = None,
) -> str | None:
    """Resolve the most specific model identifier available for a usage entry.

    Args:
        usage_entry: Usage breakdown entry.
        default_model_id: Default model identifier fallback.

    Returns:
        Optional[str]: Best available model identifier.
    """
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
    """Build the cost payload for a single-model usage payload.

    Args:
        tokens: Normalized token payload.
        pricing_by_model: Pricing catalog or wrapped pricing config.
        model_id: Provider-qualified model identifier.

    Returns:
        Dict[str, Any]: Stable single-model cost payload.
    """
    pricing = resolve_model_pricing(pricing_by_model, model_id)
    input_price = pricing.get("input") if pricing else None
    output_price = pricing.get("output") if pricing else None
    cached_price = pricing.get("input_cached") if pricing else None

    input_cost = (tokens["input_tokens"] / 1_000_000) * input_price if input_price is not None else 0.0
    output_cost = (tokens["output_tokens"] / 1_000_000) * output_price if output_price is not None else 0.0
    total_tokens = tokens["total_tokens"]
    pricing_complete = total_tokens == 0 or (input_price is not None and output_price is not None)
    missing_models = []
    if total_tokens > 0 and not pricing_complete:
        missing_models = [model_id] if model_id else ["unknown_model"]

    return {
        "model_id": model_id,
        "pricing_lookup_key": pricing.get("pricing_lookup_key") if pricing else None,
        "pricing_found": pricing is not None or total_tokens == 0,
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
    """Build a stable cost payload from usage and pricing.

    Args:
        usage_payload: Usage payload or raw usage dict.
        pricing_by_model: Pricing catalog or wrapped pricing config.
        model_id: Optional override for the aggregate model id.

    Returns:
        Dict[str, Any]: Stable cost payload.
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
        aggregate_model_ids: list[str] = []
        aggregate_pricing_lookup_keys: list[str] = []
        aggregate_input_prices: list[float] = []
        aggregate_output_prices: list[float] = []
        aggregate_cached_input_prices: list[float] = []
        attributed_tokens = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        for entry in breakdown:
            entry_tokens = normalize_token_usage(entry.get("tokens", entry))
            attributed_tokens["input_tokens"] += entry_tokens["input_tokens"]
            attributed_tokens["output_tokens"] += entry_tokens["output_tokens"]
            attributed_tokens["total_tokens"] += entry_tokens["total_tokens"]
            entry_model_id = resolve_usage_model_id(
                entry,
                usage_payload.get("model_id"),
            )
            event_cost = _build_single_model_cost_payload(
                entry_tokens,
                pricing_by_model,
                entry_model_id,
            )
            aggregate_model_ids.append(event_cost.get("model_id"))
            aggregate_pricing_lookup_keys.append(event_cost.get("pricing_lookup_key"))
            aggregate_input_prices.append(event_cost.get("input_per_million_usd"))
            aggregate_output_prices.append(event_cost.get("output_per_million_usd"))
            aggregate_cached_input_prices.append(event_cost.get("cached_input_per_million_usd"))
            total_input_cost += event_cost["input_cost_usd"]
            total_output_cost += event_cost["output_cost_usd"]
            missing_pricing_models.extend(event_cost.get("missing_pricing_models", []))
            pricing_found = pricing_found and bool(
                event_cost.get("pricing_found", False) or entry_tokens["total_tokens"] == 0
            )
            pricing_complete = pricing_complete and bool(
                event_cost.get("pricing_complete", False) or entry_tokens["total_tokens"] == 0
            )
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

        aggregate_tokens = normalize_token_usage(usage_payload.get("tokens", usage_payload))
        remainder_tokens = {
            "input_tokens": max(aggregate_tokens["input_tokens"] - attributed_tokens["input_tokens"], 0),
            "output_tokens": max(aggregate_tokens["output_tokens"] - attributed_tokens["output_tokens"], 0),
            "total_tokens": max(
                aggregate_tokens["total_tokens"] - attributed_tokens["total_tokens"],
                max(aggregate_tokens["input_tokens"] - attributed_tokens["input_tokens"], 0)
                + max(aggregate_tokens["output_tokens"] - attributed_tokens["output_tokens"], 0),
            ),
        }
        if remainder_tokens["total_tokens"] > 0:
            remainder_model_id = model_id or usage_payload.get("model_id")
            if not remainder_model_id:
                breakdown_model_ids = [
                    resolved_model_id
                    for resolved_model_id in (
                        resolve_usage_model_id(entry, usage_payload.get("model_id"))
                        for entry in breakdown
                    )
                    if resolved_model_id
                ]
                unique_breakdown_model_ids = list(dict.fromkeys(breakdown_model_ids))
                if len(unique_breakdown_model_ids) == 1:
                    remainder_model_id = unique_breakdown_model_ids[0]

            remainder_cost = _build_single_model_cost_payload(
                remainder_tokens,
                pricing_by_model,
                remainder_model_id,
            )
            aggregate_model_ids.append(remainder_cost.get("model_id"))
            aggregate_pricing_lookup_keys.append(remainder_cost.get("pricing_lookup_key"))
            aggregate_input_prices.append(remainder_cost.get("input_per_million_usd"))
            aggregate_output_prices.append(remainder_cost.get("output_per_million_usd"))
            aggregate_cached_input_prices.append(remainder_cost.get("cached_input_per_million_usd"))
            total_input_cost += remainder_cost["input_cost_usd"]
            total_output_cost += remainder_cost["output_cost_usd"]
            missing_pricing_models.extend(remainder_cost.get("missing_pricing_models", []))
            pricing_found = pricing_found and bool(
                remainder_cost.get("pricing_found", False) or remainder_tokens["total_tokens"] == 0
            )
            pricing_complete = pricing_complete and bool(
                remainder_cost.get("pricing_complete", False) or remainder_tokens["total_tokens"] == 0
            )

            remainder_provider = None
            remainder_model = None
            if remainder_model_id and "::" in str(remainder_model_id):
                remainder_provider, remainder_model = str(remainder_model_id).split("::", 1)

            cost_breakdown.append(
                {
                    "call_index": len(cost_breakdown) + 1,
                    "agent_name": "unattributed",
                    "provider": remainder_provider,
                    "model": remainder_model,
                    "model_id": remainder_model_id,
                    "tokens": deepcopy(remainder_cost["tokens"]),
                    "pricing_lookup_key": remainder_cost.get("pricing_lookup_key"),
                    "pricing_found": remainder_cost.get("pricing_found", False),
                    "pricing_complete": remainder_cost.get("pricing_complete", False),
                    "input_cost_usd": remainder_cost.get("input_cost_usd", 0.0),
                    "output_cost_usd": remainder_cost.get("output_cost_usd", 0.0),
                    "total_cost_usd": remainder_cost.get("total_cost_usd", 0.0),
                }
            )

        return {
            "model_id": _single_non_null_value(aggregate_model_ids) or model_id or usage_payload.get("model_id"),
            "pricing_lookup_key": _single_non_null_value(aggregate_pricing_lookup_keys),
            "pricing_found": pricing_found,
            "pricing_complete": pricing_complete,
            "tokens": deepcopy(usage_payload.get("tokens", {})),
            "input_per_million_usd": _single_non_null_value(aggregate_input_prices),
            "output_per_million_usd": _single_non_null_value(aggregate_output_prices),
            "cached_input_per_million_usd": _single_non_null_value(aggregate_cached_input_prices),
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
    """Combine multiple cost payloads into a single aggregate record.

    Args:
        payloads: Cost payloads to combine.

    Returns:
        Dict[str, Any]: Aggregate cost payload.
    """
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
    aggregate_model_ids: list[str] = []
    aggregate_pricing_lookup_keys: list[str] = []
    aggregate_input_prices: list[float] = []
    aggregate_output_prices: list[float] = []
    aggregate_cached_input_prices: list[float] = []

    for payload in payloads:
        tokens = normalize_token_usage(payload.get("tokens", payload))
        combined_tokens["input_tokens"] += tokens["input_tokens"]
        combined_tokens["output_tokens"] += tokens["output_tokens"]
        combined_tokens["total_tokens"] += tokens["total_tokens"]
        total_input_cost += float(payload.get("input_cost_usd", 0.0) or 0.0)
        total_output_cost += float(payload.get("output_cost_usd", 0.0) or 0.0)
        aggregate_model_ids.append(payload.get("model_id"))
        aggregate_pricing_lookup_keys.append(payload.get("pricing_lookup_key"))
        aggregate_input_prices.append(payload.get("input_per_million_usd"))
        aggregate_output_prices.append(payload.get("output_per_million_usd"))
        aggregate_cached_input_prices.append(payload.get("cached_input_per_million_usd"))

        payload_tokens = tokens["total_tokens"]
        pricing_found = pricing_found and bool(payload.get("pricing_found", False) or payload_tokens == 0)
        pricing_complete = pricing_complete and bool(payload.get("pricing_complete", False) or payload_tokens == 0)
        missing_pricing_models.extend(payload.get("missing_pricing_models", []))
        if isinstance(payload.get("llm_cost_breakdown"), list):
            cost_breakdown.extend(deepcopy(payload["llm_cost_breakdown"]))

    result = {
        "model_id": _single_non_null_value(aggregate_model_ids),
        "pricing_lookup_key": _single_non_null_value(aggregate_pricing_lookup_keys),
        "pricing_found": pricing_found,
        "pricing_complete": pricing_complete,
        "tokens": combined_tokens,
        "input_per_million_usd": _single_non_null_value(aggregate_input_prices),
        "output_per_million_usd": _single_non_null_value(aggregate_output_prices),
        "cached_input_per_million_usd": _single_non_null_value(aggregate_cached_input_prices),
        "input_cost_usd": _round_money(total_input_cost),
        "output_cost_usd": _round_money(total_output_cost),
        "total_cost_usd": _round_money(total_input_cost + total_output_cost),
        "missing_pricing_models": sorted({model for model in missing_pricing_models if model}),
    }
    if cost_breakdown:
        result["llm_cost_breakdown"] = cost_breakdown
    return result


@lru_cache(maxsize=4)
def load_pricing_catalog(catalog_path: str | Path | None = None) -> dict[str, Any]:
    """Load a versioned local pricing catalog from disk.

    Args:
        catalog_path: Optional explicit JSON path. When omitted, the default
            runtime catalog under ``data/pricing/llm_model_pricing.json`` is used.

    Returns:
        Dict[str, Any]: Parsed pricing config, or an empty dict if unavailable.
    """
    if catalog_path is None:
        root = Path(__file__).resolve().parents[2]
        target_path = root / "data" / "pricing" / "llm_model_pricing.json"
    else:
        target_path = Path(catalog_path)

    try:
        with target_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}

    return payload if isinstance(payload, dict) else {}


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    import math

    print("\033[1m" + "=" * 68 + "\033[0m")
    print("\033[1m🧪 Usage & Cost Utilities Smoke Test\033[0m")
    print("\033[1m" + "=" * 68 + "\033[0m")

    counters = {"passed": 0, "failed": 0}

    def _check(condition: bool, label: str) -> None:
        if condition:
            counters["passed"] += 1
            print(f"\033[1;32m✅ PASS\033[0m: {label}")
        else:
            counters["failed"] += 1
            print(f"\033[1;31m❌ FAIL\033[0m: {label}")

    pricing_catalog = load_pricing_catalog()
    metadata = get_pricing_metadata(pricing_catalog)
    models = pricing_catalog.get("models", {}) if isinstance(pricing_catalog, dict) else {}

    _check(bool(models), "Pricing catalog loads with model entries")
    _check(bool(metadata.get("pricing_snapshot_date")), "Pricing snapshot date is present")

    expected_prices = {
        "azure::deepseek-r1": (1.35, 5.4),
        "azure::phi-4-reasoning-plus": (0.125, 0.5),
        "azure::grok-4": (3.0, 15.0),
        "azure::llama-3.3-70b": (0.71, 0.71),
        "azure::kimi-k2.5": (0.6, 3.0),
        "azure::claude-sonnet-4.5": (3.0, 15.0),
    }

    for model_id, (expected_input, expected_output) in expected_prices.items():
        pricing = resolve_model_pricing(pricing_catalog, model_id)
        _check(pricing is not None, f"Pricing resolved for {model_id}")
        if pricing:
            _check(
                math.isclose(float(pricing.get("input", 0.0) or 0.0), expected_input, rel_tol=0, abs_tol=1e-9)
                and math.isclose(float(pricing.get("output", 0.0) or 0.0), expected_output, rel_tol=0, abs_tol=1e-9),
                f"Expected input/output pricing for {model_id}",
            )

    deepseek_usage = build_usage_payload(
        {"input_tokens": 1_000_000, "output_tokens": 100_000, "total_tokens": 1_100_000},
        model_id="azure::deepseek-r1",
        call_count=1,
    )
    deepseek_cost = build_cost_payload(deepseek_usage, pricing_catalog)
    _check(
        math.isclose(deepseek_cost["total_cost_usd"], 1.89, rel_tol=0, abs_tol=1e-9),
        "DeepSeek R1 cost payload matches expected USD total",
    )

    local_usage = build_usage_payload(
        {"input_tokens": 800, "output_tokens": 200, "total_tokens": 1_000},
        model_id="lmstudio::demo/local-model",
        call_count=1,
    )
    local_cost = build_cost_payload(local_usage, pricing_catalog)
    _check(local_cost["pricing_complete"] is True, "Local providers default to explicit zero-cost pricing")
    _check(math.isclose(local_cost["total_cost_usd"], 0.0, rel_tol=0, abs_tol=1e-12), "Local provider total cost is zero")

    multi_model_usage = build_usage_payload(
        {
            "tokens": {"input_tokens": 2_000, "output_tokens": 400, "total_tokens": 2_400},
            "call_count": 2,
            "usage_available": True,
            "llm_usage_breakdown": [
                {
                    "call_index": 1,
                    "agent_name": "supervisor",
                    "model_id": "azure::gpt-5-mini",
                    "tokens": {"input_tokens": 1_000, "output_tokens": 200, "total_tokens": 1_200},
                    "usage_available": True,
                },
                {
                    "call_index": 2,
                    "agent_name": "researcher",
                    "model_id": "azure::claude-haiku-4.5",
                    "tokens": {"input_tokens": 1_000, "output_tokens": 200, "total_tokens": 1_200},
                    "usage_available": True,
                },
            ],
        }
    )
    multi_model_cost = build_cost_payload(multi_model_usage, pricing_catalog)
    _check(
        len(multi_model_cost.get("llm_cost_breakdown", [])) == 2,
        "Multi-model usage produces a per-call cost breakdown",
    )

    print("\n\033[1mSummary:\033[0m")
    print(f"   Passed: {counters['passed']}")
    print(f"   Failed: {counters['failed']}")
    if counters["failed"]:
        raise SystemExit(1)
    print("\n\033[1;32m✅ Usage & cost utilities smoke test passed!\033[0m")
