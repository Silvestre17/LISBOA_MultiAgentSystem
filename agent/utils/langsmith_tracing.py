# ==========================================================================
# Master Thesis
#   - André Filipe Gomes Silvestre, 20240502
#
# LangSmith tracing bootstrap helpers.
# Safely enables tracing only when it is explicitly requested and the
# credentials pass a lightweight preflight check. If tracing is misconfigured,
# it gracefully falls back to no-op tracing primitives to avoid noisy runtime
# errors during local development and test runs.
# ==========================================================================

# Required libraries:
# pip install langsmith

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Callable, Dict, MutableMapping, Optional

logger = logging.getLogger(__name__)

_PLACEHOLDER_TOKENS = (
    "your_",
    "placeholder",
    "changeme",
    "example",
    "<your",
    "replace_me",
)
_DEFAULT_LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
_TRUTHY_VALUES = {"1", "true", "yes", "on"}


def _noop_traceable(*args, **kwargs):
    """Return a no-op decorator compatible with LangSmith's traceable API."""

    def decorator(func):
        return func

    return decorator


def _noop_get_current_run_tree() -> None:
    """Return None when LangSmith tracing is disabled."""
    return None


def _get_env_value(env: MutableMapping[str, str], *names: str) -> Optional[str]:
    """Return the first non-empty environment variable value for the given names."""
    for name in names:
        value = env.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _env_flag(env: MutableMapping[str, str], *names: str, default: bool = False) -> bool:
    """Parse boolean-like environment variables using a small truthy set."""
    value = _get_env_value(env, *names)
    if value is None:
        return default
    return value.strip().lower() in _TRUTHY_VALUES


def _looks_like_placeholder(value: Optional[str]) -> bool:
    """Return True when a configuration value still looks like template text."""
    if value is None:
        return True

    normalized = value.strip()
    if not normalized:
        return True

    lowered = normalized.lower()
    return any(token in lowered for token in _PLACEHOLDER_TOKENS)


def _disable_tracing_env(env: MutableMapping[str, str]) -> None:
    """Turn off LangSmith tracing flags in-place for the current process."""
    env["LANGCHAIN_TRACING_V2"] = "false"
    env["LANGSMITH_TRACING"] = "false"


def _disabled_status(reason: str, requested: bool) -> Dict[str, Any]:
    """Build the standard disabled tracing payload."""
    return {
        "enabled": False,
        "requested": requested,
        "reason": reason,
        "traceable": _noop_traceable,
        "get_current_run_tree": _noop_get_current_run_tree,
        "RunTree": None,
        "ContextThreadPoolExecutor": ThreadPoolExecutor,
    }


def _load_langsmith_symbols() -> Optional[Dict[str, Any]]:
    """Import LangSmith runtime symbols only when tracing is actually requested."""
    try:
        from langsmith import Client, ContextThreadPoolExecutor
        from langsmith.run_helpers import get_current_run_tree, traceable
        from langsmith.run_trees import RunTree

        return {
            "Client": Client,
            "traceable": traceable,
            "get_current_run_tree": get_current_run_tree,
            "RunTree": RunTree,
            "ContextThreadPoolExecutor": ContextThreadPoolExecutor,
        }
    except ImportError:
        return None


def _probe_langsmith_access(
    client_cls: Any,
    endpoint: str,
    api_key: str,
) -> tuple[bool, str]:
    """Validate LangSmith credentials with a short, read-only API probe."""
    try:
        client = client_cls(
            api_url=endpoint,
            api_key=api_key,
            timeout_ms=(1000, 2500),
            auto_batch_tracing=False,
        )
        next(iter(client.list_projects(limit=1)), None)
        return True, "LangSmith tracing enabled"
    except Exception as exc:
        message = str(exc)
        lowered = message.lower()
        if "403" in lowered or "forbidden" in lowered:
            return False, "LangSmith tracing disabled: API key is forbidden for this endpoint"
        if "401" in lowered or "unauthorized" in lowered:
            return False, "LangSmith tracing disabled: API key is unauthorized"
        return False, f"LangSmith tracing disabled: preflight check failed ({exc.__class__.__name__})"


def resolve_langsmith_tracing_status(
    env: Optional[MutableMapping[str, str]] = None,
    imported_symbols: Optional[Dict[str, Any]] = None,
    probe: Optional[Callable[[Any, str, str], tuple[bool, str]]] = None,
) -> Dict[str, Any]:
    """Resolve whether LangSmith tracing should be active for this process.

    Args:
        env: Optional mutable environment mapping for dependency injection.
        imported_symbols: Optional injected LangSmith symbol table for tests.
        probe: Optional injected preflight probe function for tests.

    Returns:
        Dict[str, Any]: Tracing status plus the correct tracing primitives.
    """
    runtime_env = env if env is not None else os.environ
    tracing_requested = _env_flag(runtime_env, "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING")

    if not tracing_requested:
        _disable_tracing_env(runtime_env)
        return _disabled_status("LangSmith tracing is disabled by environment", requested=False)

    symbols = imported_symbols if imported_symbols is not None else _load_langsmith_symbols()
    if not symbols:
        _disable_tracing_env(runtime_env)
        return _disabled_status("LangSmith package is not installed", requested=True)

    api_key = _get_env_value(runtime_env, "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")
    if _looks_like_placeholder(api_key):
        _disable_tracing_env(runtime_env)
        return _disabled_status(
            "LangSmith tracing disabled: API key is missing or still a placeholder",
            requested=True,
        )

    endpoint = _get_env_value(
        runtime_env,
        "LANGSMITH_ENDPOINT",
        "LANGCHAIN_ENDPOINT",
    ) or _DEFAULT_LANGSMITH_ENDPOINT

    if _looks_like_placeholder(endpoint):
        _disable_tracing_env(runtime_env)
        return _disabled_status(
            "LangSmith tracing disabled: endpoint is missing or still a placeholder",
            requested=True,
        )

    auth_probe = probe or _probe_langsmith_access
    is_valid, reason = auth_probe(symbols["Client"], endpoint, api_key)
    if not is_valid:
        _disable_tracing_env(runtime_env)
        logger.warning(reason)
        return _disabled_status(reason, requested=True)

    return {
        "enabled": True,
        "requested": True,
        "reason": reason,
        "traceable": symbols["traceable"],
        "get_current_run_tree": symbols["get_current_run_tree"],
        "RunTree": symbols["RunTree"],
        "ContextThreadPoolExecutor": symbols["ContextThreadPoolExecutor"],
    }


@lru_cache(maxsize=1)
def get_langsmith_tracing_status() -> Dict[str, Any]:
    """Return the process-wide LangSmith tracing status with memoization."""
    return resolve_langsmith_tracing_status()


def get_langsmith_display_state(
    status: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Classify tracing status into a small set of UI-friendly sidebar states.

    Args:
        status: Optional pre-resolved LangSmith tracing status.

    Returns:
        Dict[str, str]: A compact payload with ``state`` and ``reason``.
    """
    resolved = status or get_langsmith_tracing_status()
    reason = str(resolved.get("reason", "") or "")
    lowered = reason.lower()

    if resolved.get("enabled"):
        return {"state": "active", "reason": reason}

    if not resolved.get("requested"):
        return {"state": "disabled", "reason": reason}

    if "endpoint is" in lowered or "endpoint missing" in lowered or "preflight check failed" in lowered:
        return {"state": "auto_disabled_invalid_configuration", "reason": reason}

    if any(token in lowered for token in ("api key", "forbidden", "unauthorized", "placeholder")):
        return {"state": "auto_disabled_invalid_credentials", "reason": reason}

    if "package is not installed" in lowered:
        return {"state": "auto_disabled_missing_package", "reason": reason}

    return {"state": "auto_disabled", "reason": reason}


LANGSMITH_STATUS = get_langsmith_tracing_status()
LANGSMITH_AVAILABLE = bool(LANGSMITH_STATUS["enabled"])
traceable = LANGSMITH_STATUS["traceable"]
get_current_run_tree = LANGSMITH_STATUS["get_current_run_tree"]
RunTree = LANGSMITH_STATUS["RunTree"]
ContextThreadPoolExecutor = LANGSMITH_STATUS["ContextThreadPoolExecutor"]


__all__ = [
    "ContextThreadPoolExecutor",
    "LANGSMITH_AVAILABLE",
    "LANGSMITH_STATUS",
    "RunTree",
    "get_current_run_tree",
    "get_langsmith_display_state",
    "get_langsmith_tracing_status",
    "resolve_langsmith_tracing_status",
    "traceable",
]
