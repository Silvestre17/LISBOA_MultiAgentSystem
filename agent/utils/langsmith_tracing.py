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
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Callable, Dict, MutableMapping, Optional, Sequence

from dotenv import load_dotenv

load_dotenv()

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
_DEFAULT_NONINTERACTIVE_TRACING_OPT_IN_ENV = "LISBOA_ENABLE_CLI_LANGSMITH"
_TRUTHY_VALUES = {"1", "true", "yes", "on"}
_LAST_LANGSMITH_RUNTIME_FAILURE: Optional[Dict[str, str]] = None
_LANGSMITH_RUNTIME_HANDLER_INSTALLED = False


def _noop_traceable(*args, **kwargs):
    """Return a no-op decorator compatible with LangSmith's traceable API."""

    def decorator(func):
        return func

    return decorator


def _noop_get_current_run_tree() -> None:
    """Return None when LangSmith tracing is disabled."""
    return None


@contextmanager
def _noop_tracing_context(**_kwargs):
    """Provide a no-op tracing context manager when LangSmith is unavailable."""
    yield


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


def _enable_tracing_env(
    env: MutableMapping[str, str],
    *,
    api_key: str,
    endpoint: str,
    project_name: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> None:
    """Synchronize canonical and legacy tracing env vars for downstream SDKs."""
    env["LANGSMITH_TRACING"] = "true"
    env["LANGCHAIN_TRACING_V2"] = "true"
    env["LANGSMITH_API_KEY"] = api_key
    env["LANGCHAIN_API_KEY"] = api_key
    env["LANGSMITH_ENDPOINT"] = endpoint
    env["LANGCHAIN_ENDPOINT"] = endpoint

    if project_name:
        env["LANGSMITH_PROJECT"] = project_name
        env["LANGCHAIN_PROJECT"] = project_name

    if workspace_id:
        env["LANGSMITH_WORKSPACE_ID"] = workspace_id


def _disabled_status(reason: str, requested: bool) -> Dict[str, Any]:
    """Build the standard disabled tracing payload."""
    return {
        "enabled": False,
        "requested": requested,
        "reason": reason,
        "project_name": None,
        "endpoint": None,
        "workspace_id": None,
        "traceable": _noop_traceable,
        "tracing_context": _noop_tracing_context,
        "get_current_run_tree": _noop_get_current_run_tree,
        "RunTree": None,
        "ContextThreadPoolExecutor": ThreadPoolExecutor,
    }


def _classify_langsmith_runtime_failure(message: str) -> str:
    """Classify a LangSmith runtime persistence failure into a compact state."""
    lowered = (message or "").lower()
    if any(token in lowered for token in ("quota", "credit", "credits", "billing", "429", "rate limit")):
        return "failed_remote_quota"
    if any(token in lowered for token in ("401", "403", "unauthorized", "forbidden", "api key")):
        return "failed_remote_auth"
    if any(token in lowered for token in ("timeout", "connection", "dns", "network", "ssl", "temporarily unavailable")):
        return "failed_remote_network"
    return "failed_remote"


def _record_langsmith_runtime_failure(message: str) -> None:
    """Persist the last observed LangSmith runtime failure for later summaries."""
    global _LAST_LANGSMITH_RUNTIME_FAILURE

    normalized_message = str(message or "").strip()
    if not normalized_message:
        return

    _LAST_LANGSMITH_RUNTIME_FAILURE = {
        "message": normalized_message,
        "persistence_state": _classify_langsmith_runtime_failure(normalized_message),
    }


def get_last_langsmith_runtime_failure() -> Optional[Dict[str, str]]:
    """Return the latest captured LangSmith runtime failure, if any."""
    if not _LAST_LANGSMITH_RUNTIME_FAILURE:
        return None
    return dict(_LAST_LANGSMITH_RUNTIME_FAILURE)


class _LangSmithRuntimeFailureHandler(logging.Handler):
    """Capture LangSmith SDK runtime persistence failures without noisy console spam."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            message = str(getattr(record, "msg", "") or "")

        lowered = message.lower()
        if not any(token in lowered for token in ("langsmith", "smith.langchain", "run", "trace", "project", "ingest", "batch")):
            return
        _record_langsmith_runtime_failure(message)


def _install_langsmith_runtime_handler() -> None:
    """Attach a lightweight runtime-error capture handler to LangSmith loggers once."""
    global _LANGSMITH_RUNTIME_HANDLER_INSTALLED

    if _LANGSMITH_RUNTIME_HANDLER_INSTALLED:
        return

    handler = _LangSmithRuntimeFailureHandler(level=logging.ERROR)
    for logger_name in ("langsmith", "langsmith.client", "langsmith.run_helpers"):
        target_logger = logging.getLogger(logger_name)
        target_logger.setLevel(logging.ERROR)
        target_logger.addHandler(handler)

    _LANGSMITH_RUNTIME_HANDLER_INSTALLED = True


def _load_langsmith_symbols() -> Optional[Dict[str, Any]]:
    """Import LangSmith runtime symbols only when tracing is actually requested."""
    try:
        _install_langsmith_runtime_handler()

        from langsmith import Client, ContextThreadPoolExecutor
        from langsmith.run_helpers import (
            get_current_run_tree,
            traceable,
            tracing_context,
        )
        from langsmith.run_trees import RunTree

        return {
            "Client": Client,
            "traceable": traceable,
            "tracing_context": tracing_context,
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
    workspace_id: Optional[str] = None,
) -> tuple[bool, str]:
    """Validate LangSmith credentials with a short, read-only API probe."""
    try:
        client_kwargs = {
            "api_url": endpoint,
            "api_key": api_key,
            "timeout_ms": (1500, 5000),
            "auto_batch_tracing": False,
        }
        if workspace_id:
            client_kwargs["workspace_id"] = workspace_id

        client = client_cls(**client_kwargs)
        next(iter(client.list_projects(limit=1)), None)
        return True, "LangSmith tracing enabled"
    except Exception as exc:
        message = str(exc)
        lowered = message.lower()
        if "403" in lowered or "forbidden" in lowered:
            return False, "LangSmith tracing disabled: API key is forbidden for this endpoint"
        if "401" in lowered or "unauthorized" in lowered:
            return False, "LangSmith tracing disabled: API key is unauthorized"
        if any(token in lowered for token in ("workspace", "tenant", "multiple workspaces")):
            if workspace_id:
                return False, "LangSmith tracing disabled: workspace is invalid or inaccessible"
            return False, "LangSmith tracing disabled: API key requires LANGSMITH_WORKSPACE_ID"
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
    tracing_requested = _env_flag(runtime_env, "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")

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
    project_name = _get_env_value(
        runtime_env,
        "LANGSMITH_PROJECT",
        "LANGCHAIN_PROJECT",
    )
    workspace_id = _get_env_value(runtime_env, "LANGSMITH_WORKSPACE_ID")

    if _looks_like_placeholder(endpoint):
        _disable_tracing_env(runtime_env)
        return _disabled_status(
            "LangSmith tracing disabled: endpoint is missing or still a placeholder",
            requested=True,
        )

    auth_probe = probe or _probe_langsmith_access
    try:
        is_valid, reason = auth_probe(symbols["Client"], endpoint, api_key, workspace_id)
    except TypeError:
        is_valid, reason = auth_probe(symbols["Client"], endpoint, api_key)

    if not is_valid:
        _disable_tracing_env(runtime_env)
        logger.warning(reason)
        return _disabled_status(reason, requested=True)

    _enable_tracing_env(
        runtime_env,
        api_key=api_key,
        endpoint=endpoint,
        project_name=project_name,
        workspace_id=workspace_id,
    )

    return {
        "enabled": True,
        "requested": True,
        "reason": reason,
        "project_name": project_name or "default",
        "endpoint": endpoint,
        "workspace_id": workspace_id,
        "traceable": symbols["traceable"],
        "tracing_context": symbols.get("tracing_context", _noop_tracing_context),
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


def get_langsmith_project_name(
    status: Optional[Dict[str, Any]] = None,
) -> str:
    """Return the resolved LangSmith project name using canonical env aliases."""
    resolved = status or get_langsmith_tracing_status()
    project_name = resolved.get("project_name")
    if isinstance(project_name, str) and project_name.strip():
        return project_name.strip()

    return (
        _get_env_value(os.environ, "LANGSMITH_PROJECT", "LANGCHAIN_PROJECT")
        or "default"
    )


def get_langsmith_request_tracking_status(
    status: Optional[Dict[str, Any]] = None,
    run_tree: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return a per-request LangSmith tracking summary for terminal analytics.

    This helper reports whether the current request is attached to an active
    LangSmith run context. LangSmith persistence is asynchronous, so the result
    intentionally describes whether a save attempt is in progress, not whether
    remote ingestion or quota acceptance has been synchronously confirmed.

    Args:
        status: Optional pre-resolved LangSmith tracing status.
        run_tree: Optional injected current run tree for tests.

    Returns:
        Dict[str, Any]: Compact per-request tracking metadata.
    """
    resolved = status or get_langsmith_tracing_status()
    reason = str(resolved.get("reason", "") or "")
    current_run = run_tree
    if current_run is None:
        try:
            current_run = get_current_run_tree()
        except Exception:
            current_run = None

    current_run_attached = current_run is not None
    project_name = (
        get_langsmith_project_name(resolved)
        if resolved.get("enabled") or resolved.get("requested")
        else None
    )
    runtime_failure = get_last_langsmith_runtime_failure()

    run_id = None
    if current_run_attached:
        for attr_name in ("id", "run_id", "trace_id"):
            value = getattr(current_run, attr_name, None)
            if value is not None and str(value).strip():
                run_id = str(value).strip()
                break

    if resolved.get("enabled") and current_run_attached and runtime_failure:
        return {
            "tracking_state": "tracking_request_failed_remote",
            "status_label": "enabled",
            "save_attempted": True,
            "persistence_state": runtime_failure.get("persistence_state", "failed_remote"),
            "current_run_attached": True,
            "project_name": project_name,
            "run_id": run_id,
            "reason": reason,
            "note": runtime_failure.get("message", reason),
        }

    if resolved.get("enabled") and current_run_attached:
        return {
            "tracking_state": "tracking_request",
            "status_label": "enabled",
            "save_attempted": True,
            "persistence_state": "unconfirmed",
            "current_run_attached": True,
            "project_name": project_name,
            "run_id": run_id,
            "reason": reason,
            "note": (
                "Run context is attached locally. LangSmith persistence remains unconfirmed and may fail asynchronously, "
                "for example because of remote quota or ingestion limits."
            ),
        }

    if resolved.get("enabled"):
        return {
            "tracking_state": "enabled_no_active_run",
            "status_label": "enabled",
            "save_attempted": False,
            "persistence_state": "not_attached",
            "current_run_attached": False,
            "project_name": project_name,
            "run_id": None,
            "reason": reason,
            "note": "LangSmith is enabled, but this request has no active run context attached.",
        }

    if resolved.get("requested"):
        return {
            "tracking_state": "auto_disabled",
            "status_label": "auto-disabled",
            "save_attempted": False,
            "persistence_state": "not_active",
            "current_run_attached": False,
            "project_name": project_name,
            "run_id": None,
            "reason": reason,
            "note": reason,
        }

    return {
        "tracking_state": "disabled",
        "status_label": "disabled",
        "save_attempted": False,
        "persistence_state": "disabled",
        "current_run_attached": False,
        "project_name": None,
        "run_id": None,
        "reason": reason,
        "note": reason,
    }


def get_langsmith_scoped_project_name(
    scope_label: str,
    *,
    env_name: Optional[str] = None,
    env: Optional[MutableMapping[str, str]] = None,
) -> str:
    """Return a dedicated LangSmith project name for a non-chat workload.

    Args:
        scope_label: Human-friendly scope label such as ``Benchmark`` or
            ``Ablation``.
        env_name: Optional environment variable that explicitly overrides the
            derived project name.
        env: Optional environment mapping for tests or dependency injection.

    Returns:
        str: A project name distinct from the base interactive project.
    """
    runtime_env = env if env is not None else os.environ
    env_names = [env_name] if env_name else []
    explicit_project = _get_env_value(runtime_env, *env_names)
    if explicit_project:
        return explicit_project

    base_project = (
        _get_env_value(runtime_env, "LANGSMITH_PROJECT", "LANGCHAIN_PROJECT")
        or "default"
    )
    normalized_scope = str(scope_label or "").strip()
    if not normalized_scope:
        return base_project

    if base_project == "default":
        return f"LISBOA {normalized_scope}"

    normalized_base = base_project.strip()
    lowered_base = normalized_base.lower()
    lowered_scope = normalized_scope.lower()
    if lowered_base == lowered_scope or lowered_base.endswith(f"- {lowered_scope}"):
        return normalized_base

    return f"{normalized_base} - {normalized_scope}"


def is_langsmith_tracing_opted_in(
    *names: str,
    env: Optional[MutableMapping[str, str]] = None,
    default: bool = False,
) -> bool:
    """Return whether tracing was explicitly opted into for a non-interactive workload.

    Args:
        *names: Optional environment-variable names to check.
        env: Optional environment mapping for tests or dependency injection.
        default: Default value when none of the environment variables are set.

    Returns:
        bool: ``True`` only when one of the opt-in flags is truthy.
    """
    runtime_env = env if env is not None else os.environ
    env_names = names or (_DEFAULT_NONINTERACTIVE_TRACING_OPT_IN_ENV,)
    return _env_flag(runtime_env, *env_names, default=default)


@contextmanager
def tracing_disabled_unless_opted_in(
    *enable_env_names: str,
    env: Optional[MutableMapping[str, str]] = None,
    default: bool = False,
):
    """Disable LangSmith tracing unless a specific opt-in environment flag is set.

    This is intended for offline or low-signal workloads such as tests,
    benchmarks, and verification harnesses. Interactive app traffic can keep
    using the normal tracing configuration, while these batch paths fail closed
    unless the caller explicitly re-enables tracing.

    Args:
        *enable_env_names: Optional environment-variable names that re-enable
            tracing for the wrapped scope.
        env: Optional environment mapping for tests or dependency injection.
        default: Default value when none of the environment variables are set.
    """
    runtime_env = env if env is not None else os.environ
    if is_langsmith_tracing_opted_in(
        *enable_env_names,
        env=runtime_env,
        default=default,
    ):
        yield
        return

    with tracing_context(enabled=False):
        yield


@contextmanager
def tracing_project_override(
    project_name: Optional[str],
    *,
    env: Optional[MutableMapping[str, str]] = None,
):
    """Temporarily route traces into a dedicated LangSmith project.

    This is useful for keeping benchmark, ablation, and other offline studies
    separate from the main interactive chat project while preserving the same
    LangSmith workspace and credentials.

    Args:
        project_name: The project name to apply for the wrapped scope.
        env: Optional environment mapping for tests or dependency injection.
    """
    normalized_project = str(project_name or "").strip()
    if not normalized_project:
        yield
        return

    runtime_env = env if env is not None else os.environ
    had_langsmith_project = "LANGSMITH_PROJECT" in runtime_env
    had_langchain_project = "LANGCHAIN_PROJECT" in runtime_env
    previous_langsmith_project = runtime_env.get("LANGSMITH_PROJECT")
    previous_langchain_project = runtime_env.get("LANGCHAIN_PROJECT")

    runtime_env["LANGSMITH_PROJECT"] = normalized_project
    runtime_env["LANGCHAIN_PROJECT"] = normalized_project
    try:
        with tracing_context(project_name=normalized_project):
            yield
    finally:
        if had_langsmith_project and previous_langsmith_project is not None:
            runtime_env["LANGSMITH_PROJECT"] = previous_langsmith_project
        else:
            runtime_env.pop("LANGSMITH_PROJECT", None)

        if had_langchain_project and previous_langchain_project is not None:
            runtime_env["LANGCHAIN_PROJECT"] = previous_langchain_project
        else:
            runtime_env.pop("LANGCHAIN_PROJECT", None)


def annotate_current_run(
    *,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[Sequence[str]] = None,
) -> bool:
    """Safely attach metadata and tags to the active LangSmith run."""
    try:
        run_tree = get_current_run_tree()
    except Exception as exc:
        logger.debug("Could not access current LangSmith run tree", exc_info=exc)
        return False

    if not run_tree:
        return False

    try:
        if metadata:
            filtered_metadata = {
                key: value
                for key, value in metadata.items()
                if value is not None
            }
            if filtered_metadata:
                add_metadata = getattr(run_tree, "add_metadata", None)
                if callable(add_metadata):
                    add_metadata(filtered_metadata)
                else:
                    current_metadata = getattr(run_tree, "metadata", None)
                    if not isinstance(current_metadata, dict):
                        current_metadata = dict(current_metadata or {})
                        run_tree.metadata = current_metadata
                    current_metadata.update(filtered_metadata)

        if tags:
            normalized_tags = [
                normalized_tag
                for tag in tags
                if (normalized_tag := str(tag).strip())
            ]
            if normalized_tags:
                add_tags = getattr(run_tree, "add_tags", None)
                if callable(add_tags):
                    existing_tags = list(getattr(run_tree, "tags", []) or [])
                    missing_tags = [tag for tag in normalized_tags if tag not in existing_tags]
                    if missing_tags:
                        add_tags(missing_tags)
                else:
                    current_tags = list(getattr(run_tree, "tags", []) or [])
                    for normalized_tag in normalized_tags:
                        if normalized_tag not in current_tags:
                            current_tags.append(normalized_tag)
                    run_tree.tags = current_tags

        return True
    except Exception as exc:
        logger.debug("Failed to annotate current LangSmith run", exc_info=exc)
        return False


LANGSMITH_STATUS = get_langsmith_tracing_status()
LANGSMITH_AVAILABLE = bool(LANGSMITH_STATUS["enabled"])
traceable = LANGSMITH_STATUS["traceable"]
get_current_run_tree = LANGSMITH_STATUS["get_current_run_tree"]
RunTree = LANGSMITH_STATUS["RunTree"]
ContextThreadPoolExecutor = LANGSMITH_STATUS["ContextThreadPoolExecutor"]
tracing_context = LANGSMITH_STATUS["tracing_context"]


__all__ = [
    "annotate_current_run",
    "ContextThreadPoolExecutor",
    "LANGSMITH_AVAILABLE",
    "LANGSMITH_STATUS",
    "RunTree",
    "get_current_run_tree",
    "get_langsmith_display_state",
    "get_langsmith_project_name",
    "get_langsmith_request_tracking_status",
    "get_langsmith_scoped_project_name",
    "get_langsmith_tracing_status",
    "is_langsmith_tracing_opted_in",
    "resolve_langsmith_tracing_status",
    "traceable",
    "tracing_disabled_unless_opted_in",
    "tracing_project_override",
    "tracing_context",
]


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    print("\033[1m" + "=" * 68 + "\033[0m")
    print("\033[1m🧪 LangSmith Tracing Smoke Test\033[0m")
    print("\033[1m" + "=" * 68 + "\033[0m")

    counters = {"passed": 0, "failed": 0}

    def _check(condition: bool, label: str) -> None:
        if condition:
            counters["passed"] += 1
            print(f"\033[1;32m✅ PASS\033[0m: {label}")
        else:
            counters["failed"] += 1
            print(f"\033[1;31m❌ FAIL\033[0m: {label}")

    class _FakeExecutor:
        pass

    class _FakeRunTree:
        id = "run_123"

    class _FakeClient:
        pass

    def _fake_traceable(*_args, **_kwargs):
        def decorator(func):
            return func
        return decorator

    fake_symbols = {
        "Client": _FakeClient,
        "traceable": _fake_traceable,
        "tracing_context": _noop_tracing_context,
        "get_current_run_tree": _noop_get_current_run_tree,
        "RunTree": _FakeRunTree,
        "ContextThreadPoolExecutor": _FakeExecutor,
    }

    disabled_status = resolve_langsmith_tracing_status(
        env={"LANGCHAIN_TRACING_V2": "false", "LANGCHAIN_API_KEY": "dummy"},
        imported_symbols=fake_symbols,
    )
    enabled_status = resolve_langsmith_tracing_status(
        env={
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_API_KEY": "lsv2-real-looking-key",
            "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
            "LANGCHAIN_PROJECT": "thesis-traces",
        },
        imported_symbols=fake_symbols,
        probe=lambda client_cls, endpoint, api_key: (True, "LangSmith tracing enabled"),
    )
    request_status = get_langsmith_request_tracking_status(
        status=enabled_status,
        run_tree=_FakeRunTree(),
    )

    _check(disabled_status["enabled"] is False and disabled_status["requested"] is False, "Disabled tracing stays off when env flag is false")
    _check(enabled_status["enabled"] is True and enabled_status["project_name"] == "thesis-traces", "Tracing enables with valid injected credentials")
    _check(get_langsmith_display_state(enabled_status)["state"] == "active", "Display state resolves to active when tracing is enabled")
    _check(request_status["save_attempted"] is True and request_status["run_id"] == "run_123", "Per-request tracking reports an attached active run")
    _check(is_langsmith_tracing_opted_in(env={"LISBOA_ENABLE_CLI_LANGSMITH": "true"}) is True, "Opt-in helper recognises truthy batch tracing flag")

    print("\n\033[1mSummary:\033[0m")
    print(f"   Passed: {counters['passed']}")
    print(f"   Failed: {counters['failed']}")
    if counters["failed"]:
        raise SystemExit(1)
    print("\n\033[1;32m✅ LangSmith tracing smoke test passed!\033[0m")
