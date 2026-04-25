# ==========================================================================
# Master Thesis - Deployment Freshness Utilities
#   - André Filipe Gomes Silvestre, 20240502
#
#   Runtime helpers that keep the Streamlit app aligned with the checked-out
#   GitHub commit and local data files after Community Cloud updates.
# ==========================================================================

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

RUNTIME_MARKER_SCHEMA_VERSION = 1
DEFAULT_MARKER_RELATIVE_PATH = Path(".streamlit") / "runtime_fingerprint.json"
DEFAULT_WATCHED_RELATIVE_PATHS = (
    "app.py",
    "config.py",
    "requirements.txt",
    "pyproject.toml",
    ".streamlit/config.toml",
    "data/pricing/llm_model_pricing.json",
    "data_collection/webscraping/events.json",
    "data_collection/webscraping/places.json",
    "data_collection/webscraping/lisbon_datasets_clean.json",
)
IMPORT_CACHE_PREFIXES = ("agent", "tools", "config")


def _normalise_relative_path(path_value: str | os.PathLike[str]) -> str:
    """Return a POSIX-style relative path for fingerprint payloads.

    Args:
        path_value: Relative path value to normalize.

    Returns:
        POSIX-style path string.
    """
    return Path(path_value).as_posix().lstrip("./")


def _file_stat_payload(root_dir: Path, relative_path: str) -> dict[str, Any]:
    """Return stable file metadata used by the deployment fingerprint.

    Args:
        root_dir: Repository root directory.
        relative_path: Repository-relative path to inspect.

    Returns:
        Dict with path, existence flag, size, and mtime metadata.
    """
    target_path = root_dir / relative_path
    try:
        stat_result = target_path.stat()
    except OSError:
        return {"path": relative_path, "exists": False}

    return {
        "path": relative_path,
        "exists": True,
        "size": int(stat_result.st_size),
        "mtime_ns": int(stat_result.st_mtime_ns),
    }


def _git_commit_from_env() -> Optional[str]:
    """Return a commit SHA exposed by the hosting environment, if available.

    Returns:
        Commit SHA string, or None when no supported variable is present.
    """
    for env_name in ("GITHUB_SHA", "COMMIT_SHA", "SOURCE_VERSION"):
        value = str(os.getenv(env_name) or "").strip()
        if value:
            return value
    return None


def _git_commit_from_command(root_dir: Path) -> Optional[str]:
    """Return the current Git commit by calling git directly.

    Args:
        root_dir: Repository root directory.

    Returns:
        Commit SHA string, or None if git is unavailable.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(root_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    commit_sha = completed.stdout.strip()
    return commit_sha or None


def _git_commit_from_files(root_dir: Path) -> Optional[str]:
    """Return the current Git commit by reading .git metadata.

    Args:
        root_dir: Repository root directory.

    Returns:
        Commit SHA string, or None if metadata is unavailable.
    """
    head_path = root_dir / ".git" / "HEAD"
    try:
        head_value = head_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None

    if not head_value.startswith("ref:"):
        return head_value or None

    ref_name = head_value.split("ref:", 1)[1].strip()
    if not ref_name:
        return None
    try:
        return (root_dir / ".git" / ref_name).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def resolve_git_commit(root_dir: str | os.PathLike[str]) -> Optional[str]:
    """Resolve the checked-out Git commit for the running app.

    Args:
        root_dir: Repository root directory.

    Returns:
        Commit SHA string, or None when unavailable.
    """
    root_path = Path(root_dir).resolve()
    return (
        _git_commit_from_env()
        or _git_commit_from_command(root_path)
        or _git_commit_from_files(root_path)
    )


def compute_deployment_fingerprint(
    root_dir: str | os.PathLike[str],
    watched_relative_paths: Optional[Iterable[str | os.PathLike[str]]] = None,
) -> dict[str, Any]:
    """Build a deterministic fingerprint for the deployed code and data.

    Args:
        root_dir: Repository root directory.
        watched_relative_paths: Optional repository-relative file list to
            include. Defaults to app, configuration, pricing, and scraped data.

    Returns:
        Dict containing the fingerprint hash and the source manifest.
    """
    root_path = Path(root_dir).resolve()
    relative_paths = tuple(
        _normalise_relative_path(path_value)
        for path_value in (watched_relative_paths or DEFAULT_WATCHED_RELATIVE_PATHS)
    )
    files = [_file_stat_payload(root_path, relative_path) for relative_path in relative_paths]
    manifest = {
        "schema_version": RUNTIME_MARKER_SCHEMA_VERSION,
        "git_commit": resolve_git_commit(root_path),
        "files": files,
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**manifest, "fingerprint": hashlib.sha256(encoded).hexdigest()}


def runtime_marker_path(
    root_dir: str | os.PathLike[str],
    marker_path: Optional[str | os.PathLike[str]] = None,
) -> Path:
    """Return the file used to persist the loaded deployment fingerprint.

    Args:
        root_dir: Repository root directory.
        marker_path: Optional explicit marker path.

    Returns:
        Absolute marker path.
    """
    if marker_path is not None:
        return Path(marker_path).resolve()
    return (Path(root_dir).resolve() / DEFAULT_MARKER_RELATIVE_PATH).resolve()


def read_runtime_marker(marker_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read the persisted runtime marker.

    Args:
        marker_path: Marker JSON path.

    Returns:
        Parsed marker dictionary, or an empty dict when unavailable.
    """
    try:
        payload = json.loads(Path(marker_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_runtime_marker(
    marker_path: str | os.PathLike[str],
    fingerprint_payload: dict[str, Any],
) -> bool:
    """Persist the deployment fingerprint marker.

    Args:
        marker_path: Marker JSON path.
        fingerprint_payload: Current fingerprint payload.

    Returns:
        True when the marker was written successfully.
    """
    marker = Path(marker_path)
    payload = {
        "schema_version": RUNTIME_MARKER_SCHEMA_VERSION,
        "fingerprint": fingerprint_payload.get("fingerprint"),
        "git_commit": fingerprint_payload.get("git_commit"),
    }
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(payload, sort_keys=True, indent=2),
            encoding="utf-8",
        )
    except OSError:
        return False
    return True


def fingerprint_changed(
    current_payload: dict[str, Any],
    stored_payload: dict[str, Any],
) -> bool:
    """Return whether the current deployment differs from the stored marker.

    Args:
        current_payload: Current fingerprint payload.
        stored_payload: Persisted marker payload.

    Returns:
        True when no marker exists or the fingerprint changed.
    """
    current = str(current_payload.get("fingerprint") or "").strip()
    stored = str(stored_payload.get("fingerprint") or "").strip()
    return bool(current and current != stored)


def _clear_callable_cache(module_name: str, attribute_name: str) -> bool:
    """Clear a functools cache on a named callable when present.

    Args:
        module_name: Module to import.
        attribute_name: Callable attribute with an optional cache_clear method.

    Returns:
        True when a cache was cleared.
    """
    try:
        module = importlib.import_module(module_name)
        target = getattr(module, attribute_name, None)
        cache_clear = getattr(target, "cache_clear", None)
        if callable(cache_clear):
            cache_clear()
            return True
    except Exception:
        return False
    return False


def _clear_optimization_caches() -> list[str]:
    """Clear shared TTL caches used by API tooling.

    Returns:
        Names of caches that were cleared.
    """
    cleared: list[str] = []
    try:
        optimization = importlib.import_module("agent.utils.optimization")
    except Exception:
        return cleared

    for cache_name in ("weather_cache", "transport_cache", "static_cache"):
        cache = getattr(optimization, cache_name, None)
        cache_clear = getattr(cache, "clear", None)
        if callable(cache_clear):
            try:
                cache_clear()
                cleared.append(cache_name)
            except Exception:
                continue
    return cleared


def _reset_visitlisboa_cache() -> bool:
    """Reset VisitLisboa vector-store and JSON enrichment singletons.

    Returns:
        True when reset logic was applied.
    """
    try:
        module = importlib.import_module("tools.visitlisboa_api")
    except Exception:
        return False

    reset_function = getattr(module, "reset_visitlisboa_runtime_cache", None)
    if callable(reset_function):
        try:
            reset_function()
            return True
        except Exception:
            return False

    try:
        lock = getattr(module, "_vector_store_lock", None)
        if lock is not None:
            with lock:
                setattr(module, "_vector_store", None)
        else:
            setattr(module, "_vector_store", None)
        setattr(module, "_places_cache", None)
    except Exception:
        return False
    return True


def clear_known_runtime_caches(streamlit_module: Optional[Any] = None) -> list[str]:
    """Clear Streamlit and LISBOA runtime caches after a deployment change.

    Args:
        streamlit_module: Optional imported Streamlit module. Passing it avoids
            importing Streamlit from this utility during tests.

    Returns:
        Names of cache groups that were cleared.
    """
    cleared: list[str] = []
    if streamlit_module is not None:
        for cache_name in ("cache_data", "cache_resource"):
            cache_api = getattr(streamlit_module, cache_name, None)
            cache_clear = getattr(cache_api, "clear", None)
            if callable(cache_clear):
                try:
                    cache_clear()
                    cleared.append(f"streamlit.{cache_name}")
                except Exception:
                    continue

    for module_name, attribute_name in (
        ("agent.utils.usage_costs", "load_pricing_catalog"),
        ("agent.utils.langsmith_tracing", "get_langsmith_tracing_status"),
        ("agent.agents.transport_agent", "_get_metro_station_name_map"),
        ("agent.agents.transport_agent", "_get_cp_station_name_map"),
        ("tools.location_resolver", "_fetch_nominatim_results_cached"),
        ("tools.location_resolver", "_get_metro_station_lookup"),
        ("tools.location_resolver", "_get_cp_station_lookup"),
    ):
        if _clear_callable_cache(module_name, attribute_name):
            cleared.append(f"{module_name}.{attribute_name}")

    cleared.extend(f"agent.utils.optimization.{name}" for name in _clear_optimization_caches())
    if _reset_visitlisboa_cache():
        cleared.append("tools.visitlisboa_api.runtime_cache")
    return cleared


def purge_lisboa_import_cache(prefixes: Sequence[str] = IMPORT_CACHE_PREFIXES) -> list[str]:
    """Remove LISBOA modules from Python's import cache before a rerun.

    Args:
        prefixes: Top-level module/package names to purge.

    Returns:
        Sorted module names removed from ``sys.modules``.
    """
    normalized_prefixes = tuple(str(prefix).strip(".") for prefix in prefixes if str(prefix).strip("."))
    removed: list[str] = []
    for module_name in list(sys.modules):
        if any(module_name == prefix or module_name.startswith(f"{prefix}.") for prefix in normalized_prefixes):
            sys.modules.pop(module_name, None)
            removed.append(module_name)
    return sorted(removed)
