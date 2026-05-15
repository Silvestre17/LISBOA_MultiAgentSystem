# ==========================================================================
# Master Thesis - Transport Runtime Release Artifact Loader
#   - André Filipe Gomes Silvestre, 20240502
#
#   Restores last-known-good transport runtime databases from GitHub Release
#   assets when hosted deployments start without local GTFS SQLite files and
#   the live operator feed is temporarily unavailable.
# ==========================================================================

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# Source: https://docs.github.com/pt/rest/about-the-rest-api/api-versions?apiVersion=2026-03-10#specifying-an-api-version
GITHUB_API_BASE_URL = "https://api.github.com"
DEFAULT_RELEASE_TAG = "transport-data-latest"
DEFAULT_RELEASE_REPO = "Silvestre17/LISBOA_MultiAgentSystem"
GITHUB_API_VERSION = "2022-11-28"                                           
_LOCKS: dict[str, threading.Lock] = {}


@dataclass(frozen=True)
class RuntimeReleaseStatus:
    """Status returned by a transport runtime release restore attempt."""

    ok: bool
    attempted: bool
    restored: bool
    message: str
    target_dir: Path
    repo: str
    tag: str
    asset: str
    sha256: str | None = None
    size_bytes: int | None = None


@dataclass(frozen=True)
class _RuntimeReleaseConfig:
    """Resolved configuration for a transport runtime release asset."""

    enabled: bool
    repo: str
    tag: str
    asset: str
    token: str | None
    timeout_seconds: int


def _env_bool(name: str, default: bool) -> bool:
    """Return a boolean parsed from an environment variable."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    return default


def _first_env(names: Sequence[str], default: str | None = None) -> str | None:
    """Return the first non-empty environment variable value from ``names``."""
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return default


def _env_int(names: Sequence[str], default: int) -> int:
    """Return the first valid integer environment value from ``names``."""
    value = _first_env(names)
    if value is None:
        return default

    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _resolve_config(env_prefix: str, default_asset: str) -> _RuntimeReleaseConfig:
    """Resolve release configuration for an operator-specific runtime asset."""
    repo = _first_env(
        [
            f"{env_prefix}_REPO",
            "TRANSPORT_DATA_RELEASE_REPO",
            "VECTOR_DB_RELEASE_REPO",
            "GITHUB_REPOSITORY",
        ],
        DEFAULT_RELEASE_REPO,
    )
    tag = _first_env(
        [f"{env_prefix}_TAG", "TRANSPORT_DATA_RELEASE_TAG"],
        DEFAULT_RELEASE_TAG,
    )
    asset = _first_env([f"{env_prefix}_ASSET"], default_asset)
    token = _first_env(
        [
            f"{env_prefix}_TOKEN",
            "TRANSPORT_DATA_RELEASE_TOKEN",
            "VECTOR_DB_RELEASE_TOKEN",
            "GITHUB_RELEASE_TOKEN",
            "GITHUB_TOKEN",
        ],
    )
    timeout_seconds = _env_int(
        [
            f"{env_prefix}_TIMEOUT_SECONDS",
            "TRANSPORT_DATA_RELEASE_TIMEOUT_SECONDS",
            "VECTOR_DB_RELEASE_TIMEOUT_SECONDS",
        ],
        120,
    )

    return _RuntimeReleaseConfig(
        enabled=_env_bool(f"{env_prefix}_ENABLED", True),
        repo=repo or DEFAULT_RELEASE_REPO,
        tag=tag or DEFAULT_RELEASE_TAG,
        asset=asset or default_asset,
        token=token,
        timeout_seconds=timeout_seconds,
    )


def _required_files_present(base_dir: Path, required_files: Sequence[str]) -> bool:
    """Return whether all required files exist and are non-empty."""
    for filename in required_files:
        path = base_dir / filename
        if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
            return False
    return True


def _build_headers(token: str | None, accept: str) -> dict[str, str]:
    """Build GitHub API request headers."""
    headers = {
        "Accept": accept,
        "User-Agent": "lisboa-transport-runtime-loader",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _request_json(url: str, config: _RuntimeReleaseConfig) -> dict[str, Any]:
    """Request JSON from the GitHub API."""
    request = urllib.request.Request(
        url,
        headers=_build_headers(config.token, "application/vnd.github+json"),
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _resolve_asset_download_url(config: _RuntimeReleaseConfig) -> str:
    """Resolve the API URL for the configured release asset."""
    if "/" not in config.repo:
        raise ValueError("Transport release repo must be in 'owner/repo' format.")

    release_url = f"{GITHUB_API_BASE_URL}/repos/{config.repo}/releases/tags/{config.tag}"
    release = _request_json(release_url, config)

    for asset in release.get("assets", []):
        if asset.get("name") == config.asset and asset.get("url"):
            return str(asset["url"])

    raise FileNotFoundError(
        f"Release asset '{config.asset}' was not found in '{config.repo}@{config.tag}'."
    )


def _download_file(
    url: str,
    destination: Path,
    config: _RuntimeReleaseConfig,
) -> tuple[str, int]:
    """Download a release asset and return SHA-256 plus byte size."""
    request = urllib.request.Request(
        url,
        headers=_build_headers(config.token, "application/octet-stream"),
        method="GET",
    )
    digest = hashlib.sha256()
    size_bytes = 0

    with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
        with destination.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                size_bytes += len(chunk)

    return digest.hexdigest(), size_bytes


def _safe_extract_zip(zip_path: Path, destination: Path) -> None:
    """Extract a zip archive while preventing path traversal."""
    destination_resolved = destination.resolve()

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise ValueError(f"Unsafe path in transport runtime archive: {member.filename}")
        archive.extractall(destination)


def _merge_extracted_files(source_dir: Path, target_dir: Path) -> None:
    """Copy extracted runtime files into the target directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.iterdir():
        target = target_dir / source.name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)


def _utc_now_iso() -> str:
    """Return current UTC time in ISO-8601 form."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _annotate_metadata(
    *,
    target_dir: Path,
    metadata_filename: str,
    operator_name: str,
    config: _RuntimeReleaseConfig,
    sha256: str,
    size_bytes: int,
) -> None:
    """Add release-restore metadata to the operator metadata file."""
    metadata_path = target_dir / metadata_filename
    metadata: dict[str, Any] = {}

    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}

    metadata["_runtime_release_restore"] = {
        "operator": operator_name,
        "repo": config.repo,
        "tag": config.tag,
        "asset": config.asset,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "restored_at": _utc_now_iso(),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def ensure_runtime_data_from_release(
    *,
    operator_name: str,
    target_dir: Path,
    required_files: Sequence[str],
    env_prefix: str,
    default_asset: str,
    metadata_filename: str = "metadata.json",
) -> RuntimeReleaseStatus:
    """Restore an operator runtime database from a GitHub Release asset.

    Args:
        operator_name: Human-readable operator name used in diagnostics.
        target_dir: Runtime directory where files should be installed.
        required_files: Files that must exist after restore.
        env_prefix: Prefix for operator-specific environment variables.
        default_asset: Default release asset name.
        metadata_filename: Metadata file to annotate after restore.

    Returns:
        Restore status with diagnostic details.
    """
    config = _resolve_config(env_prefix=env_prefix, default_asset=default_asset)
    lock = _LOCKS.setdefault(env_prefix, threading.Lock())

    with lock:
        if _required_files_present(target_dir, required_files):
            return RuntimeReleaseStatus(
                ok=True,
                attempted=False,
                restored=False,
                message=f"{operator_name} runtime files are already available.",
                target_dir=target_dir,
                repo=config.repo,
                tag=config.tag,
                asset=config.asset,
            )

        if not config.enabled:
            return RuntimeReleaseStatus(
                ok=False,
                attempted=False,
                restored=False,
                message=f"{operator_name} runtime release restore is disabled.",
                target_dir=target_dir,
                repo=config.repo,
                tag=config.tag,
                asset=config.asset,
            )

        try:
            asset_url = _resolve_asset_download_url(config)
        except (OSError, ValueError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            return RuntimeReleaseStatus(
                ok=False,
                attempted=True,
                restored=False,
                message=f"Could not resolve {operator_name} runtime release asset: {exc}",
                target_dir=target_dir,
                repo=config.repo,
                tag=config.tag,
                asset=config.asset,
            )

        with tempfile.TemporaryDirectory(prefix=f"lisboa_{env_prefix.lower()}_") as temp_root:
            temp_dir = Path(temp_root)
            archive_path = temp_dir / config.asset
            extract_dir = temp_dir / "extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)

            try:
                sha256, size_bytes = _download_file(asset_url, archive_path, config)
                _safe_extract_zip(archive_path, extract_dir)

                if not _required_files_present(extract_dir, required_files):
                    raise FileNotFoundError(
                        f"{operator_name} runtime asset is missing required files: "
                        f"{', '.join(required_files)}"
                    )

                _merge_extracted_files(extract_dir, target_dir)

                if not _required_files_present(target_dir, required_files):
                    raise FileNotFoundError(
                        f"{operator_name} runtime files were not installed correctly."
                    )

                _annotate_metadata(
                    target_dir=target_dir,
                    metadata_filename=metadata_filename,
                    operator_name=operator_name,
                    config=config,
                    sha256=sha256,
                    size_bytes=size_bytes,
                )
            except (
                OSError,
                ValueError,
                zipfile.BadZipFile,
                urllib.error.URLError,
                urllib.error.HTTPError,
            ) as exc:
                return RuntimeReleaseStatus(
                    ok=False,
                    attempted=True,
                    restored=False,
                    message=f"Could not restore {operator_name} runtime release asset: {exc}",
                    target_dir=target_dir,
                    repo=config.repo,
                    tag=config.tag,
                    asset=config.asset,
                )

        return RuntimeReleaseStatus(
            ok=True,
            attempted=True,
            restored=True,
            message=f"{operator_name} runtime files restored from GitHub Release asset.",
            target_dir=target_dir,
            repo=config.repo,
            tag=config.tag,
            asset=config.asset,
            sha256=sha256,
            size_bytes=size_bytes,
        )
