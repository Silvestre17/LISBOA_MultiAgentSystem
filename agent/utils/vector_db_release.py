# ==========================================================================
# Master Thesis - Vector Database Release Artifact Loader
#   - André Filipe Gomes Silvestre, 20240502
#
#   Downloads the latest vector database artifact from a GitHub Release when
#   hosted deployments do not carry data/vector_db in the application image,
#   and fetches a newer artifact into a sibling directory while the app runs.
# ==========================================================================

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import threading
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import Config


GITHUB_API_BASE_URL = "https://api.github.com"
CHROMA_SQLITE_FILENAME = "chroma.sqlite3"
# Written next to chroma.sqlite3 so a later check can tell which release asset
# a directory holds. ChromaDB ignores files it does not know.
RELEASE_MARKER_FILENAME = ".lisboa_release.json"
_VECTOR_DB_DOWNLOAD_LOCK = threading.Lock()
_VECTOR_DB_READY_THIS_PROCESS = False
# ETag of the release payload that matches the installed data, sent back as
# If-None-Match so an unchanged release costs no download.
_RELEASE_ETAG: str | None = None


@dataclass(frozen=True)
class VectorDbReleaseStatus:
    """Status returned by the vector DB release artifact loader."""

    ok: bool
    attempted: bool
    message: str
    path: Path
    downloaded: bool = False
    sha256: str | None = None
    size_bytes: int | None = None


def _vector_db_is_present(vector_db_dir: Path) -> bool:
    """Return whether the local ChromaDB vector store appears usable."""
    chroma_db = vector_db_dir / CHROMA_SQLITE_FILENAME
    return chroma_db.exists() and chroma_db.is_file() and chroma_db.stat().st_size > 0


def _build_headers(accept: str) -> dict[str, str]:
    """Build GitHub API headers for release metadata or asset download."""
    headers = {
        "Accept": accept,
        "User-Agent": "lisboa-vector-db-loader",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = getattr(Config, "VECTOR_DB_RELEASE_TOKEN", None)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _request_release(timeout: int, etag: str | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """Request the configured release from the GitHub API.

    Args:
        timeout: Request timeout in seconds.
        etag: ETag of a previous response, sent as ``If-None-Match``.

    Returns:
        tuple: The release payload and its ETag, or ``(None, etag)`` when the
        release is unchanged (HTTP 304).
    """
    repo = str(getattr(Config, "VECTOR_DB_RELEASE_REPO", "") or "").strip()
    tag = str(getattr(Config, "VECTOR_DB_RELEASE_TAG", "") or "").strip()
    if not repo or "/" not in repo:
        raise ValueError("VECTOR_DB_RELEASE_REPO must be in 'owner/repo' format.")
    if not tag:
        raise ValueError("VECTOR_DB_RELEASE_TAG cannot be empty.")

    headers = _build_headers("application/vnd.github+json")
    if etag:
        headers["If-None-Match"] = etag
    request = urllib.request.Request(
        f"{GITHUB_API_BASE_URL}/repos/{repo}/releases/tags/{tag}",
        headers=headers,
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
            response_etag = response.headers.get("ETag")
    except urllib.error.HTTPError as exc:
        if exc.code == 304:
            return None, etag
        raise
    return json.loads(payload), response_etag


def _download_file(url: str, destination: Path, timeout: int) -> tuple[str, int]:
    """Download a GitHub release asset and return its SHA-256 and byte size."""
    request = urllib.request.Request(
        url,
        headers=_build_headers("application/octet-stream"),
        method="GET",
    )
    digest = hashlib.sha256()
    size_bytes = 0

    with urllib.request.urlopen(request, timeout=timeout) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            digest.update(chunk)
            size_bytes += len(chunk)

    return digest.hexdigest(), size_bytes


def _find_release_asset(release: dict[str, Any]) -> dict[str, Any]:
    """Return the configured vector DB asset from a release payload."""
    asset_name = str(getattr(Config, "VECTOR_DB_RELEASE_ASSET", "") or "").strip()
    if not asset_name:
        raise ValueError("VECTOR_DB_RELEASE_ASSET cannot be empty.")

    for asset in release.get("assets", []):
        if asset.get("name") == asset_name and asset.get("url"):
            return asset

    raise FileNotFoundError(
        f"Release asset '{asset_name}' was not found in release '{release.get('tag_name', '')}'."
    )


def _read_release_marker(vector_db_dir: Path) -> dict[str, Any]:
    """Return the release marker of an installed vector DB, or an empty dict."""
    try:
        payload = json.loads((vector_db_dir / RELEASE_MARKER_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _marker_matches_asset(marker: dict[str, Any], asset: dict[str, Any]) -> bool:
    """Return whether an installed vector DB holds this exact release asset."""
    return (
        marker.get("asset_id") == asset.get("id")
        and marker.get("updated_at") == asset.get("updated_at")
    )


def _safe_extract_zip(zip_path: Path, destination: Path) -> None:
    """Extract a zip archive while preventing path traversal."""
    destination_resolved = destination.resolve()

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise ValueError(f"Unsafe path in vector DB archive: {member.filename}")
        archive.extractall(destination)


def _install_vector_db(extract_path: Path, vector_db_dir: Path) -> None:
    """Install an extracted vector DB directory into the configured location.

    Windows OneDrive-backed folders can deny recursive deletion of cloud-backed
    Chroma collection directories. A clean replacement is preferred, but a
    merge fallback is acceptable because ChromaDB uses ``chroma.sqlite3`` as the
    authoritative index and ignores orphaned collection folders not referenced
    by that database.
    """
    if vector_db_dir.exists():
        try:
            shutil.rmtree(vector_db_dir)
        except OSError:
            vector_db_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(extract_path, vector_db_dir, dirs_exist_ok=True)
            return

    shutil.move(str(extract_path), str(vector_db_dir))


def _download_and_install(asset: dict[str, Any], vector_db_dir: Path, timeout: int) -> tuple[str, int]:
    """Download a release asset, install it into a directory and mark it.

    Args:
        asset: Release asset payload from the GitHub API.
        vector_db_dir: Directory that receives the ChromaDB files.
        timeout: Request timeout in seconds.

    Returns:
        tuple[str, int]: SHA-256 and byte size of the downloaded archive.
    """
    vector_db_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lisboa_vector_db_") as temp_root:
        temp_path = Path(temp_root)
        archive_path = temp_path / "vector_db.zip"
        extract_path = temp_path / "vector_db"
        extract_path.mkdir(parents=True, exist_ok=True)

        sha256, size_bytes = _download_file(str(asset["url"]), archive_path, timeout=timeout)
        _safe_extract_zip(archive_path, extract_path)
        if not _vector_db_is_present(extract_path):
            raise FileNotFoundError("Downloaded vector DB archive does not contain chroma.sqlite3.")

        marker = {
            "asset_id": asset.get("id"),
            "updated_at": asset.get("updated_at"),
            "sha256": sha256,
            "size_bytes": size_bytes,
            "installed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (extract_path / RELEASE_MARKER_FILENAME).write_text(json.dumps(marker, indent=2), encoding="utf-8")
        _install_vector_db(extract_path=extract_path, vector_db_dir=vector_db_dir)

    return sha256, size_bytes


def ensure_vector_db_from_release() -> VectorDbReleaseStatus:
    """Ensure the runtime vector DB exists, downloading it from a release if needed."""
    global _RELEASE_ETAG, _VECTOR_DB_READY_THIS_PROCESS

    vector_db_dir = Path(Config.VECTOR_DB_DIR)

    with _VECTOR_DB_DOWNLOAD_LOCK:
        force_download = bool(getattr(Config, "VECTOR_DB_RELEASE_FORCE_DOWNLOAD", False))

        if _VECTOR_DB_READY_THIS_PROCESS and _vector_db_is_present(vector_db_dir):
            return VectorDbReleaseStatus(
                ok=True,
                attempted=False,
                message="Vector database already prepared in this process.",
                path=vector_db_dir,
            )

        if _vector_db_is_present(vector_db_dir) and not force_download:
            _VECTOR_DB_READY_THIS_PROCESS = True
            return VectorDbReleaseStatus(
                ok=True,
                attempted=False,
                message="Local vector database is already available.",
                path=vector_db_dir,
            )

        if not bool(getattr(Config, "VECTOR_DB_RELEASE_ENABLED", True)):
            ok = _vector_db_is_present(vector_db_dir)
            _VECTOR_DB_READY_THIS_PROCESS = ok
            return VectorDbReleaseStatus(
                ok=ok,
                attempted=False,
                message="Vector DB release download is disabled.",
                path=vector_db_dir,
            )

        timeout = int(getattr(Config, "VECTOR_DB_RELEASE_TIMEOUT_SECONDS", 120))

        # URLError and HTTPError are OSError subclasses.
        try:
            release, etag = _request_release(timeout=timeout)
            asset = _find_release_asset(release or {})
        except (OSError, ValueError) as exc:
            ok = _vector_db_is_present(vector_db_dir)
            _VECTOR_DB_READY_THIS_PROCESS = ok
            return VectorDbReleaseStatus(
                ok=ok,
                attempted=True,
                message=f"Could not resolve vector DB release asset: {exc}",
                path=vector_db_dir,
            )

        try:
            sha256, size_bytes = _download_and_install(asset, vector_db_dir, timeout=timeout)
        except (OSError, zipfile.BadZipFile, ValueError) as exc:
            ok = _vector_db_is_present(vector_db_dir)
            _VECTOR_DB_READY_THIS_PROCESS = ok
            return VectorDbReleaseStatus(
                ok=ok,
                attempted=True,
                message=f"Could not download vector DB release asset: {exc}",
                path=vector_db_dir,
            )

        _RELEASE_ETAG = etag
        _VECTOR_DB_READY_THIS_PROCESS = True
        return VectorDbReleaseStatus(
            ok=True,
            attempted=True,
            downloaded=True,
            message="Vector database downloaded from GitHub Release asset.",
            path=vector_db_dir,
            sha256=sha256,
            size_bytes=size_bytes,
        )


def refresh_vector_db_from_release(active_dir: Path) -> Path | None:
    """Download a newer vector DB release next to the active one.

    The active directory stays untouched while searches use it. A newer asset
    goes to a new ``<VECTOR_DB_DIR>-<asset id>-<random>`` directory; the caller
    switches to it and later deletes the old copy with
    :func:`remove_retired_vector_db`.

    Args:
        active_dir: Directory the running KnowledgeBase searches.

    Returns:
        Path | None: The directory holding the newer release, or None when the
        active data is current, the release download is disabled or the check
        failed (the failure is logged and the current data stays in use).
    """
    global _RELEASE_ETAG

    if not bool(getattr(Config, "VECTOR_DB_RELEASE_ENABLED", True)):
        return None

    timeout = int(getattr(Config, "VECTOR_DB_RELEASE_TIMEOUT_SECONDS", 120))
    with _VECTOR_DB_DOWNLOAD_LOCK:
        try:
            release, etag = _request_release(timeout=timeout, etag=_RELEASE_ETAG)
            if release is None:
                return None
            asset = _find_release_asset(release)
        except (OSError, ValueError) as exc:
            print(f"⚠️ Vector DB freshness check failed; keeping the current data ({exc})", flush=True)
            return None

        if _marker_matches_asset(_read_release_marker(active_dir), asset):
            _RELEASE_ETAG = etag
            return None

        configured_dir = Path(Config.VECTOR_DB_DIR)
        configured_dir.parent.mkdir(parents=True, exist_ok=True)
        target_dir = Path(
            tempfile.mkdtemp(prefix=f"{configured_dir.name}-{asset.get('id')}-", dir=configured_dir.parent)
        )
        try:
            _download_and_install(asset, target_dir, timeout=timeout)
        except (OSError, zipfile.BadZipFile, ValueError) as exc:
            print(f"⚠️ Vector DB refresh download failed; keeping the current data ({exc})", flush=True)
            shutil.rmtree(target_dir, ignore_errors=True)
            return None

        _RELEASE_ETAG = etag
        return target_dir


def remove_retired_vector_db(retired_dir: Path) -> None:
    """Delete a vector DB copy made by a refresh once no search uses it.

    Only copies made by :func:`refresh_vector_db_from_release` are deleted; the
    configured ``VECTOR_DB_DIR`` is the cold-start copy and stays.
    """
    configured_dir = Path(Config.VECTOR_DB_DIR)
    if retired_dir.parent == configured_dir.parent and retired_dir.name.startswith(f"{configured_dir.name}-"):
        shutil.rmtree(retired_dir, ignore_errors=True)
