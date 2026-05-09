# ==========================================================================
# Master Thesis - Vector Database Release Artifact Loader
#   - André Filipe Gomes Silvestre, 20240502
#
#   Downloads the latest vector database artifact from a GitHub Release when
#   hosted deployments do not carry data/vector_db in the application image.
# ==========================================================================

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import Config


GITHUB_API_BASE_URL = "https://api.github.com"
CHROMA_SQLITE_FILENAME = "chroma.sqlite3"


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


def _request_json(url: str, timeout: int) -> dict[str, Any]:
    """Request JSON from the GitHub API."""
    request = urllib.request.Request(
        url,
        headers=_build_headers("application/vnd.github+json"),
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


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


def _resolve_asset_download_url(timeout: int) -> str:
    """Resolve the GitHub API asset URL for the configured vector DB release."""
    repo = str(getattr(Config, "VECTOR_DB_RELEASE_REPO", "") or "").strip()
    tag = str(getattr(Config, "VECTOR_DB_RELEASE_TAG", "") or "").strip()
    asset_name = str(getattr(Config, "VECTOR_DB_RELEASE_ASSET", "") or "").strip()

    if not repo or "/" not in repo:
        raise ValueError("VECTOR_DB_RELEASE_REPO must be in 'owner/repo' format.")
    if not tag:
        raise ValueError("VECTOR_DB_RELEASE_TAG cannot be empty.")
    if not asset_name:
        raise ValueError("VECTOR_DB_RELEASE_ASSET cannot be empty.")

    release_url = f"{GITHUB_API_BASE_URL}/repos/{repo}/releases/tags/{tag}"
    release = _request_json(release_url, timeout=timeout)

    for asset in release.get("assets", []):
        if asset.get("name") == asset_name and asset.get("url"):
            return str(asset["url"])

    raise FileNotFoundError(f"Release asset '{asset_name}' was not found in '{repo}@{tag}'.")


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


def ensure_vector_db_from_release() -> VectorDbReleaseStatus:
    """Ensure the runtime vector DB exists, downloading it from a release if needed."""
    vector_db_dir = Path(Config.VECTOR_DB_DIR)
    force_download = bool(getattr(Config, "VECTOR_DB_RELEASE_FORCE_DOWNLOAD", False))

    if _vector_db_is_present(vector_db_dir) and not force_download:
        return VectorDbReleaseStatus(
            ok=True,
            attempted=False,
            message="Local vector database is already available.",
            path=vector_db_dir,
        )

    if not bool(getattr(Config, "VECTOR_DB_RELEASE_ENABLED", True)):
        return VectorDbReleaseStatus(
            ok=_vector_db_is_present(vector_db_dir),
            attempted=False,
            message="Vector DB release download is disabled.",
            path=vector_db_dir,
        )

    timeout = int(getattr(Config, "VECTOR_DB_RELEASE_TIMEOUT_SECONDS", 120))
    vector_db_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        asset_url = _resolve_asset_download_url(timeout=timeout)
    except (OSError, ValueError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        return VectorDbReleaseStatus(
            ok=_vector_db_is_present(vector_db_dir),
            attempted=True,
            message=f"Could not resolve vector DB release asset: {exc}",
            path=vector_db_dir,
        )

    with tempfile.TemporaryDirectory(prefix="lisboa_vector_db_") as temp_root:
        temp_path = Path(temp_root)
        archive_path = temp_path / str(getattr(Config, "VECTOR_DB_RELEASE_ASSET", "vector_db.zip"))
        extract_path = temp_path / "vector_db"
        extract_path.mkdir(parents=True, exist_ok=True)

        try:
            sha256, size_bytes = _download_file(asset_url, archive_path, timeout=timeout)
            _safe_extract_zip(archive_path, extract_path)

            if not _vector_db_is_present(extract_path):
                raise FileNotFoundError("Downloaded vector DB archive does not contain chroma.sqlite3.")

            _install_vector_db(extract_path=extract_path, vector_db_dir=vector_db_dir)
        except (OSError, zipfile.BadZipFile, ValueError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            return VectorDbReleaseStatus(
                ok=_vector_db_is_present(vector_db_dir),
                attempted=True,
                message=f"Could not download vector DB release asset: {exc}",
                path=vector_db_dir,
            )

    return VectorDbReleaseStatus(
        ok=True,
        attempted=True,
        downloaded=True,
        message="Vector database downloaded from GitHub Release asset.",
        path=vector_db_dir,
        sha256=sha256,
        size_bytes=size_bytes,
    )
