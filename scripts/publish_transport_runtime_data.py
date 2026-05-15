# ==========================================================================
# Master Thesis - Publish Transport Runtime Data Artifacts
#   - André Filipe Gomes Silvestre, 20240502
#
#   Builds replaceable GitHub Release assets containing last-known-good CP and
#   Carris Urban GTFS SQLite runtime files. The app can restore these assets
#   when live GTFS downloads fail during hosted startup.
# ==========================================================================

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "transport_runtime_artifacts"
DEFAULT_RELEASE_MAX_BYTES = 1_800_000_000


@dataclass
class OperatorArtifact:
    """Metadata about an operator runtime artifact."""

    operator: str
    asset: str
    ok: bool
    db_path: str | None = None
    zip_path: str | None = None
    metadata_path: str | None = None
    size_bytes: int | None = None
    stops: int | None = None
    routes: int | None = None
    trips: int | None = None
    stop_times: int | None = None
    error: str | None = None


def _ensure_repo_import_path() -> None:
    """Ensure repository modules can be imported when this script runs directly."""
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _utc_now_iso() -> str:
    """Return the current UTC timestamp as ISO-8601 text."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _count_rows(db_path: Path, table: str) -> int:
    """Count rows in a SQLite table, returning 0 when unavailable."""
    try:
        with sqlite3.connect(str(db_path)) as connection:
            row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    except sqlite3.Error:
        return 0

    return int(row[0] or 0) if row else 0


def _zip_files(asset_path: Path, files: dict[Path, str]) -> int:
    """Create a zip asset from a mapping of local files to archive names."""
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    if asset_path.exists():
        asset_path.unlink()

    with zipfile.ZipFile(asset_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source, archive_name in files.items():
            if source.exists() and source.is_file():
                archive.write(source, archive_name)

    return asset_path.stat().st_size


def _release_max_bytes() -> int:
    """Return the configured maximum release asset size."""
    raw_value = os.getenv("TRANSPORT_DATA_RELEASE_MAX_BYTES", str(DEFAULT_RELEASE_MAX_BYTES))
    try:
        value = int(raw_value)
    except ValueError:
        return DEFAULT_RELEASE_MAX_BYTES
    return value if value > 0 else DEFAULT_RELEASE_MAX_BYTES


def _validate_asset_size(asset_path: Path, size_bytes: int) -> None:
    """Raise when a generated release asset is too large to publish safely."""
    max_bytes = _release_max_bytes()
    if size_bytes > max_bytes:
        raise ValueError(
            f"{asset_path.name} is too large: {size_bytes} bytes. "
            f"Configured limit: {max_bytes} bytes."
        )


def _build_carris_artifact(artifact_dir: Path) -> OperatorArtifact:
    """Build the Carris Urban runtime release asset."""
    from tools.carris_api import CarrisGTFSManager

    asset_name = os.getenv("CARRIS_RUNTIME_RELEASE_ASSET", "carris_runtime.zip")
    manager = CarrisGTFSManager()

    if not manager.ensure_database(force_update=True):
        return OperatorArtifact(
            operator="carris_urban",
            asset=asset_name,
            ok=False,
            error="Carris Urban GTFS database could not be built from the live feed.",
        )

    db_path = Path(manager.db_path)
    metadata_path = Path(manager.metadata_path)
    stops = _count_rows(db_path, "stops")
    routes = _count_rows(db_path, "routes")
    trips = _count_rows(db_path, "trips")
    stop_times = _count_rows(db_path, "stop_times")

    if stops <= 0:
        return OperatorArtifact(
            operator="carris_urban",
            asset=asset_name,
            ok=False,
            db_path=str(db_path),
            error="Carris Urban SQLite database has no stops.",
        )

    asset_size = _zip_files(
        artifact_dir / asset_name,
        {
            db_path: "carris.db",
            metadata_path: "metadata.json",
        },
    )
    _validate_asset_size(artifact_dir / asset_name, asset_size)
    return OperatorArtifact(
        operator="carris_urban",
        asset=asset_name,
        ok=True,
        db_path=str(db_path),
        metadata_path=str(metadata_path),
        size_bytes=asset_size,
        stops=stops,
        routes=routes,
        trips=trips,
        stop_times=stop_times,
    )


def _build_cp_artifact(artifact_dir: Path) -> OperatorArtifact:
    """Build the CP runtime release asset."""
    from tools.cp_api import get_gtfs_manager

    asset_name = os.getenv("CP_RUNTIME_RELEASE_ASSET", "cp_runtime.zip")
    manager = get_gtfs_manager()

    if not manager.ensure_database(force_refresh=True):
        return OperatorArtifact(
            operator="cp",
            asset=asset_name,
            ok=False,
            error="CP GTFS database could not be built from the live feed.",
        )

    db_path = Path(manager.db_path)
    gtfs_zip_path = Path(manager.gtfs_zip_path)
    metadata_path = Path(manager.metadata_path)
    stops = _count_rows(db_path, "stops")
    routes = _count_rows(db_path, "routes")
    trips = _count_rows(db_path, "trips")
    stop_times = _count_rows(db_path, "stop_times")

    if stops <= 0:
        return OperatorArtifact(
            operator="cp",
            asset=asset_name,
            ok=False,
            db_path=str(db_path),
            error="CP SQLite database has no stops.",
        )

    asset_size = _zip_files(
        artifact_dir / asset_name,
        {
            db_path: "cp_gtfs.db",
            gtfs_zip_path: "gtfs.zip",
            metadata_path: "metadata.json",
        },
    )
    _validate_asset_size(artifact_dir / asset_name, asset_size)
    return OperatorArtifact(
        operator="cp",
        asset=asset_name,
        ok=True,
        db_path=str(db_path),
        zip_path=str(gtfs_zip_path),
        metadata_path=str(metadata_path),
        size_bytes=asset_size,
        stops=stops,
        routes=routes,
        trips=trips,
        stop_times=stop_times,
    )


def _write_manifest(artifact_dir: Path, artifacts: list[OperatorArtifact]) -> None:
    """Write a manifest describing all generated transport runtime artifacts."""
    manifest = {
        "generated_at": _utc_now_iso(),
        "source_repository": os.getenv("GITHUB_REPOSITORY"),
        "source_commit": os.getenv("GITHUB_SHA"),
        "workflow_run_id": os.getenv("GITHUB_RUN_ID"),
        "artifacts": [asdict(artifact) for artifact in artifacts],
    }
    (artifact_dir / "transport_runtime_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _run_builder(
    name: str,
    builder: Callable[[Path], OperatorArtifact],
    artifact_dir: Path,
) -> OperatorArtifact:
    """Run an operator artifact builder without stopping other operators."""
    try:
        artifact = builder(artifact_dir)
    except Exception as exc:
        artifact = OperatorArtifact(
            operator=name,
            asset="unknown",
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    status = "OK" if artifact.ok else "FAILED"
    print(f"[{status}] {artifact.operator}: {artifact.error or artifact.asset}", flush=True)
    return artifact


def main() -> int:
    """Build transport runtime artifacts for GitHub Release publication."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory where release assets should be written.",
    )
    args = parser.parse_args()

    _ensure_repo_import_path()
    os.environ.setdefault("CARRIS_RUNTIME_RELEASE_ENABLED", "false")
    os.environ.setdefault("CP_RUNTIME_RELEASE_ENABLED", "false")

    artifact_dir = args.artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    artifacts = [
        _run_builder("carris_urban", _build_carris_artifact, artifact_dir),
        _run_builder("cp", _build_cp_artifact, artifact_dir),
    ]
    _write_manifest(artifact_dir, artifacts)

    if not any(artifact.ok for artifact in artifacts):
        print("No transport runtime assets were regenerated; keeping previous release assets if available.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
