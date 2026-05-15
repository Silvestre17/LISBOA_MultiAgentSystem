# ==========================================================================
# Master Thesis - Shared Runtime Startup Resources
#   - André Filipe Gomes Silvestre, 20240502
#
#   Shared preload helpers for runtime entrypoints.
#   Features:
#     - Warm vector-store resources before first request
#     - Prepare static transport support datasets and caches
#     - Return structured preload status for app and script callers
# ==========================================================================

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict


STARTUP_LOG_SEPARATOR = "=" * 72
_STARTUP_PRELOAD_LOCK = threading.Lock()
_STARTUP_PRELOAD_STATUS: Dict[str, Any] | None = None


def _startup_log(message: str = "") -> None:
    """Print a startup diagnostic line to container logs."""
    print(message, flush=True)


def _format_size_mb(path: str | Path) -> str:
    """Return a human-readable file size for a local path."""
    try:
        size_mb = Path(path).stat().st_size / (1024 * 1024)
    except OSError:
        return "missing"
    return f"{size_mb:.1f} MB"


def _file_mtime_ns(path: str | Path) -> int | None:
    """Return a file modification timestamp in nanoseconds when available."""
    try:
        return Path(path).stat().st_mtime_ns
    except OSError:
        return None


def _load_json_metadata(path: str | Path) -> Dict[str, Any]:
    """Load a JSON metadata file, returning an empty dict when unavailable."""
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}

    return payload if isinstance(payload, dict) else {}


def _count_sqlite_rows(db_path: str, table_name: str) -> int:
    """Return the number of rows available in a SQLite table."""
    if not db_path or not os.path.exists(db_path):
        return 0

    try:
        with sqlite3.connect(db_path) as connection:
            cursor = connection.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row = cursor.fetchone()
    except sqlite3.Error:
        return 0

    return int(row[0]) if row and row[0] is not None else 0


def _sqlite_table_counts(db_path: str | Path, table_names: list[str]) -> Dict[str, int]:
    """Return row counts for a list of SQLite tables."""
    resolved_path = str(db_path)
    return {
        table_name: _count_sqlite_rows(resolved_path, table_name)
        for table_name in table_names
    }


def _format_counts(counts: Dict[str, int]) -> str:
    """Format SQLite/API counts for startup diagnostics."""
    return ", ".join(f"{name}={value:,}" for name, value in counts.items())


def _metadata_restore_changed(before: Dict[str, Any], after: Dict[str, Any]) -> bool:
    """Return whether metadata shows a new runtime release restore."""
    before_restore = before.get("_runtime_release_restore") if before else None
    after_restore = after.get("_runtime_release_restore") if after else None
    return bool(after_restore and after_restore != before_restore)


def _format_runtime_restore(metadata: Dict[str, Any]) -> str | None:
    """Format release-restore metadata for startup diagnostics."""
    restore = metadata.get("_runtime_release_restore")
    if not isinstance(restore, dict):
        return None

    asset = restore.get("asset", "unknown")
    tag = restore.get("tag", "unknown")
    restored_at = restore.get("restored_at", "unknown")
    size_bytes = restore.get("size_bytes")
    size_text = ""
    if isinstance(size_bytes, int) and size_bytes > 0:
        size_text = f", size={size_bytes / (1024 * 1024):.1f} MB"

    return f"asset={asset}, tag={tag}, restored_at={restored_at}{size_text}"


def _localized_kb_status(kb_ok: bool, language: str) -> str:
    """Return the KnowledgeBase status message in the requested language."""
    if kb_ok:
        return (
            "Base de conhecimento pronta."
            if language == "pt"
            else "Knowledge base ready."
        )

    return (
        "Não foi possível carregar a base de conhecimento."
        if language == "pt"
        else "Could not load the knowledge base."
    )


def _copy_startup_status_for_language(
    status: Dict[str, Any],
    language: str,
) -> Dict[str, Any]:
    """Return a shallow startup status copy localized for the caller."""
    localized_status = dict(status)
    localized_status["kb_status"] = _localized_kb_status(
        bool(localized_status.get("kb_ok", False)),
        language,
    )
    return localized_status


def _print_startup_header() -> None:
    """Print a startup resource diagnostic header."""
    _startup_log(STARTUP_LOG_SEPARATOR)
    _startup_log("🚀 LISBOA STARTUP RESOURCE CHECK")
    _startup_log(STARTUP_LOG_SEPARATOR)


def _print_carris_urban_report(
    *,
    ok: bool,
    action: str,
    db_path: str,
    metadata_path: str,
    gtfs_url: str,
    realtime_url: str,
    counts: Dict[str, int],
    message: str,
) -> None:
    """Print Carris Urban startup diagnostics."""
    metadata = _load_json_metadata(metadata_path)
    status_icon = "✅" if ok else "❌"
    _startup_log(f"{status_icon} Carris Urban GTFS: {message}")
    _startup_log(f"   🔄 Startup action: {action}")
    _startup_log(f"   🌐 Static GTFS: {gtfs_url}")
    _startup_log(f"   📡 GTFS-RT feed: {realtime_url} (loaded on demand during live queries)")
    _startup_log(f"   🗄️ SQLite DB: {db_path} ({_format_size_mb(db_path)})")
    _startup_log(f"   📄 Metadata: {metadata_path}")
    if metadata:
        _startup_log(
            "   🧾 Snapshot: "
            f"gtfs_date={metadata.get('gtfs_date', 'unknown')}, "
            f"updated_at={metadata.get('updated_at', 'unknown')}, "
            f"schema={metadata.get('schema_version', 'unknown')}"
        )
        restore_snapshot = _format_runtime_restore(metadata)
        if restore_snapshot:
            _startup_log(f"   🧯 Release backup: {restore_snapshot}")
    _startup_log(f"   📊 Tables: {_format_counts(counts)}")


def _print_cp_report(
    *,
    ok: bool,
    action: str,
    db_path: str | Path,
    zip_path: str | Path,
    metadata_path: str | Path,
    gtfs_url: str,
    stations_url: str,
    vehicles_url: str,
    counts: Dict[str, int],
    aml_station_count: int,
    message: str,
) -> None:
    """Print CP startup diagnostics."""
    metadata = _load_json_metadata(metadata_path)
    status_icon = "✅" if ok else "❌"
    _startup_log(f"{status_icon} CP GTFS + live station layer: {message}")
    _startup_log(f"   🔄 Startup action: {action}")
    _startup_log(f"   🌐 Static GTFS: {gtfs_url}")
    _startup_log(f"   📡 Stations API: {stations_url}")
    _startup_log(f"   📡 Vehicles API: {vehicles_url} (loaded on demand during live queries)")
    _startup_log(f"   📦 GTFS ZIP: {zip_path} ({_format_size_mb(zip_path)})")
    _startup_log(f"   🗄️ SQLite DB: {db_path} ({_format_size_mb(db_path)})")
    _startup_log(f"   📄 Metadata: {metadata_path}")
    if metadata:
        _startup_log(
            "   🧾 Snapshot: "
            f"last_download={metadata.get('last_download', 'unknown')}, "
            f"db_created={metadata.get('db_created', 'unknown')}, "
            f"last_modified={metadata.get('last_modified', 'unknown')}"
        )
        restore_snapshot = _format_runtime_restore(metadata)
        if restore_snapshot:
            _startup_log(f"   🧯 Release backup: {restore_snapshot}")
    _startup_log(f"   📊 Tables: {_format_counts(counts)}")
    _startup_log(f"   🚉 AML station cache: {aml_station_count:,} stations")


def _print_metro_report(
    *,
    ok: bool,
    mode: str,
    station_count: int,
    api_base: str,
    status_url: str,
    message: str,
) -> None:
    """Print Metro startup diagnostics."""
    status_icon = "✅" if ok else "⚠️"
    _startup_log(f"{status_icon} Metro de Lisboa layer: {message}")
    _startup_log(f"   🌐 Official API: {api_base}")
    _startup_log(f"   📡 Line status fallback/API: {status_url}")
    _startup_log(f"   🚇 Station source: {mode}")
    _startup_log(f"   📊 Stations loaded: {station_count:,}")


def _print_carris_metropolitana_report(
    *,
    ok: bool,
    stops_url: str,
    lines_url: str,
    routes_url: str,
    counts: Dict[str, int],
    message: str,
) -> None:
    """Print Carris Metropolitana startup diagnostics."""
    status_icon = "✅" if ok else "❌"
    _startup_log(f"{status_icon} Carris Metropolitana reference layer: {message}")
    _startup_log(f"   🌐 Stops API: {stops_url}")
    _startup_log(f"   🌐 Lines API: {lines_url}")
    _startup_log(f"   🌐 Routes API: {routes_url}")
    _startup_log(f"   📊 Loaded: {_format_counts(counts)}")


def _invoke_startup_tool_without_tracing(tool: Any, args: Dict[str, Any] | None = None) -> Any:
    """Invoke a LangChain tool for startup checks without creating a LangSmith trace."""
    resolved_args = args if isinstance(args, dict) else {}
    try:
        from agent.utils.langsmith_tracing import tracing_context

        with tracing_context(enabled=False):
            return tool.invoke(resolved_args)
    except ImportError:
        return tool.invoke(resolved_args)


def format_transport_layer_summary(details: Dict[str, Dict[str, Any]], overall_ok: bool) -> str:
    """Build a compact transport-layer readiness summary for app and scripts."""
    metro_detail = details.get("metro", {})
    cp_detail = details.get("cp", {})
    carris_detail = details.get("carris", {})
    carris_met_detail = details.get("carris_metropolitana", {})

    metro_mode = metro_detail.get("mode") or "unreachable"
    cp_stops = int(cp_detail.get("stops") or 0)
    carris_stops = int(carris_detail.get("stops") or 0)
    carris_met_status = "reachable" if carris_met_detail.get("ok") else "unreachable"
    prefix = "✅" if overall_ok else "⚠️"

    return (
        f"{prefix} Transport layer {'ready' if overall_ok else 'incomplete'}: "
        f"Metro [{metro_mode}], "
        f"CP [{cp_stops} stops], "
        f"Carris Urban [{carris_stops} stops], "
        f"Carris Met. [{carris_met_status}]"
    )


def _check_carris_urban_readiness() -> Dict[str, Any]:
    """Validate that the Carris Urban GTFS database is available and populated."""
    try:
        from tools.carris_api import (
            CARRIS_DB_PATH,
            CARRIS_GTFS_RT_URL,
            CARRIS_GTFS_URL,
            CARRIS_METADATA_PATH,
            CarrisGTFSManager,
        )

        manager = CarrisGTFSManager()
        db_mtime_before = _file_mtime_ns(CARRIS_DB_PATH)
        metadata_before = _load_json_metadata(CARRIS_METADATA_PATH)
        database_ready = manager.ensure_database(force_update=False)
        db_mtime_after = _file_mtime_ns(CARRIS_DB_PATH)
        metadata_after = _load_json_metadata(CARRIS_METADATA_PATH)
        stop_count = _count_sqlite_rows(CARRIS_DB_PATH, "stops") if database_ready else 0
        counts = (
            _sqlite_table_counts(
                CARRIS_DB_PATH,
                ["agency", "routes", "stops", "trips", "stop_times", "shapes"],
            )
            if database_ready
            else {}
        )
        ok = bool(database_ready and stop_count > 0)
        message = (
            f"Carris Urban ready ({stop_count} stops)"
            if ok
            else "Carris Urban GTFS database is not populated"
        )
        action = (
            "restored last-known-good GitHub Release backup"
            if _metadata_restore_changed(metadata_before, metadata_after)
            else
            "downloaded GTFS and rebuilt SQLite"
            if db_mtime_before != db_mtime_after or metadata_before != metadata_after
            else "remote GTFS freshness check completed; local SQLite reused"
            if database_ready
            else "startup check failed before a usable SQLite DB was available"
        )
        _print_carris_urban_report(
            ok=ok,
            action=action,
            db_path=CARRIS_DB_PATH,
            metadata_path=CARRIS_METADATA_PATH,
            gtfs_url=CARRIS_GTFS_URL,
            realtime_url=CARRIS_GTFS_RT_URL,
            counts=counts,
            message=message,
        )
        return {
            "ok": ok,
            "stops": stop_count,
            "counts": counts,
            "message": message,
        }
    except Exception as exc:
        _startup_log(f"❌ Carris Urban GTFS: failed to load ({exc})")
        return {
            "ok": False,
            "stops": 0,
            "message": f"Carris Urban GTFS database failed to load: {exc}",
        }


def _check_metro_readiness() -> Dict[str, Any]:
    """Validate Metro stations plus line-status availability or fallback mode."""
    try:
        from tools.metrolisboa_api import (
            METRO_API_BASE,
            METRO_STATUS_URL,
            get_metro_status,
            load_metro_stations,
        )

        stations = load_metro_stations(force_reload=False)
        station_count = len(stations)
        status_snapshot = str(_invoke_startup_tool_without_tracing(get_metro_status, {}) or "").strip()
        metro_mode = "live" if "Official API" in status_snapshot else "fallback"
        ok = bool(station_count > 0 and status_snapshot and not status_snapshot.startswith("❌"))
        message = (
            f"Metro ready ({metro_mode}, {station_count} stations)"
            if ok
            else "Metro line status is unavailable"
        )
        _print_metro_report(
            ok=ok,
            mode=metro_mode if ok else "unreachable",
            station_count=station_count,
            api_base=METRO_API_BASE,
            status_url=METRO_STATUS_URL,
            message=message,
        )
        return {
            "ok": ok,
            "mode": metro_mode if ok else "unreachable",
            "stations": station_count,
            "message": message,
        }
    except Exception as exc:
        _startup_log(f"❌ Metro de Lisboa layer: failed to load ({exc})")
        return {
            "ok": False,
            "mode": "unreachable",
            "stations": 0,
            "message": f"Metro readiness check failed: {exc}",
        }


def _check_cp_readiness() -> Dict[str, Any]:
    """Validate that the CP GTFS SQLite database exists and contains stops."""
    try:
        from tools.cp_api import (
            CP_GTFS_URL,
            CP_STATIONS_URL,
            CP_VEHICLES_URL,
            get_gtfs_manager,
            load_cp_aml_stations,
        )

        manager = get_gtfs_manager()
        db_mtime_before = _file_mtime_ns(manager.db_path)
        zip_mtime_before = _file_mtime_ns(manager.gtfs_zip_path)
        metadata_before = _load_json_metadata(manager.metadata_path)
        database_ready = manager.ensure_database(force_refresh=False)
        db_mtime_after = _file_mtime_ns(manager.db_path)
        zip_mtime_after = _file_mtime_ns(manager.gtfs_zip_path)
        metadata_after = _load_json_metadata(manager.metadata_path)
        stop_count = _count_sqlite_rows(str(manager.db_path), "stops") if database_ready else 0
        aml_stations = load_cp_aml_stations(force_reload=False) if database_ready else {}
        counts = (
            _sqlite_table_counts(
                manager.db_path,
                ["agency", "routes", "stops", "trips", "stop_times", "shapes"],
            )
            if database_ready
            else {}
        )
        ok = bool(database_ready and stop_count > 0)
        message = (
            f"CP ready ({stop_count} stops, {len(aml_stations)} AML stations)"
            if ok
            else "CP GTFS SQLite database is not populated"
        )
        action = (
            "restored last-known-good GitHub Release backup"
            if _metadata_restore_changed(metadata_before, metadata_after)
            else
            "downloaded GTFS and rebuilt SQLite"
            if (
                db_mtime_before != db_mtime_after
                or zip_mtime_before != zip_mtime_after
                or metadata_before != metadata_after
            )
            else "remote GTFS freshness check completed; local SQLite reused"
            if database_ready
            else "startup check failed before a usable SQLite DB was available"
        )
        _print_cp_report(
            ok=ok,
            action=action,
            db_path=manager.db_path,
            zip_path=manager.gtfs_zip_path,
            metadata_path=manager.metadata_path,
            gtfs_url=CP_GTFS_URL,
            stations_url=CP_STATIONS_URL,
            vehicles_url=CP_VEHICLES_URL,
            counts=counts,
            aml_station_count=len(aml_stations),
            message=message,
        )
        return {
            "ok": ok,
            "stops": stop_count,
            "aml_stations": len(aml_stations),
            "counts": counts,
            "message": message,
        }
    except Exception as exc:
        _startup_log(f"❌ CP GTFS + live station layer: failed to load ({exc})")
        return {
            "ok": False,
            "stops": 0,
            "aml_stations": 0,
            "message": f"CP GTFS SQLite database failed to load: {exc}",
        }


def _check_carris_metropolitana_readiness() -> Dict[str, Any]:
    """Validate that Carris Metropolitana reference datasets are reachable."""
    try:
        from tools.carrismetropolitana_api import (
            CARRIS_LINES_URL,
            CARRIS_ROUTES_URL,
            CARRIS_STOPS_URL,
            load_carris_metropolitana_lines,
            load_carris_metropolitana_routes,
            load_carris_metropolitana_stops,
        )

        stops = load_carris_metropolitana_stops(force_reload=False)
        lines = load_carris_metropolitana_lines(force_reload=False)
        routes = load_carris_metropolitana_routes(force_reload=False)
        ok = bool(stops and lines and routes)
        message = (
            "Carris Metropolitana reachable "
            f"({len(stops)} stops, {len(lines)} lines, {len(routes)} routes)"
            if ok
            else "Carris Metropolitana API is unreachable"
        )
        counts = {"stops": len(stops), "lines": len(lines), "routes": len(routes)}
        _print_carris_metropolitana_report(
            ok=ok,
            stops_url=CARRIS_STOPS_URL,
            lines_url=CARRIS_LINES_URL,
            routes_url=CARRIS_ROUTES_URL,
            counts=counts,
            message=message,
        )
        return {
            "ok": ok,
            "stops": len(stops),
            "lines": len(lines),
            "routes": len(routes),
            "counts": counts,
            "message": message,
        }
    except Exception as exc:
        _startup_log(f"❌ Carris Metropolitana reference layer: failed to load ({exc})")
        return {
            "ok": False,
            "stops": 0,
            "lines": 0,
            "routes": 0,
            "message": f"Carris Metropolitana API check failed: {exc}",
        }


def pre_warm_vector_store() -> bool:
    """Load the VisitLisboa vector store once before the first request."""
    try:
        from agent.utils.vector_db_release import ensure_vector_db_from_release
        from tools.visitlisboa_api import initialize_vector_store

        release_status = ensure_vector_db_from_release()
        if not release_status.ok:
            print(f"⚠️ Vector DB artifact unavailable: {release_status.message}", flush=True)
            return False
        if release_status.downloaded:
            size_mb = (release_status.size_bytes or 0) / (1024 * 1024)
            print(f"✅ Vector DB artifact downloaded ({size_mb:.1f} MB)", flush=True)
        else:
            print(
                f"✅ Vector DB artifact ready at {release_status.path} "
                f"({_format_size_mb(release_status.path / 'chroma.sqlite3')})",
                flush=True,
            )

        initialize_vector_store()
        print("✅ KnowledgeBase object initialized and ready for semantic search", flush=True)
        return True
    except Exception as exc:
        print(f"❌ KnowledgeBase initialization failed: {exc}", flush=True)
        return False


def pre_warm_transport_networks() -> Dict[str, Any]:
    """Warm the static transport datasets required by first-turn routing."""
    _print_startup_header()
    details = {
        "carris": _check_carris_urban_readiness(),
        "metro": _check_metro_readiness(),
        "cp": _check_cp_readiness(),
        "carris_metropolitana": _check_carris_metropolitana_readiness(),
    }
    statuses = {
        network_name: str(network_detail.get("message") or "")
        for network_name, network_detail in details.items()
    }
    overall_ok = all(bool(network_detail.get("ok")) for network_detail in details.values())
    summary = format_transport_layer_summary(details, overall_ok)
    _startup_log(summary)
    _startup_log(STARTUP_LOG_SEPARATOR)

    return {
        "ok": overall_ok,
        "statuses": statuses,
        "details": details,
        "summary": summary,
    }


def run_startup_preload(language: str = "pt", force_refresh: bool = False) -> Dict[str, Any]:
    """Load one-time shared resources needed by the multi-agent runtime.

    Args:
        language: UI language code used for user-facing status messages.
        force_refresh: Whether to bypass the successful in-process preload cache.

    Returns:
        Structured startup readiness status for the app or a container entrypoint.
    """
    global _STARTUP_PRELOAD_STATUS

    with _STARTUP_PRELOAD_LOCK:
        if (
            not force_refresh
            and _STARTUP_PRELOAD_STATUS is not None
            and bool(_STARTUP_PRELOAD_STATUS.get("ok", False))
        ):
            return _copy_startup_status_for_language(_STARTUP_PRELOAD_STATUS, language)

        transport_preload = pre_warm_transport_networks()
        transport_ok = bool(transport_preload.get("ok", False))
        transport_status = str(transport_preload.get("summary") or "")
        kb_ok = pre_warm_vector_store()
        startup_status = {
            "transport_ok": transport_ok,
            "transport_status": transport_status,
            "transport_summary": transport_preload.get("summary"),
            "transport_details": transport_preload.get("statuses", {}),
            "transport_network_details": transport_preload.get("details", {}),
            "kb_ok": kb_ok,
            "kb_status": _localized_kb_status(kb_ok, language),
            "ok": transport_ok and kb_ok,
        }

        if bool(startup_status.get("ok", False)):
            _STARTUP_PRELOAD_STATUS = dict(startup_status)

        return startup_status
