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

import os
import sqlite3
from typing import Any, Dict, Optional, Tuple

from config import Config


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


def _invoke_startup_tool_without_tracing(tool: Any, args: Optional[Dict[str, Any]] = None) -> Any:
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
        from tools.carris_api import CARRIS_DB_PATH, CarrisGTFSManager

        manager = CarrisGTFSManager()
        database_ready = manager.ensure_database(force_update=False)
        stop_count = _count_sqlite_rows(CARRIS_DB_PATH, "stops") if database_ready else 0
        ok = bool(database_ready and stop_count > 0)
        message = (
            f"Carris Urban ready ({stop_count} stops)"
            if ok
            else "Carris Urban GTFS database is not populated"
        )
        return {"ok": ok, "stops": stop_count, "message": message}
    except Exception as exc:
        return {
            "ok": False,
            "stops": 0,
            "message": f"Carris Urban GTFS database failed to load: {exc}",
        }


def _check_metro_readiness() -> Dict[str, Any]:
    """Validate Metro stations plus line-status availability or fallback mode."""
    try:
        from tools.metrolisboa_api import get_metro_status, load_metro_stations

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
        return {
            "ok": ok,
            "mode": metro_mode if ok else "unreachable",
            "stations": station_count,
            "message": message,
        }
    except Exception as exc:
        return {
            "ok": False,
            "mode": "unreachable",
            "stations": 0,
            "message": f"Metro readiness check failed: {exc}",
        }


def _check_cp_readiness() -> Dict[str, Any]:
    """Validate that the CP GTFS SQLite database exists and contains stops."""
    try:
        from tools.cp_api import get_gtfs_manager, load_cp_aml_stations

        manager = get_gtfs_manager()
        database_ready = manager.ensure_database(force_refresh=False)
        stop_count = _count_sqlite_rows(str(manager.db_path), "stops") if database_ready else 0
        aml_stations = load_cp_aml_stations(force_reload=False) if database_ready else {}
        ok = bool(database_ready and stop_count > 0)
        message = (
            f"CP ready ({stop_count} stops, {len(aml_stations)} AML stations)"
            if ok
            else "CP GTFS SQLite database is not populated"
        )
        return {
            "ok": ok,
            "stops": stop_count,
            "aml_stations": len(aml_stations),
            "message": message,
        }
    except Exception as exc:
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
        return {
            "ok": ok,
            "stops": len(stops),
            "lines": len(lines),
            "routes": len(routes),
            "message": message,
        }
    except Exception as exc:
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
        from tools.visitlisboa_api import initialize_vector_store

        initialize_vector_store()
        return True
    except Exception:
        return False


def prepare_transport_database() -> Tuple[bool, str]:
    """Prepare the Carris Urban support database used by runtime tools."""
    try:
        carris_detail = _check_carris_urban_readiness()
        if not carris_detail["ok"]:
            return False, str(carris_detail["message"])

        from tools.carris_api import CARRIS_DB_PATH

        db_size_mb = os.path.getsize(CARRIS_DB_PATH) / (1024 * 1024)
        return True, f"Base de dados pronta ({db_size_mb:.0f} MB, {carris_detail['stops']} paragens)"
    except Exception:
        return False, "Não foi possível preparar a base de dados de transportes"


def pre_warm_transport_networks() -> Dict[str, Any]:
    """Warm the static transport datasets required by first-turn routing."""
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

    return {
        "ok": overall_ok,
        "statuses": statuses,
        "details": details,
        "summary": summary,
    }


def run_startup_preload(
    language: str = "pt",
    use_multi_agent: Optional[bool] = None,
) -> Dict[str, Any]:
    """Load one-time shared resources needed by runtime entrypoints."""
    if use_multi_agent is None:
        use_multi_agent = bool(getattr(Config, "USE_MULTI_AGENT", False))

    transport_preload = pre_warm_transport_networks()
    transport_ok = bool(transport_preload.get("ok", False))
    transport_status = str(transport_preload.get("summary") or "")
    kb_ok = True
    kb_status: Optional[str] = None

    if use_multi_agent:
        kb_ok = pre_warm_vector_store()
        kb_status = (
            "Base de conhecimento pronta."
            if kb_ok and language == "pt"
            else "Knowledge base ready."
            if kb_ok
            else "Não foi possível carregar a base de conhecimento."
            if language == "pt"
            else "Could not load the knowledge base."
        )

    return {
        "transport_ok": transport_ok,
        "transport_status": transport_status,
        "transport_summary": transport_preload.get("summary"),
        "transport_details": transport_preload.get("statuses", {}),
        "transport_network_details": transport_preload.get("details", {}),
        "kb_ok": kb_ok,
        "kb_status": kb_status,
        "ok": transport_ok and kb_ok,
    }
