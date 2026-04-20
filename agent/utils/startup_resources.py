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
from typing import Any, Dict, Optional, Tuple

from config import Config


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
        from tools.carris_api import CARRIS_DB_PATH, CarrisGTFSManager

        manager = CarrisGTFSManager()
        db_valid = False
        if os.path.exists(CARRIS_DB_PATH):
            needs_update, _ = manager.check_for_updates()
            if not needs_update:
                db_valid = True
        if not db_valid:
            manager.ensure_database(force_update=False)
        db_size_mb = os.path.getsize(CARRIS_DB_PATH) / (1024 * 1024)
        return True, f"Base de dados pronta ({db_size_mb:.0f} MB)"
    except Exception:
        return False, "Não foi possível preparar a base de dados de transportes"


def pre_warm_transport_networks() -> Dict[str, Any]:
    """Warm the static transport datasets required by first-turn routing."""
    statuses: Dict[str, str] = {}
    overall_ok = True

    carris_ok, carris_status = prepare_transport_database()
    statuses["carris"] = carris_status
    overall_ok = overall_ok and carris_ok

    try:
        from tools.metrolisboa_api import load_metro_stations

        metro_stations = load_metro_stations()
        if metro_stations:
            statuses["metro"] = f"Metro pronto ({len(metro_stations)} estações)"
        else:
            statuses["metro"] = "Não foi possível carregar as estações do Metro"
            overall_ok = False
    except Exception:
        statuses["metro"] = "Não foi possível carregar as estações do Metro"
        overall_ok = False

    try:
        from tools.cp_api import get_gtfs_manager, load_cp_aml_stations

        cp_manager = get_gtfs_manager()
        cp_db_ready = cp_manager.ensure_database(force_refresh=False)
        cp_stations = load_cp_aml_stations(force_reload=False)
        if cp_db_ready and cp_stations:
            statuses["cp"] = f"CP pronta ({len(cp_stations)} estações AML)"
        else:
            statuses["cp"] = "Não foi possível preparar os dados da CP"
            overall_ok = False
    except Exception:
        statuses["cp"] = "Não foi possível preparar os dados da CP"
        overall_ok = False

    try:
        from tools.carrismetropolitana_api import (
            load_carris_metropolitana_lines,
            load_carris_metropolitana_routes,
            load_carris_metropolitana_stops,
        )

        cm_stops = load_carris_metropolitana_stops(force_reload=False)
        cm_lines = load_carris_metropolitana_lines(force_reload=False)
        cm_routes = load_carris_metropolitana_routes(force_reload=False)
        if cm_stops and cm_lines and cm_routes:
            statuses["carris_metropolitana"] = (
                "Carris Metropolitana pronta "
                f"({len(cm_stops)} paragens, {len(cm_lines)} linhas, {len(cm_routes)} rotas)"
            )
        else:
            statuses["carris_metropolitana"] = (
                "Não foi possível preparar os dados da Carris Metropolitana"
            )
            overall_ok = False
    except Exception:
        statuses["carris_metropolitana"] = (
            "Não foi possível preparar os dados da Carris Metropolitana"
        )
        overall_ok = False

    return {
        "ok": overall_ok,
        "statuses": statuses,
        "summary": " | ".join(statuses.values()),
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
        "transport_details": transport_preload.get("statuses", {}),
        "kb_ok": kb_ok,
        "kb_status": kb_status,
        "ok": transport_ok and kb_ok,
    }
