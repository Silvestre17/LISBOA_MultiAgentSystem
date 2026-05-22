# ==========================================================================
# Master Thesis - Tools Package
#   - André Filipe Gomes Silvestre, 20240502
#
#   Authoritative export registry for the runtime tool layer.
#   The `__all__` list below defines the 45 exported LangChain tools that
#   public documentation and coverage manifests should count.
#
#   This package intentionally uses lazy exports. Importing one tool module
#   must not import every optional dependency used by unrelated tools.
# ==========================================================================

from __future__ import annotations

from importlib import import_module
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    # Weather (IPMA)
    from tools.ipma_api import (
        get_weather_warnings,
        get_weather_forecast,
        get_current_weather_summary,
        get_portugal_weather_overview,
    )
    # Transport - Metro
    from tools.metrolisboa_api import (
        get_metro_status,
        get_metro_wait_time,
        get_metro_line_wait_times,
        find_nearest_metro,
        get_metro_frequency,
        get_all_metro_stations,
    )
    # Transport - Bus (Carris Metropolitana)
    from tools.carrismetropolitana_api import (
        get_real_time_bus_positions,
        get_carris_metropolitana_alerts,
        get_carris_metropolitana_stop_info,
        search_carris_metropolitana_lines,
        find_bus_routes,
        get_bus_realtime_locations,
        get_bus_next_departures,
        find_direct_bus_lines,
    )
    # Transport - Train (CP)
    from tools.cp_api import (
        get_train_status,
        search_cp_stations,
        get_train_schedule,
        get_cp_routes,
        plan_train_trip,
        get_train_frequency,
    )
    # Transport - Multi-modal
    from tools.transport_api import (
        get_transport_summary,
        get_route_between_stations,
    )
    # Open Data (Lisboa Aberta)
    from tools.dados_abertos import (
        find_nearby_services,
        list_available_datasets,
        get_dataset_details,
        find_place_in_datasets,
        list_service_categories,
    )
    # VisitLisboa (Events & Places)
    from tools.visitlisboa_api import (
        search_cultural_events,
        search_places_attractions,
        get_event_categories,
        get_place_categories,
        search_lisbon_knowledge,
    )
    # Transport - Carris Urban (Buses & Trams)
    from tools.carris_api import (
        carris_get_stops,
        carris_get_routes,
        carris_get_next_departures,
        carris_find_routes_between,
        carris_get_realtime_vehicles,
        carris_get_arrivals,
        carris_vehicle_eta,
        carris_get_service_frequency,
    )
    # Web Knowledge (History, Culture)
    from tools.web_knowledge import (
        search_history_culture,
    )


_TOOL_MODULES: dict[str, str] = {
    # Weather (IPMA) - 4 tools
    "get_weather_warnings": "tools.ipma_api",
    "get_weather_forecast": "tools.ipma_api",
    "get_current_weather_summary": "tools.ipma_api",
    "get_portugal_weather_overview": "tools.ipma_api",

    # Transport - Metro - 6 tools
    "get_metro_status": "tools.metrolisboa_api",
    "get_metro_wait_time": "tools.metrolisboa_api",
    "get_metro_line_wait_times": "tools.metrolisboa_api",
    "find_nearest_metro": "tools.metrolisboa_api",
    "get_metro_frequency": "tools.metrolisboa_api",
    "get_all_metro_stations": "tools.metrolisboa_api",

    # Transport - Bus (Carris Metropolitana) - 8 tools
    "get_real_time_bus_positions": "tools.carrismetropolitana_api",
    "get_carris_metropolitana_alerts": "tools.carrismetropolitana_api",
    "get_carris_metropolitana_stop_info": "tools.carrismetropolitana_api",
    "search_carris_metropolitana_lines": "tools.carrismetropolitana_api",
    "find_bus_routes": "tools.carrismetropolitana_api",
    "get_bus_realtime_locations": "tools.carrismetropolitana_api",
    "get_bus_next_departures": "tools.carrismetropolitana_api",
    "find_direct_bus_lines": "tools.carrismetropolitana_api",

    # Transport - Train (CP) - 6 tools
    "get_train_status": "tools.cp_api",
    "search_cp_stations": "tools.cp_api",
    "get_train_schedule": "tools.cp_api",
    "get_cp_routes": "tools.cp_api",
    "plan_train_trip": "tools.cp_api",
    "get_train_frequency": "tools.cp_api",

    # Transport - Multi-modal - 2 tools
    "get_transport_summary": "tools.transport_api",
    "get_route_between_stations": "tools.transport_api",

    # Open Data (Lisboa Aberta) - 5 tools
    "find_nearby_services": "tools.dados_abertos",
    "list_available_datasets": "tools.dados_abertos",
    "get_dataset_details": "tools.dados_abertos",
    "find_place_in_datasets": "tools.dados_abertos",
    "list_service_categories": "tools.dados_abertos",

    # VisitLisboa (Events & Places) - 5 tools
    "search_cultural_events": "tools.visitlisboa_api",
    "search_places_attractions": "tools.visitlisboa_api",
    "get_event_categories": "tools.visitlisboa_api",
    "get_place_categories": "tools.visitlisboa_api",
    "search_lisbon_knowledge": "tools.visitlisboa_api",

    # Transport - Carris Urban (Buses & Trams) - 8 tools
    "carris_get_stops": "tools.carris_api",
    "carris_get_routes": "tools.carris_api",
    "carris_get_next_departures": "tools.carris_api",
    "carris_find_routes_between": "tools.carris_api",
    "carris_get_realtime_vehicles": "tools.carris_api",
    "carris_get_arrivals": "tools.carris_api",
    "carris_vehicle_eta": "tools.carris_api",
    "carris_get_service_frequency": "tools.carris_api",

    # Web Knowledge (History, Culture) - 1 tool
    "search_history_culture": "tools.web_knowledge",
}

__all__ = [
    # Weather (IPMA) - 4 tools
    "get_weather_warnings",
    "get_weather_forecast",
    "get_current_weather_summary",
    "get_portugal_weather_overview",

    # Transport - Metro - 6 tools
    "get_metro_status",
    "get_metro_wait_time",
    "get_metro_line_wait_times",
    "find_nearest_metro",
    "get_metro_frequency",
    "get_all_metro_stations",

    # Transport - Bus (Carris Metropolitana) - 8 tools
    "get_real_time_bus_positions",
    "get_carris_metropolitana_alerts",
    "get_carris_metropolitana_stop_info",
    "search_carris_metropolitana_lines",
    "find_bus_routes",
    "get_bus_realtime_locations",
    "get_bus_next_departures",
    "find_direct_bus_lines",

    # Transport - Train (CP) - 6 tools
    "get_train_status",
    "search_cp_stations",
    "get_train_schedule",
    "get_cp_routes",
    "plan_train_trip",
    "get_train_frequency",

    # Transport - Multi-modal - 2 tools
    "get_transport_summary",
    "get_route_between_stations",

    # Open Data (Lisboa Aberta) - 5 tools
    "find_nearby_services",
    "list_available_datasets",
    "get_dataset_details",
    "find_place_in_datasets",
    "list_service_categories",

    # VisitLisboa (Events & Places) - 5 tools
    "search_cultural_events",
    "search_places_attractions",
    "get_event_categories",
    "get_place_categories",
    "search_lisbon_knowledge",

    # Transport - Carris Urban (Buses & Trams) - 8 tools
    "carris_get_stops",
    "carris_get_routes",
    "carris_get_next_departures",
    "carris_find_routes_between",
    "carris_get_realtime_vehicles",
    "carris_get_arrivals",
    "carris_vehicle_eta",
    "carris_get_service_frequency",

    # Web Knowledge (History, Culture) - 1 tool
    "search_history_culture",
]


def __getattr__(name: str) -> Any:
    """Load exported tool objects only when they are explicitly requested.

    Args:
        name: Attribute name requested from the package.

    Returns:
        The exported tool object from its owning module.

    Raises:
        AttributeError: If the name is not part of the public tool registry.
    """
    module_name = _TOOL_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module 'tools' has no attribute {name!r}")

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the package attributes plus lazy exported tool names."""
    return sorted(set(globals()) | set(__all__))
