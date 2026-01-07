# ==========================================================================
# Master Thesis - Tools Package
#   - André Filipe Gomes Silvestre, 20240502
# ==========================================================================

from tools.ipma_api import (
    get_weather_warnings,
    get_weather_forecast,
    get_current_weather_summary
)

from tools.transport_api import (
    get_metro_status,
    get_carris_alerts,
    get_carris_stop_info,
    search_carris_lines,
    get_train_status,
    get_transport_summary,
    get_route_between_stations
)

from tools.dados_abertos import (
    find_nearby_services,
    list_available_datasets,
    get_dataset_details
)

from tools.visitlisboa_api import (
    search_cultural_events,
    search_places_attractions,
    get_event_categories,
    get_place_categories,
    search_lisbon_knowledge
)

__all__ = [
    # Weather
    "get_weather_warnings",
    "get_weather_forecast",
    "get_current_weather_summary",
    
    # Transport
    "get_metro_status",
    "get_carris_alerts",
    "get_carris_stop_info",
    "search_carris_lines",
    "get_train_status",
    "get_transport_summary",
    "get_route_between_stations",  # NEW: Routing tool
    
    # Open Data (Lisboa Aberta)
    "find_nearby_services",
    "list_available_datasets",
    "get_dataset_details",
    
    # VisitLisboa (Events & Places)
    "search_cultural_events",
    "search_places_attractions",
    "get_event_categories",
    "get_place_categories",
    "search_lisbon_knowledge",
]
