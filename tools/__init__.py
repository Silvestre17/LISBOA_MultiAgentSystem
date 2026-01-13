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
    get_metro_wait_time,           # Real-time metro wait times
    get_metro_line_wait_times,     # Wait times for entire line
    find_nearest_metro,            # Find nearest metro station by GPS
    get_metro_frequency,           # Train frequency schedules
    get_all_metro_stations,        # List all metro stations
    get_carris_alerts,
    get_carris_stop_info,
    search_carris_lines,
    get_train_status,
    get_transport_summary,
    get_route_between_stations,
    find_bus_routes,               # Bus routing between locations
    get_bus_realtime_locations,    # Real-time bus GPS locations
    get_bus_schedule,              # Bus route schedule/stops
    search_cp_stations             # Search CP train stations in AML
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
    
    # Transport - Metro
    "get_metro_status",
    "get_metro_wait_time",
    "get_metro_line_wait_times",
    "find_nearest_metro",
    "get_metro_frequency",
    "get_all_metro_stations",
    
    # Transport - Bus (Carris Metropolitana)
    "get_carris_alerts",
    "get_carris_stop_info",
    "search_carris_lines",
    "find_bus_routes",
    "get_bus_realtime_locations",
    "get_bus_schedule",
    
    # Transport - Train (CP)
    "get_train_status",
    "search_cp_stations",
    
    # Transport - Multi-modal
    "get_transport_summary",
    "get_route_between_stations",
    
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
