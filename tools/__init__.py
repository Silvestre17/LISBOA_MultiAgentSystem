# ==========================================================================
# Master Thesis - Tools Package
#   - André Filipe Gomes Silvestre, 20240502
#
#   Authoritative export registry for the runtime tool layer.
#   The `__all__` list below defines the 45 exported LangChain tools that
#   public documentation and coverage manifests should count.
# ==========================================================================

from tools.carris_api import (
    carris_find_routes_between,
    carris_get_arrivals,  # Real-time arrivals at a stop
    carris_get_next_departures,
    carris_get_realtime_vehicles,
    carris_get_routes,
    carris_get_service_frequency,  # Bus/tram service frequency (headway)
    carris_get_stops,
    carris_vehicle_eta,  # ETA calculation for specific route
)

# Carris Metropolitana (Suburban buses)
from tools.carrismetropolitana_api import (
    find_bus_routes,  # Bus routing between locations
    find_direct_bus_lines,  # Direct bus line connections
    get_bus_next_departures,  # Bus route schedule/stops
    get_bus_realtime_locations,  # Real-time bus GPS locations
    get_carris_metropolitana_alerts,
    get_carris_metropolitana_stop_info,
    get_real_time_bus_positions,
    search_carris_metropolitana_lines,
)

# CP (Comboios de Portugal) - Trains
from tools.cp_api import (
    get_cp_routes,  # CP train routes/lines
    get_train_frequency,  # Train service frequency (headway)
    get_train_schedule,  # GTFS-based schedule departures
    get_train_status,  # Real-time train status from comboios.live
    plan_train_trip,  # Plan a train trip between stations
    search_cp_stations,  # Search CP train stations in AML
)
from tools.dados_abertos import (
    find_nearby_services,
    find_place_in_datasets,  # Search places by name across datasets
    get_dataset_details,
    list_available_datasets,
    list_service_categories,  # Browse service categories from Lisboa Aberta
)
from tools.ipma_api import (
    get_current_weather_summary,
    get_portugal_weather_overview,  # Weather for all Portugal locations
    get_weather_forecast,
    get_weather_warnings,
)

# Metro de Lisboa (Official API with OAuth2)
from tools.metrolisboa_api import (
    find_nearest_metro,  # Find nearest metro station by GPS
    get_all_metro_stations,  # List all metro stations
    get_metro_frequency,  # Train frequency schedules
    get_metro_line_wait_times,  # Wait times for entire line
    get_metro_status,
    get_metro_wait_time,  # Real-time metro wait times
)

# Multi-modal transport (requires all transport APIs)
from tools.transport_api import get_route_between_stations, get_transport_summary
from tools.visitlisboa_api import (
    get_event_categories,
    get_place_categories,
    search_cultural_events,
    search_lisbon_knowledge,
    search_places_attractions,
)

# Web Knowledge (History, Culture, Real-time facts)
from tools.web_knowledge import search_history_culture

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
