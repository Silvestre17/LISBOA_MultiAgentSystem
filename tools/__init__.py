# ==========================================================================
# Master Thesis - Tools Package
#   - André Filipe Gomes Silvestre, 20240502
# ==========================================================================

from tools.ipma_api import (
    get_weather_warnings,
    get_weather_forecast,
    get_current_weather_summary,
    get_portugal_weather_overview   # Weather for all Portugal locations
)

# Metro de Lisboa (Official API with OAuth2)
from tools.metrolisboa_api import (
    get_metro_status,
    get_metro_wait_time,           # Real-time metro wait times
    get_metro_line_wait_times,     # Wait times for entire line
    find_nearest_metro,            # Find nearest metro station by GPS
    get_metro_frequency,           # Train frequency schedules
    get_all_metro_stations,        # List all metro stations
)

# Carris Metropolitana (Suburban buses)
from tools.carrismetropolitana_api import (
    get_carris_metropolitana_alerts,
    get_carris_metropolitana_stop_info,
    search_carris_metropolitana_lines,
    find_bus_routes,               # Bus routing between locations
    get_bus_realtime_locations,    # Real-time bus GPS locations
    get_bus_next_departures,       # Bus route schedule/stops
)

# CP (Comboios de Portugal) - Trains
from tools.cp_api import (
    get_train_status,              # Real-time train status from comboios.live
    search_cp_stations,            # Search CP train stations in AML
    get_train_schedule,            # GTFS-based schedule departures
    get_cp_routes,                 # CP train routes/lines
    initialize_cp_gtfs,            # Initialize/update GTFS database
)

# Multi-modal transport (requires all transport APIs)
from tools.transport_api import (
    get_transport_summary,
    get_route_between_stations,
)

from tools.dados_abertos import (
    find_nearby_services,
    list_available_datasets,
    get_dataset_details,
    find_place_in_datasets          # Search places by name across datasets
)

from tools.visitlisboa_api import (
    search_cultural_events,
    search_places_attractions,
    get_event_categories,
    get_place_categories,
    search_lisbon_knowledge
)

from tools.carris_api import (
    carris_get_stops,
    carris_get_routes,
    carris_get_next_departures,
    carris_find_routes_between,
    carris_get_realtime_vehicles,
)

__all__ = [
    # Weather (IPMA)
    "get_weather_warnings",
    "get_weather_forecast",
    "get_current_weather_summary",
    "get_portugal_weather_overview",
    
    # Transport - Metro
    "get_metro_status",
    "get_metro_wait_time",
    "get_metro_line_wait_times",
    "find_nearest_metro",
    "get_metro_frequency",
    "get_all_metro_stations",
    
    # Transport - Bus (Carris Metropolitana)
    "get_carris_metropolitana_alerts",
    "get_carris_metropolitana_stop_info",
    "search_carris_metropolitana_lines",
    "find_bus_routes",
    "get_bus_realtime_locations",
    "get_bus_next_departures",
    
    # Transport - Train (CP)
    "get_train_status",
    "search_cp_stations",
    "get_train_schedule",
    "get_cp_routes",
    "initialize_cp_gtfs",
    
    # Transport - Multi-modal
    "get_transport_summary",
    "get_route_between_stations",
    
    # Open Data (Lisboa Aberta)
    "find_nearby_services",
    "list_available_datasets",
    "get_dataset_details",
    "find_place_in_datasets",
    
    # VisitLisboa (Events & Places)
    "search_cultural_events",
    "search_places_attractions",
    "get_event_categories",
    "get_place_categories",
    "search_lisbon_knowledge",
    
    # Transport - Carris (Urban Lisbon Buses & Trams)
    "carris_get_stops",
    "carris_get_routes",
    "carris_get_next_departures",
    "carris_find_routes_between",
    "carris_get_realtime_vehicles",
]
