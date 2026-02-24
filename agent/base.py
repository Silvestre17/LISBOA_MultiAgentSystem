# ==========================================================================
# Master Thesis - Agent Tools Configuration
#   - André Filipe Gomes Silvestre, 20240502
# 
#   Central registry for all tools used by the agentic system.
#   Imports tools from specialized API modules and exposes them for
#   agent binding.
# ==========================================================================

import os
import sys

# Add parent directory to path for imports when running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Carris Urban (City buses & trams) (8)
from tools.carris_api import (
    carris_find_routes_between,
    carris_get_arrivals,
    carris_get_next_departures,
    carris_get_realtime_vehicles,
    carris_get_routes,
    carris_get_service_frequency,
    carris_get_stops,
    carris_vehicle_eta,
)

# Carris Metropolitana (Suburban buses) (8)
from tools.carrismetropolitana_api import (
    find_bus_routes,
    find_direct_bus_lines,
    get_bus_next_departures,
    get_bus_realtime_locations,
    get_carris_metropolitana_alerts,
    get_carris_metropolitana_stop_info,
    get_real_time_bus_positions,
    search_carris_metropolitana_lines,
)

# CP (Comboios de Portugal) - Trains (6)
from tools.cp_api import (
    get_cp_routes,
    get_train_frequency,
    get_train_schedule,
    get_train_status,
    plan_train_trip,
    search_cp_stations,
)

# Dados Abertos (Lisboa Aberta) (5)
from tools.dados_abertos import (
    find_nearby_services,
    find_place_in_datasets,
    get_dataset_details,
    list_available_datasets,
    list_service_categories,
)

# IPMA Weather Tools (4)
from tools.ipma_api import (
    get_current_weather_summary,
    get_portugal_weather_overview,
    get_weather_forecast,
    get_weather_warnings,
)

# Metro de Lisboa (Official API with OAuth2) (6)
from tools.metrolisboa_api import (
    find_nearest_metro,
    get_all_metro_stations,
    get_metro_frequency,
    get_metro_line_wait_times,
    get_metro_status,
    get_metro_wait_time,
)

# Multi-modal transport routing (2)
from tools.transport_api import get_route_between_stations, get_transport_summary

# VisitLisboa Tools (Events & Places) (5)
from tools.visitlisboa_api import (
    get_event_categories,
    get_place_categories,
    search_cultural_events,
    search_lisbon_knowledge,
    search_places_attractions,
)

# Web Knowledge (1)
from tools.web_knowledge import search_history_culture


def get_tools():
    """
    Returns a list of all available tools for the agent.
    
    This collection represents the full capabilities of the system,
    spanning transport, weather, open data, tourism knowledge, and web search.
    
    Returns:
        List[Tool]: List of 45 LangChain tools.
    """
    return [
        # IPMA Weather Tools (4)
        get_weather_warnings,
        get_weather_forecast,
        get_current_weather_summary,
        get_portugal_weather_overview,

        # Transport - Metro de Lisboa (6)
        get_metro_status,
        get_metro_wait_time,
        get_metro_line_wait_times,
        find_nearest_metro,
        get_metro_frequency,
        get_all_metro_stations,

        # Transport - Carris Metropolitana (Suburban buses) (8)
        get_carris_metropolitana_alerts,
        get_carris_metropolitana_stop_info,
        search_carris_metropolitana_lines,
        find_direct_bus_lines,
        get_real_time_bus_positions,
        get_bus_realtime_locations,
        get_bus_next_departures,
        find_bus_routes,

        # Transport - Carris Urban (City buses & trams) (8)
        carris_get_stops,
        carris_get_routes,
        carris_get_next_departures,
        carris_find_routes_between,
        carris_get_realtime_vehicles,
        carris_get_arrivals,
        carris_vehicle_eta,
        carris_get_service_frequency,

        # Transport - CP (Comboios de Portugal) (6)
        get_train_status,
        search_cp_stations,
        get_train_schedule,
        get_cp_routes,
        plan_train_trip,
        get_train_frequency,

        # Transport - Multi-modal (2)
        get_transport_summary,
        get_route_between_stations,

        # Dados Abertos (Lisboa Aberta) (5)
        find_nearby_services,
        list_available_datasets,
        get_dataset_details,
        find_place_in_datasets,
        list_service_categories,

        # VisitLisboa Tools (Events & Places) (5)
        search_cultural_events,
        search_places_attractions,
        get_event_categories,
        get_place_categories,
        search_lisbon_knowledge,

        # Web Knowledge (1)
        search_history_culture,
    ]


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 Agent Tools Registry Test\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")
    
    tools = get_tools()
    print(f"\n✅ Successfully loaded {len(tools)} tools.")
    
    # Group by source module (heuristic)
    import inspect
    modules = {}
    
    for t in tools:
        mod = inspect.getmodule(t.func).__name__ if hasattr(t, 'func') else "unknown"
        if mod not in modules:
            modules[mod] = []
        modules[mod].append(t.name)
        
    for mod, tools_list in modules.items():
        clean_mod = mod.split('.')[-1]
        print(f"\n\033[1;36m📦 {clean_mod} ({len(tools_list)} tools):\033[0m")
        for tool_name in tools_list:
            print(f"   - {tool_name}")
            
    print("\n" + "=" * 60)
