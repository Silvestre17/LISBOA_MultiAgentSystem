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

from langchain_core.tools import tool

# Import tools from respective modules
from tools.transport_api import (
    get_metro_status,
    get_metro_wait_time,
    get_metro_line_wait_times,
    find_nearest_metro,
    get_metro_frequency,
    get_all_metro_stations,
    get_carris_metropolitana_alerts,
    get_carris_metropolitana_stop_info,
    search_carris_metropolitana_lines,
    get_bus_realtime_locations,
    get_bus_next_departures,
    get_train_status,
    search_cp_stations,
    get_route_between_stations,
    find_bus_routes,
    get_transport_summary
)

from tools.carris_api import (
    carris_get_stops,
    carris_get_routes,
    carris_get_arrivals,
    carris_get_next_departures,
    carris_find_routes_between,
    carris_get_realtime_vehicles,
    carris_vehicle_eta
)

from tools.ipma_api import (
    get_weather_warnings,
    get_weather_forecast,
    get_portugal_weather_overview,
    get_current_weather_summary
)

from tools.dados_abertos import (
    find_nearby_services,
    list_available_datasets,
    get_dataset_details,
    find_place_in_datasets
)

from tools.visitlisboa_api import (
    search_cultural_events,
    search_places_attractions,
    get_event_categories,
    get_place_categories,
    search_lisbon_knowledge
)

def get_tools():
    """
    Returns a list of all available tools for the agent.
    
    This collection represents the full capabilities of the system,
    spanning transport, weather, open data, and tourism knowledge.
    
    Returns:
        List[Tool]: List of LangChain tools.
    """
    return [
        # Transport Tools (Metro, Bus, Train)
        get_metro_status,
        get_metro_wait_time,
        get_metro_line_wait_times,
        find_nearest_metro,
        get_metro_frequency,
        get_all_metro_stations,
        get_carris_metropolitana_alerts,
        get_carris_metropolitana_stop_info,
        search_carris_metropolitana_lines,
        get_bus_realtime_locations,
        get_bus_next_departures,
        get_train_status,
        search_cp_stations,
        get_route_between_stations,
        find_bus_routes,
        get_transport_summary,
        
        # Carris Urban Tools (Trams & City Buses)
        carris_get_stops,
        carris_get_routes,
        carris_get_arrivals,
        carris_get_next_departures,
        carris_find_routes_between,
        carris_get_realtime_vehicles,
        carris_vehicle_eta,
        
        # IPMA Weather Tools
        get_weather_warnings,
        get_weather_forecast,
        get_portugal_weather_overview,
        get_current_weather_summary,
        
        # Dados Abertos Tools (Public Services)
        find_nearby_services,
        list_available_datasets,
        get_dataset_details,
        find_place_in_datasets,
        
        # VisitLisboa Tools (Cultural Events & Attractions)
        search_cultural_events,
        search_places_attractions,
        get_event_categories,
        get_place_categories,
        search_lisbon_knowledge
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
