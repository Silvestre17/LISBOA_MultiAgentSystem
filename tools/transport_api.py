# ==========================================================================
# Master Thesis - Multi-Modal Transport API Tools
#   - André Filipe Gomes Silvestre, 20240502
# 
#   Multi-modal routing and transport summary for Lisbon Metropolitan Area.
#   This module combines data from:
#     - Metro de Lisboa (metrolisboa_api.py)
#     - Carris Metropolitana (carrismetropolitana_api.py)
#     - CP Trains (cp_api.py)
#     - Carris Urban (carris_api.py)
# 
#   For individual transport APIs, use the specific modules:
#     - tools.metrolisboa_api: Metro stations, wait times, status
#     - tools.carrismetropolitana_api: Suburban bus routes, stops, alerts
#     - tools.cp_api: Train status, stations, delays
#     - tools.carris_api: Urban Lisbon buses and trams
# ==========================================================================

# Required libraries:
# pip install requests langchain-core

import os
import sys
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from collections import defaultdict

import requests
from langchain_core.tools import tool

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

# Import from the split modules
from tools.metrolisboa_api import (
    get_metro_status,
    METRO_LINES,
    METRO_STATIONS,
    LISBON_LANDMARKS,
    get_station_lines,
    get_landmark_info,
    fetch_json_with_retry,
)

from tools.carrismetropolitana_api import (
    CARRIS_LIMITATION_NOTICE,
    resolve_location,
    find_stops_near_coordinates,
    find_common_routes,
    is_within_lisbon_city,
    both_locations_in_lisbon_city,
    load_carris_metropolitana_stops,
)

from tools.cp_api import (
    CP_LINES,
    CP_STATIONS,
    get_cp_station_info,
    get_cp_aml_trains,
    load_cp_aml_stations,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metro fallback URL
METRO_STATUS_URL = "https://app.metrolisboa.pt/status/getLinhas.php"


# ==========================================================================
# Helper Functions
# ==========================================================================

def _get_metro_direction(line_id: str, start: str, end: str) -> str:
    """Helper to determine direction (terminal station) on a Metro line."""
    stations = METRO_LINES.get(line_id, {}).get("stations", [])
    if not stations:
        return ""
    
    import unicodedata
    def norm(t):
        return ''.join(c for c in unicodedata.normalize('NFD', t) if unicodedata.category(c) != 'Mn').lower().strip()
    
    start_c = next((s for s in stations if norm(s) == norm(start)), None)
    if not start_c:
        start_c = next((s for s in stations if norm(start) in norm(s) or norm(s) in norm(start)), start)
    
    end_c = next((s for s in stations if norm(s) == norm(end)), None)
    if not end_c:
        end_c = next((s for s in stations if norm(end) in norm(s) or norm(s) in norm(end)), end)
    
    try:
        idx_start = stations.index(start_c)
        idx_end = stations.index(end_c)
        if idx_start < idx_end:
            return f"(direction {stations[-1].title()})"
        else:
            return f"(direction {stations[0].title()})"
    except ValueError:
        return ""


# ==========================================================================
# LangChain Tools
# ==========================================================================

@tool
def get_route_between_stations(origin: str, destination: str) -> str:
    """
    Plans a multi-modal route between two locations using Metro, buses, and trains.
    
    This is the MAIN ROUTING TOOL for planning trips across Lisbon. It:
    - Detects Metro stations and shows direct/transfer routes
    - Identifies Lisbon landmarks (Colombo, Belém, etc.) and suggests best transport
    - Identifies CP train stations and suggests train connections
    - Recommends bus alternatives where appropriate
    
    For BUS-ONLY routes, use `find_bus_routes` instead.
    
    Args:
        origin: Starting location (Metro station, train station, or landmark).
        destination: Destination location.
    
    Returns:
        str: Multi-modal route suggestions with Metro, train, and bus options.
    """
    origin_lower = origin.lower().strip()
    dest_lower = destination.lower().strip()
    
    response = f"🗺️ **Route: {origin.title()} → {destination.title()}**\n"
    response += "=" * 50 + "\n\n"
    
    # Check if origin or destination is a known landmark
    origin_landmark = get_landmark_info(origin)
    dest_landmark = get_landmark_info(destination)
    
    # Check if both are Metro stations
    origin_lines = get_station_lines(origin)
    dest_lines = get_station_lines(destination)
    
    # Check if they are CP train stations
    origin_cp = get_cp_station_info(origin)
    dest_cp = get_cp_station_info(destination)
    
    has_metro = bool(origin_lines or dest_lines)
    has_train = bool(origin_cp or dest_cp)
    has_landmarks = bool(origin_landmark or dest_landmark)
    
    # Handle landmarks first
    if has_landmarks:
        response += "📍 **LOCATION INFORMATION**\n"
        response += "-" * 30 + "\n"
        
        if origin_landmark:
            response += f"**{origin_landmark['name']}**\n"
            if origin_landmark.get('metro'):
                line = origin_landmark.get('line', '')
                line_emoji = METRO_LINES.get(line.split('/')[0], {}).get('emoji', '🚇')
                response += f"   🚇 Nearest Metro: **{origin_landmark['metro'].title()}** ({line_emoji} {line.title()} Line)\n"
            elif origin_landmark.get('alternative'):
                response += f"   ⚠️ No direct Metro!\n"
                response += f"   🚌 Alternative: {origin_landmark['alternative']}\n"
            response += f"   ℹ️ {origin_landmark.get('description', '')}\n\n"
        
        if dest_landmark:
            response += f"**{dest_landmark['name']}**\n"
            if dest_landmark.get('metro'):
                line = dest_landmark.get('line', '')
                line_emoji = METRO_LINES.get(line.split('/')[0], {}).get('emoji', '🚇')
                response += f"   🚇 Nearest Metro: **{dest_landmark['metro'].title()}** ({line_emoji} {line.title()} Line)\n"
            elif dest_landmark.get('alternative'):
                response += f"   ⚠️ No direct Metro!\n"
                response += f"   🚌 Alternative: {dest_landmark['alternative']}\n"
            response += f"   ℹ️ {dest_landmark.get('description', '')}\n\n"
        
        # Calculate route between landmark metros
        if origin_landmark and dest_landmark:
            origin_metro = origin_landmark.get('metro')
            dest_metro = dest_landmark.get('metro')
            
            if origin_metro and dest_metro:
                origin_lines = get_station_lines(origin_metro)
                dest_lines = get_station_lines(dest_metro)
                
                response += "🚇 **METRO ROUTE**\n"
                response += "-" * 30 + "\n"
                
                common_lines = set(origin_lines) & set(dest_lines)
                if common_lines:
                    response += f"✅ **Direct Route Available**\n\n"
                    for line in common_lines:
                        line_info = METRO_LINES.get(line, {})
                        direction = _get_metro_direction(line, origin_metro, dest_metro)
                        response += f"   {line_info.get('emoji', '')} Take **{line.title()} Line**\n"
                        response += f"   1. Walk from {origin.title()} to **{origin_metro.title()}**\n"
                        response += f"   2. Board at **{origin_metro.title()}** {direction}\n"
                        response += f"   3. Exit at **{dest_metro.title()}**\n"
                        response += f"   4. Walk to {dest_landmark['name']}\n\n"
                else:
                    # Transfer logic
                    hubs = [
                        ("Marquês de Pombal", ["amarela", "azul"]),
                        ("Saldanha", ["amarela", "vermelha"]),
                        ("Alameda", ["verde", "vermelha"]),
                        ("Baixa-Chiado", ["azul", "verde"]),
                        ("Campo Grande", ["amarela", "verde"]),
                        ("São Sebastião", ["vermelha", "azul"]),
                    ]
                    
                    transfer_hub = None
                    for station, lines in hubs:
                        if set(origin_lines) & set(lines) and set(dest_lines) & set(lines):
                            transfer_hub = station
                            break
                    
                    if transfer_hub:
                        hub_lines = next(lines for st, lines in hubs if st == transfer_hub)
                        l1 = list(set(origin_lines) & set(hub_lines))[0]
                        l2 = list(set(dest_lines) & set(hub_lines))[0]
                        l1_info = METRO_LINES[l1]
                        l2_info = METRO_LINES[l2]
                        
                        response += f"🔄 **Transfer Required**\n"
                        response += f"   💡 **Suggested Transfer**: {transfer_hub} ({l1_info['emoji']} ↔ {l2_info['emoji']})\n\n"
                        response += f"   **Full Route**:\n"
                        response += f"   1. Walk from {origin.title()} to **{origin_metro.title()}**\n"
                        
                        dir1 = _get_metro_direction(l1, origin_metro, transfer_hub)
                        response += f"   2. {l1_info['emoji']} Board at **{origin_metro.title()}** {dir1}\n"
                        response += f"   3. Exit at **{transfer_hub}**\n"
                        
                        dir2 = _get_metro_direction(l2, transfer_hub, dest_metro)
                        response += f"   4. {l2_info['emoji']} Transfer to **{l2_info['name']}** {dir2}\n"
                        response += f"   5. Exit at **{dest_metro.title()}**\n"
                        response += f"   6. Walk to {dest_landmark['name']}\n\n"
                    else:
                        response += f"⚠️ Complex route. Check [Metro map](https://www.metrolisboa.pt/viajar/mapas-e-diagramas/).\n\n"
                
                return response
            
            elif origin_metro and not dest_metro:
                response += "📋 **RECOMMENDATION**\n"
                response += "-" * 30 + "\n"
                response += f"Since {dest_landmark['name']} has no nearby Metro:\n"
                response += f"   👉 {dest_landmark.get('alternative', 'Use bus or train')}\n\n"
                return response
    
    # Handle Metro stations
    if origin_lines and dest_lines:
        response += "🚇 **METRO ROUTE**\n"
        response += "-" * 30 + "\n"
        
        common_lines = set(origin_lines) & set(dest_lines)
        if common_lines:
            response += f"✅ **Direct Route Available**\n\n"
            for line in common_lines:
                line_info = METRO_LINES.get(line, {})
                emoji = line_info.get('emoji', '')
                name = line_info.get('name', line.title())
                direction = _get_metro_direction(line, origin, destination)
                
                response += f"   {emoji} Take **{line.title()} Line** ({name})\n"
                response += f"   📍 Board at: {origin.title()} {direction}\n"
                response += f"   📍 Exit at: {destination.title()}\n\n"
        else:
            response += f"🔄 **Transfer Required**\n\n"
            
            transfer_stations = [
                ("Marquês de Pombal", ["amarela", "azul"]),
                ("Saldanha", ["amarela", "vermelha"]),
                ("Alameda", ["verde", "vermelha"]),
                ("Baixa-Chiado", ["azul", "verde"]),
                ("Campo Grande", ["amarela", "verde"]),
                ("São Sebastião", ["vermelha", "azul"]),
            ]
            
            best_hub = None
            common_hub_lines = None
            
            for station, lines in transfer_stations:
                if set(origin_lines) & set(lines) and set(dest_lines) & set(lines):
                    best_hub = station
                    l1 = list(set(origin_lines) & set(lines))[0]
                    l2 = list(set(dest_lines) & set(lines))[0]
                    common_hub_lines = (l1, l2)
                    break
            
            if best_hub and common_hub_lines:
                l1, l2 = common_hub_lines
                l1_info = METRO_LINES[l1]
                l2_info = METRO_LINES[l2]
                
                response += f"   💡 **Suggested Transfer**: {best_hub} ({l1_info['emoji']} ↔ {l2_info['emoji']})\n\n"
                response += f"   **Full Route**:\n"
                
                dir1 = _get_metro_direction(l1, origin, best_hub)
                response += f"   1. {l1_info['emoji']} Board at **{origin.title()}** {dir1}\n"
                response += f"   2. Exit at **{best_hub}**\n"
                
                dir2 = _get_metro_direction(l2, best_hub, destination)
                response += f"   3. {l2_info['emoji']} Transfer to **{l2_info['name']}** {dir2}\n"
                response += f"   4. Exit at **{destination.title()}**\n\n"
            else:
                response += f"   📍 From: {origin.title()} ({', '.join([METRO_LINES[l]['emoji'] + ' ' + l.title() for l in origin_lines])})\n"
                response += f"   📍 To: {destination.title()} ({', '.join([METRO_LINES[l]['emoji'] + ' ' + l.title() for l in dest_lines])})\n"
                response += f"   ⚠️ Route requires complex transfer. Please check [Metro map](https://www.metrolisboa.pt/viajar/mapas-e-diagramas/).\n\n"
    
    elif origin_lines:
        response += f"🚇 **Origin is a Metro station**: {origin.title()}\n"
        response += f"   Lines: {', '.join([METRO_LINES[l]['emoji'] + ' ' + l.title() + ' Line' for l in origin_lines])}\n\n"
        response += f"❌ Destination '{destination.title()}' is not a known Metro station.\n"
        response += "   Consider using Carris buses or CP trains.\n\n"
    
    elif dest_lines:
        response += f"❌ Origin '{origin.title()}' is not a known Metro station.\n\n"
        response += f"🚇 **Destination is a Metro station**: {destination.title()}\n"
        response += f"   Lines: {', '.join([METRO_LINES[l]['emoji'] + ' ' + l.title() + ' Line' for l in dest_lines])}\n\n"
        response += "   Consider using Carris buses or CP trains to reach the Metro.\n\n"
    
    elif not has_landmarks:
        response += "❌ Neither location is a known Metro station.\n\n"
    
    # Check for CP Train options
    if origin_cp or dest_cp:
        response += "🚆 **CP TRAINS**\n"
        response += "-" * 30 + "\n"
        
        if origin_cp and dest_cp:
            common_lines = set(origin_cp.get("lines", [])) & set(dest_cp.get("lines", []))
            
            if common_lines:
                response += f"✅ **Direct Train Route Available**\n\n"
                for line in common_lines:
                    line_info = CP_LINES.get(line, {"name": line.title()})
                    response += f"   🚆 Take **{line_info['name']}**\n"
                    response += f"   📍 Board at: {origin.title()}\n"
                    response += f"   📍 Exit at: {destination.title()}\n"
                    if line_info.get("frequency"):
                        response += f"   🕒 Frequency: {line_info['frequency']}\n"
                    response += "\n"
                return response
            else:
                response += f"⚠️ No direct train line linking {origin.title()} and {destination.title()}.\n"
                response += "   You may need to transfer at a major hub (e.g., Entrecampos, Oriente, Sete Rios).\n\n"
        
        if origin_cp:
            lines_str = ", ".join([CP_LINES[l]["name"] for l in origin_cp.get("lines", [])])
            response += f"✅ **{origin.title()}** is a train station\n"
            response += f"   📍 {origin_cp.get('description', 'N/A')}\n"
            response += f"   🚆 Lines: {lines_str}\n"
            if origin_cp.get("metro"):
                metro_line = METRO_LINES.get(origin_cp["metro"], {})
                response += f"   🚇 Metro connection: {metro_line.get('emoji', '')} {origin_cp['metro'].title()}\n"
            response += "\n"
        
        if dest_cp:
            lines_str = ", ".join([CP_LINES[l]["name"] for l in dest_cp.get("lines", [])])
            response += f"✅ **{destination.title()}** is a train station\n"
            response += f"   📍 {dest_cp.get('description', 'N/A')}\n"
            response += f"   🚆 Lines: {lines_str}\n"
            if dest_cp.get("metro"):
                metro_line = METRO_LINES.get(dest_cp["metro"], {})
                response += f"   🚇 Metro connection: {metro_line.get('emoji', '')} {dest_cp['metro'].title()}\n"
            response += "\n"
    
    # Add suggestion to check official sources
    response += "-" * 30 + "\n"
    response += "💡 **More information:**\n"
    response += "   • Metro: metrolisboa.pt\n"
    response += "   • Buses (Lisbon): carris.pt\n"
    response += "   • Buses (Metropolitan): carrismetropolitana.pt\n"
    response += "   • Trains: cp.pt\n"
    
    return response


@tool
def get_transport_summary() -> str:
    """
    Gets a quick summary of all public transport status in Lisbon.
    Combines Metro, buses, and trains into a single overview.
    
    Returns:
        str: Combined transport status summary.
    """
    response = "🚇 Lisbon Transport Summary\n"
    response += "=" * 40 + "\n\n"
    
    # 1. Metro Status
    response += "🚇 METRO DE LISBOA\n"
    response += "-" * 20 + "\n"
    
    metro_data = fetch_json_with_retry(METRO_STATUS_URL)
    if metro_data and metro_data.get('resposta'):
        resp = metro_data['resposta']
        all_ok = True
        for line_key, line_info in METRO_LINES.items():
            status = resp.get(line_key, 'Unknown').strip()
            if status.lower() != 'ok':
                all_ok = False
                response += f"   {line_info['emoji']} {line_key.title()}: ⚠️ {status}\n"
        
        if all_ok:
            response += "   ✅ All lines operating normally\n"
    else:
        response += "   ❌ Status unavailable\n"
    
    response += "\n"
    
    # 2. Carris (Urban Lisbon)
    response += "🚋 CARRIS (LISBON URBAN)\n"
    response += "-" * 20 + "\n"
    
    try:
        from tools.carris_api import fetch_gtfs_rt_vehicles, _get_db_connection
        
        vehicles = fetch_gtfs_rt_vehicles()
        if vehicles:
            response += f"   ✅ {len(vehicles)} vehicles tracked\n"
        else:
            response += "   ⚠️ Real-time data unavailable\n"
    except Exception as e:
        logger.warning(f"Carris Urban data failed: {e}")
        response += "   ⚠️ Urban data unavailable\n"
    
    response += "\n"
    
    # 3. Carris Metropolitana (Suburban)
    response += "🚌 CARRIS METROPOLITANA (SUBURBAN)\n"
    response += "-" * 20 + "\n"
    
    try:
        from tools.carrismetropolitana_api import get_carris_metropolitana_alerts, CARRIS_ALERTS_URL
        
        alerts_data = fetch_json_with_retry(CARRIS_ALERTS_URL)
        if alerts_data:
            # API returns a list directly, not a dict with 'entity' key
            alerts = alerts_data if isinstance(alerts_data, list) else alerts_data.get('entity', [])
            if alerts:
                response += f"   ⚠️ {len(alerts)} active alert(s)\n"
            else:
                response += "   ✅ No active alerts\n"
        else:
            response += "   ⚠️ Alert data unavailable\n"
    except Exception as e:
        logger.warning(f"Carris Metropolitana alerts failed: {e}")
        response += "   ⚠️ Suburban data unavailable\n"
    
    response += "\n"
    
    # 4. CP Trains (AML)
    response += "🚆 CP TRAINS (AML)\n"
    response += "-" * 20 + "\n"
    
    try:
        aml_trains = get_cp_aml_trains()
        if aml_trains:
            total = len(aml_trains)
            delayed = sum(1 for t in aml_trains if (t.get('delay') or 0) > 60)
            
            response += f"   📊 {total} trains serving AML\n"
            if delayed > 0:
                response += f"   ⚠️ {delayed} train(s) with delays > 1 min\n"
            else:
                response += "   ✅ Trains operating normally\n"
        else:
            response += "   ⚠️ Train data unavailable\n"
    except Exception as e:
        logger.warning(f"CP train data failed: {e}")
        response += "   ⚠️ Train data unavailable\n"
    
    response += "\n"
    
    # Footer
    response += "-" * 40 + "\n"
    response += f"📅 Updated: {datetime.now().strftime('%H:%M:%S')}\n"
    response += "💡 Use specific tools for detailed info.\n"
    
    return response


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 MULTI-MODAL TRANSPORT API - TEST SUITE\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")
    
    print("\n1. Testing get_transport_summary...")
    result = get_transport_summary.invoke({})
    print(result)
    
    print("\n2. Testing get_route_between_stations (Metro)...")
    result = get_route_between_stations.invoke({
        "origin": "Aeroporto",
        "destination": "Baixa-Chiado"
    })
    print(result[:800])
    
    print("\n3. Testing get_route_between_stations (Landmark)...")
    result = get_route_between_stations.invoke({
        "origin": "Colombo",
        "destination": "Oriente"
    })
    print(result[:800])
    
    print("\n\033[1;32m✅ Multi-modal transport API tests complete!\033[0m")
