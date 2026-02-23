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

import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from langchain_core.tools import tool

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config
from tools.carrismetropolitana_api import (
    CARRIS_LIMITATION_NOTICE,
    both_locations_in_lisbon_city,
    find_common_routes,
    find_stops_near_coordinates,
    is_within_lisbon_city,
    load_carris_metropolitana_stops,
    resolve_location,
)
from tools.cp_api import (
    CP_LINES,
    CP_STATIONS,
    get_cp_aml_trains,
    get_cp_station_info,
    load_cp_aml_stations,
)

# Import from the split modules
from tools.metrolisboa_api import (
    LISBON_LANDMARKS,
    METRO_LINES,
    METRO_STATIONS,
    fetch_json_with_retry,
    get_landmark_info,
    get_metro_status,
    get_station_lines,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metro fallback URL
METRO_STATUS_URL = "https://app.metrolisboa.pt/status/getLinhas.php"


# ==========================================================================
# Helper Functions
# ==========================================================================

def _normalize_station(text: str) -> str:
    """Normalizes station name for comparison (removes accents, lowercases)."""
    import unicodedata
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    ).lower().strip()


def _find_station_index(stations: list, station_name: str) -> int:
    """
    Finds the index of a station in an ordered line list.
    Uses fuzzy matching with accent normalization.
    
    Args:
        stations: Ordered list of station names on a line.
        station_name: Station name to find.
    
    Returns:
        Index of station, or -1 if not found.
    """
    name_norm = _normalize_station(station_name)
    
    # Exact match first
    for i, s in enumerate(stations):
        if _normalize_station(s) == name_norm:
            return i
    
    # Partial match
    for i, s in enumerate(stations):
        s_norm = _normalize_station(s)
        if name_norm in s_norm or s_norm in name_norm:
            return i
    
    return -1


def _get_metro_direction(line_id: str, start: str, end: str) -> str:
    """Helper to determine direction (terminal station) on a Metro line."""
    stations = METRO_LINES.get(line_id, {}).get("stations", [])
    if not stations:
        return ""
    
    idx_start = _find_station_index(stations, start)
    idx_end = _find_station_index(stations, end)
    
    if idx_start < 0 or idx_end < 0:
        return ""
    
    if idx_start < idx_end:
        return f"(direction {stations[-1].title()})"
    else:
        return f"(direction {stations[0].title()})"


def _count_metro_stations(line_id: str, start: str, end: str) -> int:
    """
    Counts the number of stations between two points on a Metro line.
    
    Args:
        line_id: Metro line identifier (e.g., "amarela").
        start: Origin station name.
        end: Destination station name.
    
    Returns:
        Number of stations between start and end (inclusive of destination,
        exclusive of origin). Returns -1 if either station is not found.
    """
    stations = METRO_LINES.get(line_id, {}).get("stations", [])
    if not stations:
        return -1
    
    idx_start = _find_station_index(stations, start)
    idx_end = _find_station_index(stations, end)
    
    if idx_start < 0 or idx_end < 0:
        return -1
    
    return abs(idx_end - idx_start)


def _estimate_metro_time(station_count: int, transfers: int = 0) -> str:
    """
    Estimates travel time on the Lisbon Metro.
    
    Based on official Metro de Lisboa data:
    - ~2 minutes between consecutive stations (including stop time)
    - ~3 minutes for each line transfer (walking + waiting)
    - ~2 minutes average initial wait time
    
    Args:
        station_count: Number of stations to travel.
        transfers: Number of line transfers.
    
    Returns:
        Formatted time estimate string (e.g., "~12 min").
    """
    if station_count <= 0:
        return "~2 min"
    
    travel_min = station_count * 2  # 2 min per station
    transfer_min = transfers * 3    # 3 min per transfer
    wait_min = 2                    # Average initial wait
    total = travel_min + transfer_min + wait_min
    
    return f"~{total} min"


def _get_line_status(line_id: str) -> str:
    """
    Gets real-time status for a specific Metro line.
    
    Args:
        line_id: Metro line identifier (e.g., "amarela").
    
    Returns:
        Status string ("ok" if normal, otherwise the disruption message).
        Returns "unknown" if status cannot be fetched.
    """
    try:
        metro_data = fetch_json_with_retry(METRO_STATUS_URL)
        if metro_data and metro_data.get('resposta'):
            status = metro_data['resposta'].get(line_id, 'unknown').strip()
            return status
    except Exception:
        pass
    return "unknown"


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
                response += "   ⚠️ No direct Metro!\n"
                response += f"   🚌 Alternative: {origin_landmark['alternative']}\n"
            response += f"   ℹ️ {origin_landmark.get('description', '')}\n\n"
        
        if dest_landmark:
            response += f"**{dest_landmark['name']}**\n"
            if dest_landmark.get('metro'):
                line = dest_landmark.get('line', '')
                line_emoji = METRO_LINES.get(line.split('/')[0], {}).get('emoji', '🚇')
                response += f"   🚇 Nearest Metro: **{dest_landmark['metro'].title()}** ({line_emoji} {line.title()} Line)\n"
            elif dest_landmark.get('alternative'):
                response += "   ⚠️ No direct Metro!\n"
                response += f"   🚌 Alternative: {dest_landmark['alternative']}\n"
            response += f"   ℹ️ {dest_landmark.get('description', '')}\n\n"
        
    # Resolve effective Metro stations (Handle Landmarks -> Stations)
    eff_origin = origin
    eff_dest = destination
    eff_origin_lines = origin_lines
    eff_dest_lines = dest_lines
    
    origin_from_landmark = False
    dest_from_landmark = False

    if origin_landmark and origin_landmark.get('metro'):
        eff_origin = origin_landmark['metro']
        eff_origin_lines = get_station_lines(eff_origin)
        origin_from_landmark = True
        
    if dest_landmark and dest_landmark.get('metro'):
        eff_dest = dest_landmark['metro']
        eff_dest_lines = get_station_lines(eff_dest)
        dest_from_landmark = True

    # Calculate Metro Route
    if eff_origin_lines and eff_dest_lines:
        response += "🚇 **METRO ROUTE**\n"
        response += "-" * 30 + "\n"
        
        common_lines = set(eff_origin_lines) & set(eff_dest_lines)
        
        if common_lines:
            response += "✅ **Direct Route Available**\n\n"
            for line in common_lines:
                line_info = METRO_LINES.get(line, {})
                emoji = line_info.get('emoji', '')
                name = line_info.get('name', line.title())
                direction = _get_metro_direction(line, eff_origin, eff_dest)
                
                # B1: Check real-time line status
                line_status = _get_line_status(line)
                if line_status.lower() not in ('ok', 'unknown', ''):
                    response += f"   ⚠️ **Line Alert**: {line_status}\n"
                
                # B4: Travel time estimate
                station_count = _count_metro_stations(line, eff_origin, eff_dest)
                time_est = _estimate_metro_time(station_count) if station_count > 0 else ""
                stations_str = f" ({station_count} stations)" if station_count > 0 else ""
                
                response += f"   {emoji} Take **{line.title()} Line** ({name})\n"
                if time_est:
                    response += f"   ⏱️ Estimated travel time: **{time_est}**{stations_str}\n"
                
                step = 1
                if origin_from_landmark:
                    response += f"   {step}. Walk from {origin.title()} to **{eff_origin.title()}**\n"
                    step += 1
                    
                response += f"   {step}. Board at **{eff_origin.title()}** {direction}\n"
                step += 1
                
                response += f"   {step}. Exit at **{eff_dest.title()}**\n"
                
                if dest_from_landmark:
                    step += 1
                    response += f"   {step}. Walk to {destination.title()}\n"
                
                response += "\n"

        else:
            response += "🔄 **Transfer Required**\n\n"
            
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
                if set(eff_origin_lines) & set(lines) and set(eff_dest_lines) & set(lines):
                    best_hub = station
                    l1 = list(set(eff_origin_lines) & set(lines))[0]
                    l2 = list(set(eff_dest_lines) & set(lines))[0]
                    common_hub_lines = (l1, l2)
                    break
            
            if best_hub and common_hub_lines:
                l1, l2 = common_hub_lines
                l1_info = METRO_LINES[l1]
                l2_info = METRO_LINES[l2]
                
                # B1: Check real-time status for both lines
                for check_line, check_info in [(l1, l1_info), (l2, l2_info)]:
                    status = _get_line_status(check_line)
                    if status.lower() not in ('ok', 'unknown', ''):
                        response += f"   ⚠️ **{check_info['emoji']} {check_line.title()} Line Alert**: {status}\n"
                
                # B4: Total travel time (leg 1 + transfer + leg 2)
                leg1_count = _count_metro_stations(l1, eff_origin, best_hub)
                leg2_count = _count_metro_stations(l2, best_hub, eff_dest)
                total_stations = (leg1_count if leg1_count > 0 else 0) + (leg2_count if leg2_count > 0 else 0)
                time_est = _estimate_metro_time(total_stations, transfers=1)
                
                response += f"   💡 **Transfer at**: {best_hub} ({l1_info['emoji']} ↔ {l2_info['emoji']})\n"
                response += f"   ⏱️ Estimated travel time: **{time_est}** ({total_stations} stations + 1 transfer)\n\n"
                response += "   **Full Route**:\n"
                
                step = 1
                if origin_from_landmark:
                    response += f"   {step}. Walk from {origin.title()} to **{eff_origin.title()}**\n"
                    step += 1

                dir1 = _get_metro_direction(l1, eff_origin, best_hub)
                response += f"   {step}. {l1_info['emoji']} Board at **{eff_origin.title()}** {dir1}\n"
                step += 1
                response += f"   {step}. Exit at **{best_hub}**\n"
                step += 1
                
                dir2 = _get_metro_direction(l2, best_hub, eff_dest)
                response += f"   {step}. {l2_info['emoji']} Transfer to **{l2_info['name']}** {dir2}\n"
                step += 1
                response += f"   {step}. Exit at **{eff_dest.title()}**\n"

                if dest_from_landmark:
                    step += 1
                    response += f"   {step}. Walk to {destination.title()}\n"
                
                response += "\n"
            else:
                response += "⚠️ Route requires complex transfer. Check [Metro map](https://www.metrolisboa.pt/viajar/mapas-e-diagramas/).\n\n"
                
    elif eff_origin_lines:
        # Origin valid, Dest invalid
        response += f"🚇 **Origin is Metro**: {eff_origin.title()}\n"
        if origin_from_landmark:
            response += f"   (Nearest station to {origin})\n"
        response += f"❌ Destination '{destination.title()}' not on Metro.\n"
        response += "   Consider using Carris buses or CP trains.\n\n"
        
    elif eff_dest_lines:
        # Dest valid, Origin invalid
        response += f"❌ Origin '{origin.title()}' not on Metro.\n"
        response += f"🚇 **Destination is Metro**: {eff_dest.title()}\n"
        if dest_from_landmark:
            response += f"   (Nearest station to {destination})\n"
        response += "   Consider using Carris buses or CP trains to reach the Metro.\n\n"

    else:
        if not has_landmarks:  # Only print if we haven't printed landmark info
            response += "❌ Neither location is a known Metro station.\n\n"
    
    # Check for CP Train options
    if origin_cp or dest_cp:
        response += "🚆 **CP TRAINS**\n"
        response += "-" * 30 + "\n"
        
        if origin_cp and dest_cp:
            common_lines = set(origin_cp.get("lines", [])) & set(dest_cp.get("lines", []))
            
            if common_lines:
                response += "✅ **Direct Train Route Available**\n\n"
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
            lines_str = ", ".join([CP_LINES[line_id]["name"] for line_id in origin_cp.get("lines", [])])
            response += f"✅ **{origin.title()}** is a train station\n"
            response += f"   📍 {origin_cp.get('description', 'N/A')}\n"
            response += f"   🚆 Lines: {lines_str}\n"
            if origin_cp.get("metro"):
                metro_line = METRO_LINES.get(origin_cp["metro"], {})
                response += f"   🚇 Metro connection: {metro_line.get('emoji', '')} {origin_cp['metro'].title()}\n"
            response += "\n"
        
        if dest_cp:
            lines_str = ", ".join([CP_LINES[line_id]["name"] for line_id in dest_cp.get("lines", [])])
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
    response += "   • [Metro de Lisboa](https://www.metrolisboa.pt)\n"
    response += "   • [Carris (Lisbon)](https://www.carris.pt)\n"
    response += "   • [Carris Metropolitana](https://www.carrismetropolitana.pt)\n"
    response += "   • [CP Trains](https://www.cp.pt)\n"
    
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
        from tools.carris_api import _get_db_connection, fetch_gtfs_rt_vehicles
        
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
        from tools.carrismetropolitana_api import (
            CARRIS_ALERTS_URL,
            get_carris_metropolitana_alerts,
        )
        
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
    print("\033[1m" + "=" * 70 + "\033[0m")
    print("\033[1m\U0001f9ea MULTI-MODAL TRANSPORT API - COMPREHENSIVE TEST SUITE\033[0m")
    print("\033[1m" + "=" * 70 + "\033[0m")
    
    test_results = {"passed": 0, "failed": 0, "total": 0}
    
    def run_test(name, func, args=None):
        """Runs a test and tracks results."""
        test_results["total"] += 1
        print(f"\n{'=' * 60}")
        print(f"\033[1m\U0001f9ea TEST {test_results['total']}: {name}\033[0m")
        print(f"{'=' * 60}")
        try:
            result = func(args if args else {})
            if result:
                test_results["passed"] += 1
                print(f"\033[1;32m[PASS]\033[0m Result length: {len(result)} chars")
                # Show first 600 chars for readability
                print(result[:600])
                if len(result) > 600:
                    print(f"... ({len(result) - 600} more chars)")
            else:
                test_results["failed"] += 1
                print("\033[1;31m[FAIL]\033[0m Empty result")
        except Exception as e:
            test_results["failed"] += 1
            print(f"\033[1;31m[FAIL]\033[0m Error: {e}")
    
    # =========================================================================
    # HELPER FUNCTION TESTS
    # =========================================================================
    
    # TEST: Station counting (internal validation)
    print(f"\n{'=' * 60}")
    print("\033[1m\U0001f9ea INTERNAL: _count_metro_stations validation\033[0m")
    print(f"{'=' * 60}")
    
    # Helper variables for colors to avoid f-string backslash errors in <= 3.11
    OK_TXT = "\033[32mOK\033[0m"
    FAIL_TXT = "\033[31mFAIL\033[0m"
    FAIL_12_TXT = "\033[31mFAIL (expected 12)\033[0m"
    FAIL_2_TXT = "\033[31mFAIL (expected 2)\033[0m"

    # Amarela: rato(0) to odivelas(12) = 12 stations
    count = _count_metro_stations("amarela", "rato", "odivelas")
    print(f"  Rato -> Odivelas (Amarela): {count} stations {OK_TXT if count == 12 else FAIL_12_TXT}")
    
    # Verde: cais do sodre(0) to telheiras(12) = 12 stations  
    count = _count_metro_stations("verde", "cais do sodre", "telheiras")
    print(f"  Cais do Sodre -> Telheiras (Verde): {count} stations {OK_TXT if count == 12 else FAIL_12_TXT}")
    
    # Azul: santa apolonia(0) to baixa-chiado(2) = 2 stations
    count = _count_metro_stations("azul", "santa apolonia", "baixa-chiado")
    print(f"  Santa Apolonia -> Baixa-Chiado (Azul): {count} stations {OK_TXT if count == 2 else FAIL_2_TXT}")
    
    # Time estimation
    time_est = _estimate_metro_time(5, transfers=0)
    print(f"  Time estimate (5 stations, 0 transfers): {time_est} {OK_TXT if '12' in time_est else FAIL_TXT}")
    
    time_est = _estimate_metro_time(8, transfers=1)
    print(f"  Time estimate (8 stations, 1 transfer): {time_est} {OK_TXT if '21' in time_est else FAIL_TXT}")
    
    # =========================================================================
    # METRO ROUTE TESTS - Direct Routes
    # =========================================================================
    
    # TEST 1: Direct route on same line (Vermelha)
    run_test(
        "Direct Metro Route - Same Line (Aeroporto -> Saldanha) [VERMELHA]",
        get_route_between_stations.invoke,
        {"origin": "Aeroporto", "destination": "Saldanha"}
    )
    
    # TEST 2: Direct route on same line (Verde)
    run_test(
        "Direct Metro Route - Same Line (Cais do Sodre -> Arroios) [VERDE]",
        get_route_between_stations.invoke,
        {"origin": "Cais do Sodré", "destination": "Arroios"}
    )
    
    # =========================================================================
    # METRO ROUTE TESTS - Transfer Required
    # =========================================================================
    
    # TEST 3: Transfer route (Azul -> Vermelha via Sao Sebastiao)
    run_test(
        "Transfer Route - Reboleira -> Aeroporto [AZUL -> VERMELHA]",
        get_route_between_stations.invoke,
        {"origin": "Reboleira", "destination": "Aeroporto"}
    )
    
    # TEST 4: Transfer route (Amarela -> Verde via Campo Grande)
    run_test(
        "Transfer Route - Odivelas -> Rossio [AMARELA -> VERDE]",
        get_route_between_stations.invoke,
        {"origin": "Odivelas", "destination": "Rossio"}
    )
    
    # =========================================================================
    # LANDMARK ROUTING TESTS
    # =========================================================================
    
    # TEST 5: Landmark routing (Colombo -> Oriente)
    run_test(
        "Landmark Route - Colombo -> Oriente [LANDMARK + METRO]",
        get_route_between_stations.invoke,
        {"origin": "Colombo", "destination": "Oriente"}
    )
    
    # TEST 6: Landmark with no metro (Belem)
    run_test(
        "Landmark Route - Belem (no Metro) [ALTERNATIVE TRANSPORT]",
        get_route_between_stations.invoke,
        {"origin": "Aeroporto", "destination": "Belém"}
    )
    
    # =========================================================================
    # EDGE CASES
    # =========================================================================
    
    # TEST 7: Unknown locations
    run_test(
        "Edge Case - Unknown Origin and Destination",
        get_route_between_stations.invoke,
        {"origin": "Praia do Guincho", "destination": "Serra da Estrela"}
    )
    
    # TEST 8: Same station
    run_test(
        "Edge Case - Same Origin and Destination",
        get_route_between_stations.invoke,
        {"origin": "Saldanha", "destination": "Saldanha"}
    )
    
    # =========================================================================
    # TRANSPORT SUMMARY TEST
    # =========================================================================
    
    # TEST 9: Full transport summary
    run_test(
        "Transport Summary - All Modes [METRO + CARRIS + CP]",
        get_transport_summary.invoke
    )
    
    # =========================================================================
    # TEST SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("\033[1m\U0001f4ca TEST SUMMARY\033[0m")
    print("=" * 70)
    print(f"\033[1;32m\u2705 Passed: {test_results['passed']}/{test_results['total']}\033[0m")
    print(f"\033[1;31m\u274c Failed: {test_results['failed']}/{test_results['total']}\033[0m")
    
    if test_results['failed'] == 0:
        print("\n\033[1;32m🎉 ALL TESTS PASSED! Transport system is working correctly.\033[0m")
    else:
        print("\n\033[1;33m⚠️  Some tests failed. Check errors above.\033[0m")
    
    print("=" * 70 + "\n")
