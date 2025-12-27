# ==========================================================================
# Master Thesis - Transport API Tools
#   - André Filipe Gomes Silvestre, 20240502
# 
#   Real-time transport data for Lisbon Metropolitan Area.
#   Features:
#     - Metro de Lisboa: Line status
#     - Carris Metropolitana: Alerts, stops, lines, real-time arrivals
#     - CP (Comboios de Portugal): Train status and delays
# 
#   APIs:
#     - Metro: https://app.metrolisboa.pt/status/getLinhas.php
#     - Carris: https://api.carrismetropolitana.pt/
#     - CP: https://comboios.live/api/
# ==========================================================================

# Required libraries:
# pip install requests langchain-core

import os
import sys
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any, List

import requests
from langchain_core.tools import tool

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request configuration
REQUEST_TIMEOUT = 15  # seconds
MAX_RETRIES = 3
BACKOFF_FACTOR = 2

# ==========================================================================
# API Endpoints
# ==========================================================================

# Metro de Lisboa
METRO_STATUS_URL = "https://app.metrolisboa.pt/status/getLinhas.php"

# Carris Metropolitana
CARRIS_BASE_URL = "https://api.carrismetropolitana.pt/v2"
CARRIS_ALERTS_URL = f"{CARRIS_BASE_URL}/alerts"
CARRIS_STOPS_URL = f"{CARRIS_BASE_URL}/stops"
CARRIS_LINES_URL = f"{CARRIS_BASE_URL}/lines"

# CP (Comboios de Portugal)
CP_STATIONS_URL = "https://comboios.live/api/stations"
CP_VEHICLES_URL = "https://comboios.live/api/vehicles"

# Metro line colors and names
METRO_LINES = {
    "amarela": {"name": "Yellow Line (Rato ↔ Odivelas)", "emoji": "🟡", "color": "#FFCD41"},
    "azul": {"name": "Blue Line (Santa Apolónia ↔ Reboleira)", "emoji": "🔵", "color": "#0075BF"},
    "verde": {"name": "Green Line (Telheiras ↔ Cais do Sodré)", "emoji": "🟢", "color": "#00A651"},
    "vermelha": {"name": "Red Line (S. Sebastião ↔ Aeroporto)", "emoji": "🔴", "color": "#ED1C24"}
}


# ==========================================================================
# Helper Functions
# ==========================================================================

def fetch_json_with_retry(url: str, timeout: int = REQUEST_TIMEOUT) -> Optional[Any]:
    """
    Fetches JSON data from a URL with retry logic.
    
    Args:
        url (str): URL to fetch from.
        timeout (int): Request timeout in seconds.
        
    Returns:
        Optional[Any]: JSON data if successful, None otherwise.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            wait_time = BACKOFF_FACTOR ** attempt
            logger.warning(f"Timeout. Retrying in {wait_time}s...")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait_time)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_FACTOR ** attempt)
        except ValueError:
            logger.error("Invalid JSON response")
            return None
    return None


def format_timestamp(ts: int) -> str:
    """
    Converts Unix timestamp (milliseconds) to readable format.
    
    Args:
        ts (int): Unix timestamp in milliseconds.
        
    Returns:
        str: Formatted datetime string.
    """
    try:
        dt = datetime.fromtimestamp(ts / 1000)
        return dt.strftime("%H:%M:%S")
    except (ValueError, TypeError, OSError):
        return "N/A"


# ==========================================================================
# Metro de Lisboa Tools
# ==========================================================================

@tool
def get_metro_status() -> str:
    """
    Gets the current operational status of all Lisbon Metro lines.
    
    Returns:
        str: Status of each metro line (Yellow, Blue, Green, Red).
        
    Example:
        >>> get_metro_status()
    """
    data = fetch_json_with_retry(METRO_STATUS_URL)
    
    if not data:
        return "❌ Failed to fetch Metro status. The API may be temporarily unavailable."
    
    response_data = data.get('resposta', {})
    
    if not response_data:
        return "❌ Unexpected response format from Metro API."
    
    response = "🚇 Metro de Lisboa Status\n"
    response += "=" * 40 + "\n\n"
    
    all_ok = True
    
    for line_key, line_info in METRO_LINES.items():
        status = response_data.get(line_key, 'Unknown').strip()
        emoji = line_info['emoji']
        name = line_info['name']
        
        if status.lower() == 'ok':
            status_emoji = "✅"
            status_text = "Normal service"
        else:
            status_emoji = "⚠️"
            status_text = status
            all_ok = False
        
        response += f"{emoji} {name}\n"
        response += f"   {status_emoji} {status_text}\n\n"
    
    if all_ok:
        response += "✅ All lines operating normally."
    else:
        response += "⚠️ Some lines have service disruptions."
    
    return response


# ==========================================================================
# Carris Metropolitana Tools
# ==========================================================================

@tool
def get_carris_alerts() -> str:
    """
    Gets active service alerts from Carris Metropolitana (bus network).
    Includes information about route disruptions, detours, and service changes.
    
    Returns:
        str: List of active alerts with affected routes and timing.
        
    Example:
        >>> get_carris_alerts()
    """
    data = fetch_json_with_retry(CARRIS_ALERTS_URL)
    
    if not data:
        return "❌ Failed to fetch Carris alerts. The API may be temporarily unavailable."
    
    if not isinstance(data, list):
        return "❌ Unexpected response format from Carris API."
    
    if not data:
        return "✅ No active alerts from Carris Metropolitana.\n\n🚌 Bus services operating normally."
    
    # Filter for currently active alerts
    now = datetime.now().timestamp()
    active_alerts = []
    
    for alert in data:
        # Check if alert is currently active
        active_periods = alert.get('active_period', [])
        is_active = False
        
        for period in active_periods:
            start = period.get('start', 0)
            end = period.get('end', float('inf'))
            if start <= now <= end:
                is_active = True
                break
        
        if is_active or not active_periods:
            active_alerts.append(alert)
    
    if not active_alerts:
        return "✅ No currently active alerts from Carris Metropolitana."
    
    response = f"🚌 Carris Metropolitana Alerts ({len(active_alerts)} active)\n"
    response += "=" * 50 + "\n\n"
    
    for i, alert in enumerate(active_alerts[:10], 1):  # Limit to 10 alerts
        # Get alert details
        effect = alert.get('effect', 'UNKNOWN')
        cause = alert.get('cause', 'UNKNOWN')
        
        # Get description (try Portuguese first)
        header = alert.get('header_text', {})
        description = alert.get('description_text', {})
        
        header_text = "N/A"
        desc_text = ""
        
        if header:
            translations = header.get('translation', [])
            for t in translations:
                if t.get('language') == 'pt':
                    header_text = t.get('text', 'N/A')
                    break
            if header_text == "N/A" and translations:
                header_text = translations[0].get('text', 'N/A')
        
        if description:
            translations = description.get('translation', [])
            for t in translations:
                if t.get('language') == 'pt':
                    desc_text = t.get('text', '')
                    break
        
        # Get affected routes
        informed = alert.get('informed_entity', [])
        routes = [e.get('route_id', '') for e in informed if e.get('route_id')]
        routes_str = ", ".join(routes[:5]) if routes else "N/A"
        if len(routes) > 5:
            routes_str += f" (+{len(routes) - 5} more)"
        
        # Effect emoji
        effect_emojis = {
            "NO_SERVICE": "🚫",
            "REDUCED_SERVICE": "⚠️",
            "SIGNIFICANT_DELAYS": "🕐",
            "DETOUR": "↩️",
            "STOP_MOVED": "📍",
            "OTHER_EFFECT": "ℹ️"
        }
        effect_emoji = effect_emojis.get(effect, "📢")
        
        response += f"{i}. {effect_emoji} {header_text}\n"
        response += f"   📍 Routes: {routes_str}\n"
        response += f"   🔸 Cause: {cause.replace('_', ' ').title()}\n"
        response += f"   🔸 Effect: {effect.replace('_', ' ').title()}\n"
        if desc_text and len(desc_text) < 200:
            response += f"   📝 {desc_text}\n"
        response += "\n"
    
    if len(active_alerts) > 10:
        response += f"... and {len(active_alerts) - 10} more alerts.\n"
    
    return response


@tool
def get_carris_stop_info(stop_id: str) -> str:
    """
    Gets information about a specific Carris bus stop including real-time arrivals.
    
    Args:
        stop_id (str): The stop ID (e.g., '060001' for a specific stop).

    Returns:
        str: Stop information and upcoming arrivals.
        
    Example:
        >>> get_carris_stop_info("060001")
    """
    # Get stop details
    stop_url = f"{CARRIS_STOPS_URL}/{stop_id}"
    stop_data = fetch_json_with_retry(stop_url)
    
    if not stop_data:
        return f"❌ Could not find stop with ID: {stop_id}"
    
    # Get real-time arrivals
    realtime_url = f"{CARRIS_STOPS_URL}/{stop_id}/realtime"
    realtime_data = fetch_json_with_retry(realtime_url)
    
    response = "🚏 Bus Stop Information\n"
    response += "=" * 40 + "\n\n"
    
    # Stop details
    name = stop_data.get('name', 'N/A')
    locality = stop_data.get('locality', '')
    lat = stop_data.get('lat', 'N/A')
    lon = stop_data.get('lon', 'N/A')
    lines = stop_data.get('lines', [])
    
    response += f"📍 {name}\n"
    if locality:
        response += f"   📌 {locality}\n"
    response += f"   🗺️ ({lat}, {lon})\n"
    response += f"   🚌 Lines: {', '.join(lines[:10])}"
    if len(lines) > 10:
        response += f" (+{len(lines) - 10} more)"
    response += "\n\n"
    
    # Real-time arrivals
    if realtime_data and isinstance(realtime_data, list) and realtime_data:
        response += "⏱️ Upcoming Arrivals:\n"
        
        for i, arrival in enumerate(realtime_data[:8], 1):
            line_id = arrival.get('line_id', 'N/A')
            headsign = arrival.get('headsign', 'N/A')
            estimated = arrival.get('estimated_arrival', '')
            scheduled = arrival.get('scheduled_arrival', '')
            
            # Calculate minutes until arrival
            arrival_time = estimated or scheduled
            if arrival_time:
                try:
                    arr_dt = datetime.fromisoformat(arrival_time.replace('Z', '+00:00'))
                    now = datetime.now(arr_dt.tzinfo)
                    mins = int((arr_dt - now).total_seconds() / 60)
                    time_str = f"{mins} min" if mins > 0 else "Now"
                except (ValueError, TypeError):
                    time_str = arrival_time[:5] if len(arrival_time) >= 5 else "N/A"
            else:
                time_str = "N/A"
            
            response += f"   {i}. Line {line_id} → {headsign}\n"
            response += f"      ⏰ {time_str}\n"
    else:
        response += "ℹ️ No real-time arrival data available.\n"
    
    return response


@tool
def search_carris_lines(query: str) -> str:
    """
    Searches for Carris bus lines by name or number.
    
    Args:
        query (str): Line number or name to search (e.g., '728', 'aeroporto').

    Returns:
        str: Matching lines with details.
        
    Example:
        >>> search_carris_lines("728")
        >>> search_carris_lines("aeroporto")
    """
    data = fetch_json_with_retry(CARRIS_LINES_URL)
    
    if not data:
        return "❌ Failed to fetch Carris lines data."
    
    if not isinstance(data, list):
        return "❌ Unexpected response format."
    
    # Search for matching lines
    query_lower = query.lower()
    matches = []
    
    for line in data:
        short_name = line.get('short_name', '')
        long_name = line.get('long_name', '')
        line_id = line.get('id', '')
        
        if (query_lower in short_name.lower() or 
            query_lower in long_name.lower() or
            query_lower in line_id.lower()):
            matches.append(line)
    
    if not matches:
        return f"❌ No lines found matching: '{query}'"
    
    response = f"🚌 Carris Lines matching '{query}' ({len(matches)} found)\n"
    response += "=" * 50 + "\n\n"
    
    for i, line in enumerate(matches[:10], 1):
        short_name = line.get('short_name', 'N/A')
        long_name = line.get('long_name', 'N/A')
        municipalities = line.get('municipalities', [])
        
        response += f"{i}. Line {short_name}\n"
        response += f"   📍 {long_name}\n"
        if municipalities:
            response += f"   🏘️ {', '.join(municipalities[:3])}\n"
        response += "\n"
    
    if len(matches) > 10:
        response += f"... and {len(matches) - 10} more lines.\n"
    
    return response


# ==========================================================================
# CP (Comboios de Portugal) Tools
# ==========================================================================

@tool
def get_train_status() -> str:
    """
    Gets real-time status of trains from CP (Comboios de Portugal).
    Shows trains currently in transit with their delays.
    
    Returns:
        str: List of trains with status, delays, and positions.
        
    Example:
        >>> get_train_status()
    """
    data = fetch_json_with_retry(CP_VEHICLES_URL)
    
    if not data:
        return "❌ Failed to fetch train status. The API may be temporarily unavailable."
    
    vehicles = data.get('vehicles', [])
    
    if not vehicles:
        return "ℹ️ No trains currently in service (or data unavailable)."
    
    # Filter for active trains
    active_trains = [v for v in vehicles if v.get('status') == 'IN_TRANSIT']
    
    if not active_trains:
        active_trains = vehicles[:10]  # Show any available if none in transit
    
    response = "🚆 CP Train Status\n"
    response += "=" * 40 + "\n\n"
    response += f"📊 {len(active_trains)} trains currently tracked\n\n"
    
    # Sort by delay (most delayed first)
    active_trains.sort(key=lambda x: x.get('delay', 0) or 0, reverse=True)
    
    for i, train in enumerate(active_trains[:10], 1):
        train_number = train.get('trainNumber', 'N/A')
        delay = train.get('delay', 0) or 0
        status = train.get('status', 'Unknown')
        lat = train.get('latitude')
        lon = train.get('longitude')
        
        # Origin and destination
        origin = train.get('origin', {})
        destination = train.get('destination', {})
        origin_name = origin.get('designation', 'N/A') if origin else 'N/A'
        dest_name = destination.get('designation', 'N/A') if destination else 'N/A'
        
        # Service type
        service = train.get('service', {})
        service_name = service.get('designation', 'N/A') if service else 'N/A'
        
        # Delay indicator
        if delay == 0:
            delay_str = "✅ On time"
        elif delay > 0:
            delay_mins = delay // 60
            delay_str = f"⚠️ {delay_mins} min late" if delay_mins > 0 else f"⚠️ {delay}s late"
        else:
            delay_str = "✅ Ahead of schedule"
        
        response += f"{i}. {service_name} #{train_number}\n"
        response += f"   🚉 {origin_name} → {dest_name}\n"
        response += f"   {delay_str}\n"
        response += f"   📍 Status: {status.replace('_', ' ').title()}\n"
        # Only show coordinates if they are valid floats
        if lat is not None and lon is not None:
            try:
                response += f"   🗺️ ({float(lat):.4f}, {float(lon):.4f})\n"
            except (ValueError, TypeError):
                pass
        response += "\n"
    
    if len(active_trains) > 10:
        response += f"... and {len(active_trains) - 10} more trains.\n"
    
    return response


@tool
def get_transport_summary() -> str:
    """
    Gets a quick summary of all public transport status in Lisbon.
    Combines Metro, buses, and trains into a single overview.
    
    Returns:
        str: Combined transport status summary.
        
    Example:
        >>> get_transport_summary()
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
    
    # 2. Carris Alerts
    response += "🚌 CARRIS METROPOLITANA\n"
    response += "-" * 20 + "\n"
    
    carris_data = fetch_json_with_retry(CARRIS_ALERTS_URL)
    if carris_data is not None:
        if isinstance(carris_data, list) and carris_data:
            response += f"   ⚠️ {len(carris_data)} active alerts\n"
        else:
            response += "   ✅ No active alerts\n"
    else:
        response += "   ❌ Status unavailable\n"
    
    response += "\n"
    
    # 3. Train Status
    response += "🚆 CP TRAINS\n"
    response += "-" * 20 + "\n"
    
    train_data = fetch_json_with_retry(CP_VEHICLES_URL)
    if train_data and train_data.get('vehicles'):
        vehicles = train_data['vehicles']
        delayed = sum(1 for v in vehicles if (v.get('delay') or 0) > 60)
        total = len(vehicles)
        
        if delayed > 0:
            response += f"   ⚠️ {delayed}/{total} trains delayed\n"
        else:
            response += f"   ✅ {total} trains on schedule\n"
    else:
        response += "   ❌ Status unavailable\n"
    
    response += "\n" + "=" * 40 + "\n"
    response += "📅 Updated: " + datetime.now().strftime("%H:%M:%S")
    
    return response


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 Transport API Tools Test\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")
    
    # Test 1: Transport Summary
    print("\n\033[1m🚇 Test 1: Transport Summary\033[0m")
    print("-" * 40)
    result = get_transport_summary.invoke({})
    print(result)
    
    # Test 2: Metro Status
    print("\n\033[1m🚇 Test 2: Metro Status\033[0m")
    print("-" * 40)
    result = get_metro_status.invoke({})
    print(result)
    
    # Test 3: Carris Alerts
    print("\n\033[1m🚌 Test 3: Carris Alerts\033[0m")
    print("-" * 40)
    result = get_carris_alerts.invoke({})
    print(result[:800] + "..." if len(result) > 800 else result)
    
    # Test 4: Train Status
    print("\n\033[1m🚆 Test 4: Train Status\033[0m")
    print("-" * 40)
    result = get_train_status.invoke({})
    print(result)
