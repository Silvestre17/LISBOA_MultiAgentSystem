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

# Key Metro Stations with their lines (for routing assistance)
# Based on official Metro de Lisboa GeoJSON data
# All stations listed in order along each line
METRO_STATIONS = {
    # Yellow Line (Amarela) - Rato ↔ Odivelas
    "rato": ["amarela"],
    "marquês de pombal": ["amarela", "azul"],
    "marques de pombal": ["amarela", "azul"],
    "marques pombal": ["amarela", "azul"],
    "marquês pombal": ["amarela", "azul"],
    "picoas": ["amarela"],
    "saldanha": ["amarela", "vermelha"],
    "campo pequeno": ["amarela"],
    "entre campos": ["amarela"],
    "entrecampos": ["amarela"],
    "cidade universitária": ["amarela"],
    "cidade universitaria": ["amarela"],
    "campo grande": ["amarela", "verde"],
    "quinta das conchas": ["amarela"],
    "lumiar": ["amarela"],
    "ameixoeira": ["amarela"],
    "senhor roubado": ["amarela"],
    "odivelas": ["amarela"],
    
    # Blue Line (Azul) - Santa Apolónia ↔ Reboleira
    "santa apolónia": ["azul"],
    "santa apolonia": ["azul"],
    "terreiro do paço": ["azul"],
    "terreiro do paco": ["azul"],
    "baixa-chiado": ["azul", "verde"],
    "baixa chiado": ["azul", "verde"],
    "restauradores": ["azul"],
    "avenida": ["azul"],
    # "marquês de pombal": ["amarela", "azul"],  # Already listed above
    "parque": ["azul"],
    "são sebastião": ["azul", "vermelha"],
    "sao sebastiao": ["azul", "vermelha"],
    "s. sebastião": ["azul", "vermelha"],
    "praça de espanha": ["azul"],
    "praca de espanha": ["azul"],
    "jardim zoológico": ["azul"],
    "jardim zoologico": ["azul"],
    "laranjeiras": ["azul"],
    "alto dos moinhos": ["azul"],
    "colégio militar": ["azul"],
    "colegio militar": ["azul"],
    "carnide": ["azul"],
    "pontinha": ["azul"],
    "alfornelos": ["azul"],
    "amadora este": ["azul"],
    "reboleira": ["azul"],
    
    # Green Line (Verde) - Cais do Sodré ↔ Telheiras
    "cais do sodré": ["verde"],
    "cais do sodre": ["verde"],
    # "baixa-chiado": ["azul", "verde"],  # Already listed above
    "rossio": ["verde"],
    "martim moniz": ["verde"],
    "intendente": ["verde"],
    "anjos": ["verde"],
    "arroios": ["verde"],
    "alameda": ["verde", "vermelha"],
    "areeiro": ["verde"],
    "roma": ["verde"],
    "alvalade": ["verde"],
    # "campo grande": ["amarela", "verde"],  # Already listed above
    "telheiras": ["verde"],
    
    # Red Line (Vermelha) - S. Sebastião ↔ Aeroporto
    # "são sebastião": ["azul", "vermelha"],  # Already listed above
    # "saldanha": ["amarela", "vermelha"],  # Already listed above
    # "alameda": ["verde", "vermelha"],  # Already listed above
    "olaias": ["vermelha"],
    "bela vista": ["vermelha"],
    "chelas": ["vermelha"],
    "olivais": ["vermelha"],
    "cabo ruivo": ["vermelha"],
    "oriente": ["vermelha"],
    "moscavide": ["vermelha"],
    "encarnação": ["vermelha"],
    "encarnacao": ["vermelha"],
    "aeroporto": ["vermelha"],
}

# CP Train Stations in Lisbon (from official GeoJSON)
# Stations serve different lines depending on the service
CP_STATIONS = {
    # Major hubs (multiple lines)
    "oriente": {
        "lines": ["azambuja", "norte", "beira_alta"],
        "description": "Parque das Nações - Major rail hub",
        "metro": "vermelha"
    },
    "santa apolónia": {
        "santa apolonia": "santa apolónia",
        "lines": ["azambuja", "norte"],
        "description": "Historic central station",
        "metro": "azul"
    },
    "entrecampos": {
        "lines": ["azambuja", "sintra"],
        "description": "North Lisbon hub",
        "metro": "amarela"
    },
    "sete rios": {
        "lines": ["sintra", "azambuja"],
        "description": "Connection to Sintra line",
        "metro": None
    },
    
    # Cascais Line (from Cais do Sodré)
    "cais do sodré": {
        "cais do sodre": "cais do sodré",
        "lines": ["cascais"],
        "description": "Cascais line terminus",
        "metro": "verde"
    },
    "santos": {
        "lines": ["cascais"],
        "description": "Cascais line",
        "metro": None
    },
    "alcântara mar": {
        "alcantara mar": "alcântara mar",
        "lines": ["cascais"],
        "description": "Cascais line",
        "metro": None
    },
    "alcântara terra": {
        "alcantara terra": "alcântara terra",
        "lines": ["cascais"],
        "description": "Cascais line",
        "metro": None
    },
    "belém": {
        "belem": "belém",
        "lines": ["cascais"],
        "description": "Cascais line - Belém area",
        "metro": None
    },
    
    # Sintra Line (from Rossio/Oriente)
    "rossio": {
        "lines": ["sintra"],
        "description": "Sintra line terminus (city center)",
        "metro": "verde"
    },
    "campolide": {
        "lines": ["sintra"],
        "description": "Sintra line",
        "metro": None
    },
    "benfica": {
        "lines": ["sintra"],
        "description": "Sintra line",
        "metro": None
    },
    
    # Azambuja Line (from Santa Apolónia/Oriente)
    "roma areeiro": {
        "roma-areeiro": "roma areeiro",
        "lines": ["azambuja"],
        "description": "Azambuja line",
        "metro": "verde"  # Close to Areeiro metro
    },
    "braço de prata": {
        "braco de prata": "braço de prata",
        "lines": ["azambuja"],
        "description": "Azambuja line",
        "metro": None
    },
    "marvila": {
        "lines": ["azambuja"],
        "description": "Azambuja line",
        "metro": None
    },
    "chelas": {
        "lines": ["azambuja"],
        "description": "Azambuja line",
        "metro": "vermelha"
    },
}

# CP Train Lines
CP_LINES = {
    "cascais": {
        "name": "Linha de Cascais",
        "description": "Cais do Sodré ↔ Cascais (coastal route)",
        "emoji": "🚆",
        "terminus": ["Cais do Sodré", "Cascais"],
        "frequency": "~20 min"
    },
    "sintra": {
        "name": "Linha de Sintra",
        "description": "Rossio/Oriente ↔ Sintra (historic town)",
        "emoji": "🚆",
        "terminus": ["Rossio", "Sintra"],
        "frequency": "~20 min"
    },
    "azambuja": {
        "name": "Linha de Azambuja",
        "description": "Santa Apolónia/Oriente ↔ Azambuja (north suburbs)",
        "emoji": "🚆",
        "terminus": ["Santa Apolónia", "Azambuja"],
        "frequency": "~30 min"
    },
    "sado": {
        "name": "Linha do Sado",
        "description": "Regional line to south (Setúbal, Évora)",
        "emoji": "🚆",
        "terminus": ["Entrecampos", "South regions"],
        "frequency": "Variable"
    },
}


# ==========================================================================
# Helper Functions
# ==========================================================================

def get_station_lines(station_name: str) -> List[str]:
    """
    Returns the metro lines that serve a given station.
    
    Args:
        station_name (str): Name of the station (case-insensitive).
        
    Returns:
        List[str]: List of line names (e.g., ['amarela', 'azul']).
    """
    station_lower = station_name.lower().strip()
    return METRO_STATIONS.get(station_lower, [])


def get_cp_station_info(station_name: str) -> Optional[Dict[str, Any]]:
    """
    Returns information about a CP train station.
    
    Args:
        station_name (str): Name of the station (case-insensitive).
        
    Returns:
        Optional[Dict]: Station information or None if not found.
    """
    station_lower = station_name.lower().strip()
    return CP_STATIONS.get(station_lower, None)

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
def get_route_between_stations(origin: str, destination: str) -> str:
    """
    Provides routing information between two locations in Lisbon using Metro, Carris, and CP trains.
    
    CRITICAL: This tool checks ALL transport modes (Metro, Carris buses, CP trains) and provides
    accurate line information for metro stations.
    
    Args:
        origin (str): Starting location (e.g., "Entrecampos", "Aeroporto").
        destination (str): Destination location (e.g., "Marquês de Pombal", "Cais do Sodré").
    
    Returns:
        str: Detailed routing suggestions with metro lines, bus options, and train alternatives.
    
    Examples:
        >>> get_route_between_stations("Entrecampos", "Marquês de Pombal")
        >>> get_route_between_stations("Aeroporto", "Rossio")
    """
    origin_lower = origin.lower().strip()
    dest_lower = destination.lower().strip()
    
    response = f"🗺️ **Route: {origin.title()} → {destination.title()}**\n"
    response += "=" * 50 + "\n\n"
    
    # Check if both are Metro stations
    origin_lines = get_station_lines(origin)
    dest_lines = get_station_lines(destination)
    
    # Check if they are CP train stations
    origin_cp = get_cp_station_info(origin)
    dest_cp = get_cp_station_info(destination)
    
    has_metro = bool(origin_lines or dest_lines)
    has_train = bool(origin_cp or dest_cp)
    
    if origin_lines and dest_lines:
        # Both are Metro stations
        response += "🚇 **METRO ROUTE**\n"
        response += "-" * 30 + "\n"
        
        # Check for direct line
        common_lines = set(origin_lines) & set(dest_lines)
        if common_lines:
            response += f"✅ **Direct Route Available**\n\n"
            for line in common_lines:
                line_info = METRO_LINES.get(line, {})
                emoji = line_info.get('emoji', '')
                name = line_info.get('name', line.title())
                response += f"   {emoji} Take the **{line.title()} Line** ({name})\n"
                response += f"   📍 Board at: {origin.title()}\n"
                response += f"   📍 Alight at: {destination.title()}\n\n"
        else:
            # Need to transfer
            response += f"🔄 **Transfer Required**\n\n"
            response += f"   📍 From: {origin.title()} ({', '.join([METRO_LINES[l]['emoji'] + ' ' + l.title() for l in origin_lines])})\n"
            response += f"   📍 To: {destination.title()} ({', '.join([METRO_LINES[l]['emoji'] + ' ' + l.title() for l in dest_lines])})\n\n"
            
            # Suggest transfer stations
            transfer_stations = [
                ("Marquês de Pombal", ["amarela", "azul"]),
                ("Saldanha", ["amarela", "vermelha"]),
                ("Alameda", ["verde", "vermelha"]),
                ("Baixa-Chiado", ["azul", "verde"]),
                ("Jardim Zoológico", ["azul", "verde"]),
                ("São Sebastião", ["vermelha", "azul"]),
            ]
            
            for station, lines in transfer_stations:
                if set(origin_lines) & set(lines) and set(dest_lines) & set(lines):
                    response += f"   💡 **Suggested Transfer**: {station}\n"
                    response += f"      {', '.join([METRO_LINES[l]['emoji'] for l in lines])}\n\n"
    
    elif origin_lines:
        response += f"🚇 **Origin is a Metro station**: {origin.title()}\n"
        response += f"   Lines: {', '.join([METRO_LINES[l]['emoji'] + ' ' + l.title() for l in origin_lines])}\n\n"
        response += f"❌ Destination '{destination.title()}' is not a known Metro station.\n"
        response += "   Consider using Carris buses or CP trains.\n\n"
    
    elif dest_lines:
        response += f"❌ Origin '{origin.title()}' is not a known Metro station.\n\n"
        response += f"🚇 **Destination is a Metro station**: {destination.title()}\n"
        response += f"   Lines: {', '.join([METRO_LINES[l]['emoji'] + ' ' + l.title() for l in dest_lines])}\n\n"
        response += "   Consider using Carris buses or CP trains to reach the Metro.\n\n"
    
    else:
        response += "❌ Neither location is a known Metro station.\n\n"
    
    # Check for CP Train options
    if origin_cp or dest_cp:
        response += "🚆 **CP TRAINS (COMBOIOS)**\n"
        response += "-" * 30 + "\n"
        
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
        
        # Check if both are on the same train line
        if origin_cp and dest_cp:
            origin_train_lines = set(origin_cp.get("lines", []))
            dest_train_lines = set(dest_cp.get("lines", []))
            common_train_lines = origin_train_lines & dest_train_lines
            
            if common_train_lines:
                response += "✅ **Direct train route available!**\n"
                for line in common_train_lines:
                    line_info = CP_LINES.get(line, {})
                    response += f"   {line_info.get('emoji', '🚆')} {line_info.get('name', line.title())}\n"
                    response += f"   ⏱️ Frequency: {line_info.get('frequency', 'Check schedule')}\n"
                response += "\n"
    
    # Add general transport options (only if not covered above)
    if not has_metro and not has_train:
        response += "🚌 **CARRIS BUSES**\n"
    response += "-" * 30 + "\n"
    response += "Check bus routes and real-time arrivals with:\n"
    response += f"   • Search for bus stops near '{origin.title()}'\n"
    response += f"   • Search for bus stops near '{destination.title()}'\n\n"
    
    response += "🚆 **CP TRAINS (Comboios de Portugal)**\n"
    response += "-" * 30 + "\n"
    response += "For longer distances or connections to suburbs:\n"
    response += "   • Check train schedules from nearby stations\n"
    response += "   • Lines: Sintra, Cascais, Azambuja, Sado lines\n\n"
    
    response += "💡 **RECOMMENDATION**\n"
    response += "-" * 30 + "\n"
    response += "For the fastest route, combine:\n"
    response += "   1. Metro (if both locations are near stations)\n"
    response += "   2. Carris buses (for short distances or first/last mile)\n"
    response += "   3. CP trains (for suburban connections)\n\n"
    
    # Add current transport status
    response += "📊 **CURRENT TRANSPORT STATUS**\n"
    response += "-" * 30 + "\n"
    
    # Quick metro status
    metro_data = fetch_json_with_retry(METRO_STATUS_URL)
    if metro_data and metro_data.get('resposta'):
        resp = metro_data['resposta']
        all_ok = all(resp.get(line, 'Unknown').strip().lower() == 'ok' for line in METRO_LINES.keys())
        if all_ok:
            response += "   🚇 Metro: ✅ All lines operating normally\n"
        else:
            response += "   🚇 Metro: ⚠️ Some disruptions reported\n"
    
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
