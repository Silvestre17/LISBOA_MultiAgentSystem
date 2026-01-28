# ==========================================================================
# Master Thesis - Carris Metropolitana API Tools
#   - André Filipe Gomes Silvestre, 20240502
# 
#   Real-time suburban bus data for Lisbon Metropolitan Area.
#   Features:
#     - Bus stop information and real-time arrivals
#     - Line search and route discovery
#     - Geocoding (OpenStreetMap/Nominatim)
#     - Smart location resolution for routing
#     - Real-time vehicle GPS tracking
#     - Bus route planning between any locations
# 
#   API Documentation: https://github.com/carrismetropolitana/api
#   API Base: https://api.carrismetropolitana.pt
# ==========================================================================

# Required libraries:
# pip install requests langchain-core

import os
import sys
import logging
import time
import math
from datetime import datetime, timedelta
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
MAX_RETRIES = 3       # number of retries for API calls
BACKOFF_FACTOR = 2    # exponential backoff factor

# Cache expiration time (24 hours - stop/line data doesn't change frequently)
CACHE_EXPIRATION_HOURS = 24

# ==========================================================================
# API Endpoints
# ==========================================================================

# Carris Metropolitana (Suburban buses - AML region)
CARRIS_BASE_URL = "https://api.carrismetropolitana.pt"
CARRIS_ALERTS_URL = f"{CARRIS_BASE_URL}/alerts"
CARRIS_STOPS_URL = f"{CARRIS_BASE_URL}/stops"
CARRIS_LINES_URL = f"{CARRIS_BASE_URL}/lines"
CARRIS_ROUTES_URL = f"{CARRIS_BASE_URL}/routes"
CARRIS_PATTERNS_URL = f"{CARRIS_BASE_URL}/patterns"
CARRIS_VEHICLES_URL = f"{CARRIS_BASE_URL}/vehicles"

# Nominatim (OpenStreetMap) - Free geocoding service
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# ==========================================================================
# Cache Variables
# ==========================================================================

_carris_metropolitana_stops_cache: Optional[List[Dict[str, Any]]] = None
_carris_metropolitana_stops_last_load: Optional[datetime] = None
_carris_metropolitana_lines_cache: Optional[List[Dict[str, Any]]] = None
_carris_metropolitana_lines_last_load: Optional[datetime] = None
_carris_metropolitana_routes_cache: Optional[List[Dict[str, Any]]] = None
_carris_metropolitana_routes_last_load: Optional[datetime] = None

# ==========================================================================
# Carris Urban vs Metropolitan Limitation Notice
# ==========================================================================

CARRIS_LIMITATION_NOTICE = """
⚠️ **IMPORTANT: Urban Lisbon Bus Limitation**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Carris Metropolitana API** (this search) covers SUBURBAN buses only:
• Municipalities: Sintra, Cascais, Oeiras, Amadora, Loures, Odivelas, etc.
• Great for: Traveling TO/FROM Lisbon suburbs

**Carris (Urban Lisbon)** buses are NOT included:
• Routes like 28E (tram), 738, 732, 15E are managed by Carris
• For trips WITHIN central Lisbon, please check carris.pt

💡 TIP: For central Lisbon destinations, the Metro is usually faster!
"""


# ==========================================================================
# Helper Functions
# ==========================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates the great-circle distance between two points on Earth.
    
    Args:
        lat1, lon1: First point coordinates in degrees.
        lat2, lon2: Second point coordinates in degrees.
        
    Returns:
        float: Distance in kilometers.
    """
    R = 6371  # Earth radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _is_cache_valid(last_load: Optional[datetime]) -> bool:
    """Checks if the cache is still valid (not expired)."""
    if last_load is None:
        return False
    hours_elapsed = (datetime.now() - last_load).total_seconds() / 3600
    return hours_elapsed < CACHE_EXPIRATION_HOURS


def fetch_json_with_retry(url: str, timeout: int = REQUEST_TIMEOUT) -> Optional[Any]:
    """
    Fetches JSON data from a URL with retry logic.
    
    Args:
        url: URL to fetch from.
        timeout: Request timeout in seconds.
        
    Returns:
        JSON data if successful, None otherwise.
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
        ts: Unix timestamp in milliseconds.
        
    Returns:
        Formatted datetime string.
    """
    try:
        dt = datetime.fromtimestamp(ts / 1000)
        return dt.strftime("%H:%M:%S")
    except (ValueError, TypeError, OSError):
        return "N/A"


def is_within_lisbon_city(lat: Optional[float], lon: Optional[float]) -> bool:
    """
    Checks if coordinates are within central Lisbon city limits.
    
    Central Lisbon boundaries (approximate):
    - Latitude: 38.70 to 38.80
    - Longitude: -9.20 to -9.10
    
    Args:
        lat: Latitude.
        lon: Longitude.
        
    Returns:
        True if within central Lisbon.
    """
    if lat is None or lon is None:
        return False
    return 38.70 <= lat <= 38.80 and -9.20 <= lon <= -9.10


def both_locations_in_lisbon_city(
    o_lat: Optional[float], o_lon: Optional[float],
    d_lat: Optional[float], d_lon: Optional[float]
) -> bool:
    """Checks if both origin and destination are within central Lisbon."""
    return is_within_lisbon_city(o_lat, o_lon) and is_within_lisbon_city(d_lat, d_lon)


# ==========================================================================
# Geocoding Functions
# ==========================================================================

def geocode_location(location_name: str) -> Optional[Dict[str, Any]]:
    """
    Geocodes a location name to GPS coordinates using OpenStreetMap/Nominatim.
    
    Tries multiple query variations to find the best match within the
    Lisbon metropolitan area.
    
    Args:
        location_name: Name of the location (e.g., 'Colombo', 'Torre de Belém').
        
    Returns:
        Dict with name, lat, lon, type, address or None if not found.
        
    Example:
        >>> geocode_location("Colombo")
        {'name': 'Centro Comercial Colombo', 'lat': 38.7515, 'lon': -9.1882, ...}
    """
    clean_name = location_name.strip()
    
    search_queries = [
        f"{clean_name}, Lisboa, Portugal",
        f"{clean_name}, Lisbon, Portugal",
        f"{clean_name}, Portugal"
    ]
    
    headers = {
        "User-Agent": "LisbonUrbanAssistant/1.0 (research@novaims.pt)"
    }
    
    for query in search_queries:
        try:
            params = {
                "q": query,
                "format": "json",
                "limit": 5,
                "addressdetails": 1
            }
            
            response = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            results = response.json()
            
            if not results:
                continue
            
            # Filter results to Lisbon metropolitan area
            lisbon_results = []
            for r in results:
                lat = float(r.get("lat", 0))
                lon = float(r.get("lon", 0))
                
                if 38.4 <= lat <= 39.1 and -9.6 <= lon <= -8.7:
                    lisbon_results.append(r)
            
            if lisbon_results:
                best = lisbon_results[0]
                address = best.get("address", {})
                
                result = {
                    "name": best.get("display_name", location_name),
                    "lat": float(best.get("lat")),
                    "lon": float(best.get("lon")),
                    "type": best.get("type", "unknown"),
                    "class": best.get("class", "unknown"),
                    "importance": float(best.get("importance", 0)),
                    "address": {
                        "road": address.get("road", ""),
                        "suburb": address.get("suburb", ""),
                        "city": address.get("city", address.get("town", "")),
                        "municipality": address.get("municipality", ""),
                        "postcode": address.get("postcode", "")
                    },
                    "query_used": query
                }
                
                logger.info(f"Geocoded '{location_name}' → ({result['lat']:.4f}, {result['lon']:.4f})")
                return result
        
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout geocoding '{query}'")
            continue
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error geocoding '{query}': {e}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error geocoding '{query}': {e}")
            continue
    
    logger.warning(f"Could not geocode '{location_name}'")
    return None


# ==========================================================================
# Stop/Line/Route Cache Functions
# ==========================================================================

def load_carris_metropolitana_stops(force_reload: bool = False) -> List[Dict[str, Any]]:
    """
    Loads all Carris Metropolitana bus stops into memory cache.
    
    This function fetches ~5000 bus stops from the Carris Metropolitana API.
    
    Args:
        force_reload: Force refresh even if cache is valid.
        
    Returns:
        List of stop dictionaries with id, name, lat, lon, lines.
    """
    global _carris_metropolitana_stops_cache, _carris_metropolitana_stops_last_load
    
    if not force_reload and _carris_metropolitana_stops_cache and _is_cache_valid(_carris_metropolitana_stops_last_load):
        logger.info(f"Using cached Carris Metropolitana stops ({len(_carris_metropolitana_stops_cache)} stops)")
        return _carris_metropolitana_stops_cache
    
    logger.info("Loading all Carris Metropolitana stops from API...")
    
    try:
        response = requests.get(CARRIS_STOPS_URL, timeout=30)
        response.raise_for_status()
        raw_stops = response.json()
        
        if not isinstance(raw_stops, list):
            logger.error("Unexpected response format from Carris Metropolitana stops API")
            return _carris_metropolitana_stops_cache or []
        
        processed_stops = []
        for stop in raw_stops:
            try:
                lat = float(stop.get("lat")) if stop.get("lat") else None
                lon = float(stop.get("lon")) if stop.get("lon") else None
            except (ValueError, TypeError):
                lat, lon = None, None
            
            processed_stop = {
                "id": stop.get("id", ""),
                "name": stop.get("name", "Unknown"),
                "short_name": stop.get("short_name", ""),
                "lat": lat,
                "lon": lon,
                "municipality": stop.get("municipality_name", ""),
                "locality": stop.get("locality", ""),
                "lines": stop.get("lines", []),
                "facilities": stop.get("facilities", [])
            }
            
            if processed_stop["lat"] and processed_stop["lon"]:
                processed_stops.append(processed_stop)
        
        _carris_metropolitana_stops_cache = processed_stops
        _carris_metropolitana_stops_last_load = datetime.now()
        
        logger.info(f"\033[1;32m✅ Loaded {len(processed_stops)} Carris Metropolitana stops\033[0m")
        return processed_stops
        
    except requests.exceptions.Timeout:
        logger.error("Timeout loading Carris Metropolitana stops (30s)")
        return _carris_metropolitana_stops_cache or []
    except requests.exceptions.RequestException as e:
        logger.error(f"Error loading Carris Metropolitana stops: {e}")
        return _carris_metropolitana_stops_cache or []
    except Exception as e:
        logger.error(f"Unexpected error loading Carris Metropolitana stops: {e}")
        return _carris_metropolitana_stops_cache or []


def load_carris_metropolitana_lines(force_reload: bool = False) -> List[Dict[str, Any]]:
    """
    Loads all Carris Metropolitana bus lines into memory cache.
    
    Args:
        force_reload: Force refresh even if cache is valid.
        
    Returns:
        List of line dictionaries with id, name, color, municipalities.
    """
    global _carris_metropolitana_lines_cache, _carris_metropolitana_lines_last_load
    
    if not force_reload and _carris_metropolitana_lines_cache and _is_cache_valid(_carris_metropolitana_lines_last_load):
        logger.info(f"Using cached Carris Metropolitana lines ({len(_carris_metropolitana_lines_cache)} lines)")
        return _carris_metropolitana_lines_cache
    
    logger.info("Loading all Carris Metropolitana lines from API...")
    
    try:
        response = requests.get(CARRIS_LINES_URL, timeout=30)
        response.raise_for_status()
        raw_lines = response.json()
        
        if not isinstance(raw_lines, list):
            logger.error("Unexpected response format from Carris Metropolitana lines API")
            return _carris_metropolitana_lines_cache or []
        
        processed_lines = []
        for line in raw_lines:
            processed_line = {
                "id": line.get("id", ""),
                "short_name": line.get("short_name", ""),
                "long_name": line.get("long_name", ""),
                "color": line.get("color", "#CCCCCC"),
                "text_color": line.get("text_color", "#FFFFFF"),
                "municipalities": line.get("municipalities", []),
                "localities": line.get("localities", []),
                "routes": line.get("routes", []),
                "patterns": line.get("patterns", [])
            }
            processed_lines.append(processed_line)
        
        _carris_metropolitana_lines_cache = processed_lines
        _carris_metropolitana_lines_last_load = datetime.now()
        
        logger.info(f"\033[1;32m✅ Loaded {len(processed_lines)} Carris Metropolitana lines\033[0m")
        return processed_lines
        
    except requests.exceptions.Timeout:
        logger.error("Timeout loading Carris Metropolitana lines (30s)")
        return _carris_metropolitana_lines_cache or []
    except requests.exceptions.RequestException as e:
        logger.error(f"Error loading Carris Metropolitana lines: {e}")
        return _carris_metropolitana_lines_cache or []
    except Exception as e:
        logger.error(f"Unexpected error loading Carris Metropolitana lines: {e}")
        return _carris_metropolitana_lines_cache or []


def load_carris_metropolitana_routes(force_reload: bool = False) -> List[Dict[str, Any]]:
    """
    Loads all Carris Metropolitana bus routes into memory cache.
    
    Args:
        force_reload: Force refresh even if cache is valid.
        
    Returns:
        List of route dictionaries with id, line_id, name, patterns.
    """
    global _carris_metropolitana_routes_cache, _carris_metropolitana_routes_last_load
    
    if not force_reload and _carris_metropolitana_routes_cache and _is_cache_valid(_carris_metropolitana_routes_last_load):
        logger.info(f"Using cached Carris Metropolitana routes ({len(_carris_metropolitana_routes_cache)} routes)")
        return _carris_metropolitana_routes_cache
    
    logger.info("Loading all Carris Metropolitana routes from API...")
    
    try:
        response = requests.get(CARRIS_ROUTES_URL, timeout=30)
        response.raise_for_status()
        raw_routes = response.json()
        
        if not isinstance(raw_routes, list):
            logger.error("Unexpected response format from Carris Metropolitana routes API")
            return _carris_metropolitana_routes_cache or []
        
        processed_routes = []
        for route in raw_routes:
            processed_route = {
                "id": route.get("id", ""),
                "line_id": route.get("line_id", ""),
                "short_name": route.get("short_name", ""),
                "long_name": route.get("long_name", ""),
                "color": route.get("color", "#CCCCCC"),
                "text_color": route.get("text_color", "#FFFFFF"),
                "patterns": route.get("patterns", []),
                "municipalities": route.get("municipalities", []),
                "localities": route.get("localities", [])
            }
            processed_routes.append(processed_route)
        
        _carris_metropolitana_routes_cache = processed_routes
        _carris_metropolitana_routes_last_load = datetime.now()
        
        logger.info(f"\033[1;32m✅ Loaded {len(processed_routes)} Carris Metropolitana routes\033[0m")
        return processed_routes
        
    except requests.exceptions.Timeout:
        logger.error("Timeout loading Carris Metropolitana routes (30s)")
        return _carris_metropolitana_routes_cache or []
    except requests.exceptions.RequestException as e:
        logger.error(f"Error loading Carris Metropolitana routes: {e}")
        return _carris_metropolitana_routes_cache or []
    except Exception as e:
        logger.error(f"Unexpected error loading Carris Metropolitana routes: {e}")
        return _carris_metropolitana_routes_cache or []


# ==========================================================================
# Stop Search Functions
# ==========================================================================

def find_stops_near_coordinates(
    lat: float, 
    lon: float, 
    radius_km: float = 0.5,
    max_results: int = 10
) -> List[Dict[str, Any]]:
    """
    Finds bus stops within a given radius of GPS coordinates.
    
    Args:
        lat: Latitude of the search center.
        lon: Longitude of the search center.
        radius_km: Search radius in kilometers (default: 0.5km).
        max_results: Maximum number of stops to return.
        
    Returns:
        List of nearby stops, sorted by distance.
    """
    stops = load_carris_metropolitana_stops()
    
    if not stops:
        logger.warning("No Carris Metropolitana stops available for proximity search")
        return []
    
    nearby_stops = []
    for stop in stops:
        stop_lat = stop.get("lat")
        stop_lon = stop.get("lon")
        
        if stop_lat is None or stop_lon is None:
            continue
        
        distance = haversine_distance(lat, lon, stop_lat, stop_lon)
        
        if distance <= radius_km:
            nearby_stops.append({
                "id": stop["id"],
                "name": stop["name"],
                "lat": stop_lat,
                "lon": stop_lon,
                "distance_km": round(distance, 3),
                "municipality": stop.get("municipality", ""),
                "lines": stop.get("lines", [])
            })
    
    nearby_stops.sort(key=lambda x: x["distance_km"])
    return nearby_stops[:max_results]


def find_stops_by_name(
    name_query: str,
    max_results: int = 10
) -> List[Dict[str, Any]]:
    """
    Finds bus stops matching a name query (fuzzy search).
    
    Args:
        name_query: Search query for stop name.
        max_results: Maximum number of results.
        
    Returns:
        List of matching stops.
    """
    stops = load_carris_metropolitana_stops()
    
    if not stops:
        logger.warning("No Carris Metropolitana stops available for name search")
        return []
    
    query_lower = name_query.lower().strip()
    
    matching_stops = []
    for stop in stops:
        stop_name = stop.get("name", "").lower()
        
        if query_lower in stop_name:
            matching_stops.append({
                "id": stop["id"],
                "name": stop["name"],
                "lat": stop.get("lat"),
                "lon": stop.get("lon"),
                "municipality": stop.get("municipality", ""),
                "lines": stop.get("lines", [])
            })
    
    return matching_stops[:max_results]


def find_common_routes(
    origin_stop_ids: List[str],
    dest_stop_ids: List[str]
) -> List[Dict[str, Any]]:
    """
    Finds bus lines that serve both origin and destination stops.
    
    Args:
        origin_stop_ids: List of stop IDs near origin.
        dest_stop_ids: List of stop IDs near destination.
        
    Returns:
        List of route options with line info and stops.
    """
    stops = load_carris_metropolitana_stops()
    lines = load_carris_metropolitana_lines()
    
    if not stops:
        return []
    
    stop_map = {s["id"]: s for s in stops}
    
    # Get lines serving origin stops
    origin_lines = set()
    origin_stop_lines = {}
    for stop_id in origin_stop_ids:
        stop = stop_map.get(stop_id)
        if stop:
            for line in stop.get("lines", []):
                origin_lines.add(line)
                if line not in origin_stop_lines:
                    origin_stop_lines[line] = stop
    
    # Get lines serving destination stops
    dest_lines = set()
    dest_stop_lines = {}
    for stop_id in dest_stop_ids:
        stop = stop_map.get(stop_id)
        if stop:
            for line in stop.get("lines", []):
                dest_lines.add(line)
                if line not in dest_stop_lines:
                    dest_stop_lines[line] = stop
    
    # Find common lines (direct routes)
    common_lines = origin_lines & dest_lines
    
    if not common_lines:
        return []
    
    line_map = {l["id"]: l for l in lines}
    
    route_options = []
    for line_id in common_lines:
        line_info = line_map.get(line_id, {})
        
        route_option = {
            "line_id": line_id,
            "short_name": line_info.get("short_name", line_id),
            "long_name": line_info.get("long_name", ""),
            "color": line_info.get("color", "#CCCCCC"),
            "text_color": line_info.get("text_color", "#FFFFFF"),
            "localities": line_info.get("localities", []),
            "origin_stop": origin_stop_lines.get(line_id, {}),
            "dest_stop": dest_stop_lines.get(line_id, {})
        }
        route_options.append(route_option)
    
    route_options.sort(key=lambda x: x.get("short_name", ""))
    return route_options


def resolve_location(
    location_name: str,
    search_radius_km: float = 0.5,
    max_stops: int = 5
) -> Dict[str, Any]:
    """
    Intelligently resolves a location name to bus stops.
    
    Uses a multi-step approach:
    1. Try direct name match in Carris stops
    2. If few results, use geocoding to get coordinates
    3. Find stops near the geocoded coordinates
    
    Args:
        location_name: Any location name (POI, address, landmark).
        search_radius_km: Radius for GPS search.
        max_stops: Maximum stops to return.
        
    Returns:
        Dict with method, location, stops, and success flag.
    """
    result = {
        "method": None,
        "location": None,
        "stops": [],
        "success": False,
        "query": location_name
    }
    
    clean_name = location_name.strip()
    
    # Step 1: Try direct name match
    name_matches = find_stops_by_name(clean_name, max_results=max_stops)
    
    if len(name_matches) >= 3:
        result["method"] = "name_match"
        result["stops"] = name_matches
        result["success"] = True
        logger.info(f"Resolved '{clean_name}' via name match ({len(name_matches)} stops)")
        return result
    
    # Step 2: Try geocoding
    geocoded = geocode_location(clean_name)
    
    if geocoded:
        result["location"] = geocoded
        
        nearby_stops = find_stops_near_coordinates(
            geocoded["lat"],
            geocoded["lon"],
            radius_km=search_radius_km,
            max_results=max_stops
        )
        
        if nearby_stops:
            result["method"] = "geocoding"
            result["stops"] = nearby_stops
            result["success"] = True
            logger.info(f"Resolved '{clean_name}' via geocoding ({len(nearby_stops)} stops)")
            return result
        
        # Try larger radius
        if not nearby_stops and search_radius_km < 1.0:
            nearby_stops = find_stops_near_coordinates(
                geocoded["lat"],
                geocoded["lon"],
                radius_km=1.0,
                max_results=max_stops
            )
            
            if nearby_stops:
                result["method"] = "geocoding_expanded"
                result["stops"] = nearby_stops
                result["success"] = True
                logger.info(f"Resolved '{clean_name}' via expanded geocoding ({len(nearby_stops)} stops)")
                return result
    
    # Step 3: Fallback to name matches
    if name_matches:
        result["method"] = "name_match_fallback"
        result["stops"] = name_matches
        result["success"] = True
        logger.info(f"Resolved '{clean_name}' via fallback name match ({len(name_matches)} stops)")
        return result
    
    logger.warning(f"Could not resolve location '{clean_name}'")
    return result


# ==========================================================================
# LangChain Tools
# ==========================================================================

@tool
def get_carris_metropolitana_alerts() -> str:
    """
    Gets current service alerts from Carris Metropolitana (suburban buses).
    
    Returns:
        str: Formatted list of active service alerts.
    """
    data = fetch_json_with_retry(CARRIS_ALERTS_URL)
    
    if not data:
        return "❌ Failed to fetch Carris Metropolitana alerts."
    
    # API returns a list directly, not a dict with 'entity' key
    alerts = data if isinstance(data, list) else data.get('entity', [])
    
    if not alerts:
        return "✅ No active alerts from Carris Metropolitana."
    
    response = "⚠️ **Carris Metropolitana Service Alerts**\n"
    response += "=" * 45 + "\n\n"
    
    for i, alert in enumerate(alerts[:10], 1):
        # Handle both old format (nested under 'alert') and new format (flat)
        alert_data = alert.get('alert', alert)
        
        # Header
        header_text = alert_data.get('header_text', {})
        header = header_text.get('translation', [{}])[0].get('text', 'No title')
        
        # Description
        desc_text = alert_data.get('description_text', {})
        desc = desc_text.get('translation', [{}])[0].get('text', 'No details')
        
        # Time period
        active_period = alert_data.get('active_period', [{}])[0]
        start = format_timestamp(active_period.get('start', 0))
        end = format_timestamp(active_period.get('end', 0))
        
        # Cause and effect
        cause = alert_data.get('cause', 'UNKNOWN').replace('_', ' ').title()
        effect = alert_data.get('effect', 'UNKNOWN').replace('_', ' ').title()
        
        # Affected routes
        informed_entity = alert_data.get('informed_entity', [])
        routes = [e.get('route_id', '') for e in informed_entity if e.get('route_id')]
        routes_str = ', '.join(routes[:5]) if routes else 'All routes'
        
        response += f"**{i}. {header}**\n"
        response += f"   📝 {desc[:200]}{'...' if len(desc) > 200 else ''}\n"
        response += f"   🚌 Routes: {routes_str}\n"
        response += f"   ⚠️ Cause: {cause} | Effect: {effect}\n"
        response += f"   ⏰ Period: {start} - {end}\n\n"
    
    if len(alerts) > 10:
        response += f"... and {len(alerts) - 10} more alerts.\n"
    
    return response


@tool
def get_carris_metropolitana_stop_info(stop_id: str) -> str:
    """
    Gets detailed information about a Carris Metropolitana bus stop.
    
    Includes real-time arrival information when available.
    
    Args:
        stop_id: The stop ID (e.g., '060001', '090101').

    Returns:
        str: Stop details including lines served and upcoming arrivals.
    """
    stop_url = f"{CARRIS_STOPS_URL}/{stop_id}"
    stop_data = fetch_json_with_retry(stop_url)
    
    if not stop_data:
        return f"❌ Could not find stop with ID: {stop_id}"
    
    realtime_url = f"{CARRIS_STOPS_URL}/{stop_id}/realtime"
    realtime_data = fetch_json_with_retry(realtime_url)
    
    response = "🚏 Bus Stop Information\n"
    response += "=" * 40 + "\n\n"
    
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
    
    if realtime_data and isinstance(realtime_data, list) and realtime_data:
        response += "⏱️ Upcoming Arrivals:\n"
        
        for i, arrival in enumerate(realtime_data[:8], 1):
            line_id = arrival.get('line_id', 'N/A')
            headsign = arrival.get('headsign', 'N/A')
            estimated = arrival.get('estimated_arrival', '')
            scheduled = arrival.get('scheduled_arrival', '')
            
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
def search_carris_metropolitana_lines(query: str) -> str:
    """
    Searches for Carris Metropolitana (suburban) bus lines.
    
    IMPORTANT: This searches SUBURBAN bus lines only. Urban Lisbon buses
    like lines 28E, 738, 732 are NOT included.
    
    Args:
        query: Line number, destination name, or area to search.
               Examples: '1718', 'Sintra', 'Cascais', 'Almada', 'Oriente'

    Returns:
        str: Matching lines with route details.
    """
    data = fetch_json_with_retry(CARRIS_LINES_URL)
    
    if not data:
        return "❌ Failed to fetch Carris Metropolitana lines data."
    
    if not isinstance(data, list):
        return "❌ Unexpected response format."
    
    query_lower = query.lower()
    query_normalized = query_lower.replace('é', 'e').replace('ã', 'a').replace('õ', 'o').replace('ç', 'c')
    
    matches = []
    
    for line in data:
        short_name = line.get('short_name', '')
        long_name = line.get('long_name', '')
        line_id = line.get('id', '')
        municipalities = line.get('municipalities', [])
        
        long_name_norm = long_name.lower().replace('é', 'e').replace('ã', 'a').replace('õ', 'o').replace('ç', 'c')
        muni_str = ' '.join(municipalities).lower()
        
        if (query_lower in short_name.lower() or 
            query_normalized in long_name_norm or
            query_lower in line_id.lower() or
            query_lower in muni_str):
            matches.append(line)
    
    if not matches:
        urban_keywords = ['rossio', 'baixa', 'chiado', 'alfama', 'bairro alto', '28e', '738', '732', '15e']
        if any(kw in query_lower for kw in urban_keywords):
            return (f"❌ No Carris Metropolitana lines found for '{query}'\n\n"
                    + CARRIS_LIMITATION_NOTICE)
        return f"❌ No lines found matching: '{query}'\n\n💡 Try searching by area: Sintra, Cascais, Almada, Oeiras, Loures"
    
    response = f"🚌 **Carris Metropolitana lines matching '{query}'** ({len(matches)} found)\n"
    response += "=" * 50 + "\n\n"
    
    for i, line in enumerate(matches[:10], 1):
        short_name = line.get('short_name', 'N/A')
        long_name = line.get('long_name', 'N/A')
        municipalities = line.get('municipalities', [])
        
        response += f"{i}. **Line {short_name}**\n"
        response += f"   📍 {long_name}\n"
        if municipalities:
            response += f"   🏘️ {', '.join(municipalities[:3])}\n"
        response += "\n"
    
    if len(matches) > 10:
        response += f"... and {len(matches) - 10} more lines.\n"
    
    response += "\n" + "-" * 40 + "\n"
    response += "💡 Use `get_bus_schedule` with line number for full route details.\n"
    
    return response


@tool
def get_bus_realtime_locations(line_id: Optional[str] = None) -> str:
    """
    Gets real-time GPS locations of Carris Metropolitana buses.
    
    IMPORTANT: Only works for suburban buses, not urban Lisbon.
    
    Args:
        line_id: Filter by line ID (e.g., '1718', '3703').
                 If None, shows overview of all active buses.
    
    Returns:
        str: Real-time locations with speed, status, and next stop.
    """
    data = fetch_json_with_retry(CARRIS_VEHICLES_URL)
    
    if not data:
        return "❌ Failed to fetch real-time bus locations."
    
    if not isinstance(data, list):
        return "❌ Unexpected response format from vehicles API."
    
    if not data:
        return "ℹ️ No active buses reported at this time."
    
    if line_id:
        filtered = [v for v in data if v.get('line_id') == line_id]
        if not filtered:
            return f"ℹ️ No active buses found on line {line_id} at this time.\n\n" \
                   f"💡 The line may not be operating right now."
        buses = filtered
        response = f"🚌 **Real-Time Bus Locations - Line {line_id}**\n"
    else:
        buses = data
        response = f"🚌 **Real-Time Bus Locations Overview**\n"
    
    response += "=" * 50 + "\n"
    response += f"📊 Active buses: {len(buses)}\n"
    response += f"🕐 Updated: {datetime.now().strftime('%H:%M:%S')}\n"
    response += "=" * 50 + "\n\n"
    
    if line_id:
        for i, bus in enumerate(buses[:15], 1):
            lat = bus.get('lat', 0)
            lon = bus.get('lon', 0)
            speed = bus.get('speed', 0)
            bearing = bus.get('bearing', 0)
            status = bus.get('current_status', 'UNKNOWN')
            stop_id = bus.get('stop_id', 'N/A')
            
            status_emoji = {
                'IN_TRANSIT_TO': '🚌➡️',
                'STOPPED_AT': '🚏',
                'INCOMING_AT': '📍'
            }.get(status, '🚌')
            
            directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
            dir_idx = int((bearing + 22.5) % 360 / 45)
            direction = directions[dir_idx] if bearing else '?'
            
            response += f"**{i}. Bus {bus.get('id', 'N/A')[:20]}**\n"
            response += f"   {status_emoji} Status: {status.replace('_', ' ').title()}\n"
            response += f"   📍 Position: ({lat:.5f}, {lon:.5f})\n"
            response += f"   🧭 Direction: {direction} | Speed: {speed:.1f} km/h\n"
            response += f"   🚏 Next stop ID: {stop_id}\n\n"
        
        if len(buses) > 15:
            response += f"... and {len(buses) - 15} more buses on this line.\n"
    else:
        from collections import Counter
        line_counts = Counter(v.get('line_id', 'Unknown') for v in buses)
        
        response += "**Top 15 lines by active buses:**\n\n"
        for line, count in line_counts.most_common(15):
            response += f"   🚌 Line **{line}**: {count} buses\n"
        
        response += f"\n... {len(line_counts)} lines total with active buses.\n"
    
    response += "\n" + "-" * 40 + "\n"
    response += "💡 Use `get_bus_realtime_locations('LINE_ID')` for detailed tracking.\n"
    
    return response


@tool
def get_bus_next_departures(line_id: str, stop_id: str = "", start_time: str = "") -> str:
    """
    Gets next scheduled departures for a Carris Metropolitana bus line.
    
    Args:
        line_id: The line ID (e.g., '1718', '3703').
        stop_id: Specific stop ID to filter.
        start_time: Time (HH:MM) to see schedule for a specific time.
    
    Returns:
        str: Upcoming departures information.
    """
    lines_data = fetch_json_with_retry(CARRIS_LINES_URL)
    
    if not lines_data:
        return "❌ Failed to fetch line information."
    
    line_info = None
    for line in lines_data:
        if line.get('id') == line_id or line.get('short_name') == line_id:
            line_info = line
            break
    
    if not line_info:
        return f"❌ Line '{line_id}' not found.\n\n" \
               f"💡 Use `search_carris_metropolitana_lines` to find the correct line ID."
    
    patterns = line_info.get('patterns', [])
    if not patterns:
        return f"❌ No route patterns found for line {line_id}."
    
    response = f"🚌 **Line {line_info.get('short_name', line_id)} Schedule**\n"
    response += "=" * 50 + "\n"
    response += f"📍 {line_info.get('long_name', 'N/A')}\n"
    response += "=" * 50 + "\n\n"
    
    pattern_id = patterns[0]
    pattern_url = f"{CARRIS_PATTERNS_URL}/{pattern_id}"
    pattern_data = fetch_json_with_retry(pattern_url)
    
    if not pattern_data:
        return f"❌ Failed to fetch schedule for pattern {pattern_id}."
    
    headsign = pattern_data.get('headsign', 'N/A')
    trips = pattern_data.get('trips', [])
    
    response += f"**Direction**: {headsign}\n\n"
    
    if start_time:
        try:
            datetime.strptime(start_time, "%H:%M")
            ref_time = f"{start_time}:00"
            ref_time_display = start_time
        except ValueError:
            return "Invalid time format. Use HH:MM."
    else:
        now_dt = datetime.now()
        ref_time = now_dt.strftime('%H:%M:%S')
        ref_time_display = "NOW"

    today = datetime.now().strftime('%Y%m%d')
    today_trips = [t for t in trips if today in t.get('dates', [])]
    
    if today_trips:
        response += f"**🕐 Departures after {ref_time_display}**:\n"
        response += "-" * 30 + "\n"
        
        departures = []
        for trip in today_trips:
            schedule = trip.get('schedule', [])
            if schedule:
                first_time = schedule[0].get('arrival_time', 'N/A')
                departures.append(first_time)
        
        departures.sort()
        
        upcoming = [d for d in departures if d > ref_time]
        
        if upcoming:
            response += f"   {', '.join(upcoming[:8])}\n"
            if len(upcoming) > 8:
                response += f"   ... and {len(upcoming) - 8} more.\n"
        else:
            response += "ℹ️ No more departures found for today.\n"
        
        if stop_id:
            path = pattern_data.get('path', [])
            stop_idx = next((i for i, s in enumerate(path) if s.get('stop', {}).get('id') == stop_id), None)
            
            if stop_idx is not None:
                stop_name = path[stop_idx].get('stop', {}).get('name', stop_id)
                response += f"\n**⏱️ At stop {stop_name}:**\n"
                
                stop_times = []
                for trip in today_trips:
                    schedule = trip.get('schedule', [])
                    if len(schedule) > stop_idx:
                        time_at = schedule[stop_idx].get('arrival_time', 'N/A')
                        stop_times.append(time_at)
                
                stop_times.sort()
                upcoming_at = [t for t in stop_times if t > ref_time]
                
                if upcoming_at:
                    response += f"   Next: {', '.join(upcoming_at[:6])}\n"
                else:
                    response += "   No more stops today.\n"
            else:
                response += f"\n❌ Stop {stop_id} not found on this line's path.\n"
    else:
        response += f"ℹ️ Line not operating today ({today}).\n"
    
    return response


@tool
def find_bus_routes(
    origin: str,
    destination: str,
    origin_lat: Optional[float] = None,
    origin_lon: Optional[float] = None,
    dest_lat: Optional[float] = None,
    dest_lon: Optional[float] = None,
    search_radius_km: float = 0.5
) -> str:
    """
    Finds Carris bus routes between two locations in Lisbon Metropolitan Area.
    
    Uses SMART LOCATION RESOLUTION to understand any place name:
    - "Colombo" → finds Centro Comercial Colombo → finds nearby stops
    - "Torre de Belém" → finds the monument → finds nearby stops
    
    Args:
        origin: Starting location (any place name, landmark, or address).
        destination: Ending location.
        origin_lat: Override origin with GPS coordinates.
        origin_lon: Override origin with GPS coordinates.
        dest_lat: Override destination with GPS coordinates.
        dest_lon: Override destination with GPS coordinates.
        search_radius_km: Search radius in km (default: 0.5km).
        
    Returns:
        str: Bus route options including direct lines connecting locations.
    """
    response = "🚌 **BUS ROUTE FINDER**\n"
    response += "=" * 50 + "\n"
    response += f"📍 From: {origin}\n"
    response += f"📍 To: {destination}\n"
    response += "=" * 50 + "\n\n"
    
    # Resolve origin
    response += "🔍 **Resolving origin location...**\n"
    
    if origin_lat is not None and origin_lon is not None:
        origin_stops = find_stops_near_coordinates(
            origin_lat, origin_lon, 
            radius_km=search_radius_km, 
            max_results=10
        )
        if origin_stops:
            response += f"   ✅ Using provided coordinates ({origin_lat:.4f}, {origin_lon:.4f})\n"
            response += f"   📍 Found {len(origin_stops)} stops within {search_radius_km}km\n"
            for stop in origin_stops[:3]:
                response += f"      • {stop['name']} ({stop['distance_km']*1000:.0f}m)\n"
            origin_resolved = {"stops": origin_stops, "method": "gps_provided"}
        else:
            response += f"   ⚠️ No stops within {search_radius_km}km of coordinates\n"
            origin_resolved = {"stops": [], "method": "gps_provided", "success": False}
    else:
        origin_resolved = resolve_location(origin, search_radius_km, max_stops=10)
        
        if origin_resolved["success"]:
            method = origin_resolved["method"]
            stops = origin_resolved["stops"]
            
            if method == "geocoding" or method == "geocoding_expanded":
                loc = origin_resolved.get("location", {})
                response += f"   🌍 Geocoded '{origin}' → {loc.get('name', 'Unknown')[:60]}\n"
                response += f"   📍 Coordinates: ({loc.get('lat', 0):.4f}, {loc.get('lon', 0):.4f})\n"
                response += f"   ✅ Found {len(stops)} bus stops nearby\n"
            else:
                response += f"   ✅ Found {len(stops)} stops matching '{origin}'\n"
            
            for stop in stops[:3]:
                dist = stop.get('distance_km')
                if dist:
                    response += f"      • {stop['name']} ({dist*1000:.0f}m)\n"
                else:
                    response += f"      • {stop['name']}\n"
        else:
            response += f"   ❌ Could not resolve '{origin}'\n"
    
    origin_stops = origin_resolved.get("stops", [])
    
    if not origin_stops:
        origin_loc = origin_resolved.get("location")
        if origin_loc and is_within_lisbon_city(origin_loc.get("lat"), origin_loc.get("lon")):
            response += f"\n📍 **'{origin}' is in central Lisbon**\n"
            response += "   🚋 Try using **Carris Urbana** (carris.pt) for urban routes.\n\n"
        else:
            response += f"\n❌ **No bus stops found near '{origin}'**\n"
            response += "   💡 Try adding 'Lisboa' to your search.\n"
        return response
    
    response += "\n"
    
    # Resolve destination
    response += "🔍 **Resolving destination location...**\n"
    
    if dest_lat is not None and dest_lon is not None:
        dest_stops = find_stops_near_coordinates(
            dest_lat, dest_lon, 
            radius_km=search_radius_km, 
            max_results=10
        )
        if dest_stops:
            response += f"   ✅ Using provided coordinates ({dest_lat:.4f}, {dest_lon:.4f})\n"
            response += f"   📍 Found {len(dest_stops)} stops within {search_radius_km}km\n"
            for stop in dest_stops[:3]:
                response += f"      • {stop['name']} ({stop['distance_km']*1000:.0f}m)\n"
            dest_resolved = {"stops": dest_stops, "method": "gps_provided"}
        else:
            response += f"   ⚠️ No stops within {search_radius_km}km of coordinates\n"
            dest_resolved = {"stops": [], "method": "gps_provided", "success": False}
    else:
        dest_resolved = resolve_location(destination, search_radius_km, max_stops=10)
        
        if dest_resolved["success"]:
            method = dest_resolved["method"]
            stops = dest_resolved["stops"]
            
            if method == "geocoding" or method == "geocoding_expanded":
                loc = dest_resolved.get("location", {})
                response += f"   🌍 Geocoded '{destination}' → {loc.get('name', 'Unknown')[:60]}\n"
                response += f"   📍 Coordinates: ({loc.get('lat', 0):.4f}, {loc.get('lon', 0):.4f})\n"
                response += f"   ✅ Found {len(stops)} bus stops nearby\n"
            else:
                response += f"   ✅ Found {len(stops)} stops matching '{destination}'\n"
            
            for stop in stops[:3]:
                dist = stop.get('distance_km')
                if dist:
                    response += f"      • {stop['name']} ({dist*1000:.0f}m)\n"
                else:
                    response += f"      • {stop['name']}\n"
        else:
            response += f"   ❌ Could not resolve '{destination}'\n"
    
    dest_stops = dest_resolved.get("stops", [])
    
    if not dest_stops:
        dest_loc = dest_resolved.get("location")
        if dest_loc and is_within_lisbon_city(dest_loc.get("lat"), dest_loc.get("lon")):
            response += f"\n📍 **'{destination}' is in central Lisbon**\n"
            response += "   🚋 Try using **Carris Urbana** (carris.pt) for urban routes.\n\n"
        else:
            response += f"\n❌ **No bus stops found near '{destination}'**\n"
            response += "   💡 Try using a more specific name or address.\n"
        return response
    
    response += "\n"
    
    # Find common routes
    response += "🔍 **Finding direct bus routes...**\n\n"
    
    origin_stop_ids = [s["id"] for s in origin_stops]
    dest_stop_ids = [s["id"] for s in dest_stops]
    
    route_options = find_common_routes(origin_stop_ids, dest_stop_ids)
    
    if route_options:
        grouped_routes = {}
        for route in route_options:
            origin_stop = route.get("origin_stop", {})
            dest_stop = route.get("dest_stop", {})
            key = (origin_stop.get("name", "?"), dest_stop.get("name", "?"))
            
            if key not in grouped_routes:
                grouped_routes[key] = {
                    "origin_stop": origin_stop,
                    "dest_stop": dest_stop,
                    "lines": []
                }
            grouped_routes[key]["lines"].append(route.get("short_name", "?"))
        
        response += f"✅ **{len(grouped_routes)} ROUTE OPTION(S) FOUND** ({len(route_options)} lines total)\n"
        response += "-" * 40 + "\n\n"
        
        for i, ((origin_name, dest_name), group) in enumerate(list(grouped_routes.items())[:5], 1):
            lines_str = ", ".join(sorted(group["lines"]))
            
            response += f"🚌 **Option {i}**\n"
            response += f"   🚏 Board at: {origin_name}\n"
            response += f"   🚏 Alight at: {dest_name}\n"
            response += f"   🚍 Lines: **{lines_str}**\n"
            response += "\n"
        
        if len(grouped_routes) > 5:
            response += f"   ... and {len(grouped_routes) - 5} more route options available.\n\n"
        
        response += "⚠️ **Note:** Verify that buses run in your intended direction. Check schedules at carrismetropolitana.pt\n\n"
    else:
        response += "❌ **No direct bus routes found**\n\n"
        response += "💡 **Suggestions:**\n"
        response += "   • You may need to transfer buses\n"
        response += "   • Consider using Metro + Bus combination\n"
        response += "   • Check routes from a nearby major stop\n\n"
        
        response += "📊 **Lines available near your locations:**\n\n"
        
        origin_all_lines = set()
        for stop in origin_stops:
            origin_all_lines.update(stop.get("lines", []))
        
        dest_all_lines = set()
        for stop in dest_stops:
            dest_all_lines.update(stop.get("lines", []))
        
        if origin_all_lines:
            response += f"   At {origin}: {', '.join(sorted(list(origin_all_lines))[:10])}"
            if len(origin_all_lines) > 10:
                response += f" (+{len(origin_all_lines)-10} more)"
            response += "\n"
        
        if dest_all_lines:
            response += f"   At {destination}: {', '.join(sorted(list(dest_all_lines))[:10])}"
            if len(dest_all_lines) > 10:
                response += f" (+{len(dest_all_lines)-10} more)"
            response += "\n"
    
    # Check if in Lisbon city
    origin_loc = origin_resolved.get("location")
    dest_loc = dest_resolved.get("location")
    
    if origin_loc:
        o_lat, o_lon = origin_loc.get("lat"), origin_loc.get("lon")
    elif origin_lat and origin_lon:
        o_lat, o_lon = origin_lat, origin_lon
    elif origin_stops:
        o_lat, o_lon = origin_stops[0].get("lat"), origin_stops[0].get("lon")
    else:
        o_lat, o_lon = None, None
    
    if dest_loc:
        d_lat, d_lon = dest_loc.get("lat"), dest_loc.get("lon")
    elif dest_lat and dest_lon:
        d_lat, d_lon = dest_lat, dest_lon
    elif dest_stops:
        d_lat, d_lon = dest_stops[0].get("lat"), dest_stops[0].get("lon")
    else:
        d_lat, d_lon = None, None
    
    if not route_options and both_locations_in_lisbon_city(o_lat, o_lon, d_lat, d_lon):
        response += "\n" + "-" * 50 + "\n"
        response += CARRIS_LIMITATION_NOTICE
        response += "\n"
    
    response += "\n" + "-" * 40 + "\n"
    response += "💡 **Tips:**\n"
    response += "   • Check carrismetropolitana.pt or carris.pt for detailed schedules\n"
    response += "   • Metro may be faster for longer trips\n"
    
    return response


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 CARRIS METROPOLITANA API - TEST SUITE\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")
    
    print("\n1. Testing get_carris_metropolitana_alerts...")
    result = get_carris_metropolitana_alerts.invoke({})
    print(result[:500])
    
    print("\n2. Testing search_carris_metropolitana_lines...")
    result = search_carris_metropolitana_lines.invoke({"query": "Sintra"})
    print(result[:500])
    
    print("\n3. Testing geocode_location...")
    loc = geocode_location("Colombo")
    if loc:
        print(f"   ✅ Colombo: ({loc['lat']:.4f}, {loc['lon']:.4f})")
    
    print("\n4. Testing find_bus_routes...")
    result = find_bus_routes.invoke({"origin": "Colombo", "destination": "Oriente"})
    print(result[:800])
    
    print("\n\033[1;32m✅ Carris Metropolitana API tests complete!\033[0m")
