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

import logging
import math
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
from langchain_core.tools import tool

try:
    from config import Config
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from config import Config

logger = logging.getLogger(__name__)

try:
    from tools.utils import fetch_json_with_retry, haversine_distance
except ImportError:
    from utils import fetch_json_with_retry, haversine_distance

# Request configuration
REQUEST_TIMEOUT = 15  # seconds
MAX_RETRIES = 3  # number of retries for API calls
BACKOFF_FACTOR = 2  # exponential backoff factor

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

# Cache for real-time vehicle data (30 seconds TTL)
_vehicle_cache: Dict[str, Any] = {"data": None, "timestamp": 0, "ttl": 30}

# Nominatim (OpenStreetMap) - Free geocoding service
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# ==========================================================================
# Cache Variables

_carris_metropolitana_stops_cache: Optional[List[Dict[str, Any]]] = None
_carris_metropolitana_stops_last_load: Optional[datetime] = None
_carris_metropolitana_lines_cache: Optional[List[Dict[str, Any]]] = None
_carris_metropolitana_lines_last_load: Optional[datetime] = None
_carris_metropolitana_routes_cache: Optional[List[Dict[str, Any]]] = None
_carris_metropolitana_routes_last_load: Optional[datetime] = None

# Real-time vehicle positions cache
_vehicle_positions_cache: Optional[List[Dict[str, Any]]] = None
_vehicle_positions_last_load: Optional[datetime] = None
_VEHICLE_CACHE_TTL_SECONDS = 30  # Vehicle positions update every 30 seconds

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


def _is_cache_valid(last_load: Optional[datetime]) -> bool:
    """Checks if the cache is still valid (not expired)."""
    if last_load is None:
        return False
    hours_elapsed = (datetime.now() - last_load).total_seconds() / 3600
    return hours_elapsed < CACHE_EXPIRATION_HOURS


def _is_vehicle_cache_valid(last_load: Optional[datetime]) -> bool:
    """Checks if the vehicle positions cache is still valid (30 second TTL)."""
    if last_load is None:
        return False
    seconds_elapsed = (datetime.now() - last_load).total_seconds()
    return seconds_elapsed < _VEHICLE_CACHE_TTL_SECONDS


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
    o_lat: Optional[float],
    o_lon: Optional[float],
    d_lat: Optional[float],
    d_lon: Optional[float],
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
        f"{clean_name}, Portugal",
    ]

    headers = {"User-Agent": "LisbonUrbanAssistant/1.0 (research@novaims.pt)"}

    for query in search_queries:
        try:
            params = {"q": query, "format": "json", "limit": 5, "addressdetails": 1}

            response = requests.get(
                NOMINATIM_URL, params=params, headers=headers, timeout=10
            )
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
                        "postcode": address.get("postcode", ""),
                    },
                    "query_used": query,
                }

                logger.info(
                    f"Geocoded '{location_name}' → ({result['lat']:.4f}, {result['lon']:.4f})"
                )
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

    if (
        not force_reload
        and _carris_metropolitana_stops_cache
        and _is_cache_valid(_carris_metropolitana_stops_last_load)
    ):
        logger.info(
            f"Using cached Carris Metropolitana stops ({len(_carris_metropolitana_stops_cache)} stops)"
        )
        return _carris_metropolitana_stops_cache

    logger.info("Loading all Carris Metropolitana stops from API...")

    try:
        response = requests.get(CARRIS_STOPS_URL, timeout=30)
        response.raise_for_status()
        raw_stops = response.json()

        if not isinstance(raw_stops, list):
            logger.error(
                "Unexpected response format from Carris Metropolitana stops API"
            )
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
                "facilities": stop.get("facilities", []),
            }

            if processed_stop["lat"] and processed_stop["lon"]:
                processed_stops.append(processed_stop)

        _carris_metropolitana_stops_cache = processed_stops
        _carris_metropolitana_stops_last_load = datetime.now()

        logger.info(
            f"\033[1;32m✅ Loaded {len(processed_stops)} Carris Metropolitana stops\033[0m"
        )
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

    if (
        not force_reload
        and _carris_metropolitana_lines_cache
        and _is_cache_valid(_carris_metropolitana_lines_last_load)
    ):
        logger.info(
            f"Using cached Carris Metropolitana lines ({len(_carris_metropolitana_lines_cache)} lines)"
        )
        return _carris_metropolitana_lines_cache

    logger.info("Loading all Carris Metropolitana lines from API...")

    try:
        response = requests.get(CARRIS_LINES_URL, timeout=30)
        response.raise_for_status()
        raw_lines = response.json()

        if not isinstance(raw_lines, list):
            logger.error(
                "Unexpected response format from Carris Metropolitana lines API"
            )
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
                "patterns": line.get("patterns", []),
            }
            processed_lines.append(processed_line)

        _carris_metropolitana_lines_cache = processed_lines
        _carris_metropolitana_lines_last_load = datetime.now()

        logger.info(
            f"\033[1;32m✅ Loaded {len(processed_lines)} Carris Metropolitana lines\033[0m"
        )
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


def load_carris_metropolitana_routes(
    force_reload: bool = False,
) -> List[Dict[str, Any]]:
    """
    Loads all Carris Metropolitana bus routes into memory cache.

    Args:
        force_reload: Force refresh even if cache is valid.

    Returns:
        List of route dictionaries with id, line_id, name, patterns.
    """
    global _carris_metropolitana_routes_cache, _carris_metropolitana_routes_last_load

    if (
        not force_reload
        and _carris_metropolitana_routes_cache
        and _is_cache_valid(_carris_metropolitana_routes_last_load)
    ):
        logger.info(
            f"Using cached Carris Metropolitana routes ({len(_carris_metropolitana_routes_cache)} routes)"
        )
        return _carris_metropolitana_routes_cache

    logger.info("Loading all Carris Metropolitana routes from API...")

    try:
        response = requests.get(CARRIS_ROUTES_URL, timeout=30)
        response.raise_for_status()
        raw_routes = response.json()

        if not isinstance(raw_routes, list):
            logger.error(
                "Unexpected response format from Carris Metropolitana routes API"
            )
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
                "localities": route.get("localities", []),
            }
            processed_routes.append(processed_route)

        _carris_metropolitana_routes_cache = processed_routes
        _carris_metropolitana_routes_last_load = datetime.now()

        logger.info(
            f"\033[1;32m✅ Loaded {len(processed_routes)} Carris Metropolitana routes\033[0m"
        )
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
# Real-Time Vehicle Functions
# ==========================================================================


def load_carris_metropolitana_vehicles(
    force_reload: bool = False,
) -> List[Dict[str, Any]]:
    """
    Loads real-time vehicle positions from Carris Metropolitana API.

    Each vehicle includes:
        - id: Vehicle identifier
        - lat, lon: GPS coordinates
        - bearing: Direction in degrees
        - speed: Speed in km/h (when available)
        - timestamp: Last update timestamp (Unix ms)
        - line_id: Current line being served
        - route_id: Current route
        - trip_id: Current trip
        - pattern_id: Current pattern
        - stop_id: Next/current stop
        - current_status: INCOMING_AT, STOPPED_AT, IN_TRANSIT_TO
        - schedule_relationship: SCHEDULED, etc.
        - shift_id, block_id: Operational identifiers
        - door_status: OPEN/CLOSED
        - vehicle metadata: make, model, license_plate, capacity

    Args:
        force_reload: Force refresh even if cache is valid.

    Returns:
        List of vehicle dictionaries with real-time position data.
    """
    global _vehicle_positions_cache, _vehicle_positions_last_load

    if (
        not force_reload
        and _vehicle_positions_cache
        and _is_vehicle_cache_valid(_vehicle_positions_last_load)
    ):
        age = (
            (datetime.now() - _vehicle_positions_last_load).total_seconds()
            if _vehicle_positions_last_load
            else 0
        )
        logger.info(
            f"Using cached vehicle positions ({len(_vehicle_positions_cache)} vehicles, age: {age:.1f}s)"
        )
        return _vehicle_positions_cache

    logger.info("Fetching real-time vehicle positions from Carris Metropolitana API...")

    try:
        response = requests.get(CARRIS_VEHICLES_URL, timeout=15)
        response.raise_for_status()
        vehicles = response.json()

        if not isinstance(vehicles, list):
            logger.error(
                "Unexpected response format from Carris Metropolitana vehicles API"
            )
            return _vehicle_positions_cache or []

        processed_vehicles = []
        for vehicle in vehicles:
            # Process only if vehicle has position data
            lat = vehicle.get("lat")
            lon = vehicle.get("lon")

            if lat is None or lon is None:
                continue

            processed_vehicle = {
                "id": vehicle.get("id", ""),
                "lat": float(lat),
                "lon": float(lon),
                "bearing": vehicle.get("bearing", 0),
                "speed": vehicle.get("speed"),  # May be None
                "timestamp": vehicle.get("timestamp", 0),
                "line_id": vehicle.get("line_id", ""),
                "route_id": vehicle.get("route_id", ""),
                "trip_id": vehicle.get("trip_id", ""),
                "pattern_id": vehicle.get("pattern_id", ""),
                "stop_id": vehicle.get("stop_id", ""),
                "current_status": vehicle.get("current_status", "UNKNOWN"),
                "schedule_relationship": vehicle.get("schedule_relationship", ""),
                "shift_id": vehicle.get("shift_id", ""),
                "block_id": vehicle.get("block_id", ""),
                "door_status": vehicle.get("door_status", ""),
                # Vehicle metadata
                "vehicle_make": vehicle.get("make", ""),
                "vehicle_model": vehicle.get("model", ""),
                "license_plate": vehicle.get("license_plate", ""),
                "capacity_total": vehicle.get("capacity_total", 0),
                "capacity_seated": vehicle.get("capacity_seated", 0),
                "wheelchair_accessible": vehicle.get("wheelchair_accessible", False),
                "bikes_allowed": vehicle.get("bikes_allowed", False),
                "contactless": vehicle.get("contactless", False),
            }
            processed_vehicles.append(processed_vehicle)

        _vehicle_positions_cache = processed_vehicles
        _vehicle_positions_last_load = datetime.now()

        logger.info(
            f"\033[1;32m✅ Loaded {len(processed_vehicles)} real-time vehicle positions\033[0m"
        )
        return processed_vehicles

    except requests.exceptions.Timeout:
        logger.error("Timeout loading vehicle positions (15s)")
        return _vehicle_positions_cache or []
    except requests.exceptions.RequestException as e:
        logger.error(f"Error loading vehicle positions: {e}")
        return _vehicle_positions_cache or []
    except Exception as e:
        logger.error(f"Unexpected error loading vehicle positions: {e}")
        return _vehicle_positions_cache or []


def get_vehicles_near_location(
    lat: float, lon: float, radius_km: float = 1.0, max_results: int = 10
) -> List[Dict[str, Any]]:
    """
    Finds vehicles currently near a specific GPS location.

    Args:
        lat: Latitude of search center.
        lon: Longitude of search center.
        radius_km: Search radius in kilometers.
        max_results: Maximum number of vehicles to return.

    Returns:
        List of nearby vehicles with distance and line info.
    """
    vehicles = load_carris_metropolitana_vehicles()
    lines = load_carris_metropolitana_lines()

    if not vehicles:
        return []

    line_map = {line_data["id"]: line_data for line_data in lines} if lines else {}

    nearby_vehicles = []
    for vehicle in vehicles:
        vehicle_lat = vehicle.get("lat")
        vehicle_lon = vehicle.get("lon")

        if vehicle_lat is None or vehicle_lon is None:
            continue

        distance = haversine_distance(lat, lon, vehicle_lat, vehicle_lon)

        if distance <= radius_km:
            # Enrich with line info
            line_id = vehicle.get("line_id", "")
            line_info = line_map.get(line_id, {})

            vehicle_with_distance = vehicle.copy()
            vehicle_with_distance["distance_km"] = round(distance, 3)
            vehicle_with_distance["line_short_name"] = line_info.get(
                "short_name", line_id
            )
            vehicle_with_distance["line_long_name"] = line_info.get("long_name", "")
            vehicle_with_distance["line_color"] = line_info.get("color", "#CCCCCC")

            nearby_vehicles.append(vehicle_with_distance)

    # Sort by distance
    nearby_vehicles.sort(key=lambda x: x["distance_km"])
    return nearby_vehicles[:max_results]


def get_vehicles_for_line(line_id: str) -> List[Dict[str, Any]]:
    """
    Gets all real-time vehicles currently serving a specific line.

    Args:
        line_id: The line ID (e.g., '1001', '1503').

    Returns:
        List of vehicles for that line with position and status.
    """
    vehicles = load_carris_metropolitana_vehicles()

    if not vehicles:
        return []

    line_vehicles = [v for v in vehicles if v.get("line_id") == line_id]

    # Sort by timestamp (most recent first)
    line_vehicles.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

    return line_vehicles


# ==========================================================================
# Stop Search Functions
# ==========================================================================


def find_stops_near_coordinates(
    lat: float, lon: float, radius_km: float = 0.5, max_results: int = 10
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
            nearby_stops.append(
                {
                    "id": stop["id"],
                    "name": stop["name"],
                    "lat": stop_lat,
                    "lon": stop_lon,
                    "distance_km": round(distance, 3),
                    "municipality": stop.get("municipality", ""),
                    "lines": stop.get("lines", []),
                }
            )

    nearby_stops.sort(key=lambda x: x["distance_km"])
    return nearby_stops[:max_results]


def find_stops_by_name(name_query: str, max_results: int = 10) -> List[Dict[str, Any]]:
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
            matching_stops.append(
                {
                    "id": stop["id"],
                    "name": stop["name"],
                    "lat": stop.get("lat"),
                    "lon": stop.get("lon"),
                    "municipality": stop.get("municipality", ""),
                    "lines": stop.get("lines", []),
                }
            )

    return matching_stops[:max_results]


def find_common_routes(
    origin_stop_ids: List[str], dest_stop_ids: List[str]
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

    line_map = {line_data["id"]: line_data for line_data in lines}

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
            "dest_stop": dest_stop_lines.get(line_id, {}),
        }
        route_options.append(route_option)

    route_options.sort(key=lambda x: x.get("short_name", ""))
    return route_options


def resolve_location(
    location_name: str, search_radius_km: float = 0.5, max_stops: int = 5
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
        "query": location_name,
    }

    clean_name = location_name.strip()

    # Step 1: Try direct name match
    name_matches = find_stops_by_name(clean_name, max_results=max_stops)

    if len(name_matches) >= 3:
        result["method"] = "name_match"
        result["stops"] = name_matches
        result["success"] = True
        logger.info(
            f"Resolved '{clean_name}' via name match ({len(name_matches)} stops)"
        )
        return result

    # Step 2: Try geocoding
    geocoded = geocode_location(clean_name)

    if geocoded:
        result["location"] = geocoded

        nearby_stops = find_stops_near_coordinates(
            geocoded["lat"],
            geocoded["lon"],
            radius_km=search_radius_km,
            max_results=max_stops,
        )

        if nearby_stops:
            result["method"] = "geocoding"
            result["stops"] = nearby_stops
            result["success"] = True
            logger.info(
                f"Resolved '{clean_name}' via geocoding ({len(nearby_stops)} stops)"
            )
            return result

        # Try larger radius
        if not nearby_stops and search_radius_km < 1.0:
            nearby_stops = find_stops_near_coordinates(
                geocoded["lat"], geocoded["lon"], radius_km=1.0, max_results=max_stops
            )

            if nearby_stops:
                result["method"] = "geocoding_expanded"
                result["stops"] = nearby_stops
                result["success"] = True
                logger.info(
                    f"Resolved '{clean_name}' via expanded geocoding ({len(nearby_stops)} stops)"
                )
                return result

    # Step 3: Fallback to name matches
    if name_matches:
        result["method"] = "name_match_fallback"
        result["stops"] = name_matches
        result["success"] = True
        logger.info(
            f"Resolved '{clean_name}' via fallback name match ({len(name_matches)} stops)"
        )
        return result

    logger.warning(f"Could not resolve location '{clean_name}'")
    return result


# ==========================================================================
# LangChain Tools
# ==========================================================================


@tool
def get_real_time_bus_positions(
    line_id: Optional[str] = None,
    location: Optional[str] = None,
    radius_km: float = 1.0,
) -> str:
    """
    Gets real-time GPS positions of Carris Metropolitana buses.

    Shows live locations of buses with their current line, route, speed,
    and status (at stop, incoming, or in transit).

    Args:
        line_id: Optional line ID to filter (e.g., '1001', '1503'). If None, shows all lines.
        location: Optional location name to find nearby buses (e.g., 'Colombo', 'Oriente').
        radius_km: Search radius in km when using location (default: 1.0).

    Returns:
        str: Formatted list of buses with GPS positions, status, and vehicle info.

    Examples:
        >>> get_real_time_bus_positions()  # All buses
        >>> get_real_time_bus_positions("1503")  # Line 1503 only
        >>> get_real_time_bus_positions(location="Colombo", radius_km=0.5)  # Buses near Colombo
    """
    vehicles = load_carris_metropolitana_vehicles()
    lines = load_carris_metropolitana_lines()

    if not vehicles:
        return "❌ Failed to fetch real-time vehicle positions."

    line_map = {line_data["id"]: line_data for line_data in lines} if lines else {}

    # Filter vehicles
    filtered_vehicles = vehicles

    if line_id:
        filtered_vehicles = [v for v in vehicles if v.get("line_id") == line_id]
        if not filtered_vehicles:
            return f"❌ No active vehicles found for line {line_id}."

    if location:
        geocoded = geocode_location(location)
        if not geocoded:
            return f"❌ Could not geocode location: {location}"

        nearby = get_vehicles_near_location(
            geocoded["lat"], geocoded["lon"], radius_km=radius_km, max_results=15
        )
        if not nearby:
            return f"❌ No buses found within {radius_km}km of {location}."
        filtered_vehicles = nearby

    # Build response
    if line_id:
        line_info = line_map.get(line_id, {})
        line_name = line_info.get("short_name", line_id)
        line_long = line_info.get("long_name", "")
        response = f"🚌 Real-Time Positions - Line {line_name}\n"
        if line_long:
            response += f"📍 {line_long}\n"
    elif location:
        response = f"🚌 Real-Time Buses Near {location.title()}\n"
        response += f"📍 Radius: {radius_km}km | {len(filtered_vehicles)} buses found\n"
    else:
        response = "🚌 Real-Time Bus Positions - All Lines\n"
        response += f"📊 {len(filtered_vehicles)} active vehicles\n"

    response += "=" * 50 + "\n\n"

    if not filtered_vehicles:
        response += "ℹ️ No vehicles currently active.\n"
        return response

    # Status icons
    status_icons = {"INCOMING_AT": "🚏", "STOPPED_AT": "🛑", "IN_TRANSIT_TO": "🚌"}

    # Show vehicles
    for i, vehicle in enumerate(filtered_vehicles[:10], 1):
        v_line_id = vehicle.get("line_id", "N/A")
        line_info = line_map.get(v_line_id, {})
        line_short = line_info.get("short_name", v_line_id)

        status = vehicle.get("current_status", "UNKNOWN")
        status_icon = status_icons.get(status, "🚌")

        lat = vehicle.get("lat", 0)
        lon = vehicle.get("lon", 0)
        speed = vehicle.get("speed")
        bearing = vehicle.get("bearing", 0)
        license_plate = vehicle.get("license_plate", "N/A")
        vehicle_model = vehicle.get("vehicle_model", "")
        door_status = vehicle.get("door_status", "")

        # Format timestamp
        timestamp = vehicle.get("timestamp", 0)
        if timestamp:
            last_update = datetime.fromtimestamp(timestamp / 1000)
            time_ago = (datetime.now() - last_update).total_seconds()
            if time_ago < 60:
                time_str = f"{int(time_ago)}s ago"
            else:
                time_str = f"{int(time_ago / 60)}m ago"
        else:
            time_str = "N/A"

        response += f"{i}. {status_icon} Line {line_short}\n"
        response += f"   🚗 {license_plate}"
        if vehicle_model:
            response += f" ({vehicle_model})"
        response += "\n"
        response += f"   📍 GPS: {lat:.5f}, {lon:.5f}\n"
        if speed is not None:
            response += f"   💨 Speed: {speed} km/h | Bearing: {bearing}°\n"
        response += f"   📡 Last update: {time_str}\n"

        if status == "STOPPED_AT":
            response += "   🛑 Currently stopped"
            if door_status:
                response += f" (Doors: {door_status})"
            response += "\n"
        elif status == "INCOMING_AT":
            response += "   🚏 Approaching next stop\n"

        if "distance_km" in vehicle:
            response += (
                f"   📏 Distance from search point: {vehicle['distance_km']} km\n"
            )

        response += "\n"

    if len(filtered_vehicles) > 10:
        response += f"... and {len(filtered_vehicles) - 10} more vehicles.\n"

    return response


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
    alerts = data if isinstance(data, list) else data.get("entity", [])

    if not alerts:
        return "✅ No active alerts from Carris Metropolitana."

    response = "⚠️ **Carris Metropolitana Service Alerts**\n"
    response += "=" * 45 + "\n\n"

    for i, alert in enumerate(alerts[:10], 1):
        # Handle both old format (nested under 'alert') and new format (flat)
        alert_data = alert.get("alert", alert)

        # Header
        header_text = alert_data.get("header_text", {})
        header = header_text.get("translation", [{}])[0].get("text", "No title")

        # Description
        desc_text = alert_data.get("description_text", {})
        desc = desc_text.get("translation", [{}])[0].get("text", "No details")

        # Time period
        active_period = alert_data.get("active_period", [{}])[0]
        start = format_timestamp(active_period.get("start", 0))
        end = format_timestamp(active_period.get("end", 0))

        # Cause and effect
        cause = alert_data.get("cause", "UNKNOWN").replace("_", " ").title()
        effect = alert_data.get("effect", "UNKNOWN").replace("_", " ").title()

        # Affected routes
        informed_entity = alert_data.get("informed_entity", [])
        routes = [e.get("route_id", "") for e in informed_entity if e.get("route_id")]
        routes_str = ", ".join(routes[:5]) if routes else "All routes"

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

    name = stop_data.get("name", "N/A")
    locality = stop_data.get("locality", "")
    lat = stop_data.get("lat", "N/A")
    lon = stop_data.get("lon", "N/A")
    lines = stop_data.get("lines", [])

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
            line_id = arrival.get("line_id", "N/A")
            headsign = arrival.get("headsign", "N/A")
            estimated = arrival.get("estimated_arrival", "")
            scheduled = arrival.get("scheduled_arrival", "")

            arrival_time = estimated or scheduled
            if arrival_time:
                try:
                    arr_dt = datetime.fromisoformat(arrival_time.replace("Z", "+00:00"))
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
def find_direct_bus_lines(origin: str, destination: str) -> str:
    """
    Finds Carris Metropolitana bus lines that connect two locations directly.

    This is the BEST TOOL for answering "How do I get from X to Y by bus?" questions.
    It searches the line data to find buses that serve BOTH the origin AND destination.
    Also checks for service alerts affecting the routes.

    Works with:
        - Town/city names: 'Montijo', 'Alcochete', 'Loures'
        - Transport hubs: 'Oriente', 'Campo Grande', 'Cais do Sodré'
        - Neighborhoods: 'Parque das Nações', 'Moscavide'

    Args:
        origin: Starting location name.
        destination: Ending location name.

    Returns:
        str: List of direct bus lines with line numbers, route details, and alerts.
    """
    data = fetch_json_with_retry(CARRIS_LINES_URL)

    if not data:
        return "❌ Failed to fetch Carris Metropolitana lines data."

    if not isinstance(data, list):
        return "❌ Unexpected response format."

    def normalize(text: str) -> str:
        """Normalize text for comparison."""
        return (
            text.lower()
            .replace("é", "e")
            .replace("ã", "a")
            .replace("õ", "o")
            .replace("ç", "c")
            .replace("á", "a")
            .replace("ó", "o")
            .replace("í", "i")
            .replace("ú", "u")
        )

    origin_norm = normalize(origin)
    dest_norm = normalize(destination)

    direct_lines = []

    for line in data:
        short_name = line.get("short_name", "")
        long_name = line.get("long_name") or ""
        line_id = line.get("id", "")
        localities = line.get("localities") or []
        municipalities = line.get("municipalities") or []

        # Normalize all searchable fields
        long_name_norm = normalize(long_name)
        localities_norm = [normalize(loc) for loc in localities if loc]
        muni_norm = [normalize(m) for m in municipalities if m]

        # Check if origin is served
        origin_match = (
            origin_norm in long_name_norm
            or any(origin_norm in loc for loc in localities_norm)
            or any(origin_norm in m for m in muni_norm)
        )

        # Check if destination is served
        dest_match = (
            dest_norm in long_name_norm
            or any(dest_norm in loc for loc in localities_norm)
            or any(dest_norm in m for m in muni_norm)
        )

        if origin_match and dest_match:
            direct_lines.append(
                {
                    "id": line_id,
                    "short_name": short_name,
                    "long_name": long_name,
                    "localities": [loc for loc in localities if loc],
                    "municipalities": [m for m in municipalities if m],
                }
            )

    if not direct_lines:
        response = f"❌ **Sem linhas diretas entre '{origin}' e '{destination}'**\n\n"
        response += "💡 **Sugestões:**\n"
        response += "   • Pode ser necessário fazer transbordo\n"
        response += "   • Considere combinar Metro + Autocarro\n"
        response += "   • Consulte carrismetropolitana.pt para mais opções\n"
        return response

    # Fetch alerts to check if any affect these lines
    alerts_data = fetch_json_with_retry(CARRIS_ALERTS_URL)
    alerts_list = (
        alerts_data
        if isinstance(alerts_data, list)
        else (alerts_data.get("entity", []) if alerts_data else [])
    )

    # Build set of affected line IDs
    affected_lines = set()
    relevant_alerts = []
    for alert in alerts_list:
        alert_data = alert.get("alert", alert)
        informed_entity = alert_data.get("informed_entity", [])
        for entity in informed_entity:
            route_id = entity.get("route_id", "")
            if route_id:
                affected_lines.add(route_id)
        # Check if alert affects any of our direct lines
        for line in direct_lines:
            if line["id"] in affected_lines or line["short_name"] in affected_lines:
                header_text = alert_data.get("header_text", {})
                header = header_text.get("translation", [{}])[0].get("text", "")
                if header and header not in [a["header"] for a in relevant_alerts]:
                    relevant_alerts.append(
                        {
                            "header": header,
                            "lines": [
                                e.get("route_id", "")
                                for e in informed_entity
                                if e.get("route_id")
                            ],
                        }
                    )

    # Build response - RESPECT the user's direction: origin → destination
    response = f"🚌 **Autocarros: {origin.title()} → {destination.title()}**\n"
    response += "=" * 50 + "\n\n"

    # Show alerts first if any
    if relevant_alerts:
        response += "⚠️ **ALERTAS DE SERVIÇO:**\n"
        for alert in relevant_alerts[:3]:
            response += f"   • {alert['header']}\n"
        response += "\n"

    response += f"✅ **{len(direct_lines)} linha(s) direta(s) encontrada(s):**\n\n"

    for i, line in enumerate(direct_lines[:6], 1):
        short_name = line["short_name"]
        long_name = line["long_name"]
        localities = line["localities"]
        line_id = line["id"]

        # Check if this line has alerts
        has_alert = line_id in affected_lines or short_name in affected_lines
        alert_icon = " ⚠️" if has_alert else ""

        response += f"**{i}. 🚍 Linha {short_name}**{alert_icon}\n"

        # Show the official route terminals (not direction, just terminals served)
        if long_name:
            # Replace " - " with " ↔ " to show it's bidirectional
            display_name = long_name.replace(" - ", " ↔ ")
            response += f"   📍 Terminais: {display_name}\n"

        # Show localities if available, ordered from origin to destination when possible
        if localities:
            origin_idx = -1
            dest_idx = -1
            for idx, loc in enumerate(localities):
                loc_norm = normalize(loc)
                if origin_idx < 0 and (
                    origin_norm in loc_norm or loc_norm in origin_norm
                ):
                    origin_idx = idx
                if dest_idx < 0 and (dest_norm in loc_norm or loc_norm in dest_norm):
                    dest_idx = idx

            if origin_idx >= 0 and dest_idx >= 0:
                if origin_idx < dest_idx:
                    key_stops = localities[origin_idx : dest_idx + 1]
                else:
                    key_stops = localities[dest_idx : origin_idx + 1][::-1]

                if len(key_stops) > 6:
                    display_stops = key_stops[:6]
                    response += (
                        f"   🚏 {' → '.join(display_stops)} (+{len(key_stops) - 6})\n"
                    )
                else:
                    response += f"   🚏 {' → '.join(key_stops)}\n"
            else:
                key_stops = localities[:6]
                response += f"   🚏 Passa por: {', '.join(key_stops)}"
                if len(localities) > 6:
                    response += f" (+{len(localities) - 6})"
                response += "\n"
        response += "\n"

    if len(direct_lines) > 6:
        other_lines = [line_data["short_name"] for line_data in direct_lines[6:]]
        response += f"📋 Outras linhas: {', '.join(other_lines)}\n\n"

    response += "-" * 50 + "\n"
    response += "💡 **Como usar:**\n"
    response += f"   • Procure pelo número da linha (ex: **{direct_lines[0]['short_name']}**) na paragem\n"
    response += f"   • Verifique a direção do autocarro ({origin.title()} → {destination.title()})\n"
    response += "   • Horários e paragens: carrismetropolitana.pt\n"

    return response


@tool
def search_carris_metropolitana_lines(query: str) -> str:
    """
    Searches for Carris Metropolitana (suburban) bus lines.

    .IMPORTANT: This searches SUBURBAN bus lines only. Urban Lisbon buses
    like lines 28E, 738, 732 are NOT included.

    Searches in:
        - Line number (short_name)
        - Line description (long_name)
        - Municipalities served
        - Localities served (includes landmarks like 'Oriente', 'Parque das Nações')

    Args:
        query: Line number, destination name, or area to search.
               Examples: '1718', 'Sintra', 'Cascais', 'Almada', 'Oriente', 'Montijo'

    Returns:
        str: Matching lines with route details and localities served.
    """
    data = fetch_json_with_retry(CARRIS_LINES_URL)

    if not data:
        return "❌ Failed to fetch Carris Metropolitana lines data."

    if not isinstance(data, list):
        return "❌ Unexpected response format."

    query_lower = query.lower()
    query_normalized = (
        query_lower.replace("é", "e")
        .replace("ã", "a")
        .replace("õ", "o")
        .replace("ç", "c")
    )

    matches = []

    for line in data:
        short_name = line.get("short_name", "")
        long_name = line.get("long_name") or ""
        line_id = line.get("id", "")
        municipalities = line.get("municipalities") or []
        localities = line.get("localities") or []

        # Normalize long_name for accent-insensitive search
        long_name_norm = (
            long_name.lower()
            .replace("é", "e")
            .replace("ã", "a")
            .replace("õ", "o")
            .replace("ç", "c")
        )

        # Build searchable strings (filter None values)
        muni_str = " ".join([m for m in municipalities if m]).lower()
        localities_str = " ".join([loc for loc in localities if loc]).lower()
        localities_norm = (
            localities_str.replace("é", "e")
            .replace("ã", "a")
            .replace("õ", "o")
            .replace("ç", "c")
        )

        # Search in all fields: short_name, long_name, line_id, municipalities, localities
        if (
            query_lower in short_name.lower()
            or query_normalized in long_name_norm
            or query_lower in line_id.lower()
            or query_lower in muni_str
            or query_normalized in localities_norm
        ):
            matches.append(line)

    if not matches:
        urban_keywords = [
            "rossio",
            "baixa",
            "chiado",
            "alfama",
            "bairro alto",
            "28e",
            "738",
            "732",
            "15e",
        ]
        if any(kw in query_lower for kw in urban_keywords):
            return (
                f"❌ No Carris Metropolitana lines found for '{query}'\n\n"
                + CARRIS_LIMITATION_NOTICE
            )
        return f"❌ No lines found matching: '{query}'\n\n💡 Try searching by area: Sintra, Cascais, Almada, Oeiras, Loures, Montijo, Oriente"

    response = (
        f"🚌 **Carris Metropolitana lines matching '{query}'** ({len(matches)} found)\n"
    )
    response += "=" * 50 + "\n\n"

    for i, line in enumerate(matches[:15], 1):
        short_name = line.get("short_name", "N/A")
        long_name = line.get("long_name") or "N/A"
        municipalities = line.get("municipalities") or []
        localities = line.get("localities") or []

        response += f"{i}. **Line {short_name}**\n"
        response += f"   📍 {long_name}\n"
        if municipalities:
            muni_list = [m for m in municipalities[:4] if m]
            response += f"   🏘️ Municipalities: {', '.join(muni_list)}\n"
        if localities:
            # Show key localities that might match the query
            key_localities = [loc for loc in localities[:8] if loc]
            response += f"   📌 Localities: {', '.join(key_localities)}\n"
        response += "\n"

    if len(matches) > 15:
        response += f"... and {len(matches) - 15} more lines.\n"

    response += "\n" + "-" * 40 + "\n"
    response += "💡 Podes perguntar por rotas diretas entre dois locais ou horários de uma linha específica.\n"

    return response


@tool
def get_bus_realtime_locations(line_id: Optional[str] = None) -> str:
    """
    Gets real-time GPS locations of Carris Metropolitana buses.

    .IMPORTANT: Only works for suburban buses, not urban Lisbon.

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
        filtered = [v for v in data if v.get("line_id") == line_id]
        if not filtered:
            return (
                f"ℹ️ No active buses found on line {line_id} at this time.\n\n"
                f"💡 The line may not be operating right now."
            )
        buses = filtered
        response = f"🚌 **Real-Time Bus Locations - Line {line_id}**\n"
    else:
        buses = data
        response = "🚌 **Real-Time Bus Locations Overview**\n"

    response += "=" * 50 + "\n"
    response += f"📊 Active buses: {len(buses)}\n"
    response += f"🕐 Updated: {datetime.now().strftime('%H:%M:%S')}\n"
    response += "=" * 50 + "\n\n"

    if line_id:
        for i, bus in enumerate(buses[:15], 1):
            lat = bus.get("lat", 0)
            lon = bus.get("lon", 0)
            speed = bus.get("speed", 0)
            bearing = bus.get("bearing", 0)
            status = bus.get("current_status", "UNKNOWN")
            stop_id = bus.get("stop_id", "N/A")

            status_emoji = {
                "IN_TRANSIT_TO": "🚌➡️",
                "STOPPED_AT": "🚏",
                "INCOMING_AT": "📍",
            }.get(status, "🚌")

            directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            dir_idx = int((bearing + 22.5) % 360 / 45)
            direction = directions[dir_idx] if bearing else "?"

            response += f"**{i}. Bus {bus.get('id', 'N/A')[:20]}**\n"
            response += (
                f"   {status_emoji} Status: {status.replace('_', ' ').title()}\n"
            )
            response += f"   📍 Position: ({lat:.5f}, {lon:.5f})\n"
            response += f"   🧭 Direction: {direction} | Speed: {speed:.1f} km/h\n"
            response += f"   🚏 Next stop ID: {stop_id}\n\n"

        if len(buses) > 15:
            response += f"... and {len(buses) - 15} more buses on this line.\n"
    else:
        from collections import Counter

        line_counts = Counter(v.get("line_id", "Unknown") for v in buses)

        response += "**Top 15 lines by active buses:**\n\n"
        for line, count in line_counts.most_common(15):
            response += f"   🚌 Line **{line}**: {count} buses\n"

        response += f"\n... {len(line_counts)} lines total with active buses.\n"

    response += "\n" + "-" * 40 + "\n"
    response += (
        "💡 Podes perguntar pela localização em tempo real de uma linha específica.\n"
    )

    return response


@tool
def get_bus_next_departures(
    line_id: str, stop_id: str = "", start_time: str = ""
) -> str:
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
        if line.get("id") == line_id or line.get("short_name") == line_id:
            line_info = line
            break

    if not line_info:
        return (
            f"❌ Line '{line_id}' not found.\n\n"
            f"💡 Verifica o identificador da linha. Podes perguntar-me pelas linhas disponíveis."
        )

    patterns = line_info.get("patterns", [])
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

    headsign = pattern_data.get("headsign", "N/A")
    trips = pattern_data.get("trips", [])

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
        ref_time = now_dt.strftime("%H:%M:%S")
        ref_time_display = "NOW"

    today = datetime.now().strftime("%Y%m%d")
    today_trips = [t for t in trips if today in t.get("dates", [])]

    if today_trips:
        response += f"**🕐 Departures after {ref_time_display}**:\n"
        response += "-" * 30 + "\n"

        departures = []
        for trip in today_trips:
            schedule = trip.get("schedule", [])
            if schedule:
                first_time = schedule[0].get("arrival_time", "N/A")
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
            path = pattern_data.get("path", [])
            stop_idx = next(
                (
                    i
                    for i, s in enumerate(path)
                    if s.get("stop", {}).get("id") == stop_id
                ),
                None,
            )

            if stop_idx is not None:
                stop_name = path[stop_idx].get("stop", {}).get("name", stop_id)
                response += f"\n**⏱️ At stop {stop_name}:**\n"

                stop_times = []
                for trip in today_trips:
                    schedule = trip.get("schedule", [])
                    if len(schedule) > stop_idx:
                        time_at = schedule[stop_idx].get("arrival_time", "N/A")
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
    search_radius_km: float = 0.5,
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
            origin_lat, origin_lon, radius_km=search_radius_km, max_results=10
        )
        if origin_stops:
            response += f"   ✅ Using provided coordinates ({origin_lat:.4f}, {origin_lon:.4f})\n"
            response += (
                f"   📍 Found {len(origin_stops)} stops within {search_radius_km}km\n"
            )
            for stop in origin_stops[:3]:
                response += (
                    f"      • {stop['name']} ({stop['distance_km'] * 1000:.0f}m)\n"
                )
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
                response += (
                    f"   🌍 Geocoded '{origin}' → {loc.get('name', 'Unknown')[:60]}\n"
                )
                response += f"   📍 Coordinates: ({loc.get('lat', 0):.4f}, {loc.get('lon', 0):.4f})\n"
                response += f"   ✅ Found {len(stops)} bus stops nearby\n"
            else:
                response += f"   ✅ Found {len(stops)} stops matching '{origin}'\n"

            for stop in stops[:3]:
                dist = stop.get("distance_km")
                if dist:
                    response += f"      • {stop['name']} ({dist * 1000:.0f}m)\n"
                else:
                    response += f"      • {stop['name']}\n"
        else:
            response += f"   ❌ Could not resolve '{origin}'\n"

    origin_stops = origin_resolved.get("stops", [])

    if not origin_stops:
        origin_loc = origin_resolved.get("location")
        if origin_loc and is_within_lisbon_city(
            origin_loc.get("lat"), origin_loc.get("lon")
        ):
            response += f"\n📍 **'{origin}' is in central Lisbon**\n"
            response += (
                "   🚋 Try using **Carris Urbana** (carris.pt) for urban routes.\n\n"
            )
        else:
            response += f"\n❌ **No bus stops found near '{origin}'**\n"
            response += "   💡 Try adding 'Lisboa' to your search.\n"
        return response

    response += "\n"

    # Resolve destination
    response += "🔍 **Resolving destination location...**\n"

    if dest_lat is not None and dest_lon is not None:
        dest_stops = find_stops_near_coordinates(
            dest_lat, dest_lon, radius_km=search_radius_km, max_results=10
        )
        if dest_stops:
            response += (
                f"   ✅ Using provided coordinates ({dest_lat:.4f}, {dest_lon:.4f})\n"
            )
            response += (
                f"   📍 Found {len(dest_stops)} stops within {search_radius_km}km\n"
            )
            for stop in dest_stops[:3]:
                response += (
                    f"      • {stop['name']} ({stop['distance_km'] * 1000:.0f}m)\n"
                )
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
                dist = stop.get("distance_km")
                if dist:
                    response += f"      • {stop['name']} ({dist * 1000:.0f}m)\n"
                else:
                    response += f"      • {stop['name']}\n"
        else:
            response += f"   ❌ Could not resolve '{destination}'\n"

    dest_stops = dest_resolved.get("stops", [])

    if not dest_stops:
        dest_loc = dest_resolved.get("location")
        if dest_loc and is_within_lisbon_city(dest_loc.get("lat"), dest_loc.get("lon")):
            response += f"\n📍 **'{destination}' is in central Lisbon**\n"
            response += (
                "   🚋 Try using **Carris Urbana** (carris.pt) for urban routes.\n\n"
            )
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
                    "lines": [],
                }
            grouped_routes[key]["lines"].append(route.get("short_name", "?"))

        response += f"✅ **{len(grouped_routes)} ROUTE OPTION(S) FOUND** ({len(route_options)} lines total)\n"
        response += "-" * 40 + "\n\n"

        for i, ((origin_name, dest_name), group) in enumerate(
            list(grouped_routes.items())[:5], 1
        ):
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
            response += (
                f"   At {origin}: {', '.join(sorted(list(origin_all_lines))[:10])}"
            )
            if len(origin_all_lines) > 10:
                response += f" (+{len(origin_all_lines) - 10} more)"
            response += "\n"

        if dest_all_lines:
            response += (
                f"   At {destination}: {', '.join(sorted(list(dest_all_lines))[:10])}"
            )
            if len(dest_all_lines) > 10:
                response += f" (+{len(dest_all_lines) - 10} more)"
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
    response += (
        "   • Check carrismetropolitana.pt or carris.pt for detailed schedules\n"
    )
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
