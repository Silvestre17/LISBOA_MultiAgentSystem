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
#   Usage:
#     > python tools/carrismetropolitana_api.py
#       Run the manual Carris Metropolitana tool test suite against line, stop, route, and realtime helpers.
#
#   API Documentation: https://github.com/carrismetropolitana/api
#   API Base: https://api.carrismetropolitana.pt
# ==========================================================================

# Required libraries:
# pip install requests langchain-core

import logging
import os
import re
import unicodedata
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from langchain_core.tools import tool

try:
    import config as _project_config
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
else:
    del _project_config

logger = logging.getLogger(__name__)

CARRIS_METROPOLITANA_CITY_FALLBACKS = {
    "alcochete": (38.7553, -8.9609),
    "almada": (38.6765, -9.1654),
    "amadora": (38.7597, -9.2397),
    "barreiro": (38.6631, -9.0724),
    "cascais": (38.6979, -9.4215),
    "lisboa": (38.7223, -9.1393),
    "loures": (38.8309, -9.1685),
    "mafra": (38.9379, -9.3276),
    "moita": (38.6508, -8.9904),
    "montijo": (38.7067, -8.9739),
    "odivelas": (38.7927, -9.1838),
    "oeiras": (38.6970, -9.3017),
    "palmela": (38.5690, -8.9013),
    "seixal": (38.6400, -9.1015),
    "sesimbra": (38.4445, -9.1015),
    "setubal": (38.5244, -8.8882),
    "setúbal": (38.5244, -8.8882),
    "sintra": (38.8029, -9.3817),
    "vila franca": (38.9553, -8.9897),
    "vila franca de xira": (38.9553, -8.9897),
}

try:
    from tools.utils import fetch_json_with_retry, haversine_distance
except ImportError:
    from utils import fetch_json_with_retry, haversine_distance

try:
    from tools.location_resolver import build_location_ambiguity_preamble
except ImportError:
    from location_resolver import build_location_ambiguity_preamble

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
_vehicle_feed_meta: Dict[str, Any] = {
    "source": "uninitialized",
    "generated_at": None,
    "data_age_seconds": None,
    "last_error": None,
    "missing_coordinates": 0,
    "vehicle_count": 0,
}

# ==========================================================================
# Carris Urban vs Metropolitan Limitation Notice
# ==========================================================================

CARRIS_LIMITATION_NOTICE = """
⚠️ **IMPORTANT: Carris Metropolitana Scope Note**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Carris Metropolitana API** (this search) covers AML metropolitan / intermunicipal buses:
• It serves the 18 AML municipalities: Alcochete, Almada, Amadora, Barreiro, Cascais, Lisboa, Loures, Mafra, Moita, Montijo, Odivelas, Oeiras, Palmela, Seixal, Sesimbra, Setúbal, Sintra and Vila Franca de Xira
• Many lines also enter **Lisbon municipality** and hubs such as Colégio Militar, Marquês de Pombal, Belém or Oriente

**Carris (Urban Lisbon)** routes are still separate:
• Urban-only routes like 28E (tram), 15E, 732 or 738 are managed by Carris
• For Lisbon city-only trips, always cross-check Carris Urban when relevant

💡 TIP: For mixed Lisbon + suburb trips, compare Carris Metropolitana, Carris Urban, Metro and CP.
"""


def _update_vehicle_feed_meta(
    *,
    source: str,
    generated_at: Optional[datetime] = None,
    data_age_seconds: Optional[float] = None,
    last_error: Optional[str] = None,
    missing_coordinates: int = 0,
    vehicle_count: int = 0,
) -> None:
    """Stores the latest metadata about the realtime vehicle feed."""
    _vehicle_feed_meta.update(
        {
            "source": source,
            "generated_at": generated_at.isoformat() if isinstance(generated_at, datetime) else None,
            "data_age_seconds": data_age_seconds,
            "last_error": last_error,
            "missing_coordinates": missing_coordinates,
            "vehicle_count": vehicle_count,
        }
    )


def _build_vehicle_freshness_note(include_missing_coordinates: bool = True) -> str:
    """Formats a short freshness note for realtime vehicle responses."""
    source = _vehicle_feed_meta.get("source")
    age = _vehicle_feed_meta.get("data_age_seconds")
    last_error = _vehicle_feed_meta.get("last_error")
    missing_coordinates = int(_vehicle_feed_meta.get("missing_coordinates", 0) or 0)

    lines = []

    if source == "live":
        lines.append("📡 Data freshness: live Carris Metropolitana feed snapshot.")
    elif source == "cache":
        age_text = f"{int(age)}s" if age is not None else "unknown age"
        lines.append(f"📡 Data freshness: cached Carris Metropolitana vehicle snapshot ({age_text} old).")
    elif source == "stale_cache":
        age_text = f"{int(age)}s" if age is not None else "unknown age"
        lines.append(
            f"⚠️ Data freshness: using cached vehicle data ({age_text} old) because the live Carris Metropolitana endpoint is temporarily unavailable."
        )
    elif source == "unavailable":
        lines.append("⚠️ Data freshness: live Carris Metropolitana vehicle data is temporarily unavailable.")

    if include_missing_coordinates and missing_coordinates:
        lines.append(
            f"ℹ️ {missing_coordinates} vehicle(s) were omitted because the API response did not include usable GPS coordinates."
        )

    if last_error and source in {"stale_cache", "unavailable"}:
        lines.append(f"ℹ️ Last realtime feed issue: {last_error}")

    return "\n".join(lines)


def _append_carris_scope_footer(response: str, include_freshness: bool = False) -> str:
    """Appends a compact scope note for Carris Metropolitana outputs."""
    parts = []
    if include_freshness:
        freshness_note = _build_vehicle_freshness_note()
        if freshness_note:
            parts.append(freshness_note)

    parts.append(
        "⚠️ Scope: Carris Metropolitana covers AML metropolitan / intermunicipal buses and many lines entering Lisbon municipality, but not Carris Urban-only routes such as 28E, 15E, 732 or 738."
    )
    parts.append(
        "💡 For Lisbon city-only bus or tram trips, cross-check Carris Urban or Metro data."
    )

    cleaned_parts = [part for part in parts if part]
    if not cleaned_parts:
        return response

    return response.rstrip() + "\n\n" + "\n".join(cleaned_parts) + "\n"


def _format_vehicle_relative_time(ts: Any) -> tuple[Optional[str], bool]:
    """Formats a vehicle timestamp into a relative age and flags stale upstream values."""
    last_update = _parse_unix_timestamp(ts)
    if last_update is None:
        return None, False

    time_ago = (datetime.now() - last_update).total_seconds()

    # Negative values beyond a small skew and very old values are usually upstream issues.
    if time_ago < -120 or time_ago > 6 * 3600:
        return None, True

    time_ago = max(0, time_ago)
    if time_ago < 60:
        return f"{int(time_ago)}s ago", False
    return f"{int(time_ago / 60)}m ago", False


def _normalize_carris_display_value(text: str) -> str:
    """Normalizes noisy upstream display values for deduplication only."""
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9\s-]", " ", normalized)
    normalized = re.sub(r"([a-z])\1+", r"\1", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _normalize_area_key(text: str) -> str:
    """Normalize a municipality or area name for fallback matching."""
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9\s-]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _bare_municipality_key(text: str) -> str:
    """Return a municipality key only when the user wrote just that broad area."""
    normalized = _normalize_area_key(text)
    if not normalized:
        return ""
    if normalized == "vila franca":
        return "vila franca de xira"
    return normalized if normalized in CARRIS_METROPOLITANA_CITY_FALLBACKS else ""


def _clean_carris_display_list(
    values: List[Any],
    *,
    max_items: Optional[int] = None,
    drop_numeric_only: bool = True,
) -> List[str]:
    """Removes empty, numeric-only, and duplicate upstream values while preserving order."""
    cleaned: List[str] = []
    seen: set[str] = set()

    for raw_value in values:
        value = str(raw_value or "").strip()
        if not value:
            continue
        if drop_numeric_only and re.fullmatch(r"[0-9\s,.-]+", value):
            continue

        key = _normalize_carris_display_value(value)
        if not key or key in seen:
            continue

        seen.add(key)
        cleaned.append(value)

        if max_items is not None and len(cleaned) >= max_items:
            break

    return cleaned


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


def _parse_unix_timestamp(ts: Any) -> Optional[datetime]:
    """Parses Unix timestamps that may arrive in seconds or milliseconds."""
    try:
        timestamp = float(ts)
    except (TypeError, ValueError):
        return None

    if timestamp <= 0:
        return None

    # Heuristic: timestamps above 1e11 are almost certainly milliseconds.
    if timestamp >= 100_000_000_000:
        timestamp /= 1000.0

    try:
        return datetime.fromtimestamp(timestamp)
    except (ValueError, TypeError, OSError, OverflowError):
        return None


def format_timestamp(ts: int) -> str:
    """
    Converts Unix timestamp (seconds or milliseconds) to readable format.

    Args:
        ts: Unix timestamp in seconds or milliseconds.

    Returns:
        Formatted datetime string.
    """
    dt = _parse_unix_timestamp(ts)
    return dt.strftime("%H:%M:%S") if dt else "N/A"


def is_within_lisbon_city(lat: Optional[float], lon: Optional[float]) -> bool:
    """
    Checks if coordinates are within central Lisbon city limits.

    Lisbon city boundaries (approximate):
    - Latitude: 38.68 to 38.80
    - Longitude: -9.24 to -9.10

    Args:
        lat: Latitude.
        lon: Longitude.

    Returns:
        True if within central Lisbon.
    """
    if lat is None or lon is None:
        return False
    return 38.68 <= lat <= 38.80 and -9.24 <= lon <= -9.10


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
    Geocodes a location name to GPS coordinates using the shared resolver.

    Args:
        location_name: Name of the location (e.g., 'Colombo', 'Torre de Belém').

    Returns:
        Dict with name, lat, lon, type, address or None if not found.
    """
    try:
        from tools.location_resolver import resolve_location_query
    except ImportError:
        from location_resolver import resolve_location_query

    resolved = resolve_location_query(
        location_name,
        prefer_city=False,
        allow_aml=True,
    )
    if not resolved.get("success") or resolved.get("lat") is None or resolved.get("lon") is None:
        logger.warning("Could not geocode '%s'", location_name)
        return None

    return {
        "name": resolved["display_name"],
        "lat": resolved["lat"],
        "lon": resolved["lon"],
        "type": resolved.get("type", "unknown"),
        "class": resolved.get("class", "unknown"),
        "importance": float(resolved.get("importance", 0.0)),
        "address": resolved.get("address", {}),
        "query_used": resolved.get("query_used", location_name),
        "scope": resolved.get("scope", "unknown"),
        "confidence": resolved.get("confidence", 0.0),
    }


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
        _update_vehicle_feed_meta(
            source="cache",
            generated_at=_vehicle_positions_last_load,
            data_age_seconds=age,
            last_error=None,
            missing_coordinates=int(_vehicle_feed_meta.get("missing_coordinates", 0) or 0),
            vehicle_count=len(_vehicle_positions_cache),
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
        missing_coordinates = 0
        for vehicle in vehicles:
            # Process only if vehicle has position data
            lat = vehicle.get("lat")
            lon = vehicle.get("lon")

            if lat is None or lon is None:
                missing_coordinates += 1
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
        _update_vehicle_feed_meta(
            source="live",
            generated_at=_vehicle_positions_last_load,
            data_age_seconds=0,
            last_error=None,
            missing_coordinates=missing_coordinates,
            vehicle_count=len(processed_vehicles),
        )

        logger.info(
            f"\033[1;32m✅ Loaded {len(processed_vehicles)} real-time vehicle positions\033[0m"
        )
        return processed_vehicles

    except requests.exceptions.Timeout:
        logger.error("Timeout loading vehicle positions (15s)")
        cache_age = (
            (datetime.now() - _vehicle_positions_last_load).total_seconds()
            if _vehicle_positions_last_load
            else None
        )
        _update_vehicle_feed_meta(
            source="stale_cache" if _vehicle_positions_cache else "unavailable",
            generated_at=_vehicle_positions_last_load,
            data_age_seconds=cache_age,
            last_error="request timeout",
            missing_coordinates=int(_vehicle_feed_meta.get("missing_coordinates", 0) or 0),
            vehicle_count=len(_vehicle_positions_cache or []),
        )
        return _vehicle_positions_cache or []
    except requests.exceptions.RequestException as e:
        logger.error(f"Error loading vehicle positions: {e}")
        cache_age = (
            (datetime.now() - _vehicle_positions_last_load).total_seconds()
            if _vehicle_positions_last_load
            else None
        )
        _update_vehicle_feed_meta(
            source="stale_cache" if _vehicle_positions_cache else "unavailable",
            generated_at=_vehicle_positions_last_load,
            data_age_seconds=cache_age,
            last_error=str(e),
            missing_coordinates=int(_vehicle_feed_meta.get("missing_coordinates", 0) or 0),
            vehicle_count=len(_vehicle_positions_cache or []),
        )
        return _vehicle_positions_cache or []
    except Exception as e:
        logger.error(f"Unexpected error loading vehicle positions: {e}")
        cache_age = (
            (datetime.now() - _vehicle_positions_last_load).total_seconds()
            if _vehicle_positions_last_load
            else None
        )
        _update_vehicle_feed_meta(
            source="stale_cache" if _vehicle_positions_cache else "unavailable",
            generated_at=_vehicle_positions_last_load,
            data_age_seconds=cache_age,
            last_error=str(e),
            missing_coordinates=int(_vehicle_feed_meta.get("missing_coordinates", 0) or 0),
            vehicle_count=len(_vehicle_positions_cache or []),
        )
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
    Finds direction-confirmed bus lines between origin and destination stops.

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

    origin_stop_id_set = {str(stop_id) for stop_id in origin_stop_ids}
    dest_stop_id_set = {str(stop_id) for stop_id in dest_stop_ids}

    # Get lines serving origin stops
    origin_lines = set()
    origin_stop_lines = {}
    for stop_id in origin_stop_ids:
        stop = stop_map.get(str(stop_id))
        if stop:
            for line in stop.get("lines", []):
                origin_lines.add(line)
                if line not in origin_stop_lines:
                    origin_stop_lines[line] = stop

    # Get lines serving destination stops
    dest_lines = set()
    dest_stop_lines = {}
    for stop_id in dest_stop_ids:
        stop = stop_map.get(str(stop_id))
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

    def _path_stop_id(path_item: Dict[str, Any]) -> str:
        """Return the stop id from a Carris Metropolitana pattern path item."""
        stop = path_item.get("stop") or {}
        return str(stop.get("id") or stop.get("stop_id") or "")

    def _pattern_stop_record(path_item: Dict[str, Any]) -> Dict[str, Any]:
        """Return a stop-like record from a pattern path item."""
        stop = path_item.get("stop") or {}
        return {
            "id": str(stop.get("id") or stop.get("stop_id") or ""),
            "name": stop.get("name") or stop.get("short_name") or stop.get("id") or "",
            "lat": stop.get("lat"),
            "lon": stop.get("lon"),
            "lines": stop.get("lines", []),
        }

    route_options: List[Dict[str, Any]] = []
    for line_id in sorted(common_lines):
        line_info = line_map.get(line_id, {})
        pattern_candidates: List[Dict[str, Any]] = []
        for pattern_id in line_info.get("patterns", []) or []:
            pattern_data = fetch_json_with_retry(f"{CARRIS_PATTERNS_URL}/{pattern_id}")
            if not isinstance(pattern_data, dict):
                continue

            indexed_path = [
                (_path_stop_id(path_item), index, path_item)
                for index, path_item in enumerate(pattern_data.get("path", []) or [])
            ]
            indexed_path = [(stop_id, index, item) for stop_id, index, item in indexed_path if stop_id]
            origin_matches = [
                (stop_id, index, item)
                for stop_id, index, item in indexed_path
                if stop_id in origin_stop_id_set
            ]
            dest_matches = [
                (stop_id, index, item)
                for stop_id, index, item in indexed_path
                if stop_id in dest_stop_id_set
            ]
            for origin_stop_id, origin_index, origin_item in origin_matches:
                for dest_stop_id, dest_index, dest_item in dest_matches:
                    if origin_index >= dest_index:
                        continue
                    origin_stop = stop_map.get(origin_stop_id) or _pattern_stop_record(origin_item)
                    dest_stop = stop_map.get(dest_stop_id) or _pattern_stop_record(dest_item)
                    pattern_candidates.append(
                        {
                            "line_id": line_id,
                            "short_name": line_info.get("short_name", line_id),
                            "long_name": line_info.get("long_name", ""),
                            "color": line_info.get("color", "#CCCCCC"),
                            "text_color": line_info.get("text_color", "#FFFFFF"),
                            "localities": line_info.get("localities", []),
                            "origin_stop": origin_stop,
                            "dest_stop": dest_stop,
                            "pattern_id": pattern_id,
                            "headsign": pattern_data.get("headsign") or "",
                            "stops_between": dest_index - origin_index,
                        }
                    )

        if pattern_candidates:
            route_options.append(
                min(
                    pattern_candidates,
                    key=lambda item: (
                        int(item.get("stops_between") or 999),
                        str(item.get("short_name") or ""),
                    ),
                )
            )
            continue

        route_option = {
            "line_id": line_id,
            "short_name": line_info.get("short_name", line_id),
            "long_name": line_info.get("long_name", ""),
            "color": line_info.get("color", "#CCCCCC"),
            "text_color": line_info.get("text_color", "#FFFFFF"),
            "localities": line_info.get("localities", []),
            "origin_stop": origin_stop_lines.get(line_id, {}),
            "dest_stop": dest_stop_lines.get(line_id, {}),
            "headsign": "",
            "stops_between": None,
            "direction_unconfirmed": True,
        }
        route_options.append(route_option)

    route_options.sort(
        key=lambda x: (
            1 if x.get("direction_unconfirmed") else 0,
            int(x.get("stops_between") or 999),
            x.get("short_name", ""),
        )
    )
    return route_options


def resolve_location(
    location_name: str, search_radius_km: float = 0.5, max_stops: int = 5
) -> Dict[str, Any]:
    """
    Intelligently resolves a location name to bus stops.

    Uses a multi-step approach:
    1. Try precise geocoding for the requested place/area.
    2. Find stops near the geocoded coordinates.
    3. Fall back to stop-name matches when geocoding is unavailable.

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
    broad_municipality_key = _bare_municipality_key(clean_name)

    # First collect stop-name matches, but do not let broad substring matches
    # beat geocoding. A query such as "Areeiro" should resolve to the Lisbon
    # area when geocoding can identify it, not to unrelated suburban street
    # stops whose names happen to contain the same word.
    # A bare municipality such as "Oeiras" or "Almada" is a broad area, not a
    # stop-name query. Searching stops by substring first can otherwise choose
    # unrelated homonyms such as "Av Conde Oeiras 1" and present them as the
    # requested destination. For exact municipality names, resolve the area
    # centre first and only use stop-name matches as a last diagnostic fallback.
    name_matches = [] if broad_municipality_key else find_stops_by_name(clean_name, max_results=max_stops)
    if name_matches:
        result["name_match_candidates"] = name_matches

    # Step 1/2: Try geocoding and nearby stops.
    geocoded = geocode_location(clean_name)
    if not geocoded and broad_municipality_key:
        fallback_lat, fallback_lon = CARRIS_METROPOLITANA_CITY_FALLBACKS[broad_municipality_key]
        geocoded = {
            "name": clean_name,
            "lat": fallback_lat,
            "lon": fallback_lon,
            "type": "municipality",
            "class": "boundary",
            "address": {"municipality": clean_name},
            "query_used": clean_name,
            "scope": "aml",
            "confidence": 0.75,
        }

    if geocoded:
        result["location"] = geocoded
        if broad_municipality_key:
            result["broad_area"] = True
            result["area_key"] = broad_municipality_key

        nearby_stops = find_stops_near_coordinates(
            geocoded["lat"],
            geocoded["lon"],
            radius_km=search_radius_km,
            max_results=max_stops,
        )

        if nearby_stops:
            result["method"] = "broad_area_geocoding" if broad_municipality_key else "geocoding"
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
                result["method"] = "broad_area_geocoding_expanded" if broad_municipality_key else "geocoding_expanded"
                result["stops"] = nearby_stops
                result["success"] = True
                logger.info(
                    f"Resolved '{clean_name}' via expanded geocoding ({len(nearby_stops)} stops)"
                )
                return result

    # Step 3: Fallback to name matches
    if broad_municipality_key:
        name_matches = find_stops_by_name(clean_name, max_results=max_stops)

    if name_matches:
        result["method"] = "name_match_fallback"
        result["stops"] = name_matches
        result["success"] = True
        logger.info(
            f"Resolved '{clean_name}' via fallback name match ({len(name_matches)} stops)"
        )
        return result

    if geocoded and is_within_lisbon_city(geocoded.get("lat"), geocoded.get("lon")):
        logger.info(
            "Resolved '%s' to a Lisbon-city POI but found no nearby Carris Metropolitana stops",
            clean_name,
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
        if location:
            matched_stops = find_stops_by_name(location, max_results=6)
            if matched_stops:
                stop_lines = sorted(
                    {line for stop in matched_stops for line in stop.get("lines", []) if line}
                )
                response = f"🚌 **Buses in {location.title()}**\n\n"
                response += (
                    f"ℹ️ **{location.title()} is a broad area, not an exact GPS point.** "
                    "Live vehicle data is temporarily unavailable, but I found Carris Metropolitana stops and lines matching the area.\n\n"
                )
                response += "🚏 **Matching stops:**\n"
                for stop in matched_stops[:5]:
                    response += f"- **{stop['name']}**"
                    if stop.get("lines"):
                        response += f" · Lines: {', '.join(stop.get('lines', [])[:6])}"
                    response += "\n"
                if stop_lines:
                    response += f"\n🚌 **Lines found in area:** {', '.join(stop_lines[:12])}"
                    if len(stop_lines) > 12:
                        response += f" ... and {len(stop_lines) - 12} more"
                    response += "\n"
                response += "\n💡 **Tip:** Send a street, stop name, neighbourhood, or GPS point for a precise live nearby-bus check.\n"
                return _append_carris_scope_footer(response, include_freshness=True)
        return _append_carris_scope_footer(
            "❌ Live Carris Metropolitana vehicle data is temporarily unavailable.",
            include_freshness=True,
        )

    line_map = {line_data["id"]: line_data for line_data in lines} if lines else {}

    # Filter vehicles
    filtered_vehicles = vehicles

    if line_id:
        filtered_vehicles = [v for v in vehicles if v.get("line_id") == line_id]
        if not filtered_vehicles:
            return f"❌ No active vehicles found for line {line_id}."

    if location:
        area_key = _normalize_area_key(location)
        matched_stops = find_stops_by_name(location, max_results=6)
        city_center = CARRIS_METROPOLITANA_CITY_FALLBACKS.get(area_key)

        geocoded = geocode_location(location)
        if city_center:
            geocoded = {"lat": city_center[0], "lon": city_center[1], "name": location}
        if not geocoded:
            if matched_stops:
                stop_lines = sorted(
                    {line for stop in matched_stops for line in stop.get("lines", []) if line}
                )
                response = f"🚌 **Buses in {location.title()}**\n\n"
                response += (
                    f"ℹ️ **{location.title()} is a broad area, not an exact GPS point.** "
                    "I found Carris Metropolitana stops and lines matching the area, but I need a street, stop, neighbourhood, or GPS point for live nearby buses.\n\n"
                )
                response += "🚏 **Matching stops:**\n"
                for stop in matched_stops[:5]:
                    response += f"- **{stop['name']}**"
                    if stop.get("lines"):
                        response += f" · Lines: {', '.join(stop.get('lines', [])[:6])}"
                    response += "\n"
                if stop_lines:
                    response += f"\n🚌 **Lines found in area:** {', '.join(stop_lines[:12])}"
                    if len(stop_lines) > 12:
                        response += f" ... and {len(stop_lines) - 12} more"
                    response += "\n"
                return _append_carris_scope_footer(response)
            return f"❌ Could not geocode location: {location}"

        nearby = get_vehicles_near_location(
            geocoded["lat"], geocoded["lon"], radius_km=radius_km, max_results=15
        )
        if not nearby:
            if matched_stops or city_center:
                response = f"🚌 **Buses near {location.title()}**\n\n"
                response += (
                    f"ℹ️ **{location.title()} is a broad area, not an exact stop.** "
                    f"I checked a central fallback point within {radius_km} km and did not find live buses with usable GPS right there.\n\n"
                )
                if matched_stops:
                    response += "🚏 **Area stop matches:**\n"
                    for stop in matched_stops[:5]:
                        response += f"- **{stop['name']}**"
                        if stop.get("lines"):
                            response += f" · Lines: {', '.join(stop.get('lines', [])[:6])}"
                        response += "\n"
                response += "\n💡 **Tip:** Send a street, stop name, neighbourhood, or GPS point for a precise live nearby-bus check.\n"
                return _append_carris_scope_footer(response)
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
        if CARRIS_METROPOLITANA_CITY_FALLBACKS.get(_normalize_area_key(location)):
            response += "ℹ️ Broad area fallback: using the municipality centre. Send a street, stop, neighbourhood, or GPS point for higher precision.\n"
    else:
        response = "🚌 Real-Time Bus Positions - All Lines\n"
        response += f"📊 {len(filtered_vehicles)} active vehicles\n"

    response += "=" * 50 + "\n\n"
    freshness_note = _build_vehicle_freshness_note(include_missing_coordinates=not line_id)
    if freshness_note:
        response += freshness_note + "\n\n"

    if not filtered_vehicles:
        response += "ℹ️ No vehicles currently active.\n"
        return _append_carris_scope_footer(response)

    # Status icons
    status_icons = {"INCOMING_AT": "🚏", "STOPPED_AT": "🛑", "IN_TRANSIT_TO": "🚌"}
    stale_vehicle_timestamps = 0

    # Show vehicles
    for _i, vehicle in enumerate(filtered_vehicles[:10], 1):
        v_line_id = vehicle.get("line_id") or "unknown line"
        line_info = line_map.get(v_line_id, {})
        line_short = line_info.get("short_name", v_line_id)

        status = vehicle.get("current_status", "UNKNOWN")
        status_icon = status_icons.get(status, "🚌")

        lat = vehicle.get("lat", 0)
        lon = vehicle.get("lon", 0)
        speed = vehicle.get("speed")
        bearing = vehicle.get("bearing", 0)
        license_plate = vehicle.get("license_plate") or "vehicle id unavailable"
        vehicle_model = vehicle.get("vehicle_model", "")
        door_status = vehicle.get("door_status", "")

        # Format timestamp
        timestamp = vehicle.get("timestamp", 0)
        time_str, timestamp_stale = _format_vehicle_relative_time(timestamp)
        if timestamp_stale:
            stale_vehicle_timestamps += 1

        response += f"- {status_icon} **Line {line_short}**\n"
        response += f"    - 🚗 **Vehicle:** {license_plate}"
        if vehicle_model:
            response += f" ({vehicle_model})"
        response += "\n"
        response += f"    - 📍 **Position:** ({lat:.5f}, {lon:.5f})\n"
        if speed is not None:
            response += f"    - 💨 **Speed:** {float(speed):.1f} km/h · **Bearing:** {bearing}°\n"
        if time_str:
            response += f"    - 📡 **Last update:** {time_str}\n"

        if status == "STOPPED_AT":
            response += "    - 🛑 **Currently stopped**"
            if door_status:
                response += f" (Doors: {door_status})"
            response += "\n"
        elif status == "INCOMING_AT":
            response += "    - 🚏 **Approaching next stop**\n"

        if "distance_km" in vehicle:
            response += (
                f"    - 📏 **Distance from search point:** {vehicle['distance_km']} km\n"
            )

        response += "\n"

    if len(filtered_vehicles) > 10:
        response += f"... and {len(filtered_vehicles) - 10} more vehicles.\n"

    if stale_vehicle_timestamps:
        response += (
            f"\n⚠️ Vehicle-level timestamp note: {stale_vehicle_timestamps} shown vehicle(s) reported stale timestamps in the upstream API, so exact per-vehicle age could not be trusted.\n"
        )

    return _append_carris_scope_footer(response)


def _normalize_alert_area(text: str) -> str:
    """Normalizes alert text for municipality/area filtering."""
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9\s-]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


@tool
def get_carris_metropolitana_alerts(
    area: Optional[str] = None,
    line: Optional[str] = None,
    language: str = "pt",
) -> str:
    """
    Gets current service alerts from Carris Metropolitana (suburban buses).

    Args:
        area: Optional municipality or area filter, such as "Almada".
        line: Optional Carris Metropolitana line filter, such as "3001".
        language: Output language, either "pt" or "en".

    Returns:
        str: Formatted list of active service alerts.
    """
    data = fetch_json_with_retry(CARRIS_ALERTS_URL)
    is_pt = (language or "pt").lower().startswith("pt")

    if not data:
        return "❌ Não foi possível obter alertas da Carris Metropolitana." if is_pt else "❌ Failed to fetch Carris Metropolitana alerts."

    # API returns a list directly, not a dict with 'entity' key
    alerts = data if isinstance(data, list) else data.get("entity", [])

    if not alerts:
        return "✅ Não há alertas ativos da Carris Metropolitana." if is_pt else "✅ No active alerts from Carris Metropolitana."

    line_filter = str(line or "").strip().upper()
    if not line_filter and area:
        area_line_match = re.search(r"\b(?:linha|line)?\s*(?P<line>\d{3,4}[A-Z]?)\b", str(area), flags=re.IGNORECASE)
        if area_line_match:
            line_filter = area_line_match.group("line").upper()
            area = None

    def alert_route_ids(alert_data: Dict[str, Any]) -> List[str]:
        route_ids: List[str] = []
        for entity in alert_data.get("informed_entity", []):
            route_id = str(entity.get("route_id", "") or "").strip()
            if not route_id:
                continue
            clean_route = route_id.split("_", 1)[0].upper()
            if clean_route and clean_route not in route_ids:
                route_ids.append(clean_route)
        header_text = alert_data.get("header_text", {})
        header = header_text.get("translation", [{}])[0].get("text", "")
        for route_id in re.findall(r"\b\d{3,4}[A-Z]?\b", header.upper()):
            if route_id not in route_ids:
                route_ids.append(route_id)
        return route_ids

    if line_filter:
        filtered_alerts = []
        for alert in alerts:
            alert_data = alert.get("alert", alert)
            route_ids = alert_route_ids(alert_data)
            desc_text = alert_data.get("description_text", {})
            desc = desc_text.get("translation", [{}])[0].get("text", "")
            if line_filter in route_ids or re.search(rf"\b{re.escape(line_filter)}\b", desc, flags=re.IGNORECASE):
                filtered_alerts.append(alert)
        alerts = filtered_alerts
        if not alerts:
            return (
                f"✅ Não encontrei alertas ativos da Carris Metropolitana para a linha {line_filter}."
                if is_pt
                else f"✅ No active Carris Metropolitana service alerts found for line {line_filter}."
            )

    if area:
        area_norm = _normalize_alert_area(area)
        filtered_alerts = []
        for alert in alerts:
            alert_data = alert.get("alert", alert)
            header_text = alert_data.get("header_text", {})
            header = header_text.get("translation", [{}])[0].get("text", "")
            desc_text = alert_data.get("description_text", {})
            desc = desc_text.get("translation", [{}])[0].get("text", "")
            combined = _normalize_alert_area(f"{header} {desc}")
            if area_norm and area_norm in combined:
                filtered_alerts.append(alert)
        alerts = filtered_alerts
        if not alerts:
            return (
                f"✅ Não encontrei alertas ativos da Carris Metropolitana para {area}."
                if is_pt
                else f"✅ No active Carris Metropolitana service alerts found for {area}."
            )

    if line_filter:
        scope = f" — linha {line_filter}" if is_pt else f" — line {line_filter}"
    else:
        scope = f" — {area}" if area else ""
    visible_alert_limit = 5
    visible_count = min(len(alerts), visible_alert_limit)
    if is_pt:
        total_label = "alerta ativo" if len(alerts) == 1 else "alertas ativos"
        visible_label = "esse alerta" if visible_count == 1 else f"os primeiros **{visible_count}**"
        response = f"### ⚠️ **Alertas ativos da Carris Metropolitana{scope}**\n\n"
        response += f"✅ **Resposta direta:** encontrei **{len(alerts)}** {total_label}; mostro {visible_label}.\n\n"
        response += "---\n\n"
    else:
        total_label = "active alert" if len(alerts) == 1 else "active alerts"
        visible_label = "that alert" if visible_count == 1 else f"the first **{visible_count}**"
        response = f"### ⚠️ **Carris Metropolitana service alerts{scope}**\n\n"
        response += f"✅ **Direct answer:** I found **{len(alerts)}** {total_label}; showing {visible_label}.\n\n"
        response += "---\n\n"

    for _i, alert in enumerate(alerts[:visible_alert_limit], 1):
        # Handle both old format (nested under 'alert') and new format (flat)
        alert_data = alert.get("alert", alert)

        # Header
        header_text = alert_data.get("header_text", {})
        header = header_text.get("translation", [{}])[0].get("text", "Sem título" if is_pt else "No title")
        header = re.sub(r"\s+\*\*\s*$", "", header).strip()

        # Description
        desc_text = alert_data.get("description_text", {})
        desc = desc_text.get("translation", [{}])[0].get("text", "Sem detalhes" if is_pt else "No details")

        # Time period
        active_period = alert_data.get("active_period", [{}])[0]
        start = format_timestamp(active_period.get("start", 0))
        end = format_timestamp(active_period.get("end", 0))

        # Affected routes
        route_ids = alert_route_ids(alert_data)
        if line_filter and line_filter not in route_ids:
            route_ids.append(line_filter)
        if line_filter and line_filter in route_ids:
            route_ids = [line_filter, *[route_id for route_id in route_ids if route_id != line_filter]]
        routes_str = ", ".join(route_ids[:8]) if route_ids else ("Todas as linhas" if is_pt else "All routes")

        response += f"- **⚠️ {header}**\n"
        response += f"  - 📝 {desc[:220]}{'...' if len(desc) > 220 else ''}\n"
        if line_filter:
            consulted_label = "Linha consultada" if is_pt else "Requested line"
            response += f"  - 🚌 **{consulted_label}:** {line_filter}\n"
            other_routes = [
                route_id for route_id in route_ids
                if route_id != line_filter
            ]
            if other_routes:
                other_label = "Outras linhas no mesmo alerta oficial" if is_pt else "Other lines in the same official alert"
                response += f"  - ℹ️ **{other_label}:** {', '.join(other_routes[:7])}\n"
        else:
            route_label = "Linhas afetadas" if is_pt else "Routes"
            response += f"  - 🚌 **{route_label}:** {routes_str}\n"
        period_label = "Período" if is_pt else "Period"
        if start != "N/A" or end != "N/A":
            response += f"  - ⏰ **{period_label}:** {start} - {end}\n"
        response += "\n"

    if len(alerts) > visible_alert_limit:
        if is_pt:
            response += f"- ℹ️ E mais {len(alerts) - visible_alert_limit} alertas.\n"
        else:
            response += f"- ℹ️ And {len(alerts) - visible_alert_limit} more alerts.\n"

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

    def _should_fallback_to_route_finder(text: str) -> bool:
        normalized = normalize(text)
        if not normalized:
            return False
        place_markers = (
            "museum",
            "museu",
            "restaurant",
            "restaurante",
            "hotel",
            "hospital",
            "pharmacy",
            "farmacia",
            "farmácia",
            "tower",
            "torre",
            "church",
            "igreja",
            "monument",
            "monumento",
            "fundacao",
            "fundação",
            "avenida",
            "rua",
            "praca",
            "praça",
            "campus",
            "airport",
            "aeroporto",
            "city centre",
            "city center",
            "centre of lisbon",
            "center of lisbon",
            "centro de lisboa",
            "centro da cidade",
        )
        return any(marker in normalized for marker in place_markers)

    direct_lines = []
    broad_lines = []

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

        origin_precise = origin_norm in long_name_norm or any(origin_norm in loc for loc in localities_norm)
        dest_precise = dest_norm in long_name_norm or any(dest_norm in loc for loc in localities_norm)
        origin_broad = origin_precise or any(origin_norm in m for m in muni_norm)
        dest_broad = dest_precise or any(dest_norm in m for m in muni_norm)

        line_record = {
            "id": line_id,
            "short_name": short_name,
            "long_name": long_name,
            "localities": [loc for loc in localities if loc],
            "municipalities": [m for m in municipalities if m],
        }
        if origin_precise and dest_precise:
            direct_lines.append(line_record)
        elif origin_broad and dest_broad:
            broad_lines.append(line_record)

    if not direct_lines and broad_lines:
        response = f"🚌 **Broad Carris Metropolitana candidates: {origin.title()} → {destination.title()}**\n\n"
        response += "ℹ️ I found lines serving both requested municipalities/areas, but not enough stop-level evidence to call them direct door-to-door options. Use a specific stop or street for a precise board/alight route.\n\n"
        for line in broad_lines[:5]:
            response += f"- 🚍 **Line {line['short_name']}**\n"
            if line["long_name"]:
                response += f"    - 📍 **Terminals:** {line['long_name'].replace(' - ', ' ↔ ')}\n"
            municipalities = _clean_carris_display_list(line.get("municipalities") or [], max_items=4, drop_numeric_only=True)
            if municipalities:
                response += f"    - 🏘️ **Municipalities:** {', '.join(municipalities)}\n"
        if len(broad_lines) > 5:
            response += f"... and {len(broad_lines) - 5} more broad matches.\n"
        return _append_carris_scope_footer(response)

    if not direct_lines:
        if _should_fallback_to_route_finder(origin) or _should_fallback_to_route_finder(destination):
            search_radius_km = 0.8 if any(
                marker in origin_norm or marker in dest_norm
                for marker in (
                    "city centre",
                    "city center",
                    "centre of lisbon",
                    "center of lisbon",
                    "centro de lisboa",
                    "centro da cidade",
                )
            ) else 0.5
            return str(
                find_bus_routes.invoke(
                    {
                        "origin": origin,
                        "destination": destination,
                        "search_radius_km": search_radius_km,
                    }
                )
            )

        response = f"❌ **Sem linhas diretas entre '{origin}' e '{destination}'**\n\n"
        response += "💡 **Sugestões:**\n"
        response += "   • Pode ser necessário fazer transbordo\n"
        response += "   • Considere combinar Metro + Autocarro\n"
        response += "   • Consulte carrismetropolitana.pt para mais opções\n"
        return _append_carris_scope_footer(response)

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
    response = f"🚌 **Autocarros: {origin.title()} → {destination.title()}**\n\n"

    # Show alerts first if any
    if relevant_alerts:
        response += "⚠️ **Alertas de serviço:**\n"
        for alert in relevant_alerts[:3]:
            response += f"- {alert['header']}\n"
        response += "\n"

    response += f"✅ **{len(direct_lines)} direct line(s) found**\n\n"

    for line in direct_lines[:5]:
        short_name = line["short_name"]
        long_name = line["long_name"]
        localities = line["localities"]
        line_id = line["id"]

        # Check if this line has alerts
        has_alert = line_id in affected_lines or short_name in affected_lines
        alert_icon = " ⚠️" if has_alert else ""

        response += f"- 🚍 **Linha {short_name}**{alert_icon}\n"

        # Show the official route terminals (not direction, just terminals served)
        if long_name:
            # Replace " - " with " ↔ " to show it's bidirectional
            display_name = long_name.replace(" - ", " ↔ ")
            response += f"    - 📍 **Terminals:** {display_name}\n"

        # Show localities if available, ordered from origin to destination when possible
        if localities:
            cleaned_localities = _clean_carris_display_list(localities, max_items=6)
            origin_idx = -1
            dest_idx = -1
            for idx, loc in enumerate(cleaned_localities):
                loc_norm = normalize(loc)
                if origin_idx < 0 and (
                    origin_norm in loc_norm or loc_norm in origin_norm
                ):
                    origin_idx = idx
                if dest_idx < 0 and (dest_norm in loc_norm or loc_norm in dest_norm):
                    dest_idx = idx

            if origin_idx >= 0 and dest_idx >= 0:
                if origin_idx < dest_idx:
                    key_stops = cleaned_localities[origin_idx : dest_idx + 1]
                else:
                    key_stops = cleaned_localities[dest_idx : origin_idx + 1][::-1]

                if len(key_stops) > 6:
                    display_stops = key_stops[:6]
                    response += f"    - 🚏 **Path:** {' → '.join(display_stops)} (+{len(key_stops) - 6})\n"
                else:
                    response += f"    - 🚏 **Path:** {' → '.join(key_stops)}\n"
            else:
                key_stops = cleaned_localities[:6]
                response += f"    - 🚏 **Passes through:** {', '.join(key_stops)}"
                if len(cleaned_localities) > 6:
                    response += f" (+{len(cleaned_localities) - 6})"
                response += "\n"
        response += "\n"

    if len(direct_lines) > 5:
        other_lines = [line_data["short_name"] for line_data in direct_lines[5:]]
        response += f"... and {len(other_lines)} more direct lines: {', '.join(other_lines[:10])}\n\n"

    response += "💡 **How to use it:** check the direction shown at the stop before boarding.\n"

    return _append_carris_scope_footer(response)


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

    scored_matches = []

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

        score = 0
        if query_lower == short_name.lower() or query_lower == line_id.lower():
            score = 100
        elif query_normalized in long_name_norm:
            score = 80
        elif any(query_normalized == _normalize_alert_area(loc) for loc in localities if loc):
            score = 60
        elif query_normalized in localities_norm:
            score = 45
        elif query_lower in muni_str:
            score = 25
        if score:
            scored_matches.append((score, line))

    scored_matches.sort(
        key=lambda item: (
            -item[0],
            str(item[1].get("short_name") or item[1].get("id") or ""),
        )
    )
    matches = [line for _, line in scored_matches]

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
        f"### 🚌 **Carris Metropolitana lines serving '{query}'** ({min(len(matches), 5)} shown of {len(matches)})\n"
    )

    for line in matches[:5]:
        short_name = line.get("short_name", "N/A")
        long_name = line.get("long_name") or "N/A"
        municipalities = _clean_carris_display_list(
            line.get("municipalities") or [],
            max_items=4,
            drop_numeric_only=True,
        )
        localities = _clean_carris_display_list(
            line.get("localities") or [],
            max_items=8,
            drop_numeric_only=True,
        )

        response += f"**🚌 Line {short_name}**\n"
        response += f"    - 📍 {long_name}\n"
        if municipalities:
            response += f"    - 🏘️ **Municipalities:** {', '.join(municipalities)}\n"
        if localities:
            response += f"    - 📌 **Localities:** {', '.join(localities)}\n"
        response += "\n"

    if len(matches) > 5:
        response += f"... and {len(matches) - 5} more lines.\n"

    response += "\n💡 Ask for a direct route between two stops/places if you need board and alight points.\n"

    return _append_carris_scope_footer(response)


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
    data = load_carris_metropolitana_vehicles()

    if not data:
        return _append_carris_scope_footer(
            "❌ Live Carris Metropolitana bus locations are temporarily unavailable.",
            include_freshness=True,
        )

    normalized_line_id = str(line_id or "").strip().upper() or None

    if normalized_line_id:
        filtered = [v for v in data if str(v.get("line_id") or "").upper() == normalized_line_id]
        if not filtered:
            return _append_carris_scope_footer(
                f"ℹ️ No active buses found on line {normalized_line_id} at this time.\n\n"
                f"💡 The line may not be operating right now.",
            )
        buses = filtered
    else:
        buses = data

    lines_data = load_carris_metropolitana_lines()
    stops_data = load_carris_metropolitana_stops()
    line_map = {
        str(line.get("id") or line.get("short_name") or "").upper(): line
        for line in lines_data
    }
    stop_map = {str(stop.get("id") or ""): stop for stop in stops_data}
    freshness_note = _build_vehicle_freshness_note(include_missing_coordinates=not normalized_line_id)
    updated_at = datetime.now().strftime("%H:%M")

    if normalized_line_id:
        line_info = line_map.get(normalized_line_id, {})
        route_name = line_info.get("long_name") or "route terminals not available in the live feed"
        response = f"### 🚌 **Carris Metropolitana Line {normalized_line_id} - Live Buses**\n\n"
        response += f"**Short answer:** I found **{len(buses)} active bus{'es' if len(buses) != 1 else ''}** currently reported on line **{normalized_line_id}**.\n\n"
        response += "**Current snapshot**\n"
        response += f"    - 🧭 **Route:** {route_name}\n"
        response += f"    - 📊 **Active buses:** {len(buses)}\n"
        response += f"    - 🕐 **Updated:** {updated_at}\n"
        if freshness_note:
            response += f"    - {freshness_note.splitlines()[0]}\n"
        response += "\n**Live vehicles**\n"

        for bus in buses[:5]:
            lat = bus.get("lat", 0)
            lon = bus.get("lon", 0)
            speed = bus.get("speed", 0)
            bearing = bus.get("bearing", 0)
            status = bus.get("current_status", "UNKNOWN")
            stop_id = str(bus.get("stop_id") or "").strip()
            stop_info = stop_map.get(stop_id, {}) if stop_id else {}
            stop_name = stop_info.get("name") or "not reported by the live feed"

            status_emoji = {
                "IN_TRANSIT_TO": "🚌➡️",
                "STOPPED_AT": "🚏",
                "INCOMING_AT": "📍",
            }.get(status, "🚌")

            directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            dir_idx = int((bearing + 22.5) % 360 / 45)
            direction = directions[dir_idx] if bearing else "?"
            maps_link = f"https://www.google.com/maps/search/?api=1&query={lat:.5f}%2C{lon:.5f}"
            vehicle_id = str(bus.get("id", "N/A"))[:20]

            response += f"- **🚌 Bus {vehicle_id}**\n"
            response += f"    - {status_emoji} **Status:** {status.replace('_', ' ').title()}\n"
            response += f"    - 📍 **Live position:** [Open map]({maps_link})\n"
            response += f"    - 🧭 **Direction:** {direction} · **Speed:** {speed:.1f} km/h\n"
            response += f"    - 🚏 **Next stop:** {stop_name}\n"
            response += "\n"

        if len(buses) > 5:
            response += f"... and {len(buses) - 5} more buses on this line.\n\n"

        response += "💡 **Tip:** If you tell me your exact stop, I can narrow this to the most relevant vehicle and direction.\n"
    else:
        from collections import Counter

        line_counts = Counter(v.get("line_id", "Unknown") for v in buses)

        response = "### 🚌 **Carris Metropolitana Live Buses**\n\n"
        response += f"    - 📊 **Active buses:** {len(buses)}\n"
        response += f"    - 🕐 **Updated:** {updated_at}\n"
        if freshness_note:
            response += f"    - {freshness_note.splitlines()[0]}\n"
        response += "\n**Top lines by active buses**\n"
        for line, count in line_counts.most_common(15):
            response += f"- 🚌 **Line {line}:** {count} buses\n"

        response += f"\n... {len(line_counts)} lines total with active buses.\n"
        response += "\n💡 **Tip:** Ask for a line number, stop, street, or GPS point for a narrower live view.\n"

    return _append_carris_scope_footer(response)


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
    response += f"📍 {line_info.get('long_name') or 'route details unavailable in the live feed'}\n"
    response += "=" * 50 + "\n\n"

    pattern_id = patterns[0]
    pattern_url = f"{CARRIS_PATTERNS_URL}/{pattern_id}"
    pattern_data = fetch_json_with_retry(pattern_url)

    if not pattern_data:
        return f"❌ Failed to fetch schedule for pattern {pattern_id}."

    headsign = pattern_data.get("headsign") or "direction unavailable"
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
                first_time = schedule[0].get("arrival_time")
                if first_time:
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
                        time_at = schedule[stop_idx].get("arrival_time")
                        if time_at:
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

    return _append_carris_scope_footer(response)


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
    header = f"### 🚌 **Carris Metropolitana: {origin} → {destination}**\n\n"
    response = header
    ambiguity_note = build_location_ambiguity_preamble(origin, destination, language="pt")
    if ambiguity_note:
        response += f"{ambiguity_note}\n\n"

    if origin_lat is not None and origin_lon is not None:
        origin_stops = find_stops_near_coordinates(
            origin_lat, origin_lon, radius_km=search_radius_km, max_results=10
        )
        if origin_stops:
            origin_resolved = {"stops": origin_stops, "method": "gps_provided"}
        else:
            origin_resolved = {"stops": [], "method": "gps_provided", "success": False}
    else:
        origin_resolved = resolve_location(origin, search_radius_km, max_stops=10)

    origin_stops = origin_resolved.get("stops", [])
    if origin_resolved.get("broad_area"):
        response += (
            f"ℹ️ **{origin}** é uma área/município amplo. Usei paragens perto "
            "do centro de referência; indica uma morada, estação ou paragem se "
            "quiseres uma rota porta-a-porta.\n\n"
        )

    if not origin_stops:
        origin_loc = origin_resolved.get("location")
        if origin_loc and is_within_lisbon_city(
            origin_loc.get("lat"), origin_loc.get("lon")
        ):
            response += (
                f"ℹ️ **{origin}** appears to be inside Lisbon city, where this trip may be better served by **Carris** / Carris Urban (carris.pt) instead of Carris Metropolitana.\n"
            )
        else:
            response += f"❌ **No Carris Metropolitana stops found near {origin}.**\n"
            response += "\n💡 **Tip:** try a more specific street, stop, neighbourhood, or GPS point.\n"
        return response

    if dest_lat is not None and dest_lon is not None:
        dest_stops = find_stops_near_coordinates(
            dest_lat, dest_lon, radius_km=search_radius_km, max_results=10
        )
        if dest_stops:
            dest_resolved = {"stops": dest_stops, "method": "gps_provided"}
        else:
            dest_resolved = {"stops": [], "method": "gps_provided", "success": False}
    else:
        dest_resolved = resolve_location(destination, search_radius_km, max_stops=10)

    dest_stops = dest_resolved.get("stops", [])
    if dest_resolved.get("broad_area"):
        response += (
            f"ℹ️ **{destination}** é uma área/município amplo. Usei paragens perto "
            "do centro de referência; indica uma morada, estação ou paragem se "
            "quiseres uma rota porta-a-porta.\n\n"
        )

    if not dest_stops:
        dest_loc = dest_resolved.get("location")
        if dest_loc and is_within_lisbon_city(dest_loc.get("lat"), dest_loc.get("lon")):
            response += (
                f"ℹ️ **{destination}** appears to be inside Lisbon city, where this trip may be better served by **Carris** / Carris Urban (carris.pt) instead of Carris Metropolitana.\n"
            )
        else:
            response += f"❌ **No Carris Metropolitana stops found near {destination}.**\n"
            response += "\n💡 **Tip:** try a more specific name, address, stop, or GPS point.\n"
        return response

    origin_stop_ids = [s["id"] for s in origin_stops]
    dest_stop_ids = [s["id"] for s in dest_stops]

    route_options = find_common_routes(origin_stop_ids, dest_stop_ids)
    fallback_resolution_note = ""
    if not route_options:
        fallback_sets = []
        origin_name_matches = origin_resolved.get("name_match_candidates") or []
        dest_name_matches = dest_resolved.get("name_match_candidates") or []
        if dest_name_matches:
            fallback_sets.append((origin_stops, dest_name_matches, destination))
        if origin_name_matches:
            fallback_sets.append((origin_name_matches, dest_stops, origin))
        if origin_name_matches and dest_name_matches:
            fallback_sets.append((origin_name_matches, dest_name_matches, f"{origin} / {destination}"))

        seen_fallbacks = set()
        for candidate_origin_stops, candidate_dest_stops, fallback_label in fallback_sets:
            origin_ids = tuple(str(s.get("id") or "") for s in candidate_origin_stops if s.get("id"))
            dest_ids = tuple(str(s.get("id") or "") for s in candidate_dest_stops if s.get("id"))
            key = (origin_ids, dest_ids)
            if not origin_ids or not dest_ids or key in seen_fallbacks:
                continue
            seen_fallbacks.add(key)
            candidate_options = find_common_routes(list(origin_ids), list(dest_ids))
            if candidate_options:
                route_options = candidate_options
                fallback_resolution_note = (
                    f"ℹ️ Usei paragens cujo nome corresponde a **{fallback_label}** "
                    "porque a geocodificação inicial não produziu uma ligação direta confirmada.\n\n"
                )
                break

    if route_options:
        if fallback_resolution_note:
            response += fallback_resolution_note

        def _coord_suffix(stop: Dict[str, Any]) -> str:
            """Return an internal coordinate suffix for downstream linkification."""
            lat = stop.get("lat")
            lon = stop.get("lon")
            if lat is None or lon is None:
                return ""
            try:
                return f" | coords: {float(lat):.6f},{float(lon):.6f}"
            except (TypeError, ValueError):
                return ""

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
            grouped_routes[key]["lines"].append(
                {
                    "short_name": route.get("short_name", "?"),
                    "headsign": route.get("headsign", ""),
                    "stops_between": route.get("stops_between"),
                    "direction_unconfirmed": bool(route.get("direction_unconfirmed")),
                }
            )

        response += f"✅ **{len(grouped_routes)} direct route option(s) found** ({len(route_options)} line match(es)).\n\n"

        for i, ((origin_name, dest_name), group) in enumerate(
            list(grouped_routes.items())[:5], 1
        ):
            line_labels = []
            has_unconfirmed_direction = False
            stop_counts = []
            for line in group["lines"]:
                short_name = str(line.get("short_name") or "?")
                headsign = str(line.get("headsign") or "").strip()
                if headsign:
                    line_labels.append(f"{short_name} → {headsign}")
                else:
                    line_labels.append(short_name)
                if line.get("direction_unconfirmed"):
                    has_unconfirmed_direction = True
                if line.get("stops_between") is not None:
                    stop_counts.append(int(line["stops_between"]))
            lines_str = ", ".join(sorted(set(line_labels)))

            response += f"- 🚌 **Option {i}**\n"
            response += f"    - 🚏 **Board at:** {origin_name}{_coord_suffix(group['origin_stop'])}\n"
            response += f"    - 🚏 **Alight at:** {dest_name}{_coord_suffix(group['dest_stop'])}\n"
            response += f"    - 🚍 **Lines:** {lines_str}\n"
            if stop_counts:
                response += f"    - 🧭 **Confirmed direction:** origin stop appears before destination stop in the official pattern ({min(stop_counts)} stops on-board).\n"
            if has_unconfirmed_direction:
                response += "    - ⚠️ **Direction:** line overlap found, but the official stop order could not be confirmed from patterns.\n"
            response += "\n"

        if len(grouped_routes) > 5:
            response += f"... and {len(grouped_routes) - 5} more route options available.\n\n"

        if any(route.get("direction_unconfirmed") for route in route_options):
            response += "⚠️ **Note:** verify that the bus runs in your intended direction on carrismetropolitana.pt for options marked as unconfirmed.\n\n"
    else:
        response += "❌ **Sem rota direta de autocarro confirmada**\n\n"
        response += (
            "ℹ️ **Limitação:** não encontrei uma ligação direta de autocarro "
            "entre estes pontos na Carris Metropolitana. "
            "Se o destino for uma zona ampla, indica uma paragem, estação ou "
            "morada mais precisa para procurar o próximo autocarro concreto.\n"
        )
        return response

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
        response += "\n"
        response += CARRIS_LIMITATION_NOTICE
        response += "\n"

    response += "\n💡 **Tips:**\n"
    response += "    - Check carrismetropolitana.pt or carris.pt for detailed schedules.\n"
    response += "    - Metro may be faster for longer trips.\n"

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
    # print(result[:500])
    print(result)

    print("\n2. Testing search_carris_metropolitana_lines...")
    result = search_carris_metropolitana_lines.invoke({"query": "Sintra"})
    # print(result[:500])
    print(result)

    print("\n3. Testing geocode_location...")
    loc = geocode_location("Colombo")
    if loc:
        print(f"   ✅ Colombo: ({loc['lat']:.4f}, {loc['lon']:.4f})")

    print("\n4. Testing find_bus_routes...")
    result = find_bus_routes.invoke({"origin": "Colombo", "destination": "Oriente"})
    # print(result[:800])
    print(result)

    print("\n5. Testing get_carris_metropolitana_stop_info...")
    result = get_carris_metropolitana_stop_info.invoke({"stop_id": "020037"})
    # print(result[:500])
    print(result)

    print("\n6. Testing find_direct_bus_lines...")
    result = find_direct_bus_lines.invoke({"origin": "Oeiras", "destination": "Amadora"})
    # print(result[:800])
    print(result)

    print("\n7. Testing get_real_time_bus_positions near Almada...")
    result = get_real_time_bus_positions.invoke({"location": "Almada", "radius_km": 1.0})
    # print(result[:800])
    print(result)

    print("\n8. Testing get_bus_realtime_locations for line 3001...")
    result = get_bus_realtime_locations.invoke({"line_id": "3001"})
    # print(result[:800])
    print(result)

    print("\n9. Testing get_bus_next_departures for line 3001 at stop 020037...")
    result = get_bus_next_departures.invoke({"line_id": "3001", "stop_id": "020037"})
    # print(result[:800])
    print(result)

    print("\n\033[1;32m✅ Carris Metropolitana API tests complete!\033[0m")
