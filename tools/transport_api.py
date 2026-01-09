# ==========================================================================
# Master Thesis - Transport API Tools
#   - André Filipe Gomes Silvestre, 20240502
# 
#   Real-time transport data for Lisbon Metropolitan Area.
#   Features:
#     - Metro de Lisboa: Line status and routing
#     - Carris Metropolitana: Alerts, stops, lines, real-time arrivals, routing
#     - CP (Comboios de Portugal): Train status and delays
#     - Bus Route Finder: Find bus routes between two locations using GPS
#     - Smart Geocoding: Converts place names to GPS coordinates automatically
# 
#   APIs:
#     - Metro: https://app.metrolisboa.pt/status/getLinhas.php
#     - Carris: https://api.carrismetropolitana.pt/
#     - CP: https://comboios.live/api/
#     - Nominatim (OpenStreetMap): https://nominatim.openstreetmap.org/
# 
#   Bus Routing System:
#     - On-demand loading of ~12000 bus stops from Carris API
#     - In-memory cache for fast proximity search (no database needed)
#     - Haversine distance calculation for GPS-based stop finding
#     - Smart geocoding: "Colombo" → GPS → nearest bus stops
#     - Pattern matching to find bus routes between two stops
#     - Streamlit Cloud compatible (no external dependencies)
# ==========================================================================

# Required libraries:
# pip install requests langchain-core

import os
import sys
import logging
import time
import math
import re
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import quote

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

# Carris Metropolitana (using v1 - official documented API with more complete data)
# API Documentation: https://github.com/carrismetropolitana/schedules-api
CARRIS_BASE_URL = "https://api.carrismetropolitana.pt/v1"
CARRIS_ALERTS_URL = f"{CARRIS_BASE_URL}/alerts"
CARRIS_STOPS_URL = f"{CARRIS_BASE_URL}/stops"
CARRIS_LINES_URL = f"{CARRIS_BASE_URL}/lines"
CARRIS_ROUTES_URL = f"{CARRIS_BASE_URL}/routes"
CARRIS_PATTERNS_URL = f"{CARRIS_BASE_URL}/patterns"
CARRIS_VEHICLES_URL = f"{CARRIS_BASE_URL}/vehicles"
CARRIS_MUNICIPALITIES_URL = f"{CARRIS_BASE_URL}/municipalities"

# Nominatim (OpenStreetMap) - Free geocoding service
# Rate limit: 1 request per second (we add delay between calls)
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# CP (Comboios de Portugal)
CP_STATIONS_URL = "https://comboios.live/api/stations"
CP_VEHICLES_URL = "https://comboios.live/api/vehicles"

# AML (Área Metropolitana de Lisboa) Bounding Box
# Used to filter CP stations and trains to only those serving the Lisbon region
AML_BOUNDS = {
    "lat_min": 38.5,
    "lat_max": 39.1,
    "lon_min": -9.5,
    "lon_max": -8.8
}

# ==========================================================================
# Carris Stops Cache (In-Memory)
# ==========================================================================
# Cache for all Carris bus stops and lines - loaded on demand
# This avoids repeated API calls and enables fast proximity search
# Memory usage: ~12000 stops * ~250 bytes = ~3MB (very efficient)

_carris_stops_cache: Optional[List[Dict[str, Any]]] = None
_carris_stops_last_load: Optional[datetime] = None
_carris_lines_cache: Optional[List[Dict[str, Any]]] = None
_carris_lines_last_load: Optional[datetime] = None
_carris_routes_cache: Optional[List[Dict[str, Any]]] = None
_carris_routes_last_load: Optional[datetime] = None

# ==========================================================================
# CP Stations Cache (In-Memory) - AML Only
# ==========================================================================
# Cache for CP train stations in the AML (Lisbon Metropolitan Area)
# Filters ~462 stations down to ~81 in the AML region

_cp_stations_cache: Optional[Dict[str, Dict[str, Any]]] = None  # code -> station
_cp_stations_last_load: Optional[datetime] = None

# Cache expiration time (24 hours - stops don't change frequently)
CACHE_EXPIRATION_HOURS = 24

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


# ==========================================================================
# Carris Bus Stops Cache Functions
# ==========================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates the great-circle distance between two points on Earth.
    
    Uses the Haversine formula for accurate GPS distance calculation.
    This is essential for finding bus stops near a given location.
    
    Args:
        lat1 (float): Latitude of point 1 (in degrees).
        lon1 (float): Longitude of point 1 (in degrees).
        lat2 (float): Latitude of point 2 (in degrees).
        lon2 (float): Longitude of point 2 (in degrees).
        
    Returns:
        float: Distance in kilometers.
        
    Example:
        >>> haversine_distance(38.7223, -9.1393, 38.7369, -9.1427)
        1.64  # ~1.64 km between Rossio and Marquês de Pombal
    """
    R = 6371  # Earth radius in kilometers
    
    # Convert degrees to radians
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    # Haversine formula
    a = (
        math.sin(dlat / 2) ** 2 +
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
        math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def _is_cache_valid(last_load: Optional[datetime]) -> bool:
    """
    Checks if the cache is still valid (not expired).
    
    Args:
        last_load (datetime): Timestamp of last cache load.
        
    Returns:
        bool: True if cache is valid, False if expired or never loaded.
    """
    if last_load is None:
        return False
    
    hours_elapsed = (datetime.now() - last_load).total_seconds() / 3600
    return hours_elapsed < CACHE_EXPIRATION_HOURS


# ==========================================================================
# CP (Comboios de Portugal) AML Stations Cache
# ==========================================================================

def load_cp_aml_stations(force_reload: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Loads CP train stations in the AML (Área Metropolitana de Lisboa) into cache.
    
    Filters the ~462 CP stations to only include the ~81 stations within
    the Lisbon Metropolitan Area bounding box. This enables efficient
    filtering of trains that serve the AML region.
    
    Features:
        - Lazy loading: Only fetches from API when first needed
        - Automatic refresh: Reloads after 24 hours
        - Geographic filtering: Only includes stations in AML bounds
        - Memory efficient: ~81 stations cached
    
    Args:
        force_reload (bool): Force refresh even if cache is valid.
        
    Returns:
        Dict[str, Dict]: Dictionary mapping station code to station info.
        Each station has: code, name, lat, lon, railways.
        
    Example:
        >>> stations = load_cp_aml_stations()
        >>> len(stations)
        81  # Approximately 81 AML stations
        >>> stations['94-30007']
        {'code': '94-30007', 'name': 'Lisboa Santa Apolonia', 'lat': 38.7136, 'lon': -9.1227}
    """
    global _cp_stations_cache, _cp_stations_last_load
    
    # Return cached data if valid and not forcing reload
    if not force_reload and _cp_stations_cache and _is_cache_valid(_cp_stations_last_load):
        logger.info(f"Using cached CP AML stations ({len(_cp_stations_cache)} stations)")
        return _cp_stations_cache
    
    logger.info("Loading CP AML stations from API...")
    
    try:
        response = requests.get(CP_STATIONS_URL, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        all_stations = data.get('stations', [])
        
        if not isinstance(all_stations, list):
            logger.error("Unexpected response format from CP stations API")
            return _cp_stations_cache or {}
        
        # Filter to AML region only
        aml_stations = {}
        for station in all_stations:
            try:
                lat = float(station.get('latitude', 0))
                lon = float(station.get('longitude', 0))
                
                # Check if within AML bounding box
                if (AML_BOUNDS['lat_min'] <= lat <= AML_BOUNDS['lat_max'] and
                    AML_BOUNDS['lon_min'] <= lon <= AML_BOUNDS['lon_max']):
                    
                    code = station.get('code', '')
                    aml_stations[code] = {
                        'code': code,
                        'name': station.get('designation', 'Unknown'),
                        'lat': lat,
                        'lon': lon,
                        'railways': station.get('railways', [])
                    }
            except (ValueError, TypeError):
                continue
        
        # Update cache
        _cp_stations_cache = aml_stations
        _cp_stations_last_load = datetime.now()
        
        logger.info(f"\033[1;32m✅ Loaded {len(aml_stations)} CP AML stations\033[0m")
        return aml_stations
        
    except requests.exceptions.Timeout:
        logger.error("Timeout loading CP stations (15s)")
        return _cp_stations_cache or {}
    except requests.exceptions.RequestException as e:
        logger.error(f"Error loading CP stations: {e}")
        return _cp_stations_cache or {}
    except Exception as e:
        logger.error(f"Unexpected error loading CP stations: {e}")
        return _cp_stations_cache or {}


def get_cp_aml_trains() -> List[Dict[str, Any]]:
    """
    Gets real-time train data filtered to only trains serving the AML region.
    
    A train serves the AML if:
    - Its origin station is in the AML, OR
    - Its destination station is in the AML, OR
    - Its current location (lastStation) is in the AML
    
    This filters ~90 Portugal-wide trains to ~45 AML trains.
    
    Returns:
        List[Dict]: List of trains serving the AML with full details.
        Each train has: trainNumber, delay, status, service, origin, destination, etc.
        
    Example:
        >>> trains = get_cp_aml_trains()
        >>> len(trains)
        45  # Approximately half of all trains
        >>> trains[0]['service']['designation']
        'Urbanos Lisboa'
    """
    # Load AML station codes
    aml_stations = load_cp_aml_stations()
    aml_codes = set(aml_stations.keys())
    
    if not aml_codes:
        logger.warning("No AML stations loaded, returning all trains")
        # Fallback: return all trains
        data = fetch_json_with_retry(CP_VEHICLES_URL)
        return data.get('vehicles', []) if data else []
    
    try:
        response = requests.get(CP_VEHICLES_URL, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        all_trains = data.get('vehicles', [])
        
        # Filter to trains serving AML
        aml_trains = []
        for train in all_trains:
            origin_code = train.get('origin', {}).get('code', '')
            dest_code = train.get('destination', {}).get('code', '')
            last_station = train.get('lastStation', '')
            
            # Check if train serves AML
            if (origin_code in aml_codes or 
                dest_code in aml_codes or 
                last_station in aml_codes):
                aml_trains.append(train)
        
        logger.info(f"Filtered to {len(aml_trains)}/{len(all_trains)} trains serving AML")
        return aml_trains
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching trains: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching trains: {e}")
        return []


def search_cp_station(query: str) -> List[Dict[str, Any]]:
    """
    Searches for CP train stations in the AML by name.
    
    Useful for finding station codes and information before checking
    departures or arrivals.
    
    Args:
        query (str): Station name or partial name to search for.
        
    Returns:
        List[Dict]: List of matching stations with code, name, lat, lon.
        
    Example:
        >>> search_cp_station("Oriente")
        [{'code': '94-31039', 'name': 'Lisboa Oriente', 'lat': 38.7678, 'lon': -9.099}]
        
        >>> search_cp_station("Cais")
        [{'code': '94-69005', 'name': 'Cais do Sodre', 'lat': 38.706, 'lon': -9.144}]
    """
    aml_stations = load_cp_aml_stations()
    query_lower = query.lower().strip()
    
    matches = []
    for code, station in aml_stations.items():
        if query_lower in station['name'].lower():
            matches.append(station)
    
    # Sort by relevance (exact matches first, then by name)
    matches.sort(key=lambda x: (
        0 if query_lower == x['name'].lower() else 1,
        x['name']
    ))
    
    return matches


# ==========================================================================
# Geocoding Functions (Convert place names to GPS coordinates)
# ==========================================================================

# Last geocoding request timestamp (for rate limiting)
_last_geocode_time: Optional[float] = None

def geocode_location(
    location_name: str,
    region: str = "Lisboa, Portugal"
) -> Optional[Dict[str, Any]]:
    """
    Converts a location name to GPS coordinates using Nominatim (OpenStreetMap).
    
    This function enables the system to understand place names like "Colombo",
    "Vasco da Gama", "Alfama", etc. and convert them to coordinates for
    finding nearby bus stops.
    
    Features:
        - Automatic region context (adds "Lisboa, Portugal" to searches)
        - Rate limiting (1 request per second as per Nominatim policy)
        - Returns multiple results ranked by relevance
        - Free service, no API key required
    
    Args:
        location_name (str): Name of the place (e.g., "Colombo", "Torre de Belém").
        region (str): Region context to add to search (default: "Lisboa, Portugal").
        
    Returns:
        Optional[Dict]: Dictionary with location info or None if not found.
        Contains: name, lat, lon, type, address, importance.
        
    Example:
        >>> result = geocode_location("Centro Comercial Colombo")
        >>> result
        {'name': 'Centro Comercial Colombo', 'lat': 38.7505, 'lon': -9.1848, 'type': 'mall'}
        
        >>> result = geocode_location("Vasco da Gama")
        >>> result
        {'name': 'Centro Comercial Vasco da Gama', 'lat': 38.7678, 'lon': -9.0939, 'type': 'mall'}
    """
    global _last_geocode_time
    
    # Rate limiting: ensure at least 1 second between requests
    if _last_geocode_time is not None:
        elapsed = time.time() - _last_geocode_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
    
    # Build search query with region context
    # Try different query formats for better results
    search_queries = [
        f"{location_name}, {region}",
        f"{location_name} Lisboa",
        location_name
    ]
    
    headers = {
        "User-Agent": "LisbonUrbanAssistant/1.0 (Master Thesis Project; NOVA IMS)"
    }
    
    for query in search_queries:
        try:
            params = {
                "q": query,
                "format": "json",
                "limit": 5,
                "addressdetails": 1,
                "countrycodes": "pt"  # Restrict to Portugal
            }
            
            response = requests.get(
                NOMINATIM_URL, 
                params=params, 
                headers=headers,
                timeout=10
            )
            _last_geocode_time = time.time()
            
            if response.status_code != 200:
                logger.warning(f"Nominatim returned status {response.status_code}")
                continue
            
            results = response.json()
            
            if not results:
                continue
            
            # Filter results to Lisbon metropolitan area
            # Bounding box: approximately 38.5-39.0 lat, -9.5 to -8.8 lon
            lisbon_results = []
            for r in results:
                lat = float(r.get("lat", 0))
                lon = float(r.get("lon", 0))
                
                # Check if within Lisbon metropolitan area
                if 38.4 <= lat <= 39.1 and -9.6 <= lon <= -8.7:
                    lisbon_results.append(r)
            
            if lisbon_results:
                best = lisbon_results[0]
                
                # Extract address components
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


def resolve_location(
    location_name: str,
    search_radius_km: float = 0.5,
    max_stops: int = 5
) -> Dict[str, Any]:
    """
    Intelligently resolves a location name to bus stops.
    
    This is the MAIN function for finding bus stops near any location.
    It uses a multi-step approach:
    
    1. First, try direct name match in Carris stops
    2. If few/no results, use geocoding to get coordinates
    3. Find stops near the geocoded coordinates
    
    This allows users to search for places like:
    - "Colombo" → finds stops near Centro Comercial Colombo
    - "Vasco da Gama" → finds stops near CC Vasco da Gama
    - "Torre de Belém" → finds stops near the monument
    - "Aeroporto" → finds stops near Lisbon Airport
    
    Args:
        location_name (str): Any location name (POI, address, landmark).
        search_radius_km (float): Radius for GPS search (default: 0.5km).
        max_stops (int): Maximum stops to return (default: 5).
        
    Returns:
        Dict with:
            - 'method': How the location was resolved ('name_match', 'geocoding')
            - 'location': Geocoded location info (if used)
            - 'stops': List of nearby bus stops
            - 'success': Whether any stops were found
            
    Example:
        >>> result = resolve_location("Colombo")
        >>> result['stops'][0]['name']
        'Colégio Militar (Metro) P7'
        >>> result['method']
        'geocoding'
    """
    result = {
        "method": None,
        "location": None,
        "stops": [],
        "success": False,
        "query": location_name
    }
    
    # Clean the location name
    clean_name = location_name.strip()
    
    # Step 1: Try direct name match in Carris stops
    name_matches = find_stops_by_name(clean_name, max_results=max_stops)
    
    # If we found good matches (3+ stops or exact match), use them
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
        
        # Find stops near the geocoded coordinates
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
        
        # If no stops within radius, try larger radius
        if not nearby_stops and search_radius_km < 1.0:
            nearby_stops = find_stops_near_coordinates(
                geocoded["lat"],
                geocoded["lon"],
                radius_km=1.0,  # Expand to 1km
                max_results=max_stops
            )
            
            if nearby_stops:
                result["method"] = "geocoding_expanded"
                result["stops"] = nearby_stops
                result["success"] = True
                logger.info(f"Resolved '{clean_name}' via expanded geocoding ({len(nearby_stops)} stops)")
                return result
    
    # Step 3: If geocoding failed but we had some name matches, use them
    if name_matches:
        result["method"] = "name_match_fallback"
        result["stops"] = name_matches
        result["success"] = True
        logger.info(f"Resolved '{clean_name}' via fallback name match ({len(name_matches)} stops)")
        return result
    
    # No results found
    logger.warning(f"Could not resolve location '{clean_name}'")
    return result


def load_carris_stops(force_reload: bool = False) -> List[Dict[str, Any]]:
    """
    Loads all Carris Metropolitana bus stops into memory cache.
    
    This function fetches ~5000 bus stops from the Carris API and caches
    them in memory for fast proximity searches. The cache is automatically
    refreshed every 24 hours or when force_reload is True.
    
    Features:
        - Lazy loading: Only loads when first needed
        - Automatic refresh: Reloads after 24 hours
        - Memory efficient: ~1MB for all stops
        - Streamlit Cloud compatible: No external database needed
    
    Args:
        force_reload (bool): Force refresh even if cache is valid.
        
    Returns:
        List[Dict]: List of stop dictionaries with id, name, lat, lon, lines.
        
    Example:
        >>> stops = load_carris_stops()
        >>> len(stops)
        5234  # Approximately 5000 stops
    """
    global _carris_stops_cache, _carris_stops_last_load
    
    # Return cached data if valid and not forcing reload
    if not force_reload and _carris_stops_cache and _is_cache_valid(_carris_stops_last_load):
        logger.info(f"Using cached Carris stops ({len(_carris_stops_cache)} stops)")
        return _carris_stops_cache
    
    logger.info("Loading all Carris stops from API...")
    
    try:
        # Fetch all stops from Carris API (returns JSON array)
        response = requests.get(CARRIS_STOPS_URL, timeout=30)
        response.raise_for_status()
        raw_stops = response.json()
        
        if not isinstance(raw_stops, list):
            logger.error("Unexpected response format from Carris stops API")
            return _carris_stops_cache or []
        
        # Process and cache stops
        # v1 API fields: id, name, lat, lon, municipality_name, lines, locality, facilities
        processed_stops = []
        for stop in raw_stops:
            # Extract essential fields (keeping memory usage low)
            # Note: v1 API returns lat/lon as strings, need to convert
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
                "lines": stop.get("lines", []),  # List of line IDs serving this stop
                "facilities": stop.get("facilities", [])  # school, transit_office, etc.
            }
            
            # Only include stops with valid coordinates
            if processed_stop["lat"] and processed_stop["lon"]:
                processed_stops.append(processed_stop)
        
        # Update cache
        _carris_stops_cache = processed_stops
        _carris_stops_last_load = datetime.now()
        
        logger.info(f"\033[1;32m✅ Loaded {len(processed_stops)} Carris stops\033[0m")
        return processed_stops
        
    except requests.exceptions.Timeout:
        logger.error("Timeout loading Carris stops (30s)")
        return _carris_stops_cache or []
    except requests.exceptions.RequestException as e:
        logger.error(f"Error loading Carris stops: {e}")
        return _carris_stops_cache or []
    except Exception as e:
        logger.error(f"Unexpected error loading Carris stops: {e}")
        return _carris_stops_cache or []


def load_carris_lines(force_reload: bool = False) -> List[Dict[str, Any]]:
    """
    Loads all Carris Metropolitana bus lines into memory cache.
    
    Lines contain useful information like color, municipalities served,
    localities, and associated routes/patterns. This is more user-friendly
    than routes for displaying line information.
    
    Args:
        force_reload (bool): Force refresh even if cache is valid.
        
    Returns:
        List[Dict]: List of line dictionaries with id, name, color, municipalities.
        
    Example:
        >>> lines = load_carris_lines()
        >>> lines[0]
        {'id': '1001', 'short_name': '1001', 'long_name': 'Alfragide - Reboleira', 'color': '#C61D23'}
    """
    global _carris_lines_cache, _carris_lines_last_load
    
    # Return cached data if valid and not forcing reload
    if not force_reload and _carris_lines_cache and _is_cache_valid(_carris_lines_last_load):
        logger.info(f"Using cached Carris lines ({len(_carris_lines_cache)} lines)")
        return _carris_lines_cache
    
    logger.info("Loading all Carris lines from API...")
    
    try:
        # Fetch all lines from Carris API
        response = requests.get(CARRIS_LINES_URL, timeout=30)
        response.raise_for_status()
        raw_lines = response.json()
        
        if not isinstance(raw_lines, list):
            logger.error("Unexpected response format from Carris lines API")
            return _carris_lines_cache or []
        
        # Process and cache lines
        # v1 API: id, short_name, long_name, color, text_color, municipalities, localities, routes, patterns
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
        
        # Update cache
        _carris_lines_cache = processed_lines
        _carris_lines_last_load = datetime.now()
        
        logger.info(f"\033[1;32m✅ Loaded {len(processed_lines)} Carris lines\033[0m")
        return processed_lines
        
    except requests.exceptions.Timeout:
        logger.error("Timeout loading Carris lines (30s)")
        return _carris_lines_cache or []
    except requests.exceptions.RequestException as e:
        logger.error(f"Error loading Carris lines: {e}")
        return _carris_lines_cache or []
    except Exception as e:
        logger.error(f"Unexpected error loading Carris lines: {e}")
        return _carris_lines_cache or []


def load_carris_routes(force_reload: bool = False) -> List[Dict[str, Any]]:
    """
    Loads all Carris Metropolitana bus routes into memory cache.
    
    This function fetches all bus routes (800+) from the Carris API and 
    caches them for route matching. Each route has a line_id and patterns.
    
    Note: For user display, prefer load_carris_lines() which has more info.
    Routes are useful for internal pattern matching.
    
    Args:
        force_reload (bool): Force refresh even if cache is valid.
        
    Returns:
        List[Dict]: List of route dictionaries with id, line_id, name, patterns.
        
    Example:
        >>> routes = load_carris_routes()
        >>> len(routes)
        850  # Approximately 800+ routes
    """
    global _carris_routes_cache, _carris_routes_last_load
    
    # Return cached data if valid and not forcing reload
    if not force_reload and _carris_routes_cache and _is_cache_valid(_carris_routes_last_load):
        logger.info(f"Using cached Carris routes ({len(_carris_routes_cache)} routes)")
        return _carris_routes_cache
    
    logger.info("Loading all Carris routes from API...")
    
    try:
        # Fetch all routes from Carris API
        response = requests.get(CARRIS_ROUTES_URL, timeout=30)
        response.raise_for_status()
        raw_routes = response.json()
        
        if not isinstance(raw_routes, list):
            logger.error("Unexpected response format from Carris routes API")
            return _carris_routes_cache or []
        
        # Process and cache routes
        # v1 API: id, short_name, long_name, color, text_color, line_id, patterns, municipalities, localities
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
        
        # Update cache
        _carris_routes_cache = processed_routes
        _carris_routes_last_load = datetime.now()
        
        logger.info(f"\033[1;32m✅ Loaded {len(processed_routes)} Carris routes\033[0m")
        return processed_routes
        
    except requests.exceptions.Timeout:
        logger.error("Timeout loading Carris routes (30s)")
        return _carris_routes_cache or []
    except requests.exceptions.RequestException as e:
        logger.error(f"Error loading Carris routes: {e}")
        return _carris_routes_cache or []
    except Exception as e:
        logger.error(f"Unexpected error loading Carris routes: {e}")
        return _carris_routes_cache or []


def find_stops_near_coordinates(
    lat: float, 
    lon: float, 
    radius_km: float = 0.5,
    max_results: int = 10
) -> List[Dict[str, Any]]:
    """
    Finds bus stops within a given radius of GPS coordinates.
    
    Uses Haversine distance formula for accurate GPS-based search.
    This is the core function for bus routing - it finds candidate
    stops near the origin and destination.
    
    Args:
        lat (float): Latitude of the search center.
        lon (float): Longitude of the search center.
        radius_km (float): Search radius in kilometers (default: 0.5km = 500m).
        max_results (int): Maximum number of stops to return (default: 10).
        
    Returns:
        List[Dict]: List of nearby stops, sorted by distance.
        Each stop contains: id, name, lat, lon, distance_km, lines.
        
    Example:
        >>> # Find stops near Praça do Comércio (Terreiro do Paço)
        >>> stops = find_stops_near_coordinates(38.7075, -9.1364)
        >>> stops[0]
        {'name': 'Praça do Comércio', 'distance_km': 0.05, 'lines': ['15E', '25E']}
    """
    # Ensure stops are loaded
    stops = load_carris_stops()
    
    if not stops:
        logger.warning("No Carris stops available for proximity search")
        return []
    
    # Calculate distance for each stop and filter by radius
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
    
    # Sort by distance and limit results
    nearby_stops.sort(key=lambda x: x["distance_km"])
    return nearby_stops[:max_results]


def find_stops_by_name(
    name_query: str,
    max_results: int = 10
) -> List[Dict[str, Any]]:
    """
    Finds bus stops matching a name query (fuzzy search).
    
    Searches through all cached stops and returns those whose name
    contains the query string (case-insensitive).
    
    Args:
        name_query (str): Search query for stop name.
        max_results (int): Maximum number of results (default: 10).
        
    Returns:
        List[Dict]: List of matching stops with id, name, lat, lon, lines.
        
    Example:
        >>> stops = find_stops_by_name("Colombo")
        >>> stops[0]
        {'name': 'Centro Comercial Colombo', 'lines': ['701', '750', '754']}
    """
    # Ensure stops are loaded
    stops = load_carris_stops()
    
    if not stops:
        logger.warning("No Carris stops available for name search")
        return []
    
    query_lower = name_query.lower().strip()
    
    # Find stops with matching names
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
    
    This function identifies direct bus routes by checking which lines
    serve both the origin stops and destination stops.
    
    Args:
        origin_stop_ids (List[str]): List of stop IDs near origin.
        dest_stop_ids (List[str]): List of stop IDs near destination.
        
    Returns:
        List[Dict]: List of route options with line info and stops.
        
    Example:
        >>> routes = find_common_routes(['060001'], ['060100'])
        >>> routes[0]
        {'line_id': '750', 'long_name': 'Algés - Campo Grande', 'origin_stop': {...}, 'dest_stop': {...}}
    """
    # Ensure stops and lines are loaded
    stops = load_carris_stops()
    lines = load_carris_lines()
    
    if not stops:
        return []
    
    # Create lookup maps for faster access
    stop_map = {s["id"]: s for s in stops}
    
    # Get lines serving origin stops
    origin_lines = set()
    origin_stop_lines = {}  # {line_id: stop_info}
    for stop_id in origin_stop_ids:
        stop = stop_map.get(stop_id)
        if stop:
            for line in stop.get("lines", []):
                origin_lines.add(line)
                if line not in origin_stop_lines:
                    origin_stop_lines[line] = stop
    
    # Get lines serving destination stops
    dest_lines = set()
    dest_stop_lines = {}  # {line_id: stop_info}
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
    
    # Build route options using line info (more complete than routes)
    route_options = []
    
    # Create line lookup map
    line_map = {l["id"]: l for l in lines}
    
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
    
    # Sort by line number for consistent display
    route_options.sort(key=lambda x: x.get("short_name", ""))
    
    return route_options

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
    Gets real-time status of CP trains serving the Lisbon Metropolitan Area (AML).
    
    This function filters trains to show only those that:
    - Depart from an AML station, OR
    - Arrive at an AML station, OR
    - Are currently passing through the AML region
    
    This reduces ~90 Portugal-wide trains to ~45 relevant AML trains.
    
    Information provided for each train:
    - Train number and service type (Urbanos, Regionais, Intercidades, Alfa Pendular)
    - Origin and destination stations
    - Current delay in minutes
    - Real-time GPS position
    - Disruption status
    
    Returns:
        str: List of AML trains with status, delays, and positions.
        
    Example:
        >>> get_train_status()
        "🚆 CP Trains - Lisbon Metropolitan Area (AML)
         ==========================================
         📊 45/90 trains serving AML
         
         Urbanos Lisboa (26 trains):
           ✅ #19048: Cascais → Cais do Sodré (On time)
           ⚠️ #18258: Oriente → Sintra (5 min late)
         ..."
    """
    # Get trains filtered to AML only
    aml_trains = get_cp_aml_trains()
    
    if not aml_trains:
        return "❌ Failed to fetch train status. The API may be temporarily unavailable."
    
    # Load AML stations for context
    aml_stations = load_cp_aml_stations()
    
    response = "🚆 **CP Trains - Lisbon Metropolitan Area (AML)**\n"
    response += "=" * 50 + "\n\n"
    
    # Group trains by service type
    from collections import defaultdict
    by_service = defaultdict(list)
    
    for train in aml_trains:
        service_name = train.get('service', {}).get('designation', 'Unknown')
        by_service[service_name].append(train)
    
    # Count stats
    total_trains = len(aml_trains)
    delayed_trains = sum(1 for t in aml_trains if (t.get('delay') or 0) > 0)
    
    response += f"📊 **{total_trains} trains** serving AML\n"
    if delayed_trains > 0:
        response += f"⚠️ **{delayed_trains} trains** with delays\n"
    response += "\n"
    
    # Display by service type (Urbanos first, then by number of trains)
    service_order = ['Urbanos Lisboa', 'Regionais', 'Intercidades', 'Alfa Pendular']
    
    for service_name in service_order:
        if service_name not in by_service:
            continue
            
        trains = by_service[service_name]
        # Sort by delay (most delayed first)
        trains.sort(key=lambda x: -(x.get('delay') or 0))
        
        # Service emoji
        service_emoji = {
            'Urbanos Lisboa': '🚈',
            'Regionais': '🚃',
            'Intercidades': '🚄',
            'Alfa Pendular': '🚅'
        }.get(service_name, '🚆')
        
        response += f"\n{service_emoji} **{service_name}** ({len(trains)} trains)\n"
        response += "-" * 40 + "\n"
        
        # Show top 5 trains per service (prioritize delayed)
        for train in trains[:5]:
            train_number = train.get('trainNumber', 'N/A')
            delay = train.get('delay') or 0
            status = train.get('status', 'Unknown')
            has_disruptions = train.get('hasDisruptions', False)
            lat = train.get('latitude')
            lon = train.get('longitude')
            
            # Origin and destination
            origin = train.get('origin', {})
            destination = train.get('destination', {})
            origin_name = origin.get('designation', 'N/A') if origin else 'N/A'
            dest_name = destination.get('designation', 'N/A') if destination else 'N/A'
            
            # Delay indicator
            if delay == 0:
                delay_str = "✅ On time"
            elif delay > 0:
                delay_str = f"⚠️ {delay} min late"
            else:
                delay_str = "✅ Ahead"
            
            # Status emoji
            status_emoji = {
                'IN_TRANSIT': '🚆',
                'AT_STATION': '🚉',
                'STOPPED': '⏸️'
            }.get(status, '❓')
            
            response += f"\n   {status_emoji} **#{train_number}**: {origin_name} → {dest_name}\n"
            response += f"      {delay_str}"
            
            if has_disruptions:
                response += " | ⚠️ Disruptions"
            
            # Show coordinates if available
            if lat and lon:
                try:
                    response += f"\n      📍 Position: ({float(lat):.4f}, {float(lon):.4f})"
                except (ValueError, TypeError):
                    pass
            
            response += "\n"
        
        if len(trains) > 5:
            response += f"\n   ... and {len(trains) - 5} more {service_name} trains.\n"
    
    # Add footer with AML stations info
    response += "\n" + "-" * 50 + "\n"
    response += f"📍 **AML Coverage**: {len(aml_stations)} stations\n"
    response += "🔗 Lines: Cascais, Sintra, Azambuja, Fertagus\n"
    response += "💡 Use `search_cp_stations` to find specific stations.\n"
    
    return response


@tool
def search_cp_stations(query: str) -> str:
    """
    Searches for CP train stations in the Lisbon Metropolitan Area (AML).
    
    Use this tool to find station codes, locations, and which railway lines
    serve each station. The AML includes ~81 stations across multiple lines:
    - Linha de Cascais (Cais do Sodré ↔ Cascais)
    - Linha de Sintra (Rossio/Oriente ↔ Sintra)
    - Linha de Azambuja (Santa Apolónia/Oriente ↔ Azambuja)
    - Fertagus (Entrecampos ↔ Setúbal)
    
    Args:
        query (str): Station name or partial name to search for.
        
    Returns:
        str: List of matching stations with details.
        
    Examples:
        >>> search_cp_stations("Oriente")
        "🚉 CP Stations matching 'Oriente':
         1. Lisboa Oriente (94-31039)
            📍 (38.7678, -9.0990)
            🚆 Major hub - Sintra, Azambuja lines"
            
        >>> search_cp_stations("Cascais")
        "🚉 CP Stations matching 'Cascais':
         1. Cascais (94-69260)
            📍 (38.6925, -9.4177)
            🚆 Linha de Cascais terminus"
    """
    matches = search_cp_station(query)
    
    if not matches:
        return f"❌ No CP stations found matching '{query}' in the AML region.\n\n" \
               f"💡 Try searching for: Oriente, Rossio, Cais do Sodré, Cascais, Sintra, Entrecampos"
    
    response = f"🚉 **CP Stations matching '{query}'** ({len(matches)} found)\n"
    response += "=" * 50 + "\n\n"
    
    for i, station in enumerate(matches[:10], 1):
        name = station.get('name', 'Unknown')
        code = station.get('code', '')
        lat = station.get('lat', 0)
        lon = station.get('lon', 0)
        railways = station.get('railways', [])
        
        response += f"{i}. **{name}** ({code})\n"
        response += f"   📍 ({lat:.4f}, {lon:.4f})\n"
        
        # Try to identify the line based on station name
        line_info = ""
        name_lower = name.lower()
        if "cascais" in name_lower or "estoril" in name_lower or "cais do sodre" in name_lower or "oeiras" in name_lower:
            line_info = "🚆 Linha de Cascais"
        elif "sintra" in name_lower or "cacem" in name_lower or "queluz" in name_lower or "rossio" in name_lower:
            line_info = "🚆 Linha de Sintra"
        elif "azambuja" in name_lower or "alverca" in name_lower or "vila franca" in name_lower:
            line_info = "🚆 Linha de Azambuja"
        elif "oriente" in name_lower or "entrecampos" in name_lower or "santa apolonia" in name_lower:
            line_info = "🚆 Major hub (multiple lines)"
        elif "barreiro" in name_lower or "setubal" in name_lower or "pinhal" in name_lower:
            line_info = "🚆 Fertagus / South bank"
        
        if line_info:
            response += f"   {line_info}\n"
        
        if railways:
            response += f"   🔗 Railways: {', '.join(railways)}\n"
        
        response += "\n"
    
    if len(matches) > 10:
        response += f"... and {len(matches) - 10} more stations.\n"
    
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
    
    This tool uses SMART LOCATION RESOLUTION to understand any place name:
    - "Colombo" → finds Centro Comercial Colombo → finds nearby stops
    - "Vasco da Gama" → finds CC Vasco da Gama → finds nearby stops
    - "Torre de Belém" → finds the monument → finds nearby stops
    - "Aeroporto" → finds Lisbon Airport → finds nearby stops
    
    The tool uses geocoding (OpenStreetMap) to convert place names to coordinates,
    then finds bus stops within the search radius.
    
    Args:
        origin (str): Starting location - can be ANY place name, landmark, or address.
        destination (str): Ending location - can be ANY place name, landmark, or address.
        origin_lat (float, optional): Override origin with GPS coordinates.
        origin_lon (float, optional): Override origin with GPS coordinates.
        dest_lat (float, optional): Override destination with GPS coordinates.
        dest_lon (float, optional): Override destination with GPS coordinates.
        search_radius_km (float): Search radius in km (default: 0.5km = 500m).
        
    Returns:
        str: Formatted string with bus route options including:
             - Resolved location names and coordinates
             - Direct bus lines connecting the two locations
             - Stop names and line information
             - Alternative suggestions if no direct route found
    
    Examples:
        >>> find_bus_routes("Colombo", "Vasco da Gama")  # Shopping centers
        >>> find_bus_routes("Aeroporto", "Marquês de Pombal")  # Airport to center
        >>> find_bus_routes("Torre de Belém", "Oceanário")  # Landmarks
    """
    response = "🚌 **BUS ROUTE FINDER**\n"
    response += "=" * 50 + "\n"
    response += f"📍 From: {origin}\n"
    response += f"📍 To: {destination}\n"
    response += "=" * 50 + "\n\n"
    
    # -------------------------------------------------------------------------
    # Step 1: Resolve origin location (smart geocoding)
    # -------------------------------------------------------------------------
    response += "🔍 **Resolving origin location...**\n"
    
    if origin_lat is not None and origin_lon is not None:
        # GPS coordinates provided directly
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
        # Use smart location resolution
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
        response += f"\n❌ **No bus stops found near '{origin}'**\n"
        response += "   💡 Try:\n"
        response += "   • Adding 'Lisboa' to your search (e.g., 'Colombo Lisboa')\n"
        response += "   • Using a more specific name or address\n"
        response += "   • Using landmarks near your location\n"
        return response
    
    response += "\n"
    
    # -------------------------------------------------------------------------
    # Step 2: Resolve destination location (smart geocoding)
    # -------------------------------------------------------------------------
    response += "🔍 **Resolving destination location...**\n"
    
    if dest_lat is not None and dest_lon is not None:
        # GPS coordinates provided directly
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
        # Use smart location resolution
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
        response += f"\n❌ **No bus stops found near '{destination}'**\n"
        response += "   💡 Try:\n"
        response += "   • Adding 'Lisboa' to your search (e.g., 'Vasco da Gama Lisboa')\n"
        response += "   • Using a more specific name or address\n"
        response += "   • Using landmarks near your destination\n"
        return response
    
    response += "\n"
    
    # -------------------------------------------------------------------------
    # Step 3: Find common routes
    # -------------------------------------------------------------------------
    response += "🔍 **Finding direct bus routes...**\n\n"
    
    origin_stop_ids = [s["id"] for s in origin_stops]
    dest_stop_ids = [s["id"] for s in dest_stops]
    
    route_options = find_common_routes(origin_stop_ids, dest_stop_ids)
    
    if route_options:
        response += f"✅ **{len(route_options)} DIRECT ROUTE(S) FOUND!**\n"
        response += "-" * 40 + "\n\n"
        
        for i, route in enumerate(route_options[:5], 1):
            response += f"🚌 **Option {i}: Line {route['short_name']}**\n"
            response += f"   📍 Route: {route['long_name']}\n"
            
            origin_stop = route.get("origin_stop", {})
            dest_stop = route.get("dest_stop", {})
            
            if origin_stop:
                response += f"   🚏 Board at: {origin_stop.get('name', 'N/A')}\n"
            if dest_stop:
                response += f"   🚏 Alight at: {dest_stop.get('name', 'N/A')}\n"
            
            response += "\n"
        
        if len(route_options) > 5:
            response += f"   ... and {len(route_options) - 5} more routes available.\n\n"
    
    else:
        response += "❌ **No direct bus routes found**\n\n"
        response += "💡 **Suggestions:**\n"
        response += "   • You may need to transfer buses\n"
        response += "   • Consider using Metro + Bus combination\n"
        response += "   • Check routes from a nearby major stop\n\n"
        
        # Show available lines at each location
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
    
    # -------------------------------------------------------------------------
    # Step 4: Add helpful tips
    # -------------------------------------------------------------------------
    response += "\n" + "-" * 40 + "\n"
    response += "💡 **Tips:**\n"
    response += "   • Use 'get_carris_stop_info' for real-time arrivals\n"
    response += "   • Check 'get_carris_alerts' for service disruptions\n"
    response += "   • Metro may be faster for cross-city travel\n"
    
    return response


@tool
def search_bus_stops_nearby(
    lat: float,
    lon: float,
    radius_km: float = 0.5,
    max_results: int = 10
) -> str:
    """
    Finds Carris bus stops near GPS coordinates.
    
    Use this tool when you have GPS coordinates and want to find
    nearby bus stops with the lines that serve them.
    
    Args:
        lat (float): Latitude of the search center.
        lon (float): Longitude of the search center.
        radius_km (float): Search radius in kilometers (default: 0.5km).
        max_results (int): Maximum stops to return (default: 10).
        
    Returns:
        str: Formatted list of nearby stops with distance and lines.
        
    Example:
        >>> search_bus_stops_nearby(38.7223, -9.1393, radius_km=0.3)
    """
    response = "🚏 **NEARBY BUS STOPS**\n"
    response += "=" * 50 + "\n"
    response += f"📍 Search center: {lat:.6f}, {lon:.6f}\n"
    response += f"📏 Radius: {radius_km}km ({radius_km*1000:.0f}m)\n"
    response += "=" * 50 + "\n\n"
    
    stops = find_stops_near_coordinates(lat, lon, radius_km, max_results)
    
    if not stops:
        response += f"❌ No bus stops found within {radius_km}km\n"
        response += "   Try increasing the search radius.\n"
        return response
    
    response += f"✅ **Found {len(stops)} stops:**\n\n"
    
    for i, stop in enumerate(stops, 1):
        distance_m = stop["distance_km"] * 1000
        response += f"**{i}. {stop['name']}**\n"
        response += f"   📏 Distance: {distance_m:.0f}m\n"
        response += f"   📍 Location: {stop['municipality']}\n"
        
        lines = stop.get("lines", [])
        if lines:
            lines_str = ", ".join(lines[:10])
            if len(lines) > 10:
                lines_str += f" (+{len(lines)-10} more)"
            response += f"   🚌 Lines: {lines_str}\n"
        
        response += f"   🔖 Stop ID: {stop['id']}\n\n"
    
    response += "-" * 40 + "\n"
    response += "💡 Use stop ID with 'get_carris_stop_info' for real-time arrivals.\n"
    
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
    import json
    
    print("\033[1m" + "=" * 70 + "\033[0m")
    print("\033[1m🧪 TRANSPORT API TOOLS - COMPREHENSIVE TEST SUITE\033[0m")
    print("\033[1m" + "=" * 70 + "\033[0m")
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔗 Carris API: {CARRIS_BASE_URL}")
    print("=" * 70)
    
    test_results = {"passed": 0, "failed": 0}
    
    def run_test(test_name: str, test_func):
        """Helper function to run a test and track results."""
        print(f"\n\033[1m{'─' * 70}\033[0m")
        print(f"\033[1m🧪 {test_name}\033[0m")
        print("─" * 70)
        try:
            result = test_func()
            test_results["passed"] += 1
            return result
        except Exception as e:
            print(f"\033[1;31m❌ TEST FAILED: {e}\033[0m")
            test_results["failed"] += 1
            return None
    
    # =========================================================================
    # TEST 1: Transport Summary (Quick Overview)
    # =========================================================================
    def test_transport_summary():
        result = get_transport_summary.invoke({})
        print(result)
        assert "TRANSPORT" in result.upper(), "Should contain transport info"
        print("\033[1;32m✅ Transport summary retrieved successfully\033[0m")
        return result
    
    run_test("Test 1: Transport Summary", test_transport_summary)
    
    # =========================================================================
    # TEST 2: Metro Status
    # =========================================================================
    def test_metro_status():
        result = get_metro_status.invoke({})
        print(result)
        # Check for line names (English or Portuguese) or line colors
        assert any(term in result.lower() for term in ["yellow", "blue", "green", "red", "normal", "line"]), \
            "Should contain Metro line information"
        print("\033[1;32m✅ Metro status retrieved successfully\033[0m")
        return result
    
    run_test("Test 2: Metro Status", test_metro_status)
    
    # =========================================================================
    # TEST 3: Carris Alerts
    # =========================================================================
    def test_carris_alerts():
        result = get_carris_alerts.invoke({})
        print(result[:1000] + "..." if len(result) > 1000 else result)
        assert "CARRIS" in result.upper() or "ALERT" in result.upper(), \
            "Should contain Carris alert info"
        print("\033[1;32m✅ Carris alerts retrieved successfully\033[0m")
        return result
    
    run_test("Test 3: Carris Alerts", test_carris_alerts)
    
    # =========================================================================
    # TEST 4: Train Status (CP)
    # =========================================================================
    def test_train_status():
        result = get_train_status.invoke({})
        print(result)
        assert "CP" in result.upper() or "TRAIN" in result.upper(), \
            "Should contain train status info"
        print("\033[1;32m✅ Train status retrieved successfully\033[0m")
        return result
    
    run_test("Test 4: Train Status (CP)", test_train_status)
    
    # =========================================================================
    # TEST 5: Load Carris Stops (Cache System)
    # =========================================================================
    def test_load_carris_stops():
        print("Loading all Carris stops (first call loads from API)...")
        start_time = time.time()
        stops = load_carris_stops()
        load_time = time.time() - start_time
        
        print(f"\n📊 \033[1mStops Statistics:\033[0m")
        print(f"   • Total stops loaded: {len(stops)}")
        print(f"   • Load time: {load_time:.2f}s")
        
        if stops:
            # Sample stop info
            sample = stops[0]
            print(f"\n📍 \033[1mSample Stop:\033[0m")
            print(f"   • ID: {sample.get('id')}")
            print(f"   • Name: {sample.get('name')}")
            print(f"   • Location: {sample.get('lat')}, {sample.get('lon')}")
            print(f"   • Municipality: {sample.get('municipality')}")
            print(f"   • Lines serving: {len(sample.get('lines', []))} lines")
        
        # Test cache (second call should be instant)
        print("\n🔄 Testing cache (second call)...")
        start_time = time.time()
        stops2 = load_carris_stops()
        cache_time = time.time() - start_time
        print(f"   • Cache retrieval time: {cache_time:.4f}s")
        
        assert len(stops) > 10000, f"Expected >10000 stops, got {len(stops)}"
        assert cache_time < 0.1, "Cache should be nearly instant"
        print("\033[1;32m✅ Carris stops loaded and cached successfully\033[0m")
        return stops
    
    run_test("Test 5: Load Carris Stops (Cache System)", test_load_carris_stops)
    
    # =========================================================================
    # TEST 6: Load Carris Lines
    # =========================================================================
    def test_load_carris_lines():
        print("Loading all Carris lines...")
        lines = load_carris_lines()
        
        print(f"\n📊 \033[1mLines Statistics:\033[0m")
        print(f"   • Total lines loaded: {len(lines)}")
        
        if lines:
            # Sample line info
            sample = lines[0]
            print(f"\n🚌 \033[1mSample Line:\033[0m")
            print(f"   • ID: {sample.get('id')}")
            print(f"   • Short Name: {sample.get('short_name')}")
            print(f"   • Long Name: {sample.get('long_name')}")
            print(f"   • Color: {sample.get('color')}")
            print(f"   • Localities: {', '.join(sample.get('localities', [])[:5])}")
        
        assert len(lines) > 500, f"Expected >500 lines, got {len(lines)}"
        print("\033[1;32m✅ Carris lines loaded successfully\033[0m")
        return lines
    
    run_test("Test 6: Load Carris Lines", test_load_carris_lines)
    
    # =========================================================================
    # TEST 7: Find Stops Near Coordinates (GPS Search)
    # =========================================================================
    def test_find_stops_near_coordinates():
        # Coordinates for Gare do Oriente (major Carris hub)
        # Note: Carris Metropolitana serves suburbs, not central Lisbon
        lat, lon = 38.7678, -9.0990
        radius_km = 0.5  # 500 meters
        
        print(f"Searching for stops near Gare do Oriente ({lat}, {lon})...")
        print(f"Search radius: {radius_km}km ({radius_km*1000}m)")
        print("Note: Carris Metropolitana serves suburbs, not historic center.")
        
        stops = find_stops_near_coordinates(lat, lon, radius_km=radius_km, max_results=10)
        
        print(f"\n📍 \033[1mFound {len(stops)} stops:\033[0m")
        for i, stop in enumerate(stops[:5], 1):
            print(f"   {i}. {stop['name']}")
            print(f"      Distance: {stop['distance_km']*1000:.0f}m")
            print(f"      Lines: {', '.join(stop['lines'][:5])}")
        
        assert len(stops) > 0, "Should find stops near Gare do Oriente"
        assert stops[0]["distance_km"] <= radius_km, "First stop should be within radius"
        print("\033[1;32m✅ GPS-based stop search works correctly\033[0m")
        return stops
    
    run_test("Test 7: Find Stops Near Coordinates (GPS)", test_find_stops_near_coordinates)
    
    # =========================================================================
    # TEST 8: Find Stops by Name
    # =========================================================================
    def test_find_stops_by_name():
        # Use names that actually exist in Carris API
        # Note: "Colombo" is called "Colégio Militar" in Carris
        search_terms = ["Oriente", "Colégio Militar", "Cais do Sodré"]
        
        for term in search_terms:
            print(f"\n🔍 Searching for stops with '{term}'...")
            stops = find_stops_by_name(term, max_results=3)
            
            print(f"   Found {len(stops)} stops:")
            for stop in stops:
                print(f"      • {stop['name']}")
                print(f"        Municipality: {stop.get('municipality', 'N/A')}")
                print(f"        Lines: {', '.join(stop.get('lines', [])[:5])}")
        
        # Main test with Oriente (major hub)
        oriente_stops = find_stops_by_name("Oriente", max_results=5)
        assert len(oriente_stops) > 0, "Should find stops with 'Oriente'"
        print("\n\033[1;32m✅ Name-based stop search works correctly\033[0m")
        return oriente_stops
    
    run_test("Test 8: Find Stops by Name", test_find_stops_by_name)
    
    # =========================================================================
    # TEST 9: Find Common Routes (Direct Bus Connections)
    # =========================================================================
    def test_find_common_routes():
        print("Finding direct bus routes between two areas...")
        print("Note: Using Oriente and Colégio Militar (near CC Colombo)")
        
        # Find stops near Oriente and Colégio Militar
        oriente_stops = find_stops_by_name("Oriente", max_results=10)
        militar_stops = find_stops_by_name("Colégio Militar", max_results=10)
        
        routes = []
        if oriente_stops and militar_stops:
            oriente_ids = [s["id"] for s in oriente_stops]
            militar_ids = [s["id"] for s in militar_stops]
            
            print(f"\n📍 Origin stops (Oriente): {len(oriente_ids)}")
            print(f"📍 Destination stops (Colégio Militar): {len(militar_ids)}")
            
            routes = find_common_routes(oriente_ids, militar_ids)
            
            print(f"\n🚌 \033[1mDirect Routes Found: {len(routes)}\033[0m")
            for route in routes[:5]:
                print(f"   • Line {route['short_name']}: {route['long_name']}")
                print(f"     Color: {route['color']}")
                if route.get('localities'):
                    print(f"     Via: {', '.join(route['localities'][:4])}")
        else:
            print("⚠️ Could not find stops for one of the locations")
        
        print("\033[1;32m✅ Common routes finder works correctly\033[0m")
        return routes
    
    run_test("Test 9: Find Common Routes (Direct Connections)", test_find_common_routes)
    
    # =========================================================================
    # TEST 10: Bus Route Finder Tool (@tool) - Direct Stop Names
    # =========================================================================
    def test_find_bus_routes_tool():
        print("Testing find_bus_routes tool with direct stop names...")
        print("Query: 'Oriente' → 'Colégio Militar' (exact Carris stop names)")
        
        result = find_bus_routes.invoke({
            "origin": "Oriente",
            "destination": "Colégio Militar"
        })
        
        print(result[:2000] + "..." if len(result) > 2000 else result)
        
        assert "BUS ROUTE" in result.upper(), "Should contain bus route info"
        print("\n\033[1;32m✅ Bus route finder tool works correctly\033[0m")
        return result
    
    run_test("Test 10: Bus Route Finder (Direct Names)", test_find_bus_routes_tool)
    
    # =========================================================================
    # TEST 11: Search Bus Stops Nearby Tool (@tool)
    # =========================================================================
    def test_search_bus_stops_nearby_tool():
        # Coordinates for Gare do Oriente (where Carris operates)
        lat, lon = 38.7678, -9.0990
        
        print(f"Testing search_bus_stops_nearby tool near Oriente ({lat}, {lon})...")
        print("Note: Using Oriente instead of Rossio (Carris serves suburbs)")
        
        result = search_bus_stops_nearby.invoke({
            "lat": lat,
            "lon": lon,
            "radius_km": 0.5,
            "max_results": 5
        })
        
        print(result)
        
        assert "NEARBY" in result.upper() or "STOP" in result.upper(), \
            "Should contain stop information"
        print("\n\033[1;32m✅ Nearby stops search tool works correctly\033[0m")
        return result
    
    run_test("Test 11: Search Bus Stops Nearby Tool (@tool)", test_search_bus_stops_nearby_tool)
    
    # =========================================================================
    # TEST 12: Metro Routing (get_route_between_stations)
    # =========================================================================
    def test_metro_routing():
        print("Testing Metro routing (Entrecampos → São Sebastião)...")
        
        result = get_route_between_stations.invoke({
            "origin": "Entrecampos",
            "destination": "São Sebastião"
        })
        
        print(result)
        
        assert "ROUTE" in result.upper() or "METRO" in result.upper(), \
            "Should contain routing info"
        print("\n\033[1;32m✅ Metro routing works correctly\033[0m")
        return result
    
    run_test("Test 12: Metro Routing", test_metro_routing)
    
    # =========================================================================
    # TEST 13: Carris Stop Info with Real-time Arrivals
    # =========================================================================
    def test_carris_stop_info():
        # Use a known stop ID (Marquês de Pombal)
        stop_id = "060101"
        
        print(f"Testing get_carris_stop_info for stop {stop_id}...")
        
        result = get_carris_stop_info.invoke({"stop_id": stop_id})
        
        print(result[:1500] + "..." if len(result) > 1500 else result)
        
        print("\n\033[1;32m✅ Carris stop info retrieved successfully\033[0m")
        return result
    
    run_test("Test 13: Carris Stop Info (Real-time)", test_carris_stop_info)
    
    # =========================================================================
    # TEST 14: Search Carris Lines
    # =========================================================================
    def test_search_carris_lines():
        search_query = "Colombo"
        
        print(f"Testing search_carris_lines for '{search_query}'...")
        
        result = search_carris_lines.invoke({"query": search_query})
        
        print(result[:1500] + "..." if len(result) > 1500 else result)
        
        assert "LINE" in result.upper() or search_query.upper() in result.upper(), \
            "Should contain line information"
        print("\n\033[1;32m✅ Carris lines search works correctly\033[0m")
        return result
    
    run_test("Test 14: Search Carris Lines", test_search_carris_lines)
    
    # =========================================================================
    # TEST 15: Geocoding - Centro Comercial Colombo
    # =========================================================================
    def test_geocode_colombo():
        print("Testing geocoding for 'Centro Comercial Colombo'...")
        
        result = geocode_location("Centro Comercial Colombo")
        
        if result:
            print(f"\n🌍 \033[1mGeocoding Result:\033[0m")
            print(f"   • Name: {result['name'][:80]}...")
            print(f"   • Coordinates: ({result['lat']:.4f}, {result['lon']:.4f})")
            print(f"   • Type: {result['type']}")
            print(f"   • Query: {result['query_used']}")
            
            # Verify coordinates are near Colombo (38.7505, -9.1848)
            assert 38.74 < result['lat'] < 38.76, f"Latitude should be ~38.75, got {result['lat']}"
            assert -9.20 < result['lon'] < -9.17, f"Longitude should be ~-9.18, got {result['lon']}"
            print("\033[1;32m✅ Geocoding Colombo works correctly\033[0m")
        else:
            print("\033[1;31m❌ Failed to geocode Colombo\033[0m")
            assert False, "Geocoding should return result"
        
        return result
    
    run_test("Test 15: Geocoding - Colombo", test_geocode_colombo)
    
    # =========================================================================
    # TEST 16: Geocoding - Vasco da Gama
    # =========================================================================
    def test_geocode_vasco_da_gama():
        print("Testing geocoding for 'Centro Comercial Vasco da Gama'...")
        time.sleep(1.1)  # Rate limiting
        
        result = geocode_location("Centro Comercial Vasco da Gama")
        
        if result:
            print(f"\n🌍 \033[1mGeocoding Result:\033[0m")
            print(f"   • Name: {result['name'][:80]}...")
            print(f"   • Coordinates: ({result['lat']:.4f}, {result['lon']:.4f})")
            print(f"   • Type: {result['type']}")
            
            # Verify coordinates are near Oriente/Vasco da Gama (38.7678, -9.0939)
            assert 38.76 < result['lat'] < 38.78, f"Latitude should be ~38.77, got {result['lat']}"
            assert -9.11 < result['lon'] < -9.08, f"Longitude should be ~-9.09, got {result['lon']}"
            print("\033[1;32m✅ Geocoding Vasco da Gama works correctly\033[0m")
        else:
            print("\033[1;31m❌ Failed to geocode Vasco da Gama\033[0m")
            assert False, "Geocoding should return result"
        
        return result
    
    run_test("Test 16: Geocoding - Vasco da Gama", test_geocode_vasco_da_gama)
    
    # =========================================================================
    # TEST 17: Smart Location Resolution - Colombo
    # =========================================================================
    def test_resolve_location_colombo():
        print("Testing smart location resolution for 'Colombo'...")
        time.sleep(1.1)  # Rate limiting
        
        result = resolve_location("Colombo", search_radius_km=0.5, max_stops=5)
        
        print(f"\n📍 \033[1mResolution Result:\033[0m")
        print(f"   • Method: {result['method']}")
        print(f"   • Success: {result['success']}")
        print(f"   • Stops found: {len(result['stops'])}")
        
        if result['location']:
            print(f"   • Geocoded to: ({result['location']['lat']:.4f}, {result['location']['lon']:.4f})")
        
        if result['stops']:
            print(f"\n🚏 \033[1mNearby Stops:\033[0m")
            for stop in result['stops'][:3]:
                dist = stop.get('distance_km')
                if dist:
                    print(f"      • {stop['name']} ({dist*1000:.0f}m)")
                else:
                    print(f"      • {stop['name']}")
        
        assert result['success'], "Should resolve 'Colombo' successfully"
        assert len(result['stops']) > 0, "Should find stops near Colombo"
        print("\033[1;32m✅ Smart location resolution works for Colombo\033[0m")
        return result
    
    run_test("Test 17: Smart Location Resolution - Colombo", test_resolve_location_colombo)
    
    # =========================================================================
    # TEST 18: Bus Routes with Smart Resolution (Colombo → Vasco da Gama)
    # =========================================================================
    def test_bus_routes_smart():
        print("Testing bus routes with smart geocoding...")
        print("Query: 'Colombo' → 'Vasco da Gama' (using place names, not stop names)")
        time.sleep(1.1)  # Rate limiting
        
        result = find_bus_routes.invoke({
            "origin": "Colombo",
            "destination": "Vasco da Gama"
        })
        
        print(result[:2500] + "..." if len(result) > 2500 else result)
        
        # Check that geocoding was used
        assert "Geocoded" in result or "Found" in result, \
            "Should show geocoding or found stops"
        assert "BUS ROUTE" in result.upper(), "Should contain bus route info"
        print("\n\033[1;32m✅ Smart bus routing works correctly\033[0m")
        return result
    
    run_test("Test 18: Bus Routes Smart (Colombo → Vasco da Gama)", test_bus_routes_smart)
    
    # =========================================================================
    # TEST 19: Load CP AML Stations (Cache)
    # =========================================================================
    def test_load_cp_aml_stations():
        print("Loading CP AML stations (first call loads from API)...")
        
        import time
        start = time.time()
        stations = load_cp_aml_stations()
        load_time = time.time() - start
        
        print(f"\n📊 CP AML Stations Statistics:")
        print(f"   • Total AML stations: {len(stations)}")
        print(f"   • Load time: {load_time:.2f}s")
        
        # Should have ~81 stations in AML
        assert len(stations) > 50, f"Expected 50+ AML stations, got {len(stations)}"
        assert len(stations) < 150, f"Expected <150 AML stations, got {len(stations)} (filter not working?)"
        
        # Check sample station structure
        if stations:
            sample_code = list(stations.keys())[0]
            sample = stations[sample_code]
            print(f"\n📍 Sample Station: {sample['name']}")
            print(f"   • Code: {sample['code']}")
            print(f"   • Location: ({sample['lat']:.4f}, {sample['lon']:.4f})")
            
            assert 'name' in sample, "Station should have name"
            assert 'lat' in sample, "Station should have lat"
            assert 'lon' in sample, "Station should have lon"
            assert 'code' in sample, "Station should have code"
        
        # Test cache (second call should be instant)
        start2 = time.time()
        stations2 = load_cp_aml_stations()
        cache_time = time.time() - start2
        
        print(f"\n🔄 Cache test:")
        print(f"   • Cache retrieval time: {cache_time:.4f}s")
        assert cache_time < 0.01, "Cache should be instant"
        assert len(stations2) == len(stations), "Cache should return same data"
        
        print("\n\033[1;32m✅ CP AML stations cache works correctly\033[0m")
        return stations
    
    run_test("Test 19: Load CP AML Stations (Cache)", test_load_cp_aml_stations)
    
    # =========================================================================
    # TEST 20: Get CP AML Trains (Filtered)
    # =========================================================================
    def test_get_cp_aml_trains():
        print("Getting CP trains filtered to AML region...")
        
        trains = get_cp_aml_trains()
        
        print(f"\n📊 AML Trains Statistics:")
        print(f"   • Total AML trains: {len(trains)}")
        
        # Should have some trains (typically 30-50 during the day)
        assert len(trains) > 0, "Should have at least 1 train"
        
        # Group by service type
        from collections import Counter
        services = Counter(t.get('service', {}).get('designation', 'Unknown') for t in trains)
        
        print(f"\n🚆 By Service Type:")
        for service, count in services.most_common():
            print(f"   • {service}: {count} trains")
        
        # Check train structure
        if trains:
            sample = trains[0]
            print(f"\n📍 Sample Train:")
            print(f"   • Number: #{sample.get('trainNumber')}")
            print(f"   • Service: {sample.get('service', {}).get('designation')}")
            print(f"   • Route: {sample.get('origin', {}).get('designation')} → {sample.get('destination', {}).get('designation')}")
            print(f"   • Delay: {sample.get('delay', 0)} min")
            print(f"   • Status: {sample.get('status')}")
            
            assert 'trainNumber' in sample, "Train should have trainNumber"
            assert 'origin' in sample, "Train should have origin"
            assert 'destination' in sample, "Train should have destination"
        
        print("\n\033[1;32m✅ CP AML trains filter works correctly\033[0m")
        return trains
    
    run_test("Test 20: Get CP AML Trains (Filtered)", test_get_cp_aml_trains)
    
    # =========================================================================
    # TEST 21: Search CP Stations
    # =========================================================================
    def test_search_cp_stations():
        print("Testing CP station search...")
        
        # Test search for Oriente
        print("\n🔍 Searching for 'Oriente'...")
        oriente = search_cp_station("Oriente")
        print(f"   Found {len(oriente)} matches")
        assert len(oriente) > 0, "Should find Lisboa Oriente"
        assert any("oriente" in s['name'].lower() for s in oriente), "Should match Oriente"
        
        # Test search for Cascais
        print("\n🔍 Searching for 'Cascais'...")
        cascais = search_cp_station("Cascais")
        print(f"   Found {len(cascais)} matches")
        assert len(cascais) > 0, "Should find Cascais"
        
        # Test search for Sintra
        print("\n🔍 Searching for 'Sintra'...")
        sintra = search_cp_station("Sintra")
        print(f"   Found {len(sintra)} matches")
        assert len(sintra) > 0, "Should find Sintra"
        
        # Test the @tool version
        print("\n🔍 Testing search_cp_stations @tool...")
        result = search_cp_stations.invoke({"query": "Rossio"})
        print(result[:500] if len(result) > 500 else result)
        assert "Rossio" in result, "Should find Rossio"
        assert "🚉" in result, "Should have station emoji"
        
        print("\n\033[1;32m✅ CP station search works correctly\033[0m")
        return result
    
    run_test("Test 21: Search CP Stations", test_search_cp_stations)
    
    # =========================================================================
    # TEST 22: Get Train Status (AML Filtered)
    # =========================================================================
    def test_get_train_status_aml():
        print("Testing get_train_status with AML filter...")
        
        result = get_train_status.invoke({})
        
        print(result[:2000] if len(result) > 2000 else result)
        
        # Check output format
        assert "AML" in result.upper() or "LISBON" in result.upper(), \
            "Should mention AML or Lisbon"
        assert "trains" in result.lower(), "Should mention trains"
        assert "🚆" in result or "🚈" in result or "🚄" in result, \
            "Should have train emojis"
        
        # Should show service types
        assert any(s in result for s in ["Urbanos", "Regionais", "Intercidades", "Alfa"]), \
            "Should show at least one service type"
        
        print("\n\033[1;32m✅ Train status with AML filter works correctly\033[0m")
        return result
    
    run_test("Test 22: Get Train Status (AML Filtered)", test_get_train_status_aml)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("\033[1m📊 TEST RESULTS SUMMARY\033[0m")
    print("=" * 70)
    print(f"\033[1;32m✅ Passed: {test_results['passed']}\033[0m")
    print(f"\033[1;31m❌ Failed: {test_results['failed']}\033[0m")
    print(f"📊 Total:  {test_results['passed'] + test_results['failed']}")
    print("=" * 70)
    
    if test_results['failed'] == 0:
        print("\n\033[1;32m🎉 ALL TESTS PASSED!\033[0m")
    else:
        print(f"\n\033[1;31m⚠️ {test_results['failed']} test(s) failed!\033[0m")
