# ==========================================================================
# Master Thesis - IPMA Weather API Tool
#   - André Filipe Gomes Silvestre, 20240502
# 
#   Real-time weather data from IPMA (Instituto Português do Mar e da Atmosfera).
#   Features:
#     - Weather warnings (up to 3 days)
#     - Daily forecast (up to 5 days)
#     - Human-readable output
# 
#   API Documentation: https://api.ipma.pt/
# ==========================================================================

# Required libraries:
# pip install requests langchain-core

import logging
import os
from datetime import datetime
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

# Request configuration
REQUEST_TIMEOUT = 10  # seconds

# ==========================================================================
# IPMA API Endpoints
# ==========================================================================

# -----------------------------------------------------------------------------
# Weather Warnings (up to 3 days)
# -----------------------------------------------------------------------------
# Endpoint: https://api.ipma.pt/open-data/forecast/warnings/warnings_www.json
# 
# Response Fields:
#   - text: Warning description (only filled for yellow, orange, red levels)
#   - awarenessTypeName: Warning type (e.g., "Trovoada", "Agitação Marítima", 
#                        "Precipitação", "Vento", "Nevoeiro", "Neve", 
#                        "Tempo Frio", "Tempo Quente")
#   - awarenessLevelID: Warning level/color ("green", "yellow", "orange", "red")
#                       Note: Only non-green warnings are actual alerts
#   - startTime: Warning start datetime (ISO format)
#   - endTime: Warning end datetime (ISO format)
#   - idAreaAviso: Area identifier (see distrits-islands.json for codes)
#
# Example Response:
#   [{"text": "", "awarenessTypeName": "Agitação Marítima", "idAreaAviso": "BGC",
#     "startTime": "2021-03-25T07:25:00", "awarenessLevelID": "green", 
#     "endTime": "2021-03-28T07:00:00"}, ...]
# -----------------------------------------------------------------------------
IPMA_WARNINGS_URL = "https://api.ipma.pt/open-data/forecast/warnings/warnings_www.json"

# -----------------------------------------------------------------------------
# Daily Weather Forecast (up to 5 days) by Location
# -----------------------------------------------------------------------------
# Endpoint: https://api.ipma.pt/open-data/forecast/meteorology/cities/daily/{globalIdLocal}.json
# Note: Only daily data available. Updates hourly.
#
# Response Fields:
#   - forecastDate: Forecast date (YYYY-MM-DD)
#   - dataUpdate: File update timestamp (hourly refresh)
#   - globalIdLocal: Location identifier (see distrits-islands.json)
#   - idWeatherType: Weather type code (see weather-type-classe.json)
#   - tMin: Daily minimum temperature (°C)
#   - tMax: Daily maximum temperature (°C)
#   - classWindSpeed: Wind intensity class (see wind-speed-daily-classe.json)
#   - predWindDir: Predominant wind direction (N, NE, E, SE, S, SW, W, NW)
#   - precipitaProb: Precipitation probability (%)
#   - classPrecInt: Precipitation intensity class (see precipitation-classe.json)
#   - latitude: Location latitude
#   - longitude: Location longitude
#
# Example Response:
#   {"owner": "IPMA", "country": "PT", "globalIdLocal": 1110600,
#    "dataUpdate": "2018-01-26T09:02:03",
#    "data": [{"precipitaProb": "0.0", "tMin": "7.6", "tMax": "13.3", 
#              "predWindDir": "N", "idWeatherType": 2, "classWindSpeed": 2,
#              "classPrecInt": 0, "forecastDate": "2018-01-26", 
#              "latitude": "38.8", "longitude": "-9.1"}, ...]}
# -----------------------------------------------------------------------------
IPMA_FORECAST_URL = "https://api.ipma.pt/open-data/forecast/meteorology/cities/daily/{global_id}.json"

# -----------------------------------------------------------------------------
# Aggregated Daily Forecast (All Locations for a Specific Day)
# -----------------------------------------------------------------------------
# Endpoint: https://api.ipma.pt/open-data/forecast/meteorology/cities/daily/hp-daily-forecast-day{idDay}.json
# 
# This endpoint returns forecast data for ALL districts/islands at once,
# aggregated by day. Useful for getting a Portugal-wide overview.
#
# Parameters:
#   - idDay: 0=today, 1=tomorrow, 2=day after tomorrow
#
# Response Fields (same as daily forecast):
#   - globalIdLocal: Location identifier
#   - idWeatherType: Weather type code
#   - tMin, tMax: Temperature range
#   - precipitaProb: Precipitation probability
#   - predWindDir: Wind direction
#   - classWindSpeed: Wind speed class
# -----------------------------------------------------------------------------
IPMA_FORECAST_AGGREGATED_URL = "https://api.ipma.pt/open-data/forecast/meteorology/cities/daily/hp-daily-forecast-day{id_day}.json"

# Weather type mapping (IPMA codes to descriptions)
# Source: https://api.ipma.pt/open-data/weather-type-classe.json
WEATHER_TYPES = {
    -99: "--",
    0: "No information",
    1: "Clear sky",
    2: "Partly cloudy",
    3: "Sunny intervals",
    4: "Cloudy",
    5: "Cloudy (High cloud)",
    6: "Showers/rain",
    7: "Light showers/rain",
    8: "Heavy showers/rain",
    9: "Rain/showers",
    10: "Light rain",
    11: "Heavy rain/showers",
    12: "Intermittent rain",
    13: "Intermittent light rain",
    14: "Intermittent heavy rain",
    15: "Drizzle",
    16: "Mist",
    17: "Fog",
    18: "Snow",
    19: "Thunderstorms",
    20: "Showers and thunderstorms",
    21: "Hail",
    22: "Frost",
    23: "Rain and thunderstorms",
    24: "Convective clouds",
    25: "Partly cloudy",
    26: "Fog",
    27: "Cloudy",
    28: "Snow showers",
    29: "Rain and snow",
    30: "Rain and snow"
}

# Warning level colors and severity
# Source: https://www.ipma.pt/pt/enciclopedia/otempo/sam/
WARNING_LEVELS = {
    "green": {"emoji": "🟢", "severity": 0, "description": "No warning"},
    "yellow": {"emoji": "🟡", "severity": 1, "description": "Be aware"},
    "orange": {"emoji": "🟠", "severity": 2, "description": "Be prepared"},
    "red": {"emoji": "🔴", "severity": 3, "description": "Take action"}
}

# Warning types mapping (Portuguese to English with specific emojis)
# Source: https://api.ipma.pt/open-data/forecast/warnings/warnings_www.json
WARNING_TYPES = {
    "Precipitação": {"en": "Precipitation", "emoji": "🌧️"},
    "Trovoada": {"en": "Thunderstorm", "emoji": "⛈️"},
    "Agitação Marítima": {"en": "Rough Sea", "emoji": "🌊"},
    "Vento": {"en": "Wind", "emoji": "💨"},
    "Nevoeiro": {"en": "Fog", "emoji": "🌫️"},
    "Neve": {"en": "Snow", "emoji": "❄️"},
    "Tempo Frio": {"en": "Cold Weather", "emoji": "🥶"},
    "Tempo Quente": {"en": "Hot Weather", "emoji": "🥵"}
}

# Wind direction mapping
WIND_DIRECTIONS = {
    "N": "North", "NE": "Northeast", "E": "East", "SE": "Southeast",
    "S": "South", "SW": "Southwest", "W": "West", "NW": "Northwest",
    "": "Variable"
}

# Wind Speed Classes
# Source: https://api.ipma.pt/open-data/wind-speed-daily-classe.json
WIND_SPEED_CLASSES = {
    -99: "--",
    1: "Weak",
    2: "Moderate",
    3: "Strong",
    4: "Very strong"
}

# Precipitation Intensity Classes
# Source: https://api.ipma.pt/open-data/precipitation-classe.json
PRECIPITATION_INTENSITY_CLASSES = {
    -99: "--",
    0: "No precipitation",
    1: "Weak",
    2: "Moderate",
    3: "Strong"
}


# ==========================================================================
# Helper Functions with Optimizations
# ==========================================================================

# Import optimization utilities for caching and connection pooling
try:
    from agent.utils.optimization import TTLCache, http_pool, weather_cache
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    weather_cache = None


def fetch_json(url: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
    """
    Fetches JSON data from a URL with timeout handling.
    Uses connection pooling and optional caching for performance.
    
    Args:
        url (str): URL to fetch from.
        use_cache (bool): Whether to use cached results (default True).
        
    Returns:
        Optional[Dict]: JSON data if successful, None otherwise.
    """
    # Check cache first (5 minute TTL for weather data)
    if use_cache and OPTIMIZATION_AVAILABLE and weather_cache:
        import hashlib
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cached_result = weather_cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for {url}")
            return cached_result
    
    try:
        # Use pooled connection if available
        if OPTIMIZATION_AVAILABLE:
            response = http_pool.get(url, timeout=REQUEST_TIMEOUT)
        else:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
        
        response.raise_for_status()
        data = response.json()
        
        # Cache the result
        if use_cache and OPTIMIZATION_AVAILABLE and weather_cache:
            weather_cache.set(cache_key, data, ttl=300)  # 5 minutes
        
        return data
    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching {url}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        return None
    except ValueError:
        logger.error("Invalid JSON response")
        return None


def get_weather_description(weather_type_id: int) -> str:
    """
    Converts IPMA weather type ID to human-readable description.
    
    Args:
        weather_type_id (int): IPMA weather type code.
        
    Returns:
        str: Weather description.
    """
    return WEATHER_TYPES.get(weather_type_id, f"Unknown ({weather_type_id})")


def get_wind_speed_description(class_wind_speed: int) -> str:
    """
    Converts IPMA wind speed class to human-readable description.
    
    Args:
        class_wind_speed (int): IPMA wind speed class code.
        
    Returns:
        str: Wind speed description.
    """
    return WIND_SPEED_CLASSES.get(class_wind_speed, f"Unknown ({class_wind_speed})")


def get_precipitation_intensity_description(class_prec_int: int) -> str:
    """
    Converts IPMA precipitation intensity class to human-readable description.
    
    Args:
        class_prec_int (int): IPMA precipitation intensity class code.
        
    Returns:
        str: Precipitation intensity description.
    """
    return PRECIPITATION_INTENSITY_CLASSES.get(class_prec_int, f"Unknown ({class_prec_int})")


def format_date(date_str: str) -> str:
    """
    Formats a date string for display.
    
    Args:
        date_str (str): Date in YYYY-MM-DD format.
        
    Returns:
        str: Formatted date (e.g., 'Monday, Dec 27').
    """
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%A, %b %d")
    except ValueError:
        return date_str


def precipitation_to_text(prob: str) -> str:
    """
    Converts precipitation probability to qualitative text.
    
    Args:
        prob (str): Precipitation probability as string percentage.
        
    Returns:
        str: Qualitative description.
    """
    try:
        p = float(prob)
        if p == 0:
            return "No rain expected"
        elif p < 20:
            return "Very unlikely"
        elif p < 40:
            return "Unlikely"
        elif p < 60:
            return "Possible"
        elif p < 80:
            return "Likely"
        else:
            return "Very likely"
    except (ValueError, TypeError):
        return "Unknown"


# ==========================================================================
# LangChain Tools
# ==========================================================================

@tool
def get_weather_warnings(area: str = "LSB") -> str:
    """
    Gets active weather warnings for Lisbon from IPMA.
    Only returns yellow, orange, or red warnings (ignores green/no warning).
    
    Args:
        area (str): Area code for warnings (default: 'LSB' for Lisbon).
                   Other codes: 'BGC' (Bragança), 'VCT' (Viana do Castelo), 
                   'PTO' (Porto), 'FAR' (Faro), etc.
                   See: https://api.ipma.pt/open-data/distrits-islands.json

    Returns:
        str: Formatted list of active weather warnings with severity and timing.
        
    Examples:
        >>> get_weather_warnings()
        >>> get_weather_warnings("LSB")
    """
    data = fetch_json(IPMA_WARNINGS_URL)
    
    if not data:
        return "❌ Failed to fetch weather warnings from IPMA."
    
    if not isinstance(data, list):
        return "❌ Unexpected response format from IPMA warnings API."
    
    # Filter warnings for requested area and non-green levels
    active_warnings = []
    for warning in data:
        warning_area = warning.get('idAreaAviso', '')
        level = warning.get('awarenessLevelID', 'green').lower()
        
        # Skip if not for requested area or if green (no warning)
        if warning_area != area or level == 'green':
            continue
        
        active_warnings.append(warning)
    
    if not active_warnings:
        return f"✅ No active weather warnings for area '{area}'.\n\n🌤️ Weather conditions are normal."
    
    # Sort by severity (red > orange > yellow), then by start time
    active_warnings.sort(
        key=lambda x: (
            -WARNING_LEVELS.get(x.get('awarenessLevelID', 'green').lower(), {}).get('severity', 0),
            x.get('startTime', '')
        )
    )
    
    # Format response
    response = f"⚠️ Active Weather Warnings ({area}):\n"
    response += "=" * 40 + "\n\n"
    
    for warning in active_warnings:
        level = warning.get('awarenessLevelID', 'unknown').lower()
        level_info = WARNING_LEVELS.get(level, {"emoji": "⚪", "description": "Unknown"})
        
        warning_type_pt = warning.get('awarenessTypeName', 'Unknown')
        warning_type_info = WARNING_TYPES.get(warning_type_pt, {"en": warning_type_pt, "emoji": "⚠️"})
        
        text = warning.get('text', '')
        start_time = warning.get('startTime', 'N/A')
        end_time = warning.get('endTime', 'N/A')
        
        # Format times
        try:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            time_str = f"{start_dt.strftime('%b %d, %H:%M')} → {end_dt.strftime('%b %d, %H:%M')}"
        except (ValueError, AttributeError):
            time_str = f"{start_time} → {end_time}"
        
        response += f"{level_info['emoji']} {warning_type_info['emoji']} {warning_type_info['en'].upper()}\n"
        response += f"   Level: {level_info['description']}\n"
        response += f"   ⏰ {time_str}\n"
        if text:
            response += f"   📝 {text}\n"
        response += "\n"
    
    response += "� Fonte: [IPMA](https://www.ipma.pt) - Instituto Português do Mar e da Atmosfera"
    
    return response


@tool
def get_weather_forecast(days: int = 3) -> str:
    """
    Gets the daily weather forecast for Lisbon from IPMA.
    
    ⚠️ IMPORTANT LIMITATIONS:
    - Maximum forecast range: 5 DAYS from today
    - If user asks for a date BEYOND 5 days, you MUST inform them that
      weather data is only available for the next 5 days
    - NEVER invent or hallucinate weather data for dates outside this range
    
    Args:
        days (int): Number of days to forecast (1-5, default: 3).

    Returns:
        str: Formatted weather forecast with temperatures, precipitation, 
             and conditions for each day.
        
    Examples:
        >>> get_weather_forecast()
        >>> get_weather_forecast(5)
    """

    url = IPMA_FORECAST_URL.format(global_id=Config.LISBON_GLOBAL_ID)
    data = fetch_json(url)
    
    if not data:
        return "❌ Failed to fetch weather forecast from IPMA."
    
    forecast_data = data.get('data', [])
    
    if not forecast_data:
        return "❌ No forecast data available."
    
    # Limit to requested days
    days = min(max(1, days), len(forecast_data), 5)
    forecast_data = forecast_data[:days]
    
    # Get update time
    update_time = data.get('dataUpdate', 'N/A')
    
    response = "🌤️ Weather Forecast for Lisbon\n"
    response += f"{'=' * 40}\n"
    response += f"📅 Updated: {update_time}\n\n"
    
    for day in forecast_data:
        date = day.get('forecastDate', 'N/A')
        t_min = day.get('tMin', 'N/A')
        t_max = day.get('tMax', 'N/A')
        precip_prob = day.get('precipitaProb', '0')
        wind_dir = day.get('predWindDir', '')
        weather_type = day.get('idWeatherType', 0)
        wind_speed_class = day.get('classWindSpeed', -99)
        precip_intensity_class = day.get('classPrecInt', -99)
        
        # Format output
        formatted_date = format_date(date)
        weather_desc = get_weather_description(weather_type)
        wind_dir_desc = WIND_DIRECTIONS.get(wind_dir, wind_dir)
        wind_speed_desc = get_wind_speed_description(wind_speed_class)
        precip_text = precipitation_to_text(precip_prob)
        precip_intensity_desc = get_precipitation_intensity_description(precip_intensity_class)
        
        # Weather emoji based on type
        if weather_type in [1, 2, 3]:
            emoji = "☀️"
        elif weather_type in [4, 5, 25, 27]:
            emoji = "☁️"
        elif weather_type in [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 28, 29, 30]:
            emoji = "🌧️"
        elif weather_type in [18]:
            emoji = "❄️"
        elif weather_type in [19, 20, 23]:
            emoji = "⛈️"
        elif weather_type in [16, 17, 26]:
            emoji = "🌫️"
        else:
            emoji = "🌡️"
        
        response += f"{emoji} {formatted_date}\n"
        response += f"   🌡️ {t_min}°C to {t_max}°C\n"
        response += f"   🌤️ {weather_desc}\n"
        response += f"   💧 Rain: {precip_text} ({precip_prob}%)"
        if precip_intensity_class not in [-99, 0]:
            response += f" | Intensity: {precip_intensity_desc}"
        response += "\n"
        if wind_dir_desc:
            response += f"   💨 Wind: {wind_dir_desc} ({wind_speed_desc})\n"
        response += "\n"
    
    return response


@tool
def get_portugal_weather_overview(day: int = 0) -> str:
    """
    Gets weather forecast for all Portugal locations for a specific day.
    
    Returns an aggregated view of weather across all districts and islands,
    useful for comparing conditions across the country.
    
    Args:
        day (int): Day index: 0=today, 1=tomorrow, 2=day after tomorrow.

    Returns:
        str: Formatted weather overview for all locations in Portugal.
        
    Examples:
        >>> get_portugal_weather_overview()  # Today's weather across Portugal
        >>> get_portugal_weather_overview(1)  # Tomorrow's weather
    """
    # Validate day parameter
    if day not in [0, 1, 2]:
        return "❌ Invalid day parameter. Use 0 (today), 1 (tomorrow), or 2 (day after tomorrow)."
    
    day_names = {0: "Today", 1: "Tomorrow", 2: "Day After Tomorrow"}
    url = IPMA_FORECAST_AGGREGATED_URL.format(id_day=day)
    
    data = fetch_json(url)
    
    if not data:
        return "❌ Failed to fetch aggregated forecast from IPMA."
    
    forecast_data = data.get('data', [])
    
    if not forecast_data:
        return "❌ No aggregated forecast data available."
    
    # Get update time and forecast date
    update_time = data.get('dataUpdate', 'N/A')
    forecast_date = data.get('forecastDate', 'N/A')
    
    response = f"🇵🇹 Portugal Weather Overview - {day_names[day]}\n"
    response += f"{'=' * 50}\n"
    response += f"📅 Forecast Date: {forecast_date}\n"
    response += f"🔄 Updated: {update_time}\n"
    response += f"📊 Locations: {len(forecast_data)}\n\n"
    
    # Find Lisbon in the data (globalIdLocal = 1110600)
    lisbon_data = None
    for loc in forecast_data:
        if loc.get('globalIdLocal') == Config.LISBON_GLOBAL_ID:
            lisbon_data = loc
            break
    
    if lisbon_data:
        response += "🏙️ **LISBON (Focus Area)**\n"
        response += "-" * 30 + "\n"
        
        weather_type = lisbon_data.get('idWeatherType', 0)
        weather_desc = get_weather_description(weather_type)
        t_min = lisbon_data.get('tMin', 'N/A')
        t_max = lisbon_data.get('tMax', 'N/A')
        precip_prob = lisbon_data.get('precipitaProb', '0')
        wind_dir = lisbon_data.get('predWindDir', '')
        wind_dir_desc = WIND_DIRECTIONS.get(wind_dir, wind_dir)
        
        response += f"   🌤️ {weather_desc}\n"
        response += f"   🌡️ {t_min}°C to {t_max}°C\n"
        response += f"   💧 Precipitation: {precip_prob}%\n"
        response += f"   💨 Wind: {wind_dir_desc}\n\n"
    
    # Summary statistics across all locations
    temps_min = [float(loc.get('tMin', 0)) for loc in forecast_data if loc.get('tMin')]
    temps_max = [float(loc.get('tMax', 0)) for loc in forecast_data if loc.get('tMax')]
    
    if temps_min and temps_max:
        response += "📈 **Portugal Summary**\n"
        response += "-" * 30 + "\n"
        response += f"   🌡️ Temperature range: {min(temps_min):.0f}°C to {max(temps_max):.0f}°C\n"
        response += f"   📍 Coldest: {min(temps_min):.0f}°C | Warmest: {max(temps_max):.0f}°C\n"
    
    return response


@tool
def get_current_weather_summary() -> str:
    """
    Gets a quick summary of today's weather and any active warnings for Lisbon.
    Combines forecast and warnings into a single response.

    Returns:
        str: Combined weather summary including today's forecast and any warnings.
        
    Example:
        >>> get_current_weather_summary()
    """
    # Get today's forecast
    url = IPMA_FORECAST_URL.format(global_id=Config.LISBON_GLOBAL_ID)
    forecast_data = fetch_json(url)
    
    # Get warnings
    warnings_data = fetch_json(IPMA_WARNINGS_URL)
    
    response = "🌤️ Lisbon Weather Summary\n"
    response += "=" * 40 + "\n\n"
    
    # Today's forecast
    if forecast_data and forecast_data.get('data'):
        today = forecast_data['data'][0]
        t_min = today.get('tMin', 'N/A')
        t_max = today.get('tMax', 'N/A')
        weather_type = today.get('idWeatherType', 0)
        precip_prob = today.get('precipitaProb', '0')
        wind_dir = today.get('predWindDir', '')
        wind_speed_class = today.get('classWindSpeed', -99)
        precip_intensity_class = today.get('classPrecInt', -99)
        
        weather_desc = get_weather_description(weather_type)
        wind_dir_desc = WIND_DIRECTIONS.get(wind_dir, wind_dir)
        wind_speed_desc = get_wind_speed_description(wind_speed_class)
        precip_intensity_desc = get_precipitation_intensity_description(precip_intensity_class)
        
        response += f"📅 Today ({today.get('forecastDate', 'N/A')}):\n"
        response += f"   🌡️ Temperature: {t_min}°C to {t_max}°C\n"
        response += f"   🌤️ Conditions: {weather_desc}\n"
        response += f"   💧 Rain probability: {precip_prob}%"
        if precip_intensity_class not in [-99, 0]:
            response += f" ({precip_intensity_desc})"
        response += "\n"
        if wind_dir_desc:
            response += f"   💨 Wind: {wind_dir_desc} ({wind_speed_desc})\n"
        response += "\n"
    else:
        response += "❌ Could not fetch today's forecast.\n\n"
    
    # Active warnings
    if warnings_data and isinstance(warnings_data, list):
        active = [w for w in warnings_data 
                  if w.get('idAreaAviso') == Config.LISBON_AREA_AVISO 
                  and w.get('awarenessLevelID', 'green').lower() != 'green']
        
        if active:
            response += "⚠️ Active Warnings:\n"
            for w in active[:3]:  # Max 3 warnings
                level = w.get('awarenessLevelID', 'unknown').lower()
                level_info = WARNING_LEVELS.get(level, {"emoji": "⚪"})
                warning_type_pt = w.get('awarenessTypeName', 'Unknown')
                warning_type_info = WARNING_TYPES.get(warning_type_pt, {"en": warning_type_pt, "emoji": "⚠️"})
                response += f"   {level_info['emoji']} {warning_type_info['emoji']} {warning_type_info['en']}\n"
        else:
            response += "✅ No active weather warnings.\n"
    else:
        response += "⚠️ Could not fetch weather warnings.\n"
    
    return response


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    import time
    
    print("\033[1m" + "=" * 70 + "\033[0m")
    print("\033[1m🧪 IPMA Weather API Tool - Comprehensive Test Suite\033[0m")
    print("\033[1m" + "=" * 70 + "\033[0m")
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Target Area: Lisbon (LSB) | Global ID: {Config.LISBON_GLOBAL_ID}")
    print("\033[1m" + "=" * 70 + "\033[0m")
    
    # Track test results
    tests_passed = 0
    tests_failed = 0
    
    # =========================================================================
    # Test 1: API Connectivity - Forecast Endpoint
    # =========================================================================
    print("\n\033[1m📡 Test 1: API Connectivity - Forecast Endpoint\033[0m")
    print("-" * 50)
    
    url = IPMA_FORECAST_URL.format(global_id=Config.LISBON_GLOBAL_ID)
    print(f"   URL: {url}")
    
    start_time = time.time()
    data = fetch_json(url)
    elapsed = time.time() - start_time
    
    if data:
        print(f"\033[1;32m   ✅ Connection successful ({elapsed:.2f}s)\033[0m")
        print(f"   📊 Data points received: {len(data.get('data', []))}")
        print(f"   🕐 Last update: {data.get('dataUpdate', 'N/A')}")
        tests_passed += 1
    else:
        print("\033[1;31m   ❌ Connection failed\033[0m")
        tests_failed += 1
    
    # =========================================================================
    # Test 2: API Connectivity - Warnings Endpoint
    # =========================================================================
    print("\n\033[1m📡 Test 2: API Connectivity - Warnings Endpoint\033[0m")
    print("-" * 50)
    
    print(f"   URL: {IPMA_WARNINGS_URL}")
    
    start_time = time.time()
    warnings_data = fetch_json(IPMA_WARNINGS_URL)
    elapsed = time.time() - start_time
    
    if warnings_data:
        print(f"\033[1;32m   ✅ Connection successful ({elapsed:.2f}s)\033[0m")
        print(f"   📊 Total warnings in response: {len(warnings_data)}")
        
        # Count warnings by area
        lisbon_warnings = [w for w in warnings_data if w.get('idAreaAviso') == 'LSB']
        active_lisbon = [w for w in lisbon_warnings if w.get('awarenessLevelID', 'green').lower() != 'green']
        print(f"   🏙️ Lisbon (LSB) entries: {len(lisbon_warnings)} total, {len(active_lisbon)} active")
        tests_passed += 1
    else:
        print("\033[1;31m   ❌ Connection failed\033[0m")
        tests_failed += 1
    
    # =========================================================================
    # Test 3: Weather Forecast Tool (3 days - default)
    # =========================================================================
    print("\n\033[1m🌤️ Test 3: Weather Forecast Tool (3 days - default)\033[0m")
    print("-" * 50)
    
    try:
        result = get_weather_forecast.invoke({})
        print(result)
        if "❌" not in result:
            print("\n\033[1;32m   ✅ Tool executed successfully\033[0m")
            tests_passed += 1
        else:
            print("\n\033[1;31m   ❌ Tool returned error\033[0m")
            tests_failed += 1
    except Exception as e:
        print(f"\033[1;31m   ❌ Exception: {e}\033[0m")
        tests_failed += 1
    
    # =========================================================================
    # Test 4: Weather Forecast Tool (5 days - max)
    # =========================================================================
    print("\n\033[1m📅 Test 4: Weather Forecast Tool (5 days - max)\033[0m")
    print("-" * 50)
    
    try:
        result = get_weather_forecast.invoke({"days": 5})
        print(result)
        if "❌" not in result:
            print("\n\033[1;32m   ✅ Tool executed successfully\033[0m")
            tests_passed += 1
        else:
            print("\n\033[1;31m   ❌ Tool returned error\033[0m")
            tests_failed += 1
    except Exception as e:
        print(f"\033[1;31m   ❌ Exception: {e}\033[0m")
        tests_failed += 1
    
    # =========================================================================
    # Test 5: Weather Warnings Tool (Lisbon)
    # =========================================================================
    print("\n\033[1m⚠️ Test 5: Weather Warnings Tool (Lisbon - LSB)\033[0m")
    print("-" * 50)
    
    try:
        result = get_weather_warnings.invoke({"area": "LSB"})
        print(result)
        if "❌" not in result:
            print("\n\033[1;32m   ✅ Tool executed successfully\033[0m")
            tests_passed += 1
        else:
            print("\n\033[1;31m   ❌ Tool returned error\033[0m")
            tests_failed += 1
    except Exception as e:
        print(f"\033[1;31m   ❌ Exception: {e}\033[0m")
        tests_failed += 1
    
    # =========================================================================
    # Test 6: Weather Warnings Tool (Porto - different area)
    # =========================================================================
    print("\n\033[1m⚠️ Test 6: Weather Warnings Tool (Porto - PTO)\033[0m")
    print("-" * 50)
    
    try:
        result = get_weather_warnings.invoke({"area": "PTO"})
        print(result)
        if "❌" not in result:
            print("\n\033[1;32m   ✅ Tool executed successfully\033[0m")
            tests_passed += 1
        else:
            print("\n\033[1;31m   ❌ Tool returned error\033[0m")
            tests_failed += 1
    except Exception as e:
        print(f"\033[1;31m   ❌ Exception: {e}\033[0m")
        tests_failed += 1
    
    # =========================================================================
    # Test 7: Current Weather Summary Tool
    # =========================================================================
    print("\n\033[1m📊 Test 7: Current Weather Summary Tool\033[0m")
    print("-" * 50)
    
    try:
        result = get_current_weather_summary.invoke({})
        print(result)
        if "❌" not in result or "Could not fetch" not in result:
            print("\n\033[1;32m   ✅ Tool executed successfully\033[0m")
            tests_passed += 1
        else:
            print("\n\033[1;31m   ❌ Tool returned error\033[0m")
            tests_failed += 1
    except Exception as e:
        print(f"\033[1;31m   ❌ Exception: {e}\033[0m")
        tests_failed += 1
    
    # =========================================================================
    # Test 8: Helper Functions Validation
    # =========================================================================
    print("\n\033[1m🔧 Test 8: Helper Functions Validation\033[0m")
    print("-" * 50)
    
    helper_tests_passed = 0
    
    # Test weather description
    desc = get_weather_description(6)
    expected = "Showers/rain"
    if desc == expected:
        print(f"   ✅ get_weather_description(6) = '{desc}'")
        helper_tests_passed += 1
    else:
        print(f"   ❌ get_weather_description(6) = '{desc}' (expected '{expected}')")
    
    # Test unknown weather type
    desc = get_weather_description(999)
    if "Unknown" in desc:
        print(f"   ✅ get_weather_description(999) = '{desc}' (handles unknown)")
        helper_tests_passed += 1
    else:
        print("   ❌ get_weather_description(999) should return 'Unknown'")
    
    # Test wind speed description
    desc = get_wind_speed_description(2)
    expected = "Moderate"
    if desc == expected:
        print(f"   ✅ get_wind_speed_description(2) = '{desc}'")
        helper_tests_passed += 1
    else:
        print(f"   ❌ get_wind_speed_description(2) = '{desc}' (expected '{expected}')")
    
    # Test precipitation intensity
    desc = get_precipitation_intensity_description(3)
    expected = "Strong"
    if desc == expected:
        print(f"   ✅ get_precipitation_intensity_description(3) = '{desc}'")
        helper_tests_passed += 1
    else:
        print(f"   ❌ get_precipitation_intensity_description(3) = '{desc}' (expected '{expected}')")
    
    # Test precipitation to text
    text = precipitation_to_text("85")
    if text == "Very likely":
        print(f"   ✅ precipitation_to_text('85') = '{text}'")
        helper_tests_passed += 1
    else:
        print(f"   ❌ precipitation_to_text('85') = '{text}' (expected 'Very likely')")
    
    # Test date formatting
    formatted = format_date("2026-01-13")
    if "Monday" in formatted or "Jan" in formatted:
        print(f"   ✅ format_date('2026-01-13') = '{formatted}'")
        helper_tests_passed += 1
    else:
        print(f"   ❌ format_date('2026-01-13') = '{formatted}'")
    
    if helper_tests_passed == 6:
        print(f"\n\033[1;32m   ✅ All helper functions working correctly ({helper_tests_passed}/6)\033[0m")
        tests_passed += 1
    else:
        print(f"\n\033[1;31m   ❌ Some helper functions failed ({helper_tests_passed}/6)\033[0m")
        tests_failed += 1
    
    # =========================================================================
    # Test 9: Dictionary Completeness Check
    # =========================================================================
    print("\n\033[1m📚 Test 9: Dictionary Completeness Check\033[0m")
    print("-" * 50)
    
    print(f"   WEATHER_TYPES: {len(WEATHER_TYPES)} entries (IDs -99 to 30)")
    print(f"   WARNING_LEVELS: {len(WARNING_LEVELS)} entries (green, yellow, orange, red)")
    print(f"   WARNING_TYPES: {len(WARNING_TYPES)} entries (8 warning categories)")
    print(f"   WIND_DIRECTIONS: {len(WIND_DIRECTIONS)} entries (N, NE, E, SE, S, SW, W, NW, '')")
    print(f"   WIND_SPEED_CLASSES: {len(WIND_SPEED_CLASSES)} entries (IDs -99, 1-4)")
    print(f"   PRECIPITATION_INTENSITY_CLASSES: {len(PRECIPITATION_INTENSITY_CLASSES)} entries (IDs -99, 0-3)")
    
    # Validate counts
    if (len(WEATHER_TYPES) >= 30 and len(WARNING_LEVELS) == 4 and
            len(WARNING_TYPES) == 8 and len(WIND_SPEED_CLASSES) == 5 and
            len(PRECIPITATION_INTENSITY_CLASSES) == 5):
        print("\n\033[1;32m   ✅ All dictionaries complete\033[0m")
        tests_passed += 1
    else:
        print("\n\033[1;31m   ❌ Some dictionaries incomplete\033[0m")
        tests_failed += 1
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "\033[1m" + "=" * 70 + "\033[0m")
    print("\033[1m📊 TEST SUMMARY\033[0m")
    print("\033[1m" + "=" * 70 + "\033[0m")
    
    total_tests = tests_passed + tests_failed
    
    if tests_failed == 0:
        print(f"\n\033[1;32m✅ ALL TESTS PASSED: {tests_passed}/{total_tests}\033[0m")
    else:
        print(f"\n\033[1;32m✅ Tests Passed: {tests_passed}\033[0m")
        print(f"\033[1;31m❌ Tests Failed: {tests_failed}\033[0m")
        print(f"\n\033[1mTotal: {tests_passed}/{total_tests}\033[0m")
    
    print("\n" + "\033[1m" + "=" * 70 + "\033[0m")
