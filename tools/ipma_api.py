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

import os
import sys
import logging
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
REQUEST_TIMEOUT = 10  # seconds

# ==========================================================================
# IPMA API Endpoints
# ==========================================================================
IPMA_WARNINGS_URL = "https://api.ipma.pt/open-data/forecast/warnings/warnings_www.json"
IPMA_FORECAST_URL = "https://api.ipma.pt/open-data/forecast/meteorology/cities/daily/{global_id}.json"

# Weather type mapping (IPMA codes to descriptions)
WEATHER_TYPES = {
    0: "No information",
    1: "Clear sky",
    2: "Partly cloudy",
    3: "Sunny intervals",
    4: "Cloudy",
    5: "Cloudy (high clouds)",
    6: "Showers",
    7: "Light showers",
    8: "Heavy showers",
    9: "Rain",
    10: "Light rain",
    11: "Heavy rain",
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
    27: "Cloudy"
}

# Warning level colors and severity
WARNING_LEVELS = {
    "green": {"emoji": "🟢", "severity": 0, "description": "No warning"},
    "yellow": {"emoji": "🟡", "severity": 1, "description": "Be aware"},
    "orange": {"emoji": "🟠", "severity": 2, "description": "Be prepared"},
    "red": {"emoji": "🔴", "severity": 3, "description": "Take action"}
}

# Wind direction mapping
WIND_DIRECTIONS = {
    "N": "North", "NE": "Northeast", "E": "East", "SE": "Southeast",
    "S": "South", "SW": "Southwest", "W": "West", "NW": "Northwest",
    "": "Variable"
}


# ==========================================================================
# Helper Functions
# ==========================================================================

def fetch_json(url: str) -> Optional[Dict[str, Any]]:
    """
    Fetches JSON data from a URL with timeout handling.
    
    Args:
        url (str): URL to fetch from.
        
    Returns:
        Optional[Dict]: JSON data if successful, None otherwise.
    """
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
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
                   Other codes: 'BGC' (Bragança), 'VCT' (Viana do Castelo), etc.

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
    
    # Filter warnings for Lisbon area and non-green levels
    active_warnings = []
    for warning in data:
        warning_area = warning.get('idAreaAviso', '')
        level = warning.get('awarenessLevelID', 'green').lower()
        
        # Skip if not for requested area or if green (no warning)
        if warning_area != area or level == 'green':
            continue
        
        active_warnings.append(warning)
    
    if not active_warnings:
        return f"✅ No active weather warnings for Lisbon ({area}).\n\n🌤️ Weather conditions are normal."
    
    # Sort by severity (red > orange > yellow)
    active_warnings.sort(
        key=lambda x: WARNING_LEVELS.get(x.get('awarenessLevelID', 'green').lower(), {}).get('severity', 0),
        reverse=True
    )
    
    # Format response
    response = f"⚠️ Active Weather Warnings for Lisbon:\n\n"
    
    for i, warning in enumerate(active_warnings, 1):
        level = warning.get('awarenessLevelID', 'unknown').lower()
        level_info = WARNING_LEVELS.get(level, {"emoji": "⚪", "description": "Unknown"})
        
        warning_type = warning.get('awarenessTypeName', 'Unknown')
        text = warning.get('text', '')
        start_time = warning.get('startTime', 'N/A')
        end_time = warning.get('endTime', 'N/A')
        
        # Format times
        try:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            time_str = f"{start_dt.strftime('%b %d, %H:%M')} to {end_dt.strftime('%b %d, %H:%M')}"
        except (ValueError, AttributeError):
            time_str = f"{start_time} to {end_time}"
        
        response += f"{level_info['emoji']} {warning_type.upper()} ({level_info['description']})\n"
        response += f"   ⏰ {time_str}\n"
        if text:
            response += f"   📝 {text}\n"
        response += "\n"
    
    response += "💡 Check IPMA.pt for detailed information."
    
    return response


@tool
def get_weather_forecast(days: int = 3) -> str:
    """
    Gets the daily weather forecast for Lisbon from IPMA.
    
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
    
    response = f"🌤️ Weather Forecast for Lisbon\n"
    response += f"{'=' * 40}\n"
    response += f"📅 Updated: {update_time}\n\n"
    
    for day in forecast_data:
        date = day.get('forecastDate', 'N/A')
        t_min = day.get('tMin', 'N/A')
        t_max = day.get('tMax', 'N/A')
        precip_prob = day.get('precipitaProb', '0')
        wind_dir = day.get('predWindDir', '')
        weather_type = day.get('idWeatherType', 0)
        
        # Format output
        formatted_date = format_date(date)
        weather_desc = get_weather_description(weather_type)
        wind_desc = WIND_DIRECTIONS.get(wind_dir, wind_dir)
        precip_text = precipitation_to_text(precip_prob)
        
        # Weather emoji based on type
        if weather_type in [1, 2, 3]:
            emoji = "☀️"
        elif weather_type in [4, 5, 25, 27]:
            emoji = "☁️"
        elif weather_type in [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
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
        response += f"   💧 Rain: {precip_text} ({precip_prob}%)\n"
        if wind_desc:
            response += f"   💨 Wind: {wind_desc}\n"
        response += "\n"
    
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
        
        weather_desc = get_weather_description(weather_type)
        
        response += f"📅 Today ({today.get('forecastDate', 'N/A')}):\n"
        response += f"   🌡️ Temperature: {t_min}°C to {t_max}°C\n"
        response += f"   🌤️ Conditions: {weather_desc}\n"
        response += f"   💧 Rain probability: {precip_prob}%\n\n"
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
                warning_type = w.get('awarenessTypeName', 'Unknown')
                response += f"   {level_info['emoji']} {warning_type}\n"
        else:
            response += "✅ No active weather warnings.\n"
    else:
        response += "⚠️ Could not fetch weather warnings.\n"
    
    return response


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 IPMA Weather API Tool Test\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")
    
    # Test 1: Weather Summary
    print("\n\033[1m🌤️ Test 1: Current Weather Summary\033[0m")
    print("-" * 40)
    result = get_current_weather_summary.invoke({})
    print(result)
    
    # Test 2: Weather Warnings
    print("\n\033[1m⚠️ Test 2: Weather Warnings\033[0m")
    print("-" * 40)
    result = get_weather_warnings.invoke({"area": "LSB"})
    print(result)
    
    # Test 3: 5-day Forecast
    print("\n\033[1m📅 Test 3: 5-Day Forecast\033[0m")
    print("-" * 40)
    result = get_weather_forecast.invoke({"days": 5})
    print(result)
