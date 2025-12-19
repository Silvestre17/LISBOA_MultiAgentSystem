# ==========================================================================
# Master Thesis - Dados Abertos Smart Tool
#   - André Filipe Gomes Silvestre, 2025
# 
#   Semantic search over Open Data metadata with dynamic GeoJSON fetching.
#   - 15 second timeout for requests
#   - Retry logic with exponential backoff
#   - GeoJSON validation
# ==========================================================================

import json
import requests
import pandas as pd
import math
import logging
from typing import Optional, Dict, Any
from langchain_core.tools import tool
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request configuration
REQUEST_TIMEOUT = 15  # seconds
MAX_RETRIES = 3
BACKOFF_FACTOR = 2

# Load metadata once into memory for speed
try:
    with open(Config.PATH_DADOS_ABERTOS_METADATA, 'r', encoding='utf-8') as f:
        DATASETS_METADATA = json.load(f)
    # Convert to simple list of dicts for easier searching
    DF_METADATA = pd.DataFrame(DATASETS_METADATA)
    logger.info(f"Loaded {len(DF_METADATA)} datasets from Dados Abertos")
except Exception as e:
    logger.error(f"Error loading Dados Abertos metadata: {e}")
    DF_METADATA = pd.DataFrame()

def is_valid_geojson(data: Any) -> bool:
    """
    Validates if data is valid GeoJSON.
    
    Args:
        data: Data to validate.
        
    Returns:
        bool: True if valid GeoJSON structure.
    """
    if not isinstance(data, dict):
        return False
    
    if "type" not in data:
        return False
    
    valid_types = ["FeatureCollection", "Feature", "Point", "LineString", 
                  "Polygon", "MultiPoint", "MultiLineString", "MultiPolygon", 
                  "GeometryCollection"]
    
    return data["type"] in valid_types

def fetch_geojson_with_retry(url: str) -> Optional[Dict[str, Any]]:
    """
    Fetches GeoJSON from URL with retry logic and timeout.
    
    Args:
        url (str): URL to fetch from.
        
    Returns:
        Optional[Dict]: GeoJSON data if successful, None otherwise.
        
    Notes:
        - Uses 15 second timeout per request
        - Implements exponential backoff (2s, 4s, 8s)
        - Validates GeoJSON structure
    """
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Fetching GeoJSON (attempt {attempt + 1}/{MAX_RETRIES}): {url}")
            
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            # Parse JSON
            data = response.json()
            
            # Validate GeoJSON
            if not is_valid_geojson(data):
                logger.error("Invalid GeoJSON structure")
                return None
            
            logger.info(f"Successfully fetched GeoJSON with {len(data.get('features', []))} features")
            return data
            
        except requests.exceptions.Timeout:
            wait_time = BACKOFF_FACTOR ** attempt
            logger.warning(f"Timeout after {REQUEST_TIMEOUT}s. Retrying in {wait_time}s...")
            if attempt < MAX_RETRIES - 1:
                import time
                time.sleep(wait_time)
                
        except requests.exceptions.RequestException as e:
            wait_time = BACKOFF_FACTOR ** attempt
            logger.warning(f"Request error: {e}")
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying in {wait_time}s...")
                import time
                time.sleep(wait_time)
            else:
                logger.error(f"Failed after {MAX_RETRIES} attempts")
                
        except json.JSONDecodeError:
            logger.error("Response is not valid JSON")
            return None
    
    return None

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates distance between two points (in km)."""
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

@tool
def find_dataset_and_query(query_theme: str, user_lat: float = None, user_lon: float = None, max_results: int = 5) -> str:
    """
    Search for open data datasets in Lisbon and dynamically fetch GeoJSON data.
    Returns locations near the user if coordinates are provided.
    
    Args:
        query_theme (str): Keyword to search (e.g., 'farmácias', 'metro', 'jardins', 'wifi').
        user_lat (float, optional): User's latitude for proximity search.
        user_lon (float, optional): User's longitude for proximity search.
        max_results (int): Maximum number of results to return (default: 5).

    Returns:
        str: Text summary of findings with names, addresses, and distances.
        
    Notes:
        - Always fetches fresh data from the portal (no cached files)
        - 15 second timeout per request with retry logic
        - Automatically filters results by proximity if coordinates provided
        
    Example:
        >>> find_dataset_and_query("farmácias", user_lat=38.7223, user_lon=-9.1393)
        "Found 3 pharmacies near you:
         1. Farmácia Central (0.2 km) - Rua Augusta 123
         2. Farmácia São Paulo (0.5 km) - Av. Liberdade 45
         ..."
    """
    if DF_METADATA.empty:
        return "❌ Error: Metadata not loaded."

    # 1. Search metadata (case-insensitive)
    logger.info(f"Searching for: {query_theme}")
    match = DF_METADATA[
        DF_METADATA['title'].str.contains(query_theme, case=False, na=False) | 
        DF_METADATA['description'].str.contains(query_theme, case=False, na=False)
    ]
    
    if match.empty:
        return f"❌ No datasets found for: {query_theme}"
    
    # Pick the first/best match
    dataset = match.iloc[0]
    title = dataset['title']
    stable_url = dataset.get('stable_url')
    
    if not stable_url or stable_url == "N/A":
        return f"❌ Dataset found but no URL available: {title}"
    
    logger.info(f"Found dataset: {title}")
    
    # 2. Fetch GeoJSON dynamically (with 15s timeout and retry)
    geojson_data = fetch_geojson_with_retry(stable_url)
    
    if not geojson_data:
        return f"❌ Failed to fetch data from: {title}\nURL: {stable_url}"
    
    # 3. Extract features
    features = geojson_data.get('features', [])
    
    if not features:
        return f"✓ Dataset '{title}' loaded but contains no features."
    
    logger.info(f"Loaded {len(features)} features from {title}")
    
    # 4. Process features and calculate distances if coordinates provided
    results = []
    
    for feature in features:
        try:
            properties = feature.get('properties', {})
            geometry = feature.get('geometry', {})
            
            # Extract coordinates
            coords = geometry.get('coordinates', [])
            if not coords or len(coords) < 2:
                continue
            
            # Handle different geometry types
            if geometry.get('type') == 'Point':
                lon, lat = coords[0], coords[1]
            else:
                # For MultiPoint, LineString, etc., take first coordinate
                lon, lat = coords[0][0], coords[0][1]
            
            # Calculate distance if user coordinates provided
            distance = None
            if user_lat is not None and user_lon is not None:
                distance = haversine_distance(user_lat, user_lon, lat, lon)
            
            # Extract name/address from properties
            name = properties.get('name') or properties.get('designacao') or properties.get('Nome') or 'N/A'
            address = properties.get('address') or properties.get('morada') or properties.get('Morada') or ''
            
            results.append({
                'name': name,
                'address': address,
                'lat': lat,
                'lon': lon,
                'distance': distance,
                'properties': properties
            })
            
        except Exception as e:
            logger.warning(f"Error processing feature: {e}")
            continue
    
    # 5. Sort by distance if coordinates provided
    if user_lat is not None and user_lon is not None and results:
        results = [r for r in results if r['distance'] is not None]
        results.sort(key=lambda x: x['distance'])
        results = results[:max_results]
    else:
        results = results[:max_results]
    
    # 6. Format response
    if not results:
        return f"✓ Dataset '{title}' loaded with {len(features)} features, but couldn't extract location data."
    
    response = f"✓ Found {len(results)} results from '{title}':\n\n"
    
    for i, result in enumerate(results, 1):
        response += f"{i}. {result['name']}\n"
        if result['address']:
            response += f"   📍 {result['address']}\n"
        if result['distance'] is not None:
            response += f"   📏 Distance: {result['distance']:.2f} km\n"
        response += f"   🗺️ Coordinates: {result['lat']:.6f}, {result['lon']:.6f}\n"
        response += "\n"
    
    return response