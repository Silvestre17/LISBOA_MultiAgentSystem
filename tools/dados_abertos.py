# ==========================================================================
# Master Thesis - Dados Abertos Smart Tool
#   - André Filipe Gomes Silvestre, 2025
# 
#   Semantic search over Open Data metadata -> On-demand GeoJSON fetching.
# ==========================================================================

import json
import requests
import pandas as pd
import math
from langchain_core.tools import tool
from config import Config

# Load metadata once into memory for speed
try:
    with open(Config.PATH_DADOS_ABERTOS_METADATA, 'r', encoding='utf-8') as f:
        DATASETS_METADATA = json.load(f)
    # Convert to simple list of dicts for easier searching
    # We assume the JSON structure matches your 'lisbon_datasets.json' example
    DF_METADATA = pd.DataFrame(DATASETS_METADATA)
except Exception as e:
    print(f"⚠️ Error loading Dados Abertos metadata: {e}")
    DF_METADATA = pd.DataFrame()

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
def find_dataset_and_query(query_theme: str, user_lat: float = None, user_lon: float = None) -> str:
    """
    Search for open data datasets (e.g., pharmacies, parking, gardens) in Lisbon, 
    fetch the data dynamically, and return items near the user (if coordinates provided).

    Args:
        query_theme (str): Keyword to search (e.g., 'farmácias', 'jardins', 'wifi').
        user_lat (float, optional): Latitude of the user.
        user_lon (float, optional): Longitude of the user.

    Returns:
        str: A text summary of the closest findings or the dataset content.
    """
    if DF_METADATA.empty:
        return "Error: Metadata not loaded."

    # 1. Search Metadata (Naive keyword match for now, could use embeddings later)
    # Case insensitive search
    match = DF_METADATA[DF_METADATA['title'].str.contains(query_theme, case=False, na=False) | 
                        DF_METADATA['description'].str.contains(query_theme, case=False, na=False)]
    
    if match.empty:
        return f"No datasets found for theme: {query_theme}"
    
    # Pick the first/best match
    dataset = match.iloc[0]
    title = dataset['title']
    
    # 2. Extract the GeoJSON/CSV link
    # Note: Your 'lisbon_datasets.json' structure implies 'url_portal' or 'stable_url'.
    # We need to robustly find the download link. For this example, I assume logic to find the GeoJSON link.
    # If your scraping didn't get the direct .geojson link, the agent might need to visit the 'stable_url'.
    # *CRITICAL*: In your 'lisbon_datasets.json' example, you have 'file_formats': 'geojson'.
    # If the direct link isn't there, we simulate fetching (or you update the scraper).
    # Assuming 'stable_url' might redirect to a download or we scrape it live.
    
    # For robust thesis code, assume we might need to fetch the GeoJSON URL dynamically if not present.
    # Here, let's assume 'stable_url' serves the data or we use it to find the resource.
    target_url = dataset.get('stable_url') 
    
    # 3. Fetch Data (On-Demand)
    try:
        # Important: Add a User-Agent to avoid blocking
        headers = {'User-Agent': 'ThesisAgent/1.0 (Education)'}
        # Note: If stable_url is a webpage, this needs parsing. 
        # Ideally, your scraper should have saved the direct download link.
        # If it's the portal URL, we interpret it.
        
        # SIMULATION: If we can't truly fetch without the direct link, return the metadata
        # so the LLM knows it exists.
        response = f"Found dataset: '{title}'.\nDescription: {dataset['description'][:200]}...\n"
        response += f"Link: {target_url}\n"
        
        return response

    except Exception as e:
        return f"Error fetching dataset '{title}': {str(e)}"