# ==========================================================================
# Master Thesis - Dados Abertos Smart Tool
#   - André Filipe Gomes Silvestre, 20240502
# 
#   Semantic search over Lisboa Aberta Open Data with dynamic GeoJSON fetching.
#   Features:
#     - Keyword-based dataset discovery
#     - Dynamic GeoJSON fetching with retry logic
#     - Proximity-based filtering with Haversine distance
#     - Multiple specialized query functions
# 
#   Data Source: https://dados.gov.pt/pt/datasets/?geozone=pt%3Aconcelho%3A1106 / https://dados.cm-lisboa.pt/
# ==========================================================================

# Required libraries:
# pip install requests pandas langchain-core

import json
import math
import time
import logging
import os
import sys
from typing import Optional, Dict, Any, List, Tuple

import requests
import pandas as pd
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
# Data Loading
# ==========================================================================

def load_metadata() -> pd.DataFrame:
    """
    Loads the Dados Abertos metadata from the local JSON file.
    
    Returns:
        pd.DataFrame: Metadata DataFrame with dataset information.
    """
    try:
        with open(Config.PATH_DADOS_ABERTOS_METADATA, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        logger.info(f"\033[1;32m✅ Loaded {len(df)} datasets from Dados Abertos\033[0m")
        return df
    except FileNotFoundError:
        logger.error(f"\033[1;31m❌ Metadata file not found: {Config.PATH_DADOS_ABERTOS_METADATA}\033[0m")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"\033[1;31m❌ Error loading metadata: {e}\033[0m")
        return pd.DataFrame()

# Load metadata once at module import
DF_METADATA = load_metadata()


# ==========================================================================
# Helper Functions
# ==========================================================================

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
    
    valid_types = [
        "FeatureCollection", "Feature", "Point", "LineString", 
        "Polygon", "MultiPoint", "MultiLineString", "MultiPolygon", 
        "GeometryCollection"
    ]
    
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
            logger.info(f"Fetching GeoJSON (attempt {attempt + 1}/{MAX_RETRIES}): {url[:80]}...")
            
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            
            if not is_valid_geojson(data):
                logger.error("Invalid GeoJSON structure")
                return None
            
            feature_count = len(data.get('features', []))
            logger.info(f"\033[1;32m✅ Fetched {feature_count} features\033[0m")
            return data
            
        except requests.exceptions.Timeout:
            wait_time = BACKOFF_FACTOR ** attempt
            logger.warning(f"Timeout after {REQUEST_TIMEOUT}s. Retrying in {wait_time}s...")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait_time)
                
        except requests.exceptions.RequestException as e:
            wait_time = BACKOFF_FACTOR ** attempt
            logger.warning(f"Request error: {e}")
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed after {MAX_RETRIES} attempts")
                
        except json.JSONDecodeError:
            logger.error("Response is not valid JSON")
            return None
    
    return None


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates the great-circle distance between two points on Earth.
    
    Args:
        lat1 (float): Latitude of point 1.
        lon1 (float): Longitude of point 1.
        lat2 (float): Latitude of point 2.
        lon2 (float): Longitude of point 2.
        
    Returns:
        float: Distance in kilometers.
    """
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2 +
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
        math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def extract_coordinates(geometry: Dict) -> Optional[Tuple[float, float]]:
    """
    Extracts latitude and longitude from GeoJSON geometry.
    
    Args:
        geometry (Dict): GeoJSON geometry object.
        
    Returns:
        Optional[Tuple[float, float]]: (latitude, longitude) or None.
    """
    if not geometry:
        return None
    
    coords = geometry.get('coordinates', [])
    if not coords:
        return None
    
    geo_type = geometry.get('type', '')
    
    if geo_type == 'Point' and len(coords) >= 2:
        return (coords[1], coords[0])  # GeoJSON is [lon, lat]
    elif geo_type in ['MultiPoint', 'LineString'] and coords and len(coords[0]) >= 2:
        return (coords[0][1], coords[0][0])
    elif geo_type == 'Polygon' and coords and coords[0] and len(coords[0][0]) >= 2:
        return (coords[0][0][1], coords[0][0][0])
    
    return None


def extract_name(properties: Dict) -> str:
    """
    Extracts the best available name from feature properties.
    
    Args:
        properties (Dict): GeoJSON feature properties.
        
    Returns:
        str: Best available name or 'N/A'.
    """
    name_fields = [
        'name', 'nome', 'Nome', 'designacao', 'Designacao', 
        'DESIGNACAO', 'designação', 'título', 'titulo', 'title',
        'NOME', 'NAME', 'INF_NOME', 'NOME_PARQU', 'NOME_JARDIM'
    ]
    
    for field in name_fields:
        if field in properties and properties[field]:
            return str(properties[field])
    
    return "N/A"


def extract_address(properties: Dict) -> str:
    """
    Extracts the best available address from feature properties.
    
    Args:
        properties (Dict): GeoJSON feature properties.
        
    Returns:
        str: Best available address or empty string.
    """
    address_fields = [
        'address', 'morada', 'Morada', 'MORADA', 'endereco',
        'rua', 'Rua', 'local', 'Local', 'localizacao', 'INF_MORADA'
    ]
    
    for field in address_fields:
        if field in properties and properties[field]:
            return str(properties[field])
    
    return ""


def search_datasets(query: str) -> pd.DataFrame:
    """
    Searches metadata for datasets matching the query.
    
    Args:
        query (str): Search term(s).
        
    Returns:
        pd.DataFrame: Matching datasets.
    """
    if DF_METADATA.empty:
        return pd.DataFrame()
    
    query_lower = query.lower()
    
    # Search in title and description
    mask = (
        DF_METADATA['title'].str.lower().str.contains(query_lower, na=False) |
        DF_METADATA['description'].str.lower().str.contains(query_lower, na=False)
    )
    
    return DF_METADATA[mask]


# ==========================================================================
# LangChain Tools
# ==========================================================================

@tool
def find_nearby_services(
    service_type: str,
    user_lat: float = None,
    user_lon: float = None,
    max_results: int = 5
) -> str:
    """
    Search for public services in Lisbon (pharmacies, hospitals, schools, etc.) 
    and optionally filter by proximity to user location.
    
    Args:
        service_type (str): Type of service to search (e.g., 'farmácias', 'hospitais', 
                           'escolas', 'metro', 'wifi', 'jardins', 'parques', 'fontanários').
        user_lat (float, optional): User's latitude for proximity filtering.
        user_lon (float, optional): User's longitude for proximity filtering.
        max_results (int): Maximum number of results to return (default: 5).

    Returns:
        str: Formatted list of services with names, addresses, and distances.
        
    Examples:
        >>> find_nearby_services("farmácias", user_lat=38.7223, user_lon=-9.1393)
        >>> find_nearby_services("wifi")
        >>> find_nearby_services("jardins", user_lat=38.72, user_lon=-9.14, max_results=3)
    """
    if DF_METADATA.empty:
        return "❌ Error: Metadata not loaded. Check if lisbon_datasets_clean.json exists."

    # Search for matching datasets
    matches = search_datasets(service_type)
    
    if matches.empty:
        # Try alternative search terms
        alternatives = {
            'pharmacy': 'farmácia', 'hospital': 'hospital', 'school': 'escola',
            'park': 'jardim', 'garden': 'jardim', 'wifi': 'wifi', 'metro': 'metro',
            'fountain': 'fontanário', 'parking': 'estacionamento'
        }
        alt_term = alternatives.get(service_type.lower(), service_type)
        matches = search_datasets(alt_term)
    
    if matches.empty:
        return f"❌ No datasets found for: '{service_type}'\n💡 Try: farmácias, hospitais, escolas, jardins, wifi, metro, fontanários"
    
    # Pick the best matching dataset
    dataset = matches.iloc[0]
    title = dataset['title']
    stable_url = dataset.get('stable_url')
    
    if not stable_url or stable_url == "N/A":
        return f"❌ Dataset '{title}' found but no URL available."
    
    # Fetch GeoJSON data
    geojson_data = fetch_geojson_with_retry(stable_url)
    
    if not geojson_data:
        return f"❌ Failed to fetch data from: {title}\n   URL: {stable_url}"
    
    features = geojson_data.get('features', [])
    
    if not features:
        return f"✓ Dataset '{title}' loaded but contains no features."
    
    # Process features
    results = []
    
    for feature in features:
        try:
            properties = feature.get('properties', {})
            geometry = feature.get('geometry', {})
            
            coords = extract_coordinates(geometry)
            if not coords:
                continue
            
            lat, lon = coords
            
            # Calculate distance if user coordinates provided
            distance = None
            if user_lat is not None and user_lon is not None:
                distance = haversine_distance(user_lat, user_lon, lat, lon)
            
            name = extract_name(properties)
            address = extract_address(properties)
            
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
    
    # Sort by distance if coordinates provided
    if user_lat is not None and user_lon is not None and results:
        results = [r for r in results if r['distance'] is not None]
        results.sort(key=lambda x: x['distance'])
    
    results = results[:max_results]
    
    if not results:
        return f"✓ Dataset '{title}' loaded ({len(features)} features) but couldn't extract location data."
    
    # Format response
    response = f"📍 Found {len(results)} results from '{title}':\n\n"
    
    for i, r in enumerate(results, 1):
        response += f"{i}. {r['name']}\n"
        if r['address']:
            response += f"   📍 {r['address']}\n"
        if r['distance'] is not None:
            response += f"   📏 {r['distance']:.2f} km away\n"
        response += f"   🗺️ ({r['lat']:.6f}, {r['lon']:.6f})\n\n"
    
    return response


@tool
def list_available_datasets(category: str = None) -> str:
    """
    Lists all available open data datasets from Lisboa Aberta.
    Optionally filter by category keyword.
    
    Args:
        category (str, optional): Filter datasets by category/keyword 
                                 (e.g., 'saúde', 'educação', 'ambiente', 'transportes').

    Returns:
        str: Formatted list of available datasets with titles and descriptions.
        
    Examples:
        >>> list_available_datasets()
        >>> list_available_datasets("saúde")
        >>> list_available_datasets("ambiente")
    """
    if DF_METADATA.empty:
        return "❌ Error: Metadata not loaded."
    
    df = DF_METADATA.copy()
    
    if category:
        df = search_datasets(category)
        if df.empty:
            return f"❌ No datasets found for category: '{category}'"
    
    # Format response
    response = f"📂 Available Datasets ({len(df)} total):\n\n"
    
    for i, (_, row) in enumerate(df.head(20).iterrows(), 1):
        title = row.get('title', 'N/A')
        desc = row.get('description', '')
        if desc and len(desc) > 100:
            desc = desc[:100] + "..."
        
        response += f"{i}. {title}\n"
        if desc:
            response += f"   {desc}\n"
        response += "\n"
    
    if len(df) > 20:
        response += f"... and {len(df) - 20} more datasets.\n"
        response += "💡 Use a category filter to narrow results."
    
    return response


@tool
def get_dataset_details(dataset_name: str) -> str:
    """
    Gets detailed information about a specific dataset including 
    schema inspection and sample data.
    
    Args:
        dataset_name (str): Name or keyword to identify the dataset.

    Returns:
        str: Detailed information about the dataset including available fields.
        
    Example:
        >>> get_dataset_details("farmácias")
    """
    if DF_METADATA.empty:
        return "❌ Error: Metadata not loaded."
    
    matches = search_datasets(dataset_name)
    
    if matches.empty:
        return f"❌ No dataset found matching: '{dataset_name}'"
    
    dataset = matches.iloc[0]
    title = dataset['title']
    description = dataset.get('description', 'N/A')
    stable_url = dataset.get('stable_url', 'N/A')
    last_updated = dataset.get('last_updated', 'N/A')
    
    response = f"📊 Dataset: {title}\n"
    response += f"{'=' * 50}\n\n"
    response += f"📝 Description: {description}\n\n"
    response += f"🔗 URL: {stable_url}\n"
    response += f"📅 Last Updated: {last_updated}\n\n"
    
    # Try to fetch and inspect schema
    if stable_url and stable_url != "N/A":
        geojson_data = fetch_geojson_with_retry(stable_url)
        
        if geojson_data:
            features = geojson_data.get('features', [])
            response += f"📦 Total Features: {len(features)}\n\n"
            
            if features:
                # Inspect first feature's properties
                sample = features[0].get('properties', {})
                response += "🔍 Available Fields:\n"
                for key, value in list(sample.items())[:15]:
                    val_type = type(value).__name__
                    response += f"   • {key} ({val_type})\n"
                
                if len(sample) > 15:
                    response += f"   ... and {len(sample) - 15} more fields\n"
    
    return response


def _search_place_in_datasets_logic(query: str, max_results: int = 5) -> str:
    """
    Core logic for searching places in open datasets.
    This is a regular function (not a tool) that can be called directly.
    
    Args:
        query (str): The name of the place to find.
        max_results (int): Maximum number of results to return.
    
    Returns:
        str: Formatted string with found places or empty string if nothing found.
    """
    if DF_METADATA.empty:
        return ""
    
    query_lower = query.lower()
    found_places = []
    
    # 1. Identify potential datasets
    # Strategy: Map common keywords to specific datasets + default keyword search
    
    potential_datasets = pd.DataFrame()
    
    # Comprehensive Mapping of Keywords to Datasets
    keyword_map = {
        # Shopping & Commerce
        'shopping': ['Centros Comerciais', 'Mercados', 'Quiosques e Bancas', 'Lojas Sociais de Lisboa'],
        'centro comercial': ['Centros Comerciais'],
        'mercado': ['Mercados', 'Feiras'],
        'feira': ['Feiras'],
        'loja': ['Lojas Sociais de Lisboa', 'Comercialização de Hardware e Software e Serviços', 'Quiosques e Bancas'],
        'quiosque': ['Quiosques e Bancas'],
        
        # Health & Emergency
        'hospital': ['Hospitais Públicos', 'Hospitais Privados', 'Hospitais Militares', 'Centros de Saúde', 'Prestação de Cuidados'],
        'saude': ['Centros de Saúde', 'Hospitais Públicos', 'Hospitais Privados'],
        'clinica': ['Hospitais Privados', 'Prestação de Cuidados'],
        'farmacia': ['Farmácias e Parafarmácias'],
        'bombeiros': ['Bombeiros'],
        'policia': ['Polícia Municipal', 'Polícia de Segurança Pública', 'GNR', 'Defesa e Segurança'],
        'seguranca': ['Polícia Municipal', 'Polícia de Segurança Pública'],
        'proteccao civil': ['Protecção Civil', 'Lisboa. Pontos de encontro - Emergência'],
        
        # Education
        'escola': ['Escolas Públicas - 1º Ciclo', 'Escolas Públicas - 2º e 3º Ciclo', 'Escolas Públicas - Secundário', 'Escolas Públicas - Pré-Escolar', 'Agrupamentos de Escolas de Lisboa', 'Escolas Privadas - 1º Ciclo', 'Escolas Privadas - 2º e 3º Ciclo', 'Escolas Privadas - Secundárias', 'Equipamentos Escolares'],
        'colegio': ['Escolas Privadas - 1º Ciclo', 'Escolas Privadas - 2º e 3º Ciclo', 'Escolas Privadas - Secundárias'],
        'universidade': ['Ensino Superior', 'Faculdades, Escolas e Institutos'],
        'faculdade': ['Ensino Superior', 'Faculdades, Escolas e Institutos'],
        'instituto': ['Institutos', 'Instituições'],
        'creche': ['Escolas Públicas - Pré-Escolar', 'Escolas Privadas - Pré-Escolar'],
        
        # Culture & Tourism
        'museu': ['Museus', 'Museus, Bibliotecas e Arquivos'],
        'biblioteca': ['Bibliotecas Arquivos e Centros de Documentação', 'Medidas de desempenho da Rede de Bibliotecas de Lisboa'],
        'teatro': ['Teatros', 'Artes Performativas - Teatro, Dança e Música'],
        'cinema': ['Cinemas', 'Cinema e Video'],
        'galeria': ['Galerias de Arte', 'Galerias Municipais', 'Espaços e Bairros Criativos'],
        'monumento': ['Monumentos Nacionais', 'Imóveis e Monumentos de Interesse Público', 'Estatuária', 'Património Mundial'],
        'miradouro': ['Miradouros'],
        'igreja': ['Arquitetura Religiosa', 'Localização e identificação das Casas Religiosas de Lisboa existentes em 2015'],
        'hotel': ['Capacidade de Alojamento', 'Alojamento'],
        'turismo': ['Postos de Turismo', 'Turismo Náutico'],
        'wi-fi': ['Rede LoRa'], # Approximate
        
        # Outdoors & Leisure
        'jardim': ['Jardins - Parques Urbanos', 'Grandes Parques e Jardins de Lisboa', 'Espaços Verdes'],
        'parque': ['Grandes Parques e Jardins de Lisboa', 'Jardins - Parques Urbanos', 'Parques Infantis', 'Parques de Merendas', 'Parques Caninos'],
        'praia': [], # Not many open datasets for beaches in CML directly besides river ones
        'desporto': ['Instalações Desportivas', 'Centros Desportivos', 'Equipamentos de Fitness ao Ar Livre\u200b', 'Programa Desporto Mexe Comigo'],
        'piscina': ['Instalações Desportivas', 'Programa de Apoio à Natação Curricular'],
        
        # Services & Amenities
        'wc': ['Instalações Sanitárias', 'Instalações Sanitárias Públicas Automáticas', 'Balneários'],
        'banheiro': ['Instalações Sanitárias', 'Instalações Sanitárias Públicas Automáticas'],
        'estacionamento': ['Parques de estacionamento na via pública', 'EMEL - Parques de estacionamento na via pública', 'Lugares de estacionamento na via pública para residentes ou público em geral', 'Zonas reguladas de estacionamento na via pública'],
        'embaixada': ['Embaixadas'],
        'ctt': [], # Post offices
        'cemiterio': ['Cemitérios'],
        'loja cidadao': ['Loja do Cidadão'],
        'camara': ['CM Lisboa - Paços do Concelho', 'CM Lisboa - Atendimento', 'Juntas de Freguesia'],
        'junta': ['Juntas de Freguesia'],
        
        # Streets & Locations
        'rua': ['Toponímia de Lisboa', 'Topónimos', 'Eixos de Via'],
        'avenida': ['Toponímia de Lisboa', 'Topónimos'],
        'praca': ['Toponímia de Lisboa', 'Topónimos'],
        'largo': ['Toponímia de Lisboa', 'Topónimos'],
        'bairro': ['Bairros e Zonas de Intervenção Prioritária', 'Localização e identificação das Casas Religiosas de Lisboa existentes em 2015'], # Proxy
    }

    # Add matched datasets from mapping
    for key, titles in keyword_map.items():
        if key in query_lower:
            for title in titles:
                matches = DF_METADATA[DF_METADATA['title'] == title]
                if not matches.empty:
                    potential_datasets = pd.concat([potential_datasets, matches])

    # Keywords to ignore (stopwords)
    ignore_words = {'de', 'do', 'da', 'em', 'para', 'com', 'the', 'in', 'at', 'lisboa', 'lisbon', 'perto', 'near', 'proximo', 'onde', 'fica', 'existe', 'ha'}
    tokens = [w for w in query_lower.split() if w not in ignore_words and len(w) > 3]
    
    
    if not tokens:
        tokens = [query_lower]
        
    for token in tokens:
        matches = search_datasets(token)
        if not matches.empty:
            potential_datasets = pd.concat([potential_datasets, matches])
    
    # Also handle specific cases where category might be implied
    if any(x in query_lower for x in ['shopping', 'centro comercial', 'mall']):
        matches = search_datasets('comerciais')
        potential_datasets = pd.concat([potential_datasets, matches])
    
    if potential_datasets.empty:
        return ""
        
    potential_datasets = potential_datasets.drop_duplicates(subset='stable_url')
    
    # Limit to top 5 datasets to ensure responsiveness
    for idx, dataset in potential_datasets.head(5).iterrows():
        title = dataset['title']
        url = dataset.get('stable_url')
        
        if not url or url == "N/A":
            continue
            
        # Optimization: Skip likely irrelevant large datasets based on title
        if any(x in title.lower() for x in ['limites', 'rede', 'carta', 'zonamento']):
            continue
            
        data = fetch_geojson_with_retry(url)
        if not data:
            continue
            
        features = data.get('features', [])
        for feature in features:
            properties = feature.get('properties', {})
            
            # Extract name and address
            name = extract_name(properties)
            address = extract_address(properties)
            
            # Check match: Name contains query token OR query contains Name
            if name == "N/A":
                continue
                
            match_score = 0
            name_lower = name.lower()
            
            # Full match check
            if query_lower in name_lower or name_lower in query_lower:
                match_score = 100
            else:
                # Token match check
                matches = sum(1 for t in tokens if t in name_lower)
                if matches > 0:
                    match_score = (matches / len(tokens)) * 100
            
            if match_score > 50:  # Threshold
                # Extract coordinates
                coords = extract_coordinates(feature.get('geometry', {}))
                lat, lon = coords if coords else (None, None)
                
                found_places.append({
                    'title': name,
                    'category': title, # Use dataset title as category
                    'location': address,
                    'lat': lat,
                    'lon': lon,
                    'short_description': f"Found in open data dataset: {title}",
                    'score': match_score
                })
    
    # Deduplicate by name
    unique_places = {}
    for p in found_places:
        if p['title'] not in unique_places:
            unique_places[p['title']] = p
        else:
            # Keep the one with better info
            if len(p['location']) > len(unique_places[p['title']]['location']):
                unique_places[p['title']] = p
                
    results = sorted(unique_places.values(), key=lambda x: x['score'], reverse=True)[:max_results]
    
    if not results:
        return ""

    # Format output compatible with VisitLisboa style
    output_parts = [f"🏛️ **Found {len(results)} Places in Open Data (Lisboa Aberta):**\n"]
    
    for i, place in enumerate(results, 1):
        output_parts.append(f"{i}. 🏛️ **{place['title']}**")
        output_parts.append(f"   📂 Category: {place['category']}")
        output_parts.append(f"   📝 {place['short_description']}")
        
        if place['location']:
            output_parts.append(f"   📍 {place['location']}")
        if place['lat'] and place['lon']:
            output_parts.append(f"   🗺️ Coordinates: ({place['lat']:.5f}, {place['lon']:.5f})")
            
    return "\n".join(output_parts)


@tool
def find_place_in_datasets(query: str, max_results: int = 5) -> str:
    """
    Searches for a specific place by name across relevant open datasets.
    Useful when standard place search fails but the place might exist in open data catalogs
    (e.g., specific shopping malls, markets, public facilities).

    Args:
        query (str): The name of the place to find (e.g., "Centro Comercial Colombo").
        max_results (int): Maximum number of results to return.

    Returns:
        str: Formatted string with found places or empty string if nothing found.
    """
    return _search_place_in_datasets_logic(query, max_results)


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 Dados Abertos Tool Test\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")
    
    # Test 1: List datasets
    print("\n\033[1m📂 Test 1: List Available Datasets\033[0m")
    print("-" * 40)
    result = list_available_datasets.invoke({"category": "saúde"})
    print(result[:500] + "..." if len(result) > 500 else result)
    
    # Test 2: Find nearby services
    print("\n\033[1m📍 Test 2: Find Nearby Services\033[0m")
    print("-" * 40)
    result = find_nearby_services.invoke({
        "service_type": "farmácias",
        "user_lat": Config.LISBON_LATITUDE,
        "user_lon": Config.LISBON_LONGITUDE,
        "max_results": 3
    })
    print(result)
    
    # Test 3: Get dataset details
    print("\n\033[1m📊 Test 3: Dataset Details\033[0m")
    print("-" * 40)
    result = get_dataset_details.invoke({"dataset_name": "jardins"})
    print(result)