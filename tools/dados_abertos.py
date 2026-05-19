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
#   Usage:
#     > python tools/dados_abertos.py
#       Run the manual Lisboa Aberta dataset-discovery and GeoJSON tool test suite.
#
#   Data Source: https://dados.gov.pt/pt/datasets/?geozone=pt%3Aconcelho%3A1106 / https://dados.cm-lisboa.pt/
# ==========================================================================

# Required libraries:
# pip install requests pandas langchain-core

import json
import logging
import os
import re
import time
import unicodedata
from html import unescape
from urllib.parse import quote_plus
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from langchain_core.tools import tool
import contextlib

try:
    from config import Config
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from config import Config

logger = logging.getLogger(__name__)

try:
    from tools.utils import haversine_distance
except ImportError:
    from utils import haversine_distance

# Request configuration
REQUEST_TIMEOUT = 15  # seconds
MAX_RETRIES = 3
BACKOFF_FACTOR = 2
TRANSIENT_DATASET_UNAVAILABLE_TTL_SECONDS = 15 * 60
_UNAVAILABLE_DATASET_URLS: Dict[str, Dict[str, Any] | str] = {}

_KNOWN_REFERENCE_COORDINATES: Dict[str, Tuple[float, float]] = {
    "rossio": (38.7139, -9.1394),
    "praca dom pedro iv": (38.7139, -9.1394),
    "praça dom pedro iv": (38.7139, -9.1394),
    "central lisbon": (38.7139, -9.1394),
    "central lisboa": (38.7139, -9.1394),
    "lisbon centre": (38.7139, -9.1394),
    "lisbon center": (38.7139, -9.1394),
    "city centre lisbon": (38.7139, -9.1394),
    "city center lisbon": (38.7139, -9.1394),
    "lisbon city centre": (38.7139, -9.1394),
    "lisbon city center": (38.7139, -9.1394),
    "downtown lisbon": (38.7139, -9.1394),
    "centro de lisboa": (38.7139, -9.1394),
    "baixa": (38.7106, -9.1401),
    "baixa lisboa": (38.7106, -9.1401),
    "marques de pombal": (38.7257, -9.1490),
    "marquês de pombal": (38.7257, -9.1490),
    "belem": (38.6975, -9.2063),
    "belém": (38.6975, -9.2063),
    "oriente": (38.7688, -9.0988),
    "rato": (38.7168, -9.1527),
    "saldanha": (38.7351, -9.1457),
    "areeiro": (38.7423, -9.1339),
    "praca do areeiro": (38.7423, -9.1339),
    "praça do areeiro": (38.7423, -9.1339),
    "roma areeiro": (38.7457, -9.1383),
    "roma-areeiro": (38.7457, -9.1383),
    "alvalade": (38.7533, -9.1435),
    "avenida de roma": (38.7474, -9.1396),
    "roma": (38.7474, -9.1396),
    "benfica": (38.7506, -9.2029),
    "santos": (38.7064, -9.1567),
    "alcantara": (38.7067, -9.1741),
    "alcântara": (38.7067, -9.1741),
    "baixa chiado": (38.7106, -9.1401),
    "baixa-chiado": (38.7106, -9.1401),
}


def _normalize_reference_location_name(location_name: str) -> str:
    """Normalize a reference place name for deterministic coordinate lookup."""
    normalized = unicodedata.normalize("NFKD", str(location_name or ""))
    normalized = normalized.encode("ascii", "ignore").decode("ascii").lower()
    return " ".join(normalized.replace("-", " ").split())


def _fold_open_data_text(value: object) -> str:
    """Return accent-insensitive text for Lisboa Aberta matching."""
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    ascii_text = re.sub(r"[^a-z0-9]+", " ", ascii_text)
    return re.sub(r"\s+", " ", ascii_text).strip()


_DATASET_QUERY_STOPWORDS = {
    "a",
    "ao",
    "aos",
    "as",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "near",
    "of",
    "os",
    "para",
    "perto",
    "publica",
    "publicas",
    "publico",
    "publicos",
    "the",
}


def _resolve_reference_coordinates(location_name: str) -> Optional[Tuple[float, float]]:
    """Resolve a named Lisbon reference point without using service datasets.

    Open-data service datasets are useful for finding the target services, but
    they are a poor geocoder for landmarks such as Rossio because the first
    matching service record may be kilometres away. This helper keeps common
    central anchors deterministic and then falls back to the shared location
    resolver before using the older service-data lookup.
    """
    key = _normalize_reference_location_name(location_name)
    if not key:
        return None

    if key in _KNOWN_REFERENCE_COORDINATES:
        return _KNOWN_REFERENCE_COORDINATES[key]

    try:
        from tools.location_resolver import resolve_location_query

        resolved = resolve_location_query(location_name, prefer_city=True, allow_aml=True)
        if resolved.get("success") and resolved.get("lat") is not None and resolved.get("lon") is not None:
            return float(resolved["lat"]), float(resolved["lon"])
    except Exception as exc:
        logger.info("Shared geocoder could not resolve '%s': %s", location_name, exc)

    return None


def _get_unavailable_dataset_reason(url: str) -> Optional[str]:
    """Return cached unavailability reason, expiring transient failures when needed."""
    cached = _UNAVAILABLE_DATASET_URLS.get(url)
    if cached is None:
        return None
    if isinstance(cached, str):
        return cached

    expires_at = cached.get("expires_at")
    if isinstance(expires_at, (int, float)) and expires_at <= time.time():
        _UNAVAILABLE_DATASET_URLS.pop(url, None)
        return None
    return str(cached.get("reason") or "unavailable")


def _mark_dataset_url_unavailable(url: str, reason: str, status_code: Optional[int] = None) -> None:
    """Cache unavailable dataset URLs while allowing transient HTTP failures to recover."""
    cache_entry: Dict[str, Any] = {"reason": reason, "status_code": status_code}
    if status_code == 429 or (status_code is not None and status_code >= 500):
        cache_entry["expires_at"] = time.time() + TRANSIENT_DATASET_UNAVAILABLE_TTL_SECONDS
    _UNAVAILABLE_DATASET_URLS[url] = cache_entry

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
    unavailable_reason = _get_unavailable_dataset_reason(url)
    if unavailable_reason:
        logger.warning("Skipping unavailable Lisboa Aberta dataset URL: %s (%s)", url, unavailable_reason)
        return None

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Fetching GeoJSON (attempt {attempt + 1}/{MAX_RETRIES}): {url[:80]}...")

            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code >= 400:
                reason = f"HTTP {response.status_code}"
                _mark_dataset_url_unavailable(url, reason, response.status_code)
                logger.warning("Lisboa Aberta dataset unavailable: %s -> %s", url, reason)
                return None
            response.raise_for_status()

            data = response.json()

            if not is_valid_geojson(data):
                logger.error("Invalid GeoJSON structure")
                return None

            feature_count = len(data.get('features', []))
            logger.info(f"\033[1;32m✅ Fetched {feature_count} features\033[0m")
            return data

        except requests.exceptions.Timeout:
            reason = "network timeout"
            _mark_dataset_url_unavailable(url, reason, 503)
            logger.warning("Lisboa Aberta dataset temporarily unavailable: %s -> %s", url, reason)
            return None

        except requests.exceptions.RequestException as e:
            reason = "network unavailable"
            _mark_dataset_url_unavailable(url, reason, 503)
            logger.warning("Lisboa Aberta dataset temporarily unavailable: %s -> %s", url, reason)
            logger.debug("Lisboa Aberta request exception for %s: %s", url, e)
            return None

        except json.JSONDecodeError:
            logger.error("Response is not valid JSON")
            return None

    return None


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
    Handles diverse GeoJSON schemas from Lisboa Aberta datasets.

    Args:
        properties (Dict): GeoJSON feature properties.

    Returns:
        str: Best available name or 'N/A'.
    """
    # Priority 1: Direct name fields (most common)
    primary_name_fields = [
        # Standard names
        'name', 'nome', 'Nome', 'NOME', 'NAME',
        # Portuguese variations
        'designacao', 'Designacao', 'DESIGNACAO', 'designação',
        'título', 'titulo', 'title', 'TITLE',
        # Dataset-specific name fields
        'NOME_ESCOLA', 'NOME_PARQU', 'NOME_JARDIM', 'NOME_EQUIP',
        'NOME_HOSPITAL', 'NOME_FARMACIA', 'NOME_MERCADO',
        'NOME_LOCAL', 'NOME_RUA', 'INF_NOME',
        # Other common patterns
        'ENTIDADE', 'entidade', 'Entidade',
        'ESTABELECIMENTO', 'estabelecimento',
        'LOCAL', 'local', 'Local',
    ]

    for field in primary_name_fields:
        if field in properties and properties[field]:
            return str(properties[field]).strip()

    # Priority 2: Composite name construction from descriptive fields
    # Useful for datasets without traditional name fields (e.g., parking, bike stations)
    descriptive_fields = [
        ('TIPO_ESTACIONAMENTO', 'MODELO'),  # Parking
        ('TIPOLOGIA', 'AGRUPAMENTO'),  # Schools
        ('TIPO', 'SUBTIPO'),  # Generic
        ('CATEGORIA', 'SUBCATEGORIA'),  # Categories
        ('EQUIPAMENTO_SERVIDO',),  # Equipment served
    ]

    for field_combo in descriptive_fields:
        parts = []
        for field in field_combo:
            if field in properties and properties[field]:
                parts.append(str(properties[field]).strip())
        if parts:
            return ' - '.join(parts)

    # Priority 3: Fallback to address-based identification
    address_fields = ['MORADA', 'morada', 'Morada', 'address', 'RUA', 'LOCALIZACAO']
    for field in address_fields:
        if field in properties and properties[field]:
            addr = str(properties[field]).strip()
            if len(addr) > 5:  # Only use meaningful addresses
                return f"Local: {addr[:50]}" if len(addr) > 50 else f"Local: {addr}"

    # Priority 4: Use any field containing 'nome' or 'name' (case-insensitive)
    for key, value in properties.items():
        if value and ('nome' in key.lower() or 'name' in key.lower()):
            return str(value).strip()

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


# Semantic expansion mapping for category searches
CATEGORY_SYNONYMS = {
    # Education
    'educação': ['escola', 'universidade', 'faculdade', 'ensino', 'agrupamento', 'creche', 'instituto', 'formação'],
    'education': ['escola', 'universidade', 'faculdade', 'ensino', 'agrupamento', 'creche', 'instituto'],
    'escola': ['escolas', 'secundário', 'ciclo', 'agrupamento', 'pré-escolar'],
    'school': ['escola', 'escolas', 'secundário', 'ciclo'],

    # Health
    'saúde': ['hospital', 'farmácia', 'centro de saúde', 'clínica', 'urgência', 'prestação de cuidados'],
    'health': ['hospital', 'farmácia', 'centro de saúde', 'clínica'],
    'hospital': ['hospitais', 'público', 'privado', 'militar'],
    'farmacia': ['farmácia', 'farmácias', 'farmacias', 'pharmacy', 'pharmacies', 'parafarmácia'],
    'farmacias': ['farmácia', 'farmácias', 'farmacias', 'pharmacy', 'pharmacies', 'parafarmácia'],
    'pharmacy': ['farmácia', 'farmácias', 'farmacias', 'pharmacy', 'pharmacies', 'parafarmácia'],
    'pharmacies': ['farmácia', 'farmácias', 'farmacias', 'pharmacy', 'pharmacies', 'parafarmácia'],

    # Environment
    'ambiente': ['jardim', 'parque', 'espaço verde', 'árvore', 'floresta', 'reciclagem', 'ecoponto', 'pilhão', 'pilhoes', 'papeleira', 'bebedouro'],
    'environment': ['jardim', 'parque', 'espaço verde', 'árvore'],
    'ecopontos': ['ecoponto', 'reciclagem', 'residuos', 'residuo', 'waste', 'recycling'],
    'recycling': ['ecoponto', 'reciclagem', 'residuos', 'residuo'],
    'pilhoes': ['pilhões', 'pilhoes', 'pilhão', 'pilhao', 'pilhas', 'baterias'],
    'pilhão': ['pilhões', 'pilhoes', 'pilhao', 'pilhas', 'baterias'],
    'papeleiras': ['papeleiras', 'papeleira', 'lixo', 'caixote', 'resíduos', 'residuos', 'waste bins'],
    'bebedouros': ['bebedouro', 'bebedouros', 'fontanário', 'fontanarios', 'chafariz', 'elementos de água', 'agua'],
    'parques caninos': ['parques caninos', 'parque canino', 'canino', 'dog park', 'dog parks'],

    # Transport
    'transportes': ['metro', 'autocarro', 'comboio', 'estacionamento', 'bicicleta', 'bicicletas', 'velocípede', 'velocipede', 'mobilidade', 'gira'],
    'transport': ['metro', 'autocarro', 'comboio', 'estacionamento', 'bicicleta', 'velocipede'],
    'estacionamento de bicicletas': ['estacionamento de velocípedes', 'estacionamento de velocipedes', 'velocípede', 'velocipede', 'velocípedes', 'velocipedes', 'bicicleta', 'bicicletas', 'bike parking', 'bicycle parking'],
    'bicicletas': ['estacionamento de velocípedes', 'estacionamento de velocipedes', 'velocípede', 'velocipede', 'bicicleta', 'bicicletas'],

    # Culture
    'cultura': ['museu', 'biblioteca', 'teatro', 'cinema', 'galeria', 'monumento', 'património'],
    'culture': ['museu', 'biblioteca', 'teatro', 'cinema', 'galeria', 'monumento'],

    # Tourism
    'turismo': ['hotel', 'alojamento', 'miradouro', 'monumento', 'posto de turismo'],
    'tourism': ['hotel', 'alojamento', 'miradouro', 'monumento'],

    # Security
    'segurança': ['polícia', 'bombeiros', 'proteção civil', 'emergência', 'pontos de encontro'],
    'security': ['polícia', 'bombeiros', 'proteção civil', 'emergência', 'emergency meeting points'],
    'psp': ['Polícia de Segurança Pública', 'esquadra'],
    'emergencia': ['emergência', 'pontos de encontro', 'ponto de encontro', 'proteção civil', 'emergency meeting points'],
    'pontos de encontro de emergencia': ['pontos de encontro', 'ponto de encontro', 'emergência', 'emergencia', 'proteção civil'],

    # Commerce
    'comércio': ['mercado', 'feira', 'loja', 'centro comercial', 'quiosque'],
    'commerce': ['mercado', 'feira', 'loja', 'centro comercial'],

    # Amenities
    'wc': ['instalações sanitárias', 'instalações sanitárias públicas automáticas', 'sanitários', 'casas de banho', 'toilet', 'restroom'],
    'instalacoes sanitarias': ['instalações sanitárias', 'instalações sanitárias públicas automáticas', 'sanitários', 'wc', 'casas de banho', 'toilet', 'restroom'],
    'sanitarios': ['instalações sanitárias', 'instalações sanitárias públicas automáticas', 'wc', 'casas de banho', 'toilet', 'restroom'],
    'sanitários': ['instalações sanitárias', 'instalações sanitárias públicas automáticas', 'wc', 'casas de banho', 'toilet', 'restroom'],
    'casa de banho': ['instalações sanitárias', 'instalações sanitárias públicas automáticas', 'wc', 'sanitários', 'toilet', 'restroom'],
    'toilet': ['instalações sanitárias', 'instalações sanitárias públicas automáticas', 'wc', 'restroom'],
    'restroom': ['instalações sanitárias', 'instalações sanitárias públicas automáticas', 'wc', 'toilet'],
}


# ==========================================================================
# Category Taxonomy (structured grouping of 168 datasets)
# ==========================================================================

CATEGORY_TAXONOMY = {
    "saúde": {
        "en": "Health",
        "keywords": ["hospital", "farmácia", "centro de saúde", "clínica", "prestação de cuidados", "saúde"],
        "description": "Hospitais, farmácias, centros de saúde, clínicas",
        "description_en": "Hospitals, pharmacies, health centres, clinics",
    },
    "educação": {
        "en": "Education",
        "keywords": ["escola", "universidade", "faculdade", "ensino", "agrupamento", "creche", "instituto", "formação", "educação"],
        "description": "Escolas, universidades, institutos, creches",
        "description_en": "Schools, universities, institutes, nurseries",
    },
    "segurança": {
        "en": "Safety & Emergency",
        "keywords": ["polícia", "bombeiros", "proteção civil", "emergência", "segurança", "defesa", "gnr"],
        "description": "Polícia, bombeiros, proteção civil, emergências",
        "description_en": "Police, firefighters, civil protection, emergencies",
    },
    "cultura": {
        "en": "Culture & Heritage",
        "keywords": ["museu", "biblioteca", "teatro", "cinema", "galeria", "monumento", "património", "cultura", "arquivo"],
        "description": "Museus, bibliotecas, teatros, cinemas, monumentos",
        "description_en": "Museums, libraries, theatres, cinemas, monuments",
    },
    "ambiente": {
        "en": "Environment & Green Spaces",
        "keywords": ["jardim", "parque", "espaço verde", "árvore", "reciclagem", "ecoponto", "ambiente", "floresta"],
        "description": "Jardins, parques, espaços verdes, reciclagem",
        "description_en": "Gardens, parks, green spaces, recycling points",
    },
    "transportes": {
        "en": "Transport & Mobility",
        "keywords": ["metro", "autocarro", "comboio", "estacionamento", "bicicleta", "mobilidade", "gira", "transporte"],
        "description": "Estacionamento, bicicletas, mobilidade urbana",
        "description_en": "Parking, bicycles, GIRA, urban mobility",
    },
    "turismo": {
        "en": "Tourism & Accommodation",
        "keywords": ["hotel", "alojamento", "miradouro", "posto de turismo", "turismo"],
        "description": "Hotéis, alojamento, miradouros, postos de turismo",
        "description_en": "Hotels, local accommodation, viewpoints, tourist offices",
    },
    "comércio": {
        "en": "Commerce & Markets",
        "keywords": ["mercado", "feira", "loja", "centro comercial", "quiosque", "comércio"],
        "description": "Mercados, feiras, centros comerciais, lojas",
        "description_en": "Municipal markets, fairs, shopping centres, shops",
    },
    "serviços": {
        "en": "Public Services",
        "keywords": ["junta", "câmara", "loja do cidadão", "embaixada", "cemitério", "instalações sanitárias", "wc"],
        "description": "Juntas de freguesia, câmara municipal, embaixadas, WC públicos",
        "description_en": "Parish councils, city services, embassies, public toilets",
    },
    "desporto": {
        "en": "Sports & Leisure",
        "keywords": ["desporto", "piscina", "fitness", "instalações desportivas", "recreation"],
        "description": "Instalações desportivas, piscinas, fitness ao ar livre",
        "description_en": "Sports facilities, swimming pools, outdoor fitness",
    },
}


def get_datasets_for_category(category: str) -> pd.DataFrame:
    """
    Returns datasets matching a taxonomy category.

    Args:
        category: Category key (e.g., 'saúde', 'educação') or English name.

    Returns:
        pd.DataFrame: Matching datasets.
    """
    if DF_METADATA.empty:
        return pd.DataFrame()

    category_key = _fold_open_data_text(category)

    # Find matching taxonomy entry
    keywords = []
    for cat_key, cat_info in CATEGORY_TAXONOMY.items():
        if _fold_open_data_text(cat_key) == category_key or _fold_open_data_text(cat_info["en"]) == category_key:
            keywords = cat_info["keywords"]
            break

    if not keywords:
        # Fallback to search_datasets
        return search_datasets(category)

    # Search using all category keywords
    combined_mask = pd.Series([False] * len(DF_METADATA), index=DF_METADATA.index)
    metadata_basis = (
        DF_METADATA["title"].fillna("").astype(str)
        + " "
        + DF_METADATA["description"].fillna("").astype(str)
    ).map(_fold_open_data_text)
    for kw in keywords:
        folded_kw = _fold_open_data_text(kw)
        if folded_kw:
            combined_mask = combined_mask | metadata_basis.str.contains(re.escape(folded_kw), na=False)

    return DF_METADATA[combined_mask]


def expand_search_terms(query: str) -> List[str]:
    """
    Expands a search query with semantic synonyms.

    Args:
        query (str): Original search term.

    Returns:
        List[str]: List of search terms including synonyms.
    """
    query_lower = str(query or "").lower()
    query_folded = _fold_open_data_text(query)
    terms = [query_lower, query_folded]

    # Check if query matches any category and expand
    for category, synonyms in CATEGORY_SYNONYMS.items():
        category_folded = _fold_open_data_text(category)
        if category_folded and (
            category_folded in query_folded
            or query_folded in category_folded
        ):
            terms.extend(synonyms)

    return [term for term in dict.fromkeys(terms) if term]  # Remove duplicates


def search_datasets(query: str) -> pd.DataFrame:
    """
    Searches metadata for datasets matching the query with semantic expansion.

    Args:
        query (str): Search term(s).

    Returns:
        pd.DataFrame: Matching datasets.
    """
    if DF_METADATA.empty:
        return pd.DataFrame()

    # Expand search terms semantically
    search_terms = expand_search_terms(query)

    # Build combined mask for all terms, using accent-insensitive matching.
    combined_mask = pd.Series([False] * len(DF_METADATA), index=DF_METADATA.index)
    metadata_basis = (
        DF_METADATA["title"].fillna("").astype(str)
        + " "
        + DF_METADATA["description"].fillna("").astype(str)
    ).map(_fold_open_data_text)

    for term in search_terms:
        folded_term = _fold_open_data_text(term)
        if folded_term:
            combined_mask = combined_mask | metadata_basis.str.contains(re.escape(folded_term), na=False)

    query_tokens = [
        token
        for token in _fold_open_data_text(query).split()
        if len(token) > 2 and token not in _DATASET_QUERY_STOPWORDS
    ]
    if query_tokens:
        token_mask = pd.Series([True] * len(DF_METADATA), index=DF_METADATA.index)
        for token in query_tokens:
            token_mask = token_mask & metadata_basis.str.contains(rf"\b{re.escape(token)}", regex=True, na=False)
        combined_mask = combined_mask | token_mask

    return DF_METADATA[combined_mask]


# ==========================================================================
# LangChain Tools
# ==========================================================================

@tool
def list_service_categories(language: str = "pt") -> str:
    """
    Lists all available service categories from Lisboa Aberta open data.

    Args:
        language: Output language, either ``"en"`` or ``"pt"``.

    Returns:
        str: Formatted list of categories with descriptions and dataset counts.

    Examples:
        >>> list_service_categories("en")
    """
    if DF_METADATA.empty:
        return "❌ Error: Metadata not loaded."

    is_pt = str(language or "pt").lower().startswith("pt")
    title = "Categorias de Serviços Disponíveis" if is_pt else "Available Public-Service Categories"
    response = f"### 📂 **{title} (Lisboa Aberta)**\n\n"
    emoji_map = {
        "saúde": "🏥", "educação": "🎓", "segurança": "🚔",
        "cultura": "🏛️", "ambiente": "🌳", "transportes": "🚇",
        "turismo": "🏨", "comércio": "🛒", "serviços": "🏢",
        "desporto": "⚽",
    }

    for cat_key, cat_info in CATEGORY_TAXONOMY.items():
        matches = get_datasets_for_category(cat_key)
        count = len(matches)
        emoji = emoji_map.get(cat_key, "📁")
        label = cat_key.capitalize() if is_pt else cat_info["en"]
        description = cat_info["description"] if is_pt else cat_info["description_en"]

        response += f"- {emoji} **{label}** ({count} datasets)\n"
        response += f"    - {description}.\n"

    response += "\n"
    if is_pt:
        response += "💡 **Dica:** Podes perguntar por uma categoria específica para obter resultados mais detalhados.\n\n"
        response += f"📊 **Total:** {len(DF_METADATA)} datasets disponíveis."
    else:
        response += "💡 **Tip:** Ask for a specific category to get more detailed locations or records.\n\n"
        response += f"📊 **Total:** {len(DF_METADATA)} datasets available."

    return response


@tool
def find_nearby_services(
    service_type: str,
    user_lat: Optional[float] = None,
    user_lon: Optional[float] = None,
    near_location_name: Optional[str] = None,
    max_results: int = 5,
    category: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    """
    Search for public services in Lisbon (pharmacies, hospitals, schools, etc.)
    and optionally filter by proximity to user location or a specific place name.

    Args:
        service_type (str): Type of service to search (e.g., 'farmácias', 'hospitais',
                           'escolas', 'metro', 'wifi', 'jardins', 'parques', 'fontanários').
        user_lat (float, optional): User's latitude for proximity filtering.
        user_lon (float, optional): User's longitude for proximity filtering.
        near_location_name (str, optional): Name of a place to filter by proximity (e.g., "Martim Moniz").
                                           Used if user_lat/lon are not provided.
        max_results (int): Maximum number of results to return (default: 5).
        category (str, optional): Filter by taxonomy category (e.g., 'saúde', 'educação', 'cultura').
                                 Use list_service_categories() to see all categories.
        language (str, optional): Output language. Use ``"pt"`` for PT-PT labels,
            ``"en"`` for English labels. If omitted, inferred from the query text.

    Returns:
        str: Formatted list of services with names, addresses, and distances.

    Examples:
        >>> find_nearby_services("farmácias", user_lat=38.7223, user_lon=-9.1393)
        >>> find_nearby_services("hospitais", near_location_name="Martim Moniz")
        >>> find_nearby_services("museu", category="cultura")
    """
    def _fold_text(value: object) -> str:
        """Return accent-insensitive lowercase text for matching."""
        return _fold_open_data_text(value)

    normalized_language = (language or "").lower().strip()
    if normalized_language not in {"pt", "en"}:
        language_probe = f"{service_type} {near_location_name or ''} {category or ''}"
        probe_norm = _fold_text(language_probe)
        pt_service_markers = (
            "perto",
            "farmacia",
            "farmacias",
            "hospital",
            "hospitais",
            "saude",
            "escola",
            "escolas",
            "biblioteca",
            "bibliotecas",
            "jardim",
            "jardins",
            "parque",
            "parques",
            "fontanario",
            "fontanarios",
            "estacionamento",
            "municipal",
            "municipais",
            "ecoponto",
            "ecopontos",
            "reciclagem",
            "residuo",
            "residuos",
        )
        normalized_language = "pt" if any(marker in probe_norm for marker in pt_service_markers) else "en"
    is_pt = normalized_language == "pt"

    if DF_METADATA.empty:
        return "❌ Erro: metadados não carregados. Verifica o ficheiro lisbon_datasets_clean.json." if is_pt else "❌ Error: Metadata not loaded. Check if lisbon_datasets_clean.json exists."

    def _clean_display_address(raw_address: object) -> str:
        """Return a one-line address suitable for Markdown links and maps."""
        cleaned = unescape(str(raw_address or ""))
        cleaned = re.sub(r"(?i)<br\s*/?>", ", ", cleaned)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;\n\t")
        return re.sub(r"\s+,", ",", cleaned)

    def _record_matches_requested_service(
        *,
        name: object,
        address: object,
        requested_service: str,
    ) -> bool:
        """Filter mixed cultural datasets without leaking unrelated records."""
        requested_norm = _fold_text(requested_service)
        cultural_policies = [
            (
                ("biblioteca", "bibliotecas", "library", "libraries"),
                ("bibliot", "library"),
                ("museu", "museum", "arquivo", "archive"),
            ),
            (
                ("museu", "museus", "museum", "museums"),
                ("museu", "museum"),
                ("bibliot", "library", "arquivo", "archive"),
            ),
            (
                ("arquivo", "arquivos", "archive", "archives"),
                ("arquivo", "archive"),
                ("bibliot", "library", "museu", "museum"),
            ),
        ]

        required_markers: tuple[str, ...] = ()
        excluded_markers: tuple[str, ...] = ()
        for request_markers, candidate_required, candidate_excluded in cultural_policies:
            if any(marker in requested_norm for marker in request_markers):
                required_markers = candidate_required
                excluded_markers = candidate_excluded
                break

        if not required_markers:
            return True

        record_text = _fold_text(f"{name or ''} {address or ''}")
        if any(marker in record_text for marker in required_markers):
            return True
        if any(marker in record_text for marker in excluded_markers):
            return False
        return False

    def _service_icon_and_heading(dataset_title: str, requested_service: str) -> tuple[str, str]:
        """Build a localized heading for any selected municipal service dataset.

        The requested service and selected dataset define the service family.
        Proximity labels such as "(perto de Jardim da Estrela)" must not affect
        classification, otherwise toilets near a garden become "parks".
        """

        def _normalize_basis(value: str) -> str:
            normalized = unicodedata.normalize("NFKD", value or "")
            return normalized.encode("ascii", "ignore").decode("ascii").lower()

        clean_dataset_title = re.sub(
            r"\s+\((?:perto de|near|ordenado por distância|sorted by distance)\s+.*?\)\s*$",
            "",
            dataset_title or "",
            flags=re.IGNORECASE,
        )
        requested_basis = _normalize_basis(requested_service or "")
        dataset_basis = _normalize_basis(clean_dataset_title)
        category_basis = _normalize_basis(category or "")
        location = near_location_name.strip() if near_location_name else ""

        service_families = [
            (("estacionamento de bicicletas", "estacionamento de velocipedes", "velocipede", "velocipedes", "bike parking", "bicycle parking"), "🚲", "Estacionamento de bicicletas", "Bicycle parking"),
            (("parking", "estacion", "car park", "parques de estacionamento"), "🅿️", "Estacionamento", "Parking"),
            (("farm", "pharmac"), "💊", "Farmácias", "Pharmacies"),
            (("hospit",), "🏥", "Hospitais", "Hospitals"),
            (("cuidados", "saude", "health", "clinica", "clinic"), "🏥", "Serviços de saúde", "Health services"),
            (("escola", "school", "educa", "universidade", "faculdade"), "🎓", "Serviços de educação", "Education services"),
            (("biblioteca", "library", "leitura"), "📚", "Bibliotecas", "Libraries"),
            (("museu", "museum", "cultura", "cultural"), "🏛️", "Equipamentos culturais", "Cultural venues"),
            (("pilhao", "pilhoes", "pilha", "pilhas", "bateria", "baterias"), "🔋", "Pilhões", "Battery recycling points"),
            (("papeleira", "papeleiras", "waste bin", "litter bin"), "🗑️", "Papeleiras", "Waste bins"),
            (("ecoponto", "reciclag", "residuo", "residuos", "waste", "recycling"), "♻️", "Ecopontos e reciclagem", "Recycling points"),
            (("parques infantis", "parque infantil", "playground", "playgrounds", "infantil"), "🛝", "Parques infantis", "Playgrounds"),
            (("parque canino", "parques caninos", "canino", "dog park", "dog parks"), "🐾", "Parques caninos", "Dog parks"),
            (("fontan", "bebedouro", "chafariz", "lago", "fountain", "water", "agua", "arquitetura da agua", "elementos de agua"), "🚰", "Fontanários e água", "Fountains and water points"),
            (("jardim", "parque", "garden", "green space", "verde"), "🌳", "Jardins e parques", "Gardens and parks"),
            (("policia", "police", "psp", "seguranca"), "👮", "Serviços de segurança", "Public safety services"),
            (("bombeir", "fire"), "🚒", "Bombeiros", "Fire services"),
            (("pontos de encontro", "ponto de encontro", "emergencia", "emergency meeting"), "🆘", "Pontos de encontro de emergência", "Emergency meeting points"),
            (("mercado", "market", "feira"), "🛒", "Mercados", "Markets"),
            (("correio", "postal", "ctt"), "✉️", "Serviços postais", "Postal services"),
            (("loja cidadao", "citizen", "atendimento"), "🏢", "Serviços municipais", "Municipal services"),
            (("wifi", "internet"), "📶", "Pontos Wi-Fi", "Wi-Fi points"),
            (
                ("wc", "sanitario", "sanitaria", "sanitarias", "instalacoes sanitarias", "casa de banho", "casas de banho", "toilet", "restroom"),
                "🚻",
                "Instalações sanitárias",
                "Restrooms",
            ),
            (("metro", "transport", "transporte", "paragem", "stop"), "🚇", "Transportes", "Transport services"),
        ]

        icon = "📍"
        pt_label = "Serviços públicos"
        en_label = "Public services"

        def _match_family(basis: str) -> tuple[str, str, str] | None:
            for markers, candidate_icon, candidate_pt, candidate_en in service_families:
                if any(marker in basis for marker in markers):
                    return candidate_icon, candidate_pt, candidate_en
            return None

        combined_basis = f"{requested_basis} {dataset_basis} {category_basis}"
        matched_family = (
            _match_family(requested_basis)
            or _match_family(dataset_basis)
            or _match_family(category_basis)
        )
        if matched_family:
            icon, pt_label, en_label = matched_family
            if pt_label == "Hospitais" and any(
                public_marker in combined_basis
                for public_marker in ("public", "publico", "publicos", "publica", "publicas")
            ):
                pt_label = "Hospitais públicos"
                en_label = "Public hospitals"

        if is_pt:
            if location:
                return icon, f"{pt_label} perto de {location}"
            found_label = "encontradas" if re.search(r"(?:as|ões)$", pt_label.lower()) else "encontrados"
            return icon, f"{pt_label} {found_label}"
        return icon, f"{en_label} near {location}" if location else f"{en_label} found"

    def _rank_service_datasets(candidate_matches: pd.DataFrame, requested_service: str) -> pd.DataFrame:
        """Sort service datasets by semantic fit before trying GeoJSON URLs."""
        if candidate_matches.empty:
            return candidate_matches

        requested_norm = unicodedata.normalize("NFKD", requested_service or "")
        requested_norm = requested_norm.encode("ascii", "ignore").decode("ascii").lower()
        requested_tokens = {
            token for token in re.split(r"[^a-z0-9]+", requested_norm)
            if len(token) > 3 and token not in {"near", "perto", "para", "with"}
        }
        parking_request = bool(
            re.search(r"\b(parking|estacion|car\s*park|parque\s+de\s+estacionamento)\b", requested_norm)
        )
        car_parking_request = bool(
            parking_request
            and re.search(r"\b(car|cars|auto|automovel|automoveis|carro|viatura|vehicle|vehicles)\b", requested_norm)
        )

        def _is_non_car_parking_dataset(row: pd.Series) -> bool:
            basis = unicodedata.normalize(
                "NFKD",
                f"{row.get('title', '')} {row.get('description', '')}",
            )
            basis = basis.encode("ascii", "ignore").decode("ascii").lower()
            return bool(re.search(r"\b(tuktuk|tuk\s*tuk|bicic\w*|velociped\w*|bike|bicycle|motocicl\w*|scooter\w*)\b", basis))

        def _score(row: pd.Series) -> int:
            basis = unicodedata.normalize(
                "NFKD",
                f"{row.get('title', '')} {row.get('description', '')}",
            )
            basis = basis.encode("ascii", "ignore").decode("ascii").lower()
            title = unicodedata.normalize("NFKD", str(row.get("title", "")))
            title = title.encode("ascii", "ignore").decode("ascii").lower()
            score = 0
            for token in requested_tokens:
                if token in title:
                    score += 18
                elif token in basis:
                    score += 8
            if parking_request:
                if re.search(r"\b(parques?\s+de\s+estacionamento|car\s*parks?|parking\s+facilit)", basis):
                    score += 100
                if re.search(r"\b(lugares?\s+de\s+estacionamento|zonas?\s+reguladas?\s+de\s+estacionamento)\b", basis):
                    score += 60
                if _is_non_car_parking_dataset(row):
                    score -= 500 if car_parking_request else 80
            return score

        ranked = candidate_matches.copy()
        if car_parking_request:
            ranked = ranked[~ranked.apply(_is_non_car_parking_dataset, axis=1)]
        ranked["_service_rank"] = ranked.apply(_score, axis=1)
        return ranked.sort_values("_service_rank", ascending=False, kind="mergesort").drop(columns=["_service_rank"])

    # Geocoding Logic: Resolve location name if coordinates missing
    if near_location_name and (user_lat is None or user_lon is None):
        resolved_reference = _resolve_reference_coordinates(near_location_name)
        if resolved_reference:
            user_lat, user_lon = resolved_reference
            logger.info("Resolved '%s' to (%s, %s)", near_location_name, user_lat, user_lon)

    if near_location_name and (user_lat is None or user_lon is None):
        logger.info(f"Geocoding '{near_location_name}' via Open Data...")
        places = _search_places_raw(near_location_name, max_results=1)

        if places and places[0]['lat'] and places[0]['lon']:
            user_lat = places[0]['lat']
            user_lon = places[0]['lon']
            logger.info(f"✅ Geocoded '{near_location_name}' to ({user_lat}, {user_lon})")
        else:
            # Fallback to Nominatim (Carris Metropolitana API)
            try:
                logger.info(f"Open Data lookup failed for '{near_location_name}'. Trying Nominatim fallback...")
                from tools.carrismetropolitana_api import geocode_location
                loc = geocode_location(near_location_name)

                if loc:
                    user_lat = loc['lat']
                    user_lon = loc['lon']
                    logger.info(f"✅ Geocoded '{near_location_name}' via Nominatim to ({user_lat}, {user_lon})")
                else:
                    if is_pt:
                        return f"❌ Não consegui resolver a localização '{near_location_name}'. Tentei Lisboa Aberta e geocodificação. Indica coordenadas ou uma referência mais específica."
                    return f"❌ Could not resolve location '{near_location_name}'. Tried Open Data and geocoding. Please provide coordinates."
            except ImportError:
                if is_pt:
                    return f"❌ Não consegui resolver a localização '{near_location_name}' na Lisboa Aberta. O geocoder externo está indisponível."
                return f"❌ Could not resolve location '{near_location_name}' in Open Data. External geocoder unavailable."

    # Search for matching datasets (with optional category filtering)
    if category:
        # Filter within the specified taxonomy category first
        category_datasets = get_datasets_for_category(category)
        if not category_datasets.empty:
            # Search within category datasets
            search_terms = expand_search_terms(service_type)
            combined_mask = pd.Series([False] * len(category_datasets), index=category_datasets.index)
            category_basis = (
                category_datasets["title"].fillna("").astype(str)
                + " "
                + category_datasets["description"].fillna("").astype(str)
            ).map(_fold_open_data_text)
            for term in search_terms:
                folded_term = _fold_open_data_text(term)
                if folded_term:
                    combined_mask = combined_mask | category_basis.str.contains(re.escape(folded_term), na=False)
            query_tokens = [
                token
                for token in _fold_open_data_text(service_type).split()
                if len(token) > 2 and token not in _DATASET_QUERY_STOPWORDS
            ]
            if query_tokens:
                token_mask = pd.Series([True] * len(category_datasets), index=category_datasets.index)
                for token in query_tokens:
                    token_mask = token_mask & category_basis.str.contains(rf"\b{re.escape(token)}", regex=True, na=False)
                combined_mask = combined_mask | token_mask
            matches = category_datasets[combined_mask]
            if matches.empty:
                # Fall back to service-specific search instead of broad category noise.
                matches = search_datasets(service_type)
        else:
            matches = search_datasets(service_type)
    else:
        matches = search_datasets(service_type)

    if matches.empty:
        # Try alternative search terms
        alternatives = {
            'pharmacy': 'farmácia', 'hospital': 'hospital', 'school': 'escola',
            'pharmacies': 'farmácia', 'farmacias': 'farmácia', 'farmacia': 'farmácia',
            'bibliotecas municipais': 'bibliotecas',
            'biblioteca municipal': 'biblioteca',
            'municipal libraries': 'libraries',
            'municipal library': 'libraries',
            'park': 'jardim', 'garden': 'jardim', 'wifi': 'wifi', 'metro': 'metro',
            'fountain': 'fontanário', 'parking': 'estacionamento', 'car parking': 'estacionamento',
            'ecopontos': 'ecoponto', 'recycling': 'ecoponto',
            'pilhoes': 'pilhões', 'pilhao': 'pilhões',
            'bebedouros': 'fontanário', 'bebedouro': 'fontanário',
            'papeleiras': 'papeleiras', 'papeleira': 'papeleiras',
            'parques caninos': 'parques caninos', 'parque canino': 'parques caninos',
            'estacionamento de bicicletas': 'estacionamento de velocípedes',
            'bike parking': 'estacionamento de velocípedes',
            'bicycle parking': 'estacionamento de velocípedes',
            'pontos de encontro de emergencia': 'pontos de encontro emergência',
            'emergency meeting points': 'pontos de encontro emergência',
            'wc públicos': 'instalações sanitárias', 'wc publicos': 'instalações sanitárias',
            'instalacoes sanitarias': 'instalações sanitárias',
            'sanitários': 'instalações sanitárias', 'sanitarios': 'instalações sanitárias',
            'casas de banho públicas': 'instalações sanitárias',
            'casas de banho publicas': 'instalações sanitárias',
            'toilet': 'instalações sanitárias', 'toilets': 'instalações sanitárias',
            'restroom': 'instalações sanitárias', 'restrooms': 'instalações sanitárias',
        }
        alt_term = alternatives.get(service_type.lower(), service_type)
        matches = search_datasets(alt_term)

    if matches.empty:
        if is_pt:
            return f"❌ Não encontrei fontes de dados para: '{service_type}'\n💡 Experimenta: farmácias, hospitais, escolas, jardins, metro, fontanários"
        return f"❌ No data sources found for: '{service_type}'\n💡 Try: pharmacies, hospitals, schools, parks, metro, fountains"

    matches = _rank_service_datasets(matches.drop_duplicates(subset="stable_url"), service_type)
    if matches.empty:
        if is_pt:
            return f"❌ Não encontrei uma fonte de dados compatível para: '{service_type}'"
        return f"❌ No compatible data source found for: '{service_type}'"

    selected_title = ""
    selected_feature_count = 0
    results = []
    dataset_errors: List[str] = []

    for _, dataset in matches.iterrows():
        title = dataset['title']
        stable_url = dataset.get('stable_url')
        if not stable_url or stable_url == "N/A":
            dataset_errors.append(f"{title}: no URL available")
            continue

        geojson_data = fetch_geojson_with_retry(stable_url)
        if not geojson_data:
            if _get_unavailable_dataset_reason(stable_url):
                dataset_errors.append(f"{title}: dataset temporarily unavailable")
            else:
                dataset_errors.append(f"{title}: fetch failed")
            continue

        features = geojson_data.get('features', [])
        if not features:
            dataset_errors.append(f"{title}: no features")
            continue

        candidate_results = []
        for feature in features:
            try:
                properties = feature.get('properties', {})
                geometry = feature.get('geometry', {})

                coords = extract_coordinates(geometry)
                if not coords:
                    continue

                lat, lon = coords

                distance = None
                if user_lat is not None and user_lon is not None:
                    distance = haversine_distance(user_lat, user_lon, lat, lon)

                name = extract_name(properties)
                address = extract_address(properties)

                if not _record_matches_requested_service(
                    name=name,
                    address=address,
                    requested_service=service_type,
                ):
                    continue

                candidate_results.append({
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

        if user_lat is not None and user_lon is not None and candidate_results:
            candidate_results = [r for r in candidate_results if r['distance'] is not None]
            candidate_results.sort(key=lambda x: x['distance'])

        candidate_results = candidate_results[:max_results]
        if candidate_results:
            selected_title = title
            selected_feature_count = len(features)
            results = candidate_results
            break

        dataset_errors.append(f"{title}: no usable location records")

    if not results:
        if dataset_errors:
            return (
                (
                    f"❌ Não consegui carregar dados utilizáveis para '{service_type}'.\n"
                    f"🧪 Fontes testadas: {', '.join(dataset_errors[:3])}"
                ) if is_pt else (
                    f"❌ Could not load usable data for '{service_type}'.\n"
                    f"🧪 Tried data sources: {', '.join(dataset_errors[:3])}"
                )
            )
        return (
            f"❌ Não encontrei fontes de dados para: '{service_type}'"
            if is_pt
            else f"❌ No data sources found for: '{service_type}'"
        )

    # Sort by distance if coordinates provided
    if user_lat is not None and user_lon is not None and results:
        # Add header about proximity
        if near_location_name:
            selected_title += f" ({'perto de' if is_pt else 'near'} {near_location_name})"
        else:
            selected_title += " (ordenado por distância)" if is_pt else " (sorted by distance)"

    if not results:
        if is_pt:
            return f"✓ Fonte de dados '{selected_title}' carregada ({selected_feature_count} registos), mas não foi possível extrair dados de localização."
        return f"✓ Data source '{selected_title}' loaded ({selected_feature_count} features) but couldn't extract location data."

    for item in results:
        raw_name = str(item.get("name") or "").strip()
        if not raw_name or raw_name.lower() in {"n/a", "na", "none", "null", "unknown"}:
            item["name"] = "Ponto de reciclagem" if is_pt else "Recycling point"

    name_counts: Dict[str, int] = {}
    for item in results:
        base_name = str(item.get("name") or "").strip()
        name_counts[base_name] = name_counts.get(base_name, 0) + 1
    duplicate_name_indexes: Dict[str, int] = {}

    def _display_result_name(base_name: object) -> str:
        """Return a stable display name when a dataset uses generic repeated names."""
        name = str(base_name or "").strip()
        if not name:
            name = "Ponto de reciclagem" if is_pt else "Recycling point"
        if name_counts.get(name, 0) <= 1:
            return name
        duplicate_name_indexes[name] = duplicate_name_indexes.get(name, 0) + 1
        return f"{name} {duplicate_name_indexes[name]}"

    item_icon, heading = _service_icon_and_heading(selected_title, service_type)
    count_label = "resultado" if len(results) == 1 else "resultados"
    response = f"### {item_icon} **{heading}**\n\n"

    if near_location_name and results:
        nearest = results[0]
        nearest_distance = nearest.get("distance")
        if nearest_distance is not None:
            nearest_name = _display_result_name(nearest.get("name"))
            duplicate_name_indexes.clear()
            walk_minutes = max(1, round(float(nearest_distance) * 12))
            if is_pt:
                response += (
                    f"- ✅ **Mais perto:** {nearest_name} "
                    f"({nearest_distance:.2f} km de {near_location_name}; cerca de {walk_minutes} min a pé)\n\n"
                )
            else:
                response += (
                    f"- ✅ **Nearest:** {nearest_name} "
                    f"({nearest_distance:.2f} km from {near_location_name}; about {walk_minutes} min walking)\n\n"
                )

    if is_pt:
        response += f"- 🧭 **Fonte dos dados:** {selected_title}\n"
        response += f"- 📊 **Resultados:** {len(results)} {count_label}\n\n"
        if not near_location_name:
            response += "- ⚠️ **Cobertura:** esta pesquisa mostra apenas os primeiros resultados disponíveis; não é uma listagem exaustiva da AML e horários/contactos só aparecem quando constam dos dados.\n\n"
    else:
        response += f"- 🧭 **Data source:** {selected_title}\n"
        response += f"- 📊 **Results:** {len(results)} result(s)\n\n"
        if not near_location_name:
            response += "- ⚠️ **Coverage:** this search shows only the first available results; it is not an exhaustive AML-wide list, and hours/contacts appear only when present in the data.\n\n"
    water_context = any(
        marker in unicodedata.normalize("NFKD", f"{selected_title} {service_type}").encode("ascii", "ignore").decode("ascii").lower()
        for marker in ("arquitetura da agua", "elementos de agua", "bebedouro", "fontan", "chafariz", "water")
    )
    if water_context:
        response += (
            "- ⚠️ **Nota:** estes registos identificam elementos/fontes de água no espaço público; a potabilidade não é confirmada pelos dados disponíveis.\n\n"
            if is_pt
            else "- ⚠️ **Note:** these records identify public water/fountain features; drinkability is not confirmed by the available data.\n\n"
        )

    for r in results:
        display_name = _display_result_name(r.get("name"))
        response += f"- {item_icon} **{display_name}**\n"
        cleaned_address = _clean_display_address(r.get('address'))
        if cleaned_address:
            if r.get('lat') is not None and r.get('lon') is not None:
                map_query = f"{r['lat']:.6f},{r['lon']:.6f}"
            elif "lisboa" in cleaned_address.lower():
                map_query = cleaned_address
            else:
                map_query = f"{cleaned_address}, Lisboa, Portugal"
            map_url = f"https://www.google.com/maps/search/?api=1&query={quote_plus(map_query)}"
            address_label = "Morada" if is_pt else "Address"
            response += f"    - 📍 **{address_label}:** [{cleaned_address}]({map_url})\n"
        elif r.get('lat') is not None and r.get('lon') is not None:
            map_url = f"https://www.google.com/maps/search/?api=1&query={r['lat']:.6f}%2C{r['lon']:.6f}"
            map_label = "Localização" if is_pt else "Location"
            open_label = "Abrir localização" if is_pt else "Open location"
            response += f"    - 🗺️ **{map_label}:** [{open_label}]({map_url})\n"
        if r['distance'] is not None:
            distance_label = "Distância" if is_pt else "Distance"
            response += f"    - 📏 **{distance_label}:** {r['distance']:.2f} km\n"
            if near_location_name:
                walk_minutes = max(1, round(float(r['distance']) * 12))
                walk_label = "Tempo a pé estimado" if is_pt else "Estimated walking time"
                walk_value = f"cerca de {walk_minutes} min" if is_pt else f"about {walk_minutes} min"
                response += f"    - 🚶 **{walk_label}:** {walk_value}\n"
        response += "\n"

    return response


@tool
def list_available_datasets(category: Optional[str] = None) -> str:
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
        response += "💡 Filtra por categoria para resultados mais específicos."

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


def _search_places_raw(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search for places and return raw data (lat/lon).
    """
    if DF_METADATA.empty:
        return []

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
        'psp': ['Polícia de Segurança Pública'],
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
        'wi-fi': ['Rede LoRa'],  # Approximate

        # Outdoors & Leisure
        'jardim': ['Jardins - Parques Urbanos', 'Grandes Parques e Jardins de Lisboa', 'Espaços Verdes'],
        'parque infantil': ['Parques Infantis'],
        'parques infantis': ['Parques Infantis'],
        'playground': ['Parques Infantis'],
        'parque': ['Grandes Parques e Jardins de Lisboa', 'Jardins - Parques Urbanos', 'Parques Infantis', 'Parques de Merendas', 'Parques Caninos'],
        'praia': [],  # Not many open datasets for beaches in CML directly besides river ones
        'desporto': ['Instalações Desportivas', 'Centros Desportivos', 'Equipamentos de Fitness ao Ar Livre\u200b', 'Programa Desporto Mexe Comigo'],
        'piscina': ['Instalações Desportivas', 'Programa de Apoio à Natação Curricular'],
        'bebedouro': ['Arquitetura da Água', 'Elementos de Água'],
        'bebedouros': ['Arquitetura da Água', 'Elementos de Água'],
        'ponto de agua': ['Arquitetura da Água', 'Elementos de Água'],
        'pontos de agua': ['Arquitetura da Água', 'Elementos de Água'],
        'ponto de água': ['Arquitetura da Água', 'Elementos de Água'],
        'pontos de água': ['Arquitetura da Água', 'Elementos de Água'],
        'fontanario': ['Arquitetura da Água', 'Elementos de Água'],
        'fontanarios': ['Arquitetura da Água', 'Elementos de Água'],
        'fontanário': ['Arquitetura da Água', 'Elementos de Água'],
        'fontanários': ['Arquitetura da Água', 'Elementos de Água'],
        'chafariz': ['Arquitetura da Água', 'Elementos de Água'],
        'chafarizes': ['Arquitetura da Água', 'Elementos de Água'],
        'arquitetura da agua': ['Arquitetura da Água'],
        'arquitetura da água': ['Arquitetura da Água'],

        # Services & Amenities
        'wc': ['Instalações Sanitárias', 'Instalações Sanitárias Públicas Automáticas', 'Balneários'],
        'banheiro': ['Instalações Sanitárias', 'Instalações Sanitárias Públicas Automáticas'],
        'estacionamento': ['Parques de estacionamento na via pública', 'EMEL - Parques de estacionamento na via pública', 'Lugares de estacionamento na via pública para residentes ou público em geral', 'Zonas reguladas de estacionamento na via pública'],
        'embaixada': ['Embaixadas'],
        'ctt': [],  # Post offices
        'cemiterio': ['Cemitérios'],
        'loja cidadao': ['Loja do Cidadão'],
        'camara': ['CM Lisboa - Paços do Concelho', 'CM Lisboa - Atendimento', 'Juntas de Freguesia'],
        'junta': ['Juntas de Freguesia'],

        # Streets & Locations
        'rua': ['Toponímia de Lisboa', 'Topónimos', 'Eixos de Via'],
        'avenida': ['Toponímia de Lisboa', 'Topónimos'],
        'praca': ['Toponímia de Lisboa', 'Topónimos'],
        'largo': ['Toponímia de Lisboa', 'Topónimos'],
        'bairro': ['Bairros e Zonas de Intervenção Prioritária', 'Localização e identificação das Casas Religiosas de Lisboa existentes em 2015'],  # Proxy
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
        return []

    potential_datasets = potential_datasets.drop_duplicates(subset='stable_url')

    # Limit to top 5 datasets to ensure responsiveness
    for _idx, dataset in potential_datasets.head(5).iterrows():
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
                    'category': title,  # Use dataset title as category
                    'location': address,
                    'lat': lat,
                    'lon': lon,
                    'short_description': f"Found in Lisbon municipal open data: {title}",
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

    return sorted(unique_places.values(), key=lambda x: x['score'], reverse=True)[:max_results]


def _search_place_in_datasets_logic(query: str, max_results: int = 5) -> str:
    """
    Search wrapper that returns formatted string (for VisitLisboa integration).
    """
    results = _search_places_raw(query, max_results)

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
    import sys
    with contextlib.suppress(AttributeError):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("\n" + "=" * 70)
    print("\033[1m🧪 COMPREHENSIVE TEST: Dados Abertos Lisboa Tools\033[0m")
    print("=" * 70)

    test_results = {"passed": 0, "failed": 0, "total": 0}

    def run_test(test_name: str, test_func, *args, **kwargs):
        """Helper to run tests with error handling."""
        test_results["total"] += 1
        print(f"\n\033[1m{'─' * 70}\033[0m")
        print(f"\033[1;36m🔬 TEST {test_results['total']}: {test_name}\033[0m")
        print(f"{'─' * 70}")
        try:
            result = test_func(*args, **kwargs)
            # # Truncate long outputs for readability
            # if len(result) > 800:
            #     print(result[:800] + "\n\n... (truncated for readability)")
            # else:
            print(result)
            test_results["passed"] += 1
            print("\n\033[1;32m✅ PASSED\033[0m")
            return result
        except Exception as e:
            print(f"\n\033[1;31m❌ FAILED: {str(e)}\033[0m")
            test_results["failed"] += 1
            return None

    # =========================================================================
    # DATASET DISCOVERY TESTS
    # =========================================================================
    # TEST 1: List all datasets without filter
    run_test(
        "List All Datasets (No Filter)",
        list_available_datasets.invoke,
        {}
    )

    # TEST 2: List datasets filtered by 'saúde'
    run_test(
        "List Datasets - Filter by 'saúde'",
        list_available_datasets.invoke,
        {"category": "saúde"}
    )

    # TEST 3: List datasets filtered by 'ambiente'
    run_test(
        "List Datasets - Filter by 'ambiente'",
        list_available_datasets.invoke,
        {"category": "ambiente"}
    )

    # TEST 4: List datasets filtered by 'educação'
    run_test(
        "List Datasets - Filter by 'educação'",
        list_available_datasets.invoke,
        {"category": "educação"}
    )

    # =========================================================================
    # DATASET DETAILS TESTS
    # =========================================================================
    # TEST 5: Get details for 'jardins' dataset
    run_test(
        "Get Dataset Details - Jardins (Parks)",
        get_dataset_details.invoke,
        {"dataset_name": "jardins"}
    )

    # TEST 6: Get details for 'farmácias' dataset
    run_test(
        "Get Dataset Details - Farmácias (Pharmacies)",
        get_dataset_details.invoke,
        {"dataset_name": "farmácias"}
    )

    # TEST 7: Get details for 'hospitais' dataset
    run_test(
        "Get Dataset Details - Hospitais (Hospitals)",
        get_dataset_details.invoke,
        {"dataset_name": "hospitais"}
    )

    # =========================================================================
    # NEARBY SERVICES TESTS (Without User Location)
    # =========================================================================
    # TEST 8: Find pharmacies without user location
    run_test(
        "Find Services - Farmácias (No Location)",
        find_nearby_services.invoke,
        {"service_type": "farmácias", "max_results": 3}
    )

    # TEST 9: Find gardens/parks without user location
    run_test(
        "Find Services - Jardins (No Location)",
        find_nearby_services.invoke,
        {"service_type": "jardins", "max_results": 3}
    )

    # TEST 10: Find public WiFi spots without user location
    run_test(
        "Find Services - WiFi Público (No Location)",
        find_nearby_services.invoke,
        {"service_type": "wifi", "max_results": 3}
    )

    # TEST 11: Find hospitals without user location
    run_test(
        "Find Services - Hospitais (No Location)",
        find_nearby_services.invoke,
        {"service_type": "hospitais", "max_results": 3}
    )

    # =========================================================================
    # NEARBY SERVICES TESTS (With User Location - Proximity Filtering)
    # =========================================================================

    # Test coordinates: Lisbon City Center (Rossio)
    LISBON_CENTER_LAT = 38.7139
    LISBON_CENTER_LON = -9.1395

    # TEST 12: Find pharmacies near Rossio
    run_test(
        "Find Services - Farmácias Near Rossio (WITH Location)",
        find_nearby_services.invoke,
        {
            "service_type": "farmácias",
            "user_lat": LISBON_CENTER_LAT,
            "user_lon": LISBON_CENTER_LON,
            "max_results": 3
        }
    )

    # TEST 13: Find hospitals near Rossio
    run_test(
        "Find Services - Jardins Near Rossio (WITH Location)",
        find_nearby_services.invoke,
        {
            "service_type": "jardins",
            "user_lat": LISBON_CENTER_LAT,
            "user_lon": LISBON_CENTER_LON,
            "max_results": 3
        }
    )

    # TEST 14: Find schools near Rossio
    run_test(
        "Find Services - Escolas Near Rossio (WITH Location)",
        find_nearby_services.invoke,
        {
            "service_type": "escolas",
            "user_lat": LISBON_CENTER_LAT,
            "user_lon": LISBON_CENTER_LON,
            "max_results": 3
        }
    )

    # =========================================================================
    # PLACE SEARCH TESTS (find_place_in_datasets)
    # =========================================================================
    # TEST 15: Search for "Centro Comercial Colombo"
    run_test(
        "Search Place - Centro Comercial Colombo",
        find_place_in_datasets.invoke,
        {"query": "Centro Comercial Colombo", "max_results": 3}
    )

    # TEST 16: Search for "Mercado da Ribeira"
    run_test(
        "Search Place - Jardim da Estrela",
        find_place_in_datasets.invoke,
        {"query": "Jardim da Estrela", "max_results": 3}
    )

    # TEST 17: Search for "Hospital Santa Maria"
    run_test(
        "Search Place - Hospital Santa Maria",
        find_place_in_datasets.invoke,
        {"query": "Hospital Santa Maria", "max_results": 3}
    )

    # TEST 18: Search for "Mercado da Ribeira"
    run_test(
        "Search Place - Mercado da Ribeira",
        find_place_in_datasets.invoke,
        {"query": "Mercado da Ribeira", "max_results": 3}
    )

    # TEST 19: Search for "centro comercial" (shopping centers)
    run_test(
        "Search Place - Centro Comercial (Shopping Centers)",
        find_place_in_datasets.invoke,
        {"query": "centro comercial", "max_results": 5}
    )

    # =========================================================================
    # ALTERNATIVE SEARCH TERMS TESTS (English Keywords)
    # =========================================================================
    # TEST 20: Find services using English terms
    run_test(
        "Find Services - 'pharmacy' (English Term)",
        find_nearby_services.invoke,
        {"service_type": "pharmacy", "max_results": 3}
    )

    # TEST 21: Find services using English terms
    run_test(
        "Find Services - 'park' (English Term)",
        find_nearby_services.invoke,
        {"service_type": "park", "max_results": 3}
    )

    # TEST 22: Find services using English terms
    run_test(
        "Find Services - 'hospital' (English Term)",
        find_nearby_services.invoke,
        {"service_type": "hospital", "max_results": 3}
    )

    # =========================================================================
    # EDGE CASES & ERROR HANDLING
    # =========================================================================
    # TEST 23: Nonexistent service type
    run_test(
        "Edge Case - Nonexistent Service Type",
        find_nearby_services.invoke,
        {"service_type": "xyzabc123nonexistent", "max_results": 3}
    )

    # TEST 24: Nonexistent dataset category
    run_test(
        "Edge Case - Nonexistent Dataset Category",
        list_available_datasets.invoke,
        {"category": "categoria_inexistente_xyz"}
    )

    # TEST 25: Nonexistent dataset details
    run_test(
        "Edge Case - Nonexistent Dataset Details",
        get_dataset_details.invoke,
        {"dataset_name": "dataset_que_nao_existe"}
    )

    # TEST 26: Empty place query
    run_test(
        "Edge Case - Empty Place Query",
        find_place_in_datasets.invoke,
        {"query": "", "max_results": 3}
    )

    # =========================================================================
    # SPECIAL SERVICES TESTS
    # =========================================================================
    # TEST 27: Find Services - Estacionamento (Parking)
    run_test(
        "Find Services - Estacionamento (Parking)",
        find_nearby_services.invoke,
        {"service_type": "estacionamento", "max_results": 3}
    )

    # TEST 28: Find Services - Bibliotecas (Libraries)
    run_test(
        "Find Services - Bibliotecas (Libraries)",
        find_nearby_services.invoke,
        {"service_type": "bibliotecas", "max_results": 3}
    )

    # TEST 29: Find Services - Miradouros (Viewpoints)
    run_test(
        "Find Services - Miradouros (Viewpoints)",
        find_nearby_services.invoke,
        {"service_type": "miradouros", "max_results": 3}
    )

    # =========================================================================
    # CATEGORY TAXONOMY TESTS
    # =========================================================================

    # TEST 30: Validate CATEGORY_TAXONOMY structure
    def _test_taxonomy_structure():
        errors = []
        if not isinstance(CATEGORY_TAXONOMY, dict):
            raise AssertionError("CATEGORY_TAXONOMY is not a dict")

        expected_categories = [
            "saúde", "educação", "segurança", "cultura", "ambiente",
            "transportes", "turismo", "comércio", "serviços", "desporto"
        ]

        for cat in expected_categories:
            if cat not in CATEGORY_TAXONOMY:
                errors.append(f"Missing category: {cat}")
            else:
                info = CATEGORY_TAXONOMY[cat]
                if "en" not in info:
                    errors.append(f"{cat}: missing 'en' key")
                if "keywords" not in info:
                    errors.append(f"{cat}: missing 'keywords' key")
                elif not isinstance(info["keywords"], list) or len(info["keywords"]) == 0:
                    errors.append(f"{cat}: 'keywords' must be non-empty list")
                if "description" not in info:
                    errors.append(f"{cat}: missing 'description' key")

        if errors:
            raise AssertionError("\n".join(errors))

        return f"✅ {len(CATEGORY_TAXONOMY)} categories validated\n" + \
               "\n".join(f"   {k}: {v['en']} ({len(v['keywords'])} keywords)"
                         for k, v in CATEGORY_TAXONOMY.items())

    run_test("CATEGORY_TAXONOMY Structure Validation", _test_taxonomy_structure)

    # TEST 31: get_datasets_for_category() - saúde
    def _test_get_datasets_saude():
        df = get_datasets_for_category("saúde")
        if df.empty:
            raise AssertionError("No datasets found for 'saúde'")
        result = f"Found {len(df)} datasets for 'saúde':\n"
        result += "\n".join(f"   - {t}" for t in df['title'].head(5).tolist())
        return result

    run_test("get_datasets_for_category('saúde')", _test_get_datasets_saude)

    # TEST 32: get_datasets_for_category() - English name
    def _test_get_datasets_english():
        df = get_datasets_for_category("Health")
        if df.empty:
            raise AssertionError("No datasets found for 'Health' (English)")
        return f"Found {len(df)} datasets for 'Health' (English lookup)"

    run_test("get_datasets_for_category('Health') - English Name", _test_get_datasets_english)

    # TEST 33: get_datasets_for_category() - educação
    def _test_get_datasets_educacao():
        df = get_datasets_for_category("educação")
        if df.empty:
            raise AssertionError("No datasets found for 'educação'")
        return f"Found {len(df)} datasets for 'educação'"

    run_test("get_datasets_for_category('educação')", _test_get_datasets_educacao)

    # TEST 34: get_datasets_for_category() - unknown category fallback
    def _test_get_datasets_unknown():
        df = get_datasets_for_category("nonexistent_xyz")
        return f"Fallback search returned {len(df)} datasets (expected: 0 or small)"

    run_test("get_datasets_for_category('nonexistent_xyz') - Fallback", _test_get_datasets_unknown)

    # TEST 35: list_service_categories tool
    run_test(
        "list_service_categories Tool Output",
        list_service_categories.invoke,
        {}
    )

    # TEST 36: list_service_categories validates content
    def _test_categories_content():
        result = list_service_categories.invoke({})
        checks = {
            "saúde": "Saúde" in result or "Health" in result,
            "educação": "Educação" in result or "Education" in result,
            "segurança": "Segurança" in result or "Safety" in result,
            "cultura": "Cultura" in result or "Culture" in result,
            "desporto": "Desporto" in result or "Sports" in result,
            "has_counts": "dataset" in result.lower(),
        }
        errors = [k for k, v in checks.items() if not v]
        if errors:
            raise AssertionError(f"Missing in output: {errors}")
        return f"✅ All expected categories present in output ({len(result)} chars)"

    run_test("list_service_categories Content Validation", _test_categories_content)

    # TEST 37: find_nearby_services WITH category='saúde'
    run_test(
        "find_nearby_services + category='saúde' (Pharmacies near Rossio)",
        find_nearby_services.invoke,
        {
            "service_type": "farmácias",
            "user_lat": LISBON_CENTER_LAT,
            "user_lon": LISBON_CENTER_LON,
            "max_results": 3,
            "category": "saúde"
        }
    )

    # TEST 38: find_nearby_services WITH category='cultura'
    run_test(
        "find_nearby_services + category='cultura' (Museums/Libraries)",
        find_nearby_services.invoke,
        {
            "service_type": "bibliotecas",
            "user_lat": LISBON_CENTER_LAT,
            "user_lon": LISBON_CENTER_LON,
            "max_results": 3,
            "category": "cultura"
        }
    )

    # TEST 39: find_nearby_services WITH category='ambiente' (Parks/Gardens)
    run_test(
        "find_nearby_services + category='ambiente' (Parks/Gardens)",
        find_nearby_services.invoke,
        {
            "service_type": "jardins",
            "user_lat": LISBON_CENTER_LAT,
            "user_lon": LISBON_CENTER_LON,
            "max_results": 3,
            "category": "ambiente"
        }
    )

    # TEST 40: find_nearby_services with non-matching category (should still work)
    run_test(
        "find_nearby_services + category='desporto' + service='farmácias' (Cross-category)",
        find_nearby_services.invoke,
        {
            "service_type": "farmácias",
            "max_results": 3,
            "category": "desporto"
        }
    )

    # =========================================================================
    # NEAR LOCATION NAME TESTS
    # =========================================================================
    # TEST 41: find_nearby_services with near_location_name
    run_test(
        "find_nearby_services near 'Praça do Comércio'",
        find_nearby_services.invoke,
        {
            "service_type": "farmácias",
            "near_location_name": "Praça do Comércio",
            "max_results": 3
        }
    )

    # =========================================================================
    # TEST SUMMARY
    # =========================================================================

    print("\n" + "=" * 70)
    print("\033[1m📊 TEST SUMMARY\033[0m")
    print("=" * 70)
    print(f"\033[1;32m✅ Passed: {test_results['passed']}/{test_results['total']}\033[0m")
    print(f"\033[1;31m❌ Failed: {test_results['failed']}/{test_results['total']}\033[0m")

    if test_results['failed'] == 0:
        print("\n\033[1;32m🎉 ALL TESTS PASSED! Dados Abertos system is working correctly.\033[0m")
    else:
        print("\n\033[1;33m⚠️  Some tests failed. Check errors above.\033[0m")

    print("=" * 70 + "\n")
