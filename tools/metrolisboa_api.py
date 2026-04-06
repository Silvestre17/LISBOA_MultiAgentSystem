# ==========================================================================
# Master Thesis - Metro de Lisboa API Tools
#   - André Filipe Gomes Silvestre, 20240502
#
#   Real-time metro data for Lisbon's underground network.
#   Features:
#     - Official Metro API with OAuth2 authentication
#       * Real-time waiting times per station/line
#       * Line status and disruptions
#       * Station information with GPS coordinates
#       * Service frequency/intervals
#       * Automatic token refresh
#     - Fallback to unofficial API when credentials unavailable
#
#   API Documentation: https://api.metrolisboa.pt/store/
#   API Base: https://api.metrolisboa.pt:8243/estadoServicoML/1.0.1/
# ==========================================================================

# Required libraries:
# pip install requests langchain-core

import base64
import logging
import os
import re
import socket
import ssl
import tempfile
import time
import unicodedata
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, cast

import certifi
import requests
import urllib3
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.x509.oid import AuthorityInformationAccessOID, ExtensionOID
from langchain_core.tools import tool

try:
    import config as _project_config
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
else:
    del _project_config

try:
    from tools.utils import haversine_distance
except ImportError:
    from utils import haversine_distance

logger = logging.getLogger(__name__)

# Request configuration
REQUEST_TIMEOUT = 15  # seconds
MAX_RETRIES = 3  # number of retries for API calls
BACKOFF_FACTOR = 2  # exponential backoff factor

# ==========================================================================
# API Endpoints
# ==========================================================================

# Metro de Lisboa - Official API (OAuth2 authenticated)
METRO_API_BASE = "https://api.metrolisboa.pt:8243/estadoServicoML/1.0.1"
METRO_TOKEN_URL = "https://api.metrolisboa.pt:8243/token"
METRO_CONSUMER_KEY = os.getenv("METRO_CONSUMER_KEY", "")
METRO_CONSUMER_SECRET = os.getenv("METRO_CONSUMER_SECRET", "")
METRO_CA_BUNDLE = os.getenv("METRO_CA_BUNDLE", "").strip()
_METRO_SSL_VERIFY_RAW = os.getenv("METRO_SSL_VERIFY", "").strip().lower()
_METRO_SSL_ALLOW_INSECURE_FALLBACK = (
    os.getenv("METRO_SSL_ALLOW_INSECURE_FALLBACK", "").strip().lower()
    in {"1", "true", "yes", "on"}
)
_METRO_INSECURE_WARNING_PATTERN = (
    r"Unverified HTTPS request is being made to host 'api\.metrolisboa\.pt'.*"
)
METRO_API_HOST = "api.metrolisboa.pt"
METRO_API_PORT = 8243

# Metro de Lisboa - Fallback (unofficial, no auth required)
METRO_STATUS_URL = "https://app.metrolisboa.pt/status/getLinhas.php"

# Nominatim (OpenStreetMap) - Free geocoding service
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# Cache expiration time (24 hours - station data doesn't change frequently)
CACHE_EXPIRATION_HOURS = 24

# ==========================================================================
# Metro OAuth2 Token & Station Cache
# ==========================================================================

_metro_access_token: Optional[str] = None
_metro_token_expiry: Optional[datetime] = None
_metro_stations_cache: Optional[List[Dict[str, Any]]] = None
_metro_stations_last_load: Optional[datetime] = None
_metro_runtime_ca_bundle: Optional[str] = None
_metro_runtime_ca_bundle_leaf_fingerprint: Optional[str] = None
_metro_runtime_state: Dict[str, Optional[str]] = {
    "token_status": "unknown",
    "token_error": None,
    "request_status": "unknown",
    "request_error": None,
}

# ==========================================================================
# Metro Line Configuration
# ==========================================================================

METRO_LINES = {
    "amarela": {
        "name": "Yellow Line (Rato ↔ Odivelas)",
        "emoji": "🟡",
        "color": "#F7A71C",
        "stations": [
            "rato",
            "marquês de pombal",
            "picoas",
            "saldanha",
            "campo pequeno",
            "entrecampos",
            "cidade universitária",
            "campo grande",
            "quinta das conchas",
            "lumiar",
            "ameixoeira",
            "senhor roubado",
            "odivelas",
        ],
    },
    "azul": {
        "name": "Blue Line (Santa Apolónia ↔ Reboleira)",
        "emoji": "🔵",
        "color": "#3877BD",
        "stations": [
            "santa apolónia",
            "terreiro do paço",
            "baixa-chiado",
            "restauradores",
            "avenida",
            "marquês de pombal",
            "parque",
            "são sebastião",
            "praça de espanha",
            "jardim zoológico",
            "laranjeiras",
            "alto dos moinhos",
            "colégio militar/luz",
            "carnide",
            "pontinha",
            "alfornelos",
            "amadora este",
            "reboleira",
        ],
    },
    "verde": {
        "name": "Green Line (Telheiras ↔ Cais do Sodré)",
        "emoji": "🟢",
        "color": "#00A497",
        "stations": [
            "cais do sodré",
            "baixa-chiado",
            "rossio",
            "martim moniz",
            "intendente",
            "anjos",
            "arroios",
            "alameda",
            "areeiro",
            "roma",
            "alvalade",
            "campo grande",
            "telheiras",
        ],
    },
    "vermelha": {
        "name": "Red Line (S. Sebastião ↔ Aeroporto)",
        "emoji": "🔴",
        "color": "#E81775",
        "stations": [
            "são sebastião",
            "saldanha",
            "alameda",
            "olaias",
            "bela vista",
            "chelas",
            "olivais",
            "cabo ruivo",
            "oriente",
            "moscavide",
            "encarnação",
            "aeroporto",
        ],
    },
}

# Key Metro Stations with their lines (for routing assistance)
METRO_STATIONS = {
    # Yellow Line (Amarela)
    "rato": ["amarela"],
    "marquês de pombal": ["amarela", "azul"],
    "marques de pombal": ["amarela", "azul"],
    "marques": ["amarela", "azul"],
    "marquês": ["amarela", "azul"],
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
    # Blue Line (Azul)
    "santa apolónia": ["azul"],
    "santa apolonia": ["azul"],
    "terreiro do paço": ["azul"],
    "terreiro do paco": ["azul"],
    "baixa-chiado": ["azul", "verde"],
    "baixa chiado": ["azul", "verde"],
    "restauradores": ["azul"],
    "avenida": ["azul"],
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
    "colégio militar/luz": ["azul"],
    "colegio militar/luz": ["azul"],
    "carnide": ["azul"],
    "pontinha": ["azul"],
    "alfornelos": ["azul"],
    "amadora este": ["azul"],
    "reboleira": ["azul"],
    # Green Line (Verde)
    "cais do sodré": ["verde"],
    "cais do sodre": ["verde"],
    "rossio": ["verde"],
    "martim moniz": ["verde"],
    "intendente": ["verde"],
    "anjos": ["verde"],
    "arroios": ["verde"],
    "alameda": ["verde", "vermelha"],
    "areeiro": ["verde"],
    "roma": ["verde"],
    "alvalade": ["verde"],
    "telheiras": ["verde"],
    # Red Line (Vermelha)
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

# Metro Station ID mapping (name -> API code)
METRO_STATION_IDS = {
    "rato": "RA",
    "marquês de pombal": "MP",
    "marques de pombal": "MP",
    "marques": "MP",
    "marquês": "MP",
    "picoas": "PI",
    "saldanha": "SA",
    "campo pequeno": "CP",
    "entre campos": "EC",
    "entrecampos": "EC",
    "cidade universitária": "CU",
    "cidade universitaria": "CU",
    "campo grande": "CG",
    "quinta das conchas": "QC",
    "lumiar": "LU",
    "ameixoeira": "AX",
    "senhor roubado": "SR",
    "odivelas": "OD",
    "santa apolónia": "SP",
    "santa apolonia": "SP",
    "terreiro do paço": "TP",
    "terreiro do paco": "TP",
    "baixa-chiado": "BC",
    "baixa chiado": "BC",
    "restauradores": "RE",
    "avenida": "AV",
    "parque": "PA",
    "são sebastião": "SS",
    "sao sebastiao": "SS",
    "s. sebastião": "SS",
    "praça de espanha": "PE",
    "praca de espanha": "PE",
    "jardim zoológico": "JZ",
    "jardim zoologico": "JZ",
    "laranjeiras": "LA",
    "alto dos moinhos": "AH",
    "colégio militar": "CM",
    "colegio militar": "CM",
    "colégio militar/luz": "CM",
    "carnide": "CA",
    "pontinha": "PO",
    "alfornelos": "AF",
    "amadora este": "AS",
    "reboleira": "RB",
    "cais do sodré": "CS",
    "cais do sodre": "CS",
    "rossio": "RO",
    "martim moniz": "MM",
    "intendente": "IN",
    "anjos": "AN",
    "arroios": "AR",
    "alameda": "AM",
    "areeiro": "AE",
    "roma": "RM",
    "alvalade": "AL",
    "telheiras": "TE",
    "olaias": "OL",
    "bela vista": "BV",
    "chelas": "CH",
    "olivais": "OS",
    "cabo ruivo": "CR",
    "oriente": "OR",
    "moscavide": "MO",
    "encarnação": "EN",
    "encarnacao": "EN",
    "aeroporto": "AP",
}

# Reverse mapping (code -> name) for display
METRO_STATION_NAMES = {v: k.title() for k, v in METRO_STATION_IDS.items() if len(k) > 2}

# Metro destination ID mapping (from /infoDestinos/todos)
METRO_DESTINATIONS = {
    "33": "Reboleira",
    "34": "Amadora Este",
    "35": "Pontinha",
    "36": "Colégio Militar/Luz",
    "37": "Laranjeiras",
    "38": "São Sebastião",
    "39": "Avenida",
    "40": "Baixa-Chiado",
    "41": "Terreiro do Paço",
    "42": "Santa Apolónia",
    "43": "Odivelas",
    "44": "Lumiar",
    "45": "Campo Grande",
    "46": "Campo Pequeno",
    "48": "Rato",
    "50": "Telheiras",
    "51": "Alvalade",
    "52": "Alameda",
    "53": "Martim Moniz",
    "54": "Cais do Sodré",
    "56": "Bela Vista",
    "57": "Chelas",
    "59": "Moscavide",
    "60": "Aeroporto",
}

# Lisbon Landmarks → Nearest Metro Station Mapping
LISBON_LANDMARKS = {
    "colombo": {
        "name": "Centro Comercial Colombo",
        "metro": "colégio militar/luz",
        "line": "azul",
        "description": "Largest shopping center in the Iberian Peninsula",
    },
    "vasco da gama": {
        "name": "Centro Comercial Vasco da Gama",
        "metro": "oriente",
        "line": "vermelha",
        "description": "Shopping center at Parque das Nações",
    },
    "el corte inglés": {
        "name": "El Corte Inglés",
        "metro": "são sebastião",
        "line": "azul/vermelha",
        "description": "Department store near Marquês roundabout",
    },
    "amoreiras": {
        "name": "Amoreiras Shopping Center",
        "metro": "marquês de pombal",
        "line": "amarela/azul",
        "description": "Shopping center at Amoreiras (15 min walk from metro)",
    },
    "aeroporto": {
        "name": "Aeroporto Humberto Delgado",
        "metro": "aeroporto",
        "line": "vermelha",
        "description": "Lisbon Airport",
    },
    "oceanário": {
        "name": "Oceanário de Lisboa",
        "metro": "oriente",
        "line": "vermelha",
        "description": "Lisbon Oceanarium at Parque das Nações",
    },
    "parque das nações": {
        "name": "Parque das Nações",
        "metro": "oriente",
        "line": "vermelha",
        "description": "Expo'98 area",
    },
    "jardim zoológico": {
        "name": "Jardim Zoológico de Lisboa",
        "metro": "jardim zoológico",
        "line": "azul",
        "description": "Lisbon Zoo",
    },
    "gulbenkian": {
        "name": "Fundação Calouste Gulbenkian",
        "metro": "são sebastião",
        "line": "azul/vermelha",
        "description": "Gulbenkian Museum and Gardens",
    },
    "belém": {
        "name": "Belém",
        "metro": None,
        "alternative": "Tram 15E (Praça da Figueira) or CP Train (from Cais do Sodré)",
        "description": "Jerónimos Monastery, Belém Tower, Padrão dos Descobrimentos",
    },
    "torre de belém": {
        "name": "Torre de Belém",
        "metro": None,
        "alternative": "Tram 15E or CP Train to Belém",
        "description": "UNESCO Monument",
    },
    "mosteiro dos jerónimos": {
        "name": "Mosteiro dos Jerónimos",
        "metro": None,
        "alternative": "Tram 15E or CP Train to Belém",
        "description": "UNESCO Monument",
    },
    "castelo de são jorge": {
        "name": "Castelo de São Jorge",
        "metro": "rossio",
        "line": "verde",
        "alternative": "From Rossio metro, walk up through Alfama (15 min) or Tram 28E",
        "description": "Medieval castle with panoramic views",
    },
    "alfama": {
        "name": "Alfama",
        "metro": "terreiro do paço",
        "line": "azul",
        "alternative": "Tram 28E crosses Alfama",
        "description": "Lisbon's oldest historic neighborhood",
    },
    "jardim da estrela": {
        "name": "Jardim da Estrela",
        "short_name": "Jardim da Estrela",
        "display_name": "Jardim da Estrela",
        "metro": "rato",
        "line": "amarela",
        "description": "Historic garden in Estrela, near the Basilica and Campo de Ourique",
        "walking_hint_pt": "ao jardim",
        "walking_hint_en": "to the garden",
        "metro_walk_minutes": 9,
        "train_station": "Santos",
        "train_walk_minutes": 12,
    },
    "biblioteca nacional": {
        "name": "Biblioteca Nacional de Portugal",
        "short_name": "Biblioteca Nacional",
        "display_name": "Biblioteca Nacional de Portugal",
        "metro": "entre campos",
        "line": "amarela",
        "description": "Portugal's national library in the Campo Grande university area",
        "walking_hint_pt": "à biblioteca",
        "walking_hint_en": "to the library",
        "metro_walk_minutes": 6,
        "train_station": "Entrecampos",
        "train_walk_minutes": 9,
    },
    "biblioteca nacional de portugal": {
        "name": "Biblioteca Nacional de Portugal",
        "short_name": "Biblioteca Nacional",
        "display_name": "Biblioteca Nacional de Portugal",
        "metro": "entre campos",
        "line": "amarela",
        "description": "Portugal's national library in the Campo Grande university area",
        "walking_hint_pt": "à biblioteca",
        "walking_hint_en": "to the library",
        "metro_walk_minutes": 6,
        "train_station": "Entrecampos",
        "train_walk_minutes": 9,
    },
    "faculdade de ciências": {
        "name": "Faculdade de Ciências da Universidade de Lisboa (FCUL)",
        "short_name": "Faculdade de Ciências",
        "display_name": "Faculdade de Ciências da Universidade de Lisboa (FCUL)",
        "metro": "campo grande",
        "line": "amarela/verde",
        "description": "Science faculty of the University of Lisbon in Campo Grande",
        "walking_hint_pt": "à faculdade",
        "walking_hint_en": "to the faculty",
        "metro_walk_minutes": 4,
    },
    "faculdade de ciencias": {
        "name": "Faculdade de Ciências da Universidade de Lisboa (FCUL)",
        "short_name": "Faculdade de Ciências",
        "display_name": "Faculdade de Ciências da Universidade de Lisboa (FCUL)",
        "metro": "campo grande",
        "line": "amarela/verde",
        "description": "Science faculty of the University of Lisbon in Campo Grande",
        "walking_hint_pt": "à faculdade",
        "walking_hint_en": "to the faculty",
        "metro_walk_minutes": 4,
    },
    "faculdade de ciências da universidade de lisboa": {
        "name": "Faculdade de Ciências da Universidade de Lisboa (FCUL)",
        "short_name": "Faculdade de Ciências",
        "display_name": "Faculdade de Ciências da Universidade de Lisboa (FCUL)",
        "metro": "campo grande",
        "line": "amarela/verde",
        "description": "Science faculty of the University of Lisbon in Campo Grande",
        "walking_hint_pt": "à faculdade",
        "walking_hint_en": "to the faculty",
        "metro_walk_minutes": 4,
    },
    "faculdade de ciencias da universidade de lisboa": {
        "name": "Faculdade de Ciências da Universidade de Lisboa (FCUL)",
        "short_name": "Faculdade de Ciências",
        "display_name": "Faculdade de Ciências da Universidade de Lisboa (FCUL)",
        "metro": "campo grande",
        "line": "amarela/verde",
        "description": "Science faculty of the University of Lisbon in Campo Grande",
        "walking_hint_pt": "à faculdade",
        "walking_hint_en": "to the faculty",
        "metro_walk_minutes": 4,
    },
    "fcul": {
        "name": "Faculdade de Ciências da Universidade de Lisboa (FCUL)",
        "short_name": "FCUL",
        "display_name": "Faculdade de Ciências da Universidade de Lisboa (FCUL)",
        "metro": "campo grande",
        "line": "amarela/verde",
        "description": "Science faculty of the University of Lisbon in Campo Grande",
        "walking_hint_pt": "à faculdade",
        "walking_hint_en": "to the faculty",
        "metro_walk_minutes": 4,
    },
    "campo de ourique": {
        "name": "Campo de Ourique",
        "short_name": "Campo de Ourique",
        "display_name": "Campo de Ourique",
        "metro": "rato",
        "line": "amarela",
        "description": "Residential neighborhood west of Estrela and Amoreiras",
        "walking_hint_pt": "ao bairro",
        "walking_hint_en": "to the neighbourhood",
        "metro_walk_minutes": 14,
        "train_station": "Alcantara - Terra",
        "train_walk_minutes": 12,
    },
    "ajuda": {
        "name": "Ajuda",
        "short_name": "Ajuda",
        "display_name": "Ajuda",
        "metro": None,
        "alternative": "CP Train to Belém plus Carris connections uphill to Ajuda",
        "description": "Historic hillside district near Ajuda Palace, Belém, and the university campus",
        "train_station": "Belem",
        "train_walk_minutes": 12,
    },
    "oeiras": {
        "name": "Oeiras",
        "short_name": "Oeiras",
        "display_name": "Oeiras",
        "metro": None,
        "alternative": "CP Train via Oeiras on the Cascais Line",
        "description": "Municipality west of Lisbon served directly by CP suburban trains",
        "train_station": "Oeiras",
        "train_walk_minutes": 3,
    },
    "nova ims": {
        "name": "NOVA IMS - Information Management School",
        "short_name": "NOVA IMS",
        "display_name": "NOVA IMS - Information Management School",
        "metro": "são sebastião",
        "line": "azul/vermelha",
        "description": "Information Management School at Universidade NOVA de Lisboa's Campolide campus",
        "walking_hint_pt": "ao campus de Campolide",
        "walking_hint_en": "to the Campolide campus",
        "metro_walk_minutes": 6,
        "train_station": "Campolide",
        "train_walk_minutes": 9,
    },
    "campus de campolide": {
        "name": "Campus de Campolide da Universidade NOVA de Lisboa",
        "short_name": "Campus de Campolide",
        "display_name": "Campus de Campolide da Universidade NOVA de Lisboa",
        "metro": "são sebastião",
        "line": "azul/vermelha",
        "description": "Main Universidade NOVA de Lisboa campus in Campolide",
        "walking_hint_pt": "ao campus",
        "walking_hint_en": "to the campus",
        "metro_walk_minutes": 6,
        "train_station": "Campolide",
        "train_walk_minutes": 9,
    },
    "hospital santa maria": {
        "name": "Hospital de Santa Maria",
        "short_name": "Hospital Santa Maria",
        "display_name": "Hospital de Santa Maria",
        "metro": "cidade universitária",
        "line": "amarela",
        "description": "Major Lisbon central hospital near Cidade Universitária",
        "walking_hint_pt": "ao hospital",
        "walking_hint_en": "to the hospital",
        "metro_walk_minutes": 8,
    },
    "instituto superior tecnico": {
        "name": "Instituto Superior Técnico",
        "short_name": "Instituto Superior Técnico",
        "display_name": "Instituto Superior Técnico",
        "metro": "alameda",
        "line": "verde/vermelha",
        "description": "Main Técnico campus near Alameda",
        "walking_hint_pt": "ao campus",
        "walking_hint_en": "to the campus",
        "metro_walk_minutes": 6,
    },
    "ist": {
        "name": "Instituto Superior Técnico",
        "short_name": "Instituto Superior Técnico",
        "display_name": "Instituto Superior Técnico",
        "metro": "alameda",
        "line": "verde/vermelha",
        "description": "Main Técnico campus near Alameda",
        "walking_hint_pt": "ao campus",
        "walking_hint_en": "to the campus",
        "metro_walk_minutes": 6,
    },
    "estádio da luz": {
        "name": "Estádio da Luz",
        "short_name": "Estádio da Luz",
        "display_name": "Estádio da Luz",
        "metro": "colégio militar/luz",
        "line": "azul",
        "description": "Sport Lisboa e Benfica stadium",
        "walking_hint_pt": "ao estádio",
        "walking_hint_en": "to the stadium",
        "metro_walk_minutes": 5,
    },
    "estadio da luz": {
        "name": "Estádio da Luz",
        "short_name": "Estádio da Luz",
        "display_name": "Estádio da Luz",
        "metro": "colégio militar/luz",
        "line": "azul",
        "description": "Sport Lisboa e Benfica stadium",
        "walking_hint_pt": "ao estádio",
        "walking_hint_en": "to the stadium",
        "metro_walk_minutes": 5,
    },
    "estádio josé alvalade": {
        "name": "Estádio José Alvalade",
        "short_name": "Estádio José Alvalade",
        "display_name": "Estádio José Alvalade",
        "metro": "campo grande",
        "line": "amarela/verde",
        "description": "Sporting Clube de Portugal stadium",
        "walking_hint_pt": "ao estádio",
        "walking_hint_en": "to the stadium",
        "metro_walk_minutes": 6,
    },
    "estadio jose alvalade": {
        "name": "Estádio José Alvalade",
        "short_name": "Estádio José Alvalade",
        "display_name": "Estádio José Alvalade",
        "metro": "campo grande",
        "line": "amarela/verde",
        "description": "Sporting Clube de Portugal stadium",
        "walking_hint_pt": "ao estádio",
        "walking_hint_en": "to the stadium",
        "metro_walk_minutes": 6,
    },
    "meo arena": {
        "name": "MEO Arena",
        "short_name": "MEO Arena",
        "display_name": "MEO Arena",
        "metro": "oriente",
        "line": "vermelha",
        "description": "Major arena at Parque das Nações",
        "walking_hint_pt": "à arena",
        "walking_hint_en": "to the arena",
        "metro_walk_minutes": 7,
    },
    "altice arena": {
        "name": "Altice Arena",
        "short_name": "Altice Arena",
        "display_name": "Altice Arena",
        "metro": "oriente",
        "line": "vermelha",
        "description": "Major arena at Parque das Nações",
        "walking_hint_pt": "à arena",
        "walking_hint_en": "to the arena",
        "metro_walk_minutes": 7,
    },
}


# ==========================================================================
# Helper Functions
# ==========================================================================


def _is_cache_valid(last_load: Optional[datetime]) -> bool:
    """Checks if the cache is still valid (not expired)."""
    if last_load is None:
        return False
    hours_elapsed = (datetime.now() - last_load).total_seconds() / 3600
    return hours_elapsed < CACHE_EXPIRATION_HOURS


def _load_x509_certificate_from_bytes(certificate_bytes: bytes) -> Optional[x509.Certificate]:
    """Loads a PEM or DER X.509 certificate from raw bytes."""
    if not certificate_bytes:
        return None

    try:
        return x509.load_pem_x509_certificate(certificate_bytes)
    except ValueError:
        try:
            return x509.load_der_x509_certificate(certificate_bytes)
        except ValueError:
            return None


def _fetch_metro_leaf_certificate() -> Optional[x509.Certificate]:
    """Fetches the Metro API leaf certificate without relying on CA validation.

    This is used only to inspect the Authority Information Access extension so
    the missing intermediate certificates can be completed dynamically.
    """
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    with socket.create_connection((METRO_API_HOST, METRO_API_PORT), timeout=REQUEST_TIMEOUT) as sock:
        with context.wrap_socket(sock, server_hostname=METRO_API_HOST) as secure_socket:
            certificate_der = secure_socket.getpeercert(binary_form=True)

    if not certificate_der:
        return None

    return x509.load_der_x509_certificate(certificate_der)


def _extract_ca_issuer_urls(certificate: Optional[x509.Certificate]) -> List[str]:
    """Returns CA issuer URLs from the certificate AIA extension."""
    if certificate is None:
        return []

    try:
        access_descriptions = cast(
            Any,
            certificate.extensions.get_extension_for_oid(
                ExtensionOID.AUTHORITY_INFORMATION_ACCESS
            ).value,
        )
    except Exception:
        return []

    issuer_urls: List[str] = []
    for description in access_descriptions:
        if description.access_method != AuthorityInformationAccessOID.CA_ISSUERS:
            continue
        url = getattr(description.access_location, "value", "")
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            issuer_urls.append(url)

    return issuer_urls


def _download_certificate_from_url(url: str) -> Optional[x509.Certificate]:
    """Downloads an issuer certificate from an AIA URL."""
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.info("Unable to download Metro issuer certificate from %s: %s", url, exc)
        return None

    return _load_x509_certificate_from_bytes(response.content)


def _build_runtime_metro_ca_bundle(force_refresh: bool = False) -> Optional[str]:
    """Builds a runtime CA bundle using certifi plus dynamically discovered issuers.

    The Metro gateway currently serves an incomplete TLS chain. Instead of
    pinning a repository PEM that can go stale, inspect the live leaf
    certificate, follow its AIA issuer links, and cache the resulting bundle in
    the system temp directory.
    """
    global _metro_runtime_ca_bundle, _metro_runtime_ca_bundle_leaf_fingerprint

    try:
        leaf_certificate = _fetch_metro_leaf_certificate()
        if leaf_certificate is None:
            return None

        leaf_fingerprint = leaf_certificate.fingerprint(hashes.SHA256()).hex()[:16]
        if (
            not force_refresh
            and _metro_runtime_ca_bundle
            and _metro_runtime_ca_bundle_leaf_fingerprint == leaf_fingerprint
            and os.path.isfile(_metro_runtime_ca_bundle)
        ):
            return _metro_runtime_ca_bundle

        issuer_certificates: List[x509.Certificate] = []
        seen_fingerprints = {leaf_certificate.fingerprint(hashes.SHA256()).hex()}
        current_certificate = leaf_certificate

        for _ in range(3):
            next_certificate = None
            for issuer_url in _extract_ca_issuer_urls(current_certificate):
                candidate = _download_certificate_from_url(issuer_url)
                if candidate is None:
                    continue

                candidate_fingerprint = candidate.fingerprint(hashes.SHA256()).hex()
                if candidate_fingerprint in seen_fingerprints:
                    continue
                if candidate.subject != current_certificate.issuer:
                    continue

                seen_fingerprints.add(candidate_fingerprint)
                issuer_certificates.append(candidate)
                next_certificate = candidate
                break

            if next_certificate is None or next_certificate.issuer == next_certificate.subject:
                break

            current_certificate = next_certificate

        if not issuer_certificates:
            return None

        bundle_path = os.path.join(
            tempfile.gettempdir(),
            f"lisboa_metro_ca_bundle_{leaf_fingerprint}.pem",
        )

        with open(certifi.where(), "r", encoding="utf-8") as file:
            certifi_bundle = file.read().rstrip()

        issuer_bundle = "\n".join(
            cert.public_bytes(serialization.Encoding.PEM).decode("ascii").strip()
            for cert in issuer_certificates
        )
        combined_bundle = f"{certifi_bundle}\n{issuer_bundle}\n"

        existing_bundle = None
        if os.path.isfile(bundle_path):
            with open(bundle_path, "r", encoding="utf-8") as file:
                existing_bundle = file.read()

        if existing_bundle != combined_bundle:
            with open(bundle_path, "w", encoding="utf-8", newline="\n") as file:
                file.write(combined_bundle)

        _metro_runtime_ca_bundle = bundle_path
        _metro_runtime_ca_bundle_leaf_fingerprint = leaf_fingerprint
        return bundle_path
    except Exception as e:
        logger.info("Unable to build dynamic Metro CA bundle: %s", e)
        return None


def _resolve_metro_ssl_verify() -> bool | str:
    """Resolve SSL verification mode for the Metro API.

    Preferred secure path:
        - Set METRO_CA_BUNDLE to a PEM bundle for api.metrolisboa.pt
        - Or set METRO_SSL_VERIFY=true to use the standard trust store only
        - By default, use standard certificate verification and dynamically
          complete any missing intermediate certificates when needed

    Explicit insecure fallback:
        - Set METRO_SSL_ALLOW_INSECURE_FALLBACK=true only as a last resort.
    """
    if METRO_CA_BUNDLE:
        if os.path.isfile(METRO_CA_BUNDLE):
            return METRO_CA_BUNDLE
        logger.warning(
            "METRO_CA_BUNDLE does not exist: %s. Falling back to standard certificate verification.",
            METRO_CA_BUNDLE,
        )

    if _METRO_SSL_VERIFY_RAW in {"1", "true", "yes"}:
        return True
    if _METRO_SSL_VERIFY_RAW in {"0", "false", "no"}:
        return False

    return True


METRO_SSL_VERIFY = _resolve_metro_ssl_verify()


def _metro_request(method: str, url: str, **kwargs) -> requests.Response:
    """Perform a Metro API request with narrow SSL warning handling.

    If certificate verification fails against the Metro host, first try to
    complete the missing issuer chain dynamically from the leaf certificate's
    AIA metadata. Only retry insecurely when the caller explicitly opted into
    METRO_SSL_ALLOW_INSECURE_FALLBACK.
    """
    verify = kwargs.pop("verify", METRO_SSL_VERIFY)

    def _perform_request(verify_mode: bool | str) -> requests.Response:
        if verify_mode is False:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=_METRO_INSECURE_WARNING_PATTERN,
                    category=urllib3.exceptions.InsecureRequestWarning,
                )
                return requests.request(method, url, verify=False, **kwargs)

        return requests.request(method, url, verify=verify_mode, **kwargs)

    try:
        return _perform_request(verify)
    except requests.exceptions.SSLError as exc:
        if "api.metrolisboa.pt" not in url.lower():
            raise

        ssl_error = exc
        if verify is not False:
            dynamic_bundle = _build_runtime_metro_ca_bundle(force_refresh=True)
            if dynamic_bundle and dynamic_bundle != verify:
                try:
                    return _perform_request(dynamic_bundle)
                except requests.exceptions.SSLError as dynamic_exc:
                    ssl_error = dynamic_exc

        if verify is False or not _METRO_SSL_ALLOW_INSECURE_FALLBACK:
            raise

        logger.warning(
            "Metro API SSL verification could not be completed securely for %s. Retrying insecurely because METRO_SSL_ALLOW_INSECURE_FALLBACK is enabled: %s",
            url,
            ssl_error,
        )
        return _perform_request(False)


def fetch_json_with_retry(url: str, timeout: int = REQUEST_TIMEOUT, use_cache: bool = True) -> Optional[Any]:
    """
    Fetches JSON data from a URL with retry logic.
    Uses connection pooling and optional caching for performance.

    Args:
        url: URL to fetch from.
        timeout: Request timeout in seconds.
        use_cache: Whether to use caching (default True for real-time data with 60s TTL).

    Returns:
        JSON data if successful, None otherwise.
    """
    # Import optimization utilities for caching and connection pooling
    try:
        import hashlib

        from agent.utils.optimization import http_pool, transport_cache
        OPTIMIZATION_AVAILABLE = True
    except ImportError:
        OPTIMIZATION_AVAILABLE = False
        http_pool = None
        transport_cache = None

    # Check cache first (1 minute TTL for transport data)
    if use_cache and OPTIMIZATION_AVAILABLE and transport_cache:
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cached_result = transport_cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for {url}")
            return cached_result

    for attempt in range(MAX_RETRIES):
        try:
            # Use pooled connection if available
            if OPTIMIZATION_AVAILABLE and http_pool:
                response = http_pool.get(url, timeout=timeout)
            else:
                response = requests.get(url, timeout=timeout)

            response.raise_for_status()
            data = response.json()

            # Cache the result
            if use_cache and OPTIMIZATION_AVAILABLE and transport_cache:
                transport_cache.set(cache_key, data, ttl=60)  # 1 minute

            return data
        except requests.exceptions.Timeout:
            wait_time = BACKOFF_FACTOR**attempt
            logger.warning(f"Timeout. Retrying in {wait_time}s...")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait_time)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_FACTOR**attempt)
        except ValueError:
            logger.error("Invalid JSON response")
            return None
    return None


def get_station_lines(station_name: str) -> List[str]:
    """
    Returns the metro lines that serve a given station.

    Args:
        station_name: Name of the station (case-insensitive).

    Returns:
        List of line names (e.g., ['amarela', 'azul']).
    """
    station_lower = station_name.lower().strip()
    return METRO_STATIONS.get(station_lower, [])


def get_landmark_info(location: str) -> Optional[Dict[str, Any]]:
    """
    Returns transport information for a known or dynamically resolved Lisbon place.

    Args:
        location: Location name (case-insensitive).

    Returns:
        Landmark info with nearest metro or alternative transport.
    """
    try:
        from tools.location_resolver import (
            build_dynamic_landmark_info,
            normalize_location_text,
        )
    except ImportError:
        from location_resolver import (
            build_dynamic_landmark_info,
            normalize_location_text,
        )

    location_norm = normalize_location_text(location)

    for key, info in LISBON_LANDMARKS.items():
        if normalize_location_text(key) == location_norm:
            return info

    for key, info in LISBON_LANDMARKS.items():
        key_norm = normalize_location_text(key)
        if key_norm in location_norm or location_norm in key_norm:
            return info

    try:
        dynamic_info = build_dynamic_landmark_info(
            location,
            prefer_city=True,
            allow_aml=True,
        )
        if dynamic_info:
            return dynamic_info
    except Exception as exc:
        logger.debug("Dynamic landmark resolution failed for '%s': %s", location, exc)

    return None


def _get_metro_runtime_issue_kind() -> Optional[str]:
    """Summarizes the current official Metro API issue, if any."""
    if not METRO_CONSUMER_KEY or not METRO_CONSUMER_SECRET:
        return "missing_credentials"

    token_status = _metro_runtime_state.get("token_status")
    request_status = _metro_runtime_state.get("request_status")

    for status in (request_status, token_status):
        if status and status not in {"ok", "unknown", "token_unavailable"}:
            return status

    if token_status == "token_unavailable":
        return "unavailable"

    return None


def _build_metro_fallback_notice() -> str:
    """Builds a user-facing note when the official Metro API is unavailable."""
    issue_kind = _get_metro_runtime_issue_kind()
    if issue_kind == "missing_credentials":
        return (
            "ℹ️ Official Metro de Lisboa real-time API credentials are not configured in this environment. "
            "Showing line status from the public fallback endpoint instead.\n\n"
        )

    if issue_kind:
        return (
            "ℹ️ Official Metro de Lisboa real-time API is currently unavailable or timing out. "
            "Showing line status from the public fallback endpoint instead.\n\n"
        )

    return ""


def _build_metro_realtime_unavailable_message(subject: str) -> str:
    """Builds a clear user-facing message for unavailable official Metro real-time data."""
    issue_kind = _get_metro_runtime_issue_kind()
    if issue_kind == "missing_credentials":
        return (
            f"❌ {subject} temporarily unavailable because official Metro API credentials are not configured.\n"
            "Configure METRO_CONSUMER_KEY and METRO_CONSUMER_SECRET in .env\n"
            "Register at: https://api.metrolisboa.pt/store/"
        )

    return (
        f"❌ {subject} temporarily unavailable because the official Metro de Lisboa API is not responding right now.\n"
        "The public fallback endpoint still provides line status, but not live wait-time or frequency data."
    )


# ==========================================================================
# OAuth2 Authentication
# ==========================================================================


def _get_metro_access_token(force_refresh: bool = False) -> Optional[str]:
    """
    Gets a valid access token for the Metro de Lisboa API.

    Implements OAuth2 Client Credentials flow with automatic token refresh.

    Args:
        force_refresh: Force token refresh even if current token is valid.

    Returns:
        Access token if successful, None otherwise.
    """
    global _metro_access_token, _metro_token_expiry

    if not METRO_CONSUMER_KEY or not METRO_CONSUMER_SECRET:
        _metro_runtime_state["token_status"] = "missing_credentials"
        _metro_runtime_state["token_error"] = "missing credentials"
        logger.warning(
            "Metro API credentials not configured. Set METRO_CONSUMER_KEY and METRO_CONSUMER_SECRET in .env"
        )
        return None

    if not force_refresh and _metro_access_token and _metro_token_expiry:
        if datetime.now() < _metro_token_expiry - timedelta(minutes=5):
            return _metro_access_token

    try:
        credentials = f"{METRO_CONSUMER_KEY}:{METRO_CONSUMER_SECRET}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()

        headers = {
            "Authorization": f"Basic {encoded_credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        data = {"grant_type": "client_credentials"}

        response = _metro_request(
            "post",
            METRO_TOKEN_URL,
            headers=headers,
            data=data,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code != 200:
            logger.error(
                f"Failed to get Metro access token: HTTP {response.status_code}"
            )
            return None

        token_data = response.json()
        _metro_access_token = token_data.get("access_token")
        expires_in = token_data.get("expires_in", 3600)
        _metro_token_expiry = datetime.now() + timedelta(seconds=expires_in)
        _metro_runtime_state["token_status"] = "ok"
        _metro_runtime_state["token_error"] = None

        logger.info(f"Got new Metro access token (expires in {expires_in}s)")
        return _metro_access_token

    except requests.exceptions.Timeout as e:
        _metro_runtime_state["token_status"] = "timeout"
        _metro_runtime_state["token_error"] = str(e)
        logger.error(f"Error getting Metro token: {e}")
        return None
    except requests.exceptions.RequestException as e:
        _metro_runtime_state["token_status"] = "request_error"
        _metro_runtime_state["token_error"] = str(e)
        logger.error(f"Error getting Metro token: {e}")
        return None
    except Exception as e:
        _metro_runtime_state["token_status"] = "unavailable"
        _metro_runtime_state["token_error"] = str(e)
        logger.error(f"Error getting Metro token: {e}")
        return None


def _metro_api_request(
    endpoint: str, params: Optional[Dict] = None
) -> Optional[Dict[str, Any]]:
    """
    Makes an authenticated request to the Metro de Lisboa Official API.

    Args:
        endpoint: API endpoint (e.g., '/tempoEspera/Estacao/CG').
        params: Optional query parameters.

    Returns:
        JSON response if successful, None otherwise.
    """
    token = _get_metro_access_token()

    if not token:
        _metro_runtime_state["request_status"] = "token_unavailable"
        _metro_runtime_state["request_error"] = _metro_runtime_state.get("token_error")
        logger.warning("No Metro API token available")
        return None

    url = f"{METRO_API_BASE}{endpoint}"
    headers = {"accept": "application/json", "Authorization": f"Bearer {token}"}

    try:
        response = _metro_request(
            "get",
            url,
            headers=headers,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code == 401:
            logger.info("Metro token expired, refreshing...")
            token = _get_metro_access_token(force_refresh=True)
            if token:
                headers["Authorization"] = f"Bearer {token}"
                response = _metro_request(
                    "get",
                    url,
                    headers=headers,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )

        if response.status_code != 200:
            _metro_runtime_state["request_status"] = "http_error"
            _metro_runtime_state["request_error"] = f"HTTP {response.status_code}"
            logger.error(f"Metro API error: HTTP {response.status_code} for {endpoint}")
            return None

        _metro_runtime_state["request_status"] = "ok"
        _metro_runtime_state["request_error"] = None
        return response.json()

    except requests.exceptions.Timeout as e:
        _metro_runtime_state["request_status"] = "timeout"
        _metro_runtime_state["request_error"] = str(e)
        logger.error(f"Error calling Metro API {endpoint}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        _metro_runtime_state["request_status"] = "request_error"
        _metro_runtime_state["request_error"] = str(e)
        logger.error(f"Error calling Metro API {endpoint}: {e}")
        return None
    except Exception as e:
        _metro_runtime_state["request_status"] = "unavailable"
        _metro_runtime_state["request_error"] = str(e)
        logger.error(f"Error calling Metro API {endpoint}: {e}")
        return None


def _is_metro_api_available() -> bool:
    """Checks if the Metro Official API is available and configured."""
    return bool(METRO_CONSUMER_KEY and METRO_CONSUMER_SECRET)


# ==========================================================================
# Station Data Functions
# ==========================================================================


def load_metro_stations(force_reload: bool = False) -> List[Dict[str, Any]]:
    """
    Loads all Metro de Lisboa stations with GPS coordinates.

    Args:
        force_reload: Force refresh even if cache is valid.

    Returns:
        List of station dictionaries with stop_id, stop_name, coordinates, etc.
    """
    global _metro_stations_cache, _metro_stations_last_load

    if (
        not force_reload
        and _metro_stations_cache is not None
        and _is_cache_valid(_metro_stations_last_load)
    ):
        logger.info(
            f"Using cached Metro stations ({len(_metro_stations_cache)} stations)"
        )
        return _metro_stations_cache

    if _is_metro_api_available():
        data = _metro_api_request("/infoEstacao/todos")
        if data and data.get("codigo") == "200":
            response_data = data.get("resposta", [])
            result: List[Dict[str, Any]] = (
                response_data if response_data is not None else []
            )
            _metro_stations_cache = result
            _metro_stations_last_load = datetime.now()
            logger.info(f"Loaded {len(result)} Metro stations from Official API")
            return result

    logger.warning("Metro Official API unavailable, using static station data")

    # Fallback to hardcoded data
    _metro_stations_cache = [
        {
            "stop_id": "AM",
            "stop_name": "Alameda",
            "stop_lat": "38.7373",
            "stop_lon": "-9.13409",
            "linha": "[Verde, Vermelha]",
            "zone_id": "L",
        },
        {
            "stop_id": "AF",
            "stop_name": "Alfornelos",
            "stop_lat": "38.7606",
            "stop_lon": "-9.20471",
            "linha": "[Azul]",
            "zone_id": "C",
        },
        {
            "stop_id": "AH",
            "stop_name": "Alto dos Moinhos",
            "stop_lat": "38.7496",
            "stop_lon": "-9.17995",
            "linha": "[Azul]",
            "zone_id": "L",
        },
        {
            "stop_id": "AL",
            "stop_name": "Alvalade",
            "stop_lat": "38.7535",
            "stop_lon": "-9.14388",
            "linha": "[Verde]",
            "zone_id": "L",
        },
        {
            "stop_id": "AS",
            "stop_name": "Amadora Este",
            "stop_lat": "38.7584",
            "stop_lon": "-9.21917",
            "linha": "[Azul]",
            "zone_id": "C",
        },
        {
            "stop_id": "AX",
            "stop_name": "Ameixoeira",
            "stop_lat": "38.7799",
            "stop_lon": "-9.15999",
            "linha": "[Amarela]",
            "zone_id": "L",
        },
        {
            "stop_id": "AN",
            "stop_name": "Anjos",
            "stop_lat": "38.7266",
            "stop_lon": "-9.13503",
            "linha": "[Verde]",
            "zone_id": "L",
        },
        {
            "stop_id": "AE",
            "stop_name": "Areeiro",
            "stop_lat": "38.7426",
            "stop_lon": "-9.13381",
            "linha": "[Verde]",
            "zone_id": "L",
        },
        {
            "stop_id": "AR",
            "stop_name": "Arroios",
            "stop_lat": "38.7335",
            "stop_lon": "-9.13445",
            "linha": "[Verde]",
            "zone_id": "L",
        },
        {
            "stop_id": "AV",
            "stop_name": "Avenida",
            "stop_lat": "38.7201",
            "stop_lon": "-9.14582",
            "linha": "[Azul]",
            "zone_id": "L",
        },
        {
            "stop_id": "BC",
            "stop_name": "Baixa/Chiado",
            "stop_lat": "38.7107",
            "stop_lon": "-9.13909",
            "linha": "[Azul, Verde]",
            "zone_id": "L",
        },
        {
            "stop_id": "BV",
            "stop_name": "Bela Vista",
            "stop_lat": "38.7477",
            "stop_lon": "-9.11855",
            "linha": "[Vermelha]",
            "zone_id": "L",
        },
        {
            "stop_id": "CR",
            "stop_name": "Cabo Ruivo",
            "stop_lat": "38.7632",
            "stop_lon": "-9.10409",
            "linha": "[Vermelha]",
            "zone_id": "L",
        },
        {
            "stop_id": "CS",
            "stop_name": "Cais do Sodré",
            "stop_lat": "38.7062",
            "stop_lon": "-9.14503",
            "linha": "[Verde]",
            "zone_id": "L",
        },
        {
            "stop_id": "CG",
            "stop_name": "Campo Grande",
            "stop_lat": "38.7599",
            "stop_lon": "-9.15794",
            "linha": "[Amarela, Verde]",
            "zone_id": "L",
        },
        {
            "stop_id": "CP",
            "stop_name": "Campo Pequeno",
            "stop_lat": "38.7414",
            "stop_lon": "-9.14703",
            "linha": "[Amarela]",
            "zone_id": "L",
        },
        {
            "stop_id": "CA",
            "stop_name": "Carnide",
            "stop_lat": "38.7593",
            "stop_lon": "-9.19281",
            "linha": "[Azul]",
            "zone_id": "L",
        },
        {
            "stop_id": "CH",
            "stop_name": "Chelas",
            "stop_lat": "38.7553",
            "stop_lon": "-9.11414",
            "linha": "[Vermelha]",
            "zone_id": "L",
        },
        {
            "stop_id": "CU",
            "stop_name": "Cidade Universitária",
            "stop_lat": "38.7519",
            "stop_lon": "-9.15863",
            "linha": "[Amarela]",
            "zone_id": "L",
        },
        {
            "stop_id": "CM",
            "stop_name": "Colégio Militar/Luz",
            "stop_lat": "38.7533",
            "stop_lon": "-9.18866",
            "linha": "[Azul]",
            "zone_id": "L",
        },
        {
            "stop_id": "EC",
            "stop_name": "Entre Campos",
            "stop_lat": "38.7479",
            "stop_lon": "-9.14856",
            "linha": "[Amarela]",
            "zone_id": "L",
        },
        {
            "stop_id": "IN",
            "stop_name": "Intendente",
            "stop_lat": "38.7222",
            "stop_lon": "-9.13531",
            "linha": "[Verde]",
            "zone_id": "L",
        },
        {
            "stop_id": "JZ",
            "stop_name": "Jardim Zoológico",
            "stop_lat": "38.7422",
            "stop_lon": "-9.16872",
            "linha": "[Azul]",
            "zone_id": "L",
        },
        {
            "stop_id": "LA",
            "stop_name": "Laranjeiras",
            "stop_lat": "38.7485",
            "stop_lon": "-9.17243",
            "linha": "[Azul]",
            "zone_id": "L",
        },
        {
            "stop_id": "LU",
            "stop_name": "Lumiar",
            "stop_lat": "38.7728",
            "stop_lon": "-9.1597",
            "linha": "[Amarela]",
            "zone_id": "L",
        },
        {
            "stop_id": "MP",
            "stop_name": "Marquês de Pombal",
            "stop_lat": "38.7249",
            "stop_lon": "-9.15081",
            "linha": "[Amarela, Azul]",
            "zone_id": "L",
        },
        {
            "stop_id": "MM",
            "stop_name": "Martim Moniz",
            "stop_lat": "38.7168",
            "stop_lon": "-9.13575",
            "linha": "[Verde]",
            "zone_id": "L",
        },
        {
            "stop_id": "OD",
            "stop_name": "Odivelas",
            "stop_lat": "38.7932",
            "stop_lon": "-9.17322",
            "linha": "[Amarela]",
            "zone_id": "C",
        },
        {
            "stop_id": "OL",
            "stop_name": "Olaias",
            "stop_lat": "38.7392",
            "stop_lon": "-9.12366",
            "linha": "[Vermelha]",
            "zone_id": "L",
        },
        {
            "stop_id": "OS",
            "stop_name": "Olivais",
            "stop_lat": "38.7613",
            "stop_lon": "-9.11204",
            "linha": "[Vermelha]",
            "zone_id": "L",
        },
        {
            "stop_id": "OR",
            "stop_name": "Oriente",
            "stop_lat": "38.7678",
            "stop_lon": "-9.09977",
            "linha": "[Vermelha]",
            "zone_id": "L",
        },
        {
            "stop_id": "PA",
            "stop_name": "Parque",
            "stop_lat": "38.7297",
            "stop_lon": "-9.15028",
            "linha": "[Azul]",
            "zone_id": "L",
        },
        {
            "stop_id": "PI",
            "stop_name": "Picoas",
            "stop_lat": "38.7306",
            "stop_lon": "-9.1465",
            "linha": "[Amarela]",
            "zone_id": "L",
        },
        {
            "stop_id": "PO",
            "stop_name": "Pontinha",
            "stop_lat": "38.7624",
            "stop_lon": "-9.19693",
            "linha": "[Azul]",
            "zone_id": "C",
        },
        {
            "stop_id": "PE",
            "stop_name": "Praça de Espanha",
            "stop_lat": "38.7377",
            "stop_lon": "-9.15845",
            "linha": "[Azul]",
            "zone_id": "L",
        },
        {
            "stop_id": "QC",
            "stop_name": "Quinta das Conchas",
            "stop_lat": "38.7671",
            "stop_lon": "-9.15546",
            "linha": "[Amarela]",
            "zone_id": "L",
        },
        {
            "stop_id": "RA",
            "stop_name": "Rato",
            "stop_lat": "38.7201",
            "stop_lon": "-9.15411",
            "linha": "[Amarela]",
            "zone_id": "L",
        },
        {
            "stop_id": "RE",
            "stop_name": "Restauradores",
            "stop_lat": "38.7151",
            "stop_lon": "-9.14162",
            "linha": "[Azul]",
            "zone_id": "L",
        },
        {
            "stop_id": "RM",
            "stop_name": "Roma",
            "stop_lat": "38.7485",
            "stop_lon": "-9.14135",
            "linha": "[Verde]",
            "zone_id": "L",
        },
        {
            "stop_id": "RO",
            "stop_name": "Rossio",
            "stop_lat": "38.7138",
            "stop_lon": "-9.13896",
            "linha": "[Verde]",
            "zone_id": "L",
        },
        {
            "stop_id": "SA",
            "stop_name": "Saldanha",
            "stop_lat": "38.7353",
            "stop_lon": "-9.14558",
            "linha": "[Amarela, Vermelha]",
            "zone_id": "L",
        },
        {
            "stop_id": "SP",
            "stop_name": "Santa Apolónia",
            "stop_lat": "38.7138",
            "stop_lon": "-9.12256",
            "linha": "[Azul]",
            "zone_id": "L",
        },
        {
            "stop_id": "SS",
            "stop_name": "São Sebastião",
            "stop_lat": "38.7348",
            "stop_lon": "-9.15423",
            "linha": "[Azul, Vermelha]",
            "zone_id": "L",
        },
        {
            "stop_id": "SR",
            "stop_name": "Senhor Roubado",
            "stop_lat": "38.7858",
            "stop_lon": "-9.17215",
            "linha": "[Amarela]",
            "zone_id": "C",
        },
        {
            "stop_id": "TE",
            "stop_name": "Telheiras",
            "stop_lat": "38.7604",
            "stop_lon": "-9.16606",
            "linha": "[Verde]",
            "zone_id": "L",
        },
        {
            "stop_id": "TP",
            "stop_name": "Terreiro do Paço",
            "stop_lat": "38.7072",
            "stop_lon": "-9.13335",
            "linha": "[Azul]",
            "zone_id": "L",
        },
        {
            "stop_id": "MO",
            "stop_name": "Moscavide",
            "stop_lat": "38.7748",
            "stop_lon": "-9.10266",
            "linha": "[Vermelha]",
            "zone_id": "L",
        },
        {
            "stop_id": "EN",
            "stop_name": "Encarnação",
            "stop_lat": "38.775",
            "stop_lon": "-9.11498",
            "linha": "[Vermelha]",
            "zone_id": "L",
        },
        {
            "stop_id": "AP",
            "stop_name": "Aeroporto",
            "stop_lat": "38.7686",
            "stop_lon": "-9.12833",
            "linha": "[Vermelha]",
            "zone_id": "L",
        },
        {
            "stop_id": "RB",
            "stop_name": "Reboleira",
            "stop_lat": "38.7522",
            "stop_lon": "-9.22414",
            "linha": "[Azul]",
            "zone_id": "L",
        },
    ]
    _metro_stations_last_load = datetime.now()
    return _metro_stations_cache


def find_nearest_metro_station(
    lat: float, lon: float, max_results: int = 3, max_dist_km: float = 50.0
) -> List[Dict[str, Any]]:
    """
    Finds the nearest Metro stations to given GPS coordinates.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        max_results: Maximum stations to return.
        max_dist_km: Maximum distance in km to search.

    Returns:
        List of nearest stations with distance.
    """
    stations = load_metro_stations()
    if not stations:
        return []

    stations_with_distance = []
    for station in stations:
        try:
            station_lat = float(station.get("stop_lat", 0))
            station_lon = float(station.get("stop_lon", 0))
            distance_km = haversine_distance(lat, lon, station_lat, station_lon)
            if distance_km > max_dist_km:
                continue
            stations_with_distance.append(
                {
                    **station,
                    "distance_km": round(distance_km, 2),
                    "distance_m": int(distance_km * 1000),
                }
            )
        except (ValueError, TypeError):
            continue

    stations_with_distance.sort(key=lambda x: x["distance_m"])
    return stations_with_distance[:max_results]


def get_station_id(station_name: str) -> Optional[str]:
    """
    Gets the Metro API station ID from a station name.

    Args:
        station_name: Station name (e.g., 'Campo Grande', 'Aeroporto').

    Returns:
        Station ID (e.g., 'CG', 'AP') or None if not found.
    """
    station_lower = station_name.lower().strip()

    if station_lower in METRO_STATION_IDS:
        return METRO_STATION_IDS[station_lower]

    for name, code in METRO_STATION_IDS.items():
        if station_lower in name or name in station_lower:
            return code

    stations = load_metro_stations()
    for station in stations:
        if station_lower in station.get("stop_name", "").lower():
            return station.get("stop_id")

    return None


def _format_wait_time(seconds: int) -> str:
    """Formats waiting time in seconds to human-readable string."""
    if seconds <= 0:
        return "arriving"
    elif seconds < 60:
        return f"{seconds}s"
    elif seconds < 120:
        return f"1 min {seconds - 60}s"
    else:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        if remaining_seconds > 0:
            return f"{minutes} min {remaining_seconds}s"
        return f"{minutes} min"


def _normalize_metro_text(text: str) -> str:
    """Normalizes metro station/direction text for robust comparisons."""
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    normalized = normalized.lower().strip()
    normalized = normalized.replace("s. ", "sao ")
    normalized = re.sub(r"[^a-z0-9\s/-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _find_station_index_on_line(line_id: str, station_name: str) -> Optional[int]:
    """Returns the normalized station index on a line, if available."""
    stations = METRO_LINES.get(line_id, {}).get("stations", [])
    target = _normalize_metro_text(station_name)
    for idx, station in enumerate(stations):
        normalized_station = _normalize_metro_text(station)
        if target == normalized_station or target in normalized_station or normalized_station in target:
            return idx
    return None


def _infer_line_from_station_and_direction(station_name: str, direction: str) -> Optional[str]:
    """Infers the relevant line for a station+direction pair."""
    station_lines = get_station_lines(station_name)
    direction_normalized = _normalize_metro_text(direction)

    for line_id in station_lines:
        if any(direction_normalized == _normalize_metro_text(station) for station in METRO_LINES.get(line_id, {}).get("stations", [])):
            return line_id

    return station_lines[0] if len(station_lines) == 1 else None


def _resolve_requested_direction(
    station_name: str,
    requested_direction: str,
    available_destinations: List[str],
) -> tuple[Optional[str], Optional[str]]:
    """Resolves a requested direction to the best matching API destination label.

    Returns:
        Tuple of (resolved_destination_label, fallback_note).
    """
    if not requested_direction:
        return None, None

    requested_norm = _normalize_metro_text(requested_direction)
    for destination in available_destinations:
        destination_norm = _normalize_metro_text(destination)
        if requested_norm == destination_norm or requested_norm in destination_norm or destination_norm in requested_norm:
            return destination, None

    line_id = _infer_line_from_station_and_direction(station_name, requested_direction)
    if not line_id:
        return None, None

    station_idx = _find_station_index_on_line(line_id, station_name)
    requested_idx = _find_station_index_on_line(line_id, requested_direction)
    if station_idx is None or requested_idx is None or station_idx == requested_idx:
        return None, None

    matching_side = []
    for destination in available_destinations:
        dest_idx = _find_station_index_on_line(line_id, destination)
        if dest_idx is None:
            continue
        if requested_idx > station_idx and dest_idx > station_idx:
            matching_side.append((abs(dest_idx - requested_idx), destination))
        elif requested_idx < station_idx and dest_idx < station_idx:
            matching_side.append((abs(dest_idx - requested_idx), destination))

    if not matching_side:
        return None, None

    matching_side.sort(key=lambda item: item[0])
    resolved = matching_side[0][1]
    note = f"Platform indicator currently shows {resolved}."
    return resolved, note


def _get_metro_direction(line_id: str, start: str, end: str) -> str:
    """Helper to determine direction (terminal station) on a Metro line."""
    stations = METRO_LINES.get(line_id, {}).get("stations", [])
    if not stations:
        return ""

    import unicodedata

    def norm(t):
        return (
            "".join(
                c
                for c in unicodedata.normalize("NFD", t)
                if unicodedata.category(c) != "Mn"
            )
            .lower()
            .strip()
        )

    start_c = next((s for s in stations if norm(s) == norm(start)), None)
    if not start_c:
        start_c = next(
            (s for s in stations if norm(start) in norm(s) or norm(s) in norm(start)),
            start,
        )

    end_c = next((s for s in stations if norm(s) == norm(end)), None)
    if not end_c:
        end_c = next(
            (s for s in stations if norm(end) in norm(s) or norm(s) in norm(end)), end
        )

    try:
        idx_start = stations.index(start_c)
        idx_end = stations.index(end_c)
        if idx_start < idx_end:
            return f"→ direção **{stations[-1].title()}**"
        else:
            return f"→ direção **{stations[0].title()}**"
    except ValueError:
        return ""


# ==========================================================================
# LangChain Tools
# ==========================================================================


@tool
def get_metro_status() -> str:
    """
    Gets the current operational status of all Lisbon Metro lines.

    Returns:
        str: Status of each metro line (Yellow, Blue, Green, Red)
             with service disruption details if any.
    """
    if _is_metro_api_available():
        data = _metro_api_request("/estadoLinha/todos")

        if data and data.get("codigo") == "200":
            response_data = data.get("resposta", {})

            response = "🚇 Metro de Lisboa Status (Official API)\n"
            response += "=" * 45 + "\n\n"

            all_ok = True

            for line_key, line_info in METRO_LINES.items():
                status = response_data.get(line_key, "Unknown").strip()
                status_short = response_data.get(f"{line_key}_curta", "unknown")
                emoji = line_info["emoji"]
                name = line_info["name"]

                if status.lower() == "ok" or status_short.lower() == "normal":
                    status_emoji = "✅"
                    status_text = "Normal service"
                else:
                    status_emoji = "⚠️"
                    status_text = status if status.lower() != "ok" else status_short
                    all_ok = False

                response += f"{emoji} {name}\n"
                response += f"   {status_emoji} {status_text}\n\n"

            if all_ok:
                response += "✅ All lines operating normally."
            else:
                response += "⚠️ Some lines have service disruptions."

            return response

    # Fallback to unofficial API
    data = fetch_json_with_retry(METRO_STATUS_URL)

    if not data:
        return (
            "❌ Failed to fetch Metro status. The API may be temporarily unavailable."
        )

    response_data = data.get("resposta", {})

    if not response_data:
        return "❌ Unexpected response format from Metro API."

    response = "🚇 Metro de Lisboa Status\n"
    response += "=" * 40 + "\n\n"
    response += _build_metro_fallback_notice()

    all_ok = True

    for line_key, line_info in METRO_LINES.items():
        status = response_data.get(line_key, "Unknown").strip()
        emoji = line_info["emoji"]
        name = line_info["name"]

        if status.lower() == "ok":
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


@tool
def get_metro_wait_time(station: str, direction: Optional[str] = None) -> str:
    """
    Gets real-time waiting times for the next metro trains at a specific station.

    Args:
        station: Station name (e.g., 'Campo Grande', 'Aeroporto', 'Baixa-Chiado').
        direction: Optional requested direction/terminal to filter to one platform only.

    Returns:
        str: Formatted waiting times for all platforms at the station.
    """
    if not _is_metro_api_available():
        return _build_metro_realtime_unavailable_message("Metro wait times are")

    station_id = get_station_id(station)

    if not station_id:
        stations = load_metro_stations()
        suggestions = [
            s["stop_name"]
            for s in stations
            if station.lower()[:3] in s["stop_name"].lower()
        ][:5]

        return (
            f"❌ Station '{station}' not found.\n\n"
            f"Did you mean one of these?\n" + "\n".join(f"  • {s}" for s in suggestions)
            if suggestions
            else "Use station names like: Campo Grande, Aeroporto, Baixa-Chiado, Rossio"
        )

    data = _metro_api_request(f"/tempoEspera/Estacao/{station_id}")

    if not data or data.get("codigo") != "200":
        return (
            _build_metro_realtime_unavailable_message("Metro wait times are")
            + f"\n\nStation requested: {station}."
        )

    wait_data = data.get("resposta", [])

    if not wait_data:
        return f"❌ No waiting time data available for {station}."

    station_name = METRO_STATION_NAMES.get(station_id, station.title())
    available_destinations = [
        METRO_DESTINATIONS.get(entry.get("destino", ""), f"Destination {entry.get('destino', '')}")
        for entry in wait_data
    ]
    resolved_direction = None
    direction_note = None
    if direction:
        resolved_direction, direction_note = _resolve_requested_direction(
            station_name=station_name,
            requested_direction=direction,
            available_destinations=available_destinations,
        )

    response = f"🚇 Metro Wait Times at {station_name}\n"
    response += "=" * 50 + "\n\n"

    destinations_seen = {}

    for entry in wait_data:
        dest_id = entry.get("destino", "")
        dest_name = METRO_DESTINATIONS.get(dest_id, f"Destination {dest_id}")

        if resolved_direction and _normalize_metro_text(dest_name) != _normalize_metro_text(resolved_direction):
            continue

        try:
            wait1 = int(entry.get("tempoChegada1", "0"))
            wait2 = int(entry.get("tempoChegada2", "0"))
            wait3 = int(entry.get("tempoChegada3", "0"))
        except (ValueError, TypeError):
            continue

        if wait1 == 0 and "--" in str(entry.get("tempoChegada1", "")):
            continue

        time1 = _format_wait_time(wait1)
        time2 = _format_wait_time(wait2)
        time3 = _format_wait_time(wait3)

        line_emoji = "🚇"
        dest_lower = dest_name.lower()
        if dest_lower in ["odivelas", "rato", "campo grande", "lumiar"]:
            line_emoji = "🟡"
        elif dest_lower in ["reboleira", "santa apolónia", "terreiro do paço"]:
            line_emoji = "🔵"
        elif dest_lower in ["telheiras", "cais do sodré"]:
            line_emoji = "🟢"
        elif dest_lower in ["aeroporto", "são sebastião", "alameda"]:
            line_emoji = "🔴"

        if dest_name not in destinations_seen:
            destinations_seen[dest_name] = True

            display_direction = direction if direction else dest_name
            response += f"{line_emoji} Direction: {display_direction}\n"
            if direction_note:
                response += f"   ℹ️ {direction_note}\n"
            response += f"   ⏱️ Next train: {time1}\n"
            response += f"   ⏳ Following: {time2}, {time3}\n\n"

    if not destinations_seen:
        return f"❌ No trains currently scheduled at {station_name}."

    response += f"📍 Updated: {datetime.now().strftime('%H:%M:%S')}"

    return response


@tool
def get_metro_line_wait_times(line: str) -> str:
    """
    Gets real-time waiting times for all stations on a specific Metro line.

    Args:
        line: Line name - 'Amarela'/'Yellow', 'Azul'/'Blue',
              'Verde'/'Green', or 'Vermelha'/'Red'.

    Returns:
        str: Formatted waiting times for all stations on the line.
    """
    if not _is_metro_api_available():
        return _build_metro_realtime_unavailable_message("Metro line wait times are")

    line_map = {
        "amarela": "Amarela",
        "yellow": "Amarela",
        "amarelo": "Amarela",
        "azul": "Azul",
        "blue": "Azul",
        "verde": "Verde",
        "green": "Verde",
        "vermelha": "Vermelha",
        "red": "Vermelha",
        "vermelho": "Vermelha",
    }

    line_normalized = line_map.get(line.lower().strip())

    if not line_normalized:
        return (
            f"❌ Unknown line '{line}'.\n\n"
            "Available lines:\n"
            "  🟡 Amarela (Yellow) - Rato ↔ Odivelas\n"
            "  🔵 Azul (Blue) - Santa Apolónia ↔ Reboleira\n"
            "  🟢 Verde (Green) - Cais do Sodré ↔ Telheiras\n"
            "  🔴 Vermelha (Red) - São Sebastião ↔ Aeroporto"
        )

    data = _metro_api_request(f"/tempoEspera/Linha/{line_normalized}")

    if not data or data.get("codigo") != "200":
        return (
            _build_metro_realtime_unavailable_message("Metro line wait times are")
            + f"\n\nLine requested: {line_normalized}."
        )

    wait_data = data.get("resposta", [])

    if not wait_data:
        return f"❌ No waiting time data available for {line_normalized} line."

    line_key = line_normalized.lower()
    line_info = METRO_LINES.get(line_key, {})
    emoji = line_info.get("emoji", "🚇")
    name = line_info.get("name", line_normalized)

    response = f"{emoji} {name} - Wait Times\n"
    response += "=" * 55 + "\n\n"

    stations_data = {}

    for entry in wait_data:
        station_id = entry.get("stop_id", "")
        station_name = METRO_STATION_NAMES.get(station_id, station_id)
        dest_id = entry.get("destino", "")
        dest_name = METRO_DESTINATIONS.get(dest_id, "?")

        try:
            wait1 = int(entry.get("tempoChegada1", "0"))
        except (ValueError, TypeError):
            wait1 = 0

        if station_name not in stations_data:
            stations_data[station_name] = []

        stations_data[station_name].append({"dest": dest_name, "wait": wait1})

    for station_name in sorted(stations_data.keys()):
        directions = stations_data[station_name]

        response += f"📍 {station_name}\n"
        for d in directions[:2]:
            time_str = _format_wait_time(d["wait"])
            response += f"   → {d['dest']}: {time_str}\n"
        response += "\n"

    response += f"📍 Updated: {datetime.now().strftime('%H:%M:%S')}"

    return response


@tool
def find_nearest_metro(
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    near_location_name: Optional[str] = None,
) -> str:
    """
    Finds the nearest Metro stations to a GPS location or named place.

    Args:
        latitude: GPS latitude (e.g., 38.7548).
        longitude: GPS longitude (e.g., -9.1867).
        near_location_name: Name of a place (e.g., "Colombo", "Martim Moniz").

    Returns:
        str: Formatted list of nearest metro stations with distances.
    """
    # Resolve location if name provided
    if near_location_name and (latitude is None or longitude is None):
        from tools.carrismetropolitana_api import geocode_location

        loc = geocode_location(near_location_name)
        if loc:
            latitude = loc["lat"]
            longitude = loc["lon"]
        else:
            return f"❌ Could not resolve location '{near_location_name}'. Please provide coordinates."

    if latitude is None or longitude is None:
        return "❌ Please provide either coordinates or a location name."

    nearest = find_nearest_metro_station(latitude, longitude, max_results=5)

    if not nearest:
        return (
            "❌ Could not find nearby Metro stations.\n"
            "Make sure coordinates are within Lisbon area."
        )

    response = "🚇 Nearest Metro Stations\n"
    response += "=" * 45 + "\n\n"

    for i, station in enumerate(nearest, 1):
        name = station.get("stop_name", "Unknown")
        distance_m = station.get("distance_m", 0)
        lines = station.get("linha", "[]")

        if distance_m < 1000:
            dist_str = f"{distance_m}m"
        else:
            dist_str = f"{distance_m / 1000:.1f}km"

        walk_min = max(1, distance_m // 83)

        line_emoji = "🚇"
        if "Amarela" in lines:
            line_emoji = "🟡"
        elif "Azul" in lines:
            line_emoji = "🔵"
        elif "Verde" in lines:
            line_emoji = "🟢"
        elif "Vermelha" in lines:
            line_emoji = "🔴"

        lines_clean = lines.replace("[", "").replace("]", "")

        response += f"{i}. {line_emoji} {name}\n"
        response += f"   📏 Distance: {dist_str} (~{walk_min} min walk)\n"
        response += f"   🚇 Lines: {lines_clean}\n\n"

    return response


@tool
def get_metro_frequency(line: str, day_type: str = "weekday") -> str:
    """
    Gets the service frequency (intervals between trains) for a Metro line.

    Args:
        line: Line name - 'Amarela', 'Azul', 'Verde', or 'Vermelha'.
        day_type: 'weekday' for Monday-Friday, 'weekend' for Saturday/Sunday/Holidays.

    Returns:
        str: Formatted frequency schedule for the line.
    """
    if not _is_metro_api_available():
        return _build_metro_realtime_unavailable_message("Metro frequency data is")

    line_map = {
        "amarela": "amarela",
        "yellow": "amarela",
        "azul": "azul",
        "blue": "azul",
        "verde": "verde",
        "green": "verde",
        "vermelha": "vermelha",
        "red": "vermelha",
    }

    line_normalized = line_map.get(line.lower().strip())

    if not line_normalized:
        return f"❌ Unknown line '{line}'. Use: Amarela, Azul, Verde, or Vermelha."

    day_code = (
        "S" if day_type.lower() in ["weekday", "s", "semana", "week", "du"] else "F"
    )
    day_label = "Weekdays" if day_code == "S" else "Weekends/Holidays"

    data = _metro_api_request(f"/infoIntervalos/{line_normalized}/{day_code}")

    if not data or data.get("codigo") != "200":
        return (
            _build_metro_realtime_unavailable_message("Metro frequency data is")
            + f"\n\nLine requested: {line_normalized}."
        )

    intervals = data.get("resposta", [])

    if not intervals:
        return f"❌ No frequency data available for {line_normalized} line."

    line_key = line_normalized.lower()
    line_info = METRO_LINES.get(line_key, {})
    emoji = line_info.get("emoji", "🚇")
    name = line_info.get("name", line_normalized)

    response = f"{emoji} {name}\n"
    response += f"📅 Service Frequency ({day_label})\n"
    response += "=" * 50 + "\n\n"

    for interval in intervals:
        start = interval.get("HoraInicio", "")
        end = interval.get("HoraFim", "")
        freq = interval.get("Intervalo", "")

        try:
            parts = freq.split(":")
            minutes = int(parts[0])
            seconds = int(parts[1]) if len(parts) > 1 else 0
            freq_str = f"{minutes}:{seconds:02d}" if seconds else f"{minutes} min"
        except Exception:
            freq_str = freq

        start_short = start[:5] if start else ""
        end_short = end[:5] if end else ""

        response += f"⏰ {start_short} - {end_short}\n"
        response += f"   🚇 Train every {freq_str}\n\n"

    response += "💡 Tip: Trains are more frequent during rush hours (7:15-9:30 and 16:45-19:00)."

    return response


@tool
def get_all_metro_stations() -> str:
    """
    Lists all Metro de Lisboa stations with their lines.

    Returns:
        str: Formatted list of all 55 Metro stations organized by line.
    """
    stations = load_metro_stations()

    if not stations:
        return "❌ Failed to load Metro stations data."

    response = "🚇 Metro de Lisboa - All Stations\n"
    response += "=" * 50 + "\n\n"

    line_stations = {"Amarela": [], "Azul": [], "Verde": [], "Vermelha": []}

    for station in stations:
        name = station.get("stop_name", "")
        lines = station.get("linha", "[]")

        for line in ["Amarela", "Azul", "Verde", "Vermelha"]:
            if line in lines:
                line_stations[line].append(name)

    line_display = [
        ("Amarela", "🟡", "Yellow Line (Rato ↔ Odivelas)"),
        ("Azul", "🔵", "Blue Line (Santa Apolónia ↔ Reboleira)"),
        ("Verde", "🟢", "Green Line (Cais do Sodré ↔ Telheiras)"),
        ("Vermelha", "🔴", "Red Line (São Sebastião ↔ Aeroporto)"),
    ]

    for line_key, emoji, description in line_display:
        stations_list = sorted(line_stations[line_key])
        response += f"{emoji} {description}\n"
        response += f"   {', '.join(stations_list)}\n\n"

    response += f"📊 Total: {len(stations)} stations across 4 lines\n"
    response += "💡 Interchange stations: Campo Grande, Alameda, Saldanha, Marquês de Pombal, Baixa-Chiado, São Sebastião"

    return response


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 METRO DE LISBOA API - TEST SUITE\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")

    print("\n1. Testing get_metro_status...")
    result = get_metro_status.invoke({})
    print(result[:500])

    print("\n2. Testing get_all_metro_stations...")
    result = get_all_metro_stations.invoke({})
    print(result[:500])

    if _is_metro_api_available():
        print("\n3. Testing get_metro_wait_time...")
        result = get_metro_wait_time.invoke({"station": "Campo Grande"})
        print(result[:500])

        print("\n4. Testing get_metro_wait_time with explicit direction (Saldanha -> Odivelas)...")
        result = get_metro_wait_time.invoke({"station": "Saldanha", "direction": "Odivelas"})
        print(result[:500])

        print("\n5. Testing get_metro_line_wait_times...")
        result = get_metro_line_wait_times.invoke({"line": "amarela"})
        print(result[:500])

        print("\n6. Testing get_metro_frequency...")
        result = get_metro_frequency.invoke({"line": "verde", "day_type": "weekday"})
        print(result[:500])

        print("\n7. Testing find_nearest_metro...")
        result = find_nearest_metro.invoke({"latitude": 38.7548, "longitude": -9.1867})
        print(result[:500])
    else:
        print("\n⚠️ Metro API credentials not configured, skipping OAuth2 tests")

    print("\n\033[1;32m✅ Metro de Lisboa API tests complete!\033[0m")
