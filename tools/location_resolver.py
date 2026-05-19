# ==========================================================================
# Master Thesis
#   - André Filipe Gomes Silvestre, 20240502
#
# Shared location resolution utilities for Lisbon and the AML.
#
# Centralizes:
#   - text normalization
#   - Nominatim geocoding
#   - scope classification (Lisbon city vs AML)
#   - nearest Metro / CP enrichment
#   - dynamic landmark-style transport hints
# ==========================================================================

# Required libraries:
# pip install requests

import logging
import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import requests

from agent.utils.geographic_scope import (
    AML_MUNICIPALITY_CENTROIDS,
    AML_MUNICIPALITY_NAMES,
    normalize_scope_text,
)

try:
    from tools.utils import haversine_distance
except ImportError:
    from utils import haversine_distance

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 10
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_USER_AGENT = "LisbonUrbanAssistant/1.0 (research@novaims.pt)"
PHOTON_URL = "https://photon.komoot.io/api/"

LISBON_CITY_BOUNDS = {
    "lat_min": 38.68,
    "lat_max": 38.80,
    "lon_min": -9.24,
    "lon_max": -9.10,
}

AML_BOUNDS = {
    "lat_min": 38.40,
    "lat_max": 39.00,
    "lon_min": -9.50,
    "lon_max": -8.70,
}

MAX_DYNAMIC_METRO_WALK_KM = 1.35
MAX_DYNAMIC_CP_WALK_KM = 1.85

AML_OUTSIDE_LISBON_TOKENS = {
    normalize_scope_text(name)
    for name in AML_MUNICIPALITY_NAMES
    if normalize_scope_text(name) != "lisboa"
} | {
    "vila franca",
    "cacilhas",
    "costa da caparica",
    "caparica",
}

_CURATED_QUERY_VARIANTS = {
    "centre": ["Rossio, Lisboa, Portugal", "Baixa-Chiado, Lisboa, Portugal"],
    "center": ["Rossio, Lisboa, Portugal", "Baixa-Chiado, Lisboa, Portugal"],
    "centro": ["Rossio, Lisboa, Portugal", "Baixa-Chiado, Lisboa, Portugal"],
    "centre of lisbon": ["Rossio, Lisboa, Portugal", "Baixa-Chiado, Lisboa, Portugal"],
    "center of lisbon": ["Rossio, Lisboa, Portugal", "Baixa-Chiado, Lisboa, Portugal"],
    "city centre": ["Rossio, Lisboa, Portugal", "Baixa-Chiado, Lisboa, Portugal"],
    "city center": ["Rossio, Lisboa, Portugal", "Baixa-Chiado, Lisboa, Portugal"],
    "centro de lisboa": ["Rossio, Lisboa, Portugal", "Baixa-Chiado, Lisboa, Portugal"],
    "centro da cidade": ["Rossio, Lisboa, Portugal", "Baixa-Chiado, Lisboa, Portugal"],
    "marques": ["Marquês de Pombal, Lisboa, Portugal"],
    "marquês": ["Marquês de Pombal, Lisboa, Portugal"],
    "marques pombal": ["Marquês de Pombal, Lisboa, Portugal"],
    "marquês pombal": ["Marquês de Pombal, Lisboa, Portugal"],
    "marques de pombal": ["Marquês de Pombal, Lisboa, Portugal"],
    "marquês de pombal": ["Marquês de Pombal, Lisboa, Portugal"],
    "jardim da estrela": ["Jardim da Estrela, Lisboa, Portugal"],
    "biblioteca nacional": ["Biblioteca Nacional de Portugal, Campo Grande, Lisboa, Portugal"],
    "biblioteca nacional de portugal": ["Biblioteca Nacional de Portugal, Campo Grande, Lisboa, Portugal"],
    "faculdade de ciencias": [
        "Faculdade de Ciências da Universidade de Lisboa, Campo Grande, Lisboa, Portugal",
        "FCUL, Campo Grande, Lisboa, Portugal",
    ],
    "faculdade de ciências": [
        "Faculdade de Ciências da Universidade de Lisboa, Campo Grande, Lisboa, Portugal",
        "FCUL, Campo Grande, Lisboa, Portugal",
    ],
    "faculdade de ciencias da universidade de lisboa": [
        "Faculdade de Ciências da Universidade de Lisboa, Campo Grande, Lisboa, Portugal",
    ],
    "faculdade de ciências da universidade de lisboa": [
        "Faculdade de Ciências da Universidade de Lisboa, Campo Grande, Lisboa, Portugal",
    ],
    "fcul": ["FCUL, Campo Grande, Lisboa, Portugal"],
    "campo de ourique": ["Campo de Ourique, Lisboa, Portugal"],
    "ajuda": ["Ajuda, Lisboa, Portugal"],
    "oeiras": ["Oeiras, Portugal"],
    "tour de belem": ["Torre de Belém, Lisboa, Portugal"],
    "tour de belém": ["Torre de Belém, Lisboa, Portugal"],
    "le chateau de sao jorge": ["Castelo de São Jorge, Lisboa, Portugal"],
    "le château de são jorge": ["Castelo de São Jorge, Lisboa, Portugal"],
    "jeronimos": ["Mosteiro dos Jerónimos, Lisboa, Portugal"],
    "jeronimos monastery": ["Mosteiro dos Jerónimos, Lisboa, Portugal"],
    "mosteiro dos jeronimos": ["Mosteiro dos Jerónimos, Lisboa, Portugal"],
    "gulbenkiam": ["Museu Calouste Gulbenkian, Lisboa, Portugal"],
    "gulbenkian": ["Museu Calouste Gulbenkian, Lisboa, Portugal"],
    "maat": ["MAAT, Lisboa, Portugal", "Museu de Arte, Arquitetura e Tecnologia, Lisboa, Portugal"],
    "ccb": ["Centro Cultural de Belém, Lisboa, Portugal"],
}

_CURATED_DISPLAY_NAMES = {
    "centre": "Rossio",
    "center": "Rossio",
    "centro": "Rossio",
    "centre of lisbon": "Rossio",
    "center of lisbon": "Rossio",
    "city centre": "Rossio",
    "city center": "Rossio",
    "centro de lisboa": "Rossio",
    "centro da cidade": "Rossio",
    "marques": "Marquês de Pombal",
    "marquês": "Marquês de Pombal",
    "marques pombal": "Marquês de Pombal",
    "marquês pombal": "Marquês de Pombal",
    "marques de pombal": "Marquês de Pombal",
    "marquês de pombal": "Marquês de Pombal",
    "jardim da estrela": "Jardim da Estrela",
    "biblioteca nacional": "Biblioteca Nacional de Portugal",
    "biblioteca nacional de portugal": "Biblioteca Nacional de Portugal",
    "faculdade de ciencias": "Faculdade de Ciências da Universidade de Lisboa (FCUL)",
    "faculdade de ciências": "Faculdade de Ciências da Universidade de Lisboa (FCUL)",
    "faculdade de ciencias da universidade de lisboa": "Faculdade de Ciências da Universidade de Lisboa (FCUL)",
    "faculdade de ciências da universidade de lisboa": "Faculdade de Ciências da Universidade de Lisboa (FCUL)",
    "fcul": "Faculdade de Ciências da Universidade de Lisboa (FCUL)",
    "campo de ourique": "Campo de Ourique",
    "ajuda": "Ajuda",
    "oeiras": "Oeiras",
    "tour de belem": "Torre de Belém",
    "tour de belém": "Torre de Belém",
    "le chateau de sao jorge": "Castelo de São Jorge",
    "le château de são jorge": "Castelo de São Jorge",
    "jeronimos": "Mosteiro dos Jerónimos",
    "jeronimos monastery": "Mosteiro dos Jerónimos",
    "mosteiro dos jeronimos": "Mosteiro dos Jerónimos",
    "gulbenkiam": "Museu Calouste Gulbenkian",
    "gulbenkian": "Museu Calouste Gulbenkian",
    "maat": "MAAT",
    "ccb": "Centro Cultural de Belém",
}

_CURATED_LOCATION_POINTS: Dict[str, Dict[str, Any]] = {
    "museu nacional do azulejo": {
        "display_name": "Museu Nacional do Azulejo",
        "lat": 38.7243,
        "lon": -9.1138,
        "class": "tourism",
        "type": "museum",
        "aliases": ["national tile museum", "national museum of azulejo", "museu do azulejo"],
    },
    "museu calouste gulbenkian": {
        "display_name": "Museu Calouste Gulbenkian",
        "lat": 38.7377,
        "lon": -9.1546,
        "class": "tourism",
        "type": "museum",
        "aliases": ["gulbenkian museum", "gulbenkian", "fundacao calouste gulbenkian", "fundação calouste gulbenkian"],
    },
    "maat": {
        "display_name": "MAAT",
        "lat": 38.6958,
        "lon": -9.1941,
        "class": "tourism",
        "type": "museum",
        "aliases": [
            "maat museum of art architecture and technology",
            "maat museu de arte arquitetura e tecnologia",
            "museu de arte arquitetura e tecnologia",
            "museu de arte arquitectura e tecnologia",
        ],
    },
    "museu do fado": {
        "display_name": "Museu do Fado",
        "lat": 38.7118,
        "lon": -9.1293,
        "class": "tourism",
        "type": "museum",
        "aliases": ["fado museum"],
    },
    "museu nacional de arte antiga": {
        "display_name": "Museu Nacional de Arte Antiga",
        "lat": 38.7045,
        "lon": -9.1613,
        "class": "tourism",
        "type": "museum",
        "aliases": ["national museum of ancient art", "mnAA", "mnaa"],
    },
    "museu arqueologico do carmo": {
        "display_name": "Carmo Archaeological Museum",
        "lat": 38.7121,
        "lon": -9.1409,
        "class": "tourism",
        "type": "museum",
        "aliases": ["carmo archaeological museum", "museu arqueológico do carmo", "convento do carmo"],
    },
    "carris museum": {
        "display_name": "Carris Museum",
        "lat": 38.7037,
        "lon": -9.1788,
        "class": "tourism",
        "type": "museum",
        "aliases": ["museu da carris", "carris museum public transport museum"],
    },
    "lisbon story centre": {
        "display_name": "Lisbon Story Centre",
        "lat": 38.7075,
        "lon": -9.1367,
        "class": "tourism",
        "type": "museum",
        "aliases": ["lisboa story centre"],
    },
    "national museum of sport": {
        "display_name": "National Museum of Sport",
        "lat": 38.7154,
        "lon": -9.1419,
        "class": "tourism",
        "type": "museum",
        "aliases": ["museu nacional do desporto"],
    },
    "museu de sao roque": {
        "display_name": "Museu de São Roque",
        "lat": 38.7138,
        "lon": -9.1433,
        "class": "tourism",
        "type": "museum",
        "aliases": ["museu de são roque", "sao roque museum", "são roque museum"],
    },
    "museu colecao berardo": {
        "display_name": "Museu Coleção Berardo / CCB",
        "lat": 38.6955,
        "lon": -9.2094,
        "class": "tourism",
        "type": "museum",
        "aliases": [
            "museu coleção berardo",
            "berardo collection museum",
            "museu berardo",
            "ccb museum",
        ],
    },
    "museu nacional dos coches": {
        "display_name": "Museu Nacional dos Coches",
        "lat": 38.6978,
        "lon": -9.1984,
        "class": "tourism",
        "type": "museum",
        "aliases": ["national coach museum", "museu dos coches"],
    },
    "museu do oriente": {
        "display_name": "Museu do Oriente",
        "lat": 38.7031,
        "lon": -9.1733,
        "class": "tourism",
        "type": "museum",
        "aliases": ["orient museum", "oriente museum"],
    },
    "museu nacional de historia natural e da ciencia": {
        "display_name": "Museu Nacional de História Natural e da Ciência",
        "lat": 38.7189,
        "lon": -9.1509,
        "class": "tourism",
        "type": "museum",
        "aliases": [
            "museu nacional de história natural e da ciência",
            "national museum of natural history and science",
        ],
    },
    "museu das ilusoes lisboa": {
        "display_name": "Museu das Ilusões Lisboa",
        "lat": 38.7100,
        "lon": -9.1417,
        "class": "tourism",
        "type": "museum",
        "aliases": [
            "museu das ilusões lisboa",
            "museu das ilusoes",
            "museu das ilusões",
            "museum of illusions lisbon",
            "museum of illusions",
            "moi lisboa",
        ],
    },
    "avenida de roma": {
        "display_name": "Avenida de Roma",
        "lat": 38.7446,
        "lon": -9.1399,
        "class": "highway",
        "type": "tertiary",
        "aliases": ["av roma", "av. roma", "avenida roma"],
    },
    "avenidas novas": {
        "display_name": "Avenidas Novas",
        "lat": 38.7359,
        "lon": -9.1489,
        "class": "place",
        "type": "neighbourhood",
        "aliases": [
            "avenidas novas lisboa",
            "bairro das avenidas novas",
            "zona das avenidas novas",
            "av novas",
        ],
    },
    "amoreiras shopping center": {
        "display_name": "Amoreiras Shopping Center",
        "lat": 38.72306,
        "lon": -9.16222,
        "class": "shop",
        "type": "mall",
        "aliases": [
            "amoreiras",
            "centro comercial amoreiras",
            "centro comercial das amoreiras",
            "amoreiras shopping",
            "amoreiras centro comercial",
            "rua carlos alberto da mota pinto",
            "rua carlos alberto da mota pinto lisboa",
        ],
    },
    "nova ims": {
        "display_name": "NOVA IMS",
        "lat": 38.732462,
        "lon": -9.159921,
        "class": "amenity",
        "type": "university",
        "aliases": [
            "nova ims university",
            "nova information management school",
            "campus de campolide nova ims",
            "information management school",
        ],
    },
    "alcantara": {
        "display_name": "Alcântara",
        "lat": 38.7047,
        "lon": -9.1742,
        "class": "place",
        "type": "neighbourhood",
        "aliases": ["alcântara", "bairro de alcantara", "bairro de alcântara"],
    },
    "jardim zoologico de lisboa": {
        "display_name": "Jardim Zoológico de Lisboa",
        "lat": 38.7422,
        "lon": -9.1687,
        "class": "tourism",
        "type": "zoo",
        "aliases": ["jardim zoológico de lisboa", "lisbon zoo", "zoo de lisboa"],
    },
    "cacilhas": {
        "display_name": "Cacilhas",
        "lat": 38.6876,
        "lon": -9.1483,
        "scope": "aml",
        "class": "place",
        "type": "transport_hub",
        "aliases": ["cacilhas terminal", "terminal de cacilhas"],
    },
    "cristo rei": {
        "display_name": "Cristo Rei",
        "lat": 38.6780,
        "lon": -9.1714,
        "scope": "aml",
        "class": "tourism",
        "type": "monument",
        "aliases": ["santuario cristo rei", "santuário cristo rei", "almada cristo rei"],
    },
    "almada": {
        "display_name": "Almada",
        "lat": 38.6765,
        "lon": -9.1651,
        "scope": "aml",
        "class": "place",
        "type": "municipality",
        "aliases": ["almada centro", "centro de almada"],
    },
    "barreiro": {
        "display_name": "Barreiro",
        "lat": 38.6631,
        "lon": -9.0724,
        "scope": "aml",
        "class": "place",
        "type": "municipality",
        "aliases": ["barreiro centro", "centro do barreiro"],
    },
    "setubal": {
        "display_name": "Setúbal",
        "lat": 38.5244,
        "lon": -8.8882,
        "scope": "aml",
        "class": "place",
        "type": "municipality",
        "aliases": ["setúbal", "setubal centro", "centro de setubal", "centro de setúbal"],
    },
    "costa da caparica": {
        "display_name": "Costa da Caparica",
        "lat": 38.6446,
        "lon": -9.2356,
        "scope": "aml",
        "class": "place",
        "type": "town",
        "aliases": ["caparica"],
    },
    "cascais": {
        "display_name": "Cascais",
        "lat": 38.6979,
        "lon": -9.4215,
        "scope": "aml",
        "class": "place",
        "type": "municipality",
        "aliases": ["cascais centro", "centro de cascais"],
    },
    "sintra": {
        "display_name": "Sintra",
        "lat": 38.7989,
        "lon": -9.3869,
        "scope": "aml",
        "class": "place",
        "type": "municipality",
        "aliases": ["sintra centro", "centro de sintra"],
    },
    "oeiras": {
        "display_name": "Oeiras",
        "lat": 38.6971,
        "lon": -9.3017,
        "scope": "aml",
        "class": "place",
        "type": "municipality",
        "aliases": ["oeiras centro", "centro de oeiras"],
    },
}


def _register_official_aml_municipalities() -> None:
    """Register all official AML municipalities as broad geocoding fallbacks.

    This is not a curated case workaround: it is the system's geographic scope
    contract. These anchors are used only when live geocoding cannot resolve a
    municipality name, and the output remains marked as a broad municipality
    centre rather than an exact address.
    """
    for municipality, (lat, lon) in AML_MUNICIPALITY_CENTROIDS.items():
        key = normalize_scope_text(municipality)
        aliases = {
            key,
            municipality,
            f"{key} centro",
            f"centro de {key}",
            f"{municipality} centro",
            f"centro de {municipality}",
            f"{municipality}, Portugal",
        }
        if municipality == "Setúbal":
            aliases.update({"setubal", "setubal centro", "centro de setubal"})
        if municipality == "Vila Franca de Xira":
            aliases.update({"vila franca", "vila franca centro", "centro de vila franca"})

        existing = _CURATED_LOCATION_POINTS.get(key)
        if existing:
            existing.setdefault("scope", "aml" if key != "lisboa" else "lisbon_city")
            existing.setdefault("class", "place")
            existing.setdefault("type", "municipality")
            merged_aliases = set(existing.get("aliases", [])) | aliases
            existing["aliases"] = sorted(alias for alias in merged_aliases if alias)
            continue

        _CURATED_LOCATION_POINTS[key] = {
            "display_name": municipality,
            "lat": lat,
            "lon": lon,
            "scope": "aml" if key != "lisboa" else "lisbon_city",
            "class": "place",
            "type": "municipality",
            "aliases": sorted(alias for alias in aliases if alias),
        }


_register_official_aml_municipalities()


def _build_nominatim_search_params(query: str) -> Dict[str, Any]:
    """Builds a Nominatim query bounded to Portugal and the AML viewbox.

    Restricting the search window reduces false positives for similarly named
    places outside the system scope and keeps geocoding aligned with the AML.
    """
    viewbox = (
        f"{AML_BOUNDS['lon_min']},{AML_BOUNDS['lat_max']},"
        f"{AML_BOUNDS['lon_max']},{AML_BOUNDS['lat_min']}"
    )
    return {
        "q": query,
        "format": "jsonv2",
        "limit": 8,
        "addressdetails": 1,
        "countrycodes": "pt",
        "viewbox": viewbox,
        "bounded": 1,
    }


def normalize_location_text(text: str) -> str:
    """Normalizes free-form location text for matching.

    Args:
        text: Raw location text.

    Returns:
        Normalized accent-insensitive lowercase text.
    """
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9\s/-]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _parse_coordinate_pair(text: str) -> Optional[Tuple[float, float]]:
    """Parses a latitude/longitude pair from a free-form location string."""
    match = re.search(
        r"(?P<lat>-?\d{1,2}(?:\.\d+)?)\s*[,;/]\s*(?P<lon>-?\d{1,3}(?:\.\d+)?)",
        str(text or ""),
    )
    if not match:
        return None

    lat = _safe_float(match.group("lat"))
    lon = _safe_float(match.group("lon"))
    if lat is None or lon is None:
        return None
    if not is_within_aml(lat, lon):
        return None
    return lat, lon


_LOCATION_ADDRESS_HINT_RE = re.compile(
    r"\b(?:rua|avenida|av\.?|largo|pra[cç]a|travessa|estrada|alameda|cal[cç]ada|"
    r"campo|campus|jardim|parque|museu|biblioteca|hospital|farm[aá]cia|centro\s+comercial|"
    r"shopping|universidade|faculdade|escola|teatro|cinema|est[aá]dio|mercado|igreja|"
    r"mosteiro|torre|castelo|pal[aá]cio|palacio)\b|\b\d{4}-\d{3}\b",
    re.IGNORECASE,
)

_LOCATION_VENUE_LABEL_RE = re.compile(
    r"\b(?:centro\s+comercial|shopping|farm[aá]cia|loja|restaurante|caf[eé]|hotel|"
    r"museu|biblioteca|hospital|universidade|faculdade|escola|teatro|cinema|aeroporto|mercado)\b",
    re.IGNORECASE,
)

_LOCATION_EXACT_ADDRESS_RE = re.compile(r"\b\d+[A-Za-z]?\b|\b\d{4}-\d{3}\b")

_LOCATION_CONTEXT_EXTRACTORS: Tuple[re.Pattern, ...] = (
    re.compile(
        r"\b(?:perto|junto|ao\s+p[eé]|mais\s+pr[oó]xim[ao]s?|closest|nearest|near|around)\s+"
        r"(?:de|do|da|dos|das|to|the)?\s+(?P<location>.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:estou|tou|i(?:'m| am))\s+(?:na|no|em|at|in|on)\s+(?P<location>.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:ir|vou|quero\s+ir|preciso\s+(?:de\s+)?ir|go|travel|get)\s+"
        r"(?:desde|a\s+partir\s+(?:de|do|da|dos|das)|dos|das|do|da|de|from)?\s*"
        r"(?P<location>.+?)\s+(?:para|ao|a|à|at[eé]|to)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:from|desde|a\s+partir\s+(?:de|do|da|dos|das))\s+(?P<location>.+?)\s+"
        r"(?:to|para|ao|a|à|at[eé])\b",
        re.IGNORECASE,
    ),
)

_LOCATION_SUFFIX_SPLITTERS: Tuple[re.Pattern, ...] = (
    re.compile(r"\s*\?\s*.*$"),
    re.compile(
        r"\s+(?:e|and)\s+(?:quanto\s+(?:tempo\s+)?(?:demoro|demora|leva)|"
        r"how\s+long|tempo\s+(?:a\s+p[eé]|de\s+caminhada)|walking\s+time)\b.*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s*[,;]\s*(?:qual|quais|onde|como|diz|diga|d[aá][- ]?me|mostra|"
        r"tell|show|how|where|which|what)\b.*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s+(?:e|and)\s+(?:quero|preciso|tenho|vou|gostava|como|depois|a\s+seguir|"
        r"de\s+seguida|diz|diga|d[aá][- ]?me|mostra|h[aá]|existem|tem|t[eê]m|"
        r"need|want|then|after|show|tell|can|are\s+there|is\s+there)\b.*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s+(?:e|and)\s+(?:h[aá]|existem|tem|t[eê]m|are\s+there|is\s+there)\s+"
        r"(?:atrasos?|perturba[cç][oõ]es|avisos?|alertas?|delays?|disruptions?|warnings?|alerts?)\b.*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s+(?:para|to)\s+(?:chegar|chegares|chegarmos|ir|go|get|arrive|estar|usar|use)\b.*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s+(?:com|with)\s+(?:morada|address|endere[cç]o|dist[aâ]ncia|distance|hor[aá]rio|hours?|telefone|phone)\b.*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s+(?:sem\s+ser\s+a\s+p[eé]|without\s+walking|de\s+metro|de\s+autocarro|"
        r"de\s+comboio|by\s+metro|by\s+bus|by\s+train|op[cç][aã]o|option)\b.*$",
        re.IGNORECASE,
    ),
)

_LOCATION_LEADING_PREFIX_RE = re.compile(
    r"^(?:(?:estou|tou|i(?:'m| am))\s+(?:na|no|em|at|in|on)\s+|"
    r"(?:perto|junto)\s+(?:de|do|da|dos|das)?\s+|"
    r"(?:mais\s+pr[oó]xim[ao]s?|closest|nearest)\s+(?:de|do|da|to)?\s+|"
    r"(?:desde|from|a\s+partir\s+(?:de|do|da|dos|das)|em|na|no|at|in|para|to)\s+|"
    r"(?:o|a|os|as|the)\s+)+",
    re.IGNORECASE,
)


def _is_generic_service_request_fragment(value: str) -> bool:
    """Return whether a fragment names a service type, not a location.

    This prevents the geocoder from treating requests such as
    ``"farmácia mais próxima"`` as an actual place name. Fragments that include
    an address, area, or named venue remain eligible for normal resolution.
    """
    raw = str(value or "").strip()
    if not raw:
        return False
    if _LOCATION_EXACT_ADDRESS_RE.search(raw) or re.search(
        r"\b(?:rua|avenida|av\.?|largo|pra[cç]a|travessa|estrada|alameda|cal[cç]ada|campo|campus)\b",
        raw,
        flags=re.IGNORECASE,
    ):
        return False

    normalized = normalize_location_text(raw)
    if not normalized:
        return False

    if not re.search(
        r"\b(?:farmacia|farmacias|pharmacy|pharmacies|biblioteca|bibliotecas|library|libraries|"
        r"hospital|hospitais|school|schools|escola|escolas|mercado|mercados|market|markets|"
        r"parque|parques|park|parks|wc|restroom|restrooms|casa de banho|casas de banho|"
        r"servico|servicos|serviço|serviços|service|services)\b",
        normalized,
    ):
        return False

    remainder = re.sub(
        r"\b(?:farmacia|farmacias|pharmacy|pharmacies|biblioteca|bibliotecas|library|libraries|"
        r"hospital|hospitais|school|schools|escola|escolas|mercado|mercados|market|markets|"
        r"parque|parques|park|parks|wc|restroom|restrooms|casa de banho|casas de banho|"
        r"servico|servicos|serviço|serviços|service|services|mais|proximo|proxima|proximos|"
        r"proximas|nearest|closest|nearby|perto|junto|de|do|da|dos|das|o|a|os|as|the)\b",
        " ",
        normalized,
    )
    return not re.sub(r"\s+", " ", remainder).strip()


def _strip_location_query_tail(value: str) -> str:
    """Strip intent clauses that are not part of a location name."""
    cleaned = re.sub(r"\s+", " ", str(value or "")).strip(" .?!,;:")
    for splitter in _LOCATION_SUFFIX_SPLITTERS:
        cleaned = splitter.sub("", cleaned).strip(" .?!,;:")
    return cleaned


def clean_location_query_fragment(location_name: str) -> str:
    """Extract the location phrase from a free-form user/tool argument.

    The geocoder should receive a place, street, landmark, shop, or coordinate,
    not a whole request such as ``"Largo Camões e quero ir à Biblioteca..."``.
    This helper is deliberately generic: it removes request/transport clauses
    while preserving concrete names for Nominatim-first resolution.

    Args:
        location_name: Free-form location or accidental sentence fragment.

    Returns:
        Cleaned location phrase, or an empty string when none is recoverable.
    """
    raw = re.sub(r"\s+", " ", str(location_name or "")).strip(" .?!,;:")
    if not raw:
        return ""
    if _is_generic_service_request_fragment(raw):
        return ""

    coordinate_pair = _parse_coordinate_pair(raw)
    if coordinate_pair:
        return f"{coordinate_pair[0]},{coordinate_pair[1]}"

    located_at_match = re.match(
        r"^(?P<label>.+?)\s+que\s+fica\s+(?:na|no|em|at)\s+(?P<tail>.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    if located_at_match:
        label = _strip_location_query_tail(located_at_match.group("label"))
        tail = _strip_location_query_tail(located_at_match.group("tail"))
        if _LOCATION_VENUE_LABEL_RE.search(label) and _LOCATION_EXACT_ADDRESS_RE.search(tail):
            raw = f"{label}, {tail}"
        elif _LOCATION_VENUE_LABEL_RE.search(label):
            raw = label
        else:
            raw = tail if _LOCATION_ADDRESS_HINT_RE.search(tail) else label

    embedded_address_match = re.match(
        r"^(?P<label>.{2,80}?)\s+(?:da|do|das|dos|na|no|em)\s+"
        r"(?P<tail>(?:rua|avenida|av\.?|largo|pra[cç]a|travessa|estrada|alameda|"
        r"cal[cç]ada|campo|campus|jardim|parque)\b.*)$",
        raw,
        flags=re.IGNORECASE,
    )
    if embedded_address_match:
        label = _strip_location_query_tail(embedded_address_match.group("label"))
        tail = _strip_location_query_tail(embedded_address_match.group("tail"))
        tail = re.sub(r",?\s+em\s+([^,]+)$", r", \1", tail, flags=re.IGNORECASE)
        if _is_generic_service_request_fragment(label):
            raw = tail
        elif label and tail:
            raw = f"{label}, {tail}"

    for extractor in _LOCATION_CONTEXT_EXTRACTORS:
        match = extractor.search(raw)
        if match:
            candidate = _strip_location_query_tail(match.group("location"))
            candidate = _LOCATION_LEADING_PREFIX_RE.sub("", candidate).strip(" .?!,;:")
            if candidate:
                return candidate

    cleaned = _strip_location_query_tail(raw)
    cleaned = _LOCATION_LEADING_PREFIX_RE.sub("", cleaned).strip(" .?!,;:")
    return cleaned


_AMBIGUOUS_LOCATION_HINTS = {
    "madeira": {
        "pt": [
            "- A) 🏝️ **Ilha da Madeira** — não é acessível por transportes urbanos de Lisboa; requer avião.",
            "- B) 🚇 **Rua Humberto Madeira / Av. Ilha da Madeira, em Lisboa** — continuo abaixo com a opção urbana.",
        ],
        "en": [
            "- A) 🏝️ **Madeira island** — not reachable by Lisbon urban transport; it requires a flight.",
            "- B) 🚇 **Rua Humberto Madeira / Avenida da Ilha da Madeira, Lisbon** — continuing below with the urban option.",
        ],
    },
}

_AMBIGUITY_SAME_SITE_RADIUS_KM = 0.35
_AMBIGUITY_EXCLUDED_RESULT_TYPES = {"fast_food", "restaurant", "cafe", "café"}
_AMBIGUITY_GENERIC_LOCATION_TERMS = {
    "a", "as", "o", "os", "um", "uma", "uns", "umas",
    "de", "da", "das", "do", "dos", "em", "na", "nas", "no", "nos",
    "ao", "aos", "perto", "near", "in", "at", "the",
    "lisboa", "lisbon", "portugal",
    "rua", "avenida", "av", "praça", "praca", "largo", "estrada",
    "loja", "store", "cafe", "café", "restaurante", "restaurant",
    "taberna", "bar", "hotel", "hospital", "clinica", "clínica",
    "padaria", "farmacia", "farmácia", "centro", "comercial",
    "shopping", "shop", "mais", "proximo", "proxima",
    "proximos", "proximas", "nearest", "closest", "nearby",
}
_AMBIGUITY_LOCALITY_FIELDS = (
    "suburb", "village", "town", "city", "municipality", "county", "neighbourhood",
)
_LOCATION_LOCALITY_TERMS = {
    "alcochete", "almada", "amadora", "barreiro", "cascais", "lisboa",
    "loures", "mafra", "moita", "montijo", "odivelas", "oeiras",
    "palmela", "seixal", "sesimbra", "setubal", "setúbal", "sintra",
    "vila", "franca", "xira", "alges", "algés", "belem", "belém",
}
_REQUESTED_PLACE_TYPE_COMPATIBILITY = {
    "bar": {"bar", "pub", "restaurant", "cafe", "café", "amenity", "shop"},
    "cafe": {"cafe", "café", "restaurant", "bar", "amenity"},
    "café": {"cafe", "café", "restaurant", "bar", "amenity"},
    "clinica": {"clinic", "doctors", "hospital", "veterinary", "healthcare", "amenity"},
    "clínica": {"clinic", "doctors", "hospital", "veterinary", "healthcare", "amenity"},
    "farmacia": {"pharmacy", "chemist", "healthcare", "amenity", "shop"},
    "farmácia": {"pharmacy", "chemist", "healthcare", "amenity", "shop"},
    "hospital": {"hospital", "clinic", "veterinary", "healthcare", "amenity"},
    "hotel": {"hotel", "hostel", "guest_house", "motel", "tourism"},
    "loja": {"shop", "mall", "retail", "commercial", "furniture", "supermarket"},
    "padaria": {"bakery", "pastry", "shop", "amenity"},
    "restaurant": {"restaurant", "bar", "pub", "cafe", "amenity"},
    "restaurante": {"restaurant", "bar", "pub", "cafe", "amenity"},
    "store": {"shop", "mall", "retail", "commercial", "furniture", "supermarket"},
    "taberna": {"restaurant", "bar", "pub", "cafe", "amenity"},
}


def _location_is_specific_enough(raw_location: str) -> bool:
    """Return whether a location is specific enough to avoid ambiguity prompts."""
    cleaned = clean_location_query_fragment(raw_location)
    normalized = normalize_location_text(cleaned)
    if not cleaned or not normalized:
        return True
    if _parse_coordinate_pair(cleaned):
        return True
    # Multi-word acronyms such as "NOVA IMS" are usually institutional labels.
    # A single all-caps token can also be a retail brand (for example IKEA),
    # so keep it eligible for dynamic ambiguity checks instead of treating it
    # as automatically specific.
    if _looks_like_acronym_label(cleaned) and len(normalized.split()) > 1:
        return True
    if _LOCATION_ADDRESS_HINT_RE.search(cleaned):
        return True
    if normalized in _CURATED_QUERY_VARIANTS or normalized in _CURATED_DISPLAY_NAMES:
        return True
    if normalized in _CURATED_LOCATION_POINTS:
        return True
    distilled_query = _distilled_location_query(normalized)
    if distilled_query and distilled_query != normalized:
        return False
    if len(normalized.split()) > 4:
        return True
    return False


def _format_ambiguity_candidate(result: Dict[str, Any]) -> str:
    """Return a compact user-facing label for an ambiguous geocoder candidate."""
    display = str(result.get("display_name") or "").strip()
    if not display:
        return ""
    parts = [part.strip() for part in display.split(",") if part.strip()]
    return ", ".join(parts[:3]) if parts else display


def _candidate_coordinates(candidate: Dict[str, Any]) -> Tuple[float, float] | None:
    """Return candidate coordinates when both latitude and longitude are valid."""
    lat = _safe_float(candidate.get("lat"))
    lon = _safe_float(candidate.get("lon"))
    if lat is None or lon is None:
        return None
    return lat, lon


def _candidate_normalized_text(candidate: Dict[str, Any]) -> str:
    """Return normalized searchable text for one geocoder candidate."""
    address = candidate.get("address", {}) or {}
    address_parts = [
        str(value)
        for value in address.values()
        if isinstance(value, (str, int, float)) and str(value).strip()
    ]
    return normalize_location_text(
        " ".join([str(candidate.get("display_name") or ""), *address_parts])
    )


def _cluster_ambiguity_candidates(
    candidates: List[Dict[str, Any]],
    *,
    radius_km: float = _AMBIGUITY_SAME_SITE_RADIUS_KM,
) -> List[List[Dict[str, Any]]]:
    """Group geocoder candidates that point to the same physical site."""
    clusters: List[List[Dict[str, Any]]] = []
    for candidate in candidates:
        coordinates = _candidate_coordinates(candidate)
        if not coordinates:
            continue
        lat, lon = coordinates
        matched_cluster: List[Dict[str, Any]] | None = None
        for cluster in clusters:
            for existing in cluster:
                existing_coordinates = _candidate_coordinates(existing)
                if not existing_coordinates:
                    continue
                distance_km = haversine_distance(
                    lat,
                    lon,
                    existing_coordinates[0],
                    existing_coordinates[1],
                )
                if distance_km <= radius_km:
                    matched_cluster = cluster
                    break
            if matched_cluster is not None:
                break
        if matched_cluster is None:
            clusters.append([candidate])
        else:
            matched_cluster.append(candidate)
    return clusters


def _best_cluster_candidate(cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Choose the most useful representative from a candidate cluster."""
    preferred = [
        candidate for candidate in cluster
        if str(candidate.get("type") or "").lower() not in _AMBIGUITY_EXCLUDED_RESULT_TYPES
    ]
    pool = preferred or cluster
    return max(
        pool,
        key=lambda item: (
            float(item.get("score", 0.0)),
            _safe_float(item.get("importance")) or 0.0,
            len(str(item.get("display_name") or "")),
        ),
    )


def _candidate_locality_score(candidate: Dict[str, Any]) -> int:
    """Score whether a candidate contains useful locality details."""
    address = candidate.get("address", {}) or {}
    score = 0
    seen: set[str] = set()
    for field in _AMBIGUITY_LOCALITY_FIELDS:
        value = str(address.get(field) or "").strip()
        normalized_value = normalize_location_text(value)
        if not normalized_value or normalized_value in {"portugal", "lisboa", "lisbon"}:
            continue
        if normalized_value in seen:
            continue
        seen.add(normalized_value)
        score += 1
    return score


def _best_brand_cluster_candidate(cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Choose the representative that best identifies a distinct brand site."""
    preferred = [
        candidate for candidate in cluster
        if str(candidate.get("type") or "").lower() not in _AMBIGUITY_EXCLUDED_RESULT_TYPES
    ]
    pool = preferred or cluster
    return max(
        pool,
        key=lambda item: (
            _candidate_locality_score(item),
            float(item.get("score", 0.0)),
            _safe_float(item.get("importance")) or 0.0,
        ),
    )


def _meaningful_location_query_terms(normalized_query: str) -> set[str]:
    """Return query terms that can distinguish one place from another."""
    return {
        term for term in normalized_query.split()
        if len(term) >= 3 and term not in _AMBIGUITY_GENERIC_LOCATION_TERMS
    }


def _distilled_location_query(normalized_query: str) -> str:
    """Return a compact brand/place query after removing generic request words."""
    meaningful_terms = _meaningful_location_query_terms(normalized_query)
    if not meaningful_terms:
        return ""
    ordered_terms = [
        term for term in normalized_query.split()
        if term in meaningful_terms
    ]
    return " ".join(ordered_terms).strip()


def _candidate_matches_all_terms(candidate: Dict[str, Any], terms: set[str]) -> bool:
    """Return whether a candidate contains every distinguishing query term."""
    candidate_text = _candidate_normalized_text(candidate)
    return bool(terms) and all(term in candidate_text for term in terms)


def _query_terms_missing_from_display(raw_query: str, display_name: str) -> bool:
    """Return whether the resolved display name lost user-provided qualifiers."""
    query_terms = _meaningful_location_query_terms(normalize_location_text(raw_query))
    if not query_terms:
        return False
    display_norm = normalize_location_text(display_name)
    return any(term not in display_norm for term in query_terms)


def _candidate_matches_distinctive_name_terms(
    raw_query: str,
    candidate: Dict[str, Any],
) -> bool:
    """Return whether a candidate preserves the requested place name."""
    query_terms = _meaningful_location_query_terms(normalize_location_text(raw_query))
    if not query_terms:
        return True
    name_terms = {
        term for term in query_terms
        if term not in _LOCATION_LOCALITY_TERMS
    }
    required_terms = name_terms or query_terms
    candidate_text = _candidate_normalized_text(candidate)
    return all(term in candidate_text for term in required_terms)


def _candidate_matches_requested_place_type(
    raw_query: str,
    candidate: Dict[str, Any],
) -> bool:
    """Return whether a candidate's OSM type fits an explicit venue type."""
    normalized_query = normalize_location_text(raw_query)
    requested_types = [
        term for term in _REQUESTED_PLACE_TYPE_COMPATIBILITY
        if re.search(rf"\b{re.escape(term)}\b", normalized_query)
    ]
    if not requested_types:
        return True

    candidate_text = _candidate_normalized_text(candidate)
    candidate_type_blob = normalize_location_text(
        " ".join(
            [
                str(candidate.get("class") or ""),
                str(candidate.get("type") or ""),
                candidate_text,
            ]
        )
    )
    for requested_type in requested_types:
        if requested_type in candidate_text:
            continue
        compatible_tokens = _REQUESTED_PLACE_TYPE_COMPATIBILITY[requested_type]
        if not any(token in candidate_type_blob for token in compatible_tokens):
            return False
    return True


def _query_mentions_explicit_place_type(raw_query: str) -> bool:
    """Return whether the query names a venue/service type."""
    normalized_query = normalize_location_text(raw_query)
    return any(
        re.search(rf"\b{re.escape(term)}\b", normalized_query)
        for term in _REQUESTED_PLACE_TYPE_COMPATIBILITY
    )


def _candidate_is_plausible_for_location_query(
    raw_query: str,
    candidate: Dict[str, Any],
) -> bool:
    """Return whether a geocoder result is safe enough for routing."""
    return (
        _candidate_matches_distinctive_name_terms(raw_query, candidate)
        and _candidate_matches_requested_place_type(raw_query, candidate)
    )


def _format_brand_ambiguity_candidate(candidate: Dict[str, Any], brand_norm: str) -> str:
    """Format a brand candidate with locality, not duplicate road variants."""
    display = str(candidate.get("display_name") or "").strip()
    if not display:
        return ""
    parts = [part.strip() for part in display.split(",") if part.strip()]
    primary = parts[0] if parts else display
    address = candidate.get("address", {}) or {}

    locality_parts: List[str] = []
    seen: set[str] = set()
    for field in _AMBIGUITY_LOCALITY_FIELDS:
        value = str(address.get(field) or "").strip()
        normalized_value = normalize_location_text(value)
        if not normalized_value or normalized_value in {"portugal", "lisboa", "lisbon"}:
            continue
        if normalized_value in seen:
            continue
        seen.add(normalized_value)
        locality_parts.append(value)

    if not locality_parts:
        for part in parts[1:]:
            normalized_part = normalize_location_text(part)
            if (
                not normalized_part
                or normalized_part in {"portugal", "lisboa", "lisbon"}
                or re.fullmatch(r"\d{4}-\d{3}", normalized_part)
                or re.search(r"\b(?:rua|avenida|estrada|en|viaduto)\b", normalized_part)
            ):
                continue
            if normalized_part not in seen:
                seen.add(normalized_part)
                locality_parts.append(part)
            if len(locality_parts) == 2:
                break
    if not locality_parts:
        for part in parts[1:]:
            normalized_part = normalize_location_text(part)
            if (
                not normalized_part
                or normalized_part in {"portugal", "lisboa", "lisbon"}
                or re.fullmatch(r"\d{4}-\d{3}", normalized_part)
            ):
                continue
            locality_parts.append(part)
            break

    primary_norm = normalize_location_text(primary)
    if primary_norm == brand_norm and locality_parts:
        label_parts = [f"{primary} {locality_parts[0]}"]
        locality_parts = locality_parts[1:]
    else:
        label_parts = [primary]

    label_norm = normalize_location_text(" ".join(label_parts))
    for locality in locality_parts:
        locality_norm = normalize_location_text(locality)
        if locality_norm and locality_norm not in label_norm:
            label_parts.append(locality)
            label_norm = normalize_location_text(" ".join(label_parts))
        if len(label_parts) == 3:
            break
    return ", ".join(label_parts)


def _build_dynamic_location_ambiguity_hints(raw_location: str) -> List[str]:
    """Find generic Nominatim alternatives for underspecified place names."""
    cleaned = clean_location_query_fragment(raw_location)
    normalized = normalize_location_text(cleaned)
    if _location_is_specific_enough(cleaned):
        return []

    try:
        if _resolve_known_metro_station(cleaned) or _resolve_known_cp_station(cleaned):
            return []
    except NameError:
        pass
    try:
        from tools.metrolisboa_api import get_landmark_info

        landmark = get_landmark_info(cleaned)
        if landmark and not landmark.get("dynamic"):
            return []
    except Exception:
        pass

    candidates: List[Dict[str, Any]] = []
    seen: set[Tuple[float, float, str]] = set()
    distilled_query = _distilled_location_query(normalized)
    query_variants = _build_query_variants(cleaned)
    if distilled_query and distilled_query != cleaned:
        query_variants = [
            distilled_query,
            f"{distilled_query}, Lisboa, Portugal",
            f"{distilled_query}, Lisbon, Portugal",
            *query_variants,
        ]
    for query in list(dict.fromkeys(query_variants))[:6]:
        for result in _fetch_nominatim_results_cached(query):
            lat = _safe_float(result.get("lat"))
            lon = _safe_float(result.get("lon"))
            if lat is None or lon is None:
                continue
            scope = classify_coordinate_scope(lat, lon)
            if scope == "outside_scope":
                continue
            key = (
                round(lat, 5),
                round(lon, 5),
                normalize_location_text(result.get("display_name", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            scored = dict(result)
            scored["lat"] = lat
            scored["lon"] = lon
            scored["scope"] = scope
            scored["score"] = _score_nominatim_result(
                scored,
                query_norm=normalized,
                prefer_city=True,
            )
            candidates.append(scored)
        for result in _fetch_photon_results_cached(query):
            lat = _safe_float(result.get("lat"))
            lon = _safe_float(result.get("lon"))
            if lat is None or lon is None:
                continue
            key = (
                round(lat, 5),
                round(lon, 5),
                normalize_location_text(result.get("display_name", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            scored = dict(result)
            scored["score"] = _score_nominatim_result(
                scored,
                query_norm=normalized,
                prefer_city=True,
            )
            candidates.append(scored)

    plausible_candidates = [
        candidate for candidate in candidates
        if _candidate_is_plausible_for_location_query(cleaned, candidate)
    ]
    if plausible_candidates:
        candidates = plausible_candidates
    elif _query_mentions_explicit_place_type(cleaned) and _meaningful_location_query_terms(normalized):
        return [f"__NO_CLEAR_MATCH__:{cleaned}"]

    if len(candidates) < 2:
        return []

    candidates.sort(
        key=lambda item: (
            item.get("score", 0.0),
            _safe_float(item.get("importance")) or 0.0,
        ),
        reverse=True,
    )

    def _primary_display_key(candidate: Dict[str, Any]) -> str:
        """Return the normalized leading display-name segment for ambiguity ranking."""
        primary = normalize_location_text(str(candidate.get("display_name") or "").split(",")[0])
        return re.sub(r"^(?:a|o|the)\s+", "", primary).strip()

    query_terms = _meaningful_location_query_terms(normalized)
    exact_term_candidates: List[Dict[str, Any]] = []
    if len(normalized.split()) > 1 and query_terms:
        exact_term_candidates = [
            candidate for candidate in candidates[:12]
            if _candidate_matches_all_terms(candidate, query_terms)
        ]
        if exact_term_candidates:
            exact_term_clusters = _cluster_ambiguity_candidates(exact_term_candidates)
            if len(exact_term_clusters) == 1:
                return []

    brand_query_norm = distilled_query or normalized
    if len(brand_query_norm.split()) <= 3 and len(brand_query_norm) >= 3:
        brand_like_candidates = [
            candidate for candidate in candidates[:12]
            if (
                _primary_display_key(candidate) == brand_query_norm
                or _primary_display_key(candidate).startswith(f"{brand_query_norm} ")
            )
        ]
        if not brand_like_candidates and len(normalized.split()) > 1 and not exact_term_candidates:
            partial_brand_candidates = [
                candidate for candidate in candidates[:12]
                if brand_query_norm.startswith(f"{_primary_display_key(candidate)} ")
            ]
            if partial_brand_candidates:
                preferred_partial_candidates = [
                    candidate for candidate in partial_brand_candidates
                    if str(candidate.get("type") or "").lower() not in _AMBIGUITY_EXCLUDED_RESULT_TYPES
                ]
                partial_clusters = _cluster_ambiguity_candidates(
                    preferred_partial_candidates or partial_brand_candidates
                )
                if len(partial_clusters) == 1:
                    return []
                return [f"__NO_CLEAR_MATCH__:{cleaned}"]
        preferred_brand_candidates = [
            candidate for candidate in brand_like_candidates
            if str(candidate.get("type") or "").lower() not in _AMBIGUITY_EXCLUDED_RESULT_TYPES
        ]
        if len(preferred_brand_candidates) >= 2:
            brand_clusters = _cluster_ambiguity_candidates(preferred_brand_candidates)
            labels: List[str] = []
            seen_labels: set[str] = set()
            for cluster in brand_clusters:
                label = _format_brand_ambiguity_candidate(
                    _best_brand_cluster_candidate(cluster),
                    brand_query_norm,
                )
                label_key = normalize_location_text(label)
                if label and label_key not in seen_labels:
                    seen_labels.add(label_key)
                    labels.append(label)
                if len(labels) == 3:
                    break
            if len(labels) >= 2:
                return labels

    top_score_for_exact = float(candidates[0].get("score", 0.0))
    exact_primary_candidates = [
        candidate for candidate in candidates[:10]
        if _primary_display_key(candidate) == normalized
        and top_score_for_exact - float(candidate.get("score", 0.0)) <= 0.5
    ]
    landmark_types = {
        "mall", "attraction", "museum", "university", "college", "school",
        "hospital", "pharmacy", "library", "theatre", "restaurant", "bakery",
        "supermarket", "hotel", "monument", "administrative", "city", "town",
        "village", "municipality", "suburb",
    }
    if (
        len(exact_primary_candidates) == 1
        and str(exact_primary_candidates[0].get("type") or "").lower() in landmark_types
    ):
        return []
    if len(candidates) >= 2:
        top_candidate = candidates[0]
        second_score = float(candidates[1].get("score", 0.0))
        top_score_value = float(top_candidate.get("score", 0.0))
        top_display_first = _primary_display_key(top_candidate)
        exact_named_landmark = bool(
            top_display_first
            and (
                top_display_first == normalized
                or normalized in top_display_first
            )
            and (top_score_value - second_score) >= 3.0
        )
        if exact_named_landmark:
            return []
    top_score = float(candidates[0].get("score", 0.0))
    close_candidates = [
        candidate
        for candidate in candidates[:5]
        if top_score - float(candidate.get("score", 0.0)) <= 1.25
    ]
    labels: List[str] = []
    seen_labels: set[str] = set()
    for cluster in _cluster_ambiguity_candidates(close_candidates):
        label = _format_ambiguity_candidate(_best_cluster_candidate(cluster))
        label_key = normalize_location_text(label)
        if label and label_key not in seen_labels:
            seen_labels.add(label_key)
            labels.append(label)
        if len(labels) == 3:
            break
    if labels and query_terms:
        relevant_labels = [
            label for label in labels
            if any(term in normalize_location_text(label) for term in query_terms)
        ]
        if not relevant_labels and re.search(
            r"\b(?:loja|store|cafe|café|restaurante|restaurant|taberna|bar|hotel|"
            r"hospital|cl[ií]nica|padaria|farm[áa]cia)\b",
            cleaned,
            flags=re.IGNORECASE,
        ):
            return [f"__NO_CLEAR_MATCH__:{cleaned}"]
        labels = relevant_labels or labels
    return labels if len(labels) >= 2 else []


def build_location_ambiguity_preamble(
    origin: str = "",
    destination: str = "",
    *,
    language: str = "pt",
) -> str:
    """Build a user-facing disambiguation note for bare ambiguous locations."""
    selected_language = "pt" if language == "pt" else "en"
    blocks: list[str] = []
    seen: set[str] = set()

    for raw_location in (origin, destination):
        token = normalize_location_text(raw_location)
        if token in {"lisbon", "lisboa", "lisboa portugal", "lisbon portugal"}:
            continue
        if token in seen:
            continue
        seen.add(token)
        if token in _AMBIGUOUS_LOCATION_HINTS:
            hints = _AMBIGUOUS_LOCATION_HINTS[token][selected_language]
        else:
            dynamic_hints = _build_dynamic_location_ambiguity_hints(raw_location)
            if not dynamic_hints:
                continue
            if len(dynamic_hints) == 1 and dynamic_hints[0].startswith("__NO_CLEAR_MATCH__:"):
                requested = dynamic_hints[0].split(":", 1)[1].strip() or raw_location
                if selected_language == "pt":
                    blocks.append(
                        f"⚠️ **Preciso de confirmar '{raw_location}':** não encontrei uma correspondência clara para "
                        f"**{requested}** em Lisboa/AML. Indica a morada, centro comercial, zona ou ponto de referência exato."
                    )
                else:
                    blocks.append(
                        f"⚠️ **I need to confirm '{raw_location}':** I could not find a clear match for "
                        f"**{requested}** in Lisbon/AML. Provide the exact address, shopping centre, area, or landmark."
                    )
                continue
            if selected_language == "pt":
                hints = [
                    f"- {chr(65 + index)}) 📍 **{hint}**"
                    for index, hint in enumerate(dynamic_hints)
                ]
                hints.append("- Indica a morada, zona ou ponto de referência se nenhuma destas opções for a pretendida.")
            else:
                hints = [
                    f"- {chr(65 + index)}) 📍 **{hint}**"
                    for index, hint in enumerate(dynamic_hints)
                ]
                hints.append("- Specify the address, area, or landmark if none of these options is what you mean.")
        heading = (
            f"⚠️ **Ambiguidade em '{raw_location}':** posso estar a interpretar uma destas opções:"
            if selected_language == "pt"
            else f"⚠️ **Ambiguity in '{raw_location}':** I may be interpreting one of these options:"
        )
        blocks.append("\n".join([heading, *hints]))

    return "\n\n".join(blocks)


def _looks_like_acronym_label(text: str) -> bool:
    """Returns whether a label looks like an acronym that should preserve casing.

    Args:
        text: Candidate label.

    Returns:
        True when the label should keep its original casing.
    """
    stripped = str(text or "").strip()
    return bool(re.fullmatch(r"(?:[A-Z0-9]{2,}(?:[\s/-][A-Z0-9]{2,})*)", stripped))


def _safe_float(value: Any) -> Optional[float]:
    """Safely converts a value to float.

    Args:
        value: Value to convert.

    Returns:
        Parsed float or None.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_within_lisbon_city(lat: Optional[float], lon: Optional[float]) -> bool:
    """Returns whether the coordinates are within Lisbon city bounds.

    Args:
        lat: Latitude.
        lon: Longitude.

    Returns:
        True if coordinates are inside the configured Lisbon city box.
    """
    if lat is None or lon is None:
        return False
    return (
        LISBON_CITY_BOUNDS["lat_min"] <= lat <= LISBON_CITY_BOUNDS["lat_max"]
        and LISBON_CITY_BOUNDS["lon_min"] <= lon <= LISBON_CITY_BOUNDS["lon_max"]
    )


def is_within_aml(lat: Optional[float], lon: Optional[float]) -> bool:
    """Returns whether the coordinates are within AML bounds.

    Args:
        lat: Latitude.
        lon: Longitude.

    Returns:
        True if coordinates are inside the configured AML box.
    """
    if lat is None or lon is None:
        return False
    return (
        AML_BOUNDS["lat_min"] <= lat <= AML_BOUNDS["lat_max"]
        and AML_BOUNDS["lon_min"] <= lon <= AML_BOUNDS["lon_max"]
    )


def classify_coordinate_scope(lat: Optional[float], lon: Optional[float]) -> str:
    """Classifies coordinates into Lisbon city, AML, or outside scope.

    Args:
        lat: Latitude.
        lon: Longitude.

    Returns:
        One of: lisbon_city, aml, outside_scope.
    """
    if is_within_lisbon_city(lat, lon):
        return "lisbon_city"
    if is_within_aml(lat, lon):
        return "aml"
    return "outside_scope"


def _query_mentions_aml_outside_lisbon(query_norm: str) -> bool:
    """Detects whether the query explicitly mentions AML locations outside Lisbon.

    Args:
        query_norm: Normalized query string.

    Returns:
        True if the text clearly mentions AML municipalities outside Lisbon city.
    """
    return any(token in query_norm for token in AML_OUTSIDE_LISBON_TOKENS)


def _looks_like_non_station_poi_query(query_norm: str) -> bool:
    """Return whether a query names a POI that should not fuzzy-match a station.

    Fuzzy station matching is useful for typos such as "Marques Pombal", but it
    is unsafe when the text clearly describes a venue, for example
    "Museu do Oriente". In that case, matching the embedded token "Oriente" as
    a station corrupts downstream distance and route calculations.
    """
    if not query_norm:
        return False

    station_context = re.search(
        r"\b(?:station|estacao|estacao de|metro|train|comboio|cp|terminal|stop|paragem)\b",
        query_norm,
    )
    if station_context:
        return False

    # Street, neighbourhood, and venue wording should be resolved as places
    # first. Otherwise fuzzy matching can incorrectly map "Avenida de Roma" to
    # the Metro station "Avenida", or broad districts such as "Alcântara" to a
    # specific rail platform without the user having asked for trains.
    exact_station_safe_names = {"avenida", "oriente", "rossio", "santos", "belem"}
    if query_norm not in exact_station_safe_names and re.search(
        r"\b(?:rua|avenida|av|praca|praça|largo|estrada|travessa|bairro)\b",
        query_norm,
    ):
        return True

    poi_patterns = [
        r"\bmuseu\b",
        r"\bmuseum\b",
        r"\bzoo\b",
        r"\bzoologico\b",
        r"\bmonumento\b",
        r"\bmonument\b",
        r"\bmosteiro\b",
        r"\bmonastery\b",
        r"\bigreja\b",
        r"\bchurch\b",
        r"\bcastelo\b",
        r"\bcastle\b",
        r"\bpalacio\b",
        r"\bpalace\b",
        r"\bcentro cultural\b",
        r"\bcultural centre\b",
        r"\bcultural center\b",
        r"\buniversidade\b",
        r"\buniversity\b",
        r"\bfaculdade\b",
        r"\bcampus\b",
        r"\bhospital\b",
        r"\bclinica\b",
        r"\bclinic\b",
        r"\balcantara\b",
    ]
    return any(re.search(pattern, query_norm) for pattern in poi_patterns)


def _build_query_variants(location_name: str) -> List[str]:
    """Builds a small ordered list of Nominatim query variants.

    Args:
        location_name: Raw location query.

    Returns:
        Ordered query variants for Nominatim.
    """
    clean_name = str(location_name or "").strip()
    normalized_name = normalize_location_text(clean_name)
    distilled_name = _distilled_location_query(normalized_name)
    distilled_variants: List[str] = []
    if distilled_name and distilled_name != normalized_name:
        distilled_variants.extend(
            [
                distilled_name,
                f"{distilled_name}, Lisboa, Portugal",
                f"{distilled_name}, Lisbon, Portugal",
            ]
        )
    comma_variants: List[str] = []
    comma_parts = [part.strip() for part in clean_name.split(",") if part.strip()]
    if len(comma_parts) > 1:
        for start_index in range(1, len(comma_parts)):
            tail = ", ".join(comma_parts[start_index:])
            if _LOCATION_ADDRESS_HINT_RE.search(tail) or _LOCATION_EXACT_ADDRESS_RE.search(tail):
                comma_variants.extend(
                    [
                        tail,
                        f"{tail}, Lisboa, Portugal",
                        f"{tail}, Lisbon, Portugal",
                    ]
                )
                break

        label_without_type = re.sub(
            r"^(?:farm[aá]cia|loja|restaurante|caf[eé]|hotel|museu|biblioteca|hospital)\s+",
            "",
            comma_parts[0],
            flags=re.IGNORECASE,
        ).strip()
        if label_without_type and label_without_type != comma_parts[0]:
            comma_variants.append(label_without_type)

    variants = [
        clean_name,
        *distilled_variants,
        *comma_variants,
        f"{clean_name}, Lisboa, Portugal",
        f"{clean_name}, Lisbon, Portugal",
        f"{clean_name}, Portugal",
        *_CURATED_QUERY_VARIANTS.get(normalized_name, []),
    ]
    return [variant for variant in dict.fromkeys(v.strip() for v in variants if v.strip())]


def _resolve_curated_location_point(location_name: str) -> Optional[Dict[str, Any]]:
    """Resolve stable Lisbon landmarks from the local gazetteer as a fallback."""
    query_clean = str(location_name or "").strip()
    query_norm = normalize_location_text(query_clean)
    if not query_norm:
        return None

    best_key: Optional[str] = None
    best_score = 0.0
    for canonical_key, payload in _CURATED_LOCATION_POINTS.items():
        aliases = [canonical_key, *payload.get("aliases", [])]
        for alias in aliases:
            alias_norm = normalize_location_text(str(alias))
            if not alias_norm:
                continue
            if query_norm == alias_norm:
                score = 1.0
            elif alias_norm in query_norm:
                score = 0.90
            else:
                score = SequenceMatcher(None, query_norm, alias_norm).ratio()
            if score > best_score:
                best_key = canonical_key
                best_score = score

    if not best_key or best_score < 0.88:
        return None

    payload = _CURATED_LOCATION_POINTS[best_key]
    lat = _safe_float(payload.get("lat"))
    lon = _safe_float(payload.get("lon"))
    if lat is None or lon is None:
        return None

    scope = str(payload.get("scope") or classify_coordinate_scope(lat, lon))

    return {
        "display_name": str(payload.get("display_name") or query_clean),
        "full_display_name": str(payload.get("display_name") or query_clean),
        "lat": lat,
        "lon": lon,
        "type": str(payload.get("type") or "unknown"),
        "class": str(payload.get("class") or "unknown"),
        "importance": 1.0,
        "address": {},
        "query_used": best_key,
        "scope": scope,
        "match_source": "curated_gazetteer",
        "confidence": round(min(0.99, best_score), 2),
    }


@lru_cache(maxsize=256)
def _fetch_nominatim_results_cached(query: str) -> List[Dict[str, Any]]:
    """Fetches and caches raw Nominatim results for a query.

    Args:
        query: Nominatim query string.

    Returns:
        Raw list of result dictionaries.
    """
    headers = {"User-Agent": NOMINATIM_USER_AGENT}
    params = _build_nominatim_search_params(query)

    try:
        response = requests.get(
            NOMINATIM_URL,
            params=params,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            return []
        return payload
    except ValueError as exc:
        logger.info("Invalid Nominatim JSON for '%s': %s", query, exc)
        return []
    except requests.RequestException as exc:
        logger.info("Nominatim lookup failed for '%s': %s", query, exc)
        return []
    except Exception as exc:
        logger.info("Unexpected Nominatim error for '%s': %s", query, exc)
        return []


def _format_photon_display(properties: Dict[str, Any]) -> str:
    """Build a compact display name for a Photon feature."""
    street = str(properties.get("street") or "").strip()
    number = str(properties.get("housenumber") or "").strip()
    name = str(properties.get("name") or "").strip()
    postcode = str(properties.get("postcode") or "").strip()
    city = str(
        properties.get("city")
        or properties.get("district")
        or properties.get("county")
        or "Lisboa"
    ).strip()
    street_line = " ".join(part for part in (street, number) if part).strip()
    parts = [part for part in (name, street_line, postcode, city) if part]
    return ", ".join(dict.fromkeys(parts))


def _photon_feature_to_result(feature: Dict[str, Any], query: str) -> Optional[Dict[str, Any]]:
    """Convert a Photon feature into the internal geocoder-result shape."""
    geometry = feature.get("geometry") or {}
    coordinates = geometry.get("coordinates") or []
    if len(coordinates) < 2:
        return None
    lon = _safe_float(coordinates[0])
    lat = _safe_float(coordinates[1])
    if lat is None or lon is None:
        return None
    scope = classify_coordinate_scope(lat, lon)
    if scope == "outside_scope":
        return None

    properties = feature.get("properties") or {}
    display_name = _format_photon_display(properties) or query
    return {
        "display_name": display_name,
        "lat": lat,
        "lon": lon,
        "class": str(properties.get("osm_key") or "place"),
        "type": str(properties.get("osm_value") or properties.get("type") or "unknown"),
        "importance": 0.45,
        "address": {
            "road": properties.get("street"),
            "house_number": properties.get("housenumber"),
            "postcode": properties.get("postcode"),
            "city": properties.get("city") or properties.get("district"),
        },
        "query_used": query,
        "scope": scope,
        "match_source": "photon",
    }


@lru_cache(maxsize=256)
def _fetch_photon_results_cached(query: str) -> List[Dict[str, Any]]:
    """Fetch and cache Photon/OSM results as a fallback to Nominatim."""
    headers = {"User-Agent": NOMINATIM_USER_AGENT}
    params = {
        "q": query,
        "limit": 8,
        "lat": 38.7223,
        "lon": -9.1393,
    }
    try:
        response = requests.get(PHOTON_URL, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        features = payload.get("features") if isinstance(payload, dict) else None
        if not isinstance(features, list):
            return []
        results: List[Dict[str, Any]] = []
        for feature in features:
            if isinstance(feature, dict):
                result = _photon_feature_to_result(feature, query)
                if result:
                    results.append(result)
        return results
    except ValueError as exc:
        logger.info("Invalid Photon JSON for '%s': %s", query, exc)
        return []
    except requests.RequestException as exc:
        logger.info("Photon lookup failed for '%s': %s", query, exc)
        return []
    except Exception as exc:
        logger.info("Unexpected Photon error for '%s': %s", query, exc)
        return []


def _score_nominatim_result(
    result: Dict[str, Any],
    query_norm: str,
    prefer_city: bool,
) -> float:
    """Scores a Nominatim candidate for transport-aware location resolution.

    Args:
        result: Raw Nominatim result.
        query_norm: Normalized query.
        prefer_city: Whether Lisbon city matches should be preferred.

    Returns:
        Ranking score where higher is better.
    """
    display_norm = normalize_location_text(result.get("display_name", ""))
    address = result.get("address", {}) or {}
    lat = _safe_float(result.get("lat"))
    lon = _safe_float(result.get("lon"))
    scope = classify_coordinate_scope(lat, lon)

    tokens = [token for token in query_norm.split() if len(token) > 1]
    importance = _safe_float(result.get("importance")) or 0.0

    score = importance * 10.0

    if display_norm == query_norm:
        score += 8.0
    elif query_norm and query_norm in display_norm:
        score += 5.0

    if tokens:
        overlap = sum(token in display_norm for token in tokens)
        score += overlap * 1.2

    municipality = normalize_location_text(
        address.get("municipality")
        or address.get("city")
        or address.get("town")
        or address.get("village")
        or ""
    )

    if scope == "lisbon_city":
        score += 4.0 if prefer_city else 2.0
    elif scope == "aml":
        score += 1.5
    else:
        score -= 50.0

    if municipality == "lisboa":
        score += 1.5

    return score


def geocode_location_name(
    location_name: str,
    prefer_city: bool = True,
    allow_aml: bool = True,
) -> Optional[Dict[str, Any]]:
    """Geocodes a place name and returns the best AML-scoped match.

    Args:
        location_name: Free-form place name.
        prefer_city: Whether Lisbon city matches should be preferred.
        allow_aml: Whether AML matches outside Lisbon city are allowed.

    Returns:
        Best geocoded result dictionary or None if unresolved.
    """
    raw_query = str(location_name or "").strip()
    if not raw_query:
        return None

    coordinate_pair = _parse_coordinate_pair(raw_query)
    if coordinate_pair:
        lat, lon = coordinate_pair
        return {
            "display_name": raw_query,
            "full_display_name": raw_query,
            "lat": lat,
            "lon": lon,
            "type": "coordinate",
            "class": "place",
            "importance": 1.0,
            "address": {},
            "query_used": raw_query,
            "raw_query": raw_query,
            "cleaned_query": raw_query,
            "scope": classify_coordinate_scope(lat, lon),
            "match_source": "coordinate_pair",
            "confidence": 1.0,
        }

    query_clean = clean_location_query_fragment(raw_query)
    if not query_clean:
        return None

    query_norm = normalize_location_text(query_clean)
    effective_prefer_city = prefer_city and not _query_mentions_aml_outside_lisbon(
        query_norm
    )

    candidates: List[Dict[str, Any]] = []
    seen = set()

    for query in _build_query_variants(query_clean):
        for result in _fetch_nominatim_results_cached(query):
            lat = _safe_float(result.get("lat"))
            lon = _safe_float(result.get("lon"))
            if lat is None or lon is None:
                continue

            scope = classify_coordinate_scope(lat, lon)
            if scope == "outside_scope":
                continue
            if not allow_aml and scope != "lisbon_city":
                continue

            dedupe_key = (
                round(lat, 5),
                round(lon, 5),
                normalize_location_text(result.get("display_name", "")),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            scored = dict(result)
            scored["lat"] = lat
            scored["lon"] = lon
            scored["scope"] = scope
            scored["query_used"] = query
            scored["score"] = _score_nominatim_result(
                scored,
                query_norm=query_norm,
                prefer_city=effective_prefer_city,
            )
            candidates.append(scored)
        for result in _fetch_photon_results_cached(query):
            lat = _safe_float(result.get("lat"))
            lon = _safe_float(result.get("lon"))
            if lat is None or lon is None:
                continue

            scope = classify_coordinate_scope(lat, lon)
            if scope == "outside_scope":
                continue
            if not allow_aml and scope != "lisbon_city":
                continue

            dedupe_key = (
                round(lat, 5),
                round(lon, 5),
                normalize_location_text(result.get("display_name", "")),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            scored = dict(result)
            scored["lat"] = lat
            scored["lon"] = lon
            scored["scope"] = scope
            scored["query_used"] = query
            scored["score"] = _score_nominatim_result(
                scored,
                query_norm=query_norm,
                prefer_city=effective_prefer_city,
            )
            candidates.append(scored)

    distinctive_candidates = [
        candidate for candidate in candidates
        if _candidate_is_plausible_for_location_query(query_clean, candidate)
    ]
    if distinctive_candidates:
        candidates = distinctive_candidates
    elif _meaningful_location_query_terms(query_norm):
        candidates = []

    if not candidates:
        curated = _resolve_curated_location_point(query_clean)
        if curated:
            if not allow_aml and curated["scope"] != "lisbon_city":
                return None
            if _candidate_is_plausible_for_location_query(query_clean, curated):
                curated["raw_query"] = raw_query
                curated["cleaned_query"] = query_clean
                return curated
        return None

    candidates.sort(
        key=lambda item: (
            item.get("score", 0.0),
            _safe_float(item.get("importance")) or 0.0,
        ),
        reverse=True,
    )
    best = candidates[0]

    short_display = str(best.get("display_name") or query_clean).split(",")[0].strip()

    return {
        "display_name": short_display or query_clean,
        "full_display_name": str(best.get("display_name") or query_clean).strip(),
        "lat": best["lat"],
        "lon": best["lon"],
        "type": str(best.get("type") or "unknown"),
        "class": str(best.get("class") or "unknown"),
        "importance": _safe_float(best.get("importance")) or 0.0,
        "address": best.get("address", {}) or {},
        "query_used": str(best.get("query_used") or query_clean),
        "raw_query": raw_query,
        "cleaned_query": query_clean,
        "scope": str(best.get("scope") or "unknown"),
        "match_source": str(best.get("match_source") or "nominatim"),
        "confidence": round(min(0.99, 0.55 + (best.get("score", 0.0) / 25.0)), 2),
    }


def _resolve_named_candidate(
    fragment: str,
    candidate_map: Dict[str, str],
    minimum_score: float = 0.82,
) -> Optional[str]:
    """Resolves a free-form fragment against a candidate map with fuzzy matching.

    Args:
        fragment: User fragment.
        candidate_map: Normalized candidate map.
        minimum_score: Minimum score threshold.

    Returns:
        Canonical candidate string or None.
    """
    normalized_fragment = normalize_location_text(fragment)
    if not normalized_fragment:
        return None

    if normalized_fragment in candidate_map:
        return candidate_map[normalized_fragment]

    fragment_tokens = set(normalized_fragment.split())
    best_value = None
    best_score = 0.0

    for candidate_key, canonical in candidate_map.items():
        candidate_tokens = set(candidate_key.split())
        score = SequenceMatcher(None, normalized_fragment, candidate_key).ratio()

        if normalized_fragment in candidate_key or candidate_key in normalized_fragment:
            score += 0.15

        if fragment_tokens and candidate_tokens:
            overlap = len(fragment_tokens & candidate_tokens) / max(
                len(fragment_tokens),
                len(candidate_tokens),
            )
            score += overlap * 0.25

        if score > best_score:
            best_score = score
            best_value = canonical

    return best_value if best_score >= minimum_score else None


@lru_cache(maxsize=1)
def _get_metro_station_lookup() -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    """Builds a reusable metro alias map and canonical station index.

    Returns:
        Tuple with alias map and canonical station data map.
    """
    from tools.metrolisboa_api import (
        METRO_STATION_IDS,
        get_station_lines,
        load_metro_stations,
    )

    alias_map: Dict[str, str] = {}
    station_map: Dict[str, Dict[str, Any]] = {}
    stations_by_id: Dict[str, str] = {}

    for station in load_metro_stations():
        stop_id = str(station.get("stop_id") or "").strip()
        stop_name = str(station.get("stop_name") or "").strip()
        if not stop_name:
            continue

        if stop_id:
            stations_by_id[stop_id] = stop_name

        station_map[normalize_location_text(stop_name)] = {
            "name": stop_name,
            "lat": _safe_float(station.get("stop_lat")),
            "lon": _safe_float(station.get("stop_lon")),
            "lines": list(dict.fromkeys(get_station_lines(stop_name))),
        }

    for alias, station_id in METRO_STATION_IDS.items():
        canonical_name = stations_by_id.get(station_id) or alias.title()
        alias_map[normalize_location_text(alias)] = canonical_name
        alias_map.setdefault(normalize_location_text(canonical_name), canonical_name)

    return alias_map, station_map


def _resolve_known_metro_station(location_name: str) -> Optional[Dict[str, Any]]:
    """Resolves exact or fuzzy Metro station mentions.

    Args:
        location_name: Raw station candidate.

    Returns:
        Resolved station payload or None.
    """
    alias_map, station_map = _get_metro_station_lookup()
    normalized_query = normalize_location_text(location_name)
    canonical_name = alias_map.get(normalized_query)
    if not canonical_name and _looks_like_non_station_poi_query(normalized_query):
        return None
    if not canonical_name:
        canonical_name = _resolve_named_candidate(
            location_name,
            alias_map,
            minimum_score=0.84,
        )
    if not canonical_name:
        return None

    station = station_map.get(normalize_location_text(canonical_name), {})
    lines = list(station.get("lines", []))

    return {
        "query": str(location_name or "").strip(),
        "normalized_query": normalize_location_text(location_name),
        "display_name": canonical_name,
        "full_display_name": canonical_name,
        "lat": station.get("lat"),
        "lon": station.get("lon"),
        "scope": "lisbon_city",
        "match_source": "metro_station",
        "confidence": 0.98,
        "lines": lines,
        "nearest_metro": {
            "name": canonical_name,
            "distance_km": 0.0,
            "lines": lines,
        },
        "nearest_cp": None,
        "warnings": [],
        "success": True,
    }


@lru_cache(maxsize=1)
def _get_cp_station_lookup() -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    """Builds a reusable CP alias map and canonical station index.

    Returns:
        Tuple with alias map and canonical station data map.
    """
    from tools.cp_api import CP_KEY_STATIONS, load_cp_aml_stations

    alias_map: Dict[str, str] = {}
    station_map: Dict[str, Dict[str, Any]] = {}

    for station in load_cp_aml_stations().values():
        canonical_name = str(station.get("name") or "").strip()
        if not canonical_name:
            continue

        normalized_name = normalize_location_text(canonical_name)
        alias_map[normalized_name] = canonical_name
        station_map[normalized_name] = {
            "name": canonical_name,
            "lat": _safe_float(station.get("lat")),
            "lon": _safe_float(station.get("lon")),
            "railways": list(station.get("railways", [])),
        }

    for key, info in CP_KEY_STATIONS.items():
        canonical_name = str(info.get("name") or key.replace("_", " ").title()).strip()
        alias_map[normalize_location_text(key.replace("_", " "))] = canonical_name
        alias_map[normalize_location_text(canonical_name)] = canonical_name

        station_map.setdefault(
            normalize_location_text(canonical_name),
            {
                "name": canonical_name,
                "lat": None,
                "lon": None,
                "railways": list(info.get("lines", [])),
            },
        )

    return alias_map, station_map


def _resolve_known_cp_station(location_name: str) -> Optional[Dict[str, Any]]:
    """Resolves exact or fuzzy CP station mentions.

    Args:
        location_name: Raw station candidate.

    Returns:
        Resolved station payload or None.
    """
    alias_map, station_map = _get_cp_station_lookup()
    normalized_query = normalize_location_text(location_name)
    canonical_name = alias_map.get(normalized_query)
    if not canonical_name and _looks_like_non_station_poi_query(normalized_query):
        return None
    if not canonical_name:
        canonical_name = _resolve_named_candidate(
            location_name,
            alias_map,
            minimum_score=0.82,
        )
    if not canonical_name:
        return None

    station = station_map.get(normalize_location_text(canonical_name), {})
    railways = list(station.get("railways", []))
    station_lat = station.get("lat")
    station_lon = station.get("lon")
    if station_lat is None or station_lon is None:
        return None
    scope = classify_coordinate_scope(station_lat, station_lon)

    return {
        "query": str(location_name or "").strip(),
        "normalized_query": normalize_location_text(location_name),
        "display_name": canonical_name,
        "full_display_name": canonical_name,
        "lat": station_lat,
        "lon": station_lon,
        "scope": scope,
        "match_source": "cp_station",
        "confidence": 0.96,
        "lines": railways,
        "nearest_metro": None,
        "nearest_cp": {
            "name": canonical_name,
            "distance_km": 0.0,
            "railways": railways,
        },
        "warnings": [],
        "success": True,
    }


def _find_nearest_metro_context(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Returns the nearest Metro station context to a coordinate pair.

    Args:
        lat: Latitude.
        lon: Longitude.

    Returns:
        Nearest station payload or None.
    """
    from tools.metrolisboa_api import find_nearest_metro_station, get_station_lines

    nearest = find_nearest_metro_station(
        lat,
        lon,
        max_results=1,
        max_dist_km=8.0,
    )
    if not nearest:
        return None

    station = nearest[0]
    station_name = str(station.get("stop_name") or "").strip()
    if not station_name:
        return None

    return {
        "name": station_name,
        "distance_km": float(station.get("distance_km", 0.0)),
        "lines": list(dict.fromkeys(get_station_lines(station_name))),
    }


def _find_nearest_cp_context(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Returns the nearest CP AML station context to a coordinate pair.

    Args:
        lat: Latitude.
        lon: Longitude.

    Returns:
        Nearest CP station payload or None.
    """
    from tools.cp_api import load_cp_aml_stations

    best_station = None
    best_distance = None

    for station in load_cp_aml_stations().values():
        station_lat = _safe_float(station.get("lat"))
        station_lon = _safe_float(station.get("lon"))
        if station_lat is None or station_lon is None:
            continue

        distance_km = haversine_distance(lat, lon, station_lat, station_lon)
        if best_distance is None or distance_km < best_distance:
            best_distance = distance_km
            best_station = station

    if not best_station or best_distance is None:
        return None

    return {
        "name": str(best_station.get("name") or "").strip(),
        "distance_km": round(best_distance, 2),
        "railways": list(best_station.get("railways", [])),
    }


def resolve_location_query(
    location_name: str,
    prefer_city: bool = True,
    allow_aml: bool = True,
) -> Dict[str, Any]:
    """Resolves a free-form location into a transport-aware payload.

    Args:
        location_name: Free-form location query.
        prefer_city: Whether Lisbon city matches should be preferred.
        allow_aml: Whether AML matches outside Lisbon city are allowed.

    Returns:
        Resolution payload with scope, match source, confidence, and nearest nodes.
    """
    raw_query = str(location_name or "").strip()
    query = clean_location_query_fragment(raw_query)
    normalized_query = normalize_location_text(query)

    if not raw_query or not query:
        return {
            "query": query,
            "raw_query": raw_query,
            "normalized_query": "",
            "success": False,
            "display_name": "",
            "scope": "unknown",
            "match_source": "none",
            "confidence": 0.0,
            "nearest_metro": None,
            "nearest_cp": None,
            "warnings": [],
        }

    metro_match = _resolve_known_metro_station(query)
    if metro_match:
        return metro_match

    cp_match = _resolve_known_cp_station(query)
    if cp_match:
        return cp_match

    geocoded = geocode_location_name(
        query,
        prefer_city=prefer_city,
        allow_aml=allow_aml,
    )
    if not geocoded:
        return {
            "query": query,
            "raw_query": raw_query,
            "normalized_query": normalized_query,
            "success": False,
            "display_name": query,
            "scope": "unknown",
            "match_source": "none",
            "confidence": 0.0,
            "nearest_metro": None,
            "nearest_cp": None,
            "warnings": ["Could not resolve location confidently."],
        }

    lat = geocoded["lat"]
    lon = geocoded["lon"]
    scope = geocoded["scope"]

    warnings: List[str] = []
    if scope == "aml":
        warnings.append("Location resolved in the AML, outside Lisbon city.")
    elif scope == "outside_scope":
        warnings.append("Location appears outside the supported Lisbon/AML scope.")

    return {
        "query": query,
        "raw_query": raw_query,
        "normalized_query": normalized_query,
        "success": True,
        "display_name": geocoded["display_name"],
        "full_display_name": geocoded["full_display_name"],
        "lat": lat,
        "lon": lon,
        "scope": scope,
        "match_source": geocoded["match_source"],
        "confidence": geocoded["confidence"],
        "type": geocoded.get("type", "unknown"),
        "class": geocoded.get("class", "unknown"),
        "importance": geocoded.get("importance", 0.0),
        "address": geocoded.get("address", {}),
        "query_used": geocoded.get("query_used", query),
        "nearest_metro": _find_nearest_metro_context(lat, lon),
        "nearest_cp": _find_nearest_cp_context(lat, lon),
        "warnings": warnings,
    }


def _estimate_walk_minutes(distance_km: float) -> int:
    """Estimates walking minutes using a practical urban pace.

    Args:
        distance_km: Walking distance in kilometers.

    Returns:
        Estimated walking minutes.
    """
    return max(3, int(round(distance_km * 12.0)))


def _build_walking_hints(osm_class: str, osm_type: str) -> Tuple[str, str]:
    """Returns PT/EN walking hints based on OSM class/type.

    Args:
        osm_class: OSM class.
        osm_type: OSM type.

    Returns:
        Tuple with PT and EN walking hints.
    """
    label = f"{normalize_location_text(osm_class)}:{normalize_location_text(osm_type)}"

    if any(token in label for token in ["university", "college", "school"]):
        return "ao campus", "to the campus"
    if any(token in label for token in ["hospital", "clinic"]):
        return "ao hospital", "to the hospital"
    if "library" in label:
        return "à biblioteca", "to the library"
    if any(token in label for token in ["garden", "park"]):
        return "ao jardim ou parque", "to the garden or park"
    if any(token in label for token in ["suburb", "neighbourhood", "neighborhood", "quarter", "residential", "administrative"]):
        return "ao bairro", "to the neighbourhood"
    if any(token in label for token in ["stadium", "sports", "pitch"]):
        return "ao recinto", "to the venue"
    if any(token in label for token in ["museum", "monument", "attraction", "viewpoint"]):
        return "ao local", "to the site"
    if any(token in label for token in ["mall", "retail", "commercial"]):
        return "ao centro comercial", "to the shopping centre"

    return "ao local", "to the place"


def build_dynamic_landmark_info(
    location_name: str,
    prefer_city: bool = True,
    allow_aml: bool = True,
) -> Optional[Dict[str, Any]]:
    """Builds a landmark-like transport payload for non-curated places.

    Args:
        location_name: Raw place name.
        prefer_city: Whether Lisbon city matches should be preferred.
        allow_aml: Whether AML matches outside Lisbon city are allowed.

    Returns:
        Landmark-like dictionary or None if no plausible transport anchor exists.
    """
    resolved = resolve_location_query(
        location_name,
        prefer_city=prefer_city,
        allow_aml=allow_aml,
    )
    if not resolved.get("success"):
        return None

    if resolved.get("match_source") in {"metro_station", "cp_station"}:
        return None

    nearest_metro = resolved.get("nearest_metro")
    nearest_cp = resolved.get("nearest_cp")

    if nearest_metro and nearest_metro.get("distance_km", 999.0) > MAX_DYNAMIC_METRO_WALK_KM:
        nearest_metro = None
    if nearest_cp and nearest_cp.get("distance_km", 999.0) > MAX_DYNAMIC_CP_WALK_KM:
        nearest_cp = None

    if not nearest_metro and not nearest_cp:
        return None

    raw_query = str(location_name or "").strip()
    display_name = str(resolved.get("display_name") or raw_query).strip()
    if _query_terms_missing_from_display(raw_query, display_name):
        display_name = raw_query
    short_name = (
        raw_query
        if _looks_like_acronym_label(raw_query)
        or _query_terms_missing_from_display(raw_query, display_name)
        else display_name
    )

    info: Dict[str, Any] = {
        "name": display_name,
        "short_name": short_name,
        "display_name": display_name,
        "description": "Resolved dynamically via OpenStreetMap/Nominatim",
        "dynamic": True,
        "match_source": resolved.get("match_source", "nominatim"),
        "scope": resolved.get("scope", "unknown"),
        "warnings": resolved.get("warnings", []),
    }

    if nearest_metro:
        station_name = str(nearest_metro["name"]).strip()
        station_name_lower = station_name.lower()
        lines = list(nearest_metro.get("lines", []))
        walking_hint_pt, walking_hint_en = _build_walking_hints(
            str(resolved.get("class") or ""),
            str(resolved.get("type") or ""),
        )

        info["metro"] = station_name_lower
        info["line"] = "/".join(lines) if lines else ""
        info["walking_hint_pt"] = walking_hint_pt
        info["walking_hint_en"] = walking_hint_en
        info["metro_walk_minutes"] = _estimate_walk_minutes(
            float(nearest_metro["distance_km"])
        )

    if nearest_cp:
        info["train_station"] = str(nearest_cp["name"]).strip()
        info["train_walk_minutes"] = _estimate_walk_minutes(
            float(nearest_cp["distance_km"])
        )
        if not nearest_metro:
            info["alternative"] = f"CP Train via {info['train_station']}"

    return info


def get_location_display_name(location_name: str, detailed: bool = False) -> str:
    """Returns the best available display label for a free-form location.

    Args:
        location_name: Raw location name.
        detailed: Unused for now, kept for API symmetry.

    Returns:
        Best available user-facing display label.
    """
    _ = detailed
    raw = str(location_name or "").strip()
    if not raw:
        return raw

    if _looks_like_acronym_label(raw):
        return raw

    resolved = resolve_location_query(raw, prefer_city=True, allow_aml=True)
    if resolved.get("success"):
        curated_display_name = _CURATED_DISPLAY_NAMES.get(normalize_location_text(raw))
        if curated_display_name:
            return curated_display_name
        cleaned_query = clean_location_query_fragment(raw)
        if cleaned_query and normalize_location_text(cleaned_query) != normalize_location_text(raw):
            return str(resolved.get("display_name") or cleaned_query).strip()
        if resolved.get("match_source") in {"metro_station", "cp_station"}:
            resolved_display = str(resolved.get("display_name") or raw).strip()
            if normalize_location_text(raw) == normalize_location_text(resolved_display):
                return resolved_display
            return raw.title()
        resolved_display = str(resolved.get("display_name") or raw).strip()
        if _query_terms_missing_from_display(raw, resolved_display):
            return raw
        return resolved_display

    curated_display_name = _CURATED_DISPLAY_NAMES.get(normalize_location_text(raw))
    if curated_display_name:
        return curated_display_name

    return raw.title()


if __name__ == "__main__":
    # ========================
    # Test Block
    # ========================
    import sys

    PASS = "\033[1;32m✅\033[0m"
    FAIL = "\033[1;31m❌\033[0m"
    errors = 0

    # --- Test: normalize_location_text with Portuguese diacritics ---
    normalized = normalize_location_text("Marquês de Pombal")
    if normalized == "marques de pombal":
        print(f"{PASS} normalizes Portuguese diacritics")
    else:
        print(f"{FAIL} normalizes Portuguese diacritics — got: {normalized!r}")
        errors += 1

    # --- Edge case: empty input ---
    empty = normalize_location_text("")
    if empty == "":
        print(f"{PASS} handles empty input")
    else:
        print(f"{FAIL} handles empty input — got: {empty!r}")
        errors += 1

    # --- Edge case: ambiguous destination ---
    ambiguity = build_location_ambiguity_preamble("Rossio", "Madeira", language="pt")
    if "Ilha da Madeira" in ambiguity and "Rua Humberto Madeira" in ambiguity:
        print(f"{PASS} surfaces Madeira ambiguity")
    else:
        print(f"{FAIL} surfaces Madeira ambiguity — got: {ambiguity!r}")
        errors += 1

    # --- Edge case: explicit address should not trigger ambiguity ---
    explicit = build_location_ambiguity_preamble(
        "Rossio",
        "Avenida da Ilha da Madeira",
        language="pt",
    )
    if explicit == "":
        print(f"{PASS} avoids false ambiguity for explicit Madeira address")
    else:
        print(f"{FAIL} avoids false ambiguity for explicit address — got: {explicit!r}")
        errors += 1

    # --- Test: acronym display labels are preserved ---
    acronym = get_location_display_name("NOVA IMS")
    if acronym == "NOVA IMS":
        print(f"{PASS} preserves acronym labels")
    else:
        print(f"{FAIL} preserves acronym labels — got: {acronym!r}")
        errors += 1

    sys.exit(errors)
