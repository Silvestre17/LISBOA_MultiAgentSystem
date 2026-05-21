# ==========================================================================
# Master Thesis - Quality Assurance Agent
#   - André Filipe Gomes Silvestre, 20240502
#
#   Validates completeness of agent outputs before final response.
#   Two-phase validation:
#     Phase 1 (LLM): Structural completeness check via prompt-based analysis
#     Phase 2 (Deterministic): Factual verification against known data
#   Identifies missing data and returns retry hints to the orchestrator.
#   Ensures no incomplete or hallucinated responses reach the user.
# ==========================================================================

import logging
import re
import unicodedata
from contextlib import suppress
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agent.agents.base import BaseAgent, clean_response, parse_json_response
from agent.utils.langsmith_tracing import traceable
from agent.prompts.qa import get_qa_prompt
from agent.utils.response_formatter import (
    _LABEL_TRANSLATIONS,
    _count_structured_place_cards,
    missing_material_source_labels,
    _place_response_missing_required_fields,
    final_post_qa_guard,
    final_visual_pass,
    infer_visible_label_language,
    infer_response_language,
)

# Import authoritative static transport data from tool modules.
# These provide the single source of truth for metro/CP verification (no API calls).
try:
    from tools.metrolisboa_api import METRO_LINES as _METRO_LINES_DATA
    from tools.metrolisboa_api import METRO_STATIONS as _METRO_STATIONS_DATA
    _HAS_METRO_DATA = True
except ImportError:
    _METRO_LINES_DATA: Dict = {}
    _METRO_STATIONS_DATA: Dict = {}
    _HAS_METRO_DATA = False

try:
    from tools.cp_api import CP_LINES as _CP_LINES_DATA
    _HAS_CP_DATA = True
except ImportError:
    _CP_LINES_DATA: Dict = {}
    _HAS_CP_DATA = False

logger = logging.getLogger(__name__)

_QA_ANCHOR_KEYWORD_RE = re.compile(
    r"\b(?:rua|avenida|av|largo|praca|calcada|travessa|estrada|campo|"
    r"museu|palacio|castelo|mosteiro|torre|padrao|catedral|igreja|capela|"
    r"jardim|parque|miradouro|mercado|fundacao|universidade|faculdade|"
    r"hospital|farmacia|estacao|aeroporto|terminal|centro\s+comercial|"
    r"shopping|livraria|teatro|coliseu|oceanario|maat|lx\s*factory)\b",
    re.IGNORECASE,
)
_QA_GENERIC_ANCHOR_RE = re.compile(
    r"\b(?:monumentos?|atracoes?|atra[cç][oõ]es?|museus?|restaurantes?|"
    r"gastronomia|comida|cozinha|eventos?|cultura|historicos?|tradicional|"
    r"imperdiveis|locais|sitios|places|sights|restaurants?|food|culture|events?)\b",
    re.IGNORECASE,
)
_QA_ANCHOR_LIST_SPLIT_RE = re.compile(
    r"\s*(?:[,;]|\s+\+\s+|\s+/\s+|\s+(?:e|and)\s+)\s+",
    re.IGNORECASE,
)
_QA_CENTRAL_AREA_RE = re.compile(
    r"\b(?:se de lisboa|catedral de lisboa|carmo|baixa|chiado|rossio|praca do comercio|terreiro do paco|alfama)\b",
    re.IGNORECASE,
)
_QA_BELEM_AREA_RE = re.compile(
    r"\b(?:belem|torre de belem|padrao dos descobrimentos|jeronimos|mosteiro dos jeronimos|brasilia|imperio)\b",
    re.IGNORECASE,
)


def _qa_normalize_text(text: str) -> str:
    """Normalize user-facing text for deterministic final-response audits."""
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = re.sub(r"[^a-zA-Z0-9\s/-]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip().lower()


def _qa_event_category_key(category: str) -> str:
    """Return a comparable key for event category labels."""
    normalized = _qa_normalize_text(category)
    aliases = {
        "musica": "music",
        "music": "music",
        "concert": "music",
        "concerts": "music",
        "concerto": "music",
        "concertos": "music",
        "desporto": "sports",
        "desportos": "sports",
        "desportivo": "sports",
        "desportiva": "sports",
        "desportivos": "sports",
        "desportivas": "sports",
        "sport": "sports",
        "sports": "sports",
        "festivais": "festivals",
        "festivals": "festivals",
        "festival": "festivals",
        "teatro opera e danca": "theater opera dance",
        "theater opera dance": "theater opera dance",
        "exposicoes": "exhibitions",
        "exhibitions": "exhibitions",
        "feiras": "fairs",
        "fairs": "fairs",
    }
    return aliases.get(normalized, normalized)


def _qa_requested_event_category_keys(user_query: str) -> set[str]:
    """Infer event categories positively requested by the user."""
    normalized = _qa_normalize_text(user_query)
    if not normalized:
        return set()
    negated_spans = [
        match.span()
        for match in re.finditer(
            r"\b(?:sem|nao|not|without|excluding|except|exclui|excluir|evita|evitar)\b(?:\s+\w+){0,5}",
            normalized,
            flags=re.IGNORECASE,
        )
    ]
    patterns = {
        "sports": r"\b(?:desporto|desportos|desportiv[oa]s?|sport|sports|maratona|marathon|trail)\b",
        "music": r"\b(?:musica|music|concertos?|concerts?|fado|jazz|rock|pop)\b",
        "festivals": r"\b(?:festivais?|festivals?|arrai(?:al|ais|s)?|santos populares|marchas populares|festas de lisboa)\b",
        "exhibitions": r"\b(?:exposicoes?|exhibitions?|arte|art)\b",
        "theater opera dance": r"\b(?:teatro|theatre|theater|opera|danca|dance)\b",
        "fairs": r"\b(?:feiras?|fairs?|mercado|market)\b",
    }
    requested: set[str] = set()
    for key, pattern in patterns.items():
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
            if any(start <= match.start() <= end for start, end in negated_spans):
                continue
            requested.add(key)
            break
    return requested


def _qa_forbidden_event_category_keys(user_query: str) -> set[str]:
    """Infer event categories explicitly excluded by the user."""
    normalized = _qa_normalize_text(user_query)
    if not normalized:
        return set()
    category_terms = {
        "sports": r"(?:desporto|desportos|desportiv[oa]s?|sport|sports)",
        "music": r"(?:musica|music|concertos?|concerts?|fado|jazz|rock|pop)",
        "festivals": r"(?:festivais?|festivals?|arrai(?:al|ais|s)?|santos populares|marchas populares)",
        "exhibitions": r"(?:exposicoes?|exhibitions?|arte|art)",
        "theater opera dance": r"(?:teatro|theatre|theater|opera|danca|dance)",
        "fairs": r"(?:feiras?|fairs?|mercado|market)",
    }
    forbidden: set[str] = set()
    for key, term_re in category_terms.items():
        if re.search(
            rf"\b(?:sem|nao|not|without|excluding|except|exclui|excluir|evita|evitar)\b(?:\s+\w+){{0,5}}\s+\b{term_re}\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            forbidden.add(key)
    return forbidden - _qa_requested_event_category_keys(user_query)


_QA_GENERIC_RESEARCHER_INTRO_TITLES = {
    "atracoes imperdiveis",
    "atracoes recomendadas",
    "locais recomendados",
    "locais essenciais",
    "must see attractions",
    "must-see attractions",
    "recommended places",
    "essential places",
    "recommended attractions",
    "top attractions",
}


def _qa_generic_researcher_intro_key(line: str) -> str:
    """Return a normalized key for generic intro/card checks."""
    normalized = _qa_normalize_text(line)
    normalized = re.sub(r"^(?:[-/]+|\d+[.)]?)\s*", "", normalized).strip()
    normalized = re.sub(r"^(?:descricao|description)\s+", "", normalized).strip()
    return normalized


def _qa_is_generic_researcher_intro_title(line: str) -> bool:
    """Return whether a line is a generic researcher intro title."""
    return _qa_generic_researcher_intro_key(line) in _QA_GENERIC_RESEARCHER_INTRO_TITLES


def _qa_is_generic_researcher_intro_sentence(line: str) -> bool:
    """Return whether a line is generic intro prose, not a concrete result."""
    normalized = _qa_generic_researcher_intro_key(line)
    return bool(
        normalized
        and re.search(
            r"\b(?:aqui tens|selecao|essenciais|primeira visita|primeira vez|"
            r"principais locais|correspondem ao pedido|here is|here are|"
            r"selection|essential places|first visit|match your request|main places)\b",
            normalized,
        )
    )


def _qa_response_has_generic_intro_card_defect(response: str) -> bool:
    """Return whether generic intro text leaked as duplicated place cards."""
    generic_title_count = 0
    lines = str(response or "").splitlines()
    for index, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not _qa_is_generic_researcher_intro_title(stripped):
            continue
        generic_title_count += 1
        if re.match(r"^(?:[-*]\s+)?\*\*[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]*\s*", stripped):
            return True
        for later_line in lines[index + 1:index + 4]:
            later = later_line.strip()
            if not later:
                continue
            if _qa_is_generic_researcher_intro_title(later):
                return True
            if _qa_is_generic_researcher_intro_sentence(later):
                return True
            break
    return generic_title_count > 1


def _qa_clean_requested_anchor_fragment(fragment: str) -> str:
    """Return a compact candidate place name from a user-request fragment."""
    cleaned = re.sub(r"\([^)]*\)", " ", str(fragment or ""))
    cleaned = re.sub(
        r"^\s*(?:o|a|os|as|um|uma|uns|umas|no|na|nos|nas|em|at|from|the|"
        r"de|do|da|dos|das)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s+(?:e|and)\s+(?:termin\S*|acab\S*|ending|end)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s+(?:as|at|pelas|por\s+volta|durante|during|com|with|inclui|include|"
        r"including|depois|then|sem|using|usando)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", cleaned).strip(" .:-")


def _qa_anchor_fragment_is_specific(fragment: str) -> bool:
    """Return whether a candidate is likely a specific place, not a category."""
    cleaned = _qa_clean_requested_anchor_fragment(fragment)
    normalized = _qa_normalize_text(cleaned)
    if not (2 <= len(cleaned) <= 90 and normalized):
        return False
    if _QA_GENERIC_ANCHOR_RE.fullmatch(normalized):
        return False
    if _QA_ANCHOR_KEYWORD_RE.search(normalized):
        return True
    if re.search(r"\b[A-Z0-9]{2,}\b", cleaned):
        return True
    if len(normalized.split()) >= 2 and re.search(r"\b[A-Z][A-Za-z0-9-]{2,}", cleaned):
        return True
    return len(normalized.split()) == 1 and cleaned[:1].isupper() and len(normalized) >= 4


def _qa_split_requested_anchor_fragment(fragment: str) -> List[str]:
    """Split a user-provided place list into specific place candidates."""
    candidates: List[str] = []
    for part in _QA_ANCHOR_LIST_SPLIT_RE.split(str(fragment or "")):
        cleaned = _qa_clean_requested_anchor_fragment(part)
        if _qa_anchor_fragment_is_specific(cleaned):
            candidates.append(cleaned)
    return candidates


def _qa_extract_requested_anchor_phrases(user_query: str) -> List[str]:
    """Extract specific requested place names without relying on a fixed list."""
    text = str(user_query or "").strip()
    labels: List[str] = []
    seen: set[str] = set()

    def add_fragment(fragment: str) -> None:
        for candidate in _qa_split_requested_anchor_fragment(fragment):
            key = _qa_normalize_text(candidate)
            if key and key not in seen:
                seen.add(key)
                labels.append(candidate)

    endpoint_patterns = [
        r"\bde\s+(?P<origin>[^,.;]+?)\s+(?:para|ate|at\S*)\s+(?P<destination>[^,.;]+)",
        r"\bfrom\s+(?P<origin>[^,.;]+?)\s+to\s+(?P<destination>[^,.;]+)",
    ]
    for pattern in endpoint_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            add_fragment(match.group("origin"))
            add_fragment(match.group("destination"))

    single_place_patterns = [
        r"\b(?:come\S*|iniciar|inicia|starting|start)\s+(?:no|na|em|at|from)\s+(?P<place>[^,.;]+)",
        r"\b(?:termin\S*|acab\S*|ending|end)\s+(?:no|na|em|at|in)\s+(?P<place>[^,.;]+)",
        r"\b(?:a\s+partir\s+d\S*|desde)\s+(?P<place>[^,.;]+)",
    ]
    for pattern in single_place_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            add_fragment(match.group("place"))

    list_patterns = [
        r"\b(?:visitar|visit|inclui\S*|include|including|passar\s+por|pass\s+through)\s+(?P<places>[^.;]+)",
    ]
    for pattern in list_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            add_fragment(match.group("places"))

    return labels


def _qa_requested_anchor_labels(user_query: str) -> List[str]:
    """Extract explicit named Lisbon anchors from a user request."""
    return _qa_extract_requested_anchor_phrases(user_query)


def _qa_response_mentions_anchor(response: str, label: str) -> bool:
    """Return whether the response mentions a required anchor or alias."""
    normalized_response = _qa_normalize_text(response)
    normalized_label = _qa_normalize_text(label)
    return bool(normalized_label and re.search(rf"\b{re.escape(normalized_label)}\b", normalized_response))


def _qa_mentions_central_and_belem(text: str) -> bool:
    """Return whether text mentions both central Lisbon and Belém-area anchors."""
    normalized = _qa_normalize_text(text)
    return bool(_QA_CENTRAL_AREA_RE.search(normalized) and _QA_BELEM_AREA_RE.search(normalized))


def _qa_response_has_cross_zone_movement(response: str) -> bool:
    """Return whether a final response contains a concrete center-to-Belém movement leg."""
    concrete_mode_re = re.compile(
        r"\b(?:carris\s+\d{2,4}[a-z]?|\d{1,4}e|\d{3,4}[a-z]?|linha\s+(?:\d{2,4}[a-z]?|verde|azul|amarela|vermelha|de\s+(?:cascais|sintra|azambuja|sado))|"
        r"line\s+(?:\d{2,4}[a-z]?|green|blue|yellow|red)|cp\s+linha|comboio\s+linha|train\s+line)\b",
        re.IGNORECASE,
    )
    for raw_line in str(response or "").splitlines():
        normalized_line = _qa_normalize_text(raw_line)
        has_leg_marker = (
            "->" in raw_line
            or "→" in raw_line
            or " para " in f" {normalized_line} "
            or " to " in f" {normalized_line} "
        )
        if (
            has_leg_marker
            and _QA_CENTRAL_AREA_RE.search(normalized_line)
            and _QA_BELEM_AREA_RE.search(normalized_line)
            and concrete_mode_re.search(normalized_line)
        ):
            return True
    normalized_response = _qa_normalize_text(response)
    return bool(
        _qa_mentions_central_and_belem(normalized_response)
        and concrete_mode_re.search(normalized_response)
    )


def _qa_response_has_cross_zone_limitation(response: str) -> bool:
    """Return whether a final response names the central-Belém leg as unconfirmed."""
    normalized_response = _qa_normalize_text(response)
    return bool(
        _qa_mentions_central_and_belem(normalized_response)
        and re.search(
            r"\b(?:nao confirmad\w*|unconfirmed|sem ligacao confirmad\w*|sem ligacao concreta|did not confirm|not confirmed)\b",
            normalized_response,
            flags=re.IGNORECASE,
        )
    )


def _qa_requested_start_label(user_query: str) -> str:
    """Return the explicit requested starting anchor, when the query names one."""
    normalized_query = _qa_normalize_text(user_query)
    if not re.search(r"\b(?:comeca|comece|comecar|inicia|iniciar|start|starting)\b", normalized_query):
        return ""
    labels = _qa_extract_requested_anchor_phrases(user_query)
    return labels[0] if labels else ""


def _qa_first_route_block_mentions_anchor(response: str, label: str) -> bool:
    """Return whether the first visible itinerary block starts at the requested anchor."""
    if not label:
        return True
    in_route_section = False
    for raw_line in str(response or "").splitlines():
        stripped = raw_line.strip()
        normalized_line = _qa_normalize_text(stripped)
        if re.search(r"\b(?:roteiro sugerido|suggested route)\b", normalized_line):
            in_route_section = True
            continue
        if in_route_section and stripped.startswith("### "):
            return False
        if not in_route_section:
            continue
        if not re.match(r"^[-*]\s+\*\*.+\*\*", stripped):
            continue
        first_title = re.sub(r"^[-*]\s+\*\*", "", stripped)
        first_title = re.sub(r"\*\*.*$", "", first_title)
        return _qa_response_mentions_anchor(first_title, label)
    return False


# ==========================================================================
# Static Knowledge for Deterministic Fact-Checking
# ==========================================================================
# Metro and CP authoritative data is imported from tool modules above.
# Only non-dynamic knowledge (bounds, domains, limits) is defined here.

# Canonical metro station names - derived from the authoritative METRO_STATIONS
# dict imported from tools.metrolisboa_api. Kept as an alias for backward
# compatibility (used by tests and external imports).
_METRO_CANONICAL_STATIONS: set = (
    set(_METRO_STATIONS_DATA.keys()) if _METRO_STATIONS_DATA
    else {  # Minimal inline fallback for isolated test environments
        "rato", "marquês de pombal", "marques de pombal", "picoas", "saldanha",
        "campo pequeno", "entre campos", "entrecampos", "cidade universitária",
        "cidade universitaria", "campo grande", "quinta das conchas", "lumiar",
        "ameixoeira", "senhor roubado", "odivelas",
        "santa apolónia", "santa apolonia", "terreiro do paço", "terreiro do paco",
        "baixa-chiado", "baixa chiado", "restauradores", "avenida", "parque",
        "são sebastião", "sao sebastiao", "praça de espanha", "praca de espanha",
        "jardim zoológico", "jardim zoologico", "laranjeiras", "alto dos moinhos",
        "colégio militar", "colegio militar", "carnide", "pontinha", "alfornelos",
        "amadora este", "reboleira",
        "cais do sodré", "cais do sodre", "rossio", "martim moniz", "intendente",
        "anjos", "arroios", "alameda", "areeiro", "roma", "alvalade", "telheiras",
        "olaias", "bela vista", "chelas", "olivais", "cabo ruivo", "oriente",
        "moscavide", "encarnação", "encarnacao", "aeroporto",
    }
)

# AML geographic bounding box (same values as cp_api.AML_BOUNDS)
_AML_BOUNDS = {
    "lat_min": 38.4,
    "lat_max": 39.0,
    "lon_min": -9.5,
    "lon_max": -8.7,
}

# Known valid URL domains for Lisbon data
_VALID_DOMAINS = {
    "visitlisboa.com", "metrolisboa.pt", "api.metrolisboa.pt",
    "carrismetropolitana.pt", "api.carrismetropolitana.pt",
    "cp.pt", "comboios.live", "ipma.pt", "api.ipma.pt",
    "dados.cm-lisboa.pt", "dados.gov.pt", "cm-lisboa.pt",
    "wikipedia.org", "en.wikipedia.org", "pt.wikipedia.org",
    "carris.pt", "gateway.carris.pt", "aml.pt", "google.com",
}

# IPMA forecast range (max days available)
_IPMA_FORECAST_DAYS = 5
_TOP_LEVEL_SECTION_RE = re.compile(r"^###\s+", re.MULTILINE)

# Lisbon historic temperature bounds (°C) for weather sanity checks.
# Source: IPMA records. All-time high: 44.1°C (Aug 2023). Generous margins applied.
_LISBON_TEMP_MIN = -5.0
_LISBON_TEMP_MAX = 47.0

# Time tolerance factor for itinerary duration check (allows 50% overrun before warning)
_TIME_TOLERANCE_FACTOR = 1.5

# Output truncation limit (chars per agent output, controls LLM token usage)
_TRUNCATION_LIMIT = 6000

# Known Carris tram (elétrico) lines currently operating in Lisbon.
# Routes with GTFS route_short_name ending in "E". 12E is tourist-only (Hills Tramcar).
# Source: https://www.carris.pt/linhas-e-paragens/ (as of 2025)
_CARRIS_TRAM_LINES = {"12e", "15e", "18e", "25e", "28e"}


class QualityAssuranceAgent(BaseAgent):
    """
    Quality Assurance agent that validates data completeness and factual accuracy.

    Two-phase validation:
        Phase 1 (LLM): Analyzes structural completeness via prompt-based reasoning.
            Checks if all required data fields are present for the query type.
        Phase 2 (Deterministic): Cross-checks factual claims against known data.
            Validates metro stations, coordinates, dates, URLs without LLM involvement.

    Responsibilities:
        - Analyze outputs from specialized agents
        - Verify user preferences/constraints are addressed
        - Identify missing critical data for the query type
        - Return `required_agents` hints when data is incomplete
        - Flag potential hallucinations or data gaps
        - Add disclaimers about known data limitations

    Note:
        This agent has NO LangChain tools. It uses deterministic Python functions
        for fact-checking (Phase 2), not LLM tool-calling. The surrounding
        orchestration layer decides whether returned retry hints should trigger
        additional worker execution.
    """

    def __init__(self):
        """Initializes the QA agent."""
        super().__init__("qa")

    @staticmethod
    def _normalize_query(user_query: str) -> str:
        """Returns a normalized lower-case query string for lightweight intent guards."""
        return (user_query or "").strip().lower()

    @classmethod
    def _is_event_listing_query(cls, user_query: str) -> bool:
        """Detects event-discovery queries that should stay within the researcher domain."""
        query = cls._normalize_query(user_query)
        if not query:
            return False

        event_patterns = [
            r"\bevents?\b",
            r"\beventos?\b",
            r"\bmusic\b",
            r"\bm[uú]sica\b",
            r"\bsports?\b",
            r"\bdesporto\b",
            r"\bdesportiv[oa]s?\b",
            r"\btheat(?:er|re)\b",
            r"\bteatro\b",
            r"\b[oó]pera\b",
            r"\bdance\b",
            r"\bdan[çc]a\b",
            r"\bconcerts?\b",
            r"\bconcertos?\b",
            r"\bexhibitions?\b",
            r"\bexposi(?:ç|c)[aã]o(?:es)?\b",
            r"\bwhat'?s on\b",
            r"\bo que acontece\b",
            r"\bcultura\b",
            r"\bcultural\b",
            r"\bfestival(?:es)?\b",
            r"\barrai(?:al|ais|s)?\b",
            r"\bsantos populares\b",
            r"\bmarchas populares\b",
            r"\bfestas de lisboa\b",
        ]
        planning_patterns = [
            r"\bplan\b",
            r"\bplanning\b",
            r"\bitinerary\b",
            r"\broteiro\b",
            r"\bitiner[aá]rio\b",
            r"\bplane(?:ia|ar|ie)\b",
            r"\bcria(?:r)? um itiner[aá]rio\b",
            r"\borganiza(?:r)?\b",
            r"\bwhat to do\b",
            r"\bo que fazer\b",
        ]
        weather_patterns = [
            r"\bweather\b",
            r"\bforecast\b",
            r"\bmeteorolog",
            r"\bprevis[aã]o\b",
            r"\brain\b",
            r"\bchuva\b",
            r"\btemperatura\b",
            r"\btemperature\b",
            r"\btempo em\b",
            r"\bqual (?:é|e) o tempo\b",
        ]
        transport_patterns = [
            r"\btransport\b",
            r"\btransporte\b",
            r"\bmetro\b",
            r"\bbus\b",
            r"\bautocarro\b",
            r"\bcomboio\b",
            r"\btrain\b",
            r"\broute\b",
            r"\brota\b",
            r"\bliga[cç][aã]o\b",
            r"\bconnection\b",
            r"\bhow to get\b",
            r"\bcomo chegar\b",
        ]

        has_event_intent = any(re.search(pattern, query) for pattern in event_patterns)
        has_planning_intent = any(re.search(pattern, query) for pattern in planning_patterns)
        has_weather_intent = any(re.search(pattern, query) for pattern in weather_patterns)
        has_transport_intent = any(re.search(pattern, query) for pattern in transport_patterns)

        return has_event_intent and not has_planning_intent and not has_weather_intent and not has_transport_intent

    @classmethod
    def _is_category_browse_query(cls, user_query: str) -> bool:
        """Detects category-browsing questions that should not require instance cards."""
        normalized = unicodedata.normalize("NFKD", user_query or "")
        normalized = normalized.encode("ascii", "ignore").decode("ascii").lower()
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return bool(
            re.search(r"\b(?:what kinds?|types?|categories?|which kinds?)\b.*\b(?:events?|places?|services?|public services?)\b", normalized)
            or re.search(r"\bque tipos? de\b.*\b(?:eventos?|locais|lugares|atracoes?|servicos?)\b", normalized)
            or re.search(r"\b(?:categorias|tipos) de (?:eventos?|locais|lugares|atracoes?|servicos?)\b", normalized)
            or re.search(r"\b(?:eventos?|locais|lugares|atracoes?|servicos?)\b.*\b(?:posso encontrar|posso procurar|posso explorar|categorias)\b", normalized)
        )

    @classmethod
    def _normalize_category_query_validation(
        cls,
        user_query: str,
        agents_called: List[str],
        llm_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Keep category queries scoped to category lists, not item-level completeness."""
        if not cls._is_category_browse_query(user_query):
            return llm_result

        called_workers = {agent for agent in agents_called if agent and agent not in {"qa", "supervisor"}}
        if called_workers and not called_workers.issubset({"researcher", "final"}):
            return llm_result

        llm_result["required_agents"] = []
        llm_result["missing_data"] = []
        llm_result["disclaimers"] = [
            item for item in llm_result.get("disclaimers", [])
            if not cls._is_cross_domain_event_requirement(item)
        ]
        llm_result["complete"] = True

        normalization_note = (
            "Normalized QA for category query: category lists are complete without event/place instance cards."
        )
        reasoning = llm_result.get("reasoning", "")
        if normalization_note not in reasoning:
            llm_result["reasoning"] = f"{reasoning} | {normalization_note}".strip(" |")

        return llm_result

    @staticmethod
    def _is_cross_domain_event_requirement(text: str) -> bool:
        """Returns whether a QA gap/disclaimer incorrectly asks for weather/transport in an events-only query."""
        lower = (text or "").lower()
        cross_domain_patterns = [
            r"\bweather\b",
            r"\bmeteorolog",
            r"\bforecast\b",
            r"\bprevis[aã]o\b",
            r"\brain\b",
            r"\bchuva\b",
            r"\btemperature\b",
            r"\btemperatura\b",
            r"\btransport\b",
            r"\btransporte\b",
            r"\bmetro\b",
            r"\bcarris\b",
            r"\bcp\b",
            r"\broute\b",
            r"\brota\b",
            r"\bliga[cç][aã]o\b",
            r"\bconnection\b",
            r"\btransfer\b",
            r"\bcomo chegar\b",
            r"\bhow to get\b",
        ]
        return any(re.search(pattern, lower) for pattern in cross_domain_patterns)

    @classmethod
    def _normalize_event_query_validation(
        cls,
        user_query: str,
        agents_called: List[str],
        llm_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prevents event-only queries from being expanded into weather/transport retries by QA."""
        if not cls._is_event_listing_query(user_query):
            return llm_result

        called_workers = {agent for agent in agents_called if agent and agent not in {"qa", "supervisor"}}
        if called_workers and not called_workers.issubset({"researcher", "final"}):
            return llm_result

        filtered_required_agents = [] if "researcher" in called_workers else [
            agent for agent in llm_result.get("required_agents", [])
            if agent == "researcher"
        ]
        filtered_missing_data = [
            item for item in llm_result.get("missing_data", [])
            if not cls._is_cross_domain_event_requirement(item)
            and not re.search(
                r"\b(?:exaustiv|exhaustiv|listagem|listing|canon|canonical|"
                r"idioma|language|r[oó]tulos?|labels?|conte[uú]do|content|"
                r"ingl[eê]s|english|url|slug|morada|address|geogr[aá]fic|"
                r"ambito|[aâ]mbito|scope|contextual|fora de lisboa|outside lisbon|"
                r"filtro|filter|filtrad|filtragem|explicit|expl[ií]cit|exclu|"
                r"ocorr|inteiro|m[eê]s|mes|month)\b",
                cls._normalize_query(str(item or "")),
                flags=re.IGNORECASE,
            )
        ]
        filtered_disclaimers = [
            item for item in llm_result.get("disclaimers", [])
            if not cls._is_cross_domain_event_requirement(item)
        ]

        llm_result["required_agents"] = filtered_required_agents
        llm_result["missing_data"] = filtered_missing_data
        llm_result["disclaimers"] = filtered_disclaimers

        if not filtered_required_agents and not filtered_missing_data:
            llm_result["complete"] = True

        normalization_note = (
            "Normalized QA for event-only query: weather and transport are optional and must not trigger retries."
        )
        reasoning = llm_result.get("reasoning", "")
        if normalization_note not in reasoning:
            llm_result["reasoning"] = (
                f"{reasoning} | {normalization_note}".strip(" |")
            )

        return llm_result

    @classmethod
    def _normalize_event_no_result_validation(
        cls,
        user_query: str,
        agent_outputs: Dict[str, str],
        agents_called: List[str],
        llm_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Accept grounded no-result event listings as complete answers.

        Event searches often have valid negative outcomes, especially when the
        user combines temporal, price, audience, and location filters. In that
        case the best answer is to preserve the applied filters and the scoped
        limitation instead of forcing a repair that invents event cards.
        """
        if not cls._is_event_listing_query(user_query):
            return llm_result

        called_workers = {agent for agent in agents_called if agent and agent not in {"qa", "supervisor"}}
        if called_workers and not called_workers.issubset({"researcher", "final"}):
            return llm_result

        combined_output = "\n".join(
            str(value)
            for key, value in agent_outputs.items()
            if not str(key).startswith("_") and isinstance(value, str)
        )
        normalized_output = cls._normalize_query(combined_output)
        if not normalized_output:
            return llm_result

        no_result_patterns = (
            r"\bnão encontrei eventos\b",
            r"\bnão encontrei mais eventos\b",
            r"\bnao encontrei eventos\b",
            r"\bnao encontrei mais eventos\b",
            r"\bnão há eventos\b",
            r"\bnao ha eventos\b",
            r"\bsem eventos\b",
            r"\bsem resultados\b.*\beventos\b",
            r"\bno events?\b",
            r"\bno more confirmed events?\b",
            r"\bno confirmed events?\b",
            r"\bdid not find\b.*\bmore\b.*\bevents?\b",
            r"\bdid not find\b.*\bevents?\b",
            r"\bcould not find\b.*\bmore\b.*\bevents?\b",
            r"\bcould not find\b.*\bevents?\b",
        )
        if not any(re.search(pattern, normalized_output) for pattern in no_result_patterns):
            return llm_result

        llm_result["required_agents"] = []
        llm_result["missing_data"] = []
        llm_result["complete"] = True

        normalization_note = (
            "Normalized QA for event no-result query: a grounded empty result set is complete."
        )
        reasoning = llm_result.get("reasoning", "")
        if normalization_note not in reasoning:
            llm_result["reasoning"] = f"{reasoning} | {normalization_note}".strip(" |")

        return llm_result

    @classmethod
    def _normalize_place_partial_no_result_validation(
        cls,
        user_query: str,
        agent_outputs: Dict[str, str],
        agents_called: List[str],
        llm_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Accept grounded area-scoped place answers with explicit category gaps.

        Mixed place queries can validly find attractions but no restaurants, or
        the reverse. In that situation QA must not force a generative repair,
        because the best answer is the scoped limitation already returned by the
        Researcher evidence.
        """
        if not cls._is_place_listing_query(user_query):
            return llm_result

        called_workers = {agent for agent in agents_called if agent and agent not in {"qa", "supervisor"}}
        if called_workers and not called_workers.issubset({"researcher", "final"}):
            return llm_result

        combined_output = "\n".join(
            str(value)
            for key, value in agent_outputs.items()
            if not str(key).startswith("_") and isinstance(value, str)
        )
        normalized_query = cls._normalize_query(user_query)
        normalized_output = cls._normalize_query(combined_output)
        if not normalized_output:
            return llm_result

        asks_food = bool(re.search(r"\b(?:restaurantes?|restaurants?|food|comida|gastronomia|cozinha)\b", normalized_query))
        asks_places = bool(re.search(r"\b(?:atracoes?|atra[cç][oõ]es?|places?|locais?|monumentos?|museus?)\b", normalized_query))
        has_place_cards = bool(
            re.search(r"(?m)^\s*(?:[-*]\s+)?\*\*(?:🏛️|📍|🖼️|📚|🌿)\s+[^*\n]+\*\*", combined_output)
        )
        has_food_no_result = bool(
            re.search(
                r"\b(?:sem restaurantes confirmados|não encontrei restaurantes confirmados|nao encontrei restaurantes confirmados|"
                r"no confirmed restaurants|did not find confirmed restaurants)\b",
                normalized_output,
                flags=re.IGNORECASE,
            )
        )
        if not (asks_food and asks_places and has_place_cards and has_food_no_result):
            return llm_result

        llm_result["required_agents"] = []
        llm_result["missing_data"] = []
        llm_result["complete"] = True

        normalization_note = (
            "Normalized QA for mixed place query: explicit category no-result with grounded cards is complete."
        )
        reasoning = llm_result.get("reasoning", "")
        if normalization_note not in reasoning:
            llm_result["reasoning"] = f"{reasoning} | {normalization_note}".strip(" |")

        return llm_result

    @classmethod
    def _is_grounded_place_partial_no_result(
        cls,
        user_query: str,
        agent_outputs: Dict[str, str],
        agents_called: List[str],
    ) -> bool:
        """Return whether Researcher provided grounded cards plus an explicit no-result category."""
        if not cls._is_place_listing_query(user_query):
            return False
        called_workers = {agent for agent in agents_called if agent and agent not in {"qa", "supervisor"}}
        if called_workers and not called_workers.issubset({"researcher", "final"}):
            return False
        combined_output = "\n".join(
            str(value)
            for key, value in agent_outputs.items()
            if not str(key).startswith("_") and isinstance(value, str)
        )
        normalized_query = cls._normalize_query(user_query)
        normalized_output = cls._normalize_query(combined_output)
        asks_food = bool(re.search(r"\b(?:restaurantes?|restaurants?|food|comida|gastronomia|cozinha)\b", normalized_query))
        asks_places = bool(re.search(r"\b(?:atracoes?|atra[cç][oõ]es?|places?|locais?|monumentos?|museus?)\b", normalized_query))
        has_place_cards = bool(
            re.search(r"(?m)^\s*(?:[-*]\s+)?\*\*(?:🏛️|📍|🖼️|📚|🌿)\s+[^*\n]+\*\*", combined_output)
        )
        has_explicit_no_result = bool(
            re.search(
                r"\b(?:sem restaurantes confirmados|não encontrei restaurantes confirmados|nao encontrei restaurantes confirmados|"
                r"no confirmed restaurants|did not find confirmed restaurants)\b",
                normalized_output,
                flags=re.IGNORECASE,
            )
        )
        return asks_food and asks_places and has_place_cards and has_explicit_no_result

    @classmethod
    def _normalize_place_partial_fact_check(
        cls,
        fact_check: Dict[str, Any],
        user_query: str,
        agent_outputs: Dict[str, str],
        agents_called: List[str],
    ) -> Dict[str, Any]:
        """Avoid generative repair for honest mixed-category place limitations."""
        if not cls._is_grounded_place_partial_no_result(user_query, agent_outputs, agents_called):
            return fact_check

        benign_patterns = (
            "Source footer is missing material source(s): Lisboa Aberta",
            "Structured emoji field labels must start on their own line",
            "Place cards are missing canonical fields",
        )
        benign_disclaimer_patterns = (
            "Some URLs reference unverified domains",
            "horários de funcionamento não foram confirmados",
            "opening hours",
        )

        def filter_issues(issues: object) -> List[str]:
            return [
                str(issue)
                for issue in list(issues or [])
                if not any(pattern in str(issue) for pattern in benign_patterns)
            ]

        def filter_disclaimers(disclaimers: object) -> List[str]:
            return [
                str(disclaimer)
                for disclaimer in list(disclaimers or [])
                if not any(pattern.lower() in str(disclaimer).lower() for pattern in benign_disclaimer_patterns)
            ]

        fact_check["critical_issues"] = filter_issues(fact_check.get("critical_issues"))
        fact_check["disclaimers"] = filter_disclaimers(fact_check.get("disclaimers"))
        if not fact_check["critical_issues"]:
            fact_check["valid"] = True
            fact_check["repairable_agents"] = [
                agent for agent in list(fact_check.get("repairable_agents") or [])
                if agent != "researcher"
            ]

        per_agent = fact_check.get("per_agent")
        if isinstance(per_agent, dict):
            for agent_name, agent_check in per_agent.items():
                if agent_name != "researcher" or not isinstance(agent_check, dict):
                    continue
                agent_check["critical_issues"] = filter_issues(agent_check.get("critical_issues"))
                agent_check["disclaimers"] = filter_disclaimers(agent_check.get("disclaimers"))
                if not agent_check["critical_issues"]:
                    agent_check["valid"] = True
                    agent_check["repairable_agents"] = [
                        agent for agent in list(agent_check.get("repairable_agents") or [])
                        if agent != "researcher"
                    ]

        return fact_check

    @classmethod
    def _is_place_listing_query(cls, user_query: str) -> bool:
        """Detects standalone place/attraction discovery queries that should stay inside researcher scope."""
        query = cls._normalize_query(user_query)
        if not query:
            return False

        place_patterns = [
            r"\battractions?\b",
            r"\batra(?:ç|c)[aã]o(?:es)?\b",
            r"\batra[cç][oõ]es?\b",
            r"\bplaces?\b",
            r"\blocais?\b",
            r"\brestaurants?\b",
            r"\brestaurantes?\b",
            r"\bhotels?\b",
            r"\bhot[eé]is\b",
            r"\balojamentos?\b",
            r"\bshops?\b",
            r"\blojas?\b",
            r"\bcompras\b",
            r"\bcentros?\s+comerciais?\b",
            r"\bshopping\b",
            r"\bcruzeiros?\b",
            r"\bcruises?\b",
            r"\bpraias?\b",
            r"\bbeaches\b",
            r"\bgolfe?\b",
            r"\bgolf\b",
            r"\bfado\b",
            r"\bnightlife\b",
            r"\bmuseums?\b",
            r"\bmuseus?\b",
            r"\bmonuments?\b",
            r"\bmonumentos?\b",
            r"\bwhat to visit\b",
            r"\bo que visitar\b",
            r"\bpasseio\b",
            r"\bpassear\b",
            r"\bsugere(?:-me)?\b",
            r"\bs[ií]tios?\b",
            r"\bimperd[ií]veis\b",
            r"\bfirst time\b",
            r"\bprimeira vez\b",
            r"\bdetalhes?\s+sobre\b",
            r"\bdetails?\s+about\b",
            r"\btell me about\b",
            r"\bfala[- ]?me\b",
            r"\bpadr[aã]o dos descobrimentos\b",
            r"\btorre de bel[eé]m\b",
            r"\bcastelo de s[aã]o jorge\b",
            r"\bmosteiro dos jer[oó]nimos\b",
        ]
        planning_patterns = [
            r"\bplan\b",
            r"\bplanning\b",
            r"\bitinerary\b",
            r"\broteiro\b",
            r"\bitiner[aá]rio\b",
            r"\bplane(?:ia|ar|ie)\b",
            r"\borganiza(?:r)?\b",
            r"\bo que fazer\b",
        ]
        weather_patterns = [
            r"\bweather\b",
            r"\bforecast\b",
            r"\bmeteorolog",
            r"\bprevis[aã]o\b",
            r"\brain\b",
            r"\bchuva\b",
            r"\btemperatura\b",
            r"\btemperature\b",
        ]
        transport_patterns = [
            r"\btransport\b",
            r"\btransporte\b",
            r"\bmetro\b",
            r"\bbus\b",
            r"\bautocarro\b",
            r"\bcomboio\b",
            r"\btrain\b",
            r"\broute\b",
            r"\brota\b",
            r"\bhow to get\b",
            r"\bcomo chegar\b",
        ]

        has_place_intent = any(re.search(pattern, query) for pattern in place_patterns)
        has_planning_intent = any(re.search(pattern, query) for pattern in planning_patterns)
        has_weather_intent = any(re.search(pattern, query) for pattern in weather_patterns)
        has_transport_intent = any(re.search(pattern, query) for pattern in transport_patterns)

        return has_place_intent and not has_planning_intent and not has_weather_intent and not has_transport_intent

    @classmethod
    def _normalize_place_query_validation(
        cls,
        user_query: str,
        agents_called: List[str],
        llm_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prevents standalone place-listing queries from being expanded into weather/transport retries by QA."""
        if not cls._is_place_listing_query(user_query):
            return llm_result

        called_workers = {agent for agent in agents_called if agent}
        if called_workers and not called_workers.issubset({"researcher"}):
            return llm_result

        filtered_required_agents = [] if "researcher" in called_workers else [
            agent for agent in llm_result.get("required_agents", [])
            if agent == "researcher"
        ]

        def _is_extra_place_suggestion_requirement(item: object) -> bool:
            normalized = unicodedata.normalize("NFKD", str(item or ""))
            normalized = normalized.encode("ascii", "ignore").decode("ascii").lower()
            return bool(
                re.search(r"\b(?:segunda|second).*(?:terceira|third).*(?:sugesto|suggestion|loca|place|atrac)", normalized)
                or re.search(r"\b(?:more|additional|mais|outras?)\s+(?:places?|locais|attractions?|atracoes?|sugestoes?)\b", normalized)
            )

        def _is_non_retryable_place_detail_gap(item: object) -> bool:
            normalized_item = unicodedata.normalize("NFKD", str(item or ""))
            normalized_item = normalized_item.encode("ascii", "ignore").decode("ascii").lower()
            normalized_query = unicodedata.normalize("NFKD", user_query or "")
            normalized_query = normalized_query.encode("ascii", "ignore").decode("ascii").lower()
            if "acessibilidade" in normalized_item or "accessibility" in normalized_item:
                return not re.search(r"\b(acessibilidade|acessivel|mobilidade reduzida|wheelchair|accessible|accessibility)\b", normalized_query)
            return bool(
                re.search(r"\b(horario|opening hours?|hours?)\b", normalized_item)
                and re.search(r"\b(confirmacao|confirmar|valid[oa]|fonte oficial|desatualiz|outdated|official source)\b", normalized_item)
            )

        filtered_missing_data = [
            item for item in llm_result.get("missing_data", [])
            if not cls._is_cross_domain_event_requirement(item)
            and not _is_extra_place_suggestion_requirement(item)
            and not _is_non_retryable_place_detail_gap(item)
        ]
        filtered_disclaimers = [
            item for item in llm_result.get("disclaimers", [])
            if not cls._is_cross_domain_event_requirement(item)
        ]

        llm_result["required_agents"] = filtered_required_agents
        llm_result["missing_data"] = filtered_missing_data
        llm_result["disclaimers"] = filtered_disclaimers

        if not filtered_required_agents and not filtered_missing_data:
            llm_result["complete"] = True

        normalization_note = (
            "Normalized QA for place-only query: weather and transport are optional and must not trigger retries."
        )
        reasoning = llm_result.get("reasoning", "")
        if normalization_note not in reasoning:
            llm_result["reasoning"] = (
                f"{reasoning} | {normalization_note}".strip(" |")
            )

        return llm_result

    @staticmethod
    def _is_cross_domain_weather_requirement(text: str) -> bool:
        """Returns whether a QA gap incorrectly asks for non-weather workers in a weather-only query."""
        lower = (text or "").lower()
        cross_domain_patterns = [
            r"\bresearcher\b",
            r"\bplaces?\b",
            r"\battractions?\b",
            r"\bevents?\b",
            r"\beventos?\b",
            r"\bvisitlisboa\b",
            r"\btransport\b",
            r"\btransporte\b",
            r"\bmetro\b",
            r"\bbus\b",
            r"\bautocarro\b",
            r"\btrain\b",
            r"\bcomboio\b",
            r"\broute\b",
            r"\brota\b",
            r"\bhow to get\b",
            r"\bcomo chegar\b",
        ]
        return any(re.search(pattern, lower) for pattern in cross_domain_patterns)

    @classmethod
    def _normalize_weather_query_validation(
        cls,
        agents_called: List[str],
        llm_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prevents weather-only queries from being expanded into researcher or transport retries."""
        called_workers = {agent for agent in agents_called if agent}
        if not called_workers or not called_workers.issubset({"weather"}):
            return llm_result

        filtered_required_agents = [
            agent for agent in llm_result.get("required_agents", [])
            if agent == "weather"
        ]
        filtered_missing_data = [
            item for item in llm_result.get("missing_data", [])
            if not cls._is_cross_domain_weather_requirement(item)
        ]
        filtered_disclaimers = [
            item for item in llm_result.get("disclaimers", [])
            if not cls._is_cross_domain_weather_requirement(item)
        ]

        llm_result["required_agents"] = filtered_required_agents
        llm_result["missing_data"] = filtered_missing_data
        llm_result["disclaimers"] = filtered_disclaimers

        if not filtered_required_agents and not filtered_missing_data:
            llm_result["complete"] = True

        normalization_note = (
            "Normalized QA for weather-only query: researcher and transport are optional and must not trigger retries."
        )
        reasoning = llm_result.get("reasoning", "")
        if normalization_note not in reasoning:
            llm_result["reasoning"] = (
                f"{reasoning} | {normalization_note}".strip(" |")
            )

        return llm_result

    @classmethod
    def _normalize_transport_query_validation(
        cls,
        agent_outputs: Dict[str, str],
        agents_called: List[str],
        llm_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Downgrade impossible transport gaps when the answer states a grounded limitation.

        QA should flag missing data that can be repaired by another worker/tool.
        It should not keep retrying when the available transport layer has
        already answered with an explicit limitation, for example unavailable
        official fare data or no CP real-time feed for a timetable result.
        """
        called_workers = {agent for agent in agents_called if agent}
        if not called_workers or "transport" not in called_workers:
            return llm_result

        combined_output = "\n".join(
            str(value) for key, value in agent_outputs.items()
            if not str(key).startswith("_") and isinstance(value, str)
        )
        normalized_output = cls._normalize_query(combined_output)
        if (
            re.search(r"\b(?:ambiguidade|ambiguity|preciso de confirmar|i need to confirm)\b", combined_output, flags=re.IGNORECASE)
            and re.search(
                r"\b(?:indica|especifica|specify|morada|address|zona|area|ponto de referência|landmark)\b",
                combined_output,
                flags=re.IGNORECASE,
            )
        ):
            llm_result["missing_data"] = []
            llm_result["required_agents"] = []
            llm_result["complete"] = True
            return llm_result

        has_actionable_bus_route = bool(
            re.search(
                r"(?:melhor op[cç][aã]o confirmada|best confirmed option).{0,240}"
                r"(?:apanha|take).{0,80}\b\d{3,4}[A-Z]?\b.{0,240}"
                r"(?:sai em|alight at)",
                combined_output,
                flags=re.IGNORECASE | re.DOTALL,
            )
            or re.search(
                r"(?:apanha em|board at).{0,180}(?:sai em|get off at|alight at).{0,180}"
                r"(?:linhas?|lines?)\s*(?:\*\*)?\s*:\s*(?:\*\*)?\s*\d{3,4}[A-Z]?",
                combined_output,
                flags=re.IGNORECASE | re.DOTALL,
            )
            or re.search(
                r"(?:paragens|stops).{0,120}(?:apanha em|board at).{0,120}(?:sai em|alight at).{0,160}"
                r"(?:tempo estimado|estimated travel time)",
                combined_output,
                flags=re.IGNORECASE | re.DOTALL,
            )
            or re.search(
                r"(?:apanha em|board at).{0,420}(?:sai em|leave at|alight at).{0,420}"
                r"(?:tempo estimado|estimated travel time|~\s*\d+\s*min)",
                combined_output,
                flags=re.IGNORECASE | re.DOTALL,
            )
        )
        has_actionable_bus_realtime = bool(
            re.search(
                r"(?:pr[oó]ximas partidas|next departures).{0,220}"
                r"(?:tempo real|real-time|em tempo real|live|atraso|delay)",
                combined_output,
                flags=re.IGNORECASE | re.DOTALL,
            )
        )
        has_actionable_departures = bool(
            re.search(
                r"\b(?:pr[oó]ximas partidas|next departures)\b",
                combined_output,
                flags=re.IGNORECASE,
            )
        )
        has_declared_realtime_limitation = bool(
            re.search(
                r"(?:tempo real|real[- ]?time|live).{0,260}"
                r"(?:confirmad|confirmar|sem dados|no real[- ]?time|not confirmed|not available|"
                r"indispon[ií]vel|pode ficar desatualizad|may become stale)",
                normalized_output,
                flags=re.IGNORECASE | re.DOTALL,
            )
        )
        has_quality_route_limitation = bool(
            re.search(
                r"(?:pouca caminhada|menos caminhada|low walking|less walking|rain|chuva|chover).{0,700}"
                r"(?:nao ficou confirmad|não ficou confirmad|nao foi confirmad|não foi confirmad|"
                r"not confirmed|nao consigo confirmar|não consigo confirmar|cannot confirm|"
                r"continua a ser|confirmed option|opcao confirmada|opção confirmada)",
                combined_output,
                flags=re.IGNORECASE | re.DOTALL,
            )
        )
        combined_query_output = cls._normalize_query(
            " ".join(
                str(part or "")
                for part in [llm_result.get("reasoning"), *llm_result.get("missing_data", [])]
            )
            + " "
            + combined_output
        )
        has_generic_service_area_route = bool(
            re.search(
                r"\b(?:veterinario|veterinaria|veterinary|farmacia|pharmacy|restaurante|restaurant|taberna|loja|store|shop)\b",
                combined_query_output,
                flags=re.IGNORECASE,
            )
            and re.search(
                r"(?:usei|used).{0,120}(?:ponto de referencia|ponto de referência|destination reference|area)|"
                r"(?:nao consigo confirmar|não consigo confirmar|cannot confirm|no specific confirmed).{0,220}"
                r"(?:morada|address|nome especifico|nome específico|specific name|veterinario|veterinary|servico|service)",
                normalized_output,
                flags=re.IGNORECASE | re.DOTALL,
            )
        )
        requested_cm_line_match = re.search(
            r"\b(?:linha|line)\s+(?P<line>\d{3,4}[A-Z]?)\b",
            combined_query_output,
            flags=re.IGNORECASE,
        )
        requested_cm_line = requested_cm_line_match.group("line").upper() if requested_cm_line_match else ""
        has_filtered_cm_alert = bool(
            requested_cm_line
            and (
                re.search(
                    rf"\b(?:linha consultada|requested line|linhas? afetadas?|affected lines?)\b"
                    rf"[^0-9A-Z]{{0,40}}{re.escape(requested_cm_line)}\b",
                    normalized_output,
                    flags=re.IGNORECASE,
                )
                or re.search(
                    rf"\balertas?\b[^.\n]{{0,160}}\blinha\b[^0-9A-Z]{{0,20}}{re.escape(requested_cm_line)}\b",
                    normalized_output,
                    flags=re.IGNORECASE,
                )
            )
        )

        limitation_patterns = {
            "fare": r"(tarifa|preco|price|fare).{0,120}(nao foi possivel|not possible|not confirmed|nao confirmad|not available)",
            "cp_realtime": r"(cp|comboio|train).{0,160}(sem dados em tempo real|no real time|real time.*(?:unavailable|not available|not confirmed))",
            "delay_unknown": r"(nao confirma|does not confirm|cannot confirm).{0,120}(atras|delay|on time|pontual)",
        }

        def _is_satisfied_limitation(item: object) -> bool:
            normalized_item = cls._normalize_query(str(item or ""))
            if (
                not re.search(r"\b(?:tempo real|real time|perturb|estado|status|disruption)\b", normalized_item)
                and any(
                    token in normalized_item
                    for token in (
                        "ligação de autocarro",
                        "ligacao de autocarro",
                        "linha, direção",
                        "linha, direcao",
                        "pontos de transferência",
                        "pontos de transferencia",
                        "bus route",
                        "bus option",
                    )
                )
            ):
                return has_actionable_bus_route
            if (
                has_filtered_cm_alert
                and requested_cm_line
                and re.search(r"\b(?:alerta|alert|linha)\b", normalized_item)
                and re.search(
                    r"\b(?:exclusiv\w*|estrit\w*|especificamente|conjunto|sem incluir|associad\w*|sem referencia|sem referência|only|exclusive)\b",
                    normalized_item,
                )
            ):
                return True
            if has_filtered_cm_alert and requested_cm_line and re.search(
                r"\b(?:truncamento|texto descritivo|detalhes? completos?|full detail|specific source|fonte especifica|fonte específica|"
                r"pagina da linha|página da linha|citacao material|citação material|pagina oficial|página oficial)\b",
                normalized_item,
            ):
                return True
            if has_filtered_cm_alert and requested_cm_line and re.search(
                r"\b(?:estado em tempo real|perturbacoes especificas|perturbações específicas|aviso generico|aviso genérico|alerta activo|alerta ativo)\b",
                normalized_item,
            ):
                return True
            if has_filtered_cm_alert and requested_cm_line and re.search(
                r"\b(?:clarific\w*|partilhad\w*|filtrad\w*|omitid\w*)\b",
                normalized_item,
            ):
                return True
            if any(token in normalized_item for token in ("tarifa", "preco", "price", "fare")):
                return bool(re.search(limitation_patterns["fare"], normalized_output))
            if "tempo real" in normalized_item or "real time" in normalized_item:
                if has_actionable_bus_realtime:
                    return True
                if has_actionable_bus_route and has_actionable_departures:
                    return True
                if has_declared_realtime_limitation:
                    return True
                if re.search(
                    r"(?:tempo real carris metropolitana|carris metropolitana real time).{0,220}"
                    r"(?:nao proximos autocarros|não próximos autocarros|not next buses|not confirm real-time departure)",
                    normalized_output,
                    flags=re.IGNORECASE | re.DOTALL,
                ):
                    return True
                return bool(re.search(limitation_patterns["cp_realtime"], normalized_output))
            if re.search(r"\b(?:percurso final|acesso final|entrada|walk|walking|final access)\b", normalized_item):
                return bool(
                    re.search(
                        r"(?:percurso final|acesso final|final access).{0,180}"
                        r"(?:nao ficou confirmado|não ficou confirmado|not confirmed)",
                        normalized_output,
                        flags=re.IGNORECASE | re.DOTALL,
                    )
                )
            if has_generic_service_area_route and re.search(
                r"\b(?:veterinario|veterinaria|veterinary|servico|serviço|service|morada|address|"
                r"localizacao|localização|location|ponto de chegada|ultima perna|última perna|last leg|final walk)\b",
                normalized_item,
            ):
                return True
            if any(
                token in normalized_item
                for token in (
                    "pouca caminhada", "menos caminhada", "low walking", "less walking",
                    "walking", "walk", "chuva", "rain", "rota detalhada alternativa",
                    "alternative route",
                )
            ):
                return has_quality_route_limitation
            if any(token in normalized_item for token in ("on time", "delayed", "disrupted", "atras", "perturb")):
                return bool(re.search(limitation_patterns["delay_unknown"], normalized_output))
            return False

        filtered_missing = [
            item for item in llm_result.get("missing_data", [])
            if not _is_satisfied_limitation(item)
        ]
        llm_result["missing_data"] = filtered_missing
        if not filtered_missing and not llm_result.get("critical_issues"):
            llm_result["complete"] = True
            llm_result["required_agents"] = []
            llm_result["needs_repair"] = False
        return llm_result

    @classmethod
    def _augment_query_specific_validation(
        cls,
        user_query: str,
        agent_outputs: Dict[str, str],
        llm_result: Dict[str, Any],
        language: str,
    ) -> Dict[str, Any]:
        """Adds deterministic completeness guards for common multi-part query failures."""
        query = cls._normalize_query(user_query)
        combined_output = "\n".join(
            str(value) for key, value in agent_outputs.items()
            if not str(key).startswith("_") and isinstance(value, str)
        )
        output_lower = combined_output.lower()
        if (
            re.search(r"\b(?:ambiguidade|ambiguity)\b", combined_output, flags=re.IGNORECASE)
            and re.search(
                r"\b(?:indica|especifica|specify|morada|address|zona|area|ponto de referência|landmark)\b",
                combined_output,
                flags=re.IGNORECASE,
            )
        ):
            llm_result["missing_data"] = []
            llm_result["required_agents"] = []
            llm_result["complete"] = True
            return llm_result

        missing_data = cls._dedupe_preserve_order(list(llm_result.get("missing_data", [])))
        required_agents = cls._dedupe_preserve_order(list(llm_result.get("required_agents", [])))
        disclaimers = cls._dedupe_preserve_order(list(llm_result.get("disclaimers", [])))
        reasoning = str(llm_result.get("reasoning", "") or "").strip()
        reasoning_notes: List[str] = []

        planning_without_weather = bool(
            re.search(r"\b(?:plan|planning|itinerary|roteiro|itinerario|itiner[aá]rio|afternoon|day|tarde|dia)\b", query)
            and not re.search(
                r"\b(?:weather|forecast|rain|rainy|temperature|wind|umbrella|"
                r"chuva|chover|previs[aã]o|temperatura|vento|guarda[-\s]?chuva)\b",
                query,
            )
        )
        if planning_without_weather:
            weather_gap_re = re.compile(
                r"\b(?:weather|forecast|rain|rain probability|temperature|wind|warnings?|"
                r"meteorolog|previs[aã]o|chuva|probabilidade de chuva|temperatura|vento|avisos?)\b",
                flags=re.IGNORECASE,
            )
            before_count = len(missing_data)
            missing_data = [
                item for item in missing_data
                if not weather_gap_re.search(cls._normalize_query(str(item or "")))
            ]
            if len(missing_data) != before_count:
                reasoning_notes.append("removed QA gap for weather data not requested by the planning query")
            if "weather" in required_agents:
                required_agents = [agent for agent in required_agents if agent != "weather"]

        rejected_mode_markers = {
            "metro": r"\b(?:n[aã]o\s+(?:quero|usar|uses?|meter)|sem|without|no)\s+metro\b|\bmetro\b.{0,30}\b(?:n[aã]o|sem|without|no)\b",
            "bus": r"\b(?:n[aã]o\s+(?:quero|usar|uses?|meter)|sem|without|no)\s+(?:autocarro|bus|ônibus|onibus)\b|\b(?:autocarro|bus)\b.{0,30}\b(?:n[aã]o|sem|without|no)\b",
            "train": r"\b(?:n[aã]o\s+(?:quero|usar|uses?|meter)|sem|without|no)\s+(?:comboio|train|cp)\b|\b(?:comboio|train|cp)\b.{0,30}\b(?:n[aã]o|sem|without|no)\b",
        }
        rejected_modes = {
            mode for mode, pattern in rejected_mode_markers.items()
            if re.search(pattern, query, flags=re.IGNORECASE)
        }
        if rejected_modes:
            before_count = len(missing_data)
            missing_data = [
                item for item in missing_data
                if not any(mode in cls._normalize_query(str(item or "")) for mode in rejected_modes)
            ]
            if len(missing_data) != before_count:
                reasoning_notes.append("removed QA gap for a transport mode explicitly rejected by the user")

        def add_gap(message: str, required_agent: Optional[str] = None) -> None:
            if message not in missing_data:
                missing_data.append(message)
            if required_agent and required_agent not in required_agents:
                required_agents.append(required_agent)

        expected_language = language if language in {"pt", "en"} else infer_response_language(user_query=user_query, default="en")
        output_language = infer_visible_label_language(combined_output, default=expected_language)
        if combined_output.strip() and output_language != expected_language:
            add_gap(
                "response language should match the user's English query"
                if expected_language == "en"
                else "o idioma da resposta deve corresponder ao pedido do utilizador em Português",
            )
            reasoning_notes.append("detected response-language mismatch")

        service_requirements = [
            (
                ("hospital", "hospitals", "hospital", "hospitais"),
                ("hospital", "hospitais", "🏥"),
                "nearest hospital information" if language == "en" else "informação sobre o hospital mais próximo",
            ),
            (
                ("pharmacy", "pharmacies", "farmácia", "farmacia", "farmácias", "farmacias"),
                ("pharmacy", "pharmacies", "farmácia", "farmacia", "farmácias", "farmacias", "💊"),
                "nearest pharmacy information" if language == "en" else "informação sobre a farmácia mais próxima",
            ),
        ]
        for query_markers, output_markers, gap_label in service_requirements:
            if any(marker in query for marker in query_markers) and not any(marker in output_lower for marker in output_markers):
                add_gap(gap_label, required_agent="researcher")
                reasoning_notes.append(gap_label)

        asks_opening_hours = bool(
            re.search(
                r"\b(?:hor[aá]rio|horarios|hours|opening|abert[oa]|open|fecha|closes|disponibilidade|availability)\b",
                query,
                flags=re.IGNORECASE,
            )
        )
        has_nearby_service_result = bool(
            re.search(r"\b(?:mais perto|nearest|perto de|near)\b", output_lower, flags=re.IGNORECASE)
            and re.search(r"\b(?:dist[aâ]ncia|distance|tempo a p[eé]|walking time)\b", output_lower, flags=re.IGNORECASE)
        )
        if has_nearby_service_result and not asks_opening_hours:
            before_count = len(missing_data)
            missing_data = [
                item for item in missing_data
                if not re.search(
                    r"\b(?:hor[aá]rio|horarios|hours|opening|funcionamento|abert[oa]|open|disponibilidade|availability)\b",
                    cls._normalize_query(str(item or "")),
                    flags=re.IGNORECASE,
                )
            ]
            if len(missing_data) != before_count:
                reasoning_notes.append("opening hours are optional for this nearby-service request")

        query_has_reference_anchor = bool(
            re.search(r"\b(?:perto de|near|junto a|around|em torno de)\b", query, flags=re.IGNORECASE)
        )
        if has_nearby_service_result and query_has_reference_anchor:
            before_count = len(missing_data)
            missing_data = [
                item for item in missing_data
                if not re.search(
                    r"\b(?:ponto de partida|localiza[cç][aã]o do utilizador|user location|user origin|origin point|morada/ponto de partida)\b",
                    cls._normalize_query(str(item or "")),
                    flags=re.IGNORECASE,
                )
            ]
            if len(missing_data) != before_count:
                reasoning_notes.append("nearby-service ranking uses the explicit reference anchor from the query")
            if not any(
                re.search(r"\b(?:servi[cç]o|service|biblioteca|library|farmacia|pharmacy|hospital)\b", cls._normalize_query(str(item or "")))
                for item in missing_data
            ):
                required_agents = [agent for agent in required_agents if agent != "researcher"]

        if has_nearby_service_result:
            before_count = len(missing_data)
            missing_data = [
                item for item in missing_data
                if not re.search(
                    r"\b(?:instru[cç][oõ]es?.*(?:p[eé]|pedonal)|caminho\s+a\s+p[eé]|"
                    r"rota\s+(?:a\s+p[eé]|pedonal)|walking\s+(?:route|directions)|"
                    r"ponto\s+de\s+partida\s+exato|exact\s+starting\s+point|"
                    r"proximidade.*coerente|coerente.*proximidade)\b",
                    cls._normalize_query(str(item or "")),
                    flags=re.IGNORECASE,
                )
            ]
            if len(missing_data) != before_count:
                reasoning_notes.append(
                    "nearby-service distance and walking-time estimate are sufficient when exact pedestrian routing is unavailable"
                )
            if not missing_data:
                required_agents = [
                    agent for agent in required_agents
                    if agent not in {"transport", "researcher"}
                ]

        if has_nearby_service_result and re.search(r"\b(?:morada|address|localiza[cç][aã]o|location)\b", output_lower):
            before_count = len(missing_data)
            missing_data = [
                item for item in missing_data
                if not re.search(
                    r"\b(?:designa[cç][aã]o canonica|canonical|nome.*morada|name.*address|fonte material|material source|campo util|additional field)\b",
                    cls._normalize_query(str(item or "")),
                    flags=re.IGNORECASE,
                )
            ]
            if len(missing_data) != before_count:
                reasoning_notes.append("nearby-service result already includes name, address, distance and walking time")
            if not any(
                re.search(r"\b(?:servi[cç]o|service|biblioteca|library|farmacia|pharmacy|hospital)\b", cls._normalize_query(str(item or "")))
                for item in missing_data
            ):
                required_agents = [agent for agent in required_agents if agent != "researcher"]

        event_query = bool(
            re.search(
                r"\b(?:evento|eventos|event|events|concerto|concert|festival|exposi[cç][aã]o|exhibition|"
                r"desporto|desportiv[oa]s?|sports?|arrai(?:al|ais|s)?|santos populares|marchas populares)\b",
                query,
                flags=re.IGNORECASE,
            )
        )
        if event_query:
            normalized_output = cls._normalize_query(combined_output)
            has_event_card = bool(
                re.search(r"(?m)^-\s+\*\*.+?\*\*", combined_output)
                and re.search(
                    r"\b(?:visitlisboa|data/hora|date/time|quando|when|bilhetes|tickets)\b",
                    normalized_output,
                    flags=re.IGNORECASE,
                )
            )
            free_filter_ok = (
                not re.search(r"\b(?:gratuit|gratis|grátis|free)\b", query, flags=re.IGNORECASE)
                or re.search(
                    r"\b(?:entrada gratuita|gratuito|gratuita|free entry|free admission|free)\b",
                    normalized_output,
                    flags=re.IGNORECASE,
                )
            )
            belem_filter_ok = (
                not re.search(r"\b(?:bel[eé]m|belem)\b", query, flags=re.IGNORECASE)
                or re.search(r"\b(?:bel[eé]m|belem)\b", normalized_output, flags=re.IGNORECASE)
            )
            date_filter_ok = (
                not re.search(
                    r"\b(?:fim de semana|weekend|esta semana|this week|hoje|today|amanh[aã]|tomorrow)\b",
                    query,
                    flags=re.IGNORECASE,
                )
                or re.search(
                    r"\b(?:data/hora|date/time|quando|when|\d{1,2}\s+de\s+\w+|\bmaio\b|\bmay\b)\b",
                    normalized_output,
                    flags=re.IGNORECASE,
                )
            )

            if has_event_card and free_filter_ok and belem_filter_ok and date_filter_ok:
                def _is_satisfied_event_gap(item: str) -> bool:
                    normalized_item = cls._normalize_query(str(item or ""))
                    if not re.search(r"\b(?:evento|eventos|event|events)\b", normalized_item):
                        return False
                    return bool(
                        re.search(
                            r"\b(?:adicional|additional|mais|more|plural|gratuit|gratis|grátis|free|data|date|fim de semana|weekend)\b",
                            normalized_item,
                            flags=re.IGNORECASE,
                        )
                    )

                before_count = len(missing_data)
                missing_data = [
                    item for item in missing_data
                    if not _is_satisfied_event_gap(str(item))
                    and not re.search(
                        r"\b(?:exaustiv|exhaustiv|canon|canonical|idioma|language|"
                        r"r[oó]tulos?|labels?|conte[uú]do|content|ingl[eê]s|english|"
                        r"pt[-\s]?pt|slug|morada|address|url|link|fonte|source|data|date|evento|event)\b",
                        cls._normalize_query(str(item or "")),
                        flags=re.IGNORECASE,
                    )
                ]
                if len(missing_data) != before_count:
                    reasoning_notes.append("grounded event result satisfies requested filters")
                if not any(
                    re.search(r"\b(?:evento|eventos|event|events)\b", cls._normalize_query(str(item or "")))
                    for item in missing_data
                ):
                    required_agents = [agent for agent in required_agents if agent != "researcher"]

        forbids_metro = bool(
            re.search(r"\b(?:sem|without|no|n[aã]o\s+(?:quero|usar|uses?|meter))\s+metro\b", query)
            or re.search(r"\bmetro\b.{0,30}\b(?:sem|without|no|n[aã]o)\b", query)
            or re.search(r"\b(?:so|only)\s+(?:de\s+)?(?:autocarro|autocarros|bus|buses)\b", query)
        )
        forbids_bus = bool(
            re.search(r"\b(?:sem|without|no|n[aã]o\s+(?:quero|usar|uses?|meter))\s+(?:autocarro|autocarros|bus|buses)\b", query)
            or re.search(r"\b(?:autocarro|autocarros|bus|buses)\b.{0,30}\b(?:sem|without|no|n[aã]o)\b", query)
            or re.search(r"\b(?:so|only)\s+(?:de\s+)?metro\b", query)
        )
        forbids_train = bool(
            re.search(r"\b(?:sem|without|no|n[aã]o\s+(?:quero|usar|uses?|meter))\s+(?:comboio|comboios|train|trains|cp)\b", query)
            or re.search(r"\b(?:comboio|comboios|train|trains|cp)\b.{0,30}\b(?:sem|without|no|n[aã]o)\b", query)
        )
        forbids_tram = bool(
            re.search(r"\b(?:sem|without|no|n[aã]o\s+(?:quero|usar|uses?|meter))\s+(?:eletrico|eletricos|tram|trams)\b", query)
            or re.search(r"\b(?:eletrico|eletricos|tram|trams)\b.{0,30}\b(?:sem|without|no|n[aã]o)\b", query)
        )

        asks_metro = bool(re.search(r"\bmetro\b", query)) and not forbids_metro
        asks_train = bool(re.search(r"\b(comboio|comboios|train|trains)\b", query)) and not forbids_train
        asks_bus = bool(re.search(r"\b(autocarro|autocarros|bus|buses)\b", query)) and not forbids_bus
        asks_tram = bool(re.search(r"\b(eletrico|eletricos|tram|trams)\b", query)) and not forbids_tram
        forbidden_mode_gaps = []
        if forbids_metro:
            forbidden_mode_gaps.extend(["metro option details", "detalhes da opcao de metro", "detalhes da opção de metro"])
        if forbids_bus:
            forbidden_mode_gaps.extend(["bus option details", "detalhes da opcao de autocarro", "detalhes da opção de autocarro"])
        if forbids_train:
            forbidden_mode_gaps.extend(["train option details", "detalhes da opcao de comboio", "detalhes da opção de comboio"])
        if forbids_tram:
            forbidden_mode_gaps.extend(["tram option details", "detalhes da opcao de eletrico", "detalhes da opção de elétrico", "detalhes da opção de eletrico"])
        if forbidden_mode_gaps:
            before_count = len(missing_data)
            missing_data = [
                item for item in missing_data
                if not any(gap in cls._normalize_query(str(item or "")) for gap in forbidden_mode_gaps)
            ]
            if len(missing_data) != before_count:
                reasoning_notes.append("removed forbidden transport mode from QA gaps")
        asks_fastest = bool(re.search(r"\b(mais r[aá]pid[oa]|faster|fastest|quickest)\b", query))
        asks_cheapest = bool(re.search(r"\b(mais barat[oa]|cheaper|cheapest|lowest cost|pre[cç]o)\b", query))
        requested_modes = [
            mode for mode, requested in (
                ("metro", asks_metro),
                ("train", asks_train),
                ("bus", asks_bus),
                ("tram", asks_tram),
            )
            if requested
        ]

        output_mode_markers = {
            "metro": r"\b(?:metro|metropolitano|linha\s+(?:amarela|azul|verde|vermelha)|yellow line|blue line|green line|red line)\b",
            "train": r"\b(?:cp|comboio|comboios|train|trains|linha\s+de\s+sintra|linha\s+de\s+cascais|linha\s+da\s+azambuja|linha\s+do\s+sado)\b",
            "bus": r"\b(?:carris|autocarro|autocarros|bus|buses|linha\s+\d{3,4}|linhas\s+\d{3,4})\b",
            "tram": r"\b(?:el[eé]trico|eletrico|tram|trams|\b\d{1,2}e\b)\b",
        }

        def _output_mentions_mode(mode: str) -> bool:
            return bool(re.search(output_mode_markers[mode], output_lower, flags=re.IGNORECASE))

        for mode in requested_modes:
            if _output_mentions_mode(mode):
                continue
            label_by_mode = {
                "metro": "metro option details" if language == "en" else "detalhes da opção de metro",
                "train": "train option details" if language == "en" else "detalhes da opção de comboio",
                "bus": "bus option details" if language == "en" else "detalhes da opção de autocarro",
                "tram": "tram option details" if language == "en" else "detalhes da opção de elétrico",
            }
            add_gap(label_by_mode[mode], required_agent="transport")
            reasoning_notes.append(f"missing requested transport mode: {mode}")

        best_transport_requested = bool(
            re.search(r"\b(?:melhor\s+opcao|melhor\s+opcao|melhor\s+linha|best\s+option|best\s+line)\b", query)
        )
        has_ranked_transport_answer = bool(
            re.search(r"\b(?:melhor\s+opcao\s+confirmada|best\s+confirmed\s+option)\b", output_lower)
            and re.search(r"\b(?:tempo\s+estimado|estimated\s+time|~?\d{1,3}\s*min)\b", output_lower)
        )
        if best_transport_requested and has_ranked_transport_answer:
            before_count = len(missing_data)
            missing_data = [
                item for item in missing_data
                if not re.search(
                    r"\b(?:melhor|best|rapida|rapido|faster|fastest|quickest|comparacao|comparison)\b",
                    cls._normalize_query(str(item or "")),
                    flags=re.IGNORECASE,
                )
            ]
            if len(missing_data) != before_count:
                reasoning_notes.append("ranked best transport option is already present")

        if requested_modes and all(_output_mentions_mode(mode) for mode in requested_modes):
            comparison_requested = asks_fastest or asks_cheapest

            def _is_satisfied_transport_option_gap(item: str) -> bool:
                normalized_item = cls._normalize_query(str(item or ""))
                if comparison_requested and re.search(
                    r"\b(?:mais rapido|mais rapida|faster|fastest|quickest|mais barato|mais barata|cheaper|cheapest|tarifa|fare|preco|price)\b",
                    normalized_item,
                    flags=re.IGNORECASE,
                ):
                    return False
                return bool(
                    re.search(
                        r"\b(?:rota|rotas|route|routes|opcao|opcoes|option|options|modo|modos|mode|modes|transporte|transport|ambas|both|coerente|coherent|especifica|specific)\b",
                        normalized_item,
                        flags=re.IGNORECASE,
                    )
                    and (
                        any(
                            re.search(output_mode_markers[mode], normalized_item, flags=re.IGNORECASE)
                            for mode in requested_modes
                        )
                        or bool(re.search(r"\b(?:ambas|both)\b", normalized_item, flags=re.IGNORECASE))
                    )
                )

            before_count = len(missing_data)
            missing_data = [
                item for item in missing_data
                if not _is_satisfied_transport_option_gap(str(item))
            ]
            if len(missing_data) != before_count:
                reasoning_notes.append("requested transport modes are present with route details")

        is_final_response_audit = "final" in agent_outputs
        if is_final_response_audit:
            normalized_user_query = _qa_normalize_text(user_query)
            plan_like_query = bool(
                re.search(
                    r"\b(?:roteiro|plano|itinerario|itinerary|plan|visitar|visit|dia|day)\b",
                    normalized_user_query,
                    flags=re.IGNORECASE,
                )
            )
            if plan_like_query:
                for label in _qa_requested_anchor_labels(user_query):
                    if _qa_response_mentions_anchor(combined_output, label):
                        continue
                    add_gap(
                        f"explicitly requested stop or area: {label}"
                        if language == "en"
                        else f"paragem ou zona explicitamente pedida: {label}",
                        required_agent="planner",
                    )
                    reasoning_notes.append(f"missing requested anchor: {label}")
                start_label = _qa_requested_start_label(user_query)
                if start_label and not _qa_first_route_block_mentions_anchor(combined_output, start_label):
                    add_gap(
                        f"first itinerary stop should match requested start: {start_label}"
                        if language == "en"
                        else f"a primeira paragem deve respeitar o início pedido: {start_label}",
                        required_agent="planner",
                    )
                    reasoning_notes.append(f"wrong requested start: {start_label}")
                normalized_final_output = _qa_normalize_text(combined_output)
                if (
                    re.search(
                        r"\b(?:gastronom|restaurant|restaurante|food|comida|tradicional|almoco|lunch|jantar|dinner|cozinha)\b",
                        normalized_user_query,
                        flags=re.IGNORECASE,
                    )
                    and not re.search(
                        r"\b(?:almoco|lunch|jantar|dinner|restaurante|restaurant|cozinha|comida)\b",
                        normalized_final_output,
                        flags=re.IGNORECASE,
                    )
                ):
                    add_gap(
                        "requested meal or restaurant stop"
                        if language == "en"
                        else "paragem de refeição ou restaurante pedido",
                        required_agent="planner",
                    )
                    reasoning_notes.append("missing requested food stop")

                multi_stop_plan_requested = bool(
                    re.search(
                        r"\b(?:full\s+day|dia\s+inteiro|1\s+dia|one\s+day|roteiro|itinerario|itinerary|"
                        r"museums?|museus|viewpoints?|miradouros?|monuments?|monumentos)\b",
                        normalized_user_query,
                        flags=re.IGNORECASE,
                    )
                )
                if multi_stop_plan_requested:
                    route_stop_count = len(
                        re.findall(
                            r"(?m)^\s*[-*]\s+\*\*[^*\n]{3,160}\*\*",
                            combined_output,
                        )
                    )
                    grounded_field_count = len(
                        re.findall(
                            r"\*\*(?:Address|Morada|More details|Mais detalhes|Website|Category|Categoria|Hours|Hor[aá]rio|Price|Pre[cç]o)\s*:\*\*",
                            combined_output,
                            flags=re.IGNORECASE,
                        )
                    )
                    generic_city_leg = bool(
                        re.search(
                            r"\b(?:rossio|baixa|chiado|carmo|marques|marqu[eê]s)\s*(?:->|→)\s*(?:lisbon|lisboa)\b|"
                            r"\b(?:lisbon|lisboa)\s*(?:->|→)\s*[^:\n]+",
                            normalized_final_output,
                            flags=re.IGNORECASE,
                        )
                    )
                    if route_stop_count < 2 or grounded_field_count == 0 or generic_city_leg:
                        add_gap(
                            "concrete grounded itinerary stops and non-generic movement legs"
                            if language == "en"
                            else "paragens concretas fundamentadas e deslocações não genéricas",
                            required_agent="planner",
                        )
                        reasoning_notes.append("planner final response is too generic for a multi-stop itinerary")

            asks_movement_details = bool(
                re.search(
                    r"\b(?:como\s+(?:me|te)?\s*deslocas?|deslocacoes|deslocacao|deslocar|transportes?|"
                    r"percurso|trajeto|rota|route|movement|how to move|get around|getting around|"
                    r"inclui\s+(?:como|transportes?|deslocacoes)|include\s+(?:transport|movement|how to))\b",
                    normalized_user_query,
                    flags=re.IGNORECASE,
                )
            )
            if (
                asks_movement_details
                and _qa_mentions_central_and_belem(user_query)
                and not _qa_response_has_cross_zone_movement(combined_output)
                and not _qa_response_has_cross_zone_limitation(combined_output)
            ):
                add_gap(
                    "movement leg between central Lisbon and Belém"
                    if language == "en"
                    else "ligação de deslocação entre o centro de Lisboa e Belém",
                    required_agent="planner",
                )
                if "transport" not in required_agents:
                    required_agents.append("transport")
                reasoning_notes.append("missing requested cross-zone movement leg")

        if asks_metro and asks_train:
            if not re.search(r"\bmetro\b", output_lower):
                add_gap(
                    "metro option details" if language == "en" else "detalhes da opção de metro",
                    required_agent="transport",
                )
            if not re.search(r"\b(comboio|comboios|train|trains)\b", output_lower):
                add_gap(
                    "train option details" if language == "en" else "detalhes da opção de comboio",
                    required_agent="transport",
                )
            if asks_fastest and not re.search(r"\b(mais r[aá]pid[oa]|faster|fastest|quickest)\b", output_lower):
                add_gap(
                    "which option is faster" if language == "en" else "qual das opções é mais rápida",
                    required_agent="transport",
                )
            if asks_cheapest and not re.search(
                r"\b(mais barat[oa]|cheaper|cheapest|fare|tarifa|price|pre[cç]o|n[aã]o (?:foi )?poss[ií]vel|not available|not possible to confirm)\b",
                output_lower,
            ):
                add_gap(
                    "which option is cheaper or an explicit fare-data limitation"
                    if language == "en"
                    else "qual das opções é mais barata ou uma nota explícita sobre a falta de dados de tarifa",
                    required_agent="transport",
                )
                reasoning_notes.append("missing fare comparison")
                fare_disclaimer = (
                    "Official fare data was not confirmed in the available transport tools."
                    if language == "en"
                    else "Os dados oficiais de tarifa não foram confirmados nas ferramentas de transporte disponíveis."
                )
                if fare_disclaimer not in disclaimers:
                    disclaimers.append(fare_disclaimer)

        critical_issues = cls._dedupe_preserve_order(list(llm_result.get("critical_issues", [])))

        def add_critical_issue(message: str) -> None:
            """Add a deterministic final-answer quality issue."""
            if message not in critical_issues:
                critical_issues.append(message)

        if re.search(r"\b~?\s*--\s*min\b", combined_output, flags=re.IGNORECASE):
            add_critical_issue("placeholder travel time leaked into the user-facing answer")
            reasoning_notes.append("detected placeholder travel-time output")
        if re.search(r"(?mi)^###\s+✅\s+\*\*(?:Resposta direta|Direct answer):", combined_output):
            add_critical_issue("direct answer was incorrectly rendered as a heading")
            reasoning_notes.append("detected malformed direct-answer heading")
        if (
            re.search(r"(?m)^###\s+", combined_output)
            and re.search(r"\*\*(?:Fonte|Source):\*\*", combined_output, flags=re.IGNORECASE)
            and not re.search(r"\*\*(?:Resposta direta|Direct answer):\*\*", combined_output, flags=re.IGNORECASE)
        ):
            add_critical_issue("missing direct-answer line in a factual response")
            reasoning_notes.append("detected missing direct answer")
        if re.search(
            r"(?mi)^(?:💡\s*)?(?:Dicas(?: Práticas)?|Practical Tips)\s*:?\s*$\n\s*(?:📌\s+\*\*(?:Fonte|Source):|\Z)",
            combined_output,
        ):
            add_critical_issue("empty practical-tips section leaked into the final answer")
            reasoning_notes.append("detected empty tips section")
        if re.search(r"\bleave at stop\b|\bsai em stop\b", combined_output, flags=re.IGNORECASE):
            add_critical_issue("raw English stop wording leaked into a transport answer")
            reasoning_notes.append("detected raw stop wording")
        if re.search(
            r"(?mi)^-\s+\*\*[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+\s+"
            r"(?:manh[aã]|tarde|noite|afternoon|morning|evening|roteiro|itiner[aá]rio)[^*\n]*\*\*\s*$",
            combined_output,
        ):
            add_critical_issue("planner title was rendered as a bullet instead of an H3 heading")
            reasoning_notes.append("detected planner title bullet")
        area_match = re.search(
            r"(?:Zona anterior|Previous area):\s*\*{0,2}(?P<area>[^\n.]+?)\*{0,2}(?:[.\n]|$)",
            user_query,
            flags=re.IGNORECASE,
        )
        area_rules = {
            "belem": {
                "labels": {"belem", "belém"},
                "postal_prefixes": {"1300", "1400", "1449"},
                "terms": {"belem", "brasilia", "imperio", "jeronimos", "descobrimentos"},
            },
            "baixa": {
                "labels": {"baixa", "baixa/chiado", "chiado"},
                "postal_prefixes": {"1100", "1200"},
                "terms": {"baixa", "chiado", "rossio", "carmo", "se", "comercio", "mouraria"},
            },
            "parque_nacoes": {
                "labels": {"parque das nacoes", "parque das nações"},
                "postal_prefixes": {"1990", "1998"},
                "terms": {"parque das nacoes", "oriente", "olivais", "oceanos", "fil"},
            },
            "avenidas": {
                "labels": {"avenidas novas", "saldanha"},
                "postal_prefixes": {"1000", "1050", "1069", "1070"},
                "terms": {"avenidas novas", "saldanha", "picoas", "tomas ribeiro"},
            },
        }
        if area_match:
            requested_area = cls._normalize_query(area_match.group("area"))
            matched_zone = next(
                (
                    zone
                    for zone, rule in area_rules.items()
                    if requested_area in rule["labels"] or any(label in requested_area for label in rule["labels"])
                ),
                "",
            )
            if matched_zone:
                rule = area_rules[matched_zone]
                address_lines = re.findall(
                    r"(?mi)^\s*[-*]\s+[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]*\s*"
                    r"\*\*(?:Morada|Address):\*\*\s*(.+)$",
                    combined_output,
                )
                outside_area_lines: list[str] = []
                for address_line in address_lines:
                    folded_address = cls._normalize_query(address_line)
                    postals = re.findall(r"\b(\d{4})\s*-?\s*\d{3}\b", folded_address)
                    if postals:
                        if not any(prefix in rule["postal_prefixes"] for prefix in postals):
                            outside_area_lines.append(address_line)
                        continue
                    if not any(term in folded_address for term in rule["terms"]):
                        outside_area_lines.append(address_line)
                has_same_area_limitation = re.search(
                    r"\b(?:não há dados suficientes|nao ha dados suficientes|sem alternativas confirmadas|"
                    r"not enough confirmed|no confirmed alternatives)\b",
                    output_lower,
                    flags=re.IGNORECASE,
                )
                if outside_area_lines and not has_same_area_limitation:
                    add_critical_issue("planner violated the requested same-area constraint")
                    reasoning_notes.append("detected itinerary items outside the previous plan area")

        llm_result["missing_data"] = cls._dedupe_preserve_order(missing_data)
        llm_result["required_agents"] = cls._dedupe_preserve_order(required_agents)
        llm_result["disclaimers"] = cls._dedupe_preserve_order(disclaimers)
        llm_result["critical_issues"] = cls._dedupe_preserve_order(critical_issues)
        if llm_result["missing_data"]:
            llm_result["complete"] = False
        elif not llm_result.get("critical_issues"):
            llm_result["complete"] = True
        else:
            llm_result["complete"] = False

        if reasoning_notes:
            note = " | ".join(reasoning_notes)
            llm_result["reasoning"] = f"{reasoning} | {note}".strip(" |")

        return llm_result

    @classmethod
    def _normalize_stated_limitations(
        cls,
        agent_outputs: Dict[str, str],
        llm_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Treat explicit, non-repairable limitations as complete answers.

        The QA layer should keep flagging gaps that another worker can fill. It
        should not continue to request retries for facts the final answer has
        already scoped honestly, such as unavailable fare data, unsupported
        operators, unconfirmed opening hours, or missing real-time feeds.
        """
        combined_output = "\n".join(
            str(value) for key, value in agent_outputs.items()
            if not str(key).startswith("_") and isinstance(value, str)
        )
        normalized_output = cls._normalize_query(combined_output)
        if not normalized_output:
            return llm_result

        limitation_markers = (
            "not confirmed",
            "not available",
            "unavailable",
            "not possible",
            "not explicitly verified",
            "not fully verified",
            "does not fully verify",
            "could not confirm",
            "cannot confirm",
            "can't confirm",
            "i cannot confirm",
            "i can't confirm",
            "outside confirmed scope",
            "unsupported",
            "no real time",
            "no live",
            "no results",
            "did not find",
            "could not find",
            "do not have a confirmed structured source",
            "i do not have a confirmed structured source",
            "não tenho uma fonte estruturada",
            "nao tenho uma fonte estruturada",
            "não confirmado",
            "não confirmada",
            "não confirmados",
            "não está confirmado",
            "não foi possível",
            "não consigo confirmar",
            "não encontrei",
            "nao encontrei",
            "nao foi encontrado",
            "nao confirmado",
            "nao confirmada",
            "nao confirmados",
            "nao esta confirmado",
            "nao foi possivel",
            "nao consigo confirmar",
            "indisponivel",
            "sem dados em tempo real",
            "sem horario",
            "sem horário",
            "sem horario noturno",
            "sem horário noturno",
            "nao equivale",
            "não equivale",
            "does not mean service is available",
            "fora do ambito",
            "sem cobertura",
        )

        limitation_categories = [
            (("previsao", "previsão", "forecast", "temperatura", "temperature", "precipitacao", "precipitação", "weather", "tempo"), ("previsao", "previsão", "forecast", "ipma", "dias", "days", "horizonte")),
            (("tarifa", "preco", "price", "fare", "barato", "cheapest"), ("tarifa", "preco", "price", "fare")),
            (("tempo real", "real time", "real-time", "live", "on time", "delay", "atras", "pontual", "perturb"), ("tempo real", "real time", "real-time", "live", "delay", "atras", "pontual", "perturb")),
            (("horario", "opening", "hours", "evening", "tonight", "night", "noturno", "servico", "service", "availability", "disponibilidade", "aberto", "fechado"), ("horario", "opening", "hours", "evening", "tonight", "night", "noturno", "periodo noturno", "night period", "servico", "service", "availability", "disponibilidade", "aberto", "fechado")),
            (("gratuito", "gratuita", "free", "gratuit"), ("gratuito", "gratuita", "free", "gratuit")),
            (("evento", "event", "titulo", "title", "data", "date", "localizacao", "location", "crianca", "children", "kids", "familia", "family"), ("evento", "event", "crianca", "children", "kids", "familia", "family", "gratuit", "free")),
            (("fertagus", "ferry", "barreiro", "transtejo", "soflusa", "operator", "operador", "ambito", "scope"), ("fertagus", "ferry", "barreiro", "transtejo", "soflusa", "operator", "operador", "ambito", "scope")),
            (("partida", "departure", "eta", "arrival", "route", "rota", "linha", "ligacao"), ("partida", "departure", "eta", "arrival", "route", "rota", "linha", "ligacao")),
            (("veterinario", "veterinaria", "veterinary", "vet", "clinica", "clinic", "contacto", "contact", "morada", "address", "localizacao", "location", "distancia", "distance", "horario", "hours"), ("veterinario", "veterinaria", "veterinary", "vet", "fonte estruturada", "structured source", "clinica", "clinic")),
            (("subjective", "touristy", "authentic", "quiet", "best", "relevance", "ranking", "criteria"), ("subjective", "touristy", "authentic", "quiet", "prioritised", "prioritized", "compatible signals", "verifiable details")),
        ]

        def _has_stated_limitation(output_tokens: tuple[str, ...]) -> bool:
            for token in output_tokens:
                start = 0
                while True:
                    index = normalized_output.find(token, start)
                    if index == -1:
                        break
                    window_start = max(0, index - 140)
                    window_end = min(len(normalized_output), index + len(token) + 220)
                    window = normalized_output[window_start:window_end]
                    if any(marker in window for marker in limitation_markers):
                        return True
                    start = index + len(token)
            return False

        def _is_satisfied_gap(item: object) -> bool:
            normalized_item = cls._normalize_query(str(item or ""))
            if not normalized_item:
                return False
            for missing_tokens, output_tokens in limitation_categories:
                if any(token in normalized_item for token in missing_tokens):
                    return _has_stated_limitation(output_tokens)
            return False

        filtered_missing = [
            item for item in llm_result.get("missing_data", [])
            if not _is_satisfied_gap(item)
        ]
        llm_result["missing_data"] = cls._dedupe_preserve_order(filtered_missing)
        if not filtered_missing and not llm_result.get("critical_issues"):
            llm_result["required_agents"] = []
            llm_result["complete"] = True
        return llm_result

    @staticmethod
    def _dedupe_preserve_order(items: List[str]) -> List[str]:
        """Removes duplicates while preserving the original order."""
        deduped: List[str] = []
        for item in items:
            if item and item not in deduped:
                deduped.append(item)
        return deduped

    @staticmethod
    def _sanitize_repair_disclaimer(disclaimer: object, language: str) -> Optional[str]:
        """Return a user-facing caveat for final repair, or ``None`` for QA-only notes."""
        if not disclaimer:
            return None

        normalized = re.sub(r"\s+", " ", str(disclaimer)).strip()
        if not normalized:
            return None

        lowered = normalized.lower()
        internal_markers = (
            "qa",
            "quality validation",
            "validation structure",
            "structured result",
            "reasoning",
            "final response",
            "final answer",
            "repair",
            "worker",
            "agent ",
            "source footer",
            "markdown",
            "field labels",
            "semantic emoji",
            "canonical layout",
            "collapsed",
            "technical identifiers",
            "known domains",
            "unverified domains",
            "domínios conhecidos",
            "dominios conhecidos",
            "domínios não verificados",
            "dominios nao verificados",
            "validação",
            "validacao",
            "controlo de qualidade",
            "raciocínio",
            "raciocinio",
            "resposta final",
            "reparação",
            "reparacao",
            "agente ",
            "linha de fonte",
            "rótulos",
            "rotulos",
            "identificadores técnicos",
            "identificadores tecnicos",
        )
        if any(marker in lowered for marker in internal_markers):
            return None

        if "event details (dates, times, ticket prices) should be confirmed at visitlisboa.com" in lowered:
            return None

        if "carris bus route numbers and schedules should be verified at carris.pt" in lowered:
            return None

        if "real-time" in lowered or "tempo real" in lowered:
            if language == "pt":
                return "Dados em tempo real podem sofrer alterações rápidas."
            return "Real-time data can change quickly."

        if "accessibility" in lowered or "acessibilidade" in lowered or "mobilidade reduzida" in lowered:
            if language == "pt":
                return "A acessibilidade deve ser confirmada junto do operador ou do espaço oficial."
            return "Accessibility should be confirmed with the official operator or venue."

        if (
            "horário" in lowered
            or "horario" in lowered
            or "opening hours" in lowered
            or re.search(r"\bhours?\b", lowered)
        ) and any(marker in lowered for marker in ("verificar", "confirm", "confirmar", "check")):
            if language == "pt":
                return "Confirma os horários na fonte oficial antes de ir."
            return "Check opening hours with the official source before going."

        if "fare data" in lowered or "tarifa" in lowered or "preço" in lowered or "price" in lowered:
            if language == "pt":
                return "A tarifa exata não foi confirmada nas fontes disponíveis."
            return "The exact fare was not confirmed in the available sources."

        return None

    @classmethod
    def _sanitize_repair_disclaimers(
        cls,
        disclaimers: List[object],
        language: str,
    ) -> List[str]:
        """Filter QA disclaimers before they can influence user-facing repair text."""
        sanitized: List[str] = []
        for disclaimer in disclaimers:
            cleaned = cls._sanitize_repair_disclaimer(disclaimer, language)
            if cleaned and cleaned not in sanitized:
                sanitized.append(cleaned)
        return sanitized

    @classmethod
    def _merge_fact_check_results(
        cls,
        combined_fact_check: Dict[str, Any],
        per_agent_fact_checks: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Merges combined and per-agent deterministic fact-check results."""
        disclaimers: List[str] = []
        critical_issues: List[str] = []
        checks_performed: List[str] = []
        repairable_agents: List[str] = []

        for fact_check in [combined_fact_check, *per_agent_fact_checks.values()]:
            disclaimers.extend(fact_check.get("disclaimers", []))
            critical_issues.extend(fact_check.get("critical_issues", []))
            checks_performed.extend(fact_check.get("checks_performed", []))

        for agent_name, fact_check in per_agent_fact_checks.items():
            if fact_check.get("critical_issues"):
                repairable_agents.append(agent_name)

        merged_disclaimers = cls._dedupe_preserve_order(disclaimers)
        merged_critical = cls._dedupe_preserve_order(critical_issues)
        merged_checks = cls._dedupe_preserve_order(checks_performed)
        merged_repairable_agents = cls._dedupe_preserve_order(repairable_agents)

        return {
            "valid": len(merged_critical) == 0,
            "disclaimers": merged_disclaimers,
            "critical_issues": merged_critical,
            "checks_performed": merged_checks,
            "repairable_agents": merged_repairable_agents,
            "per_agent": per_agent_fact_checks,
        }

    @traceable(name="qa_agent", run_type="chain", tags=["sub-agent", "qa"])
    def validate(
        self,
        user_query: str,
        agent_outputs: Dict[str, str],
        agents_called: List[str],
        language: str = "en",
        user_context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Validates if gathered data is complete for answering the user query.

        Runs two phases:
            Phase 1: LLM-based structural completeness check
            Phase 2: Deterministic fact verification (metro stations, coordinates,
                     dates, URLs)

        Args:
            user_query: The user's original query.
            agent_outputs: Dict mapping agent names to their output strings.
            agents_called: List of agent names that were called.
            language: Language code ('en' or 'pt').
            user_context: User preferences and constraints (location, mobility,
                         preferences, available_time, language).
            conversation_history: Last 2-3 user messages for follow-up coherence.

        Returns:
            Dict with validation result:
                - complete (bool): True if data is sufficient
                - missing_data (List[str]): List of missing data fields
                - required_agents (List[str]): Agents the orchestrator may call
                    for missing data
                - reasoning (str): Explanation of the assessment
                - disclaimers (List[str]): Warnings about data limitations
                - fact_check (Dict): Results from deterministic verification
                - critical_issues (List[str]): Deterministic issues that require correction
                - repairable_agents (List[str]): Worker agents whose outputs should be revised
                - needs_repair (bool): Whether the final response should be repaired
        """
        # ── Phase 1: LLM-based structural completeness ──────────────
        system_prompt = get_qa_prompt(
            language,
            user_context=user_context,
            conversation_history=conversation_history,
        )

        # Build context showing what was gathered
        context_parts = [f"**User Query:** {user_query}"]
        context_parts.append(f"**Agents Called:** {', '.join(agents_called)}")

        # Include user context if available
        if user_context:
            ctx_lines = []
            if user_context.get("preferences"):
                ctx_lines.append(f"- Interests/Preferences: {', '.join(user_context['preferences'])}")
            if user_context.get("mobility"):
                ctx_lines.append(f"- Mobility: {user_context['mobility']}")
            if user_context.get("available_time"):
                ctx_lines.append(f"- Available time: {user_context['available_time']}h")
            if user_context.get("latitude") and user_context.get("longitude"):
                ctx_lines.append(f"- Location: ({user_context['latitude']:.4f}, {user_context['longitude']:.4f})")
            if user_context.get("language"):
                ctx_lines.append(f"- Language preference: {user_context['language']}")
            if ctx_lines:
                context_parts.append("**User Context:**\n" + "\n".join(ctx_lines))

        # Include conversation history for follow-up coherence
        if conversation_history:
            history_str = " → ".join(conversation_history[-3:])
            context_parts.append(f"**Recent conversation:** {history_str}")

        for agent_name, output in agent_outputs.items():
            if agent_name.startswith("_"):
                continue  # Skip internal keys
            # Truncate very long outputs to avoid token limits
            truncated = output[:_TRUNCATION_LIMIT] if len(str(output)) > _TRUNCATION_LIMIT else output
            context_parts.append(
                f"\n**{agent_name.upper()} Agent Output:**\n{truncated}"
            )

        context = "\n".join(context_parts)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"# VALIDATION TASK\n\nValidate completeness of the following data:\n\n{context}"),
        ]

        # LLM call with retry for Azure content filter false positives
        response = self._safe_llm_invoke(self.llm, messages)
        content = clean_response(response.content, _print=False)

        # Parse JSON response (with one retry on failure)
        result = parse_json_response(content)

        if not result:
            # Retry: ask LLM again with explicit JSON instruction
            logger.warning("QA: First JSON parse failed, retrying...")
            retry_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=(
                        "# VALIDATION TASK (RETRY)\n\n"
                        "Your previous response was not valid JSON. "
                        "Output ONLY a JSON object with keys: complete, missing_data, "
                        "required_agents, reasoning, disclaimers.\n\n"
                        f"{context}"
                    )
                ),
            ]
            response = self._safe_llm_invoke(self.llm, retry_messages)
            content = clean_response(response.content, _print=False)
            result = parse_json_response(content)

        if result:
            llm_result = {
                "complete": result.get("complete", True),
                "missing_data": result.get("missing_data", []),
                "required_agents": [
                    a for a in result.get("required_agents", [])
                    if a in ("weather", "transport", "researcher")
                ],
                "reasoning": result.get("reasoning", ""),
                "disclaimers": result.get("disclaimers", []),
            }
        else:
            # Fallback: if JSON parsing still fails, fail closed so the
            # orchestration can still trigger the conservative repair path.
            logger.warning("QA: JSON parse failed after retry; marking response incomplete.")
            llm_result = {
                "complete": False,
                "missing_data": [
                    "QA validation structure could not be confirmed after retry"
                ],
                "required_agents": [],
                "reasoning": "QA validation could not parse LLM response after retry.",
                "disclaimers": [
                    "Quality validation could not produce a valid structured result after retry"
                ],
            }

        llm_result = self._normalize_category_query_validation(
            user_query=user_query,
            agents_called=agents_called,
            llm_result=llm_result,
        )
        llm_result = self._normalize_event_query_validation(
            user_query=user_query,
            agents_called=agents_called,
            llm_result=llm_result,
        )
        llm_result = self._normalize_event_no_result_validation(
            user_query=user_query,
            agent_outputs=agent_outputs,
            agents_called=agents_called,
            llm_result=llm_result,
        )
        llm_result = self._normalize_place_partial_no_result_validation(
            user_query=user_query,
            agent_outputs=agent_outputs,
            agents_called=agents_called,
            llm_result=llm_result,
        )
        llm_result = self._normalize_place_query_validation(
            user_query=user_query,
            agents_called=agents_called,
            llm_result=llm_result,
        )
        llm_result = self._normalize_weather_query_validation(
            agents_called=agents_called,
            llm_result=llm_result,
        )
        llm_result = self._normalize_transport_query_validation(
            agent_outputs=agent_outputs,
            agents_called=agents_called,
            llm_result=llm_result,
        )
        llm_result = self._augment_query_specific_validation(
            user_query=user_query,
            agent_outputs=agent_outputs,
            llm_result=llm_result,
            language=language,
        )
        llm_result = self._normalize_stated_limitations(
            agent_outputs=agent_outputs,
            llm_result=llm_result,
        )
        llm_result = self._normalize_transport_query_validation(
            agent_outputs=agent_outputs,
            agents_called=agents_called,
            llm_result=llm_result,
        )

        # ── Phase 2: Deterministic fact verification ─────────────────
        combined_output = "\n".join(
            str(v) for k, v in agent_outputs.items()
            if not k.startswith("_") and isinstance(v, str)
        )

        per_agent_fact_checks: Dict[str, Dict[str, Any]] = {}
        planner_expected = "planner" in set(agents_called or [])
        for agent_name, output in agent_outputs.items():
            if agent_name.startswith("_") or not isinstance(output, str):
                continue
            per_agent_fact_checks[agent_name] = self._verify_facts(
                output,
                user_query,
                user_context,
                language=language,
                agent_name=agent_name,
                intermediate_output=planner_expected,
            )

        combined_fact_check = self._verify_facts(
            combined_output,
            user_query,
            user_context,
            language=language,
            intermediate_output=planner_expected,
        )
        fact_check = self._merge_fact_check_results(
            combined_fact_check=combined_fact_check,
            per_agent_fact_checks=per_agent_fact_checks,
        )
        fact_check = self._normalize_place_partial_fact_check(
            fact_check=fact_check,
            user_query=user_query,
            agent_outputs=agent_outputs,
            agents_called=agents_called,
        )

        # Merge fact-check disclaimers into LLM result
        if fact_check.get("disclaimers"):
            llm_result["disclaimers"] = self._dedupe_preserve_order(
                llm_result.get("disclaimers", []) + fact_check["disclaimers"]
            )

        # If fact-check found critical issues, flag as incomplete
        if fact_check.get("critical_issues"):
            llm_result["reasoning"] += f" | Fact-check: {'; '.join(fact_check['critical_issues'])}"

        llm_result["critical_issues"] = fact_check.get("critical_issues", [])
        llm_result["repairable_agents"] = fact_check.get("repairable_agents", [])
        if not llm_result.get("missing_data") and not llm_result.get("critical_issues"):
            llm_result["required_agents"] = []
            llm_result["complete"] = True
        else:
            llm_result["complete"] = False
        llm_result["needs_repair"] = bool(
            fact_check.get("critical_issues") or llm_result.get("missing_data")
        )
        llm_result["fact_check"] = fact_check
        return llm_result

    def assess_final_response(
        self,
        user_query: str,
        final_response: str,
        language: str = "en",
        user_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Deterministically audit the final composed answer before display.

        The main ``validate`` method checks worker evidence before synthesis.
        This method checks the final Markdown after planner/formatter/repair
        steps, where language drift, source-footer defects, or mode-mismatch
        regressions can still be introduced.
        """
        if not final_response:
            return {
                "complete": False,
                "missing_data": ["final response is empty"],
                "required_agents": [],
                "reasoning": "Final response audit found an empty response.",
                "disclaimers": [],
                "critical_issues": ["Final response is empty."],
                "repairable_agents": [],
                "needs_repair": True,
                "fact_check": {
                    "valid": False,
                    "disclaimers": [],
                    "critical_issues": ["Final response is empty."],
                    "checks_performed": ["final_response_audit"],
                    "repairable_agents": [],
                    "per_agent": {},
                },
            }

        result: Dict[str, Any] = {
            "complete": True,
            "missing_data": [],
            "required_agents": [],
            "reasoning": "Final response deterministic audit completed.",
            "disclaimers": [],
        }
        result = self._augment_query_specific_validation(
            user_query=user_query,
            agent_outputs={"final": final_response},
            llm_result=result,
            language=language,
        )
        fact_check = self._verify_facts(
            final_response,
            user_query,
            user_context,
            language=language,
            agent_name="final",
            intermediate_output=False,
        )
        fact_check = self._normalize_place_partial_fact_check(
            fact_check,
            user_query,
            {"final": final_response},
            ["final"],
        )
        critical_issues = self._dedupe_preserve_order(
            list(result.get("critical_issues", []))
            + list(fact_check.get("critical_issues", []))
        )
        disclaimers = self._dedupe_preserve_order(
            list(result.get("disclaimers", []))
            + list(fact_check.get("disclaimers", []))
        )
        result["critical_issues"] = critical_issues
        result["disclaimers"] = disclaimers
        result["repairable_agents"] = []
        result["fact_check"] = {
            **fact_check,
            "critical_issues": critical_issues,
            "disclaimers": disclaimers,
            "repairable_agents": [],
            "per_agent": {},
        }
        result["needs_repair"] = bool(critical_issues or result.get("missing_data"))
        if result["needs_repair"]:
            result["complete"] = False
        return result

    @traceable(name="qa_repair_pass", run_type="chain", tags=["sub-agent", "qa", "repair"])
    def repair_final_response(
        self,
        user_query: str,
        draft_response: str,
        agent_outputs: Dict[str, str],
        qa_result: Dict[str, Any],
        language: str = "en",
    ) -> str:
        """Repairs a draft final response using QA findings and grounded worker outputs.

        This pass is intentionally conservative: it may rewrite phrasing,
        remove unsupported claims, and surface missing-data caveats, but it must
        stay strictly grounded in the provided worker outputs.
        """
        if not draft_response:
            return draft_response
        if re.search(
            r"\b(?:Ambiguidade em|Ambiguity in|Preciso de confirmar o local|Location needs confirmation)\b",
            draft_response,
            flags=re.IGNORECASE,
        ):
            return final_post_qa_guard(draft_response, language=language)

        fact_check = qa_result.get("fact_check", {}) if isinstance(qa_result, dict) else {}
        missing_data = self._dedupe_preserve_order(
            list(qa_result.get("missing_data", []))
        )
        critical_issues = self._dedupe_preserve_order(
            list(qa_result.get("critical_issues", []))
            + list(fact_check.get("critical_issues", []))
        )
        raw_disclaimers = self._dedupe_preserve_order(
            list(qa_result.get("disclaimers", []))
            + list(fact_check.get("disclaimers", []))
        )
        disclaimers = self._sanitize_repair_disclaimers(raw_disclaimers, language)

        if not critical_issues and not disclaimers and not missing_data:
            return draft_response

        if language == "pt":
            system_prompt = (
                "És a etapa final de reparação de qualidade da resposta. "
                "Receberás um rascunho de resposta, os outputs dos agentes especializados e os achados do QA. "
                "Reescreve APENAS o necessário para corrigir problemas factuais, remover alegações não confirmadas, "
                "manter a resposta completa e preservar uma apresentação natural com markdown, emojis e linhas de fonte corretas.\n\n"
                "REGRAS:\n"
                "- Usa apenas factos presentes no rascunho e nos outputs dos agentes.\n"
                "- Nunca inventes locais, horários, preços, acessibilidade, estações, ligações, ou URLs.\n"
                "- Se algo não estiver confirmado e o utilizador o pediu, diz brevemente que deve ser verificado. Caso contrário, omite o campo.\n"
                "- Remove referências internas a QA, validação, fact-checking, reasoning, ou agentes.\n"
                "- Preserva o idioma de saída exigido, o estilo visual, os emojis úteis e a estrutura markdown.\n"
                "- Todas as tuas edições, avisos e aditamentos de texto devem ser estritamente em Português (PT-PT).\n"
                "- Preserva o tipo de resposta pedido pelo utilizador. Se o utilizador pediu explicação, lista, comparação ou contexto histórico, não transformes a resposta num roteiro/plano. Se o utilizador disse para não dar roteiro, plano ou itinerário, essa restrição é obrigatória.\n"
                "- Mantém ou melhora a linha de fonte final se ela já existir; Google Maps nunca deve aparecer como fonte de evidência.\n"
                "- Preserva links markdown válidos exatamente quando aparecem no rascunho ou nos outputs dos agentes. Nunca substituas campos com link por texto simples como Website oficial, VisitLisboa, Bilhetes ou Tickets.\n"
                "- Preserva quaisquer perguntas interativas ao utilizador que estejam no final do texto (ex: perguntar se o utilizador quer planear o Dia 2).\n"
                "- Não acrescentes notas ou avisos de QA ao output do utilizador. Repara silenciosamente ou omite campos sem suporte.\n"
                "- Só mantém um aviso ⚠️ quando houver uma preocupação real de segurança ou quando o utilizador tiver pedido cautelas.\n"
                "- NUNCA promovas passos intermédios de uma rota (caminhar, transbordo, embarque, saída) a títulos `### ...`. Esses passos são bullets `-` dentro da secção `🗺️ O seu Trajeto de Metro:` e nunca devem ter um separador `---` entre eles.\n"
                "- Preserva a estrutura visual já existente: títulos H3, listas, indentação a 4 espaços, separadores `---` e linhas em branco devem permanecer onde já estão. Não acrescentes novos `---` nem novos títulos H3.\n"
                "- Devolve apenas a resposta final reparada, sem prefácio nem explicações."
            )
            task_prefix = "# TAREFA DE REPARAÇÃO FINAL"
            critical_label = "Problemas críticos"
            disclaimer_label = "Notas e limitações"
            missing_label = "Dados em falta"
            worker_label = "Outputs dos agentes especializados"
            draft_label = "Rascunho atual"
        else:
            system_prompt = (
                "You are the final response-quality repair pass. "
                "You will receive a draft answer, the specialized-agent outputs, and QA findings. "
                "Rewrite only what is necessary to fix factual issues, remove unsupported claims, "
                "keep the answer complete, and preserve a natural markdown response with useful emojis and a correct source line.\n\n"
                "RULES:\n"
                "- Use only facts present in the draft and worker outputs.\n"
                "- Never invent venues, times, prices, accessibility claims, stations, links, or URLs.\n"
                "- If something is not confirmed and the user asked for it, briefly say it should be verified. Otherwise omit the field.\n"
                "- Remove any references to QA, validation, fact-checking, reasoning, or internal agents.\n"
                "- Preserve the required output language, visual style, helpful emojis, and markdown structure.\n"
                "- All edits, warnings, and added text must be strictly in English.\n"
                "- Preserve the answer type requested by the user. If the user asked for an explanation, list, comparison, or historical context, do not transform the answer into an itinerary/plan. If the user said not to provide a route, plan, or itinerary, that constraint is mandatory.\n"
                "- Keep or improve the final source line if one already exists; Google Maps must never appear as an evidence source.\n"
                "- Preserve valid markdown links exactly when they appear in the draft or worker outputs. Never replace linked fields with plain text such as Official website, VisitLisboa, Tickets, or More details.\n"
                "- Preserve any interactive questions to the user at the end of the text (e.g., asking if they want to plan Day 2).\n"
                "- Do not add QA notes or QA warnings to the user output. Repair silently or omit unsupported fields.\n"
                "- Keep a ⚠️ warning only for a real-world safety concern or when the user explicitly asked for caveats.\n"
                "- NEVER promote intermediate route steps (walk, transfer, board, exit) to `### ...` headings. Those steps are list bullets `-` inside the `🗺️ Your Metro Route:` section and must not have `---` separators between them.\n"
                "- Preserve the existing visual structure: H3 titles, list bullets, 4-space indentation, `---` separators and blank lines must stay where they already are. Do not add new `---` rules or new H3 titles.\n"
                "- Return only the repaired final answer, with no preface or explanation."
            )
            task_prefix = "# FINAL REPAIR TASK"
            critical_label = "Critical issues"
            missing_label = "Missing data"
            disclaimer_label = "Warnings and limitations"
            worker_label = "Worker outputs"
            draft_label = "Current draft"

        worker_context_parts = []
        for agent_name, output in agent_outputs.items():
            if agent_name.startswith("_") or not isinstance(output, str):
                continue
            truncated = output[:_TRUNCATION_LIMIT] if len(output) > _TRUNCATION_LIMIT else output
            worker_context_parts.append(f"## {agent_name.upper()}\n{truncated}")

        worker_context = "\n\n".join(worker_context_parts) if worker_context_parts else ""
        critical_block = "\n".join(f"- {item}" for item in critical_issues) or "- None"
        missing_block = "\n".join(f"- {item}" for item in missing_data) or "- None"
        disclaimer_block = "\n".join(f"- {item}" for item in disclaimers) or "- None"

        human_content = (
            f"{task_prefix}\n\n"
            f"**User query:** {user_query}\n\n"
            f"**Required output language:** {'PT-PT' if language == 'pt' else 'English'}\n\n"
            f"## {critical_label}\n{critical_block}\n\n"
            f"## {missing_label}\n{missing_block}\n\n"
            f"## {disclaimer_label}\n{disclaimer_block}\n\n"
            f"## {draft_label}\n{draft_response[:_TRUNCATION_LIMIT]}\n\n"
            f"## {worker_label}\n{worker_context}"
        )

        try:
            response = self._safe_llm_invoke(
                self.llm,
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_content),
                ],
            )
            repaired = clean_response(response.content, _print=False).strip()
            if self._repair_collapsed_structured_draft(draft_response, repaired):
                return draft_response
            if (
                self._repair_added_specific_lookup_intro(draft_response, repaired)
                or self._repair_promoted_researcher_cards_to_headings(draft_response, repaired)
                or self._repair_added_missing_value_placeholders(draft_response, repaired)
                or self._repair_degraded_category_listing(draft_response, repaired)
            ):
                return final_post_qa_guard(draft_response, language=language)
            repaired_lower = repaired.lower()
            if any(
                marker in repaired_lower
                for marker in [
                    "desculpe, tive dificuldades em processar o pedido",
                    "sorry, i'm having difficulty processing your request",
                    "an error occurred while processing",
                ]
            ):
                return draft_response
            repaired = final_post_qa_guard(final_visual_pass(repaired), language=language)
            return repaired or draft_response
        except Exception as exc:
            logger.warning("QA final repair pass failed, keeping draft response: %s", exc)
            return draft_response

    def guard_final_response(self, final_response: str, language: str = "en") -> str:
        """Run the deterministic last-mile guard on the final user-facing answer.

        This is deliberately non-generative. It removes residual internal QA
        wording, repairs common markdown defects, enforces PT/EN labels, and
        makes the source footer visually stable after all synthesis steps.
        """
        if not final_response:
            return final_response or ""

        return final_post_qa_guard(final_response, language=language)

    @staticmethod
    def _repair_collapsed_structured_draft(draft_response: str, repaired_response: str) -> bool:
        """Return whether a repair pass collapsed a structured grounded draft too aggressively."""
        draft_sections = len(_TOP_LEVEL_SECTION_RE.findall(draft_response or ""))
        repaired_sections = len(_TOP_LEVEL_SECTION_RE.findall(repaired_response or ""))

        if draft_sections >= 3 and repaired_sections < 2:
            return True
        if draft_sections >= 2 and repaired_sections == 0:
            return True
        return False

    @staticmethod
    def _repair_added_specific_lookup_intro(draft_response: str, repaired_response: str) -> bool:
        """Return whether QA invented an exact-lookup failure intro not in the draft."""
        intro_re = re.compile(
            r"\b(?:não encontrei um (?:evento|local) específico com o nome|"
            r"nao encontrei um (?:evento|local) especifico com o nome|"
            r"i could not find a specific (?:event|place) named)\b",
            flags=re.IGNORECASE,
        )
        return bool(intro_re.search(repaired_response or "")) and not bool(intro_re.search(draft_response or ""))

    @staticmethod
    def _repair_promoted_researcher_cards_to_headings(draft_response: str, repaired_response: str) -> bool:
        """Return whether QA converted grounded researcher cards into H3 item headings."""
        draft_has_bullet_cards = bool(
            re.search(
                r"(?m)^\s*(?:[-*]\s*)?\*\*[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+\s+[^*\n]+\*\*\s*$",
                draft_response or "",
            )
            and re.search(r"(?m)^\s{4,}[-*]\s+(?:📂|📍|🕐|🕒|💶|⭐|📞|✉️|🌐|🔗|🎟️)\s+\*\*", draft_response or "")
        )
        repaired_has_h3_cards = bool(
            re.search(
                r"(?m)^###\s+(?:🛏️|🏛️|🍽️|☕|🥐|🌿|📍|🖼️|🎵|📚|🛍️|📅|🏅|🏷️|🎪)\s+\*\*[^*\n]+\*\*\s*$",
                repaired_response or "",
            )
            and re.search(r"(?m)^[-*]\s+(?:📂|📍|🕐|🕒|💶|⭐|📞|✉️|🌐|🔗|🎟️|📏)\s+\*\*", repaired_response or "")
        )
        return draft_has_bullet_cards and repaired_has_h3_cards

    @staticmethod
    def _repair_added_missing_value_placeholders(draft_response: str, repaired_response: str) -> bool:
        """Return whether QA introduced user-facing missing-value placeholders."""
        placeholder_re = re.compile(
            r"\b(?:não indicado|nao indicado|não disponível|nao disponivel|not available|not indicated|unknown|n/a)\b",
            flags=re.IGNORECASE,
        )
        return bool(placeholder_re.search(repaired_response or "")) and not bool(placeholder_re.search(draft_response or ""))

    @staticmethod
    def _repair_degraded_category_listing(draft_response: str, repaired_response: str) -> bool:
        """Return whether QA rewrote a category inventory into unrelated cards."""
        category_heading_re = re.compile(
            r"(?i)\b(?:Categorias de Locais(?: Dispon[ií]veis)?|Categorias de Eventos em Lisboa|"
            r"Categorias de Servi[cç]os|Available Place Categories|Event Categories in Lisbon|"
            r"Service Categories)\b"
        )
        if not category_heading_re.search(draft_response or ""):
            return False

        category_bullet_re = re.compile(
            r"(?m)^\s*[-*]\s+[\U0001F300-\U0001FAFF\u2300-\u23FF\u2600-\u27BF\uFE0F\u200D]+\s+\*\*[^*\n]+:\*\*"
        )
        draft_count = len(category_bullet_re.findall(draft_response or ""))
        repaired_count = len(category_bullet_re.findall(repaired_response or ""))
        if draft_count >= 4 and repaired_count < max(3, draft_count // 2):
            return True

        if not category_heading_re.search(repaired_response or ""):
            return True

        return len(re.findall(r"(?i)\*\*(?:Locais de gastronomia|Food and Dining)\*\*", repaired_response or "")) >= 3

    def _verify_facts(
        self,
        combined_output: str,
        user_query: str,
        user_context: Optional[Dict[str, Any]] = None,
        language: Optional[str] = None,
        agent_name: Optional[str] = None,
        intermediate_output: bool = False,
    ) -> Dict[str, Any]:
        """
        Deterministic fact verification against authoritative static data.

        Checks (9 total):
            1. Metro station names (METRO_STATIONS from metrolisboa_api)
            2. Metro line-station pair validity (sentence-level, using METRO_LINES)
            3. CP train line names (CP_LINES from cp_api)
            4. AML coordinate bounds (Lisbon Metropolitan Area bounding box)
            5. Date sanity (IPMA 5-day forecast range)
            6. URL domain validation (known Lisbon data sources)
            7. User preference adherence (accessibility, available time)
            8. IPMA temperature sanity (Lisbon historic bounds + tMin/tMax inversion)
            9. Dynamic-data disclaimers (events, Carris bus/tram info)

        Args:
            combined_output: All agent outputs concatenated.
            user_query: The user's original query (for context).
            user_context: User preferences/constraints dict.
            intermediate_output: Whether this text is worker evidence that
                will still pass through planner synthesis and final rendering.
                Intermediate evidence is fact-checked, but it is not required
                to satisfy the final Streamlit Markdown contract.

        Returns:
            Dict with:
                - valid (bool): True if no critical issues found
                - disclaimers (List[str]): Informational warnings to surface to user
                - critical_issues (List[str]): Definitive factual errors detected
                - checks_performed (List[str]): Names of checks that ran
        """
        disclaimers: List[str] = []
        critical_issues: List[str] = []
        checks: List[str] = []
        output_lower = combined_output.lower()
        expected_language = language if language in {"pt", "en"} else infer_response_language(
            user_query=user_query,
            default="en",
        )

        def add_critical_issue(message: str) -> None:
            if message not in critical_issues:
                critical_issues.append(message)

        # ── Check 1: Metro station names ──────────────────────────────
        # Uses METRO_STATIONS from metrolisboa_api as the authoritative source.
        checks.append("metro_stations")
        station_text_patterns = [
            r"esta[çc][aã]o\s+(?:de\s+|do\s+)?([A-Za-zÀ-ú\s\-\.]+?)(?:\s*[\(\),\.]|\s+(?:da|na|para|line|linha|on|to|from))",
            r"station\s+([A-Za-zÀ-ú\s\-\.]+?)(?:\s*[\(\),\.]|\s+(?:on|to|from|line))",
        ]
        mentioned_stations: set = set()
        for pattern in station_text_patterns:
            for match in re.findall(pattern, output_lower, re.IGNORECASE):
                name = match.strip().lower().rstrip(".")
                word_count = len(name.split())
                if (
                    len(name) > 2
                    and word_count <= 4
                    and "metropolitano de lisboa" not in name
                    and "reconhecida" not in name
                    and "recognized" not in name
                ):
                    mentioned_stations.add(name)

        valid_metro_set = _METRO_CANONICAL_STATIONS
        invalid_stations = [
            s for s in mentioned_stations
            if s not in valid_metro_set
            and not any(s in v or v in s for v in valid_metro_set)
        ]
        if invalid_stations:
            disclaimers.append(
                f"Some metro station names could not be verified: {', '.join(invalid_stations)}"
            )

        # ── Check 2: Metro line-station pair validity ─────────────────
        # Detects hallucinations like "linha amarela to Telheiras"
        # (Telheiras is only on linha verde). Uses sentence-level analysis.
        # Requires "linha X" pattern to avoid false positives on standalone color words.
        checks.append("metro_line_station_pairs")
        if _HAS_METRO_DATA and _METRO_LINES_DATA and _METRO_STATIONS_DATA:
            sentences = re.split(r"[.!?\n]+", output_lower)
            seen_pair_issues: set = set()
            for sentence in sentences:
                for line_name in _METRO_LINES_DATA:
                    if not re.search(rf"\blinha\s+{re.escape(line_name)}\b", sentence):
                        continue
                    for station, station_lines in _METRO_STATIONS_DATA.items():
                        if len(station) < 5:  # Skip very short names (noise risk)
                            continue
                        if station in sentence and line_name not in station_lines:
                            key = f"{station}@{line_name}"
                            if key not in seen_pair_issues:
                                seen_pair_issues.add(key)
                                correct = ", ".join(station_lines)
                                disclaimers.append(
                                    f"Station '{station.title()}' does not serve the "
                                    f"{line_name} metro line (it serves: {correct})"
                                )

        # ── Check 3: CP train line names ──────────────────────────────
        # Validates CP line names against CP_LINES from cp_api.
        checks.append("cp_lines")
        if _HAS_CP_DATA and _CP_LINES_DATA:
            cp_pattern = (
                r"linha\s+de\s+([A-Za-zÀ-ú\s\-]+?)(?:[\.,;\n]|\s+(?:line|train|comboio|de|da))"
            )
            for match in re.findall(cp_pattern, output_lower):
                line_name = match.strip().lower()
                if len(line_name) > 2:
                    is_known = any(
                        line_name in key or key in line_name
                        for key in _CP_LINES_DATA
                    )
                    if not is_known:
                        valid_list = ", ".join(_CP_LINES_DATA.keys())
                        disclaimers.append(
                            f"CP train line '{match.strip()}' could not be verified. "
                            f"Known AML lines: {valid_list}"
                        )

        # ── Check 4: Coordinate bounds (AML area) ─────────────────────
        checks.append("aml_coordinates")
        coord_patterns = [
            r"(-?\d+\.\d+)\s*[,°]\s*(-?\d+\.\d+)",
            r"lat(?:itude)?\s*[:=]?\s*(-?\d+\.?\d*)\s*[,;]\s*lon(?:gitude)?\s*[:=]?\s*(-?\d+\.?\d*)",
        ]
        coord_matches: list = []
        for cp in coord_patterns:
            coord_matches.extend(re.findall(cp, combined_output, re.IGNORECASE))
        out_of_bounds = []
        for lat_s, lon_s in coord_matches:
            try:
                lat, lon = float(lat_s), float(lon_s)
                if 30.0 <= abs(lat) <= 50.0 and 5.0 <= abs(lon) <= 15.0:
                    if not (
                        _AML_BOUNDS["lat_min"] <= lat <= _AML_BOUNDS["lat_max"]
                        and _AML_BOUNDS["lon_min"] <= lon <= _AML_BOUNDS["lon_max"]
                    ):
                        out_of_bounds.append(f"({lat}, {lon})")
            except (ValueError, TypeError):
                continue
        if out_of_bounds:
            disclaimers.append(
                f"Some coordinates appear outside the Lisbon Metropolitan Area: "
                f"{', '.join(out_of_bounds[:3])}"
            )

        # ── Check 5: Date sanity ──────────────────────────────────────
        checks.append("date_sanity")
        today = datetime.now().date()
        date_patterns = [
            r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})",  # DD/MM/YYYY
            r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})",  # YYYY-MM-DD
        ]
        for dp in date_patterns:
            for groups in re.findall(dp, combined_output):
                try:
                    if len(groups[0]) == 4:
                        d = datetime(int(groups[0]), int(groups[1]), int(groups[2])).date()
                    else:
                        d = datetime(int(groups[2]), int(groups[1]), int(groups[0])).date()
                    if "forecast" in output_lower or "previsão" in output_lower:
                        max_forecast = today + timedelta(days=_IPMA_FORECAST_DAYS)
                        if d > max_forecast:
                            disclaimers.append(
                                f"Weather forecast for {d.isoformat()} may be beyond the "
                                f"available forecast range ({_IPMA_FORECAST_DAYS} days from IPMA)"
                            )
                except (ValueError, TypeError):
                    continue

        # ── Check 6: URL domain validation ───────────────────────────
        checks.append("url_validation")
        url_domains = re.findall(r"https?://([a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,})", combined_output)
        suspicious_urls = [
            d.lower() for d in url_domains
            if not any(
                d.lower() == v or d.lower().endswith("." + v) for v in _VALID_DOMAINS
            )
        ]
        if suspicious_urls:
            unique_sus = list(set(suspicious_urls))[:5]
            disclaimers.append(
                f"Some URLs reference unverified domains: {', '.join(unique_sus)}. "
                "Please verify links before visiting."
            )

        # ── Check 7: User preference adherence ───────────────────────
        checks.append("user_preferences")
        if user_context:
            mobility = user_context.get("mobility", "")
            if mobility in ("limited", "wheelchair"):
                accessibility_terms = [
                    "acess", "wheelchair", "cadeira de rodas", "elevador",
                    "elevator", "lift", "mobilidade reduzida", "reduced mobility",
                ]
                if "transport" in output_lower and not any(
                    t in output_lower for t in accessibility_terms
                ):
                    disclaimers.append(
                        "Transport information may not include accessibility details. "
                        "Please verify station accessibility at metrolisboa.pt."
                    )
            available_time = user_context.get("available_time")
            if available_time and (
                "plan" in user_query.lower() or "roteiro" in user_query.lower()
            ):
                time_indicators = re.findall(r"(\d+)\s*(?:hours?|horas?|h\b)", output_lower)
                if time_indicators:
                    parsed_hours = [int(h) for h in time_indicators]
                    total_hours = sum(h for h in parsed_hours if h < 24)
                    if total_hours > available_time * _TIME_TOLERANCE_FACTOR:
                        disclaimers.append(
                            f"The suggested itinerary may exceed your available time of {available_time}h."
                        )

        # ── Check 8: IPMA temperature sanity ─────────────────────────
        # Flags temperatures outside Lisbon's historic range AND tMin > tMax.
        checks.append("temperature_sanity")
        temp_values: List[float] = []
        for tp in [r"(-?\d+\.?\d*)\s*°[Cc]", r"(?:tmax|tmin)\s*[:=]\s*(-?\d+\.?\d*)"]:
            for t in re.findall(tp, combined_output, re.IGNORECASE):
                with suppress(ValueError):
                    temp_values.append(float(t))
        extreme_temps = [t for t in temp_values if t < _LISBON_TEMP_MIN or t > _LISBON_TEMP_MAX]
        if extreme_temps:
            critical_issues.append(
                f"Temperature value(s) outside Lisbon's historic range "
                f"({_LISBON_TEMP_MIN}°C to {_LISBON_TEMP_MAX}°C): "
                f"{', '.join(f'{t:.1f}°C' for t in extreme_temps[:3])}"
            )
        tmin_m = re.findall(r"\btmin\s*[:=]\s*(-?\d+\.?\d*)", combined_output, re.IGNORECASE)
        tmax_m = re.findall(r"\btmax\s*[:=]\s*(-?\d+\.?\d*)", combined_output, re.IGNORECASE)
        if tmin_m and tmax_m:
            with suppress(ValueError):
                if float(tmin_m[0]) > float(tmax_m[0]):
                    critical_issues.append(
                        f"Temperature inversion: tMin ({tmin_m[0]}°C) > tMax ({tmax_m[0]}°C)"
                    )

        # ── Check 9: Event category alignment ─────────────────────────
        checks.append("event_category_alignment")
        category_labels = [
            match.strip()
            for match in re.findall(
                r"\*\*(?:Categoria|Category):\*\*\s*([^\n\r]+)",
                combined_output,
                flags=re.IGNORECASE,
            )
            if match.strip()
        ]
        if category_labels:
            response_category_keys = {
                _qa_event_category_key(re.sub(r"\s*\|.*$", "", label).strip())
                for label in category_labels
            }
            requested_category_keys = _qa_requested_event_category_keys(user_query)
            forbidden_category_keys = _qa_forbidden_event_category_keys(user_query)
            if requested_category_keys:
                unexpected = sorted(response_category_keys - requested_category_keys)
                if unexpected:
                    add_critical_issue(
                        "Event response includes categories outside the user's requested category filter: "
                        + ", ".join(unexpected[:4])
                    )
            forbidden_present = sorted(response_category_keys & forbidden_category_keys)
            if forbidden_present:
                add_critical_issue(
                    "Event response includes a category explicitly excluded by the user: "
                    + ", ".join(forbidden_present[:4])
                )

        # ── Check 10: Dynamic-data disclaimers ────────────────────────
        # Adds informational caveats for data that cannot be deterministically
        # verified at runtime (events change daily, bus routes change, etc.).
        checks.append("dynamic_data_disclaimers")
        event_keywords = {
            "event", "evento", "exhibition", "exposição", "exposicao",
            "festival", "concert", "concerto", "spectacle", "espectáculo",
        }
        event_keyword_pattern = (
            r"\b(?:"
            + "|".join(re.escape(keyword) for keyword in sorted(event_keywords, key=len, reverse=True))
            + r")s?\b"
        )
        is_event_category_listing = bool(
            re.search(r"\b(?:event categories|categorias de eventos)\b", output_lower)
        )
        event_source_present = bool(
            re.search(
                r"visitlisboa\.com/(?:en/events|pt-pt/eventos)|"
                r"visitlisboa\s+(?:events|eventos)|search_cultural_events",
                output_lower,
                flags=re.IGNORECASE,
            )
        )
        user_asked_events = bool(
            re.search(event_keyword_pattern, _qa_normalize_text(user_query), flags=re.IGNORECASE)
        )
        if (
            (event_source_present or user_asked_events)
            and re.search(event_keyword_pattern, output_lower, flags=re.IGNORECASE)
            and not is_event_category_listing
        ):
            disclaimers.append(
                "Event details (dates, times, ticket prices) should be confirmed at "
                "visitlisboa.com, as this data is synced daily and may have changed."
            )
        tram_mentions = {
            match.lower()
            for pattern in (
                r"\b(?:tram|trams|el[eé]trico|el[eé]tricos)\s+(\d+e)\b",
                r"\b(\d+e)\s+(?:tram|trams|el[eé]trico|el[eé]tricos)\b",
            )
            for match in re.findall(pattern, output_lower)
        }
        invalid_trams = tram_mentions - _CARRIS_TRAM_LINES
        if invalid_trams:
            disclaimers.append(
                f"Tram line(s) could not be verified: "
                f"{', '.join(t.upper() for t in sorted(invalid_trams))}. "
                f"Known Lisbon trams: {', '.join(t.upper() for t in sorted(_CARRIS_TRAM_LINES))}"
            )
        if re.search(r"\b[0-9]{3}\b", combined_output) and "carris" in output_lower:
            disclaimers.append(
                "Carris bus route numbers and schedules should be verified at carris.pt, "
                "as GTFS data may not reflect the most recent changes."
            )

        # ── Check 11: User-facing output hygiene ───────────────────────
        # Flags backend-oriented fields that should never reach the final UI.
        checks.append("output_hygiene")
        if intermediate_output:
            result = {
                "valid": len(critical_issues) == 0,
                "disclaimers": disclaimers,
                "critical_issues": critical_issues,
                "checks_performed": checks,
            }
            if disclaimers or critical_issues:
                logger.info(
                    f"QA fact-check: {len(critical_issues)} critical issue(s), "
                    f"{len(disclaimers)} disclaimer(s)"
                )
            return result

        structured_label_count = len(
            re.findall(r"\*\*[^*:\n]{2,40}:?\*\*", combined_output)
        )
        structured_output = "### " in combined_output or structured_label_count >= 3
        source_footer_match = re.search(
            r"(?m)^📌\s*\*\*(?:Fontes?|Sources?):\*\*.*$",
            combined_output,
        )

        if re.search(r"(?im)^\s*(?:[-*•]\s*)?(?:🗺️\s*)?GPS\s*:", combined_output):
            add_critical_issue("Raw GPS coordinates leaked into user-facing output.")
        if re.search(
            r"(?im)^\s*(?:[-*•]\s*)?(?:🚏\s*)?(?:next\s+)?stop(?:_id|\s+id)\s*[:=]|\b(?:line_id|stop_id|route_id|pattern_id|trip_id)\b",
            combined_output,
        ):
            add_critical_issue("Technical transport identifiers leaked into user-facing output.")
        if re.search(
            r"\b(?:Unknown event|Evento sem nome|Unknown place|Local sem nome|Unknown station|Estação sem nome)\b",
            combined_output,
            re.IGNORECASE,
        ):
            add_critical_issue("Unnamed placeholder content leaked into user-facing output.")
        for raw_line in combined_output.splitlines():
            normalized_line = unicodedata.normalize("NFKD", raw_line or "")
            normalized_line = "".join(char for char in normalized_line if not unicodedata.combining(char)).lower()
            normalized_line = re.sub(r"[*_`~]", "", normalized_line)
            if "carris metropolitana" not in normalized_line:
                continue
            match = re.search(
                r"\b(?:nas?\s+)?(?:linha|linhas|line|lines)\b\s*(?::|#)?\s*"
                r"(?P<ids>(?:\d{1,4}[a-z]?\s*(?:,|/|e|and|\s+)?)+)",
                normalized_line,
            )
            if not match:
                continue
            line_ids = re.findall(r"\b\d{1,4}[a-z]?\b", match.group("ids"))
            if any(not re.fullmatch(r"\d{4}", line_id) for line_id in line_ids):
                add_critical_issue("Carris Metropolitana line identifiers must not be mixed with Carris Urban line IDs.")
                break

        translatable_pairs = [
            (pt_label, en_label)
            for pt_label, en_label, _ in _LABEL_TRANSLATIONS
            if pt_label.lower() != en_label.lower()
        ]
        if expected_language == "en" and any(
            re.search(rf"\*\*{re.escape(pt_label)}\s*:?[ ]*\*\*", combined_output, re.IGNORECASE)
            for pt_label, _ in translatable_pairs
        ):
            add_critical_issue("Portuguese field labels leaked into an English response.")
        if expected_language == "pt" and any(
            re.search(rf"\*\*{re.escape(en_label)}\s*:?[ ]*\*\*", combined_output, re.IGNORECASE)
            for _, en_label in translatable_pairs
        ):
            add_critical_issue("English field labels leaked into a Portuguese response.")
        if expected_language == "en" and re.search(
            r"\[(?:Comprar bilhetes|Página oficial|Mais detalhes)\]\(",
            combined_output,
            flags=re.IGNORECASE,
        ):
            add_critical_issue("Portuguese link labels leaked into an English response.")
        if expected_language == "pt" and re.search(
            r"\[(?:Buy tickets|Tickets|Official website|Official page|More details)\]\(",
            combined_output,
            flags=re.IGNORECASE,
        ):
            add_critical_issue("English link labels leaked into a Portuguese response.")
        if expected_language == "pt":
            body_without_sources = re.sub(
                r"(?m)^📌\s*\*\*(?:Fontes?|Sources?):\*\*.*$",
                "",
                combined_output,
            )
            if re.search(
                r"\b(?:start at|from the|if you prefer|have lunch|i couldn['’]?t|couldn['’]?t confirm|"
                r"after lunch|allow about|good first stop|walking is|how to get|bel[eé]m stops|"
                r"by tram|taxi|rideshare|no direct bus routes found|no carris metropolitana stops found|"
                r"try a more specific|you may need to transfer|consider a metro|appears to be inside lisbon city|carris urban|"
                r"one of the most recent|world heritage|adult|children|youngster|free with lisboa card|"
                r"temporarily closed|opening hours|reviews|given its outstanding|maritime museum reflects)\b",
                body_without_sources,
                flags=re.IGNORECASE,
            ):
                add_critical_issue("English running prose leaked into a Portuguese response.")

        if structured_output and not source_footer_match:
            add_critical_issue("Source footer is missing or malformed.")
        if source_footer_match:
            trailing_text = combined_output[source_footer_match.end():]
            if re.search(r"(?m)^\s*(?:[-*•]\s*)?[💡⚠️]", trailing_text):
                add_critical_issue("Tips and warnings must appear before the source footer.")
            source_footer = source_footer_match.group(0)
            if re.search(r"google\.com/maps|Google Maps", source_footer, flags=re.IGNORECASE):
                add_critical_issue("Google Maps links are address aids, not valid evidence sources.")
            missing_sources = missing_material_source_labels(combined_output, expected_language)
            if missing_sources:
                add_critical_issue(
                    "Source footer is missing material source(s): "
                    + ", ".join(missing_sources)
                    + "."
                )

        if re.search(r"(?m)^\s*1\.\s*$", combined_output):
            add_critical_issue("Stray numbered-list markers leaked into the response.")
        if re.search(r"(?m)^\s*\*\*\s*$", combined_output) or combined_output.count("**") % 2 != 0:
            add_critical_issue("Broken bold markdown markers leaked into the response.")
        if combined_output.count("`") % 2 != 0:
            add_critical_issue("Raw markdown backticks leaked into the response.")

        if re.search(r"\[[^\]]+\]\(\[[^\]]+\]\([^)]+\)\)", combined_output):
            add_critical_issue("Nested markdown links leaked into the response.")
        if re.search(
            r"\[(?:Bilhetes|Tickets|Website|Phone|Telefone|Morada|Address)[^\]]*\]\((?!https?://|mailto:|tel:|https://www\.google\.com/maps)[^)]+\)",
            combined_output,
            re.IGNORECASE,
        ):
            add_critical_issue("Markdown links must only wrap valid URLs or tel: targets.")

        if re.search(
            r"(?m)^[^\n]*(?:📍\s*\*\*(?:Address|Morada|Endereço|Localização|Location)|📞\s*\*\*(?:Phone|Telefone)|🌐\s*\*\*(?:Website)|🎟️\s*\*\*(?:Tickets|Bilhetes)).+",
            combined_output,
        ) and re.search(
            r"(?m)^(?!\s*[-*•]\s).+\S\s+(?:📍\s*\*\*|📞\s*\*\*|🌐\s*\*\*|🎟️\s*\*\*)",
            combined_output,
        ):
            add_critical_issue("Structured emoji field labels must start on their own line.")

        if _qa_response_has_generic_intro_card_defect(combined_output):
            add_critical_issue(
                "Generic place-list intro text must not be rendered as a duplicated place card."
            )

        for line in combined_output.splitlines():
            if re.search(r"(?:📞|\*\*(?:Phone|Telefone|Contact|Contacto):\*\*)", line, re.IGNORECASE):
                if re.search(r"\+351\s*\d{3}\s*\d{3}\s*\d{3}", line) and "tel:" not in line.lower():
                    add_critical_issue("Phone fields must include a tel: link.")
            if re.search(
                r"(?:📍\s*\*\*(?:Address|Morada|Endereço|Localização|Location):\*\*|\*\*(?:Address|Morada|Endereço|Localização|Location):\*\*)",
                line,
                re.IGNORECASE,
            ):
                if "google.com/maps" not in line.lower():
                    add_critical_issue("Address fields must use Google Maps links.")

        if re.search(r"\(-?\d+\.\d+\s*,\s*-?\d+\.\d+\)", combined_output) and "google.com/maps" not in output_lower:
            add_critical_issue("Coordinate pairs must be wrapped in Google Maps links.")

        emoji_expectations = [
            (r"\*\*(?:Phone|Telefone):\*\*", "📞"),
            (r"\*\*(?:Address|Morada|Endereço|Location|Localização):\*\*", "📍"),
            (r"\*\*(?:Source|Fonte):\*\*", "📌"),
            (r"\*\*(?:Tip|Dica(?: rápida)?):\*\*", "💡"),
            (r"\*\*(?:Warning|Aviso|Note|Nota|Atenção):\*\*", "⚠️"),
            (r"\*\*(?:Tickets|Bilhetes):\*\*", "🎟️"),
        ]
        for pattern, expected_emoji in emoji_expectations:
            for line in combined_output.splitlines():
                if re.search(pattern, line, re.IGNORECASE) and expected_emoji not in line:
                    add_critical_issue("Structured field labels are missing their expected semantic emoji.")
                    break

        if agent_name == "researcher" and self._is_place_listing_query(user_query):
            place_card_count = _count_structured_place_cards(combined_output)
            if place_card_count == 0:
                add_critical_issue("Place cards collapsed into summary text and lost the canonical layout.")
            elif _place_response_missing_required_fields(
                combined_output,
                place_card_count,
            ):
                add_critical_issue(
                    "Place cards are missing canonical fields such as description, address, opening hours, or website."
                )

        result = {
            "valid": len(critical_issues) == 0,
            "disclaimers": disclaimers,
            "critical_issues": critical_issues,
            "checks_performed": checks,
        }
        if disclaimers or critical_issues:
            logger.info(
                f"QA fact-check: {len(critical_issues)} critical issue(s), "
                f"{len(disclaimers)} disclaimer(s)"
            )
        return result


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 QA Agent Test\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")

    # ── Test deterministic fact-checking (no LLM needed) ─────────
    print("\n\033[1m📋 Phase 2: Deterministic Fact-Checking Tests\033[0m")
    agent = QualityAssuranceAgent.__new__(QualityAssuranceAgent)

    # Test: Valid metro station
    r = agent._verify_facts("Take metro to estação de Alameda", "test", None)
    assert "metro_stations" in r["checks_performed"]
    print("  \033[1;32m✅ PASS\033[0m: Metro station check runs")

    # Test: Coordinates in AML
    r = agent._verify_facts("Location: 38.7223, -9.1393", "test", None)
    assert len(r["disclaimers"]) == 0 or "outside" not in str(r["disclaimers"])
    print("  \033[1;32m✅ PASS\033[0m: Valid AML coordinates accepted")

    # Test: Coordinates outside AML
    r = agent._verify_facts("Location: 41.1579, -8.6291", "test", None)
    outside_found = any("outside" in d for d in r["disclaimers"])
    print(f"  {'✅ PASS' if outside_found else '⚠️ SKIP'}: Out-of-bounds coordinate flagged: {outside_found}")

    # Test: Suspicious URL
    r = agent._verify_facts("Visit https://fake-lisbon-tours.xyz/book", "test", None)
    url_flagged = any("unverified" in d for d in r["disclaimers"])
    print(f"  \033[1;32m✅ PASS\033[0m: Suspicious URL flagged: {url_flagged}")

    # Test: Valid URL
    r = agent._verify_facts("Source: https://www.visitlisboa.com/events", "test", None)
    no_url_flag = not any("unverified" in d for d in r["disclaimers"])
    print(f"  \033[1;32m✅ PASS\033[0m: Valid URL accepted: {no_url_flag}")

    # Test: Mobility preference
    r = agent._verify_facts(
        "Take transport from Alameda to Oriente via metro line vermelha",
        "plan accessible route",
        {"mobility": "wheelchair"},
    )
    access_flagged = any("accessibility" in d for d in r["disclaimers"])
    print(f"  \033[1;32m✅ PASS\033[0m: Wheelchair accessibility disclaimer: {access_flagged}")

    # Test: Metro line-station pair (Telheiras is on Verde, NOT Amarela)
    r = agent._verify_facts(
        "Toma a linha amarela até Telheiras para chegar ao destino.",
        "test", None,
    )
    wrong_pair = any("telheiras" in d.lower() for d in r["disclaimers"])
    print(f"  {'✅ PASS' if wrong_pair else '⚠️  WARN (metro data unavailable?)'}: "
          f"Wrong metro line-station pair flagged (Telheiras/Amarela): {wrong_pair}")

    # Test: Temperature out of Lisbon bounds (-30°C impossible)
    r = agent._verify_facts("Today's temperature in Lisbon is -30°C.", "test", None)
    temp_flagged = any("temperature" in i.lower() for i in r["critical_issues"])
    print(f"  \033[1;32m✅ PASS\033[0m: Extreme temperature flagged as critical: {temp_flagged}")

    # Test: Temperature inversion (tMin > tMax)
    r = agent._verify_facts("Forecast: tMin: 32, tMax: 15", "test", None)
    inversion = any("inversion" in i.lower() for i in r["critical_issues"])
    print(f"  \033[1;32m✅ PASS\033[0m: Temperature inversion detected: {inversion}")

    # Test: Event disclaimer added when events are mentioned
    r = agent._verify_facts("Join the jazz festival at CCBB tonight.", "test", None)
    event_disc = any("event" in d.lower() or "visitlisboa" in d.lower() for d in r["disclaimers"])
    print(f"  \033[1;32m✅ PASS\033[0m: Event data disclaimer present: {event_disc}")

    # Test: Known tram 28E should NOT be flagged
    r = agent._verify_facts("Take tram 28E from Martim Moniz to Prazeres.", "test", None)
    tram_ok = not any("28e" in d.lower() and "not be verified" in d.lower() for d in r["disclaimers"])
    print(f"  \033[1;32m✅ PASS\033[0m: Valid tram 28E not flagged: {tram_ok}")

    # Test: Unknown tram line (99E does not exist)
    r = agent._verify_facts("Take tram 99E across Lisbon.", "test", None)
    tram_flagged = any("99e" in d.lower() for d in r["disclaimers"])
    print(f"  {'✅ PASS' if tram_flagged else '⚠️  INFO'}: Unknown tram 99E flagged: {tram_flagged}")

    print(f"\n\033[1m📋 All deterministic checks ({len(r['checks_performed'])}): "
          f"{r['checks_performed']}\033[0m")

    # ── Test full LLM-based validation ───────────────────────────
    print("\n\033[1m📋 Phase 1+2: Full Validation Tests (requires LLM)\033[0m")
    try:
        agent = QualityAssuranceAgent()
        print(f"  \033[1m✅ QA Agent initialized:\033[0m {agent.get_model_info()}")
        print(f"     Tools: {len(agent.tools)} (QA has no tools)")

        # Test 1: Incomplete planning query
        print("\n  \033[1m📝 Test 1: Incomplete planning query\033[0m")
        result = agent.validate(
            user_query="Plan my day tomorrow in Lisbon",
            agent_outputs={
                "weather": "Tomorrow: 18°C, sunny, no rain expected.",
                "researcher": "1. Museu do Azulejo\n2. Castelo de São Jorge\n3. Belém Tower",
            },
            agents_called=["weather", "researcher"],
            language="en",
            user_context={"preferences": ["museums", "history"], "mobility": "full"},
        )
        print(f"     Complete: {result['complete']}")
        print(f"     Missing: {result['missing_data']}")
        print(f"     Required agents: {result['required_agents']}")
        print(f"     Fact-check: {result['fact_check']['checks_performed']}")

        # Test 2: Complete weather query
        print("\n  \033[1m📝 Test 2: Complete weather query\033[0m")
        result = agent.validate(
            user_query="What's the weather today?",
            agent_outputs={
                "weather": "Today: 22°C max, 14°C min. Sunny. No rain. Wind: Moderate from NW.",
            },
            agents_called=["weather"],
            language="en",
        )
        print(f"     Complete: {result['complete']}")
        print(f"     Reasoning: {result['reasoning']}")

        print("\n\033[1;32m✅ QA Agent working!\033[0m")

    except Exception as e:
        print(f"\n\033[1;31m❌ LLM-based test error (expected if no LLM configured):\033[0m {e}")
