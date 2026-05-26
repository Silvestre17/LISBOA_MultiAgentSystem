# ==========================================================================
# Master Thesis - Researcher Agent
#   - André Filipe Gomes Silvestre, 20240502
#
#   RAG-based researcher for places, events, and local knowledge.
#   Uses semantic search over vector store.
#   Uses BaseAgent.execute_react_loop() for tool execution.
# ==========================================================================

import re
import unicodedata
import uuid
from copy import deepcopy
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from tools.visitlisboa_api import (
    _extract_specific_event_lookup_phrase,
    _extract_specific_place_lookup_phrase,
    _load_places_json,
    _score_specific_place_lookup_match,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

from agent.agents.base import BaseAgent, parse_json_response
from agent.prompts.researcher import get_researcher_prompt
from agent.utils.langsmith_tracing import traceable
from agent.state import AgentState
from agent.utils.geographic_scope import (
    extract_aml_municipality_mentions,
    normalize_scope_text,
)
from agent.utils.langgraph_compat import ToolNode
from agent.utils.response_formatter import (
    finalize_worker_response,
    infer_response_language,
    resolve_output_language,
)

# Words that start interrogative sentences — not place names when they appear at token[0].
_QUESTION_STARTER_WORDS: frozenset = frozenset({
    "is", "are", "was", "were", "do", "does", "did",
    "have", "has", "had", "can", "could", "should",
    "would", "will", "shall",
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    # PT-PT interrogative starters
    "é", "estão", "fica", "ficam", "tem", "têm", "existe", "existem",
    "qual", "quais", "quando", "onde", "como", "por",
})

# Tokens that are capitalised in English sentence position but are NOT proper place names.
_NON_PROPER_PLACE_WORDS: frozenset = frozenset({
    "is", "are", "was", "were", "do", "does", "did",
    "have", "has", "had", "can", "could", "should", "would", "will", "shall",
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "the", "a", "an", "this", "that", "these", "those", "it", "its",
    "in", "at", "on", "near", "to", "from", "of", "with", "for", "by",
    "about", "open", "closed", "free", "paid", "available", "visit",
    "wheelchair", "accessible", "accessibility", "step",
    "place", "places", "museum", "museums", "monument", "monuments",
    "restaurant", "restaurants", "hotel", "hotels", "attraction", "attractions",
    "there", "any", "some", "many", "much", "every", "all", "no", "not",
    "and", "or", "but", "so", "if", "then", "too", "also",
    "me", "you", "we", "they", "he", "she", "our", "your", "their",
    # English day and month names (capitalise but are not place proper nouns)
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mondays", "tuesdays", "wednesdays", "thursdays", "fridays", "saturdays", "sundays",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
})

_STRUCTURED_QUERY_PLAN_INTENTS: frozenset[str] = frozenset({
    "place_lookup",
    "event_lookup",
    "nearby_services",
    "dataset_search",
    "dataset_details",
    "knowledge_search",
    "unknown",
})

_RESEARCHER_FOOD_INTENT_RE = re.compile(
    r"\b(?:restaurant|restaurante|restaurants|restaurantes|food|comida|comer|cuisine|cozinha|"
    r"gastronom\w*|tradicional|traditional|almoco|almocar|almoço|almoçar|lunch|jantar|dinner|"
    r"seafood|marisco|fish|peixe|cafe|café|coffee|pastelaria|pastry|padaria|bakery|bar|bars)\b",
    re.IGNORECASE,
)

_RESEARCHER_CAFE_INTENT_RE = re.compile(
    r"\b(?:cafe|café|coffee|pastelaria|pastry|padaria|bakery)\b",
    re.IGNORECASE,
)


def _normalize_researcher_intent_text(value: str) -> str:
    """Return accent-insensitive lowercase text for intent regex matching."""
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    normalized = normalized.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"\s+", " ", normalized).strip()


_STRUCTURED_SERVICE_TYPE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "pharmacies": {
        "tool_label": "farmácias",
        "category": "saúde",
        "dataset_term": "Farmácias e Parafarmácias",
        "aliases": {"pharmacies", "pharmacy", "farmacias", "farmacia", "farmácias", "farmácia"},
    },
    "hospitals": {
        "tool_label": "hospitais",
        "category": "saúde",
        "dataset_term": "Hospitais Públicos",
        "aliases": {"hospitals", "hospital", "hospitais", "hospitalares"},
    },
    "schools": {
        "tool_label": "escolas",
        "category": "educação",
        "dataset_term": "Escolas Públicas",
        "aliases": {"schools", "school", "escolas", "escola"},
    },
    "libraries": {
        "tool_label": "bibliotecas",
        "category": "cultura",
        "dataset_term": "Bibliotecas Arquivos e Centros de Documentação",
        "aliases": {"libraries", "library", "bibliotecas", "biblioteca"},
    },
    "gardens": {
        "tool_label": "jardins",
        "category": "ambiente",
        "dataset_term": "Jardins - Parques Urbanos",
        "aliases": {"gardens", "garden", "jardins", "jardim", "parks", "park", "parques", "parque"},
    },
    "playgrounds": {
        "tool_label": "parques infantis",
        "category": "ambiente",
        "dataset_term": "Parques Infantis",
        "aliases": {"playgrounds", "playground", "parques infantis", "parque infantil", "infantil"},
    },
    "water_points": {
        "tool_label": "Arquitetura da Água",
        "category": "ambiente",
        "dataset_term": "Arquitetura da Água",
        "aliases": {
            "water points",
            "water point",
            "drinking fountains",
            "drinking fountain",
            "fountains",
            "fountain",
            "bebedouros",
            "bebedouro",
            "pontos de água",
            "ponto de água",
            "pontos de agua",
            "ponto de agua",
            "fontanários",
            "fontanario",
            "chafarizes",
            "chafariz",
        },
    },
    "police": {
        "tool_label": "polícia",
        "category": None,
        "dataset_term": "Polícia Municipal",
        "aliases": {"police", "policia", "polícia", "psp", "esquadra"},
    },
    "public_restrooms": {
        "tool_label": "Instalações Sanitárias",
        "category": None,
        "dataset_term": "Instalações Sanitárias",
        "aliases": {
            "public_restrooms", "public_restroom", "public_toilet", "public_toilets",
            "restroom", "restrooms", "wc", "sanitarios", "sanitario", "sanitários", "sanitário",
            "casas_de_banho_publicas", "casa_de_banho_publica",
        },
    },
    "wifi": {
        "tool_label": "wifi",
        "category": "serviços",
        "dataset_term": "Wi-Fi",
        "aliases": {"wifi", "wi-fi", "wi fi", "public wifi", "public wi-fi", "internet"},
    },
    "bike_parking": {
        "tool_label": "Estacionamento de velocípedes",
        "category": None,
        "dataset_term": "Estacionamento de velocípedes",
        "aliases": {
            "bike_parking", "bicycle_parking", "bikeparking", "bicycleparking",
            "estacionamento_de_bicicletas", "estacionamento_de_velocipedes", "bicicletas",
        },
    },
    "battery_recycling": {
        "tool_label": "pilhões",
        "category": "ambiente",
        "dataset_term": "Pilhões",
        "aliases": {
            "battery_recycling", "battery_bins", "battery_bin", "battery_points",
            "pilhoes", "pilhao", "pilhões", "pilhão", "pilhas", "baterias",
        },
    },
    "waste_bins": {
        "tool_label": "papeleiras",
        "category": "ambiente",
        "dataset_term": "Papeleiras",
        "aliases": {
            "waste_bins", "waste_bin", "litter_bins", "litter_bin",
            "papeleiras", "papeleira", "caixote_do_lixo", "caixotes_do_lixo",
        },
    },
    "dog_parks": {
        "tool_label": "parques caninos",
        "category": "ambiente",
        "dataset_term": "Parques Caninos",
        "aliases": {
            "dog_parks", "dog_park", "parques_caninos", "parque_canino",
            "canino", "caes", "cães",
        },
    },
    "emergency_meeting_points": {
        "tool_label": "pontos de encontro de emergência",
        "category": "segurança",
        "dataset_term": "Lisboa. Pontos de encontro - Emergência.",
        "aliases": {
            "emergency_meeting_points", "emergency_meeting_point",
            "pontos_de_encontro_de_emergencia", "ponto_de_encontro_de_emergencia",
            "pontos_de_encontro", "ponto_de_encontro", "emergencia", "emergência",
            "proteccao_civil", "protecao_civil", "proteção_civil",
        },
    },
    "cemeteries": {
        "tool_label": "cemitérios",
        "category": None,
        "dataset_term": "Cemitérios",
        "aliases": {"cemeteries", "cemetery", "cemiterios", "cemiterio", "cemitérios", "cemitério"},
    },
    "firefighters": {
        "tool_label": "bombeiros",
        "category": None,
        "dataset_term": "Bombeiros",
        "aliases": {"firefighters", "firefighter", "bombeiros", "bombeiro"},
    },
    "parking": {
        "tool_label": "estacionamento",
        "category": None,
        "dataset_term": "Parques de estacionamento na via pública",
        "aliases": {"parking", "estacionamento", "car_park", "carpark"},
    },
    "markets": {
        "tool_label": "mercados",
        "category": None,
        "dataset_term": "Mercados",
        "aliases": {"markets", "market", "mercados", "mercado", "feiras", "feira"},
    },
    "embassies": {
        "tool_label": "embaixadas",
        "category": None,
        "dataset_term": "Embaixadas",
        "aliases": {"embassies", "embassy", "embaixadas", "embaixada"},
    },
    "metro_stations": {
        "tool_label": "Estações de Metro",
        "category": None,
        "dataset_term": "Estações de Metro",
        "aliases": {"metro_stations", "metro_station", "estacoes_de_metro", "estacao_de_metro", "estações_de_metro", "estação_de_metro"},
    },
    "sports_facilities": {
        "tool_label": "Instalações Desportivas",
        "category": None,
        "dataset_term": "Instalações Desportivas",
        "aliases": {"sports_facilities", "sports_facility", "instalacoes_desportivas", "instalações_desportivas"},
    },
    "citizen_shops": {
        "tool_label": "Loja do Cidadão",
        "category": None,
        "dataset_term": "Loja do Cidadão",
        "aliases": {"citizen_shops", "citizen_shop", "loja_do_cidadao", "loja_do_cidadão"},
    },
}

_STRUCTURED_DATE_MONTHS: Dict[str, int] = {
    "january": 1, "janeiro": 1, "jan": 1,
    "february": 2, "fevereiro": 2, "feb": 2, "fev": 2,
    "march": 3, "marco": 3, "março": 3, "mar": 3,
    "april": 4, "abril": 4, "apr": 4, "abr": 4,
    "may": 5, "maio": 5,
    "june": 6, "junho": 6, "jun": 6,
    "july": 7, "julho": 7, "jul": 7,
    "august": 8, "agosto": 8, "aug": 8, "ago": 8,
    "september": 9, "setembro": 9, "sep": 9, "set": 9,
    "october": 10, "outubro": 10, "oct": 10, "out": 10,
    "november": 11, "novembro": 11, "nov": 11,
    "december": 12, "dezembro": 12, "dec": 12, "dez": 12,
}


class ResearcherAgent(BaseAgent):
    """
    RAG researcher agent for places, events, and local knowledge.

    Uses 11 retrieval tools loaded via get_agent_tools:
        - search_places_attractions
        - search_cultural_events
        - search_lisbon_knowledge
        - find_nearby_services (pharmacies, hospitals, etc.)
        - get_event_categories / get_place_categories
        - search_history_culture (web search for history/facts)
        - list_available_datasets / get_dataset_details / find_place_in_datasets
        - list_service_categories

    Notes:
        This agent combines semantic retrieval over the vector store, on-demand
        open-data lookup, and web fallback search. It is the main worker for
        places, events, essential services, and Lisbon knowledge queries.
    """

    def __init__(self):
        """Initializes the researcher agent."""
        super().__init__("researcher")
        self.system_prompt = get_researcher_prompt()
        self._last_search_context: Optional[dict] = None
        self._pending_deterministic_replay: Optional[dict] = None
        self._pending_pagination_replay: Optional[dict] = None
        self._place_title_localization_cache: Dict[str, str] = {}

    @staticmethod
    def _place_title_needs_llm_localization(title: str) -> bool:
        """Return whether a place title still appears English in a PT answer."""
        normalized = unicodedata.normalize("NFKD", str(title or ""))
        ascii_title = normalized.encode("ascii", "ignore").decode("ascii")
        lowered = f" {ascii_title.lower()} "
        english_markers = (
            " museum ",
            " palace ",
            " castle ",
            " monastery ",
            " church ",
            " cathedral ",
            " tower ",
            " monument ",
            " gardens ",
            " garden ",
            " national ",
            " archaeological ",
            " maritime ",
            " tile ",
            " design ",
            " history ",
            " science ",
            " natural ",
        )
        portuguese_markers = (
            " museu ",
            " palacio ",
            " palácio ",
            " castelo ",
            " mosteiro ",
            " igreja ",
            " se ",
            " sé ",
            " torre ",
            " monumento ",
            " jardim ",
            " jardins ",
            " nacional ",
        )
        if any(marker in lowered for marker in portuguese_markers):
            return False
        return any(marker in lowered for marker in english_markers)

    @staticmethod
    def _valid_localized_place_title(original: str, candidate: str) -> bool:
        """Validate conservative LLM title localization output."""
        value = re.sub(r"\s+", " ", str(candidate or "").strip().strip('"“”'))
        if not value or value.upper() == "SAME":
            return False
        if "\n" in value or "[" in value or "]" in value or "http" in value.lower():
            return False
        if len(value) > max(90, len(str(original or "")) * 2 + 20):
            return False
        lowered = value.lower()
        blocked = ("tradução", "translation", "não sei", "unknown", "official")
        return not any(token in lowered for token in blocked)

    def _localize_place_title_with_llm(self, title: str, *, category: str = "", url: str = "") -> str:
        """Use the Researcher LLM to localize one place-card title for PT output."""
        raw_title = re.sub(r"\s+", " ", str(title or "").strip())
        if not raw_title or not self._place_title_needs_llm_localization(raw_title):
            return raw_title

        cache_key = f"{raw_title}|{category}|{url}"
        if cache_key in self._place_title_localization_cache:
            return self._place_title_localization_cache[cache_key]

        prompt = (
            "Localize this Lisbon tourism place title to European Portuguese.\n"
            "Return only the official/common Portuguese title, with no explanation.\n"
            "If the title is a brand, already official as written, or you are not confident, return SAME.\n\n"
            f"Title: {raw_title}\n"
            f"Category: {category or 'unknown'}\n"
            f"URL: {url or 'unknown'}"
        )
        try:
            response = self._safe_llm_invoke(
                self.llm,
                [
                    SystemMessage(content="You are a conservative Lisbon tourism title localizer. Do not invent uncertain names."),
                    HumanMessage(content=prompt),
                ],
            )
            candidate = str(getattr(response, "content", "") or "").strip()
        except Exception:
            candidate = ""

        if self._valid_localized_place_title(raw_title, candidate):
            localized = re.sub(r"\s+", " ", candidate.strip().strip('"“”'))
        else:
            localized = raw_title
        self._place_title_localization_cache[cache_key] = localized
        return localized

    def _localize_place_card_titles_with_llm(self, text: str, language: str) -> str:
        """Localize unknown English place-card titles in PT Researcher outputs."""
        if language != "pt" or not text or not re.search(r"(?i)(locais e atra|places and attractions|visitlisboa)", text):
            return text

        lines = str(text).splitlines()
        output_lines: List[str] = []
        card_re = re.compile(
            r"^(?P<prefix>\s*(?:[-*]\s+)?)\*\*(?P<icon>🏛️|🍽️|☕|🥐|🌿|📍|🖼️|🎵|📚|🛍️)\s+"
            r"(?P<title>[^*\n]+?)\*\*(?P<suffix>\s*)$"
        )
        h3_re = re.compile(
            r"^(?P<prefix>\s*#{1,6}\s+)(?P<icon>🏛️|🍽️|☕|🥐|🌿|📍|🖼️|🎵|📚|🛍️)\s+"
            r"(?P<title>.+?)(?P<suffix>\s*)$"
        )
        for index, line in enumerate(lines):
            match = card_re.match(line) or h3_re.match(line)
            if not match:
                output_lines.append(line)
                continue
            title = match.group("title").strip()
            nearby = "\n".join(lines[index + 1:index + 8])
            category_match = re.search(r"\*\*(?:Categoria|Category):\*\*\s*([^\n]+)", nearby)
            url_match = re.search(r"https://www\.visitlisboa\.com/[^\s)]+", nearby)
            localized = self._localize_place_title_with_llm(
                title,
                category=category_match.group(1).strip() if category_match else "",
                url=url_match.group(0) if url_match else "",
            )
            if card_re.match(line):
                output_lines.append(f"{match.group('prefix')}**{match.group('icon')} {localized}**{match.group('suffix')}")
            else:
                output_lines.append(f"{match.group('prefix')}{match.group('icon')} {localized}{match.group('suffix')}")
        return "\n".join(output_lines)

    def reset_conversation_context(self) -> None:
        """Clears cached result-window context for this session."""
        self._last_search_context = None
        self._pending_deterministic_replay = None
        self._pending_pagination_replay = None

    @staticmethod
    def _normalize_structured_plan_text(value: Any) -> Optional[str]:
        """Normalize optional JSON string fields returned by the structured query planner."""
        text = str(value or "").strip()
        if not text or text.lower() in {"null", "none", "unknown", "n/a"}:
            return None
        normalized = re.sub(r"\s+", " ", text).strip(" .?!,;:")
        return normalized or None

    @staticmethod
    def _normalize_structured_service_token(value: Any) -> str:
        """Normalize free-form service labels into a comparison-friendly token."""
        normalized = unicodedata.normalize("NFKD", str(value or ""))
        normalized = normalized.encode("ascii", "ignore").decode("ascii").lower()
        normalized = normalized.replace("-", "_").replace("/", "_")
        normalized = re.sub(r"[^a-z0-9_ ]+", "", normalized)
        return re.sub(r"\s+", "_", normalized).strip("_")

    @classmethod
    def _normalize_structured_service_types(cls, values: Any) -> List[str]:
        """Map LLM-emitted service labels to a compact canonical enum set."""
        if values is None:
            return []

        raw_values = values if isinstance(values, list) else [values]
        normalized_services: List[str] = []
        seen: set[str] = set()
        for raw_value in raw_values:
            normalized_token = cls._normalize_structured_service_token(raw_value)
            if not normalized_token:
                continue
            for canonical, definition in _STRUCTURED_SERVICE_TYPE_DEFINITIONS.items():
                aliases = {
                    cls._normalize_structured_service_token(alias)
                    for alias in definition.get("aliases", set())
                }
                aliases.add(cls._normalize_structured_service_token(canonical))
                if normalized_token in aliases:
                    if canonical not in seen:
                        seen.add(canonical)
                        normalized_services.append(canonical)
                    break
        return normalized_services

    @staticmethod
    def _normalize_structured_date_filter(value: Any, user_message: str) -> Optional[str]:
        """Normalize date filters from the structured planner into tool-friendly values."""
        raw_value = str(value or "").strip()
        if raw_value.lower() in {"", "null", "none", "unknown", "n/a"}:
            raw_value = ""

        current_year = datetime.now().year
        candidate = raw_value or user_message
        normalized = unicodedata.normalize("NFKD", candidate)
        normalized = normalized.encode("ascii", "ignore").decode("ascii").lower()
        normalized = re.sub(r"\s+", " ", normalized).strip()

        iso_match = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", normalized)
        if iso_match:
            return iso_match.group(0)

        compact_match = re.search(r"\b(\d{1,2})[/-](\d{1,2})\b", normalized)
        if compact_match:
            day = int(compact_match.group(1))
            month = int(compact_match.group(2))
            if 1 <= day <= 31 and 1 <= month <= 12:
                return f"{current_year}-{month:02d}-{day:02d}"

        long_match = re.search(r"\b(\d{1,2})\s+de\s+([a-z]+)\b", normalized)
        if long_match:
            day = int(long_match.group(1))
            month = _STRUCTURED_DATE_MONTHS.get(long_match.group(2))
            if month and 1 <= day <= 31:
                return f"{current_year}-{month:02d}-{day:02d}"

        for month_name in _STRUCTURED_DATE_MONTHS:
            if re.search(rf"\b{re.escape(month_name)}\b", normalized):
                return month_name

        fallback = raw_value.strip()
        return fallback or None

    @staticmethod
    def _has_explicit_calendar_reference(user_message: str) -> bool:
        """Return whether a query mentions a concrete day or month that merits structured extraction."""
        normalized = unicodedata.normalize("NFKD", user_message or "")
        normalized = normalized.encode("ascii", "ignore").decode("ascii").lower()
        if re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", normalized):
            return True
        if re.search(r"\b\d{1,2}\s+de\s+[a-z]+\b", normalized):
            return True
        return any(re.search(rf"\b{re.escape(month_name)}\b", normalized) for month_name in _STRUCTURED_DATE_MONTHS)

    def _build_structured_query_plan_prompt(self) -> str:
        """Build a compact JSON-only prompt for LLM-assisted routing of hard researcher queries."""
        current_year = datetime.now().year
        intents = ", ".join(f'"{intent}"' for intent in sorted(_STRUCTURED_QUERY_PLAN_INTENTS))
        service_types = ", ".join(f'"{name}"' for name in _STRUCTURED_SERVICE_TYPE_DEFINITIONS)
        return (
            "You convert Lisbon local-information queries into a compact routing plan. "
            "Return ONLY valid JSON with keys: intent, subject, near_location, service_types, dataset_name, date_filter, category_hint. "
            f"intent must be one of [{intents}]. "
            f"service_types must be an array using only [{service_types}]. "
            "Restaurants, cafes, bars, fado venues, hotels, museums, attractions, viewpoints, and tourist places "
            "are place_lookup requests with category_hint, not nearby_services. "
            "Use nearby_services/service_types only for explicit municipal or public-service facilities such as "
            "pharmacies, hospitals, markets, restrooms, police, schools, libraries, parking, Wi-Fi, or recycling. "
            f"For day-month references without a year, assume {current_year} and output YYYY-MM-DD. "
            "For month-only filters, keep the month name exactly as written by the user. "
            "For generic nearby-service queries, keep subject null unless the user names a specific facility. "
            "Do not copy the nearby location into subject. Use null for unknown values and [] for no service types."
        )

    def _should_try_structured_query_plan(self, user_message: str) -> bool:
        """Gate the extra LLM parsing step to only the researcher slices where regex heuristics are weak."""
        normalized = unicodedata.normalize("NFKD", user_message or "")
        normalized = normalized.encode("ascii", "ignore").decode("ascii").lower()
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized:
            return False

        dataset_markers = [" dataset", " datasets", "dados abertos", "lisboa aberta"]
        unsupported_service_markers = [
            "casa de banho", "casas de banho", "sanitario", "sanitarios",
            "biciclet", "velociped", "cemiter", "bombeir", "embaix",
            "pilh", "bateria", "papeleir", "caixote", "parque canino",
            "canino", "ponto de encontro", "emergencia", "protecao civil",
            "loja do cidada", "estacoes de metro", "estacao de metro",
        ]
        named_service_with_nearby = bool(
            self._extract_near_location_name(user_message)
            and self._extract_place_focus_query(user_message)
            and self._extract_service_types(user_message)
        )
        event_with_explicit_calendar = (
            self._is_direct_event_lookup_query(user_message)
            and self._extract_event_date_filter(user_message) is None
            and self._has_explicit_calendar_reference(user_message)
        )
        researcher_domain_marker = bool(
            re.search(
                r"\b(?:eventos?|events?|restaurantes?|restaurants?|fado|museus?|museums?|"
                r"monumentos?|monuments?|miradouros?|viewpoints?|locais?|places?|"
                r"farm[aá]cias?|pharmacies|hospitais?|hospitals?|bibliotecas?|libraries|"
                r"mercados?|markets?|servi[cç]os?|services)\b",
                normalized,
                flags=re.IGNORECASE,
            )
        )
        preference_or_exclusion = bool(
            re.search(
                r"\b(?:sem|nao|não|evitar|excluir|excepto|exceto|menos|prefiro|prefer|"
                r"not|without|avoid|excluding|except|best|melhor|recomenda|recomendar|"
                r"sugere|suggest|autentico|autêntico|authentic|local|barato|cheap|"
                r"orcamento|orçamento|budget|vista|view|tejo|tagus|familia|família|"
                r"family|criancas|crianças|kids|acessivel|acessível|accessible)\b",
                normalized,
                flags=re.IGNORECASE,
            )
        )
        complex_researcher_query = (
            researcher_domain_marker
            and preference_or_exclusion
            and len(normalized.split()) >= 8
        )

        return (
            any(marker in normalized for marker in dataset_markers)
            or any(marker in normalized for marker in unsupported_service_markers)
            or named_service_with_nearby
            or event_with_explicit_calendar
            or complex_researcher_query
        )

    def _extract_structured_query_plan(self, user_message: str) -> Optional[Dict[str, Any]]:
        """Use the configured researcher LLM to parse hard natural-language queries into canonical routing fields."""
        if not user_message or not user_message.strip():
            return None

        response = self._safe_llm_invoke(
            self.llm,
            [
                SystemMessage(content=self._build_structured_query_plan_prompt()),
                HumanMessage(content=user_message),
            ],
            retries=1,
        )
        payload = parse_json_response(str(getattr(response, "content", response) or ""))
        if not isinstance(payload, dict):
            return None

        intent = str(payload.get("intent") or "").strip().lower()
        if intent not in _STRUCTURED_QUERY_PLAN_INTENTS or intent == "unknown":
            return None

        subject = self._normalize_structured_plan_text(payload.get("subject"))
        near_location = self._normalize_structured_plan_text(payload.get("near_location"))
        dataset_name = self._normalize_structured_plan_text(payload.get("dataset_name"))
        category_hint = self._normalize_structured_plan_text(payload.get("category_hint"))
        service_types = self._normalize_structured_service_types(payload.get("service_types"))
        date_filter = self._normalize_structured_date_filter(payload.get("date_filter"), user_message)

        if self._query_should_use_visitlisboa_place_catalog(user_message, category_hint):
            inferred_place_category = self._infer_place_category_hint(user_message)
            normalized_message = _normalize_researcher_intent_text(user_message)
            intent = "place_lookup"
            service_types = []
            category_hint = (
                "Restaurants"
                if inferred_place_category == "Restaurants" and _RESEARCHER_FOOD_INTENT_RE.search(normalized_message)
                else self._canonical_structured_place_category_hint(category_hint)
                or inferred_place_category
                or category_hint
            )

        if subject and near_location and subject.lower() == near_location.lower():
            subject = None

        return {
            "intent": intent,
            "subject": subject,
            "near_location": near_location,
            "service_types": service_types,
            "dataset_name": dataset_name,
            "date_filter": date_filter,
            "category_hint": category_hint,
        }

    @staticmethod
    def _canonical_structured_place_category_hint(value: Any) -> Optional[str]:
        """Map an LLM-emitted category hint to a supported VisitLisboa category."""
        normalized = ResearcherAgent._normalize_event_preference_text(str(value or ""))
        if not normalized:
            return None
        exact_categories = {
            "museums & monuments": "Museums & Monuments",
            "museums": "Museums & Monuments",
            "monuments": "Museums & Monuments",
            "restaurants": "Restaurants",
            "restaurant": "Restaurants",
            "hotels": "Hotels",
            "hotel": "Hotels",
            "view points": "View Points",
            "viewpoints": "View Points",
            "beaches": "Beaches",
            "shopping": "Shopping",
            "tourist offices": "Tourist Offices",
            "tourist office": "Tourist Offices",
            "nightlife": "Nightlife",
            "parks & gardens": "Parks & Gardens",
            "parks": "Parks & Gardens",
            "gardens": "Parks & Gardens",
            "tours": "Tours",
            "tejo cruises": "Tejo Cruises",
        }
        if normalized in exact_categories:
            return exact_categories[normalized]
        alias_patterns = (
            ("Museums & Monuments", r"\b(?:museu|museus|museum|monumento|monument|palacio|palace|castelo|castle)\b"),
            ("Restaurants", r"\b(?:restaurante|restaurant|gastronomia|food|dining|fado|marisco|seafood)\b"),
            ("View Points", r"\b(?:miradouro|viewpoint|vista|view)\b"),
            ("Parks & Gardens", r"\b(?:jardim|garden|parque|park)\b"),
            ("Beaches", r"\b(?:praia|beach|surf)\b"),
            ("Tourist Offices", r"\b(?:tourist\s+offices?|postos?\s+de\s+turismo|informa[cç][aã]o\s+tur[ií]stica)\b"),
            ("Shopping", r"\b(?:compras|shopping|lojas|shops?)\b"),
            ("Nightlife", r"\b(?:nightlife|vida\s+noturna|bar|bars)\b"),
            ("Tours", r"\b(?:tour|tours|visita|visitas|experience|experiencia|experiência)\b"),
        )
        for category, pattern in alias_patterns:
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                return category
        return None

    @staticmethod
    def _structured_subject_looks_like_generic_place_query(value: Any) -> bool:
        """Return whether a structured subject is a category query, not a name."""
        normalized = ResearcherAgent._normalize_event_preference_text(str(value or ""))
        if not normalized:
            return False
        if len(normalized.split()) > 6:
            return True
        return bool(
            re.search(
                r"\b(?:restaurants?|restaurantes?|museums?|museus?|monuments?|monumentos?|"
                r"events?|eventos?|places?|locais?|attractions?|atracoes?|atrações?|"
                r"best|melhor|recomenda|suggest|with|com|near|perto|view|vista|"
                r"tejo|tagus|food|seafood|marisco|cheap|barato|budget|orcamento|orçamento|"
                r"without|sem|not|nao|não|avoid|evitar)\b",
                normalized,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _query_should_use_visitlisboa_place_catalog(user_message: str, category_hint: Any = None) -> bool:
        """Return whether an LLM structured plan must stay in the tourism/place catalogue."""
        normalized = _normalize_researcher_intent_text(user_message)
        if not normalized:
            return False

        canonical_hint = ResearcherAgent._canonical_structured_place_category_hint(category_hint)
        if canonical_hint in {
            "Restaurants",
            "Museums & Monuments",
            "Hotels",
            "View Points",
            "Beaches",
            "Shopping",
            "Nightlife",
            "Parks & Gardens",
            "Tours",
            "Tejo Cruises",
        }:
            return True

        explicit_dataset_or_service = bool(
            re.search(
                r"\b(?:lisboa aberta|dados abertos|dataset|datasets|municipal|public service|"
                r"servico municipal|servicos municipais|serviço municipal|serviços municipais)\b",
                normalized,
            )
        )
        restaurant_or_food_request = bool(_RESEARCHER_FOOD_INTENT_RE.search(normalized))
        restaurant_word_present = bool(
            re.search(r"\b(?:restaurant|restaurants|restaurante|restaurantes|dining|jantar|dinner|almoco|almoço|lunch)\b", normalized)
        )
        market_only_request = bool(
            re.search(r"\b(?:market|markets|mercado|mercados|feira|feiras)\b", normalized)
            and not restaurant_word_present
        )
        if restaurant_or_food_request and not explicit_dataset_or_service and not market_only_request:
            return True

        return bool(
            re.search(
                r"\b(?:museu|museus|museum|museums|monumento|monumentos|monument|monuments|"
                r"miradouro|miradouros|viewpoint|viewpoints|hotel|hotels|fado|"
                r"attraction|attractions|atracao|atracoes|atração|atrações|tourist places|locais turisticos)\b",
                normalized,
            )
            and not explicit_dataset_or_service
        )

    @staticmethod
    def _append_subjective_place_preference_caveat(result: str, user_message: str, language: str) -> str:
        """Add a caveat when requested place preferences are subjective and unverifiable."""
        if not result:
            return result or ""

        normalized_query = _normalize_researcher_intent_text(user_message)
        if not re.search(
            r"\b(?:not overly touristy|not touristy|less touristy|touristy|tourist trap|"
            r"hidden gem|authentic|local vibe|local feel|quiet|not crowded|cosy|cozy|romantic|"
            r"pouco turistico|menos turistico|turistico|armadilha turistica|autentico|"
            r"ambiente local|calmo|pouco cheio|romantico)\b",
            normalized_query,
        ):
            return result

        visible_result = _normalize_researcher_intent_text(result)
        if re.search(
            r"\b(?:subjective filter|subjective criteria|not fully verified|not explicitly verified|"
            r"criterios subjetivos|nao permitem verificar|não permitem verificar|nao ficou confirmado)\b",
            visible_result,
        ):
            return result

        if (language or "").lower().startswith("pt"):
            caveat = (
                "⚠️ **Limitação:** critérios subjetivos como serem pouco turísticos, autênticos ou calmos "
                "não ficam explicitamente confirmados nos dados disponíveis; priorizei locais com sinais "
                "compatíveis e detalhes verificáveis."
            )
        else:
            caveat = (
                "⚠️ **Limitation:** subjective filters such as how touristy, authentic, or quiet a place feels "
                "are not explicitly verified in the available place data; I prioritised venues with compatible "
                "signals and verifiable details."
            )
        return f"{result.rstrip()}\n\n{caveat}"

    @staticmethod
    def _structured_service_tool_label(service_type: str) -> Optional[str]:
        """Resolve a canonical structured service enum to the best nearby-service tool label."""
        definition = _STRUCTURED_SERVICE_TYPE_DEFINITIONS.get(service_type)
        if not definition:
            return None
        return str(definition.get("tool_label") or "").strip() or None

    @staticmethod
    def _service_type_identity(service_type: str) -> str:
        """Return a stable identity for equivalent Lisboa Aberta service labels."""
        normalized = unicodedata.normalize("NFKD", service_type or "")
        normalized = normalized.encode("ascii", "ignore").decode("ascii").lower()
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
        if any(marker in normalized for marker in ("sanitario", "instalac", "restroom", "toilet", "wc")):
            return "public_restrooms"
        if any(marker in normalized for marker in ("wifi", "wi fi", "internet")):
            return "wifi"
        if "farmac" in normalized:
            return "pharmacies"
        if "hospital" in normalized:
            return "hospitals"
        if "bibliotec" in normalized or "librar" in normalized:
            return "libraries"
        if "mercado" in normalized or "market" in normalized:
            return "markets"
        return normalized

    @staticmethod
    def _structured_service_category(service_type: str) -> Optional[str]:
        """Resolve a canonical structured service enum to the best Lisboa Aberta taxonomy hint."""
        definition = _STRUCTURED_SERVICE_TYPE_DEFINITIONS.get(service_type)
        if not definition:
            return None
        category = definition.get("category")
        return str(category).strip() if isinstance(category, str) and category.strip() else None

    @staticmethod
    def _structured_dataset_search_term(structured_plan: Dict[str, Any], user_message: str) -> Optional[str]:
        """Choose the best dataset-search term from a structured plan."""
        for field_name in ("dataset_name", "subject", "category_hint"):
            value = ResearcherAgent._normalize_structured_plan_text(structured_plan.get(field_name))
            if value:
                return value
        for service_type in structured_plan.get("service_types", []):
            definition = _STRUCTURED_SERVICE_TYPE_DEFINITIONS.get(service_type)
            dataset_term = str(definition.get("dataset_term") or "").strip() if definition else ""
            if dataset_term:
                return dataset_term
        return ResearcherAgent._normalize_structured_plan_text(user_message)

    def _run_structured_dataset_lookup(self, user_message: str, language: str, structured_plan: Dict[str, Any]) -> Optional[str]:
        """Execute dataset-search or dataset-details intents resolved by the structured planner."""
        intent = str(structured_plan.get("intent") or "").strip()
        if intent not in {"dataset_search", "dataset_details"}:
            return None

        tool_name = "get_dataset_details" if intent == "dataset_details" else "list_available_datasets"
        tool = self._get_tool_by_name(tool_name)
        if not tool:
            return None

        if intent == "dataset_details":
            dataset_name = self._normalize_structured_plan_text(structured_plan.get("dataset_name"))
            dataset_name = dataset_name or self._structured_dataset_search_term(structured_plan, user_message)
            if not dataset_name:
                return None
            result = str(self._invoke_tool(tool, {"dataset_name": dataset_name}, tool_name=tool_name)).strip()
        else:
            search_term = self._structured_dataset_search_term(structured_plan, user_message)
            if not search_term:
                return None
            result = str(self._invoke_tool(tool, {"category": search_term}, tool_name=tool_name)).strip()

        if not result:
            return None
        if "Lisboa Aberta" not in result:
            result = f"{result}\n\n{self._build_open_data_services_source_line(language)}".strip()
        return result

    def _maybe_run_structured_query_plan(self, user_message: str, language: str) -> Optional[str]:
        """Try a low-overhead LLM-assisted routing pass for researcher queries that regex heuristics underspecify."""
        if not self._should_try_structured_query_plan(user_message):
            return None

        if self._is_transactional_place_lookup_query(user_message):
            return self._run_direct_place_lookup(user_message, language)

        structured_plan = self._extract_structured_query_plan(user_message)
        if not structured_plan:
            return None

        intent = structured_plan.get("intent")
        if intent in {"dataset_search", "dataset_details"}:
            return self._run_structured_dataset_lookup(user_message, language, structured_plan)
        if intent == "event_lookup" and (structured_plan.get("date_filter") or structured_plan.get("subject")):
            return self._run_direct_event_lookup(user_message, language, structured_plan=structured_plan)
        if intent in {"nearby_services", "place_lookup"} and (
            structured_plan.get("service_types")
            or structured_plan.get("subject")
            or structured_plan.get("near_location")
        ):
            return self._run_direct_place_lookup(user_message, language, structured_plan=structured_plan)
        return None

    @staticmethod
    def _is_transactional_place_lookup_query(user_message: str) -> bool:
        """Return whether a booking/buying request targets a specific place."""
        return bool(
            _extract_specific_place_lookup_phrase(user_message)
            and re.search(
                r"\b(?:book|reserve|buy|purchase|tickets?|table|hotel|room|"
                r"reservar|reserva|comprar|compra|bilhetes?|entradas?|mesa|marcar|marca)\b",
                user_message or "",
                flags=re.IGNORECASE,
            )
        )

    def _replay_same_deterministic_response_once(self, user_message: str) -> Optional[str]:
        """Return a cached deterministic response once when the same message is retried immediately."""
        normalized_message = (user_message or "").strip().lower()
        pending_replay = getattr(self, "_pending_deterministic_replay", None)
        if pending_replay and pending_replay.get("message") == normalized_message:
            self._pending_deterministic_replay = None
            replayed = str(pending_replay.get("response") or "").strip()
            return replayed or None
        self._pending_deterministic_replay = None
        return None

    def _remember_deterministic_response_for_retry(self, user_message: str, response: str) -> str:
        """Stage a deterministic response so a same-message retry can replay it once."""
        self._pending_deterministic_replay = {
            "message": (user_message or "").strip().lower(),
            "response": response,
        }
        return response

    # Anaphoric lookback patterns; conservative to avoid misrouting factual queries.
    _CONVERSATIONAL_RECALL_PATTERNS: Tuple[re.Pattern, ...] = (
        re.compile(
            r"\b(?:qual\s+(?:foi|era|é)|que\s+(?:foi|era))\s+"
            r"(?:o|a|os|as)?\s*\w+\s+(?:que|q)\s+"
            r"(?:suger[ie][a-z]{0,4}|indica[a-z]{0,4}|recomenda[a-z]{0,4}|"
            r"menciona[a-z]{0,4}|diss?es[a-z]{0,3}|disse|dize[a-z]{0,3}|"
            r"propu[a-z]{0,4}|propus[a-z]{0,4}|prop[oõ]e[a-z]{0,3}|"
            r"refer[ei][a-z]{0,4}|fala[a-z]{0,4}|aponta[a-z]{0,4}|"
            r"d[aá]s|d[aá]|mostra[a-z]{0,4})\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:que|qual)\s+\w+\s+(?:é\s+que\s+)?"
            r"(?:suger[ie][a-z]{0,4}|indica[a-z]{0,4}|recomenda[a-z]{0,4}|"
            r"menciona[a-z]{0,4}|diss?es[a-z]{0,3}|disse|dize[a-z]{0,3}|"
            r"propu[a-z]{0,4}|refer[ei][a-z]{0,4}|fala[a-z]{0,4}|"
            r"aponta[a-z]{0,4}|d[aá]s|d[aá]|mostra[a-z]{0,4})\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:what|which)\s+(?:was|were|is|are)?\s*(?:the\s+)?\w+\s+"
            # Allow optional auxiliary (did/do/does/have/had) before "you".
            r"(?:(?:did|do|does|have|had)\s+)?(?:that\s+)?you\s+"
            r"(?:suggested|recommended|mentioned|said|proposed|"
            r"referred(?:\s+to)?|talked\s+about|pointed\s+(?:out|to)|"
            r"suggest|recommend|mention|propose|say|refer\s+to|"
            r"talk\s+about|point\s+(?:out|to))\b",
            re.IGNORECASE,
        ),
    )

    _PREVIOUS_ASSISTANT_CONTEXT_RE = re.compile(
        r"Previous assistant answer[^:]*:\s*(.+?)(?:\n\n[A-Z][a-z]+ [a-z]+ [a-z]+:|\Z)",
        re.IGNORECASE | re.DOTALL,
    )

    _LANGUAGE_AWARE_TOOL_NAMES: frozenset[str] = frozenset(
        {
            "search_places_attractions",
            "search_cultural_events",
            "search_history_culture",
            "search_lisbon_knowledge",
            "find_nearby_services",
            "get_place_categories",
            "get_event_categories",
            "list_service_categories",
        }
    )

    def _preprocess_tool_args(self, tool_name: str, tool_args: dict) -> dict:
        """Force language-aware researcher tools to match the active response language."""
        args = super()._preprocess_tool_args(tool_name, dict(tool_args or {}))
        active_language = str(getattr(self, "_active_response_language", "") or "").lower()
        if tool_name in self._LANGUAGE_AWARE_TOOL_NAMES and active_language in {"pt", "en"}:
            args["language"] = active_language
        return args

    @staticmethod
    def _summarize_previous_cards_for_recall(
        user_message: str,
        previous_assistant_text: str,
        language: str,
    ) -> str:
        """Extract concise recalled card titles from a previous answer."""
        normalized_message = _normalize_researcher_intent_text(user_message)

        if _RESEARCHER_FOOD_INTENT_RE.search(normalized_message):
            target_re = re.compile(
                r"\b(?:almoço|almoco|lunch|jantar|dinner|restaurant|restaurante|gastron[oó]m\w*|food)\b",
                re.IGNORECASE,
            )
            title = "restaurante" if language == "pt" else "restaurant"
        elif re.search(r"\b(?:museu|museum|monumento|monument|atra[cç]ao|atração|attraction|local|place)\b", normalized_message):
            target_re = re.compile(
                r"\b(?:museu|museum|monumento|monument|paragem hist[oó]rica|historical stop|torre|padr[aã]o|mosteiro|carmo|s[eé])\b",
                re.IGNORECASE,
            )
            title = "locais" if language == "pt" else "places"
        elif re.search(r"\b(?:evento|event|concerto|concert|exposi[cç][aã]o|exhibition)\b", normalized_message):
            target_re = re.compile(r"\b(?:evento|event|concerto|concert|exposi[cç][aã]o|exhibition)\b", re.IGNORECASE)
            title = "eventos" if language == "pt" else "events"
        else:
            target_re = re.compile(r".")
            title = "itens" if language == "pt" else "items"

        candidates: list[str] = []
        card_patterns = (
            r"(?m)^\*\*(?:🏷️\s*)?(?P<title>[^*\n]{3,140})\*\*\s*$",
            r"(?m)^-\s*(?:[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+\s*)?\*\*(?P<title>[^*\n]{3,140})\*\*",
            r"(?m)^#{2,4}\s+(?:[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+\s*)?\*\*(?P<title>[^*\n]{3,140})\*\*",
        )
        for pattern in card_patterns:
            for match in re.finditer(pattern, previous_assistant_text or ""):
                raw_title = re.sub(r"\s+", " ", match.group("title")).strip(" .:-")
                if not raw_title or re.search(r"\b(?:resposta direta|direct answer|fonte|source|dicas|tips|notas finais|final notes)\b", raw_title, re.IGNORECASE):
                    continue
                if not target_re.search(raw_title):
                    continue
                display = raw_title
                if ":" in raw_title and re.search(r"\b(?:almo|lunch|jantar|dinner|paragem|stop)\b", raw_title, re.IGNORECASE):
                    display = raw_title.split(":", 1)[1].strip()
                if display and display not in candidates:
                    candidates.append(display)
                if len(candidates) >= 5:
                    break
            if len(candidates) >= 5:
                break

        if not candidates:
            return ""
        joined = "; ".join(f"**{item}**" for item in candidates)
        if language == "pt":
            return f"✅ **Resposta direta:** os {title} que referi foram: {joined}."
        return f"✅ **Direct answer:** the {title} I referred to were: {joined}."

    def _maybe_answer_conversational_recall(
        self, user_message: str, context: str, language: str
    ) -> Optional[str]:
        """Answer recall-style follow-ups from the previous assistant turn.

        Returns a formatted answer string when the user asks "what restaurant
        did you suggest?" / "qual foi o restaurante que indicaste?" and a
        previous assistant message is available in ``context``. Returns None
        when no recall intent is detected or no previous assistant text was
        injected by the orchestrator.
        """
        message_text = (user_message or "").strip()
        if not message_text:
            return None
        if not any(pattern.search(message_text) for pattern in self._CONVERSATIONAL_RECALL_PATTERNS):
            return None

        match = self._PREVIOUS_ASSISTANT_CONTEXT_RE.search(context or "")
        if not match:
            return None
        previous_assistant_text = match.group(1).strip()
        if not previous_assistant_text:
            return None
        # Aggressively strip any leading separator markers, blank lines, and
        # stray dash characters so the quoted text does not collide with our
        # own intro separator above.
        previous_assistant_text = re.sub(
            r"^(?:[\s\-]*-{3,}[\s\-]*)+",
            "",
            previous_assistant_text,
        ).lstrip()
        # Also drop a leading "✅ Resposta direta" / "Direct answer" line so the
        # recall card does not echo the prior intro verbatim.
        previous_assistant_text = re.sub(
            r"^(?:✅\s*\*\*\s*(?:Resposta direta|Direct answer)[^\n]*\n+)",
            "",
            previous_assistant_text,
            flags=re.IGNORECASE,
        )
        # Re-strip in case the intro removal left another separator behind.
        previous_assistant_text = re.sub(
            r"^(?:[\s\-]*-{3,}[\s\-]*)+",
            "",
            previous_assistant_text,
        ).lstrip()

        summarized = self._summarize_previous_cards_for_recall(
            message_text,
            previous_assistant_text,
            language,
        )
        if summarized:
            return summarized

        if language == "pt":
            intro = "✅ **Resposta direta:** aqui está a recomendação anterior referida.\n\n"
        else:
            intro = "✅ **Direct answer:** here is the previous recommendation you referenced.\n\n"
        return intro + previous_assistant_text[:900]

    @staticmethod
    def _is_content_filter_error(error: Exception) -> bool:
        """Returns whether an exception is an Azure content-filter false positive."""
        error_str = str(error).lower()
        return (
            "content_filter" in error_str
            or "responsibleaipolicyviolation" in error_str
            or "jailbreak" in error_str
        )

    @staticmethod
    def _infer_research_query_language(user_message: str) -> str:
        """Resolve the researcher reply language via the shared PT/EN router."""
        resolved_language, _requires_note, _detected_language = resolve_output_language(
            user_query=user_message,
            ui_default="en",
        )
        return resolved_language

    @staticmethod
    def _build_messages(
        system_prompt: str,
        user_message: str,
        context: str = "",
        language: str | None = None,
    ) -> list:
        """Builds the message list for a researcher invocation.

        Args:
            system_prompt: System prompt text.
            user_message: The user's query.
            context: Additional orchestrator context.
            language: Pre-resolved language code ('pt' or 'en'). When ``None``,
                the language is inferred from *user_message* as a fallback.
        """
        resolved_language = language or ResearcherAgent._infer_research_query_language(user_message)
        language_instruction = (
            "Respond ENTIRELY in Portuguese (PT-PT)."
            if resolved_language == "pt"
            else "Respond ENTIRELY in English."
        )

        messages = [
            SystemMessage(content=system_prompt),
            SystemMessage(content=language_instruction),
        ]

        if context:
            messages.append(SystemMessage(content=f"Context from other agents:\n{context}"))

        messages.append(HumanMessage(content=user_message))
        return messages

    def _get_tool_by_name(self, tool_name: str):
        """Returns a loaded tool by name, or None if not found."""
        for tool in getattr(self, "tools", []):
            if getattr(tool, "name", "") == tool_name:
                return tool
        return None

    @staticmethod
    def _is_accessibility_place_query(user_message: str) -> bool:
        """Detects high-risk accessibility place queries that should skip free-form synthesis."""
        query = (user_message or "").lower()
        accessibility_terms = [
            "wheelchair", "accessible", "accessibility", "step-free",
            "cadeira de rodas", "acessível", "acessivel", "mobilidade reduzida",
        ]
        place_terms = [
            "museum", "museu", "monument", "monumento", "place", "places",
            "attraction", "attractions", "visit", "visita", "visitar",
            "passeio", "sugere", "recomenda", "short visit", "visita curta",
            "belem", "belém",
        ]
        return any(term in query for term in accessibility_terms) and (
            any(term in query for term in place_terms)
            or bool(ResearcherAgent._extract_place_focus_query(user_message))
        )

    @staticmethod
    def _extract_current_location_anchor(user_message: str) -> Optional[str]:
        """Extract a compact current-location anchor such as "estou no Rossio"."""
        patterns = [
            r"\b(?:estou|encontro[-\s]?me|i\s+am|i'm)\s+"
            r"(?:no|na|em|at)\s+"
            r"(?P<location>[A-ZÀ-Ýa-zà-ÿ0-9][A-ZÀ-Ýa-zà-ÿ0-9 '\-/]{1,60})",
            r"\b(?:mobilidade\s+reduzida|cadeira\s+de\s+rodas|wheelchair|reduced\s+mobility)\b"
            r".{0,60}?\b(?:no|na|em|at)\s+"
            r"(?P<location>[A-ZÀ-Ýa-zà-ÿ0-9][A-ZÀ-Ýa-zà-ÿ0-9 '\-/]{1,60})",
        ]
        match = next(
            (
                found
                for pattern in patterns
                for found in [re.search(pattern, user_message or "", flags=re.IGNORECASE)]
                if found
            ),
            None,
        )
        if not match:
            return None
        location = re.sub(
            r"\s*(?:;|,|\.)?\s*(?:sugere|recomenda|suggest|recommend|quero|i\s+want|"
            r"com\s+|with\s+|para\s+|for\s+).*$",
            "",
            match.group("location"),
            flags=re.IGNORECASE,
        )
        location = ResearcherAgent._clean_place_focus_subject(location.strip(" .,:;?!"))
        return location or None

    @staticmethod
    def _is_accessibility_focus_noise(focus_query: Optional[str], user_message: str) -> bool:
        """Return whether an extracted focus is only accessibility phrasing noise."""
        if not focus_query:
            return False
        normalized_focus = _normalize_researcher_intent_text(focus_query)
        if normalized_focus in {
            "uso", "use", "using", "tenho", "have", "i", "im", "i am",
            "cadeira", "wheelchair", "mobilidade", "accessibility",
        }:
            return True
        normalized_message = _normalize_researcher_intent_text(user_message)
        return bool(
            normalized_focus in {"rodas", "reduced", "mobility"}
            and re.search(r"\b(?:cadeira de rodas|wheelchair|mobilidade reduzida|reduced mobility)\b", normalized_message)
        )

    def _run_accessibility_place_lookup(self, user_message: str, language: str) -> str:
        """Runs a deterministic place lookup for accessibility-focused queries."""
        tool = self._get_tool_by_name("search_places_attractions")
        if not tool:
            return self._run_direct_tool_fallback(user_message, language)

        focus_query = self._extract_place_focus_query(user_message)
        if self._is_accessibility_focus_noise(focus_query, user_message):
            focus_query = None
        current_location = self._extract_current_location_anchor(user_message)
        if focus_query:
            query = focus_query
        elif current_location:
            query = (
                f"locais e atrações perto de {current_location}"
                if language == "pt"
                else f"places and attractions near {current_location}"
            )
        else:
            query = user_message
        args = {"query": query, "max_results": 5, "offset": 0, "language": language}
        specific_lookup = _extract_specific_place_lookup_phrase(user_message)
        specific_tokens = re.findall(r"[a-z0-9]+", (specific_lookup or "").lower())
        broad_type_tokens = {"museum", "museums", "museu", "museus", "monument", "monuments"}
        broad_category_lookup = len(specific_tokens) <= 2 and any(
            token in broad_type_tokens for token in specific_tokens
        )
        if specific_lookup and not broad_category_lookup and focus_query:
            args["specific_lookup"] = True
        category_hint = self._infer_place_category_hint(user_message) or self._infer_place_category_hint(focus_query or "")
        generic_accessibility_visit = current_location and not focus_query and not re.search(
            r"\b(?:restaurantes?|restaurants?|jantar|dinner|almo[cç]o|lunch|comer|eat)\b",
            user_message,
            flags=re.IGNORECASE,
        )
        if generic_accessibility_visit:
            category_hint = "Museums & Monuments"
        if category_hint:
            args["category"] = category_hint

        result = str(self._invoke_tool(tool, args, tool_name="search_places_attractions")).strip()
        base_args = {key: value for key, value in args.items() if key not in {"max_results", "offset"}}
        self._remember_search_context(
            domain="places",
            tool_name="search_places_attractions",
            base_args=base_args,
            page_size=int(args["max_results"]),
            shown_count=self._count_ranked_results(result),
            language=language,
            source_query=user_message,
            offset=0,
        )
        source_line = self._build_places_source_line(result, language)
        if language == "pt":
            note = (
                "⚠️ **Mobilidade reduzida:** os dados de locais confirmam nomes, moradas e detalhes turísticos "
                "quando disponíveis; acessos sem degraus, elevadores operacionais e apoio no local devem ser "
                "confirmados diretamente com o espaço antes da deslocação."
            )
            if current_location and not focus_query:
                note += " Como o pedido é uma recomendação de visita e não uma rota para um destino escolhido, não calculei uma rota porta-a-porta."
        else:
            note = (
                "⚠️ **Reduced mobility:** place data confirms names, addresses, and tourism details where available; "
                "step-free access, working lifts, and on-site assistance should be confirmed directly before travelling."
            )
            if current_location and not focus_query:
                note += " Because this is a visit recommendation rather than a route to a chosen destination, I did not calculate a door-to-door route."
        return f"{result}\n\n{note}\n\n{source_line}".strip()

    @staticmethod
    def _extract_visit_confirmation_target(user_message: str) -> Optional[str]:
        """Extract the named venue from pre-visit checklist questions."""
        query = str(user_message or "").strip()
        patterns = [
            r"\b(?:quero\s+ir|vou|pretendo\s+ir|i\s+want\s+to\s+go|i\s+am\s+going|i'?m\s+going)\s+"
            r"(?:ao|à|a|para\s+o|para\s+a|to)\s+"
            r"(?P<target>[A-ZÀ-Ýa-zà-ÿ0-9][A-ZÀ-Ýa-zà-ÿ0-9 '&’\-/]{2,90}?)"
            r"(?=\s+(?:com|with|antes|before|e\s+|and\s+|o\s+que|what\s+|"
            r"hoje|amanh[ãa]|today|tomorrow|este|esta|this|next|pr[óo]xim[oa])|[.?!;,]|$)",
            r"\b(?:visitar|visit)\s+(?:o\s+|a\s+|os\s+|as\s+)?"
            r"(?P<target>[A-ZÀ-Ýa-zà-ÿ0-9][A-ZÀ-Ýa-zà-ÿ0-9 '&’\-/]{2,90}?)"
            r"(?=\s+(?:com|with|antes|before|e\s+|and\s+|o\s+que|what\s+|"
            r"hoje|amanh[ãa]|today|tomorrow|este|esta|this|next|pr[óo]xim[oa])|[.?!;,]|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, query, flags=re.IGNORECASE)
            if not match:
                continue
            target = ResearcherAgent._clean_place_focus_subject(match.group("target").strip(" .,:;?!"))
            target = re.sub(
                r"\s+\b(?:hoje|amanh[ãa]|today|tomorrow|este|esta|this|next|pr[óo]xim[oa])\b.*$",
                "",
                target,
                flags=re.IGNORECASE,
            ).strip()
            if target:
                return target
        return ResearcherAgent._extract_place_focus_query(user_message)

    @staticmethod
    def _is_visit_confirmation_checklist_query(user_message: str) -> bool:
        """Detect venue-specific "what should I confirm before going?" questions."""
        normalized = (user_message or "").lower()
        if not re.search(
            r"\b(?:o\s+que\s+(?:devo\s+)?confirmar|que\s+(?:devo\s+)?confirmar|"
            r"confirmar\s+antes|what\s+should\s+i\s+(?:check|confirm)|"
            r"what\s+to\s+(?:check|confirm)|check\s+before|confirm\s+before)\b",
            normalized,
        ):
            return False
        return bool(ResearcherAgent._extract_visit_confirmation_target(user_message))

    def _run_visit_confirmation_checklist_lookup(self, user_message: str, language: str) -> str:
        """Lookup the named venue and render a grounded pre-visit checklist."""
        tool = self._get_tool_by_name("search_places_attractions")
        if not tool:
            return self._run_direct_tool_fallback(user_message, language)

        target = self._extract_visit_confirmation_target(user_message) or user_message
        args = {
            "query": target,
            "specific_lookup": True,
            "max_results": 1,
            "offset": 0,
            "language": language,
        }
        result = str(self._invoke_tool(tool, args, tool_name="search_places_attractions")).strip()
        base_args = {key: value for key, value in args.items() if key not in {"max_results", "offset"}}
        self._remember_search_context(
            domain="places",
            tool_name="search_places_attractions",
            base_args=base_args,
            page_size=int(args["max_results"]),
            shown_count=self._count_ranked_results(result),
            language=language,
            source_query=user_message,
            offset=0,
        )
        child_context = bool(re.search(r"\b(?:criança|crianças|filh[ao]s?|kids?|children|child)\b", user_message, re.IGNORECASE))
        if language == "pt":
            checklist = [
                "### ✅ **O que confirmar antes da visita**",
                "- 🕒 **Horário e última entrada:** confirma se há alterações no dia da visita.",
                "- 🎟️ **Bilhetes e preço:** confirma preço, compra online e regras de entrada.",
            ]
            if child_context:
                checklist.append("- 👧 **Com criança:** confirma idade recomendada, atividades infantis, alimentação, WC e zonas de descanso.")
            checklist.extend([
                "- ♿ **Acessibilidade e deslocação:** confirma elevadores, acessos sem degraus e a melhor entrada se isso for relevante.",
                "- 🌦️ **Condições no dia:** para espaços com zonas exteriores, confirma meteorologia e eventuais alterações operacionais.",
            ])
            intro = f"### 🧾 **Antes da visita: {target}**\n\n✅ **Resposta direta:** confirma estes pontos práticos antes da visita; abaixo deixo os dados confirmados que encontrei."
        else:
            checklist = [
                "### ✅ **What to confirm before visiting**",
                "- 🕒 **Opening hours and last entry:** check whether the schedule changes on your visit day.",
                "- 🎟️ **Tickets/price:** confirm price, online purchase, and entry rules.",
            ]
            if child_context:
                checklist.append("- 👧 **With a child:** confirm recommended ages, child-friendly activities, food, toilets, and rest areas.")
            checklist.extend([
                "- ♿ **Accessibility and route:** confirm lifts, step-free access, and the best entrance if relevant.",
                "- 🌦️ **Day-of conditions:** for outdoor areas, check weather and operational changes.",
            ])
            intro = f"### 🧾 **Before visiting {target}**\n\n✅ **Direct answer:** confirm these practical points before going; below are the confirmed details I found."
        source_line = self._build_places_source_line(result, language)
        return f"{intro}\n\n---\n\n{result}\n\n---\n\n" + "\n".join(checklist) + f"\n\n{source_line}"

    @staticmethod
    def _infer_place_category_signals(user_message: str) -> List[str]:
        """Extract all recognized place-category hints from the query.

        Returns all possible category matches so callers can avoid over-filtering when
        a query asks for multiple types (for example one museum + one viewpoint).
        """
        query = (user_message or "").lower()
        signals: List[str] = []

        if any(term in query for term in ["museum", "museu", "monument", "monumento"]):
            signals.append("Museums & Monuments")
        if any(
            term in query
            for term in [
                "restaurant",
                "restaurants",
                "restaurante",
                "restaurantes",
                "seafood",
                "marisco",
                "marisqueira",
                "food",
                "dining",
                "gastronomy",
                "gastronomia",
                "cuisine",
                "lunch",
                "dinner",
                "almoco",
                "almoço",
                "jantar",
                "refeicao",
                "refeição",
                "brunch",
                "cafe",
                "café",
                "casa de fado",
                "casas de fado",
                "fado house",
                "fado houses",
            ]
        ):
            signals.append("Restaurants")
        if any(
            term in query
            for term in [
                "hotel",
                "hotels",
                "hoteis",
                "hotéis",
                "accommodation",
                "lodging",
                "stay",
                "alojamento",
                "hostel",
                "hostels",
                "guest house",
                "guest houses",
                "pousada",
                "pousadas",
                "apartamento",
                "apartamentos",
            ]
        ):
            signals.append("Hotels")
        if any(term in query for term in ["viewpoint", "view point", "miradouro", "scenic view"]):
            signals.append("View Points")
        if any(term in query for term in ["garden", "jardim", "parque", "park"]):
            signals.append("Parks & Gardens")
        if any(
            term in query
            for term in [
                "tourist office",
                "tourist offices",
                "posto de turismo",
                "postos de turismo",
                "informação turística",
                "informacao turistica",
            ]
        ):
            signals.append("Tourist Offices")
        if any(
            term in query
            for term in [
                "shopping",
                "shop",
                "shops",
                "loja",
                "lojas",
                "compras",
                "centro comercial",
                "centros comerciais",
                "mall",
                "malls",
            ]
        ):
            signals.append("Shopping")
        if any(term in query for term in ["cruise", "cruises", "cruzeiro", "cruzeiros", "boat tour", "river tour"]):
            signals.append("Tejo Cruises")
        if any(term in query for term in ["beach", "beaches", "praia", "praias", "surf"]):
            signals.append("Beaches")
        if any(term in query for term in ["golf", "golfe", "campo de golfe", "campos de golfe"]):
            signals.append("Golf")
        elif any(term in query for term in ["running", "corrida"]):
            signals.append("Running")
        elif any(term in query for term in ["sports", "sport", "desporto", "desportos"]):
            signals.append("Sports")
        fado_venue_request = bool(
            re.search(
                r"\b(?:casas?\s+de\s+fado|fado\s+houses?|bar(?:es)?\s+de\s+fado|fado\s+venues?)\b",
                query,
                flags=re.IGNORECASE,
            )
        )
        if "fado" in query and ("Restaurants" not in signals or fado_venue_request):
            signals.append("Fado")
        elif any(term in query for term in ["nightlife", "bar", "bars", "vida noturna", "vida nocturna"]):
            signals.append("Nightlife")

        return signals

    @classmethod
    def _infer_place_category_hint(cls, user_message: str) -> Optional[str]:
        """Return a single category hint only when intent is unambiguous.

        Mixed-category requests (e.g., one museum + one viewpoint) skip category filtering
        so retrieval remains broader and can return the requested mix.
        """
        signals = cls._infer_place_category_signals(user_message)
        if (
            "Restaurants" in signals
            and _RESEARCHER_FOOD_INTENT_RE.search(user_message or "")
            and len(signals) == 1
        ):
            return "Restaurants"
        if len(signals) != 1:
            return None
        return signals[0]

    @classmethod
    def _is_generic_research_discovery_query(cls, user_message: str) -> bool:
        """Detect broad, non-specific Lisbon discovery prompts that still need tool-backed output."""
        normalized = cls._normalize_for_deterministic_routing(user_message)
        if "lisbon" not in normalized and "lisboa" not in normalized:
            return False
        discovery_markers = [
            "tell me",
            "what should",
            "what can i do",
            "what can i recommend",
            "recommend something",
            "something useful",
            "some useful",
            "ideas",
            "what is worth",
            "worth seeing",
            "what to do",
            "what to see",
            "places to go",
            "places i should",
            "suggestions",
            "good options",
        ]
        return any(marker in normalized for marker in discovery_markers)

    @staticmethod
    def _build_places_source_line(result: str, language: str) -> str:
        """Builds the right source line for direct place lookups, including hybrid open-data results."""
        if (
            "Open Data:" in result
            or "Lisboa Aberta:" in result
            or re.search(r"\b[1-9]\d*\s+from Lisboa Aberta\b", result or "")
        ):
            if language == "pt":
                return "📌 **Fonte:** [*VisitLisboa Locais*](https://www.visitlisboa.com/pt-pt/locais) e [*Lisboa Aberta*](https://dados.cm-lisboa.pt/)"
            return "📌 **Source:** [*VisitLisboa Places*](https://www.visitlisboa.com/en/places) and [*Lisboa Aberta*](https://dados.cm-lisboa.pt/)"

        if language == "pt":
            return "📌 **Fonte:** [*VisitLisboa Locais*](https://www.visitlisboa.com/pt-pt/locais)"
        return "📌 **Source:** [*VisitLisboa Places*](https://www.visitlisboa.com/en/places)"

    @staticmethod
    def _build_events_source_line(language: str) -> str:
        """Builds the right source line for direct event lookups."""
        if language == "pt":
            return "📌 **Fonte:** [*VisitLisboa Eventos*](https://www.visitlisboa.com/pt-pt/eventos)"
        return "📌 **Source:** [*VisitLisboa Events*](https://www.visitlisboa.com/en/events)"

    @staticmethod
    def _extract_first_area_after_pattern(user_message: str, pattern: str) -> Optional[str]:
        """Extract a compact area fragment following a component-specific pattern."""
        match = re.search(
            pattern
            + r".{0,50}?\b(?:em|no|na|in|near|perto\s+de|perto\s+do|perto\s+da)\s+"
            r"(?P<area>[A-ZÀ-Ýa-zà-ÿ0-9][A-ZÀ-Ýa-zà-ÿ0-9 '\-/]{1,60})",
            user_message or "",
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        area = re.sub(r"\s+", " ", match.group("area")).strip(" .?!,;:")
        area = re.sub(
            r"\s+(?:e|and|com|with|para|for|perto|near)\b.*$",
            "",
            area,
            flags=re.IGNORECASE,
        ).strip(" .?!,;:")
        return area or None

    @staticmethod
    def _extract_primary_near_area(user_message: str) -> Optional[str]:
        """Extract the first explicit nearby area for component-level searches."""
        match = re.search(
            r"\b(?:perto\s+de|perto\s+do|perto\s+da|near)\s+"
            r"(?P<area>[A-ZÀ-Ýa-zà-ÿ0-9][A-ZÀ-Ýa-zà-ÿ0-9 '\-/]{1,60})",
            user_message or "",
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        area = re.sub(r"\s+", " ", match.group("area")).strip(" .?!,;:")
        area = re.sub(
            r"\s+(?:e|and|com|with|para|for|miradouro|bar|casas?\s+de\s+fado)\b.*$",
            "",
            area,
            flags=re.IGNORECASE,
        ).strip(" .?!,;:")
        return area or None

    @classmethod
    def _build_multi_place_component_specs(cls, user_message: str, language: str) -> list[dict[str, str]]:
        """Build component-level place searches for explicit multi-place requests."""
        signals = list(dict.fromkeys(cls._infer_place_category_signals(user_message)))
        if len(signals) < 2:
            return []

        is_pt = (language or "").lower().startswith("pt")
        near_area = cls._extract_primary_near_area(user_message)
        fado_area = cls._extract_first_area_after_pattern(
            user_message,
            r"\b(?:casas?\s+de\s+fado|fado\s+houses?|bar(?:es)?\s+de\s+fado|fado)\b",
        )
        specs: list[dict[str, str]] = []

        def add(category: str, label_pt: str, label_en: str, query: str) -> None:
            specs.append({
                "category": category,
                "label": label_pt if is_pt else label_en,
                "query": query.strip() or user_message,
            })

        if "Restaurants" in signals:
            area = near_area or cls._extract_place_area_filter(user_message)
            add(
                "Restaurants",
                "Restaurante / refeição",
                "Restaurant / meal",
                f"restaurante jantar barato perto de {area}" if area else user_message,
            )
        if "View Points" in signals:
            area = near_area or cls._extract_place_area_filter(user_message)
            add(
                "View Points",
                "Miradouro",
                "Viewpoint",
                f"miradouro perto de {area}" if area else user_message,
            )
        if "Fado" in signals:
            add(
                "Fado",
                "Fado",
                "Fado",
                f"bar de fado em {fado_area}" if fado_area else user_message,
            )
        if "Parks & Gardens" in signals:
            area = near_area or cls._extract_place_area_filter(user_message)
            add(
                "Parks & Gardens",
                "Jardim / parque",
                "Garden / park",
                f"jardim perto de {area}" if area else user_message,
            )

        return specs if len(specs) >= 2 else []

    def _run_multi_component_place_lookup(self, places_tool: Any, user_message: str, language: str) -> str:
        """Run separate grounded searches for explicit multi-component place requests."""
        specs = self._build_multi_place_component_specs(user_message, language)
        if not specs:
            return ""

        is_pt = (language or "").lower().startswith("pt")
        title = "### 🔎 **Locais pedidos**" if is_pt else "### 🔎 **Requested places**"
        direct = (
            "✅ **Resposta direta:** procurei separadamente cada componente pedido para não reduzir o pedido a uma só categoria."
            if is_pt
            else "✅ **Direct answer:** I searched each requested component separately instead of reducing the request to one category."
        )
        sections: list[str] = [title, "", direct, "", "---"]
        used_material = False
        for spec in specs:
            args = {
                "query": spec["query"],
                "category": spec["category"],
                "max_results": 1,
                "offset": 0,
                "language": language,
            }
            result = str(self._invoke_tool(places_tool, args, tool_name="search_places_attractions")).strip()
            result = re.sub(r"(?mi)^ðŸ“Œ\s*\*\*(?:Fonte|Source):\*\*.*$", "", result).strip()
            sections.extend(["", f"### **{spec['label']}**", ""])
            if result and not result.startswith(("Error:", "\u274c")) and not re.match(r"(?i)^n[aã]o (?:foram|foi|encontrei)", result):
                used_material = True
                sections.append(result)
            else:
                sections.append(
                    f"- **Limitação:** não encontrei um resultado confirmado para **{spec['label']}** com os filtros pedidos."
                    if is_pt
                    else f"- **Limitation:** I did not find a confirmed result for **{spec['label']}** with the requested filters."
                )

        sections.extend(["", self._build_places_source_line("\n".join(sections), language)])
        return "\n".join(sections).strip() if used_material else "\n".join(sections).strip()

    @staticmethod
    def _has_specific_lookup_fallback_intro(result: str) -> bool:
        """Return whether a tool result starts with an exact-match miss plus alternatives."""
        normalized = unicodedata.normalize("NFKD", str(result or "")).encode("ascii", "ignore").decode("ascii").lower()
        return any(
            marker in normalized
            for marker in [
                "nao encontrei um evento especifico com o nome",
                "nao encontrei um local especifico com o nome",
                "i could not find a specific event named",
                "i could not find a specific place named",
            ]
        )

    @staticmethod
    def _extract_pagination_request(user_message: str) -> Optional[dict]:
        """Extracts a simple next-more pagination intent from a follow-up query."""
        query = (user_message or "").lower().strip()
        if not query:
            return None

        normalized_query = unicodedata.normalize("NFKD", query)
        normalized_query = normalized_query.encode("ascii", "ignore").decode("ascii").lower()
        nearest_patterns = [
            r"\bmais\s+pert[oa]s?\b",
            r"\bmais\s+proxim[oa]s?\b",
            r"\bnearest\b",
            r"\bclosest\b",
            r"\bnearby\b",
            r"\bperto\s+d[eo]\b",
        ]
        if any(re.search(pattern, normalized_query) for pattern in nearest_patterns):
            return None

        explicit_pagination_patterns = [
            r"\b(?:next|following|another)\s+(?:\d{1,2}\s+)?(?:results?|places?|events?|options?)\b",
            r"\bmore\s+(?:\d{1,2}\s+)?(?:results?|places?|events?|options?)\b",
            r"\b(?:mais|proxim[oa]s?|seguintes?)\s+(?:(?:\d{1,2}|um|uma|dois|duas|tr[eê]s|tres|quatro|cinco|seis|sete|oito|nove|dez)\s+)?(?:resultados?|locais?|eventos?|op[cç][oõ]es|opcoes)\b",
            r"\b(?:mostra|mostre|d[aá]-me|quero)\s+mais\b",
            r"\b(?:outro|outra|outros|outras)\s+(?:(?:\d{1,2}|um|uma|dois|duas|tr[eê]s|tres|quatro|cinco|seis|sete|oito|nove|dez)\s+)?(?:resultados?|locais?|eventos?|op[cç][oõ]es|opcoes|atra[cç][oõ]es|atracoes|restaurantes?|museus?)\b",
            r"^\s*(?:outro|outra|outros|outras|mais\s+um|mais\s+uma|another|one\s+more)\s*$",
        ]
        if not any(re.search(pattern, normalized_query) for pattern in explicit_pagination_patterns):
            return None

        explicit_window_hint = True
        generic_more_hint = any(token in normalized_query for token in ["more", "mais"])
        result_nouns = [
            "result", "results", "event", "events", "evento", "eventos",
            "place", "places", "local", "locais", "attraction", "attractions",
            "atração", "atrações", "atracao", "atracoes",
            "option", "options", "opção", "opções", "opcao", "opcoes",
            "museum", "museums", "museu", "museus", "hospital", "hospitals",
            "restaurant", "restaurants", "restaurante", "restaurantes",
        ]
        if not explicit_window_hint and not (generic_more_hint and any(noun in normalized_query for noun in result_nouns)):
            return None

        word_to_number = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "um": 1,
            "uma": 1,
            "dois": 2,
            "duas": 2,
            "três": 3,
            "tres": 3,
            "quatro": 4,
            "cinco": 5,
            "seis": 6,
            "sete": 7,
            "oito": 8,
            "nove": 9,
            "dez": 10,
        }

        explicit_count: Optional[int] = None
        number_match = re.search(r"\b(\d{1,2})\b", query)
        if number_match:
            explicit_count = max(1, int(number_match.group(1)))
        else:
            for word, value in word_to_number.items():
                if re.search(rf"\b{re.escape(word)}\b", query):
                    explicit_count = value
                    break

        return {"count": explicit_count}

    @staticmethod
    def _infer_search_domain_from_query(user_message: str) -> Optional[str]:
        """Infers whether a follow-up refers to events or places."""
        query = (user_message or "").lower()
        event_terms = [
            "event", "events", "evento", "eventos", "concert", "concerto",
            "festival", "exhibition", "exposição", "exposicao", "show", "music",
            "música", "musica", "theatre", "teatro", "dance", "dança", "danca",
            "desporto", "desportivo", "desportivos", "sports", "sport",
            "arraial", "arraiais", "arrais", "santos populares", "marchas populares",
            "festas de lisboa",
        ]
        place_terms = [
            "place", "places", "local", "locais", "museum", "museu", "monument",
            "monumento", "attraction", "attractions", "restaurant", "restaurante",
            "hotel", "viewpoint", "miradouro", "pharmacy", "farmácia", "farmacia",
            "hospital", "library", "biblioteca", "park", "jardim", "garden",
        ]
        if any(term in query for term in event_terms):
            return "events"
        if any(term in query for term in place_terms):
            return "places"
        return None

    @staticmethod
    def _is_named_lookup_followup(user_message: str) -> bool:
        """Detects short follow-ups such as 'what about "Book Fair"?' or 'e do MAAT?'."""
        query = (user_message or "").strip()
        if not query:
            return False

        query_lower = query.lower()
        tokens = re.findall(r"[a-zA-ZÀ-ÿ0-9]+", query)
        if any(symbol in query for symbol in ['"', '“', '”']):
            return True
        return len(tokens) <= 10 and any(
            marker in query_lower
            for marker in [
                "what about", "how about", "tell me about", "more about",
                "sobre", "fala-me de", "fala me de", "e do", "e da", "e o", "e a",
            ]
        )

    @staticmethod
    def _count_ranked_results(result: str) -> int:
        """Counts item cards in raw VisitLisboa-style outputs."""
        text = str(result or "")
        numbered_count = len(re.findall(r"(?m)^\s*\d+\.\s+", text))
        if numbered_count:
            return numbered_count
        bullet_card_count = len(re.findall(r"(?m)^\s*[-*]\s+\*\*[^\n*]+\*\*", text))
        if bullet_card_count:
            return bullet_card_count
        return len(re.findall(r"(?m)^\s*\*\*[^\n*]+\*\*\s*$", text))

    @staticmethod
    def _trim_ranked_result_cards(result: str, max_cards: int) -> str:
        """Keep only the first requested item cards in a Markdown result."""
        text = str(result or "")
        if max_cards <= 0 or not text.strip():
            return text

        category_only_labels = {
            "attractions",
            "compras",
            "exposicoes",
            "exposições",
            "food & dining",
            "locais e atracoes",
            "locais e atrações",
            "museums & monuments",
            "museus e monumentos",
            "restaurants",
            "restaurantes",
            "shopping",
        }
        card_start_re = re.compile(r"^\s*(?:[-*]\s+)?\*\*[^*\n]+?\*\*\s*$")

        def is_item_card_heading(line: str) -> bool:
            if not card_start_re.match(line.strip()):
                return False
            label = re.sub(r"^\s*(?:[-*]\s+)?\*\*", "", line.strip())
            label = re.sub(r"\*\*\s*$", "", label).strip()
            normalized = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode("ascii")
            normalized = re.sub(r"[^a-zA-Z0-9& ]+", " ", normalized).strip().lower()
            normalized = re.sub(r"\s+", " ", normalized)
            return normalized not in category_only_labels

        lines = text.splitlines()
        first_card_index = next(
            (index for index, line in enumerate(lines) if is_item_card_heading(line)),
            None,
        )
        if first_card_index is None:
            return text

        intro = lines[:first_card_index]
        blocks: List[List[str]] = []
        tail: List[str] = []
        current_block: List[str] = []
        for line in lines[first_card_index:]:
            stripped = line.strip()
            if stripped.startswith("📌"):
                if current_block:
                    blocks.append(current_block)
                    current_block = []
                tail.append(line)
                continue
            if is_item_card_heading(line):
                if current_block:
                    blocks.append(current_block)
                current_block = [line]
                continue
            if current_block:
                current_block.append(line)
            else:
                tail.append(line)
        if current_block:
            blocks.append(current_block)

        if len(blocks) <= max_cards:
            return text

        kept_lines: List[str] = [*intro]
        for block in blocks[:max_cards]:
            kept_lines.extend(block)
        kept_lines.extend(tail)
        return "\n".join(kept_lines).strip()

    def _remember_search_context(
        self,
        *,
        domain: str,
        tool_name: str,
        base_args: dict,
        page_size: int,
        shown_count: int,
        language: str,
        source_query: str,
        offset: int = 0,
    ) -> None:
        """Stores enough context to continue a prior event/place result batch."""
        safe_page_size = max(1, int(page_size or 5))
        safe_offset = max(0, int(offset or 0))
        safe_shown = max(0, int(shown_count or 0))
        self._last_search_context = {
            "domain": domain,
            "tool_name": tool_name,
            "base_args": deepcopy(base_args),
            "page_size": safe_page_size,
            "offset": safe_offset,
            "next_offset": safe_offset + safe_shown,
            "language": language,
            "source_query": source_query,
        }

    @staticmethod
    def _localize_continued_search_filter_token(value: str, language: str) -> str:
        """Localizes common event/place filter tokens in pagination summaries."""
        token = str(value or "").strip()
        if not token:
            return ""
        normalized = unicodedata.normalize("NFKD", token).encode("ascii", "ignore").decode("ascii").lower()
        pt_map = {
            "this week": "esta semana",
            "next week": "próxima semana",
            "this weekend": "este fim de semana",
            "next weekend": "próximo fim de semana",
            "today": "hoje",
            "tomorrow": "amanhã",
            "music": "Música",
            "exhibitions": "Exposições",
            "theater": "Teatro",
            "theatre": "Teatro",
            "dance": "Dança",
            "cinema": "Cinema",
            "sports": "Desporto",
            "fairs": "Feiras",
            "festivals": "Festivais",
            "gastronomy": "Gastronomia",
            "main events": "Grandes eventos",
            "museums & monuments": "Museus e monumentos",
            "restaurants": "Restaurantes",
        }
        en_map = {
            "esta semana": "this week",
            "proxima semana": "next week",
            "este fim de semana": "this weekend",
            "proximo fim de semana": "next weekend",
            "hoje": "today",
            "amanha": "tomorrow",
            "musica": "Music",
            "exposicoes": "Exhibitions",
            "teatro": "Theater",
            "danca": "Dance",
            "desporto": "Sports",
            "feiras": "Fairs",
            "festivais": "Festivals",
            "gastronomia": "Gastronomy",
            "museus e monumentos": "Museums & Monuments",
            "restaurantes": "Restaurants",
        }
        mapping = pt_map if language == "pt" else en_map
        return mapping.get(normalized, token)

    @staticmethod
    def _describe_continued_search_filter(base_args: dict, language: str) -> str:
        """Builds a compact user-facing description of the previous search filter."""
        labels: list[str] = []
        date_filter = str(base_args.get("date_filter") or "").strip()
        category = str(base_args.get("category") or "").strip()
        exclude_categories = str(base_args.get("exclude_categories") or "").strip()
        query = str(base_args.get("query") or "").strip()
        service_type = str(base_args.get("service_type") or "").strip()
        if date_filter:
            labels.append(date_filter)
        if category:
            labels.append(category)
        if exclude_categories:
            for excluded_category in re.split(r"[,;/|]+", exclude_categories):
                localized_exclusion = ResearcherAgent._localize_continued_search_filter_token(
                    excluded_category.strip(),
                    language,
                )
                if localized_exclusion:
                    labels.append(
                        f"sem {localized_exclusion}"
                        if language == "pt"
                        else f"excluding {localized_exclusion}"
                    )
        if service_type:
            labels.append(service_type)
        if query:
            labels.append(query[:120])
        if not labels:
            return "pesquisa anterior" if language == "pt" else "previous search"
        localized = [
            ResearcherAgent._localize_continued_search_filter_token(label, language)
            for label in labels
        ]
        return ", ".join(dict.fromkeys(label for label in localized if label))

    @classmethod
    def _build_no_more_pagination_response(
        cls,
        *,
        domain: str,
        base_args: dict,
        language: str,
        source_line: str,
    ) -> str:
        """Builds a grounded continuation answer when the next page has no cards."""
        filter_text = cls._describe_continued_search_filter(base_args, language)
        if language == "pt":
            heading = "### 🔵 **Eventos encontrados**" if domain == "events" else "### 🔵 **Locais e atrações**"
            noun = "eventos" if domain == "events" else "locais"
            body = [
                heading,
                "",
                f"✅ **Resposta direta:** não encontrei mais {noun} confirmados para continuar esta lista com o filtro anterior.",
                "",
                "---",
                "",
                f"🧭 **Filtro mantido:** {filter_text}",
                "⚠️ A página seguinte da fonte não devolveu novos resultados confirmados; por isso não inventei resultados.",
                "",
                source_line,
            ]
            return "\n".join(body).strip()

        heading = "### 🔵 **Events Found**" if domain == "events" else "### 🔵 **Places and Attractions**"
        noun = "events" if domain == "events" else "places"
        body = [
            heading,
            "",
            f"✅ **Direct answer:** I did not find more confirmed {noun} to continue this list with the previous filter.",
            "",
            "---",
            "",
            f"🧭 **Filter kept:** {filter_text}",
            "⚠️ The next source page did not return new confirmed results, so I did not invent results.",
            "",
            source_line,
        ]
        return "\n".join(body).strip()

    def _maybe_continue_previous_search(self, user_message: str, language: str) -> Optional[str]:
        """Continue the last event/place search when the user asks for more results."""
        pagination_request = self._extract_pagination_request(user_message)
        if not pagination_request or not self._last_search_context:
            self._pending_pagination_replay = None
            return None

        normalized_message = (user_message or "").strip().lower()
        pending_replay = getattr(self, "_pending_pagination_replay", None)
        if pending_replay and pending_replay.get("message") == normalized_message:
            self._pending_pagination_replay = None
            return str(pending_replay.get("response") or "").strip() or None
        self._pending_pagination_replay = None

        explicit_domain = self._infer_search_domain_from_query(user_message)
        cached_domain = str(self._last_search_context.get("domain") or "").strip()
        if explicit_domain and explicit_domain != cached_domain:
            return None

        tool_name = str(self._last_search_context.get("tool_name") or "").strip()
        tool = self._get_tool_by_name(tool_name)
        if not tool:
            return None

        count = max(1, int(pagination_request.get("count") or self._last_search_context.get("page_size") or 5))
        offset = max(0, int(self._last_search_context.get("next_offset") or 0))
        base_args = deepcopy(self._last_search_context.get("base_args") or {})
        if cached_domain == "events":
            additional_exclusions = self._extract_excluded_event_categories(user_message)
            if additional_exclusions:
                existing_exclusions = [
                    value.strip()
                    for value in re.split(r"[,;/|]+", str(base_args.get("exclude_categories") or ""))
                    if value.strip()
                ]
                merged_exclusions = list(dict.fromkeys([*existing_exclusions, *additional_exclusions]))
                base_args["exclude_categories"] = ", ".join(merged_exclusions)
        args = {**base_args, "max_results": count, "offset": offset}

        result = str(self._invoke_tool(tool, args, tool_name=tool_name)).strip()
        shown_count = self._count_ranked_results(result)
        self._remember_search_context(
            domain=cached_domain,
            tool_name=tool_name,
            base_args=base_args,
            page_size=count,
            shown_count=shown_count,
            language=language,
            source_query=str(self._last_search_context.get("source_query") or user_message),
            offset=offset,
        )

        if cached_domain == "events":
            source_line = self._build_events_source_line(language)
        else:
            source_line = self._build_places_source_line(result, language)
        if shown_count == 0:
            response = self._build_no_more_pagination_response(
                domain=cached_domain,
                base_args=base_args,
                language=language,
                source_line=source_line,
            )
        else:
            response = f"{result}\n\n{source_line}".strip()
        self._pending_pagination_replay = {
            "message": normalized_message,
            "response": response,
        }
        return response

    @staticmethod
    def _is_broad_attractions_query(user_message: str) -> bool:
        """Detects broad attraction-list queries that should bypass free-form synthesis."""
        query = (user_message or "").lower()
        if re.search(r"\b(?:near|nearby|around|perto\s+d[eoa]?|junto\s+d[eoa]?)\b", query):
            return False
        attraction_phrases = [
            "atrações imperdíveis",
            "atracoes imperdiveis",
            "locais imperdíveis",
            "locais imperdiveis",
            "sítios imperdíveis",
            "sitios imperdiveis",
            "atrações",
            "atracoes",
            "must-see",
            "must see",
            "first time",
            "first visit",
            "primeira vez",
            "primeira visita",
            "top attractions",
            "highly recommended attractions",
            "main attractions",
            "o que visitar",
            "what should i visit",
        ]
        planning_terms = ["itinerary", "roteiro", "plan", "plano", "schedule", "agenda"]
        return any(phrase in query for phrase in attraction_phrases) and not any(
            term in query for term in planning_terms
        )

    @staticmethod
    def _mentions_place_or_attraction_intent(user_message: str) -> bool:
        """Return whether the query asks for places or attractions."""
        normalized = _normalize_researcher_intent_text(user_message)
        return bool(
            re.search(
                r"\b(?:atrac(?:ao|oes)|attraction|attractions|places?|locais?|local|"
                r"monumentos?|monuments?|museus?|museums?|visitar|visit|sights?|"
                r"pontos?\s+de\s+interesse)\b",
                normalized,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _is_area_mixed_food_place_query(user_message: str) -> bool:
        """Return whether the query asks for food and places in the same area."""
        return bool(
            _RESEARCHER_FOOD_INTENT_RE.search(user_message or "")
            and ResearcherAgent._mentions_place_or_attraction_intent(user_message)
        )

    @staticmethod
    def _extract_area_scoped_card_blocks(markdown: str, area_label: str) -> List[str]:
        """Extract formatted place cards whose fields mention the requested area.

        This prevents area-scoped requests such as "restaurants in Setúbal" from
        being filled with good but geographically irrelevant Lisbon results.
        """
        area_key = normalize_scope_text(area_label)
        if not area_key:
            return []

        card_start_re = re.compile(r"^\s*(?:[-*]\s+)?\*\*[^*\n]+?\*\*\s*$")
        blocks: List[List[str]] = []
        current_block: List[str] = []
        for line in str(markdown or "").splitlines():
            if card_start_re.match(line):
                if current_block:
                    blocks.append(current_block)
                current_block = [line]
                continue
            if current_block:
                current_block.append(line)
        if current_block:
            blocks.append(current_block)

        scoped_blocks: List[str] = []
        for block_lines in blocks:
            block = "\n".join(block_lines).strip()
            if area_key in normalize_scope_text(block):
                scoped_blocks.append(block)
        return scoped_blocks

    @staticmethod
    def _build_area_no_results_line(area_label: str, category_label: str, language: str) -> str:
        """Build a compact limitation when an area-scoped category has no cards."""
        if language == "pt":
            return f"- ⚠️ Não encontrei {category_label} confirmados em **{area_label}** nos dados disponíveis."
        return f"- ⚠️ I did not find confirmed {category_label} in **{area_label}** in the available data."

    def _run_area_scoped_mixed_place_lookup(
        self,
        places_tool: Any,
        user_message: str,
        area_label: str,
        language: str,
    ) -> str:
        """Run separate area-scoped attraction and restaurant searches."""
        if language == "pt":
            attraction_query = f"atrações em {area_label}"
            restaurant_query = f"restaurantes em {area_label}"
            title = f"### 📍 **Locais em {area_label}**"
            attractions_heading = "### 🏛️ **Atrações confirmadas**"
            restaurants_heading = "### 🍽️ **Restaurantes confirmados**"
            attraction_label = "atrações"
            restaurant_label = "restaurantes"
        else:
            attraction_query = f"attractions in {area_label}"
            restaurant_query = f"restaurants in {area_label}"
            title = f"### 📍 **Places in {area_label}**"
            attractions_heading = "### 🏛️ **Confirmed attractions**"
            restaurants_heading = "### 🍽️ **Confirmed restaurants**"
            attraction_label = "attractions"
            restaurant_label = "restaurants"

        attraction_args = {
            "query": attraction_query,
            "category": "Museums & Monuments",
            "max_results": 6,
            "offset": 0,
            "language": language,
        }
        restaurant_args = {
            "query": restaurant_query,
            "category": "Restaurants",
            "max_results": 6,
            "offset": 0,
            "language": language,
        }
        attraction_result = str(
            self._invoke_tool(places_tool, attraction_args, tool_name="search_places_attractions")
        ).strip()
        restaurant_result = str(
            self._invoke_tool(places_tool, restaurant_args, tool_name="search_places_attractions")
        ).strip()

        attraction_cards = self._extract_area_scoped_card_blocks(attraction_result, area_label)
        restaurant_cards = self._extract_area_scoped_card_blocks(restaurant_result, area_label)
        attraction_count = len(attraction_cards)
        restaurant_count = len(restaurant_cards)

        if language == "pt":
            summary_parts = []
            if attraction_count == 1:
                summary_parts.append("1 atração confirmada")
            elif attraction_count:
                summary_parts.append(f"{attraction_count} atrações confirmadas")
            else:
                summary_parts.append("sem atrações confirmadas")
            if restaurant_count == 1:
                summary_parts.append("1 restaurante confirmado")
            elif restaurant_count:
                summary_parts.append(f"{restaurant_count} restaurantes confirmados")
            else:
                summary_parts.append("sem restaurantes confirmados")
            direct = f"encontrei {', '.join(summary_parts)} em **{area_label}** nos dados disponíveis."
        else:
            summary_parts = []
            if attraction_count == 1:
                summary_parts.append("1 confirmed attraction")
            elif attraction_count:
                summary_parts.append(f"{attraction_count} confirmed attractions")
            else:
                summary_parts.append("no confirmed attractions")
            if restaurant_count == 1:
                summary_parts.append("1 confirmed restaurant")
            elif restaurant_count:
                summary_parts.append(f"{restaurant_count} confirmed restaurants")
            else:
                summary_parts.append("no confirmed restaurants")
            direct = f"I found {', '.join(summary_parts)} in **{area_label}** in the available data."

        attraction_section = "\n\n".join(attraction_cards) if attraction_cards else self._build_area_no_results_line(area_label, attraction_label, language)
        restaurant_section = "\n\n".join(restaurant_cards) if restaurant_cards else self._build_area_no_results_line(area_label, restaurant_label, language)
        source_line = self._build_places_source_line(f"{attraction_result}\n{restaurant_result}", language)

        return "\n".join(
            [
                title,
                "",
                ("✅ **Resposta direta:** " if language == "pt" else "✅ **Direct answer:** ") + direct,
                "",
                "---",
                "",
                attractions_heading,
                "",
                attraction_section,
                "",
                restaurants_heading,
                "",
                restaurant_section,
                "",
                source_line,
            ]
        ).strip()

    def _filter_area_scoped_place_result(
        self,
        result: str,
        area_label: str,
        language: str,
        category_label: str,
    ) -> str:
        """Keep only cards that match the requested AML municipality."""
        scoped_cards = self._extract_area_scoped_card_blocks(result, area_label)
        if scoped_cards:
            return "\n\n".join(scoped_cards).strip()

        title = f"### 📍 **Locais em {area_label}**" if language == "pt" else f"### 📍 **Places in {area_label}**"
        line = self._build_area_no_results_line(area_label, category_label, language)
        return f"{title}\n\n{line}".strip()

    @staticmethod
    def _is_direct_event_lookup_query(user_message: str) -> bool:
        """Detects event-discovery queries that are safer to answer directly from tools."""
        query = (user_message or "").lower()
        normalized_query = ResearcherAgent._normalize_event_preference_text(user_message)
        if any(term in query for term in ["history", "história", "historia"]):
            return False
        if ResearcherAgent._is_direct_place_lookup_query(user_message):
            return False

        specific_lookup = _extract_specific_event_lookup_phrase(user_message)
        planning_terms = [
            "plan", "plano", "roteiro", "itinerary", "agenda",
            "combine", "combinar", "day plan", "plan my day",
        ]
        event_terms = [
            "event", "events", "evento", "eventos", "concert", "concerto", "concertos",
            "festival", "festivals", "festivais", "exhibition", "exposição", "exposicao",
            "exposições", "exposicoes",
            "music", "música", "musica", "show", "theatre", "teatro",
            "dance", "dança", "danca", "cinema", "what's on", "o que há", "o que ha",
            "fair", "fairs", "feira", "feiras", "book fair",
            "summit", "conference", "congress", "forum", "expo", "games week", "gaming",
            "sport", "sports", "desporto", "desportivo", "desportivos",
            "arraial", "arraiais", "arrais", "santos populares", "marchas populares",
            "festas de lisboa", "gastronomia", "gastronomy", "família", "familia",
            "famílias", "familias", "crianças", "criancas", "kids", "children",
        ]
        named_lookup_markers = [
            "tell me about", "what about", "more about", "details about", "information about",
            "sobre", "fala-me de", "fala me de", "fala-me do", "fala me do", "fala-me da", "fala me da",
            "diz-me sobre", "diz me sobre", "diz-me do", "diz me do", "diz-me da", "diz me da",
            "e do", "e da",
        ]
        category_discovery_request = bool(
            ResearcherAgent._infer_event_category_hint(user_message)
            and (
                ResearcherAgent._extract_event_date_filters(user_message)
                or re.search(
                    r"\b(?:mostra|mostrar|lista|listar|diz\s+me|dizme|quais|que|show|list|give|find|procura|procurar)\b",
                    normalized_query,
                    flags=re.IGNORECASE,
                )
            )
        )
        return (
            (
                bool(specific_lookup)
                and not ResearcherAgent._is_direct_place_lookup_query(user_message)
            )
            or
            any(term in query for term in event_terms)
            or
            category_discovery_request
            or (
                any(marker in query for marker in named_lookup_markers)
                and ResearcherAgent._infer_event_category_hint(user_message) is not None
            )
            or (
                any(symbol in user_message for symbol in ['"', '“', '”'])
                and ResearcherAgent._infer_event_category_hint(user_message) is not None
            )
        ) and not any(
            term in query for term in planning_terms
        )

    @staticmethod
    def _is_mixed_event_place_query(user_message: str) -> bool:
        """Detects mixed queries that ask for both places and events in the same turn."""
        query = (user_message or "").lower()
        normalized_query = ResearcherAgent._normalize_event_preference_text(user_message)
        event_terms = [
            "event", "events", "evento", "eventos", "concert", "concerto",
            "festival", "festivals", "exhibition", "exposição", "exposicao",
            "music", "música", "musica", "show", "theatre", "teatro",
            "dance", "dança", "danca", "cinema", "what's on", "o que há", "o que ha",
            "fair", "fairs", "feira", "feiras",
            "sport", "sports", "desporto", "desportivo", "desportivos",
            "arraial", "arraiais", "arrais", "santos populares", "marchas populares",
            "festas de lisboa",
        ]
        place_terms = [
            "museum", "museums", "museu", "museus",
            "restaurant", "restaurants", "restaurante", "restaurantes",
            "pharmacy", "pharmacies", "farmácia", "farmacias", "farmácias",
            "hospital", "hospitals", "attraction", "attractions", "place", "places",
            "monument", "monuments", "monumento", "monumentos",
        ]
        has_event_terms = any(term in query for term in event_terms)
        has_place_terms = any(term in query for term in place_terms)
        if not (has_event_terms and has_place_terms):
            return False

        place_term_re = (
            r"(?:museums?|museus?|restaurants?|restaurantes?|pharmacies|pharmacy|"
            r"farmacias?|hospitals?|attractions?|places?|monuments?|monumentos?)"
        )
        negated_place_re = (
            r"\b(?:sem|nao|not|without|avoid|excluding|except|menos|"
            r"nao\s+me\s+mostres?|nao\s+me\s+sugiras?|do\s+not\s+show|"
            r"don\s+t\s+show|do\s+not\s+suggest|don\s+t\s+suggest)\b"
            rf"(?:\s+\w+){{0,6}}\s+\b{place_term_re}\b"
        )
        stripped_query = re.sub(negated_place_re, " ", normalized_query, flags=re.IGNORECASE)
        stripped_query = re.sub(r"\bno\s+(?:more\s+)?museums?\b", " ", stripped_query, flags=re.IGNORECASE)
        positive_place_terms = [
            term for term in place_terms
            if re.search(rf"\b{re.escape(term)}\b", stripped_query, flags=re.IGNORECASE)
        ]
        return bool(positive_place_terms)

    @staticmethod
    def _extract_event_date_filter(user_message: str) -> Optional[str]:
        """Extracts a lightweight date filter for direct event tool lookups."""
        query = (user_message or "").lower()
        query_key = unicodedata.normalize("NFKD", query)
        query_key = query_key.encode("ascii", "ignore").decode("ascii")
        mappings = [
            (["this weekend", "este fim de semana", "fim de semana"], "this weekend"),
            (["next week", "próxima semana", "proxima semana"], "next week"),
            (["this week", "esta semana"], "this week"),
            (["tomorrow", "amanhã", "amanha"], "tomorrow"),
            (["today", "hoje"], "today"),
            (["next month", "próximo mês", "proximo mes"], "next month"),
            (["this month", "este mês", "este mes"], "this month"),
        ]
        for terms, date_filter in mappings:
            if any(term in query for term in terms):
                return date_filter
        weekday_mappings = [
            (["monday", "segunda feira", "segunda"], "monday"),
            (["tuesday", "terca feira", "terca"], "tuesday"),
            (["wednesday", "quarta feira", "quarta"], "wednesday"),
            (["thursday", "quinta feira", "quinta"], "thursday"),
            (["friday", "sexta feira", "sexta"], "friday"),
            (["saturday", "sabado"], "saturday"),
            (["sunday", "domingo"], "sunday"),
        ]
        for terms, date_filter in weekday_mappings:
            if any(re.search(rf"\b{re.escape(term)}\b", query_key) for term in terms):
                return date_filter
        return None

    @staticmethod
    def _is_category_date_event_discovery(user_message: str) -> bool:
        """Detects category/date event browsing rather than a named event lookup."""
        query = (user_message or "").lower()
        if not ResearcherAgent._extract_event_date_filters(user_message):
            return False
        if not ResearcherAgent._infer_event_category_hint(user_message):
            return False
        specific_lookup = _extract_specific_event_lookup_phrase(user_message)
        if specific_lookup and not ResearcherAgent._specific_event_lookup_is_category_noise(specific_lookup):
            return False
        named_lookup_markers = [
            "tell me about",
            "what about",
            "more about",
            "details about",
            "information about",
            "sobre ",
            "fala-me",
            "fala me",
        ]
        has_quoted_title = any(symbol in (user_message or "") for symbol in ['"', '“', '”'])
        return not has_quoted_title and not any(marker in query for marker in named_lookup_markers)

    @staticmethod
    def _is_category_event_discovery(user_message: str) -> bool:
        """Detect event category browsing even when the user does not name a date."""
        if not ResearcherAgent._infer_event_category_hint(user_message):
            return False
        specific_lookup = _extract_specific_event_lookup_phrase(user_message)
        if specific_lookup and not ResearcherAgent._specific_event_lookup_is_category_noise(specific_lookup):
            return False
        if any(symbol in (user_message or "") for symbol in ['"', '“', '”']):
            return False
        normalized = ResearcherAgent._normalize_event_preference_text(user_message)
        named_lookup_re = re.compile(
            r"\b(?:tell me about|details about|information about|"
            r"fala(?:-|\s)?me\s+(?:de|do|da)|sobre\s+(?:o|a|os|as)\s+\w{3,}|"
            r"diz(?:-|\s)?me\s+sobre)\b",
            flags=re.IGNORECASE,
        )
        if named_lookup_re.search(normalized):
            return False
        return bool(
            re.search(
                r"\b(?:eventos?|events?|todos?|todas?|all|lista|list|mostra|show|"
                r"quais|which|desporto|desportiv[oa]s?|sports?|musica|music|"
                r"teatro|exposicoes?|exhibitions?|festivais|festivals?|"
                r"gastronomia|gastronomic[oa]s?|culinari[oa]s?|food|wine|vinho|"
                r"familias?|family|kids|children|criancas|cinema|films?|movies?|"
                r"feiras?|fairs?|market|mercado|principais|main|summit|conference|"
                r"congress|forum|expo)\b",
                normalized,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _is_outdoor_event_query(user_message: str) -> bool:
        """Detects explicit outdoor-event discovery requests."""
        normalized_query = unicodedata.normalize("NFKD", user_message or "")
        normalized_query = normalized_query.encode("ascii", "ignore").decode("ascii").lower()
        if re.search(
            r"\b(?:sem|nao|no|not|without|avoid|evitar|excluir|excluding|except|menos)\b"
            r"(?:\s+\w+){0,6}\s+\b(?:outdoor|outdoors|open\s+air|outside|ao\s+ar\s+livre|ar\s+livre|exterior)\b"
            r"|\b(?:que\s+nao\s+sejam|que\s+nao\s+seja|not\s+outdoors?|not\s+outside)\b",
            normalized_query,
            flags=re.IGNORECASE,
        ):
            return False
        outdoor_terms = [
            "outdoor", "outdoors", "open air", "open-air", "outside",
            "ao ar livre", "ar livre", "exterior",
        ]
        event_terms = ["event", "events", "evento", "eventos", "festival", "concert", "concerto"]
        return any(term in normalized_query for term in outdoor_terms) and any(
            term in normalized_query for term in event_terms
        )

    @staticmethod
    def _extract_event_focus_query(user_message: str) -> Optional[str]:
        """Drops generic event phrasing so broad date-based event searches keep high recall."""
        specific_lookup = _extract_specific_event_lookup_phrase(user_message)
        if ResearcherAgent._specific_event_lookup_is_category_noise(specific_lookup):
            specific_lookup = None
        if specific_lookup:
            return specific_lookup

        quoted_match = re.search(r'"([^"\n]{2,120})"|“([^”\n]{2,120})”', user_message or "")
        if quoted_match:
            quoted_subject = next((group for group in quoted_match.groups() if group), "").strip(" .?!")
            if quoted_subject:
                return quoted_subject

        query = (user_message or "").lower()
        normalized_query = ResearcherAgent._normalize_event_preference_text(user_message)
        if re.search(
            r"\b(?:arrai(?:al|ais|s)?|santos populares|marchas populares|festas de lisboa)\b",
            normalized_query,
            flags=re.IGNORECASE,
        ):
            return "santos populares arraiais marchas populares festas de lisboa"
        if any(phrase in query for phrase in ["música ao vivo", "musica ao vivo", "live music"]):
            return "música ao vivo" if any(term in query for term in ["música", "musica"]) else "live music"

        named_lookup_markers = [
            "tell me about", "what about", "more about", "details about", "information about",
            "sobre", "fala-me de", "fala me de", "e do", "e da",
        ]
        if any(marker in query for marker in named_lookup_markers) and ResearcherAgent._infer_event_category_hint(user_message):
            subject = re.sub(
                r"\b(?:tell me about|what about|more about|details about|information about|sobre(?: o| a| os| as)?|fala me de|e do|e da)\b",
                " ",
                query,
            )
            subject = re.sub(r"\b(?:event|events|evento|eventos)\b", " ", subject)
            subject = re.sub(r"\s+", " ", subject).strip(" .?!")
            if subject:
                return subject

        specific_interest_terms = [
            "music", "música", "musica", "concert", "concerto", "concertos", "fado",
            "jazz", "rock", "pop", "festival", "festivais", "exhibition", "exposição",
            "exposicao", "theatre", "teatro", "dance", "dança", "danca", "cinema",
            "art", "arte", "family", "família", "familia", "kids", "children",
            "child", "miúdos", "miudos", "crianças", "criancas", "night",
            "nightlife", "evening", "noite", "food", "gastronomia",
            "sports", "desporto", "desportos", "market", "mercado", "fair", "feira",
            "outdoor", "outdoors", "open air", "open-air", "outside", "ar livre",
            "free", "gratuito", "gratuitos", "gratuita", "gratuitas", "gratis", "grátis",
        ]
        if not any(term in query for term in specific_interest_terms):
            return None

        generic_terms = {
            "que", "quais", "what", "which", "major", "great", "grandes", "large",
            "find", "search", "show", "mostrar", "mostra", "encontra", "encontre",
            "procura", "procure", "descobre", "discover",
            "event", "events", "evento", "eventos", "this", "week", "esta", "semana",
            "este",
            "today", "hoje", "tomorrow", "amanhã", "amanha", "next", "weekend",
            "fim", "de", "semana", "local", "locais", "culture", "cultura", "cultural",
            "culturais", "quero", "queria", "gostava", "mas", "but", "avoid", "evita",
            "evitar", "without", "sem", "excluding", "exclui", "excluir",
            "nem", "nor", "neither",
            "explore", "explorar", "lisbon", "lisboa", "temos", "there", "happening", "have",
            "algo", "interessante", "fazer", "para", "perto", "near", "around", "theres",
            "are", "should", "bring", "umbrella", "weather", "rain", "chuva",
        }
        tokens = [token for token in re.findall(r"[a-zA-ZÀ-ÿ0-9]+", query) if len(token) >= 3]
        meaningful_tokens = [token for token in tokens if token not in generic_terms]
        return " ".join(dict.fromkeys(meaningful_tokens)) if meaningful_tokens else None

    @staticmethod
    def _normalize_event_preference_text(text: str) -> str:
        """Normalize user preference text for category-inclusion/exclusion checks."""
        normalized = unicodedata.normalize("NFKD", text or "")
        normalized = normalized.encode("ascii", "ignore").decode("ascii").lower()
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def _specific_event_lookup_is_category_noise(phrase: Optional[str]) -> bool:
        """Return whether a supposed event title is only category/filter wording."""
        normalized = ResearcherAgent._normalize_event_preference_text(phrase or "")
        if not normalized:
            return False
        tokens = re.findall(r"[a-z0-9]+", normalized)
        if not tokens:
            return False
        category_noise = {
            "de", "do", "da", "dos", "das", "em", "in", "o", "a", "os", "as",
            "e", "and", "que", "what", "which", "event", "events", "evento", "eventos", "todos", "todas",
            "all", "not", "no", "nao", "sem", "without", "except",
            "excluding", "exclui", "excluir", "menos", "procura", "procurar", "encontra", "encontrar",
            "search", "find", "show", "mostrar", "mostra", "mostres", "lista", "list", "diz",
            "me", "sabes", "conheces", "podes", "dizer", "quero", "queria", "mas", "but",
            "music", "musica", "live", "vivo", "ao",
            "concert", "concerts", "concerto", "concertos", "sport",
            "sports", "desporto", "desportos", "desportivo", "desportiva",
            "desportivos", "desportivas", "festival", "festivals",
            "festivais", "teatro", "theatre", "theater", "danca", "dança",
            "dance", "exposicao",
            "exposicoes", "exhibition", "exhibitions", "categoria",
            "category", "tipo", "types", "kinds", "lisboa", "lisbon",
            "museum", "museums", "museu", "museus",
            "gastronomia", "gastronomico", "gastronomicos", "gastronomica",
            "gastronomicas", "gastronomy", "gastronomic", "culinaria",
            "culinario", "culinary", "food", "wine", "vinho", "familia",
            "familias", "family", "kids", "children", "child", "criancas",
            "cinema", "film", "films", "movie", "movies", "fair", "fairs",
            "feira", "feiras", "market", "mercado", "main", "principais",
            "principal", "summit", "conference", "congress", "forum", "expo",
            "este", "esta", "fim", "weekend", "mes", "month", "semana", "week", "hoje",
            "today", "amanha", "tomorrow", "junho", "june", "maio", "may",
            "para", "com", "perto", "near", "fado", "jazz", "rock", "pop",
        }
        return all(token in category_noise for token in tokens)

    @staticmethod
    def _event_category_key(category: str) -> str:
        """Return a comparable key for VisitLisboa event category labels."""
        normalized = ResearcherAgent._normalize_event_preference_text(category)
        aliases = {
            "musica": "music",
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
            "festival": "festivals",
            "festivais": "festivals",
            "teatro": "theater opera dance",
            "danca": "theater opera dance",
            "dance": "theater opera dance",
            "exposicao": "exhibitions",
            "exposicoes": "exhibitions",
            "feira": "fairs",
            "feiras": "fairs",
        }
        return aliases.get(normalized, normalized)

    @classmethod
    def _extract_excluded_event_categories(cls, user_message: str) -> list[str]:
        """Infer event categories explicitly excluded by the user."""
        normalized = cls._normalize_event_preference_text(user_message)
        if not normalized:
            return []
        normalized = re.sub(r"\bno\s+(?:more\s+)?museums?\b", " ", normalized, flags=re.IGNORECASE)

        category_terms = {
            "Fado": ("fado",),
            "Music": (
                "music", "musica", "concert", "concerts", "concerto", "concertos",
            ),
            "Exhibitions": (
                "exhibition", "exhibitions", "exposicao", "exposicoes", "arte",
            ),
            "Theater Opera & Dance": (
                "theater", "theatre", "teatro", "opera", "dance", "danca", "ballet",
            ),
            "Family & Kids": (
                "family", "familia", "kids", "children", "child", "miudos", "criancas",
            ),
            "Cinema": ("cinema", "film", "films", "movie", "movies"),
            "Sports": ("sport", "sports", "desporto", "desportos", "desportivo", "desportiva", "desportivos", "desportivas"),
            "Fairs": ("fair", "fairs", "feira", "feiras", "market", "mercado"),
            "Festivals": ("festival", "festivals", "festivais"),
            "Gastronomy": (
                "food", "gastronomy", "gastronomia", "gastronomico", "gastronomicos",
                "gastronomica", "gastronomicas", "culinaria", "culinario", "wine", "vinho",
            ),
        }
        negation_re = (
            r"(?:avoid|without|excluding|except|no|not|"
            r"sem|evita|evitar|exclui|excluir|menos|nao|"
            r"nao\s+quero|que\s+nao\s+sejam|que\s+nao\s+seja)"
        )
        excluded: list[str] = []
        for category, terms in category_terms.items():
            term_re = r"(?:%s)" % "|".join(re.escape(term) for term in terms)
            negation_before = re.search(
                rf"\b{negation_re}\b(?:\s+\w+){{0,5}}\s+\b{term_re}\b",
                normalized,
                flags=re.IGNORECASE,
            )
            term_before_negation = re.search(
                rf"\b{term_re}\b(?:\s+\w+){{0,5}}\s+\b(?:avoid|excluded|excluding|evitar|excluir|excluido|excluida)\b",
                normalized,
                flags=re.IGNORECASE,
            )
            if negation_before or term_before_negation:
                excluded.append(category)
        included_category_keys = {
            cls._event_category_key(category)
            for category in cls._infer_included_event_categories(user_message)
        }
        return [
            category
            for category in dict.fromkeys(excluded)
            if cls._event_category_key(category) not in included_category_keys
        ]

    @classmethod
    def _infer_included_event_categories(cls, user_message: str) -> list[str]:
        """Infer event categories that the user positively requests."""
        query = cls._normalize_event_preference_text(user_message or "")
        if not query:
            return []
        query = re.sub(r"\bno\s+(?:more\s+)?museums?\b", " ", query, flags=re.IGNORECASE)

        category_patterns = (
            ("Music", r"\b(?:music|musica|concerts?|concertos?|fado|jazz|rock|pop)\b"),
            ("Theater Opera & Dance", r"\b(?:theatre|theater|teatro|opera|dance|danca|ballet)\b"),
            ("Exhibitions", r"\b(?:exhibition|exhibitions|exposicao|exposicoes|art|arte|gallery|galeria)\b"),
            ("Family & Kids", r"\b(?:family|familia|kids|children|child|miudos|criancas)\b"),
            ("Festivals", r"\b(?:festival|festivals|festivais|arrai(?:al|ais|s)?|santos populares|marchas populares|festas de lisboa)\b"),
            ("Sports", r"\b(?:sport|sports|desporto|desportos|desportiv[oa]s?|marathon|maratona|trail|surf|csio)\b"),
            ("Cinema", r"\b(?:cinema|films?|movies?)\b"),
            ("Fairs", r"\b(?:fair|fairs|feira|feiras|market|mercado)\b"),
            ("Gastronomy", r"\b(?:food|gastronomy|gastronomia|gastronomic[oa]s?|culinari[oa]s?|wine|vinho)\b"),
            ("Main Events", r"\b(?:main\s+events?|principais\s+eventos|eventos?\s+principais|summit|conference|congress|forum|expo|technology|tech|startup)\b"),
        )
        excluded_spans = [
            match.span()
            for match in re.finditer(
                r"\b(?:avoid|without|excluding|except|no|not|sem|evita|evitar|exclui|excluir|menos|nao)"
                r"\b(?:\s+\w+){0,5}",
                query,
                flags=re.IGNORECASE,
            )
        ]

        included: list[str] = []
        for category, pattern in category_patterns:
            for match in re.finditer(pattern, query, flags=re.IGNORECASE):
                if any(start <= match.start() <= end for start, end in excluded_spans):
                    continue
                included.append(category)
                break
        return list(dict.fromkeys(included))

    @classmethod
    def _clean_event_focus_for_exclusions(
        cls,
        focus_query: Optional[str],
        excluded_categories: list[str],
        user_message: str = "",
    ) -> Optional[str]:
        """Remove filters/categories from the text focus while preserving real themes."""
        if not focus_query:
            return focus_query
        normalized_focus = cls._normalize_event_preference_text(focus_query)
        if not normalized_focus:
            return None
        blocked_tokens = {
            "avoid", "without", "excluding", "except", "no", "not", "sem",
            "nem", "nor", "neither",
            "evita", "evitar", "exclui", "excluir", "menos", "nao",
            "seja", "sejam", "tambem", "tambem", "que",
            "quero", "queria", "gostava", "cultural", "culturais", "mas", "but",
            "mostra", "mostrar", "mais", "dois", "outro", "outros",
            "event", "events", "evento", "eventos", "concert", "concerts",
            "concerto", "concertos", "music", "musica", "museu", "museus",
            "museum", "museums", "children", "child", "kids", "family",
            "familia", "criancas", "miudos", "sports", "sport", "desporto",
            "desportos", "exhibition", "exhibitions", "exposicao", "exposicoes",
            "domingo", "sunday", "sabado", "saturday", "segunda", "monday",
            "terca", "tuesday", "quarta", "wednesday", "quinta", "thursday",
            "sexta", "friday", "livre", "outdoor", "outdoors", "open", "air",
            "outside", "exterior", "gratis", "gratuito", "gratuitos", "gratuita",
            "gratuitas", "free",
        }
        if cls._query_excludes_museum_venues(user_message):
            blocked_tokens.update({"museu", "museus", "museum", "museums"})
        if "Fado" in excluded_categories:
            blocked_tokens.update({"fado"})
        if "Music" in excluded_categories:
            blocked_tokens.update({"music", "musica", "concert", "concerts", "concerto", "concertos", "fado", "jazz", "rock", "pop"})
        if "Exhibitions" in excluded_categories:
            blocked_tokens.update({"exhibition", "exhibitions", "exposicao", "exposicoes", "arte"})
        if "Theater Opera & Dance" in excluded_categories:
            blocked_tokens.update({"theater", "theatre", "teatro", "opera", "dance", "danca", "ballet"})
        if "Family & Kids" in excluded_categories:
            blocked_tokens.update({"family", "familia", "kids", "children", "child", "miudos", "criancas"})
        if "Cinema" in excluded_categories:
            blocked_tokens.update({"cinema", "film", "films", "movie", "movies"})
        if "Sports" in excluded_categories:
            blocked_tokens.update({"sport", "sports", "desporto", "desportos", "desportivo", "desportiva", "desportivos", "desportivas"})
        if "Fairs" in excluded_categories:
            blocked_tokens.update({"fair", "fairs", "feira", "feiras", "market", "mercado"})
        if "Festivals" in excluded_categories:
            blocked_tokens.update({"festival", "festivals", "festivais"})
        if "Gastronomy" in excluded_categories:
            blocked_tokens.update({
                "food", "gastronomy", "gastronomia", "gastronomico", "gastronomicos",
                "gastronomica", "gastronomicas", "culinaria", "culinario", "wine", "vinho",
            })
        remaining = [
            token for token in re.findall(r"[a-z0-9]+", normalized_focus)
            if token not in blocked_tokens and len(token) >= 3
        ]
        return " ".join(dict.fromkeys(remaining)) if remaining else None

    @staticmethod
    def _infer_event_category_hint(user_message: str) -> Optional[str]:
        """Infers a VisitLisboa event category hint from common PT/EN event queries."""
        query = ResearcherAgent._normalize_event_preference_text(user_message or "")
        excluded_categories = set(ResearcherAgent._extract_excluded_event_categories(user_message))
        requested_categories = ResearcherAgent._infer_included_event_categories(user_message)
        category_terms = (
            ("Music", r"\b(?:music|musica|concerts?|concertos?|fado|jazz|rock|pop)\b"),
            ("Theater Opera & Dance", r"\b(?:theatre|theater|teatro|opera|dance|danca|ballet)\b"),
            ("Exhibitions", r"\b(?:exhibition|exhibitions|exposicao|exposicoes|art|arte|gallery|galeria)\b"),
            ("Family & Kids", r"\b(?:family|familia|kids|children|child|miudos|criancas)\b"),
            ("Festivals", r"\b(?:festival|festivals|festivais|arrai(?:al|ais|s)?|santos populares|marchas populares|festas de lisboa)\b"),
            ("Sports", r"\b(?:sport|sports|desporto|desportos|desportiv[oa]s?|marathon|maratona|trail|surf|csio)\b"),
            ("Cinema", r"\b(?:cinema|films?|movies?)\b"),
            ("Fairs", r"\b(?:fair|fairs|feira|feiras|market|mercado)\b"),
            ("Gastronomy", r"\b(?:food|gastronomy|gastronomia|gastronomic[oa]s?|culinari[oa]s?|wine|vinho)\b"),
            ("Main Events", r"\b(?:main\s+events?|principais\s+eventos|eventos?\s+principais|summit|conference|congress|forum|expo|technology|tech|startup)\b"),
        )
        for category, pattern in category_terms:
            if (
                category not in excluded_categories
                and category not in requested_categories
                and re.search(pattern, query, flags=re.IGNORECASE)
            ):
                requested_categories.append(category)
        return ", ".join(dict.fromkeys(requested_categories)) if requested_categories else None

    @staticmethod
    def _extract_event_date_filters(user_message: str) -> list[str]:
        """Extract one or more event date filters from natural PT/EN phrasing."""
        primary = ResearcherAgent._extract_event_date_filter(user_message)
        if primary:
            return [primary]

        normalized = ResearcherAgent._normalize_event_preference_text(user_message)
        month_terms = (
            ("January", r"\b(?:janeiro|january)\b"),
            ("February", r"\b(?:fevereiro|february)\b"),
            ("March", r"\b(?:marco|março|march)\b"),
            ("April", r"\b(?:abril|april)\b"),
            ("May", r"\b(?:maio|may)\b"),
            ("June", r"\b(?:junho|june)\b"),
            ("July", r"\b(?:julho|july)\b"),
            ("August", r"\b(?:agosto|august)\b"),
            ("September", r"\b(?:setembro|september)\b"),
            ("October", r"\b(?:outubro|october)\b"),
            ("November", r"\b(?:novembro|november)\b"),
            ("December", r"\b(?:dezembro|december)\b"),
        )
        filters: list[str] = []
        for label, pattern in month_terms:
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                filters.append(label)
        return list(dict.fromkeys(filters))

    @staticmethod
    def _clean_place_focus_subject(subject: Optional[str]) -> Optional[str]:
        """Remove generic request modifiers from an extracted place subject."""
        value = re.sub(r"\s+", " ", str(subject or "")).strip(" .?!,;:")
        if not value:
            return None
        suffix_patterns = (
            r"\s*,?\s+(?:with|including|plus)\s+(?:practical\s+)?(?:visit\s+)?"
            r"(?:info(?:rmation)?|details|opening\s+hours?|tickets?)\b.*$",
            r"\s+(?:and|plus|including)\s+nearby\s+(?:cultural\s+)?"
            r"(?:stops?|places?|attractions?|museums?|monuments?)\b.*$",
            r"\s*,?\s+(?:com|incluindo)\s+(?:informação|informacao|dados|detalhes)\s+"
            r"(?:prática|pratica|de\s+visita)\b.*$",
            r"\s+(?:e|mais)\s+(?:locais|paragens|atrações|atracoes)\s+"
            r"(?:culturais\s+)?(?:próxim[oa]s|proxim[oa]s)\b.*$",
        )
        for pattern in suffix_patterns:
            value = re.sub(pattern, "", value, flags=re.IGNORECASE).strip(" .?!,;:")
        normalized_value = _normalize_researcher_intent_text(value)
        if normalized_value in {
            "info",
            "information",
            "details",
            "practical info",
            "practical visit info",
            "visit info",
            "informacao",
            "informacao pratica",
            "detalhes",
            "detalhes praticos",
        }:
            return None
        return value or None

    @staticmethod
    def _extract_place_focus_query(user_message: str) -> Optional[str]:
        """Extracts a focused place subject from broader PT/EN lookup phrasings."""
        query = (user_message or "").strip()
        if not query:
            return None

        quoted_match = re.search(r'"([^"\n]{2,120})"|“([^”\n]{2,120})”', query)
        if quoted_match:
            quoted_subject = next((group for group in quoted_match.groups() if group), "").strip(" .?!")
            if quoted_subject:
                cleaned_subject = ResearcherAgent._clean_place_focus_subject(quoted_subject)
                if cleaned_subject:
                    return cleaned_subject

        visit_area_match = re.search(
            r"\b(?:visit|visiting|explore|visitar|conhecer|explorar)\s+(?:a\s+|o\s+|os\s+|as\s+)?(?P<subject>[A-ZÀ-Ýa-zà-ÿ0-9][A-ZÀ-Ýa-zà-ÿ0-9 '\-/]{1,80}?)(?=\s+(?:tomorrow|today|tonight|amanh[aã]|hoje|esta noite|this week|this weekend)\b|[\?\!\.,]|$)",
            query,
            flags=re.IGNORECASE,
        )
        if visit_area_match:
            subject = visit_area_match.group("subject").strip(" .?!,")
            subject = re.sub(
                r"^(?:in|near|around|em|no|na|nos|nas|perto\s+d[eoa]?|junto\s+d[eoa]?)\s+",
                "",
                subject,
                flags=re.IGNORECASE,
            )
            if subject:
                cleaned_subject = ResearcherAgent._clean_place_focus_subject(subject)
                if cleaned_subject:
                    return cleaned_subject

        named_lookup_re = re.compile(
            r"\b(?:tell me about|what about|more about|details about|information about|where is|where's|find|show me|d[aá][- ]?me(?: mais)? detalhes? sobre(?: o| a| os| as)?|detalhes? sobre(?: o| a| os| as)?|sobre(?: o| a| os| as)?|fala[- ]?me(?: mais)? sobre(?: o| a| os| as)?|fala[- ]?me(?: mais)? (?:de|do|da|dos|das)|fale[- ]?me(?: mais)? (?:de|do|da|dos|das)|diz[- ]?me(?: mais)? sobre(?: o| a| os| as)?|diz[- ]?me (?:de|do|da|dos|das)|diz[- ]?me onde(?: e| é| fica)(?: o| a| os| as)?|onde(?: e| é| fica)(?: o| a| os| as)?|encontra(?:r)?|mostrar(?:-me)?)\b",
            re.IGNORECASE,
        )
        if named_lookup_re.search(query):
            subject = named_lookup_re.sub(" ", query)
            subject = re.sub(r"\s+", " ", subject).strip(" .?!")
            if subject:
                cleaned_subject = ResearcherAgent._clean_place_focus_subject(subject)
                if cleaned_subject:
                    return cleaned_subject

        tokens = re.findall(r"[a-zA-ZÀ-ÿ0-9']+", query)
        has_title_like_casing = any(char.isupper() for char in query)
        if has_title_like_casing and 1 <= len(tokens) <= 6:
            lowered = query.lower()
            if not any(term in lowered for term in ["event", "events", "evento", "eventos"]):
                first_lower = tokens[0].lower() if tokens else ""
                if first_lower in _QUESTION_STARTER_WORDS:
                    proper_nouns = [
                        token
                        for token in tokens
                        if any(char.isupper() for char in token) and token.lower() not in _NON_PROPER_PLACE_WORDS
                    ]
                    if 1 <= len(proper_nouns) <= 4:
                        cleaned_subject = ResearcherAgent._clean_place_focus_subject(" ".join(proper_nouns))
                        if cleaned_subject:
                            return cleaned_subject
                    return None
                cleaned_subject = ResearcherAgent._clean_place_focus_subject(query.strip(" .?!"))
                if cleaned_subject:
                    return cleaned_subject

        specific_lookup = _extract_specific_place_lookup_phrase(user_message)
        if specific_lookup:
            cleaned_subject = ResearcherAgent._clean_place_focus_subject(specific_lookup)
            if cleaned_subject:
                return cleaned_subject

        return None

    @staticmethod
    def _extract_place_area_filter(user_message: str) -> Optional[str]:
        """Extract an area filter from broad place listing prompts."""
        query = str(user_message or "").strip()
        if not query:
            return None
        matches = list(
            re.finditer(
                r"\b(?:in|near|around|em|no|na|nos|nas|perto\s+de|perto\s+do|perto\s+da)\s+"
                r"(?P<area>[A-ZÀ-Ýa-zà-ÿ0-9][A-ZÀ-Ýa-zà-ÿ0-9 '\-/]{1,80})",
                query,
                flags=re.IGNORECASE,
            )
        )
        for match in reversed(matches):
            area = re.sub(r"\s+", " ", match.group("area")).strip(" .?!,;:")
            area = re.sub(
                r"\s+(?:with|including|for|that|where|which|com|incluindo|para|que|"
                r"e\s+(?:com|usando|transporte)|and\s+(?:with|using|transport))\b.*$",
                "",
                area,
                flags=re.IGNORECASE,
            ).strip(" .?!,;:")
            cleaned_area = ResearcherAgent._clean_place_focus_subject(area)
            if cleaned_area:
                return cleaned_area
        return None

    @staticmethod
    def _is_direct_place_lookup_query(user_message: str) -> bool:
        """Detects straightforward place and service lookups that are safer to answer directly from tools."""
        query = (user_message or "").lower()
        history_keywords = ["history", "historical", "história", "historia", "culture", "cultura"]
        event_keywords = [
            "event", "events", "evento", "eventos", "concert", "concerto",
            "festival", "exhibition", "exposição", "exposicao", "show",
            "fair", "fairs", "feira", "feiras", "book fair",
            "summit", "conference", "congress", "forum", "expo", "games week",
        ]
        has_place_like_query = bool(
            re.search(
                r"\b(?:museum|museums|museu|museus|monument|monuments|monumento|monumentos|"
                r"restaurant|restaurants|restaurante|restaurantes|place|places|local|locais|"
                r"attraction|attractions|atra[cç][aã]o|atra[cç][oõ]es|viewpoint|viewpoints|"
                r"miradouro|miradouros|palace|palaces|pal[aá]cio|pal[aá]cios|castle|castles|"
                r"castelo|castelos|church|churches|igreja|igrejas|cathedral|cathedrals|"
                r"catedral|catedrais|sé|monastery|monasteries|"
                r"mosteiro|mosteiros|tower|towers|torre|torres|heritage|patrim[oó]nio|"
                r"hotels?|hot[eé]is|alojamentos?|hostels?|guest\s+houses?|pousadas?|"
                r"shops?|lojas?|compras|centros?\s+comerciais?|shopping|malls?|"
                r"cruises?|cruzeiros?|beaches|praias?|golfe?|golf|fado|nightlife|bars?)\b",
                query,
            )
        )
        if has_place_like_query:
            event_keywords = [keyword for keyword in event_keywords if keyword != "show"]
        if ResearcherAgent._query_excludes_museum_venues(user_message) and re.search(
            r"\b(?:live\s+music|m[uú]sica\s+ao\s+vivo|music|m[uú]sica|concerts?|concertos?|events?|eventos?)\b",
            query,
            flags=re.IGNORECASE,
        ):
            return False
        focus_query = ResearcherAgent._extract_place_focus_query(user_message) or ResearcherAgent._extract_place_area_filter(user_message)
        directed_lookup_markers = [
            "where is", "where's", "onde fica", "onde é", "onde e", "tell me about", "what about",
            "more about", "details about", "information about", "sobre",
            "fala-me de", "fala me de", "fala-me do", "fala me do", "fala-me da", "fala me da",
            "diz-me sobre", "diz me sobre", "diz-me do", "diz me do", "diz-me da", "diz me da",
            "diz-me onde é", "diz me onde e", "diz-me onde fica", "diz me onde fica",
            "closest to", "nearest to", "near ",
            "mais perto", "mais próximo", "perto de", "perto do", "perto da",
        ]
        if any(keyword in query for keyword in event_keywords):
            return False
        if any(keyword in query for keyword in history_keywords) and not (focus_query and has_place_like_query):
            return False
        if ResearcherAgent._extract_service_types(user_message):
            return bool(ResearcherAgent._extract_near_location_name(user_message) or focus_query)

        place_category_signals = ResearcherAgent._infer_place_category_signals(user_message)
        has_place_hint = ResearcherAgent._infer_place_category_hint(user_message) is not None
        has_multi_place_intent = len(place_category_signals) >= 2
        has_directional_lookup = any(marker in query for marker in directed_lookup_markers)
        has_field_specific_lookup = has_place_hint and bool(
            re.search(
                r"\b(?:opening hours?|hours?|hor[aá]rios?|tickets?|bilhetes?|website|site|phone|telefone|email|e-mail)\b",
                query,
            )
        )
        has_recommendation_lookup = bool(
            re.search(
                r"\b(?:best|top|recommended|recommend|suggest|suggested|good|quiet|hidden|lesser[-\s]?known|"
                r"calm|calmos?|tranquilos?|escondidos?|menos\s+conhecidos|melhores|principais|recomenda|sugere)\b",
                query,
            )
            or re.search(r"\bwhat are\b.*\b(?:museums?|monuments?|restaurants?|hotels?|viewpoints?)\b", query)
            or re.search(r"\bquais s(?:ã|a)o\b.*\b(?:museus|monumentos|restaurantes|hot[eé]is|miradouros)\b", query)
        )
        has_area_filter = bool(
            re.search(
                r"\b(?:in|near|around|em|no|na|nos|nas|perto de|perto do|perto da)\s+"
                r"[a-zà-ÿ0-9][a-zà-ÿ0-9 '\-/]{1,80}",
                query,
            )
        )
        has_listing_lookup = has_place_like_query and bool(
            re.search(
                r"\b(?:tell me|show me|list|show|give me|recommend|suggest|diz[- ]?me|"
                r"mostra|lista|indica|recomenda|sugere|fala[- ]?me|quais|que|h[aá]|existem?)\b",
                query,
            )
        )

        if focus_query and any(marker in query for marker in [
            "where is", "where's", "onde fica", "onde é", "onde e", "tell me about", "what about", "more about",
            "details about", "information about", "sobre", "fala-me de", "fala me de", "fala-me do",
            "fala me do", "fala-me da", "fala me da", "diz-me sobre", "diz me sobre", "diz-me do",
            "diz me do", "diz-me da", "diz me da", "diz-me onde é", "diz me onde e", "diz-me onde fica", "diz me onde fica",
        ]):
            return True

        if has_place_hint and has_directional_lookup:
            return True
        if has_field_specific_lookup:
            return True
        if has_place_hint and has_recommendation_lookup:
            return True
        if has_place_hint and has_place_like_query and (has_area_filter or has_listing_lookup):
            return True
        if has_multi_place_intent and has_place_like_query and (has_area_filter or has_listing_lookup):
            return True

        return False

    @staticmethod
    def _extract_requested_result_count(user_message: str) -> Optional[int]:
        """Extract a small explicit result count from place-discovery prompts."""
        query = (user_message or "").lower()
        word_counts = {
            "one": 1,
            "um": 1,
            "uma": 1,
            "two": 2,
            "dois": 2,
            "duas": 2,
            "three": 3,
            "tres": 3,
            "três": 3,
            "four": 4,
            "quatro": 4,
            "five": 5,
            "cinco": 5,
        }
        count_pattern = (
            r"(?:museums?|museus|places?|locais|attractions?|atra[cç][oõ]es|restaurants?|restaurantes|"
            r"monuments?|monumentos|options?|op[cç][oõ]es|resultados?|results?|sugest[oõ]es|suggestions?)"
        )
        digit_match = re.search(rf"\b([1-5])\s+{count_pattern}\b", query)
        if digit_match:
            return int(digit_match.group(1))
        for word, count in word_counts.items():
            if re.search(rf"\b{re.escape(word)}\s+{count_pattern}\b", query):
                return count
        return None

    @staticmethod
    def _is_visit_place_context_query(user_message: str) -> bool:
        """Detects visit-area prompts where Researcher should return local place context."""
        query = (user_message or "").lower()
        visit_terms = ["visit", "visiting", "explore", "visitar", "conhecer", "explorar"]
        if not any(term in query for term in visit_terms):
            return False
        if any(term in query for term in ["history", "história", "historia", "culture", "cultura"]):
            return False
        return bool(ResearcherAgent._extract_place_focus_query(user_message))

    @staticmethod
    def _is_history_culture_query(user_message: str) -> bool:
        """Detects history/culture questions that should use the dedicated grounded lookup path."""
        normalized = unicodedata.normalize("NFKD", user_message or "")
        normalized = normalized.encode("ascii", "ignore").decode("ascii").lower()
        negated_event_cue = bool(
            re.search(
                r"\b(?:sem\s+eventos?|nao\s+(?:me\s+)?(?:mostres?|sugiras?|incluas?)\s+eventos?|"
                r"do\s+not\s+(?:suggest|show|include)\s+events?|"
                r"don'?\s*t\s+(?:suggest|show|include)\s+events?|"
                r"no\s+events?|without\s+events?|not\s+events?)\b",
                normalized,
            )
        )
        if (
            re.search(
                r"\b(event|events|evento|eventos|festival|festivals|concert|concerto|show|what's on|o que ha|esta semana|this week|fim de semana|weekend)\b",
                normalized,
            )
            and not negated_event_cue
        ):
            return False
        if re.search(r"\b(history|historical|historia|culture|cultura)\b", normalized):
            return True
        return bool(
            re.search(r"\b(?:explica|explicar|explain|resumo|resume|summari[sz]e|o que era|what was)\b", normalized)
            and re.search(r"\b(?:lisboa|lisbon)\b", normalized)
            and re.search(r"\b(?:por volta de|around|cerca de|seculo|sec\.?|1[5-9]\d{2}|20\d{2})\b", normalized)
        )

    @staticmethod
    def _extract_near_location_name(user_message: str) -> Optional[str]:
        """Extracts a nearby-location target from simple PT/EN service phrasings."""
        central_match = re.search(
            r"\b(?:in|around|within|no|na|em)\s+(?P<location>central\s+lisbon|central\s+lisboa|downtown\s+lisbon|centro\s+de\s+lisboa)\b",
            user_message,
            flags=re.IGNORECASE,
        )
        if central_match:
            return central_match.group("location").strip()

        service_location_match = re.search(
            r"\b(?:farm[aá]cias?|pharmac(?:y|ies)|hospitais?|hospitals?|cl[ií]nicas?|clinics?|"
            r"bibliotecas?|libraries|escolas?|schools?|mercados?|markets?|jardins?|gardens?|"
            r"parques?|parks?|sanit[aá]rios?|toilets?|restrooms?|casas?\s+de\s+banho|"
            r"wi[-\s]?fi|internet|pol[ií]cia|police|bombeiros?|firefighters?|"
            r"estacionamento|parking|embaixadas?|embassies)\b"
            r".{0,80}?\b(?:perto\s+de|perto\s+do|perto\s+da|near|em|no|na|nos|nas|in|at)\s+"
            r"(?P<location>.+?)(?:\s+(?:com|with|que|which|abert[oa]s?|open|morada|address|"
            r"perto|near|recomendas|recommend|usar|use)\b|[\?\!\.,;]|$)",
            user_message,
            flags=re.IGNORECASE,
        )
        if service_location_match:
            location = ResearcherAgent._clean_nearby_location_text(
                service_location_match.group("location")
            )
            if location:
                return location

        patterns = [
            r"\b(?:como\s+(?:chego|vou|posso\s+ir)|how\s+(?:do|can)\s+i\s+(?:get|go))\b.{0,100}"
            r"\b(?:desde|from|a\s+partir\s+(?:de|do|da))\s+(?P<location>.+?)(?:[\?\!\.,;]|$)",
            r"\b(?:tenho|have)\s+\d+\s*(?:minutos?|mins?|minutes?)\s+(?:no|na|em|at|in)\s+"
            r"(?P<location>.+?)(?:\s*[,;]\s*|\s+(?:e|and|diz|diga|mostra|tell|show)\b|[\?\!\.]|$)",
            r"\b(?:chamad[ao]|called|named)\s+(?P<location>.+?)(?:\s*,|\s+consegues\b|\s+can\b|[\?\!\.]|$)",
            r"\b(?:fica|located)\s+(?:na|no|em|at)\s+(?P<location>.+?)(?:\s*,\s*(?:diz|leva|tell|take)\b|[\?\!\.]|$)",
            r"\b(?:estou|tou)\s+(?:na|no|em|à|a)\s+(?P<location>.+?)(?:\s*[,;]\s*(?:qual|quais|onde|há|ha|existe|consegues|diz|which|what|where)\b|\s+(?:qual|quais|onde|há|ha|existe)\b|[\?\!\.]|$)",
            r"\b(?:i(?:'m| am))\s+(?:at|in|on)\s+(?P<location>.+?)(?:\s*,\s*(?:which|what|where|can)\b|[\?\!\.]|$)",
            r"\bstart(?:ing)?\s+(?:in|from|at)\s+(?P<location>.+?)(?:\s*,|\s+and\b|\s+then\b|\s+to\b|[\?\!\.]|$)",
            r"\bfrom\s+(?P<location>.+?)\s+(?:i|we)\s+(?:need|want|have)\b",
            r"\bcome(?:çar|car)\s+(?:em|no|na|do|da)\s+(?P<location>.+?)(?:\s*,|\s+e\b|\s+depois\b|\s+para\b|[\?\!\.]|$)",
            r"\ba\s+partir\s+(?:de|do|da)\s+(?P<location>.+?)(?:\s*,|\s+e\b|\s+depois\b|\s+para\b|[\?\!\.]|$)",
            r"\bnearest\s+(?:\w+\s+){0,4}to\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bclosest\s+(?:\w+\s+){0,4}to\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bclosest to\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bnearest to\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bnear the\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bnear\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bmais perto de\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bmais perto do\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bmais perto da\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bmais próximo de\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bmais próximo do\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bmais próximo da\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bmais próxima de\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bmais próxima do\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bmais próxima da\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bperto de\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bperto do\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bperto da\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, user_message, flags=re.IGNORECASE)
            if match:
                location = match.group("location").strip(" .?!,")
                location = ResearcherAgent._clean_nearby_location_text(location)
                return location or None
        return None

    @staticmethod
    def _clean_nearby_location_text(location: object) -> Optional[str]:
        """Remove temporal or intent suffixes from a nearby-location phrase."""
        if location is None:
            return None
        cleaned = re.sub(r"\s+", " ", str(location or "")).strip(" .?!,;:")
        if not cleaned:
            return None
        normalized_cleaned = unicodedata.normalize("NFKD", cleaned)
        normalized_cleaned = normalized_cleaned.encode("ascii", "ignore").decode("ascii").lower()
        normalized_cleaned = re.sub(r"\s+", " ", normalized_cleaned).strip()
        if normalized_cleaned in {
            "mim",
            "me",
            "my",
            "eu",
            "aqui",
            "here",
            "minha localizacao",
            "minha localizacao atual",
            "localizacao atual",
            "current location",
            "my location",
            "where i am",
            "onde estou",
        }:
            return None
        try:
            from tools.location_resolver import clean_location_query_fragment

            cleaned = clean_location_query_fragment(cleaned) or cleaned
        except Exception:
            pass

        split_patterns = (
            r"\b(?:posso|podemos|consigo|devo|can\s+i|can\s+we|should\s+i)\s+"
            r"(?:usar|utilizar|visitar|entrar|ir|use|visit|enter|go)\b.*$",
            r"\b(?:i|we)\s+(?:can|could|should)\s+(?:use|visit|enter|go)\b.*$",
            r"\b(?:posso|podemos|consigo|devo|i\s+can|we\s+can|can\s+i|can\s+we|should\s+i)\s*$",
            r"\b(?:recomendas|aconselhas|sugeres|recommend|suggest)\b.*$",
            r"\s*[,;:]\s*(?:qual|quais|onde|há|ha|existe|consegues|diz|mostra|which|what|where|show|tell)\b",
            r"\b(?:e|and)\s+(?:quanto\s+(?:tempo\s+)?(?:demoro|demora|leva)|how\s+long|tempo\s+(?:a\s+p[eé]|de\s+caminhada)|walking\s+time)\b.*$",
            r"\b(?:if|when|se|quando)\b",
            r"\b(?:that|which|who|where|still|open|useful|que|e que)\b",
            r"\b(?:e|and)\s+(?:quero|preciso|tenho|vou|gostava|need|want|can)\b",
            r"\b(?:e|and)\s+(?:a\s+)?(?:agua|água|water)\s+(?:é|e|is)\b",
            r"\b(?:abert[ao]s?|open)\s+(?:ao|aos|à|as|on)\s+(?:domingo|sunday|sábado|sabado|saturday)\b",
            r"\b(?:ao|aos|à|as|on)\s+(?:domingo|sunday|sábado|sabado|saturday)\b",
            r"\b(?:esta\s+noite|ao\s+fim\s+do\s+dia|mais\s+tarde|hoje|amanh[ãa]|agora|tonight|this\s+evening|today|tomorrow|now|later)\b",
            r"\b(?:para|to)\s+(?:usar|use|ir|go|chegar|reach|esta\s+noite|hoje|amanh[ãa]|agora|mais\s+tarde|tonight|this\s+evening|today|tomorrow|now|later)\b",
            r"\b(?:for|during)\s+(?:tonight|this\s+evening|today|tomorrow|later|the\s+evening)\b",
            r"\b(?:com|with)\s+(?:morada|address|endereço|distância|distance|horário|hours?|telefone|phone)\b.*$",
            r"\b(?:nos|nas|com|usando|usa|usar|limita(?:r|ndo)?(?:-te)?\s+a|apenas\s+com)\s+"
            r"(?:os\s+)?(?:dados|fontes?|datasets?|lisboa\s+aberta)\b.*$",
            r"\b(?:in|from|using|with|limited\s+to|only\s+from)\s+(?:the\s+)?"
            r"(?:available\s+)?(?:data|datasets?|sources?|open\s+data)\b.*$",
            r"\b(?:dados|fontes?|datasets?|data|sources?)\s+(?:da|do|de|from)\s+"
            r"(?:lisboa\s+aberta|available\s+data|open\s+data)\b.*$",
        )
        for pattern in split_patterns:
            parts = re.split(pattern, cleaned, maxsplit=1, flags=re.IGNORECASE)
            cleaned = parts[0].strip(" .?!,;:")
        normalized_cleaned = unicodedata.normalize("NFKD", cleaned)
        normalized_cleaned = normalized_cleaned.encode("ascii", "ignore").decode("ascii").lower()
        normalized_cleaned = re.sub(r"\s+", " ", normalized_cleaned).strip()
        if normalized_cleaned in {"mim", "me", "aqui", "here", "minha localizacao", "current location", "my location"}:
            return None
        return cleaned or None

    @staticmethod
    def _query_references_user_current_location(user_message: str) -> bool:
        """Return whether the user asks for proximity to their own location."""
        normalized = unicodedata.normalize("NFKD", user_message or "")
        normalized = normalized.encode("ascii", "ignore").decode("ascii").lower()
        return bool(
            re.search(
                r"\b(?:perto\s+de\s+mim|perto\s+da\s+minha\s+localizacao|"
                r"perto\s+daqui|near\s+me|near\s+my\s+location|nearby|close\s+to\s+me)\b",
                normalized,
            )
        )

    @staticmethod
    def _build_current_location_clarification(language: str) -> str:
        """Ask for a usable reference point instead of geocoding 'me'."""
        if language == "pt":
            return "\n".join(
                [
                    "### 🧭 **Preciso da tua localização de referência**",
                    "",
                    "✅ **Resposta direta:** consigo procurar serviços perto de ti, mas preciso de uma morada, zona, ponto de referência ou estação/paragem próxima.",
                    "",
                    "---",
                    "",
                    "- Exemplo: “estou no Rossio”, “perto da NOVA IMS” ou “junto ao Cais do Sodré”.",
                ]
            )
        return "\n".join(
            [
                "### 🧭 **I Need Your Reference Location**",
                "",
                "✅ **Direct answer:** I can search services near you, but I need an address, area, landmark, or nearby station/stop.",
                "",
                "---",
                "",
                "- Example: “I am at Rossio”, “near NOVA IMS”, or “by Cais do Sodré”.",
            ]
        )

    @staticmethod
    def _is_unsupported_private_service_query(user_message: str) -> bool:
        """Detect private-service discovery requests not covered by LISBOA datasets."""
        normalized = unicodedata.normalize("NFKD", user_message or "")
        normalized = normalized.encode("ascii", "ignore").decode("ascii").lower()
        if not re.search(r"\b(?:veterinari\w*|veterinary|vet|vets)\b", normalized):
            return False
        return bool(
            re.search(
                r"\b(?:onde|where|perto|near|em|in|no|na|find|encontra|mostra|"
                r"lista|recommend|recomenda|hospital|clinica|clinic)\b",
                normalized,
            )
        )

    @staticmethod
    def _build_unsupported_private_service_response(user_message: str, language: str) -> str:
        """Build a conservative answer for private services without structured coverage."""
        nearby_location = ResearcherAgent._clean_nearby_location_text(
            ResearcherAgent._extract_near_location_name(user_message)
        )
        if not nearby_location:
            location_match = re.search(
                r"\b(?:veterinari\w*|veterinary|vet|vets)\b.{0,80}"
                r"\b(?:em|no|na|nos|nas|in|near|perto\s+de|perto\s+do|perto\s+da)\s+"
                r"(?P<location>.+?)(?:[\?\!\.,;]|$)",
                user_message or "",
                flags=re.IGNORECASE,
            )
            if location_match:
                nearby_location = ResearcherAgent._clean_nearby_location_text(
                    location_match.group("location")
                )
        scope_pt = f" em **{nearby_location}**" if nearby_location else ""
        scope_en = f" in **{nearby_location}**" if nearby_location else ""
        if language == "pt":
            lines = [
                "### 🐾 **Serviço não confirmado nos dados disponíveis**",
                "",
                f"✅ **Resposta direta:** não tenho uma fonte estruturada confirmada para listar veterinários{scope_pt}.",
                "",
                "---",
                "",
                "- Não vou inventar clínicas, moradas, horários ou contactos sem fonte local confirmada.",
                "- Se tiveres o **nome** ou a **morada** do local, posso usar essa informação para ajudar com o percurso ou com contexto prático.",
                "- Para disponibilidade clínica, urgências ou marcações, confirma diretamente com o estabelecimento.",
            ]
            return "\n".join(lines)
        lines = [
            "### 🐾 **Service Not Confirmed In Available Data**",
            "",
            f"✅ **Direct answer:** I do not have a confirmed structured source for veterinary clinics{scope_en}.",
            "",
            "---",
            "",
            "- I will not invent clinics, addresses, opening hours, or contacts without confirmed local data.",
            "- If you provide the **name** or **address**, I can use that information for directions or practical context.",
            "- For clinical availability, emergencies, or appointments, confirm directly with the provider.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _extract_event_location_constraint(user_message: str) -> Optional[str]:
        """Extract a location constraint from event discovery wording."""
        nearby_location = ResearcherAgent._extract_near_location_name(user_message)
        if nearby_location:
            return nearby_location

        patterns = (
            r"\b(?:perto|junto)\s+(?:de|do|da|dos|das)?\s+(?P<location>.+?)(?:\s+(?:hoje|amanh[ãa]|esta\s+noite|à\s+noite|a\s+noite|tonight|today|tomorrow)\b|[,;?.!]|$)",
            r"\b(?:near|around)\s+(?P<location>.+?)(?:\s+(?:tonight|today|tomorrow|this\s+evening)\b|[,;?.!]|$)",
            r"\b(?:no|na|em|in|at)\s+(?P<location>.+?)(?:\s+(?:hoje|amanh[ãa]|esta\s+noite|à\s+noite|a\s+noite|tonight|today|tomorrow|this\s+evening)\b|[,;?.!]|$)",
        )
        for pattern in patterns:
            match = re.search(pattern, user_message or "", flags=re.IGNORECASE)
            if not match:
                continue
            location = ResearcherAgent._strip_event_temporal_suffix(
                ResearcherAgent._clean_nearby_location_text(match.group("location")) or ""
            )
            if location and not ResearcherAgent._event_location_constraint_is_filter(location):
                location = re.sub(r"\s+ao\s+vivo\b", "", location, flags=re.IGNORECASE).strip(" .?!,;:")
                return location or None
        return None

    @staticmethod
    def _strip_event_temporal_suffix(location: str) -> str:
        """Remove trailing date/window words from an extracted event location."""
        value = str(location or "").strip(" .?!,;:")
        if not value:
            return ""
        temporal_suffix_re = re.compile(
            r"\s+(?:este\s+m[eê]s|esta\s+semana|este\s+fim\s+de\s+semana|"
            r"pr[oó]xima\s+semana|pr[oó]ximo\s+m[eê]s|hoje|amanh[aã]|"
            r"segunda(?:-|\s+)?feira|ter[cç]a(?:-|\s+)?feira|quarta(?:-|\s+)?feira|"
            r"quinta(?:-|\s+)?feira|sexta(?:-|\s+)?feira|s[aá]bado|domingo|"
            r"janeiro|fevereiro|mar[cç]o|abril|maio|junho|julho|agosto|"
            r"setembro|outubro|novembro|dezembro|this\s+month|this\s+week|"
            r"this\s+weekend|next\s+week|next\s+month|today|tomorrow|"
            r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
            r"january|february|march|april|may|june|july|august|"
            r"september|october|november|december)\b.*$",
            flags=re.IGNORECASE,
        )
        value = temporal_suffix_re.sub("", value).strip(" .?!,;:")
        return re.sub(r"\s+(?:no|na|nos|nas|ao|aos|à|às|on|at)$", "", value, flags=re.IGNORECASE).strip(" .?!,;:")

    @staticmethod
    def _event_location_constraint_is_filter(location: str) -> bool:
        """Return whether an extracted event location is actually a date/filter token."""
        normalized = ResearcherAgent._normalize_event_preference_text(location)
        if not normalized:
            return True
        if normalized in {"lisboa", "lisbon"}:
            return True

        tokens = re.findall(r"[a-z0-9]+", normalized)
        month_tokens = {
            "janeiro", "january", "fevereiro", "february", "marco", "march",
            "abril", "april", "maio", "may", "junho", "june", "julho", "july",
            "agosto", "august", "setembro", "september", "outubro", "october",
            "novembro", "november", "dezembro", "december",
        }
        connector_tokens = {"e", "and", "a", "ate", "to", "through", "until"}
        date_tokens = {
            "hoje", "today", "amanha", "tomorrow", "semana", "week", "mes", "month",
            "fim", "weekend", "noite", "tonight", "esta", "este", "this", "next",
            "proxima", "proximo",
        }
        generic_event_tokens = {
            "evento", "eventos", "event", "events", "festival", "festivals", "festivais",
            "musica", "music", "desporto", "sports", "feira", "feiras", "fairs",
            "exposicao", "exposicoes", "exhibition", "exhibitions", "teatro",
            "theatre", "theater", "danca", "dance", "cinema", "concertos",
            "concert", "concerts",
        }
        filter_clause_tokens = {
            "mas", "but", "sem", "without", "nao", "not", "menos", "except", "excluding",
            "excluindo", "exclude", "excluir", "fado", "rock", "pop", "jazz", "classico",
            "classical", "live", "indoor", "outdoor", "exterior", "interior",
            "gratis", "gratuito", "free", "pago",
        }
        normalized_for_filter = re.sub(r"\bao\s+vivo\b", "", normalized).strip()
        if not normalized_for_filter:
            return True
        filter_tokens = re.findall(r"[a-z0-9]+", normalized_for_filter)
        if any(token in filter_clause_tokens for token in filter_tokens):
            return True
        if any(token in month_tokens for token in tokens):
            return all(token in month_tokens | connector_tokens | generic_event_tokens for token in tokens)
        return bool(tokens and all(token in date_tokens | connector_tokens | generic_event_tokens for token in tokens))

    @staticmethod
    def _query_requests_evening_events(user_message: str) -> bool:
        """Return whether the user specifically asks for evening/night events."""
        normalized = _normalize_researcher_intent_text(user_message)
        return bool(
            re.search(
                r"\b(?:esta noite|a noite|ao fim do dia|noite|tonight|this evening|evening)\b",
                normalized,
            )
        )

    @staticmethod
    def _event_result_has_evening_time(result: str) -> bool:
        """Return whether an event result exposes at least one evening time."""
        for hour, minute in re.findall(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", result or ""):
            if int(hour) >= 18:
                return True
        return False

    @staticmethod
    def _event_result_mentions_location(result: str, location: str) -> bool:
        """Return whether an event result visibly mentions the requested location."""
        normalized_result = _normalize_researcher_intent_text(result)
        normalized_location = _normalize_researcher_intent_text(location)
        if not normalized_location:
            return True
        return bool(re.search(rf"\b{re.escape(normalized_location)}\b", normalized_result))

    def _event_constraint_limitation(
        self,
        *,
        language: str,
        location: str,
        needs_evening: bool,
        requests_free: bool = False,
    ) -> str:
        """Build a conservative event limitation when constraints cannot be verified."""
        if language == "pt":
            title = (
                f"### 🎭 **Eventos perto de {location}**"
                if location
                else "### 🎭 **Eventos encontrados**"
            )
            constraints = []
            if location:
                constraints.append(f"perto de **{location}**")
            if needs_evening:
                constraints.append("hoje à noite")
            scope = " e ".join(constraints) if constraints else "com esses critérios"
            event_noun = "evento gratuito" if requests_free else "evento"
            return (
                f"{title}\n\n"
                f"✅ **Resposta direta:** não consegui confirmar um {event_noun} {scope} com os dados disponíveis.\n\n"
                "---\n\n"
                "⚠️ Para não inventar proximidade, horário ou disponibilidade, não apresento eventos genéricos de Lisboa como se cumprissem esses critérios.\n\n"
                f"{self._build_events_source_line(language)}"
            ).strip()

        title = (
            f"### 🎭 **Events near {location}**"
            if location
            else "### 🎭 **Events found**"
        )
        constraints = []
        if location:
            constraints.append(f"near **{location}**")
        if needs_evening:
            constraints.append("tonight")
        scope = " and ".join(constraints) if constraints else "with those criteria"
        event_noun = "free event" if requests_free else "event"
        return (
            f"{title}\n\n"
            f"✅ **Direct answer:** I could not confirm an {event_noun} {scope} with the available data.\n\n"
            "---\n\n"
            "⚠️ To avoid inventing proximity, schedule, or availability, I am not presenting generic Lisbon events as if they matched those criteria.\n\n"
            f"{self._build_events_source_line(language)}"
        ).strip()

    @staticmethod
    def _event_date_filter_label(date_filter: str, language: str) -> str:
        """Return a compact user-facing label for an event date filter."""
        normalized = str(date_filter or "").strip()
        if language != "pt":
            return normalized or "Requested period"
        month_labels = {
            "January": "janeiro",
            "February": "fevereiro",
            "March": "março",
            "April": "abril",
            "May": "maio",
            "June": "junho",
            "July": "julho",
            "August": "agosto",
            "September": "setembro",
            "October": "outubro",
            "November": "novembro",
            "December": "dezembro",
            "this month": "este mês",
            "next month": "próximo mês",
            "this week": "esta semana",
            "next week": "próxima semana",
            "this weekend": "este fim de semana",
        }
        return month_labels.get(normalized, normalized or "período pedido")

    @staticmethod
    def _strip_event_result_heading(result: str) -> str:
        """Remove per-tool headings/filter summaries before composing multi-period answers."""
        cleaned_lines: list[str] = []
        for raw_line in str(result or "").splitlines():
            stripped = raw_line.strip()
            if not stripped:
                if cleaned_lines and cleaned_lines[-1] != "":
                    cleaned_lines.append("")
                continue
            if stripped.startswith("### "):
                continue
            if re.match(r"^🧭\s+\*\*(?:Filtro aplicado|Filter used):\*\*", stripped, flags=re.IGNORECASE):
                continue
            if re.match(r"^📅\s+\*\*(?:Data de referência|Reference date):\*\*", stripped, flags=re.IGNORECASE):
                continue
            if stripped.startswith("💡 "):
                continue
            cleaned_lines.append(raw_line.rstrip())
        return "\n".join(cleaned_lines).strip()

    @staticmethod
    def _event_result_no_confirmed_events(result: str) -> bool:
        """Return whether an event block is an explicit no-result answer."""
        normalized = ResearcherAgent._normalize_event_preference_text(result)
        return bool(
            re.search(
                r"\b(?:nao encontrei eventos|nao ha eventos|sem eventos|"
                r"no confirmed events|no events found|did not find events)\b",
                normalized,
                flags=re.IGNORECASE,
            )
        )

    @classmethod
    def _format_multi_date_event_results(
        cls,
        date_results: list[tuple[str, str]],
        language: str,
    ) -> str:
        """Compose multiple date-filter tool results without duplicate section headings."""
        heading = "### 🎭 **Eventos encontrados**" if language == "pt" else "### 🎭 **Events found**"
        period_summaries: list[str] = []
        sections: list[str] = []

        for date_filter, raw_result in date_results:
            label = cls._event_date_filter_label(date_filter, language)
            stripped_result = str(raw_result or "").strip()
            body = cls._strip_event_result_heading(stripped_result)
            is_no_result = cls._event_result_no_confirmed_events(stripped_result)
            if is_no_result:
                sentence = next(
                    (
                        line.strip()
                        for line in body.splitlines()
                        if line.strip().startswith("❌")
                    ),
                    "❌ Não encontrei eventos confirmados nesse período."
                    if language == "pt"
                    else "❌ I did not find confirmed events in that period.",
                )
                period_summaries.append(
                    f"**{label}:** sem eventos confirmados"
                    if language == "pt"
                    else f"**{label}:** no confirmed events"
                )
                sections.append(f"- **📅 {label}:** {sentence}")
                continue

            period_summaries.append(
                f"**{label}:** eventos encontrados"
                if language == "pt"
                else f"**{label}:** events found"
            )
            sections.append(body)

        direct_label = "Resposta direta" if language == "pt" else "Direct answer"
        direct_text = "; ".join(period_summaries)
        return (
            f"{heading}\n\n"
            f"✅ **{direct_label}:** {direct_text}.\n\n"
            "---\n\n"
            + "\n\n".join(section for section in sections if section.strip())
        ).strip()

    def _run_direct_event_lookup(
        self,
        user_message: str,
        language: str,
        structured_plan: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Runs a deterministic VisitLisboa event lookup with explicit date parsing."""
        events_tool = self._get_tool_by_name("search_cultural_events")
        if not events_tool:
            return (
                "Não consegui aceder à pesquisa de eventos neste momento."
                if language == "pt"
                else "I couldn't access the event search tool right now."
            )

        requested_count = 10 if re.search(
            r"\b(?:todos?|todas?|all|everything|tudo|lista(?:r)?|list)\b",
            self._normalize_event_preference_text(user_message),
            flags=re.IGNORECASE,
        ) else 5
        args = {"max_results": requested_count, "language": language, "offset": 0}
        date_filters = [] if structured_plan else self._extract_event_date_filters(user_message)
        inherited_date_filter: Optional[str] = None
        if not structured_plan and not date_filters:
            last_search_context = getattr(self, "_last_search_context", None)
            last_base_args = (
                last_search_context.get("base_args", {})
                if isinstance(last_search_context, dict)
                and last_search_context.get("domain") == "events"
                and isinstance(last_search_context.get("base_args"), dict)
                else {}
            )
            if (
                last_base_args.get("date_filter")
                and (
                    self._infer_event_category_hint(user_message)
                    or self._extract_excluded_event_categories(user_message)
                    or self._is_named_lookup_followup(user_message)
                )
            ):
                inherited_date_filter = str(last_base_args["date_filter"]).strip() or None
        date_filter = self._normalize_structured_date_filter(
            structured_plan.get("date_filter"),
            user_message,
        ) if structured_plan else (date_filters[0] if len(date_filters) == 1 else inherited_date_filter)
        excluded_categories = self._extract_excluded_event_categories(user_message)
        category_hint = self._infer_event_category_hint(user_message)
        if category_hint:
            requested_category_keys = {
                self._event_category_key(category)
                for category in re.split(r"[,;/|]+", category_hint)
                if category.strip()
            }
            excluded_categories = [
                category for category in excluded_categories
                if self._event_category_key(category) not in requested_category_keys
            ]
        outdoor_event_query = self._is_outdoor_event_query(user_message)
        raw_extracted_focus_query = self._extract_event_focus_query(user_message)
        location_constraint = self._extract_event_location_constraint(user_message)
        if not location_constraint and re.search(
            r"\b(?:em|in|no|na|within)\s+(?:lisboa|lisbon)\b",
            self._normalize_event_preference_text(user_message),
            flags=re.IGNORECASE,
        ):
            location_constraint = "Lisboa"
        needs_evening_event = self._query_requests_evening_events(user_message)
        broad_date_discovery = (
            not structured_plan
            and bool(date_filter)
            and category_hint is None
            and not outdoor_event_query
            and not raw_extracted_focus_query
        )
        category_date_discovery = (
            not structured_plan
            and self._is_category_date_event_discovery(user_message)
            and not outdoor_event_query
        )
        category_discovery = (
            not structured_plan
            and self._is_category_event_discovery(user_message)
            and not outdoor_event_query
        )
        specific_lookup = None if structured_plan else _extract_specific_event_lookup_phrase(user_message)
        if self._specific_event_lookup_is_category_noise(specific_lookup):
            specific_lookup = None
        focus_query = self._normalize_structured_plan_text(structured_plan.get("subject")) if structured_plan else None
        strict_theme_focus = bool(
            raw_extracted_focus_query
            and re.search(
                r"\b(?:santos populares|arraiais|marchas populares|festas de lisboa)\b",
                self._normalize_event_preference_text(raw_extracted_focus_query),
                flags=re.IGNORECASE,
            )
        )
        cleaned_raw_focus_query = self._clean_event_focus_for_exclusions(
            raw_extracted_focus_query,
            excluded_categories,
            user_message,
        )
        precise_theme_focus = bool(
            cleaned_raw_focus_query
            and re.search(
                r"\b(?:jazz|rock|pop|fado|santos|arraiais|marchas)\b",
                self._normalize_event_preference_text(cleaned_raw_focus_query),
                flags=re.IGNORECASE,
            )
        )
        if broad_date_discovery:
            extracted_focus_query = None
        elif (
            category_date_discovery or category_discovery
        ) and not strict_theme_focus and not precise_theme_focus and str(raw_extracted_focus_query or "").lower() not in {"live music", "música ao vivo", "musica ao vivo"}:
            extracted_focus_query = None
        else:
            extracted_focus_query = cleaned_raw_focus_query
        if broad_date_discovery or category_date_discovery or category_discovery:
            specific_lookup = None
        elif not structured_plan and date_filter and not extracted_focus_query:
            specific_lookup = None
        focus_query = focus_query or specific_lookup or extracted_focus_query
        if outdoor_event_query:
            normalized_focus = self._normalize_event_preference_text(focus_query or "")
            non_generic_focus = re.sub(
                r"\b(?:eventos?|events?|gratuit[oa]s?|gratis|free|livre|outdoor|outdoors|ar|ao)\b",
                " ",
                normalized_focus,
                flags=re.IGNORECASE,
            )
            if not focus_query or not re.sub(r"\s+", " ", non_generic_focus).strip():
                focus_query = "eventos ao ar livre" if language == "pt" else "outdoor events"
        requests_free_event = bool(
            re.search(
                r"\b(?:gratuito|gratuitos|gratuita|gratuitas|gratis|gr[aá]tis|free|free entry|entrada gratuita)\b",
                user_message,
                flags=re.IGNORECASE,
            )
        )
        if requests_free_event and not focus_query:
            focus_query = "eventos gratuitos" if language == "pt" else "free events"
        elif requests_free_event and focus_query and not re.search(r"\b(?:gratuit|gratis|gr[aá]tis|free)\b", focus_query, flags=re.IGNORECASE):
            focus_query = f"{focus_query} gratuitos" if language == "pt" else f"{focus_query} free"
        excludes_outdoor_event = bool(
            re.search(
                r"\b(?:sem|nao|não|no|not|without|avoid|evitar|excluir|excluding|except|menos)\b"
                r"(?:\s+\w+){0,6}\s+\b(?:outdoor|outdoors|open\s+air|outside|ao\s+ar\s+livre|ar\s+livre|exterior)\b"
                r"|\b(?:que\s+nao\s+sejam|que\s+não\s+sejam|que\s+nao\s+seja|que\s+não\s+seja|not\s+outdoors?|not\s+outside)\b",
                self._normalize_event_preference_text(user_message),
                flags=re.IGNORECASE,
            )
        )
        if excludes_outdoor_event and not focus_query:
            focus_query = "sem eventos ao ar livre" if language == "pt" else "not outdoor events"
        elif excludes_outdoor_event and focus_query and not re.search(r"\b(?:ar\s+livre|outdoor|outside)\b", focus_query, flags=re.IGNORECASE):
            focus_query = f"{focus_query} sem eventos ao ar livre" if language == "pt" else f"{focus_query} not outdoor"
        if location_constraint:
            if requests_free_event:
                focus_query = (
                    f"{location_constraint} eventos gratuitos"
                    if language == "pt"
                    else f"{location_constraint} free events"
                )
            elif focus_query:
                if not specific_lookup:
                    focus_query = f"{location_constraint} {focus_query}"
            else:
                focus_query = f"{location_constraint} eventos" if language == "pt" else f"{location_constraint} events"

        if date_filter:
            args["date_filter"] = date_filter
        if category_hint:
            args["category"] = category_hint
        if excluded_categories:
            args["exclude_categories"] = ", ".join(excluded_categories)
        if focus_query:
            args["query"] = focus_query
        if specific_lookup:
            args["specific_lookup"] = True

        if len(date_filters) > 1:
            dated_result_blocks: list[tuple[str, str]] = []
            for scoped_date_filter in date_filters:
                scoped_args = dict(args)
                scoped_args["date_filter"] = scoped_date_filter
                scoped_result = str(
                    self._invoke_tool(events_tool, scoped_args, tool_name="search_cultural_events")
                ).strip()
                if scoped_result:
                    dated_result_blocks.append((scoped_date_filter, scoped_result))
            result = self._format_multi_date_event_results(dated_result_blocks, language)
        else:
            result = str(self._invoke_tool(events_tool, args, tool_name="search_cultural_events")).strip()
        if language == "pt":
            result = result.replace("Family & Kids", "Família e Crianças")
        result = self._filter_event_result_for_excluded_museum_venues(result, user_message, language)
        if re.search(r"\b(?:m[uú]sica\s+ao\s+vivo|musica\s+ao\s+vivo|live\s+music)\b", user_message, flags=re.IGNORECASE):
            live_note = (
                "⚠️ **Nota:** os dados confirmam a categoria **Música**; quando a página não explicita atuação ao vivo, confirma no detalhe do evento antes de ir."
                if language == "pt"
                else "⚠️ **Note:** the data confirms the **Music** category; when the page does not explicitly say it is live, confirm on the event page before going."
            )
            if live_note not in result:
                result = f"{result.rstrip()}\n\n{live_note}"
        if location_constraint and not self._event_result_mentions_location(result, location_constraint):
            if not self._has_specific_lookup_fallback_intro(result):
                return self._event_constraint_limitation(
                    language=language,
                    location=location_constraint,
                    needs_evening=needs_evening_event,
                    requests_free=requests_free_event,
                )
        if needs_evening_event and not self._event_result_has_evening_time(result):
            return self._event_constraint_limitation(
                language=language,
                location=location_constraint or "",
                needs_evening=True,
                requests_free=requests_free_event,
            )
        if result.startswith("❌") and not result.lstrip().startswith("###"):
            heading = "### 🎭 **Eventos encontrados**" if language == "pt" else "### 🎭 **Events found**"
            result = f"{heading}\n\n{result}"
        shown_count = self._count_ranked_results(result)
        remembered_page_size = shown_count if (specific_lookup and shown_count and self._has_specific_lookup_fallback_intro(result)) else int(args["max_results"])
        base_args = {key: value for key, value in args.items() if key not in {"max_results", "offset"}}
        self._remember_search_context(
            domain="events",
            tool_name="search_cultural_events",
            base_args=base_args,
            page_size=remembered_page_size,
            shown_count=shown_count,
            language=language,
            source_query=user_message,
            offset=0,
        )
        source_line = self._build_events_source_line(language)
        return f"{result}\n\n{source_line}".strip()

    @classmethod
    def _query_excludes_museum_venues(cls, user_message: str) -> bool:
        """Return whether the user explicitly excludes museums from event results."""
        normalized = cls._normalize_event_preference_text(user_message)
        broad_negation = bool(
            re.search(
                r"\b(?:sem|nao|not|without|avoid|excluding|except|menos|"
                r"nao\s+me\s+mostres?|nao\s+me\s+sugiras?|do\s+not\s+show|"
                r"don\s+t\s+show|do\s+not\s+suggest|don\s+t\s+suggest)\b"
                r"(?:\s+\w+){0,6}\s+\b(?:museus?|museums?)\b",
                normalized,
                flags=re.IGNORECASE,
            )
        )
        targeted_en = bool(
            re.search(
                r"\bno\s+(?:more\s+)?museums?\b",
                normalized,
                flags=re.IGNORECASE,
            )
        )
        return broad_negation or targeted_en

    @classmethod
    def _filter_event_result_for_excluded_museum_venues(
        cls,
        result: str,
        user_message: str,
        language: str,
    ) -> str:
        """Remove event cards hosted at venues explicitly excluded as museums."""
        if not result or not cls._query_excludes_museum_venues(user_message):
            return result

        lines = result.splitlines()
        intro: list[str] = []
        cards: list[list[str]] = []
        current: list[str] = []
        for line in lines:
            if re.match(r"^\s*[-*]\s+\*\*.+\*\*\s*$", line):
                if current:
                    cards.append(current)
                current = [line]
            elif current:
                current.append(line)
            else:
                intro.append(line)
        if current:
            cards.append(current)
        if not cards:
            return result

        kept_cards: list[list[str]] = []
        removed = 0
        for card in cards:
            block = "\n".join(card)
            if re.search(r"\b(?:museu|museus|museum|museums)\b", block, flags=re.IGNORECASE):
                removed += 1
                continue
            kept_cards.append(card)
        if not removed:
            return result
        place_result = bool(
            re.search(
                r"\b(?:VisitLisboa Locais|Locais e atra[cç][oõ]es|Places and attractions|"
                r"Local encontrado|Place found)\b",
                result,
                flags=re.IGNORECASE,
            )
        )
        if not kept_cards:
            if place_result:
                heading = "### 🏛️ **Sem locais confirmados**" if language == "pt" else "### 🏛️ **No confirmed places**"
            else:
                heading = "### 🎭 **Sem eventos confirmados**" if language == "pt" else "### 🎭 **No confirmed events**"
            direct_label = "Resposta direta" if language == "pt" else "Direct answer"
            if place_result:
                direct = (
                    f"Não encontrei locais que respeitem a exclusão de museus nos dados disponíveis; omiti {removed} resultado(s) associado(s) a museus."
                    if language == "pt"
                    else f"I did not find places that respect the museum exclusion in the available data; I omitted {removed} museum-associated result(s)."
                )
            else:
                direct = (
                    f"Não encontrei eventos que respeitem a exclusão de museus nos dados disponíveis; omiti {removed} resultado(s) associado(s) a museus."
                    if language == "pt"
                    else f"I did not find events that respect the museum exclusion in the available data; I omitted {removed} museum-associated result(s)."
                )
            return f"{heading}\n\n✅ **{direct_label}:** {direct}"

        if place_result:
            note = (
                f"⚠️ Omiti {removed} local(is) associado(s) a museus, conforme pediste."
                if language == "pt"
                else f"⚠️ I omitted {removed} place(s) associated with museums, as requested."
            )
        else:
            note = (
                f"⚠️ Omiti {removed} evento(s) associado(s) a espaços identificados como museu, conforme pediste."
                if language == "pt"
                else f"⚠️ I omitted {removed} event(s) associated with venues identified as museums, as requested."
            )
        output_lines = intro
        while output_lines and not output_lines[-1].strip():
            output_lines.pop()
        output_lines.extend(["", note, ""])
        for index, card in enumerate(kept_cards):
            if index:
                output_lines.append("")
            output_lines.extend(card)
        return "\n".join(output_lines).strip()

    @staticmethod
    def _should_add_place_history_context(user_message: str, result: str) -> bool:
        """Return whether a specific place answer should include concise historical context."""
        normalized_query = unicodedata.normalize("NFKD", user_message or "")
        normalized_query = normalized_query.encode("ascii", "ignore").decode("ascii").lower()
        asks_about_place = any(
            marker in normalized_query
            for marker in (
                "fala-me",
                "fala me",
                "fale-me",
                "fale me",
                "tell me about",
                "talk to me about",
                "details about",
                "information about",
                "detalhes",
                "sobre",
                "historia",
                "history",
                "historical",
            )
        )
        if not asks_about_place:
            return False

        normalized_result = unicodedata.normalize("NFKD", result or "")
        normalized_result = normalized_result.encode("ascii", "ignore").decode("ascii").lower()
        historical_markers = (
            "monument",
            "monumento",
            "museum",
            "museu",
            "palace",
            "palacio",
            "castle",
            "castelo",
            "unesco",
            "patrimonio",
            "heritage",
        )
        return any(marker in normalized_result for marker in historical_markers)

    @staticmethod
    def _extract_primary_place_title(result: str) -> str:
        """Extract the first canonical place-card title from a formatted tool result."""
        generic_titles = {
            "locais e atracoes",
            "places and attractions",
            "lisbon places and attractions",
        }

        def _canonical_title(raw_title: str) -> str:
            title = re.sub(r"^\*+|\*+$", "", raw_title or "").strip()
            normalized = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii").lower()
            return title if normalized not in generic_titles else ""

        for pattern in (
            r"(?m)^\s*\*\*(?:[\U0001F300-\U0001FAFF\u2300-\u27BF\uFE0F\u200D]+\s+)?(?P<title>[^*\n]+?)\*\*\s*$",
            r"(?m)^\s*[-*]\s+\*\*(?:[\U0001F300-\U0001FAFF\u2300-\u27BF\uFE0F\u200D]+\s+)?(?P<title>[^*\n]+?)\*\*\s*$",
        ):
            card_match = re.search(pattern, result or "")
            if not card_match:
                continue
            title = _canonical_title(card_match.group("title"))
            if title:
                return title

        for match in re.finditer(r"(?m)^###\s+(?:[\U0001F300-\U0001FAFF\u2300-\u27BF\uFE0F\u200D]+\s+)?(.+?)\s*$", result or ""):
            title = _canonical_title(match.group(1))
            if title:
                return title
        return ""

    @staticmethod
    def _known_place_history_facts(subject: str, language: str) -> List[str]:
        """Return curated Lisbon historical facts for landmarks where web snippets are noisy."""
        normalized_subject = unicodedata.normalize("NFKD", subject or "")
        normalized_subject = normalized_subject.encode("ascii", "ignore").decode("ascii").lower()

        if "jeronimos" in normalized_subject:
            if language == "pt":
                return [
                    "- O Mosteiro dos Jerónimos foi mandado construir por D. Manuel I no início do século XVI e é um dos exemplos mais importantes da arquitetura manuelina.",
                    "- Está associado aos Descobrimentos portugueses, à memória de Vasco da Gama e à antiga comunidade monástica de Santa Maria de Belém.",
                    "- Integra, com a Torre de Belém, a classificação de Património Mundial da UNESCO atribuída em 1983.",
                ]
            return [
                "- It was commissioned by King Manuel I in the early 16th century and is one of the most important examples of Manueline architecture.",
                "- It is closely linked to Portugal's maritime expansion, Vasco da Gama's memory, and the former monastic community of Santa Maria de Belém.",
                "- Together with Belém Tower, it has been a UNESCO World Heritage Site since 1983.",
            ]

        if "marvila" in normalized_subject:
            if language == "pt":
                return [
                    "- Marvila situa-se na zona oriental de Lisboa, numa área historicamente marcada pela ligação ao Tejo e por antigas quintas, conventos e núcleos industriais.",
                    "- A sua leitura urbana combina uma frente ribeirinha de armazéns e atividade produtiva com bairros residenciais desenvolvidos ao longo do século XX.",
                    "- Nas últimas décadas, antigas áreas industriais e logísticas de Marvila e do Beato ganharam nova centralidade cultural, criativa e residencial.",
                ]
            return [
                "- Marvila is in eastern Lisbon, in an area historically shaped by the Tagus waterfront and by former estates, convents, warehouses, and industrial uses.",
                "- Its urban fabric combines riverside productive spaces with residential neighbourhoods that expanded through the twentieth century.",
                "- In recent decades, former industrial and logistics areas around Marvila and Beato have gained renewed cultural, creative, and residential importance.",
            ]

        if "torre de belem" in normalized_subject or "belem tower" in normalized_subject:
            if language == "pt":
                return [
                    "- Foi construída no início do século XVI como estrutura defensiva na entrada do Tejo.",
                    "- Tornou-se um símbolo da Lisboa marítima e da arquitetura manuelina.",
                    "- Integra, com o Mosteiro dos Jerónimos, a classificação de Património Mundial da UNESCO atribuída em 1983.",
                ]
            return [
                "- It was built in the early 16th century as a defensive structure at the Tagus entrance.",
                "- It became a symbol of maritime Lisbon and Manueline architecture.",
                "- Together with Jerónimos Monastery, it has been a UNESCO World Heritage Site since 1983.",
            ]

        if "castelo de sao jorge" in normalized_subject or "sao jorge" in normalized_subject or "st george" in normalized_subject:
            if language == "pt":
                return [
                    "- A colina do castelo foi ocupada desde períodos antigos e ganhou importância estratégica durante o domínio islâmico de Lisboa.",
                    "- A fortificação medieval consolidou-se como alcáçova e cidadela no período islâmico, com referências importantes a partir de meados do século XI.",
                    "- Após a conquista de Lisboa em 1147, tornou-se um núcleo simbólico do poder régio e integrou a área palaciana usada pela corte portuguesa.",
                    "- Mais tarde, com a transferência progressiva da corte para a zona ribeirinha, o castelo perdeu centralidade palaciana, mas preservou muralhas, vistas e vestígios arqueológicos essenciais para ler várias camadas da história urbana de Lisboa.",
                ]
            return [
                "- The castle hill has been occupied since ancient periods and became strategically important during Islamic rule in Lisbon.",
                "- The medieval fortification developed as an alcáçova and citadel in the Islamic period, with important references from around the mid-11th century.",
                "- After the conquest of Lisbon in 1147, it became a symbolic seat of royal power and part of the palatial area used by the Portuguese court.",
                "- As court life later shifted toward the riverfront, the castle lost palatial centrality, but its walls, views, and archaeological remains still reveal several layers of Lisbon's urban history.",
            ]

        if "padrao dos descobrimentos" in normalized_subject or "discoveries" in normalized_subject:
            if language == "pt":
                return [
                    "- Foi concebido como monumento evocativo dos Descobrimentos portugueses e da relação histórica de Belém com a navegação oceânica.",
                    "- A forma de caravela e o conjunto escultórico destacam figuras ligadas à expansão marítima portuguesa.",
                    "- A localização ribeirinha reforça a ligação simbólica entre Lisboa, o Tejo e as viagens atlânticas.",
                ]
            return [
                "- It was conceived as a monument evoking the Portuguese Discoveries and Belém's historical link with ocean navigation.",
                "- Its caravel-like shape and sculptural group highlight figures connected with Portuguese maritime expansion.",
                "- Its riverside location reinforces the symbolic link between Lisbon, the Tagus, and Atlantic voyages.",
            ]

        return []

    @staticmethod
    def _compact_history_result(raw_result: str, language: str, subject: str = "") -> str:
        """Build a short bullet list from a history/web result without leaking raw search noise."""
        text = str(raw_result or "").strip()
        if not text or text.startswith(("❌", "Error:")):
            return ""
        normalized_subject = unicodedata.normalize("NFKD", subject or "")
        normalized_subject = normalized_subject.encode("ascii", "ignore").decode("ascii").lower()
        if re.search(r"\b(?:lisboa|lisbon)\b", normalized_subject) and re.search(
            r"\b(?:1800|seculo\s+xviii|seculo\s+xix|18th\s+century|19th\s+century|pombal|pombaline)\b",
            normalized_subject,
        ):
            if language == "pt":
                return "\n".join(
                    [
                        "- Por volta de 1800, Lisboa ainda era marcada pela reconstrução pombalina posterior ao terramoto de 1755.",
                        "- A Baixa afirmava-se como centro administrativo e comercial, com ruas mais regulares e funções urbanas reorganizadas.",
                        "- O porto e o Tejo continuavam a estruturar a economia, a circulação de mercadorias e a ligação imperial atlântica.",
                        "- A cidade combinava modernização iluminista com desigualdades sociais, freguesias densas e forte presença religiosa.",
                        "- No início do século XIX, Lisboa aproximava-se das invasões francesas e das tensões políticas que mudariam o país.",
                    ]
                )
            return "\n".join(
                [
                    "- Around 1800, Lisbon was still shaped by the Pombaline reconstruction after the 1755 earthquake.",
                    "- Baixa had become a more regular administrative and commercial core, with reorganised urban functions.",
                    "- The port and the Tagus continued to structure trade, mobility, and Portugal's Atlantic imperial links.",
                    "- The city mixed Enlightenment-era modernisation with social inequality, dense parishes, and strong religious institutions.",
                    "- By the early nineteenth century, Lisbon was approaching the French invasions and political tensions that would reshape Portugal.",
                ]
            )
        curated_facts = ResearcherAgent._known_place_history_facts(subject, language)
        subject_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKD", subject or "").encode("ascii", "ignore").decode("ascii").lower())
            if len(token) >= 4 and token not in {"historia", "history", "lisboa", "lisbon", "portugal", "volta", "around", "roteiro"}
        }
        text = re.sub(r"(?m)^\s*📌\s+\*\*(?:Fontes?|Sources?):.*$", "", text)
        text = re.sub(
            r"(?im)^\s*[📚🌐🔎]*\s*(?:\*\*)?(?:Wikip[eé]dia|Wikipedia):\s*[^*\n]+(?:\*\*)?\s*",
            "",
            text,
        )
        text = re.sub(r"(?m)^\s*🔗\s*\[[^\]]+\]\((?:[^()]|\([^()]*\))+\)\s*$", "", text)
        text = re.sub(r"(?m)^\s*🔗\s*(?:URL:\s*)?https?://\S+\s*$", "", text)
        text = re.sub(
            r"\[[^\]]+\]\((?:[^()]|\([^()]*\))+\)",
            lambda match: match.group(0).split("](", 1)[0].lstrip("["),
            text,
        )
        text = re.sub(r"(?m)^#{1,6}\s+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+", text)
        bullets: List[str] = []
        previous_sentence_was_related = False
        for sentence in sentences:
            cleaned = re.sub(r"\*+", "", sentence).strip(" -*•\t")
            if len(cleaned) < 45 or len(cleaned) > 320:
                previous_sentence_was_related = False
                continue
            if re.search(r"\b(?:http|source|fonte|search|pesquisa)\b", cleaned, flags=re.IGNORECASE):
                previous_sentence_was_related = False
                continue
            normalized_sentence = unicodedata.normalize("NFKD", cleaned)
            normalized_sentence = normalized_sentence.encode("ascii", "ignore").decode("ascii").lower()
            has_subject_reference = not subject_tokens or any(token in normalized_sentence for token in subject_tokens)
            has_historical_cue = bool(
                re.search(
                    r"\b(?:manuel|unesco|patrimonio|heritage|descobr|maritim|maritime|constru|built|seculo|century|monast|mosteiro|arquitet|architecture|defens|royal|regio|tejo|tagus|terramoto|earthquake|pombal|pombaline|population|populacao|economia|economy|urban|reconstruction|reconstrucao|1800)\b",
                    normalized_sentence,
                    flags=re.IGNORECASE,
                )
            )
            if subject_tokens:
                if not has_subject_reference and not (previous_sentence_was_related and has_historical_cue):
                    previous_sentence_was_related = False
                    continue
            bullets.append(f"- {cleaned}")
            previous_sentence_was_related = True
            if len(bullets) == 4:
                break
        if curated_facts:
            return "\n".join(curated_facts)
        if bullets:
            return "\n".join(bullets)
        return ""

    @staticmethod
    def _extend_place_source_line_with_history(source_line: str) -> str:
        """Add the historical-context source to a VisitLisboa place footer."""
        addition = "[*Wikipedia/Web*](https://www.wikipedia.org/)"
        if addition in source_line:
            return source_line
        return f"{source_line} | {addition}" if source_line else addition

    def _build_place_history_section(
        self,
        subject: str,
        user_message: str,
        result: str,
        language: str,
    ) -> tuple[str, bool]:
        """Fetch and format a concise historical context section for a specific place.

        Returns:
            A tuple with the formatted section and whether an external history source was
            queried. Curated facts are intentionally not attributed to the web footer.
        """
        if not subject or not self._should_add_place_history_context(user_message, result):
            return "", False
        canonical_subject = self._extract_primary_place_title(result) or subject
        used_external_history_source = False
        compact_history = ""

        knowledge_tool = self._get_tool_by_name("search_lisbon_knowledge")
        if knowledge_tool:
            raw_knowledge = str(
                self._invoke_tool(
                    knowledge_tool,
                    {"query": canonical_subject, "max_results": 3},
                    tool_name="search_lisbon_knowledge",
                )
            ).strip()
            if raw_knowledge and not raw_knowledge.startswith(("❌", "Error:")) and len(raw_knowledge) > 80:
                compact_history = self._compact_history_result(raw_knowledge, language, canonical_subject)

        if not compact_history:
            curated_facts = self._known_place_history_facts(canonical_subject, language)
            if curated_facts:
                compact_history = "\n".join(curated_facts)

        if not compact_history:
            history_tool = self._get_tool_by_name("search_history_culture")
            if not history_tool:
                return "", False
            raw_history = str(
                self._invoke_tool(
                    history_tool,
                    {"query": canonical_subject, "language": language},
                    tool_name="search_history_culture",
                )
            ).strip()
            compact_history = self._compact_history_result(raw_history, language, canonical_subject)
            used_external_history_source = bool(compact_history)
        if not compact_history:
            return "", False
        heading = f"### 📜 Factos Históricos de {canonical_subject}" if language == "pt" else f"### 📜 Historical Facts About {canonical_subject}"
        return f"---\n\n{heading}\n\n{compact_history}", used_external_history_source

    @staticmethod
    def _maybe_answer_after_hours_culture_query(user_message: str, language: str) -> Optional[str]:
        """Answer museum/monument recommendation queries whose requested window is after indoor closing times."""
        normalized = unicodedata.normalize("NFKD", user_message or "")
        normalized = normalized.encode("ascii", "ignore").decode("ascii").lower()
        asks_culture = any(token in normalized for token in ("museu", "museum", "monumento", "monument"))
        asks_recommendation = any(token in normalized for token in ("recomendas", "recommend", "sugere", "suggest", "qual"))
        time_matches = re.findall(r"\b(\d{1,2})\s*(?:h|:00)?\b", normalized)
        requested_hours = [int(value) for value in time_matches if 0 <= int(value) <= 23]
        late_window = bool(requested_hours and max(requested_hours) >= 19)
        if not (asks_culture and asks_recommendation and late_window):
            return None

        requested_start = min(requested_hours) if requested_hours else 19
        requested_end = max(requested_hours) if requested_hours else requested_start + 1
        if requested_start == requested_end:
            requested_end = min(requested_start + 1, 23)
        requested_window = f"{requested_start:02d}:00-{requested_end:02d}:00"

        def normalize(value: Any) -> str:
            text = unicodedata.normalize("NFKD", str(value or ""))
            return text.encode("ascii", "ignore").decode("ascii").lower()

        places = _load_places_json()
        candidate_scores: list[tuple[int, Dict[str, Any]]] = []
        outside_lisbon_markers = (
            "vila franca de xira", "sintra", "cascais", "oeiras", "mafra",
            "loures", "odivelas", "almada", "seixal", "barreiro", "setubal",
            "setubal", "montijo", "alcochete", "palmela", "sesimbra",
        )
        preferred_outdoor_titles = (
            "torre de belem", "padrao dos descobrimentos", "arco da rua augusta",
            "se de lisboa", "castelo de sao jorge", "praca do comercio",
        )
        for place in places:
            title = str(place.get("title") or "")
            category = normalize(place.get("category"))
            address_text = normalize(" ".join(str(place.get(key) or "") for key in ("address", "location")))
            text = normalize(" ".join(str(place.get(key) or "") for key in ("title", "category", "short_description", "description")))
            normalized_title = normalize(title)
            if any(marker in address_text for marker in outside_lisbon_markers):
                continue
            if address_text and "lisboa" not in address_text and "belem" not in address_text:
                continue
            if any(proxy in category or proxy in normalize(title) for proxy in ("tram", "tour", "cruise", "hotel", "restaurant", "shopping")):
                continue
            if "museum" in category or "museu" in category:
                continue
            if "monument" not in category and "monumento" not in category:
                continue
            score = 0
            if any(preferred in normalized_title for preferred in preferred_outdoor_titles):
                score += 80
            if any(term in text for term in ("outdoor", "exterior", "view", "vista", "river", "tejo", "tagus", "miradouro", "panoramic")):
                score += 35
            if any(term in text for term in ("façade", "facade", "fachada", "monument", "monumento", "square", "praça")):
                score += 20
            if any(term in text for term in ("events", "eventos", "available for events", "palace halls")):
                score -= 20
            if place.get("address") or place.get("location"):
                score += 10
            if place.get("url"):
                score += 5
            candidate_scores.append((score, place))

        if not candidate_scores:
            return None

        _, selected = max(candidate_scores, key=lambda item: item[0])
        title = str(selected.get("title") or "monumento exterior").strip()
        address = str(selected.get("address") or selected.get("location") or "").strip()
        description = str(selected.get("short_description") or selected.get("description") or "").strip()
        url = str(selected.get("url") or "").strip()
        if len(description) > 220:
            description = description[:217].rsplit(" ", 1)[0] + "..."
        pt_description = (
            "Monumento visitável por fora, adequado para uma janela curta ao fim do dia quando a entrada interior em museus pode já não estar disponível."
        )
        en_description = (
            "Outdoor-viewable monument, suitable for a short evening window when indoor museum entry may no longer be available."
        )

        timestamp = datetime.now().strftime("%H:%M")
        maps_url = f"https://www.google.com/maps/search/?api=1&query={address.replace(' ', '+')}" if address else ""
        if language == "pt":
            lines = [
                f"### 🏛️ **Recomendação para {requested_window}**",
                "",
                "Para essa janela, eu evitaria recomendar **museus interiores**, porque muitos fecham antes ou perto das 18:00. A opção mais segura é um monumento exterior ou visitável por fora.",
                "",
                f"- 🏛️ **{title} (exterior)**",
            ]
            lines.append(f"    - 📝 **Porque faz sentido:** {pt_description}")
            if address and maps_url:
                lines.append(f"    - 📍 **Morada:** [{address}]({maps_url})")
            if url:
                lines.append(f"    - 🌐 **Mais detalhes:** [VisitLisboa]({url})")
            lines.extend([
                f"    - 🕐 **Janela recomendada:** {requested_window}, visita exterior.",
                "    - ℹ️ **Nota:** se quiseres entrada interior noutro museu ou monumento, confirma primeiro o horário oficial do próprio dia.",
                "",
                f"📌 **Fonte:** [*VisitLisboa Locais*](https://www.visitlisboa.com/pt-pt/locais) | **Atualizado:** {timestamp}",
            ])
            return "\n".join(lines)

        lines = [
            f"### 🏛️ **Recommendation for {requested_window}**",
            "",
            "For that window, I would avoid recommending **indoor museums**, because many close before or around 18:00. The safer option is an outdoor monument or a place that still works from outside.",
            "",
            f"- 🏛️ **{title} (outside)**",
        ]
        lines.append(f"    - 📝 **Why it fits:** {en_description}")
        if address and maps_url:
            lines.append(f"    - 📍 **Address:** [{address}]({maps_url})")
        if url:
            lines.append(f"    - 🌐 **More details:** [VisitLisboa]({url})")
        lines.extend([
            f"    - 🕐 **Recommended window:** {requested_window}, outside visit.",
            "    - ℹ️ **Note:** if you want indoor entry somewhere else, confirm the official opening hours for that exact day first.",
            "",
            f"📌 **Source:** [*VisitLisboa Places*](https://www.visitlisboa.com/en/places) | **Updated:** {timestamp}",
        ])
        return "\n".join(lines)

    @staticmethod
    def _is_free_museum_event_query(user_message: str) -> bool:
        """Detect mixed free-museum plus free-event requests.

        These queries are high risk because place and event cards can be
        accidentally merged by free-form synthesis. Keep the answer separated:
        museum availability is only stated when confirmed; event suggestions
        come from the event source.
        """
        normalized = unicodedata.normalize("NFKD", user_message or "")
        normalized = normalized.encode("ascii", "ignore").decode("ascii").lower()
        has_museum = any(token in normalized for token in ("museu", "museus", "museum", "museums"))
        has_event = any(token in normalized for token in ("evento", "event", "events", "festival", "concerto", "concert"))
        has_free = any(token in normalized for token in ("gratuito", "gratuitos", "gratis", "free"))
        has_weekend = any(token in normalized for token in ("fim de semana", "weekend", "sabado", "domingo", "saturday", "sunday"))
        return has_museum and has_event and has_free and has_weekend

    def _run_free_museum_event_guard(self, language: str) -> str:
        """Answer free-museum + free-event questions without merging cards."""
        events_tool = self._get_tool_by_name("search_cultural_events")
        timestamp = datetime.now().strftime("%H:%M")
        event_result = ""
        if events_tool:
            try:
                event_result = str(self._invoke_tool(
                    events_tool,
                    {
                        "query": "free events" if language != "pt" else "eventos gratuitos",
                        "date_filter": "this weekend",
                        "max_results": 3,
                        "language": language,
                        "offset": 0,
                    },
                    tool_name="search_cultural_events",
                )).strip()
            except Exception:
                event_result = ""
        event_result = re.sub(r"(?mi)^📌\s*\*\*(?:Fonte|Source):\*\*.*$", "", event_result).strip()

        if language == "pt":
            lines = [
                "### 🏛️ **Museus e eventos gratuitos em Lisboa**",
                "",
                "✅ **Resposta direta:** não vou juntar museus e eventos no mesmo cartão. Nos dados disponíveis, não confirmei uma lista segura de **museus com entrada gratuita exatamente neste fim de semana**.",
                "",
                "⚠️ **Limitação:** benefícios como \"gratuito com Lisboa Card\" não são o mesmo que entrada gratuita geral; por isso não os trato como museus gratuitos.",
            ]
            if event_result and not event_result.startswith(("❌", "Error:")):
                lines.extend(["", "### 🎭 **Eventos gratuitos encontrados**", "", event_result])
            else:
                lines.extend(["", "### 🎭 **Eventos gratuitos encontrados**", "", "- Não encontrei eventos gratuitos confirmados para este filtro nos dados disponíveis."])
            lines.extend(["", f"📌 **Fonte:** [*VisitLisboa Eventos*](https://www.visitlisboa.com/pt-pt/eventos) | **Atualizado:** {timestamp}"])
            return "\n".join(lines)

        lines = [
            "### 🏛️ **Free museums and free events in Lisbon**",
            "",
            "✅ **Direct answer:** I will not merge museum and event data into one card. In the available data, I could not safely confirm a list of **museums with general free entry exactly this weekend**.",
            "",
            "⚠️ **Limitation:** benefits such as \"free with Lisboa Card\" are not the same as general free entry, so I do not label them as free museums.",
        ]
        if event_result and not event_result.startswith(("❌", "Error:")):
            lines.extend(["", "### 🎭 **Free events found**", "", event_result])
        else:
            lines.extend(["", "### 🎭 **Free events found**", "", "- I did not find confirmed free events for this filter in the available data."])
        lines.extend(["", f"📌 **Source:** [*VisitLisboa Events*](https://www.visitlisboa.com/en/events) | **Updated:** {timestamp}"])
        return "\n".join(lines)

    def _run_direct_place_lookup(
        self,
        user_message: str,
        language: str,
        structured_plan: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Runs a deterministic tool path for simple place and multi-service lookups."""
        places_tool = self._get_tool_by_name("search_places_attractions")
        nearby_tool = self._get_tool_by_name("find_nearby_services")
        structured_subject = self._normalize_structured_plan_text(structured_plan.get("subject")) if structured_plan else None
        place_focus_query = structured_subject or self._extract_place_focus_query(user_message) or self._extract_place_area_filter(user_message)
        transactional_lookup = bool(
            re.search(
                r"\b(?:book|reserve|reservation|booking|buy|purchase|"
                r"reservar|reserva|marcar|marca|comprar|compra)\b",
                user_message or "",
                flags=re.IGNORECASE,
            )
        )
        specific_lookup = _extract_specific_place_lookup_phrase(user_message)
        if (
            structured_subject
            and not specific_lookup
            and not self._structured_subject_looks_like_generic_place_query(structured_subject)
        ):
            specific_lookup = structured_subject
        structured_category_hint = self._canonical_structured_place_category_hint(
            structured_plan.get("category_hint") if structured_plan else None
        )
        service_types = self._extract_service_types(user_message)
        for structured_service in structured_plan.get("service_types", []) if structured_plan else []:
            tool_label = self._structured_service_tool_label(structured_service)
            existing_service_ids = {self._service_type_identity(service_type) for service_type in service_types}
            if tool_label and self._service_type_identity(tool_label) not in existing_service_ids:
                service_types.append(tool_label)
        nearby_location = self._normalize_structured_plan_text(structured_plan.get("near_location")) if structured_plan else None
        nearby_location = self._clean_nearby_location_text(nearby_location)
        nearby_location = nearby_location or self._extract_near_location_name(user_message)
        service_types = self._filter_location_anchored_service_types(service_types, nearby_location)
        if service_types and self._query_references_user_current_location(user_message) and not nearby_location:
            return self._build_current_location_clarification(language)
        is_broad_attractions = self._is_broad_attractions_query(user_message)
        area_labels = extract_aml_municipality_mentions(user_message)
        area_label = area_labels[0] if len(area_labels) == 1 else ""
        if service_types and area_label and self._normalize_for_deterministic_routing(area_label) != "lisboa":
            return self._build_area_service_coverage_limitation(
                service_types=service_types,
                area_label=area_label,
                language=language,
            )

        if (
            places_tool
            and area_label
            and not service_types
            and self._is_area_mixed_food_place_query(user_message)
        ):
            return self._run_area_scoped_mixed_place_lookup(
                places_tool=places_tool,
                user_message=user_message,
                area_label=area_label,
                language=language,
            )

        if places_tool and not service_types and not specific_lookup:
            multi_component_response = self._run_multi_component_place_lookup(
                places_tool=places_tool,
                user_message=user_message,
                language=language,
            )
            if multi_component_response:
                return multi_component_response

        if places_tool and place_focus_query and specific_lookup and not is_broad_attractions and not service_types:
            exact_args = {
                "query": place_focus_query,
                "max_results": 5,
                "offset": 0,
                "language": language,
                "specific_lookup": True,
            }
            exact_category_hint = self._infer_place_category_hint(user_message) or structured_category_hint
            if exact_category_hint:
                exact_args["category"] = exact_category_hint

            exact_result = str(self._invoke_tool(places_tool, exact_args, tool_name="search_places_attractions")).strip()
            exact_result = self._localize_place_card_titles_with_llm(exact_result, language)
            # Accept clean exact match OR fallback-with-ranked-alternatives to avoid a costlier broader retry.
            if exact_result and not exact_result.startswith("Error:"):
                has_fallback_intro = self._has_specific_lookup_fallback_intro(exact_result)
                shown_count = self._count_ranked_results(exact_result)
                accept_clean_exact = not has_fallback_intro and not exact_result.startswith("❌")
                accept_fallback_with_alternatives = has_fallback_intro and shown_count > 0
                if transactional_lookup and accept_clean_exact:
                    exact_result = self._filter_transactional_place_contact_fields(exact_result)
                    source_line = self._build_places_source_line(exact_result, language)
                    if language == "pt":
                        direct_note = (
                            "### ⚠️ **Reserva não suportada**\n\n"
                            "✅ **Resposta direta:** não consigo fazer a reserva diretamente; abaixo estão apenas os detalhes confirmados do local para contactares pelos canais oficiais.\n\n"
                            "---"
                        )
                    else:
                        direct_note = (
                            "### ⚠️ **Booking Not Supported**\n\n"
                            "✅ **Direct answer:** I cannot make the booking directly; below are only the confirmed venue details so you can contact the official channels.\n\n"
                            "---"
                        )
                    return f"{direct_note}\n\n{exact_result}\n\n{source_line}".strip()
                if transactional_lookup and not accept_clean_exact:
                    source_line = self._build_places_source_line(exact_result, language)
                    target_label = place_focus_query or specific_lookup or ("o local pedido" if language == "pt" else "the requested venue")
                    if language == "pt":
                        return (
                            "### ⚠️ **Reserva não suportada**\n\n"
                            f"✅ **Resposta direta:** não consigo fazer reservas e não encontrei detalhes oficiais confirmados para **{target_label}** nos dados disponíveis.\n\n"
                            "---\n\n"
                            "- Posso ajudar com morada, contactos ou transporte se indicares o website oficial ou a morada do local.\n\n"
                            f"{source_line}"
                        ).strip()
                    return (
                        "### ⚠️ **Booking Not Supported**\n\n"
                        f"✅ **Direct answer:** I cannot make bookings, and I could not confirm official details for **{target_label}** in the available data.\n\n"
                        "---\n\n"
                        "- I can still help with address, contact details, or transport if you provide the official website or venue address.\n\n"
                        f"{source_line}"
                    ).strip()
                if accept_clean_exact or accept_fallback_with_alternatives:
                    base_args = {key: value for key, value in exact_args.items() if key not in {"max_results", "offset"}}
                    remembered_page_size = (
                        shown_count if (accept_fallback_with_alternatives and shown_count) else int(exact_args["max_results"])
                    )
                    self._remember_search_context(
                        domain="places",
                        tool_name="search_places_attractions",
                        base_args=base_args,
                        page_size=remembered_page_size,
                        shown_count=shown_count,
                        language=language,
                        source_query=user_message,
                        offset=0,
                    )
                    history_section, used_external_history_source = self._build_place_history_section(
                        place_focus_query,
                        user_message,
                        exact_result,
                        language,
                    )
                    source_line = self._build_places_source_line(exact_result, language)
                    exact_result = self._append_subjective_place_preference_caveat(
                        exact_result,
                        user_message,
                        language,
                    )
                    if history_section:
                        if used_external_history_source:
                            source_line = self._extend_place_source_line_with_history(source_line)
                        return f"{exact_result}\n\n{history_section}\n\n{source_line}".strip()
                    return f"{exact_result}\n\n{source_line}".strip()

        if nearby_tool and service_types:
            service_blocks: List[str] = []
            missing_services: List[str] = []

            for service_type in service_types:
                service_args = {
                    "service_type": service_type,
                    "max_results": 5,
                    "language": language,
                }
                if nearby_location:
                    service_args["near_location_name"] = nearby_location
                category_hint = self._service_category_for_type(service_type)
                if not category_hint and structured_plan:
                    normalized_structured_services = structured_plan.get("service_types", [])
                    for structured_service in normalized_structured_services:
                        structured_label = self._structured_service_tool_label(structured_service)
                        if (
                            structured_label
                            and self._service_type_identity(structured_label) == self._service_type_identity(service_type)
                        ):
                            category_hint = self._structured_service_category(structured_service)
                            break
                if category_hint:
                    service_args["category"] = category_hint

                result = str(
                    self._invoke_tool(
                        nearby_tool,
                        service_args,
                        tool_name="find_nearby_services",
                    )
                ).strip()

                if result and not result.startswith(("❌", "Error:")):
                    service_blocks.append(result)
                else:
                    missing_services.append(service_type)

            if service_blocks:
                combined = "\n\n".join(service_blocks).strip()
                combined = self._strip_lisboa_aberta_source_lines(combined)
                nearest_summary = self._build_nearby_services_direct_summary(
                    service_blocks,
                    language=language,
                )
                if nearest_summary:
                    combined = f"{nearest_summary}\n\n---\n\n{combined}"
                time_sensitive_requested = bool(re.search(
                    r"\b(evening|tonight|late|after\s+hours|this\s+evening|now|right\s+now|open|opened|"
                    r"hours?|opening\s+hours?|hor[áa]rios?|hor[áa]rio\s+atual|"
                    r"noite|esta\s+noite|ao\s+fim\s+do\s+dia|mais\s+tarde|agora|abert[ao]s?|funciona|dispon[ií]vel)\b",
                    user_message,
                    re.IGNORECASE,
                ))
                if time_sensitive_requested:
                    health_requested = bool(re.search(
                        r"\b(pharmacy|pharmacies|farm[áa]cia|farm[áa]cias|hospital|hospitais|health|sa[úu]de)\b",
                        user_message,
                        re.IGNORECASE,
                    ))
                    if language == "pt":
                        if health_requested:
                            combined += "\n\n⚠️ A Lisboa Aberta confirma localização e proximidade, mas não confirma horário atual, turno de farmácia ou disponibilidade clínica. Em urgência, contacta o 112; para farmácia, confirma por telefone antes de te deslocares."
                        else:
                            combined += "\n\n⚠️ A Lisboa Aberta confirma localização e proximidade, mas não confirma horário atual nem disponibilidade ao fim do dia. Confirma diretamente antes de te deslocares."
                    else:
                        if health_requested:
                            combined += "\n\n⚠️ Lisboa Aberta confirms location and proximity, but not current opening hours, pharmacy duty status, or clinical availability. For emergencies call 112; for pharmacies, confirm by phone before going."
                        else:
                            combined += "\n\n⚠️ Lisboa Aberta confirms location and proximity, but not current opening hours or evening availability. Confirm directly before going."
                accessibility_requested = bool(re.search(
                    r"\b(accessible|accessibility|wheelchair|step[-\s]?free|acess[ií]vel|acessibilidade|cadeira\s+de\s+rodas|mobilidade\s+reduzida)\b",
                    user_message,
                    re.IGNORECASE,
                ))
                if accessibility_requested:
                    if language == "pt":
                        combined += "\n\n⚠️ A Lisboa Aberta confirma localização e proximidade; características de acessibilidade só estão confirmadas quando aparecem explicitamente nos campos acima."
                    else:
                        combined += "\n\n⚠️ Lisboa Aberta confirms location and proximity; accessibility features are confirmed only when explicitly shown in the fields above."
                broad_catalog_requested = not nearby_location and bool(re.search(
                    r"\b(todas?|todos?|all|every|exaustiv[ao]s?|complet[ao]s?)\b",
                    user_message,
                    re.IGNORECASE,
                ))
                if broad_catalog_requested:
                    if language == "pt":
                        combined += "\n\n⚠️ Esta não é uma listagem exaustiva da AML; mostro apenas os primeiros resultados disponíveis no dataset consultado."
                    else:
                        combined += "\n\n⚠️ This is not an exhaustive AML-wide list; I am only showing the first available results from the consulted dataset."
                if missing_services:
                    missing_label = ", ".join(missing_services)
                    if language == "pt":
                        combined += f"\n\n⚠️ Não foi possível confirmar resultados para: {missing_label}."
                    else:
                        combined += f"\n\n⚠️ I could not confirm results for: {missing_label}."
                combined += f"\n\n{self._build_open_data_services_source_line(language)}"
                return combined.strip()
            if missing_services:
                return self._build_missing_services_limitation(
                    service_types=missing_services,
                    nearby_location=nearby_location,
                    language=language,
                )

        if not places_tool:
            return self._run_direct_tool_fallback(user_message, language)
        category_hint = self._infer_place_category_hint(user_message) or structured_category_hint
        requested_count = self._extract_requested_result_count(user_message)
        query_text = user_message if category_hint and not specific_lookup else (place_focus_query or user_message)
        max_results = requested_count or 5
        overfetch_result_count = requested_count if (
            requested_count and category_hint and place_focus_query and not specific_lookup
        ) else None
        tool_max_results = max_results
        if overfetch_result_count:
            tool_max_results = max(max_results, min(8, overfetch_result_count + 2))
        if category_hint and place_focus_query and requested_count and not specific_lookup:
            if category_hint == "Restaurants":
                query_text = (
                    f"restaurantes em {place_focus_query}"
                    if language == "pt"
                    else f"restaurants in {place_focus_query}"
                )
            elif category_hint == "Museums & Monuments":
                wants_monuments = bool(re.search(r"\b(?:monument|monuments|monumento|monumentos)\b", user_message, re.IGNORECASE))
                label = "monumentos conhecidos" if wants_monuments and language == "pt" else "well-known monuments" if wants_monuments else "museus e monumentos" if language == "pt" else "museums and monuments"
                connector = "em" if language == "pt" else "in"
                query_text = f"{label} {connector} {place_focus_query}"
            else:
                connector = "em" if language == "pt" else "in"
                query_text = f"{category_hint} {connector} {place_focus_query}"
        if is_broad_attractions:
            query_text = "must-see attractions first time visitors Lisbon iconic monuments museums palaces castles historic sites"
            max_results = 6
        elif self._is_visit_place_context_query(user_message) and place_focus_query and not category_hint:
            query_text = f"near {place_focus_query}"

        args = {"query": query_text, "max_results": tool_max_results, "offset": 0, "language": language}
        if specific_lookup and not is_broad_attractions:
            args["specific_lookup"] = True
        if is_broad_attractions:
            args["category"] = "Museums & Monuments"
        elif self._is_visit_place_context_query(user_message) and place_focus_query and not category_hint:
            args["category"] = "Museums & Monuments"
        elif category_hint:
            args["category"] = category_hint

        result = str(self._invoke_tool(places_tool, args, tool_name="search_places_attractions")).strip()
        if overfetch_result_count:
            result = self._trim_ranked_result_cards(result, overfetch_result_count)
        if is_broad_attractions and language == "pt":
            rewrite_result = getattr(self, "_rewrite_broad_attractions_result", None)
            if callable(rewrite_result):
                result = str(rewrite_result(result, user_message, language)).strip()
        result = self._localize_place_card_titles_with_llm(result, language)
        if area_label and not result.startswith(("❌", "Error:")):
            if category_hint == "Restaurants":
                category_label = "restaurantes" if language == "pt" else "restaurants"
            elif category_hint:
                category_label = "locais" if language == "pt" else "places"
            elif is_broad_attractions:
                category_label = "atrações" if language == "pt" else "attractions"
            else:
                category_label = "locais" if language == "pt" else "places"
            result = self._filter_area_scoped_place_result(
                result=result,
                area_label=area_label,
                language=language,
                category_label=category_label,
            )
        shown_count = self._count_ranked_results(result)
        remembered_page_size = shown_count if (specific_lookup and shown_count and self._has_specific_lookup_fallback_intro(result)) else int(args["max_results"])
        base_args = {key: value for key, value in args.items() if key not in {"max_results", "offset"}}
        self._remember_search_context(
            domain="places",
            tool_name="search_places_attractions",
            base_args=base_args,
            page_size=remembered_page_size,
            shown_count=shown_count,
            language=language,
            source_query=user_message,
            offset=0,
        )
        source_line = self._build_places_source_line(result, language)
        result = self._append_subjective_place_preference_caveat(
            result,
            user_message,
            language,
        )
        return f"{result}\n\n{source_line}".strip()

    @staticmethod
    def _filter_transactional_place_contact_fields(result: str) -> str:
        """Keep only non-transactional contact/navigation fields in place cards."""
        if not result:
            return result or ""
        blocked_field_re = re.compile(
            r"^\s*[-*]\s+.*?\*\*(?:"
            r"Hor[aá]rio|Opening hours|Hours|Pre[cç]o|Price|Bilhetes|Tickets|"
            r"Avalia[cç][aã]o|Rating|Disponibilidade|Availability"
            r"):\*\*",
            flags=re.IGNORECASE,
        )
        kept_lines = [line for line in result.splitlines() if not blocked_field_re.match(line)]
        return "\n".join(kept_lines).strip()

    def _run_direct_tool_fallback(self, user_message: str, language: str) -> str:
        """
        Runs a deterministic tool-only fallback when Azure blocks both prompt
        attempts. This avoids failing benign queries like 'Museums in Lisbon'.
        """
        message_lower = user_message.lower()

        history_keywords = ["history", "historical", "história", "historia", "culture", "cultura"]
        event_keywords = [
            "event", "events", "evento", "eventos", "concert", "concerto",
            "festival", "exhibition", "exposição", "exposicao", "show",
        ]
        category_keywords = ["categories", "categorias", "service categories", "tipos de serviços"]

        if any(keyword in message_lower for keyword in category_keywords):
            tool = self._get_tool_by_name("list_service_categories")
            if tool:
                return str(self._invoke_tool(tool, {}, tool_name="list_service_categories"))

        service_types = self._extract_service_types(user_message)
        if service_types:
            area_labels = extract_aml_municipality_mentions(user_message)
            area_label = area_labels[0] if len(area_labels) == 1 else ""
            if area_label and self._normalize_for_deterministic_routing(area_label) != "lisboa":
                return self._build_area_service_coverage_limitation(
                    service_types=service_types,
                    area_label=area_label,
                    language=language,
                )
            tool = self._get_tool_by_name("find_nearby_services")
            if tool:
                nearby_location = self._clean_nearby_location_text(
                    self._extract_near_location_name(user_message)
                )
                service_types = self._filter_location_anchored_service_types(service_types, nearby_location)
                if service_types and self._query_references_user_current_location(user_message) and not nearby_location:
                    return self._build_current_location_clarification(language)
                blocks: List[str] = []
                missing_services: List[str] = []

                for service_type in service_types:
                    args = {
                        "service_type": service_type,
                        "max_results": 5,
                        "language": language,
                    }
                    if nearby_location:
                        args["near_location_name"] = nearby_location
                    category_hint = self._service_category_for_type(service_type)
                    if category_hint:
                        args["category"] = category_hint

                    result = str(
                        self._invoke_tool(
                            tool,
                            args,
                            tool_name="find_nearby_services",
                        )
                    ).strip()
                    if result and not result.startswith(("❌", "Error:")):
                        blocks.append(result)
                    else:
                        missing_services.append(service_type)

                if blocks:
                    combined = "\n\n".join(blocks)
                    combined = self._strip_lisboa_aberta_source_lines(combined)
                    if any("parking" in service_type for service_type in service_types) and re.search(
                        r"\b(municipal|car\s+parks?|park\s+my\s+car|municipais|carro|viatura)\b",
                        message_lower,
                    ):
                        parking_note = (
                            "I found parking-related locations near the requested area in Lisboa Aberta. "
                            "The available data does not confirm municipal ownership or complete car-park coverage for each result."
                            if language == "en"
                            else "Encontrei locais relacionados com estacionamento perto da zona pedida na Lisboa Aberta. "
                            "Os dados disponíveis não confirmam a titularidade municipal nem cobertura completa de parques para automóveis em cada resultado."
                        )
                        combined = f"{parking_note}\n\n{combined}"
                    if missing_services:
                        if language == "pt":
                            combined += f"\n\n⚠️ Não foi possível confirmar resultados para: {', '.join(missing_services)}."
                        else:
                            combined += f"\n\n⚠️ I could not confirm results for: {', '.join(missing_services)}."
                    normalized_services = " ".join(
                        unicodedata.normalize("NFKD", service or "")
                        .encode("ascii", "ignore")
                        .decode("ascii")
                        .lower()
                        for service in service_types
                    )
                    health_requested = bool(re.search(r"\b(farmacia|farmacias|hospital|hospitais)\b", normalized_services))
                    time_sensitive_requested = bool(re.search(
                        r"\b(evening|tonight|late|after\s+hours|this\s+evening|now|right\s+now|availability|available|open|still\s+useful|noite|esta\s+noite|ao\s+fim\s+do\s+dia|mais\s+tarde|agora|disponibilidade|disponivel|abert[ao]s?|funciona)\b",
                        user_message,
                        re.IGNORECASE,
                    ))
                    if time_sensitive_requested:
                        if language == "pt":
                            if health_requested:
                                combined += "\n\nAviso: a Lisboa Aberta confirma localização e proximidade, mas não confirma disponibilidade em tempo real, urgência, atendimento atual ou farmácias de serviço. Confirma diretamente antes de te deslocares."
                            else:
                                combined += "\n\nAviso: a Lisboa Aberta confirma localização e proximidade, mas não confirma horário atual nem disponibilidade ao fim do dia. Confirma diretamente antes de te deslocares."
                        elif health_requested:
                            combined += "\n\nNote: Lisboa Aberta confirms location and proximity, but not real-time availability, emergency capacity, current attendance, or duty-pharmacy status. Confirm directly before going."
                        else:
                            combined += "\n\nNote: Lisboa Aberta confirms location and proximity, but not current opening hours or evening availability. Confirm directly before going."
                    combined += f"\n\n{self._build_open_data_services_source_line(language)}"
                    return combined.strip()

                if missing_services:
                    return self._build_missing_services_limitation(
                        service_types=missing_services,
                        nearby_location=nearby_location,
                        language=language,
                    )

        if any(keyword in message_lower for keyword in history_keywords) or self._is_history_culture_query(user_message):
            subject = self._extract_history_culture_subject(user_message) or user_message
            knowledge_tool = self._get_tool_by_name("search_lisbon_knowledge")
            if knowledge_tool:
                raw_knowledge = str(
                    self._invoke_tool(
                        knowledge_tool,
                        {"query": subject, "max_results": 5},
                        tool_name="search_lisbon_knowledge",
                    )
                ).strip()
                raw_knowledge = re.sub(r"\n?📅\s+\*\*Related Events:\*\*.*", "", raw_knowledge, flags=re.DOTALL).strip()
                raw_knowledge = re.sub(r"\n?🏛️\s+\*\*Related Places:\*\*.*", "", raw_knowledge, flags=re.DOTALL).strip()
                compact_knowledge = self._compact_history_result(raw_knowledge, language, subject)
                if compact_knowledge and not re.search(
                    r"\b(?:Lisbon Knowledge Search Results|Guide\s*/\s*PDF Knowledge|Guia\s+Lxcard)\b",
                    compact_knowledge,
                    flags=re.IGNORECASE,
                ):
                    timestamp = datetime.now().strftime("%H:%M")
                    heading = f"### 📚 Contexto histórico: {subject}" if language == "pt" else f"### 📚 Historical Context: {subject}"
                    source = (
                        f"📌 **Fonte:** *Guia Lisboa Card* | **Atualizado:** {timestamp}"
                        if language == "pt"
                        else f"📌 **Source:** *Lisboa Card Guide* | **Updated:** {timestamp}"
                    )
                    return f"{heading}\n\n{compact_knowledge}\n\n{source}"
            tool = self._get_tool_by_name("search_history_culture")
            if tool:
                raw_history = str(
                    self._invoke_tool(
                        tool,
                        {"query": subject, "language": language},
                        tool_name="search_history_culture",
                    )
                )
                compact_history = self._compact_history_result(raw_history, language, subject)
                if compact_history:
                    has_external_source = bool(
                        re.search(r"https?://", raw_history, flags=re.IGNORECASE)
                    )
                    timestamp = datetime.now().strftime("%H:%M")
                    if language == "pt":
                        heading = f"### 📚 Contexto histórico: {subject}"
                        source = f"📌 **Fonte:** [*Wikipedia/Web*](https://www.wikipedia.org/) | **Atualizado:** {timestamp}"
                    else:
                        heading = f"### 📚 Historical Context: {subject}"
                        source = f"📌 **Source:** [*Wikipedia/Web*](https://www.wikipedia.org/) | **Updated:** {timestamp}"
                    if has_external_source:
                        return f"{heading}\n\n{compact_history}\n\n{source}"
                    return f"{heading}\n\n{compact_history}"
                return raw_history

        if any(keyword in message_lower for keyword in event_keywords):
            return self._run_direct_event_lookup(user_message, language)

        tool = self._get_tool_by_name("search_places_attractions")
        if tool:
            args = {"query": user_message, "max_results": 5, "offset": 0, "language": language}
            category_hint = self._infer_place_category_hint(user_message)
            if category_hint:
                args["category"] = category_hint

            result = str(self._invoke_tool(tool, args, tool_name="search_places_attractions")).strip()
            result = self._localize_place_card_titles_with_llm(result, language)
            base_args = {key: value for key, value in args.items() if key not in {"max_results", "offset"}}
            self._remember_search_context(
                domain="places",
                tool_name="search_places_attractions",
                base_args=base_args,
                page_size=int(args["max_results"]),
                shown_count=self._count_ranked_results(result),
                language=language,
                source_query=user_message,
                offset=0,
            )
            source_line = self._build_places_source_line(result, language)
            return f"{result}\n\n{source_line}".strip()

        return (
            "I couldn't complete the semantic search prompt flow, but the retrieval tools are available."
            if language == "en"
            else "Não consegui concluir o fluxo semântico do prompt, mas as ferramentas de pesquisa continuam disponíveis."
        )

    @staticmethod
    def _is_planner_evidence_request(user_message: str) -> bool:
        """Return whether Researcher should return cards for Planner consumption."""
        query = _normalize_researcher_intent_text(user_message)
        has_strong_planning_intent = bool(
            re.search(
                r"\b(plan|itinerary|itinerario|route plan|roteiro|planeia|planejar|plano|agenda|"
                r"full day|half day|half-day|day trip|dia inteiro|meio dia)\b",
                query,
            )
        )
        has_soft_time_planning_hint = bool(
            re.search(r"\b(morning|manha|afternoon|evening|fim de tarde|tarde|noite)\b", query)
        )
        if (
            not has_strong_planning_intent
            and has_soft_time_planning_hint
            and ResearcherAgent._is_direct_event_lookup_query(user_message)
        ):
            return False
        has_planning_intent = has_strong_planning_intent or has_soft_time_planning_hint
        has_place_need = bool(
            re.search(
                r"\b(cultural|culture|cultura|historic|historical|sights?|sightseeing|historic[oa]s?|"
                r"stop|paragem|museum|museu|gallery|galeria|monument|monumento|monumentos|"
                r"viewpoint|miradouro|restaurant|restaurante|restaurants|restaurantes|food|comida|"
                r"comer|cuisine|cozinha|gastronom\w*|tradicional|traditional|almoco|almocar|"
                r"jantar|dinner|lunch|event|evento)\b",
                query,
            )
        )
        return has_planning_intent and has_place_need

    @staticmethod
    def _extract_planner_evidence_excluded_areas(user_message: str) -> list[str]:
        """Extract explicit area exclusions for itinerary evidence search."""
        normalized = ResearcherAgent._normalize_event_preference_text(user_message)
        if not normalized:
            return []
        non_area_re = re.compile(
            r"\b(?:caminh|walk|metro|autocar|bus|comboio|train|eletrico|tram|"
            r"chuva|rain|preco|price|orcamento|budget|tempo|time|horario|"
            r"schedule|subida|escadas|stairs|muito|grandes|longas?)\b",
            flags=re.IGNORECASE,
        )
        stop_re = re.compile(
            r"\b(?:com|with|mas|but|evitando|avoiding|para|porque|because|"
            r"quero|queria|gostava|i want|i would)\b",
            flags=re.IGNORECASE,
        )
        exclusions: list[str] = []
        for pattern in (
            r"\b(?:do not repeat|dont repeat|avoid|exclude|excluding|sem repetir|nao repetir|"
            r"evita(?:r)?|exclui(?:r)?)\s+([a-z0-9][a-z0-9 /&-]{1,100})",
            r"\b(?:sem|without)\s+([a-z0-9][a-z0-9 /&-]{1,80})",
        ):
            for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
                raw = stop_re.split(match.group(1), maxsplit=1)[0]
                for piece in re.split(r"\s*(?:,|/|\+|&|\band\b|\bor\b|\be\b|\bou\b)\s*", raw, flags=re.IGNORECASE):
                    cleaned = re.sub(r"\b(?:areas?|zonas?|neighbourhoods?|neighborhoods?|bairros?)\b", " ", piece)
                    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")
                    if cleaned and not non_area_re.search(cleaned) and cleaned not in exclusions:
                        exclusions.append(cleaned)
        return exclusions[:8]

    @staticmethod
    def _planner_evidence_query_matches_exclusion(query: str, excluded_areas: list[str]) -> bool:
        """Return whether a planned evidence query conflicts with excluded areas."""
        normalized_query = ResearcherAgent._normalize_event_preference_text(query)
        for area in excluded_areas:
            normalized_area = ResearcherAgent._normalize_event_preference_text(area)
            if not normalized_area:
                continue
            if normalized_area == "belem":
                if re.search(r"\b(?:belem|jeronimos|torre de belem|padrao dos descobrimentos|imperio|brasilia)\b", normalized_query):
                    return True
            elif re.search(rf"\b{re.escape(normalized_area)}\b", normalized_query):
                return True
        return False

    @staticmethod
    def _planner_evidence_query_is_context_only(query: str) -> bool:
        """Return whether a planner evidence query is trip context, not a POI.

        Planner evidence searches should retrieve verifiable VisitLisboa place
        cards. Temporal/base phrases such as "amanhã a partir do hotel no
        Saldanha" are useful for ordering and transport, but searching them as
        Museums & Monuments creates fake itinerary stops.
        """
        normalized = ResearcherAgent._normalize_event_preference_text(query)
        if not normalized:
            return True

        has_temporal_or_start_context = bool(
            re.search(
                r"\b(?:amanha|tomorrow|hoje|today|tonight|esta noite|"
                r"full day|dia inteiro|starting|start|comecando|comecar|partir|base)\b",
                normalized,
            )
        )
        has_accommodation_context = bool(
            re.search(
                r"\b(?:hotel|hoteis|alojamento|alojamentos|accommodation|"
                r"stay|staying|my hotel|meu hotel)\b",
                normalized,
            )
        )
        if has_temporal_or_start_context and has_accommodation_context:
            return True

        return bool(
            re.fullmatch(
                r"(?:o\s+|a\s+|the\s+)?(?:meu\s+|my\s+)?"
                r"(?:hotel|alojamento|accommodation)"
                r"(?:\s+(?:no|na|em|in|near|perto|around)\s+[a-z0-9 ]{2,60})?",
                normalized,
            )
        )

    def _extract_planner_evidence_query_plan(self, user_message: str, language: str) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """Ask the LLM for VisitLisboa queries that will feed itinerary synthesis.

        The output is still verified through tools. The LLM only chooses compact
        search terms, which avoids baking one fixed itinerary into the Researcher
        while keeping deterministic seeds available if the planner call fails.
        """
        if not user_message or not user_message.strip():
            return None

        excluded_areas = self._extract_planner_evidence_excluded_areas(user_message)
        exclusion_prompt = (
            "Excluded areas: " + ", ".join(excluded_areas) + ". Do not search for those areas or their landmarks.\n"
            if excluded_areas
            else ""
        )
        prompt = (
            "Convert the user request into VisitLisboa search calls for a Lisbon itinerary planner.\n"
            "Return ONLY valid JSON with keys cultural_queries and food_queries.\n"
            "Each item must have: query, category, max_results, specific_lookup.\n"
            "Allowed categories: Museums & Monuments, View Points, Restaurants, Tours, General.\n"
            "Use concrete Lisbon/AML place names when a named itinerary needs verifiable cards; otherwise use a broad semantic query.\n"
            "For historic/gastronomy day plans, prefer 3-4 geographically coherent historical/cultural Lisbon POIs and one restaurant query.\n"
            "For one-day Lisbon historic/gastronomy plans without a user-specified zone, include at least one central Lisbon/Baixa/Chiado/Alfama/Se stop and one Belem stop only when the user did not exclude Belem or the riverfront cluster.\n"
            f"{exclusion_prompt}"
            "When using specific_lookup=true, query must be exactly one official/common place name, not a list of places.\n"
            "Never include places outside Lisbon/AML, placeholder text, or unsupported sources.\n"
            "Set specific_lookup=true only for named POIs. Keep max_results between 1 and 5.\n\n"
            f"Language: {language}\n"
            f"User request: {user_message}"
        )
        try:
            response = self._safe_llm_invoke(
                self.llm,
                [
                    SystemMessage(content="You are a conservative Lisbon tourism query planner. Output JSON only."),
                    HumanMessage(content=prompt),
                ],
                retries=1,
            )
        except Exception:
            return None

        payload = parse_json_response(str(getattr(response, "content", response) or ""))
        if not isinstance(payload, dict):
            return None

        def _normalize_entries(raw_entries: Any, *, default_category: str) -> List[Dict[str, Any]]:
            entries = raw_entries if isinstance(raw_entries, list) else []
            output: List[Dict[str, Any]] = []
            seen_queries: set[str] = set()
            allowed_categories = {
                "Museums",
                "Monuments",
                "Museums & Monuments",
                "View Points",
                "Restaurants",
                "Tours",
                "General",
            }
            known_lisbon_pois = (
                ("Mosteiro dos Jerónimos", "mosteiro dos jeronimos"),
                ("Torre de Belém", "torre de belem"),
                ("Padrão dos Descobrimentos", "padrao dos descobrimentos"),
                ("Castelo de São Jorge", "castelo de sao jorge"),
                ("Museu Arqueológico do Carmo", "museu arqueologico do carmo"),
                ("Sé Catedral de Lisboa", "se de lisboa"),
                ("Arco da Rua Augusta", "arco da rua augusta"),
                ("Casa dos Bicos", "casa dos bicos"),
            )
            for raw_entry in entries:
                if not isinstance(raw_entry, dict):
                    continue
                query = re.sub(r"\s+", " ", str(raw_entry.get("query") or "").strip())
                if not query or query.lower() in {"null", "none", "unknown", "n/a"}:
                    continue
                normalized_ascii_query = unicodedata.normalize("NFKD", query)
                normalized_ascii_query = normalized_ascii_query.encode("ascii", "ignore").decode("ascii").lower()
                normalized_ascii_query = re.sub(r"\s+", " ", normalized_ascii_query).strip()
                if self._planner_evidence_query_is_context_only(normalized_ascii_query):
                    continue
                if self._planner_evidence_query_matches_exclusion(normalized_ascii_query, excluded_areas):
                    continue
                if default_category == "Museums & Monuments" and re.fullmatch(r"se\s+de(?:\s+lisboa)?", normalized_ascii_query):
                    query = "Sé Catedral de Lisboa"
                    normalized_ascii_query = "se catedral de lisboa"

                normalized_query = query.lower()
                if normalized_query in seen_queries:
                    continue
                seen_queries.add(normalized_query)
                category = str(raw_entry.get("category") or default_category).strip()
                if category not in allowed_categories:
                    category = default_category
                if (
                    default_category == "Museums & Monuments"
                    and re.search(r"\b(?:museu|museum)\b", normalized_ascii_query)
                ):
                    category = "Museums"
                if default_category == "Museums & Monuments" and category == "General":
                    category = default_category
                try:
                    max_results = int(raw_entry.get("max_results") or 1)
                except (TypeError, ValueError):
                    max_results = 1
                specific_lookup = bool(raw_entry.get("specific_lookup"))
                if default_category == "Museums & Monuments" and category == "Museums & Monuments" and not specific_lookup:
                    has_concrete_place_marker = bool(
                        re.search(
                            r"\b(?:museu|museum|mosteiro|monastery|torre|tower|castelo|castle|"
                            r"palacio|palace|igreja|church|se de lisboa|arco|padrao|carmo|bicos|"
                            r"teatro romano|judiaria|descobrimentos|jeronimos)\b",
                            normalized_ascii_query,
                        )
                    )
                    if not has_concrete_place_marker:
                        continue
                if specific_lookup:
                    matched_pois = [
                        display_name
                        for display_name, normalized_name in known_lisbon_pois
                        if normalized_name in normalized_ascii_query
                    ]
                    if len(matched_pois) > 1:
                        for display_name in matched_pois:
                            normalized_display = display_name.lower()
                            if normalized_display in seen_queries:
                                continue
                            seen_queries.add(normalized_display)
                            output.append(
                                {
                                    "query": display_name,
                                    "category": category,
                                    "max_results": 1,
                                    "specific_lookup": True,
                                    "language": language,
                                }
                            )
                        continue
                output.append(
                    {
                        "query": query,
                        "category": category,
                        "max_results": max(1, min(max_results, 5)),
                        "specific_lookup": specific_lookup,
                        "language": language,
                    }
                )
            return output

        cultural_queries = _normalize_entries(
            payload.get("cultural_queries"),
            default_category="Museums & Monuments",
        )[:4]
        food_queries = _normalize_entries(
            payload.get("food_queries"),
            default_category="Restaurants",
        )[:2]
        if not cultural_queries and not food_queries:
            return None
        return {"cultural_queries": cultural_queries, "food_queries": food_queries}

    def _run_planner_evidence_lookup(self, user_message: str, language: str) -> str:
        """Return complete place/event cards for downstream planner synthesis."""
        places_tool = self._get_tool_by_name("search_places_attractions")
        events_tool = self._get_tool_by_name("search_cultural_events")
        if not places_tool:
            return ""

        query = user_message.strip()
        normalized = _normalize_researcher_intent_text(query)
        result_blocks: List[str] = []
        event_blocks_added = False
        place_blocks_added = False
        planner_query_plan = self._extract_planner_evidence_query_plan(query, language)
        planned_cultural_queries = list(planner_query_plan.get("cultural_queries", [])) if planner_query_plan else []
        planned_food_queries = list(planner_query_plan.get("food_queries", [])) if planner_query_plan else []
        explicit_cultural_query_keys = {
            self._normalize_for_deterministic_routing(str(item.get("query") or ""))
            for item in planned_cultural_queries
            if isinstance(item, dict) and item.get("specific_lookup")
        }
        deterministic_cultural_queries: List[Dict[str, Any]] = []
        deterministic_food_queries: List[Dict[str, Any]] = []
        try:
            from agent.agents.planner_agent import (
                _extract_compact_plan_area_anchor,
                _extract_requested_plan_area,
                _extract_requested_plan_origin,
                _planner_local_area_profile,
                _query_has_explicit_start_end_constraint,
                _requested_anchor_labels,
                _requested_plan_total_stop_count,
                _requested_plan_type_counts,
            )

            requested_anchor_queries = _requested_anchor_labels(query)
            planner_origin_anchor = _extract_requested_plan_origin(query)
            planner_area_anchor = _extract_compact_plan_area_anchor(query) or _extract_requested_plan_area(query)
            planner_area_key, planner_area_label, _planner_area_blockers = _planner_local_area_profile(query)
            planner_has_start_end = _query_has_explicit_start_end_constraint(query)
            planner_requested_counts = _requested_plan_type_counts(query)
            planner_requested_stop_count = _requested_plan_total_stop_count(query)
        except Exception:
            requested_anchor_queries = []
            planner_origin_anchor = ""
            planner_area_anchor = ""
            planner_area_key = ""
            planner_area_label = ""
            planner_has_start_end = False
            planner_requested_counts = {}
            planner_requested_stop_count = 0
        if requested_anchor_queries:
            existing_queries = {
                _normalize_researcher_intent_text(str(item.get("query") or ""))
                for item in [*planned_cultural_queries, *planned_food_queries]
                if isinstance(item, dict)
            }
            for anchor in requested_anchor_queries[:8]:
                anchor_text = re.sub(r"\s+", " ", str(anchor or "")).strip()
                anchor_key = _normalize_researcher_intent_text(anchor_text)
                if (
                    not anchor_text
                    or anchor_key in existing_queries
                    or self._planner_evidence_query_is_context_only(anchor_text)
                ):
                    continue
                existing_queries.add(anchor_key)
                is_food_anchor = bool(
                    re.search(
                        r"\b(?:restaurante|restaurant|taberna|taverna|cafe|caf[eé]|pastelaria|fado)\b",
                        anchor_key,
                    )
                )
                if re.search(r"\b(?:museu|museum)\b", anchor_key):
                    anchor_category = "Museums"
                elif re.search(r"\b(?:miradouro|viewpoint|view\s+point)\b", anchor_key):
                    anchor_category = "View Points"
                elif re.search(r"\b(?:jardim|jardins|garden|gardens|parque|park|parks)\b", anchor_key):
                    anchor_category = "Parks & Gardens"
                elif re.search(r"\b(?:mercado|market|markets)\b", anchor_key):
                    anchor_category = "Shopping"
                elif re.search(r"\b(?:teatro|theatre|theater|cinema)\b", anchor_key):
                    anchor_category = "Culture"
                else:
                    anchor_category = "Museums & Monuments"
                entry = {
                    "query": anchor_text,
                    "category": "Restaurants" if is_food_anchor else anchor_category,
                    "max_results": 1,
                    "specific_lookup": True,
                    "language": language,
                }
                if is_food_anchor:
                    deterministic_food_queries.append(entry)
                else:
                    deterministic_cultural_queries.append(entry)
                    explicit_cultural_query_keys.add(
                        self._normalize_for_deterministic_routing(anchor_text)
                    )
            planned_cultural_queries = [*deterministic_cultural_queries, *planned_cultural_queries]
            planned_food_queries = [*deterministic_food_queries, *planned_food_queries]

        has_historic_monument_need = bool(
            re.search(r"\b(historic|historical|sights?|sightseeing|hist[oó]ric[oa]s?|monument|monumento|monumentos|patrim[oó]nio|heritage)\b", normalized)
        )
        requests_food_stop = bool(_RESEARCHER_FOOD_INTENT_RE.search(normalized))
        requests_traditional_food = bool(
            requests_food_stop
            and re.search(
                r"\b(?:gastronomia\s+tradicional|cozinha\s+tradicional|cozinha\s+portuguesa|"
                r"comida\s+portuguesa|tradicional|traditional\s+portuguese|portuguese\s+cuisine)\b",
                normalized,
            )
        )
        requests_budget_food = bool(
            requests_food_stop
            and re.search(
                r"\b(?:barat\w*|econ[oó]mic\w*|baixo\s+custo|low\s+cost|"
                r"cheap|budget|under\s+20|<\s*20|menos\s+de\s+20)\b",
                normalized,
                flags=re.IGNORECASE,
            )
        )
        requests_pastry_stop = bool(_RESEARCHER_CAFE_INTENT_RE.search(normalized))
        pastry_area = "Belém" if re.search(r"\b(?:bel[eé]m|belem)\b", normalized) else "Lisbon"
        requests_event_stop = bool(
            re.search(
                r"\b(?:event|events|evento|eventos|concert|concerto|concertos|festival|festivais|exhibition|exhibitions|exposi[cç][aã]o|exposi[cç][oõ]es|teatro|show|cultural programme|programa cultural)\b",
                normalized,
            )
        )
        explicit_place_need = bool(
            re.search(
                r"\b(?:historic|historical|hist[oó]ric[oa]s?|sights?|sightseeing|museum|museu|gallery|galeria|monument|monumento|monumentos|patrim[oó]nio|heritage|viewpoint|miradouro|attraction|atra[cç][aã]o|atra[cç][oõ]es)\b",
                normalized,
            )
        )
        if requests_event_stop and not explicit_place_need:
            planned_cultural_queries = []
        end_area = ""
        end_area_patterns = (
            r"\b(?:terminar|acabar|finish|end)\s+"
            r"(?:perto\s+d(?:e|o|a|os|as)|em|no|na|near|around|at)\s+"
            r"(?P<area>[^,.;]+?)(?:\s+no\s+segundo\s+dia|\s+on\s+day\s+\d+|[,.;]|$)",
            r"\b(?:terminando|terminar|acabar|acabando|finishing|ending|finish|end)\s+"
            r"(?:(?:o|a|no|na|on|the)\s+)?(?:primeiro|segundo|terceiro|\d+)(?:\s+dia|\s+day)?\s+"
            r"(?:perto\s+d(?:e|o|a|os|as)|em|no|na|near|around|at)\s+"
            r"(?P<area>[^,.;]+?)(?:[,.;]|$)",
        )
        for end_area_pattern in end_area_patterns:
            end_area_match = re.search(end_area_pattern, query, flags=re.IGNORECASE)
            if end_area_match:
                end_area = re.sub(r"\s+", " ", end_area_match.group("area")).strip(" .:-")
                break
        if not end_area:
            area_match = re.search(
                r"\b(?:em|no|na|around|near)\s+"
                r"(?P<area>[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ]*(?:\s+d[aeo]s?\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ]*){0,3})"
                r"(?=\s+(?:com|e|para|durante|inclui|including|with)|[,.;]|$)",
                query,
            )
            if area_match:
                candidate_area = re.sub(r"\s+", " ", area_match.group("area")).strip(" .:-")
                candidate_key = self._normalize_for_deterministic_routing(candidate_area)
                prefix_window = _normalize_researcher_intent_text(
                    query[max(0, area_match.start() - 50):area_match.start()]
                )
                is_accommodation_area = bool(
                    re.search(
                        r"\b(?:hotel|alojamento|accommodation|stay|staying|base)\b",
                        prefix_window,
                    )
                )
                is_start_area = bool(
                    re.search(
                        r"\b(?:comecar|comecando|iniciar|iniciando|start|starting|a\s+partir|desde)\b",
                        prefix_window,
                    )
                )
                if (
                    candidate_key
                    and candidate_key not in {"lisboa", "lisbon", "aml"}
                    and not is_accommodation_area
                    and not is_start_area
                ):
                    end_area = candidate_area
        if has_historic_monument_need and requests_food_stop:
            default_cultural_query = (
                "monumentos históricos no centro de Lisboa"
                if language == "pt"
                else "historic monuments in central Lisbon"
            )
            default_food_query = (
                "restaurantes de gastronomia tradicional em Lisboa"
                if language == "pt"
                else "traditional Portuguese restaurants in Lisbon"
            )
            if not planned_cultural_queries:
                planned_cultural_queries = [
                    {
                        "query": default_cultural_query,
                        "category": "Museums & Monuments",
                        "max_results": 5,
                        "specific_lookup": False,
                        "language": language,
                    }
                ]
            end_area_cultural_query_added = False
            if end_area and not any(
                self._normalize_for_deterministic_routing(end_area)
                in self._normalize_for_deterministic_routing(str(item.get("query") or ""))
                for item in planned_cultural_queries
            ):
                planned_cultural_queries.append(
                    {
                        "query": (
                            f"monumentos históricos em {end_area}"
                            if language == "pt"
                            else f"historic monuments in {end_area}"
                        ),
                        "category": "Museums & Monuments",
                        "max_results": 5,
                        "specific_lookup": False,
                        "language": language,
                    }
                )
                end_area_cultural_query_added = True
            if not planned_food_queries:
                planned_food_queries = [
                    {
                        "query": default_food_query,
                        "category": "Restaurants",
                        "max_results": 5,
                        "specific_lookup": False,
                        "language": language,
                    }
                ]
            elif not any(
                re.search(
                    r"\b(?:gastronomia tradicional|cozinha tradicional|cozinha portuguesa|"
                    r"traditional portuguese|portuguese cuisine)\b",
                    self._normalize_for_deterministic_routing(str(item.get("query") or "")),
                )
                for item in planned_food_queries
            ):
                planned_food_queries.insert(
                    0,
                    {
                        "query": default_food_query,
                        "category": "Restaurants",
                        "max_results": 5,
                        "specific_lookup": False,
                        "language": language,
                    },
                )
            if end_area and not any(
                self._normalize_for_deterministic_routing(end_area)
                in self._normalize_for_deterministic_routing(str(item.get("query") or ""))
                for item in planned_food_queries
            ):
                planned_food_queries.append(
                    {
                        "query": (
                            f"restaurantes de gastronomia tradicional em {end_area}"
                            if language == "pt"
                            else f"traditional Portuguese restaurants in {end_area}"
                        ),
                        "category": "Restaurants",
                        "max_results": 5,
                        "specific_lookup": False,
                        "language": language,
                    }
                )
            if deterministic_cultural_queries or explicit_cultural_query_keys:
                explicit_keys = {
                    *explicit_cultural_query_keys,
                    *{
                        self._normalize_for_deterministic_routing(str(item.get("query") or ""))
                        for item in deterministic_cultural_queries
                    },
                }
                explicit_queries = [
                    item
                    for item in planned_cultural_queries
                    if self._normalize_for_deterministic_routing(str(item.get("query") or "")) in explicit_keys
                ]
                other_queries = [
                    item
                    for item in planned_cultural_queries
                    if item not in explicit_queries
                ]
                planned_cultural_queries = explicit_queries + other_queries[: max(0, 6 - len(explicit_queries))]
            elif len(planned_cultural_queries) > 4 and end_area_cultural_query_added:
                end_area_key = self._normalize_for_deterministic_routing(end_area)
                end_area_queries = [
                    item
                    for item in planned_cultural_queries
                    if end_area_key in self._normalize_for_deterministic_routing(str(item.get("query") or ""))
                ][:1]
                other_queries = [
                    item
                    for item in planned_cultural_queries
                    if item not in end_area_queries
                ]
                planned_cultural_queries = end_area_queries + other_queries[: max(0, 4 - len(end_area_queries))]
            else:
                planned_cultural_queries = planned_cultural_queries[:4]
            if len(planned_food_queries) > 2:
                end_area_key = self._normalize_for_deterministic_routing(end_area)
                if end_area_key:
                    end_area_food_queries = [
                        item
                        for item in planned_food_queries
                        if end_area_key in self._normalize_for_deterministic_routing(str(item.get("query") or ""))
                    ][:1]
                    other_food_queries = [
                        item
                        for item in planned_food_queries
                        if item not in end_area_food_queries
                    ]
                    planned_food_queries = end_area_food_queries + other_food_queries[: max(0, 2 - len(end_area_food_queries))]
                else:
                    planned_food_queries = planned_food_queries[:2]
            else:
                planned_food_queries = planned_food_queries[:2]
        has_broader_itinerary_need = bool(
            re.search(
                r"\b(?:roteiro|itiner[aá]rio|plano|plan|dia|dias|day|days|full\s+day|"
                r"short\s+plan|plano\s+curto|afternoon|evening|manh[aã]|tarde|noite)\b",
                normalized,
            )
        )
        if has_broader_itinerary_need and not requests_event_stop:
            requests_museum_day = bool(
                re.search(r"\b(?:museum|museums|museu|museus)\b", normalized)
            )
            requests_viewpoint_day = bool(
                re.search(r"\b(?:viewpoint|viewpoints|view\s+point|views?|vista|vistas|miradouro|miradouros)\b", normalized)
            )
            default_general_cultural_query = (
                "museus em Lisboa"
                if requests_museum_day and language == "pt"
                else "museums in Lisbon"
                if requests_museum_day
                else "monumentos históricos no centro de Lisboa"
                if language == "pt"
                else "historic monuments in central Lisbon"
            )
            default_general_key = self._normalize_for_deterministic_routing(default_general_cultural_query)
            has_default_general_query = any(
                default_general_key
                in self._normalize_for_deterministic_routing(str(item.get("query") or ""))
                for item in planned_cultural_queries
            )
            if not planned_cultural_queries:
                planned_cultural_queries = [
                    {
                        "query": default_general_cultural_query,
                        "category": "Museums & Monuments",
                        "max_results": 5,
                        "specific_lookup": False,
                        "language": language,
                    }
                ]
            elif not has_default_general_query and (
                len(planned_cultural_queries) < 2
                or end_area
                or requests_museum_day
                or re.search(r"\b(?:pouco\s+esfor[cç]o|pouca\s+caminhada|senior|s[eé]nior|low\s+walking|reduced\s+mobility)\b", normalized)
            ):
                planned_cultural_queries.insert(
                    0,
                    {
                        "query": default_general_cultural_query,
                        "category": "Museums & Monuments",
                        "max_results": 5,
                        "specific_lookup": False,
                        "language": language,
                    },
                )
            if requests_museum_day:
                museum_default = {
                    "query": "museus em Lisboa" if language == "pt" else "museums in Lisbon",
                    "category": "Museums & Monuments",
                    "max_results": 5,
                    "specific_lookup": False,
                    "language": language,
                }
                museum_default_key = self._normalize_for_deterministic_routing(str(museum_default["query"]))
                planned_cultural_queries = [
                    item
                    for item in planned_cultural_queries
                    if museum_default_key
                    not in self._normalize_for_deterministic_routing(str(item.get("query") or ""))
                ]
                planned_cultural_queries.insert(0, museum_default)
            if requests_viewpoint_day:
                viewpoint_default = {
                    "query": "miradouros em Lisboa" if language == "pt" else "viewpoints in Lisbon",
                    "category": "View Points",
                    "max_results": 5,
                    "specific_lookup": False,
                    "language": language,
                }
                viewpoint_default_key = self._normalize_for_deterministic_routing(str(viewpoint_default["query"]))
                planned_cultural_queries = [
                    item
                    for item in planned_cultural_queries
                    if viewpoint_default_key
                    not in self._normalize_for_deterministic_routing(str(item.get("query") or ""))
                ]
                insert_at = 1 if requests_museum_day and planned_cultural_queries else 0
                planned_cultural_queries.insert(insert_at, viewpoint_default)
            if end_area and not any(
                self._normalize_for_deterministic_routing(end_area)
                in self._normalize_for_deterministic_routing(str(item.get("query") or ""))
                for item in planned_cultural_queries
            ):
                planned_cultural_queries.append(
                    {
                        "query": (
                            f"locais culturais e monumentos em {end_area}"
                            if language == "pt"
                            else f"cultural sites and monuments in {end_area}"
                        ),
                        "category": "Museums & Monuments",
                        "max_results": 5,
                        "specific_lookup": False,
                        "language": language,
                    }
                )
            if len(planned_cultural_queries) > 4:
                if deterministic_cultural_queries or explicit_cultural_query_keys:
                    explicit_keys = {
                        *explicit_cultural_query_keys,
                        *{
                            self._normalize_for_deterministic_routing(str(item.get("query") or ""))
                            for item in deterministic_cultural_queries
                        },
                    }
                    explicit_queries = [
                        item
                        for item in planned_cultural_queries
                        if self._normalize_for_deterministic_routing(str(item.get("query") or "")) in explicit_keys
                    ]
                    other_queries = [
                        item
                        for item in planned_cultural_queries
                        if item not in explicit_queries
                    ]
                    planned_cultural_queries = explicit_queries + other_queries[: max(0, 6 - len(explicit_queries))]
                else:
                    end_area_key = self._normalize_for_deterministic_routing(end_area)
                    if end_area_key:
                        end_area_queries = [
                            item
                            for item in planned_cultural_queries
                            if end_area_key in self._normalize_for_deterministic_routing(str(item.get("query") or ""))
                        ][:1]
                        other_queries = [
                            item
                            for item in planned_cultural_queries
                            if item not in end_area_queries
                        ]
                        planned_cultural_queries = end_area_queries + other_queries[: max(0, 4 - len(end_area_queries))]
                    else:
                        planned_cultural_queries = planned_cultural_queries[:4]
        if has_broader_itinerary_need and requests_food_stop:
            default_general_food_query = (
                "restaurantes de gastronomia tradicional em Lisboa"
                if language == "pt"
                else "traditional Portuguese restaurants in Lisbon"
            )
            if requests_budget_food:
                planned_food_queries.insert(
                    0,
                    {
                        "query": (
                            "restaurantes < 20 em Lisboa"
                            if language == "pt"
                            else "restaurants under 20 in Lisbon"
                        ),
                        "category": "Restaurants",
                        "max_results": 8,
                        "specific_lookup": False,
                        "language": language,
                    },
                )
            if not planned_food_queries:
                planned_food_queries = [
                    {
                        "query": default_general_food_query,
                        "category": "Restaurants",
                        "max_results": 5,
                        "specific_lookup": False,
                        "language": language,
                    }
                ]
            if end_area and not any(
                self._normalize_for_deterministic_routing(end_area)
                in self._normalize_for_deterministic_routing(str(item.get("query") or ""))
                for item in planned_food_queries
            ):
                planned_food_queries.append(
                    {
                        "query": (
                            f"restaurantes de gastronomia tradicional em {end_area}"
                            if language == "pt"
                            else f"traditional Portuguese restaurants in {end_area}"
                        ),
                        "category": "Restaurants",
                        "max_results": 5,
                        "specific_lookup": False,
                        "language": language,
                    }
                )
            if len(planned_food_queries) > 2:
                end_area_key = self._normalize_for_deterministic_routing(end_area)
                if end_area_key:
                    end_area_food_queries = [
                        item
                        for item in planned_food_queries
                        if end_area_key in self._normalize_for_deterministic_routing(str(item.get("query") or ""))
                    ][:1]
                    other_food_queries = [
                        item
                        for item in planned_food_queries
                        if item not in end_area_food_queries
                    ]
                    planned_food_queries = end_area_food_queries + other_food_queries[: max(0, 2 - len(end_area_food_queries))]
                else:
                    planned_food_queries = planned_food_queries[:2]
        if end_area:
            end_area_key = self._normalize_for_deterministic_routing(end_area)

            def area_query_priority(item: Dict[str, Any]) -> tuple[int, int]:
                query_text = self._normalize_for_deterministic_routing(str(item.get("query") or ""))
                is_area_specific = bool(end_area_key and end_area_key in query_text)
                is_generic_lisbon = bool(re.search(r"\b(?:lisboa|lisbon)\b", query_text))
                return (0 if is_area_specific else 1, 1 if is_generic_lisbon else 0)

            if end_area_key:
                if deterministic_cultural_queries or explicit_cultural_query_keys:
                    explicit_keys = {
                        *explicit_cultural_query_keys,
                        *{
                            self._normalize_for_deterministic_routing(str(item.get("query") or ""))
                            for item in deterministic_cultural_queries
                        },
                    }
                    explicit_queries = [
                        item
                        for item in planned_cultural_queries
                        if self._normalize_for_deterministic_routing(str(item.get("query") or "")) in explicit_keys
                    ]
                    other_queries = [
                        item
                        for item in planned_cultural_queries
                        if item not in explicit_queries
                    ]
                    planned_cultural_queries = explicit_queries + sorted(
                        other_queries,
                        key=area_query_priority,
                    )[: max(0, 6 - len(explicit_queries))]
                else:
                    planned_cultural_queries = sorted(planned_cultural_queries, key=area_query_priority)[:4]
                planned_food_queries = sorted(planned_food_queries, key=area_query_priority)[:2]

        compact_anchor = re.sub(
            r"\s+",
            " ",
            str(planner_origin_anchor or planner_area_anchor or "").strip(" .:-"),
        )
        compact_anchor_key = self._normalize_for_deterministic_routing(compact_anchor)
        is_compact_start_area_plan = bool(
            compact_anchor_key
            and compact_anchor_key not in {"lisboa", "lisbon", "aml", "centro de lisboa", "central lisbon"}
            and not planner_has_start_end
            and re.search(
                r"\b(?:meio\s+dia|half\s+day|[2-5]\s+horas?|[2-5]\s+hours?|"
                r"come[cç]ar|come[cç]ando|iniciar|iniciando|start|starting|"
                r"a\s+partir|desde|from)\b",
                normalized,
            )
        )
        if is_compact_start_area_plan:

            def query_uses_compact_proximity(item: Dict[str, Any]) -> bool:
                item_query = self._normalize_for_deterministic_routing(str(item.get("query") or ""))
                return bool(
                    compact_anchor_key
                    and compact_anchor_key in item_query
                    and re.search(
                        r"\b(?:perto|junto|zona|volta|near|nearby|around|close\s+to)\b",
                        item_query,
                    )
                )

            if requests_traditional_food:
                existing_food_keys = {
                    self._normalize_for_deterministic_routing(str(item.get("query") or ""))
                    for item in planned_food_queries
                    if isinstance(item, dict)
                }
                traditional_food_queries = [
                    (
                        f"restaurantes de gastronomia tradicional perto de {compact_anchor}"
                        if language == "pt"
                        else f"traditional Portuguese restaurants near {compact_anchor}"
                    ),
                    (
                        "restaurantes de gastronomia tradicional em Lisboa"
                        if language == "pt"
                        else "traditional Portuguese restaurants in Lisbon"
                    ),
                ]
                traditional_entries: List[Dict[str, Any]] = []
                for food_query in traditional_food_queries:
                    food_query_key = self._normalize_for_deterministic_routing(food_query)
                    if food_query_key and food_query_key not in existing_food_keys:
                        existing_food_keys.add(food_query_key)
                        traditional_entries.append(
                            {
                                "query": food_query,
                                "category": "Restaurants",
                                "max_results": 5,
                                "specific_lookup": False,
                                "language": language,
                            }
                        )
                if traditional_entries:
                    planned_food_queries = [*traditional_entries, *planned_food_queries]

            if not requests_event_stop and not any(
                query_uses_compact_proximity(item) for item in planned_cultural_queries
            ):
                planned_cultural_queries.insert(
                    0,
                    {
                        "query": (
                            f"locais culturais e monumentos perto de {compact_anchor}"
                            if language == "pt"
                            else f"cultural sites and monuments near {compact_anchor}"
                        ),
                        "category": "Museums & Monuments",
                        "max_results": 5,
                        "specific_lookup": False,
                        "language": language,
                    },
                )
            if not requests_event_stop and planner_area_key and planner_area_label:
                area_parts = [
                    re.sub(r"\s+", " ", part).strip(" .:-")
                    for part in re.split(r"\s*/\s*", planner_area_label)
                ]
                area_parts = [
                    part for part in area_parts
                    if part and self._normalize_for_deterministic_routing(part) != compact_anchor_key
                ]
                area_reference = area_parts[0] if area_parts else ""
                area_reference_key = self._normalize_for_deterministic_routing(area_reference)
                existing_cultural_keys = {
                    self._normalize_for_deterministic_routing(str(item.get("query") or ""))
                    for item in planned_cultural_queries
                    if isinstance(item, dict)
                }
                if area_reference and not any(
                    area_reference_key in query_key
                    and re.search(r"\b(?:perto|junto|zona|volta|near|nearby|around|close\s+to)\b", query_key)
                    for query_key in existing_cultural_keys
                ):
                    planned_cultural_queries.insert(
                        1,
                        {
                            "query": (
                                f"locais culturais e monumentos perto de {area_reference}"
                                if language == "pt"
                                else f"cultural sites and monuments near {area_reference}"
                            ),
                            "category": "Museums & Monuments",
                            "max_results": 5,
                            "specific_lookup": False,
                            "language": language,
                        },
                    )
            if not requests_event_stop:
                compact_experience_queries: List[Dict[str, Any]] = []
                explicit_viewpoint_request = bool(
                    re.search(r"\b(?:miradouro|miradouros|viewpoint|viewpoints|vista|views)\b", normalized)
                )
                generic_compact_route_request = bool(
                    has_broader_itinerary_need
                    and not re.search(
                        r"\b(?:museu|museus|museum|museums|monumento|monumentos|monument|monuments|"
                        r"interior|indoor|chuva|rain|evento|event)\b",
                        normalized,
                    )
                )
                if explicit_viewpoint_request or generic_compact_route_request:
                    compact_experience_queries.append(
                        {
                            "query": (
                                f"miradouros perto de {compact_anchor}"
                                if language == "pt"
                                else f"viewpoints near {compact_anchor}"
                            ),
                            "category": "View Points",
                            "max_results": 4,
                            "specific_lookup": False,
                            "language": language,
                        }
                    )
                if re.search(r"\b(?:parque|parques|jardim|jardins|park|parks|garden|gardens)\b", normalized):
                    compact_experience_queries.append(
                        {
                            "query": (
                                f"parques e jardins perto de {compact_anchor}"
                                if language == "pt"
                                else f"parks and gardens near {compact_anchor}"
                            ),
                            "category": "Parks & Gardens",
                            "max_results": 4,
                            "specific_lookup": False,
                            "language": language,
                        }
                    )
                existing_cultural_keys = {
                    self._normalize_for_deterministic_routing(str(item.get("query") or ""))
                    for item in planned_cultural_queries
                    if isinstance(item, dict)
                }
                for experience_query in reversed(compact_experience_queries):
                    experience_key = self._normalize_for_deterministic_routing(str(experience_query.get("query") or ""))
                    if experience_key and experience_key not in existing_cultural_keys:
                        planned_cultural_queries.insert(0, experience_query)
                        existing_cultural_keys.add(experience_key)
            if requests_food_stop and not any(
                query_uses_compact_proximity(item) for item in planned_food_queries
            ):
                compact_food_query = (
                    f"restaurantes de gastronomia tradicional perto de {compact_anchor}"
                    if requests_traditional_food and language == "pt"
                    else f"traditional Portuguese restaurants near {compact_anchor}"
                    if requests_traditional_food
                    else f"restaurantes perto de {compact_anchor}"
                    if language == "pt"
                    else f"restaurants near {compact_anchor}"
                )
                planned_food_queries = [
                    {
                        "query": compact_food_query,
                        "category": "Restaurants",
                        "max_results": 5,
                        "specific_lookup": False,
                        "language": language,
                    },
                    *planned_food_queries,
                ]
                area_food_query = ""
                if planner_area_key and planner_area_label:
                    area_parts = [
                        re.sub(r"\s+", " ", part).strip(" .:-")
                        for part in re.split(r"\s*/\s*", planner_area_label)
                    ]
                    area_parts = [
                        part for part in area_parts
                        if part and self._normalize_for_deterministic_routing(part) != compact_anchor_key
                    ]
                    area_reference = area_parts[0] if area_parts else ""
                    if area_reference:
                        area_food_query = (
                            f"restaurantes de gastronomia tradicional perto de {area_reference}"
                            if requests_traditional_food and language == "pt"
                            else f"traditional Portuguese restaurants near {area_reference}"
                            if requests_traditional_food
                            else f"restaurantes perto de {area_reference}"
                            if language == "pt"
                            else f"restaurants near {area_reference}"
                        )
                if area_food_query:
                    planned_food_queries.insert(
                        1,
                        {
                            "query": area_food_query,
                            "category": "Restaurants",
                            "max_results": 5,
                            "specific_lookup": False,
                            "language": language,
                        },
                    )
            elif requests_food_stop:
                canonical_food_query = (
                    f"restaurantes de gastronomia tradicional perto de {compact_anchor}"
                    if requests_traditional_food and language == "pt"
                    else f"traditional Portuguese restaurants near {compact_anchor}"
                    if requests_traditional_food
                    else f"restaurantes perto de {compact_anchor}"
                    if language == "pt"
                    else f"restaurants near {compact_anchor}"
                )
                area_food_query = ""
                if planner_area_key and planner_area_label:
                    area_parts = [
                        re.sub(r"\s+", " ", part).strip(" .:-")
                        for part in re.split(r"\s*/\s*", planner_area_label)
                    ]
                    area_parts = [
                        part for part in area_parts
                        if part and self._normalize_for_deterministic_routing(part) != compact_anchor_key
                    ]
                    area_reference = area_parts[0] if area_parts else ""
                    if area_reference:
                        area_food_query = (
                            f"restaurantes de gastronomia tradicional perto de {area_reference}"
                            if requests_traditional_food and language == "pt"
                            else f"traditional Portuguese restaurants near {area_reference}"
                            if requests_traditional_food
                            else f"restaurantes perto de {area_reference}"
                            if language == "pt"
                            else f"restaurants near {area_reference}"
                        )
                canonical_entries = [
                    {
                        "query": canonical_food_query,
                        "category": "Restaurants",
                        "max_results": 5,
                        "specific_lookup": False,
                        "language": language,
                    },
                ]
                if area_food_query:
                    canonical_entries.append(
                        {
                            "query": area_food_query,
                            "category": "Restaurants",
                            "max_results": 5,
                            "specific_lookup": False,
                            "language": language,
                        }
                    )
                planned_food_queries = [
                    *canonical_entries,
                    *[
                        item
                        for item in planned_food_queries
                        if not query_uses_compact_proximity(item)
                    ],
                ]
            if requests_budget_food:
                budget_areas = [compact_anchor]
                if planner_area_key and planner_area_label:
                    budget_areas.extend(
                        re.sub(r"\s+", " ", part).strip(" .:-")
                        for part in re.split(r"\s*/\s*", planner_area_label)
                        if part.strip()
                    )
                budget_entries: List[Dict[str, Any]] = []
                seen_budget_areas: set[str] = set()
                for area in budget_areas:
                    area_key = self._normalize_for_deterministic_routing(area)
                    if not area_key or area_key in seen_budget_areas:
                        continue
                    seen_budget_areas.add(area_key)
                    budget_entries.append(
                        {
                            "query": (
                                f"restaurantes < 20 perto de {area}"
                                if language == "pt"
                                else f"restaurants under 20 near {area}"
                            ),
                            "category": "Restaurants",
                            "max_results": 8,
                            "specific_lookup": False,
                            "language": language,
                        }
                    )
                planned_food_queries = [*budget_entries, *planned_food_queries]
            planned_cultural_queries = planned_cultural_queries[:5]
            planned_food_queries = planned_food_queries[:2]

        requested_cultural_results = max(
            5,
            min(
                8,
                max(
                    int(planner_requested_counts.get("museum", 0) or 0)
                    + int(planner_requested_counts.get("monument", 0) or 0),
                    int(planner_requested_counts.get("total", 0) or 0),
                    int(planner_requested_stop_count or 0),
                ),
            ),
        )
        requested_food_results = max(5, min(8, int(planner_requested_counts.get("food", 0) or 0)))
        museum_count_requested = int(planner_requested_counts.get("museum", 0) or 0)
        has_broad_museum_query = any(
            (
                re.search(r"\b(?:museus|museums)\b", self._normalize_for_deterministic_routing(str(item.get("query") or "")))
                or (
                    str(item.get("category") or "") == "Museums & Monuments"
                    and not bool(item.get("specific_lookup"))
                    and re.search(
                        r"\b(?:museu|museum)\b",
                        self._normalize_for_deterministic_routing(str(item.get("query") or "")),
                    )
                )
            )
            for item in planned_cultural_queries
        )
        if museum_count_requested and not has_broad_museum_query:
            planned_cultural_queries.insert(
                0,
                {
                    "query": "museus em Lisboa" if language == "pt" else "museums in Lisbon",
                    "category": "Museums & Monuments",
                    "max_results": requested_cultural_results,
                    "specific_lookup": False,
                    "language": language,
                },
            )
        if planner_requested_counts.get("viewpoint") and not any(
            re.search(r"\b(?:miradouro|viewpoint|view\s+point|views?|vista|vistas)\b", self._normalize_for_deterministic_routing(str(item.get("query") or "")))
            for item in planned_cultural_queries
        ):
            planned_cultural_queries.insert(
                0,
                {
                    "query": "miradouros em Lisboa" if language == "pt" else "viewpoints in Lisbon",
                    "category": "View Points",
                    "max_results": requested_cultural_results,
                    "specific_lookup": False,
                    "language": language,
                },
            )
        for item in planned_cultural_queries:
            if isinstance(item, dict):
                item["max_results"] = max(int(item.get("max_results") or 5), requested_cultural_results)
        for item in planned_food_queries:
            if isinstance(item, dict):
                item["max_results"] = max(int(item.get("max_results") or 5), requested_food_results)

        def dedupe_planner_queries(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Deduplicate equivalent planner search calls while preserving intent."""
            deduped: List[Dict[str, Any]] = []
            by_key: Dict[tuple[str, str, bool], Dict[str, Any]] = {}
            for raw_item in items:
                if not isinstance(raw_item, dict):
                    continue
                query_key = self._normalize_for_deterministic_routing(str(raw_item.get("query") or ""))
                category_key = self._normalize_for_deterministic_routing(str(raw_item.get("category") or ""))
                specific_lookup = bool(raw_item.get("specific_lookup"))
                if not query_key:
                    continue
                key = (query_key, category_key, specific_lookup)
                existing = by_key.get(key)
                if existing is None:
                    item = dict(raw_item)
                    by_key[key] = item
                    deduped.append(item)
                    continue
                existing["max_results"] = max(
                    int(existing.get("max_results") or 5),
                    int(raw_item.get("max_results") or 5),
                )
                existing["language"] = existing.get("language") or raw_item.get("language") or language
            return deduped

        planned_cultural_queries = dedupe_planner_queries(planned_cultural_queries)
        planned_food_queries = dedupe_planner_queries(planned_food_queries)
        if is_compact_start_area_plan and compact_anchor_key:

            def compact_query_priority(item: Dict[str, Any]) -> tuple[int, int]:
                """Prioritize local/proximity evidence before broad Lisbon recall."""
                query_text = self._normalize_for_deterministic_routing(str(item.get("query") or ""))
                mentions_anchor = bool(compact_anchor_key and compact_anchor_key in query_text)
                asks_proximity = bool(
                    re.search(
                        r"\b(?:perto|junto|zona|volta|near|nearby|around|close\s+to)\b",
                        query_text,
                    )
                )
                is_generic_lisbon = bool(
                    re.search(r"\b(?:lisboa|lisbon)\b", query_text)
                    and not mentions_anchor
                )
                if mentions_anchor and asks_proximity:
                    return (0, 0)
                if mentions_anchor:
                    return (1, 0)
                if is_generic_lisbon:
                    return (3, 0)
                return (2, 0)

            planned_cultural_queries = sorted(planned_cultural_queries, key=compact_query_priority)
            planned_food_queries = sorted(planned_food_queries, key=compact_query_priority)

        if has_historic_monument_need and planned_cultural_queries:
            planned_query_text = " ".join(str(item.get("query") or "") for item in planned_cultural_queries)
            planned_query_ascii = unicodedata.normalize("NFKD", planned_query_text)
            planned_query_ascii = planned_query_ascii.encode("ascii", "ignore").decode("ascii").lower()
            has_broad_historic_query = bool(
                re.search(
                    r"\b(?:historic|historical|historico|historicos|hist[oó]rico|monument|monumento)\b",
                    planned_query_ascii,
                )
            )
            if len(planned_cultural_queries) < 3 and not has_broad_historic_query:
                planned_cultural_queries.append(
                    {
                        "query": (
                            "monumentos históricos em Lisboa"
                            if language == "pt"
                            else "historic monuments in Lisbon"
                        ),
                        "category": "Museums & Monuments",
                        "max_results": 5,
                        "specific_lookup": False,
                        "language": language,
                    }
                )
        has_cultural_place_signal = explicit_place_need or bool(
            not requests_event_stop
            and re.search(r"\b(cultural|culture|cultura|stop|paragem)\b", normalized)
        )
        if requests_event_stop and events_tool:
            event_area = end_area.strip()
            requests_free_event = bool(
                re.search(r"\b(?:gratuito|gratuitos|gratuita|gratuitas|gratis|grátis|free)\b", normalized)
            )
            event_query = (
                f"eventos gratuitos {event_area}" if language == "pt" and requests_free_event and event_area
                else "eventos gratuitos em Lisboa" if language == "pt" and requests_free_event
                else f"eventos culturais em {event_area}" if language == "pt" and event_area
                else "eventos culturais em Lisboa" if language == "pt"
                else f"free events in {event_area}" if requests_free_event and event_area
                else "free events in Lisbon" if requests_free_event
                else f"cultural events in {event_area}" if event_area
                else "cultural events in Lisbon"
            )
            if re.search(r"\b(?:concert|concerto|concertos|music|m[uú]sica)\b", normalized):
                if language == "pt":
                    event_query = (
                        f"concertos gratuitos em {event_area}" if requests_free_event and event_area
                        else "concertos gratuitos em Lisboa" if requests_free_event
                        else f"concertos e música em {event_area}" if event_area
                        else "concertos e música em Lisboa"
                    )
                else:
                    event_query = (
                        f"free concerts in {event_area}" if requests_free_event and event_area
                        else "free concerts in Lisbon" if requests_free_event
                        else f"concerts and music in {event_area}" if event_area
                        else "concerts and music in Lisbon"
                    )
            elif re.search(r"\b(?:teatro|theatre|theater|dan[cç]a|dance)\b", normalized):
                if language == "pt":
                    event_query = (
                        f"teatro e dança gratuitos em {event_area}" if requests_free_event and event_area
                        else "teatro e dança gratuitos em Lisboa" if requests_free_event
                        else f"teatro e dança em {event_area}" if event_area
                        else "teatro e dança em Lisboa"
                    )
                else:
                    event_query = (
                        f"free theatre and dance in {event_area}" if requests_free_event and event_area
                        else "free theatre and dance in Lisbon" if requests_free_event
                        else f"theatre and dance in {event_area}" if event_area
                        else "theatre and dance in Lisbon"
                    )
            elif re.search(r"\b(?:exhibition|exhibitions|exposi[cç][aã]o|exposi[cç][oõ]es|arte|art)\b", normalized):
                if language == "pt":
                    event_query = (
                        f"exposições gratuitas em {event_area}" if requests_free_event and event_area
                        else "exposições gratuitas em Lisboa" if requests_free_event
                        else f"exposições e arte em {event_area}" if event_area
                        else "exposições e arte em Lisboa"
                    )
                else:
                    event_query = (
                        f"free exhibitions in {event_area}" if requests_free_event and event_area
                        else "free exhibitions in Lisbon" if requests_free_event
                        else f"exhibitions and art in {event_area}" if event_area
                        else "exhibitions and art in Lisbon"
                    )

            event_args: Dict[str, Any] = {
                "query": event_query,
                "max_results": 4,
                "language": language,
            }
            if re.search(r"\b(?:s[aá]bado|sabado|domingo|fim de semana|weekend)\b", normalized):
                event_args["date_filter"] = "this weekend"
            elif re.search(r"\b(?:esta semana|this week)\b", normalized):
                event_args["date_filter"] = "this week"
            elif re.search(r"\b(?:amanh[aã]|tomorrow)\b", normalized):
                event_args["date_filter"] = "tomorrow"
            elif re.search(r"\b(?:hoje|today)\b", normalized):
                event_args["date_filter"] = "today"

            event_result = str(
                self._invoke_tool(
                    events_tool,
                    event_args,
                    tool_name="search_cultural_events",
                )
            ).strip()
            if event_result and not event_result.startswith(("❌", "Error:")):
                result_blocks.append(event_result)
                event_blocks_added = True
                self._remember_search_context(
                    domain="events",
                    tool_name="search_cultural_events",
                    base_args={key: value for key, value in event_args.items() if key not in {"max_results", "offset"}},
                    page_size=int(event_args["max_results"]),
                    shown_count=self._count_ranked_results(event_result),
                    language=language,
                    source_query=query,
                    offset=0,
                )

        if (has_cultural_place_signal and not requests_event_stop) or planned_cultural_queries:
            cultural_queries: List[Dict[str, Any]]
            if planned_cultural_queries:
                cultural_queries = planned_cultural_queries
            elif has_historic_monument_need:
                cultural_queries = [
                    {
                        "query": (
                            "monumentos históricos em Lisboa"
                            if language == "pt"
                            else "historic monuments in Lisbon"
                        ),
                        "category": "Museums & Monuments",
                        "max_results": 5,
                        "specific_lookup": False,
                        "language": language,
                    },
                ]
            else:
                cultural_queries = [
                    {
                        "query": f"cultural stop museums monuments viewpoints {query}",
                        "category": "Museums & Monuments",
                        "max_results": 5,
                        "language": language,
                    }
                ]
            for cultural_args in cultural_queries:
                result = str(self._invoke_tool(places_tool, cultural_args, tool_name="search_places_attractions")).strip()
                result = self._localize_place_card_titles_with_llm(result, language)
                if result:
                    result_blocks.append(result)
                    place_blocks_added = True
                    self._remember_search_context(
                        domain="places",
                        tool_name="search_places_attractions",
                        base_args={key: value for key, value in cultural_args.items() if key not in {"max_results", "offset"}},
                        page_size=int(cultural_args["max_results"]),
                        shown_count=self._count_ranked_results(result),
                        language=language,
                        source_query=query,
                        offset=0,
                    )

        if _RESEARCHER_FOOD_INTENT_RE.search(normalized):
            default_food_query = {
                "query": "restaurantes de gastronomia tradicional em Lisboa",
                "category": "Restaurants",
                "max_results": 5,
                "language": language,
                "specific_lookup": False,
            } if language == "pt" else {
                "query": "traditional Portuguese restaurants in Lisbon",
                "category": "Restaurants",
                "max_results": 5,
                "language": language,
                "specific_lookup": False,
            }
            food_queries = planned_food_queries or [
                default_food_query
            ]
            if requests_pastry_stop:
                pastry_query = {
                    "query": f"coffee shop {pastry_area}" if language == "en" else f"pastelaria {pastry_area}",
                    "max_results": 3,
                    "language": language,
                    "specific_lookup": False,
                }
                food_queries = [
                    pastry_query,
                    *[
                        existing_query
                        for existing_query in food_queries
                        if str(existing_query.get("query") or "").strip().lower()
                        != str(pastry_query["query"]).strip().lower()
                    ],
                ]
            food_blocks_added = False
            executed_food_query_keys: set[str] = set()

            def run_food_query(food_args: Dict[str, Any]) -> None:
                nonlocal food_blocks_added, place_blocks_added
                query_key = self._normalize_for_deterministic_routing(str(food_args.get("query") or ""))
                if not query_key or query_key in executed_food_query_keys:
                    return
                executed_food_query_keys.add(query_key)
                result = str(self._invoke_tool(places_tool, food_args, tool_name="search_places_attractions")).strip()
                result = self._localize_place_card_titles_with_llm(result, language)
                if result and self._count_ranked_results(result) > 0:
                    result_blocks.append(result)
                    place_blocks_added = True
                    food_blocks_added = True

            for food_args in food_queries[:2]:
                run_food_query(food_args)
            if not food_blocks_added:
                run_food_query(default_food_query)

        if not result_blocks:
            generic_args = {"query": query, "max_results": 5, "language": language}
            result = str(self._invoke_tool(places_tool, generic_args, tool_name="search_places_attractions")).strip()
            result = self._localize_place_card_titles_with_llm(result, language)
            if result:
                result_blocks.append(result)
                place_blocks_added = True

        if not result_blocks:
            return ""

        if event_blocks_added and place_blocks_added:
            heading = "### 🎭 **Locais e eventos**" if language == "pt" else "### 🎭 **Places and events**"
        elif event_blocks_added:
            heading = "### 🎭 **Eventos culturais**" if language == "pt" else "### 🎭 **Cultural events**"
        else:
            heading = "### 🏛️ **Locais e atrações**" if language == "pt" else "### 🏛️ **Places and attractions**"
        note = (
            "✅ **Resposta direta:** Aqui tens os resultados mais relevantes que encontrei para o que pediste."
            if language == "pt"
            else "✅ **Direct answer:** Here are the most relevant results I found for what you asked."
        )
        combined = "\n\n".join(result_blocks)
        if event_blocks_added and place_blocks_added:
            source_line = (
                "📌 **Fonte:** [*VisitLisboa Eventos*](https://www.visitlisboa.com/pt-pt/eventos) | [*VisitLisboa Locais*](https://www.visitlisboa.com/pt-pt/locais)"
                if language == "pt"
                else "📌 **Source:** [*VisitLisboa Events*](https://www.visitlisboa.com/en/events) | [*VisitLisboa Places*](https://www.visitlisboa.com/en/places)"
            )
        elif event_blocks_added:
            source_line = self._build_events_source_line(language)
        else:
            source_line = self._build_places_source_line(combined, language)
        return "\n\n".join([heading, note, combined, source_line]).strip()

    @staticmethod
    def _extract_service_types(user_message: str) -> List[str]:
        """Extracts one or more practical service types from a service query."""
        normalized_query = unicodedata.normalize("NFKD", user_message or "")
        normalized_query = normalized_query.encode("ascii", "ignore").decode("ascii").lower()
        parking_context = bool(re.search(
            r"\b(?:parking|car\s+parks?|park\s+my\s+car|estacionamento|estacionar|parques?\s+de\s+estacionamento)\b",
            normalized_query,
        ))
        accommodation_with_parking_filter = bool(
            parking_context
            and re.search(
                r"\b(?:hotel|hotels|hoteis|hostels?|guest\s+houses?|accommodation|lodging|stay|"
                r"alojamentos?|pousadas?|apartamentos?)\b",
                normalized_query,
            )
        )
        if accommodation_with_parking_filter:
            return []
        car_parking_context = bool(re.search(
            r"\b(?:car|cars|carro|carros|automovel|automoveis|viatura|viaturas|automobile|vehicle)\b",
            normalized_query,
        ))
        bike_parking_context = bool(re.search(
            r"\b(?:bike|bikes|bicycle|bicycles|bicicleta|bicicletas|velocipede|velocipedes|velocipedo|velocipedos)\b",
            normalized_query,
        ))
        service_catalog = [
            (("pharmacy", "pharmacies", "farm", "pharmac"), "farm\u00e1cias"),
            (
                (
                    "hospital", "hospitals", "hospit", "clinic", "clinica", "clinicas",
                    "cl\u00ednica", "cl\u00ednicas", "health", "saude", "sa\u00fade",
                    "centro de saude", "centros de saude", "centro de sa\u00fade",
                    "centros de sa\u00fade", "servicos de saude", "servi\u00e7os de sa\u00fade",
                ),
                "hospitais",
            ),
            (("school", "schools", "escola", "escolas", "sch"), "escolas"),
            (("library", "libraries", "bibliot", "librar"), "bibliotecas"),
            (
                (
                    "pilhao", "pilhoes", "pilha", "pilhas", "bateria", "baterias",
                    "battery bin", "battery bins", "battery recycling", "battery point",
                    "battery points",
                ),
                "pilhões",
            ),
            (
                (
                    "papeleira", "papeleiras", "caixote do lixo", "caixotes do lixo",
                    "litter bin", "litter bins", "waste bin", "waste bins",
                ),
                "papeleiras",
            ),
            (
                (
                    "parque canino", "parques caninos", "dog park", "dog parks",
                    "canino", "caes", "caes soltos", "cães", "cães soltos",
                ),
                "parques caninos",
            ),
            (
                (
                    "ponto de encontro", "pontos de encontro", "ponto de encontro de emergencia",
                    "pontos de encontro de emergencia", "proteccao civil", "protecao civil",
                    "proteção civil", "emergency meeting point", "emergency meeting points",
                ),
                "pontos de encontro de emergência",
            ),
            (("ecoponto", "ecopontos", "reciclag", "recycling", "recycle", "residuo", "residuos"), "ecopontos"),
            (("parque infantil", "parques infantis", "playground", "playgrounds", "infantil"), "parques infantis"),
            (
                (
                    "bebedouro", "bebedouros", "ponto de agua", "pontos de agua",
                    "ponto de \u00e1gua", "pontos de \u00e1gua", "drinking fountain",
                    "drinking fountains", "water point", "water points", "fontanario",
                    "fontanarios", "fontan\u00e1rio", "fontan\u00e1rios", "chafariz", "chafarizes",
                ),
                "Arquitetura da \u00c1gua",
            ),
            (("wifi", "wi-fi", "wi fi", "public wifi", "public wi-fi", "internet"), "wifi"),
            (("park", "parks", "garden", "gardens", "jardim", "jardins"), "jardins"),
            (("psp", "policia de seguranca publica", "pol\u00edcia de seguran\u00e7a p\u00fablica", "esquadra"), "Pol\u00edcia de Seguran\u00e7a P\u00fablica"),
            (("police", "polic"), "pol\u00edcia"),
            (
                (
                    "bike parking", "bicycle parking", "estacionamento de bicicletas",
                    "estacionamento de velocipedes", "estacionamento de velocípedes",
                    "bicicleta", "bicicletas", "velocipede", "velocipedes",
                ),
                "Estacionamento de velocípedes",
            ),
            (("parking", "estacion", "car park", "park my car", "parque de estacionamento"), "parking"),
            (("market", "markets", "mercado", "mercados", "feira", "feiras"), "mercados"),
            (("firefighter", "firefighters", "bombeiro", "bombeiros"), "bombeiros"),
            (
                (
                    "wc", "wcs", "restroom", "restrooms", "toilet", "toilets",
                    "public restroom", "public restrooms", "public toilet", "public toilets",
                    "casa de banho", "casas de banho", "sanitario", "sanitarios",
                    "instalacoes sanitarias", "instalacao sanitaria",
                    "instala\u00e7\u00f5es sanit\u00e1rias", "instala\u00e7\u00e3o sanit\u00e1ria",
                ),
                "sanit\u00e1rios",
            ),
            (("embassy", "embassies", "embaixada", "embaixadas"), "embaixadas"),
            (
                (
                    "citizen shop", "citizen shops", "loja do cidadao", "loja do cidad\u00e3o",
                    "posto de correios", "correios", "espaco cidadao", "espa\u00e7o cidad\u00e3o",
                    "balcao municipal", "balc\u00e3o municipal",
                ),
                "Loja do Cidad\u00e3o",
            ),
        ]

        extracted: List[str] = []
        seen_identities: set[str] = set()
        for markers, normalized_service in service_catalog:
            if normalized_service == "jardins" and parking_context:
                continue
            if any(marker in normalized_query for marker in markers):
                if normalized_service == "parking" and bike_parking_context:
                    continue
                if normalized_service == "parking" and car_parking_context:
                    normalized_service = "car parking"
                identity = ResearcherAgent._service_type_identity(normalized_service)
                if identity in seen_identities:
                    continue
                seen_identities.add(identity)
                extracted.append(normalized_service)
        return extracted

    @staticmethod
    def _filter_location_anchored_service_types(
        service_types: List[str],
        nearby_location: Optional[str],
    ) -> List[str]:
        """Remove service types that are only implied by the origin landmark name."""
        if len(service_types) <= 1 or not nearby_location:
            return service_types

        normalized_location = unicodedata.normalize("NFKD", nearby_location or "")
        normalized_location = normalized_location.encode("ascii", "ignore").decode("ascii").lower()
        location_markers = {
            "bibliotecas": ("biblioteca", "library"),
            "jardins": ("jardim", "jardins", "parque", "garden", "park"),
            "parques infantis": ("parque infantil", "parques infantis", "playground", "infantil"),
            "Arquitetura da Água": ("bebedouro", "fontanario", "chafariz", "water"),
            "hospitais": ("hospital", "clinica", "clinic"),
            "farmácias": ("farmacia", "pharmacy"),
            "escolas": ("escola", "school"),
            "mercados": ("mercado", "market"),
            "wifi": ("wifi", "wi-fi", "internet"),
            "bombeiros": ("bombeiro", "firefighter"),
            "polícia": ("policia", "police"),
            "Polícia de Segurança Pública": ("psp", "esquadra", "policia"),
            "pilhões": ("pilhao", "pilhoes", "pilha", "bateria", "battery"),
            "papeleiras": ("papeleira", "papeleiras", "caixote", "litter", "waste"),
            "parques caninos": ("parque canino", "parques caninos", "dog park", "canino"),
            "pontos de encontro de emergência": ("ponto de encontro", "emergencia", "protecao civil", "emergency"),
        }

        filtered = [
            service_type for service_type in service_types
            if not any(
                marker in normalized_location
                for marker in location_markers.get(service_type, ())
            )
        ]
        return filtered or service_types

    @staticmethod
    def _service_category_for_type(service_type: str) -> Optional[str]:
        """Maps a service type to the most likely Lisboa Aberta taxonomy category."""
        normalized = unicodedata.normalize("NFKD", service_type or "")
        normalized = normalized.encode("ascii", "ignore").decode("ascii").lower()
        if normalized in {"farmacias", "hospitais"}:
            return "sa\u00fade"
        if normalized == "escolas":
            return "educa\u00e7\u00e3o"
        if normalized == "bibliotecas":
            return "cultura"
        if normalized in {
            "jardins",
            "parques infantis",
            "ecopontos",
            "arquitetura da agua",
            "pilhoes",
            "papeleiras",
            "parques caninos",
        }:
            return "ambiente"
        if normalized in {"policia", "policia de seguranca publica", "bombeiros", "pontos de encontro de emergencia"}:
            return "seguran\u00e7a"
        if "parking" in normalized or "velocipedes" in normalized or "bicicletas" in normalized:
            return "transportes"
        if normalized in {"estacionamento", "sanitarios", "loja do cidadao", "wifi"}:
            return "servi\u00e7os"
        if normalized == "mercados":
            return "com\u00e9rcio"
        return None

    @staticmethod
    def _build_nearby_services_direct_summary(service_blocks: List[str], language: str) -> str:
        """Build a direct nearest-service summary from Lisboa Aberta blocks.

        Multi-service questions such as "nearest hospital and pharmacy" should
        answer the requested nearest items first, then show the detailed cards.
        The parser intentionally uses only rendered block structure so every
        service type produced by the shared Lisboa Aberta tool inherits the same
        behaviour.
        """
        if not service_blocks:
            return ""

        is_pt = language == "pt"
        heading = "### 📍 **Serviços mais próximos**" if is_pt else "### 📍 **Nearest services**"
        summary_lines: List[str] = [heading, ""]
        seen_labels: set[str] = set()

        for block in service_blocks:
            normalized_block = unicodedata.normalize("NFKD", block or "")
            normalized_block = normalized_block.encode("ascii", "ignore").decode("ascii").lower()
            if "farm" in normalized_block or "pharmac" in normalized_block:
                icon = "💊"
                label = "Farmácia" if is_pt else "Pharmacy"
            elif "hospit" in normalized_block or "cuidados" in normalized_block:
                icon = "🏥"
                label = "Hospital"
            elif any(marker in normalized_block for marker in ("policia", "police", "psp", "esquadra")):
                icon = "👮"
                label = "Esquadra/Polícia" if is_pt else "Police station"
            else:
                icon = "📍"
                label = "Serviço" if is_pt else "Service"
            if label in seen_labels:
                continue

            nearest_match = re.search(
                r"(?m)^-\s*✅\s*\*\*(?:Mais perto|Closest):\*\*\s*(?P<name>.+?)(?:\s*\((?P<distance>[0-9]+(?:\.[0-9]+)?\s*km)[^)]*\))?\s*$",
                block or "",
                flags=re.IGNORECASE,
            )
            item_match = nearest_match or re.search(r"(?m)^-\s*(?:[^\w*]+\s*)?\*\*(?P<name>[^*]+)\*\*", block or "")
            if not item_match:
                continue
            name = item_match.group("name").strip()
            tail = (block or "")[item_match.end(): item_match.end() + 500]
            distance_match = re.search(r"\*\*(?:Distância|Distance):\*\*\s*(?P<distance>[0-9]+(?:\.[0-9]+)?\s*km)", tail, re.IGNORECASE)
            raw_distance = ""
            if nearest_match and nearest_match.groupdict().get("distance"):
                raw_distance = nearest_match.group("distance")
            elif distance_match:
                raw_distance = distance_match.group("distance")
            distance = f" ({raw_distance})" if raw_distance else ""
            walking_note = ""
            try:
                distance_km = float(str(raw_distance).split()[0].replace(",", ".")) if raw_distance else None
            except (TypeError, ValueError):
                distance_km = None
            if distance_km is not None:
                walking_minutes = max(1, round(distance_km * 12))
                walking_note = f" — cerca de {walking_minutes} min a pé" if is_pt else f" — about {walking_minutes} min walking"
            summary_lines.append(f"- {icon} **{label}:** {name}{distance}{walking_note}")
            seen_labels.add(label)

        return "\n".join(summary_lines).strip() if len(summary_lines) > 2 else ""

    @staticmethod
    def _build_open_data_services_source_line(language: str) -> str:
        """Builds a stable Lisboa Aberta source line for nearby-service answers."""
        timestamp = datetime.now().strftime("%H:%M")
        if language == "pt":
            return f"\U0001F4CC **Fonte:** [*Lisboa Aberta*](https://dados.cm-lisboa.pt/) | **Atualizado:** {timestamp}"
        return f"\U0001F4CC **Source:** [*Lisboa Aberta*](https://dados.cm-lisboa.pt/) | **Updated:** {timestamp}"

    @staticmethod
    def _strip_lisboa_aberta_source_lines(text: str) -> str:
        """Remove embedded Lisboa Aberta source footers before final caveats."""
        kept_lines = [
            line for line in str(text or "").splitlines()
            if not ("Lisboa Aberta" in line and re.search(r"\*\*(Fonte|Source):\*\*", line))
        ]
        return "\n".join(kept_lines).strip()

    @staticmethod
    def _build_missing_services_limitation(
        service_types: List[str],
        nearby_location: Optional[str],
        language: str,
    ) -> str:
        """Build a scoped limitation when a municipal service lookup has no reliable result."""
        display_names_pt = {
            "car parking": "estacionamento automóvel",
            "ecopontos": "ecopontos",
        }
        display_names_en = {
            "car parking": "car parking",
            "ecopontos": "recycling points",
        }
        display_names = display_names_pt if language == "pt" else display_names_en
        services = ", ".join(display_names.get(service, service) for service in service_types)
        location = f" perto de {nearby_location}" if language == "pt" and nearby_location else ""
        location_en = f" near {nearby_location}" if language != "pt" and nearby_location else ""
        if language == "pt":
            body = "\n".join(
                [
                    f"### 🧭 **Serviços municipais{location}**",
                    "",
                    f"⚠️ **Resposta direta:** não encontrei resultados municipais fiáveis para **{services}**{location} nos dados disponíveis da Lisboa Aberta.",
                ]
            )
        else:
            body = "\n".join(
                [
                    f"### 🧭 **Municipal services{location_en}**",
                    "",
                    f"⚠️ **Direct answer:** I could not confirm reliable municipal results for **{services}**{location_en} in the available Lisboa Aberta data.",
                ]
            )
        return f"{body}\n\n{ResearcherAgent._build_open_data_services_source_line(language)}"

    @staticmethod
    def _build_area_service_coverage_limitation(
        service_types: List[str],
        area_label: str,
        language: str,
    ) -> str:
        """Explain that Lisboa Aberta service data must not be projected to another AML municipality."""
        display_names_pt = {
            "bibliotecas": "bibliotecas",
            "farmácias": "farmácias",
            "hospitais": "hospitais",
            "escolas": "escolas",
            "jardins": "jardins",
            "mercados": "mercados",
            "polícia": "polícia",
            "car parking": "estacionamento automóvel",
            "ecopontos": "ecopontos",
            "pilhões": "pilhões",
            "papeleiras": "papeleiras",
            "parques caninos": "parques caninos",
            "pontos de encontro de emergência": "pontos de encontro de emergência",
            "Estacionamento de velocípedes": "estacionamento de bicicletas",
        }
        display_names_en = {
            "bibliotecas": "libraries",
            "farmácias": "pharmacies",
            "hospitais": "hospitals",
            "escolas": "schools",
            "jardins": "parks/gardens",
            "mercados": "markets",
            "polícia": "police",
            "car parking": "car parking",
            "ecopontos": "recycling points",
            "pilhões": "battery recycling points",
            "papeleiras": "waste bins",
            "parques caninos": "dog parks",
            "pontos de encontro de emergência": "emergency meeting points",
            "Estacionamento de velocípedes": "bicycle parking",
        }
        names = display_names_pt if language == "pt" else display_names_en
        services = ", ".join(names.get(service, service) for service in service_types)
        source_line = ResearcherAgent._build_open_data_services_source_line(language)
        if language == "pt":
            return "\n".join(
                [
                    f"### 🧭 **Serviços municipais em {area_label}**",
                    "",
                    f"⚠️ **Resposta direta:** {area_label} está dentro da AML, mas não tenho dados municipais confirmáveis para **{services}** nesse município.",
                    "",
                    "A integração de serviços municipais disponível usa dados da **Lisboa Aberta**, que não deve ser extrapolada para outros municípios da AML. Por isso não vou devolver resultados do município de Lisboa como se fossem de outro concelho.",
                    "",
                    source_line,
                ]
            )
        return "\n".join(
            [
                f"### 🧭 **Municipal services in {area_label}**",
                "",
                f"⚠️ **Direct answer:** {area_label} is inside the Lisbon Metropolitan Area, but I do not have confirmable municipal-service records for **{services}** in that municipality.",
                "",
                "The available municipal-service integration uses **Lisboa Aberta** data and should not be projected to other AML municipalities.",
                "",
                source_line,
            ]
        )

    @classmethod
    def _is_event_category_query(cls, user_message: str) -> bool:
        """Return whether the user is asking for event categories, not event instances."""
        normalized = cls._normalize_for_deterministic_routing(user_message)
        return bool(
            re.search(
                r"\b(?:what kinds?|types?|categories?|which kinds?)\b.*\bevents?\b",
                normalized,
            )
            or re.search(r"\bevents?\b.*\b(?:can i look for|can i find|available categories|categories)\b", normalized)
            or re.search(r"\bque tipos? de\b.*\beventos?\b", normalized)
            or re.search(r"\b(?:tipos|categorias) de eventos?\b", normalized)
            or re.search(
                r"\beventos?\b.*\b(?:posso encontrar|posso procurar|posso explorar|categorias disponiveis|categorias)\b",
                normalized,
            )
        )

    @classmethod
    def _is_place_category_query(cls, user_message: str) -> bool:
        """Return whether the user is asking for place categories, not place cards."""
        normalized = cls._normalize_for_deterministic_routing(user_message)
        return bool(
            re.search(
                r"\b(?:what kinds?|types?|categories?|which kinds?)\b.*\b(?:places?|locais|attractions?)\b",
                normalized,
            )
            or re.search(
                r"\b(?:places?|locais|attractions?)\b.*\b(?:can i explore|can i visit|available categories|categories)\b",
                normalized,
            )
            or re.search(r"\bque tipos? de\b.*\b(?:locais|lugares|atracoes?)\b", normalized)
            or re.search(r"\b(?:tipos|categorias) de (?:locais|lugares|atracoes?)\b", normalized)
            or re.search(
                r"\b(?:locais|lugares|atracoes?)\b.*\b(?:posso explorar|posso visitar|posso procurar|categorias disponiveis|categorias)\b",
                normalized,
            )
        )

    @classmethod
    def _is_service_category_query(cls, user_message: str) -> bool:
        """Return whether the user is asking to browse public-service categories."""
        normalized = cls._normalize_for_deterministic_routing(user_message)
        return bool(
            re.search(r"\b(?:what kinds?|types?|categories?)\b.*\b(?:public )?services?\b", normalized)
            or re.search(r"\b(?:public )?services?\b.*\b(?:can you help me find|available categories|categories)\b", normalized)
            or re.search(r"\bque tipos? de\b.*\bservicos?\b", normalized)
            or re.search(r"\b(?:tipos|categorias) de servicos?\b", normalized)
            or re.search(
                r"\bservicos?\b.*\b(?:posso procurar|posso encontrar|podes ajudar|categorias disponiveis|categorias)\b",
                normalized,
            )
        )

    def _run_category_lookup(self, user_message: str, language: str) -> Optional[str]:
        """Runs deterministic category lookups for broad browse-intent queries."""
        category_specs = [
            (self._is_event_category_query, "get_event_categories", lambda _result, lang: self._build_events_source_line(lang)),
            (self._is_place_category_query, "get_place_categories", self._build_places_source_line),
            (self._is_service_category_query, "list_service_categories", lambda _result, lang: self._build_open_data_services_source_line(lang)),
        ]
        for detector, tool_name, source_builder in category_specs:
            if not detector(user_message):
                continue
            tool = self._get_tool_by_name(tool_name)
            if not tool:
                return None
            result = str(self._invoke_tool(tool, {"language": language}, tool_name=tool_name)).strip()
            source_line = source_builder(result, language)
            return f"{result}\n\n{source_line}".strip()
        return None

    def _run_discovery_lookup(self, language: str) -> str:
        """Returns grounded suggestions for broad Lisbon discovery prompts."""
        tool = self._get_tool_by_name("search_places_attractions")
        if not tool:
            if language == "pt":
                return "Posso ajudar com Lisboa, mas a ferramenta de procura de locais está indisponível no momento."
            return "I can help with Lisbon places, but the places search tool is currently unavailable."

        args: Dict[str, Any] = {
            "query": "recommended places and activities in Lisbon",
            "max_results": 5,
            "offset": 0,
            "language": language,
        }
        result = str(self._invoke_tool(tool, args, tool_name="search_places_attractions")).strip()
        result = self._localize_place_card_titles_with_llm(result, language)
        source_line = self._build_places_source_line(result, language)
        return f"{result}\n\n{source_line}".strip()

    @classmethod
    def _is_lisboa_card_query(cls, user_message: str) -> bool:
        """Return whether the query asks about Lisboa Card benefits or inclusion."""
        normalized = cls._normalize_for_deterministic_routing(user_message)
        card_terms = ("lisboa card", "lisbon card")
        benefit_terms = (
            "included", "include", "free", "discount", "benefit", "benefits",
            "entrada", "incluido", "incluida", "gratuito", "gratuita", "gratis",
            "desconto", "beneficio", "beneficios",
            "relevant", "relevance", "apply", "applicable", "suitable", "works", "good for", "for visiting",
        )
        if not (any(term in normalized for term in card_terms) and any(term in normalized for term in benefit_terms)):
            return False
        subject = cls._extract_lisboa_card_subject(user_message).strip()
        if not subject or subject.lower() == user_message.strip().lower():
            return False
        generic_only = re.compile(
            r"^(?:what|which|whats|quais|qual|quanto|quantos|quanta|quantas|how|why|where|when|tem|h[áa]|"
            r"existe|existem|posso|pode|podem|deve|devem|precisa|precisamos|preciso)$",
            re.IGNORECASE,
        )
        meaningful_tokens = [
            tok for tok in re.split(r"\s+", subject)
            if len(tok) >= 3 and not generic_only.match(tok)
        ]
        if not meaningful_tokens:
            return False
        return True

    @classmethod
    def _extract_lisboa_card_subject(cls, user_message: str) -> str:
        """Extract the likely attraction name from a Lisboa Card benefit query."""
        subject = str(user_message or "")
        split_match = re.split(
            r"(?i)\b(?:e|and)\s+(?:que|quais|which|what)\s+(?:atra[cç][oõ]es|locais|places|attractions)\b",
            subject,
            maxsplit=1,
        )
        subject = split_match[0].strip() if split_match else subject

        english_match = re.search(
            r"(?i)\b(?:is|are)\s+(?P<subject>.+?)\s+(?:included|free|discounted)\b",
            subject,
        )
        portuguese_match = re.search(
            r"(?i)\b(?:[ée]|est[aá]|ficam?|s[aã]o)\s+(?P<subject>.+?)\s+(?:inclu[ií]d[ao]s?|gratu[ií]t[ao]s?)\b",
            subject,
        )
        portuguese_match_reverse = re.search(
            r"(?i)(?P<subject>.+?)\s+(?:[ée]|est[aá]|ficam?|s[aã]o)\s+(?:inclu[ií]d[ao]s?|gratu[ií]t[ao]s?)\b",
            subject,
        )

        candidate = subject
        if english_match:
            candidate = english_match.group("subject")
        elif portuguese_match:
            candidate = portuguese_match.group("subject")
        elif portuguese_match_reverse:
            candidate = portuguese_match_reverse.group("subject")
        else:
            include_match = re.search(
                r"(?i)\b(?:inclui|include|includes|included)\s+(?:o|a|os|as|the)?\s*(?P<subject>.+?)(?:\?|$)",
                subject,
            )
            if include_match:
                candidate = include_match.group("subject")

        cleaned = re.sub(
            r"(?i)\b(?:is|are|does|do|isn't|aren't|the|a|an|o|os|as|um|uma|[ée]|est[aá]|fica|fica-se|para|with|in|on|no|na|nos|nas|com|can|can't|cannot|o|a|os|as)\b",
            " ",
            candidate,
        )
        cleaned = re.sub(
            r"(?i)\b(?:included|include|includes|inclui|incluem|inclu[ií]d[ao]s?|free|discount|benefit|benefits|entrada|gratuito|desconto|benef[ií]cios?|relevant|relevante|apply|applicable|suitable|works|for|visiting)\b",
            " ",
            cleaned,
        )
        cleaned = re.sub(r"(?i)\b(?:lisboa card|lisbon card|card)\b", " ", cleaned)
        cleaned = re.sub(r"[^\wÀ-ÿ\s'-]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if not cleaned or len(cleaned) < 3:
            fallback = re.sub(
                r"(?i)\b(?:is|are|does|do|isn't|aren't|the|a|an|o|os|as|um|uma|[ée]|est[aá]|fica|fica-se|para|with|in|on|no|na|nos|nas|com|can|can't|cannot|o|a|os|as)\b",
                " ",
                subject,
            )
            fallback = re.sub(
                r"(?i)\b(?:included|include|includes|inclui|incluem|inclu[ií]d[ao]s?|free|discount|benefit|benefits|entrada|gratuito|desconto|benef[ií]cios?|relevant|relevante|apply|applicable|suitable|works|for|visiting)\b",
                " ",
                fallback,
            )
            fallback = re.sub(r"(?i)\b(?:lisboa card|lisbon card|card)\b", " ", fallback)
            fallback = re.sub(r"[^\wÀ-ÿ\s'-]+", " ", fallback)
            fallback = re.sub(r"\s+", " ", fallback).strip()
            return fallback or user_message

        return cleaned

    @staticmethod
    def _localize_lisboa_card_benefit(benefit: str, language: str) -> str:
        """Localize a compact Lisboa Card benefit phrase."""
        value = str(benefit or "").strip()
        if language != "pt" or not value:
            return value
        value = re.sub(r"\bFree\s+with\s+Lisboa\s+Card\b", "Gratuito com Lisboa Card", value, flags=re.IGNORECASE)
        value = re.sub(r"\bwith\s+Lisboa\s+Card\b", "com Lisboa Card", value, flags=re.IGNORECASE)
        value = re.sub(r"\bexhibitions\b", "exposições", value, flags=re.IGNORECASE)
        return value

    @staticmethod
    def _classify_lisboa_card_benefit(benefit: str) -> str:
        """Classify a Lisboa Card benefit as free admission, discount, or unknown."""
        value = _normalize_researcher_intent_text(benefit)
        if not value:
            return "unknown"
        if re.search(r"\b(?:free|gratuito|gratuita|gratuits?|incluido|incluida|included)\b", value):
            return "free"
        if re.search(r"(?:\d+\s*%|discount|desconto|reduced|reducao|reduzid[ao]|off\b)", value):
            return "discount"
        return "unknown"

    @staticmethod
    def _query_requests_nearby_lisboa_card_places(user_message: str) -> bool:
        """Return whether a Lisboa Card query asks for nearby included places."""
        normalized = _normalize_researcher_intent_text(user_message)
        return bool(
            re.search(r"\b(?:ali perto|perto|nearby|near)\b", normalized)
            and re.search(r"\b(?:atracoes|locais|places|attractions|incluidas|included|beneficios|benefits)\b", normalized)
        )

    @staticmethod
    def _nearby_lisboa_card_candidates(
        best_place: Dict[str, Any],
        places: List[Dict[str, Any]],
        *,
        max_results: int = 4,
    ) -> List[Dict[str, Any]]:
        """Find nearby Lisboa Card candidates using shared postal/area evidence."""
        location_text = str(best_place.get("location") or best_place.get("address") or "")
        title_text = str(best_place.get("title") or "")
        postal_match = re.search(r"\b(\d{4})-\d{3}\b", location_text)
        postal_prefix = postal_match.group(1) if postal_match else ""
        folded_anchor = _normalize_researcher_intent_text(f"{title_text} {location_text}")
        area_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", folded_anchor)
            if len(token) >= 4
            and token not in {
                "lisboa", "lisbon", "rua", "avenida", "av", "praca", "largo",
                "estrada", "museu", "museum", "monastery", "mosteiro", "centro",
            }
        }

        best_title = _normalize_researcher_intent_text(title_text)
        candidates: list[tuple[int, str, Dict[str, Any]]] = []
        for place in places:
            title = str(place.get("title") or "")
            if _normalize_researcher_intent_text(title) == best_title:
                continue
            benefit = str(place.get("lisboa_card_benefit") or place.get("lisboa_card_discount") or "").strip()
            if not benefit:
                continue
            candidate_location = str(place.get("location") or place.get("address") or "")
            candidate_folded = _normalize_researcher_intent_text(f"{title} {candidate_location}")
            score = 0
            if postal_prefix and re.search(rf"\b{re.escape(postal_prefix)}-\d{{3}}\b", candidate_location):
                score += 4
            shared_tokens = area_tokens.intersection(set(re.findall(r"[a-z0-9]+", candidate_folded)))
            score += min(3, len(shared_tokens))
            if score <= 0:
                continue
            benefit_rank = 0 if re.search(r"\bfree\b|gratuit", benefit, flags=re.IGNORECASE) else 1
            candidates.append((score * 10 - benefit_rank, title, place))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        return [place for _, _, place in candidates[:max_results]]

    def _run_lisboa_card_lookup(self, user_message: str, language: str) -> str:
        """Answer Lisboa Card benefit questions from VisitLisboa place data, never event search."""
        knowledge_tool = self._get_tool_by_name("search_lisbon_knowledge")
        if knowledge_tool:
            self._invoke_tool(
                knowledge_tool,
                {"query": user_message, "max_results": 5},
                tool_name="search_lisbon_knowledge",
            )

        subject = self._extract_lisboa_card_subject(user_message)
        places = _load_places_json()
        best_place: Optional[Dict[str, Any]] = None
        best_score = 0.0
        for place in places:
            score = _score_specific_place_lookup_match(place, subject)
            place_has_benefit = bool(place.get("lisboa_card_benefit") or place.get("lisboa_card_discount"))
            best_has_benefit = bool(
                best_place
                and (best_place.get("lisboa_card_benefit") or best_place.get("lisboa_card_discount"))
            )
            if score > best_score or (score == best_score and place_has_benefit and not best_has_benefit):
                best_place = place
                best_score = score

        is_pt = language == "pt"
        heading = "Cartão Lisboa" if is_pt else "Lisboa Card"
        source_line = self._build_places_source_line("", language)
        if not best_place or best_score < 55:
            if is_pt:
                return (
                    f"### 🎫 **{heading}**\n\n"
                    "⚠️ Não consegui confirmar esse benefício nos dados VisitLisboa disponíveis. "
                    "Usa o nome oficial do local para confirmar a página certa.\n\n"
                    f"{source_line}"
                )
            return (
                f"### 🎫 **{heading}**\n\n"
                "⚠️ I could not confirm that benefit in the available VisitLisboa place data. "
                "Use the official place name to confirm the exact page.\n\n"
                f"{source_line}"
            )

        title = str(best_place.get("title") or subject).strip()
        if language == "pt" and subject and _score_specific_place_lookup_match(best_place, subject) >= 55:
            title = subject
        raw_benefit = str(best_place.get("lisboa_card_benefit") or best_place.get("lisboa_card_discount") or "").strip()
        benefit_type = self._classify_lisboa_card_benefit(raw_benefit)
        benefit = self._localize_lisboa_card_benefit(raw_benefit, language)
        url = str(best_place.get("url") or "").strip()
        website = str(best_place.get("website") or "").strip()
        tickets = str(best_place.get("tickets") or best_place.get("ticket_url") or "").strip()
        address = str(best_place.get("address") or best_place.get("location") or "").strip()
        card_lines: List[str] = [f"### 🎫 **{heading}: {title}**", ""]
        if benefit:
            if benefit_type == "free" and is_pt:
                card_lines.append(f"✅ **Sim:** **{title}** está listado com **{benefit}**.")
            elif benefit_type == "free":
                card_lines.append(f"✅ **Yes:** {title} is listed with **{benefit}**.")
            elif is_pt:
                card_lines.append(f"✅ **Sim, mas como desconto:** **{title}** está listado com **{benefit}**, não como entrada gratuita.")
            else:
                card_lines.append(f"✅ **Yes, but as a discount:** {title} is listed with **{benefit}**, not free entry.")
        else:
            if is_pt:
                card_lines.append(f"⚠️ Encontrei o local, mas não encontrei um benefício Lisboa Card confirmado para **{title}** nos dados disponíveis.")
            else:
                card_lines.append(f"⚠️ I found the place, but no confirmed Lisboa Card benefit is listed for **{title}** in the available data.")
        card_lines.append("")
        if address:
            label = "Morada" if is_pt else "Address"
            card_lines.append(f"- 📍 **{label}:** {address}")
        if website:
            label = "Website oficial" if is_pt else "Official website"
            card_lines.append(f"- 🌐 **{label}:** [{label}]({website})")
        if tickets:
            label = "Bilhetes" if is_pt else "Tickets"
            card_lines.append(f"- 🎟️ **{label}:** [{label}]({tickets})")
        elif url:
            label = "Mais detalhes" if is_pt else "More details"
            link_label = "VisitLisboa" if not is_pt else "VisitLisboa"
            card_lines.append(f"- 🌐 **{label}:** [{link_label}]({url})")
        note = (
            "ℹ️ **Nota:** confirma sempre a condição atual antes de comprar o bilhete."
            if is_pt
            else "ℹ️ **Note:** always confirm the current condition before buying the ticket."
        )
        card_lines.extend(["", note])
        if self._query_requests_nearby_lisboa_card_places(user_message):
            nearby = self._nearby_lisboa_card_candidates(best_place, places)
            if nearby:
                section_label = "Atrações com benefício Lisboa Card ali perto" if is_pt else "Nearby places with Lisboa Card benefits"
                card_lines.extend(["", f"### 🎫 **{section_label}**", ""])
                for candidate in nearby:
                    candidate_title = str(candidate.get("title") or "").strip()
                    candidate_benefit = self._localize_lisboa_card_benefit(
                        str(candidate.get("lisboa_card_benefit") or candidate.get("lisboa_card_discount") or "").strip(),
                        language,
                    )
                    candidate_location = str(candidate.get("location") or candidate.get("address") or "").strip()
                    candidate_url = str(candidate.get("url") or "").strip()
                    if not candidate_title or not candidate_benefit:
                        continue
                    card_lines.append(f"- **🏛️ {candidate_title}**")
                    card_lines.append(f"    - 🎫 **Lisboa Card:** {candidate_benefit}")
                    if candidate_location:
                        loc_label = "Morada" if is_pt else "Address"
                        card_lines.append(f"    - 📍 **{loc_label}:** {candidate_location}")
                    if candidate_url:
                        details_label = "Mais detalhes" if is_pt else "More details"
                        card_lines.append(f"    - 🔗 **{details_label}:** [VisitLisboa]({candidate_url})")
            elif is_pt:
                card_lines.extend(["", "⚠️ Não encontrei outras atrações próximas com benefício Lisboa Card suficientemente confirmado nesta fonte."])
            else:
                card_lines.extend(["", "⚠️ I did not find other nearby attractions with a sufficiently confirmed Lisboa Card benefit in this source."])
        card_lines.extend(["", source_line])
        return "\n".join(card_lines).strip()

    @staticmethod
    def _build_tool_call(name: str, args: dict) -> AIMessage:
        """Creates a deterministic tool call message for the subgraph."""
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": name,
                    "args": args,
                    "id": f"auto_{uuid.uuid4().hex}",
                    "type": "tool_call",
                }
            ],
        )

    @staticmethod
    def _normalize_for_deterministic_routing(user_message: str) -> str:
        """Normalize user text for accent-insensitive deterministic routing checks."""
        normalized = unicodedata.normalize("NFKD", user_message or "")
        normalized = normalized.encode("ascii", "ignore").decode("ascii").lower()
        return re.sub(r"\s+", " ", normalized).strip()

    @classmethod
    def _extract_history_culture_subject(cls, user_message: str) -> str:
        """Extract the likely subject from a history/culture lookup query."""
        query = (user_message or "").strip()
        normalized = cls._normalize_for_deterministic_routing(query)
        subject = query
        subject_patterns = [
            r"^.*?\bhistory\s+(?:of|about)\s+",
            r"^.*?\bhistorical\s+context\s+(?:of|about|for)\s+",
            r"^.*?\bhistorical\s+(?:facts\s+)?(?:about|on)\s+",
            r"^.*?\bhistoria\s+(?:de|do|da|dos|das|sobre)\s+",
            r"^.*?\bcultura\s+(?:de|do|da|dos|das|sobre)\s+",
            r"^.*?\bculture\s+(?:of|about)\s+",
        ]
        for pattern in subject_patterns:
            match = re.search(pattern, normalized)
            if match:
                subject = query[match.end():]
                break
        subject = re.sub(
            r"(?i)\b(?:e\s+)?n[ãa]o\s+me\s+d[êe]s\s+(?:um\s+)?(?:roteiro|plano|itiner[áa]rio)\b.*$",
            "",
            subject,
        )
        subject = re.sub(
            r"(?i)\bsem\s+(?:roteiro|plano|itiner[áa]rio)\b.*$",
            "",
            subject,
        )
        subject = re.sub(
            r"(?i)\b(?:and\s+)?do\s+not\s+give\s+me\s+(?:an?\s+)?(?:route|plan|itinerary)\b.*$",
            "",
            subject,
        )
        subject = re.sub(
            r"(?i)\bwithout\s+(?:an?\s+)?(?:route|plan|itinerary)\b.*$",
            "",
            subject,
        )
        subject = re.sub(
            r"(?i)\b(?:,?\s*(?:and\s+)?)?(?:do\s+not|don't)\s+(?:suggest|show|include)\s+events?\b.*$",
            "",
            subject,
        )
        subject = re.sub(
            r"(?i)\b(?:,?\s*(?:e\s+)?)?(?:sem\s+eventos?|n[ãa]o\s+(?:me\s+)?(?:mostres?|sugiras?|incluas?)\s+eventos?)\b.*$",
            "",
            subject,
        )
        subject = re.sub(
            r"(?i)\b(?:em|in)\s+(?:pt-pt|portugu[eê]s(?:\s+europeu)?|english|ingl[eê]s)\b.*$",
            "",
            subject,
        )
        subject = re.sub(
            r"(?i)^\s*(?:explica|explique|resume|resuma|summarize|explain)\s+"
            r"(?:(?:em|in)\s+\d+\s+(?:linhas|lines)\s+)?",
            "",
            subject,
        )
        subject = re.sub(r"(?i)^\s*o\s+que\s+era\s+", "", subject)
        subject = re.sub(r"\s+", " ", subject).strip(" .?!")
        return subject or query

    @staticmethod
    def _build_language_instruction(language: str) -> str:
        """Builds a compact language instruction for subgraph LLM steps."""
        return (
            "Respond ENTIRELY in Portuguese (PT-PT)."
            if language == "pt"
            else "Respond ENTIRELY in English."
        )

    @staticmethod
    def _is_auto_category_tool_result(messages: list) -> bool:
        """Return whether the latest ToolMessage is a deterministic category lookup."""
        if len(messages) < 2 or not isinstance(messages[-1], ToolMessage):
            return False
        latest_tool_name = str(getattr(messages[-1], "name", "") or "")
        if latest_tool_name in {"get_event_categories", "get_place_categories", "list_service_categories"}:
            return True
        previous_message = messages[-2]
        if not isinstance(previous_message, AIMessage) or not getattr(previous_message, "tool_calls", None):
            return False
        tool_call = previous_message.tool_calls[0]
        tool_call_id = str(tool_call.get("id") or "")
        tool_name = str(tool_call.get("name") or "")
        return tool_call_id.startswith("auto_") and tool_name in {
            "get_event_categories",
            "get_place_categories",
            "list_service_categories",
        }

    def _finalize_auto_category_tool_result(
        self,
        messages: list,
        user_message: str,
        language: str,
    ) -> str:
        """Finalize deterministic category results without allowing a second free-form lookup."""
        previous_message = messages[-2]
        tool_name = str(getattr(messages[-1], "name", "") or "")
        if not tool_name and isinstance(previous_message, AIMessage) and getattr(previous_message, "tool_calls", None):
            tool_name = str(previous_message.tool_calls[0].get("name") or "")
        result = str(messages[-1].content or "").strip()
        if tool_name == "get_event_categories":
            result = f"{result}\n\n{self._build_events_source_line(language)}"
        elif tool_name == "get_place_categories":
            result = f"{result}\n\n{self._build_places_source_line(result, language)}"
        elif tool_name == "list_service_categories":
            result = f"{result}\n\n{self._build_open_data_services_source_line(language)}"
        return finalize_worker_response(
            result,
            agent_name="researcher",
            user_query=user_message,
            language=language,
        )

    def _ensure_subgraph_messages(self, messages: list, language: str) -> list:
        """Ensures researcher subgraph LLM calls receive system and language instructions."""
        updated_messages = list(messages)
        if not updated_messages or not isinstance(updated_messages[0], SystemMessage):
            updated_messages = [SystemMessage(content=self.system_prompt)] + updated_messages

        if not any(
            isinstance(message, SystemMessage)
            and "Respond ENTIRELY" in str(message.content)
            for message in updated_messages[:3]
        ):
            updated_messages = [
                updated_messages[0],
                SystemMessage(content=self._build_language_instruction(language)),
                *updated_messages[1:],
            ]

        return updated_messages

    @classmethod
    def _build_deterministic_subgraph_tool_call(cls, user_message: str) -> Optional[AIMessage]:
        """Routes obvious researcher queries to their canonical tool in the subgraph."""
        query = user_message.strip()
        query_lower = query.lower()
        normalized_query = cls._normalize_for_deterministic_routing(query)

        if cls._is_event_category_query(query):
            language = infer_response_language(user_query=query, default="en")
            return cls._build_tool_call("get_event_categories", {"language": language})

        if cls._is_place_category_query(query):
            language = infer_response_language(user_query=query, default="en")
            return cls._build_tool_call("get_place_categories", {"language": language})

        if cls._is_service_category_query(query):
            language = infer_response_language(user_query=query, default="en")
            return cls._build_tool_call("list_service_categories", {"language": language})

        if cls._is_lisboa_card_query(query):
            return cls._build_tool_call("search_lisbon_knowledge", {"query": query, "max_results": 5})

        if cls._is_generic_research_discovery_query(query):
            language = cls._infer_research_query_language(query)
            return cls._build_tool_call(
                "search_places_attractions",
                {
                    "query": "recommended places and activities in Lisbon",
                    "max_results": 5,
                    "language": language,
                },
            )

        if re.search(r"\b(history|historical|historia|culture|cultura)\b", normalized_query):
            subject = cls._extract_history_culture_subject(query)
            language = cls._infer_research_query_language(query)
            if re.search(
                r"\b(give me|summari[sz]e|historical importance|only use|supported details|without inventing|context for)\b",
                query_lower,
            ):
                language = "en"
            return cls._build_tool_call("search_lisbon_knowledge", {"query": subject or query, "max_results": 5})

        if cls._is_direct_event_lookup_query(query) and not cls._is_mixed_event_place_query(query):
            language = cls._infer_research_query_language(query)
            args: Dict[str, Any] = {"max_results": 5, "language": language}
            date_filters = cls._extract_event_date_filters(query)
            category_hint = cls._infer_event_category_hint(query)
            focus_query = cls._extract_event_focus_query(query)
            if date_filters:
                args["date_filter"] = date_filters[0]
            if category_hint:
                args["category"] = category_hint
            if focus_query:
                args["query"] = focus_query
            elif category_hint:
                args["query"] = category_hint
            return cls._build_tool_call("search_cultural_events", args)

        if "service categories" in query_lower or (
            re.search(r"\b(public services?|servi[cç]os p[úu]blicos)\b", query_lower)
            and re.search(r"\b(categories|types|kinds|available|help me find|can you help me find|what kinds)\b", query_lower)
        ):
            language = infer_response_language(user_query=query, default="en")
            return cls._build_tool_call("list_service_categories", {"language": language})

        if "dataset details for" in query_lower:
            dataset_name = re.sub(r"^.*dataset details for\s+", "", query, flags=re.IGNORECASE).strip(" .?!")
            return cls._build_tool_call("get_dataset_details", {"dataset_name": dataset_name or query})

        if "open datasets for" in query_lower:
            place_query = re.sub(r"^.*open datasets for\s+", "", query, flags=re.IGNORECASE).strip(" .?!")
            return cls._build_tool_call("find_place_in_datasets", {"query": place_query or query})

        if "list available lisboa aberta service datasets" in query_lower:
            return cls._build_tool_call("list_available_datasets", {})

        if "event categories" in query_lower or re.search(
            r"\b(what kinds of events|types of events|which events can i look for)\b",
            query_lower,
        ):
            language = infer_response_language(user_query=query, default="en")
            return cls._build_tool_call("get_event_categories", {"language": language})

        if "place categories" in query_lower or re.search(
            r"\b(what kinds of places|types of places|which places can i explore)\b",
            query_lower,
        ):
            language = infer_response_language(user_query=query, default="en")
            return cls._build_tool_call("get_place_categories", {"language": language})

        if "knowledge base for" in query_lower:
            search_query = re.sub(r"^.*knowledge base for\s+", "", query, flags=re.IGNORECASE).strip(" .?!")
            return cls._build_tool_call("search_lisbon_knowledge", {"query": search_query or query, "max_results": 5})

        if query_lower.startswith("encontra farmácias perto do"):
            place_name = re.sub(r"^encontra farmácias perto do\s+", "", query, flags=re.IGNORECASE).strip(" .?!")
            return cls._build_tool_call("find_nearby_services", {"service_type": "farmácias", "near_location_name": place_name or "Rossio", "max_results": 5})

        if "cultural events" in query_lower or "events in lisbon" in query_lower:
            date_filter = "this weekend" if "this weekend" in query_lower else "today" if "today" in query_lower else None
            args = {"query": "cultural events", "max_results": 5}
            if date_filter:
                args["date_filter"] = date_filter
            return cls._build_tool_call("search_cultural_events", args)

        if "attractions related to" in query_lower:
            search_query = re.sub(r"^.*attractions related to\s+", "", query, flags=re.IGNORECASE).strip(" .?!")
            return cls._build_tool_call("search_places_attractions", {"query": search_query or query, "max_results": 5})

        if cls._is_visit_place_context_query(query):
            language = infer_response_language(user_query=query, default="en")
            focus_query = cls._extract_place_focus_query(query) or query
            return cls._build_tool_call(
                "search_places_attractions",
                {
                    "query": f"attractions monuments museums in {focus_query}",
                    "category": "Museums & Monuments",
                    "max_results": 5,
                    "language": language,
                },
            )

        service_types = cls._extract_service_types(query)
        if service_types:
            language = infer_response_language(user_query=query, default="en")
            args: Dict[str, Any] = {
                "service_type": service_types[0],
                "max_results": 5,
                "language": language,
            }
            nearby_location = cls._clean_nearby_location_text(cls._extract_near_location_name(query))
            if nearby_location:
                args["near_location_name"] = nearby_location
            category_hint = cls._service_category_for_type(service_types[0])
            if category_hint:
                args["category"] = category_hint
            return cls._build_tool_call("find_nearby_services", args)

        place_keywords = [
            "museum", "museu", "monument", "monumento", "restaurant", "restaurante",
            "hotel", "viewpoint", "miradouro", "beach", "praia", "garden", "jardim",
            "park", "parque", "hospital", "pharmacy", "farmácia", "farmacia", "school",
            "escola", "library", "biblioteca", "police", "polícia", "policia", "belém", "belem",
        ]
        if any(keyword in query_lower for keyword in place_keywords):
            args = {"query": query, "max_results": 5}
            category_hint = cls._infer_place_category_hint(query)
            if category_hint:
                args["category"] = category_hint
            return cls._build_tool_call("search_places_attractions", args)

        return None

    @traceable(name="researcher_agent", run_type="chain", tags=["sub-agent", "researcher"])
    def invoke(
        self, user_message: str, context: str = "", verbose: bool = False
    ) -> str:
        """
        Processes a places/events query using semantic search.

        Args:
            user_message: The user's query.
            context: Additional context from other agents (optional).
            verbose: Whether involved tool calls should be printed.

        Returns:
            str: Places/events information response.
        """
        # Extract explicit language preference from context if provided
        import re
        language_match = re.search(r"User language:\s*(en|pt)", context, re.IGNORECASE)
        if language_match:
            language = language_match.group(1).lower()
        else:
            language = self._infer_research_query_language(user_message)
        if language == "pt" and re.search(
            r"\b(give me|summari[sz]e|historical importance|only use|supported details|without inventing|context for)\b",
            user_message.lower(),
        ):
            language = "en"
        self._active_response_language = language

        # Skip tool enforcement for greetings/thanks
        is_greeting = any(
            w in user_message.lower()
            for w in ["hello", "thanks", "obrigado", "tchau", "olá", "bom dia"]
        )

        if not is_greeting and (replayed_response := self._replay_same_deterministic_response_once(user_message)):
            return replayed_response

        if not is_greeting:
            continued_search_response = self._maybe_continue_previous_search(user_message, language)
            if continued_search_response:
                if verbose:
                    print("      [RESEARCHER] Continuing previous paginated search...")
                return self._remember_deterministic_response_for_retry(user_message, finalize_worker_response(
                    continued_search_response,
                    agent_name="researcher",
                    user_query=user_message,
                    language=language,
                ))

        # Conversational recall fast-path: if the user is asking us to recall
        # something we already mentioned (e.g. "Qual foi o restaurante que
        # indicaste?", "What restaurant did you suggest?"), answer from the
        # previous assistant turn provided in context instead of issuing a
        # fresh search that loses the original recommendation.
        if not is_greeting:
            recall_response = self._maybe_answer_conversational_recall(
                user_message, context, language
            )
            if recall_response:
                if verbose:
                    print("      [RESEARCHER] Using conversational recall fast-path...")
                return self._remember_deterministic_response_for_retry(
                    user_message,
                    finalize_worker_response(
                        recall_response,
                        agent_name="researcher",
                        user_query=user_message,
                        language=language,
                    ),
                )

        if not is_greeting and self._is_unsupported_private_service_query(user_message):
            if verbose:
                print("      [RESEARCHER] Using private-service coverage guard...")
            response = self._build_unsupported_private_service_response(user_message, language)
            return self._remember_deterministic_response_for_retry(
                user_message,
                finalize_worker_response(
                    response,
                    agent_name="researcher",
                    user_query=user_message,
                    language=language,
                ),
            )

        messages = self._build_messages(self.system_prompt, user_message, context, language=language)

        tool_enforcement_msg = "" if is_greeting else (
            "You MUST use a tool (like search_places_attractions) to get real data. "
            "Do NOT answer from your knowledge base. Call the tool now."
        )

        if not is_greeting and self._is_visit_confirmation_checklist_query(user_message):
            if verbose:
                print("      [RESEARCHER] Using deterministic place lookup for visit-confirmation checklist...")

            response = self._run_visit_confirmation_checklist_lookup(user_message, language)
            return self._remember_deterministic_response_for_retry(user_message, finalize_worker_response(
                response,
                agent_name="researcher",
                user_query=user_message,
                language=language,
            ))

        if not is_greeting and self._is_accessibility_place_query(user_message):
            if verbose:
                print("      [RESEARCHER] Using deterministic place lookup for accessibility-focused query...")

            response = self._run_accessibility_place_lookup(user_message, language)
            return self._remember_deterministic_response_for_retry(user_message, finalize_worker_response(
                response,
                agent_name="researcher",
                user_query=user_message,
                language=language,
            ))

        if not is_greeting and self._is_lisboa_card_query(user_message):
            if verbose:
                print("      [RESEARCHER] Using deterministic Lisboa Card benefit lookup...")

            response = self._run_lisboa_card_lookup(user_message, language)
            return self._remember_deterministic_response_for_retry(user_message, finalize_worker_response(
                response,
                agent_name="researcher",
                user_query=user_message,
                language=language,
            ))

        if not is_greeting and self._is_planner_evidence_request(user_message):
            if verbose:
                print("      [RESEARCHER] Returning deterministic planner evidence cards...")

            response = self._run_planner_evidence_lookup(user_message, language)
            if response:
                return self._remember_deterministic_response_for_retry(user_message, finalize_worker_response(
                    response,
                    agent_name="researcher",
                    user_query=user_message,
                    language=language,
                ))

        if not is_greeting and self._is_history_culture_query(user_message):
            if verbose:
                print("      [RESEARCHER] Using deterministic history/culture lookup...")

            knowledge_tool = self._get_tool_by_name("search_lisbon_knowledge")
            if knowledge_tool:
                subject = self._extract_history_culture_subject(user_message) or user_message
                raw_knowledge = str(
                    self._invoke_tool(
                        knowledge_tool,
                        {"query": subject, "max_results": 5},
                        tool_name="search_lisbon_knowledge",
                    )
                ).strip()
                raw_knowledge = re.sub(
                    r"\n?📅\s+\*\*Related Events:\*\*.*",
                    "",
                    raw_knowledge,
                    flags=re.DOTALL,
                ).strip()
                raw_knowledge = re.sub(
                    r"\n?🏛️\s+\*\*Related Places:\*\*.*",
                    "",
                    raw_knowledge,
                    flags=re.DOTALL,
                ).strip()
                prose_lines = [
                    line.strip() for line in raw_knowledge.splitlines()
                    if len(line.strip()) > 60
                    and not re.search(
                        r"Guia\s+Lxcard|p\.\d{1,3}\s+—|Lisboa\s+Knowledge\s+Search|"
                        r"Guide\s*/\s*PDF\s+Knowledge|🔍\s+\*\*Lisbon\s+Knowledge\s+Search",
                        line.strip(),
                        flags=re.IGNORECASE,
                    )
                ]
                has_prose_content = len(prose_lines) >= 1
                compact_knowledge = self._compact_history_result(raw_knowledge, language, subject)
                if compact_knowledge and has_prose_content:
                    timestamp = datetime.now().strftime("%H:%M")
                    heading = f"### 📚 Contexto histórico: {subject}" if language == "pt" else f"### 📚 Historical Context: {subject}"
                    source = (
                        f"📌 **Fonte:** *Guia Lisboa Card* | **Atualizado:** {timestamp}"
                        if language == "pt"
                        else f"📌 **Source:** *Lisboa Card Guide* | **Updated:** {timestamp}"
                    )
                    return self._remember_deterministic_response_for_retry(
                        user_message,
                        finalize_worker_response(
                            f"{heading}\n\n{compact_knowledge}\n\n{source}",
                            agent_name="researcher",
                            user_query=user_message,
                            language=language,
                        ),
                    )

            subject = self._extract_history_culture_subject(user_message) or user_message
            history_tool = self._get_tool_by_name("search_history_culture")
            if history_tool:
                raw_history = str(
                    self._invoke_tool(
                        history_tool,
                        {"query": subject, "language": language},
                        tool_name="search_history_culture",
                    )
                ).strip()
                compact_history = self._compact_history_result(raw_history, language, subject)
                if compact_history:
                    timestamp = datetime.now().strftime("%H:%M")
                    heading = f"### 📚 Contexto histórico: {subject}" if language == "pt" else f"### 📚 Historical Context: {subject}"
                    source = (
                        f"📌 **Fonte:** [*Wikipedia/Web*](https://www.wikipedia.org/) | **Atualizado:** {timestamp}"
                        if language == "pt"
                        else f"📌 **Source:** [*Wikipedia/Web*](https://www.wikipedia.org/) | **Updated:** {timestamp}"
                    )
                    response = f"{heading}\n\n{compact_history}\n\n{source}"
                else:
                    response = raw_history
            else:
                response = self._run_direct_tool_fallback(user_message, language)
            return self._remember_deterministic_response_for_retry(user_message, finalize_worker_response(
                response,
                agent_name="researcher",
                user_query=user_message,
                language=language,
            ))

        if not is_greeting:
            category_response = self._run_category_lookup(user_message, language)
            if category_response:
                if verbose:
                    print("      [RESEARCHER] Using deterministic category lookup...")
                return self._remember_deterministic_response_for_retry(user_message, finalize_worker_response(
                    category_response,
                    agent_name="researcher",
                    user_query=user_message,
                    language=language,
                ))

        if not is_greeting:
            after_hours_response = self._maybe_answer_after_hours_culture_query(user_message, language)
            if after_hours_response:
                if verbose:
                    print("      [RESEARCHER] Using after-hours culture recommendation guard...")
                return self._remember_deterministic_response_for_retry(user_message, finalize_worker_response(
                    after_hours_response,
                    agent_name="researcher",
                    user_query=user_message,
                    language=language,
                ))

        if not is_greeting and self._is_transactional_place_lookup_query(user_message):
            if verbose:
                print("      [RESEARCHER] Using deterministic transactional place lookup...")
            response = self._run_direct_place_lookup(user_message, language)
            return self._remember_deterministic_response_for_retry(user_message, finalize_worker_response(
                response,
                agent_name="researcher",
                user_query=user_message,
                language=language,
            ))

        if (
            not is_greeting
            and self._is_direct_event_lookup_query(user_message)
            and not self._is_mixed_event_place_query(user_message)
            and not self._is_free_museum_event_query(user_message)
        ):
            if verbose:
                print("      [RESEARCHER] Using deterministic direct event lookup...")

            response = self._run_direct_event_lookup(user_message, language)
            finalized_response = finalize_worker_response(
                response,
                agent_name="researcher",
                user_query=user_message,
                language=language,
            )
            finalized_response = self._filter_event_result_for_excluded_museum_venues(
                finalized_response,
                user_message,
                language,
            )
            return self._remember_deterministic_response_for_retry(user_message, finalized_response)

        if not is_greeting:
            structured_response = self._maybe_run_structured_query_plan(user_message, language)
            if structured_response:
                if verbose:
                    print("      [RESEARCHER] Using structured LLM-assisted deterministic routing...")
                return self._remember_deterministic_response_for_retry(user_message, finalize_worker_response(
                    structured_response,
                    agent_name="researcher",
                    user_query=user_message,
                    language=language,
                ))

        if not is_greeting and self._is_visit_place_context_query(user_message):
            if verbose:
                print("      [RESEARCHER] Using deterministic visit-area place lookup...")

            response = self._run_direct_place_lookup(user_message, language)
            return self._remember_deterministic_response_for_retry(user_message, finalize_worker_response(
                response,
                agent_name="researcher",
                user_query=user_message,
                language=language,
            ))

        event_followup_context = bool(
            getattr(self, "_last_search_context", None)
            and str((self._last_search_context or {}).get("domain") or "") == "events"
            and self._is_named_lookup_followup(user_message)
        )
        if (
            not is_greeting
            and self._extract_service_types(user_message)
            and not self._is_direct_event_lookup_query(user_message)
            and not self._is_event_category_query(user_message)
            and not event_followup_context
        ):
            if verbose:
                print("      [RESEARCHER] Using deterministic Lisboa Aberta service lookup...")

            response = self._run_direct_tool_fallback(user_message, language)
            return self._remember_deterministic_response_for_retry(user_message, finalize_worker_response(
                response,
                agent_name="researcher",
                user_query=user_message,
                language=language,
            ))

        if not is_greeting and self._is_free_museum_event_query(user_message):
            if verbose:
                print("      [RESEARCHER] Using guarded free museum/event lookup...")
            response = self._run_free_museum_event_guard(language)
            return self._remember_deterministic_response_for_retry(user_message, finalize_worker_response(
                response,
                agent_name="researcher",
                user_query=user_message,
                language=language,
            ))

        if not is_greeting and self._is_direct_event_lookup_query(user_message) and not self._is_mixed_event_place_query(user_message):
            if verbose:
                print("      [RESEARCHER] Using deterministic direct event lookup...")

            response = self._run_direct_event_lookup(user_message, language)
            finalized_response = finalize_worker_response(
                response,
                agent_name="researcher",
                user_query=user_message,
                language=language,
            )
            finalized_response = self._filter_event_result_for_excluded_museum_venues(
                finalized_response,
                user_message,
                language,
            )
            return self._remember_deterministic_response_for_retry(user_message, finalized_response)

        if not is_greeting and hasattr(self, "tools") and self._is_direct_place_lookup_query(user_message):
            if verbose:
                print("      [RESEARCHER] Using deterministic direct place lookup...")

            response = self._run_direct_place_lookup(user_message, language)
            return self._remember_deterministic_response_for_retry(user_message, finalize_worker_response(
                response,
                agent_name="researcher",
                user_query=user_message,
                language=language,
            ))

        if not is_greeting and self._is_generic_research_discovery_query(user_message):
            if verbose:
                print("      [RESEARCHER] Using deterministic discovery lookup for broad Lisbon query...")
            response = self._run_discovery_lookup(language)
            return self._remember_deterministic_response_for_retry(user_message, finalize_worker_response(
                response,
                agent_name="researcher",
                user_query=user_message,
                language=language,
            ))

        last_search_context = getattr(self, "_last_search_context", None)
        if not is_greeting and last_search_context and self._is_named_lookup_followup(user_message):
            cached_domain = str(last_search_context.get("domain") or "").strip()
            explicit_place_lookup = (
                self._is_direct_place_lookup_query(user_message)
                or self._is_broad_attractions_query(user_message)
            )
            if cached_domain == "events" and not explicit_place_lookup:
                if verbose:
                    print("      [RESEARCHER] Resolving named follow-up against previous event domain...")

                response = self._run_direct_event_lookup(user_message, language)
                return self._remember_deterministic_response_for_retry(user_message, finalize_worker_response(
                    response,
                    agent_name="researcher",
                    user_query=user_message,
                    language=language,
                ))
            if cached_domain == "places" or explicit_place_lookup:
                if verbose:
                    print("      [RESEARCHER] Resolving named follow-up against previous place domain...")

                response = self._run_direct_place_lookup(user_message, language)
                return self._remember_deterministic_response_for_retry(user_message, finalize_worker_response(
                    response,
                    agent_name="researcher",
                    user_query=user_message,
                    language=language,
                ))

        if (
            not is_greeting
            and self._is_history_culture_query(user_message)
            and not self._is_direct_place_lookup_query(user_message)
        ):
            if verbose:
                print("      [RESEARCHER] Using deterministic history/culture lookup...")

            response = self._run_direct_tool_fallback(user_message, language)
            return self._remember_deterministic_response_for_retry(user_message, finalize_worker_response(
                response,
                agent_name="researcher",
                user_query=user_message,
                language=language,
            ))

        if not is_greeting and hasattr(self, "tools") and self._is_broad_attractions_query(user_message):
            if verbose:
                print("      [RESEARCHER] Using deterministic broad attractions lookup...")

            response = self._run_direct_place_lookup(user_message, language)
            return self._remember_deterministic_response_for_retry(user_message, finalize_worker_response(
                response,
                agent_name="researcher",
                user_query=user_message,
                language=language,
            ))

        try:
            response = self.execute_react_loop(
                messages=messages,
                verbose=verbose,
                max_iterations=6,
                tool_enforcement_msg=tool_enforcement_msg,
            )
        except Exception as e:
            if not self._is_content_filter_error(e):
                raise

            if verbose:
                print("      [RESEARCHER] Retrying with safe prompt variant after content filter...")

            safe_messages = self._build_messages(
                get_researcher_prompt(safe_mode=True),
                user_message,
                context,
                language=language,
            )
            try:
                response = self.execute_react_loop(
                    messages=safe_messages,
                    verbose=verbose,
                    max_iterations=6,
                    tool_enforcement_msg=tool_enforcement_msg,
                )
            except Exception as safe_error:
                if not self._is_content_filter_error(safe_error):
                    raise

                if verbose:
                    print("      [RESEARCHER] Falling back to direct tool invocation after repeated content-filter blocks...")

                response = self._run_direct_tool_fallback(user_message, language)

        return finalize_worker_response(
            response,
            agent_name="researcher",
            user_query=user_message,
            language=language,
        )

    def build_subgraph(self) -> "CompiledStateGraph":
        """
        Builds a LangGraph subgraph for this agent.

        Returns:
            CompiledStateGraph: Compiled subgraph for researcher queries.
        """

        def agent_node(state: AgentState) -> dict:
            """Researcher agent decision node."""
            messages = list(state["messages"])

            user_message = None
            for message in reversed(messages):
                if isinstance(message, HumanMessage) and message.content:
                    user_message = str(message.content)
                    break

            language = infer_response_language(user_query=user_message or "", default="en")

            last_message = messages[-1] if messages else None
            if isinstance(last_message, ToolMessage):
                if self._is_auto_category_tool_result(messages):
                    finalized_response = self._finalize_auto_category_tool_result(
                        messages,
                        user_message or "",
                        language,
                    )
                    return {"messages": [AIMessage(content=finalized_response)]}

                response = self._safe_llm_invoke(
                    self.llm_with_tools,
                    self._ensure_subgraph_messages(messages, language),
                )
                return {"messages": [response]}

            if user_message:
                deterministic_call = self._build_deterministic_subgraph_tool_call(user_message)
                if deterministic_call is not None:
                    return {"messages": [deterministic_call]}

            response = self._safe_llm_invoke(
                self.llm_with_tools,
                self._ensure_subgraph_messages(messages, language),
            )
            return {"messages": [response]}

        def should_continue(state: AgentState) -> str:
            """Determines next step."""
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return "end"

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", "end": END}
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile()


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 Researcher Agent Test\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")

    try:
        agent = ResearcherAgent()
        print(
            f"\n\033[1m✅ Researcher Agent initialized:\033[0m {agent.get_model_info()}"
        )
        print(f"   Tools: {[t.name for t in agent.tools]}")

        print("\n\033[1m📝 Testing query:\033[0m 'Museums in Lisbon'")
        response = agent.invoke("Museums in Lisbon")
        print("\n\033[1m🤖 Response:\033[0m")
        print(response)

        print("\n\033[1;32m✅ Researcher agent working!\033[0m")

    except Exception as e:
        print(f"\n\033[1;31m❌ Error:\033[0m {e}")
        import traceback

        traceback.print_exc()
