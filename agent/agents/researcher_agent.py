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
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from tools.visitlisboa_api import (
    _extract_specific_event_lookup_phrase,
    _extract_specific_place_lookup_phrase,
    _load_places_json,
    _score_specific_place_lookup_match,
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
    "police": {
        "tool_label": "polícia",
        "category": None,
        "dataset_term": "Polícia Municipal",
        "aliases": {"police", "policia", "polícia"},
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
    "bike_parking": {
        "tool_label": "Estacionamento de velocípedes",
        "category": None,
        "dataset_term": "Estacionamento de velocípedes",
        "aliases": {
            "bike_parking", "bicycle_parking", "bikeparking", "bicycleparking",
            "estacionamento_de_bicicletas", "estacionamento_de_velocipedes", "bicicletas",
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

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

from agent.agents.base import BaseAgent, parse_json_response
from agent.prompts.researcher import get_researcher_prompt
from agent.utils.langsmith_tracing import traceable
from agent.state import AgentState
from agent.utils.langgraph_compat import ToolNode
from agent.utils.response_formatter import (
    finalize_worker_response,
    infer_response_language,
    resolve_output_language,
)


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
        # Tools are loaded by BaseAgent.__init__ via get_agent_tools("researcher")
        # which returns the full set including dados_abertos tools

    def reset_conversation_context(self) -> None:
        """Clears cached result-window context for this session."""
        self._last_search_context = None
        self._pending_deterministic_replay = None
        self._pending_pagination_replay = None

    def get_last_search_context(self) -> Optional[dict]:
        """Returns the latest cached result-window context."""
        return deepcopy(self._last_search_context)

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
        normalized = re.sub(r"\s+", "_", normalized).strip("_")
        return normalized

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

        return (
            any(marker in normalized for marker in dataset_markers)
            or any(marker in normalized for marker in unsupported_service_markers)
            or named_service_with_nearby
            or event_with_explicit_calendar
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
            verbose=False,
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
    def _structured_service_tool_label(service_type: str) -> Optional[str]:
        """Resolve a canonical structured service enum to the best nearby-service tool label."""
        definition = _STRUCTURED_SERVICE_TYPE_DEFINITIONS.get(service_type)
        if not definition:
            return None
        return str(definition.get("tool_label") or "").strip() or None

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
            "attraction", "attractions", "belem", "belém",
        ]
        return any(term in query for term in accessibility_terms) and (
            any(term in query for term in place_terms)
            or bool(ResearcherAgent._extract_place_focus_query(user_message))
        )

    def _run_accessibility_place_lookup(self, user_message: str, language: str) -> str:
        """Runs a deterministic place lookup for accessibility-focused queries."""
        tool = self._get_tool_by_name("search_places_attractions")
        if not tool:
            return self._run_direct_tool_fallback(user_message, language)

        focus_query = self._extract_place_focus_query(user_message)
        args = {"query": focus_query or user_message, "max_results": 5, "offset": 0, "language": language}
        specific_lookup = _extract_specific_place_lookup_phrase(user_message)
        specific_tokens = re.findall(r"[a-z0-9]+", (specific_lookup or "").lower())
        broad_type_tokens = {"museum", "museums", "museu", "museus", "monument", "monuments"}
        broad_category_lookup = len(specific_tokens) <= 2 and any(
            token in broad_type_tokens for token in specific_tokens
        )
        if specific_lookup and not broad_category_lookup:
            args["specific_lookup"] = True
        category_hint = self._infer_place_category_hint(user_message) or self._infer_place_category_hint(focus_query or "")
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
        return f"{result}\n\n{source_line}".strip()

    @staticmethod
    def _infer_place_category_hint(user_message: str) -> Optional[str]:
        """Infers a high-level VisitLisboa place category from common PT/EN query terms."""
        query = (user_message or "").lower()

        if any(term in query for term in ["museum", "museu", "monument", "monumento"]):
            return "Museums & Monuments"
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
                "brunch",
                "cafe",
                "café",
            ]
        ):
            return "Restaurants"
        if any(term in query for term in ["hotel", "hotels", "accommodation", "lodging", "stay", "alojamento"]):
            return "Hotels"
        if any(term in query for term in ["viewpoint", "view point", "miradouro", "scenic view"]):
            return "View Points"

        return None

    @staticmethod
    def _build_places_source_line(result: str, language: str) -> str:
        """Builds the right source line for direct place lookups, including hybrid open-data results."""
        if "Open Data:" in result or re.search(r"\b[1-9]\d*\s+from Lisboa Aberta\b", result or ""):
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
            r"\b(?:mais|proxim[oa]s?|seguintes?)\s+(?:\d{1,2}\s+)?(?:resultados?|locais?|eventos?|op[cç][oõ]es|opcoes)\b",
            r"\b(?:mostra|mostre|d[aá]-me|quero)\s+mais\b",
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
        """Counts numbered items in raw VisitLisboa-style outputs."""
        return len(re.findall(r"(?m)^\s*\d+\.\s+", str(result or "")))

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
        attraction_phrases = [
            "atrações imperdíveis",
            "atracoes imperdiveis",
            "atrações",
            "atracoes",
            "must-see",
            "must see",
            "first time",
            "primeira vez",
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
    def _is_direct_event_lookup_query(user_message: str) -> bool:
        """Detects event-discovery queries that are safer to answer directly from tools."""
        query = (user_message or "").lower()
        if any(term in query for term in ["history", "história", "historia", "culture", "cultura"]):
            return False

        specific_lookup = _extract_specific_event_lookup_phrase(user_message)
        planning_terms = [
            "plan", "plano", "roteiro", "itinerary", "agenda",
            "combine", "combinar", "day plan", "plan my day",
        ]
        event_terms = [
            "event", "events", "evento", "eventos", "concert", "concerto",
            "festival", "festivals", "exhibition", "exposição", "exposicao",
            "music", "música", "musica", "show", "theatre", "teatro",
            "dance", "dança", "danca", "cinema", "what's on", "o que há", "o que ha",
            "fair", "fairs", "feira", "feiras", "book fair",
            "summit", "conference", "congress", "forum", "expo",
        ]
        named_lookup_markers = [
            "tell me about", "what about", "more about", "details about", "information about",
            "sobre", "fala-me de", "fala me de", "fala-me do", "fala me do", "fala-me da", "fala me da",
            "diz-me sobre", "diz me sobre", "diz-me do", "diz me do", "diz-me da", "diz me da",
            "e do", "e da",
        ]
        return (
            (
                bool(specific_lookup)
                and not ResearcherAgent._is_direct_place_lookup_query(user_message)
            )
            or
            any(term in query for term in event_terms)
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
        event_terms = [
            "event", "events", "evento", "eventos", "concert", "concerto",
            "festival", "festivals", "exhibition", "exposição", "exposicao",
            "music", "música", "musica", "show", "theatre", "teatro",
            "dance", "dança", "danca", "cinema", "what's on", "o que há", "o que ha",
            "fair", "fairs", "feira", "feiras",
        ]
        place_terms = [
            "museum", "museums", "museu", "museus",
            "restaurant", "restaurants", "restaurante", "restaurantes",
            "pharmacy", "pharmacies", "farmácia", "farmacias", "farmácias",
            "hospital", "hospitals", "attraction", "attractions", "place", "places",
            "local", "locais", "monument", "monuments", "monumento", "monumentos",
        ]
        return any(term in query for term in event_terms) and any(term in query for term in place_terms)

    @staticmethod
    def _extract_event_date_filter(user_message: str) -> Optional[str]:
        """Extracts a lightweight date filter for direct event tool lookups."""
        query = (user_message or "").lower()
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
        return None

    @staticmethod
    def _is_category_date_event_discovery(user_message: str) -> bool:
        """Detects category/date event browsing rather than a named event lookup."""
        query = (user_message or "").lower()
        if not ResearcherAgent._extract_event_date_filter(user_message):
            return False
        if not ResearcherAgent._infer_event_category_hint(user_message):
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
            "diz-me",
            "diz me",
        ]
        has_quoted_title = any(symbol in (user_message or "") for symbol in ['"', '“', '”'])
        return not has_quoted_title and not any(marker in query for marker in named_lookup_markers)

    @staticmethod
    def _is_uncertain_place_recommendation_request(user_message: str) -> bool:
        """Detects requests that explicitly ask the assistant to guess uncertain places."""
        query = (user_message or "").lower()
        uncertainty_markers = [
            "not completely sure",
            "not sure",
            "não tiveres a certeza",
            "nao tiveres a certeza",
            "mesmo sem certeza",
            "even if",
            "guess",
            "invent",
        ]
        place_markers = ["hidden spots", "spots", "places", "locais", "sítios", "sitios"]
        return any(marker in query for marker in uncertainty_markers) and any(
            marker in query for marker in place_markers
        )

    @staticmethod
    def _is_outdoor_event_query(user_message: str) -> bool:
        """Detects explicit outdoor-event discovery requests."""
        normalized_query = unicodedata.normalize("NFKD", user_message or "")
        normalized_query = normalized_query.encode("ascii", "ignore").decode("ascii").lower()
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
        if specific_lookup:
            return specific_lookup

        quoted_match = re.search(r'"([^"\n]{2,120})"|“([^”\n]{2,120})”', user_message or "")
        if quoted_match:
            quoted_subject = next((group for group in quoted_match.groups() if group), "").strip(" .?!")
            if quoted_subject:
                return quoted_subject

        query = (user_message or "").lower()
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
            "explore", "explorar", "lisbon", "lisboa", "temos", "there", "happening", "have",
            "algo", "interessante", "fazer", "para", "perto", "near", "around", "theres",
            "are", "should", "bring", "umbrella", "weather", "rain", "chuva",
        }
        tokens = [token for token in re.findall(r"[a-zA-ZÀ-ÿ0-9]+", query) if len(token) >= 3]
        meaningful_tokens = [token for token in tokens if token not in generic_terms]
        return " ".join(dict.fromkeys(meaningful_tokens)) if meaningful_tokens else None

    @staticmethod
    def _infer_event_category_hint(user_message: str) -> Optional[str]:
        """Infers a VisitLisboa event category hint from common PT/EN event queries."""
        query = (user_message or "").lower()
        if any(term in query for term in ["summit", "conference", "congress", "forum", "expo", "technology", "tech", "startup"]):
            return "Main Events"
        if any(term in query for term in ["music", "música", "musica", "concert", "concerto", "fado", "jazz", "rock", "pop"]):
            return "Music"
        if any(term in query for term in ["theatre", "theater", "teatro", "opera", "dance", "dança", "danca", "ballet"]):
            return "Theater Opera & Dance"
        if any(term in query for term in ["exhibition", "exhibitions", "exposição", "exposicao", "art", "arte", "gallery", "galeria"]):
            return "Exhibitions"
        if any(term in query for term in ["festival", "festivais", "festivals"]):
            return "Festivals"
        if any(term in query for term in ["sport", "sports", "desporto", "desportos", "marathon", "maratona"]):
            return "Sports"
        if any(term in query for term in ["cinema", "film", "movie", "movies"]):
            return "Cinema"
        if any(term in query for term in ["fair", "fairs", "feira", "feiras", "market", "mercado"]):
            return "Fairs"
        if any(term in query for term in ["food", "gastronomy", "gastronomia", "wine", "vinho"]):
            return "Gastronomy"
        return None

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
                return quoted_subject

        visit_area_match = re.search(
            r"\b(?:visit|visiting|explore|visitar|conhecer|explorar)\s+(?:a\s+|o\s+|os\s+|as\s+)?(?P<subject>[A-ZÀ-Ýa-zà-ÿ0-9][A-ZÀ-Ýa-zà-ÿ0-9 '\-/]{1,80}?)(?=\s+(?:tomorrow|today|tonight|amanh[aã]|hoje|esta noite|this week|this weekend)\b|[\?\!\.,]|$)",
            query,
            flags=re.IGNORECASE,
        )
        if visit_area_match:
            subject = visit_area_match.group("subject").strip(" .?!,")
            if subject:
                return subject

        named_lookup_re = re.compile(
            r"\b(?:tell me about|what about|more about|details about|information about|where is|where's|find|show me|sobre(?: o| a| os| as)?|fala[- ]?me(?: mais)? sobre(?: o| a| os| as)?|fala[- ]?me(?: mais)? (?:de|do|da|dos|das)|fale[- ]?me(?: mais)? (?:de|do|da|dos|das)|diz[- ]?me(?: mais)? sobre(?: o| a| os| as)?|diz[- ]?me (?:de|do|da|dos|das)|diz[- ]?me onde(?: e| é| fica)(?: o| a| os| as)?|onde(?: e| é| fica)(?: o| a| os| as)?|encontra(?:r)?|mostrar(?:-me)?)\b",
            re.IGNORECASE,
        )
        if named_lookup_re.search(query):
            subject = named_lookup_re.sub(" ", query)
            subject = re.sub(r"\s+", " ", subject).strip(" .?!")
            if subject:
                return subject

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
                        return " ".join(proper_nouns)
                    return None
                return query.strip(" .?!")

        specific_lookup = _extract_specific_place_lookup_phrase(user_message)
        if specific_lookup:
            return specific_lookup

        return None

    @staticmethod
    def _is_direct_place_lookup_query(user_message: str) -> bool:
        """Detects straightforward place and service lookups that are safer to answer directly from tools."""
        query = (user_message or "").lower()
        history_keywords = ["history", "história", "historia", "culture", "cultura"]
        event_keywords = [
            "event", "events", "evento", "eventos", "concert", "concerto",
            "festival", "exhibition", "exposição", "exposicao", "show",
            "fair", "fairs", "feira", "feiras", "book fair",
        ]
        directed_lookup_markers = [
            "where is", "where's", "onde fica", "onde é", "onde e", "tell me about", "what about",
            "more about", "details about", "information about", "sobre",
            "fala-me de", "fala me de", "fala-me do", "fala me do", "fala-me da", "fala me da",
            "diz-me sobre", "diz me sobre", "diz-me do", "diz me do", "diz-me da", "diz me da",
            "diz-me onde é", "diz me onde e", "diz-me onde fica", "diz me onde fica",
            "closest to", "nearest to", "near ",
            "mais perto", "mais próximo", "perto de", "perto do", "perto da",
        ]
        if any(keyword in query for keyword in history_keywords + event_keywords):
            return False

        focus_query = ResearcherAgent._extract_place_focus_query(user_message)
        if ResearcherAgent._extract_service_types(user_message):
            return bool(ResearcherAgent._extract_near_location_name(user_message) or focus_query)

        has_place_hint = ResearcherAgent._infer_place_category_hint(user_message) is not None
        has_directional_lookup = any(marker in query for marker in directed_lookup_markers)
        has_recommendation_lookup = bool(
            re.search(
                r"\b(?:best|top|recommended|recommend|suggest|suggested|good|melhores|principais|recomenda|sugere)\b",
                query,
            )
            or re.search(r"\bwhat are\b.*\b(?:museums?|monuments?|restaurants?|hotels?|viewpoints?)\b", query)
            or re.search(r"\bquais s(?:ã|a)o\b.*\b(?:museus|monumentos|restaurantes|hot[eé]is|miradouros)\b", query)
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
        if has_place_hint and has_recommendation_lookup:
            return True

        return False

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
        return bool(re.search(r"\b(history|historical|historia|culture|cultura)\b", normalized))

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

        patterns = [
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
                location = re.split(
                    r"\b(?:that|which|who|where|still|open|useful|tonight|this evening|que|e que)\b",
                    location,
                    maxsplit=1,
                    flags=re.IGNORECASE,
                )[0].strip(" .?!,")
                return location or None
        return None

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

        args = {"max_results": 5, "language": language, "offset": 0}
        date_filter = self._normalize_structured_date_filter(
            structured_plan.get("date_filter"),
            user_message,
        ) if structured_plan else self._extract_event_date_filter(user_message)
        category_hint = self._infer_event_category_hint(user_message)
        outdoor_event_query = self._is_outdoor_event_query(user_message)
        broad_date_discovery = (
            not structured_plan
            and bool(date_filter)
            and category_hint is None
            and not outdoor_event_query
        )
        category_date_discovery = (
            not structured_plan
            and self._is_category_date_event_discovery(user_message)
            and not outdoor_event_query
        )
        specific_lookup = None if structured_plan else _extract_specific_event_lookup_phrase(user_message)
        focus_query = self._normalize_structured_plan_text(structured_plan.get("subject")) if structured_plan else None
        raw_extracted_focus_query = self._extract_event_focus_query(user_message)
        if broad_date_discovery:
            extracted_focus_query = None
        elif category_date_discovery and str(raw_extracted_focus_query or "").lower() not in {"live music", "música ao vivo", "musica ao vivo"}:
            extracted_focus_query = None
        else:
            extracted_focus_query = raw_extracted_focus_query
        if broad_date_discovery or category_date_discovery:
            specific_lookup = None
        elif not structured_plan and date_filter and not extracted_focus_query:
            specific_lookup = None
        focus_query = focus_query or specific_lookup or extracted_focus_query
        if outdoor_event_query and not focus_query:
            focus_query = "outdoor events"

        if date_filter:
            args["date_filter"] = date_filter
        if category_hint:
            args["category"] = category_hint
        if focus_query:
            args["query"] = focus_query
        if specific_lookup:
            args["specific_lookup"] = True

        result = str(self._invoke_tool(events_tool, args, tool_name="search_cultural_events")).strip()
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
        match = re.search(r"(?m)^###\s+(?:[\U0001F300-\U0001FAFF\u2300-\u27BF\uFE0F\u200D]+\s+)?(.+?)\s*$", result or "")
        return match.group(1).strip() if match else ""

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
        curated_facts = ResearcherAgent._known_place_history_facts(subject, language)
        subject_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKD", subject or "").encode("ascii", "ignore").decode("ascii").lower())
            if len(token) >= 4 and token not in {"historia", "history", "lisboa", "portugal"}
        }
        text = re.sub(r"(?m)^\s*📌\s+\*\*(?:Fonte|Source):.*$", "", text)
        text = re.sub(
            r"(?im)^\s*[📚🌐🔎]*\s*(?:\*\*)?(?:Wikip[eé]dia|Wikipedia):\s*[^*\n]+(?:\*\*)?\s*",
            "",
            text,
        )
        text = re.sub(r"\[[^\]]+\]\(([^)]+)\)", lambda match: match.group(0).split("](", 1)[0].lstrip("["), text)
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
                    r"\b(?:manuel|unesco|patrimonio|heritage|descobr|maritim|maritime|constru|built|seculo|century|monast|mosteiro|arquitet|architecture|defens|royal|regio|tejo|tagus)\b",
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
    def _extend_place_source_line_with_history(source_line: str, language: str) -> str:
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
    ) -> str:
        """Fetch and format a concise historical context section for a specific place."""
        if not subject or not self._should_add_place_history_context(user_message, result):
            return ""
        history_tool = self._get_tool_by_name("search_history_culture")
        if not history_tool:
            return ""
        canonical_subject = self._extract_primary_place_title(result) or subject
        query = canonical_subject
        raw_history = str(
            self._invoke_tool(
                history_tool,
                {"query": query, "language": language},
                tool_name="search_history_culture",
            )
        ).strip()
        compact_history = self._compact_history_result(raw_history, language, canonical_subject)
        if not compact_history:
            return ""
        heading = f"### 📜 Factos Históricos de {canonical_subject}" if language == "pt" else f"### 📜 Historical Facts About {canonical_subject}"
        return f"---\n\n{heading}\n\n{compact_history}"

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
        for place in places:
            title = str(place.get("title") or "")
            category = normalize(place.get("category"))
            text = normalize(" ".join(str(place.get(key) or "") for key in ("title", "category", "short_description", "description")))
            if any(proxy in category or proxy in normalize(title) for proxy in ("tram", "tour", "cruise", "hotel", "restaurant", "shopping")):
                continue
            if "museum" in category or "museu" in category:
                continue
            if "monument" not in category and "monumento" not in category:
                continue
            score = 0
            if any(term in text for term in ("outdoor", "exterior", "view", "vista", "river", "tejo", "tagus", "miradouro", "panoramic")):
                score += 35
            if any(term in text for term in ("façade", "facade", "fachada", "monument", "monumento", "square", "praça")):
                score += 20
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

        timestamp = datetime.now().strftime("%H:%M")
        maps_url = f"https://www.google.com/maps/search/?api=1&query={address.replace(' ', '+')}" if address else ""
        if language == "pt":
            lines = [
                f"### 🏛️ Recomendação Para {requested_window}",
                "",
                "Para essa janela, eu evitaria recomendar **museus interiores**, porque muitos fecham antes ou perto das 18:00. A opção mais segura é um monumento exterior ou visitável por fora.",
                "",
                f"- 🏛️ **{title} (exterior)**",
            ]
            if description:
                lines.append(f"    - 📝 **Porque faz sentido:** {description}")
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
            f"### 🏛️ Recommendation For {requested_window}",
            "",
            "For that window, I would avoid recommending **indoor museums**, because many close before or around 18:00. The safer option is an outdoor monument or a place that still works from outside.",
            "",
            f"- 🏛️ **{title} (outside)**",
        ]
        if description:
            lines.append(f"    - 📝 **Why it fits:** {description}")
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
        place_focus_query = structured_subject or self._extract_place_focus_query(user_message)
        specific_lookup = _extract_specific_place_lookup_phrase(user_message)
        if structured_subject and not specific_lookup:
            specific_lookup = structured_subject
        service_types = self._extract_service_types(user_message)
        for structured_service in structured_plan.get("service_types", []) if structured_plan else []:
            tool_label = self._structured_service_tool_label(structured_service)
            if tool_label and tool_label not in service_types:
                service_types.append(tool_label)
        nearby_location = self._normalize_structured_plan_text(structured_plan.get("near_location")) if structured_plan else None
        nearby_location = nearby_location or self._extract_near_location_name(user_message)
        is_broad_attractions = self._is_broad_attractions_query(user_message)

        if places_tool and place_focus_query and specific_lookup and not is_broad_attractions and not service_types:
            exact_args = {
                "query": place_focus_query,
                "max_results": 5,
                "offset": 0,
                "language": language,
                "specific_lookup": True,
            }
            exact_category_hint = self._infer_place_category_hint(user_message)
            if exact_category_hint:
                exact_args["category"] = exact_category_hint

            exact_result = str(self._invoke_tool(places_tool, exact_args, tool_name="search_places_attractions")).strip()
            # Accept the specific-lookup result when either (a) it is a clean exact
            # match, or (b) it is a "specific not found, here are alternatives"
            # response that nevertheless surfaces ranked alternatives.
            # Falling through to the broad lookup in case (b) would re-call the tool with
            # weaker arguments and return less useful data, while doubling the tool-call cost.
            if exact_result and not exact_result.startswith("Error:"):
                has_fallback_intro = self._has_specific_lookup_fallback_intro(exact_result)
                shown_count = self._count_ranked_results(exact_result)
                accept_clean_exact = not has_fallback_intro and not exact_result.startswith("❌")
                accept_fallback_with_alternatives = has_fallback_intro and shown_count > 0
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
                    history_section = self._build_place_history_section(
                        place_focus_query,
                        user_message,
                        exact_result,
                        language,
                    )
                    source_line = self._build_places_source_line(exact_result, language)
                    if history_section:
                        source_line = self._extend_place_source_line_with_history(source_line, language)
                        return f"{exact_result}\n\n{history_section}\n\n{source_line}".strip()
                    return f"{exact_result}\n\n{source_line}".strip()

        if nearby_tool and service_types:
            service_blocks: List[str] = []
            missing_services: List[str] = []

            for service_type in service_types:
                service_args = {
                    "service_type": service_type,
                    "max_results": 5,
                }
                if nearby_location:
                    service_args["near_location_name"] = nearby_location
                category_hint = self._service_category_for_type(service_type)
                if not category_hint and structured_plan:
                    normalized_structured_services = structured_plan.get("service_types", [])
                    for structured_service in normalized_structured_services:
                        if self._structured_service_tool_label(structured_service) == service_type:
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
                if missing_services:
                    missing_label = ", ".join(missing_services)
                    if language == "pt":
                        combined += f"\n\n⚠️ Não foi possível confirmar resultados para: {missing_label}."
                    else:
                        combined += f"\n\n⚠️ I could not confirm results for: {missing_label}."
                if "Lisboa Aberta" not in combined:
                    combined += f"\n\n{self._build_open_data_services_source_line(language)}"
                return combined.strip()

        if not places_tool:
            return self._run_direct_tool_fallback(user_message, language)

        query_text = place_focus_query or user_message
        max_results = 5
        if is_broad_attractions:
            query_text = f"{user_message} iconic monuments museums palaces castles historic sites"
            max_results = 6
        elif self._is_visit_place_context_query(user_message) and place_focus_query:
            query_text = f"attractions monuments museums in {place_focus_query}"

        args = {"query": query_text, "max_results": max_results, "offset": 0, "language": language}
        if specific_lookup and not is_broad_attractions:
            args["specific_lookup"] = True
        if is_broad_attractions:
            args["category"] = "Museums & Monuments"
        elif self._is_visit_place_context_query(user_message) and place_focus_query:
            args["category"] = "Museums & Monuments"
        else:
            category_hint = self._infer_place_category_hint(user_message)
            if category_hint:
                args["category"] = category_hint

        result = str(self._invoke_tool(places_tool, args, tool_name="search_places_attractions")).strip()
        if is_broad_attractions and language == "pt":
            rewrite_result = getattr(self, "_rewrite_broad_attractions_result", None)
            if callable(rewrite_result):
                result = str(rewrite_result(result, user_message, language)).strip()
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
        return f"{result}\n\n{source_line}".strip()

    def _run_direct_tool_fallback(self, user_message: str, language: str) -> str:
        """
        Runs a deterministic tool-only fallback when Azure blocks both prompt
        attempts. This avoids failing benign queries like 'Museums in Lisbon'.
        """
        message_lower = user_message.lower()

        history_keywords = ["history", "história", "historia", "culture", "cultura"]
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
            tool = self._get_tool_by_name("find_nearby_services")
            if tool:
                nearby_location = self._extract_near_location_name(user_message)
                blocks: List[str] = []
                missing_services: List[str] = []

                for service_type in service_types:
                    args = {
                        "service_type": service_type,
                        "max_results": 5,
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
                    if "parking" in service_types and re.search(
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
                    combined += f"\n\n{self._build_open_data_services_source_line(language)}"
                    return combined.strip()

        if any(keyword in message_lower for keyword in history_keywords):
            tool = self._get_tool_by_name("search_history_culture")
            if tool:
                subject = self._extract_history_culture_subject(user_message) or user_message
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
                    if language == "pt":
                        heading = f"### 📚 Contexto histórico: {subject}"
                        source = "📌 **Fonte:** [*Wikipedia/Web*](https://www.wikipedia.org/)"
                    else:
                        heading = f"### 📚 Historical Context: {subject}"
                        source = "📌 **Source:** [*Wikipedia/Web*](https://www.wikipedia.org/)"
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

        fallback_text = (
            "I couldn't complete the semantic search prompt flow, but the retrieval tools are available."
            if language == "en"
            else "Não consegui concluir o fluxo semântico do prompt, mas as ferramentas de pesquisa continuam disponíveis."
        )
        return fallback_text

    @staticmethod
    def _extract_service_types(user_message: str) -> List[str]:
        """Extracts one or more practical service types from a service query."""
        normalized_query = unicodedata.normalize("NFKD", user_message or "")
        normalized_query = normalized_query.encode("ascii", "ignore").decode("ascii").lower()
        parking_context = bool(re.search(
            r"\b(?:parking|car\s+parks?|park\s+my\s+car|estacionamento|estacionar|parques?\s+de\s+estacionamento)\b",
            normalized_query,
        ))
        service_catalog = [
            (("pharmacy", "pharmacies", "farm", "pharmac"), "farm\u00e1cias"),
            (("hospital", "hospitals", "hospit", "clinic", "clinica", "clinicas", "cl\u00ednica", "cl\u00ednicas"), "hospitais"),
            (("school", "schools", "escola", "escolas", "sch"), "escolas"),
            (("library", "libraries", "bibliot", "librar"), "bibliotecas"),
            (("park", "parks", "garden", "gardens", "jardim", "jardins", "parque infantil", "parques infantis", "playground", "infantil"), "jardins"),
            (("police", "polic"), "pol\u00edcia"),
            (("parking", "estacion", "car park", "park my car", "parque de estacionamento"), "parking"),
            (("market", "markets", "mercado", "mercados", "feira", "feiras"), "mercados"),
            (("firefighter", "firefighters", "bombeiro", "bombeiros"), "bombeiros"),
            (("restroom", "restrooms", "toilet", "toilets", "casa de banho", "sanitario", "sanitarios"), "sanit\u00e1rios"),
            (("embassy", "embassies", "embaixada", "embaixadas"), "embaixadas"),
            (("citizen shop", "loja do cidadao", "loja do cidad\u00e3o", "servi\u00e7os", "servicos", "posto de correios", "correios"), "Loja do Cidad\u00e3o"),
        ]

        extracted: List[str] = []
        for markers, normalized_service in service_catalog:
            if normalized_service == "jardins" and parking_context:
                continue
            if any(marker in normalized_query for marker in markers) and normalized_service not in extracted:
                extracted.append(normalized_service)
        return extracted

    @classmethod
    def _extract_service_type(cls, user_message: str) -> str:
        """Extracts the first practical service keyword for open-data fallback."""
        extracted = cls._extract_service_types(user_message)
        return extracted[0] if extracted else user_message

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
        if normalized == "jardins":
            return "ambiente"
        if normalized in {"policia", "bombeiros"}:
            return "seguran\u00e7a"
        if normalized == "parking":
            return "transportes"
        if normalized in {"estacionamento", "sanitarios", "loja do cidadao"}:
            return "servi\u00e7os"
        if normalized == "mercados":
            return "com\u00e9rcio"
        return None

    @staticmethod
    def _build_open_data_services_source_line(language: str) -> str:
        """Builds a stable Lisboa Aberta source line for nearby-service answers."""
        timestamp = datetime.now().strftime("%H:%M")
        if language == "pt":
            return f"\U0001F4CC **Fonte:** [*Lisboa Aberta*](https://dados.cm-lisboa.pt/) | **Atualizado:** {timestamp}"
        return f"\U0001F4CC **Source:** [*Lisboa Aberta*](https://dados.cm-lisboa.pt/) | **Updated:** {timestamp}"

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
        )

    @classmethod
    def _is_service_category_query(cls, user_message: str) -> bool:
        """Return whether the user is asking to browse public-service categories."""
        normalized = cls._normalize_for_deterministic_routing(user_message)
        return bool(
            re.search(r"\b(?:what kinds?|types?|categories?)\b.*\b(?:public )?services?\b", normalized)
            or re.search(r"\b(?:public )?services?\b.*\b(?:can you help me find|available categories|categories)\b", normalized)
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

    @classmethod
    def _is_lisboa_card_query(cls, user_message: str) -> bool:
        """Return whether the query asks about Lisboa Card benefits or inclusion."""
        normalized = cls._normalize_for_deterministic_routing(user_message)
        card_terms = ("lisboa card", "lisbon card")
        benefit_terms = (
            "included", "include", "free", "discount", "benefit", "benefits",
            "entrada", "incluido", "incluida", "gratuito", "desconto", "beneficio", "beneficios",
        )
        return any(term in normalized for term in card_terms) and any(term in normalized for term in benefit_terms)

    @classmethod
    def _extract_lisboa_card_subject(cls, user_message: str) -> str:
        """Extract the likely attraction name from a Lisboa Card benefit query."""
        subject = str(user_message or "")
        subject = re.sub(r"(?i)\b(?:is|are|does|do|o|a|os|as|est[aá]|fica|inclu[ií]d[oa]s?)\b", " ", subject)
        subject = re.sub(r"(?i)\b(?:included|include|free|discount|benefit|benefits|entrada|gratuito|desconto|benef[ií]cios?)\b", " ", subject)
        subject = re.sub(r"(?i)\b(?:in|with|on|no|na|nos|nas|com|the|lisboa card|lisbon card|card)\b", " ", subject)
        subject = re.sub(r"[^\wÀ-ÿ\s'-]+", " ", subject)
        subject = re.sub(r"\s+", " ", subject).strip()
        return subject or user_message

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
            if score > best_score:
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
        benefit = str(best_place.get("lisboa_card_benefit") or best_place.get("lisboa_card_discount") or "").strip()
        url = str(best_place.get("url") or "").strip()
        website = str(best_place.get("website") or "").strip()
        tickets = str(best_place.get("tickets") or best_place.get("ticket_url") or "").strip()
        address = str(best_place.get("address") or best_place.get("location") or "").strip()
        card_lines: List[str] = [f"### 🎫 **{heading}: {title}**", ""]
        if benefit:
            if is_pt:
                card_lines.append(f"✅ **Sim, mas como desconto:** o {title} está listado com **{benefit}**, não como entrada gratuita.")
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
        card_lines.extend(["", note, "", source_line])
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
        return subject.strip(" .?!") or query

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

        if cls._is_lisboa_card_query(query):
            return cls._build_tool_call("search_lisbon_knowledge", {"query": query, "max_results": 5})

        if re.search(r"\b(history|historical|historia|culture|cultura)\b", normalized_query):
            subject = cls._extract_history_culture_subject(query)
            language = infer_response_language(user_query=query, default="en")
            return cls._build_tool_call("search_history_culture", {"query": subject, "language": language})

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
            args: Dict[str, Any] = {"service_type": service_types[0], "max_results": 5}
            nearby_location = cls._extract_near_location_name(query)
            if nearby_location:
                args["near_location_name"] = nearby_location
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

        # Skip tool enforcement for greetings/thanks
        is_greeting = any(
            w in user_message.lower()
            for w in ["hello", "thanks", "obrigado", "tchau", "olá", "bom dia"]
        )

        if not is_greeting and (replayed_response := self._replay_same_deterministic_response_once(user_message)):
            return replayed_response

        messages = self._build_messages(self.system_prompt, user_message, context, language=language)

        tool_enforcement_msg = "" if is_greeting else (
            "You MUST use a tool (like search_places_attractions) to get real data. "
            "Do NOT answer from your knowledge base. Call the tool now."
        )

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

        last_search_context = getattr(self, "_last_search_context", None)
        if not is_greeting and last_search_context and self._is_named_lookup_followup(user_message):
            cached_domain = str(last_search_context.get("domain") or "").strip()
            if cached_domain == "events":
                if verbose:
                    print("      [RESEARCHER] Resolving named follow-up against previous event domain...")

                response = self._run_direct_event_lookup(user_message, language)
                return self._remember_deterministic_response_for_retry(user_message, finalize_worker_response(
                    response,
                    agent_name="researcher",
                    user_query=user_message,
                    language=language,
                ))
            if cached_domain == "places":
                if verbose:
                    print("      [RESEARCHER] Resolving named follow-up against previous place domain...")

                response = self._run_direct_place_lookup(user_message, language)
                return self._remember_deterministic_response_for_retry(user_message, finalize_worker_response(
                    response,
                    agent_name="researcher",
                    user_query=user_message,
                    language=language,
                ))

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

        if not is_greeting and self._is_history_culture_query(user_message):
            if verbose:
                print("      [RESEARCHER] Using deterministic history/culture lookup...")

            response = self._run_direct_tool_fallback(user_message, language)
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
            return self._remember_deterministic_response_for_retry(user_message, finalize_worker_response(
                response,
                agent_name="researcher",
                user_query=user_message,
                language=language,
            ))

        if not is_greeting and hasattr(self, "tools") and (
            self._is_direct_place_lookup_query(user_message)
            or self._is_broad_attractions_query(user_message)
        ):
            if verbose:
                print("      [RESEARCHER] Using deterministic direct place lookup...")

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
