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
from typing import TYPE_CHECKING, List, Optional

from tools.visitlisboa_api import _extract_specific_event_lookup_phrase, _extract_specific_place_lookup_phrase

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

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

from agent.agents.base import BaseAgent
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
        for tool in self.tools:
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
        if _extract_specific_place_lookup_phrase(user_message):
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

        explicit_window_hint = any(
            token in query
            for token in ["next", "following", "another", "próxim", "proxim", "seguint"]
        )
        generic_more_hint = any(token in query for token in ["more", "mais"])
        result_nouns = [
            "result", "results", "event", "events", "evento", "eventos",
            "place", "places", "local", "locais", "attraction", "attractions",
            "atração", "atrações", "atracao", "atracoes",
            "option", "options", "opção", "opções", "opcao", "opcoes",
            "museum", "museums", "museu", "museus", "hospital", "hospitals",
            "restaurant", "restaurants", "restaurante", "restaurantes",
        ]
        if not explicit_window_hint and not (generic_more_hint and any(noun in query for noun in result_nouns)):
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

        named_lookup_re = re.compile(
            r"\b(?:tell me about|what about|more about|details about|information about|where is|where's|find|show me|sobre(?: o| a| os| as)?|fala[- ]?me(?: mais)? sobre(?: o| a| os| as)?|fala[- ]?me de|diz[- ]?me(?: mais)? sobre(?: o| a| os| as)?|diz[- ]?me de|diz[- ]?me onde(?: e| é| fica)(?: o| a| os| as)?|onde(?: e| é| fica)(?: o| a| os| as)?|encontra(?:r)?|mostrar(?:-me)?)\b",
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

        if focus_query and any(marker in query for marker in [
            "where is", "where's", "onde fica", "onde é", "onde e", "tell me about", "what about", "more about",
            "details about", "information about", "sobre", "fala-me de", "fala me de", "fala-me do",
            "fala me do", "fala-me da", "fala me da", "diz-me sobre", "diz me sobre", "diz-me do",
            "diz me do", "diz-me da", "diz me da", "diz-me onde é", "diz me onde e", "diz-me onde fica", "diz me onde fica",
        ]):
            return True

        if has_place_hint and has_directional_lookup:
            return True

        return False

    @staticmethod
    def _extract_near_location_name(user_message: str) -> Optional[str]:
        """Extracts a nearby-location target from simple PT/EN service phrasings."""
        patterns = [
            r"\bclosest to\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bnearest to\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bnear\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bnear the\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bmais perto de\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bmais perto do\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bmais perto da\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bmais próximo de\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bmais próximo do\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bmais próximo da\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bperto de\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bperto do\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bperto da\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, user_message, flags=re.IGNORECASE)
            if match:
                return match.group("location").strip(" .?!,")
        return None

    def _run_direct_event_lookup(self, user_message: str, language: str) -> str:
        """Runs a deterministic VisitLisboa event lookup with explicit date parsing."""
        events_tool = self._get_tool_by_name("search_cultural_events")
        if not events_tool:
            return (
                "Não consegui aceder à pesquisa de eventos neste momento."
                if language == "pt"
                else "I couldn't access the event search tool right now."
            )

        args = {"max_results": 5, "language": language, "offset": 0}
        date_filter = self._extract_event_date_filter(user_message)
        specific_lookup = _extract_specific_event_lookup_phrase(user_message)
        focus_query = specific_lookup or self._extract_event_focus_query(user_message)
        category_hint = self._infer_event_category_hint(user_message)

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

    def _run_direct_place_lookup(self, user_message: str, language: str) -> str:
        """Runs a deterministic tool path for simple place and multi-service lookups."""
        places_tool = self._get_tool_by_name("search_places_attractions")
        nearby_tool = self._get_tool_by_name("find_nearby_services")
        place_focus_query = self._extract_place_focus_query(user_message)
        specific_lookup = _extract_specific_place_lookup_phrase(user_message)
        service_types = self._extract_service_types(user_message)
        nearby_location = self._extract_near_location_name(user_message)

        if nearby_tool and service_types and (nearby_location or not place_focus_query):
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
        if self._is_broad_attractions_query(user_message):
            query_text = f"{user_message} iconic monuments museums palaces castles historic sites"
            max_results = 6

        args = {"query": query_text, "max_results": max_results, "offset": 0, "language": language}
        if specific_lookup:
            args["specific_lookup"] = True
        if self._is_broad_attractions_query(user_message):
            args["category"] = "Museums & Monuments"
        else:
            category_hint = self._infer_place_category_hint(user_message)
            if category_hint:
                args["category"] = category_hint

        result = str(self._invoke_tool(places_tool, args, tool_name="search_places_attractions")).strip()
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
                return str(
                    self._invoke_tool(
                        tool,
                        {"query": user_message, "language": language},
                        tool_name="search_history_culture",
                    )
                )

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
        service_catalog = [
            (("pharmacy", "pharmacies", "farm", "pharmac"), "farm\u00e1cias"),
            (("hospital", "hospitals", "hospit"), "hospitais"),
            (("school", "schools", "escola", "escolas", "sch"), "escolas"),
            (("library", "libraries", "bibliot", "librar"), "bibliotecas"),
            (("park", "parks", "garden", "gardens", "jardim", "jardins"), "jardins"),
            (("police", "polic"), "pol\u00edcia"),
        ]

        extracted: List[str] = []
        for markers, normalized_service in service_catalog:
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
        return None

    @staticmethod
    def _build_open_data_services_source_line(language: str) -> str:
        """Builds a stable Lisboa Aberta source line for nearby-service answers."""
        timestamp = datetime.now().strftime("%H:%M")
        if language == "pt":
            return f"\U0001F4CC **Fonte:** [*Lisboa Aberta*](https://dados.cm-lisboa.pt/) | **Atualizado:** {timestamp}"
        return f"\U0001F4CC **Source:** [*Lisboa Aberta*](https://dados.cm-lisboa.pt/) | **Updated:** {timestamp}"

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
    def _build_language_instruction(language: str) -> str:
        """Builds a compact language instruction for subgraph LLM steps."""
        return (
            "Respond ENTIRELY in Portuguese (PT-PT)."
            if language == "pt"
            else "Respond ENTIRELY in English."
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

        if "search the web for the history of" in query_lower:
            subject = re.sub(r"^.*history of\s+", "", query, flags=re.IGNORECASE).strip(" .?!")
            return cls._build_tool_call("search_history_culture", {"query": subject or query, "language": "en"})

        if "service categories" in query_lower:
            return cls._build_tool_call("list_service_categories", {})

        if "dataset details for" in query_lower:
            dataset_name = re.sub(r"^.*dataset details for\s+", "", query, flags=re.IGNORECASE).strip(" .?!")
            return cls._build_tool_call("get_dataset_details", {"dataset_name": dataset_name or query})

        if "open datasets for" in query_lower:
            place_query = re.sub(r"^.*open datasets for\s+", "", query, flags=re.IGNORECASE).strip(" .?!")
            return cls._build_tool_call("find_place_in_datasets", {"query": place_query or query})

        if "list available lisboa aberta service datasets" in query_lower:
            return cls._build_tool_call("list_available_datasets", {})

        if "event categories" in query_lower:
            return cls._build_tool_call("get_event_categories", {})

        if "place categories" in query_lower:
            return cls._build_tool_call("get_place_categories", {})

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
