# ==========================================================================
# Master Thesis - Supervisor Agent
#   - André Filipe Gomes Silvestre, 20240502
#
#   Smart router that analyzes user intent and decides which specialized
#   agents to invoke. Only calls agents when necessary.
# ==========================================================================

import re
import unicodedata
from contextlib import suppress
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agent.agents.base import BaseAgent, clean_response, parse_json_response
from agent.prompts.supervisor import get_supervisor_prompt
from agent.utils.langsmith_tracing import traceable
from agent.utils.response_formatter import strip_unsupported_closing_offers


class SupervisorAgent(BaseAgent):
    """
    Supervisor agent that routes queries to specialized agents.

    Responsibilities:
        - Analyze user query intent
        - Decide which agents to call (can be 0, 1, or multiple)
        - Handle simple queries directly without calling agents
        - Return routing decisions as structured JSON
    """

    def __init__(self):
        """Initializes the supervisor agent."""
        super().__init__("supervisor")
        # System prompt is now dynamic per request

    @staticmethod
    def _normalize_query(user_message: str) -> str:
        """Normalizes user text for lightweight routing heuristics."""
        normalized = unicodedata.normalize("NFKD", str(user_message or ""))
        normalized = "".join(
            char for char in normalized if not unicodedata.combining(char)
        )
        normalized = re.sub(r"[!?.,;:]+", " ", normalized.lower()).strip()
        return re.sub(r"\s+", " ", normalized)

    @staticmethod
    def _query_tokens(text: str) -> List[str]:
        """Extract normalized alphanumeric tokens for lightweight fuzzy checks."""
        return [token for token in re.findall(r"[a-z0-9]+", str(text or "")) if token]

    @classmethod
    def _contains_domain_keyword(
        cls,
        user_message: str,
        keywords: List[str],
        *,
        minimum_ratio: float = 0.86,
    ) -> bool:
        """Returns whether a query matches a domain keyword exactly or with a mild typo.

        This keeps the supervisor deterministic for obvious single-domain queries
        while tolerating small spelling mistakes such as ``weathr`` or ``musems``.
        """
        normalized = cls._normalize_query(user_message)
        if not normalized:
            return False

        normalized_keywords: List[str] = []
        for keyword in keywords:
            normalized_keyword = cls._normalize_query(keyword)
            if not normalized_keyword:
                continue
            normalized_keywords.append(normalized_keyword)
            if " " in normalized_keyword:
                if normalized_keyword in normalized:
                    return True
                continue
            if re.search(rf"\b{re.escape(normalized_keyword)}\b", normalized):
                return True

        query_tokens = cls._query_tokens(normalized)
        single_token_keywords = [
            keyword for keyword in normalized_keywords if " " not in keyword and len(keyword) >= 4
        ]
        for token in query_tokens:
            if len(token) < 4:
                continue
            for keyword in single_token_keywords:
                score = SequenceMatcher(None, token, keyword).ratio()
                if token in keyword or keyword in token:
                    score += 0.08
                if score >= minimum_ratio:
                    return True

        return False

    @classmethod
    def _is_greeting_only(cls, user_message: str) -> bool:
        """Detects greeting-only messages that should bypass the LLM."""
        normalized = cls._normalize_query(user_message)
        return normalized in {
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "olá",
            "ola",
            "bom dia",
            "boa tarde",
            "boa noite",
            "thanks",
            "thank you",
            "obrigado",
            "obrigada",
        }

    @classmethod
    def _has_lisbon_context(cls, message_lower: str) -> bool:
        """Returns whether the query clearly references Lisbon/AML topics."""
        lisbon_keywords = [
            "lisbon",
            "lisboa",
            "aml",
            "metro",
            "bus",
            "autocarro",
            "comboio",
            "transport",
            "transporte",
            "weather",
            "tempo",
            "forecast",
            "previsão",
            "museum",
            "museu",
            "event",
            "evento",
            "restaurant",
            "restaurante",
            "pharmacy",
            "farmácia",
            "hospital",
            "route",
            "rota",
            "belém",
            "belem",
            "chiado",
            "alfama",
            "bairro alto",
            "rossio",
            "oriente",
            "sintra",
            "cascais",
        ]
        return cls._contains_domain_keyword(
            message_lower,
            lisbon_keywords,
            minimum_ratio=0.88,
        )

    @classmethod
    def _looks_like_weather_query(cls, message_lower: str) -> bool:
        """Detects weather queries without over-matching generic PT words like `tempo`."""
        normalized = cls._normalize_query(message_lower)
        weather_patterns = [
            r"\bweather\b",
            r"\brain\b",
            r"\btemperature\b",
            r"\bwind\b",
            r"\bforecast\b",
            r"\bumbrella\b",
            r"\bmeteo\b",
            r"\bchover\b",
            r"\bprevis[aã]o\b",
            r"\bchuva\b",
            r"\btemperatura\b",
            r"\bvento\b",
            r"\bsol\b",
            r"\bsailing\b",
            r"\bsail\b",
            r"\bsea conditions\b",
            r"\bmarine forecast\b",
            r"\bcomo est[aá] o tempo\b",
            r"\bqual (?:é|e) o tempo\b",
            r"\btempo em\b",
            r"\btempo hoje\b",
            r"\btempo amanh[aã]\b",
        ]
        if any(re.search(pattern, normalized) for pattern in weather_patterns):
            return True

        return False

    @classmethod
    def _looks_like_transport_query(cls, message_lower: str) -> bool:
        """Detects transport and routing queries from natural PT/EN phrasing, not only explicit mode words."""
        normalized = cls._normalize_query(message_lower)
        transport_patterns = [
            r"\bmetro\b",
            r"\bbus\b",
            r"\btrain\b",
            r"\bcarris\b",
            r"\bcomboio\b",
            r"\bautocarro\b",
            r"\broute\b",
            r"\brota\b",
            r"\btransporte\b",
            r"\btransport\b",
            r"\bferry\b",
            r"\bbarco\b",
            r"\bfertagus\b",
            r"\bcp\b",
            r"\bfrequencia\b",
            r"\bfrequency\b",
            r"\bheadway\b",
            r"\bintervalo\b",
            r"\bde quanto em quanto\b",
            r"\bhow often\b",
            r"\bhow long\b.*\bwait\b",
            r"\bwait\b.*\b(?:metro|line|station|platform)\b",
            r"\b(?:red|green|yellow|blue)\s+line\b",
            r"\blinha\s+(?:vermelha|verde|amarela|azul)\b",
            r"\bhow to get\b",
            r"\bhow do i get\b",
            r"\bhow can i get\b",
            r"\bget from\b",
            r"\bgo from\b",
            r"\bcomo chego\b",
            r"\bcomo vou\b",
            r"\bcomo posso ir\b",
            r"\bcomo ir\b",
            r"\ba partir do\b",
            r"\ba partir da\b",
            r"\bfrom\s+.+\s+to\s+.+",
        ]
        if any(re.search(pattern, normalized) for pattern in transport_patterns):
            return True

        return cls._contains_domain_keyword(
            normalized,
            [
                "metro",
                "autocarro",
                "comboio",
                "train",
                "bus",
                "carris",
                "transport",
                "transporte",
                "route",
                "rota",
                "station",
                "paragem",
                "tram",
                "departure",
                "departures",
                "frequency",
                "frequencia",
            ],
            minimum_ratio=0.85,
        )

    @classmethod
    def _is_obvious_out_of_scope(cls, user_message: str) -> bool:
        """Detects clearly out-of-scope trivia, math, coding, or translation queries."""
        message_lower = cls._normalize_query(user_message)
        if cls._has_lisbon_context(message_lower):
            return False

        out_of_scope_patterns = [
            r"\b\d+\s*[-+*/x]\s*\d+\b",
            r"\bcapital of\b",
            r"\bpresident of\b",
            r"\bwho won\b",
            r"\bhow do you say\b",
            r"\btranslate\b",
            r"\bcomo se diz\b",
            r"\btradu[zç][aã]o\b",
            r"\bwrite code\b",
            r"\bcoding\b",
            r"\bprogramming\b",
            r"\bpython\b",
            r"\bjavascript\b",
            r"\bsql\b",
            r"\bmandarim\b",
            r"\bjapan\b",
            r"\bjap[aã]o\b",
        ]
        return any(re.search(pattern, message_lower) for pattern in out_of_scope_patterns)

    @classmethod
    def _is_geographic_out_of_scope_route(cls, user_message: str) -> bool:
        """Detects point-to-point route requests wholly outside the Lisbon/AML scope."""
        normalized = cls._normalize_query(user_message)
        if not cls._looks_like_transport_query(normalized):
            return False
        outside_places = [
            "madrid",
            "barcelona",
            "seville",
            "sevilla",
            "paris",
            "london",
            "porto",
            "coimbra",
            "faro",
        ]
        mentioned = [place for place in outside_places if re.search(rf"\b{re.escape(place)}\b", normalized)]
        return len(mentioned) >= 2 and not cls._has_lisbon_context(normalized)

    @classmethod
    def _is_unsupported_action_request(cls, user_message: str) -> bool:
        """Detects transactional actions that LISBOA can explain but cannot perform."""
        normalized = cls._normalize_query(user_message)
        unsupported_patterns = [
            r"\b(book|reserve)\s+(?:me\s+)?(?:a\s+)?(?:table|restaurant|ticket|tickets|seat|seats|hotel|room)\b",
            r"\b(can|could|please)?\s*(?:you\s+)?(?:help\s+me\s+)?(?:book|reserve|buy|purchase)\s+(?:(?:me\s+)?(?:a|an|one)\s+)?(?:table|restaurant|ticket|tickets|seat|seats|hotel|room|flight|pass|passes)\b",
            r"\b(make|booking|do|doing)\s+(?:me\s+)?(?:a\s+)?(?:reservation|booking)\b",
            r"\bbuy\s+(?:me\s+)?(?:a\s+)?(?:ticket|tickets)\b",
            r"\breservar\s+(?:uma\s+)?(?:mesa|bilhetes?|hotel|quarto)\b",
            r"\bmarcar\s+(?:uma\s+)?(?:mesa|reserva|bilhetes?|hotel|quarto)\b",
            r"\bfazer\s+(?:uma\s+)?reserva\b",
            r"\bcomprar\s+(?:bilhetes?|entradas?)\s+(?:por mim|para mim)\b",
        ]
        return any(re.search(pattern, normalized) for pattern in unsupported_patterns)

    @staticmethod
    def _build_unsupported_action_response(language: str) -> str:
        """Builds a concise limitation response for unsupported booking actions."""
        if language == "pt":
            return (
                "### ⚠️ **Pedidos de reserva / compra**\n\n"
                "Não consigo fazer reservas, compras ou marcações diretamente.\n\n"
                "- ✅ **O que posso confirmar:** dados públicos sobre o local em Lisboa, contactos ou fontes oficiais quando estiverem disponíveis.\n"
                "- 🚫 **Não posso inventar:** disponibilidade de mesa/cadeira, preços atuais ou confirmação de reserva."
            )
        return (
            "### ⚠️ **Booking and purchase request**\n\n"
            "I can't make bookings, purchases, or reservations directly.\n\n"
            "- ✅ **What I can confirm:** public details about the Lisbon venue, contacts, or official sources when available.\n"
            "- 🚫 **I cannot invent:** table/seat availability, current prices, opening status, or booking confirmation."
        )

    @staticmethod
    def _build_greeting_response(language: str) -> str:
        """Builds a lightweight greeting response."""
        if language == "en":
            return (
                "Hello! 👋 I'm your Lisbon Urban Assistant. "
                "How can I help you today?"
            )
        return (
            "Olá! 👋 Sou o teu Assistente Urbano de Lisboa. "
            "Em que te posso ajudar hoje?"
        )

    @staticmethod
    def _build_out_of_scope_response(language: str) -> str:
        """Builds a friendly out-of-scope redirection response."""
        if language == "en":
            return (
                "Oops, that's outside my area of expertise! 😄 "
                "I'm your **Lisbon Urban Assistant** and I focus on the Lisbon Metropolitan Area 🏙️\n\n"
                "Here's what I can help you with:\n\n"
                "- 🌤️ Weather forecasts and real-time warnings\n"
                "- 🚇 Transport information (Metro, buses, trains, trams)\n"
                "- 🎭 Cultural events and activities\n"
                "- 📍 Places to visit, restaurants, and attractions\n"
                "- 🗺️ Custom itinerary planning\n"
                "- 🏥 Nearby services such as pharmacies and hospitals\n"
                "- 📚 Lisbon history and culture\n\n"
                "Ask me anything about Lisbon and I'll jump in. 🧭"
            )
        return (
            "Ups, isso fica fora da minha especialidade! 😄 "
            "Sou o teu **Assistente Urbano de Lisboa** e foco-me na Área Metropolitana de Lisboa 🏙️\n\n"
            "Posso ajudar-te com:\n\n"
            "- 🌤️ Previsões meteorológicas e avisos em tempo real\n"
            "- 🚇 Informação de transportes (Metro, autocarros, comboios, elétricos)\n"
            "- 🎭 Eventos culturais e atividades\n"
            "- 📍 Locais para visitar, restaurantes e atrações\n"
            "- 🗺️ Planeamento de itinerários à medida\n"
            "- 🏥 Serviços próximos, como farmácias e hospitais\n"
            "- 📚 História e cultura de Lisboa\n\n"
            "Pergunta-me o que quiseres sobre Lisboa e eu trato disso. 🧭"
        )

    def _direct_routing_override(self, user_message: str, language: str) -> Optional[Dict[str, Any]]:
        """Handles trivial direct responses before invoking the supervisor LLM."""
        if self._is_greeting_only(user_message):
            return {
                "reasoning": "Direct greeting override",
                "agents": [],
                "direct_response": self._sanitize_direct_response(self._build_greeting_response(language)),
            }

        if self._is_unsupported_action_request(user_message):
            return {
                "reasoning": "Direct unsupported transactional action override",
                "agents": [],
                "direct_response": self._sanitize_direct_response(self._build_unsupported_action_response(language)),
            }

        if self._is_geographic_out_of_scope_route(user_message):
            return {
                "reasoning": "Direct geographic out-of-scope route override",
                "agents": [],
                "direct_response": self._sanitize_direct_response(self._build_out_of_scope_response(language)),
            }

        if self._is_obvious_out_of_scope(user_message):
            return {
                "reasoning": "Direct out-of-scope override",
                "agents": [],
                "direct_response": self._sanitize_direct_response(self._build_out_of_scope_response(language)),
            }

        return None

    @staticmethod
    def _sanitize_direct_response(text: Optional[str]) -> Optional[str]:
        """Removes unsupported closing offers from direct supervisor responses."""
        if not text:
            return text
        return strip_unsupported_closing_offers(text).strip()

    @classmethod
    def _looks_like_follow_up(cls, user_message: str) -> bool:
        """Detects short anaphoric follow-ups that truly need previous-turn context."""
        normalized = cls._normalize_query(user_message)
        if not normalized:
            return False

        follow_up_prefixes = [
            "e ",
            "and ",
            "what about",
            "how about",
            "same ",
            "also ",
            "agora ",
            "tomorrow",
            "amanha",
            "amanhã",
        ]
        if any(normalized.startswith(prefix) for prefix in follow_up_prefixes):
            return True

        tokens = normalized.split()
        anaphoric_tokens = {"it", "that", "those", "there", "same", "isso", "isto", "essa", "esse", "aquela", "aquele"}
        return len(tokens) <= 6 and any(token in anaphoric_tokens for token in tokens)

    @classmethod
    def _single_domain_override(cls, user_message: str) -> Optional[Dict[str, Any]]:
        """Routes obvious standalone single-domain queries without letting history or the LLM over-expand them."""
        if cls._looks_like_follow_up(user_message):
            return None
        direct_weather_transport = cls._is_direct_weather_transport_query(user_message)
        message_lower = cls._normalize_query(user_message)
        if re.search(r"\b(uber|bolt|taxi|taxis|tÃ¡xi|tÃ¡xis|ride-hailing)\b", message_lower):
            return {
                "reasoning": "Direct unsupported ride-hailing transport override",
                "agents": ["transport"],
                "direct_response": None,
            }
        if cls._is_weather_only_outdoor_decision_query(user_message):
            return {
                "reasoning": "Direct weather advice override for outdoor activity",
                "agents": ["weather"],
                "direct_response": None,
            }

        # A single recommendation for one museum/monument in a constrained time
        # window is not an itinerary. Route it to Researcher so the after-hours
        # availability guard can answer conservatively instead of publishing a
        # generic planner skeleton.
        if (
            re.search(r"\b(?:recomendas?|recommend|suggest|qual|which)\b", message_lower)
            and re.search(r"\b(?:museu|museus|museum|museums|monumento|monument)\b", message_lower)
            and re.search(r"\b(?:\d{1,2}\s*(?:h|:|às|as)\s*\d{0,2}|domingo|sunday|tonight|esta noite|evening|fim do dia)\b", message_lower)
            and not re.search(r"\b(?:plano|plan|itinerary|roteiro|agenda|2 dias|dois dias|day plan)\b", message_lower)
        ):
            return {
                "reasoning": "Direct single place recommendation with time-window override",
                "agents": ["researcher"],
                "direct_response": None,
            }

        if cls._is_planning_query(user_message) and not direct_weather_transport:
            return None

        weather_hit = cls._looks_like_weather_query(message_lower)
        transport_terms = [
            "metro", "bus", "autocarro", "comboio", "train", "carris",
            "route", "rota", "station", "estação", "paragem", "departures", "wait time",
        ]
        event_terms = [
            "event", "events", "evento", "eventos", "concert", "concerto",
            "festival", "exhibition", "exposição", "exposicao", "music", "música", "musica",
            "what's on", "o que há", "o que ha",
        ]
        place_terms = [
            "attraction", "attractions", "atração", "atrações", "atracao", "atracoes",
            "museum", "museu", "monument", "monumento", "miradouro", "places", "locais",
            "restaurant", "restaurante", "what to visit", "o que visitar",
        ]
        service_terms = [
            "pharmacy", "farmácia", "farmacia", "hospital", "school", "escola",
            "library", "biblioteca", "police", "polícia", "policia",
        ]

        transport_hit = cls._looks_like_transport_query(message_lower) or cls._contains_domain_keyword(
            message_lower,
            transport_terms,
            minimum_ratio=0.85,
        )
        exact_event_hit = any(
            re.search(pattern, message_lower)
            for pattern in (
                r"\bevents?\b",
                r"\beventos?\b",
                r"\bconcerts?\b",
                r"\bconcertos?\b",
                r"\bfestivals?\b",
                r"\bexhibitions?\b",
                r"\bexposi[cç][aã]o\b",
                r"\bexposi[cç][oõ]es\b",
                r"\bwhat's on\b",
                r"\bo que h[aá]\b",
            )
        )
        event_hit = exact_event_hit or cls._contains_domain_keyword(message_lower, event_terms, minimum_ratio=0.84)
        if weather_hit and event_hit and not re.search(
            r"\b(?:weather|forecast|rain|temperature|wind|umbrella|tempo|meteo|previs[aã]o|chuva|temperatura|vento|guarda[-\s]?chuva)\b",
            message_lower,
        ):
            weather_hit = False
        if weather_hit and re.search(r"\b(?:wind|vento)\b", message_lower) and not exact_event_hit:
            event_hit = False
        exact_place_hit = any(
            re.search(pattern, message_lower)
            for pattern in (
                r"\battractions?\b",
                r"\batra[cç][oõ]es?\b",
                r"\batra[cç][aã]o\b",
                r"\bmuseums?\b",
                r"\bmuseus?\b",
                r"\bmonuments?\b",
                r"\bmonumentos?\b",
                r"\bmiradouro\b",
                r"\bplaces\b",
                r"\blocais\b",
                r"\brestaurants?\b",
                r"\brestaurantes?\b",
                r"\bwhat to visit\b",
                r"\bo que visitar\b",
            )
        )
        place_hit = exact_place_hit or cls._contains_domain_keyword(message_lower, place_terms, minimum_ratio=0.82)
        if transport_hit and not exact_place_hit:
            place_hit = False
        service_hit = cls._contains_domain_keyword(message_lower, service_terms, minimum_ratio=0.84)

        if weather_hit and not any([transport_hit, event_hit, place_hit, service_hit]):
            return {
                "reasoning": "Direct standalone weather override",
                "agents": ["weather"],
                "direct_response": None,
            }

        if weather_hit and event_hit and not any([transport_hit, place_hit, service_hit]):
            return {
                "reasoning": "Direct weather-aware event override",
                "agents": ["weather", "researcher"],
                "direct_response": None,
            }

        if weather_hit and transport_hit and not any([event_hit, service_hit]):
            agents = ["weather", "transport"]
            if cls._direct_weather_transport_query_needs_local_context(message_lower):
                agents.append("researcher")
            return {
                "reasoning": "Direct weather and transport override",
                "agents": agents,
                "direct_response": None,
            }

        if transport_hit and not any([weather_hit, event_hit, place_hit, service_hit]):
            return {
                "reasoning": "Direct standalone transport override",
                "agents": ["transport"],
                "direct_response": None,
            }

        if (event_hit or place_hit or service_hit) and not any([weather_hit, transport_hit]):
            return {
                "reasoning": "Direct standalone researcher override",
                "agents": ["researcher"],
                "direct_response": None,
            }

        return None

    @classmethod
    def _follow_up_domain_override(
        cls,
        user_message: str,
        conversation_history: Optional[List],
    ) -> Optional[Dict[str, Any]]:
        """Routes short follow-ups by reusing the current or previous domain safely."""
        if not conversation_history or not cls._looks_like_follow_up(user_message):
            return None

        event_terms = [
            "event", "events", "evento", "eventos", "concert", "concerto",
            "festival", "exhibition", "exposição", "exposicao", "music", "música",
            "musica", "what's on", "o que há", "o que ha",
        ]
        place_terms = [
            "attraction", "attractions", "atração", "atrações", "atracao", "atracoes",
            "museum", "museu", "monument", "monumento", "miradouro", "places", "locais",
            "restaurant", "restaurante", "what to visit", "o que visitar",
        ]
        service_terms = [
            "pharmacy", "farmácia", "farmacia", "hospital", "school", "escola",
            "library", "biblioteca", "police", "polícia", "policia",
        ]
        transport_terms = [
            "metro", "bus", "autocarro", "comboio", "train", "carris", "tram",
            "elétrico", "eletrico", "route", "rota", "station", "estação", "estacao",
            "stop", "paragem", "departure", "departures", "chegar", "get there",
        ]

        def infer_domain(text: str) -> Optional[str]:
            text_lower = cls._normalize_query(text)
            if cls._is_planning_query(text_lower):
                return "planner"
            if cls._looks_like_weather_query(text_lower):
                return "weather"
            if cls._looks_like_transport_query(text_lower) or cls._contains_domain_keyword(
                text_lower,
                transport_terms,
                minimum_ratio=0.85,
            ):
                return "transport"
            if cls._contains_domain_keyword(
                text_lower,
                event_terms + place_terms + service_terms,
                minimum_ratio=0.82,
            ):
                return "researcher"
            return None

        current_domain = infer_domain(user_message)
        if current_domain == "planner":
            return {
                "reasoning": "Follow-up domain override from current planning intent",
                "agents": ["planner", "researcher"],
                "direct_response": None,
            }
        if current_domain:
            return {
                "reasoning": f"Follow-up domain override from current query ({current_domain})",
                "agents": [current_domain],
                "direct_response": None,
            }

        last_user_message = None
        for msg in reversed(conversation_history):
            if isinstance(msg, HumanMessage) and msg.content:
                last_user_message = str(msg.content)
                break

        previous_domain = infer_domain(last_user_message or "")
        if previous_domain == "planner":
            return {
                "reasoning": "Follow-up domain override from previous planning query",
                "agents": ["planner", "researcher"],
                "direct_response": None,
            }
        if previous_domain:
            return {
                "reasoning": f"Follow-up domain override from previous user query ({previous_domain})",
                "agents": [previous_domain],
                "direct_response": None,
            }

        return None

    @classmethod
    def _is_planning_query(cls, user_message: str) -> bool:
        """Detects itinerary/planning intent without over-matching words like `today`."""
        message_lower = cls._normalize_query(user_message)
        if cls._is_category_browse_query(user_message):
            return False
        if cls._is_direct_weather_transport_query(user_message):
            return False
        planning_patterns = [
            r"\bplan\b",
            r"\bplan my day\b",
            r"\bday plan\b",
            r"\bitinerary\b",
            r"\broteiro\b",
            r"\bplano\b",
            r"\bagenda\b",
            r"\bschedule\b",
            r"\bday trip\b",
            r"\bpasseio\b",
            r"\bvisitar\b",
            r"\bvisitando\b",
            r"\bplane(?:ar|ia|ie)\b",
            r"\borganiza(?:r)?\b",
            r"\bir a vários? locais\b",
            r"\bvisit multiple\b",
            r"\bpasseio\b.*\b(?:\d+\s*h|\d+\s*horas?|percurso|coerente|alternativo|locais|sitios|s[ií]tios)\b",
            r"\b(?:percurso|rota)\s+(?:a pe|a p[eé]|pedonal|coerente|alternativ[oa])\b",
            r"\bwalk(?:ing)?\s+(?:route|around)\b",
            r"\bcoherent\s+walk\b",
            r"\b(?:one\s+)?cultural\s+stop\b",
            r"\bdinner\s+plus\b",
            r"\broute\b.*\b(?:history|historical|culture|cultural|pastry|custard|tart|pastel|pacing|stop)\b",
            r"\b(?:history|historical|culture|cultural|pastry|custard|tart|pastel|pacing)\b.*\broute\b",
            r"\b(?:this\s+)?afternoon\b.*\b(?:history|historical|pastry|custard|tart|pastel|stop)\b",
            r"\b\d+\s*(?:h|hours?|horas?)\b.*\b(?:walk|passeio|percurso|itinerary|roteiro)\b",
            r"\b\d+\s*minutes?\b.*\b(?:walk|cultural|dinner|stop)\b",
        ]
        if any(re.search(pattern, message_lower) for pattern in planning_patterns):
            return True

        return False

    @classmethod
    def _planning_query_has_origin_anchor(cls, user_message: str) -> bool:
        """Return whether a planning request names a starting point that affects feasibility."""
        normalized = cls._normalize_query(user_message)
        origin_patterns = [
            r"\bstarting from\b",
            r"\bstart(?:ing)? at\b",
            r"\barrive(?:s|d)? at\b",
            r"\bi arrive at\b",
            r"\bfrom\b.+\bto\b",
            r"\ba partir de\b",
            r"\bdesde\b",
            r"\bestou em\b",
            r"\bsaindo de\b",
            r"\bpartindo de\b",
            r"\bcome[cç]ar em\b",
            r"\bcome[cç]ando em\b",
        ]
        return any(re.search(pattern, normalized) for pattern in origin_patterns)

    @classmethod
    def _is_direct_weather_transport_query(cls, user_message: str) -> bool:
        """Detects operational weather-plus-route requests that should not become itineraries."""
        normalized = cls._normalize_query(user_message)
        if not normalized:
            return False

        full_planning_markers = [
            r"\bfull\s+(?:day|itinerary)\b",
            r"\bplan\s+(?:a|my|the)?\s*(?:full\s+)?(?:day|itinerary)\b",
            r"\broute\b.*\b(?:history|historical|culture|cultural|pastry|custard|tart|pastel|pacing|stop)\b",
            r"\b(?:history|historical|culture|cultural|pastry|custard|tart|pastel|pacing)\b.*\broute\b",
            r"\bplan\s+(?:a\s+)?(?:single|one|relaxed|quiet|calm)\b.*\bday\b",
            r"\bplan\b.*\b(?:museum|museu)\b.*\b(?:garden|jardim)\b",
            r"\bplan\s+\d+\s*(?:day|days)\b",
            r"\b\d+\s*(?:day|days)\b",
            r"\bplane(?:ar|ia|ie)\s+\d+\s*dias\b",
            r"\b\d+\s*dias\b",
            r"\bitinerary\b",
            r"\broteiro\b",
            r"\bplano\s+(?:do|de)?\s*dia\b",
            r"\b(?:varios|vários|multiple)\s+(?:locais|places|stops)\b",
        ]
        if any(re.search(pattern, normalized) for pattern in full_planning_markers):
            return False

        route_markers = [
            r"\bhow\s+(?:do|can)\s+i\s+(?:get|go)\b.*\b(?:from|to)\b",
            r"\bcomo\s+(?:vou|chego|posso ir)\b.*\b(?:de|do|da|para)\b",
            r"\b(?:from|de|do|da)\s+.+\b(?:to|para)\s+.+\b(?:public transport|transportes publicos|transportes públicos)",
            r"\b(?:public transport|transportes publicos|transportes públicos)\b",
        ]
        has_route_request = any(re.search(pattern, normalized) for pattern in route_markers)
        return cls._looks_like_weather_query(normalized) and cls._looks_like_transport_query(normalized) and has_route_request

    @classmethod
    def _is_weather_only_outdoor_decision_query(cls, user_message: str) -> bool:
        """Detect weather advice for an outdoor activity without a requested transport leg."""
        normalized = cls._normalize_query(user_message)
        if not normalized or not cls._looks_like_weather_query(normalized):
            return False
        if cls._is_planning_query(user_message):
            return False
        if cls._is_direct_weather_transport_query(user_message):
            return False
        if re.search(
            r"\b(?:event|events|evento|eventos|concert|concerts|concerto|concertos|festival|festivals|exhibition|exhibitions|exposi[cç][aã]o|exposi[cç][oõ]es)\b",
            normalized,
        ):
            return False
        outdoor_activity = re.search(
            r"\b(?:walk|walking|caminhar|passeio|outdoor|ar livre|viewpoint|miradouro|cycling|bicicleta)\b",
            normalized,
        )
        if not outdoor_activity:
            return False
        explicit_route = re.search(
            r"\b(?:how\s+(?:do|can)\s+i\s+(?:get|go)|como\s+(?:vou|chego|posso ir)|public transport|transportes publicos|transportes públicos|metro|bus|autocarro|comboio|train|tram|el[eé]trico)\b",
            normalized,
        )
        return not bool(explicit_route)

    @staticmethod
    def _direct_weather_transport_query_needs_local_context(normalized_message: str) -> bool:
        """Returns whether a direct weather-route request also asks for visit context."""
        return bool(
            re.search(
                r"\b(?:what\s+to\s+visit|where\s+to\s+visit|places?\s+to\s+visit|recommend|suggest|"
                r"visit(?:ing)?\s+(?:belem|bel[eé]m|a\s+place|an\s+area|a\s+neighbourhood|a\s+neighborhood)|"
                r"visitar\s+(?:belem|bel[eé]m|um\s+local|uma\s+zona|um\s+bairro)|"
                r"o\s+que\s+visitar|onde\s+visitar|locais?\s+para\s+visitar|recomenda|sugere|sugeres)\b",
                normalized_message,
            )
        )

    @classmethod
    def _is_category_browse_query(cls, user_message: str) -> bool:
        """Detects category-browsing questions that should not become itinerary plans."""
        normalized = cls._normalize_query(user_message)
        category_patterns = [
            r"\bwhat kinds? of\b.*\b(?:places?|events?|public services?|services?)\b",
            r"\bwhat types? of\b.*\b(?:places?|events?|public services?|services?)\b",
            r"\b(?:places?|events?|public services?|services?)\b.*\b(?:can i explore|can i look for|can you help me find|available categories|categories)\b",
            r"\bque tipos? de\b.*\b(?:locais|eventos|servi[cç]os)\b",
            r"\b(?:locais|eventos|servi[cç]os)\b.*\b(?:posso procurar|posso explorar|categorias disponiveis|categorias)\b",
        ]
        return any(re.search(pattern, normalized) for pattern in category_patterns)

    @staticmethod
    def _planning_query_mentions_weather(user_message: str) -> bool:
        """Detects explicit weather references inside itinerary/planning requests."""
        message_lower = (user_message or "").lower()
        weather_hints = [
            "weather",
            "forecast",
            "rain",
            "chuva",
            "previsão",
            "previsao",
            "consider the weather",
            "considera o tempo",
            "considera a meteorologia",
            "com o tempo",
        ]
        return any(hint in message_lower for hint in weather_hints)

    @traceable(name="supervisor_agent", run_type="chain", tags=["sub-agent", "supervisor"])
    def route(
        self,
        user_message: str,
        language: str = "en",
        conversation_history: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Analyzes user message and returns routing decision.

        Args:
            user_message: The user's query.
            language: Language code ('en' or 'pt').
            conversation_history: Recent conversation messages for follow-up context.

        Returns:
            Dict with:
                - reasoning: Why these agents were chosen
                - agents: List of agent names to call (can be empty)
                - direct_response: Response if no agents needed (or None)
        """
        direct_override = self._direct_routing_override(user_message, language)
        if direct_override:
            return direct_override

        single_domain_override = self._single_domain_override(user_message)
        if single_domain_override:
            return single_domain_override

        follow_up_override = self._follow_up_domain_override(user_message, conversation_history)
        if follow_up_override:
            return follow_up_override

        system_prompt = get_supervisor_prompt(language)

        messages = [SystemMessage(content=system_prompt)]

        # Inject minimal follow-up context (NOT raw messages - that confuses routing)
        if conversation_history and self._looks_like_follow_up(user_message):
            # Extract ONLY the last user query for follow-up detection
            last_user_queries = []
            for msg in reversed(conversation_history):
                if isinstance(msg, HumanMessage) and msg.content:
                    last_user_queries.append(msg.content[:150])
                    if len(last_user_queries) >= 2:
                        break
            last_user_queries.reverse()

            if last_user_queries:
                context_note = (
                    "FOLLOW-UP CONTEXT (for reference ONLY - do NOT add extra agents because of this):\n"
                    f"Previous user question(s): {' | '.join(last_user_queries)}\n"
                    "Use this ONLY to understand references like 'E amanhã?', 'E de autocarro?' etc. "
                    "Route the CURRENT query based on its OWN content. Do NOT add agents from previous topics."
                )
                messages.append(SystemMessage(content=context_note))

        messages.append(HumanMessage(content=user_message))

        # Get routing decision from LLM (with retry for Azure content filter)
        # If the model path is temporarily unavailable (rate-limit, connection,
        # provider error), fallback to deterministic routing to keep user flow.
        try:
            response = self._safe_llm_invoke(self.llm, messages)
            content = clean_response(response.content, _print=False)
        except Exception as exc:
            fallback_decision = self._fallback_routing(
                user_message=user_message,
                language=language,
            )
            fallback_decision["reasoning"] = (
                f"Fallback routing (supervisor model unavailable: {type(exc).__name__})"
            )
            return fallback_decision

        # Parse JSON response
        decision = parse_json_response(content)

        if decision:
            agents = decision.get("agents", [])
            reasoning = decision.get("reasoning", "")
            message_lower = user_message.lower()

            # Check if this is a planning query that requires weather.
            is_planning_query = self._is_planning_query(user_message)
            if re.search(r"\b(?:plan|itinerary|roteiro|planeia|planejar)\b", user_message, flags=re.IGNORECASE) and re.search(
                r"\b(?:[2-9]\s*(?:days?|dias?)|seven days|five days|7 days|5 days|weekend|fim de semana)\b",
                user_message,
                flags=re.IGNORECASE,
            ):
                is_planning_query = True
                agents = [agent for agent in ["weather", "transport", "researcher", "planner", *agents] if agent]
                agents = [agent for index, agent in enumerate(agents) if agent not in agents[:index]]
                reasoning += " (Deterministic override: multi-day planning requires full planning route)"

            weather_only_outdoor_decision = self._is_weather_only_outdoor_decision_query(user_message)
            if weather_only_outdoor_decision:
                agents = ["weather"]
                reasoning += " (Reduced to weather: no transport leg requested)"
                decision["agents"] = agents

            if is_planning_query and not weather_only_outdoor_decision:
                if "planner" not in agents:
                    agents.append("planner")
                    reasoning += " (Added planner agent: itinerary/planning query)"

                if "researcher" not in agents:
                    agents.append("researcher")
                    reasoning += " (Added researcher agent: planning needs place/activity grounding)"

                if (
                    self._looks_like_transport_query(message_lower)
                    or self._planning_query_has_origin_anchor(user_message)
                ) and "transport" not in agents:
                    agents.append("transport")
                    reasoning += " (Added transport agent: planning query includes route/origin feasibility)"

            # Force weather agent for near-future planning
            if is_planning_query and (
                self._requires_weather_for_planning(user_message)
                or self._planning_query_mentions_weather(user_message)
            ):
                if "weather" not in agents:
                    agents.append("weather")
                    reasoning += " (Added weather agent: planning for near-future date)"

            # Enforce rejection for out of scope queries even if LLM tries to answer
            reasoning_lower = reasoning.lower()
            if not agents and any(k in reasoning_lower for k in ["matemática", "math", "fora de âmbito", "out of scope", "trivia", "trivialidade"]):
                # Only override if LLM didn't provide a direct_response
                if not decision.get("direct_response"):
                    if language == "pt":
                        decision["direct_response"] = (
                            "Ups, isso fica um pouco fora da minha especialidade! 😄 "
                            "Sou o teu **Assistente Urbano de Lisboa** e estou aqui para te ajudar "
                            "a aproveitar ao máximo a Área Metropolitana de Lisboa 🏙️\n\n"
                            "Olha o que posso fazer por ti:\n\n"
                            "- 🌤️ Previsões meteorológicas e avisos em tempo real\n"
                            "- 🚇 Informação de transportes (Metro, autocarros, comboios, elétricos)\n"
                            "- 🎭 Eventos culturais e atividades\n"
                            "- 📍 Locais para visitar, restaurantes e atrações\n"
                            "- 🗺️ Planeamento de itinerários à medida\n"
                            "- 🏥 Serviços próximos (farmácias, hospitais, parques)\n"
                            "- 📚 História e cultura de Lisboa\n\n"
                            "Pergunta-me o que quiseres sobre Lisboa! 🧭"
                        )
                    else:
                        decision["direct_response"] = (
                            "Oops, that's a bit outside my expertise! 😄 "
                            "I'm your **Lisbon Urban Assistant** and I'm here to help you "
                            "make the most of the Lisbon Metropolitan Area 🏙️\n\n"
                            "Here's what I can do for you:\n\n"
                            "- 🌤️ Weather forecasts & real-time warnings\n"
                            "- 🚇 Transport info (Metro, buses, trains, trams)\n"
                            "- 🎭 Cultural events & activities\n"
                            "- 📍 Places to visit, restaurants & attractions\n"
                            "- 🗺️ Custom itinerary planning\n"
                            "- 🏥 Nearby services (pharmacies, hospitals, parks)\n"
                            "- 📚 Lisbon history & culture\n\n"
                            "Go ahead, ask me anything about Lisbon! 🧭"
                        )

            return {
                "reasoning": reasoning,
                "agents": agents,
                "direct_response": self._sanitize_direct_response(decision.get("direct_response")),
            }

        # Fallback: If JSON parsing fails, try to extract intent heuristically
        return self._fallback_routing(user_message, language)

    def _fallback_routing(self, user_message: str, language: str = "pt") -> Dict[str, Any]:
        """
        Fallback routing when JSON parsing fails.
        Uses simple keyword matching as backup.

        Args:
            user_message: Original user query.
            language: User language code ("pt" or "en"). Defaults to "pt".

        Returns:
            Dict with routing decision.
        """
        direct_override = self._direct_routing_override(user_message, language)
        if direct_override:
            return direct_override

        message_lower = user_message.lower()

        # 1. Check for Out-of-Scope keywords (Locations outside AML)
        # Note: AML includes Sintra, Cascais, Montijo, Setúbal, etc. - these are IN SCOPE!
        # CRITICAL: Use word boundaries to avoid false positives
        # e.g. "porto" must NOT match "aeroporto", "transporte", etc.
        forbidden_patterns = [
            r"\bporto\b",
            r"\baveiro\b",
            r"\bbraga\b",
            r"\bcoimbra\b",
            r"\bfaro\b",
            r"\balgarve\b",
            r"\bévora\b",
            r"\bmadrid\b",
            r"\bparis\b",
            r"\blondon\b",
            r"\bbarcelona\b",
            r"\bnew york\b",
            r"\btokyo\b",
            r"\broma\b",
            r"\brome\b",
        ]
        if any(re.search(pat, message_lower) for pat in forbidden_patterns):
            if language == "en":
                oos_msg = (
                    "That's a bit outside my area! 😊 "
                    "I'm your guide for the **Lisbon Metropolitan Area** 🏙️\n\n"
                    "But here's everything I can help you with:\n\n"
                    "- 🌤️ Weather forecasts and warnings\n"
                    "- 🚇 Real-time transport (Metro, buses, trains)\n"
                    "- 🎭 Cultural events and activities\n"
                    "- 📍 Places, museums and attractions\n"
                    "- 🗺️ Personalized itinerary planning\n"
                    "- 🏥 Essential services (pharmacies, hospitals, schools)\n\n"
                    "Want to explore Lisbon? Just ask! 🧭"
                )
            else:
                oos_msg = (
                    "Isso fica um pouco fora da minha área! 😊 "
                    "Sou o teu guia para a **Área Metropolitana de Lisboa** 🏙️\n\n"
                    "Mas olha tudo o que te posso ajudar:\n\n"
                    "- 🌤️ Previsão meteorológica e avisos\n"
                    "- 🚇 Transportes em tempo real (Metro, autocarros, comboios)\n"
                    "- 🎭 Eventos e atividades culturais\n"
                    "- 📍 Locais, museus e atrações\n"
                    "- 🗺️ Planeamento personalizado de itinerários\n"
                    "- 🏥 Serviços essenciais (farmácias, hospitais, escolas)\n\n"
                    "Queres explorar Lisboa? Pergunta-me! 🧭"
                )
            return {
                "reasoning": "Fallback: Detected out-of-scope location (outside AML)",
                "agents": [],
                "direct_response": self._sanitize_direct_response(oos_msg),
            }

        # 2. AML locations that ARE in scope - should trigger transport agent
        aml_keywords = [
            "sintra",
            "cascais",
            "oeiras",
            "amadora",
            "loures",
            "odivelas",
            "almada",
            "seixal",
            "barreiro",
            "montijo",
            "alcochete",
            "setúbal",
            "palmela",
            "sesimbra",
            "mafra",
            "vila franca",
        ]
        if any(loc in message_lower for loc in aml_keywords):
            # These are AML locations - use transport agent
            return {
                "reasoning": "Fallback: AML location detected - using transport agent",
                "agents": ["transport"],
                "direct_response": None,
            }

        # Weather keywords
        # Transport keywords
        transport_keywords = [
            "metro",
            "bus",
            "train",
            "carris",
            "comboio",
            "autocarro",
            "route",
            "rota",
            "como chego",
            "how to get",
            "transporte",
            "ferry",
            "barco",
            "fertagus",
            "cp",
            "frequência",
            "frequency",
            "headway",
            "intervalo",
            "de quanto em quanto",
            "how often",
        ]

        # Places/Events keywords
        places_keywords = [
            "museum",
            "restaurant",
            "park",
            "museu",
            "restaurante",
            "parque",
            "visit",
            "visitar",
            "event",
            "evento",
            "attraction",
            "atração",
            "what to do",
            "o que fazer",
            "places",
        ]

        # Resident services keywords (always → researcher)
        service_keywords = [
            "farmácia",
            "pharmacy",
            "hospital",
            "escola",
            "school",
            "biblioteca",
            "library",
            "bombeiros",
            "fire",
            "polícia",
            "police",
            "junta",
            "embaixada",
            "embassy",
            "cemitério",
            "wc",
            "sanitário",
            "toilet",
            "mercado",
            "market",
            "piscina",
            "desporto",
            "sports",
            "jardim",
            "garden",
            "creche",
            "estacionamento",
            "parking",
            "serviço",
            "service",
        ]

        agents = []

        # Check for keywords
        if self._looks_like_weather_query(message_lower):
            agents.append("weather")
        if self._looks_like_transport_query(message_lower) or any(kw in message_lower for kw in transport_keywords):
            agents.append("transport")
        if any(kw in message_lower for kw in places_keywords):
            agents.append("researcher")
        if any(kw in message_lower for kw in service_keywords):
            if "researcher" not in agents:
                agents.append("researcher")
        if self._is_planning_query(message_lower):
            # Itinerary queries should be grounded consistently across providers.
            if (
                self._requires_weather_for_planning(user_message)
                or self._planning_query_mentions_weather(user_message)
            ) and "weather" not in agents:
                agents.append("weather")
            if (
                self._looks_like_transport_query(message_lower)
                or self._planning_query_has_origin_anchor(user_message)
            ) and "transport" not in agents:
                agents.append("transport")
            if "researcher" not in agents:
                agents.append("researcher")
            if "planner" not in agents:
                agents.append("planner")

        # If still no agents and not a greeting, default to researcher
        if not agents:
            agents = ["researcher"]

        return {
            "reasoning": f"Fallback routing based on keywords: {agents}",
            "agents": agents,
            "direct_response": None,
        }

    def _requires_weather_for_planning(self, user_message: str) -> bool:
        """
        Detects if user is asking to plan for today, tomorrow, this week, or next few days.
        Returns True if weather data should be mandatory for the planning.

        Args:
            user_message: The user's query.

        Returns:
            bool: True if weather should be required for planning.
        """
        message_lower = user_message.lower()

        # Patterns for immediate/near-future planning (requires weather)
        immediate_patterns = [
            # Today
            r"\bhoje\b",
            r"\btoday\b",
            r"\bfor today\b",
            r"\bpara hoje\b",
            r"\bthis\s+(?:morning|afternoon|evening)\b",
            r"\btonight\b",
            r"\besta\s+(?:manh[Ã£a]|tarde|noite)\b",
            r"\blogo\s+Ã \s+noite\b",
            r"\blogo\s+a\s+noite\b",
            # Tomorrow
            r"\bamanh[ãa]\b",
            r"\btomorrow\b",
            r"\bfor tomorrow\b",
            r"\bpara amanh[ãa]\b",
            # This week
            r"\besta semana\b",
            r"\bthis week\b",
            r"\bna semana\b",
            r"\bduring this week\b",
            # Next X days
            r"\bpr[óo]ximos?\s+\d+\s+dias?\b",
            r"\bnext\s+\d+\s+days?\b",
            r"\bpr[óo]ximos?\s+(?:dois|tr[êe]s|quatro|cinco|seis|sete)\s+dias?\b",
            r"\bnext\s+(?:two|three|four|five|six|seven)\s+days?\b",
            r"\b(?:dois|tr[êe]s|quatro|cinco|seis|sete)\s+dias?\b",
            r"\b(?:two|three|four|five|six|seven)\s+days?\b",
            # Day plans
            r"\bday\s+\d+\b",
            r"\b(\d+)[oa]?\s+dia\b",
            # Weekend
            r"\bweekend\b",
            r"\bfim de semana\b",
            r"\bpr[óo]ximo fim de semana\b",
            r"\bnext weekend\b",
            # Now/immediate
            r"\bagora\b",
            r"\bnow\b",
            r"\bcurrently\b",
            r"\batualmente\b",
        ]

        # Check if any immediate pattern matches
        for pattern in immediate_patterns:
            if re.search(pattern, message_lower):
                return True

        return False


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    import sys
    with suppress(AttributeError):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 Supervisor Agent Test\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")

    passed = 0
    failed = 0

    try:
        supervisor = SupervisorAgent()
        print(
            f"\n\033[1m✅ Supervisor initialized:\033[0m {supervisor.get_model_info()}"
        )

        # =================================================================
        # ORIGINAL ROUTING TESTS
        # =================================================================
        print("\n\033[1m📋 Section 1: General Routing\033[0m")
        print("-" * 50)

        test_queries = [
            "Hello!",
            "What's the weather like?",
            "How do I get to Belém?",
            "Recommend some museums",
            "Plan my day visiting museums and considering the weather",
            "Quanto é 2+2?",
            "Que sitios posso visitar no Porto?",
            "Como está o tempo em Aveiro?",
        ]

        for query in test_queries:
            print(f"\n\033[1m📝 Query:\033[0m {query}")
            decision = supervisor.route(query)
            print(f"   \033[1mAgents:\033[0m {decision['agents']}")
            print(f"   \033[1mReason:\033[0m {decision['reasoning']}")
            if decision["direct_response"]:
                print(f"   \033[1mDirect:\033[0m {decision['direct_response']}")

        # =================================================================
        # SERVICE KEYWORD ROUTING TESTS
        # =================================================================
        print("\n\033[1m📋 Section 2: Resident Service Routing\033[0m")
        print("-" * 50)

        service_tests = [
            ("Onde fica a farmácia mais próxima?", "researcher", "pharmacy → researcher"),
            ("Where is the nearest hospital?", "researcher", "hospital → researcher"),
            ("Há alguma biblioteca perto de mim?", "researcher", "library → researcher"),
            ("I need a police station nearby", "researcher", "police → researcher"),
            ("Onde posso estacionar perto do Rossio?", "researcher", "parking → researcher"),
            ("Preciso de encontrar uma escola para o meu filho", "researcher", "school → researcher"),
            ("Where is the nearest WC?", "researcher", "wc/toilet → researcher"),
            ("Quero encontrar um mercado", "researcher", "market → researcher"),
            ("Onde ficam os bombeiros?", "researcher", "fire station → researcher"),
            ("Há piscinas municipais abertas?", "researcher", "sports/pool → researcher"),
        ]

        for query, expected_agent, description in service_tests:
            decision = supervisor._fallback_routing(query)
            agents = decision["agents"]
            if expected_agent in agents:
                passed += 1
                print(f"  \033[1;32m✅ PASS\033[0m: {description}")
                print(f"      Query: {query}")
                print(f"      Agents: {agents}")
            else:
                failed += 1
                print(f"  \033[1;31m❌ FAIL\033[0m: {description}")
                print(f"      Query: {query}")
                print(f"      Expected '{expected_agent}' in {agents}")

        # =================================================================
        # FREQUENCY KEYWORD ROUTING TESTS
        # =================================================================
        print("\n\033[1m📋 Section 3: Frequency/Headway Routing\033[0m")
        print("-" * 50)

        frequency_tests = [
            ("How often does the metro come?", "transport", "frequency EN → transport"),
            ("De quanto em quanto tempo passa o 28E?", "transport", "frequency PT → transport"),
            ("What's the headway on the Sintra line?", "transport", "headway → transport"),
            ("Qual a frequência do comboio para Cascais?", "transport", "frequência → transport"),
            ("What's the interval between buses?", "transport", "intervalo → transport"),
        ]

        for query, expected_agent, description in frequency_tests:
            decision = supervisor._fallback_routing(query)
            agents = decision["agents"]
            if expected_agent in agents:
                passed += 1
                print(f"  \033[1;32m✅ PASS\033[0m: {description}")
                print(f"      Agents: {agents}")
            else:
                failed += 1
                print(f"  \033[1;31m❌ FAIL\033[0m: {description}")
                print(f"      Expected '{expected_agent}' in {agents}")

        # =================================================================
        # OUT-OF-SCOPE FALLBACK TESTS
        # =================================================================
        print("\n\033[1m📋 Section 4: Out-of-Scope Fallback\033[0m")
        print("-" * 50)

        oos_tests = [
            ("What is the capital of Japan?", "General OOS query"),
            ("Como se diz obrigado em mandarim?", "Language/trivia OOS"),
            ("Bom dia!", "Greeting"),
        ]

        for query, description in oos_tests:
            decision = supervisor._fallback_routing(query)
            agents = decision["agents"]
            print(f"  \033[1m📝\033[0m {description}: \"{query[:40]}\"")
            print(f"      Agents: {agents} | Direct: {'Yes' if decision['direct_response'] else 'No'}")

        # =================================================================
        # SUMMARY
        # =================================================================
        total = passed + failed
        print("\n" + "=" * 60)
        print("\033[1m📊 SUPERVISOR TEST SUMMARY\033[0m")
        print("=" * 60)
        print(f"\033[1;32m✅ Passed: {passed}/{total}\033[0m")
        if failed > 0:
            print(f"\033[1;31m❌ Failed: {failed}/{total}\033[0m")
        else:
            print("\033[1;32m🎉 ALL FALLBACK ROUTING TESTS PASSED!\033[0m")
        print("=" * 60)

        print("\n\033[1;32m✅ Supervisor agent working!\033[0m")

    except Exception as e:
        print(f"\n\033[1;31m❌ Error:\033[0m {e}")
        import traceback
        traceback.print_exc()
