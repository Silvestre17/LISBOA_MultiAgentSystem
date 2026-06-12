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
from agent.utils.geographic_scope import (
    build_geographic_out_of_scope_response,
    extract_aml_municipality_mentions,
    extract_outside_aml_mentions,
    route_mentions_outside_aml,
)
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
                if token[0] != keyword[0]:
                    continue
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

    @staticmethod
    def _is_non_informative_message(user_message: str) -> bool:
        """Return whether a message has no alphanumeric user intent."""
        raw = str(user_message or "").strip()
        return not raw or not any(char.isalnum() for char in raw)

    @staticmethod
    def _build_non_informative_message_response(language: str) -> str:
        """Build a render-safe prompt for empty or punctuation-only turns."""
        if language == "pt":
            return (
                "### 🤖 **Como posso ajudar**\n\n"
                "Olá! 👋 Em que te posso ajudar na Área Metropolitana de Lisboa?\n\n"
                "Aqui está o que posso ajudar na AML/Lisboa:\n\n"
                "- 🌤️ **Meteorologia** — previsões, avisos, dados IPMA\n"
                "- 🚌 **Transportes** — metro, autocarro, elétrico, comboio e estado em tempo real\n"
                "- 🏛️ **Cultura & Eventos** — museus, exposições, festivais, concertos\n"
                "- 📍 **Locais & Serviços** — restaurantes, farmácias, hospitais, estacionamento\n"
                "- 🗓️ **Planeamento** — itinerários personalizados e planos de dia\n"
                "- 📚 **História & Conhecimento** — história de Lisboa, bairros, Guia Lisboa Card"
            )
        return (
            "### 🤖 **How I Can Help**\n\n"
            "Hi! 👋 How can I help you in the Lisbon Metropolitan Area?\n\n"
            "Here is what I can help with in Lisbon/AML:\n\n"
            "- 🌤️ **Weather** — forecasts, warnings, and IPMA data\n"
            "- 🚌 **Transport** — metro, bus, tram, train, and real-time status\n"
            "- 🏛️ **Culture & Events** — museums, exhibitions, festivals, concerts\n"
            "- 📍 **Places & Services** — restaurants, pharmacies, hospitals, parking\n"
            "- 🗓️ **Planning** — personalized itineraries and day plans\n"
            "- 📚 **History & Knowledge** — Lisbon history, neighbourhoods, Lisboa Card guide"
        )

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
        explicit_weather_clothing_advice = bool(
            re.search(
                r"\b(?:what\s+(?:should|do)\s+i\s+wear|what\s+to\s+wear|"
                r"o\s+que\s+(?:devo|posso)\s+vestir|devo\s+levar|should\s+i\s+(?:take|bring|wear))\b",
                normalized,
                flags=re.IGNORECASE,
            )
        )
        outdoor_exposure_weather_advice = bool(
            re.search(
                r"\b(?:today|tomorrow|tonight|morning|afternoon|evening|hoje|amanha|"
                r"manha|tarde|noite)\b",
                normalized,
                flags=re.IGNORECASE,
            )
            and re.search(
                r"\b(?:outside|outdoor|queue|waiting|wait|stand|standing|fila|fora|"
                r"exterior|ar\s+livre|esperar|ficar|caminhar|walk|walking)\b",
                normalized,
                flags=re.IGNORECASE,
            )
        )
        acquisition_or_discovery_lookup = bool(
            re.search(
                r"\b(?:where|onde|find|procurar|available|disponiveis|disponivel|which|quais|"
                r"shops?|stores?|lojas?|shopping|tour|tours|visita(?:s)?\s+guiada(?:s)?)\b",
                normalized,
                flags=re.IGNORECASE,
            )
            and re.search(
                r"\b(?:roupa|clothes|clothing|casacos?|jackets?|vestuario|vestu[aá]rio|"
                r"umbrella|guarda[-\s]?chuva|raincoat|impermeavel|imperme[aá]vel|poncho|"
                r"sunscreen|protetor\s+solar|hat|chapeu|chap[eé]u|walking\s+tours?|"
                r"guided\s+tours?|visita(?:s)?\s+guiada(?:s)?)\b",
                normalized,
                flags=re.IGNORECASE,
            )
        )
        if acquisition_or_discovery_lookup and not explicit_weather_clothing_advice:
            return False
        if explicit_weather_clothing_advice or outdoor_exposure_weather_advice:
            return True
        if re.search(
            r"\b(?:lojas?|shops?|shopping|comprar|buy|store|stores)\b"
            r".{0,50}\b(?:roupa|clothes|clothing|casacos?|jackets?|vestuario|vestu[aá]rio)\b|"
            r"\b(?:roupa|clothes|clothing|casacos?|jackets?|vestuario|vestu[aá]rio)\b"
            r".{0,50}\b(?:lojas?|shops?|shopping|comprar|buy|store|stores)\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            return False
        weather_patterns = [
            r"\bweather\b",
            r"\brain\b",
            r"\btemperature\b",
            r"\bwind\b",
            r"\bforecast\b",
            r"\bgood\s+weather\b",
            r"\bsuitable\b",
            r"\bumbrella\b",
            r"\baverage\s+climate\b",
            r"\bclimatology\b",
            r"\bclimate\s+average\b",
            r"\bwhat\s+(?:should|do)\s+i\s+wear\b",
            r"\bwhat\s+to\s+wear\b",
            r"\bdress(?:ing)?\b",
            r"\bwear(?:ing)?\b",
            r"\bclothing\b",
            r"\bjacket\b",
            r"\bmeteo\b",
            r"\bchover\b",
            r"\bprevis[aã]o\b",
            r"\bchuva\b",
            r"\btemperatura\b",
            r"\bvento\b",
            r"\bsol\b",
            r"\bbom\s+tempo\b",
            r"\badequad[ao]\b",
            r"\bclima\s+medio\b",
            r"\bmedias?\s+(?:historicas?\s+)?(?:do\s+)?clima\b",
            r"\bo\s+que\s+(?:devo|posso)\s+vestir\b",
            r"\bdevo\s+levar\s+(?:casaco|guarda[-\s]?chuva)\b",
            r"\bvestir\b",
            r"\broupa\b",
            r"\bcasaco\b",
            r"\bguarda[-\s]?chuva\b",
            r"\bsailing\b",
            r"\bsail\b",
            r"\bsea conditions\b",
            r"\bmarine forecast\b",
            r"\bagitacao\s+maritima\b",
            r"\bestado\s+do\s+mar\b",
            r"\bondulacao\b",
            r"\bcomo est[aá] o tempo\b",
            r"\bque tempo faz\b",
            r"\bqual (?:é|e) o tempo\b(?!\s+de\s+espera)",
            r"\btempo em\b",
            r"\btempo hoje\b",
            r"\btempo amanh[aã]\b",
        ]
        if any(re.search(pattern, normalized) for pattern in weather_patterns):
            return True

        # Guard: do not classify "tempo de espera" (wait time) as weather even
        # if a more permissive pattern matched. This is a transport phrase.
        if re.search(r"\btempos?\s+de\s+espera\b", normalized):
            return False

        return False

    @classmethod
    def _looks_like_transport_query(cls, message_lower: str) -> bool:
        """Detects transport and routing queries from natural PT/EN phrasing, not only explicit mode words."""
        normalized = cls._normalize_query(message_lower)
        has_carris_route_code = bool(
            re.search(r"\b\d{1,4}e\b", normalized, flags=re.IGNORECASE)
            or re.search(
                r"\b(?:linha|route|line|rota|tram|el[eé]trico|autocarro|bus|"
                r"o|a|do|da|no|na|the)\s+\d{2,4}[a-z]?\b",
                normalized,
                flags=re.IGNORECASE,
            )
        )
        carris_operational_route_query = bool(
            has_carris_route_code
            and re.search(
                r"\b(?:funciona|funcionar|circula|circular|passa|passar|"
                r"detalhes?|details?|rota|route|percurso|trajeto|trajecto|"
                r"paragens?|stops?|onde\s+est[aá]|where\s+is|agora|now|"
                r"apanh[ao]r?|catch|running|run|service|status|"
                r"para\b|ao\b|at[eé]\b|direction|direc[cç][aã]o|destino|"
                r"terminus|terminal|sentido)\b",
                normalized,
                flags=re.IGNORECASE,
            )
            and not cls._looks_like_weather_query(normalized)
        )
        if carris_operational_route_query:
            return True
        weather_only_transport_false_positive = (
            cls._looks_like_weather_query(normalized)
            and re.search(
                r"\b(?:previs[aã]o|forecast)\b.*\b(?:tempo|weather)\b|"
                r"\b(?:tempo|weather)\b.*\b(?:hoje|amanh[aã]|semana|dias?|week|days?)\b",
                normalized,
            )
            and not re.search(
                r"\b(?:metro|bus|autocarro|comboio|train|tram|el[eé]trico|transportes?|public transport|"
                r"como\s+(?:vou|chego|posso ir)|how\s+(?:do|can)\s+i\s+(?:get|go)|from\s+.+\s+to)\b",
                normalized,
            )
        )
        if weather_only_transport_false_positive:
            return False

        weather_warning_false_positive = (
            cls._looks_like_weather_query(normalized)
            and re.search(r"\b(?:aviso|avisos|alerta|alertas|warning|warnings)\b", normalized)
            and re.search(
                r"\b(?:vento|wind|chuva|rain|temperatura|temperature|trovoada|thunderstorm|"
                r"precipitacao|precipitation|meteo|weather|agitacao|ondulacao|"
                r"maritima|maritime|swell|waves|sea)\b",
                normalized,
            )
            and not re.search(
                r"\b(?:metro|bus|autocarro|comboio|train|tram|el[eé]trico|carris|cp|"
                r"transportes?|public transport|rota|route|"
                r"como\s+(?:vou|chego|posso ir|ir)|how\s+(?:do|can)\s+i\s+(?:get|go)|"
                r"from\s+.+\s+to|de\s+.+\s+para\s+.+\s+(?:de\s+metro|de\s+autocarro|de\s+comboio))\b",
                normalized,
            )
        )
        if weather_warning_false_positive:
            return False

        event_calendar_false_positive = (
            re.search(
                r"\b(?:eventos?|events?|concertos?|concerts?|festivais?|festivals?|"
                r"arrai(?:al|ais|s)?|santos populares|marchas populares|festas de lisboa)\b",
                normalized,
                flags=re.IGNORECASE,
            )
            and re.search(
                r"\b(?:janeiro|fevereiro|mar[cç]o|abril|maio|junho|julho|agosto|"
                r"setembro|outubro|novembro|dezembro|january|february|march|april|"
                r"may|june|july|august|september|october|november|december|"
                r"este mes|este m[eê]s|this month|esta semana|this week)\b",
                normalized,
                flags=re.IGNORECASE,
            )
            and not re.search(
                r"\b(?:metro|bus|autocarro|comboio|train|carris|transportes?|transport|"
                r"como\s+(?:vou|chego|posso ir|ir)|how\s+(?:do|can)\s+i\s+(?:get|go)|"
                r"quero\s+(?:ir|apanhar)|preciso\s+(?:de\s+)?ir|from\s+.+\s+to)\b",
                normalized,
                flags=re.IGNORECASE,
            )
        )
        if event_calendar_false_positive:
            return False

        _walk_norm_no_neg = re.sub(
            r"\b(?:no|not|without|sem|n[aã]o|nem|sans)\s+"
            r"(?:metro|bus|autocarro|autocarros|comboio|train|carris|tram|el[eé]trico|"
            r"cp\b|fertagus|transportes?|transport|barco|ferry)\b",
            "",
            normalized,
        )
        walking_context_false_positive = (
            re.search(r"\ba\s+p[eé]\b|walking|a\s+pe\b|walk\b|passeio\s+a\s+p[eé]|a\s+caminhar|caminhando", normalized)
            and not re.search(
                r"\b(?:metro|bus|autocarro|autocarros|comboio|train|carris|tram|el[eé]trico|"
                r"cp\b|fertagus|transportes?|transport|barco|ferry|paragem|esta[cç][aã]o)\b",
                _walk_norm_no_neg,
            )
        )
        if walking_context_false_positive:
            return False

        route_intent_markers = bool(
            re.search(
                r"\b(?:metro|bus|autocarro|comboio|train|tram|el[eé]trico|carris|cp|"
                r"transportes?|public transport|rota|route|percurso|trajeto|"
                r"como\s+(?:vou|chego|posso ir|ir)|how\s+(?:do|can)\s+i\s+(?:get|go)|"
                r"get from|go from|apanhar|catch|partida|departure|hor[áa]rio|schedule)\b",
                normalized,
            )
        )
        generic_de_para_false_positive = (
            re.search(
                r"\b(?:receita|recipe|cozinhar|cooking|ingredientes?|ingredients?|"
                r"bom\s+para|boa\s+para|s[ií]tio\s+indoor|sitio\s+indoor|"
                r"passar\s+\d+\s+hora|crian[cç]a|bilhetes?|tickets?|morada|address|"
                r"pre[cç]o|price|abert[oa]|website)\b",
                normalized,
            )
            and not route_intent_markers
        )
        generic_food_place_de_para_false_positive = (
            re.search(
                r"\b(?:jantar|dinner|almoco|lunch|restaurantes?|restaurants?|"
                r"casas?\s+de\s+fado|fado\s+houses?|bar(?:es)?\s+de\s+fado|"
                r"bares?|bars?|miradouros?|viewpoints?)\b",
                normalized,
                flags=re.IGNORECASE,
            )
            and not route_intent_markers
        )
        if generic_de_para_false_positive or generic_food_place_de_para_false_positive:
            return False

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
            r"\b(?:e\s+se\s+)?(?:quiser|quero|queria|preciso|vou|posso|tenho)\s+(?:de\s+)?ir\b",
            r"\ba partir do\b",
            r"\ba partir da\b",
            r"\ba partir dos\b",
            r"\ba partir das\b",
            r"\b(?:quero|queria|quiser|preciso|vou|posso|tenho)\s+(?:de\s+)?ir\s+(?:dos|das|do|da|de)\s+.+?\s+(?:para|ao|a|à|até)\s+.+",
            r"\b(?:dos|das|do|da|de)\s+.+?\s+(?:para|ao|a|à|até)\s+.+",
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
        if re.search(r"\b\d{4}-\d{3}\b", str(user_message or "")):
            return False

        out_of_scope_patterns = [
            r"\b(?:todos?|todas?|all|every)\b.{0,80}\b(?:portugal|pa[ií]s|country|national)\b",
            r"\b(?:portugal|pa[ií]s|country|national)\b.{0,80}\b(?:todos?|todas?|all|every)\b",
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
            r"\b(?:receita|recipe)\b",
            r"\b(?:cozinhar|cooking)\b",
            r"\bmandarim\b",
            r"\bjapan\b",
            r"\bjap[aã]o\b",
        ]
        return any(re.search(pattern, message_lower) for pattern in out_of_scope_patterns)

    @classmethod
    def _is_geographic_out_of_scope_route(cls, user_message: str) -> bool:
        """Detect point-to-point route requests outside LISBOA's AML scope."""
        normalized = cls._normalize_query(user_message)
        if not cls._looks_like_transport_query(normalized):
            return False
        return route_mentions_outside_aml(user_message)

    @classmethod
    def _is_geographic_out_of_scope_query(cls, user_message: str) -> bool:
        """Detect non-route Lisbon-domain requests that target places outside AML."""
        if cls._is_geographic_out_of_scope_route(user_message):
            return False
        if not route_mentions_outside_aml(user_message):
            return False

        normalized = cls._normalize_query(user_message)
        lisbon_domain_terms = (
            r"weather|forecast|tempo|previs[aã]o|avisos?|warnings?|"
            r"events?|eventos?|concertos?|concerts?|festival|exposi[cç][aã]o|"
            r"places?|locais|atra[cç][oõ]es?|attractions?|museus?|museums?|"
            r"monumentos?|monuments?|miradouros?|viewpoints?|"
            r"restaurants?|restaurantes?|caf[eé]s?|pastelarias?|"
            r"farm[aá]cias?|pharmacies|hospitals?|bibliotecas?|libraries|"
            r"pol[ií]cia|police|parking|estacionamento|servi[cç]os?|services?|"
            r"roteiro|itiner[aá]rio|itinerary|plano|plan|visitar|visit|"
            r"melhores|best|recomenda|recommend|onde|where|nearby|perto"
        )
        return bool(re.search(rf"\b(?:{lisbon_domain_terms})\b", normalized, flags=re.IGNORECASE))

    @classmethod
    def _is_unsupported_action_request(cls, user_message: str) -> bool:
        """Detects transactional actions that LISBOA can explain but cannot perform."""
        normalized = cls._normalize_query(user_message)
        unsupported_patterns = [
            r"\b(book|reserve)\s+(?:me\s+)?(?:a\s+)?(?:table|restaurant|ticket|tickets|seat|seats|hotel|room)\b",
            r"\b(can|could|please)?\s*(?:you\s+)?(?:help\s+me\s+)?(?:book|reserve|buy|purchase)\s+(?:(?:me\s+)?(?:a|an|one)\s+)?(?:table|restaurant|ticket|tickets|seat|seats|hotel|room|flight|pass|passes)\b",
            r"\b(make|booking|do|doing)\s+(?:me\s+)?(?:a\s+)?(?:reservation|booking)\b",
            r"\bbuy\s+(?:me\s+)?(?:a\s+)?(?:ticket|tickets)\b",
            r"\breserva[-\s]?me\s+(?:uma\s+)?(?:mesa|bilhetes?|hotel|quarto)\b",
            r"\breservar\s+(?:uma\s+)?(?:mesa|bilhetes?|hotel|quarto)\b",
            r"\breserva\s+(?:j[aá]\s+)?(?:uma\s+)?(?:mesa|bilhetes?|hotel|quarto)\b",
            r"\bmarca[-\s]?me\s+(?:uma\s+)?(?:mesa|reserva|bilhetes?|hotel|quarto)\b",
            r"\bmarcar\s+(?:uma\s+)?(?:mesa|reserva|bilhetes?|hotel|quarto)\b",
            r"\bmarca\s+(?:uma\s+)?(?:mesa|reserva|bilhetes?|hotel|quarto)\b",
            r"\bfazer\s+(?:uma\s+)?reserva\b",
            r"\bfaz\s+(?:uma\s+)?reserva\b",
            r"\bcompra[-\s]?me\s+(?:bilhetes?|entradas?)\b",
            r"\b(?:quero\s+)?comprar\s+(?:um\s+|uma\s+)?(?:bilhetes?|entradas?|tickets?)\b",
            r"\bcomprar\s+(?:bilhetes?|entradas?)\s+(?:por mim|para mim)\b",
            r"\bcompra\s+(?:bilhetes?|entradas?)\s+(?:por mim|para mim)\b",
            r"\b(?:consegues|consegue|podes|pode|poderias|podias)\s+(?:comprar|reservar|marcar)\s+(?:bilhetes?|entradas?|tickets?|mesa|reserva)\b",
            r"\b(?:can|could)\s+you\s+(?:buy|purchase|book|reserve)\s+(?:tickets?|seats?|a\s+table|a\s+reservation)\b",
        ]
        return any(re.search(pattern, normalized) for pattern in unsupported_patterns)

    @classmethod
    def _unsupported_action_has_supported_lookup_target(cls, user_message: str) -> bool:
        """Return whether an unsupported transaction still has useful Lisbon lookup value."""
        normalized = cls._normalize_query(user_message)
        if not normalized:
            return False
        if route_mentions_outside_aml(user_message):
            return False
        if cls._is_geographic_out_of_scope_query(user_message):
            return False
        transactional_target = bool(
            re.search(
                r"\b(?:at|for|in|near|no|na|em|para|pelo|pela)\s+"
                r"[a-z0-9][a-z0-9'\- ]{2,80}",
                normalized,
            )
        )
        lisbon_lookup_context = bool(
            cls._has_lisbon_context(normalized)
            or re.search(
                r"\b(?:restaurante|restaurant|mesa|table|bilhetes?|tickets?|"
                r"concerto|concert|evento|event|ccb|teatro|theatre|hotel|"
                r"ramiro|lisboa|lisbon)\b",
                normalized,
            )
        )
        return transactional_target and lisbon_lookup_context

    @classmethod
    def _is_broad_realtime_transport_dump_request(cls, user_message: str) -> bool:
        """Detect requests that ask for an unusably broad live transport dump."""
        normalized = cls._normalize_query(user_message)
        if not normalized:
            return False
        wait_time_catalog_request = bool(
            re.search(r"\b(?:tempos?\s+de\s+espera|wait\s+times?)\b", normalized)
            and re.search(r"\b(?:todas?|todos?|all|every)\b", normalized)
            and re.search(r"\b(?:esta[cç][oõ]es?|stations?|linhas?|lines|metro|subway)\b", normalized)
        )
        if (
            wait_time_catalog_request
            and re.search(r"\b(?:metro|subway)\b", normalized)
            and re.search(r"\b(?:linhas?|lines)\b", normalized)
            and not re.search(
                r"\b(?:paragens?|stops|esta[cç][oõ]es?|stations?|"
                r"ve[ií]culos?|vehicles?|partidas?|departures?|"
                r"comboios?|trains?|autocarros?|buses|servi[cç]os?|services?)\b",
                normalized,
            )
        ):
            return False
        if (
            not wait_time_catalog_request
            and not re.search(r"\b(?:tempo real|real[-\s]?time|live|agora|now)\b", normalized)
        ):
            return False
        has_transport_catalog_context = bool(
            re.search(
                r"\b(?:carris|autocarros?|buses|paragens?|stops|linhas?|lines|"
                r"metro|subway|esta[cç][oõ]es?|stations?|tempos?\s+de\s+espera|wait\s+times?|"
                r"cp|comboios?|trains?|partidas?|departures?|servi[cç]os?|services?)\b",
                normalized,
            )
        )
        asks_all_catalog = bool(
            re.search(
                r"\b(?:todas?|todos?|all|every)\b.*\b(?:linhas?|lines|paragens?|stops|esta[cç][oõ]es?|stations?|partidas?|departures?|comboios?|trains?|servi[cç]os?|services?|tempos?\s+de\s+espera|wait\s+times?)\b"
                r"|\b(?:linhas?|lines|paragens?|stops|esta[cç][oõ]es?|stations?|partidas?|departures?|comboios?|trains?|servi[cç]os?|services?|tempos?\s+de\s+espera|wait\s+times?)\b.*\b(?:todas?|todos?|all|every)\b",
                normalized,
            )
        )
        requested_metro_lines = set(
            re.findall(r"\b(?:linha\s+)?(amarela|azul|verde|vermelha|yellow|blue|green|red)\b", normalized)
        )
        single_metro_line_scope = len(requested_metro_lines) == 1 and re.search(
            r"\b(?:metro|esta[cç][oõ]es?|stations?|tempos?\s+de\s+espera|wait\s+times?)\b",
            normalized,
        )
        if single_metro_line_scope and not re.search(r"\b(?:todas?\s+as\s+linhas|all\s+lines|all\s+metro)\b", normalized):
            return False
        return has_transport_catalog_context and asks_all_catalog

    @staticmethod
    def _build_broad_realtime_transport_dump_response(language: str) -> str:
        """Build a direct response for broad real-time transport catalogue requests."""
        if language == "pt":
            return (
                "### 📡 **Pedido demasiado amplo para tempo real**\n\n"
                "✅ **Resposta direta:** não é útil nem fiável despejar todas as linhas, estações, paragens, partidas, veículos ou tempos de espera numa só resposta em tempo real.\n\n"
                "---\n\n"
                "- 🧭 **Para responder com qualidade:** indica uma linha, paragem, zona ou origem → destino.\n"
                "- 🚌 **Exemplos bons:** `tempo de espera na Alameda`, `próximo 758 em Amoreiras`, `autocarro de Avenidas Novas para Campo de Ourique`, `próximo CP de Entrecampos para Sete Rios`."
            )
        return (
            "### 📡 **Request Too Broad For Real-Time Data**\n\n"
            "✅ **Direct answer:** dumping every line, station, stop, departure, vehicle, or wait time in one live answer is not useful or reliable.\n\n"
            "---\n\n"
            "- 🧭 **To answer well:** give me a line, stop, area, or origin → destination.\n"
            "- 🚌 **Good examples:** `wait time at Alameda`, `next 758 at Amoreiras`, `bus from Avenidas Novas to Campo de Ourique`, `next CP train from Entrecampos to Sete Rios`."
        )

    @staticmethod
    def _build_unsupported_action_response(language: str) -> str:
        """Builds a concise limitation response for unsupported booking actions."""
        if language == "pt":
            return (
                "### ⚠️ **Reservas e Compras Não Suportadas**\n\n"
                "✅ **Resposta direta:** não consigo fazer reservas, compras ou marcações diretamente, mas posso ajudar-te a decidir com dados verificáveis sobre Lisboa.\n\n"
                "---\n\n"
                "- ✅ **Posso confirmar:** contactos, moradas, fontes oficiais e informação pública do local quando estiver disponível.\n"
                "- 🚫 **Não posso assumir:** disponibilidade de mesa/lugar, preços atuais, bilhetes ainda válidos ou confirmação de reserva."
            )
        return (
            "### ⚠️ **Booking and Purchase Requests**\n\n"
            "✅ **Direct answer:** I can't make bookings, purchases, or reservations directly, but I can help you decide with verifiable Lisbon data.\n\n"
            "---\n\n"
            "- ✅ **I can confirm:** contacts, addresses, official sources, and public venue information when available.\n"
            "- 🚫 **I cannot assume:** table/seat availability, current prices, still-valid tickets, or booking confirmation."
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
                "### 🧭 **Outside LISBOA's Scope**\n\n"
                "✅ **Direct answer:** that falls outside what I can validate with quality in this Lisbon/AML system.\n\n"
                "---\n\n"
                "LISBOA is focused on the **Lisbon Metropolitan Area**.\n\n"
                "💡 **Supported scope:**\n"
                "- 🌤️ **Weather:** forecasts, warnings, and IPMA data for Lisbon and AML\n"
                "- 🚇 **Transport:** Metro, Carris Urban, Carris Metropolitana, and CP suburban/AML\n"
                "- 🎭 **Culture & Events:** museums, exhibitions, festivals, concerts, and activities\n"
                "- 📍 **Places & Services:** restaurants, attractions, pharmacies, hospitals, parking, and public services\n"
                "- 🗺️ **Planning:** personalized itineraries and day plans for Lisbon/AML\n"
                "- 📚 **History & Knowledge:** Lisbon history, neighborhoods, culture, and Lisboa Card guide\n\n"
                "Ask me about Lisbon/AML and I will use confirmable data whenever possible."
            )
        return (
            "### 🧭 **Fora do Âmbito do LISBOA**\n\n"
            "✅ **Resposta direta:** isso fica fora do que consigo validar com qualidade neste sistema focado em Lisboa/AML.\n\n"
            "---\n\n"
            "O LISBOA está focado na **Área Metropolitana de Lisboa**.\n\n"
            "💡 **Âmbito suportado:**\n"
            "- 🌤️ **Meteorologia:** previsão, avisos e dados IPMA para Lisboa/AML\n"
            "- 🚇 **Transportes:** Metro, Carris Urban, Carris Metropolitana e CP suburbano/AML\n"
            "- 🎭 **Cultura & Eventos:** museus, exposições, festivais, concertos e atividades\n"
            "- 📍 **Locais & Serviços:** restaurantes, atrações, farmácias, hospitais, estacionamento e serviços públicos\n"
            "- 🗺️ **Planeamento:** roteiros personalizados e planos de dia para Lisboa/AML\n"
            "- 📚 **História & Conhecimento:** história de Lisboa, bairros, cultura e Guia Lisboa Card\n\n"
            "Pergunta-me por Lisboa/AML e eu respondo com dados confirmáveis sempre que possível."
        )

    @classmethod
    def _is_capability_query(cls, user_message: str) -> bool:
        """Detects queries asking what LISBOA can do, without a specific Lisbon topic."""
        normalized = cls._normalize_query(user_message)
        if not normalized:
            return False
        capability_patterns = [
            r"\bwhat\s+can\s+(?:you|lisboa|the\s+assistant)\s+(?:do|help|assist)\b",
            r"\bwhat\s+(?:do\s+you|can\s+you|are\s+you\s+able\s+to)\s+(?:do|know|help\s+with|assist\s+with)\b",
            r"\bwhat\s+(?:are\s+your|your)\s+(?:capabilities|features|functions)\b",
            r"\bwhat\s+can\s+i\s+(?:ask|request)\b",
            r"\bhow\s+can\s+(?:you|lisboa)\s+help\b",
            r"\btell\s+me\s+what\s+you\s+(?:do|can)\b",
            r"\bwhat\s+(?:topics?|areas?|domains?)\s+(?:do\s+you|can\s+you)\s+cover\b",
            r"\bo\s+que\s+(?:e\s+que\s+o?\s*)?(?:o\s+)?(?:lisboa|assistente)?\s*consegue\s+fazer\b",
            r"\bo\s+que\s+(?:sabes?|podes?)\s+fazer\b",
            r"\bque\s+(?:capacidades?|funcionalidades?|funcoes?|fun[cç][oõ]es?)\s+(?:tens?|tem)\b",
            r"\bcomo\s+(?:me\s+podes?|podes?\s+me)\s+ajudar\b",
            r"\bo\s+que\s+posso\s+(?:pedir|perguntar|pedir-te)\b",
            r"\bem\s+que\s+(?:me\s+podes?|podes?\s+me)\s+ajudar\b",
            r"\bque\s+(?:tipos?\s+de\s+)?(?:perguntas?|questoes?)\s+(?:posso|podes?)\b",
            r"\bhelp\s*$",
            r"\bajuda\s*$",
        ]
        return any(re.search(p, normalized, flags=re.IGNORECASE) for p in capability_patterns)

    @staticmethod
    def _build_full_capability_response(language: str) -> str:
        """Builds the canonical all-6-capabilities response."""
        if language == "pt":
            return (
                "### 🤖 **O que o LISBOA consegue fazer**\n\n"
                "Sou o teu **Assistente Urbano de Lisboa**, especializado na "
                "**Área Metropolitana de Lisboa (AML)**. Aqui está o que podes pedir:\n\n"
                "- 🌤️ **Meteorologia** — previsões do tempo, avisos e dados IPMA para Lisboa e AML\n"
                "- 🚌 **Transportes** — rotas de metro, autocarro, elétrico e comboio; "
                "estado do serviço e informação em tempo real\n"
                "- 🏛️ **Cultura & Eventos** — museus, exposições, festivais, concertos e atividades em Lisboa\n"
                "- 📍 **Locais & Serviços** — restaurantes, atrações, farmácias, hospitais, "
                "estacionamento e serviços públicos via dados abertos\n"
                "- 🗓️ **Planeamento** — itinerários personalizados e planos de dia para Lisboa/AML\n"
                "- 📚 **História & Conhecimento** — história de Lisboa, bairros, cultura e Guia Lisboa Card\n\n"
                "Experimenta perguntar, por exemplo: *\"Como ir de Belém ao Oriente?\"*, "
                "*\"Que eventos há este fim de semana?\"* ou *\"Planeia-me um dia em Alfama.\"*"
            )
        return (
            "### 🤖 **What LISBOA Can Do**\n\n"
            "I'm your **Lisbon Urban Assistant**, specialized in the "
            "**Lisbon Metropolitan Area (AML)**. Here's what you can ask me:\n\n"
            "- 🌤️ **Weather** — forecasts, warnings, and IPMA data for Lisbon and AML\n"
            "- 🚌 **Transport** — metro, bus, tram, and train routes; "
            "service status and real-time information\n"
            "- 🏛️ **Culture & Events** — museums, exhibitions, festivals, concerts, and activities in Lisbon\n"
            "- 📍 **Places & Services** — restaurants, attractions, pharmacies, hospitals, "
            "parking, and public services via open data\n"
            "- 🗓️ **Planning** — personalized itineraries and day plans for Lisbon/AML\n"
            "- 📚 **History & Knowledge** — Lisbon's history, neighborhoods, culture, and Lisboa Card guide\n\n"
            "Try asking: *\"How do I get from Belém to Oriente?\"*, "
            "*\"What events are on this weekend?\"*, or *\"Plan me a day in Alfama.\"*"
        )

    def _direct_routing_override(self, user_message: str, language: str) -> Optional[Dict[str, Any]]:
        """Handles trivial direct responses before invoking the supervisor LLM."""
        if self._is_non_informative_message(user_message):
            return {
                "reasoning": "Mensagem vazia/pontuação apenas, sem pedido claro",
                "agents": [],
                "direct_response": self._sanitize_direct_response(
                    self._build_non_informative_message_response(language)
                ),
            }

        if self._is_greeting_only(user_message):
            return {
                "reasoning": "Direct greeting override",
                "agents": [],
                "direct_response": self._sanitize_direct_response(self._build_greeting_response(language)),
            }

        if self._is_capability_query(user_message):
            return {
                "reasoning": "Direct capability query override",
                "agents": [],
                "direct_response": self._sanitize_direct_response(self._build_full_capability_response(language)),
            }

        if self._is_visit_confirmation_checklist_query(user_message):
            return {
                "reasoning": "Direct pre-visit checklist override; answer needs grounded place details, not planning or route synthesis.",
                "agents": ["researcher"],
                "direct_response": None,
            }

        if self._is_unsupported_action_request(user_message) and route_mentions_outside_aml(user_message):
            return {
                "reasoning": "Direct geographic out-of-scope transactional request",
                "agents": [],
                "direct_response": self._sanitize_direct_response(
                    build_geographic_out_of_scope_response(
                        user_message,
                        language=language,
                        mobility=False,
                    )
                ),
            }

        if self._is_unsupported_action_request(user_message):
            normalized_transaction = self._normalize_query(user_message)
            if re.search(r"\b(?:uber|bolt|taxi|taxis|taxi|taxis|t[aá]xi|t[aá]xis|ride\s*hailing)\b", normalized_transaction):
                return {
                    "reasoning": "Unsupported ride-hailing booking routed to transport for mobility-scope limitation.",
                    "agents": ["transport"],
                    "direct_response": None,
                }
            if re.search(r"\b(?:comboio|comboios|train|trains|cp|metro|autocarro|autocarros|bus|buses|carris)\b", normalized_transaction):
                return {
                    "reasoning": "Unsupported transport ticket purchase routed to transport for supported schedule/route information.",
                    "agents": ["transport"],
                    "direct_response": None,
                }
            if self._unsupported_action_has_supported_lookup_target(user_message):
                return {
                    "reasoning": "Unsupported transaction with a supported Lisbon lookup target",
                    "agents": ["researcher"],
                    "direct_response": None,
                }
            return {
                "reasoning": "Direct unsupported transactional action override",
                "agents": [],
                "direct_response": self._sanitize_direct_response(self._build_unsupported_action_response(language)),
            }

        if self._is_broad_realtime_transport_dump_request(user_message):
            return {
                "reasoning": "Direct broad real-time transport dump override",
                "agents": [],
                "direct_response": self._sanitize_direct_response(
                    self._build_broad_realtime_transport_dump_response(language)
                ),
            }

        if self._is_geographic_out_of_scope_route(user_message):
            return {
                "reasoning": "Direct geographic out-of-scope route override",
                "agents": [],
                "direct_response": self._sanitize_direct_response(
                    build_geographic_out_of_scope_response(user_message, language=language)
                ),
            }

        if self._is_geographic_out_of_scope_query(user_message):
            return {
                "reasoning": "Direct geographic out-of-scope domain override",
                "agents": [],
                "direct_response": self._sanitize_direct_response(
                    build_geographic_out_of_scope_response(
                        user_message,
                        language=language,
                        mobility=False,
                    )
                ),
            }

        if self._is_obvious_out_of_scope(user_message):
            return {
                "reasoning": "Direct out-of-scope override",
                "agents": [],
                "direct_response": self._sanitize_direct_response(self._build_out_of_scope_response(language)),
            }

        return None

    @classmethod
    def _is_visit_confirmation_checklist_query(cls, user_message: str) -> bool:
        """Detect venue-specific pre-visit checklist requests."""
        normalized = cls._normalize_query(user_message)
        if not re.search(
            r"\b(?:o\s+que\s+(?:devo\s+)?confirmar|que\s+(?:devo\s+)?confirmar|"
            r"confirmar\s+antes|what\s+should\s+i\s+(?:check|confirm)|"
            r"what\s+to\s+(?:check|confirm)|check\s+before|confirm\s+before)\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            return False
        return bool(
            re.search(
                r"\b(?:vou|quero\s+ir|pretendo\s+ir|visitar|visit|going|go)\b"
                r".{0,100}\b(?:ao|a|à|para|to|o|a)\b",
                normalized,
                flags=re.IGNORECASE,
            )
            or re.search(
                r"\b(?:oceanario|oceanário|jardim\s+zoologico|jardim\s+zoológico|maat|ccb|"
                r"mosteiro|museu|museum|castelo|castle|torre|tower|palacio|palácio)\b",
                normalized,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _sanitize_direct_response(text: Optional[str]) -> Optional[str]:
        """Removes unsupported closing offers from direct supervisor responses."""
        if not text:
            return text
        return strip_unsupported_closing_offers(text).strip()

    @classmethod
    def _should_route_complex_query_with_llm(cls, user_message: str) -> bool:
        """Return whether a high-entropy query should bypass single-domain rules.

        The deterministic single-domain overrides are useful for short, obvious
        requests, but they should not pre-empt the supervisor model when the
        user expresses preferences, exclusions, trade-offs, or multiple
        information needs. Hard safety/scope gates still run before this check.
        """
        normalized = cls._normalize_query(user_message)
        if not normalized:
            return False
        if cls._looks_like_follow_up(user_message) and len(normalized.split()) <= 7:
            return False
        if cls._is_weather_only_outdoor_decision_query(user_message):
            return False
        if cls._is_direct_weather_transport_query(user_message):
            return False
        if not cls._is_planning_query(user_message):
            research_terms = (
                r"\b(?:eventos?|events?|concertos?|concerts?|festivais?|festivals?|"
                r"exposi[cç][oõ]es?|exhibitions?|cultura|culture|"
                r"atra[cç][oõ]es?|attractions?|museus?|museums?|monumentos?|monuments?|"
                r"locais|places?|restaurantes?|restaurants?|fado|"
                r"farm[aá]cias?|pharmacies|hospitais?|hospitals?|bibliotecas?|libraries|"
                r"servi[cç]os?|services?)\b"
            )
            operational_terms = (
                r"\b(?:weather|tempo|previs[aã]o|chuva|rain|metro|autocarros?|bus|"
                r"comboios?|train|transportes?|transport|rota|route|como\s+(?:vou|chego|posso\s+ir))\b"
            )
            if re.search(research_terms, normalized, flags=re.IGNORECASE) and not re.search(
                operational_terms,
                normalized,
                flags=re.IGNORECASE,
            ):
                return False

        preference_markers = (
            r"\b(?:prefer|preference|personalized|personalised|avoid|without|excluding|except|not\s+too|"
            r"least|fewest|best|recommend|suggest|different|hidden|local|authentic|budget|cheap|"
            r"accessible|wheelchair|rain|rainy|children|kids|elderly|"
            r"prefiro|preferencia|preferencia|personalizado|evitar|sem|excluir|exceto|excepto|"
            r"nao\s+quero|não\s+quero|menos|melhor|recomenda|sugere|diferente|local|autentico|"
            r"autêntico|barato|orcamento|orçamento|acessivel|acessível|cadeira\s+de\s+rodas|"
            r"chuva|criancas|crianças|idosos?)\b"
        )
        multi_need_markers = (
            r"\b(?:and|also|with|including|include|plus|then|after|before|"
            r"e|tambem|também|com|inclui|incluindo|depois|antes)\b"
        )
        cardinality_markers = (
            r"\b(?:\d{1,2}|um|uma|one|dois|duas|two|tres|three|quatro|four|cinco|five|"
            r"seis|six|sete|seven|oito|eight)\s+"
            r"(?:museus?|museums?|monumentos?|monuments?|locais|lugares|sitios|sítios|"
            r"places|stops|paragens|restaurantes?|restaurants?|miradouros?|viewpoints?)\b"
        )
        anchor_constraint_markers = (
            r"\b(?:passa(?:r|ndo)?\s+(?:por|pelo|pela)|via|pass\s+through|stop\s+at|"
            r"termina|termine|terminar|acaba|acabe|acabar|finish|ending|end)\b"
        )
        domain_terms = {
            "weather": r"\b(?:weather|tempo|meteorolog|chuva|rain|forecast|previsao|previsão)\b",
            "transport": r"\b(?:metro|bus|buses|autocarro|comboio|train|cp|transport|transportes|route|rota)\b",
            "researcher": r"\b(?:event|evento|restaurant|restaurante|museum|museu|monument|monumento|fado|farmacia|farmácia|hospital|library|biblioteca|place|local)\b",
            "planner": r"\b(?:plan|itinerary|roteiro|plano|planeia|organiza|dia|day|afternoon|tarde|passa(?:r|ndo)?|terminar|acabar|stops?|paragens?)\b",
        }
        matched_domains = {
            name for name, pattern in domain_terms.items()
            if re.search(pattern, normalized, flags=re.IGNORECASE)
        }
        has_preference = bool(re.search(preference_markers, normalized, flags=re.IGNORECASE))
        has_multi_need = bool(re.search(multi_need_markers, normalized, flags=re.IGNORECASE))
        has_cardinality = bool(re.search(cardinality_markers, normalized, flags=re.IGNORECASE))
        has_anchor_constraints = bool(re.search(anchor_constraint_markers, normalized, flags=re.IGNORECASE))
        tokens = normalized.split()
        if len(tokens) < 8 and not (has_preference or has_cardinality or has_anchor_constraints):
            return False
        return has_preference or has_cardinality or has_anchor_constraints or (has_multi_need and len(matched_domains) >= 2)

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
        if len(tokens) <= 6 and any(token in anaphoric_tokens for token in tokens):
            return True
        return bool(
            re.search(
                r"\b(?:destes|destas|desses|dessas|estes|estas|esses|essas|"
                r"anteriores|previous|above|listed|these|those)\b",
                normalized,
            )
        )

    @classmethod
    def _single_domain_override(cls, user_message: str) -> Optional[Dict[str, Any]]:
        """Routes obvious standalone single-domain queries without letting history or the LLM over-expand them."""
        if cls._looks_like_follow_up(user_message) and not cls._looks_like_transport_query(user_message):
            return None
        direct_weather_transport = cls._is_direct_weather_transport_query(user_message)
        message_lower = cls._normalize_query(user_message)
        if re.search(r"\b(uber|bolt|taxi|taxis|táxi|táxis|ride-hailing)\b", message_lower):
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

        # Single time-windowed museum recommendation → researcher (not planner skeleton).
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

        if (
            re.search(
                r"\b(?:alternativa|alternative|op[cç][aã]o|option|local|sitio|s[ií]tio|place)\b",
                message_lower,
            )
            and re.search(r"\b(?:indoor|interior|interiores|cobert[oa]s?)\b", message_lower)
            and re.search(r"\b(?:perto\s+de\s+transporte|near\s+(?:public\s+)?transport)\b", message_lower)
        ):
            return {
                "reasoning": "Direct indoor-place alternative override; transport is a proximity constraint, not a route request.",
                "agents": ["researcher"],
                "direct_response": None,
            }

        weather_hit = cls._looks_like_weather_query(message_lower)
        transport_terms = [
            "metro", "bus", "autocarro", "comboio", "train", "carris",
            "route", "rota", "station", "estação", "paragem", "departures", "wait time",
        ]
        event_terms = [
            "event", "events", "evento", "eventos", "concert", "concerto",
            "festival", "exhibition", "exposição", "exposicao", "music", "música", "musica",
            "what's on", "o que há", "o que ha", "desporto", "desportivo",
            "desportivos", "sports", "sport", "arraial", "arraiais", "arrais",
            "santos populares", "marchas populares", "festas de lisboa",
        ]
        place_terms = [
            "attraction", "attractions", "atração", "atrações", "atracao", "atracoes",
            "museum", "museu", "monument", "monumento", "miradouro", "places", "locais",
            "restaurant", "restaurante", "what to visit", "o que visitar",
            "pavilhão", "pavilhao", "sítio", "sitio", "indoor", "bilhete", "bilhetes",
            "ticket", "tickets", "morada", "address", "preço", "preco", "price",
        ]
        service_terms = [
            "pharmacy", "farmácia", "farmacia", "hospital", "school", "escola",
            "library", "biblioteca", "police", "polícia", "policia", "psp", "esquadra",
            "market", "markets", "mercado", "mercados", "feira", "feiras",
            "park", "parks", "parque", "parques", "garden", "gardens", "jardim", "jardins",
            "parking", "estacionamento", "wifi", "wi-fi", "wc", "restroom", "restrooms",
            "casa de banho", "casas de banho", "instalações sanitárias",
            "posto de turismo", "tourist office",
        ]
        service_transport_request = bool(
            re.search(
                r"\b(?:metro|autocarro|autocarros|bus|buses|comboio|train|carris|"
                r"transporte|transport|rota|route|como\s+(?:vou|chego|posso\s+ir|ir)|"
                r"leva[-\s]?me|take\s+me|sem\s+ser\s+a\s+p[eé]|without\s+walking|"
                r"apanhar|catch|partida|departure|hor[áa]rio|schedule)\b",
                message_lower,
                flags=re.IGNORECASE,
            )
        )
        reduced_mobility_visit_request = bool(
            re.search(
                r"\b(?:mobilidade\s+reduzida|cadeira\s+de\s+rodas|wheelchair|"
                r"reduced\s+mobility|step[-\s]?free|acess[ií]vel|acessibilidade)\b",
                message_lower,
                flags=re.IGNORECASE,
            )
            and re.search(
                r"\b(?:sugere|recomenda|suggest|recommend|visita\s+curta|short\s+visit|"
                r"pouca\s+caminhada|little\s+walking|transporte\s+simples|simple\s+transport)\b",
                message_lower,
                flags=re.IGNORECASE,
            )
        )
        if reduced_mobility_visit_request:
            return {
                "reasoning": "Direct reduced-mobility visit recommendation override; this needs grounded place evidence, not a route from the origin to itself.",
                "agents": ["researcher"],
                "direct_response": None,
            }

        _msg_no_neg_transport = re.sub(
            r"\b(?:no|not|without|sem|n[aã]o|nem|sans)\s+"
            r"(?:metro|bus|autocarro|autocarros|comboio|train|carris|tram|el[eé]trico|"
            r"cp\b|fertagus|transportes?|transport|barco|ferry)\b",
            "",
            message_lower,
        )
        _explicit_walk_route = (
            re.search(r"\b(?:walk(?:ing)?|a\s+p[eé]|a\s+pe|caminhando|a\s+caminhar)\b", message_lower)
            and re.search(
                r"\b(?:from|de|do|da|desde)\s+.{1,60}?\s+(?:to|para|até|ate|a|à|ao)\b",
                message_lower,
            )
            and not cls._contains_domain_keyword(
                _msg_no_neg_transport, transport_terms, minimum_ratio=0.85
            )
        )
        if _explicit_walk_route:
            return {
                "reasoning": "Explicit walking route between two locations; no positive transport mode — walking guidance via planner and researcher.",
                "agents": ["planner", "researcher"],
                "direct_response": None,
            }
        transport_hit = cls._looks_like_transport_query(message_lower) or cls._contains_domain_keyword(
            _msg_no_neg_transport,
            transport_terms,
            minimum_ratio=0.85,
        )
        nearest_metro_request = bool(
            transport_hit
            and re.search(r"\b(?:metro|station|esta[cç][aã]o|esta[cç][oõ]es)\b", message_lower)
            and re.search(r"\b(?:mais\s+perto|mais\s+pr[oó]xim[ao]s?|nearest|closest)\b", message_lower)
        )
        if nearest_metro_request:
            return {
                "reasoning": "Direct nearest-metro override",
                "agents": ["transport"],
                "direct_response": None,
            }
        if weather_hit and transport_hit and cls._weather_reference_is_route_modifier(message_lower):
            weather_hit = False
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
        if event_hit and not exact_event_hit and re.search(
            r"\b(?:what\s+(?:can|should)\s+i\s+(?:visit|see)|o\s+que\s+h(?:a|\u00e1|\u00e3)?\s+(?:para\s+)?(?:ver|visitar)|"
            r"l\s+perto|nearby)\b",
            message_lower,
        ):
            event_hit = False
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
                r"\bcasas?\s+de\s+fado\b",
                r"\bfado\s+houses?\b",
                r"\bbar(?:es)?\s+de\s+fado\b",
                r"\bwhat to visit\b",
                r"\bwhat (?:can|should) i (?:visit|see)\b",
                r"\bwhat .* nearby\b",
                r"\bo que visitar\b",
                r"\bo que h(?:a|\u00e1|\u00e3)?\s+(?:para\s+)?(?:ver|visitar)\b",
                r"\bo que h\s+(?:para\s+)?(?:ver|visitar)\b",
                r"\bo que existe\s+(?:para\s+)?(?:ver|visitar)\b",
                r"\bl(?:a|\u00e1)\s+perto\b",
                r"\bl\s+perto\b",
                r"\bperto\s+(?:de\s+)?l(?:a|\u00e1)\b",
                r"\bperto\s+(?:de\s+)?l\b",
            )
        )
        shopping_hit = bool(
            re.search(
                r"\b(?:shops?|stores?|shopping|lojas?|compras?|comprar|buy|"
                r"clothes|clothing|roupa|vestu[aá]rio|casacos?|jackets?)\b",
                message_lower,
                flags=re.IGNORECASE,
            )
            and re.search(
                r"\b(?:near|perto|em|no|na|where|onde|comprar|buy|shops?|stores?|lojas?)\b",
                message_lower,
                flags=re.IGNORECASE,
            )
        )
        place_hit = exact_place_hit or cls._contains_domain_keyword(message_lower, place_terms, minimum_ratio=0.82)
        if shopping_hit:
            place_hit = True
        if transport_hit and not exact_place_hit:
            place_hit = False
        service_hit = cls._contains_domain_keyword(message_lower, service_terms, minimum_ratio=0.84)
        if (
            service_hit
            and not any([weather_hit, event_hit, place_hit])
            and re.search(
                r"\b(?:nearest|closest|nearby|mais\s+pr[oó]xim[ao]s?|"
                r"perto\s+(?:de|do|da|dos|das)|junto\s+(?:de|do|da|dos|das)|"
                r"na\s+zona\s+de|nas\s+proximidades\s+de|near)\b",
                message_lower,
                flags=re.IGNORECASE,
            )
            and not service_transport_request
        ):
            return {
                "reasoning": "Direct nearby-service override; Lisboa Aberta proximity includes distance/walking-time estimate.",
                "agents": ["researcher"],
                "direct_response": None,
            }

        if transport_hit and re.search(
            r"\b(?:(?:when|quando|a\s+que\s+horas).{0,80}\b(?:next|pr[oó]xim[ao]|bus|autocarro|linha|line)|"
            r"(?:next|pr[oó]xim[ao]|eta|arrival|chegada).{0,80}\b(?:bus|autocarro|linha|line)|"
            r"(?:bus|autocarro|linha|line)\s+\d{1,4}[a-z]?)\b",
            message_lower,
            flags=re.IGNORECASE,
        ):
            return {
                "reasoning": "Direct live bus-arrival override",
                "agents": ["transport"],
                "direct_response": None,
            }

        if service_hit and transport_hit and not any([weather_hit, event_hit, place_hit]):
            needs_service_confirmation = bool(
                re.search(r"\b(?:confirmar|confirma|h[aá]\s+uma?|chamad[ao]|called|named)\b", message_lower)
            )
            nearby_service_request = bool(
                re.search(
                    r"\b(?:nearest|closest|nearby|mais\s+pr[oó]xim[ao]s?|"
                    r"pr[oó]xim[ao]s?|perto\s+(?:de|do|da|dos|das)|"
                    r"junto\s+(?:de|do|da|dos|das)|near)\b",
                    message_lower,
                    flags=re.IGNORECASE,
                )
            )
            service_context_before_origin = re.split(
                r"\b(?:desde|from|a\s+partir\s+de)\b",
                message_lower,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            explicit_service_destination = bool(
                needs_service_confirmation
                or re.search(
                    r"\b(?:farm[aá]cia|biblioteca|hospital|mercado|escola|parque|library|pharmacy|market)"
                    r"\b.{0,100}\b(?:rua|avenida|av\.?|largo|pra[cç]a|travessa|\d{4}-\d{3})\b",
                    service_context_before_origin,
                    flags=re.IGNORECASE,
                )
                or re.search(
                    r"\b(?:farm[aá]cia|biblioteca|hospital|mercado|escola|parque|library|pharmacy|market)"
                    r"\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ'-]+",
                    user_message,
                )
            )
            has_explicit_origin = bool(
                re.search(r"\b(?:desde|a\s+partir\s+de|from|de|do|da)\s+.+\b(?:para|at[eé]|até|to)\b", message_lower)
                or re.search(r"\b(?:estou|encontro-me|encontro me|i\s+am|i'm)\s+(?:no|na|em|at)\s+.+\b(?:para|at[eé]|até|to)\b", message_lower)
            )
            if nearby_service_request and not explicit_service_destination:
                agents = ["researcher"]
            else:
                agents = ["transport"] if has_explicit_origin and not needs_service_confirmation else ["researcher", "transport"]
            return {
                "reasoning": "Direct service-destination mobility override; route/confirmation request should not become an itinerary planner task.",
                "agents": agents,
                "direct_response": None,
            }

        if transport_hit and place_hit and not any([weather_hit, event_hit, service_hit]):
            explicit_point_to_point = bool(
                re.search(
                    r"\b(?:desde|a\s+partir\s+de|from|de|do|da)\s+.+\b(?:para|at[eé]|até|to)\s+.+",
                    message_lower,
                )
            )
            if not explicit_point_to_point:
                explicit_point_to_point = bool(
                    re.search(r"\b(?:desde|from|de|do|da)\s+.+\b(?:a|ao)\s+.+", message_lower)
                )
            operational_route_request = bool(
                re.search(
                    r"\b(?:rota|route|transporte|transport|como\s+(?:vou|chego|posso\s+ir)|"
                    r"how\s+(?:do|can)\s+i\s+(?:get|go)|"
                    r"(?:quero|preciso|tenho)\s+(?:de\s+)?ir|"
                    r"a\s+que\s+horas|quando\s+devo\s+sair|sair|apanhar|catch|leave|"
                    r"chegar\s+(?:a|à|as|às))\b",
                    message_lower,
                )
            )
            nearby_place_request = bool(
                re.search(
                    r"\b(?:what\s+(?:can|should)\s+i\s+(?:visit|see)|what\s+.*nearby|"
                    r"nearby|around\s+there|o\s+que\s+h(?:a|\u00e1|\u00e3)?\s+(?:para\s+)?(?:ver|visitar)|"
                    r"o\s+que\s+h\s+(?:para\s+)?(?:ver|visitar)|"
                    r"o\s+que\s+existe\s+(?:para\s+)?(?:ver|visitar)|l(?:a|\u00e1)\s+perto|l\s+perto|"
                    r"perto\s+(?:de\s+)?l(?:a|\u00e1)|perto\s+(?:de\s+)?l)\b",
                    message_lower,
                )
            )
            if explicit_point_to_point and operational_route_request and nearby_place_request:
                return {
                    "reasoning": "Direct route plus nearby-place lookup override; answer needs transport and researcher, not planner.",
                    "agents": ["transport", "researcher"],
                    "direct_response": None,
                }
            if explicit_point_to_point and operational_route_request:
                return {
                    "reasoning": "Direct place-destination mobility override; point-to-point transport should not become an itinerary planner task.",
                    "agents": ["transport"],
                    "direct_response": None,
                }

        pure_weather_question = (
            weather_hit
            and re.search(
                r"\b(?:que\s+tempo|como\s+est[aá]\s+o\s+tempo|vai\s+chover|"
                r"previs[aã]o|forecast|weather|avisos?|warnings?)\b",
                message_lower,
            )
            and not re.search(
                r"\b(?:onde|where|recomenda|recommend|sugere|suggest|encontra|find|"
                r"eventos?|events?|museus?|museums?|restaurantes?|restaurants?|"
                r"farm[aá]cias?|pharmacies|bibliotecas?|libraries|mercados?|markets?)\b",
                message_lower,
            )
        )
        if pure_weather_question:
            return {
                "reasoning": "Direct weather override; place/service word is an incidental location fragment.",
                "agents": ["weather"],
                "direct_response": None,
            }

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

        if weather_hit and place_hit and not any([transport_hit, event_hit, service_hit]):
            needs_place_lookup = shopping_hit or bool(
                re.search(
                    r"\b(?:where|onde|which|qual|quais|recommend|recomendas?|suggest|"
                    r"find|encontra|procurar|comprar|buy|preciso|need|quero|want|"
                    r"shops?|stores?|lojas?)\b",
                    message_lower,
                    flags=re.IGNORECASE,
                )
            )
            return {
                "reasoning": (
                    "Direct weather-aware place lookup override"
                    if needs_place_lookup
                    else "Direct weather advice override with incidental place context"
                ),
                "agents": ["weather", "researcher"] if needs_place_lookup else ["weather"],
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
            "musica", "what's on", "o que há", "o que ha", "desporto",
            "desportivo", "desportivos", "sports", "sport", "arraial", "arraiais",
            "arrais", "santos populares", "marchas populares", "festas de lisboa",
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

        last_user_message = None
        for msg in reversed(conversation_history):
            if isinstance(msg, HumanMessage) and msg.content:
                last_user_message = str(msg.content)
                break

        current_domain = infer_domain(user_message)
        previous_domain = infer_domain(last_user_message or "")
        if current_domain == "planner":
            return {
                "reasoning": "Follow-up domain override from current planning intent",
                "agents": cls._planning_follow_up_agents(user_message),
                "direct_response": None,
            }
        mode_only_follow_up = bool(
            current_domain == "transport"
            and previous_domain == "planner"
            and re.search(
                r"^\s*(?:e\s+)?(?:de\s+)?(?:metro|autocarro|autocarros|bus|comboio|train|tram|el[eé]trico)\s*\??\s*$|"
                r"^\s*(?:and\s+)?(?:by\s+)?(?:metro|bus|train|tram)\s*\??\s*$",
                cls._normalize_query(user_message),
            )
        )
        if mode_only_follow_up:
            return {
                "reasoning": "Mode-only follow-up resolved against previous itinerary",
                "agents": cls._planning_follow_up_agents(user_message),
                "direct_response": None,
            }
        if previous_domain == "planner" and re.search(
            r"\b(?:adiciona|acrescenta|mant[eé]m|continua|ajusta|troca|substitui|"
            r"garante|evita|remove|add|keep|continue|adjust|replace|avoid|remove)\b",
            cls._normalize_query(user_message),
        ):
            return {
                "reasoning": "Planning revision follow-up resolved against previous itinerary context",
                "agents": cls._planning_follow_up_agents(user_message),
                "direct_response": None,
            }
        if current_domain:
            return {
                "reasoning": f"Follow-up domain override from current query ({current_domain})",
                "agents": [current_domain],
                "direct_response": None,
            }
        if previous_domain == "planner":
            return {
                "reasoning": "Follow-up domain override from previous planning query",
                "agents": cls._planning_follow_up_agents(user_message),
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
    def _planning_follow_up_agents(cls, user_message: str) -> List[str]:
        """Return a grounded planning stack for short planning follow-ups."""
        normalized = cls._normalize_query(user_message)
        agents: List[str] = []
        if (
            cls._planning_query_mentions_weather(user_message)
            or re.search(
                r"\b(?:hoje|amanh[ãa]|today|tomorrow|esta\s+(?:manh[ãa]|tarde|noite)|"
                r"this\s+(?:morning|afternoon|evening)|fim\s+de\s+semana|weekend|chuva|rain)\b",
                normalized,
            )
        ):
            agents.append("weather")
        if (
            not cls._planning_query_blocks_transport_enrichment(user_message)
            and (
                cls._planning_query_requires_transport_context(user_message)
                or re.search(
                    r"\b(?:menos\s+caminhada|pouca\s+caminhada|low\s+walking|walk\s+less|"
                    r"transporte|transportes|metro|autocarro|comboio|bus|train|return|voltar|hotel)\b",
                    normalized,
                )
            )
        ):
            agents.append("transport")
        agents.extend(["researcher", "planner"])
        return [agent for index, agent in enumerate(agents) if agent not in agents[:index]]

    @classmethod
    def _is_planning_query(cls, user_message: str) -> bool:
        """Detects itinerary/planning intent without over-matching words like `today`."""
        message_lower = cls._normalize_query(user_message)
        if cls._negates_itinerary_request(user_message):
            return False
        if cls._is_category_browse_query(user_message):
            return False
        if cls._is_direct_weather_transport_query(user_message):
            return False
        if cls._is_transport_line_route_query(user_message):
            return False
        pure_weather_request = (
            cls._looks_like_weather_query(message_lower)
            and re.search(
                r"\b(?:que\s+tempo|como\s+est[aá]\s+o\s+tempo|vai\s+chover|"
                r"previs[aã]o|forecast|weather|avisos?|warnings?)\b",
                message_lower,
            )
            and not re.search(
                r"\b(?:planeia|plan(?:ear)?|organiza|roteiro|itiner[aá]rio|itinerary|"
                r"faz\s+uma\s+(?:manh[aã]|tarde|noite)|dia\s+inteiro|full\s+day)\b",
                message_lower,
            )
        )
        if pure_weather_request:
            return False
        if re.search(
            r"\b(?:quero|queria|gostava|preciso|faz|fazer|planeia|organiza|monta)\b.*"
            r"\b(?:tarde|manh[aã]|noite|afternoon|morning|evening)\b.*"
            r"\b(?:almo[cç]o|lunch|jantar|dinner|miradouro|viewpoint|jardim|garden|"
            r"caf[eé]|coffee|restaurante|restaurant)\b.*"
            r"\b(?:almo[cç]o|lunch|jantar|dinner|miradouro|viewpoint|jardim|garden|"
            r"caf[eé]|coffee|restaurante|restaurant)\b",
            message_lower,
        ):
            return True
        simple_lookup_with_preferences = bool(
            re.search(
                r"\b(?:restaurants?|restaurantes?|museums?|museus?|monuments?|monumentos?|"
                r"places?|locais|attractions?|atra[c\u00e7][o\u00f5]es?|bibliotecas?|libraries|"
                r"farm[a\u00e1]cias?|pharmacies|hospitais?|hospitals)\b",
                message_lower,
            )
            and re.search(
                r"\b(?:near|perto|nearby|open|abert[oa]s?|cheap|barat[oa]s?|"
                r"vegetarian[oa]s?|vegan[oa]s?|accessible|acess[i\u00ed]vel|wheelchair|"
                r"cadeira\s+de\s+rodas|today|hoje|tonight|esta\s+noite)\b",
                message_lower,
            )
            and not re.search(
                r"\b(?:planeia|plan(?:ear)?|organiza|organize|itinerary|itiner[a\u00e1]rio|"
                r"roteiro|dia\s+inteiro|full\s+day|half\s+day|meio\s+dia|"
                r"\d+\s*(?:dias?|days?)|v[a\u00e1]rios?\s+locais|multiple\s+places|"
                r"ordem|order|schedule|agenda)\b",
                message_lower,
            )
        )
        if simple_lookup_with_preferences:
            return False
        multi_stop_visit_request = bool(
            re.search(
                r"\b(?:quero\s+ir|queria\s+ir|gostava\s+de\s+ir|i\s+want\s+to\s+go|"
                r"i\s+would\s+like\s+to\s+go|visitar|visit)\b",
                message_lower,
            )
            and len(
                re.findall(
                    r"\b(?:museus?|museums?|mosteiro|mosteiros|monastery|monasteries|"
                    r"torre|torres|tower|towers|maat|monumentos?|monuments?|"
                    r"palacio|palace|castelo|castle|miradouro|viewpoint|jardim|garden)\b",
                    message_lower,
                )
            )
            >= 2
        )
        if multi_stop_visit_request and re.search(
            r"\b(?:hoje|today|amanha|tomorrow|mobilidade|accessibility|accessible|"
            r"pouca\s+caminhada|low\s+walking|menos\s+caminhada|roteiro|itinerary|"
            r"almoco|lunch|jantar|dinner)\b",
            message_lower,
        ):
            return True
        planning_patterns = [
            r"\bplan my day\b",
            r"\bday plan\b",
            r"\bitinerary\b",
            r"\b(?:itiner[aá]rio|itener[aá]rio|itinerario|itenerario)\b",
            r"\broteiro\b",
            r"\bday trip\b",
            r"\bpasseio\b",
            r"\bplan\b.*\b(?:day|days|afternoon|morning|evening|itinerary|trip|route|visit|stops?|"
            r"rainy|museum|museums|restaurants?|hotel|transport|return|lisbon|lisboa|bel[eé]m)\b",
            r"\b(?:plane(?:ar|ia|ie)|organiza(?:r)?|organize|organise|organizing)\b.*\b(?:dia|day|"
            r"manh(?:a)?|morning|tarde|afternoon|noite|evening|horas?|hours?|roteiro|itiner[aá]rio|itinerary|"
            r"ordem|order|transportes?|hotel|locais|places|stops?|paragens?|visitar|visit|"
            r"comer|eat|food|meal|refei[cç][aã]o|almo[cç]o|jantar|lunch|dinner)\b",
            r"\b(?:plano|plan)\b.*\b(?:dia|day|manh[aã]|tarde|noite|viagem|trip|visita|visit)\b",
            r"\bschedule\b.*\b(?:day|itinerary|route|visits?|stops?)\b",
            r"\b(?:quero|queria|gostava|preciso|faz|fazer|planeia|organiza|monta)\b.*"
            r"\b(?:tarde|manh[aã]|noite|afternoon|morning|evening)\b.*"
            r"\b(?:almo[cç]o|lunch|jantar|dinner|miradouro|viewpoint|jardim|garden|"
            r"caf[eé]|coffee|restaurante|restaurant)\b.*"
            r"\b(?:almo[cç]o|lunch|jantar|dinner|miradouro|viewpoint|jardim|garden|"
            r"caf[eé]|coffee|restaurante|restaurant)\b",
            r"\b(?:cria|criar|monta|montar|faz|fazer)\b.*\b(?:itiner[aá]rio|itener[aá]rio|roteiro|plano|dia|manh(?:a)?|tarde)\b",
            r"\b(?:itiner[aá]rio|itener[aá]rio|roteiro|plano)\b.*\b(?:cria|criar|monta|montar|faz|fazer|inclui|incluir|comer|refei[cç][aã]o|almo[cç]o|jantar|hotel)\b",
            r"\b(?:estes|estas|esses|essas|these|those)\s+(?:locais|lugares|s[ií]tios|places|stops)\b.*\b(?:dia|day|amanh[aã]|tomorrow|almo[cç]o|jantar|lunch|dinner|hotel)\b",
            r"\b(?:inclui|incluir|with|including)\b.*\b(?:almo[cç]o|jantar|lunch|dinner)\b.*\b(?:dia|day|roteiro|itiner[aá]rio|itener[aá]rio|plano|hotel)\b",
            r"\bir a vários? locais\b",
            r"\bvisit multiple\b",
            r"\b(?:visitar|visitando)\b.*\b(?:varios|vários|multiple|locais|places|stops|paragens|"
            r"roteiro|itinerario|itinerary|percurso|rota|route|ordem|order|dia\s+inteiro|full\s+day)\b",
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
            r"\b(?:plane(?:ar|ia|ie)|organiza(?:r)?|cria|criar|faz|fazer|plan|organize|organise)\b"
            r".*\b(?:a\s+partir\s+de|desde|starting\s+from|start(?:ing)?\s+at)\b"
            r".*\b(?:comer|eat|food|meal|refei[cç][aã]o|almo[cç]o|jantar|lunch|dinner|paragens?|stops?)\b",
            r"\b(?:come[cç]a|come[cç]ar|inicia|iniciar|start|starting)\b.*"
            r"\b(?:passa(?:r|ndo)?\s+(?:por|pelo|pela)|via|termina|terminar|acaba|acabar|end|finish)\b",
            r"\b(?:passa(?:r|ndo)?\s+(?:por|pelo|pela)|via|pass\s+through)\b.*"
            r"\b(?:termina|terminar|acaba|acabar|end|finish|roteiro|itinerary|plano|plan)\b",
            # NOTE: indefinite articles (um/uma/one) are intentionally excluded from the
            # count list. "quero um restaurante" means "I want A restaurant" (a simple
            # lookup), not "1 restaurant" as an itinerary count. Real itinerary counts use
            # digits or two+ (dois/two/...). Multi-component plans like "faz uma tarde com
            # um museu e um miradouro" are still caught by the afternoon/plan patterns above.
            r"\b(?:quero|queria|gostava|ver|visitar|visit|see|show|mostra|inclui|include)\b.*"
            r"\b(?:\d{1,2}|dois|duas|two|tr[eê]s|three|quatro|four|cinco|five|seis|six|sete|seven|oito|eight)\s+"
            r"(?:museus?|museums?|monumentos?|monuments?|locais|lugares|sitios|s[ií]tios|places|stops|paragens|restaurantes?|restaurants?|miradouros?|viewpoints?)\b",
            r"\b(?:plane(?:ar|ia|ie)|planear|plan(?:eia)?)\b.*\b\d+\s*dias?\b",
            r"\b\d+\s*dias?\b.*\b(?:plane(?:ar|ia|ie)|planear|itiner[aá]rio|roteiro)\b",
            r"\bdia\s+inteiro\b",
            r"\btarde\s+inteira\b",
            r"\bmanh[aã]\s+inteira\b",
        ]
        if any(re.search(pattern, message_lower) for pattern in planning_patterns):
            return True

        return False

    @classmethod
    def _is_transport_line_route_query(cls, user_message: str) -> bool:
        """Return whether the user asks about a fixed transport line route."""
        normalized = cls._normalize_query(user_message)
        if not normalized:
            return False
        has_line = bool(
            re.search(
                r"\b(?:linha\s*)?(?:tram|eletrico|el[eé]trico|bus|autocarro|carris)?\s*\d{1,3}[a-z]?\b",
                normalized,
            )
        )
        has_route_question = bool(
            re.search(
                r"\b(?:route|rota|percurso|trajeto|stops?|paragens?|frequency|frequencia|horarios?|schedule)\b",
                normalized,
            )
        )
        has_personal_plan_intent = bool(
            re.search(r"\b(?:planeia|organiza|roteiro|itinerary|plan my|para mim|for me|dia|day)\b", normalized)
        )
        return has_line and has_route_question and not has_personal_plan_intent

    @classmethod
    def _negates_itinerary_request(cls, user_message: str) -> bool:
        """Return whether the user explicitly says not to produce a route/itinerary."""
        normalized = cls._normalize_query(user_message)
        if not normalized:
            return False
        explicit_negation = bool(
            re.search(
                r"\b(?:nao|não)\s+(?:me\s+)?(?:des|de|d[eê]s|fa[cç]as?|quero|preciso)\s+(?:um\s+|uma\s+)?(?:roteiro|itinerario|itiner[aá]rio|rota|percurso)\b",
                normalized,
            )
            or re.search(
                r"\b(?:sem|without|no)\s+(?:(?:criar|fazer|montar|gerar|produzir|give\s+me|make\s+me)\s+)?"
                r"(?:um\s+|uma\s+|an?\s+)?(?:roteiro|itinerario|itiner[aá]rio|itinerary|route|walking route|plano|plan)\b",
                normalized,
            )
            or re.search(r"\bdo\s+not\s+give\s+me\s+(?:an?\s+)?(?:itinerary|route|plan)\b", normalized)
            or re.search(r"\bdon'?t\s+give\s+me\s+(?:an?\s+)?(?:itinerary|route|plan)\b", normalized)
        )
        if not explicit_negation:
            return False
        planning_context = re.sub(
            r"\b(?:sem|without|no)\s+(?:(?:criar|fazer|montar|gerar|produzir|give\s+me|make\s+me)\s+)?"
            r"(?:um\s+|uma\s+|an?\s+)?(?:roteiro|itinerario|itiner[aá]rio|itinerary|route|walking route|plano|plan)\b",
            "",
            normalized,
        )
        planning_context = re.sub(
            r"\b(?:nao|não)\s+(?:me\s+)?(?:des|de|d[eê]s|fa[cç]as?|quero|preciso)\s+"
            r"(?:um\s+|uma\s+)?(?:roteiro|itinerario|itiner[aá]rio|rota|percurso)\b",
            "",
            planning_context,
        )
        return bool(
            re.search(
                r"\b(?:explica|explicar|resumo|resume|summarize|explain|historia|history|cultura|culture|o que era|what was)\b",
                normalized,
            )
            or not re.search(r"\b(?:planeia|planear|plan|visitar|visit|dia|day|roteiro|itinerario|itinerary|route)\b", planning_context)
        )

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
            r"\bcome[cç]ar no\b",
            r"\bcome[cç]ar na\b",
            r"\ba come[cç]ar em\b",
            r"\ba come[cç]ar no\b",
            r"\ba come[cç]ar na\b",
            r"\bcome[cç]ando em\b",
            r"\bcome[cç]ando no\b",
            r"\bcome[cç]ando na\b",
        ]
        return any(re.search(pattern, normalized) for pattern in origin_patterns)

    @classmethod
    def _planning_query_has_route_constraint(cls, user_message: str) -> bool:
        """Return whether a plan has movement constraints beyond a local start anchor."""
        normalized = cls._normalize_query(user_message)
        if not normalized:
            return False
        route_constraint_patterns = [
            r"\b(?:termina|termine|terminar|terminando|acaba|acabe|acabar|acabando|"
            r"finish|finishes|finishing|ending|ends?|end)\b",
            r"\b(?:passa(?:r|ndo)?\s+(?:por|pelo|pela)|via|pass\s+through|stop\s+at)\b",
            r"\b(?:voltar|regressar|return|back)\s+(?:ao|a|à|para|to)\b",
            r"\b(?:hotel|alojamento|accommodation)\b.*\b(?:voltar|regressar|return|back)\b",
            r"\b(?:ate|até|by)\s+\d{1,2}(?::\d{2})?\b",
            r"\b\d{1,2}(?::\d{2})?\s*(?:h|am|pm)\b.*\b(?:passa|passar|termina|acaba|finish|end)\b",
        ]
        return any(re.search(pattern, normalized) for pattern in route_constraint_patterns)

    @classmethod
    def _planning_query_explicitly_requests_transport(cls, user_message: str) -> bool:
        """Return whether the user explicitly asks for transport inside an itinerary."""
        normalized = cls._normalize_query(user_message)
        if not normalized:
            return False
        normalized = cls._strip_negative_transport_generation_instruction(normalized)
        return bool(
            re.search(
                r"\b(?:transportes?\s+p[úu]blicos?|public\s+transport|transit|metro|"
                r"autocarros?|bus|buses|comboios?|train|trains|cp|carris|el[eé]tricos?|tram|trams|"
                r"menos\s+caminhada|pouca\s+caminhada|low\s+walking|walk\s+less|sem\s+andar\s+muito)\b",
                normalized,
            )
        )

    @classmethod
    def _strip_negative_transport_generation_instruction(cls, normalized_message: str) -> str:
        """Remove negative anti-invention transport wording from positive intent probes."""
        normalized = re.sub(r"\s+", " ", str(normalized_message or "").lower()).strip()
        if not normalized:
            return ""
        transport_targets = (
            r"transportes?|transporte|rotas?|routes?|percursos?|trajetos?|trips?|"
            r"linhas?|lines?|paragens?|stops?|horarios?|schedules?|partidas?|departures?|"
            r"metro|autocarros?|bus|buses|comboios?|train|trains|tram|trams|eletricos?"
        )
        generation_verbs = (
            r"inventar(?:es)?|inventes|inventa|invent|mak(?:e|ing)\s+up|assumir|assumas|assume|"
            r"criar|create|adicionar|add"
        )
        negative_prefix = r"sem|nao|nunca|without|no|do\s+not|dont|don't"
        # Strip only the forward form ("sem inventar ... transportes") with a short
        # gap. A trailing negative after a transport noun ("quais os autocarros ...
        # sem inventar") asks for real data; stripping the noun would flip intent.
        cleaned = re.sub(
            rf"\b(?:{negative_prefix})\s+(?:{generation_verbs})\b(?:\s+\w+){{0,3}}\s+\b(?:{transport_targets})\b",
            " ",
            normalized,
        )
        return re.sub(r"\s+", " ", cleaned).strip()

    @classmethod
    def _planning_query_blocks_transport_enrichment(cls, user_message: str) -> bool:
        """Return whether transport is mentioned only to forbid invented details."""
        normalized = cls._normalize_query(user_message)
        if not normalized:
            return False
        cleaned = cls._strip_negative_transport_generation_instruction(normalized)
        if cleaned == normalized:
            return False
        return not bool(
            re.search(
                r"\b(?:metro|carris|cp|autocarros?|bus|buses|comboios?|train|trains|tram|trams|"
                r"transportes?\s+publicos?|public\s+transport|transit|"
                r"como\s+(?:vou|chego|posso\s+ir|me\s+desloco)|how\s+(?:do\s+i\s+)?get|"
                r"route\s+from|route\s+between|rota\s+de|rotas\s+entre|percurso\s+de|trajeto\s+de)\b",
                cleaned,
            )
        )

    @classmethod
    def _planning_query_requires_transport_context(cls, user_message: str) -> bool:
        """Return whether the itinerary request requires transport-tool evidence."""
        normalized = cls._normalize_query(user_message)
        has_origin_anchor = cls._planning_query_has_origin_anchor(user_message)
        has_route_constraint = cls._planning_query_has_route_constraint(user_message)
        positive_transport_probe = cls._strip_negative_transport_generation_instruction(normalized)
        if cls._planning_query_explicitly_requests_transport(user_message):
            return True
        if has_origin_anchor and has_route_constraint:
            return True
        if cls._looks_like_transport_query(positive_transport_probe) and not has_origin_anchor:
            return True

        named_far_zone_re = re.compile(
            r"\b(?:bel[eé]m|oriente|parque das nac[oõ]es|cascais|sintra|oeiras|"
            r"almada|cacilhas|montijo|mafra|ericeira)\b",
            flags=re.IGNORECASE,
        )
        mentioned_far_zones = {
            match.group(0).lower()
            for match in named_far_zone_re.finditer(normalized)
        }
        return len(mentioned_far_zones) >= 2

    @classmethod
    def _is_direct_weather_transport_query(cls, user_message: str) -> bool:
        """Detects operational weather-plus-route requests that should not become itineraries."""
        normalized = cls._normalize_query(user_message)
        if not normalized:
            return False
        if cls._weather_reference_is_route_modifier(normalized):
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
    def _weather_reference_is_route_modifier(cls, message_lower: str) -> bool:
        """Return whether weather wording is only a route preference, not a weather request."""
        normalized = cls._normalize_query(message_lower)
        if not normalized:
            return False
        has_route_condition = bool(
            re.search(
                r"\b(?:com\s+chuva|se\s+chover|se\s+estiver\s+a\s+chover|"
                r"sem\s+apanhar\s+chuva|evitar\s+chuva|with\s+rain|if\s+it\s+rains|"
                r"in\s+the\s+rain|avoid\s+rain|rainy)\b",
                normalized,
                flags=re.IGNORECASE,
            )
        )
        if not has_route_condition:
            return False
        explicit_weather_request = bool(
            re.search(
                r"\b(?:vai\s+chover|vai\s+estar|previsao|forecast|tempo|weather|meteo|"
                r"aviso|avisos|alerta|alertas|warning|warnings|temperatura|temperature|"
                r"vento|wind)\b",
                normalized,
                flags=re.IGNORECASE,
            )
        )
        return not explicit_weather_request

    @classmethod
    def _is_weather_only_outdoor_decision_query(cls, user_message: str) -> bool:
        """Detect weather advice for an outdoor activity without a requested transport leg."""
        normalized = cls._normalize_query(user_message)
        if not normalized or not cls._looks_like_weather_query(normalized):
            return False
        clothing_advice = bool(
            re.search(
                r"\b(?:what\s+(?:should|do)\s+i\s+wear|what\s+to\s+wear|dress(?:ing)?|"
                r"wear(?:ing)?|clothing|jacket|umbrella|o\s+que\s+(?:devo|posso)\s+vestir|"
                r"devo\s+levar\s+(?:casaco|guarda[-\s]?chuva)|vestir|roupa|casaco|"
                r"guarda[-\s]?chuva)\b",
                normalized,
                flags=re.IGNORECASE,
            )
        )
        outdoor_advice = bool(
            re.search(
                r"\b(?:bom\s+tempo|good\s+weather|suitable|adequad[ao]|"
                r"passeio|caminhar|walk|walking|outdoor|ar\s+livre|"
                r"ao\s+ar\s+livre)\b",
                normalized,
                flags=re.IGNORECASE,
            )
        )
        explicit_itinerary = bool(
            re.search(
                r"\b(?:itinerary|roteiro|plan(?:ear)?|planeia|organiza|agenda|day\s+plan|"
                r"plano\s+do\s+dia)\b",
                normalized,
                flags=re.IGNORECASE,
            )
        )
        shopping_lookup = bool(
            re.search(
                r"\b(?:where|onde|comprar|buy|shops?|stores?|lojas?|shopping)\b",
                normalized,
                flags=re.IGNORECASE,
            )
            and re.search(
                r"\b(?:roupa|clothes|clothing|casacos?|jackets?|vestuario|vestu[aá]rio|"
                r"umbrella|guarda[-\s]?chuva|raincoat|impermeavel|imperme[aá]vel|poncho|"
                r"sunscreen|protetor\s+solar|hat|chapeu|chap[eé]u)\b",
                normalized,
                flags=re.IGNORECASE,
            )
        )
        if shopping_lookup:
            return False
        if re.search(
            r"\b(?:walking\s+tours?|guided\s+tours?|visita(?:s)?\s+guiada(?:s)?|"
            r"que\s+visitas|which\s+tours?)\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            return False
        if cls._is_planning_query(user_message) and not (clothing_advice or outdoor_advice):
            return False
        if (clothing_advice or outdoor_advice) and explicit_itinerary:
            return False
        if cls._is_direct_weather_transport_query(user_message):
            return False
        if re.search(
            r"\b(?:event|events|evento|eventos|concert|concerts|concerto|concertos|festival|festivals|exhibition|exhibitions|exposi[cç][aã]o|exposi[cç][oõ]es)\b",
            normalized,
        ):
            return False
        outdoor_activity = re.search(
            r"\b(?:walk|walking(?:\s+around)?|caminhar|passeio|outdoor|ar livre|"
            r"viewpoint|miradouro|cycling|bicicleta|dress(?:ing)?|wear(?:ing)?|"
            r"clothing|jacket|umbrella|vestir|roupa|casaco|guarda[-\s]?chuva|"
            r"outside|queue|waiting|fila|fora|exterior|ar\s+livre)\b",
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
        if re.search(
            r"\b(?:ordem|order|sequ[eê]ncia|sequence|roteiro|itiner[aá]rio|itinerary|"
            r"plano|plan|agenda|almo[cç]o|lunch|jantar|dinner|meio\s+dia|half[-\s]?day|"
            r"manh[aã]|morning|tarde|afternoon|hor[aá]rio|schedule)\b",
            normalized,
        ):
            return False
        place_nouns = (
            r"(?:monumentos?|museus?|locais|lugares|atra[cç][oõ]es?|miradouros?|"
            r"jardins?|parques?|restaurantes?|cafes?|cafés?|pastelarias?|bairros?|"
            r"mercados?|bibliotecas?|lojas|centros?\s+comerciais?|hot[eé]is|alojamentos?|"
            r"cruzeiros?|praias?|golfe?|fado|ruas|viewpoints?|museums?|monuments?|"
            r"places?|attractions?|gardens?|parks?|restaurants?|cafes?|neighbou?rhoods?|"
            r"markets?|libraries?|shops?|malls?|hotels?|accommodations?|cruises?|beaches?|golf|streets?)"
        )
        category_patterns = [
            r"\bwhat kinds? of\b.*\b(?:places?|events?|public services?|services?)\b",
            r"\bwhat types? of\b.*\b(?:places?|events?|public services?|services?)\b",
            r"\b(?:places?|events?|public services?|services?)\b.*\b(?:can i explore|can i look for|can you help me find|available categories|categories)\b",
            r"\b(?:which|what|tell me|list|show me)\b.*\b(?:monuments?|museums?|places?|attractions?)\b.*\b(?:visit|in|around|near)\b",
            r"\b(?:monuments?|museums?|places?|attractions?)\b.*\b(?:can i visit|to visit|worth visiting)\b",
            r"\bque tipos? de\b.*\b(?:locais|eventos|servi[cç]os)\b",
            r"\b(?:locais|eventos|servi[cç]os)\b.*\b(?:posso procurar|posso explorar|categorias disponiveis|categorias)\b",
            r"\b(?:que|quais|diz-me|fala-me|lista|indica)\b.*\b(?:monumentos?|museus?|locais|atra[cç][oõ]es)\b.*\b(?:visitar|em|no|na|perto)\b",
            r"\b(?:monumentos?|museus?|locais|atra[cç][oõ]es)\b.*\b(?:posso ir visitar|posso visitar|para visitar|vale a pena visitar)\b",
            rf"\b(?:que|quais|diz-me|fala-me|lista|indica|mostra|recomenda)\b.*\b{place_nouns}\b.*\b(?:visitar|conhecer|ver|morada|endere[cç]o|perto|em|no|na|com)\b",
            rf"\b{place_nouns}\b.*\b(?:posso\s+(?:visitar|conhecer|ver)|para\s+visitar|vale\s+a\s+pena|morada|endere[cç]o)\b",
            rf"\b(?:which|what|tell me|list|show me|recommend)\b.*\b{place_nouns}\b.*\b(?:visit|see|address|near|in|around|with)\b",
            rf"\b{place_nouns}\b.*\b(?:can\s+i\s+visit|to\s+visit|worth\s+visiting|address)\b",
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

        defer_single_domain_override = self._should_route_complex_query_with_llm(user_message)
        if not defer_single_domain_override:
            single_domain_override = self._single_domain_override(user_message)
            if single_domain_override:
                return single_domain_override

        follow_up_override = self._follow_up_domain_override(user_message, conversation_history)
        if follow_up_override:
            return follow_up_override

        system_prompt = get_supervisor_prompt(language)

        messages = [SystemMessage(content=system_prompt)]

        if conversation_history and self._looks_like_follow_up(user_message):
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
            if not agents and decision.get("direct_response"):
                in_scope_override = self._single_domain_override(user_message)
                if in_scope_override and in_scope_override.get("agents"):
                    agents = list(in_scope_override["agents"])
                    decision["direct_response"] = None
                    reasoning += " (Removed direct response: current in-scope request needs grounded data)"

            if self._negates_itinerary_request(user_message):
                original_agents = list(agents)
                agents = [
                    agent for agent in agents
                    if agent != "planner" and (agent != "transport" or self._looks_like_transport_query(user_message))
                ]
                if agents != original_agents:
                    reasoning += " (Removed planner/transport: user explicitly asked not to receive an itinerary.)"

            # Check if this is a planning query that requires weather.
            is_planning_query = self._is_planning_query(user_message)
            is_multi_day = bool(
                re.search(r"\b(?:plan|itinerary|roteiro|planeia|planejar)\b", user_message, flags=re.IGNORECASE)
                and re.search(
                    r"\b(?:[2-9]\s*(?:days?|dias?)|seven days|five days|7 days|5 days|weekend|fim de semana)\b",
                    user_message,
                    flags=re.IGNORECASE,
                )
            )
            if is_multi_day:
                is_planning_query = True
                planning_stack = ["researcher", "planner"]
                if self._planning_query_requires_transport_context(user_message):
                    planning_stack.insert(0, "transport")
                if self._requires_weather_for_planning(user_message) or self._planning_query_mentions_weather(user_message):
                    planning_stack.insert(0, "weather")
                agents = [agent for agent in [*planning_stack, *agents] if agent]
                agents = [agent for index, agent in enumerate(agents) if agent not in agents[:index]]
                reasoning += " (Deterministic override: multi-day planning requires full planning route)"

            weather_only_outdoor_decision = self._is_weather_only_outdoor_decision_query(user_message)
            if weather_only_outdoor_decision:
                agents = ["weather"]
                reasoning += " (Reduced to weather: no transport leg requested)"
                decision["agents"] = agents

            if not is_planning_query and "planner" in agents:
                single_domain_guard = self._single_domain_override(user_message)
                guarded_agents = list(single_domain_guard.get("agents") or []) if single_domain_guard else []
                if guarded_agents and "planner" not in guarded_agents:
                    agents = guarded_agents
                    reasoning += " (Removed planner: current query is a direct research/status request, not an itinerary.)"
                    decision["direct_response"] = None

            if is_planning_query or weather_only_outdoor_decision:
                decision["direct_response"] = None

            if is_planning_query and not weather_only_outdoor_decision:
                if "planner" not in agents:
                    agents.append("planner")
                    reasoning += " (Added planner agent: itinerary/planning query)"

                if "researcher" not in agents:
                    agents.append("researcher")
                    reasoning += " (Added researcher agent: planning needs place/activity grounding)"

                requires_transport_context = self._planning_query_requires_transport_context(user_message)
                if requires_transport_context and "transport" not in agents:
                    agents.append("transport")
                    reasoning += " (Added transport agent: request includes explicit or cross-zone movement constraints)"
                elif not requires_transport_context and not is_multi_day and "transport" in agents:
                    agents = [agent for agent in agents if agent != "transport"]
                    reasoning += " (Removed transport agent: local itinerary start anchor can be handled by planner after POI grounding)"

            if is_planning_query and (
                self._requires_weather_for_planning(user_message)
                or self._planning_query_mentions_weather(user_message)
            ):
                if "weather" not in agents:
                    agents.append("weather")
                    reasoning += " (Added weather agent: planning for near-future date)"

            if (
                is_planning_query
                and "weather" in agents
                and not weather_only_outdoor_decision
                and not self._requires_weather_for_planning(user_message)
                and not self._planning_query_mentions_weather(user_message)
            ):
                agents = [agent for agent in agents if agent != "weather"]
                reasoning += " (Removed weather agent: itinerary duration is not a weather/date request)"

            reasoning_lower = reasoning.lower()
            if not agents and any(k in reasoning_lower for k in ["matemática", "math", "fora de âmbito", "out of scope", "trivia", "trivialidade"]):
                if not decision.get("direct_response"):
                    decision["direct_response"] = self._build_out_of_scope_response(language)

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

        # 1. Geographic scope fallback. Keep this aligned with the shared AML
        # scope helper so official AML municipalities never become false
        # out-of-scope matches through broad tokens such as "Roma" or "Porto".
        outside_aml_labels = extract_outside_aml_mentions(user_message)
        if outside_aml_labels:
            oos_msg = build_geographic_out_of_scope_response(
                user_message,
                language=language,
                mobility=self._looks_like_transport_query(message_lower),
            )
            return {
                "reasoning": "Fallback: Detected out-of-scope location (outside AML)",
                "agents": [],
                "direct_response": self._sanitize_direct_response(oos_msg),
            }

        # 2. AML municipalities are in scope. In fallback mode, route mobility
        # requests to Transport and let unsupported-data answers be expressed
        # as data/source limitations, not geographic exclusions.
        if extract_aml_municipality_mentions(user_message) and self._looks_like_transport_query(message_lower):
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
            "desporto",
            "desportivo",
            "desportivos",
            "sports",
            "sport",
            "arraial",
            "arraiais",
            "arrais",
            "santos populares",
            "marchas populares",
            "festas de lisboa",
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

        is_planning = self._is_planning_query(message_lower)
        is_multi_day = re.search(r"\b(?:plan|itinerary|roteiro|planeia|planejar)\b", user_message, flags=re.IGNORECASE) and re.search(
            r"\b(?:[2-9]\s*(?:days?|dias?)|seven days|five days|7 days|5 days|weekend|fim de semana)\b",
            user_message,
            flags=re.IGNORECASE,
        )
        if is_planning or is_multi_day:
            # Itinerary queries should be grounded consistently across providers.
            if is_multi_day:
                for agent in ["weather", "transport", "researcher", "planner"]:
                    if agent not in agents:
                        agents.append(agent)
            else:
                if (
                    self._requires_weather_for_planning(user_message)
                    or self._planning_query_mentions_weather(user_message)
                ) and "weather" not in agents:
                    agents.append("weather")
                if (
                    self._planning_query_requires_transport_context(user_message)
                ) and "transport" not in agents:
                    agents.append("transport")
                if "researcher" not in agents:
                    agents.append("researcher")
                if "planner" not in agents:
                    agents.append("planner")

        if not agents:
            if not self._has_lisbon_context(message_lower):
                return {
                    "reasoning": "Fallback: out-of-scope query without Lisbon/AML domain evidence",
                    "agents": [],
                    "direct_response": self._sanitize_direct_response(
                        self._build_out_of_scope_response(language)
                    ),
                }
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
            r"\besta\s+(?:manh[ãa]|tarde|noite)\b",
            r"\blogo\s+à\s+noite\b",
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
            # Weekend
            r"\bweekend\b",
            r"\bfim de semana\b",
            r"\bpr[óo]ximo fim de semana\b",
            r"\bnext weekend\b",
            # Named weekdays / explicit dates in planning requests
            r"\b(?:segunda|terça|terca|quarta|quinta|sexta|sábado|sabado|domingo)(?:-feira)?\b",
            r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            r"\b\d{1,2}\s+de\s+(?:janeiro|fevereiro|março|marco|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\b",
            r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b",
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
