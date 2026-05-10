# ==========================================================================
# Master Thesis - Weather Agent
#   - André Filipe Gomes Silvestre, 20240502
#
#   Specialized agent for weather-related queries using IPMA data.
#   Uses BaseAgent.execute_react_loop() for tool execution.
# ==========================================================================

import re
import unicodedata
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

from agent.agents.base import BaseAgent
from agent.prompts.weather import get_weather_prompt
from agent.utils.langsmith_tracing import traceable
from agent.state import AgentState
from agent.utils.langgraph_compat import ToolNode
from agent.utils.response_formatter import (
    finalize_worker_response,
    infer_response_language,
)

FORECAST_HORIZON_DAYS = 5
FORECAST_MAX_OFFSET = FORECAST_HORIZON_DAYS - 1

_MONTH_NAME_TO_NUMBER = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "janeiro": 1,
    "fevereiro": 2,
    "marco": 3,
    "março": 3,
    "abril": 4,
    "maio": 5,
    "junho": 6,
    "julho": 7,
    "agosto": 8,
    "setembro": 9,
    "outubro": 10,
    "novembro": 11,
    "dezembro": 12,
}

_WEEKDAY_NAME_TO_INDEX = {
    "monday": 0,
    "segunda": 0,
    "segunda feira": 0,
    "tuesday": 1,
    "terca": 1,
    "terça": 1,
    "terca feira": 1,
    "terça feira": 1,
    "wednesday": 2,
    "quarta": 2,
    "quarta feira": 2,
    "thursday": 3,
    "quinta": 3,
    "quinta feira": 3,
    "friday": 4,
    "sexta": 4,
    "sexta feira": 4,
    "saturday": 5,
    "sabado": 5,
    "sábado": 5,
    "sunday": 6,
    "domingo": 6,
}

_WARNING_QUERY_TERMS = [
    "warning",
    "warnings",
    "aviso",
    "avisos",
    "alert",
    "alerts",
    "alerta",
    "alertas",
]


class WeatherAgent(BaseAgent):
    """
    Weather specialist agent using IPMA data.

    Tools:
        - get_weather_warnings
        - get_weather_forecast
        - get_current_weather_summary
        - get_portugal_weather_overview

    Notes:
        This worker is used for weather-specific retrieval. The optional
        `context` argument is injected by the orchestrator in multi-agent
        scenarios to preserve language and follow-up hints.
    """

    def __init__(self):
        """Initializes the weather agent."""
        super().__init__("weather")
        self.system_prompt = get_weather_prompt(language="en")
        self._system_prompt_dynamic = True

    def _get_runtime_system_prompt(self, language: str, *, safe_mode: bool = False) -> str:
        """Return the prompt for the requested language while preserving explicit test overrides."""
        if safe_mode:
            return get_weather_prompt(language=language, safe_mode=True)

        if not getattr(self, "_system_prompt_dynamic", False):
            override_prompt = getattr(self, "system_prompt", "")
            if override_prompt:
                return override_prompt

        prompt = get_weather_prompt(language=language)
        self.system_prompt = prompt
        return prompt

    @staticmethod
    def _infer_weather_query_language(user_message: str) -> str:
        """Adds a small PT-PT heuristic for short follow-ups where generic language inference is weak."""
        query = (user_message or "").lower()
        pt_markers = [
            "amanhã",
            "amanha",
            "daqui",
            "semana",
            "previsão",
            "previsao",
            "avisos",
            "hoje",
            "tempo",
            "próxim",
            "proxim",
        ]
        if any(marker in query for marker in pt_markers) or re.search(r"[ãõáéíóúç]", query):
            return "pt"
        return infer_response_language(user_query=user_message, default="en")

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
    def _build_messages(
        system_prompt: str,
        user_message: str,
        context: str = "",
        language: str | None = None,
    ) -> list:
        """Builds the message list for a weather invocation.

        Args:
            system_prompt: System prompt text.
            user_message: The user's query.
            context: Additional orchestrator context.
            language: Pre-resolved language code ('pt' or 'en'). When ``None``,
                the language is inferred from *user_message* as a fallback.
        """
        resolved_language = language or WeatherAgent._infer_weather_query_language(user_message)
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
    def _has_english_language_drift(response: str, language: str) -> bool:
        """Detects when an English weather answer still leaks obvious PT-PT content."""
        if language != "en" or not response:
            return False

        drift_patterns = [
            r"\bsegunda-feira\b",
            r"\bterça-feira\b",
            r"\bquarta-feira\b",
            r"\bquinta-feira\b",
            r"\bsexta-feira\b",
            r"\bsábado\b",
            r"\bdomingo\b",
            r"\bchuva\b",
            r"\baguaceiros\b",
            r"\bfraca\b",
            r"\bnoroeste\b",
            r"\bvista casaco\b",
            r"\bguarda-chuva\b",
        ]
        matches = sum(1 for pattern in drift_patterns if re.search(pattern, response, re.IGNORECASE))
        return matches >= 2

    @staticmethod
    def _is_current_weather_query(user_message: str) -> bool:
        """Detects simple current-weather queries that should use the summary tool directly."""
        query = (user_message or "").lower()
        asks_current_wind = bool(
            re.search(r"\b(?:wind|vento)\b", query)
            and re.search(r"\b(?:today|hoje|now|agora|current|atual)\b", query)
        )
        return bool(
            asks_current_wind
            or "right now" in query
            or "current weather summary" in query
            or "current temperature" in query
            or "agora" in query
            or re.search(r"\b(weather|tempo)\b.*\b(today|hoje|now|agora)\b", query)
            or re.search(r"\b(today|hoje)\b.*\b(weather|tempo)\b", query)
        )

    @classmethod
    def _resolve_forecast_window(cls, user_message: str) -> Optional[dict[str, Any]]:
        """Resolve the smallest grounded forecast window implied by a query.

        Returns a dictionary containing ``day_offset`` and ``days`` for the
        IPMA forecast tool. The helper intentionally keeps the output window
        focused: tomorrow means only tomorrow, a named weekday means only that
        weekday, and weekend means the available Saturday/Sunday subset.
        """
        normalized = cls._normalize_weather_query(user_message)
        today = datetime.now().date()

        explicit_date = cls._extract_explicit_forecast_date(user_message)
        if explicit_date is not None:
            day_offset = (explicit_date.date() - today).days
            if 0 <= day_offset <= FORECAST_MAX_OFFSET:
                return {"day_offset": day_offset, "days": 1, "label": "explicit_date"}
            return None

        if cls._has_tomorrow_reference(normalized):
            return {"day_offset": 1, "days": 1, "label": "tomorrow"}

        if any(term in normalized for term in ["tonight", "esta noite", "hoje a noite"]):
            return {"day_offset": 0, "days": 1, "label": "tonight"}

        if any(term in normalized for term in ["today", "hoje", "agora"]) and any(
            term in normalized for term in ["forecast", "previsao", "weather", "tempo", "detalhada", "detailed"]
        ):
            return {"day_offset": 0, "days": 1, "label": "today"}

        if any(term in normalized for term in ["weekend", "fim de semana"]):
            return cls._resolve_weekend_forecast_window()

        weekday_offset = cls._extract_named_weekday_offset(user_message)
        if weekday_offset is not None and 0 <= weekday_offset <= FORECAST_MAX_OFFSET:
            return {"day_offset": weekday_offset, "days": 1, "label": "weekday"}

        explicit_days = re.search(r"\b([1-5])\s*(?:-|\s)?\s*(?:day|days|dia|dias)\b", normalized)
        if explicit_days:
            return {"day_offset": 0, "days": int(explicit_days.group(1)), "label": "range"}

        if any(term in normalized for term in ["week", "this week", "semana", "esta semana"]):
            return {"day_offset": 0, "days": FORECAST_HORIZON_DAYS, "label": "range"}

        if any(term in normalized for term in ["forecast", "previsao", "next days", "proximos dias"]):
            return {"day_offset": 0, "days": 3, "label": "range"}

        return None

    @staticmethod
    def _resolve_weekend_forecast_window() -> Optional[dict[str, Any]]:
        """Return the available Saturday/Sunday forecast subset within IPMA's horizon."""
        weekday = datetime.now().date().weekday()
        if weekday == 5:
            start_offset = 0
            desired_days = 2
        elif weekday == 6:
            start_offset = 0
            desired_days = 1
        else:
            start_offset = (5 - weekday) % 7
            desired_days = 2

        if start_offset > FORECAST_MAX_OFFSET:
            return None

        available_days = min(desired_days, FORECAST_HORIZON_DAYS - start_offset)
        if available_days <= 0:
            return None

        return {
            "day_offset": start_offset,
            "days": available_days,
            "label": "weekend",
            "partial": available_days < desired_days,
        }

    @classmethod
    def _extract_named_weekday_offset(cls, user_message: str) -> Optional[int]:
        """Return the next occurrence offset for a named weekday in PT or EN."""
        normalized = cls._normalize_weather_query(user_message)
        today_index = datetime.now().date().weekday()
        for weekday_name, target_index in sorted(_WEEKDAY_NAME_TO_INDEX.items(), key=lambda item: -len(item[0])):
            if not re.search(rf"\b{re.escape(cls._normalize_weather_query(weekday_name))}\b", normalized):
                continue
            offset = (target_index - today_index) % 7
            if re.search(rf"\bnext\s+{re.escape(cls._normalize_weather_query(weekday_name))}\b", normalized) and offset == 0:
                offset = 7
            return offset
        return None

    @staticmethod
    def _extract_explicit_forecast_date(user_message: str) -> Optional[datetime]:
        """Extracts an explicit forecast date from common numeric or month-name forms."""
        query = user_message or ""
        for pattern, fmt in (
            (r"\b(\d{4}-\d{2}-\d{2})\b", "%Y-%m-%d"),
            (r"\b(\d{2}/\d{2}/\d{4})\b", "%d/%m/%Y"),
        ):
            match = re.search(pattern, query)
            if not match:
                continue
            try:
                return datetime.strptime(match.group(1), fmt)
            except ValueError:
                continue

        normalized = WeatherAgent._normalize_weather_query(query)
        month_pattern = "|".join(re.escape(month) for month in sorted(_MONTH_NAME_TO_NUMBER, key=len, reverse=True))
        month_first = re.search(
            rf"\b({month_pattern})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,)?\s+(\d{{4}})\b",
            normalized,
        )
        day_first = re.search(
            rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:de\s+)?({month_pattern})(?:\s+de)?\s+(\d{{4}})\b",
            normalized,
        )
        for match, month_group, day_group, year_group in (
            (month_first, 1, 2, 3),
            (day_first, 2, 1, 3),
        ):
            if not match:
                continue
            month = _MONTH_NAME_TO_NUMBER.get(match.group(month_group))
            if not month:
                continue
            try:
                return datetime(int(match.group(year_group)), month, int(match.group(day_group)))
            except ValueError:
                continue
        return None

    @staticmethod
    def _normalize_weather_query(user_message: str) -> str:
        """Normalize weather query text for accent-insensitive capability checks."""
        normalized = unicodedata.normalize("NFKD", user_message or "")
        normalized = normalized.encode("ascii", "ignore").decode("ascii").lower()
        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def _has_tomorrow_reference(normalized_query: str) -> bool:
        """Return whether a normalized query refers to tomorrow, tolerating common typos."""
        tokens = re.findall(r"[a-z]+", normalized_query or "")
        if any(token in {"tomorrow", "amanha"} for token in tokens):
            return True
        # Common accent/typing corruptions observed in prompt runs: amahna,
        # amanha with swapped letters, and close edit-distance variants.
        for token in tokens:
            if token.startswith("aman") or token.startswith("amah"):
                if abs(len(token) - len("amanha")) <= 2:
                    return True
            if len(token) >= 5:
                mismatches = sum(1 for a, b in zip(token[:6], "amanha", strict=False) if a != b)
                if mismatches <= 2 and abs(len(token) - 6) <= 1:
                    return True
        return False

    @classmethod
    def _is_portugal_overview_query(cls, user_message: str) -> bool:
        """Return whether the user asks for a country-level Portugal weather overview."""
        normalized = cls._normalize_weather_query(user_message)
        return bool(
            "portugal-wide" in normalized
            or "portugal wide" in normalized
            or "portugal overview" in normalized
            or "country-wide" in normalized
            or "country wide" in normalized
            or re.search(r"\bportugal\b.*\b(weather|tempo|forecast|previsao|overview|resumo)\b", normalized)
            or re.search(r"\b(weather|tempo|forecast|previsao|overview|resumo)\b.*\bportugal\b", normalized)
        )

    @classmethod
    def _extract_unsupported_weather_location(cls, user_message: str) -> str | None:
        """Detect non-Lisbon city-specific weather requests that should not use Lisbon forecast data."""
        normalized = cls._normalize_weather_query(user_message)
        if cls._is_portugal_overview_query(user_message):
            return None

        unsupported_locations = {
            "porto": "Porto",
            "braga": "Braga",
            "coimbra": "Coimbra",
            "aveiro": "Aveiro",
            "faro": "Faro",
            "evora": "Évora",
            "madeira": "Madeira",
            "funchal": "Funchal",
            "acores": "Açores",
            "azores": "Açores",
            "ponta delgada": "Ponta Delgada",
        }
        weather_terms = [
            "weather",
            "tempo",
            "forecast",
            "previsao",
            "rain",
            "chuva",
            "temperature",
            "temperatura",
            "wind",
            "vento",
        ]
        if not any(term in normalized for term in weather_terms):
            return None

        for token, label in unsupported_locations.items():
            if re.search(rf"\b{re.escape(token)}\b", normalized):
                return label
        return None

    @staticmethod
    def _build_unsupported_location_message(location: str, language: str) -> str:
        """Build a localized response for city-specific weather outside LISBOA's grounded scope."""
        if language == "pt":
            return (
                "### 🌤️ **Âmbito Meteorológico**\n\n"
                f"⚠️ Não consigo fornecer uma previsão meteorológica específica para {location} neste sistema. "
                "O **LISBOA está focado em Lisboa/AML**, com previsões IPMA de curto prazo para Lisboa "
                "e, quando pedido, uma visão geral de Portugal. Para essa localidade, consulta diretamente o IPMA."
            )
        return (
            "### 🌤️ **Weather Scope**\n\n"
            f"⚠️ I can't provide a city-specific forecast for {location} in this system. "
            "**LISBOA is focused on Lisbon/AML**, with short-range IPMA forecasts for Lisbon and, when requested, "
            "a Portugal-wide overview. For that location, check IPMA directly."
        )

    @classmethod
    def _is_climate_average_query(cls, user_message: str) -> bool:
        """Return whether the query asks for historical climate averages, not forecasts."""
        normalized = cls._normalize_weather_query(user_message)
        climate_terms = [
            "climate",
            "climatology",
            "climatological",
            "historical weather",
            "weather history",
            "clima",
            "climatologia",
            "historico meteorologico",
        ]
        average_terms = [
            "average",
            "mean",
            "typical",
            "usual",
            "normal",
            "media",
            "medio",
            "tipica",
            "tipico",
            "normalmente",
        ]
        month_or_season_terms = [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
            "janeiro",
            "fevereiro",
            "marco",
            "abril",
            "maio",
            "junho",
            "julho",
            "agosto",
            "setembro",
            "outubro",
            "novembro",
            "dezembro",
            "summer",
            "winter",
            "spring",
            "autumn",
            "verao",
            "inverno",
            "primavera",
            "outono",
        ]
        has_climate_term = any(term in normalized for term in climate_terms)
        has_average_term = any(term in normalized for term in average_terms)
        has_month_or_season = any(term in normalized for term in month_or_season_terms)
        asks_temperature = bool(re.search(r"\b(temperature|temperatura|temp)\b", normalized))
        return has_climate_term or (has_average_term and has_month_or_season and asks_temperature)

    @staticmethod
    def _build_climate_average_limit_message(language: str) -> str:
        """Builds a localized message for unsupported climatology-average requests."""
        if language == "pt":
            return (
                "### 🌤️ **Âmbito Meteorológico**\n\n"
                "⚠️ Não tenho dados climatológicos ou médias históricas neste sistema. "
                "A cobertura meteorológica disponível é previsão IPMA de curto prazo para Lisboa, "
                "até 5 dias, e avisos atuais."
            )
        return (
            "### 🌤️ **Weather Scope**\n\n"
            "⚠️ I don't have climatology data or historical weather averages in this system. "
            "The available weather coverage is short-range IPMA forecasts for Lisbon, "
            "up to 5 days, plus current warnings."
        )

    @classmethod
    def _is_unsupported_weather_data_query(cls, user_message: str) -> bool:
        """Return whether the user asks for a weather field unavailable in current tools."""
        normalized = cls._normalize_weather_query(user_message)
        unsupported_terms = [
            "uv index",
            "indice uv",
            "indice ultravioleta",
            "ultraviolet",
            "air quality",
            "qualidade do ar",
            "pollen",
            "polen",
            "humidity",
            "humidade",
        ]
        return any(term in normalized for term in unsupported_terms)

    @staticmethod
    def _build_unsupported_weather_data_message(language: str) -> str:
        """Build a concise localized response for unsupported weather fields."""
        if language == "pt":
            return (
                "### 🌤️ **Âmbito Meteorológico**\n\n"
                "⚠️ Não consigo confirmar esse indicador com as ferramentas meteorológicas disponíveis. "
                "Neste sistema, tenho acesso a previsões IPMA de curto prazo para Lisboa, avisos meteorológicos, "
                "temperatura, precipitação, condições e vento."
            )
        return (
            "### 🌤️ **Weather Scope**\n\n"
            "⚠️ I can't confirm that indicator with the available weather tools. "
            "In this system, I have access to short-range IPMA forecasts for Lisbon, weather warnings, "
            "temperature, rain, conditions, and wind."
        )

    @classmethod
    def _is_beyond_forecast_horizon_query(cls, user_message: str) -> bool:
        """Returns whether the query clearly asks for weather beyond IPMA's 5-day horizon."""
        query = (user_message or "").lower()

        if re.search(r"\b([6-9]|\d{2,})\s*(?:-|\s)?\s*(?:day|days|dia|dias)\b", query):
            return True

        beyond_horizon_patterns = [
            r"\bin\s+(?:a|one)\s+week\b",
            r"\ba\s+week\s+from\s+now\b",
            r"\bnext\s+week\b",
            r"\bdaqui\s+a\s+uma\s+semana\b",
            r"\bpr[oó]xima\s+semana\b",
            r"\bpr[oó]ximos?\s+7\s+dias\b",
        ]
        if any(re.search(pattern, query) for pattern in beyond_horizon_patterns):
            return True

        explicit_date = cls._extract_explicit_forecast_date(user_message)
        if explicit_date is not None:
            delta_days = (explicit_date.date() - datetime.now().date()).days
            if delta_days < 0 or delta_days > FORECAST_MAX_OFFSET:
                return True

        weekday_offset = cls._extract_named_weekday_offset(user_message)
        if weekday_offset is not None and weekday_offset > FORECAST_MAX_OFFSET:
            return True

        return False

    @staticmethod
    def _build_forecast_horizon_limit_message(language: str) -> str:
        """Builds a localized message when the user asks beyond the 5-day forecast horizon."""
        if language == "pt":
            return (
                "### 🌤️ **Previsão Meteorológica**\n\n"
                "⚠️ Só tenho previsão meteorológica fiável do IPMA para Lisboa para os próximos 5 dias, "
                "por isso não consigo confirmar o tempo para esse horizonte sem inventar dados."
            )
        return (
            "### 🌤️ **Weather Forecast**\n\n"
            "⚠️ I only have reliable IPMA weather forecast data for Lisbon for the next 5 days, "
            "so I can't confirm the weather for that time horizon without inventing data."
        )

    @classmethod
    def _is_simple_forecast_query(cls, user_message: str) -> bool:
        """Detects standalone forecast/warnings queries that can skip free-form synthesis."""
        query = (user_message or "").lower()
        planning_terms = [
            "plan", "itinerary", "roteiro", "plano", "activity", "activities",
            "visit", "visitar", "museum", "museu", "restaurant", "restaurante",
        ]

        if any(term in query for term in planning_terms):
            return False

        return bool(
            cls._resolve_forecast_window(user_message)
            or any(term in query for term in _WARNING_QUERY_TERMS)
            or cls._is_weather_advice_query(user_message)
        )

    @classmethod
    def _is_weather_advice_query(cls, user_message: str) -> bool:
        """Return whether a weather query asks for practical advice based on forecast data."""
        normalized = cls._normalize_weather_query(user_message)
        advice_terms = [
            "jacket",
            "coat",
            "wear",
            "wearing",
            "clothes",
            "clothing",
            "walking",
            "walk outdoors",
            "outdoors",
            "casaco",
            "vestir",
            "roupa",
            "caminhar",
            "ar livre",
            "umbrella",
            "guarda chuva",
            "sailing",
            "sail",
            "vela",
            "boat",
            "barco",
            "safe",
            "seguro",
        ]
        return any(term in normalized for term in advice_terms)

    @classmethod
    def _is_planning_weather_context_query(cls, user_message: str) -> bool:
        """Return whether Weather was called only to supply context for a plan."""
        normalized = cls._normalize_weather_query(user_message)
        return bool(
            re.search(
                r"\b(?:plan|itinerary|roteiro|plano|planeia|cria|create|visit|visitar|tour|monument|monumento|museum|museu|restaurant|restaurante|gastronom)\b",
                normalized,
            )
        )

    def _run_direct_tool_fallback(
        self,
        user_message: str,
        *,
        force_forecast_days: Optional[int] = None,
        include_warnings: bool = False,
    ) -> str:
        """
        Runs a deterministic tool-only fallback when Azure blocks weather prompt
        attempts. This preserves real data access without relying on another
        model call.
        """
        language = self._infer_weather_query_language(user_message)
        if self._is_climate_average_query(user_message):
            return self._build_climate_average_limit_message(language)

        if self._is_beyond_forecast_horizon_query(user_message):
            return self._build_forecast_horizon_limit_message(language)

        unsupported_location = self._extract_unsupported_weather_location(user_message)
        if unsupported_location:
            return self._build_unsupported_location_message(unsupported_location, language)

        if self._is_unsupported_weather_data_query(user_message):
            if any(term in self._normalize_weather_query(user_message) for term in ["today", "hoje", "now", "agora"]):
                current_tool = self._get_tool_by_name("get_current_weather_summary")
                if current_tool:
                    self._invoke_tool(current_tool, {})
            return self._build_unsupported_weather_data_message(language)

        query = self._normalize_weather_query(user_message)
        wants_portugal_overview = self._is_portugal_overview_query(user_message)
        forecast_window = self._resolve_forecast_window(user_message)
        if force_forecast_days is not None:
            forecast_window = {"day_offset": 0, "days": force_forecast_days, "label": "range"}
        requested_forecast_days = forecast_window["days"] if forecast_window else None
        wants_warnings = include_warnings or any(term in query for term in _WARNING_QUERY_TERMS)
        wants_advice = self._is_weather_advice_query(user_message)
        wants_warnings = wants_warnings or any(term in query for term in ["sailing", "sail", "vela", "boat", "barco"])
        wants_forecast = requested_forecast_days is not None or wants_advice
        wants_warnings = wants_warnings or bool(
            wants_forecast and forecast_window and forecast_window.get("label") == "range"
        )
        wants_current = not wants_warnings and not wants_forecast and (
            any(term in query for term in ["today", "current", "now", "hoje", "agora"])
        ) or (
            any(term in query for term in ["weather", "tempo"]) and not wants_warnings and not wants_forecast
        )

        sections = []

        if wants_portugal_overview:
            overview_tool = self._get_tool_by_name("get_portugal_weather_overview")
            if overview_tool:
                sections.append(self._invoke_tool(overview_tool, {"day": 0}))
            if sections:
                return self._compose_direct_weather_response(
                    user_message=user_message,
                    sections=[str(section) for section in sections if section],
                    language=language,
                    forecast_window=forecast_window,
                )

        if wants_warnings:
            warnings_tool = self._get_tool_by_name("get_weather_warnings")
            if warnings_tool:
                sections.append(self._invoke_tool(warnings_tool, {"area": "LSB"}))

        if wants_current:
            current_tool = self._get_tool_by_name("get_current_weather_summary")
            if current_tool:
                sections.append(self._invoke_tool(current_tool, {}))

        forecast_tool = self._get_tool_by_name("get_weather_forecast")
        if forecast_tool and wants_forecast and requested_forecast_days:
            sections.append(
                self._invoke_tool(
                    forecast_tool,
                    {
                        "days": requested_forecast_days,
                        "day_offset": int(forecast_window.get("day_offset", 0)) if forecast_window else 0,
                    },
                )
            )

        if not sections:
            current_tool = self._get_tool_by_name("get_current_weather_summary")
            if current_tool:
                sections.append(self._invoke_tool(current_tool, {}))

        if not sections:
            return "Unable to retrieve weather data at the moment."

        return self._compose_direct_weather_response(
            user_message=user_message,
            sections=[str(section) for section in sections if section],
            language=language,
            forecast_window=forecast_window,
        )

    @classmethod
    def _compose_direct_weather_response(
        cls,
        *,
        user_message: str,
        sections: list[str],
        language: str,
        forecast_window: Optional[dict[str, Any]] = None,
    ) -> str:
        """Prepend a query-specific answer before grounded weather details."""
        body = "\n\n---\n\n".join(section for section in sections if section).strip()
        direct_answer = cls._build_direct_weather_answer(user_message, body, language, forecast_window)
        title = cls._weather_title_for_query(
            user_message,
            language,
            tool_text=body,
            forecast_window=forecast_window,
        )
        if not direct_answer:
            return f"{title}\n\n{body}".strip()
        if not body:
            return f"{title}\n\n{direct_answer}".strip()
        if cls._is_redundant_clear_warning_body(user_message, direct_answer, body):
            return f"{title}\n\n{direct_answer}".strip()
        return f"{title}\n\n{direct_answer}\n\n---\n\n{body}".strip()

    @classmethod
    def _is_redundant_clear_warning_body(cls, user_message: str, direct_answer: str, body: str) -> bool:
        """Return whether a warning tool body only repeats the direct answer."""
        normalized_query = cls._normalize_weather_query(user_message)
        if not any(term in normalized_query for term in _WARNING_QUERY_TERMS):
            return False

        combined_direct = cls._normalize_weather_query(direct_answer)
        combined_body = cls._normalize_weather_query(body)
        clear_warning_pattern = r"no active weather warnings|sem avisos meteorol|n[aã]o h[aá] avisos meteorol"
        direct_says_clear = bool(re.search(clear_warning_pattern, combined_direct))
        body_says_clear = bool(re.search(clear_warning_pattern, combined_body))
        body_has_active_details = bool(re.search(r"yellow|orange|red|amarelo|laranja|vermelho|rough sea|vento|chuva|agita", combined_body))
        return direct_says_clear and body_says_clear and not body_has_active_details

    @classmethod
    def _weather_title_for_query(
        cls,
        user_message: str,
        language: str,
        *,
        tool_text: str = "",
        forecast_window: Optional[dict[str, Any]] = None,
    ) -> str:
        """Build the canonical H3 weather title for the query intent."""
        normalized = cls._normalize_weather_query(user_message)
        asks_warnings = any(term in normalized for term in _WARNING_QUERY_TERMS)
        has_forecast_data = bool(forecast_window) or bool(
            re.search(
                r"\b(?:forecast|previs[aã]o|temperature|temperatura|rain|chuva|wind|vento|humidity|humidade)\b",
                tool_text,
                flags=re.IGNORECASE,
            )
        )
        has_clear_no_warning_status = bool(
            re.search(
                r"no active weather warnings|sem avisos meteorol|n[aã]o h[aá] avisos meteorol",
                tool_text,
                flags=re.IGNORECASE,
            )
        )
        has_active_warning_data = asks_warnings and not has_clear_no_warning_status and bool(
            re.search(
                r"(?:active weather warnings for|avisos meteorol[oó]gicos ativos|🟡|🟠|🔴|yellow|orange|red|amarelo|laranja|vermelho)",
                tool_text,
                flags=re.IGNORECASE,
            )
        )
        if cls._is_portugal_overview_query(user_message):
            title = "Visão Meteorológica de Portugal" if language == "pt" else "Portugal Weather Overview"
        elif has_forecast_data:
            title = "Previsão Meteorológica de Lisboa" if language == "pt" else "Lisbon Weather Forecast"
        elif asks_warnings and has_active_warning_data:
            title = "Avisos meteorológicos ativos" if language == "pt" else "Active Weather Warnings"
        elif asks_warnings:
            title = "Estado dos Avisos Meteorológicos" if language == "pt" else "Weather Warning Status"
        elif cls._is_current_weather_query(user_message):
            title = "Resumo Meteorológico de Lisboa" if language == "pt" else "Lisbon Weather Summary"
        else:
            title = "Previsão Meteorológica" if language == "pt" else "Weather Forecast"
        return f"### 🌤️ **{title}**"

    @classmethod
    def _build_direct_weather_answer(
        cls,
        user_message: str,
        tool_text: str,
        language: str,
        forecast_window: Optional[dict[str, Any]],
    ) -> str:
        """Build a concise first answer tailored to the user's weather question."""
        normalized = cls._normalize_weather_query(user_message)
        is_pt = language == "pt"
        no_warnings = bool(
            re.search(
                r"no active weather warnings|sem avisos meteorol|n[aã]o h[aá] avisos meteorol",
                tool_text,
                re.IGNORECASE,
            )
        )

        if any(term in normalized for term in _WARNING_QUERY_TERMS):
            if "weekend" in normalized or "fim de semana" in normalized:
                coverage = cls._weekend_coverage_sentence(forecast_window, is_pt)
                warning_answer = (
                    "✅ Não há **avisos meteorológicos ativos** para Lisboa neste momento."
                    if is_pt and no_warnings
                    else "✅ No, there are **no active weather warnings** for Lisbon right now."
                    if no_warnings
                    else "⚠️ Há avisos meteorológicos ativos para Lisboa."
                    if is_pt
                    else "⚠️ There are active weather warnings for Lisbon."
                )
                return f"{warning_answer}\n\n{coverage}".strip()
            out_of_horizon_day = cls._named_day_outside_horizon_sentence(user_message, is_pt)
            if no_warnings:
                warning_answer = (
                    "✅ Não, não há **avisos meteorológicos ativos** para Lisboa neste momento."
                    if is_pt
                    else "✅ No, there are **no active weather warnings** for Lisbon right now."
                )
                return f"{warning_answer}\n\n{out_of_horizon_day}".strip()
            warning_answer = (
                "⚠️ Sim, há **avisos meteorológicos ativos** para Lisboa neste momento."
                if is_pt
                else "⚠️ Yes, there are **active weather warnings** for Lisbon right now."
            )
            return f"{warning_answer}\n\n{out_of_horizon_day}".strip()

        if "tonight" in normalized or "esta noite" in normalized or "hoje a noite" in normalized:
            minimum = cls._extract_temperature_min(tool_text)
            if minimum:
                return (
                    f"🌙 Esta noite, Lisboa deverá descer até cerca de **{minimum}°C**."
                    if is_pt
                    else f"🌙 Tonight should get down to about **{minimum}°C** in Lisbon."
                )

        practical_advice_terms = [
            "suitable",
            "avoid",
            "good for",
            "wear",
            "wearing",
            "clothes",
            "clothing",
            "walking",
            "walk",
            "walk outdoors",
            "riverside",
            "outdoors",
            "adequado",
            "evitar",
            "bom para",
            "vestir",
            "roupa",
            "caminhar",
            "passeio",
            "andar ao ar livre",
            "ar livre",
            "umbrella",
            "guarda chuva",
        ]

        if (
            ("rain" in normalized or "chuva" in normalized or "chover" in normalized)
            and not any(term in normalized for term in practical_advice_terms)
        ):
            rain = cls._extract_rain_summary(tool_text)
            if rain:
                return cls._rain_direct_answer(rain, is_pt)

        if any(term in normalized for term in practical_advice_terms):
            minimum = cls._extract_temperature_min(tool_text)
            maximum = cls._extract_temperature_max(tool_text)
            rain = cls._extract_rain_summary(tool_text)
            wind = cls._extract_wind_summary(tool_text)
            rain_answer = f"{cls._rain_direct_answer(rain, is_pt)}\n\n" if rain else ""
            if is_pt:
                advice_parts = ["leva **casaco leve**"]
                if rain and float(rain.get("probability", 0) or 0) >= 35:
                    advice_parts.append("guarda-chuva compacto ou impermeável")
                if wind:
                    advice_parts.append("uma camada que corte o vento")
                temperature_note = f" porque a previsão fica entre **{minimum}°C e {maximum}°C**" if minimum and maximum else ""
                suitability = "Parece adequado para caminhar" if any(term in normalized for term in ["adequado", "bom para", "evitar", "passeio"]) else "Para caminhar ao ar livre"
                return f"{rain_answer}👟 {suitability}, {', '.join(advice_parts)}{temperature_note}."
            advice_parts = ["wear a **light jacket**"]
            if rain and float(rain.get("probability", 0) or 0) >= 35:
                advice_parts.append("carry a compact umbrella or rain shell")
            if wind:
                advice_parts.append("add a wind-resistant layer")
            temperature_note = f" because the forecast is around **{minimum}°C to {maximum}°C**" if minimum and maximum else ""
            suitability = "It looks suitable for a walk" if any(term in normalized for term in ["suitable", "good for", "avoid", "riverside"]) else "For walking outdoors"
            return f"{rain_answer}👟 {suitability}, {', '.join(advice_parts)}{temperature_note}."

        if "wind" in normalized or "vento" in normalized:
            wind = cls._extract_wind_summary(tool_text)
            if wind:
                future_reference = (
                    "tomorrow" in normalized
                    or "amanha" in normalized
                    or "amanhã" in normalized
                    or "saturday" in normalized
                    or "sunday" in normalized
                    or "sabado" in normalized
                    or "sábado" in normalized
                    or "domingo" in normalized
                )
                time_label_pt = "Na data pedida" if future_reference else "Hoje"
                time_label_en = "For the requested day" if future_reference else "Today"
                return (
                    f"💨 {time_label_pt}, o vento em Lisboa está de **{wind}**."
                    if is_pt
                    else f"💨 {time_label_en}, Lisbon's wind is **{wind}**."
                )

        if any(term in normalized for term in ["jacket", "coat", "casaco"]):
            minimum = cls._extract_temperature_min(tool_text)
            rain = cls._extract_rain_summary(tool_text)
            if is_pt:
                return f"🧥 Sim, leva um **casaco leve**{cls._jacket_reason(minimum, rain, is_pt)}."
            return f"🧥 Yes, bring a **light jacket**{cls._jacket_reason(minimum, rain, is_pt)}."

        if any(term in normalized for term in ["sailing", "sail", "vela", "boat", "barco", "safe", "seguro"]):
            wind = cls._extract_wind_summary(tool_text)
            if is_pt:
                return (
                    "⛵ Não consigo certificar se é seguro velejar só com estes dados meteorológicos urbanos. "
                    f"Usa os avisos, a chuva e o vento{f' ({wind})' if wind else ''} apenas como contexto e confirma sempre a previsão marítima oficial antes de sair."
                )
            return (
                "⛵ I cannot certify sailing safety from these urban weather data alone. "
                f"Use warnings, rain, and wind{f' ({wind})' if wind else ''} only as context, and check the official marine forecast before departing."
            )

        if cls._is_portugal_overview_query(user_message):
            return (
                "🇵🇹 Aqui está uma visão geral meteorológica de Portugal com Lisboa destacada."
                if is_pt
                else "🇵🇹 Here is a bounded Portugal-wide weather overview, with Lisbon highlighted."
            )

        return (
            "✅ **Resposta direta:** Aqui está a previsão meteorológica disponível para Lisboa."
            if is_pt
            else "✅ **Direct answer:** Here is the available weather information for Lisbon."
        )

    @staticmethod
    def _weekend_coverage_sentence(forecast_window: Optional[dict[str, Any]], is_pt: bool) -> str:
        """Explain weekend forecast coverage when Saturday/Sunday are partly outside the horizon."""
        if not forecast_window:
            return (
                "⚠️ A previsão para sábado/domingo ainda está fora do horizonte IPMA de 5 dias."
                if is_pt
                else "⚠️ The Saturday/Sunday forecast is still outside IPMA's 5-day horizon."
            )
        if forecast_window.get("partial"):
            return (
                "⚠️ O IPMA só cobre parte do fim de semana neste momento; o restante fica fora do horizonte de 5 dias."
                if is_pt
                else "⚠️ IPMA only covers part of the weekend right now; the rest is outside the 5-day horizon."
            )
        return (
            "✅ A previsão disponível cobre o fim de semana dentro do horizonte IPMA."
            if is_pt
            else "✅ The available forecast covers the weekend within IPMA's horizon."
        )

    @classmethod
    def _named_day_outside_horizon_sentence(cls, user_message: str, is_pt: bool) -> str:
        """Explain when a requested named day is outside IPMA's forecast horizon."""
        weekday_offset = cls._extract_named_weekday_offset(user_message)
        if weekday_offset is None or weekday_offset <= FORECAST_MAX_OFFSET:
            return ""
        return (
            "⚠️ O dia pedido ainda está fora do horizonte IPMA de 5 dias, por isso não consigo confirmar avisos específicos para essa data."
            if is_pt
            else "⚠️ The requested day is still outside IPMA's 5-day horizon, so I can't confirm date-specific warnings yet."
        )

    @staticmethod
    def _extract_temperature_min(tool_text: str) -> Optional[str]:
        """Extract the minimum temperature from a weather tool response."""
        match = re.search(r"(?:Temperature|Temperatura)?\D*(\d+(?:\.\d+)?)°C\s+(?:to|a)\s+\d+(?:\.\d+)?°C", tool_text, re.IGNORECASE)
        return match.group(1) if match else None

    @staticmethod
    def _extract_temperature_max(tool_text: str) -> Optional[str]:
        """Extract the maximum temperature from a weather tool response."""
        match = re.search(r"(?:Temperature|Temperatura)?\D*\d+(?:\.\d+)?°C\s+(?:to|a)\s+(\d+(?:\.\d+)?)°C", tool_text, re.IGNORECASE)
        return match.group(1) if match else None

    @staticmethod
    def _extract_wind_summary(tool_text: str) -> Optional[str]:
        """Extract a compact wind summary from a weather tool response."""
        match = re.search(r"💨\s*(?:\*\*)?(?:Wind|Vento)(?:\*\*)?:\s*([^\n]+)", tool_text, re.IGNORECASE)
        if not match:
            return None
        return match.group(1).strip().rstrip(".")

    @staticmethod
    def _extract_rain_summary(tool_text: str) -> Optional[dict[str, Any]]:
        """Extract rain probability and qualitative wording from a weather response."""
        match = re.search(
            r"💧\s*(?:\*\*)?(?:Rain|Chuva|Rain probability|Probabilidade de chuva)(?:\*\*)?:\s*([^\n]*?)(\d+(?:\.\d+)?)%([^\n]*)",
            tool_text,
            re.IGNORECASE,
        )
        if not match:
            return None
        return {
            "label": re.sub(r"[()|]", " ", f"{match.group(1)} {match.group(3)}").strip(),
            "probability": float(match.group(2)),
        }

    @staticmethod
    def _rain_direct_answer(rain: dict[str, Any], is_pt: bool) -> str:
        """Build a direct rain answer from probability data."""
        probability = rain["probability"]
        if is_pt:
            if probability < 20:
                return f"☔ Não deverá chover em Lisboa; a probabilidade é **{probability:g}%**."
            if probability < 60:
                return f"☔ Pode chover em Lisboa; a probabilidade é **{probability:g}%**."
            return f"☔ Sim, a chuva é provável em Lisboa; a probabilidade é **{probability:g}%**."
        if probability < 20:
            return f"☔ Rain is unlikely in Lisbon; the probability is **{probability:g}%**."
        if probability < 60:
            return f"☔ Rain is possible in Lisbon; the probability is **{probability:g}%**."
        return f"☔ Yes, rain is likely in Lisbon; the probability is **{probability:g}%**."

    @staticmethod
    def _jacket_reason(minimum: Optional[str], rain: Optional[dict[str, Any]], is_pt: bool) -> str:
        """Build a short jacket rationale from minimum temperature and rain probability."""
        reasons = []
        if minimum:
            reasons.append(f"mínima de cerca de {minimum}°C" if is_pt else f"a low around {minimum}°C")
        if rain and rain["probability"] >= 40:
            reasons.append("chuva possível" if is_pt else "possible rain")
        if not reasons:
            return ""
        return " porque há " + " e ".join(reasons) if is_pt else " because there is " + " and ".join(reasons)

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
    def _build_tool_calls(tool_specs: list[tuple[str, dict]]) -> AIMessage:
        """Create a deterministic multi-tool call message for the subgraph."""
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": name,
                    "args": args,
                    "id": f"auto_{uuid.uuid4().hex}",
                    "type": "tool_call",
                }
                for name, args in tool_specs
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
        """Ensures weather subgraph LLM calls receive system and language instructions."""
        updated_messages = list(messages)
        if not updated_messages or not isinstance(updated_messages[0], SystemMessage):
            updated_messages = [SystemMessage(content=self._get_runtime_system_prompt(language))] + updated_messages

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
        """Routes obvious weather queries to their canonical tool in the subgraph."""
        query = user_message.lower().strip()

        if cls._is_portugal_overview_query(user_message):
            return cls._build_tool_call("get_portugal_weather_overview", {"day": 0})

        if any(term in query for term in _WARNING_QUERY_TERMS):
            return cls._build_tool_call("get_weather_warnings", {"area": "LSB"})

        forecast_window = cls._resolve_forecast_window(user_message)
        normalized = cls._normalize_weather_query(user_message)
        safety_terms = ["sailing", "sail", "vela", "boat", "barco", "safe", "seguro"]
        if forecast_window and any(term in normalized for term in safety_terms):
            forecast_args = {
                "days": int(forecast_window.get("days", 1)),
                "day_offset": int(forecast_window.get("day_offset", 0)),
            }
            return cls._build_tool_calls(
                [
                    ("get_weather_warnings", {"area": "LSB"}),
                    ("get_weather_forecast", forecast_args),
                ]
            )

        if forecast_window:
            return cls._build_tool_call(
                "get_weather_forecast",
                {
                    "days": int(forecast_window.get("days", 1)),
                    "day_offset": int(forecast_window.get("day_offset", 0)),
                },
            )

        if (
            "current weather summary" in query
            or "right now" in query
            or ("weather" in query and "today" in query)
            or (
                re.search(r"\b(?:wind|vento)\b", query)
                and re.search(r"\b(?:today|hoje|now|agora|current|atual)\b", query)
            )
        ):
            return cls._build_tool_call("get_current_weather_summary", {})

        return None

    @traceable(name="weather_agent", run_type="chain", tags=["sub-agent", "weather"])
    def invoke(
        self, user_message: str, context: str = "", verbose: bool = False
    ) -> str:
        """
        Processes a weather-related query.

        Args:
            user_message: The user's query.
            context: Additional context from other agents (optional).
            verbose: Whether involved tool calls should be printed.

        Returns:
            str: Weather information response.
        """
        # Extract explicit language preference from context if provided
        import re
        language_match = re.search(r"User language:\s*(en|pt)", context, re.IGNORECASE)
        if language_match:
            language = language_match.group(1).lower()
        else:
            language = self._infer_weather_query_language(user_message)
        if self._is_climate_average_query(user_message):
            return finalize_worker_response(
                self._build_climate_average_limit_message(language),
                agent_name="weather",
                user_query=user_message,
                language=language,
            )

        if self._is_unsupported_weather_data_query(user_message):
            return finalize_worker_response(
                self._run_direct_tool_fallback(user_message),
                agent_name="weather",
                user_query=user_message,
                language=language,
            )

        if self._is_beyond_forecast_horizon_query(user_message):
            return finalize_worker_response(
                self._build_forecast_horizon_limit_message(language),
                agent_name="weather",
                user_query=user_message,
                language=language,
            )

        unsupported_location = self._extract_unsupported_weather_location(user_message)
        if unsupported_location:
            return finalize_worker_response(
                self._build_unsupported_location_message(unsupported_location, language),
                agent_name="weather",
                user_query=user_message,
                language=language,
            )

        if self._is_portugal_overview_query(user_message):
            response = self._run_direct_tool_fallback(user_message)
            return finalize_worker_response(
                response,
                agent_name="weather",
                user_query=user_message,
                language=language,
            )

        if self._is_planning_weather_context_query(user_message):
            context_query = (
                "Qual é a previsão para Lisboa hoje? Existem avisos ativos?"
                if language == "pt"
                else "What is today's Lisbon weather forecast? Are there active warnings?"
            )
            response = self._run_direct_tool_fallback(context_query, include_warnings=True)
            return finalize_worker_response(
                response,
                agent_name="weather",
                user_query=context_query,
                language=language,
            )

        system_prompt = self._get_runtime_system_prompt(language)
        messages = self._build_messages(system_prompt, user_message, context, language=language)
        tool_enforcement_msg = (
            "You MUST use a tool (like get_current_weather_summary) to get real data. "
            "Do NOT answer from your knowledge base. Call the tool now."
        )

        if self._is_current_weather_query(user_message) or self._is_simple_forecast_query(user_message):
            response = self._run_direct_tool_fallback(user_message)
            return finalize_worker_response(
                response,
                agent_name="weather",
                user_query=user_message,
                language=language,
            )

        try:
            response = self.execute_react_loop(
                messages=messages,
                verbose=verbose,
                max_iterations=5,
                tool_enforcement_msg=tool_enforcement_msg,
            )
        except Exception as e:
            if not self._is_content_filter_error(e):
                raise

            if verbose:
                print("      [WEATHER] Retrying with safe prompt variant after content filter...")

            safe_messages = self._build_messages(
                self._get_runtime_system_prompt(language, safe_mode=True),
                user_message,
                context,
                language=language,
            )
            try:
                response = self.execute_react_loop(
                    messages=safe_messages,
                    verbose=verbose,
                    max_iterations=5,
                    tool_enforcement_msg=tool_enforcement_msg,
                )
            except Exception as safe_error:
                if not self._is_content_filter_error(safe_error):
                    raise

                if verbose:
                    print("      [WEATHER] Falling back to direct tool invocation after repeated content-filter blocks...")

                response = self._run_direct_tool_fallback(user_message)

        if self._has_english_language_drift(response, language):
            if verbose:
                print("      [WEATHER] Detected language drift in EN response, switching to deterministic tool output...")
            response = self._run_direct_tool_fallback(user_message)

        return finalize_worker_response(
            response,
            agent_name="weather",
            user_query=user_message,
            language=language,
        )

    def build_subgraph(self) -> "CompiledStateGraph":
        """
        Builds a LangGraph subgraph for this agent.

        Returns:
            CompiledStateGraph: Compiled subgraph for weather queries.
        """

        def agent_node(state: AgentState) -> dict:
            """Weather agent decision node."""
            messages = list(state["messages"])

            user_message = None
            for message in reversed(messages):
                if isinstance(message, HumanMessage) and message.content:
                    user_message = str(message.content)
                    break

            language = self._infer_weather_query_language(user_message or "")

            if user_message and self._is_climate_average_query(user_message):
                return {
                    "messages": [
                        AIMessage(
                            content=self._build_climate_average_limit_message(language)
                        )
                    ]
                }

            if user_message and self._is_unsupported_weather_data_query(user_message):
                return {
                    "messages": [
                        AIMessage(
                            content=self._build_unsupported_weather_data_message(language)
                        )
                    ]
                }

            if user_message and self._is_beyond_forecast_horizon_query(user_message):
                return {
                    "messages": [
                        AIMessage(
                            content=self._build_forecast_horizon_limit_message(language)
                        )
                    ]
                }

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

        # Build graph
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
    print("\033[1m🧪 Weather Agent Test\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")

    try:
        agent = WeatherAgent()
        print(f"\n\033[1m✅ Weather Agent initialized:\033[0m {agent.get_model_info()}")
        print(f"   Tools: {[t.name for t in agent.tools]}")

        print("\n\033[1m📝 Testing query:\033[0m 'What is the weather in Lisbon?'")
        response = agent.invoke("What is the weather in Lisbon?")
        print("\n\033[1m🤖 Response:\033[0m")
        print(response)

        print("\n\033[1;32m✅ Weather agent working!\033[0m")

    except Exception as e:
        print(f"\n\033[1;31m❌ Error:\033[0m {e}")
        import traceback

        traceback.print_exc()
