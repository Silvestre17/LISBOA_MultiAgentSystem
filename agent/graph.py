# ==========================================================================
# Master Thesis - LangGraph Agent
#   - André Filipe Gomes Silvestre, 20240502
#
#   Implements the Lisbon Urban Assistant using LangGraph.
#   Features:
#     - ReAct agent pattern with tool calling
#     - State management for context persistence
#     - Multiple specialized tools for Lisbon data
# ==========================================================================

# Required libraries:
# pip install langgraph langchain-core

import json
import logging
import re
import unicodedata

# Always need as_completed for collecting parallel results
import time as time_module
from concurrent.futures import as_completed
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)

from agent.agents.base import (
    clean_response,
    is_local_provider,
)
from agent.utils.langsmith_tracing import (
    LANGSMITH_AVAILABLE,
    ContextThreadPoolExecutor,
    annotate_current_run,
    get_langsmith_request_tracking_status,
    traceable,
)

# Response formatting for Streamlit rendering
from agent.utils.response_formatter import (
    build_bilingual_note,
    build_bounded_planning_framework,
    canonicalize_planner_source_line,
    canonicalize_transport_terms,
    enforce_language_labels,
    ensure_response_title,
    final_post_qa_guard,
    final_visual_pass,
    finalize_worker_response,
    has_source_line,
    ensure_requested_area_limitation,
    is_overcomplex_planning_request,
    format_response,
    generate_response_title,
    operators_from_tool_names,
    reconcile_researcher_place_response,
    rebuild_transport_source_line,
    resolve_output_language,
    strip_excluded_place_cards,
)
from agent.utils.usage_costs import (
    build_cost_payload,
    build_usage_payload,
    get_pricing_metadata,
    load_pricing_catalog,
)

try:
    from config import Config  # For model info without extra LLM instantiation
except ModuleNotFoundError:
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from config import Config

from tools.carris_api import (
    carris_find_routes_between,
    carris_get_arrivals,
    carris_get_next_departures,
    carris_get_realtime_vehicles,
    carris_get_routes,
    carris_get_service_frequency,
    carris_get_stops,
    carris_vehicle_eta,
)

# Carris Metropolitana (Suburban buses)
from tools.carrismetropolitana_api import (
    find_bus_routes,
    find_direct_bus_lines,
    get_bus_next_departures,
    get_bus_realtime_locations,
    get_carris_metropolitana_alerts,
    get_carris_metropolitana_stop_info,
    get_real_time_bus_positions,
    search_carris_metropolitana_lines,
)

# CP (Comboios de Portugal) - Trains
from tools.cp_api import (
    get_cp_routes,
    get_train_frequency,
    get_train_schedule,
    get_train_status,
    plan_train_trip,
    search_cp_stations,
)
from tools.dados_abertos import (
    find_nearby_services,
    find_place_in_datasets,
    get_dataset_details,
    list_available_datasets,
    list_service_categories,
)

# Import tools
from tools.ipma_api import (
    get_current_weather_summary,
    get_portugal_weather_overview,
    get_weather_forecast,
    get_weather_warnings,
)

# Metro de Lisboa (Official API with OAuth2)
from tools.metrolisboa_api import (
    find_nearest_metro,
    get_all_metro_stations,
    get_metro_frequency,
    get_metro_line_wait_times,
    get_metro_status,
    get_metro_wait_time,
)

# Multi-modal transport routing
from tools.transport_api import get_route_between_stations, get_transport_summary
from tools.visitlisboa_api import (
    get_event_categories,
    get_place_categories,
    search_cultural_events,
    search_lisbon_knowledge,
    search_places_attractions,
)

# Web Knowledge (History, Culture, Real-time facts)
from tools.web_knowledge import search_history_culture
import contextlib

# Number of previous messages included in QA conversation_history context
_QA_HISTORY_WINDOW = 6
# Max characters per message used in QA history preview
_QA_MSG_PREVIEW_LEN = 200
# Hard cap for stored user/assistant turns to prevent unbounded session growth.
_MAX_CONVERSATION_HISTORY_MESSAGES = 60
# Maximum time to wait for all parallel workers before collecting partial results.
_WORKER_BATCH_TIMEOUT_S = 120

# ==========================================================================
# Tool Configuration
# ==========================================================================
# Note: Response cleaning utility (clean_response) is imported from
# agent.agents.base to avoid code duplication


def get_all_tools() -> List:
    """
    Returns all available tools for the agent.

    Total: 45 tools across the main categories (Weather, Transport, Open Data,
        VisitLisboa, Carris Urban, Web Knowledge).

    Returns:
        List: List of 45 LangChain tools.
    """
    return [
        # Weather Tools (IPMA) - 4 tools
        get_weather_warnings,
        get_weather_forecast,
        get_current_weather_summary,
        get_portugal_weather_overview,

        # Transport - Metro de Lisboa - 6 tools
        get_metro_status,
        get_metro_wait_time,
        get_metro_line_wait_times,
        find_nearest_metro,
        get_metro_frequency,
        get_all_metro_stations,

        # Transport - Carris Metropolitana (Suburban buses) - 8 tools
        get_carris_metropolitana_alerts,
        get_carris_metropolitana_stop_info,
        search_carris_metropolitana_lines,
        find_bus_routes,
        get_real_time_bus_positions,
        get_bus_realtime_locations,
        get_bus_next_departures,
        find_direct_bus_lines,

        # Transport - Carris Urban (Lisbon city buses & trams) - 8 tools
        carris_get_stops,
        carris_get_routes,
        carris_get_next_departures,
        carris_find_routes_between,
        carris_get_realtime_vehicles,
        carris_get_arrivals,
        carris_vehicle_eta,
        carris_get_service_frequency,

        # Transport - CP (Comboios de Portugal) - 6 tools
        get_train_status,
        search_cp_stations,
        get_train_schedule,
        get_cp_routes,
        plan_train_trip,
        get_train_frequency,

        # Transport - Multi-modal - 2 tools
        get_transport_summary,
        get_route_between_stations,

        # Open Data Tools (Lisboa Aberta) - 5 tools
        find_nearby_services,
        list_available_datasets,
        get_dataset_details,
        find_place_in_datasets,
        list_service_categories,

        # VisitLisboa Tools (Events & Places) - 5 tools
        search_cultural_events,
        search_places_attractions,
        get_event_categories,
        get_place_categories,
        search_lisbon_knowledge,

        # Web Knowledge - 1 tool
        search_history_culture,
    ]


# ==========================================================================
# Agent Interface
# ==========================================================================


def _print_final_markdown_response(final_output: str) -> None:
    """Print the final Markdown response while tolerating legacy consoles."""
    import builtins
    import sys

    def _safe_print(value: object = "") -> None:
        """Print final markdown without failing on legacy terminal encodings."""
        try:
            builtins.print(value)
        except UnicodeEncodeError:
            encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
            text = str(value).encode(encoding, errors="replace").decode(
                encoding,
                errors="replace",
            )
            builtins.print(text)

    print = _safe_print

    print("=" * 80)
    print("📝 FINAL RESPONSE (Markdown)")
    print("=" * 80)
    print(final_output)
    print("=" * 80 + "\n")

# ==========================================================================
# Multi-Agent System
# ==========================================================================


class MultiAgentAssistant:
    """
    Multi-Agent Lisbon Urban Assistant.

    Uses a Supervisor agent to route queries to specialized agents:
        - WeatherAgent: IPMA weather data
        - TransportAgent: Metro, bus, train information
        - ResearcherAgent: RAG for places and events
        - PlannerAgent: Itinerary synthesis

    Key features:
        - Smart routing: Only calls agents that are needed
        - Direct responses for simple queries (greetings, general chat)
        - Parallel agent execution for complex queries
        - Configurable models per agent via config.py
    """

    def __init__(self):
        """Initializes the multi-agent assistant."""
        from agent.agents.planner_agent import PlannerAgent
        from agent.agents.qa_agent import QualityAssuranceAgent
        from agent.agents.researcher_agent import ResearcherAgent
        from agent.agents.supervisor import SupervisorAgent
        from agent.agents.transport_agent import TransportAgent
        from agent.agents.weather_agent import WeatherAgent
        from agent.state import create_initial_state

        # Initialize agents
        self.supervisor = SupervisorAgent()
        self.qa_agent = QualityAssuranceAgent()
        self.agents = {
            "weather": WeatherAgent(),
            "transport": TransportAgent(),
            "researcher": ResearcherAgent(),
            "planner": PlannerAgent(),
        }

        # Initialize state
        self.state = create_initial_state()
        self.last_execution_summary: Dict[str, Any] | None = None

        self.model_info = {
            "supervisor": self.supervisor.get_model_info(),
            "qa": self.qa_agent.get_model_info(),
            **{name: agent.get_model_info() for name, agent in self.agents.items()},
        }

    @property
    def model_name(self) -> str:
        """Returns the active supervisor model name for display."""
        sv_info = self.model_info.get("supervisor", {})
        if isinstance(sv_info, dict):
            sv_model = sv_info.get("model", "Unknown")
        else:
            sv_model = str(sv_info) if sv_info else "Unknown"
        return f"Multi-Agent ({sv_model})"

    @staticmethod
    def _safe_metric_int(value: object) -> int:
        """Safely coerces simple numeric metrics while defaulting mocks to zero."""
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return 0
        return 0

    @classmethod
    def _normalize_usage_summary(cls, summary: object) -> Dict[str, Any]:
        """Returns a defensive usage-summary shape for partially mocked agents in tests."""
        if not isinstance(summary, dict):
            return {
                "call_count": 0,
                "usage_available": False,
                "model_id": "Unknown",
                "tokens": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
                "llm_usage_breakdown": [],
            }

        tokens = summary.get("tokens", {})
        if not isinstance(tokens, dict):
            tokens = {}

        breakdown = summary.get("llm_usage_breakdown", [])
        if not isinstance(breakdown, list):
            breakdown = []

        return {
            "call_count": cls._safe_metric_int(summary.get("call_count", 0)),
            "usage_available": bool(summary.get("usage_available", False)),
            "model_id": str(summary.get("model_id") or "Unknown"),
            "tokens": {
                "input_tokens": cls._safe_metric_int(tokens.get("input_tokens", 0)),
                "output_tokens": cls._safe_metric_int(tokens.get("output_tokens", 0)),
                "total_tokens": cls._safe_metric_int(tokens.get("total_tokens", 0)),
            },
            "llm_usage_breakdown": breakdown,
        }

    @staticmethod
    def _normalize_tool_calls_log(tool_log: object) -> List[Dict[str, Any]]:
        """Returns a list-shaped tool log even when tests inject plain mocks."""
        return list(tool_log) if isinstance(tool_log, list) else []

    @staticmethod
    def _build_orchestration_failure_fallback(
        message: str,
        language: str,
        attempted_agents: Optional[List[str]] = None,
    ) -> str:
        """Build a limitation-safe fallback when orchestration fails before finalization.

        The fallback preserves language consistency, avoids exposing internals, and
        clearly states that live checks could not be confirmed for this run.
        """
        normalized = (message or "").lower()
        attempted_agents = attempted_agents or []
        has_weather = any(
            keyword in normalized
            for keyword in (
                "weather",
                "tempo",
                "rain",
                "chuva",
                "umbrella",
                "previs",
                "forecast",
                "aviso",
                "warn",
            )
        )
        has_transport = any(
            keyword in normalized
            for keyword in (
                "metro",
                "bus",
                "comboio",
                "train",
                "autocarro",
                "tram",
                "carris",
                "carris",
                "cp",
                "route",
                "rota",
            )
        )
        has_research = any(
            keyword in normalized
            for keyword in (
                "museum",
                "museu",
                "restaurante",
                "event",
                "evento",
                "park",
                "parque",
                "pharmacy",
                "farmácia",
                "farmacia",
                "parking",
                "police",
                "polícia",
                "library",
                "biblioteca",
            )
        )

        attempted = ", ".join(attempted_agents) if attempted_agents else "none"
        if language == "pt":
            if has_weather and has_transport:
                return (
                    "### ⚠️ **Dados em Tempo Real Não Confirmados**\n\n"
                    "Não consegui confirmar dados em tempo real de **meteorologia** e **transportes** para esta pergunta. "
                    "Não vou inventar horários, avisos ou percursos.\n\n"
                    "**Ainda posso ajudar com:**\n"
                    "- Explicar alternativas de planeamento com base em contexto de Lisboa\n"
                    "- Organizar opções de visita, refeições e mobilidade sem afirmar dados ao vivo\n"
                    "- Reprocessar a pergunta para tentar confirmar meteorologia, transportes ou serviços"
                )
            if has_weather:
                return (
                    "### ⚠️ **Meteorologia Não Confirmada**\n\n"
                    "Neste momento não consigo confirmar condições de **meteorologia** para Lisboa em tempo real. "
                    "Não vou inventar temperatura, chuva ou avisos.\n\n"
                    "**Posso ajudar com:**\n"
                    "- Planos e alternativas de percurso\n"
                    "- Opções de locais e serviços turísticos\n"
                    "- Nova tentativa de confirmação meteorológica"
                )
            if has_transport:
                return (
                    "### ⚠️ **Transportes Não Confirmados**\n\n"
                    "Neste momento não consigo confirmar dados de **transportes em tempo real** para esta consulta. "
                    "Não vou inventar ligações, frequências nem atrasos.\n\n"
                    "**Posso ainda ajudar com:**\n"
                    "- Estruturas de orientação e lógica de ligação por Lisboa\n"
                    "- Nova tentativa de confirmação operacional"
                )
            if has_research:
                return (
                    "### ⚠️ **Dados de Lisboa Não Consolidados**\n\n"
                    "Posso ajudar com recomendações de Lisboa, mas nesta execução não consegui consolidar dados suficientes com qualidade.\n"
                    "Vou evitar afirmar factos que não consigo confirmar agora.\n\n"
                    "Repete a pergunta para uma nova validação dos locais, serviços ou agenda."
                )
            if attempted and ("planner" in attempted):
                return (
                    "### ⚠️ **Plano Não Consolidado**\n\n"
                    "Não consegui consolidar o plano final nesta execução por uma falha temporária no processamento.\n"
                    "Não vou inventar um itinerário fechado.\n\n"
                    "Volta a submeter o pedido para reprocessar com validação operacional."
                )
            return (
                "### ⚠️ **Dados Não Confirmados**\n\n"
                "Não consigo confirmar as fontes necessárias de Lisboa neste momento e não vou inventar detalhes indisponíveis.\n\n"
                "Volta a perguntar em breve para eu tentar novamente com dados atualizados."
            )
        return (
            "### ⚠️ **Operational Notice**\n\n"
            "I cannot confirm all live Lisbon sources for this request right now, and I won’t invent unavailable details.\n\n"
            "To stay reliable, I’m stopping with a caution-only response.\n"
            "Please retry in a few moments for the live-checked answer."
        )

    @staticmethod
    def _dedupe_preserve_order(items: List[str]) -> List[str]:
        """Removes duplicates while preserving order."""
        deduped: List[str] = []
        for item in items:
            if item and item not in deduped:
                deduped.append(item)
        return deduped

    def _agent_uses_local_provider(self, agent_name: str) -> bool:
        """Returns whether the given worker agent is backed by a local provider."""
        agent = self.agents.get(agent_name)
        provider = getattr(agent, "llm_provider", "") if agent is not None else ""
        return is_local_provider(provider)

    def _should_execute_agent_batch_in_parallel(self, agent_names: List[str]) -> bool:
        """Returns whether a worker batch should run in parallel without overloading local runtimes."""
        if len(agent_names) <= 1:
            return False

        return not any(self._agent_uses_local_provider(agent_name) for agent_name in agent_names)

    @classmethod
    def _get_agent_specific_qa_feedback(
        cls,
        qa_result: Optional[Dict[str, object]],
        agent_name: str,
    ) -> List[str]:
        """Extracts agent-specific QA issues and warnings from per-agent fact checks."""
        if not qa_result:
            return []

        fact_check = qa_result.get("fact_check", {})
        if not isinstance(fact_check, dict):
            return []

        per_agent = fact_check.get("per_agent", {})
        if not isinstance(per_agent, dict):
            return []

        agent_fact_check = per_agent.get(agent_name, {})
        if not isinstance(agent_fact_check, dict):
            return []

        return cls._dedupe_preserve_order(
            list(agent_fact_check.get("critical_issues", []))
            + list(agent_fact_check.get("disclaimers", []))
        )

    @staticmethod
    def _sanitize_single_qa_warning(warning: object, language: str) -> Optional[str]:
        """Converts raw QA warnings into concise user-facing notes and drops internal-only messages."""
        if not warning:
            return None

        normalized = re.sub(r"\s+", " ", str(warning)).strip()
        if not normalized:
            return None

        lowered = normalized.lower()
        internal_markers = [
            "agente de ",
            "agent ",
            "contradiz",
            "contradict",
            "ignore na resposta final",
            "ignored in the final response",
            "resposta final",
            "final answer",
            "títulos/categorias",
            "titles/categories",
            "rótulos consistentes",
            "consistent labels",
            "worker",
        ]
        if any(marker in lowered for marker in internal_markers):
            return None

        if "station names could not be verified" in lowered or "nomes de estações não puderam ser verificados" in lowered:
            return None

        if any(
            marker in lowered
            for marker in (
                "domínios conhecidos",
                "dominios conhecidos",
                "known domains",
                "não levantam suspeita de fabricação",
                "nao levantam suspeita de fabricacao",
                "do not raise fabrication concerns",
                "do not suggest fabrication",
            )
        ):
            return None

        if any(
            marker in lowered
            for marker in (
                "os indicadores apresentados",
                "dados de autocarros e comboios apresentados são parciais",
                "dados de autocarros e comboios apresentados sao parciais",
                "bus and train data shown are partial",
                "indicators shown",
                "indicators presented",
                "não foram fornecidos detalhes de cada alerta",
                "nao foram fornecidos detalhes de cada alerta",
                "não especificam perturbações concretas por linha ou serviço",
                "nao especificam perturbacoes concretas por linha ou servico",
                "do not specify concrete disruptions by line or service",
                "details of each alert or affected line were not provided",
            )
        ):
            return None

        if (
            "visitlisboa" in lowered
            and any(marker in lowered for marker in ("acceptable", "aceitável", "aceitavel"))
        ):
            return None

        if (
            "visitlisboa.com" in lowered
            and any(marker in lowered for marker in ("links presented use the domain", "os links apresentados usam o domínio"))
        ):
            return None

        if "some urls reference unverified domains" in lowered or "domínios não verificados" in lowered:
            return None

        if any(
            marker in lowered
            for marker in (
                "qa validation structure could not be confirmed",
                "qa validation could not parse",
                "quality validation could not produce",
                "valid structured result after retry",
            )
        ):
            return None

        if "event details (dates, times, ticket prices) should be confirmed at visitlisboa.com" in lowered:
            return None

        if "carris bus route numbers and schedules should be verified at carris.pt" in lowered:
            return None

        if "fonte explícita no output" in lowered or "explicit source in the output" in lowered:
            if language == "pt":
                return "A fonte indicada é apenas a do Metro de Lisboa; confirme Carris, Carris Metropolitana e CP nas respetivas fontes oficiais."
            return "Only Metro de Lisboa is explicitly cited here; please confirm Carris, Carris Metropolitana, and CP through their official sources."

        return normalized

    @classmethod
    def _sanitize_qa_disclaimers(
        cls,
        warnings: List[object],
        language: str,
    ) -> List[str]:
        """Drops internal QA warnings and localizes a small set of common user-facing caveats."""
        sanitized: List[str] = []
        for warning in warnings:
            cleaned = cls._sanitize_single_qa_warning(warning, language)
            if cleaned and cleaned not in sanitized:
                sanitized.append(cleaned)
        return sanitized

    def _append_assistant_message(self, content: str) -> None:
        """Append the final assistant message to conversation state.

        Args:
            content: Final user-facing assistant response.
        """
        if not content:
            return
        messages = self.state.setdefault("messages", [])
        messages.append(AIMessage(content=content))
        if len(messages) > _MAX_CONVERSATION_HISTORY_MESSAGES:
            del messages[:-_MAX_CONVERSATION_HISTORY_MESSAGES]

    def _append_user_message(self, content: str) -> None:
        """Append a user message while pruning stale conversation history."""
        if not content:
            return
        messages = self.state.setdefault("messages", [])
        messages.append(HumanMessage(content=content))
        if len(messages) > _MAX_CONVERSATION_HISTORY_MESSAGES:
            del messages[:-_MAX_CONVERSATION_HISTORY_MESSAGES]

    @staticmethod
    def _looks_like_next_day_planning_follow_up(message: str) -> bool:
        """Return whether the current user message asks to continue a plan tomorrow."""
        normalized = re.sub(r"\s+", " ", (message or "").lower()).strip()
        if not normalized:
            return False
        has_next_day = bool(
            re.search(
                r"\b(?:dia seguinte|pr[oó]ximo dia|amanh[aã]|tomorrow|next day|following day)\b",
                normalized,
            )
        )
        has_planning = bool(
            re.search(
                r"\b(?:plan|planeia|planejar|itinerary|itiner[aá]rio|itener[aá]rio|roteiro|dia|day)\b",
                normalized,
            )
        )
        return has_next_day and has_planning

    def _build_planning_follow_up_context(self, current_message: str) -> str:
        """Build compact continuity context for planner follow-up requests."""
        if not self._looks_like_next_day_planning_follow_up(current_message):
            return ""

        messages = list(self.state.get("messages", []))
        previous_user = ""
        previous_assistant = ""
        for msg in reversed(messages[:-1]):
            if not previous_assistant and isinstance(msg, AIMessage) and msg.content:
                previous_assistant = str(msg.content)
                continue
            if not previous_user and isinstance(msg, HumanMessage) and msg.content:
                previous_user = str(msg.content)
            if previous_user and previous_assistant:
                break

        combined = "\n".join(part for part in (previous_user, previous_assistant) if part).lower()
        if not re.search(r"\b(?:plan|planeia|itiner[aá]rio|itener[aá]rio|roteiro|monuments?|monumentos?|atra[cç][aã]o|atra[cç][oõ]es|places?|locais|gastronom|traditional|tradicional)\b", combined):
            return ""

        if previous_user:
            previous_user = previous_user.strip()[:350]
        previous_place_lines: list[str] = []
        if previous_assistant:
            previous_cards, _source_line = self._extract_place_cards_from_answer(previous_assistant)
            for card in previous_cards[:8]:
                title = re.sub(r"\s+", " ", str(card.get("title") or "")).strip()
                if not title:
                    continue
                previous_place_lines.append(f"- **{title}**")
                address = re.sub(r"\s+", " ", str(card.get("address") or "")).strip()
                category = re.sub(r"\s+", " ", str(card.get("category") or "")).strip()
                if address:
                    previous_place_lines.append(f"    - **Morada:** {address}")
                if category:
                    previous_place_lines.append(f"    - **Categoria:** {category}")
        if previous_assistant:
            previous_assistant = previous_assistant.strip()[:1400]

        continuity_requirement = (
            "Continuity requirement: the current request refers to the previous place set; "
            "keep those referenced places visible in the new plan, add the newly requested constraints, "
            "and include practical transport logic."
            if re.search(
                r"\b(?:estes|estas|esses|essas|desses|dessas|these|those|previous|above|listed)\b",
                self._fold_context_text(current_message),
            )
            else
            "Continuity requirement: answer the current request as a new following-day plan; "
            "preserve explicit preferences from the previous turn, avoid repeating the same main stops, "
            "and include practical transport logic."
        )

        return (
            "Previous planning request:\n"
            f"{previous_user}\n\n"
            + (
                "Previous referenced places:\n"
                + "\n".join(previous_place_lines)
                + "\n\n"
                if previous_place_lines
                else ""
            )
            +
            "Previous final plan excerpt:\n"
            f"{previous_assistant}\n\n"
            f"{continuity_requirement}"
        ).strip()

    @staticmethod
    def _researcher_query_with_planning_context(message: str, planning_context: str) -> str:
        """Add sanitized place continuity hints to Researcher planning lookups."""
        if not planning_context:
            return message
        titles = [
            re.sub(r"\s+", " ", match.group(1)).strip()
            for match in re.finditer(r"(?m)^-\s+\*\*([^*\n]+)\*\*", planning_context)
            if match.group(1).strip()
        ]
        if not titles:
            return message
        normalized_context = MultiAgentAssistant._fold_context_text(planning_context)
        area_hints: list[str] = []
        if re.search(r"\b(?:belem|torre de belem|padrao dos descobrimentos|jeronimos|museu de marinha)\b", normalized_context):
            area_hints.append("Belém")
        if re.search(r"\b(?:saldanha|hotel|picoas|avenidas novas)\b", MultiAgentAssistant._fold_context_text(message)):
            area_hints.append("Saldanha")
        suffix = (
            "\n\nContinuity places to preserve for evidence lookup: "
            + ", ".join(titles[:8])
            + "."
        )
        if area_hints:
            suffix += (
                " For requested meal stops, search restaurants near "
                + " and ".join(area_hints[:2])
                + " when evidence exists."
            )
        return f"{message}{suffix}"

    @staticmethod
    def _planner_meal_research_supplement(
        message: str,
        planning_context: str,
        existing_research: str,
        language: str,
    ) -> str:
        """Return deterministic restaurant evidence for requested meal areas.

        Researcher may call the right restaurant tools but later summarize only a
        subset. For itinerary synthesis, keep the relevant meal-area cards in
        planner evidence so the planner/QA cannot drift to a distant restaurant.
        """
        normalized_message = MultiAgentAssistant._fold_context_text(message)
        if not re.search(r"\b(?:almoco|almocar|lunch|jantar|dinner|restaurante|restaurant)\b", normalized_message):
            return ""

        normalized_context = MultiAgentAssistant._fold_context_text(
            "\n".join([message, planning_context])
        )
        queries: list[str] = []
        asks_lunch = bool(re.search(r"\b(?:almoco|almocar|lunch)\b", normalized_message))
        asks_dinner = bool(re.search(r"\b(?:jantar|dinner)\b", normalized_message))
        is_pt = (language or "").lower().startswith("pt")

        if asks_lunch and re.search(r"\b(?:belem|torre de belem|padrao dos descobrimentos|jeronimos|museu de marinha)\b", normalized_context):
            queries.append("restaurantes em Belém Lisboa almoço" if is_pt else "restaurants in Belém for lunch Lisbon")

        hotel_match = re.search(
            r"\b(?:hotel|alojamento|stay|staying|base)\s+(?:e|é|is|fica|no|na|near|perto de|em|in)?\s*(?P<area>[a-z0-9\s.'/-]{3,40})",
            normalized_message,
        )
        dinner_area = "Saldanha" if re.search(r"\b(?:saldanha|picoas|avenidas novas)\b", normalized_context) else ""
        if not dinner_area and hotel_match:
            dinner_area = re.sub(r"\s+", " ", hotel_match.group("area")).strip().title()
        if asks_dinner and dinner_area:
            queries.append(
                f"restaurantes perto de {dinner_area} Lisboa jantar"
                if is_pt
                else f"restaurants near {dinner_area} for dinner Lisbon"
            )

        if not queries:
            return ""

        try:
            from tools.visitlisboa_api import search_places_attractions

            outputs: list[str] = []
            for query in dict.fromkeys(queries):
                outputs.append(
                    str(
                        search_places_attractions.invoke(
                            {
                                "query": query,
                                "category": "Restaurants",
                                "max_results": 5,
                                "language": "pt" if is_pt else "en",
                            }
                        )
                        or ""
                    ).strip()
                )
            return "\n\n".join(output for output in outputs if output)
        except Exception:
            return ""

    def _get_conversation_anchors(self) -> Dict[str, Any]:
        """Return mutable structured anchors used for multi-turn planning follow-ups."""
        user_ctx = self.state.setdefault("user_context", {})
        anchors = user_ctx.setdefault(
            "conversation_anchors",
            {
                "last_itinerary_destinations": [],
                "current_selected_destination": "",
                "excluded_areas": [],
                "user_preferences": [],
                "last_plan_summary": "",
                "last_plan_text": "",
                "last_response_agents": [],
                "last_research_context": {},
                "last_transport_route": {},
                "pending_location_clarification": {},
                "expected_transport_destination": {},
            },
        )
        if not isinstance(anchors, dict):
            anchors = {
                "last_itinerary_destinations": [],
                "current_selected_destination": "",
                "excluded_areas": [],
                "user_preferences": [],
                "last_plan_summary": "",
                "last_plan_text": "",
                "last_response_agents": [],
                "last_research_context": {},
                "last_transport_route": {},
                "pending_location_clarification": {},
                "expected_transport_destination": {},
            }
            user_ctx["conversation_anchors"] = anchors
        anchors.setdefault("pending_location_clarification", {})
        anchors.setdefault("expected_transport_destination", {})
        return anchors

    @staticmethod
    def _merge_anchor_values(existing: object, new_values: List[str], limit: int = 12) -> List[str]:
        """Merge anchor lists while preserving order and ignoring empty values."""
        merged: List[str] = []
        seen: Set[str] = set()
        for value in list(existing or []) + new_values:
            cleaned = re.sub(r"\s+", " ", str(value or "")).strip(" .,:;\n\t")
            key = cleaned.lower()
            if cleaned and key not in seen:
                merged.append(cleaned)
                seen.add(key)
        return merged[:limit]

    @staticmethod
    def _extract_excluded_areas(message: str) -> List[str]:
        """Extract explicit areas the user asked not to repeat or include."""
        exclusions: List[str] = []
        folded = unicodedata.normalize("NFKD", message or "")
        folded = folded.encode("ascii", "ignore").decode("ascii").lower()
        folded = re.sub(r"[^a-z0-9\s,/&+-]", " ", folded)
        folded = re.sub(r"\s+", " ", folded).strip()
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
        for pattern in (
            r"(?i)(?:do not repeat|don't repeat|dont repeat|avoid|exclude|excluding|sem repetir|"
            r"nao repetir|evita(?:r)?|exclui(?:r)?)\s+([a-z0-9][a-z0-9 /&-]{1,100})",
            r"(?i)(?:sem|without)\s+([a-z0-9][a-z0-9 /&-]{1,80})",
        ):
            for match in re.finditer(pattern, folded):
                raw = stop_re.split(match.group(1), maxsplit=1)[0]
                for piece in re.split(r"\s*(?:,|\band\b|\bor\b|\be\b|\bou\b|/|\+|&)\s*", raw, flags=re.IGNORECASE):
                    cleaned = re.sub(r"\b(?:or|ou|areas?|zonas?|neighbourhoods?|neighborhoods?|bairros?)\b", "", piece, flags=re.IGNORECASE).strip(" .,:;")
                    if cleaned and not non_area_re.search(cleaned) and cleaned not in exclusions:
                        exclusions.append(cleaned)
        if exclusions:
            return exclusions[:8]
        for match in re.finditer(
            r"(?i)(?:do not repeat|don't repeat|avoid|sem repetir|não repetir|nao repetir|evita(?:r)?)\s+([^.;!?]+)",
            message or "",
        ):
            raw = match.group(1)
            for piece in re.split(r"\s*(?:,|\band\b|\bor\b|\be\b|\bou\b|/|\+)\s*", raw, flags=re.IGNORECASE):
                cleaned = re.sub(r"\b(?:or|ou|areas?|zonas?|neighbourhoods?|bairros?)\b", "", piece, flags=re.IGNORECASE).strip(" .,:;")
                if cleaned:
                    exclusions.append(cleaned)
        return exclusions[:8]

    @staticmethod
    def _extract_user_preferences(message: str) -> List[str]:
        """Extract stable planning preferences from the current message."""
        normalized = re.sub(r"\s+", " ", (message or "").lower())
        preferences: List[str] = []
        preference_patterns = [
            (r"\b(?:low walking|little walking|pouca caminhada|andar pouco|baixo declive|pouco declive)\b", "low walking"),
            (r"\b(?:indoor backup|rain[- ]?safe|if it rains|se chover|chuva|chover|interior|interiores|coberto|coberta)\b", "rain-safe indoor backup"),
            (r"\b(?:cheap|budget|barato|econ[oó]mico|mais barato)\b", "budget-conscious"),
            (r"\b(?:public transport|metro|bus|tram|autocarro|el[eé]trico|comboio)\b", "public transport"),
            (r"\b(?:history|hist[oó]ria|cultural|culture|cultura|museum|museu)\b", "culture/history"),
        ]
        for pattern, label in preference_patterns:
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                preferences.append(label)
        return preferences

    @staticmethod
    def _extract_destination_candidates_from_plan(text: str) -> List[str]:
        """Extract concrete destination anchors from a final itinerary without storing raw internals."""
        if not text:
            return []
        candidates: List[str] = []
        skip = {
            "direct answer", "constraints used", "plan blocks", "movement logic",
            "weather strategy", "limitations", "source", "fonte", "updated", "atualizado",
            "structured plan", "plano estruturado", "lisbon", "lisboa",
            "roteiro sugerido", "suggested route", "como te deslocas", "how to move",
            "dicas", "tips", "notas finais", "final notes", "resumo da viagem",
        }

        def add_candidate(raw_value: str) -> None:
            value = re.sub(r"^[\W_\d:.-]+", "", raw_value or "").strip(" .,:;*-_")
            value = re.sub(r"\s+", " ", value)
            value = re.sub(r"^\d{1,2}:\d{2}\s*[·.-]\s*", "", value).strip()
            label_match = re.match(
                r"(?i)^(?:paragem\s+hist[oó]rica|paragem\s+cultural|visita|"
                r"almo[cç]o|jantar|caf[eé]|lanche|historic\s+stop|visit|"
                r"lunch|dinner|coffee|snack)\s*:\s*(.+)$",
                value,
            )
            if label_match:
                value = label_match.group(1).strip()
            lower = value.lower()
            folded = MultiAgentAssistant._fold_context_text(value)
            if not value or lower in skip or folded in skip:
                return
            if any(token in folded for token in [
                "source", "updated", "distance", "lines", "warning", "tip", "limitation",
                "structured", "plan", "title", "temperature", "conditions", "yellow", "blue", "green",
                "origin confirmed", "transport limits", "location", "category", "note",
                "resposta direta", "direct answer", "roteiro sugerido", "como te deslocas",
                "dicas", "notas finais", "manha em", "morning in",
            ]):
                return
            if len(value.split()) > 8:
                return
            if re.search(r"[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç'’.-]+", value):
                candidates.append(value)

        for match in MultiAgentAssistant._PLANNER_CARD_NAME_RE.finditer(text):
            add_candidate(match.group("name") or "")
        for match in MultiAgentAssistant._PLANNER_SIMPLE_CARD_RE.finditer(text):
            add_candidate(match.group("name") or "")
        return MultiAgentAssistant._merge_anchor_values([], candidates, limit=10)

    # Anaphoric reference to a venue/place suggested in the previous turn,
    # for example "the restaurant you suggested" or
    # "o restaurante que sugeriste". The pattern tolerates up to a few
    # intervening words ("o almoço às horas que referes", "the lunch we were
    # talking about") and the SMS abbreviation "q" for "que". Captured noun
    # group enables intent-aware card selection later.
    _VENUE_ANAPHOR_RE = re.compile(
        r"(?:"
        r"(?:o|a|ao|aos|à|às)\s+(?P<noun_pt>restaurante|almoco|almo[cç]o|jantar|"
        r"pequeno[- ]almoco|pequeno[- ]almo[cç]o|brunch|lanche|caf[eé]|bar|"
        r"pastelaria|padaria|s[ií]tio|lugar|local|museu|monumento|atra[cç][aã]o|ponto|paragem)"
        r"(?:\s+\S+){0,5}?"
        r"\s+(?:que|q)\b\s*(?:tu\s+)?"
        # Past, present, and imperfect-progressive PT verb forms covering the
        # recurring patterns "que sugeriste", "que sugeres", "que estavas a
        # sugerir", "que tinhas sugerido", etc. Stems use the broadest common
        # prefix so both past (-iste) and present (-es) inflections match.
        r"(?:"
        r"suger[ie][a-z]{0,4}|indica[a-z]{0,4}|recomenda[a-z]{0,4}|"
        r"menciona[a-z]{0,4}|diss?es[a-z]{0,3}|disse|dize[a-z]{0,3}|"
        r"propu[a-z]{0,4}|propus[a-z]{0,4}|prop[oõ]e[a-z]{0,3}|"
        r"refer[ei][a-z]{0,4}|fala[a-z]{0,4}|aponta[a-z]{0,4}|d[ae][a-z]{0,3}|"
        r"mostra[a-z]{0,4}|d[aá]s|d[aá]"
        r"|estavas?\s+a\s+(?:sugerir|indicar|recomendar|mencionar|dizer|propor|"
        r"referir|falar|apontar|mostrar|dar)"
        r"|tinha[s]?\s+(?:sugerido|indicado|recomendado|mencionado|dito|proposto|"
        r"referido|falado|apontado|mostrado|dado)"
        r")"
        r"|"
        r"(?:the|that)\s+(?P<noun_en>restaurant|lunch|dinner|breakfast|brunch|"
        r"snack|cafe|caf[eé]|coffee\s+shop|pastry|bakery|bar|place|spot|venue|museum|landmark|attraction|monument|stop)"
        r"(?:\s+\S+){0,5}?"
        # Allow optional auxiliary verbs (did/do/does/have/had) before "you".
        r"\s+(?:(?:did|do|does|have|had)\s+)?(?:you|we|that\s+you|that\s+we)\s+"
        r"(?:suggested|recommended|mentioned|proposed|said|referred(?:\s+to)?|"
        r"talked\s+about|were\s+talking\s+about|pointed\s+(?:out|to)|"
        # Present / bare verb form: "...did you suggest", "the lunch you suggest".
        r"suggest|recommend|mention|propose|say|refer\s+to|talk\s+about|point\s+(?:out|to))"
        r")",
        re.IGNORECASE,
    )

    # Bold venue name in a list-bullet card; used to extract a concrete venue
    # name from the previous assistant answer (researcher-style cards).
    _VENUE_CARD_NAME_RE = re.compile(
        r"^\s*[\-*]\s+\*\*[^\w\s]*\s*([A-Z0-9][^\n*]{2,80})\*\*",
        re.MULTILINE,
    )

    # Planner-style itinerary card heading, e.g.
    # "**🏷️ 12:45 · Almoço tradicional: Restaurante Exemplo**" or
    # "**🏷️ 09:30 · Paragem histórica: Museu Exemplo**".
    # Capture group 1 = label segment (used for intent matching), group 2 = name.
    _PLANNER_CARD_NAME_RE = re.compile(
        r"^\s*(?:[-*]\s+)?\*\*[^\n*]*?(?P<label>"
        r"Paragem\s+hist[oó]rica|Paragem\s+cultural|Visita|Almo[cç]o|Jantar|"
        r"Pequeno[- ]almo[cç]o|Brunch|Caf[eé]|Lanche|"
        r"Historic\s+stop|Cultural\s+stop|Visit|Lunch|Dinner|Breakfast|Coffee|Snack"
        r")[^*\n]*?:\s*(?P<name>[^\n*]{3,80})\*\*",
        re.MULTILINE | re.IGNORECASE,
    )
    _PLANNER_SIMPLE_CARD_RE = re.compile(
        r"^\s*[-*]\s+\*\*[^\w\n]*(?P<name>[A-ZÀ-Ý0-9][^\n*]{2,90})\*\*",
        re.MULTILINE,
    )

    # Mapping from anaphor noun (lowercase, ASCII-folded) to ordered list of
    # planner-card label keywords that should be preferred when extracting the
    # venue name. The first list whose first match exists wins.
    _ANAPHOR_NOUN_TO_PLANNER_LABELS = {
        # Meals / restaurants
        "almoco": ("almo", "lunch"),
        "lunch": ("lunch", "almo"),
        "jantar": ("jantar", "dinner"),
        "dinner": ("dinner", "jantar"),
        "pequeno-almoco": ("pequeno", "breakfast", "brunch"),
        "pequeno almoco": ("pequeno", "breakfast", "brunch"),
        "breakfast": ("breakfast", "brunch", "pequeno"),
        "brunch": ("brunch", "breakfast", "pequeno"),
        "cafe": ("caf", "coffee", "lanche", "snack"),
        "coffee shop": ("coffee", "caf", "lanche", "snack"),
        "pastelaria": ("pastelaria", "caf", "coffee", "lanche", "snack"),
        "padaria": ("padaria", "bakery", "pastelaria", "caf", "coffee"),
        "pastry": ("pastry", "bakery", "coffee", "caf", "snack"),
        "bakery": ("bakery", "pastry", "coffee", "caf", "snack"),
        "lanche": ("lanche", "snack", "caf", "coffee"),
        "snack": ("snack", "lanche", "caf", "coffee"),
        "bar": ("bar",),
        "restaurante": ("almo", "jantar", "lunch", "dinner"),
        "restaurant": ("lunch", "dinner", "almo", "jantar"),
        # Cultural / historic
        "museu": ("museu", "museum", "hist", "visita", "visit", "cultural"),
        "museum": ("museum", "museu", "hist", "visit", "cultural"),
        "monumento": ("hist", "monumento", "monument", "visita", "visit"),
        "monument": ("hist", "monumento", "monument", "visit"),
        "atracao": ("hist", "cultural", "visit", "visita"),
        "attraction": ("hist", "cultural", "visit", "visita"),
        "ponto": ("hist", "cultural", "visit", "visita"),
        "paragem": ("hist", "cultural", "visit", "visita"),
        "stop": ("hist", "cultural", "visit", "visita"),
        # Generic
        "sitio": (),
        "lugar": (),
        "local": (),
        "place": (),
        "spot": (),
        "venue": (),
    }

    _DEMONSTRATIVE_VENUE_RE = re.compile(
        r"\b(?P<demo>esse|essa|este|esta|aquele|aquela|that|this)\s+"
        r"(?P<noun>restaurante|restaurant|almo[cç]o|almoco|lunch|jantar|dinner|"
        r"pastelaria|padaria|pastry|bakery|coffee\s+shop|"
        r"museu|museum|monumento|monument|atra[cç][aã]o|atracao|attraction|"
        r"local|s[ií]tio|sitio|lugar|place|spot|venue)\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _fold_context_text(text: str) -> str:
        """Fold accents and whitespace for context-anchor comparisons."""
        normalized = unicodedata.normalize("NFKD", str(text or ""))
        ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
        return re.sub(r"\s+", " ", ascii_text).strip().lower()

    @classmethod
    def _short_english_follow_up_uses_previous_language(cls, message: str) -> bool:
        """Return whether a compact English follow-up should keep English output."""
        folded = cls._fold_context_text(message)
        if not folded or len(folded.split()) > 7:
            return False
        if not re.match(r"^(?:and|also|what about|how about|same)\b", folded):
            return False
        return bool(
            re.search(
                r"\b(?:hat|jacket|coat|umbrella|sunscreen|suncream|sunglasses|"
                r"one|it|that|there|this|these|those|tomorrow|later)\b",
                folded,
            )
        )

    @classmethod
    def _message_needs_previous_turn_context(cls, message: str) -> bool:
        """Return whether worker prompts need previous-turn text for recall.

        Self-contained route/place/weather questions should not be biased by a
        previous itinerary or search result. Compact follow-ups such as
        "E de autocarro?", "esse restaurante", or "mais destes" still need the
        previous turn so workers can resolve the missing referent.
        """
        folded = cls._fold_context_text(message)
        if not folded:
            return False

        has_explicit_route_pair = bool(
            re.search(
                r"\b(?:from\s+.+?\s+to\s+.+|de\s+.+?\s+(?:para|ate|a|ao)\s+.+)",
                folded,
            )
        )
        anaphora_re = re.compile(
            r"\b(?:"
            r"there|that|those|these|previous|above|same|another|other|one|it|"
            r"restaurant|lunch|dinner|meal|place|stop|venue|alternative|instead|"
            r"la|ali|aqui|ai|esse|essa|isso|este|esta|aquele|aquela|destes|destas|"
            r"desses|dessas|anterior|anteriores|mesmo|mesma|outro|outra|alternativa|"
            r"restaurante|almoco|jantar|refeicao|paragem|local|sitio|lugar|mais"
            r")\b",
            flags=re.IGNORECASE,
        )
        has_anaphora = bool(anaphora_re.search(folded))

        if has_explicit_route_pair:
            return has_anaphora

        mode_only_follow_up = bool(
            len(folded.split()) <= 8
            and re.search(r"\b(?:metro|autocarro|comboio|bus|train)\b", folded)
        )
        return has_anaphora or bool(
            re.search(
                r"^(?:e\s+)?(?:de|por|para|com|sem|mais|outro|outra|another|other|and|also)\b",
                folded,
                flags=re.IGNORECASE,
            )
        ) or mode_only_follow_up

    def _build_unconfirmed_meal_revision_response(
        self,
        *,
        language: str,
        previous_area: str,
        destinations: List[str],
    ) -> str:
        """Build a safe direct answer when a plan revision asks to keep an unconfirmed meal."""
        is_pt = language == "pt"
        folded_meal_re = re.compile(
            r"\b(?:almoco|almoço|lunch|jantar|dinner|restaurante|restaurant|refeicao|refeição|meal)\b",
            flags=re.IGNORECASE,
        )
        visible_stops = [
            re.sub(r"\s+", " ", str(destination or "")).strip()
            for destination in destinations
            if str(destination or "").strip()
            and not folded_meal_re.search(self._fold_context_text(str(destination or "")))
        ][:4]
        area_label = previous_area.strip()

        if is_pt:
            stop_lines = (
                "\n".join(f"- 🏛️ **{stop}**: mantém como paragem interior/coberta prioritária." for stop in visible_stops)
                if visible_stops
                else "- 🏛️ Mantém as paragens interiores já confirmadas no roteiro anterior."
            )
            meal_line = (
                f"- 🍽️ **Refeição em {area_label}:** mantém a pausa gastronómica, mas confirma um restaurante concreto antes de fechar o plano."
                if area_label
                else "- 🍽️ **Refeição na zona anterior:** mantém a pausa gastronómica, mas confirma um restaurante concreto antes de fechar o plano."
            )
            return (
                "### ☔ **Adaptação do roteiro anterior**\n\n"
                "✅ **Resposta direta:** adapto o plano para chuva, mas **não havia um restaurante específico confirmado** no roteiro anterior; "
                "por isso mantenho a pausa de refeição como limitação e não a substituo por outro restaurante inventado.\n\n"
                "---\n\n"
                "### 📍 **O que mantém**\n\n"
                f"{stop_lines}\n"
                f"{meal_line}\n\n"
                "### ☔ **Ajuste para chuva**\n\n"
                "- Prioriza museus, monumentos visitáveis por dentro e espaços cobertos já presentes no roteiro.\n"
                "- Evita miradouros, longas caminhadas e trocas de bairro/concelho sem nova confirmação.\n"
                "- Mantém a ordem compacta do roteiro anterior para reduzir deslocações em dias de chuva."
            )

        stop_lines = (
            "\n".join(f"- 🏛️ **{stop}**: keep as a priority indoor/covered stop." for stop in visible_stops)
            if visible_stops
            else "- 🏛️ Keep the indoor stops already confirmed in the previous itinerary."
        )
        meal_line = (
            f"- 🍽️ **Meal in {area_label}:** keep the food break, but confirm a concrete restaurant before finalising the plan."
            if area_label
            else "- 🍽️ **Meal in the previous area:** keep the food break, but confirm a concrete restaurant before finalising the plan."
        )
        return (
            "### ☔ **Previous Itinerary Rain Adaptation**\n\n"
            "✅ **Direct answer:** I can adapt it for rain, but **no specific restaurant was confirmed** in the previous itinerary; "
            "I keep the meal break as a limitation instead of replacing it with an invented restaurant.\n\n"
            "---\n\n"
            "### 📍 **What Stays**\n\n"
            f"{stop_lines}\n"
            f"{meal_line}\n\n"
            "### ☔ **Rain Adjustment**\n\n"
            "- Prioritise museums, indoor monuments, and covered spaces already present in the itinerary.\n"
            "- Avoid viewpoints, long walks, and neighbourhood/municipality changes without new confirmation.\n"
            "- Keep the previous compact order to reduce movement on rainy days."
        )

    def _resolve_venue_anaphor(self, message: str) -> str:
        """Replace anaphoric venue references with the previous turn's venue name.

        For example:
            "Como vou da origem ate ao restaurante que sugeriste?"
            -> "Como vou da origem ate ao Restaurante Exemplo?"

        Only triggers when the message also contains an explicit
        transport/navigation cue. Pure recall questions such as
        "Qual foi o restaurante que indicaste?" are intentionally left
        unchanged so the researcher recall fast-path can handle them.
        """
        if not message:
            return message
        explicit_anaphor_match = self._VENUE_ANAPHOR_RE.search(message)
        demonstrative_match = self._DEMONSTRATIVE_VENUE_RE.search(message)
        anaphor_match = explicit_anaphor_match or demonstrative_match
        if not anaphor_match:
            return message
        # Only substitute when the user is actually asking how to GET to the
        # referenced venue. Otherwise this is a recall question and the
        # researcher's recall fast-path is the right path.
        normalized = message.lower()
        has_navigation_intent = bool(re.search(
            r"\b(?:como\s+(?:vou|chego|ir|fa[cç]o\s+para\s+(?:ir|chegar))|ir\s+(?:de|do|da|para|at[eé])|chegar\s+at[eé]|"
            r"how\s+do\s+i\s+get|how\s+to\s+get|get\s+to|go\s+to|travel\s+to|head\s+to)\b",
            normalized,
        ))
        if not has_navigation_intent:
            return message
        recent_msgs = self.state.get("messages", []) or []
        previous_assistant = ""
        for msg in reversed(recent_msgs):
            if isinstance(msg, AIMessage) and msg.content:
                previous_assistant = str(msg.content)
                break
        anchors = self._get_conversation_anchors()
        anchor_plan_text = str(anchors.get("last_plan_text") or "")
        if not previous_assistant and not anchor_plan_text:
            return message
        # Identify which noun the user referenced (PT or EN) and try planner
        # cards first (they encode meal-vs-monument intent in the label),
        # then fall back to generic researcher list-bullet venue cards.
        anaphor_noun = (
            anaphor_match.groupdict().get("noun_pt")
            or anaphor_match.groupdict().get("noun_en")
            or anaphor_match.groupdict().get("noun")
            or ""
        )
        ascii_noun = self._fold_context_text(anaphor_noun)
        if ascii_noun in {"restaurante", "restaurant"}:
            return message
        venue_name = self._extract_venue_from_previous_answer(previous_assistant, ascii_noun)
        if not venue_name and anchor_plan_text:
            venue_name = self._extract_venue_from_previous_answer(anchor_plan_text, ascii_noun)
        if not venue_name:
            return message
        return (
            message[: anaphor_match.start()]
            + venue_name
            + message[anaphor_match.end() :]
        )

    def _extract_venue_from_previous_answer(
        self, previous_assistant: str, ascii_noun: str
    ) -> str:
        """Extract a venue name from the previous assistant answer.

        Tries planner-card headings first using intent-aware label keywords
        (so "almoço" preferentially picks the lunch card, not the museum
        card). Falls back to a generic list-bullet venue card. Returns an
        empty string when no usable name was found.
        """
        if not previous_assistant:
            return ""
        planner_matches = list(self._PLANNER_CARD_NAME_RE.finditer(previous_assistant))
        if planner_matches:
            label_keywords = self._ANAPHOR_NOUN_TO_PLANNER_LABELS.get(ascii_noun, ())
            if label_keywords:
                for keyword in label_keywords:
                    for match in planner_matches:
                        label_segment = (match.group("label") or "").lower()
                        # Fold accents in label segment for keyword matching.
                        folded = (
                            label_segment.replace("ç", "c")
                            .replace("ã", "a")
                            .replace("á", "a")
                            .replace("é", "e")
                            .replace("ê", "e")
                            .replace("í", "i")
                            .replace("ó", "o")
                            .replace("ô", "o")
                            .replace("õ", "o")
                            .replace("ú", "u")
                        )
                        if keyword in folded:
                            name = (match.group("name") or "").strip(" .·-")
                            if len(name) >= 3:
                                return name
            # No intent mapping or no label match: take the first planner card
            # only when the noun is generic (place/spot/lugar/local).
            if ascii_noun in {"sitio", "lugar", "local", "place", "spot", "venue"}:
                first = (planner_matches[0].group("name") or "").strip(" .·-")
                if len(first) >= 3:
                    return first
        if ascii_noun in {"cafe", "coffee shop", "pastelaria", "padaria", "pastry", "bakery", "lanche", "snack"}:
            food_name = self._extract_food_venue_from_previous_answer(previous_assistant, ascii_noun)
            if food_name:
                return food_name
        # Fall back to researcher list-bullet venue cards.
        venue_match = self._VENUE_CARD_NAME_RE.search(previous_assistant)
        if venue_match:
            name = venue_match.group(1).strip(" .·-")
            if len(name) >= 3:
                return name
        return ""

    def _extract_food_venue_from_previous_answer(self, previous_assistant: str, ascii_noun: str) -> str:
        """Return a food/cafe venue from a previous card-style answer.

        Planner fallbacks often render cards as ``- **Pastéis de Belém**`` with
        the role stored in child fields such as ``Categoria: Coffee shop``.
        This parser keeps demonstrative follow-ups like "essa pastelaria" tied
        to the previous plan without hardcoding venue names.
        """
        if not previous_assistant:
            return ""
        cafe_specific = ascii_noun in {"cafe", "coffee shop", "pastelaria", "padaria", "pastry", "bakery"}
        matches = list(self._PLANNER_SIMPLE_CARD_RE.finditer(previous_assistant))
        for index, match in enumerate(matches):
            name = re.sub(r"\s+", " ", match.group("name") or "").strip(" .·-")
            if len(name) < 3:
                continue
            next_start = matches[index + 1].start() if index + 1 < len(matches) else len(previous_assistant)
            block = previous_assistant[match.end(): next_start]
            basis = self._fold_context_text(f"{name}\n{block}")
            if cafe_specific:
                signal = re.search(
                    r"\b(?:coffee shop|coffee|cafe|pastelaria|padaria|pastry|bakery|pasteis|brunch|snack|lanche)\b",
                    basis,
                )
            else:
                signal = re.search(
                    r"\b(?:restaurant|restaurante|coffee shop|coffee|cafe|pastelaria|padaria|pastry|bakery|food|cozinha|gastronomia|brunch|snack|lanche)\b",
                    basis,
                )
            if signal:
                return name
        return ""

    def _extract_meal_anchors_from_plan(self, ascii_noun: str = "restaurante") -> List[Dict[str, str]]:
        """Return planner meal venues/times matching an anaphoric noun.

        Args:
            ascii_noun: Folded noun used by the user, such as ``almoco``,
                ``jantar`` or ``restaurante``.

        Returns:
            Ordered list of matching meal anchors from the last planner answer.
        """
        plan_text = str(self._get_conversation_anchors().get("last_plan_text") or "")
        if not plan_text:
            return []
        ascii_noun = self._fold_context_text(ascii_noun)
        label_keywords = self._ANAPHOR_NOUN_TO_PLANNER_LABELS.get(ascii_noun, ("almo", "lunch", "jantar", "dinner"))
        anchors: List[Dict[str, str]] = []
        seen_anchor_keys: Set[str] = set()

        def add_anchor(name: str, time_value: str = "", address: str = "", role: str = "") -> None:
            """Add a de-duplicated meal anchor extracted from a rendered plan."""
            cleaned_name = re.sub(r"\s+", " ", str(name or "")).strip(" .:;*-_·•")
            cleaned_name = re.sub(r"^[^\w\d]+", "", cleaned_name).strip(" .:;*-_·•")
            cleaned_name = re.sub(r"^\d{1,2}:\d{2}\s*[·•.\-–—]\s*", "", cleaned_name).strip(" .:;*-_·•")
            if ":" in cleaned_name:
                prefix, suffix = cleaned_name.rsplit(":", 1)
                if re.search(
                    r"\b(?:almo|lunch|jantar|dinner|breakfast|brunch|cafe|coffee|"
                    r"lanche|snack|restaurante|restaurant|tradicional|traditional)\b",
                    self._fold_context_text(prefix),
                ):
                    cleaned_name = suffix.strip(" .:;*-_·•")
            if len(cleaned_name) < 3:
                return
            folded_name = self._fold_context_text(cleaned_name)
            if folded_name in {
                "roteiro sugerido",
                "suggested route",
                "como te deslocas",
                "how to move",
                "dicas",
                "tips",
            }:
                return
            key = f"{folded_name}|{time_value or ''}"
            if key in seen_anchor_keys:
                return
            seen_anchor_keys.add(key)
            anchors.append({
                "name": cleaned_name,
                "time": time_value or "",
                "address": str(address or "").strip(),
                "role": role,
            })

        # Formatter and QA repairs can change exact bullet shape, so parse
        # visible heading lines as well as the canonical planner regex.
        for line_match in re.finditer(r"(?m)^.+$", plan_text):
            raw_line = line_match.group(0).strip()
            if "**" not in raw_line:
                continue
            folded_line = self._fold_context_text(raw_line)
            if not any(keyword in folded_line for keyword in label_keywords):
                continue
            if not re.search(
                r"\b(?:almo|lunch|jantar|dinner|breakfast|brunch|cafe|coffee|lanche|snack|restaurante|restaurant)\b",
                folded_line,
            ):
                continue
            visible = re.sub(r"[*_`]", "", raw_line)
            visible = re.sub(r"^\s*[-#]+\s*", "", visible).strip()
            visible = re.sub(r"^[^\w\d]+", "", visible).strip()
            name = ""
            if ":" in visible:
                name = visible.rsplit(":", 1)[1]
            else:
                dash_match = re.search(r"\d{1,2}:\d{2}\s*[·•.-]\s*[^-–:]+[-–]\s*(?P<name>.+)$", visible)
                name = dash_match.group("name") if dash_match else visible
            name = re.sub(r"^(?:\d{1,2}:\d{2}\s*[·•.-]\s*)", "", name).strip()
            next_slice = plan_text[line_match.end(): line_match.end() + 900]
            address_match = re.search(
                r"-\s*(?:📍\s*)?\*\*(?:Morada|Address):\*\*\s*\[([^\]]+)\]",
                next_slice,
                flags=re.IGNORECASE,
            )
            time_match = re.search(r"\b(\d{1,2}:\d{2})\b", raw_line)
            add_anchor(
                name,
                time_match.group(1) if time_match else "",
                address_match.group(1).strip() if address_match else "",
                folded_line,
            )

        for match in self._PLANNER_CARD_NAME_RE.finditer(plan_text):
            folded = self._fold_context_text(match.group("label") or "")
            if not any(keyword in folded for keyword in label_keywords):
                continue
            full_line = match.group(0)
            time_match = re.search(r"\b(\d{1,2}:\d{2})\b", full_line)
            next_slice = plan_text[match.end(): match.end() + 900]
            address_match = re.search(
                r"-\s*📍\s*\*\*(?:Morada|Address):\*\*\s*\[([^\]]+)\]",
                next_slice,
                flags=re.IGNORECASE,
            )
            add_anchor(
                (match.group("name") or "").strip(" .·-"),
                time_match.group(1) if time_match else "",
                address_match.group(1).strip() if address_match else "",
                folded,
            )
        compact_pattern = re.compile(
            r"(?im)^\s*(?:[-#*]+\s*)?(?:\*\*)?(?:[^\w\n]{0,12}\s*)?"
            r"(?P<label>Almo[cç]o|Lunch|Jantar|Dinner|Pequeno[- ]almo[cç]o|Breakfast|Brunch|Caf[eé]|Coffee|Lanche|Snack)"
            r"(?:\s+(?:tradicional|traditional))?"
            r"(?:\s*[-–]\s*(?P<time_before>\d{1,2}:\d{2}))?"
            r"\s*:\s*(?P<name>[^\n*]{3,120}?)"
            r"(?:\s+(?:às|as|at)\s*(?P<time_after>\d{1,2}:\d{2}))?"
            r"(?:[.\n*]|$)"
        )
        for match in compact_pattern.finditer(plan_text):
            folded = self._fold_context_text(match.group("label") or "")
            if not any(keyword in folded for keyword in label_keywords):
                continue
            name = re.sub(r"\s+", " ", match.group("name") or "").strip(" .:-")
            if not name:
                continue
            time_match = match.group("time_before") or match.group("time_after")
            add_anchor(name, time_match or "", "", folded)

        if ascii_noun in {
            "almoco",
            "lunch",
            "jantar",
            "dinner",
            "restaurante",
            "restaurant",
            "cafe",
            "coffee shop",
            "pastelaria",
            "padaria",
            "pastry",
            "bakery",
            "lanche",
            "snack",
        }:
            food_signal_re = re.compile(
                r"\b(?:categoria:\s*(?:restaurante|restaurant|coffee shop|cafe|pastelaria|padaria|"
                r"pastry|bakery)|restaurante|restaurant|coffee shop|cafe|pastelaria|padaria|"
                r"pastry|bakery|cozinha|cuisine|gastronom|food|comida|refeicao|refeição|"
                r"meal|marisco|brunch|lanche|snack)\b",
                re.IGNORECASE,
            )
            top_level_card_re = re.compile(
                r"^-\s+\*\*[^\w\n]*(?P<name>[A-ZÀ-Ý0-9][^\n*]{2,90})\*\*"
            )
            lines = plan_text.splitlines()
            index = 0
            while index < len(lines):
                raw_line = lines[index]
                match = top_level_card_re.match(raw_line.strip())
                if not match:
                    index += 1
                    continue

                name = re.sub(r"\s+", " ", match.group("name") or "").strip(" .:;*-_·•")
                folded_name = self._fold_context_text(name)
                if (
                    not name
                    or folded_name.endswith(":")
                    or folded_name
                    in {"categoria", "category", "morada", "address", "horario", "hours"}
                ):
                    index += 1
                    continue

                body_lines: List[str] = []
                next_index = index + 1
                while next_index < len(lines):
                    candidate_line = lines[next_index]
                    if candidate_line.startswith("### "):
                        break
                    if top_level_card_re.match(candidate_line.strip()):
                        break
                    body_lines.append(candidate_line)
                    next_index += 1

                body = "\n".join(body_lines)
                basis = self._fold_context_text(f"{name}\n{body}")
                if food_signal_re.search(basis):
                    address_match = re.search(
                        r"-\s*(?:📍\s*)?\*\*(?:Morada|Address):\*\*\s*\[([^\]]+)\]",
                        body,
                        flags=re.IGNORECASE,
                    )
                    # In simple card layouts, child fields often contain
                    # opening hours. Treat only a time present in the card
                    # heading as the planned meal time.
                    time_match = re.search(r"\b(\d{1,2}:\d{2})\b", raw_line)
                    add_anchor(
                        name,
                        time_match.group(1) if time_match else "",
                        address_match.group(1).strip() if address_match else "",
                        basis,
                    )
                index = max(next_index, index + 1)
        return anchors

    def _extract_transport_destination_clarification(self, message: str, language: str) -> str:
        """Extract a concrete destination supplied as a route follow-up."""
        original = re.sub(r"\s+", " ", str(message or "")).strip(" .,:;?!")
        if not original:
            return ""

        address_re = re.compile(
            r"\b(?:rua|r\.|avenida|av\.|largo|pra[cç]a|travessa|tv\.|estrada|estr\.|"
            r"alameda|cal[cç]ada|campo|hospital|cl[ií]nica|taberna|restaurante|"
            r"veterin[áa]rio|veterin[áa]ria|\d{4}-\d{3})\b",
            flags=re.IGNORECASE,
        )

        def clean_destination(value: str) -> str:
            cleaned = re.sub(r"\s+", " ", str(value or "")).strip(" .,:;?!")
            cleaned = re.sub(
                r"^\s*(?:o|a|os|as|ao|à|no|na|nos|nas|em|the|at|in)\s+",
                "",
                cleaned,
                flags=re.IGNORECASE,
            ).strip(" .,:;?!")
            return cleaned

        relation_match = re.match(
            r"^(?P<label>.{2,90}?)\s+"
            r"(?:é|e|fica|situa-se|est[áa]|is|it'?s|it is)\s+"
            r"(?:na|no|em|at|in)\s+(?P<tail>.+)$",
            original,
            flags=re.IGNORECASE,
        )
        if relation_match:
            label = clean_destination(relation_match.group("label"))
            tail = clean_destination(relation_match.group("tail"))
            if tail and address_re.search(tail):
                return tail
            if label and tail and len(tail) >= 2:
                connector = "em" if language == "pt" else "in"
                return clean_destination(f"{label} {connector} {tail}")

        explicit_match = re.search(
            r"\b(?:refiro[-\s]?me\s+(?:ao|à|a|o|no|na)?|queria\s+dizer|quero\s+dizer|i\s+mean|i\s+meant)\s+"
            r"(?P<dest>[^.?!;]+)",
            original,
            flags=re.IGNORECASE,
        )
        if explicit_match:
            return clean_destination(explicit_match.group("dest"))

        correction_match = re.match(
            r"^(?:afinal\s+)?(?:é|e|fica|seria|it'?s|it is|is)\s+(?P<dest>.+)$",
            original,
            flags=re.IGNORECASE,
        )
        if correction_match:
            return clean_destination(correction_match.group("dest"))

        if address_re.search(original) and len(original.split()) <= 14:
            return clean_destination(original)
        return ""

    def _resolve_transport_destination_clarification_follow_up(
        self,
        message: str,
        language: str,
    ) -> Dict[str, Any]:
        """Resolve a place clarification against the previous transport route.

        This handles turns such as "I meant the shopping centre" after a route
        answer or a location clarification, while keeping unrelated place
        browsing queries out of the transport path.
        """
        anchors = self._get_conversation_anchors()
        last_route = anchors.get("last_transport_route")
        if not isinstance(last_route, dict) or not last_route.get("origin"):
            return {}
        last_agents = {str(agent) for agent in anchors.get("last_response_agents") or []}
        if "transport" not in last_agents:
            return {}

        normalized = self._fold_context_text(message)
        destination = self._extract_transport_destination_clarification(message, language)
        has_clarification_cue = bool(
            re.search(
                r"\b(?:afinal|refiro[-\s]?me|queria\s+dizer|quero\s+dizer|i\s+mean|i\s+meant|"
                r"o\s+centro\s+comercial|a\s+loja|esse\s+sitio|esse\s+local|that\s+place|the\s+mall|shopping\s+centre)\b",
                normalized,
            )
        )
        asks_route_or_choice = bool(
            re.search(
                r"\b(?:melhor\s+op[cç][aã]o|como\s+vou|como\s+chego|ir|rota|trajeto|"
                r"agora|sem\s+repetir|best\s+option|how\s+do\s+i\s+get|route|now)\b",
                normalized,
            )
        )
        destination_correction = bool(destination) and bool(
            has_clarification_cue
            or re.match(
                r"^(?:e\s+)?(?:é|fica|seria|it'?s|it is|is)\b",
                normalized,
                flags=re.IGNORECASE,
            )
            or re.match(
                r"^(?:e|é)?\s*(?:em|no|na|nos|nas|in|near)\b",
                normalized,
                flags=re.IGNORECASE,
            )
            or re.search(r"\b(?:rua|avenida|travessa|largo|pra[cç]a|\d{4}-\d{3})\b", normalized)
        )
        if not ((has_clarification_cue and asks_route_or_choice) or destination_correction):
            return {}

        previous_destination = str(last_route.get("destination") or "").strip()
        if not destination:
            destination = previous_destination

        if re.search(r"\b(?:centro\s+comercial|shopping|mall|shopping\s+centre)\b", normalized):
            previous_key = self._fold_context_text(previous_destination)
            destination_key = self._fold_context_text(destination)
            if "colombo" in previous_key and "colombo" not in destination_key:
                destination = "Centro Comercial Colombo" if language == "pt" else "Colombo Shopping Centre"

        destination = re.sub(r"^(?:ao|à|a|o|no|na|the)\s+", "", destination, flags=re.IGNORECASE).strip()
        if len(destination) < 3:
            return {}

        origin = str(last_route.get("origin") or "").strip()
        asks_best_now = bool(
            re.search(
                r"\b(?:melhor|agora|best|now)\b",
                normalized,
                flags=re.IGNORECASE,
            )
        )
        rewritten = (
            f"Qual é a melhor opção agora para ir de {origin} para {destination}?"
            if language == "pt" and asks_best_now
            else f"Como vou de {origin} para {destination}?"
            if language == "pt"
            else f"What is the best option now to get from {origin} to {destination}?"
            if asks_best_now
            else f"How do I get from {origin} to {destination}?"
        )
        return {
            "message": rewritten,
            "agents": ["transport"],
            "routing_reasoning": "Transport follow-up clarified the previous destination.",
        }

    def _resolve_standalone_place_after_transport_follow_up(
        self,
        message: str,
        language: str,
    ) -> Dict[str, Any]:
        """Keep compact place-only turns from inheriting stale transport context."""
        anchors = self._get_conversation_anchors()
        last_agents = {str(agent) for agent in anchors.get("last_response_agents") or []}
        if "transport" not in last_agents:
            return {}

        normalized = self._fold_context_text(message)
        if not normalized or len(normalized.split()) > 7:
            return {}

        transport_follow_up_re = re.compile(
            r"\b(?:como|chego|vou|ir|rota|trajeto|percurso|tempo|horas?|quando|"
            r"proximo|proxima|partida|partidas|apanhar|metro|autocarro|comboio|"
            r"transporte|transportes|meios|alternativa|alternativas|opcao|opcoes|outras?|outros?|sem|"
            r"how|get|go|route|time|when|next|departure|catch|bus|train|transport|"
            r"transit|modes?|option|options|alternative|without)\b",
            flags=re.IGNORECASE,
        )
        if transport_follow_up_re.search(normalized):
            return {}
        if re.search(r"\b(?:mais|more|outro|outra|outros|outras|another|other)\b", normalized):
            return {}

        return {
            "message": message,
            "agents": ["researcher"],
            "routing_reasoning": "Compact place-only follow-up after transport resolved as place information, not a reused route.",
        }

    def _extract_meal_anchor_from_plan(self, ascii_noun: str = "restaurante") -> Dict[str, str]:
        """Return the first matching planner meal venue/time from stored output."""
        anchors = self._extract_meal_anchors_from_plan(ascii_noun)
        return anchors[0] if anchors else {}

    def _meal_anchor_clarification(self, anchors: List[Dict[str, str]], language: str) -> str:
        """Build a clarification question for ambiguous meal references."""
        visible = []
        for anchor in anchors[:4]:
            name = str(anchor.get("name") or "").strip()
            time_value = str(anchor.get("time") or "").strip()
            if not name:
                continue
            visible.append(f"- **{name}**" + (f" ({time_value})" if time_value else ""))
        if language == "pt":
            return (
                "### 🧭 **Preciso de confirmar o restaurante**\n\n"
                "✅ **Resposta direta:** a referência é ambígua porque há mais do que um restaurante/refeição no plano anterior.\n\n"
                + "\n".join(visible)
                + "\n\nDiz-me qual deles queres usar e eu calculo a rota certa."
            ).strip()
        return (
            "### 🧭 **Restaurant needs confirmation**\n\n"
            "✅ **Direct answer:** the reference is ambiguous because the previous plan includes more than one restaurant/meal.\n\n"
            + "\n".join(visible)
            + "\n\nTell me which one you mean and I will calculate the correct route."
        ).strip()

    @staticmethod
    def _extract_follow_up_origin(message: str) -> str:
        """Extract an explicit origin from a compact follow-up message."""
        if not message:
            return ""
        patterns = (
            r"\b(?:desde|a partir de|from)\s+(?P<origin>.+?)(?:\s+(?:para|até|ate|to)\b|[?.,;]|$)",
            r"\b(?:de|do|da)\s+(?P<origin>.+?)\s+(?:para|até|ate|ao|à|a|to)\b",
        )
        for pattern in patterns:
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if not match:
                continue
            origin = re.sub(r"\s+", " ", match.group("origin")).strip(" .,:;?!")
            origin = re.sub(r"^(?:o|a|os|as|the)\s+", "", origin, flags=re.IGNORECASE)
            if len(origin) >= 3:
                return origin
        return ""

    @staticmethod
    def _build_meal_transport_follow_up_message(
        meal_anchor: Dict[str, str],
        language: str,
        *,
        origin: str,
        ask_departure_time: bool = False,
    ) -> str:
        """Build a grounded transport follow-up for a stored meal venue."""
        name = str(meal_anchor.get("name") or "o restaurante indicado").strip()
        address = str(meal_anchor.get("address") or "").strip()
        destination = address or name
        time_value = str(meal_anchor.get("time") or "").strip()
        if language == "pt":
            parts = [f"Como vou de {origin} para {destination}?"]
            if address and name:
                parts.append(f"O destino é o restaurante {name}.")
            if time_value and ask_departure_time:
                parts.append(f"Quero chegar às {time_value}.")
            if ask_departure_time:
                parts.append("Diz-me também a que horas devo sair.")
            return " ".join(parts)
        parts = [f"How do I get from {origin} to {destination}?"]
        if address and name:
            parts.append(f"The destination is the restaurant {name}.")
        if time_value and ask_departure_time:
            parts.append(f"I want to arrive by {time_value}.")
        if ask_departure_time:
            parts.append("Also tell me when I should leave.")
        return " ".join(parts)

    @staticmethod
    def _is_researcher_no_more_pagination_response(text: str) -> bool:
        """Return whether a Researcher answer is a grounded empty continuation page."""
        normalized = MultiAgentAssistant._fold_context_text(text)
        return bool(
            re.search(
                r"\b(?:nao encontrei mais (?:eventos|locais|lugares)|"
                r"no more confirmed (?:events|places)|"
                r"did not find more confirmed (?:events|places)|"
                r"did not find more (?:events|places))\b",
                normalized,
            )
        )

    @staticmethod
    def _extract_route_pair_from_text(text: str) -> Dict[str, str]:
        """Extract a visible origin/destination pair from a route answer or query."""
        if not text:
            return {}
        patterns = (
            r"(?m)^#{1,6}\s+.*?\*\*(?P<origin>[^*→\n]{2,120})\s*→\s*(?P<destination>[^*\n]{2,160})\*\*\s*$",
            r"(?:Op[cç][oõ]es\s+apenas\s+de\s+[^*\n]+?\s+para|Bus-only\s+options\s+for)\s*(?P<origin>[^→\n]{2,120})\s*→\s*(?P<destination>[^*\n]{2,160})",
            r"(?:Rota\s+de\s+transporte\s+p[úu]blico|Public\s+transport\s+route|Trajeto|Route):\s*(?P<origin>[^→\n]{2,120})\s*→\s*(?P<destination>[^*\n]{2,160})",
            r"\bfrom\s+(?P<origin>.+?)\s+to\s+(?P<destination>[^?.!\n]{2,160})",
            r"\b(?:de|do|da|desde|from)\s+(?P<origin>.+?)\s+(?:para|aos|às|ao|à|at[eé]|ate|a|to)\s+(?P<destination>[^?.!\n]{2,160})",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            origin = re.sub(r"[*_`#\[\]()]|https?://\S+", "", match.group("origin"))
            destination = re.sub(r"[*_`#\[\]()]|https?://\S+", "", match.group("destination"))
            origin = re.sub(
                r"^\s*(?:rota\s+de\s+transporte\s+p[úu]blico|public\s+transport\s+route|trajeto|route)\s*:\s*",
                "",
                origin,
                flags=re.IGNORECASE,
            )
            origin = re.sub(
                r"^\s*(?:uma\s+|um\s+)?(?:alternativa|op[cç][aã]o|meios?\s+de\s+transporte|"
                r"transportes?|transporte\s+p[úu]blico|transportes\s+p[úu]blicos|public\s+transport)\s+"
                r"(?:de|do|da|dos|das|from)\s+",
                "",
                origin,
                flags=re.IGNORECASE,
            )
            origin = re.sub(
                r"^\s*(?:metro|autocarros?|comboios?|bus(?:es)?|train(?:s)?)\s+"
                r"(?:de|do|da|desde|from)\s+",
                "",
                origin,
                flags=re.IGNORECASE,
            )
            origin = re.sub(r"^\s*(?:o|a|os|as|the)\s+", "", origin, flags=re.IGNORECASE)
            destination = re.sub(r"^\s*(?:o|a|os|as|the)\s+", "", destination, flags=re.IGNORECASE)
            destination = re.sub(
                r"\s+(?:de\s+metro|de\s+autocarro|de\s+comboio|by\s+metro|by\s+bus|by\s+train)\b.*$",
                "",
                destination,
                flags=re.IGNORECASE,
            )
            origin = re.sub(r"\s+", " ", origin).strip(" .,:;?!-")
            destination = re.sub(r"\s+", " ", destination).strip(" .,:;?!-")
            folded_origin = MultiAgentAssistant._fold_context_text(origin)
            folded_destination = MultiAgentAssistant._fold_context_text(destination)
            if re.search(
                r"\b(?:preciso de confirmar|confirmar|correspondencia clara|ambiguidade|nao encontrei|não encontrei)\b",
                folded_origin,
            ):
                continue
            if re.search(r"\b(?:indica a morada|specify the address|provide the exact address)\b", folded_destination):
                continue
            if MultiAgentAssistant._fold_context_text(origin) in {"metro", "autocarro", "autocarros", "bus", "comboio", "train"}:
                continue
            if len(origin) >= 2 and len(destination) >= 2:
                return {"origin": origin, "destination": destination}
        return {}

    @staticmethod
    def _transport_follow_up_mode_hint(message: str, language: str) -> str:
        """Return a transport-mode preference expressed in an elliptical follow-up."""
        normalized = MultiAgentAssistant._fold_context_text(message)
        if re.search(r"\b(?:sem\s+autocarro|without\s+(?:the\s+)?bus|no\s+bus)\b", normalized):
            return "evitando autocarro" if language == "pt" else "avoiding bus"
        if re.search(r"\b(?:sem\s+metro|without\s+(?:the\s+)?metro|no\s+metro)\b", normalized):
            return "evitando metro" if language == "pt" else "avoiding metro"
        if re.search(
            r"\b(?:(?:autocarro|bus)\s+(?:ou|or)\s+(?:comboio|train)|(?:comboio|train)\s+(?:ou|or)\s+(?:autocarro|bus))\b",
            normalized,
        ):
            return "comparar autocarro e comboio" if language == "pt" else "compare bus and train"
        if re.search(
            r"\b(?:(?:metro)\s+(?:ou|or)\s+(?:autocarro|bus)|(?:autocarro|bus)\s+(?:ou|or)\s+(?:metro))\b",
            normalized,
        ):
            return "comparar metro e autocarro" if language == "pt" else "compare metro and bus"
        if re.search(
            r"\b(?:(?:metro)\s+(?:ou|or)\s+(?:comboio|train)|(?:comboio|train)\s+(?:ou|or)\s+(?:metro))\b",
            normalized,
        ):
            return "comparar metro e comboio" if language == "pt" else "compare metro and train"
        if re.search(r"\b(?:outros?\s+meios?\s+de\s+transporte|outros?\s+transportes?|meios?\s+de\s+transporte|other\s+(?:transport|transit)\s+modes?|other\s+ways?)\b", normalized):
            return "comparar meios suportados" if language == "pt" else "compare supported modes"
        if re.search(
            r"\b(?:outros?|outras?|mais|other|more)\s+(?:autocarros?|linhas?\s+de\s+autocarro|buses|bus\s+lines?)\b",
            normalized,
        ):
            return "outros autocarros" if language == "pt" else "other buses"
        if re.search(
            r"\b(?:outros?|outras?|mais|other|more)\s+(?:eletricos?|linhas?\s+de\s+eletrico|trams|tram\s+lines?)\b",
            normalized,
        ):
            return "outros elétricos" if language == "pt" else "other trams"
        if re.search(r"\b(?:de\s+metro|metro)\b", normalized):
            return "de metro" if language == "pt" else "by metro"
        if re.search(r"\b(?:de\s+autocarro|autocarros?|bus|buses)\b", normalized):
            return "de autocarro" if language == "pt" else "by bus"
        if re.search(r"\b(?:de\s+comboio|comboio|train)\b", normalized):
            return "de comboio" if language == "pt" else "by train"
        return "com uma alternativa diferente" if language == "pt" else "with a different alternative"

    @staticmethod
    def _extract_mode_destination_follow_up(message: str, language: str) -> Dict[str, str]:
        """Extract compact follow-ups such as ``E de metro até ao Saldanha?``."""
        match = re.search(
            r"^\s*(?:e\s+)?(?:de\s+|by\s+)?"
            r"(?P<mode>metro|autocarros?|bus(?:es)?|comboios?|train(?:s)?)\s+"
            r"(?:at[eé]|para|to)\s+"
            r"(?:(?:ao|aos|à|às|a|o|os|as|the)\s+)?"
            r"(?P<destination>[^?!.;,]+)",
            str(message or ""),
            flags=re.IGNORECASE,
        )
        if not match:
            return {}
        destination = re.sub(r"\s+", " ", match.group("destination")).strip(" .,:;?!")
        if len(destination) < 2:
            return {}
        mode = MultiAgentAssistant._fold_context_text(match.group("mode"))
        if "metro" in mode:
            mode_hint = "de metro" if language == "pt" else "by metro"
        elif "comboio" in mode or "train" in mode:
            mode_hint = "de comboio" if language == "pt" else "by train"
        else:
            mode_hint = "de autocarro" if language == "pt" else "by bus"
        return {"destination": destination, "mode_hint": mode_hint}

    @staticmethod
    def _rewrite_transport_alternative_request(
        *,
        origin: str,
        destination: str,
        mode_hint: str,
        language: str,
    ) -> str:
        """Builds a natural point-to-point follow-up without turning the mode into a place."""
        if language == "pt":
            if mode_hint == "comparar meios suportados":
                return (
                    f"Quero ir de {origin} para {destination}. "
                    "Compara os meios de transporte público suportados por dados disponíveis "
                    "e dá-me outra opção se existir."
                )
            if mode_hint.startswith("comparar "):
                compared = mode_hint.removeprefix("comparar ").strip()
                return (
                    f"Quero ir de {origin} para {destination}. Compara {compared}. "
                    "Avalia as opções suportadas por dados disponíveis, explica limitações e recomenda a melhor."
                )
            if mode_hint == "de metro":
                return (
                    f"Quero ir de metro de {origin} para {destination}. "
                    "Usa Metro de Lisboa se houver uma ligação suportada e não uses autocarro como recomendação principal."
                )
            if mode_hint == "de autocarro":
                return (
                    f"Quero ir de autocarro de {origin} para {destination}. "
                    "Usa Carris quando for aplicável e não uses metro como recomendação principal."
                )
            if mode_hint == "outros autocarros":
                return (
                    f"Quero ir de {origin} para {destination}. "
                    "Dá-me outros autocarros confirmados pela Carris; "
                    "se só houver uma linha, diz isso claramente."
                )
            if mode_hint == "outros elétricos":
                return (
                    f"Quero ir de {origin} para {destination}. "
                    "Dá-me outros elétricos confirmados pela Carris; "
                    "se só houver uma linha, diz isso claramente."
                )
            if mode_hint == "de comboio":
                return (
                    f"Quero ir de comboio de {origin} para {destination}. "
                    "Usa CP suburbano quando for aplicável e indica claramente se não houver ligação suportada."
                )
            if mode_hint == "evitando autocarro":
                return f"Quero ir de {origin} para {destination}, evitando autocarro. Dá-me a melhor alternativa suportada."
            if mode_hint == "evitando metro":
                return f"Quero ir de {origin} para {destination}, evitando metro. Dá-me a melhor alternativa suportada."
            return f"Quero ir de {origin} para {destination}. Dá-me uma alternativa diferente suportada."

        if mode_hint == "compare supported modes":
            return (
                f"I want to go from {origin} to {destination}. "
                "Compare the public transport modes supported by available data and give me another option if one exists."
            )
        if mode_hint.startswith("compare "):
            compared = mode_hint.removeprefix("compare ").strip()
            return (
                f"I want to go from {origin} to {destination}. Compare {compared}. "
                "Evaluate the options supported by available data, explain limitations, and recommend the best one."
            )
        if mode_hint == "by metro":
            return (
                f"I want to go by metro from {origin} to {destination}. "
                "Use Metro de Lisboa if a supported connection exists and do not use bus as the main recommendation."
            )
        if mode_hint == "by bus":
            return (
                f"I want to go by bus from {origin} to {destination}. "
                "Use Carris when applicable and do not use metro as the main recommendation."
            )
        if mode_hint == "other buses":
            return (
                f"I want to go from {origin} to {destination}. "
                "Give me other buses confirmed by Carris; if there is only one line, say that clearly."
            )
        if mode_hint == "other trams":
            return (
                f"I want to go from {origin} to {destination}. "
                "Give me other trams confirmed by Carris; if there is only one line, say that clearly."
            )
        if mode_hint == "by train":
            return (
                f"I want to go by train from {origin} to {destination}. "
                "Use CP suburban rail when applicable and say clearly if no supported connection exists."
            )
        if mode_hint == "avoiding bus":
            return f"I want to go from {origin} to {destination}, avoiding bus. Give me the best supported alternative."
        if mode_hint == "avoiding metro":
            return f"I want to go from {origin} to {destination}, avoiding metro. Give me the best supported alternative."
        return f"I want to go from {origin} to {destination}. Give me a different supported option."

    @staticmethod
    def _extract_location_ambiguity_options(text: str) -> Dict[str, Any]:
        """Extract user-visible ambiguous-location options from a final answer."""
        if not text:
            return {}
        heading_match = re.search(
            r"(?:Ambiguidade em|Ambiguity in)\s+'(?P<fragment>[^']{2,80})'",
            text,
            flags=re.IGNORECASE,
        )
        if not heading_match:
            return {}

        options: List[Dict[str, str]] = []
        for option_match in re.finditer(
            r"(?m)^\s*[-*]\s*(?P<letter>[A-Z])\)\s*(?:[^\w*]+\s*)?\*\*(?P<label>[^*\n]{2,160})\*\*",
            text,
        ):
            label = re.sub(r"\s+", " ", option_match.group("label")).strip(" .,:;")
            if not label:
                continue
            options.append({
                "letter": option_match.group("letter").upper(),
                "label": label,
            })

        return {
            "fragment": heading_match.group("fragment").strip(),
            "options": options[:6],
        } if options else {}

    @staticmethod
    def _location_option_matches_reply(reply: str, option: Dict[str, str]) -> bool:
        """Return whether a compact user reply selects one ambiguous option."""
        normalized_reply = MultiAgentAssistant._fold_context_text(reply)
        if not normalized_reply:
            return False

        letter = str(option.get("letter") or "").strip().lower()
        if letter and re.fullmatch(rf"(?:{re.escape(letter)}|(?:opcao|option)\s+{re.escape(letter)})", normalized_reply):
            return True

        label = str(option.get("label") or "").strip()
        normalized_label = MultiAgentAssistant._fold_context_text(label)
        if not normalized_label:
            return False

        match_reply = MultiAgentAssistant._fold_location_match_text(reply)
        match_label = MultiAgentAssistant._fold_location_match_text(label)
        if match_reply == match_label:
            return True
        if len(match_label) >= 8 and match_label in match_reply:
            return True

        if normalized_reply == normalized_label:
            return True
        if len(normalized_reply) >= 3 and normalized_reply in normalized_label:
            return True

        stop_terms = {
            "and", "ate", "com", "da", "das", "de", "do", "dos", "em",
            "from", "ir", "na", "nas", "no", "nos", "of", "para", "quero",
            "the", "to", "want",
        }
        reply_terms = {
            term for term in match_reply.split()
            if len(term) >= 3 and term not in stop_terms
        }
        label_terms = {
            term for term in match_label.split()
            if len(term) >= 3 and term not in stop_terms
        }
        if len(label_terms) >= 2 and label_terms.issubset(reply_terms):
            return True
        return bool(reply_terms and reply_terms.issubset(label_terms))

    @staticmethod
    def _fold_location_match_text(text: str) -> str:
        """Normalize a location string for option and route-side matching."""
        folded = MultiAgentAssistant._fold_context_text(text)
        return re.sub(r"[^a-z0-9]+", " ", folded).strip()

    @staticmethod
    def _location_fragment_matches_route_side(fragment: str, route_side: str) -> bool:
        """Return whether an ambiguity fragment refers to one side of a route."""
        folded_fragment = MultiAgentAssistant._fold_location_match_text(fragment)
        folded_side = MultiAgentAssistant._fold_location_match_text(route_side)
        if not folded_fragment or not folded_side:
            return False
        return (
            folded_fragment == folded_side
            or folded_fragment in folded_side
            or folded_side in folded_fragment
        )

    @staticmethod
    def _standalone_query_from_malformed_pending_route(message: str, route_pair: Dict[str, str]) -> str:
        """Extract a new standalone query accidentally placed after a route ``to``."""
        candidates = [
            re.sub(r"\s+", " ", match.group("candidate")).strip(" .,:;?!")
            for match in re.finditer(
                r"\b(?:para|to)\s+(?P<candidate>.+)$",
                str(message or ""),
                flags=re.IGNORECASE | re.DOTALL,
            )
        ]
        candidates.append(re.sub(r"\s+", " ", str(route_pair.get("destination") or "")).strip(" .,:;?!"))
        candidate = next((item for item in candidates if len(item) >= 8), "")
        if len(candidate) < 8:
            return ""
        folded = MultiAgentAssistant._fold_context_text(candidate)
        if not re.match(
            r"^(?:quero|queria|gostava|podes|pode|diz|diz-me|lista|mostra|"
            r"qual|quais|que|ha|onde|como|what|where|which|show|list|give|tell|find)\b",
            folded,
        ):
            return ""
        if not re.search(
            r"\b(?:eventos?|cultura|atracoes|atra[cç][oõ]es|locais|lugares|"
            r"tempo|previsao|meteorologia|farmacias?|hospitais?|restaurantes?|"
            r"roteiro|itinerario|itinerary|events?|weather|places?|restaurants?)\b",
            folded,
        ):
            return ""
        return candidate

    @staticmethod
    def _is_new_standalone_domain_request(message: str) -> bool:
        """Return whether a pending clarification should be cleared for a new request."""
        folded = MultiAgentAssistant._fold_context_text(message)
        if not folded:
            return False
        has_request_cue = bool(
            re.match(
                r"^(?:que|quais|qual|quando|onde|ha|mostra|lista|procura|pesquisa|"
                r"recomenda|diz|diz-me|quero\s+(?:saber|ver|eventos?|locais|"
                r"restaurantes?|monumentos?|museus?|roteiro|itinerario)|"
                r"what|which|where|when|show|list|find|tell|give|recommend|"
                r"eventos?|events?|metro|comboio|train|autocarro|bus)\b",
                folded,
            )
            or "?" in str(message or "")
        )
        if not has_request_cue:
            return False
        return bool(
            re.search(
                r"\b(?:eventos?|events?|desportiv|sports?|musica|music|concertos?|"
                r"concerts?|festival|festivais|teatro|theatre|exposic|exhibit|"
                r"monumentos?|monuments?|museus?|museums?|miradouros?|viewpoints?|"
                r"restaurantes?|restaurants?|fado|gastronom|casas?\s+de\s+banho|"
                r"wc|sanitarios?|sanitarias?|farmacias?|pharmacies|hospitais?|"
                r"previsao|meteorologia|weather|roteiro|itinerario|itinerary|"
                r"metro|comboio|train|autocarro|bus|transportes?|transport|"
                r"atrasos?|delays?|perturbacoes?|disruptions?|estado|status)\b",
                folded,
            )
        )

    @staticmethod
    def _route_destination_from_location_option(label: str) -> str:
        """Prefer the address part of an ambiguity option while keeping the place name."""
        cleaned = re.sub(r"\s+", " ", str(label or "")).strip(" .,:;")
        parts = [part.strip(" .,:;") for part in cleaned.split(",") if part.strip(" .,:;")]
        if len(parts) < 2:
            return cleaned

        name = parts[0]
        address = ", ".join(parts[1:])
        address_has_street_signal = bool(
            re.search(
                r"\b(?:avenida|av\.?|rua|r\.?|largo|pra[cç]a|travessa|estrada|alameda|campo|cal[cç]ada)\b",
                address,
                flags=re.IGNORECASE,
            )
        )
        if address and name and address_has_street_signal:
            return f"{name}, {address}"
        return cleaned

    def _store_pending_location_clarification(
        self,
        message: str,
        final_output: str,
        effective_agent_set: Set[str],
    ) -> None:
        """Remember route context when the answer asks the user to disambiguate a location."""
        anchors = self._get_conversation_anchors()
        anchors["pending_location_clarification"] = {}
        ambiguity = self._extract_location_ambiguity_options(final_output)
        if not ambiguity:
            return

        route_pair = self._extract_route_pair_from_text(message)
        if "transport" not in effective_agent_set or not route_pair:
            return

        fragment = str(ambiguity.get("fragment") or route_pair.get("destination") or "").strip()
        ambiguous_field = "destination"
        if self._location_fragment_matches_route_side(fragment, route_pair.get("origin", "")):
            ambiguous_field = "origin"
        elif self._location_fragment_matches_route_side(fragment, route_pair.get("destination", "")):
            ambiguous_field = "destination"

        anchors["pending_location_clarification"] = {
            "kind": "transport_route",
            "ambiguous_field": ambiguous_field,
            "ambiguous_fragment": fragment,
            "origin": route_pair.get("origin", "") if ambiguous_field == "destination" else "",
            "destination": route_pair.get("destination", "") if ambiguous_field == "origin" else "",
            "destination_fragment": fragment,
            "options": list(ambiguity.get("options") or []),
            "source_message": message[:300],
            "language": str((self.state.get("user_context") or {}).get("language") or ""),
        }

    def _resolve_pending_location_clarification_follow_up(self, message: str, language: str) -> Dict[str, Any]:
        """Resolve a short clarification reply into the original route request."""
        anchors = self._get_conversation_anchors()
        pending = anchors.get("pending_location_clarification")
        if not isinstance(pending, dict) or not pending:
            return {}
        if pending.get("kind") != "transport_route":
            return {}

        options = [opt for opt in pending.get("options") or [] if isinstance(opt, dict)]
        matches = [opt for opt in options if self._location_option_matches_reply(message, opt)]
        if len(matches) != 1 and self._is_new_standalone_domain_request(message):
            anchors["pending_location_clarification"] = {}
            return {
                "message": message,
                "routing_reasoning": "Stale pending location clarification ignored because the user asked a new standalone question.",
            }
        fresh_route_pair = self._extract_route_pair_from_text(message)
        if fresh_route_pair and len(matches) != 1:
            standalone_query = self._standalone_query_from_malformed_pending_route(message, fresh_route_pair)
            anchors["pending_location_clarification"] = {}
            if standalone_query:
                return {
                    "message": standalone_query,
                    "routing_reasoning": "Stale pending location clarification ignored because the user asked a new standalone question.",
                }
            return {}

        ambiguous_field = str(pending.get("ambiguous_field") or "destination").strip().lower()
        if ambiguous_field not in {"origin", "destination"}:
            ambiguous_field = "destination"
        response_language = str(pending.get("language") or "").strip()
        if response_language not in {"pt", "en"}:
            response_language = language
        selected_label = ""
        if len(matches) == 1:
            selected_label = str(matches[0].get("label") or "").strip()
            resolved_location = self._route_destination_from_location_option(selected_label)
        else:
            original_reply = re.sub(r"\s+", " ", str(message or "")).strip(" .,:;?!")
            area_only_reply = bool(
                re.match(
                    r"^(?:é|e|fica|seria|it'?s|it is|is)?\s*(?:no|na|nos|nas|em|in|near)\s+",
                    self._fold_context_text(original_reply),
                )
            )
            cleaned_reply = re.sub(
                r"^\s*(?:é|e|fica|seria|it'?s|it is|is)\s+",
                "",
                original_reply,
                flags=re.IGNORECASE,
            )
            cleaned_reply = re.sub(
                r"^\s*(?:o|a|os|as|ao|à|no|na|nos|nas|em|the|in|near)\s+",
                "",
                cleaned_reply,
                flags=re.IGNORECASE,
            ).strip(" .,:;?!")
            fragment = re.sub(
                r"\s+",
                " ",
                str(pending.get("ambiguous_fragment") or pending.get("destination_fragment") or ""),
            ).strip(" .,:;?!")
            base_fragment = re.sub(
                r"\s+(?:em|no|na|nos|nas|perto\s+de|near|in)\s+.+$",
                "",
                fragment,
                flags=re.IGNORECASE,
            ).strip(" .,:;?!")
            if not cleaned_reply:
                return {}
            if area_only_reply and base_fragment and self._fold_context_text(cleaned_reply) not in self._fold_context_text(base_fragment):
                resolved_location = f"{base_fragment} no {cleaned_reply}" if response_language == "pt" else f"{base_fragment} in {cleaned_reply}"
            else:
                resolved_location = cleaned_reply
            selected_label = resolved_location

        if ambiguous_field == "origin":
            origin = resolved_location
            destination = str(pending.get("destination") or "").strip()
        else:
            origin = str(pending.get("origin") or "").strip()
            destination = resolved_location

        if not origin or not destination:
            return {}

        anchors["pending_location_clarification"] = {}
        if ambiguous_field == "destination":
            anchors["expected_transport_destination"] = {
                "label": selected_label,
                "route_destination": destination,
            }
        else:
            anchors["expected_transport_destination"] = {}
        if response_language == "pt":
            rewritten = f"Quero ir de {origin} para {destination}."
        else:
            rewritten = f"I want to go from {origin} to {destination}."
        return {
            "message": rewritten,
            "language": response_language,
            "agents": ["transport"],
            "routing_reasoning": "Pending location clarification resolved into the original point-to-point transport request.",
        }

    def _rebuild_single_transport_source_line(
        self,
        text: str,
        language: str,
        effective_agents: List[str],
    ) -> str:
        """Cite every visible operator in a single-domain transport response."""
        if "transport" not in set(effective_agents or []):
            return text
        if any(agent_name != "transport" for agent_name in effective_agents if not str(agent_name).startswith("_")):
            return text

        transport_agent = getattr(self, "agents", {}).get("transport") if isinstance(getattr(self, "agents", {}), dict) else None
        tool_names = []
        if transport_agent is not None and hasattr(transport_agent, "get_tool_calls_log"):
            tool_names = [
                call.get("tool_name")
                for call in transport_agent.get_tool_calls_log()
                if isinstance(call, dict)
            ]

        operators_used = operators_from_tool_names(tool_names)
        if "get_route_between_stations" in {str(name or "") for name in tool_names}:
            visible_transport_text = self._fold_context_text(text)
            if (
                "metro" not in operators_used
                and re.search(
                    r"\b(?:metro de lisboa|trajeto metro|linha\s+(?:amarela|azul|verde|vermelha))\b",
                    visible_transport_text,
                )
            ):
                operators_used = ["metro", *operators_used]
            if "carris metropolitana" in visible_transport_text and "carris_metropolitana" not in operators_used:
                operators_used = [*operators_used, "carris_metropolitana"]
            if (
                re.search(r"\b(?:autocarro|autocarros|bus|carris|linha\s+\d{3}|route\s+\d{3})\b", visible_transport_text)
                and "carris" not in operators_used
                and "carris_metropolitana" not in operators_used
            ):
                operators_used = [*operators_used, "carris"]

        if not operators_used:
            return text
        return final_visual_pass(rebuild_transport_source_line(text, operators_used, language=language))

    @staticmethod
    def _qa_gap_is_generic_service_area_route(
        user_message: str,
        transport_output: str,
        qa_result: Dict[str, Any],
    ) -> bool:
        """Return whether QA is asking for an unavailable exact service after an area route."""
        if not transport_output or not qa_result:
            return False
        visible = MultiAgentAssistant._fold_context_text(transport_output)
        combined = MultiAgentAssistant._fold_context_text(
            " ".join(
                [
                    user_message or "",
                    transport_output or "",
                    *[str(item or "") for item in qa_result.get("missing_data", [])],
                ]
            )
        )
        if not re.search(r"\b(?:usei|used)\b.{0,140}\b(?:ponto de referencia|ponto de referência|destination reference|area)\b", visible):
            return False
        if not re.search(r"\b(?:veterinario|veterinaria|veterinary|farmacia|pharmacy|restaurante|restaurant|taberna|loja|store|shop)\b", combined):
            return False
        if not re.search(r"\b(?:apanha em|board at|paragens|stops|proximas partidas|próximas partidas|next departures)\b", visible):
            return False
        return bool(
            re.search(
                r"\b(?:destino especifico|destino específico|morada|address|nome|ponto de chegada|final)\b",
                combined,
            )
        )

    def _resolve_onward_transport_follow_up(self, message: str, language: str) -> Dict[str, Any]:
        """Resolve short onward-route follow-ups from the previous transport destination."""
        normalized = self._fold_context_text(message)
        if not re.search(r"^\s*(?:e\s+)?(?:depois|a\s+seguir|then|and\s+then)\b", normalized):
            return {}

        anchors = self._get_conversation_anchors()
        last_agents = {str(agent) for agent in anchors.get("last_response_agents") or []}
        route = anchors.get("last_transport_route") if isinstance(anchors.get("last_transport_route"), dict) else {}
        if "transport" not in last_agents:
            return {}

        origin = str(route.get("destination") or "").strip()
        destination_match = re.search(
            r"\b(?:para|at(?:e|\u00e9)|to)\s+(?P<destination>[^?!.;,]+)",
            message,
            flags=re.IGNORECASE,
        )
        if not origin or not destination_match:
            return {}

        destination = re.sub(r"\s+", " ", destination_match.group("destination")).strip(" .,:;?!")
        destination = re.sub(
            r"\s+(?:de\s+metro|de\s+autocarro|de\s+comboio|by\s+metro|by\s+bus|by\s+train)\b.*$",
            "",
            destination,
            flags=re.IGNORECASE,
        ).strip(" .,:;?!")
        if len(destination) < 2:
            return {}

        rewritten = (
            f"Como vou de {origin} para {destination}?"
            if language == "pt"
            else f"How do I get from {origin} to {destination}?"
        )
        return {
            "message": rewritten,
            "agents": ["transport"],
            "routing_reasoning": "Onward route follow-up resolved from the previous transport destination.",
        }

    def _repair_wrong_contextual_transport_destination(self, text: str, language: str) -> str:
        """Replace a contextual route if geocoding clearly returned the wrong selected option."""
        anchors = self._get_conversation_anchors()
        expected = anchors.get("expected_transport_destination")
        if not isinstance(expected, dict) or not expected:
            return text

        label = re.sub(r"\s+", " ", str(expected.get("label") or "")).strip(" .,:;")
        if not label:
            anchors["expected_transport_destination"] = {}
            return text

        normalized_text = self._fold_context_text(text)
        normalized_label = self._fold_context_text(label)
        name = normalized_label.split(",", 1)[0].strip()
        stop_terms = {
            "avenida", "av", "centro", "comercial", "colombo", "lisboa",
            "loja", "rua", "street", "the",
        }
        specific_terms = [
            term for term in re.findall(r"[a-z0-9]+", name)
            if len(term) >= 3 and term not in stop_terms
        ]
        if not specific_terms:
            anchors["expected_transport_destination"] = {}
            return text
        if any(term in normalized_text for term in specific_terms):
            anchors["expected_transport_destination"] = {}
            return text

        anchors["expected_transport_destination"] = {}
        title = "Preciso de confirmar o destino" if language == "pt" else "I Need To Confirm The Destination"
        direct = (
            "não consegui validar uma rota fiável para o destino que escolheste; "
            "a resolução automática apontou para outro local, por isso não vou apresentar esse percurso como correto."
            if language == "pt"
            else "I could not validate a reliable route to the destination you selected; "
            "the automatic resolver pointed to another place, so I will not present that route as correct."
        )
        suggestion = (
            "Indica um ponto de referência próximo, uma estação/paragem de partida ou coordenadas, "
            "para eu recalcular sem confundir com outro local."
            if language == "pt"
            else "Send a nearby landmark, a departure station/stop, or coordinates "
            "so I can recalculate without confusing it with another place."
        )
        return (
            f"### 🧭 **{title}**\n\n"
            f"✅ **{'Resposta direta' if language == 'pt' else 'Direct answer'}:** {direct}\n\n"
            "---\n\n"
            f"- 📍 **{'Destino escolhido' if language == 'pt' else 'Selected destination'}:** {label}\n"
            f"- 💡 **{'Como corrigir' if language == 'pt' else 'How to fix it'}:** {suggestion}"
        )

    def _resolve_transport_alternative_follow_up(self, message: str, language: str) -> Dict[str, Any]:
        """Resolve short transport alternative follow-ups using the last route only."""
        normalized = self._fold_context_text(message)
        anchors = self._get_conversation_anchors()
        last_agents = {str(agent) for agent in anchors.get("last_response_agents") or []}
        route = anchors.get("last_transport_route") if isinstance(anchors.get("last_transport_route"), dict) else {}
        mode_destination = self._extract_mode_destination_follow_up(message, language)
        if mode_destination and "transport" in last_agents:
            origin = str(route.get("origin") or "").strip()
            destination = str(mode_destination.get("destination") or "").strip()
            if origin and destination:
                rewritten = self._rewrite_transport_alternative_request(
                    origin=origin,
                    destination=destination,
                    mode_hint=str(mode_destination.get("mode_hint") or ""),
                    language=language,
                )
                return {
                    "message": rewritten,
                    "agents": ["transport"],
                    "routing_reasoning": "Conversation route anchor resolved into a mode-specific destination follow-up.",
                }
        if re.search(
            r"\b(?:from\s+.+?\s+to\s+.+|de\s+.+?\s+(?:para|ate|a|ao)\s+.+)",
            normalized,
        ):
            return {}
        if not re.search(
            r"\b(?:alternativa|alternativas|outra\s+opcao|outras\s+opcoes|outro\s+caminho|"
            r"outros?\s+meios?\s+de\s+transporte|outros?\s+transportes?|meios?\s+de\s+transporte|"
            r"outros?\s+autocarros?|outras?\s+linhas?\s+de\s+autocarro|mais\s+autocarros?|"
            r"outros?\s+el[eé]tricos?|outras?\s+linhas?\s+de\s+el[eé]trico|"
            r"(?:metro|autocarro|comboio|bus|train)\s+(?:ou|or)\s+(?:metro|autocarro|comboio|bus|train)|"
            r"e\s+de\s+(?:metro|autocarro|comboio)|(?:ir|vou|preferia|prefiro|preferir|quiser)\s+de\s+(?:metro|autocarro|comboio)|"
            r"(?:preferia|prefiro|preferir|quiser)\s+(?:metro|autocarro|comboio)|"
            r"sem\s+(?:metro|autocarro|comboio)|"
            r"alternative|another\s+(?:option|route|way)|other\s+(?:transport|transit)\s+modes?|other\s+ways?|"
            r"other\s+buses?|more\s+buses?|other\s+bus\s+lines?|other\s+trams?|more\s+trams?|other\s+tram\s+lines?|"
            r"(?:go|travel|prefer|want)\s+by\s+(?:metro|bus|train)|"
            r"(?:prefer|want)\s+(?:metro|bus|train)|"
            r"without\s+(?:metro|bus|train)|by\s+(?:metro|bus|train))\b",
            normalized,
        ):
            return {}
        if "transport" not in last_agents:
            return {}
        origin = str(route.get("origin") or "").strip()
        destination = str(route.get("destination") or "").strip()
        if not origin or not destination:
            clarification = (
                "Diz-me a origem e o destino para eu comparar uma alternativa de transporte."
                if language == "pt"
                else "Tell me the origin and destination so I can compare a transport alternative."
            )
            return {"clarification": clarification}
        mode_hint = self._transport_follow_up_mode_hint(message, language)
        rewritten = self._rewrite_transport_alternative_request(
            origin=origin,
            destination=destination,
            mode_hint=mode_hint,
            language=language,
        )
        return {
            "message": rewritten,
            "agents": ["transport"],
            "routing_reasoning": "Conversation route anchor resolved into a transport alternative request.",
        }

    def _resolve_research_pagination_follow_up(self, message: str, language: str) -> Dict[str, Any]:
        """Force Researcher for short 'more/another' follow-ups when a search page exists."""
        researcher = self.agents.get("researcher")
        if researcher is None:
            return {}
        extract_pagination = getattr(researcher, "_extract_pagination_request", None)
        infer_domain = getattr(researcher, "_infer_search_domain_from_query", None)
        if not callable(extract_pagination):
            return {}
        pagination_request = extract_pagination(message)
        if not pagination_request:
            return {}
        anchors = self._get_conversation_anchors()
        cached_context = getattr(researcher, "_last_search_context", None) or anchors.get("last_research_context")
        if not isinstance(cached_context, dict) or not cached_context:
            return {}
        if getattr(researcher, "_last_search_context", None) is None:
            setattr(researcher, "_last_search_context", cached_context)
        explicit_domain = infer_domain(message) if callable(infer_domain) else None
        cached_domain = str(cached_context.get("domain") or "").strip()
        last_agents = {str(agent) for agent in anchors.get("last_response_agents") or []}
        short_generic = len((message or "").split()) <= 4 and not explicit_domain
        if short_generic and "researcher" not in last_agents:
            return {}
        if explicit_domain and cached_domain and explicit_domain != cached_domain:
            return {}
        normalized_message = re.sub(r"\s+", " ", (message or "").lower()).strip()
        rejects_museums = bool(
            re.search(
                r"\b(?:n[aã]o\s+(?:sejam|quero|incluas?)\s+museus|sem\s+museus|not\s+museums?|not\s+museum|no\s+museums?)\b",
                normalized_message,
                flags=re.IGNORECASE,
            )
        )
        if cached_domain == "places" and rejects_museums:
            base_args = cached_context.get("base_args") if isinstance(cached_context.get("base_args"), dict) else {}
            previous_query = " ".join(
                str(value or "")
                for value in (
                    base_args.get("query"),
                    cached_context.get("source_query"),
                )
            )
            location_match = re.search(
                r"\b(?:perto\s+d[eo]?\s+|near\s+|em\s+|in\s+)(Bel[eé]m|Baixa|Chiado|Alfama|Rossio|Oriente|Parque das Na[cç][oõ]es|Avenida da Igreja|Alc[aâ]ntara)\b",
                previous_query,
                flags=re.IGNORECASE,
            )
            location = location_match.group(1) if location_match else ("Lisboa" if language == "pt" else "Lisbon")
            rewritten = (
                f"Mostra jardins, miradouros e outros locais em {location}, sem museus."
                if language == "pt"
                else f"Show gardens, viewpoints, and other places in {location}, excluding museums."
            )
            setattr(researcher, "_last_search_context", None)
            return {
                "message": rewritten,
                "agents": ["researcher"],
                "routing_reasoning": "Conversation search context resolved into a new non-museum place search.",
            }
        return {
            "message": message,
            "agents": ["researcher"],
            "routing_reasoning": "Conversation search context resolved into a paginated Researcher follow-up.",
        }

    @staticmethod
    def _event_price_rank(price_text: str) -> tuple[int, float]:
        """Rank event prices with free entries first and unknown prices last."""
        normalized = MultiAgentAssistant._fold_context_text(price_text)
        if re.search(r"\b(?:gratuit\w*|gratis|free|entrada\s+livre)\b", normalized) or "0€" in normalized:
            return (0, 0.0)
        values = [
            float(value.replace(",", "."))
            for value in re.findall(r"\d+(?:[,.]\d+)?", str(price_text or ""))
        ]
        if values:
            return (1, min(values))
        return (2, 999999.0)

    @staticmethod
    def _extract_event_cards_from_answer(text: str) -> tuple[List[Dict[str, str]], str]:
        """Extract visible event cards and the source line from a previous answer."""
        cards: List[Dict[str, str]] = []
        current: Dict[str, str] | None = None
        source_line = ""
        card_re = re.compile(r"^\s*[-*]\s+\*\*(?P<title>.+?)\*\*\s*$")
        field_re = re.compile(
            r"^\s+[-*]\s+(?:[\U0001F300-\U0001FAFF\u2300-\u27BF\uFE0F\u200D]+\s*)?"
            r"\*\*(?P<label>[^:*]+):\*\*\s*(?P<value>.+?)\s*$"
        )

        for raw_line in str(text or "").splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("📌 **Fonte") or stripped.startswith("📌 **Source"):
                source_line = stripped
                continue
            match = card_re.match(raw_line)
            if match and not raw_line.startswith(("  ", "\t")):
                if current and current.get("title"):
                    cards.append(current)
                title = re.sub(r"^[\U0001F300-\U0001FAFF\u2300-\u27BF\uFE0F\u200D]+\s*", "", match.group("title")).strip()
                current = {"title": title}
                continue
            if not current:
                continue
            field_match = field_re.match(raw_line)
            if not field_match:
                if "Mais detalhes" in stripped or "More details" in stripped:
                    current["details"] = stripped
                elif "Bilhetes" in stripped or "Tickets" in stripped:
                    current["tickets"] = stripped
                continue
            label = MultiAgentAssistant._fold_context_text(field_match.group("label"))
            value = field_match.group("value").strip()
            if label in {"data/hora", "date/time", "quando", "when"}:
                current["when"] = value
            elif label in {"preco", "price"}:
                current["price"] = value
            elif label in {"categoria", "category"}:
                current["category"] = value
            elif label in {"morada", "address"}:
                current["address"] = value
            elif label in {"mais detalhes", "more details"}:
                current["details"] = stripped
            elif label in {"bilhetes", "tickets"}:
                current["tickets"] = stripped

        if current and current.get("title"):
            cards.append(current)
        event_cards = [
            card for card in cards
            if any(card.get(key) for key in ("when", "price", "details", "tickets"))
        ]
        return event_cards, source_line

    @staticmethod
    def _format_filtered_event_cards(
        cards: List[Dict[str, str]],
        *,
        language: str,
        source_line: str,
        omitted_count: int,
    ) -> str:
        """Render a compact answer for event filter follow-ups."""
        is_pt = language == "pt"
        title = "### 🎭 **Eventos encontrados**" if is_pt else "### 🎭 **Events found**"
        direct = (
            "✅ **Resposta direta:** filtrei a lista anterior e mantive apenas eventos com entrada gratuita ou preço explícito."
            if is_pt
            else "✅ **Direct answer:** I filtered the previous list and kept only events with free entry or explicit prices."
        )
        lines = [title, "", direct, "", "---", ""]
        date_label = "Data/Hora" if is_pt else "Date/Time"
        price_label = "Preço" if is_pt else "Price"
        category_label = "Categoria" if is_pt else "Category"

        for card in cards:
            lines.append(f"- **🎭 {card['title']}**")
            if card.get("when"):
                lines.append(f"    - 📅 **{date_label}:** {card['when']}")
            if card.get("price"):
                lines.append(f"    - 💰 **{price_label}:** {card['price']}")
            if card.get("category"):
                lines.append(f"    - 📂 **{category_label}:** {card['category']}")
            if card.get("details"):
                lines.append(f"    - {card['details'].lstrip('- ').strip()}")
            if card.get("tickets"):
                lines.append(f"    - {card['tickets'].lstrip('- ').strip()}")
            lines.append("")

        if omitted_count:
            note = (
                f"💡 Omiti {omitted_count} evento(s) da lista anterior sem preço confirmado."
                if is_pt
                else f"💡 I omitted {omitted_count} previous event(s) without confirmed price information."
            )
            lines.extend([note, ""])

        if not source_line:
            updated_label = "Atualizado" if is_pt else "Updated"
            source_name = "Fonte" if is_pt else "Source"
            source_line = (
                f"📌 **{source_name}:** [*VisitLisboa Eventos*](https://www.visitlisboa.com/pt-pt/eventos)"
                f" | **{updated_label}:** {datetime.now().strftime('%H:%M')}"
                if is_pt
                else f"📌 **{source_name}:** [*VisitLisboa Events*](https://www.visitlisboa.com/en/events)"
                f" | **{updated_label}:** {datetime.now().strftime('%H:%M')}"
            )
        lines.append(source_line)
        return "\n".join(lines).strip()

    def _resolve_event_filter_follow_up(self, message: str, language: str) -> Dict[str, Any]:
        """Answer price/free filters against the previous event result set."""
        normalized = self._fold_context_text(message)
        if not re.search(r"\b(?:destes|destas|desses|dessas|these|those|previous|above)\b", normalized):
            return {}
        wants_free = bool(re.search(r"\b(?:gratuit\w*|gratis|free)\b", normalized))
        wants_cheapest = bool(re.search(r"\b(?:barat\w*|cheap|cheapest|preco|price)\b", normalized))
        if not (wants_free or wants_cheapest):
            return {}

        previous_assistant = ""
        for msg in reversed(self.state.get("messages") or []):
            if isinstance(msg, AIMessage) and msg.content:
                previous_assistant = str(msg.content)
                break
        if not previous_assistant or not re.search(r"\b(?:eventos|events)\b", self._fold_context_text(previous_assistant)):
            return {}

        cards, source_line = self._extract_event_cards_from_answer(previous_assistant)
        if not cards:
            return {}

        priced_cards = [card for card in cards if card.get("price")]
        if wants_free and not wants_cheapest:
            selected = [
                card for card in priced_cards
                if self._event_price_rank(card.get("price", ""))[0] == 0
            ]
        else:
            selected = [
                card for card in priced_cards
                if self._event_price_rank(card.get("price", ""))[0] in {0, 1}
            ]
            selected.sort(key=lambda card: self._event_price_rank(card.get("price", "")))

        selected = selected[:5]
        omitted_count = max(0, len(cards) - len(selected))
        if not selected:
            response = (
                "### 🎭 **Eventos encontrados**\n\n"
                "✅ **Resposta direta:** na lista anterior não encontrei eventos com entrada gratuita ou preço confirmado; para não inventar preços, mantive a limitação explícita.\n\n"
                f"{source_line}"
                if language == "pt"
                else "### 🎭 **Events found**\n\n"
                "✅ **Direct answer:** I did not find events with free entry or confirmed prices in the previous list; I kept the limitation explicit instead of inventing prices.\n\n"
                f"{source_line}"
            ).strip()
            return {"direct_response": response}

        return {
            "direct_response": self._format_filtered_event_cards(
                selected,
                language=language,
                source_line=source_line,
                omitted_count=omitted_count,
            )
        }

    @staticmethod
    def _place_price_rank(card: Dict[str, str]) -> tuple[int, float]:
        """Rank place cards by visible price hints, keeping unknowns last."""
        text = MultiAgentAssistant._fold_context_text(
            " ".join(str(card.get(key) or "") for key in ("price", "features", "description"))
        )
        if re.search(r"(?:<|ate|até|menos de)\s*20\s*€?", text) or "< 20" in text:
            return (0, 20.0)
        if re.search(r"20\s*€?\s*a\s*50|20\s*€?\s*-\s*50|20\s+to\s+50", text):
            return (1, 50.0)
        if re.search(r">\s*50|mais de\s*50|over\s*50", text):
            return (2, 999.0)
        values = [
            float(value.replace(",", "."))
            for value in re.findall(r"\d+(?:[,.]\d+)?", text)
        ]
        if values:
            return (1, min(values))
        return (3, 999999.0)

    @staticmethod
    def _place_distance_rank(card: Dict[str, str]) -> tuple[int, float]:
        """Rank place cards by visible distance hints, keeping unknowns last."""
        text = MultiAgentAssistant._fold_context_text(str(card.get("distance") or ""))
        match = re.search(r"(\d+(?:[,.]\d+)?)\s*km\b", text)
        if not match:
            return (1, 999999.0)
        try:
            return (0, float(match.group(1).replace(",", ".")))
        except ValueError:
            return (1, 999999.0)

    @staticmethod
    def _extract_place_cards_from_answer(text: str) -> tuple[List[Dict[str, str]], str]:
        """Extract visible place/restaurant cards and the source line from a previous answer."""
        cards: List[Dict[str, str]] = []
        current: Dict[str, str] | None = None
        source_line = ""
        card_re = re.compile(r"^\s*[-*]\s+\*\*(?P<title>.+?)\*\*\s*$")
        field_re = re.compile(
            r"^\s+[-*]\s+(?:[\U0001F300-\U0001FAFF\u2300-\u27BF\uFE0F\u200D]+\s*)?"
            r"\*\*(?P<label>[^:*]+):\*\*\s*(?P<value>.+?)\s*$"
        )

        for raw_line in str(text or "").splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("📌 **Fonte") or stripped.startswith("📌 **Source"):
                source_line = stripped
                continue
            match = card_re.match(raw_line)
            if match and not raw_line.startswith(("  ", "\t")):
                if current and current.get("title"):
                    cards.append(current)
                title = re.sub(r"^[\U0001F300-\U0001FAFF\u2300-\u27BF\uFE0F\u200D]+\s*", "", match.group("title")).strip()
                current = {"title": title}
                continue
            if not current:
                continue
            field_match = field_re.match(raw_line)
            if not field_match:
                continue
            label = MultiAgentAssistant._fold_context_text(field_match.group("label"))
            value = field_match.group("value").strip()
            if label in {"caracteristicas", "features", "destaques", "highlights"}:
                current["features"] = value
            elif label in {"preco", "price"}:
                current["price"] = value
            elif label in {"website", "site", "site oficial"}:
                current["website"] = value
            elif label in {"morada", "address"}:
                current["address"] = value
            elif label in {"categoria", "category"}:
                current["category"] = value
            elif label in {"distancia", "distance"}:
                current["distance"] = value
            elif label in {"mais detalhes", "more details"}:
                current["details"] = stripped

        if current and current.get("title"):
            cards.append(current)
        place_cards = [
            card for card in cards
            if any(card.get(key) for key in ("address", "website", "features", "price", "details"))
        ]
        return place_cards, source_line

    @staticmethod
    def _format_filtered_place_cards(
        cards: List[Dict[str, str]],
        *,
        language: str,
        source_line: str,
        note: str = "",
    ) -> str:
        """Render compact filtered place cards from a previous answer."""
        is_pt = language == "pt"
        title = "### 📍 **Locais Recomendados**" if is_pt else "### 📍 **Recommended places**"
        direct = (
            "✅ **Resposta direta:** filtrei apenas os locais da lista anterior que cumprem melhor os critérios pedidos."
            if is_pt
            else "✅ **Direct answer:** I filtered only the previous places that best match your criteria."
        )
        lines = [title, "", direct, "", "---", ""]
        price_label = "Preço/indicação" if is_pt else "Price/hint"
        website_label = "Website"

        for card in cards:
            lines.append(f"- **📍 {card['title']}**")
            visible_price = card.get("price") or card.get("features")
            if visible_price:
                lines.append(f"    - 💰 **{price_label}:** {visible_price}")
            if card.get("website"):
                lines.append(f"    - 🌐 **{website_label}:** {card['website']}")
            if card.get("distance"):
                distance_label = "Distância" if is_pt else "Distance"
                lines.append(f"    - 📏 **{distance_label}:** {card['distance']}")
            if card.get("details"):
                lines.append(f"    - {card['details'].lstrip('- ').strip()}")
            lines.append("")

        if note:
            lines.extend([note, ""])
        if source_line:
            lines.append(source_line)
        return "\n".join(lines).strip()

    def _resolve_place_filter_follow_up(self, message: str, language: str) -> Dict[str, Any]:
        """Answer simple filters against previous place or restaurant cards."""
        normalized = self._fold_context_text(message)
        if not re.search(r"\b(?:destes|destas|desses|dessas|these|those|previous|above)\b", normalized):
            return {}
        wants_website = bool(re.search(r"\b(?:website|site|pagina|page)\b", normalized))
        wants_cheapest = bool(re.search(r"\b(?:barat\w*|cheap|cheapest|preco|price|econom)\b", normalized))
        wants_nearest = bool(re.search(r"\b(?:perto|proxim\w*|near|nearest|closer|closest|distance|distancia)\b", normalized))
        if not (wants_website or wants_cheapest or wants_nearest):
            return {}

        previous_assistant = ""
        for msg in reversed(self.state.get("messages") or []):
            if isinstance(msg, AIMessage) and msg.content:
                previous_assistant = str(msg.content)
                break
        if not previous_assistant:
            return {}
        if re.search(r"\b(?:eventos|events)\b", self._fold_context_text(previous_assistant)):
            return {}

        cards, source_line = self._extract_place_cards_from_answer(previous_assistant)
        if not cards:
            return {}

        selected = list(cards)
        omitted_notes: list[str] = []
        if wants_website:
            website_cards = [card for card in selected if card.get("website")]
            without_website = len(selected) - len(website_cards)
            selected = website_cards
            if without_website:
                omitted_notes.append(
                    f"sem website confirmado: {without_website}"
                    if language == "pt"
                    else f"without confirmed website: {without_website}"
                )
        if wants_cheapest:
            selected.sort(key=self._place_price_rank)
            priced_selected = [
                card for card in selected
                if self._place_price_rank(card)[0] < 3
            ]
            without_price = len(selected) - len(priced_selected)
            if without_price:
                omitted_notes.append(
                    f"sem preço confirmado: {without_price}"
                    if language == "pt"
                    else f"without confirmed price: {without_price}"
                )
            selected = [
                card for card in selected
                if self._place_price_rank(card)[0] < 3
            ] or selected
        if wants_nearest:
            selected.sort(key=self._place_distance_rank)

        selected = selected[:5]
        if not selected:
            response = (
                "✅ **Resposta direta:** na lista anterior não encontrei locais que cumpram esses filtros com dados confirmados."
                if language == "pt"
                else "✅ **Direct answer:** I did not find previous places matching those filters with confirmed data."
            )
            return {"direct_response": response}

        note = ""
        if omitted_notes:
            note = (
                "💡 Omiti cards da lista anterior com dados em falta (" + "; ".join(omitted_notes) + ")."
                if language == "pt"
                else "💡 I omitted previous cards with missing data (" + "; ".join(omitted_notes) + ")."
            )
        return {
            "direct_response": self._format_filtered_place_cards(
                selected,
                language=language,
                source_line=source_line,
                note=note,
            )
        }

    def _resolve_meal_replacement_follow_up(self, message: str, language: str) -> Dict[str, Any]:
        """Resolve meal replacement requests against the previous itinerary."""
        normalized = self._fold_context_text(message)
        if not re.search(
            r"\b(?:troca|trocar|substitui|substituir|muda|mudar|replace|swap|change)\b",
            normalized,
        ):
            return {}
        if not re.search(r"\b(?:jantar|dinner|restaurante|restaurant|meal|refeicao)\b", normalized):
            return {}
        wants_cheaper = bool(re.search(r"\b(?:mais barat\w*|barat\w*|cheaper|cheap|budget|econom\w*)\b", normalized))
        wants_same_zone = bool(re.search(r"\b(?:mesma zona|same area|same zone|perto|nearby|near)\b", normalized))
        if not (wants_cheaper or wants_same_zone):
            return {}

        meal_anchor = self._extract_meal_anchor_from_plan("jantar") or self._extract_meal_anchor_from_plan("restaurante")
        name = str(meal_anchor.get("name") or "").strip()
        if not name:
            return {}
        address = str(meal_anchor.get("address") or "").strip()
        basis = self._fold_context_text(
            " ".join(str(meal_anchor.get(key) or "") for key in ("name", "address", "role"))
        )
        timestamp = datetime.now().strftime("%H:%M")

        if wants_cheaper and re.search(r"(?:<\s*20|menos de 20|under 20|low cost|baixo custo|econom)", basis):
            if language == "pt":
                location_line = f"\n- 📍 **Zona usada:** {address}" if address else ""
                return {
                    "direct_response": (
                        "### 🍽️ **Substituição do jantar**\n\n"
                        f"✅ **Resposta direta:** não confirmo uma opção mais barata do que **{name}** com os dados disponíveis.\n\n"
                        "---\n\n"
                        f"- 🍽️ **Jantar atual:** {name}{location_line}\n"
                        "- 💶 **Motivo:** o restaurante anterior já aparece na faixa económica **< 20€**; a fonte não dá preço granular para provar uma alternativa mais barata.\n"
                        "- 🔁 **Alternativa segura:** posso trocar por outro restaurante na mesma zona, mas só devo apresentá-lo como **mesma faixa de preço**, não como mais barato.\n\n"
                        f"📌 **Fonte:** [*VisitLisboa Locais*](https://www.visitlisboa.com/pt-pt/locais) | **Atualizado:** {timestamp}"
                    )
                }
            location_line = f"\n- 📍 **Area used:** {address}" if address else ""
            return {
                "direct_response": (
                    "### 🍽️ **Dinner Replacement**\n\n"
                    f"✅ **Direct answer:** I cannot confirm a cheaper option than **{name}** from the available data.\n\n"
                    "---\n\n"
                    f"- 🍽️ **Current dinner:** {name}{location_line}\n"
                    "- 💶 **Reason:** the previous restaurant is already in the **under €20** price band; the source does not provide granular prices to prove a cheaper replacement.\n"
                    "- 🔁 **Safe alternative:** I can switch to another restaurant in the same area, but only as the **same price band**, not as cheaper.\n\n"
                    f"📌 **Source:** [*VisitLisboa Places*](https://www.visitlisboa.com/en/places) | **Updated:** {timestamp}"
                )
            }

        area_hint = self._compact_area_hint_from_address(address) or address or name
        rewritten = (
            f"Procura restaurantes de cozinha tradicional portuguesa económicos em {area_hint}, Lisboa. "
            f"Exclui {name}. "
            "Mostra só opções com preço e morada confirmados. Se não conseguires confirmar que ficam nessa zona, diz a limitação claramente."
            if language == "pt"
            else f"Find affordable traditional Portuguese restaurants in {area_hint}, Lisbon. "
            f"Exclude {name}. "
            "Show only options with confirmed price and address. If you cannot confirm they are in that area, state the limitation clearly."
        )
        return {
            "message": rewritten,
            "agents": ["researcher"],
            "routing_reasoning": "Meal replacement follow-up resolved against the previous itinerary meal anchor.",
        }

    @staticmethod
    def _compact_area_hint_from_address(address: str) -> str:
        """Extract a compact human area hint from a Lisbon address."""
        parts = [re.sub(r"\s+", " ", part).strip(" .:-") for part in re.split(r"[,;]", address or "")]
        parts = [part for part in parts if part]
        if not parts:
            return ""
        blocked_re = re.compile(
            r"\b(?:lisboa|portugal|loja|piso|andar|edificio|edif|cc|c\.c\.|centro comercial)\b",
            flags=re.IGNORECASE,
        )
        for part in reversed(parts):
            folded = MultiAgentAssistant._fold_context_text(part)
            if re.search(r"\b\d{4}\s*-?\s*\d{3}\b", folded) or folded in {"lisboa", "portugal"}:
                continue
            if any(char.isdigit() for char in part) and not re.search(r"\b(?:bairro|baixa|chiado|alfama|amoreiras|saldanha|arroios|campo|rato|belem|oriente)\b", folded):
                continue
            candidate = re.sub(blocked_re, " ", part)
            candidate = re.sub(r"\s+", " ", candidate).strip(" .:-")
            if 2 <= len(candidate) <= 50:
                return candidate
        first = re.sub(r"\s+", " ", parts[0]).strip(" .:-")
        return first if 2 <= len(first) <= 60 else ""

    def _resolve_contextual_follow_up(self, message: str, language: str) -> Dict[str, Any]:
        """Resolve compact follow-ups such as 'there' or plan revision requests."""
        # First, resolve anaphoric venue references like "the restaurant you
        # suggested" so the supervisor sees a fully grounded message and can
        # route to the correct worker(s) instead of issuing a fresh search.
        message = self._resolve_venue_anaphor(message)
        anchors = self._get_conversation_anchors()
        normalized = re.sub(r"\s+", " ", (message or "").lower()).strip()
        if not normalized:
            return {"message": message}

        malformed_route_pair = self._extract_route_pair_from_text(message)
        standalone_query = self._standalone_query_from_malformed_pending_route(message, malformed_route_pair)
        if standalone_query and self._fold_location_match_text(standalone_query) != self._fold_location_match_text(message):
            return {
                "message": standalone_query,
                "routing_reasoning": "Malformed route prefix ignored because the segment after the route connector is a standalone question.",
            }

        pending_location = self._resolve_pending_location_clarification_follow_up(message, language)
        if pending_location:
            return pending_location

        transport_alternative = self._resolve_transport_alternative_follow_up(message, language)
        if transport_alternative:
            return transport_alternative

        onward_transport = self._resolve_onward_transport_follow_up(message, language)
        if onward_transport:
            return onward_transport

        event_filter = self._resolve_event_filter_follow_up(message, language)
        if event_filter:
            return event_filter

        place_filter = self._resolve_place_filter_follow_up(message, language)
        if place_filter:
            return place_filter

        meal_replacement = self._resolve_meal_replacement_follow_up(message, language)
        if meal_replacement:
            return meal_replacement

        transport_destination_clarification = self._resolve_transport_destination_clarification_follow_up(message, language)
        if transport_destination_clarification:
            return transport_destination_clarification

        compact_route_pair = re.search(
            r"^\s*(?:e\s+)?(?:de|do|da|desde|from)\s+"
            r"(?P<origin>.+?)\s+(?:para|até|ate|to)\s+"
            r"(?P<destination>[^?!.;,]+)",
            message,
            flags=re.IGNORECASE,
        )
        if compact_route_pair:
            origin = re.sub(r"\s+", " ", compact_route_pair.group("origin")).strip(" .,:;?!")
            destination = re.sub(r"\s+", " ", compact_route_pair.group("destination")).strip(" .,:;?!")
            if len(origin) >= 2 and len(destination) >= 2:
                rewritten = (
                    f"Como vou de {origin} para {destination}?"
                    if language == "pt"
                    else f"How do I get from {origin} to {destination}?"
                )
                return {
                    "message": rewritten,
                    "agents": ["transport"],
                    "routing_reasoning": "Compact origin-destination follow-up resolved as a transport route.",
                }

        standalone_place_after_transport = self._resolve_standalone_place_after_transport_follow_up(message, language)
        if standalone_place_after_transport:
            return standalone_place_after_transport

        research_pagination = self._resolve_research_pagination_follow_up(message, language)
        if research_pagination:
            return research_pagination

        route_cue_for_recalled_venue = bool(re.search(
            r"\b(?:como\s+(?:(?:e|é)\s+que\s+)?(?:posso\s+)?(?:vou|chego|ir)|"
            r"ir\s+(?:de|do|da|desde)|chegar\s+(?:ao|a|à|ate|até)|"
            r"how\s+do\s+i\s+get|how\s+can\s+i\s+get|get\s+from|go\s+from|travel\s+from)\b",
            normalized,
        ))
        if (
            not route_cue_for_recalled_venue
            and not re.search(r"\b(?:adapta|ajusta|muda|troca|chover|chuva|rain|change|adjust|revise)\b", normalized)
            and re.search(r"\b(?:qual\s+foi|qual\s+era|que\s+restaurante|restaurante\s+q|restaurante\s+que)\b", normalized)
        ):
            meal_anchor = self._extract_meal_anchor_from_plan("restaurante")
            if meal_anchor.get("name"):
                time_text = f" às **{meal_anchor['time']}**" if meal_anchor.get("time") else ""
                if language == "pt":
                    return {"direct_response": f"✅ **Resposta direta:** o restaurante que indiquei foi **{meal_anchor['name']}**{time_text}."}
                return {"direct_response": f"✅ **Direct answer:** the restaurant I suggested was **{meal_anchor['name']}**{time_text}."}

        asks_departure_to_meal = bool(
            re.search(r"\b(?:a\s+que\s+horas|quando|when)\b.*\b(?:apanhar|sair|partir|leave|depart|take)\b", normalized)
            or re.search(r"\b(?:apanhar|sair|partir|leave|depart|take)\b.*\b(?:a\s+que\s+horas|quando|when)\b", normalized)
        )
        if asks_departure_to_meal:
            meal_anchor = self._extract_meal_anchor_from_plan("almoco")
            origin = self._extract_follow_up_origin(message)
            if meal_anchor.get("name") and origin:
                return {
                    "message": self._build_meal_transport_follow_up_message(
                        meal_anchor,
                        language,
                        origin=origin,
                        ask_departure_time=True,
                    ),
                    "agents": ["transport"],
                    "routing_reasoning": "Conversation meal anchor resolved into a point-to-point transport request.",
                }

        asks_route_to_meal = bool(
            re.search(
                r"\b(?:como\s+(?:(?:é|e)\s+que\s+)?(?:posso\s+)?(?:vou|chego|ir|fa[cç]o\s+para\s+(?:ir|chegar))|"
                r"ir\s+(?:de|do|da|desde)|chegar\s+(?:ao|à|a|ate|até)|"
                r"how\s+do\s+i\s+get|how\s+can\s+i\s+get|get\s+from|go\s+from|travel\s+from)\b",
                normalized,
            )
            and re.search(r"\b(?:almo[cç]o|almoco|lunch|jantar|dinner|restaurante|restaurant)\b", normalized)
        )
        if asks_route_to_meal:
            meal_key = "jantar" if re.search(r"\b(?:jantar|dinner)\b", normalized) else "almoco"
            if re.search(
                r"\b(?:(?:esse|essa|este|esta|aquele|aquela|that|this)\s+"
                r"(?:restaurante|restaurant)|(?:restaurante|restaurant).{0,50}?"
                r"(?:que|q|you|we|suggest|recommend|indica|suger))\b",
                normalized,
            ):
                meal_key = "restaurante"
            meal_anchors = self._extract_meal_anchors_from_plan(meal_key)
            if meal_key == "restaurante" and len(meal_anchors) > 1:
                return {"clarification": self._meal_anchor_clarification(meal_anchors, language)}
            meal_anchor = meal_anchors[0] if meal_anchors else {}
            origin = self._extract_follow_up_origin(message)
            if meal_anchor.get("name") and origin:
                mention_time = bool(re.search(r"\b(?:hora|horas|time|chegar|arrive)\b", normalized))
                return {
                    "message": self._build_meal_transport_follow_up_message(
                        meal_anchor,
                        language,
                        origin=origin,
                        ask_departure_time=mention_time,
                    ),
                    "agents": ["transport"],
                    "routing_reasoning": "Conversation meal anchor resolved into a point-to-point transport request.",
                }

        if re.search(r"\b(?:disseste|disseste-me|referiste|mencionaste).+\bhora\s+do\s+almo", normalized):
            meal_anchor = self._extract_meal_anchor_from_plan("almoco")
            if meal_anchor.get("name"):
                time_text = f" às **{meal_anchor['time']}**" if meal_anchor.get("time") else ""
                if language == "pt":
                    return {"direct_response": f"✅ **Resposta direta:** sim — referi o almoço no **{meal_anchor['name']}**{time_text}."}
                return {"direct_response": f"✅ **Direct answer:** yes — I referred to lunch at **{meal_anchor['name']}**{time_text}."}

        destination = str(anchors.get("current_selected_destination") or "").strip()
        # Treat "there" as an anaphoric destination only when it is not the
        # existential construction used in questions such as "are there any
        # disruptions?". Those prompts can also contain an explicit
        # origin-destination pair, for example "from Cais do Sodré to Cascais".
        explicit_route_pair = bool(re.search(
            r"\b(?:from\s+.+?\s+to\s+.+|de\s+.+?\s+(?:para|a|ao|à|at[eé])\s+.+)",
            normalized,
        ))
        explicit_destination_in_current_turn = bool(re.search(
            r"\b(?:get|go|travel|head|ir|chegar|viajar)\s+(?:to|para|a|ao|à)\s+"
            r"(?!there\b|la\b|lá\b|ali\b|ai\b|aí\b)[a-z0-9à-ÿ' -]{2,}",
            normalized,
        ))
        full_planning_request = bool(
            re.search(
                r"\b(?:plan|itinerary|route|roteiro|plano|planeia|planejar|"
                r"morning|afternoon|evening|manha|manhã|tarde|noite|around|em)\b",
                normalized,
            )
            and len(normalized.split()) >= 7
        )
        embedded_there_instruction = bool(
            re.search(
                r"\b(?:include|including|inclui|incluir).{0,80}"
                r"(?:how\s+(?:i|we|to)\s+(?:get|go|travel)\s+there|"
                r"como\s+(?:vou|vamos|chego|chegar|ir)\s+(?:la|lá|ali|a[ií]))\b",
                normalized,
            )
        )
        self_contained_nearest_request = bool(
            re.search(
                r"\b(?:qual|quais|onde|which|what|where).{0,120}"
                r"\b(?:mais\s+pr[oó]xim[ao]s?|nearest|closest)\b",
                normalized,
            )
            or re.search(
                r"\b(?:farmacia|farmácia|biblioteca|hospital|escola|parque|mercado|"
                r"servico|serviço|service|library|pharmacy|school|market).{0,120}"
                r"\b(?:mais\s+pr[oó]xim[ao]s?|nearest|closest)\b",
                normalized,
            )
            or re.search(
                r"\b(?:farmacia|farmácia|biblioteca|hospital|escola|parque|mercado|"
                r"servico|serviço|service|library|pharmacy|school|market|"
                r"public toilet|restroom|wc)\b.{0,140}"
                r"\b(?:pr[oó]xim[ao]s?|nearby|near|perto|junto)\b",
                normalized,
            )
        )
        existential_there = bool(
            re.search(r"\bthere\s+(?:are|is|were|was|any|no)\b", normalized)
            or re.search(r"\b(?:are|is|were|was)\s+there\b", normalized)
        )
        uses_there = (
            bool(re.search(r"\b(?:there|lá|la|ali|aí|ai)\b", normalized))
            and not existential_there
            and not explicit_route_pair
            and not explicit_destination_in_current_turn
            and not full_planning_request
            and not embedded_there_instruction
            and not self_contained_nearest_request
        )
        asks_route = bool(re.search(r"\b(?:how do i get|como chego|como vou|ir de|go from|get from|from|desde|a partir de)\b", normalized))
        if uses_there and asks_route:
            if not destination:
                clarification = (
                    "Which destination from the previous plan do you want directions to?"
                    if language != "pt"
                    else "Para que destino do plano anterior queres as indicações?"
                )
                return {"clarification": clarification}
            return {
                "message": (
                    f"{message}\n\nResolved conversation anchor: interpret 'there' as {destination}. "
                    f"Destination: {destination}."
                ),
                "agents": ["transport"],
                "routing_reasoning": "Conversation destination anchor resolved into a point-to-point transport request.",
            }

        previous_plan_context = (
            str(anchors.get("last_plan_summary") or "").strip()
            or str(anchors.get("last_plan_text") or "").strip()
        )
        revises_previous_plan = bool(
            re.search(
                r"\b(?:make it|change it|adjust it|cheaper|rain|chuva|chover|"
                r"mais\s+interiores?|interior|indoor|mais barato|barato|"
                r"suitable|adequado|adapta|ajusta|muda|troca)\b",
                normalized,
            )
            and previous_plan_context
        )
        if revises_previous_plan:
            preferences = ", ".join(anchors.get("user_preferences") or [])
            exclusions = ", ".join(anchors.get("excluded_areas") or [])
            destinations = ", ".join(str(item) for item in (anchors.get("last_itinerary_destinations") or [])[:8])
            folded_area_context = self._fold_context_text(
                " ".join(
                    [
                        destinations,
                        str(anchors.get("last_plan_summary") or ""),
                        str(anchors.get("last_plan_text") or ""),
                    ]
                )
            )
            previous_zone = str(anchors.get("last_plan_area_zone") or "").strip()
            if not previous_zone and "belem" in folded_area_context:
                previous_zone = "belem"
            previous_area = self._planner_zone_label(previous_zone, language=language) if previous_zone else ""
            rain_or_indoor_revision = bool(
                re.search(r"\b(?:rain|chuva|chover|interiores?|indoor|covered|cobert[oa]s?)\b", normalized)
            )
            preserve_previous_meal = bool(
                re.search(
                    r"\b(?:mant[eé]m|mantendo|manter|preserva|preservar|keep|keeping|same)\b"
                    r".{0,80}\b(?:restaurante|restaurant|almo[cç]o|lunch|jantar|dinner|refei[cç][aã]o|meal)\b",
                    normalized,
                    flags=re.IGNORECASE,
                )
            )
            previous_meal_anchors = self._extract_meal_anchors_from_plan("restaurante") if preserve_previous_meal else []
            previous_meal_context = ""
            if previous_meal_anchors:
                meal_parts = []
                for anchor in previous_meal_anchors[:3]:
                    anchor_name = str(anchor.get("name") or "").strip()
                    anchor_address = str(anchor.get("address") or "").strip()
                    if anchor_name:
                        meal_parts.append(anchor_name + (f" ({anchor_address})" if anchor_address else ""))
                previous_meal_context = "; ".join(meal_parts)
            previous_meal_unconfirmed = bool(
                preserve_previous_meal
                and re.search(
                    r"\b(?:nenhum restaurante especifico ficou confirmado|no specific restaurant was confirmed)\b",
                    self._fold_context_text(str(anchors.get("last_plan_text") or "")),
                    flags=re.IGNORECASE,
                )
            )
            if previous_meal_unconfirmed and rain_or_indoor_revision:
                return {
                    "direct_response": self._build_unconfirmed_meal_revision_response(
                        language=language,
                        previous_area=previous_area,
                        destinations=list(anchors.get("last_itinerary_destinations") or []),
                    )
                }
            if language == "pt":
                locality_constraint = (
                    f"Restrição de zona: mantém apenas pontos na zona **{previous_area}**; "
                    "ignora resultados cuja morada, nome ou descrição indiquem outra zona ou concelho. "
                    "Se não houver alternativas interiores confirmadas nessa zona, diz claramente que não há dados suficientes "
                    "em vez de trocar para outra zona."
                    if previous_area
                    else "Restrição de zona: preserva a zona inferida pelas paragens anteriores; não mudes para outra zona salvo pedido explícito."
                )
                meal_instruction = (
                    "Mantém a refeição anterior; se ela estava apenas a confirmar e sem restaurante específico, "
                    "diz isso claramente e não a substituas por outro restaurante novo. "
                    if preserve_previous_meal
                    else ""
                )
                if rain_or_indoor_revision:
                    revision_goal = (
                        "Objetivo: adaptar o roteiro anterior para chuva, substituindo paragens exteriores "
                        "por opções mais interiores ou cobertas, como museus, monumentos visitáveis por dentro, "
                        "centros culturais e cafés/pastelarias cobertas próximos das paragens anteriores. "
                        "A resposta direta deve dizer explicitamente que o roteiro foi adaptado para chuva "
                        "e destacar a principal alteração feita. "
                        "Mantém a mesma zona do roteiro anterior; não uses alternativas noutros bairros/concelhos "
                        "como se fossem substituições locais. Se não houver dados suficientes na mesma zona, diz isso claramente. "
                        f"{meal_instruction}{locality_constraint}"
                    )
                else:
                    revision_goal = (
                        "Objetivo: rever o roteiro anterior mantendo apenas alterações pedidas pelo utilizador. "
                        f"{meal_instruction}"
                    ).strip()
                meal_context_line = (
                    f"Refeição anterior a preservar: {previous_meal_context or 'não extraída explicitamente'}.\n"
                    f"Estado da refeição anterior: {'sem restaurante específico confirmado; preservar como limitação, não substituir' if previous_meal_unconfirmed else 'usar apenas se estiver explicitamente confirmada no contexto'}.\n"
                    if preserve_previous_meal
                    else ""
                )
                context_message = (
                    f"{message}\n\n"
                    f"Contexto do roteiro anterior: {str(anchors.get('last_plan_summary') or '')[:900]}.\n"
                    f"Paragens anteriores: {destinations or 'não extraídas explicitamente'}.\n"
                    f"Zona anterior: {previous_area or 'inferir pelas paragens anteriores'}.\n"
                    f"{meal_context_line}"
                    f"{revision_goal}\n"
                    "Não trates títulos de secção ou texto de resposta direta como paragens do roteiro.\n\n"
                    f"Preferências guardadas: {preferences or 'nenhuma preferência explícita'}.\n"
                    f"Exclusões guardadas: {exclusions or 'nenhuma exclusão explícita'}."
                )
            else:
                locality_constraint = (
                    f"Area constraint: keep only places in **{previous_area}**; ignore results whose address, name, "
                    "or description indicate another area or municipality. If there are not enough confirmed indoor "
                    "alternatives in that area, state that limitation instead of switching area."
                    if previous_area
                    else "Area constraint: preserve the area inferred from the previous stops; do not switch area unless explicitly requested."
                )
                meal_instruction = (
                    "Keep the previous meal stop; if it was only an unconfirmed meal placeholder, say that clearly "
                    "and do not replace it with a new restaurant. "
                    if preserve_previous_meal
                    else ""
                )
                if rain_or_indoor_revision:
                    revision_goal = (
                        "Goal: adapt the previous itinerary for rain by replacing outdoor stops with more indoor "
                        "or covered options, such as museums, indoor monuments, cultural centres, and covered cafés "
                        "near the previous stops. The direct answer must explicitly say that the itinerary was "
                        "adapted for rain and highlight the main change. "
                        "Keep the same area as the previous itinerary; do not use alternatives "
                        "in other neighbourhoods/municipalities as local replacements. If there is not enough same-area "
                        f"evidence, state that clearly. {meal_instruction}{locality_constraint}"
                    )
                else:
                    revision_goal = (
                        "Goal: revise the previous itinerary while preserving only the changes requested by the user. "
                        f"{meal_instruction}"
                    ).strip()
                meal_context_line = (
                    f"Previous meal stop to preserve: {previous_meal_context or 'not explicitly extracted'}.\n"
                    f"Previous meal status: {'no specific restaurant confirmed; preserve as a limitation, do not replace' if previous_meal_unconfirmed else 'use only if explicitly confirmed in context'}.\n"
                    if preserve_previous_meal
                    else ""
                )
                context_message = (
                    f"{message}\n\n"
                    f"Previous itinerary context: {str(anchors.get('last_plan_summary') or '')[:900]}.\n"
                    f"Previous stops: {destinations or 'not explicitly extracted'}.\n"
                    f"Previous area: {previous_area or 'infer from the previous stops'}.\n"
                    f"{meal_context_line}"
                    f"{revision_goal}\n"
                    "Do not treat section headings or direct-answer text as itinerary stops.\n\n"
                    f"Stored preferences: {preferences or 'none explicitly stored'}.\n"
                    f"Stored exclusions: {exclusions or 'none explicitly stored'}."
                )
            return {
                "message": context_message,
                "agents": ["weather", "researcher", "planner"],
                "routing_reasoning": "Planning follow-up asks to revise the previous itinerary using weather/context.",
            }

        return {"message": message}

    def _update_conversation_anchors(
        self,
        message: str,
        final_output: str,
        effective_agents: List[str],
    ) -> None:
        """Update structured conversation anchors after publishing a final answer."""
        anchors = self._get_conversation_anchors()
        effective_agent_set = set(effective_agents or [])
        anchors["last_response_agents"] = list(effective_agents or [])
        anchors["excluded_areas"] = self._merge_anchor_values(
            anchors.get("excluded_areas"),
            self._extract_excluded_areas(message),
        )
        anchors["user_preferences"] = self._merge_anchor_values(
            anchors.get("user_preferences"),
            self._extract_user_preferences(message),
        )

        if "researcher" in effective_agent_set:
            researcher = self.agents.get("researcher")
            search_context = getattr(researcher, "_last_search_context", None) if researcher else None
            if not search_context and researcher and hasattr(researcher, "get_tool_calls_log"):
                for call in reversed(researcher.get_tool_calls_log()):
                    if not isinstance(call, dict):
                        continue
                    tool_name = str(call.get("tool_name") or "").strip()
                    if tool_name not in {"search_cultural_events", "search_places_attractions"}:
                        continue
                    args = call.get("args") if isinstance(call.get("args"), dict) else {}
                    domain = "events" if tool_name == "search_cultural_events" else "places"
                    page_size = int(args.get("max_results") or 5)
                    offset = int(args.get("offset") or 0)
                    shown_count = max(1, min(page_size, len(re.findall(r"(?m)^\s*[-*]\s+\*\*", final_output or "")) or page_size))
                    search_context = {
                        "domain": domain,
                        "tool_name": tool_name,
                        "base_args": {key: value for key, value in args.items() if key not in {"max_results", "offset"}},
                        "page_size": page_size,
                        "offset": offset,
                        "next_offset": offset + shown_count,
                        "language": str(args.get("language") or ""),
                        "source_query": message,
                    }
                    setattr(researcher, "_last_search_context", search_context)
                    break
            if isinstance(search_context, dict) and search_context:
                anchors["last_research_context"] = {
                    "domain": search_context.get("domain"),
                    "tool_name": search_context.get("tool_name"),
                    "base_args": dict(search_context.get("base_args") or {}),
                    "page_size": search_context.get("page_size"),
                    "offset": search_context.get("offset"),
                    "next_offset": search_context.get("next_offset"),
                    "language": search_context.get("language"),
                    "source_query": str(search_context.get("source_query") or "")[:300],
                }

        if "transport" in effective_agent_set:
            route_pair = self._extract_route_pair_from_text(message) or self._extract_route_pair_from_text(final_output)
            if route_pair:
                anchors["last_transport_route"] = route_pair
            self._store_pending_location_clarification(message, final_output, effective_agent_set)
        else:
            anchors["pending_location_clarification"] = {}

        if "planner" not in effective_agent_set:
            return

        destinations = self._extract_destination_candidates_from_plan(final_output)
        excluded = {str(area).lower() for area in anchors.get("excluded_areas") or []}
        filtered_destinations = [
            destination
            for destination in destinations
            if destination.lower() not in excluded
        ]
        if filtered_destinations:
            anchors["last_itinerary_destinations"] = filtered_destinations
            anchors["current_selected_destination"] = filtered_destinations[0]
        summary_parts: list[str] = []
        if filtered_destinations:
            summary_parts.append("Destinations: " + ", ".join(filtered_destinations[:5]))
        mentioned_zones = self._planner_zones_in_text(f"{message}\n{final_output}")
        if len(mentioned_zones) == 1:
            plan_zone = next(iter(mentioned_zones))
            anchors["last_plan_area_zone"] = plan_zone
            summary_parts.append("Area: " + self._planner_zone_label(plan_zone, language="pt"))
        elif "last_plan_area_zone" in anchors:
            anchors.pop("last_plan_area_zone", None)
        if anchors.get("user_preferences"):
            summary_parts.append("Preferences: " + ", ".join(str(item) for item in anchors.get("user_preferences") or []))
        if anchors.get("excluded_areas"):
            summary_parts.append("Excluded areas: " + ", ".join(str(item) for item in anchors.get("excluded_areas") or []))
        anchors["last_plan_summary"] = ("; ".join(summary_parts) or "Previous itinerary available")[:700]
        anchors["last_plan_text"] = (final_output or "")[:4000]

    def _run_lightweight_weather_fact_check(
        self,
        user_query: str,
        weather_output: str,
        language: str,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run deterministic fact-checking for simple weather-only requests.

        This preserves the low-latency weather fast path while still checking
        for obvious factual or formatting issues. If critical issues are found,
        the caller can escalate to the full QA validation pass.

        Args:
            user_query: Original user query.
            weather_output: Weather worker output.
            language: Output language code.
            verbose: Whether to emit terminal diagnostics.

        Returns:
            Dict[str, Any]: Deterministic fact-check result plus escalation hints.
        """
        verify_facts = getattr(self.qa_agent, "_verify_facts", None)
        if not callable(verify_facts) or not weather_output:
            return {
                "performed": False,
                "requires_full_qa": False,
                "fact_check": {},
                "disclaimers": [],
            }

        try:
            fact_check = verify_facts(
                weather_output,
                user_query,
                self.state.get("user_context"),
            )
        except Exception as exc:
            if verbose:
                print(f"   [QA] Lightweight weather fact-check unavailable: {exc}")
            return {
                "performed": False,
                "requires_full_qa": False,
                "fact_check": {},
                "disclaimers": [],
            }

        if not isinstance(fact_check, dict):
            return {
                "performed": False,
                "requires_full_qa": False,
                "fact_check": {},
                "disclaimers": [],
            }

        sanitized_disclaimers = self._sanitize_qa_disclaimers(
            fact_check.get("disclaimers", []),
            language,
        )
        critical_issues = self._dedupe_preserve_order(
            list(fact_check.get("critical_issues", []))
        )

        if verbose:
            print("\n   [QA] Fast deterministic weather fact-check completed")
            if sanitized_disclaimers:
                for disclaimer in sanitized_disclaimers:
                    print(f"   [QA FACT-CHECK] {disclaimer}")
            if critical_issues:
                for issue in critical_issues:
                    print(f"   [QA FACT-CHECK] Critical: {issue}")

        return {
            "performed": True,
            "requires_full_qa": bool(critical_issues),
            "fact_check": fact_check,
            "disclaimers": sanitized_disclaimers,
        }

    @staticmethod
    def _format_usd_cost_label(cost_payload: Optional[Dict[str, Any]]) -> str:
        """Format the total USD cost using a compact terminal-friendly label.

        Args:
            cost_payload: Cost payload with ``total_cost_usd``.

        Returns:
            str: Compact cost label such as ``(0.003$)``.
        """
        if not isinstance(cost_payload, dict):
            return "(0.0000$)"

        total_cost = float(cost_payload.get("total_cost_usd", 0.0) or 0.0)
        if total_cost <= 0:
            return "(0.0000$)"
        if total_cost < 0.0001:
            return f"({total_cost:.6f}$)"
        if total_cost < 0.01:
            return f"({total_cost:.4f}$)"
        if total_cost < 1:
            return f"({total_cost:.3f}$)"
        return f"({total_cost:.2f}$)"

    @staticmethod
    def _truncate_summary_text(text: object, max_length: int = 180) -> str:
        """Trim terminal summary strings to a readable single-line preview."""
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(normalized) <= max_length:
            return normalized
        return normalized[: max_length - 3].rstrip() + "..."

    def _collect_execution_summary(
        self,
        *,
        user_request: str,
        routing_reasoning: str,
        agents_to_call: List[str],
        agent_outputs: Dict[str, Any],
        direct_response_used: bool,
        workers: List[str],
        run_workers_in_parallel: bool,
        qa_result: Optional[Dict[str, Any]],
        retry_agents_used: List[str],
        final_repair_ran: bool,
        simple_weather_fact_check: Optional[Dict[str, Any]],
        elapsed_time: float,
    ) -> Dict[str, Any]:
        """Collect runtime metrics for the terminal execution summary.

        Args:
            agents_to_call: Agents selected by the supervisor.
            agent_outputs: Worker outputs gathered for the request.
            direct_response_used: Whether the supervisor answered directly.
            workers: Worker agents executed before planner synthesis.
            run_workers_in_parallel: Whether workers ran in parallel.
            qa_result: QA validation result, if any.
            retry_agents_used: Agents retried after QA feedback.
            final_repair_ran: Whether the final QA repair pass ran.
            simple_weather_fact_check: Fast weather fact-check metadata.
            elapsed_time: End-to-end request duration in seconds.

        Returns:
            Dict[str, Any]: Structured execution summary payload.
        """
        usage_snapshot = {
            agent_name: self._normalize_usage_summary(summary)
            for agent_name, summary in self.get_llm_usage_snapshot().items()
        }
        aggregate_usage = build_usage_payload(
            self.get_llm_usage_summary(),
            by_agent=usage_snapshot,
        )

        pricing_catalog = load_pricing_catalog(str(Config.LLM_PRICING_CATALOG_PATH))
        pricing_metadata = get_pricing_metadata(pricing_catalog)
        total_cost = build_cost_payload(
            aggregate_usage,
            pricing_catalog,
            model_id=aggregate_usage.get("model_id"),
        )
        langsmith_request_status = get_langsmith_request_tracking_status()

        # Optional opt-in sync flush: when LANGSMITH_SYNC_FLUSH=true, wait for
        # local tracer queue to drain and probe the active run via
        # client.read_run. On success we upgrade the persistence_state label
        # from "unconfirmed" to "confirmed" so the execution summary reflects
        # the verified ingestion.
        try:
            from agent.utils.langsmith_tracing import (
                flush_langsmith_and_confirm as _flush_langsmith_and_confirm,
                is_langsmith_sync_flush_enabled as _is_langsmith_sync_flush_enabled,
            )
            if _is_langsmith_sync_flush_enabled() and langsmith_request_status.get("current_run_attached"):
                flush_info = _flush_langsmith_and_confirm(
                    run_id=langsmith_request_status.get("run_id"),
                )
                if flush_info.get("confirmed"):
                    langsmith_request_status = {
                        **langsmith_request_status,
                        "persistence_state": "confirmed",
                        "note": flush_info.get("message") or langsmith_request_status.get("note"),
                    }
                elif flush_info.get("message"):
                    langsmith_request_status = {
                        **langsmith_request_status,
                        "note": flush_info["message"],
                    }
        except Exception:  # pragma: no cover - defensive, flush is opt-in
            pass

        agent_objects = {
            "supervisor": self.supervisor,
            **self.agents,
            "qa": self.qa_agent,
        }
        effective_agents = self._dedupe_preserve_order(
            [
                "supervisor",
                *workers,
                *agents_to_call,
                *[name for name in agent_outputs.keys() if not str(name).startswith("_")],
                *retry_agents_used,
                "qa" if qa_result or final_repair_ran or (simple_weather_fact_check or {}).get("performed") else "",
            ]
        )

        relevant_agents: List[str] = []
        agent_tool_logs: Dict[str, List[Dict[str, Any]]] = {}
        agent_costs: Dict[str, Dict[str, Any]] = {}
        models_used: List[str] = []

        for agent_name, agent_obj in agent_objects.items():
            usage_summary = usage_snapshot.get(agent_name, self._normalize_usage_summary({}))
            tool_log = self._normalize_tool_calls_log(
                getattr(agent_obj, "get_tool_calls_log", lambda: [])()
            ) if agent_name in self.agents else []

            if (
                agent_name in effective_agents
                or usage_summary["call_count"] > 0
                or tool_log
            ):
                relevant_agents.append(agent_name)
                if tool_log:
                    agent_tool_logs[agent_name] = tool_log
                agent_costs[agent_name] = build_cost_payload(
                    usage_summary,
                    pricing_catalog,
                    model_id=usage_summary.get("model_id"),
                )
                if usage_summary["call_count"] > 0 and usage_summary["model_id"] != "Unknown":
                    if usage_summary["model_id"] not in models_used:
                        models_used.append(usage_summary["model_id"])

        if direct_response_used:
            execution_type = "direct"
        elif "planner" in agents_to_call:
            execution_type = "planner"
        elif len(workers) > 1:
            execution_type = "hybrid"
        elif len(workers) == 1:
            execution_type = "single-worker"
        else:
            execution_type = "fallback"

        worker_mode = "parallel" if workers and run_workers_in_parallel else "sequential" if workers else "n/a"

        qa_steps: List[str] = []
        if direct_response_used:
            qa_steps.append("not-applicable")
        elif simple_weather_fact_check and simple_weather_fact_check.get("performed"):
            qa_steps.append("fast-weather-fact-check")
        elif qa_result:
            qa_steps.append("validated")
        else:
            qa_steps.append("not-run")
        if retry_agents_used:
            qa_steps.append("retry")
        if final_repair_ran:
            qa_steps.append("final-repair")

        return {
            "elapsed_time": elapsed_time,
            "user_request": user_request,
            "routing_reasoning": routing_reasoning,
            "selected_agents": list(agents_to_call),
            "execution_type": execution_type,
            "worker_mode": worker_mode,
            "qa_path": " -> ".join(qa_steps),
            "langsmith": langsmith_request_status,
            "usage": aggregate_usage,
            "pricing_metadata": pricing_metadata,
            "total_cost": total_cost,
            "models_used": models_used,
            "relevant_agents": relevant_agents,
            "agent_usage": usage_snapshot,
            "agent_costs": agent_costs,
            "agent_tool_logs": agent_tool_logs,
            "total_tool_invocations": sum(len(tool_log) for tool_log in agent_tool_logs.values()),
            "retry_agents_used": retry_agents_used,
        }

    def _print_execution_summary(self, summary: Dict[str, Any]) -> None:
        """Print a compact analytical execution summary to the terminal.

        Args:
            summary: Structured summary payload returned by
                ``_collect_execution_summary``.
        """
        usage = summary.get("usage", {}) if isinstance(summary, dict) else {}
        tokens = usage.get("tokens", {}) if isinstance(usage, dict) else {}
        langsmith = summary.get("langsmith", {}) if isinstance(summary, dict) else {}
        total_cost = summary.get("total_cost", {}) if isinstance(summary, dict) else {}
        show_detailed_terminal_logs = bool(
            getattr(Config, "SHOW_DETAILED_EXECUTION_LOGS", False)
        )

        import builtins
        import sys

        def _safe_print(value: object = "") -> None:
            """Print terminal diagnostics without breaking chat responses on legacy consoles."""
            try:
                builtins.print(value)
            except UnicodeEncodeError:
                encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
                text = str(value).encode(encoding, errors="replace").decode(
                    encoding,
                    errors="replace",
                )
                builtins.print(text)

        print = _safe_print

        print("\n" + "=" * 80)
        print("📊 EXECUTION SUMMARY")
        print("=" * 80)
        print(f"⏱️  Time taken: {float(summary.get('elapsed_time', 0.0) or 0.0):.2f}s")
        request_text = self._truncate_summary_text(summary.get("user_request", ""), 220)
        if request_text:
            print(f"🗣️  User request: {request_text}")

        selected_agents = summary.get("selected_agents", []) if isinstance(summary, dict) else []
        selected_agents_label = ", ".join(selected_agents) if selected_agents else "direct response"
        print(f"🎯  Routed agents: {selected_agents_label}")

        routing_reasoning = self._truncate_summary_text(summary.get("routing_reasoning", ""), 220)
        if routing_reasoning:
            print(f"🧠  Routing reason: {routing_reasoning}")

        print(
            f"🧭  Execution: {summary.get('execution_type', 'unknown')} | "
            f"Workers: {summary.get('worker_mode', 'n/a')} | "
            f"QA: {summary.get('qa_path', 'n/a')}"
        )
        print(
            f"📈  LLM Calls: {int(usage.get('call_count', 0) or 0)} | "
            f"🪙 Tokens: {int(tokens.get('total_tokens', 0) or 0)} "
            f"(Input: {int(tokens.get('input_tokens', 0) or 0)} | Output: {int(tokens.get('output_tokens', 0) or 0)})"
        )

        models_used = summary.get("models_used", []) if isinstance(summary, dict) else []
        print(f"🧠  Models: {', '.join(models_used) if models_used else 'No LLM call'}")

        print(
            f"💵  Total Cost: {self._format_usd_cost_label(total_cost)}"
        )

        if show_detailed_terminal_logs and isinstance(langsmith, dict) and langsmith:
            project_name = str(langsmith.get("project_name") or "").strip()
            run_id = str(langsmith.get("run_id") or "").strip()
            run_context_label = "attached" if langsmith.get("current_run_attached") else "not-attached"
            persistence_state = str(langsmith.get("persistence_state") or "n/a").replace("_", " ")
            langsmith_parts = [
                f"🛰️  LangSmith: {langsmith.get('status_label', 'disabled')}",
                f"Run context: {run_context_label}",
                f"Persistence: {persistence_state}",
            ]
            if project_name:
                langsmith_parts.append(f"Project: {project_name}")
            if run_id:
                langsmith_parts.append(f"Run ID: {self._truncate_summary_text(run_id, 18)}")
            print(" | ".join(langsmith_parts))

            langsmith_note = str(langsmith.get("note") or "").strip()
            if langsmith_note:
                print(f"      {langsmith_note}")

        missing_pricing = total_cost.get("missing_pricing_models", []) if isinstance(total_cost, dict) else []
        if missing_pricing:
            print(f"⚠️  Missing pricing: {', '.join(missing_pricing)}")

        relevant_agents = summary.get("relevant_agents", []) if isinstance(summary, dict) else []
        if show_detailed_terminal_logs and relevant_agents:
            print("🧩  Agent breakdown:")
            agent_usage = summary.get("agent_usage", {})
            agent_costs = summary.get("agent_costs", {})
            agent_tool_logs = summary.get("agent_tool_logs", {})

            for agent_name in relevant_agents:
                usage_summary = agent_usage.get(agent_name, self._normalize_usage_summary({}))
                tokens_summary = usage_summary.get("tokens", {})
                tool_count = len(agent_tool_logs.get(agent_name, []))
                model_label = usage_summary.get("model_id", "Unknown")
                if usage_summary.get("call_count", 0) == 0:
                    if tool_count:
                        model_label = "tool-only"
                    elif agent_name == "supervisor":
                        model_label = "heuristic-only"
                    else:
                        model_label = "no-llm"

                display_name = "QA" if agent_name == "qa" else agent_name.title()
                print(
                    f"    │ {display_name} [{model_label}] | "
                    f"LLM {usage_summary.get('call_count', 0)} | "
                    f"Tools {tool_count} | "
                    f"Tok {int(tokens_summary.get('input_tokens', 0) or 0)}/"
                    f"{int(tokens_summary.get('output_tokens', 0) or 0)}/"
                    f"{int(tokens_summary.get('total_tokens', 0) or 0)} | "
                    f"Cost {self._format_usd_cost_label(agent_costs.get(agent_name))}"
                )

        agent_tool_logs = summary.get("agent_tool_logs", {}) if isinstance(summary, dict) else {}
        if show_detailed_terminal_logs and agent_tool_logs:
            print("🔧  Tool calls:")
            for agent_name, tool_log in agent_tool_logs.items():
                display_name = "QA" if agent_name == "qa" else agent_name.title()
                print(f"    │ {display_name} [{len(tool_log)} call(s)]")
                for item in tool_log:
                    try:
                        args_str = json.dumps(item.get("args", {}), ensure_ascii=False)
                    except Exception:
                        args_str = str(item.get("args", {}))
                    if len(args_str) > 140:
                        args_str = args_str[:137] + "..."
                    print(f"    ├──> {item.get('tool_name', 'unknown')}({args_str})")
            print(f"    ╰── Total Tool Invocations: {summary.get('total_tool_invocations', 0)}")
        elif show_detailed_terminal_logs:
            print("🔧  Tool calls: 0")

    def _finalize_chat_response(
        self,
        *,
        response: str,
        message: str,
        language: str,
        agents_to_call: List[str],
        routing_reasoning: str,
        agent_outputs: Dict[str, Any],
        direct_response_used: bool,
        start_time: float,
        workers: List[str],
        run_workers_in_parallel: bool,
        qa_result: Optional[Dict[str, Any]],
        retry_agents_used: List[str],
        final_repair_ran: bool,
        simple_weather_fact_check: Optional[Dict[str, Any]],
    ) -> str:
        """Apply final formatting, persist history, and print analytics.

        Args:
            response: Raw drafted response.
            message: Original user message.
            language: Output language code.
            agents_to_call: Agents selected by the supervisor.
            routing_reasoning: Supervisor routing explanation for this turn.
            agent_outputs: Worker outputs gathered during the run.
            direct_response_used: Whether the supervisor answered directly.
            start_time: Request start timestamp.
            workers: Workers executed before planner synthesis.
            run_workers_in_parallel: Whether workers ran in parallel.
            qa_result: QA result payload, if any.
            retry_agents_used: Agents retried after QA feedback.
            final_repair_ran: Whether the final QA repair pass ran.
            simple_weather_fact_check: Fast weather fact-check metadata.

        Returns:
            str: Final user-facing response.
        """
        from agent.utils.response_formatter import (
            canonicalize_visitlisboa_source_line,
            ensure_transport_notes_heading,
            format_researcher_card,
            infer_researcher_source_kind,
            reconcile_researcher_event_response,
            researcher_place_response_missing_requested_fields,
            normalize_transport_notes_block,
            strip_redundant_transport_status_notes,
            strip_technical_output_artifacts,
        )

        effective_agents = self._dedupe_preserve_order(
            [
                *agents_to_call,
                *[name for name in agent_outputs.keys() if not str(name).startswith("_")],
            ]
        )

        sanitized_response = clean_response(response)
        if "transport" in effective_agents:
            sanitized_response = strip_technical_output_artifacts(sanitized_response)

        formatted = format_response(sanitized_response)
        # Phase 1.2: deterministic PT/EN label repair to guarantee the final
        # answer does not mix languages, even if one worker emitted a label in
        # the other language. Operates only on bold `**Label**` tokens.
        formatted = enforce_language_labels(formatted, language)
        if "transport" in effective_agents:
            formatted = canonicalize_transport_terms(formatted, language=language)
            formatted = strip_technical_output_artifacts(formatted)
            formatted = ensure_transport_notes_heading(formatted, language=language)
            formatted = normalize_transport_notes_block(formatted)
            formatted = strip_redundant_transport_status_notes(formatted)

        if effective_agents:
            title = generate_response_title(effective_agents, message, language)
            final_output = ensure_response_title(formatted, title)
            if "transport" in effective_agents:
                final_output = canonicalize_transport_terms(final_output, language=language)
                final_output = strip_technical_output_artifacts(final_output)
                final_output = ensure_transport_notes_heading(final_output, language=language)
                final_output = normalize_transport_notes_block(final_output)
                final_output = strip_redundant_transport_status_notes(final_output)
        else:
            final_output = formatted

        # Prepend a visually formatted bilingual note when the user wrote in a
        # language other than PT or EN (e.g., FR, DE, JA). The assistant is
        # optimized for PT-PT and EN, so the final response is provided in
        # English along with a small note explaining why.
        user_ctx = self.state.get("user_context") or {}
        if user_ctx.get("requires_bilingual_note") and final_output.strip():
            detected = user_ctx.get("detected_language") or "und"
            note = build_bilingual_note(detected)
            if note and note not in final_output:
                final_output = f"{note}\n\n{final_output}"

        planner_involved = "planner" in effective_agents
        single_domain_agents = [
            agent_name for agent_name in effective_agents if agent_name in {"weather", "researcher", "transport"}
        ]
        if not planner_involved and len(single_domain_agents) == 1:
            final_output = finalize_worker_response(
                final_output,
                agent_name=single_domain_agents[0],
                user_query=message,
                language=language,
            )

        final_output = final_visual_pass(final_output)
        if "transport" in effective_agents:
            final_output = enforce_language_labels(final_output, language)
            final_output = canonicalize_transport_terms(final_output, language=language)
            final_output = ensure_transport_notes_heading(final_output, language=language)
            final_output = normalize_transport_notes_block(final_output)
            final_output = strip_redundant_transport_status_notes(final_output)
            final_output = final_visual_pass(final_output)

        if "weather" in effective_agents and "umbrella" not in final_output.lower() and "guarda-chuva" not in final_output.lower():
            umbrella_advice = self._build_umbrella_advice(
                user_query=message,
                weather_output=str(agent_outputs.get("weather") or ""),
                language=language,
            )
            if umbrella_advice:
                final_output = f"{final_output.rstrip()}\n\n---\n\n{umbrella_advice}"
                final_output = final_visual_pass(final_output)

        if "weather" in effective_agents:
            activity_advice = self._build_weather_activity_advice(
                user_query=message,
                weather_output=str(agent_outputs.get("weather") or ""),
                language=language,
            )
            if activity_advice and "viabilidade da atividade" not in final_output.lower() and "activity feasibility" not in final_output.lower():
                final_output = f"{activity_advice}\n\n---\n\n{final_output.rstrip()}"
                final_output = final_visual_pass(final_output)

        if {"researcher", "transport"}.issubset(set(effective_agents)):
            enriched_service_route_output = self._append_service_metro_route_if_missing(
                final_output,
                user_message=message,
                language=language,
                agent_outputs=agent_outputs,
            )
            if enriched_service_route_output != final_output:
                final_output = final_visual_pass(enriched_service_route_output)

        planner_scope_fallback = planner_involved and any(
            marker in final_output.lower()
            for marker in (
                "i’m limiting the",
                "i'm limiting the",
                "vou limitar o pedido",
                "request too broad for a fixed plan",
                "pedido demasiado amplo para um plano fechado",
                "simple reduced-mobility evening",
                "plano simples com mobilidade reduzida",
                "resident service plan",
                "plano residente com serviços",
                "low-walk day plan",
                "plano de dia com pouca caminhada",
                "planning framework",
                "framework de planeamento",
                "framework dos primeiros 5 dias",
                "estrutura de planeamento",
                "relaxed one-day plan",
                "plano relaxado de um dia",
                "suggested evening plan",
                "plano de fim de tarde",
            )
        )

        planner_has_structured_footer = planner_involved and has_source_line(final_output)
        if planner_has_structured_footer:
            # The structured planner renderer already cites only the source_ids
            # selected by the PlanDraft. Do not replace that precise footer with
            # a broader combined footer from every worker that happened to run.
            final_output = canonicalize_planner_source_line(final_output, language=language)
            final_output = final_visual_pass(final_output)
        elif (
            agent_outputs
            and not planner_scope_fallback
            and not self._is_unsupported_transport_scope_response(final_output)
        ):
            source_footer = self._build_combined_source_footer(agent_outputs, language)
            if source_footer:
                footer_line_re = re.compile(r"^(?:[-*•]\s*)?📌\s*\*\*(?:Fontes?|Sources?):\*\*.*$", re.IGNORECASE)
                kept_lines = [line for line in final_output.splitlines() if not footer_line_re.match(line.strip())]
                while kept_lines and not kept_lines[-1].strip():
                    kept_lines.pop()
                final_output = "\n".join(kept_lines).rstrip()
                final_output = f"{final_output}\n\n{source_footer}".strip()
                final_output = final_visual_pass(final_output)
                if planner_involved:
                    final_output = canonicalize_planner_source_line(final_output, language=language)
                    final_output = final_visual_pass(final_output)
                if "transport" in effective_agents:
                    final_output = enforce_language_labels(final_output, language)
                    final_output = canonicalize_transport_terms(final_output, language=language)
                    final_output = ensure_transport_notes_heading(final_output, language=language)
                    final_output = normalize_transport_notes_block(final_output)
                    final_output = strip_redundant_transport_status_notes(final_output)
                    final_output = final_visual_pass(final_output)

        if not planner_involved and len(single_domain_agents) == 1:
            final_output = finalize_worker_response(
                final_output,
                agent_name=single_domain_agents[0],
                user_query=message,
                language=language,
            )

        if (
            not planner_involved
            and single_domain_agents == ["researcher"]
            and isinstance(agent_outputs.get("researcher"), str)
        ):
            final_output = reconcile_researcher_place_response(
                final_output,
                agent_outputs["researcher"],
                language=language,
                user_query=message,
            )
            final_output = reconcile_researcher_event_response(
                final_output,
                agent_outputs["researcher"],
                language=language,
                user_query=message,
            )

        if (
            not planner_involved
            and single_domain_agents == ["researcher"]
            and not self._researcher_output_looks_structured_lisboa_aberta_service(final_output)
            and infer_researcher_source_kind(user_query=message, text=final_output) == "places"
            and not self._is_researcher_no_more_pagination_response(final_output)
            and researcher_place_response_missing_requested_fields(
                final_output,
                user_query=message,
            )
        ):
            researcher_agent = self.agents.get("researcher")
            if researcher_agent is not None and hasattr(researcher_agent, "_run_direct_place_lookup"):
                try:
                    direct_place_output = researcher_agent._run_direct_place_lookup(message, language)
                except Exception:
                    direct_place_output = ""
                if direct_place_output:
                    final_output = reconcile_researcher_place_response(
                        final_output,
                        direct_place_output,
                        language=language,
                        user_query=message,
                    )
                    final_output = format_researcher_card(
                        final_output,
                        language=language,
                        user_query=message,
                    )
                    final_output = final_visual_pass(final_output)
                    final_output = canonicalize_visitlisboa_source_line(
                        final_output,
                        user_query=message,
                        language=language,
                    )
                    final_output = final_visual_pass(final_output)

        if (
            not planner_involved
            and single_domain_agents == ["researcher"]
            and infer_researcher_source_kind(user_query=message, text=final_output) == "events"
            and not self._is_researcher_no_more_pagination_response(final_output)
            and not (
                callable(getattr(getattr(self, "agents", {}).get("researcher"), "_is_event_category_query", None))
                and getattr(getattr(self, "agents", {}).get("researcher"), "_is_event_category_query")(message) is True
            )
            and (
                "**Categoria:**" not in final_output
                and "**Category:**" not in final_output
                or "[Mais detalhes](" not in final_output
                and "[More details](" not in final_output
            )
        ):
            researcher_agent = self.agents.get("researcher")
            if researcher_agent is not None and hasattr(researcher_agent, "_run_direct_event_lookup"):
                try:
                    direct_event_output = researcher_agent._run_direct_event_lookup(message, language)
                except Exception:
                    direct_event_output = ""
                if direct_event_output:
                    final_output = reconcile_researcher_event_response(
                        final_output,
                        direct_event_output,
                        language=language,
                        user_query=message,
                    )

        if "transport" in effective_agents and isinstance(agent_outputs.get("transport"), str):
            transport_output = str(agent_outputs.get("transport") or "")
            ambiguous_transport_route = (
                "Ilha da Madeira" in transport_output
                and "Rua Humberto Madeira" in transport_output
                and (
                    "🗺️ **Trajeto" in transport_output
                    or "🗺️ **Route" in transport_output
                    or "**Trajeto:**" in transport_output
                    or "**Route:" in transport_output
                    or "Percurso de metro" in transport_output
                    or "METRO ROUTE" in transport_output
                )
            )
            final_dropped_route = not any(
                marker in final_output
                for marker in (
                    "🗺️ **Trajeto",
                    "🗺️ **Route",
                    "**Trajeto:**",
                    "**Route:",
                    "Percurso de metro",
                    "METRO ROUTE",
                )
            )
            if ambiguous_transport_route and final_dropped_route:
                final_output = finalize_worker_response(
                    transport_output,
                    agent_name="transport",
                    user_query=message,
                    language=language,
                )
                final_output = enforce_language_labels(final_output, language)
                final_output = canonicalize_transport_terms(final_output, language=language)
                final_output = final_visual_pass(final_output)

        final_output = self._move_location_ambiguity_preamble_first(
            response=final_output,
            user_query=message,
            language=language,
        )

        final_output = self._rebuild_single_transport_source_line(final_output, language, effective_agents)

        final_output = final_visual_pass(final_output)
        if planner_scope_fallback:
            final_output = self._rebuild_planner_scope_fallback_source_line(
                final_output,
                language=language,
                effective_agents=effective_agents,
            )

        qa_agent = getattr(self, "qa_agent", None)
        final_guard = getattr(type(qa_agent), "guard_final_response", None) if qa_agent is not None else None
        if callable(final_guard):
            guarded_output = final_guard(qa_agent, final_output, language=language)
            if guarded_output != final_output:
                final_repair_ran = True
            final_output = guarded_output
            final_output = final_visual_pass(final_output)
            final_output = self._rebuild_single_transport_source_line(final_output, language, effective_agents)
            if planner_scope_fallback:
                final_output = self._rebuild_planner_scope_fallback_source_line(
                    final_output,
                    language=language,
                    effective_agents=effective_agents,
                )

        plan_like_request = bool(
            re.search(r"\b(?:roteiro|plano|itiner[aá]rio|itinerary|plan)\b", message, flags=re.IGNORECASE)
        ) and not self.supervisor._negates_itinerary_request(message)
        plan_response_needs_rebuild = bool(planner_involved or plan_like_request)
        if plan_response_needs_rebuild:
            from agent.agents.planner_agent import (
                _build_card_based_itinerary_fallback,
                _build_structured_plan_fallback,
                _extract_visitlisboa_place_cards,
                _planner_response_has_markdown_contract_defects,
                _planner_response_has_minimum_user_value,
                _planner_response_has_transport_quality_defects,
                _planner_response_loses_transport_leg_evidence,
                _planner_response_missing_requested_movement,
                _planner_response_missing_requested_food_stop,
                _planner_response_missing_requested_day_sections,
                _planner_response_missing_requested_stops,
                _planner_response_has_unrequested_sequence_stops,
                _planner_response_violates_requested_start,
                _planner_response_matches_schema,
                _strip_irrelevant_planner_movement_items,
                _ensure_requested_origin_target_in_transport_section,
                _repair_planner_address_map_links,
                _repair_planner_visitlisboa_details_links,
                _repair_visible_transport_sources,
                _repair_meal_locality_in_response,
            )

            planner_evidence_cache: dict[str, str] = {}

            def _load_planner_evidence_once() -> str:
                """Return planner evidence without repeating identical Researcher lookups."""
                if "value" in planner_evidence_cache:
                    return planner_evidence_cache["value"]

                current_researcher_data = str(agent_outputs.get("researcher") or "")
                current_cards = (
                    _extract_visitlisboa_place_cards(current_researcher_data, max_items=8)
                    if current_researcher_data
                    else []
                )
                normalized_request = unicodedata.normalize("NFKD", message or "")
                normalized_request = "".join(
                    char for char in normalized_request if not unicodedata.combining(char)
                ).lower()
                needs_food = bool(
                    re.search(
                        r"\b(?:food|comida|restaurant|restaurante|pastry|pastelaria|"
                        r"pastel|cafe|coffee|lunch|dinner|almo[cç]o|jantar)\b",
                        normalized_request,
                    )
                )
                has_food_evidence = bool(
                    re.search(
                        r"\b(?:coffee shop|restaurant|restaurante|past[eé]is|pastelaria|"
                        r"cafe|caf[eé]|food)\b",
                        current_researcher_data,
                        flags=re.IGNORECASE,
                    )
                )
                if len(current_cards) >= 2 and (not needs_food or has_food_evidence):
                    planner_evidence_cache["value"] = current_researcher_data
                    return current_researcher_data

                researcher_agent = self.agents.get("researcher")
                evidence_lookup = getattr(researcher_agent, "_run_planner_evidence_lookup", None)
                if callable(evidence_lookup):
                    planner_evidence_cache["value"] = str(evidence_lookup(message, language) or "")
                else:
                    planner_evidence_cache["value"] = current_researcher_data
                return planner_evidence_cache["value"]

            normalized_final_output = unicodedata.normalize("NFKD", final_output or "")
            normalized_final_output = "".join(
                char for char in normalized_final_output if not unicodedata.combining(char)
            ).lower()
            if (
                _planner_response_has_markdown_contract_defects(final_output)
                or not _planner_response_matches_schema(final_output)
                or not _planner_response_has_minimum_user_value(final_output)
                or _planner_response_has_transport_quality_defects(final_output, message, str(agent_outputs.get("transport", "") or ""))
                or _planner_response_loses_transport_leg_evidence(final_output, str(agent_outputs.get("transport", "") or ""))
                or _planner_response_missing_requested_movement(final_output, message, str(agent_outputs.get("transport", "") or ""))
                or _planner_response_missing_requested_food_stop(final_output, message)
                or _planner_response_missing_requested_day_sections(final_output, message)
                or _planner_response_has_unrequested_sequence_stops(final_output, message)
                or _planner_response_violates_requested_start(final_output, message)
                or _planner_response_missing_requested_stops(
                    final_output,
                    message,
                    "\n".join([
                        str(agent_outputs.get("_conversation_context", "") or ""),
                        str(agent_outputs.get("researcher", "") or ""),
                        str(agent_outputs.get("events", "") or agent_outputs.get("_events_context", "") or ""),
                    ]),
                )
                or "restricoes nao especificadas" in normalized_final_output
                or "paragem cultural confirmavel" in normalized_final_output
            ):
                researcher_data = str(agent_outputs.get("researcher", "") or "")
                researcher_cards = _extract_visitlisboa_place_cards(researcher_data, max_items=8) if researcher_data else []
                researcher_agent = self.agents.get("researcher")
                evidence_lookup = getattr(researcher_agent, "_run_planner_evidence_lookup", None)
                if callable(evidence_lookup):
                    enriched_researcher_data = _load_planner_evidence_once()
                    enriched_cards = (
                        _extract_visitlisboa_place_cards(enriched_researcher_data, max_items=8)
                        if enriched_researcher_data
                        else []
                    )
                    if enriched_cards and len(enriched_cards) >= max(2, len(researcher_cards)):
                        researcher_data = enriched_researcher_data
                        agent_outputs["researcher"] = researcher_data
                rebuilt_plan = _build_card_based_itinerary_fallback(
                    user_message=message,
                    language=language,
                    weather_data=str(agent_outputs.get("weather", "") or ""),
                    transport_data=str(agent_outputs.get("transport", "") or ""),
                    places_data=researcher_data,
                    events_data=str(agent_outputs.get("events", "") or agent_outputs.get("_events_context", "") or ""),
                    qa_disclaimers=agent_outputs.get("_qa_disclaimers"),
                    conversation_context=str(agent_outputs.get("_conversation_context", "") or ""),
                ) or _build_structured_plan_fallback(
                    user_message=message,
                    language=language,
                    weather_data=str(agent_outputs.get("weather", "") or ""),
                    transport_data=str(agent_outputs.get("transport", "") or ""),
                    places_data=researcher_data,
                    events_data=str(agent_outputs.get("events", "") or agent_outputs.get("_events_context", "") or ""),
                    qa_disclaimers=agent_outputs.get("_qa_disclaimers"),
                    conversation_context=str(agent_outputs.get("_conversation_context", "") or ""),
                )
                if not rebuilt_plan:
                    researcher_agent = self.agents.get("researcher")
                    evidence_lookup = getattr(researcher_agent, "_run_planner_evidence_lookup", None)
                    if callable(evidence_lookup):
                        researcher_data = _load_planner_evidence_once()
                        if researcher_data:
                            agent_outputs["researcher"] = researcher_data
                            rebuilt_plan = _build_card_based_itinerary_fallback(
                                user_message=message,
                                language=language,
                                weather_data=str(agent_outputs.get("weather", "") or ""),
                                transport_data=str(agent_outputs.get("transport", "") or ""),
                                places_data=researcher_data,
                                events_data=str(agent_outputs.get("events", "") or agent_outputs.get("_events_context", "") or ""),
                                qa_disclaimers=agent_outputs.get("_qa_disclaimers"),
                                conversation_context=str(agent_outputs.get("_conversation_context", "") or ""),
                            ) or _build_structured_plan_fallback(
                                user_message=message,
                                language=language,
                                weather_data=str(agent_outputs.get("weather", "") or ""),
                                transport_data=str(agent_outputs.get("transport", "") or ""),
                                places_data=researcher_data,
                                events_data=str(agent_outputs.get("events", "") or agent_outputs.get("_events_context", "") or ""),
                                qa_disclaimers=agent_outputs.get("_qa_disclaimers"),
                                conversation_context=str(agent_outputs.get("_conversation_context", "") or ""),
                            )
                if rebuilt_plan:
                    final_output = final_post_qa_guard(final_visual_pass(rebuilt_plan), language=language)

        final_output = final_post_qa_guard(final_output, language=language)
        if "planner" in effective_agents:
            meal_places_data = "\n\n".join(
                item for item in (
                    str(agent_outputs.get("researcher", "") or ""),
                    self._planner_meal_research_supplement(
                        message,
                        "\n".join([
                            str(agent_outputs.get("_conversation_context", "") or ""),
                            final_output,
                        ]),
                        "",
                        language,
                    ),
                )
                if item
            )
            final_output = _repair_meal_locality_in_response(
                final_output,
                user_message=message,
                places_data=meal_places_data,
                language=language,
            )
            final_output = final_post_qa_guard(final_visual_pass(final_output), language=language)
        if plan_response_needs_rebuild:
            final_output = _strip_irrelevant_planner_movement_items(
                final_output,
                message,
                language,
            )
            final_output = _ensure_requested_origin_target_in_transport_section(
                final_output,
                message,
                language,
                str(agent_outputs.get("transport", "") or ""),
            )
            final_output = _repair_planner_address_map_links(final_output)
            final_output = _repair_planner_visitlisboa_details_links(
                final_output,
                places_data=str(agent_outputs.get("researcher", "") or ""),
                language=language,
            )
            final_output = _repair_visible_transport_sources(final_output)
            final_output = final_post_qa_guard(final_visual_pass(final_output), language=language)

            normalized_final_output = unicodedata.normalize("NFKD", final_output or "")
            normalized_final_output = "".join(
                char for char in normalized_final_output if not unicodedata.combining(char)
            ).lower()
            if (
                _planner_response_has_markdown_contract_defects(final_output)
                or "paragem cultural confirmavel" in normalized_final_output
                or not _planner_response_matches_schema(final_output)
                or not _planner_response_has_minimum_user_value(final_output)
                or _planner_response_has_transport_quality_defects(final_output, message, str(agent_outputs.get("transport", "") or ""))
                or _planner_response_loses_transport_leg_evidence(final_output, str(agent_outputs.get("transport", "") or ""))
                or _planner_response_missing_requested_movement(final_output, message, str(agent_outputs.get("transport", "") or ""))
                or _planner_response_missing_requested_food_stop(final_output, message)
                or _planner_response_missing_requested_day_sections(final_output, message)
                or _planner_response_has_unrequested_sequence_stops(final_output, message)
                or _planner_response_violates_requested_start(final_output, message)
                or _planner_response_missing_requested_stops(
                    final_output,
                    message,
                    "\n".join([
                        str(agent_outputs.get("_conversation_context", "") or ""),
                        str(agent_outputs.get("researcher", "") or ""),
                        str(agent_outputs.get("events", "") or agent_outputs.get("_events_context", "") or ""),
                    ]),
                )
            ):
                researcher_data = str(agent_outputs.get("researcher", "") or "")
                researcher_cards = _extract_visitlisboa_place_cards(researcher_data, max_items=8) if researcher_data else []
                researcher_agent = self.agents.get("researcher")
                evidence_lookup = getattr(researcher_agent, "_run_planner_evidence_lookup", None)
                if callable(evidence_lookup):
                    enriched_researcher_data = _load_planner_evidence_once()
                    enriched_cards = (
                        _extract_visitlisboa_place_cards(enriched_researcher_data, max_items=8)
                        if enriched_researcher_data
                        else []
                    )
                    if enriched_cards and len(enriched_cards) >= max(2, len(researcher_cards)):
                        researcher_data = enriched_researcher_data
                        agent_outputs["researcher"] = researcher_data
                rebuilt_plan = _build_card_based_itinerary_fallback(
                    user_message=message,
                    language=language,
                    weather_data=str(agent_outputs.get("weather", "") or ""),
                    transport_data=str(agent_outputs.get("transport", "") or ""),
                    places_data=researcher_data,
                    events_data=str(agent_outputs.get("events", "") or agent_outputs.get("_events_context", "") or ""),
                    qa_disclaimers=agent_outputs.get("_qa_disclaimers"),
                    conversation_context=str(agent_outputs.get("_conversation_context", "") or ""),
                ) or _build_structured_plan_fallback(
                    user_message=message,
                    language=language,
                    weather_data=str(agent_outputs.get("weather", "") or ""),
                    transport_data=str(agent_outputs.get("transport", "") or ""),
                    places_data=researcher_data,
                    events_data=str(agent_outputs.get("events", "") or agent_outputs.get("_events_context", "") or ""),
                    qa_disclaimers=agent_outputs.get("_qa_disclaimers"),
                    conversation_context=str(agent_outputs.get("_conversation_context", "") or ""),
                )
                if not rebuilt_plan:
                    researcher_agent = self.agents.get("researcher")
                    evidence_lookup = getattr(researcher_agent, "_run_planner_evidence_lookup", None)
                    if callable(evidence_lookup):
                        researcher_data = _load_planner_evidence_once()
                        if researcher_data:
                            agent_outputs["researcher"] = researcher_data
                            rebuilt_plan = _build_card_based_itinerary_fallback(
                                user_message=message,
                                language=language,
                                weather_data=str(agent_outputs.get("weather", "") or ""),
                                transport_data=str(agent_outputs.get("transport", "") or ""),
                                places_data=researcher_data,
                                events_data=str(agent_outputs.get("events", "") or agent_outputs.get("_events_context", "") or ""),
                                qa_disclaimers=agent_outputs.get("_qa_disclaimers"),
                                conversation_context=str(agent_outputs.get("_conversation_context", "") or ""),
                            ) or _build_structured_plan_fallback(
                                user_message=message,
                                language=language,
                                weather_data=str(agent_outputs.get("weather", "") or ""),
                                transport_data=str(agent_outputs.get("transport", "") or ""),
                                places_data=researcher_data,
                                events_data=str(agent_outputs.get("events", "") or agent_outputs.get("_events_context", "") or ""),
                                qa_disclaimers=agent_outputs.get("_qa_disclaimers"),
                                conversation_context=str(agent_outputs.get("_conversation_context", "") or ""),
                            )
                if rebuilt_plan:
                    final_output = final_post_qa_guard(final_visual_pass(rebuilt_plan), language=language)
        if planner_involved or plan_like_request:
            final_output = canonicalize_planner_source_line(final_output, language=language)
            final_output = final_visual_pass(final_output)
            final_output = final_post_qa_guard(final_output, language=language)

        final_output = final_post_qa_guard(final_visual_pass(final_output), language=language)

        public_agent_outputs = {
            key: value for key, value in agent_outputs.items()
            if not str(key).startswith("_") and isinstance(value, str) and str(value).strip()
        }
        final_audit = None
        if (
            not direct_response_used
            and public_agent_outputs
            and qa_agent is not None
            and hasattr(qa_agent, "assess_final_response")
        ):
            try:
                final_audit = qa_agent.assess_final_response(
                    user_query=message,
                    final_response=final_output,
                    language=language,
                    user_context=self.state.get("user_context"),
                )
            except Exception as exc:
                logger.warning("Final QA response audit failed; keeping guarded output: %s", exc)
                final_audit = None

        if final_audit and final_audit.get("needs_repair"):
            qa_result = self._merge_qa_result_payloads(qa_result, final_audit)
            try:
                final_output = qa_agent.repair_final_response(
                    user_query=message,
                    draft_response=final_output,
                    agent_outputs=agent_outputs,
                    qa_result=qa_result,
                    language=language,
                )
                final_output = final_post_qa_guard(final_visual_pass(final_output), language=language)
                if planner_involved or plan_like_request:
                    final_output = canonicalize_planner_source_line(final_output, language=language)
                    final_output = final_post_qa_guard(final_visual_pass(final_output), language=language)
                final_repair_ran = True
            except Exception as exc:
                logger.warning("Final QA response repair failed; keeping guarded output: %s", exc)

        if planner_involved or plan_like_request:
            from agent.agents.planner_agent import (
                _build_card_based_itinerary_fallback,
                _build_structured_plan_fallback,
                _extract_visitlisboa_place_cards,
                _planner_response_has_markdown_contract_defects,
                _planner_response_has_minimum_user_value,
                _planner_response_has_transport_quality_defects,
                _planner_response_loses_transport_leg_evidence,
                _planner_response_missing_requested_movement,
                _planner_response_missing_requested_food_stop,
                _planner_response_missing_requested_plan_components,
                _planner_response_missing_requested_stops,
                _planner_response_has_unrequested_sequence_stops,
                _planner_response_violates_requested_start,
                _strip_irrelevant_planner_movement_items,
                _ensure_requested_origin_target_in_transport_section,
                _repair_planner_address_map_links,
                _repair_planner_visitlisboa_details_links,
                _repair_visible_transport_sources,
            )

            final_output = _strip_irrelevant_planner_movement_items(
                final_output,
                message,
                language,
            )
            final_output = _ensure_requested_origin_target_in_transport_section(
                final_output,
                message,
                language,
                str(agent_outputs.get("transport", "") or ""),
            )
            final_output = _repair_planner_address_map_links(final_output)
            final_output = _repair_planner_visitlisboa_details_links(
                final_output,
                places_data=str(agent_outputs.get("researcher", "") or ""),
                language=language,
            )
            final_output = _repair_visible_transport_sources(final_output)
            final_output = final_post_qa_guard(final_visual_pass(final_output), language=language)

            evidence_context = "\n".join([
                str(agent_outputs.get("_conversation_context", "") or ""),
                str(agent_outputs.get("researcher", "") or ""),
                str(agent_outputs.get("events", "") or agent_outputs.get("_events_context", "") or ""),
            ])
            if (
                _planner_response_has_markdown_contract_defects(final_output)
                or not _planner_response_has_minimum_user_value(final_output)
                or _planner_response_has_transport_quality_defects(final_output, message, str(agent_outputs.get("transport", "") or ""))
                or _planner_response_loses_transport_leg_evidence(final_output, str(agent_outputs.get("transport", "") or ""))
                or _planner_response_missing_requested_movement(
                    final_output,
                    message,
                    str(agent_outputs.get("transport", "") or ""),
                )
                or _planner_response_missing_requested_stops(
                    final_output,
                    message,
                    evidence_context,
                )
                or _planner_response_missing_requested_plan_components(final_output, message)
                or _planner_response_missing_requested_food_stop(final_output, message)
                or _planner_response_has_unrequested_sequence_stops(final_output, message)
                or _planner_response_violates_requested_start(final_output, message)
            ):
                current_researcher_data = str(agent_outputs.get("researcher", "") or "")
                current_cards = _extract_visitlisboa_place_cards(current_researcher_data, max_items=8) if current_researcher_data else []
                researcher_agent = self.agents.get("researcher")
                evidence_lookup = getattr(researcher_agent, "_run_planner_evidence_lookup", None)
                if callable(evidence_lookup):
                    enriched_researcher_data = _load_planner_evidence_once()
                    enriched_cards = (
                        _extract_visitlisboa_place_cards(enriched_researcher_data, max_items=8)
                        if enriched_researcher_data
                        else []
                    )
                    if enriched_cards and len(enriched_cards) >= max(2, len(current_cards)):
                        agent_outputs["researcher"] = enriched_researcher_data
                rebuilt_plan = _build_card_based_itinerary_fallback(
                    user_message=message,
                    language=language,
                    weather_data=str(agent_outputs.get("weather", "") or ""),
                    transport_data=str(agent_outputs.get("transport", "") or ""),
                    places_data=str(agent_outputs.get("researcher", "") or ""),
                    events_data=str(agent_outputs.get("events", "") or agent_outputs.get("_events_context", "") or ""),
                    qa_disclaimers=agent_outputs.get("_qa_disclaimers"),
                    conversation_context=str(agent_outputs.get("_conversation_context", "") or ""),
                ) or _build_structured_plan_fallback(
                    user_message=message,
                    language=language,
                    weather_data=str(agent_outputs.get("weather", "") or ""),
                    transport_data=str(agent_outputs.get("transport", "") or ""),
                    places_data=str(agent_outputs.get("researcher", "") or ""),
                    events_data=str(agent_outputs.get("events", "") or agent_outputs.get("_events_context", "") or ""),
                    qa_disclaimers=agent_outputs.get("_qa_disclaimers"),
                    conversation_context=str(agent_outputs.get("_conversation_context", "") or ""),
                )
                if rebuilt_plan:
                    final_output = final_post_qa_guard(final_visual_pass(rebuilt_plan), language=language)
                    final_output = _strip_irrelevant_planner_movement_items(
                        final_output,
                        message,
                        language,
                    )
                    final_output = _ensure_requested_origin_target_in_transport_section(
                        final_output,
                        message,
                        language,
                        str(agent_outputs.get("transport", "") or ""),
                    )
                    final_output = _repair_planner_address_map_links(final_output)
                    final_output = _repair_planner_visitlisboa_details_links(
                        final_output,
                        places_data=str(agent_outputs.get("researcher", "") or ""),
                        language=language,
                    )
                    final_output = _repair_visible_transport_sources(final_output)
                    final_output = canonicalize_planner_source_line(final_output, language=language)
                    final_output = final_post_qa_guard(final_visual_pass(final_output), language=language)

            final_output = self._repair_incomplete_visible_planner_route_with_tool(
                final_output,
                message,
                language,
            )
            final_output = _repair_visible_transport_sources(final_output)
            final_output = canonicalize_planner_source_line(final_output, language=language)
            final_output = final_post_qa_guard(final_visual_pass(final_output), language=language)

        if (
            not planner_involved
            and single_domain_agents == ["researcher"]
            and not self._researcher_output_looks_structured_lisboa_aberta_service(final_output)
        ):
            final_output = format_researcher_card(
                final_output,
                language=language,
                user_query=message,
            )
            final_output = strip_excluded_place_cards(
                final_output,
                user_query=message,
                language=language,
            )
            final_output = ensure_requested_area_limitation(
                final_output,
                user_query=message,
                language=language,
            )
            final_output = final_post_qa_guard(final_visual_pass(final_output), language=language)

        if (
            not planner_involved
            and single_domain_agents == ["researcher"]
            and isinstance(agent_outputs.get("researcher"), str)
            and self._researcher_output_looks_structured_lisboa_aberta_service(agent_outputs["researcher"])
        ):
            final_output = finalize_worker_response(
                agent_outputs["researcher"],
                agent_name="researcher",
                user_query=message,
                language=language,
            )
            final_output = final_visual_pass(final_output)

        final_output = self._repair_wrong_contextual_transport_destination(final_output, language)
        final_output = self._rebuild_single_transport_source_line(final_output, language, effective_agents)
        final_output = final_post_qa_guard(final_visual_pass(final_output), language=language)
        final_output = self._rebuild_single_transport_source_line(final_output, language, effective_agents)
        final_output = final_post_qa_guard(final_output, language=language)
        if planner_involved or plan_like_request:
            from agent.agents.planner_agent import (
                _ensure_requested_return_to_origin_in_transport_section,
                _repair_meal_locality_in_response,
            )

            meal_places_data = "\n\n".join(
                item for item in (
                    str(agent_outputs.get("researcher", "") or ""),
                    self._planner_meal_research_supplement(
                        message,
                        "\n".join([
                            str(agent_outputs.get("_conversation_context", "") or ""),
                            final_output,
                        ]),
                        "",
                        language,
                    ),
                )
                if item
            )
            final_output = _repair_meal_locality_in_response(
                final_output,
                user_message=message,
                places_data=meal_places_data,
                language=language,
            )
            final_output = final_post_qa_guard(final_visual_pass(final_output), language=language)
            final_output = _ensure_requested_return_to_origin_in_transport_section(
                final_output,
                message,
                language,
            )

        self._update_conversation_anchors(message, final_output, effective_agents)

        self._append_assistant_message(final_output)

        execution_summary = self._collect_execution_summary(
            user_request=message,
            routing_reasoning=routing_reasoning,
            agents_to_call=agents_to_call,
            agent_outputs=agent_outputs,
            direct_response_used=direct_response_used,
            workers=workers,
            run_workers_in_parallel=run_workers_in_parallel,
            qa_result=qa_result,
            retry_agents_used=retry_agents_used,
            final_repair_ran=final_repair_ran,
            simple_weather_fact_check=simple_weather_fact_check,
            elapsed_time=time_module.time() - start_time,
        )
        self.last_execution_summary = execution_summary
        self._print_execution_summary(execution_summary)

        if Config.SHOW_MARKDOWN_RESPONSE_IN_TERMINAL:
            _print_final_markdown_response(final_output)

        return final_output

    @staticmethod
    def _rebuild_planner_scope_fallback_source_line(
        text: str,
        language: str,
        effective_agents: List[str],
    ) -> str:
        """Keep bounded planner fallbacks from inheriting broad worker footers."""
        if not text:
            return text

        lowered = text.lower()
        sources: List[str] = []
        if (
            "weather" in effective_agents
            and "weather was not retrieved" not in lowered
            and "meteorologia não foi consultada" not in lowered
            and (
                "no active weather warnings" in lowered
                or "sem avisos meteorológicos ativos" in lowered
                or "ipma forecast" in lowered
                or "previsão do ipma" in lowered
                or "temperature" in lowered
                or "temperatura" in lowered
                or "conditions" in lowered
                or "condições" in lowered
                or "sunny intervals" in lowered
                or "intervalos de sol" in lowered
                or "wind" in lowered
                or "vento" in lowered
                or "rain" in lowered
                or "chuva" in lowered
                or "showers" in lowered
                or "aguaceiros" in lowered
            )
        ):
            sources.append("[*IPMA*](https://www.ipma.pt)")
        if "lisboa aberta" in lowered or "mercado de campo de ourique" in lowered:
            sources.append("[*Lisboa Aberta*](https://dados.cm-lisboa.pt/)")
        if any(
            marker in lowered
            for marker in (
                "visitlisboa",
                "national coach museum",
                "museu nacional dos coches",
                "maat",
                "oceanário",
                "oceanario",
                "pavilhão do conhecimento",
                "pavilhao do conhecimento",
                "gulbenkian",
                "doca de santo",
                "museu nacional de arte antiga",
                "national museum of ancient art",
                "casa fernando pessoa",
            )
        ):
            sources.append(
                "[*VisitLisboa Locais*](https://www.visitlisboa.com/pt-pt/locais)"
                if language == "pt"
                else "[*VisitLisboa Places*](https://www.visitlisboa.com/en/places)"
            )
        if "metro" in lowered or "linha verde" in lowered or "green line" in lowered:
            sources.append("[*Metro de Lisboa*](https://www.metrolisboa.pt)")
        has_carris_urban_evidence = bool(
            re.search(r"\bcarris(?!\s+metropolitana)\b", lowered)
            or "direct urban bus" in lowered
            or "autocarro urbano" in lowered
            or "carris urbana" in lowered
            or "carris urban" in lowered
        )
        if has_carris_urban_evidence:
            sources.append("[*Carris*](https://www.carris.pt)")
        if "cp " in lowered or "linha de cascais" in lowered or "cascais line" in lowered:
            sources.append("[*CP*](https://www.cp.pt)")

        deduped_sources: List[str] = []
        for source in sources:
            if source not in deduped_sources:
                deduped_sources.append(source)

        source_re = re.compile(r"(?m)^📌\s+\*\*(?:Source|Fonte):\*\*.*$")
        if not deduped_sources:
            return source_re.sub("", text, count=1).strip()

        label = "Fonte" if language == "pt" else "Source"
        updated = "Atualizado" if language == "pt" else "Updated"
        timestamp = datetime.now().strftime("%H:%M")
        replacement = f"📌 **{label}:** {' | '.join(deduped_sources)} | **{updated}:** {timestamp}"
        source_re = re.compile(r"(?m)^📌\s+\*\*(?:Source|Fonte):\*\*.*$")
        if source_re.search(text):
            return source_re.sub(replacement, text, count=1)
        return f"{text.rstrip()}\n\n{replacement}"

    @classmethod
    def _build_qa_retry_context(
        cls,
        base_context: str,
        qa_result: Optional[Dict[str, object]],
        agent_name: str,
    ) -> str:
        """Builds targeted retry context for a worker agent after QA review."""
        context = base_context
        if not qa_result:
            return context

        missing_data = list(qa_result.get("missing_data", []))
        reasoning = str(qa_result.get("reasoning", "") or "").strip()
        if missing_data:
            context += (
                "\n\nIMPORTANT QA FEEDBACK: Your previous answer missed required data: "
                + ", ".join(missing_data)
                + "."
            )
            if reasoning:
                context += f" Reasoning: {reasoning}."
            context += " Please search specifically for the missing information and return a corrected answer."

        agent_specific_feedback = cls._get_agent_specific_qa_feedback(qa_result, agent_name)
        if agent_specific_feedback:
            context += (
                "\n\nIMPORTANT QA REPAIR FEEDBACK FOR THIS AGENT: "
                "Revise your previous answer so the issues below are corrected. "
                "Keep the same user language, stay strictly supported by the tool data, "
                "and do not mention QA, validation, or internal checks.\n- "
                + "\n- ".join(agent_specific_feedback)
            )

        return context

    @staticmethod
    def _is_usable_worker_output(output: object) -> bool:
        """Return whether a worker produced evidence worth giving to the planner."""
        text = str(output or "").strip()
        if not text:
            return False
        lowered = text.lower()
        failed_markers = (
            "error:",
            "erro:",
            "failed:",
            "timeout",
            "traceback",
            "no response",
            "sem resposta",
        )
        return not lowered.startswith(failed_markers)

    @staticmethod
    def _qa_flags_missing_transit(qa_result: Optional[Dict[str, object]]) -> bool:
        """Return whether the QA result flags missing transit/transport details."""
        if not qa_result:
            return False
        missing_data = qa_result.get("missing_data") or []
        reasoning = str(qa_result.get("reasoning") or "").lower()
        transit_keywords = [
            "transport", "transit", "route", "leg", "connection", "departure",
            "carris", "metro", "viagem", "itinerário", "deslocação", "deslocacao",
            "comboio", "train", "autocarro", "bus", "elétrico", "tram", "cp",
            "carrismetropolitana", "carris metropolitana", "linha", "line",
            "paragem", "estação", "estacao", "stop", "station"
        ]
        for item in missing_data:
            lowered_item = str(item).lower()
            if any(kw in lowered_item for kw in transit_keywords):
                return True
        if any(kw in reasoning for kw in transit_keywords):
            return True
        return False

    @classmethod
    def _filter_planner_qa_retry_agents(
        cls,
        retry_agents: List[str],
        *,
        user_message: str,
        agents_to_call: List[str],
        workers: List[str],
        agent_outputs: Dict[str, object],
        qa_result: Optional[Dict[str, object]],
    ) -> List[str]:
        """Avoid broad second worker passes before planner synthesis.

        For planner requests, already executed workers are evidence providers,
        not final renderers. QA may still request a missing domain, or retry a
        worker that failed. It should not re-run a healthy researcher or
        transport worker just because optional details remain unconfirmed; the
        planner must synthesize a bounded answer with the QA limitations.
        """
        if "planner" not in set(agents_to_call or []):
            return retry_agents

        worker_set = set(workers or [])
        filtered: List[str] = []
        skipped: List[str] = []
        for agent_name in retry_agents:
            if (
                agent_name == "weather"
                and "weather" not in worker_set
                and not cls._planner_retry_should_fetch_weather(user_message)
            ):
                skipped.append(agent_name)
                continue
            if (
                agent_name == "transport"
                and "transport" not in worker_set
                and not cls._planner_retry_should_fetch_transport(user_message)
            ):
                skipped.append(agent_name)
                continue
            if (
                agent_name == "transport"
                and (
                    cls._qa_flags_missing_transit(qa_result)
                    or "transport" in ((qa_result or {}).get("required_agents") or [])
                    or "transport" in ((qa_result or {}).get("repairable_agents") or [])
                )
            ):
                filtered.append(agent_name)
                continue
            if (
                agent_name in worker_set
                and cls._is_usable_worker_output(agent_outputs.get(agent_name))
            ):
                skipped.append(agent_name)
                continue
            filtered.append(agent_name)

        if skipped and isinstance(qa_result, dict):
            qa_result["_skipped_planner_retry_agents"] = cls._dedupe_preserve_order(skipped)
        return cls._dedupe_preserve_order(filtered)

    @staticmethod
    def _planner_retry_should_fetch_weather(user_message: str) -> bool:
        """Return whether QA may add weather to a planner route after workers."""
        normalized = re.sub(r"\s+", " ", str(user_message or "").lower())
        return bool(
            re.search(
                r"\b(?:weather|forecast|rain|rainy|temperature|wind|umbrella|chuva|previs[aã]o|temperatura|vento|guarda[-\s]?chuva)\b",
                normalized,
            )
            or re.search(
                r"\b(?:today|tonight|tomorrow|this week|weekend|hoje|esta noite|amanh[ãa]|fim de semana)\b",
                normalized,
            )
        )

    @staticmethod
    def _planner_retry_should_fetch_transport(user_message: str) -> bool:
        """Return whether QA may add transport evidence to a planner request."""
        normalized = re.sub(r"\s+", " ", str(user_message or "").lower())
        return bool(
            re.search(
                r"\b(?:metro|carris|cp|autocarro|autocarros|bus|buses|comboio|comboios|train|tram|el[eé]trico|transportes?|public transport|transporte p[uú]blico)\b",
                normalized,
            )
            or re.search(
                r"\b(?:a partir de|desde|starting from|from|come[cç]ando em|come[cç]ar em|sair de|origem)\b",
                normalized,
            )
            or re.search(
                r"\b(?:como (?:chego|vou|ir)|how (?:do i )?get|route from|rota de|percurso de)\b",
                normalized,
            )
        )

    @staticmethod
    def _should_run_final_qa_repair(
        qa_result: Optional[Dict[str, object]],
    ) -> bool:
        """Returns whether a final QA repair pass is worth running."""
        if not qa_result:
            return False
        if qa_result.get("needs_repair"):
            return True
        if qa_result.get("missing_data"):
            return True

        fact_check = qa_result.get("fact_check", {})
        if isinstance(fact_check, dict) and fact_check.get("critical_issues"):
            return True

        return False

    @classmethod
    def _merge_qa_result_payloads(
        cls,
        base: Optional[Dict[str, object]],
        extra: Optional[Dict[str, object]],
    ) -> Optional[Dict[str, object]]:
        """Merge two QA payloads while preserving retry/repair semantics."""
        if not extra:
            return base

        merged: Dict[str, object] = dict(base or {})
        if not merged:
            merged = {
                "complete": True,
                "missing_data": [],
                "required_agents": [],
                "reasoning": "",
                "disclaimers": [],
                "critical_issues": [],
                "repairable_agents": [],
                "needs_repair": False,
                "fact_check": {
                    "disclaimers": [],
                    "critical_issues": [],
                    "repairable_agents": [],
                    "per_agent": {},
                },
            }

        for key in ("missing_data", "required_agents", "disclaimers", "critical_issues", "repairable_agents"):
            merged[key] = cls._dedupe_preserve_order(
                list(merged.get(key) or []) + list(extra.get(key) or [])
            )

        reasoning_parts = [
            str(merged.get("reasoning") or "").strip(),
            str(extra.get("reasoning") or "").strip(),
        ]
        merged["reasoning"] = " | ".join(part for part in reasoning_parts if part)

        fact_check = dict(merged.get("fact_check") or {})
        extra_fact_check = extra.get("fact_check") if isinstance(extra.get("fact_check"), dict) else {}
        for key in ("disclaimers", "critical_issues", "repairable_agents", "checks_performed"):
            fact_check[key] = cls._dedupe_preserve_order(
                list(fact_check.get(key) or []) + list(extra_fact_check.get(key) or [])
            )
        if extra_fact_check.get("per_agent"):
            per_agent = dict(fact_check.get("per_agent") or {})
            per_agent.update(extra_fact_check.get("per_agent") or {})
            fact_check["per_agent"] = per_agent
        fact_check["valid"] = not bool(fact_check.get("critical_issues"))
        merged["fact_check"] = fact_check

        merged["complete"] = bool(merged.get("complete")) and bool(extra.get("complete", True))
        merged["needs_repair"] = bool(
            merged.get("needs_repair")
            or extra.get("needs_repair")
            or merged.get("missing_data")
            or merged.get("critical_issues")
        )
        return merged

    @staticmethod
    def _should_block_planner_publication(
        qa_result: Optional[Dict[str, object]],
    ) -> bool:
        """Return whether planner synthesis should be suppressed after QA.

        Missing optional details should become compact caveats inside the
        planner answer. Only critical factual issues should force the graph to
        publish the evidence-supported fallback instead of a synthesized itinerary.
        """
        if not qa_result:
            return False

        def _issue_requires_block(issue: object) -> bool:
            """Return True for factual contradictions, not missing-data caveats."""
            normalized = str(issue or "").lower()
            if not normalized:
                return False
            missing_data_markers = (
                "missing",
                "unavailable",
                "not available",
                "could not resolve",
                "couldn't resolve",
                "cannot resolve",
                "not confirmed",
                "should be verified",
                "please verify",
                "not verify",
                "not verified",
                "unverified",
                "verify links",
                "verify",
                "confirmed at",
                "should be confirmed",
                "opening hours",
                "exact address",
                "route details",
                "transport details",
                "accessibility details",
                "schedule",
                "gtfs",
                "tips and warnings",
                "source footer",
                "google maps links",
                "structured field labels",
                "semantic emoji",
            )
            factual_error_markers = (
                "hallucinat",
                "invent",
                "incorrect",
                "wrong",
                "contradict",
                "outside scope",
                "out of scope",
                "unsupported venue",
                "invalid source",
                "fabricat",
            )
            if any(marker in normalized for marker in factual_error_markers):
                return True
            if any(marker in normalized for marker in missing_data_markers):
                return False
            return True

        critical_issues = qa_result.get("critical_issues") or []
        if isinstance(critical_issues, (str, bytes)):
            critical_issues = [critical_issues]
        if any(_issue_requires_block(issue) for issue in critical_issues):
            return True

        fact_check = qa_result.get("fact_check", {})
        if isinstance(fact_check, dict):
            fact_critical_issues = fact_check.get("critical_issues") or []
            if isinstance(fact_critical_issues, (str, bytes)):
                fact_critical_issues = [fact_critical_issues]
            return any(_issue_requires_block(issue) for issue in fact_critical_issues)
        return False

    @staticmethod
    def _should_preserve_direct_researcher_answer(agent_outputs: Dict[str, Any]) -> bool:
        """Return whether a direct researcher answer should bypass planner rewriting.

        Some researcher shortcuts already answer a narrow recommendation request
        with grounded caveats and a source footer. Sending those through the
        planner can turn a concise recommendation into a malformed itinerary.
        """
        public_output_keys = {
            key for key in agent_outputs.keys() if not str(key).startswith("_")
        }
        if public_output_keys != {"researcher"}:
            return False

        researcher_output = str(agent_outputs.get("researcher") or "")
        normalized_output = researcher_output.lower()
        has_timed_recommendation = bool(
            re.search(
                r"(?:recomenda(?:ção|cao)|recommendation)\s+(?:para|for)\s+\d{1,2}:\d{2}\s*[-–]\s*\d{1,2}:\d{2}",
                normalized_output,
            )
        )
        if not has_timed_recommendation:
            return False

        return "📌" in researcher_output and any(
            marker in normalized_output
            for marker in (
                "open-air",
                "outdoor",
                "exterior",
                "monument",
                "miradouro",
                "viewpoint",
                "fachada",
                "ao ar livre",
            )
        )

    def _maybe_enrich_planner_transport_context(
        self,
        *,
        user_message: str,
        language: str,
        agent_outputs: Dict[str, Any],
        verbose: bool = False,
    ) -> None:
        """Add route-leg evidence for planner requests after POIs are known.

        Workers normally run in parallel, so the first TransportAgent pass does
        not know which POIs the Researcher will surface. For itinerary requests,
        this lightweight second pass uses the concrete researcher cards to fetch
        up to three public-transport legs before QA and Planner synthesis.

        Args:
            user_message: Original user request.
            language: Resolved output language.
            agent_outputs: Mutable worker outputs for the current request.
            verbose: Whether to print diagnostic information.
        """
        researcher_text = str(agent_outputs.get("researcher") or "")
        events_text = str(agent_outputs.get("events") or agent_outputs.get("_events_context") or "")
        if events_text.strip() and events_text not in researcher_text:
            researcher_text = f"{researcher_text.rstrip()}\n\n{events_text.strip()}".strip()
        if not researcher_text.strip():
            return

        transport_text = str(agent_outputs.get("transport") or "")
        normalized_user_message = unicodedata.normalize("NFKD", user_message or "")
        normalized_user_message = "".join(
            char for char in normalized_user_message if not unicodedata.combining(char)
        ).lower()
        normalized_transport_text = unicodedata.normalize("NFKD", transport_text or "")
        normalized_transport_text = "".join(
            char for char in normalized_transport_text if not unicodedata.combining(char)
        ).lower()
        requires_central_belem_leg = bool(
            re.search(r"\b(?:se de lisboa|carmo|baixa|chiado|rossio|praca do comercio)\b", normalized_user_message)
            and re.search(r"\b(?:belem|torre de belem|padrao dos descobrimentos|jeronimos)\b", normalized_user_message)
        )
        has_concrete_central_belem_leg = bool(
            re.search(
                r"\b(?:op[cç][oõ]es carris|carris\s+\d{1,4}[a-z]?)\b",
                normalized_transport_text,
                flags=re.IGNORECASE,
            )
        )
        requires_event_food_leg = self._planner_request_requires_event_food_route(normalized_user_message)
        if (
            self._planner_transport_has_route_leg_evidence(transport_text)
            and not (requires_central_belem_leg and not has_concrete_central_belem_leg)
            and not requires_event_food_leg
        ):
            return

        try:
            from agent.agents.planner_agent import (
                _extract_visitlisboa_place_cards,
                _extract_requested_plan_area,
                _extract_requested_plan_origin,
                _is_historic_gastronomy_day_request,
                _localize_planner_display_title,
                _normalize_planner_text,
                _order_historic_food_cards,
                _planner_card_display_name,
                _select_planner_cards_for_request,
            )
            from tools.carris_api import carris_find_routes_between
            from tools.transport_api import get_route_between_stations
        except Exception as exc:
            if verbose:
                print(f"   [PLANNER-TRANSPORT] Route-leg enrichment unavailable: {type(exc).__name__}: {exc}")
            return

        requested_origin = _extract_requested_plan_origin(user_message)
        requested_target = _extract_requested_plan_area(user_message)
        normalized_requested_origin = _normalize_planner_text(requested_origin)
        normalized_requested_target = _normalize_planner_text(requested_target)
        transport_has_requested_route_detail = bool(
            normalized_requested_origin
            and normalized_requested_target
            and normalized_requested_origin in normalized_transport_text
            and normalized_requested_target in normalized_transport_text
            and re.search(
                r"\b(?:ate|to|exit|sai|saia|alight|walk|segue|station|estacao|paragem|stop)\b",
                normalized_transport_text,
            )
        )
        mode_constrained_away_from_metro = bool(
            re.search(
                r"\b(?:autocarro|autocarros|bus|buses|carris|el[eé]trico|el[eé]tricos|tram|trams|comboio|comboios|train|trains)\b",
                normalized_user_message,
            )
            and not re.search(r"\bmetro\b", normalized_user_message)
        )
        if (
            requested_origin
            and requested_target
            and not mode_constrained_away_from_metro
            and not transport_has_requested_route_detail
        ):
            route_args = {"origin": requested_origin, "destination": requested_target}
            try:
                requested_route_output = str(get_route_between_stations.invoke(route_args) or "").strip()
            except Exception as exc:
                requested_route_output = ""
                if verbose:
                    print(f"   [PLANNER-TRANSPORT] Requested origin-target route failed: {type(exc).__name__}: {exc}")
            if requested_route_output and requested_route_output not in transport_text:
                transport_text = (
                    f"{requested_route_output}\n\n{transport_text.rstrip()}"
                    if transport_text.strip()
                    else requested_route_output
                )
                agent_outputs["transport"] = transport_text
                transport_agent = self.agents.get("transport")
                if transport_agent is not None and hasattr(transport_agent, "_record_tool_call"):
                    transport_agent._record_tool_call("get_route_between_stations", route_args)

        cards = _extract_visitlisboa_place_cards(researcher_text, max_items=8)
        if len(cards) < 2:
            researcher_agent = self.agents.get("researcher")
            evidence_lookup = getattr(researcher_agent, "_run_planner_evidence_lookup", None)
            if callable(evidence_lookup):
                try:
                    enriched_researcher_text = str(evidence_lookup(user_message, language) or "")
                except Exception as exc:
                    if verbose:
                        print(f"   [PLANNER-TRANSPORT] Evidence lookup failed: {type(exc).__name__}: {exc}")
                    enriched_researcher_text = ""
                if enriched_researcher_text.strip():
                    researcher_text = (
                        f"{researcher_text.rstrip()}\n\n{enriched_researcher_text.strip()}"
                        if researcher_text.strip()
                        else enriched_researcher_text.strip()
                    )
                    agent_outputs["researcher"] = researcher_text
                    cards = _extract_visitlisboa_place_cards(researcher_text, max_items=8)
        selected_cards = _select_planner_cards_for_request(cards, user_message)[:4]
        if _is_historic_gastronomy_day_request(_normalize_planner_text(user_message)):
            selected_cards = _order_historic_food_cards(
                _select_planner_cards_for_request(cards, user_message)
            )[:4]
        if requires_event_food_leg:
            selected_cards = self._planner_event_food_route_cards(cards, selected_cards)
        route_points: List[Dict[str, str]] = []
        for card in selected_cards:
            name = (
                _planner_card_display_name(card)
                or str(card.get("name") or "").strip()
                or str(card.get("address") or "").strip()
            )
            name = re.sub(r"\s+", " ", name).strip(" .")
            address = self._plain_markdown_location(str(card.get("address") or ""))
            if name and name not in {point["name"] for point in route_points}:
                route_points.append(
                    {
                        "name": name,
                        "query": address or name,
                        "query_candidates": self._planner_route_query_candidates(name, address),
                        "zone": self._planner_card_zone(name, address),
                    }
                )

        dynamic_route_points: List[Dict[str, str]] = []

        def add_dynamic_route_point(label: str, query: str) -> None:
            cleaned_label = re.sub(r"\s+", " ", str(label or "")).strip(" .")
            cleaned_query = re.sub(r"\s+", " ", str(query or cleaned_label)).strip(" .")
            if not cleaned_label:
                return
            dynamic_route_points.append(
                {
                    "name": cleaned_label,
                    "query": cleaned_query or cleaned_label,
                    "query_candidates": self._planner_route_query_candidates(
                        cleaned_label,
                        cleaned_query or cleaned_label,
                    ),
                    "zone": self._planner_card_zone(cleaned_label, cleaned_query),
                }
            )

        if requested_origin:
            add_dynamic_route_point(requested_origin, requested_origin)
        if requested_target:
            add_dynamic_route_point(requested_target, requested_target)

        query_route_points = [
            *dynamic_route_points,
            *self._planner_requested_route_points(normalized_user_message),
        ]
        if query_route_points:
            merged_points: List[Dict[str, str]] = []
            seen_point_keys: set[str] = set()
            for point in [*query_route_points, *route_points]:
                key = re.sub(r"\s+", " ", str(point.get("name") or "").lower()).strip()
                if not key or key in seen_point_keys:
                    continue
                seen_point_keys.add(key)
                merged_points.append(point)
            route_points = merged_points

        if len(route_points) < 2:
            return

        is_pt = language == "pt"
        heading = (
            "### 🚇 **Ligações entre paragens do roteiro**"
            if is_pt
            else "### 🚇 **Route Legs Between Itinerary Stops**"
        )
        sections: List[str] = [heading]
        transport_agent = self.agents.get("transport")
        for origin, destination in zip(route_points, route_points[1:4]):
            origin_name = _localize_planner_display_title(origin["name"], language) if is_pt else origin["name"]
            destination_name = _localize_planner_display_title(destination["name"], language) if is_pt else destination["name"]
            if origin["zone"] and origin["zone"] == destination["zone"]:
                sections.append(
                    self._planner_walking_leg_summary(
                        origin_name,
                        destination_name,
                        zone=origin["zone"],
                        is_pt=is_pt,
                    )
                )
                continue

            route_output = ""
            used_origin_query = origin["query"]
            used_destination_query = destination["query"]
            route_failed = False
            origin_candidates = origin.get("query_candidates") or [origin["query"]]
            destination_candidates = destination.get("query_candidates") or [destination["query"]]
            for origin_query in origin_candidates[:3]:
                for destination_query in destination_candidates[:3]:
                    try:
                        candidate_output = str(
                            carris_find_routes_between.invoke(
                                {
                                    "origin": origin_query,
                                    "destination": destination_query,
                                    "search_radius_km": 0.6,
                                }
                            )
                            or ""
                        ).strip()
                    except Exception as exc:
                        route_failed = True
                        if verbose:
                            print(f"   [PLANNER-TRANSPORT] Carris leg failed {origin_name} -> {destination_name}: {type(exc).__name__}: {exc}")
                        continue
                    if candidate_output and "Could not locate" not in candidate_output:
                        route_output = candidate_output
                        used_origin_query = origin_query
                        used_destination_query = destination_query
                        break
                if route_output:
                    break
            if not route_output and route_failed:
                continue

            summary = self._summarize_carris_plan_leg(
                route_output,
                origin_name=origin_name,
                destination_name=destination_name,
                is_pt=is_pt,
            )
            if not summary:
                continue
            if transport_agent is not None and hasattr(transport_agent, "_record_tool_call"):
                transport_agent._record_tool_call(
                    "carris_find_routes_between",
                    {
                        "origin": used_origin_query,
                        "destination": used_destination_query,
                        "search_radius_km": 0.6,
                    },
                )
            sections.append(summary)

        if len(sections) <= 1:
            return

        enrichment = "\n".join(sections).strip()
        agent_outputs["transport"] = (
            f"{enrichment}\n\n{transport_text.rstrip()}"
            if transport_text.strip()
            else enrichment
        )

    def _repair_incomplete_visible_planner_route_with_tool(
        self,
        response: str,
        user_message: str,
        language: str,
    ) -> str:
        """Repair a visible planner origin-target leg when the LLM dropped stop details."""
        if not response:
            return response

        try:
            from agent.agents.planner_agent import (
                _extract_requested_origin_target_transport_bullet,
                _extract_requested_plan_area,
                _extract_requested_plan_origin,
                _normalize_planner_text,
                _planner_origin_target_leg_has_movement_detail,
                _planner_transport_bullet_is_actionable,
            )
            from tools.transport_api import get_route_between_stations
        except Exception as exc:
            logger.warning("Planner route repair helpers unavailable: %s", exc)
            return response

        origin = _extract_requested_plan_origin(user_message)
        target = _extract_requested_plan_area(user_message)
        origin_norm = _normalize_planner_text(origin)
        target_norm = _normalize_planner_text(target)
        if not origin_norm or not target_norm:
            return response

        lines = response.splitlines()
        incomplete_indices = [
            index
            for index, line in enumerate(lines)
            if _planner_transport_bullet_is_actionable(line)
            and origin_norm in _normalize_planner_text(line)
            and target_norm in _normalize_planner_text(line)
            and not _planner_origin_target_leg_has_movement_detail(line)
        ]
        if not incomplete_indices:
            return response

        route_args = {"origin": origin, "destination": target}
        try:
            route_output = str(get_route_between_stations.invoke(route_args) or "").strip()
        except Exception as exc:
            logger.warning("Final planner route repair failed for %s -> %s: %s", origin, target, exc)
            return response

        confirmed_leg = _extract_requested_origin_target_transport_bullet(
            route_output,
            origin,
            target,
            language,
        )
        if not confirmed_leg:
            return response

        lines[incomplete_indices[0]] = f"- {confirmed_leg}"
        for index in reversed(incomplete_indices[1:]):
            del lines[index]

        transport_agent = self.agents.get("transport")
        if transport_agent is not None and hasattr(transport_agent, "_record_tool_call"):
            transport_agent._record_tool_call("get_route_between_stations", route_args)

        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

    @staticmethod
    def _service_route_origin_from_query(user_message: str) -> str:
        """Extract the user-visible origin/area in a service + route request."""
        patterns = [
            r"\bperto\s+d[eoa]?\s+(?P<origin>.+?)(?:\s+e\s+(?:diz|mostra|indica)|[,.;?!]|$)",
            r"\bnear\s+(?P<origin>.+?)(?:\s+and\s+(?:tell|show)|[,.;?!]|$)",
            r"\b(?:estou|tou)\s+(?:em|no|na|junto\s+d[eoa]?)\s+(?P<origin>.+?)(?:\s+e\b|[,.;?!]|$)",
            r"\b(?:i(?:'m| am))\s+(?:at|in|near)\s+(?P<origin>.+?)(?:\s+and\b|[,.;?!]|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, user_message or "", flags=re.IGNORECASE)
            if match:
                return re.sub(r"\s+", " ", match.group("origin")).strip(" .,:;?!")
        return ""

    @staticmethod
    def _primary_service_name_from_response(response: str) -> str:
        """Extract the first concrete public-service result name from a final answer."""
        patterns = [
            r"\*\*(?:Mais perto|Closest):\*\*\s*(?P<name>[^(\n]+)",
            r"^\s*-\s+\*\*[^\w\n]*(?P<name>(?:Biblioteca|Farm[aá]cia|Hospital|Escola|Mercado|Ecoponto|Parque|Police|Library|Pharmacy|School|Market)[^*\n]{2,90})\*\*",
        ]
        for pattern in patterns:
            match = re.search(response or "", pattern, flags=re.IGNORECASE | re.MULTILINE)
            if not match:
                continue
            name = re.sub(r"\s+", " ", match.group("name")).strip(" .,:;?!")
            if name and not re.search(r"\b(?:perto de|near|resultados|results|fonte|source)\b", name, re.IGNORECASE):
                return name
        return ""

    def _append_service_metro_route_if_missing(
        self,
        final_output: str,
        *,
        user_message: str,
        language: str,
        agent_outputs: Dict[str, Any],
    ) -> str:
        """Append a concrete Metro leg when a service answer found a place but no route."""
        normalized_query = unicodedata.normalize("NFKD", user_message or "")
        normalized_query = "".join(char for char in normalized_query if not unicodedata.combining(char)).lower()
        if "metro" not in normalized_query:
            return final_output
        if not re.search(r"\b(?:como\s+(?:la|lá)\s+chegar|how\s+to\s+get\s+there|diz.*chegar|tell.*get\s+there)\b", user_message or "", re.IGNORECASE):
            return final_output
        if re.search(r"\b(?:embarque|board|saia|exit|transfer[eê]ncia|transfer)\b", final_output or "", re.IGNORECASE):
            return final_output

        origin_label = self._service_route_origin_from_query(user_message)
        service_name = self._primary_service_name_from_response(final_output)
        if not origin_label or not service_name:
            return final_output

        try:
            from tools.location_resolver import resolve_location_query
            from tools.transport_api import get_route_between_stations
        except Exception as exc:
            logger.debug("Service route enrichment unavailable: %s", exc)
            return final_output

        try:
            origin_resolved = resolve_location_query(origin_label)
            service_resolved = resolve_location_query(service_name)
        except Exception as exc:
            logger.debug("Service route location resolution failed: %s", exc)
            return final_output

        def _nearest_metro_label(resolved: Dict[str, Any], fallback: str = "") -> str:
            if str(resolved.get("match_source") or "") == "metro_station":
                return str(resolved.get("display_name") or fallback).strip()
            nearest = resolved.get("nearest_metro") or {}
            return str(nearest.get("name") or fallback).strip()

        origin_station = _nearest_metro_label(origin_resolved, origin_label)
        destination_station = _nearest_metro_label(service_resolved, "")
        if not origin_station or not destination_station:
            return final_output
        if origin_station.lower() == destination_station.lower():
            return final_output

        route_args = {"origin": origin_station, "destination": destination_station}
        try:
            route_output = str(get_route_between_stations.invoke(route_args) or "").strip()
        except Exception as exc:
            logger.debug("Service route enrichment failed for %s: %s", route_args, exc)
            return final_output
        if not route_output or "not on Metro" in route_output or "não fica numa estação" in route_output.lower():
            return final_output

        transport_agent = self.agents.get("transport")
        formatted_route = route_output
        if transport_agent is not None and hasattr(transport_agent, "_format_deterministic_tool_result"):
            try:
                formatted_route = transport_agent._format_deterministic_tool_result(
                    tool_name="get_route_between_stations",
                    tool_args=route_args,
                    result=route_output,
                    language=language,
                    user_message=(
                        f"Como vou de {origin_station} para {destination_station} de metro?"
                        if language == "pt"
                        else f"How do I get from {origin_station} to {destination_station} by metro?"
                    ),
                )
            except Exception as exc:
                logger.debug("Service route formatting failed: %s", exc)
                formatted_route = route_output
            if hasattr(transport_agent, "_record_tool_call"):
                transport_agent._record_tool_call("get_route_between_stations", route_args)

        formatted_route = re.sub(
            r"\n?\s*(?:[-*•]\s*)?📌\s*\*\*(?:Fonte|Source|Fontes|Sources):\*\*.*$",
            "",
            formatted_route.strip(),
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        formatted_route_lines = formatted_route.splitlines()
        if formatted_route_lines and formatted_route_lines[0].lstrip().startswith("### "):
            formatted_route = "\n".join(formatted_route_lines[1:]).strip()
        if not formatted_route:
            return final_output

        heading = "### 🚇 **Como lá chegar de metro**" if language == "pt" else "### 🚇 **How to get there by metro**"
        agent_outputs["transport"] = (
            f"{str(agent_outputs.get('transport') or '').rstrip()}\n\n{formatted_route}".strip()
        )
        return f"{final_output.rstrip()}\n\n---\n\n{heading}\n\n{formatted_route}".strip()

    @staticmethod
    def _planner_request_requires_event_food_route(normalized_user_message: str) -> bool:
        """Return whether a planner request needs event-to-food movement evidence."""
        text = str(normalized_user_message or "")
        if not text:
            return False
        has_event_intent = bool(
            re.search(
                r"\b(?:evento|eventos|event|events|cultura|cultural|concerto|concert|festival|teatro|theatre|theater|show)\b",
                text,
            )
        )
        has_food_intent = bool(
            re.search(
                r"\b(?:jantar|dinner|almoco|lunch|restaurante|restaurant|gastronom|comida|food|cuisine|cozinha)\b",
                text,
            )
        )
        asks_movement = bool(
            re.search(
                r"\b(?:como\s+(?:me\s+)?desloco|desloc|transporte|transport|rota|route|ligacao|ligacoes|chegar|apanhar|go|get|move)\b",
                text,
            )
        )
        return has_event_intent and has_food_intent and asks_movement

    @staticmethod
    def _planner_card_search_text(card: Dict[str, str]) -> str:
        """Return accent-folded searchable text for a planner evidence card."""
        raw = " ".join(
            str(card.get(key) or "")
            for key in (
                "name",
                "category",
                "description",
                "features",
                "when",
                "duration",
                "venue",
                "address",
                "url",
                "details_url",
                "website_url",
                "tickets_url",
            )
        )
        normalized = unicodedata.normalize("NFKD", raw.lower())
        return "".join(char for char in normalized if not unicodedata.combining(char))

    @classmethod
    def _planner_card_looks_event(cls, card: Dict[str, str]) -> bool:
        """Return whether an evidence card represents a cultural event."""
        text = cls._planner_card_search_text(card)
        return bool(
            card.get("when")
            or card.get("duration")
            or card.get("tickets_url")
            or "visitlisboa.com/en/events" in text
            or re.search(
                r"\b(?:musica|music|teatro|theatre|theater|danca|dance|exposic|exhibit|feira|fair|festival|concerto|concert|evento|event)\b",
                text,
            )
        )

    @classmethod
    def _planner_card_looks_food(cls, card: Dict[str, str]) -> bool:
        """Return whether an evidence card represents a restaurant or food stop."""
        text = cls._planner_card_search_text(card)
        return bool(
            re.search(
                r"\b(?:restaurante|restaurant|gastronom|cozinha|cuisine|comida|food|tradicional|traditional|jantar|dinner|almoco|lunch)\b",
                text,
            )
        )

    @classmethod
    def _planner_event_food_route_cards(
        cls,
        cards: List[Dict[str, str]],
        selected_cards: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Prioritize the concrete event and food cards for route enrichment."""
        prioritized: List[Dict[str, str]] = []
        seen: set[str] = set()

        def add_card(card: Dict[str, str]) -> None:
            name = re.sub(r"\s+", " ", str(card.get("name") or "")).strip().lower()
            address = re.sub(r"\s+", " ", str(card.get("address") or "")).strip().lower()
            key = f"{name}|{address}"
            if not name or key in seen:
                return
            seen.add(key)
            prioritized.append(card)

        for predicate in (cls._planner_card_looks_event, cls._planner_card_looks_food):
            for card in cards:
                if predicate(card):
                    add_card(card)
                    break

        for card in selected_cards:
            add_card(card)
            if len(prioritized) >= 4:
                break

        return prioritized[:4]

    @staticmethod
    def _planner_requested_route_points(normalized_user_message: str) -> List[Dict[str, str]]:
        """Build route anchors explicitly named by the user for planner enrichment."""
        anchor_specs = [
            (r"\b(?:se de lisboa|catedral de lisboa)\b", "Sé de Lisboa", "Largo da Sé, 1, 1100-585, Lisboa", "baixa"),
            (r"\b(?:baixa|baixa chiado)\b", "Baixa", "Baixa, Lisboa", "baixa"),
            (r"\b(?:torre de belem|belem tower)\b", "Torre de Belém", "Av. Brasília, 1400-038, Lisboa", "belem"),
            (
                r"\b(?:padrao dos descobrimentos|monument to the discoveries)\b",
                "Padrão dos Descobrimentos",
                "Avenida de Brasília, 1400-038, Lisboa",
                "belem",
            ),
            (r"\b(?:mosteiro dos jeronimos|jeronimos monastery)\b", "Mosteiro dos Jerónimos", "Praça do Império, 1400-206, Lisboa", "belem"),
        ]
        points: List[Dict[str, str]] = []
        for pattern, name, query, zone in anchor_specs:
            if not re.search(pattern, normalized_user_message, flags=re.IGNORECASE):
                continue
            points.append(
                {
                    "name": name,
                    "query": query,
                    "query_candidates": [query, name],
                    "zone": zone,
                }
            )
        return points

    @staticmethod
    def _plain_markdown_location(value: str) -> str:
        """Return a plain address/location from a Markdown link field."""
        text = str(value or "").strip()
        link_match = re.match(r"^\[([^\]]+)\]\(https?://[^)]+\)$", text)
        if link_match:
            text = link_match.group(1)
        return re.sub(r"\s+", " ", text).strip(" .")

    @staticmethod
    def _planner_route_query_candidates(name: str, address: str) -> List[str]:
        """Build robust location queries for planner route-leg enrichment."""
        candidates: List[str] = []
        for value in (address, name):
            cleaned = re.sub(r"\s+", " ", str(value or "")).strip(" .")
            if not cleaned:
                continue
            candidates.append(cleaned)
            ascii_value = unicodedata.normalize("NFKD", cleaned)
            ascii_value = "".join(char for char in ascii_value if not unicodedata.combining(char))
            if ascii_value and ascii_value != cleaned:
                candidates.append(ascii_value)
        deduped: List[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = candidate.lower()
            if key not in seen:
                deduped.append(candidate)
                seen.add(key)
        return deduped

    @staticmethod
    def _researcher_output_looks_event_context(text: str) -> bool:
        """Return whether Researcher output should also feed Planner event context."""
        normalized = unicodedata.normalize("NFKD", str(text or "").lower())
        normalized = "".join(char for char in normalized if not unicodedata.combining(char))
        return bool(
            "visitlisboa.com/en/events" in normalized
            or "visitlisboa.com/pt-pt/eventos" in normalized
            or "visitlisboa eventos" in normalized
            or "visitlisboa events" in normalized
            or "eventos encontrados" in normalized
            or "events found" in normalized
        )

    @staticmethod
    def _researcher_output_looks_structured_lisboa_aberta_service(text: str) -> bool:
        """Return whether Researcher already produced a structured municipal-service answer."""
        value = str(text or "")
        if not value.strip():
            return False
        normalized = unicodedata.normalize("NFKD", value.lower())
        normalized = "".join(char for char in normalized if not unicodedata.combining(char))
        has_lisboa_aberta_source = "lisboa aberta" in normalized or "dados.cm-lisboa.pt" in normalized
        has_dataset_summary = bool(re.search(r"\b(?:fonte do dataset|dataset source|resultados|results)\b", normalized))
        has_service_card = bool(
            re.search(
                r"(?m)^-\s+[\U0001F100-\U0001F1FF\U0001F300-\U0001FAFF\u2300-\u27BF\uFE0F\u200D]+\s+\*\*",
                value,
            )
        )
        has_clean_limitation = bool(
            re.search(
                r"\b(?:nao encontrei resultados municipais|não encontrei resultados municipais|"
                r"could not confirm reliable municipal results)\b",
                normalized,
            )
        )
        return has_lisboa_aberta_source and (has_dataset_summary or has_service_card or has_clean_limitation)

    @staticmethod
    def _user_request_needs_municipal_service_listing(message: str) -> bool:
        """Return whether a hybrid answer should keep a Lisboa Aberta service list.

        Service listings are useful when the user asks to find, compare, or
        identify a public service. They are noisy when the user only names a
        service as a route destination or asks for weather suitability at a
        known place.
        """
        normalized = MultiAgentAssistant._fold_context_text(message)
        if not normalized:
            return False
        service_terms = (
            r"farmacia|farmacias|hospital|hospitais|centro de saude|saude|"
            r"balcao|balcoes|loja do cidadao|servico municipal|servicos municipais|"
            r"casa de banho|wc|sanitario|biblioteca|bibliotecas|mercado|mercados|"
            r"policia|psp|escola|escolas|jardim|jardins|parque|parques|"
            r"ponto de agua|bebedouro|wifi|wi-fi|posto|counter|public toilet|"
            r"pilhao|pilhoes|pilha|bateria|papeleira|papeleiras|caixote|"
            r"parque canino|parques caninos|ponto de encontro|emergencia|"
            r"estacionamento de bicicleta|estacionamento de bicicletas|"
            r"estacionamento de velocipede|estacionamento de velocipedes|"
            r"bicicleta|bicicletas|velocipede|velocipedes|"
            r"pharmacy|hospital|library|market|municipal service|public service|"
            r"battery|battery recycling|waste bin|waste bins|litter bin|dog park|"
            r"emergency meeting point|emergency meeting points|bike parking|bicycle parking"
        )
        discovery_terms = (
            r"mais proxim[oa]s?|perto de|junto de|onde (?:ha|há|existe|existem)|"
            r"qual (?:e|é)|quais|encontra|encontrar|procura|procurar|lista|listar|"
            r"mostra|indica|deixar|estacionar|near(?:by|est)?|closest|find|show|"
            r"list|which|where|leave|park"
        )
        return bool(
            re.search(rf"\b(?:{service_terms})\b", normalized)
            and re.search(rf"\b(?:{discovery_terms})\b", normalized)
        )

    @staticmethod
    def _prune_irrelevant_hybrid_outputs(
        agent_outputs: Dict[str, Any],
        user_message: str,
    ) -> Dict[str, Any]:
        """Remove cross-domain worker output that would lower final answer quality."""
        if not agent_outputs or "researcher" not in agent_outputs:
            return agent_outputs
        public_agents = {key for key in agent_outputs if not str(key).startswith("_")}
        if not ({"weather", "transport"} & public_agents):
            return agent_outputs
        researcher_text = str(agent_outputs.get("researcher") or "")
        if not MultiAgentAssistant._researcher_output_looks_structured_lisboa_aberta_service(researcher_text):
            return agent_outputs
        if MultiAgentAssistant._user_request_needs_municipal_service_listing(user_message):
            transport_text = str(agent_outputs.get("transport") or "")
            normalized_transport = MultiAgentAssistant._fold_context_text(transport_text)
            if re.search(
                r"\b(?:rede fora do ambito confirmado|fora do ambito confirmado|micromobility|gira|"
                r"nao consigo confirmar em tempo real|nao consigo confirmar)\b",
                normalized_transport,
            ):
                return {
                    key: value
                    for key, value in agent_outputs.items()
                    if key != "transport"
                }
            return agent_outputs

        # A structured Lisboa Aberta list is valuable for "find the nearest X";
        # it is not valuable for "route to named X" or "is this named park OK
        # tomorrow?". In those cases the relevant evidence is the transport or
        # weather worker, and the municipal list can be unrelated to the named
        # destination.
        return {
            key: value
            for key, value in agent_outputs.items()
            if key != "researcher"
        }

    @staticmethod
    def _planner_card_zone(name: str, address: str) -> str:
        """Infer broad Lisbon planning zone for lightweight leg decisions."""
        normalized = unicodedata.normalize("NFKD", f"{name} {address}".lower())
        normalized = "".join(char for char in normalized if not unicodedata.combining(char))
        if re.search(r"\b(belem|brasilia|jeronimos|descobrimentos|torre de belem|mosteiro)\b", normalized):
            return "belem"
        if re.search(
            r"\b(parque das nacoes|expo|oriente|oceanario|fil|altice arena|"
            r"rossio dos olivais|alameda dos oceanos|avenida dom joao ii)\b",
            normalized,
        ):
            return "parque_nacoes"
        if re.search(
            r"\b(carmo|chiado|baixa|rossio|se|sé|largo da se|largo da sé|mouraria|correeiros|"
            r"douradores|prata|conceicao|conceição|galerias romanas|roman galleries|figueira|comercio)\b",
            normalized,
        ):
            return "baixa"
        if re.search(r"\b(saldanha|parque eduardo vii|tomas ribeiro|avenida 5 de outubro|picoas)\b", normalized):
            return "avenidas"
        return ""

    @staticmethod
    def _planner_zones_in_text(text: str) -> set[str]:
        """Infer all broad planning zones mentioned in a text block."""
        normalized = unicodedata.normalize("NFKD", str(text or "").lower())
        normalized = "".join(char for char in normalized if not unicodedata.combining(char))
        zone_patterns = {
            "belem": r"\b(belem|brasilia|jeronimos|descobrimentos|torre de belem|mosteiro)\b",
            "parque_nacoes": r"\b(parque das nacoes|expo|oriente|oceanario|fil|altice arena|rossio dos olivais|alameda dos oceanos)\b",
            "baixa": r"\b(carmo|chiado|baixa|rossio|se|largo da se|mouraria|correeiros|douradores|prata|conceicao|figueira|comercio)\b",
            "avenidas": r"\b(saldanha|parque eduardo vii|tomas ribeiro|avenida 5 de outubro|picoas|avenidas novas)\b",
        }
        return {
            zone
            for zone, pattern in zone_patterns.items()
            if re.search(pattern, normalized)
        }

    @staticmethod
    def _planner_zone_label(zone: str, language: str = "pt") -> str:
        """Return a human label for a broad planner zone."""
        is_pt = (language or "").lower().startswith("pt")
        labels = {
            "baixa": "Baixa/Chiado",
            "belem": "Belém",
            "parque_nacoes": "Parque das Nações" if is_pt else "Parque das Nações",
            "avenidas": "Avenidas Novas",
        }
        fallback = "a mesma zona" if is_pt else "the same area"
        return labels.get(str(zone or "").strip(), fallback)

    @staticmethod
    def _planner_walking_leg_summary(
        origin_name: str,
        destination_name: str,
        *,
        zone: str,
        is_pt: bool,
    ) -> str:
        """Build a concise same-zone walking leg for planner evidence."""
        zone_label = {
            "baixa": "Baixa/Chiado",
            "belem": "Belém",
            "parque_nacoes": "Parque das Nações",
            "avenidas": "Avenidas Novas",
        }.get(zone, "a mesma zona" if is_pt else "the same area")
        if is_pt:
            return f"- 🚶 **{origin_name} → {destination_name}:** caminhada curta no eixo {zone_label}; mantém esta ligação a pé se o tempo permitir."
        return f"- 🚶 **{origin_name} → {destination_name}:** short walk in the {zone_label} area; keep this as a walking leg if conditions allow."

    @staticmethod
    def _summarize_carris_plan_leg(
        route_output: str,
        *,
        origin_name: str,
        destination_name: str,
        is_pt: bool,
    ) -> str:
        """Convert a Carris route search result into one planner-safe bullet."""
        text = str(route_output or "")
        if not text or "Direct routes found" not in text:
            return ""

        options, total_routes = MultiAgentAssistant._extract_carris_plan_options(
            text,
            is_pt=is_pt,
        )
        if not options:
            return ""

        joined = ", ".join(options[:3])
        extra_count = max(total_routes - len(options[:3]), 0)
        extra_note = ""
        if extra_count:
            extra_note = (
                f" (+{extra_count} opções diretas)"
                if is_pt
                else f" (+{extra_count} direct options)"
            )
        if is_pt:
            return f"- 🚌 **{origin_name} → {destination_name}:** opções Carris: {joined}{extra_note}. Confirma a partida no momento da deslocação."
        return f"- 🚌 **{origin_name} → {destination_name}:** Carris options: {joined}{extra_note}. Confirm the departure when you are ready to travel."

    @staticmethod
    def _extract_carris_plan_options(route_output: str, *, is_pt: bool) -> tuple[List[str], int]:
        """Extract concrete Carris route options from a route-search response.

        The route tool already ranks direct routes. This parser preserves that
        evidence instead of hard-coding a preferred line, prioritising options
        with confirmed upcoming departures and visible travel durations.
        """
        route_options: List[Dict[str, Any]] = []
        current: Dict[str, Any] | None = None

        def _flush_current() -> None:
            if current and current.get("line"):
                route_options.append(current.copy())

        for raw_line in str(route_output or "").splitlines():
            stripped = raw_line.strip()
            route_match = re.match(
                r"^(?P<line>\d{1,4}[A-Z]?)\s*:\s*(?P<headsign>.+)$",
                stripped,
                flags=re.IGNORECASE,
            )
            if route_match:
                _flush_current()
                current = {
                    "line": route_match.group("line").upper(),
                    "headsign": route_match.group("headsign").strip(),
                    "duration": "",
                    "has_next": False,
                    "has_stop_pair": False,
                    "order": len(route_options),
                }
                continue

            if current is None:
                continue
            if stripped.startswith("Next:"):
                current["has_next"] = True
            if stripped.startswith("Stops:"):
                current["has_stop_pair"] = True
            duration_match = re.search(r"~\s*(?P<duration>\d+\s*min(?:\s*\d+s)?|\d+s)\s*travel", stripped, flags=re.IGNORECASE)
            if duration_match:
                duration = re.sub(r"\s+", " ", duration_match.group("duration")).strip()
                duration = re.sub(r"(\d+)\s*min\b", r"\1 min", duration, flags=re.IGNORECASE)
                duration = re.sub(r"(\d+)\s*s\b", r"\1s", duration, flags=re.IGNORECASE)
                current["duration"] = duration

        _flush_current()
        if not route_options:
            return [], 0
        reported_total_match = re.search(
            r"Direct routes found:\*\*\s*(?P<count>\d+)",
            str(route_output or ""),
            flags=re.IGNORECASE,
        )
        reported_total = int(reported_total_match.group("count")) if reported_total_match else 0

        deduped: List[Dict[str, Any]] = []
        seen_lines: set[str] = set()
        for option in route_options:
            line = str(option.get("line") or "")
            if not line or line in seen_lines:
                continue
            seen_lines.add(line)
            deduped.append(option)

        def _duration_minutes(option: Dict[str, Any]) -> float:
            duration = str(option.get("duration") or "")
            minute_match = re.search(r"(\d+)\s*min", duration)
            if minute_match:
                return float(minute_match.group(1))
            second_match = re.search(r"(\d+)\s*s", duration)
            if second_match:
                return max(float(second_match.group(1)) / 60.0, 0.1)
            return 999.0

        ranked = sorted(
            deduped,
            key=lambda option: (
                0 if option.get("has_next") else 1,
                0 if option.get("duration") else 1,
                _duration_minutes(option),
                int(option.get("order") or 0),
            ),
        )

        formatted: List[str] = []
        for option in ranked[:4]:
            line = str(option.get("line") or "").strip()
            duration = str(option.get("duration") or "").strip()
            has_next = bool(option.get("has_next"))
            duration_text = f" (~{duration})" if duration else ""
            unconfirmed = ""
            if not has_next:
                unconfirmed = (
                    " (sem próxima partida confirmada agora)"
                    if is_pt
                    else " (no upcoming departure confirmed now)"
                )
            formatted.append(f"Carris {line}{duration_text}{unconfirmed}")

        return formatted, max(reported_total, len(deduped))

    @staticmethod
    def _planner_transport_has_route_leg_evidence(text: str) -> bool:
        """Return whether transport context already contains concrete route legs."""
        raw_text = str(text or "")
        normalized = raw_text.lower()
        if not normalized:
            return False
        if (
            re.search(
                r"\b(?:nao\s+(?:consegui\s+)?confirm|não\s+(?:consegui\s+)?confirm|"
                r"not\s+confirmed|unconfirmed|sem\s+lig|no\s+direct|not\s+on\s+metro)\b",
                normalized,
                flags=re.IGNORECASE,
            )
            and not re.search(r"\b(?:carris\s+\d{1,4}[a-z]?|op[cç][oõ]es\s+carris|direct\s+route)\b", normalized)
        ):
            return False
        if re.search(
            r"\b(?:liga[cç][oõ]es entre paragens do roteiro|route legs between itinerary stops|"
            r"op[cç][oõ]es carris|carris\s+\d{1,4}[a-z]?|caminhada curta|short walk)\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            return True
        for raw_line in raw_text.splitlines():
            stripped = raw_line.strip()
            if not re.match(r"^[-*•]\s+", stripped):
                continue
            if ("→" in stripped or "->" in stripped) and re.search(
                r"\b(?:carris|metro|cp|comboio|autocarro|bus|tram|caminhada|walk|linha|line)\b",
                stripped,
                flags=re.IGNORECASE,
            ):
                return True
        return False

    @traceable(
        name="LISBOA Chat",
        run_type="chain",
        tags=["multi-agent", "user-query"],
    )
    def chat(
        self,
        message: str,
        verbose: bool = False,
        on_status_change: Optional[Callable[[str], None]] = None,
        language: str = "en",
    ) -> str:
        """
        Processes a user message using the multi-agent system.

        Uses @traceable decorator to create a single parent trace in LangSmith
        that encompasses all agent and tool calls. The ContextThreadPoolExecutor
        ensures proper context propagation across parallel agent executions.

        INPUT for LangSmith: The 'message' parameter (user question)
        OUTPUT for LangSmith: The returned string (assistant response)


        Flow:
            1. Supervisor analyzes query and decides which agents to call
            2. If no agents needed, Supervisor provides direct response
            3. If agents needed, they are called (in sequence or parallel)
            4. If Planner is in the list, it synthesizes the final response
            5. Otherwise, agent outputs are combined

        Args:
            message (str): User message.
            verbose (bool): If True, prints routing decisions and agent calls.
            on_status_change (func, optional): Callback for UI status updates.

        Returns:
            str: Assistant response.
        """
        # Add to conversation history
        import time

        from langchain_core.messages import HumanMessage

        self._append_user_message(message)
        start_time = time.time()
        run_workers_in_parallel = False
        retry_agents_used: List[str] = []
        final_repair_ran = False
        simple_weather_fact_check: Optional[Dict[str, Any]] = None
        ui_language = language
        previous_user_ctx = self.state.get("user_context") or {}
        previous_language = str(
            previous_user_ctx.get("language")
            or previous_user_ctx.get("detected_language")
            or ""
        ).strip()
        effective_language, requires_bilingual_note, detected_language = resolve_output_language(
            user_query=message,
            ui_default=ui_language,
        )
        if (
            previous_language == "en"
            and effective_language != "en"
            and self._short_english_follow_up_uses_previous_language(message)
        ):
            effective_language = "en"
            requires_bilingual_note = False
            detected_language = "en"

        # Reset tracking for all sub-agents to capture metrics strictly for this request
        self.supervisor.reset_llm_usage_tracking()
        self.qa_agent.reset_llm_usage_tracking()
        for _, agent in self.agents.items():
            agent.reset_llm_usage_tracking()

        # Update user language preference in state
        user_ctx = self.state.get("user_context")
        if user_ctx is None:
            from agent.state import UserContext

            user_ctx = UserContext()
            user_ctx["language"] = effective_language
            user_ctx["ui_language"] = ui_language
            user_ctx["detected_language"] = detected_language or effective_language
            user_ctx["requires_bilingual_note"] = requires_bilingual_note
            self.state["user_context"] = user_ctx
        else:
            user_ctx["language"] = effective_language
            user_ctx["ui_language"] = ui_language
            user_ctx["detected_language"] = detected_language or effective_language
            user_ctx["requires_bilingual_note"] = requires_bilingual_note

        contextual_resolution = self._resolve_contextual_follow_up(message, effective_language)
        contextual_language = str(contextual_resolution.get("language") or "").strip()
        if contextual_language in {"pt", "en"} and contextual_language != effective_language:
            effective_language = contextual_language
            requires_bilingual_note = False
            detected_language = contextual_language
            user_ctx["language"] = effective_language
            user_ctx["detected_language"] = detected_language
            user_ctx["requires_bilingual_note"] = False
        forced_agents_from_context = list(contextual_resolution.get("agents") or [])
        forced_routing_reason = str(contextual_resolution.get("routing_reasoning") or "").strip()
        if contextual_resolution.get("direct_response"):
            return self._finalize_chat_response(
                response=contextual_resolution["direct_response"],
                message=message,
                language=effective_language,
                agents_to_call=[],
                routing_reasoning="Conversation anchor answered directly from the previous response.",
                agent_outputs={},
                direct_response_used=True,
                start_time=start_time,
                workers=[],
                run_workers_in_parallel=False,
                qa_result=None,
                retry_agents_used=[],
                final_repair_ran=False,
                simple_weather_fact_check=None,
            )
        if contextual_resolution.get("clarification"):
            return self._finalize_chat_response(
                response=contextual_resolution["clarification"],
                message=message,
                language=effective_language,
                agents_to_call=[],
                routing_reasoning="Conversation anchor for the requested destination was ambiguous.",
                agent_outputs={},
                direct_response_used=True,
                start_time=start_time,
                workers=[],
                run_workers_in_parallel=False,
                qa_result=None,
                retry_agents_used=[],
                final_repair_ran=False,
                simple_weather_fact_check=None,
            )
        message = contextual_resolution.get("message", message)

        if is_overcomplex_planning_request(message):
            bounded_response = build_bounded_planning_framework(effective_language)
            return self._finalize_chat_response(
                response=bounded_response,
                message=message,
                language=effective_language,
                agents_to_call=[],
                routing_reasoning="Planner request exceeds safe evidence-supported detail; returned bounded framework before worker execution.",
                agent_outputs={},
                direct_response_used=True,
                start_time=start_time,
                workers=[],
                run_workers_in_parallel=False,
                qa_result=None,
                retry_agents_used=[],
                final_repair_ran=False,
                simple_weather_fact_check=None,
            )

        if LANGSMITH_AVAILABLE:
            annotate_current_run(
                metadata={
                    "assistant_mode": "multi-agent",
                    "language": effective_language,
                    "ui_language": ui_language,
                    "detected_language": detected_language or effective_language,
                    "requires_bilingual_note": requires_bilingual_note,
                    "request_source": "user_chat",
                }
            )

        # Notify status: Routing
        if on_status_change:
            status_msg = (
                "🤔 A analisar o pedido..."
                if ui_language == "pt"
                else "🤔 Analyzing request..."
            )
            on_status_change(status_msg)

        # Step 1: Route the query (with conversation history for follow-up awareness)
        # Exclude the current message (last) from history
        history_for_routing = self.state["messages"][:-1] if len(self.state["messages"]) > 1 else None
        if forced_agents_from_context:
            routing = {
                "reasoning": forced_routing_reason or "Conversation context resolved this follow-up into a concrete worker request.",
                "agents": forced_agents_from_context,
                "direct_response": None,
            }
        else:
            try:
                routing = self.supervisor.route(
                    message,
                    language=effective_language,
                    conversation_history=history_for_routing,
                )
            except Exception as exc:
                if verbose:
                    print(f"   [ROUTING] Supervisor failed ({type(exc).__name__}): {exc}")
                routing = self.supervisor._fallback_routing(
                    user_message=message,
                    language=effective_language,
                )
                routing["reasoning"] = (
                    f"Fallback routing due supervisor error ({type(exc).__name__})"
                )
        agents_to_call = routing.get("agents", [])
        direct_response = routing.get("direct_response")
        reasoning = routing.get("reasoning", "")
        planning_follow_up_context = self._build_planning_follow_up_context(message)

        if verbose:
            print("\n   [ROUTING] Supervisor decision:")
            print(f"      Reasoning: {reasoning}")
            print(
                f"      Agents: {agents_to_call if agents_to_call else 'None (direct response)'}"
            )

        # Inject LangSmith metadata and tags based on routing decision
        if LANGSMITH_AVAILABLE:
            query_tags: list[str] = []
            if "weather" in agents_to_call:
                query_tags.append("weather")
            if "transport" in agents_to_call:
                query_tags.append("transport")
            if "researcher" in agents_to_call:
                query_tags.append("research")
            if "planner" in agents_to_call:
                query_tags.append("itinerary")
            if not agents_to_call:
                query_tags.append("direct_response")

            annotate_current_run(
                metadata={
                    "agents_called": agents_to_call,
                    "num_agents": len(agents_to_call),
                    "supervisor_reasoning": reasoning[:200] if reasoning else None,
                    "query_tags": query_tags,
                },
                tags=query_tags,
            )

        data_check_status = (
            "🔎 A consultar dados relevantes..."
            if ui_language == "pt"
            else "🔎 Checking relevant data..."
        )
        retry_data_check_status = (
            "🔄 A consultar dados adicionais..."
            if ui_language == "pt"
            else "🔄 Checking additional data..."
        )
        validation_status = (
            "🔍 A validar a resposta..."
            if ui_language == "pt"
            else "🔍 Validating the answer..."
        )
        final_response_status = (
            "✍️ A preparar a resposta final..."
            if ui_language == "pt"
            else "✍️ Preparing the final response..."
        )
        evidence_response_status = (
            "⚠️ A preparar resposta com os dados confirmados..."
            if ui_language == "pt"
            else "⚠️ Preparing an answer with confirmed data..."
        )

        # Notify status: relevant data lookup selected
        if agents_to_call:
            if on_status_change:
                on_status_change(data_check_status)

        # Step 2: Handle direct response (no agents needed)
        if direct_response and not agents_to_call:
            if verbose:
                print("      Mode: DIRECT RESPONSE (no agents called)")
            return self._finalize_chat_response(
                response=direct_response,
                message=message,
                language=effective_language,
                agents_to_call=[],
                routing_reasoning=reasoning,
                agent_outputs={},
                direct_response_used=True,
                start_time=start_time,
                workers=[],
                run_workers_in_parallel=False,
                qa_result=None,
                retry_agents_used=[],
                final_repair_ran=False,
                simple_weather_fact_check=None,
            )

        # Step 3: Execute agents (Parallelized with LangSmith context propagation)
        agent_outputs = {}
        qa_result = None

        # Identify worker agents (exclude planner which runs last)
        workers = [a for a in agents_to_call if a != "planner" and a in self.agents]

        if workers:
            run_workers_in_parallel = self._should_execute_agent_batch_in_parallel(workers)

            if verbose:
                execution_mode = "PARALLEL" if run_workers_in_parallel else "SEQUENTIAL"
                print(f"      [{execution_mode}] Executing {len(workers)} agents: {workers}")

            if on_status_change:
                on_status_change(data_check_status)

            # Context for agents: language instruction + minimal follow-up context
            # Workers should focus on the CURRENT query, not be biased by history
            agent_context = f"User language: {effective_language}. Respond in {'Portuguese (PT-PT)' if effective_language == 'pt' else 'English'}."
            if planning_follow_up_context:
                agent_outputs["_conversation_context"] = planning_follow_up_context
                agent_context += (
                    "\nPlanning follow-up context:\n"
                    f"{planning_follow_up_context[:1200]}"
                )

            # Only add last user message for follow-up context (e.g., "E amanhã?")
            recent_msgs = self.state.get("messages", [])
            needs_previous_turn_context = self._message_needs_previous_turn_context(message)
            if len(recent_msgs) > 1 and needs_previous_turn_context:
                # Find the previous user message for reference
                for msg in reversed(recent_msgs[:-1]):
                    if isinstance(msg, HumanMessage) and msg.content:
                        agent_context += f"\nPrevious user question (for context only): {msg.content[:150]}"
                        break
                # Find the previous assistant message so workers can resolve
                # anaphoric references such as "the lunch you mentioned",
                # "the restaurant you suggested", or "para o almoço".
                for msg in reversed(recent_msgs[:-1]):
                    if isinstance(msg, AIMessage) and msg.content:
                        snippet = str(msg.content).strip()
                        if snippet:
                            agent_context += (
                                "\nPrevious assistant answer (for anaphora and recall ONLY; "
                                "do not re-answer the previous question):\n"
                                f"{snippet[:1500]}"
                            )
                        break

            if run_workers_in_parallel:
                # Use ContextThreadPoolExecutor to propagate LangSmith tracing context
                with ContextThreadPoolExecutor(max_workers=len(workers)) as executor:
                    # Submit all tasks with timing
                    future_to_agent = {}
                    agent_start_times = {}

                    for agent_name in workers:
                        if verbose:
                            print(f"\n   [AGENT: {agent_name.upper()}] Starting...")
                        agent_start_times[agent_name] = time_module.time()
                        worker_message = (
                            self._researcher_query_with_planning_context(message, planning_follow_up_context)
                            if agent_name == "researcher" and planning_follow_up_context
                            else message
                        )

                        # Pass verbose=verbose to invoke
                        future_to_agent[
                            executor.submit(
                                self.agents[agent_name].invoke,
                                worker_message,
                                agent_context,  # Context with language
                                verbose,        # Verbose flag
                            )
                        ] = agent_name

                    # Collect results as they complete with latency tracking
                    try:
                        for future in as_completed(future_to_agent, timeout=_WORKER_BATCH_TIMEOUT_S):
                            agent_name = future_to_agent[future]
                            agent_latency = time_module.time() - agent_start_times[agent_name]

                            try:
                                output = future.result()
                                agent_outputs[agent_name] = output

                                # Log latency to LangSmith metadata if available
                                if LANGSMITH_AVAILABLE:
                                    annotate_current_run(
                                        metadata={
                                            f"agent_{agent_name}_latency_ms": int(agent_latency * 1000),
                                            f"agent_{agent_name}_output_chars": len(output),
                                        }
                                    )

                                if verbose:
                                    print(
                                        f"   [AGENT: {agent_name.upper()}] Finished ({len(output)} chars, {agent_latency:.2f}s)"
                                    )
                            except Exception as e:
                                error_type = type(e).__name__
                                error_msg = f"Error ({error_type}): {str(e)}"
                                agent_outputs[agent_name] = error_msg
                                if verbose:
                                    print(f"   [AGENT: {agent_name.upper()}] Failed ({error_type}): {str(e)}")
                    except TimeoutError:
                        # Some workers didn't finish within the timeout.
                        # Results for already-completed workers are preserved.
                        timed_out = [
                            name for fut, name in future_to_agent.items()
                            if name not in agent_outputs
                        ]
                        for timed_out_name in timed_out:
                            agent_outputs[timed_out_name] = f"Error (TimeoutError): Worker {timed_out_name} exceeded {_WORKER_BATCH_TIMEOUT_S}s timeout."
                        if verbose:
                            print(f"   [TIMEOUT] Workers did not finish in {_WORKER_BATCH_TIMEOUT_S}s: {timed_out}")
            else:
                for agent_name in workers:
                    if verbose:
                        print(f"\n   [AGENT: {agent_name.upper()}] Starting...")

                    agent_start = time_module.time()

                    try:
                        worker_message = (
                            self._researcher_query_with_planning_context(message, planning_follow_up_context)
                            if agent_name == "researcher" and planning_follow_up_context
                            else message
                        )
                        output = self.agents[agent_name].invoke(
                            worker_message,
                            agent_context,
                            verbose,
                        )
                        agent_outputs[agent_name] = output
                        agent_latency = time_module.time() - agent_start

                        if LANGSMITH_AVAILABLE:
                            annotate_current_run(
                                metadata={
                                    f"agent_{agent_name}_latency_ms": int(agent_latency * 1000),
                                    f"agent_{agent_name}_output_chars": len(output),
                                }
                            )

                        if verbose:
                            print(
                                f"   [AGENT: {agent_name.upper()}] Finished ({len(output)} chars, {agent_latency:.2f}s)"
                            )
                    except Exception as e:
                        error_type = type(e).__name__
                        error_msg = f"Error ({error_type}): {str(e)}"
                        agent_outputs[agent_name] = error_msg
                        if verbose:
                            print(f"   [AGENT: {agent_name.upper()}] Failed ({error_type}): {str(e)}")

        if (
            "planner" in agents_to_call
            and "researcher" in agent_outputs
            and "transport" in agents_to_call
        ):
            self._maybe_enrich_planner_transport_context(
                user_message=message,
                language=effective_language,
                agent_outputs=agent_outputs,
                verbose=verbose,
            )

        # Step 4: QA Validation (single retry if incomplete)
        skip_qa_for_simple_weather = (
            workers == ["weather"]
            and "planner" not in agents_to_call
            and (
                self.agents["weather"]._is_current_weather_query(message)
                or self.agents["weather"]._is_simple_forecast_query(message)
            )
        )
        skip_qa_for_structured_service = (
            workers == ["researcher"]
            and "planner" not in agents_to_call
            and self._user_request_needs_municipal_service_listing(message)
            and self._researcher_output_looks_structured_lisboa_aberta_service(
                str(agent_outputs.get("researcher") or "")
            )
        )

        if skip_qa_for_simple_weather:
            if verbose:
                print("\n   [QA] Skipped for simple deterministic weather query")

            simple_weather_fact_check = self._run_lightweight_weather_fact_check(
                user_query=message,
                weather_output=str(agent_outputs.get("weather", "")),
                language=effective_language,
                verbose=verbose,
            )
            if simple_weather_fact_check.get("requires_full_qa"):
                skip_qa_for_simple_weather = False
                if verbose:
                    print("   [QA] Escalating simple weather query to full QA after deterministic fact-check")
            else:
                simple_weather_disclaimers = self._sanitize_qa_disclaimers(
                    simple_weather_fact_check.get("disclaimers", []),
                    effective_language,
                )
                if simple_weather_disclaimers:
                    agent_outputs["_qa_disclaimers"] = simple_weather_disclaimers
                    if verbose:
                        for disclaimer in simple_weather_disclaimers:
                            print(f"   [QA] Warning: {disclaimer}")

                qa_result = {
                    "complete": True,
                    "missing_data": [],
                    "required_agents": [],
                    "reasoning": "Fast deterministic weather fact-check completed.",
                    "disclaimers": simple_weather_disclaimers,
                    "critical_issues": [],
                    "repairable_agents": [],
                    "needs_repair": False,
                    "fact_check": simple_weather_fact_check.get("fact_check", {}),
                }

        if skip_qa_for_structured_service:
            if verbose:
                print("\n   [QA] Skipped for structured Lisboa Aberta service lookup")
            qa_result = {
                "complete": True,
                "missing_data": [],
                "required_agents": [],
                "reasoning": "Structured Lisboa Aberta municipal-service output preserved without generative QA rewrite.",
                "disclaimers": [],
                "critical_issues": [],
                "repairable_agents": [],
                "needs_repair": False,
                "fact_check": {
                    "disclaimers": [],
                    "critical_issues": [],
                    "repairable_agents": [],
                    "per_agent": {},
                },
            }

        if agent_outputs and len(workers) > 0 and not skip_qa_for_simple_weather and not skip_qa_for_structured_service:
            if verbose:
                print("\n   [QA] Validating completeness...")

            if on_status_change:
                on_status_change(validation_status)

            messages_list = self.state.get("messages", [])
            qa_history = (
                [
                    f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: "
                    f"{m.content[:_QA_MSG_PREVIEW_LEN]}"
                    for m in messages_list[:-1][-_QA_HISTORY_WINDOW:]
                ]
                if len(messages_list) > 1 else None
            )
            try:
                qa_result = self.qa_agent.validate(
                    user_query=message,
                    agent_outputs=agent_outputs,
                    agents_called=self._dedupe_preserve_order(
                        workers + (["planner"] if "planner" in agents_to_call else [])
                    ),
                    language=effective_language,
                    user_context=self.state.get("user_context"),
                    conversation_history=qa_history,
                )
            except Exception as e:
                if verbose:
                    print(f"   [QA] Validation failed; continuing with worker outputs: {e}")
                qa_result = {
                    "complete": True,
                    "missing_data": [],
                    "required_agents": [],
                    "reasoning": "QA validation was unavailable; preserved worker outputs.",
                    "disclaimers": [],
                    "critical_issues": [],
                    "repairable_agents": [],
                    "needs_repair": False,
                    "fact_check": {
                        "disclaimers": [],
                        "critical_issues": [],
                        "repairable_agents": [],
                        "per_agent": {},
                    },
                }

            if self._qa_gap_is_generic_service_area_route(
                message,
                str(agent_outputs.get("transport") or ""),
                qa_result,
            ):
                qa_result["complete"] = True
                qa_result["missing_data"] = []
                qa_result["required_agents"] = []
                qa_result["repairable_agents"] = []
                qa_result["needs_repair"] = False
                qa_result["reasoning"] = (
                    "Generic service-area route is complete because the answer "
                    "uses the named area as a disclosed destination reference."
                )

            if verbose:
                print(f"   [QA] Complete: {qa_result['complete']}")
                if qa_result['missing_data']:
                    print(f"   [QA] Missing: {qa_result['missing_data']}")
                if qa_result['required_agents']:
                    print(f"   [QA] Need agents: {qa_result['required_agents']}")
                if qa_result.get('fact_check'):
                    fc = qa_result['fact_check']
                    if fc.get('disclaimers'):
                        for d in fc['disclaimers']:
                            print(f"   [QA FACT-CHECK] {d}")

            # Single retry: call missing agents or re-run workers with deterministic QA repair feedback
            retry_agents = self._dedupe_preserve_order(
                [
                    a for a in qa_result.get("required_agents", [])
                    if a in self.agents and a != "planner"
                ]
                + [
                    a for a in qa_result.get("repairable_agents", [])
                    if a in self.agents and a != "planner"
                ]
            )
            retry_agents = self._filter_planner_qa_retry_agents(
                retry_agents,
                user_message=message,
                agents_to_call=agents_to_call,
                workers=workers,
                agent_outputs=agent_outputs,
                qa_result=qa_result,
            )

            qa_has_retryable_issue = bool(
                qa_result.get("missing_data")
                or qa_result.get("required_agents")
                or self._should_block_planner_publication(qa_result)
            )
            if retry_agents and qa_has_retryable_issue:
                retry_agents_used = list(retry_agents)

                if retry_agents:
                    if verbose:
                        print(f"   [QA RETRY] Calling additional agents: {retry_agents}")

                    if on_status_change:
                        on_status_change(retry_data_check_status)

                    run_retry_in_parallel = self._should_execute_agent_batch_in_parallel(retry_agents)

                    if run_retry_in_parallel:
                        # Execute retry agents in parallel
                        with ContextThreadPoolExecutor(max_workers=len(retry_agents)) as executor:
                            retry_futures = {}
                            for agent_name in retry_agents:
                                # Use targeted feedback context when the agent is being retried after QA
                                ctx = self._build_qa_retry_context(
                                    base_context=agent_context,
                                    qa_result=qa_result,
                                    agent_name=agent_name,
                                )
                                retry_futures[
                                    executor.submit(
                                        self.agents[agent_name].invoke,
                                        message,
                                        ctx,
                                        verbose,
                                    )
                                ] = agent_name

                            for future in as_completed(retry_futures, timeout=_WORKER_BATCH_TIMEOUT_S):
                                agent_name = retry_futures[future]
                                try:
                                    output = future.result()
                                    if agent_name in workers and agent_name in agent_outputs:
                                        # Overwrite with the newer, more complete response from QA retry
                                        agent_outputs[agent_name] = output
                                    else:
                                        agent_outputs[agent_name] = output

                                    if verbose:
                                        print(f"   [QA RETRY: {agent_name.upper()}] Finished ({len(output)} chars)")
                                except Exception as e:
                                    if agent_name not in workers:
                                        agent_outputs[agent_name] = f"Error: {str(e)}"
                                    if verbose:
                                        print(f"   [QA RETRY: {agent_name.upper()}] Failed: {str(e)}")
                    else:
                        for agent_name in retry_agents:
                            ctx = self._build_qa_retry_context(
                                base_context=agent_context,
                                qa_result=qa_result,
                                agent_name=agent_name,
                            )
                            try:
                                output = self.agents[agent_name].invoke(
                                    message,
                                    ctx,
                                    verbose,
                                )
                                agent_outputs[agent_name] = output

                                if verbose:
                                    print(f"   [QA RETRY: {agent_name.upper()}] Finished ({len(output)} chars)")
                            except Exception as e:
                                if agent_name not in workers:
                                    agent_outputs[agent_name] = f"Error: {str(e)}"
                                if verbose:
                                    print(f"   [QA RETRY: {agent_name.upper()}] Failed: {str(e)}")

                    # Post-retry re-validation (lightweight, no further retries)
                    if verbose:
                        print("   [QA] Post-retry re-validation...")

                    try:
                        qa_result_2 = self.qa_agent.validate(
                            user_query=message,
                            agent_outputs=agent_outputs,
                            agents_called=self._dedupe_preserve_order(
                                workers
                                + retry_agents
                                + (["planner"] if "planner" in agents_to_call else [])
                            ),
                            language=effective_language,
                            user_context=self.state.get("user_context"),
                            conversation_history=qa_history,
                        )
                    except Exception as e:
                        if verbose:
                            print(f"   [QA] Post-retry validation failed; keeping previous QA result: {e}")
                        qa_result_2 = qa_result

                    if verbose:
                        print(f"   [QA] Post-retry complete: {qa_result_2['complete']}")

                    # Merge disclaimers from both QA passes
                    all_disclaimers = list(set(
                        qa_result.get("disclaimers", []) +
                        qa_result_2.get("disclaimers", [])
                    ))
                    # Merge fact_check warnings
                    fc1 = qa_result.get("fact_check", {})
                    fc2 = qa_result_2.get("fact_check", {})
                    merged_fc_disclaimers = self._dedupe_preserve_order(
                        list(fc1.get("disclaimers", [])) + list(fc2.get("disclaimers", []))
                    )
                    current_fc_critical = self._dedupe_preserve_order(
                        list(fc2.get("critical_issues", []))
                    )
                    current_repairable_agents = self._dedupe_preserve_order(
                        list(fc2.get("repairable_agents", []))
                    )
                    current_per_agent = fc2.get("per_agent", {}) if isinstance(fc2, dict) else {}

                    qa_result = qa_result_2
                    qa_result["disclaimers"] = all_disclaimers
                    qa_result["critical_issues"] = self._dedupe_preserve_order(
                        list(qa_result.get("critical_issues", []))
                    )
                    qa_result["repairable_agents"] = current_repairable_agents
                    qa_result["needs_repair"] = bool(
                        qa_result["critical_issues"] or qa_result.get("missing_data")
                    )
                    if qa_result.get("fact_check"):
                        qa_result["fact_check"]["disclaimers"] = merged_fc_disclaimers
                        qa_result["fact_check"]["critical_issues"] = current_fc_critical
                        qa_result["fact_check"]["repairable_agents"] = current_repairable_agents
                        qa_result["fact_check"]["per_agent"] = current_per_agent

            # Pass QA disclaimers as context for synthesis (internal key, filtered from output)
            all_qa_warnings = qa_result.get("disclaimers", [])
            fc_warns = qa_result.get("fact_check", {})
            if isinstance(fc_warns, dict):
                all_qa_warnings = self._dedupe_preserve_order(
                    list(all_qa_warnings) + list(fc_warns.get("disclaimers", []))
                )
            all_qa_warnings = self._sanitize_qa_disclaimers(all_qa_warnings, effective_language)
            if all_qa_warnings:
                agent_outputs["_qa_disclaimers"] = all_qa_warnings
                if verbose:
                    for d in all_qa_warnings:
                        print(f"   [QA] Warning: {d}")

        # Step 5: Filter out failed agent outputs (errors must never reach user)
        clean_outputs = {}
        for aname, aoutput in agent_outputs.items():
            if isinstance(aoutput, str) and re.match(r"^Error(?:\s*\(|:)", aoutput):
                if verbose:
                    print(f"   [FILTER] Removing failed agent output: {aname}")
                continue
            clean_outputs[aname] = aoutput
        agent_outputs = clean_outputs
        agent_outputs = self._prune_irrelevant_hybrid_outputs(agent_outputs, message)
        if "planner" in agents_to_call:
            meal_research = self._planner_meal_research_supplement(
                message,
                str(agent_outputs.get("_conversation_context", "") or ""),
                str(agent_outputs.get("researcher", "") or ""),
                effective_language,
            )
            if meal_research:
                existing_research = str(agent_outputs.get("researcher", "") or "").strip()
                agent_outputs["researcher"] = (
                    f"{existing_research}\n\n{meal_research}".strip()
                    if existing_research
                    else meal_research
                )
        if (
            "researcher" in agent_outputs
            and "_events_context" not in agent_outputs
            and self._researcher_output_looks_event_context(str(agent_outputs.get("researcher") or ""))
        ):
            agent_outputs["_events_context"] = str(agent_outputs.get("researcher") or "")

        planner_requested = "planner" in agents_to_call
        planner_blocked = planner_requested and self._should_block_planner_publication(
            qa_result
        )
        preserve_direct_researcher_answer = (
            planner_requested
            and not planner_blocked
            and self._should_preserve_direct_researcher_answer(agent_outputs)
        )
        preserve_weather_limitation_answer = (
            planner_requested
            and "weather" in agent_outputs
            and re.search(
                r"(?i)(forecast range only extends|next 5 days|horizonte.*5 dias|previs[aã]o.*5 dias|n[aã]o consigo confirmar.*previs[aã]o)",
                str(agent_outputs.get("weather", "")),
            )
            and re.search(r"(?i)\b(weather|forecast|tempo|previs[aã]o|confirm)\b", message)
        )
        response_agents_to_call = list(agents_to_call)
        planner_executed = False
        planner_fallback_used = False

        if planner_blocked:
            if verbose:
                print(
                    "\n   [QA] Blocking planner synthesis because evidence-supported data is still incomplete"
                )
            if on_status_change:
                on_status_change(evidence_response_status)

        if preserve_direct_researcher_answer:
            response_agents_to_call = [
                agent_name for agent_name in agents_to_call if agent_name != "planner"
            ]

        # Step 6: If Planner was requested and QA did not block publication,
        # synthesize the final response. Otherwise, fall back to the combined
        # worker outputs so caveats remain visible instead of publishing a
        # confident itinerary over incomplete evidence.
        if preserve_weather_limitation_answer:
            response_agents_to_call = ["weather"]
            response = str(agent_outputs.get("weather", "")).strip()
        elif planner_requested and planner_blocked:
            from agent.agents.planner_agent import _build_structured_plan_fallback

            response = _build_structured_plan_fallback(
                user_message=message,
                language=effective_language,
                weather_data=str(agent_outputs.get("weather", "") or ""),
                transport_data=str(agent_outputs.get("transport", "") or ""),
                places_data=str(agent_outputs.get("researcher", "") or ""),
                events_data=str(agent_outputs.get("events", "") or agent_outputs.get("_events_context", "") or ""),
                qa_disclaimers=getattr(qa_result, "disclaimers", None),
                conversation_context=str(agent_outputs.get("_conversation_context", "") or ""),
            )
            planner_executed = True
            planner_fallback_used = True
        elif planner_requested and not planner_blocked and not preserve_direct_researcher_answer:
            if verbose:
                print(
                    f"\n   [AGENT: PLANNER] Synthesizing from {list(agent_outputs.keys())}..."
                )

            if on_status_change:
                on_status_change(final_response_status)

            try:
                response = self.agents["planner"].synthesize(message, agent_outputs)
            except Exception as e:
                if verbose:
                    print(f"   [PLANNER] Planner synthesis failed ({type(e).__name__}): {e}")
                from agent.agents.planner_agent import _build_structured_plan_fallback

                response = _build_structured_plan_fallback(
                    user_message=message,
                    language=effective_language,
                    weather_data=str(agent_outputs.get("weather", "") or ""),
                    transport_data=str(agent_outputs.get("transport", "") or ""),
                    places_data=str(agent_outputs.get("researcher", "") or ""),
                    events_data=str(agent_outputs.get("events", "") or agent_outputs.get("_events_context", "") or ""),
                    qa_disclaimers=getattr(qa_result, "disclaimers", None),
                    conversation_context=str(agent_outputs.get("_conversation_context", "") or ""),
                )
                planner_executed = True
                planner_fallback_used = True
            planner_executed = True
        elif agent_outputs:
            # Combine agent outputs if no planner
            response = self._combine_outputs(agent_outputs, language=effective_language)
        else:
            # Fallback: Use researcher for general queries
            if verbose:
                print("\n   [FALLBACK] Using researcher agent")
            response = self.agents["researcher"].invoke(
                message,
                context=f"User language: {effective_language}",
                verbose=verbose
            )

        should_run_final_repair = self._should_run_final_qa_repair(qa_result)
        if should_run_final_repair and self.supervisor._negates_itinerary_request(message):
            should_run_final_repair = False
        if should_run_final_repair and re.search(
            r"\b(?:Ambiguidade|Ambiguity|Preciso de confirmar o local|Location needs confirmation)\b",
            str(response or ""),
            flags=re.IGNORECASE,
        ):
            should_run_final_repair = False
        if should_run_final_repair and (
            response_agents_to_call == ["transport"] or "planner" in set(response_agents_to_call or [])
        ):
            fact_check = qa_result.get("fact_check", {}) if isinstance(qa_result, dict) else {}
            critical_issues = []
            if isinstance(qa_result, dict):
                critical_issues.extend(qa_result.get("critical_issues") or [])
            if isinstance(fact_check, dict):
                critical_issues.extend(fact_check.get("critical_issues") or [])
            should_run_final_repair = bool(critical_issues)
        if should_run_final_repair and response_agents_to_call == ["researcher"] and isinstance(qa_result, dict):
            if self.supervisor._negates_itinerary_request(message):
                should_run_final_repair = False
            else:
                fact_check = qa_result.get("fact_check", {})
                critical_issues = list(qa_result.get("critical_issues") or [])
                if isinstance(fact_check, dict):
                    critical_issues.extend(fact_check.get("critical_issues") or [])
                normalized_issues = " ".join(str(issue).lower() for issue in critical_issues)
                if (
                    qa_result.get("complete") is True
                    and not qa_result.get("missing_data")
                    and not qa_result.get("needs_repair")
                    and (
                        not critical_issues
                        or all(
                            marker in normalized_issues
                            for marker in ("unverified", "domains")
                        )
                    )
                ):
                    should_run_final_repair = False

        if not planner_fallback_used and should_run_final_repair:
            if verbose:
                print("\n   [QA] Running final repair pass on the drafted response...")
            final_repair_ran = True
            response = self.qa_agent.repair_final_response(
                user_query=message,
                draft_response=response,
                agent_outputs=agent_outputs,
                qa_result=qa_result,
                language=effective_language,
            )

        if planner_executed:
            from agent.agents.planner_agent import (
                _build_card_based_itinerary_fallback,
                _build_structured_plan_fallback,
                _ensure_requested_origin_target_in_transport_section,
                _ensure_multi_day_response_quality,
                _planner_response_has_markdown_contract_defects,
                _planner_response_has_transport_quality_defects,
                _planner_response_has_local_area_drift,
                _planner_response_loses_transport_leg_evidence,
                _planner_response_missing_requested_movement,
                _planner_response_missing_requested_food_stop,
                _planner_response_missing_requested_stops,
                _planner_response_has_unrequested_sequence_stops,
                _planner_response_violates_requested_start,
                _planner_response_matches_schema,
                _repair_meal_locality_in_response,
            )
            from agent.utils.response_formatter import finalize_worker_response

            response = _repair_meal_locality_in_response(
                response,
                user_message=message,
                places_data=str(agent_outputs.get("researcher", "") or ""),
                language=effective_language,
            )
            response = _ensure_multi_day_response_quality(
                response,
                user_message=message,
                language=effective_language,
                weather_data=str(agent_outputs.get("weather", "") or ""),
                transport_data=str(agent_outputs.get("transport", "") or ""),
                places_data=str(agent_outputs.get("researcher", "") or ""),
                events_data=str(agent_outputs.get("events", "") or agent_outputs.get("_events_context", "") or ""),
                qa_disclaimers=agent_outputs.get("_qa_disclaimers"),
                conversation_context=str(agent_outputs.get("_conversation_context", "") or ""),
            )
            if not planner_fallback_used and (
                _planner_response_has_markdown_contract_defects(response)
                or _planner_response_has_transport_quality_defects(response, message, str(agent_outputs.get("transport", "") or ""))
                or _planner_response_has_local_area_drift(response, message)
                or _planner_response_loses_transport_leg_evidence(response, str(agent_outputs.get("transport", "") or ""))
                or _planner_response_missing_requested_movement(response, message, str(agent_outputs.get("transport", "") or ""))
                or _planner_response_missing_requested_food_stop(response, message)
                or _planner_response_violates_requested_start(response, message)
                or _planner_response_has_unrequested_sequence_stops(response, message)
                or _planner_response_missing_requested_stops(
                    response,
                    message,
                    "\n".join([
                        str(agent_outputs.get("_conversation_context", "") or ""),
                        str(agent_outputs.get("researcher", "") or ""),
                        str(agent_outputs.get("events", "") or agent_outputs.get("_events_context", "") or ""),
                    ]),
                )
            ):
                response = _build_card_based_itinerary_fallback(
                    user_message=message,
                    language=effective_language,
                    weather_data=str(agent_outputs.get("weather", "") or ""),
                    transport_data=str(agent_outputs.get("transport", "") or ""),
                    places_data=str(agent_outputs.get("researcher", "") or ""),
                    events_data=str(agent_outputs.get("events", "") or agent_outputs.get("_events_context", "") or ""),
                    qa_disclaimers=agent_outputs.get("_qa_disclaimers"),
                    conversation_context=str(agent_outputs.get("_conversation_context", "") or ""),
                ) or _build_structured_plan_fallback(
                    user_message=message,
                    language=effective_language,
                    weather_data=str(agent_outputs.get("weather", "") or ""),
                    transport_data=str(agent_outputs.get("transport", "") or ""),
                    places_data=str(agent_outputs.get("researcher", "") or ""),
                    events_data=str(agent_outputs.get("events", "") or agent_outputs.get("_events_context", "") or ""),
                    qa_disclaimers=agent_outputs.get("_qa_disclaimers"),
                    conversation_context=str(agent_outputs.get("_conversation_context", "") or ""),
                )
                planner_fallback_used = True
            if not planner_fallback_used and not _planner_response_matches_schema(response):
                rebuilt_plan = _build_card_based_itinerary_fallback(
                    user_message=message,
                    language=effective_language,
                    weather_data=str(agent_outputs.get("weather", "") or ""),
                    transport_data=str(agent_outputs.get("transport", "") or ""),
                    places_data=str(agent_outputs.get("researcher", "") or ""),
                    events_data=str(agent_outputs.get("events", "") or agent_outputs.get("_events_context", "") or ""),
                    qa_disclaimers=agent_outputs.get("_qa_disclaimers"),
                    conversation_context=str(agent_outputs.get("_conversation_context", "") or ""),
                ) or _build_structured_plan_fallback(
                    user_message=message,
                    language=effective_language,
                    weather_data=str(agent_outputs.get("weather", "") or ""),
                    transport_data=str(agent_outputs.get("transport", "") or ""),
                    places_data=str(agent_outputs.get("researcher", "") or ""),
                    events_data=str(agent_outputs.get("events", "") or agent_outputs.get("_events_context", "") or ""),
                    qa_disclaimers=agent_outputs.get("_qa_disclaimers"),
                    conversation_context=str(agent_outputs.get("_conversation_context", "") or ""),
                )
                response = rebuilt_plan or finalize_worker_response(response, "planner", message, effective_language)
                planner_fallback_used = bool(rebuilt_plan)
            response = _ensure_requested_origin_target_in_transport_section(
                _repair_meal_locality_in_response(
                    response,
                    user_message=message,
                    places_data=str(agent_outputs.get("researcher", "") or ""),
                    language=effective_language,
                ),
                message,
                effective_language,
                str(agent_outputs.get("transport", "") or ""),
            )
            response = self._rebuild_planner_scope_fallback_source_line(
                response,
                effective_language,
                response_agents_to_call,
            )
        elif len(response_agents_to_call) == 1 and response_agents_to_call[0] in {"researcher", "transport"}:
            from agent.utils.response_formatter import finalize_worker_response

            response = finalize_worker_response(
                response,
                response_agents_to_call[0],
                message,
                effective_language,
            )
            if response_agents_to_call[0] == "transport":
                response = self._rebuild_single_transport_source_line(
                    response,
                    effective_language,
                    response_agents_to_call,
                )
        elif len(response_agents_to_call) == 1 and response_agents_to_call[0] == "weather":
            from agent.utils.response_formatter import finalize_worker_response

            response = finalize_worker_response(
                response,
                "weather",
                message,
                effective_language,
            )
        elif len(response_agents_to_call) > 1:
            from agent.utils.response_formatter import (
                canonicalize_local_information_terms,
                infer_researcher_source_kind,
                strip_placeholder_field_lines,
            )

            response = canonicalize_local_information_terms(response, effective_language)
            if (
                "researcher" in response_agents_to_call
                and infer_researcher_source_kind(user_query=message, text=response) != "events"
            ):
                response = strip_placeholder_field_lines(response)

        response = self._move_location_ambiguity_preamble_first(
            response=response,
            user_query=message,
            language=effective_language,
        )

        try:
            return self._finalize_chat_response(
                response=response,
                message=message,
                language=effective_language,
                agents_to_call=response_agents_to_call,
                routing_reasoning=reasoning,
                agent_outputs=agent_outputs,
                direct_response_used=False,
                start_time=start_time,
                workers=workers,
                run_workers_in_parallel=run_workers_in_parallel,
                qa_result=qa_result,
                retry_agents_used=retry_agents_used,
                final_repair_ran=final_repair_ran,
                simple_weather_fact_check=simple_weather_fact_check,
            )
        except Exception as exc:
            if verbose:
                print(f"   [FORMAT] Finalization failed ({type(exc).__name__}): {exc}")
            try:
                emergency_response = final_post_qa_guard(
                    final_visual_pass(clean_response(response)),
                    language=effective_language,
                )
            except Exception:
                emergency_response = ""
            if (
                emergency_response
                and len(emergency_response.strip()) >= 80
                and "Operational Notice" not in emergency_response
                and "Dados Não Confirmados" not in emergency_response
            ):
                self._append_assistant_message(emergency_response)
                self.last_execution_summary = self._collect_execution_summary(
                    user_request=message,
                    routing_reasoning=reasoning,
                    agents_to_call=response_agents_to_call,
                    agent_outputs=agent_outputs,
                    direct_response_used=False,
                    workers=workers,
                    run_workers_in_parallel=run_workers_in_parallel,
                    qa_result=qa_result,
                    retry_agents_used=retry_agents_used,
                    final_repair_ran=final_repair_ran,
                    simple_weather_fact_check=simple_weather_fact_check,
                    elapsed_time=time_module.time() - start_time,
                )
                if Config.SHOW_MARKDOWN_RESPONSE_IN_TERMINAL:
                    self._print_execution_summary(self.last_execution_summary)
                return emergency_response
            fallback_response = self._build_orchestration_failure_fallback(
                message=message,
                language=effective_language,
                attempted_agents=response_agents_to_call,
            )
            self._append_assistant_message(fallback_response)
            self.last_execution_summary = self._collect_execution_summary(
                user_request=message,
                routing_reasoning=reasoning,
                agents_to_call=response_agents_to_call,
                agent_outputs=agent_outputs,
                direct_response_used=False,
                workers=workers,
                run_workers_in_parallel=run_workers_in_parallel,
                qa_result=qa_result,
                retry_agents_used=retry_agents_used,
                final_repair_ran=final_repair_ran,
                simple_weather_fact_check=simple_weather_fact_check,
                elapsed_time=time_module.time() - start_time,
            )
            if Config.SHOW_MARKDOWN_RESPONSE_IN_TERMINAL:
                self._print_execution_summary(self.last_execution_summary)
            return fallback_response

    @staticmethod
    def _is_unsupported_transport_scope_response(response: str) -> bool:
        """Return whether the answer is an unsupported transport-scope limitation."""
        if not response:
            return False
        normalized = unicodedata.normalize("NFKD", response).encode("ascii", "ignore").decode("ascii").lower()
        return any(
            marker in normalized
            for marker in (
                "rede fora do ambito confirmado",
                "mobilidade fora do ambito confirmado",
                "comboios cp fora do ambito",
                "fora do ambito aml",
                "outside confirmed scope",
                "outside the confirmed scope",
                "mobility outside confirmed scope",
                "cp trains outside aml",
            )
        )

    @staticmethod
    def _move_location_ambiguity_preamble_first(
        response: str,
        user_query: str,
        language: str,
    ) -> str:
        """Move bare ambiguous-location warnings before generic response headings."""
        if not response or not user_query:
            return response

        normalized_response = unicodedata.normalize("NFKD", response).encode("ascii", "ignore").decode("ascii").lower()
        normalized_query = unicodedata.normalize("NFKD", user_query).encode("ascii", "ignore").decode("ascii").lower()
        is_weather_request = bool(
            re.search(
                r"\b(?:tempo|weather|meteo|previsao|forecast|aviso|avisos|alerta|alertas|warning|warnings|"
                r"chuva|rain|vento|wind|temperatura|temperature|trovoada|thunderstorm)\b",
                normalized_query,
                flags=re.IGNORECASE,
            )
        )
        explicit_transport_intent = bool(
            re.search(
                r"\b(?:metro|autocarro|autocarros|bus|buses|comboio|comboios|train|trains|carris|cp|"
                r"transportes?|public transport|rota|route|percurso|trajeto|apanhar|apanho|take|"
                r"como\s+(?:vou|chego|ir|posso\s+ir)|how\s+(?:do|can)\s+i\s+(?:get|go)|"
                r"go from|from .+ to )\b",
                normalized_query,
                flags=re.IGNORECASE,
            )
        )
        if is_weather_request and not explicit_transport_intent:
            return response

        is_movement_request = bool(
            re.search(
                r"\b(?:como\s+(?:vou|chego|ir)|leva-me|route|rota|percurso|trajeto|trajeto|"
                r"apanhar|apanho|take|go from|from .+ to |de .+ para |entre .+ e )\b",
                normalized_query,
                flags=re.IGNORECASE,
            )
        )
        if (
            is_movement_request
            and re.search(r"\b(?:eventos?|events?|concertos?|concerts?|teatro|theatre|dance|danca|dança|exposicoes?|exhibitions?)\b", normalized_query)
            and not explicit_transport_intent
        ):
            is_movement_request = False
        if not is_movement_request and re.search(
            r"\b(?:alerta|alertas|aviso|avisos|estado|perturbacao|perturbacoes|status)\b",
            normalized_query,
            flags=re.IGNORECASE,
        ):
            return response
        if not is_movement_request:
            return response
        if re.search(
            r"\b(?:referencia\s+e\s+ambigua|reference\s+is\s+ambiguous|"
            r"preciso\s+de\s+confirmar\s+o\s+restaurante|restaurant\s+needs\s+confirmation)\b",
            normalized_response,
            flags=re.IGNORECASE,
        ):
            return response
        scope_markers = (
            "fora do ambito confirmado",
            "fora do ambito aml",
            "fora do ambito de mobilidade",
            "fora do ambito do lisboa",
            "outside confirmed scope",
            "outside the confirmed scope",
            "outside lisboa",
            "outside lisboa's mobility scope",
            "rede fora do ambito",
            "mobilidade fora do ambito",
            "comboios cp fora do ambito",
            "nao consigo confirmar",
            "nao consigo validar uma rota",
            "i can't verify",
            "i cannot verify",
            "i cannot validate a route",
            "i could not confirm",
        )
        if MultiAgentAssistant._is_unsupported_transport_scope_response(response) or any(
            marker in normalized_response for marker in scope_markers
        ):
            return response

        try:
            from agent.agents.transport_agent import _extract_route_endpoints
            from tools.location_resolver import build_location_ambiguity_preamble

            endpoints = _extract_route_endpoints(user_query)
            if not endpoints:
                return response
            preamble = build_location_ambiguity_preamble(
                endpoints[0],
                endpoints[1],
                language=language,
            )
        except Exception:
            return response

        if not preamble:
            return response

        stripped_response = response.lstrip()
        if stripped_response.startswith("⚠️") and (
            "Ambiguidade" in stripped_response[:160]
            or "Ambiguity" in stripped_response[:160]
            or "Preciso de confirmar" in stripped_response[:180]
            or "I need to confirm" in stripped_response[:180]
        ):
            return response

        cleaned_lines: List[str] = []
        skipping_ambiguity = False
        for line in response.splitlines():
            stripped_line = line.strip()
            if not skipping_ambiguity and (
                "Ambiguidade" in stripped_line or "Ambiguity" in stripped_line
            ):
                skipping_ambiguity = True
                continue

            if skipping_ambiguity:
                if not stripped_line:
                    skipping_ambiguity = False
                    continue
                if stripped_line.startswith(("A)", "B)", "- A)", "- B)")) or "Assumo" in stripped_line or "continu" in stripped_line.lower():
                    continue
                skipping_ambiguity = False

            lowered_line = stripped_line.lower()
            if stripped_line in {"###", "---"}:
                continue
            if stripped_line.startswith(("A)", "B)", "- A)", "- B)")):
                continue
            if lowered_line.startswith("- se não for") and "destino" in lowered_line:
                continue
            if "se o destino pretendido for a ilha da madeira" in lowered_line:
                continue

            cleaned_lines.append(line)

        body = "\n".join(cleaned_lines).strip()
        return f"{preamble}\n\n{body}".strip()

    @staticmethod
    def _extract_structured_section_parts(text: str) -> tuple[str, List[str], Optional[str]]:
        """Removes per-section source lines while collecting links and timestamps for a combined footer."""
        source_line_re = re.compile(
            r"^(?:[-*•]\s*)?(?:📌\s*)?(?:\*\*)?(?:Fontes?|Sources?)(?:\*\*)?:.*$",
            re.IGNORECASE,
        )
        timestamp_re = re.compile(r"(?:Atualizado|Updated):\s*(\d{2}:\d{2})", re.IGNORECASE)

        links: List[str] = []
        timestamps: List[str] = []
        body_lines: List[str] = []

        for line in (text or "").splitlines():
            stripped = line.strip()
            if source_line_re.match(stripped):
                links.extend(re.findall(r"\[[^\]]+\]\([^)]+\)", stripped))
                timestamps.extend(timestamp_re.findall(stripped))
                continue
            body_lines.append(line)

        while body_lines and not body_lines[-1].strip():
            body_lines.pop()

        timestamp = max(timestamps) if timestamps else None
        deduped_links: List[str] = []
        for link in links:
            if link not in deduped_links:
                deduped_links.append(link)

        return "\n".join(body_lines).strip(), deduped_links, timestamp

    @staticmethod
    def _strip_cross_domain_hybrid_lines(agent_name: str, body: str, available_agents: Set[str]) -> str:
        """Remove lines where one worker repeats another worker's domain in hybrid output."""
        if not body:
            return body

        weather_has_transport_leak = agent_name == "weather" and "transport" in available_agents
        researcher_has_weather_transport_leak = agent_name == "researcher" and bool(
            {"weather", "transport"} & available_agents
        )
        if not weather_has_transport_leak and not researcher_has_weather_transport_leak:
            return body

        transport_markers = [
            "transport",
            "transportes",
            "public transport",
            "metro",
            "autocarro",
            "autocarros",
            "comboio",
            "comboio +",
            "carris",
            "cp",
            "route",
            "rota",
            "trajeto",
            "percurso",
            "rossio",
        ]
        weather_markers = [
            "tempo",
            "weather",
            "meteorolog",
            "chuva",
            "rain",
            "temperatura",
            "temperature",
            "vento",
            "wind",
            "céu",
            "ceo",
            "sky",
        ]

        cleaned_lines: List[str] = []
        for line in body.splitlines():
            normalized = re.sub(r"\s+", " ", line.strip().lower())
            if weather_has_transport_leak and any(marker in normalized for marker in transport_markers):
                continue
            if researcher_has_weather_transport_leak and (
                any(marker in normalized for marker in weather_markers)
                or any(marker in normalized for marker in transport_markers)
            ):
                continue
            cleaned_lines.append(line)

        cleaned = "\n".join(cleaned_lines)
        return re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    def _render_structured_hybrid_response(self, agent_outputs: dict, language: str) -> str:
        """Builds a deterministic multi-section response for hybrid multi-agent answers."""
        filtered = {k: v for k, v in agent_outputs.items() if not k.startswith("_")}
        if not filtered:
            return ""

        section_order = ["weather", "transport", "researcher"]
        section_labels = {
            "pt": {
                "weather": "### 🌤️ Resumo Meteorológico",
                "transport": "### 🚇 Mobilidade e Ligações",
                "researcher": "### 📍 Destaques Locais",
                "notes": "**⚠️ Notas úteis**",
                "source": "📌 **Fonte:**",
                "updated": "**Atualizado:**",
            },
            "en": {
                "weather": "### 🌤️ Weather Snapshot",
                "transport": "### 🚇 Mobility and Connections",
                "researcher": "### 📍 Local Highlights",
                "notes": "**⚠️ Helpful notes**",
                "source": "📌 **Source:**",
                "updated": "**Updated:**",
            },
        }
        labels = section_labels["pt" if language == "pt" else "en"]
        qa_disclaimers = self._sanitize_qa_disclaimers(
            agent_outputs.get("_qa_disclaimers", []),
            language,
        )

        if len(filtered) == 1:
            single_output = str(list(filtered.values())[0])
            if not qa_disclaimers:
                return single_output

            body, links, timestamp = self._extract_structured_section_parts(single_output)
            notes = "\n".join(f"- ⚠️ {warning}" for warning in qa_disclaimers)
            response = (body or single_output.strip()) + f"\n\n---\n\n{labels['notes']}\n\n{notes}"
            if links:
                response += (
                    f"\n\n{labels['source']} {' | '.join(links)} | "
                    f"{labels['updated']} {timestamp or datetime.now().strftime('%H:%M')}"
                )
            return response.strip()

        sections: List[str] = []
        collected_links: List[str] = []
        collected_timestamps: List[str] = []

        ordered_agents = [name for name in section_order if name in filtered] + [
            name for name in filtered if name not in section_order
        ]

        for agent_name in ordered_agents:
            body, links, timestamp = self._extract_structured_section_parts(str(filtered[agent_name]))
            body = self._strip_cross_domain_hybrid_lines(agent_name, body, set(filtered.keys()))
            if not body:
                continue
            if links:
                for link in links:
                    if link not in collected_links:
                        collected_links.append(link)
            if timestamp:
                collected_timestamps.append(timestamp)

            title = labels.get(agent_name, f"### {agent_name.title()}")
            sections.append(f"{title}\n\n{body}")

        if qa_disclaimers:
            notes = "\n".join(f"- ⚠️ {warning}" for warning in qa_disclaimers)
            sections.append(f"{labels['notes']}\n\n{notes}")

        response = "\n\n---\n\n".join(section for section in sections if section.strip())
        if not response:
            return ""

        if collected_links:
            timestamp = max(collected_timestamps) if collected_timestamps else datetime.now().strftime("%H:%M")
            response += (
                f"\n\n{labels['source']} {' | '.join(collected_links)} | "
                f"{labels['updated']} {timestamp}"
            )

        return response.strip()

    def _build_combined_source_footer(self, agent_outputs: dict, language: str) -> Optional[str]:
        """Build a deterministic shared source footer from worker outputs when one is missing."""
        labels = {
            "pt": {"source": "📌 **Fonte:**", "updated": "**Atualizado:**"},
            "en": {"source": "📌 **Source:**", "updated": "**Updated:**"},
        }
        label_set = labels["pt" if language == "pt" else "en"]
        collected_links: List[str] = []
        collected_timestamps: List[str] = []
        transport_link_map = {
            "metro": "[*Metro de Lisboa*](https://www.metrolisboa.pt)",
            "carris": "[*Carris*](https://www.carris.pt)",
            "carris_metropolitana": "[*Carris Metropolitana*](https://www.carrismetropolitana.pt)",
            "cp": "[*CP*](https://www.cp.pt)",
        }

        for agent_name, output in (agent_outputs or {}).items():
            if str(agent_name).startswith("_") or not isinstance(output, str):
                continue
            _, links, timestamp = self._extract_structured_section_parts(output)
            has_weather_evidence = True
            if agent_name == "transport":
                agents_registry = getattr(self, "agents", {})
                transport_agent = agents_registry.get("transport") if isinstance(agents_registry, dict) else None
                tool_names = (
                    [
                        call.get("tool_name")
                        for call in transport_agent.get_tool_calls_log()
                        if isinstance(call, dict)
                    ]
                    if transport_agent is not None and hasattr(transport_agent, "get_tool_calls_log")
                    else []
                )
                operator_links = [transport_link_map[operator] for operator in operators_from_tool_names(tool_names)]
                if operator_links:
                    links = operator_links
            elif agent_name == "weather":
                weather_lower = output.lower()
                weather_fact_re = re.compile(
                    r"(?<![%\w])\d+(?:[.,]\d+)?\s*°\s*c\b"
                    r"|\b(?:warnings?|avisos?)[^\n]*(?:no active|sem avisos|active|ativos?)"
                    r"|\b(?:rain|chuva|precipita)[^\n:]{0,60}:\s*(?:\d|sem|no|muito|very|likely|prov[a-z]*|fraca|weak)"
                    r"|\b(?:wind|vento)[^\n:]{0,60}:\s*[a-z]"
                    r"|\b(?:periodos de ceu|períodos de céu|chuviscos|aguaceiros|light showers|sunny intervals|clear sky)\b",
                    re.IGNORECASE,
                )
                weather_limitation_re = re.compile(
                    r"\b(?:no detailed ipma forecast facts|no detailed weather facts|"
                    r"nao ha dados detalhados do ipma|não há dados detalhados do ipma|"
                    r"cannot confirm|can not confirm|can't confirm|nao consigo confirmar|não consigo confirmar|"
                    r"please verify the latest|confirma (?:a )?(?:previs|meteorolog|ipma))\b",
                    re.IGNORECASE,
                )
                has_weather_evidence = bool(weather_fact_re.search(weather_lower)) and not bool(
                    weather_limitation_re.search(weather_lower)
                )
                if not has_weather_evidence:
                    links = [link for link in links if "ipma.pt" not in link.lower()]
            for link in links:
                if link not in collected_links:
                    collected_links.append(link)
            if agent_name == "weather":
                if has_weather_evidence:
                    ipma_link = "[*IPMA*](https://www.ipma.pt)" if language == "pt" else "[*IPMA*](https://www.ipma.pt/en/)"
                    if not any("ipma.pt" in existing.lower() for existing in collected_links):
                        collected_links.insert(0, ipma_link)
            if timestamp:
                collected_timestamps.append(timestamp)

        if not collected_links:
            return None

        timestamp = max(collected_timestamps) if collected_timestamps else datetime.now().strftime("%H:%M")
        return (
            f"{label_set['source']} {' | '.join(collected_links)} | "
            f"{label_set['updated']} {timestamp}"
        )

    @staticmethod
    def _build_umbrella_advice(user_query: str, weather_output: str, language: str) -> str:
        """Build a concise umbrella answer when a hybrid response includes weather evidence."""
        normalized_query = re.sub(r"\s+", " ", (user_query or "").lower())
        if not any(term in normalized_query for term in ["umbrella", "guarda-chuva", "guarda chuva"]):
            return ""

        normalized_weather = re.sub(r"\s+", " ", (weather_output or "").lower())
        low_rain_signal = bool(
            re.search(r"\b(?:very unlikely|unlikely|muito improv[aá]vel|improv[aá]vel)\b", normalized_weather)
            or re.search(r"\b(?:rain|chuva)\b[^\n]{0,40}\b(?:0|1[0-9]|2[0-5])(?:\.0)?%", normalized_weather)
        )
        rain_expected = any(
            marker in normalized_weather
            for marker in ["rain", "showers", "chuva", "aguaceiros", "precipitação", "precipitacao"]
        ) and not any(
            marker in normalized_weather
            for marker in ["no rain expected", "sem precipitação", "sem precipitacao"]
        ) and not low_rain_signal

        if language == "pt":
            answer = (
                "Leva guarda-chuva: a previsão indica chuva ou aguaceiros."
                if rain_expected
                else "Não parece indispensável, mas confirma a previsão antes de sair."
            )
            return f"### ☔ Conselho de guarda-chuva\n\n- {answer}"

        answer = (
            "Bring an umbrella: rain or showers are forecast."
            if rain_expected
            else "An umbrella does not look essential, but check the forecast before leaving."
        )
        return f"### ☔ Umbrella Advice\n\n- {answer}"

    @staticmethod
    def _build_weather_activity_advice(user_query: str, weather_output: str, language: str) -> str:
        """Build a direct suitability answer for outdoor-activity weather questions."""
        normalized_query = MultiAgentAssistant._fold_context_text(user_query)
        if not normalized_query:
            return ""
        asks_decision = bool(
            re.search(
                r"\b(?:da para|d[aá]\s+para|posso|recomendas?|vale a pena|seguro|safe|"
                r"should i|can i|is it ok|is it safe)\b",
                normalized_query,
            )
        )
        outdoor_activity = bool(
            re.search(
                r"\b(?:piquenique|picnic|jardim|parque|praia|beach|caminhada|walk|"
                r"corrida|run|bicicleta|bike|vela|sailing|miradouro|esplanada|outdoor|"
                r"ar livre)\b",
                normalized_query,
            )
        )
        if not (asks_decision and outdoor_activity and weather_output.strip()):
            return ""

        normalized_weather = MultiAgentAssistant._fold_context_text(weather_output)
        percentages = [
            float(value.replace(",", "."))
            for value in re.findall(r"(\d+(?:[.,]\d+)?)\s*%", weather_output)
            if value
        ]
        max_probability = max(percentages) if percentages else None
        rain_possible = bool(
            re.search(r"\b(?:chuva|aguaceiros|precipitacao|rain|showers)\b", normalized_weather)
            and not re.search(r"\b(?:sem precipitacao|sem chuva|no rain)\b", normalized_weather)
        )
        wind_relevant = bool(
            re.search(r"\b(?:vento|wind)\b", normalized_weather)
            and re.search(r"\b(?:moderad|forte|strong|moderate)\b", normalized_weather)
        )

        risk_level = "low"
        if (max_probability is not None and max_probability >= 60) or re.search(
            r"\b(?:muito provavel|provavel|very likely|likely)\b", normalized_weather
        ):
            risk_level = "high"
        elif rain_possible or (max_probability is not None and max_probability >= 30) or wind_relevant:
            risk_level = "medium"

        rain_line = ""
        wind_line = ""
        for raw_line in weather_output.splitlines():
            line = raw_line.strip(" -")
            folded = MultiAgentAssistant._fold_context_text(line)
            if not rain_line and re.search(r"\b(?:chuva|rain|precipitacao)\b", folded):
                rain_line = line
            if not wind_line and re.search(r"\b(?:vento|wind)\b", folded):
                wind_line = line

        if language == "pt":
            if risk_level == "high":
                direct = "não é a melhor opção sem plano alternativo coberto, porque a previsão aponta para chuva relevante."
            elif risk_level == "medium":
                direct = "dá, mas eu faria com plano B coberto e confirmaria a previsão antes de sair."
            else:
                direct = "parece viável, mantendo a confirmação da previsão antes de sair."
            lines = [
                "### 🌤️ **Viabilidade da atividade**",
                "",
                f"✅ **Resposta direta:** {direct}",
            ]
            if rain_line or wind_line:
                lines.extend(["", "---", ""])
            if rain_line:
                lines.append(f"- 💧 **Chuva:** {rain_line}")
            if wind_line:
                lines.append(f"- 💨 **Vento:** {wind_line}")
            return "\n".join(lines).strip()

        if risk_level == "high":
            direct = "it is not ideal without a covered backup plan because the forecast points to meaningful rain."
        elif risk_level == "medium":
            direct = "it is possible, but I would keep a covered backup plan and recheck the forecast before leaving."
        else:
            direct = "it looks feasible, while still rechecking the forecast before leaving."
        lines = [
            "### 🌤️ **Activity Feasibility**",
            "",
            f"✅ **Direct answer:** {direct}",
        ]
        if rain_line or wind_line:
            lines.extend(["", "---", ""])
        if rain_line:
            lines.append(f"- 💧 **Rain:** {rain_line}")
        if wind_line:
            lines.append(f"- 💨 **Wind:** {wind_line}")
        return "\n".join(lines).strip()

    def _combine_outputs(self, agent_outputs: dict, language: str = "en") -> str:
        """
        Combines outputs from multiple agents into a structured response.

        Args:
            agent_outputs: Dict mapping agent names to their outputs.
            language: Language code (`en` or `pt`).

        Returns:
            str: Combined, coherent response.
        """
        if not agent_outputs:
            return "Não foi possível obter informação útil dos agentes. Por favor, reformule a sua questão." if language == "pt" else "Unable to retrieve useful information from the agents. Please rephrase your question."

        try:
            structured = self._render_structured_hybrid_response(agent_outputs, language)
            if structured:
                return structured
            filtered = {k: v for k, v in agent_outputs.items() if not k.startswith("_")}
            if len(filtered) == 1:
                return list(filtered.values())[0]
            return "\n\n---\n\n".join(filtered.values())

        except Exception:
            # Fallback to simple concatenation if structured rendering fails
            filtered = {k: v for k, v in agent_outputs.items() if not k.startswith("_")}
            sections = []
            if "weather" in filtered:
                sections.append(filtered["weather"])
            if "researcher" in filtered:
                sections.append(filtered["researcher"])
            if "transport" in filtered:
                sections.append(filtered["transport"])
            return "\n\n---\n\n".join(sections)

    def reset(self):
        """Resets the conversation state."""
        from agent.state import create_initial_state

        self.state = create_initial_state()
        self.last_execution_summary = None
        for agent in self.agents.values():
            reset_context = getattr(agent, "reset_conversation_context", None)
            if callable(reset_context):
                reset_context()

    def reset_llm_usage_tracking(self) -> None:
        """Resets LLM usage tracking across supervisor, QA, and worker agents."""
        self.supervisor.reset_llm_usage_tracking()
        self.qa_agent.reset_llm_usage_tracking()
        for agent in self.agents.values():
            agent.reset_llm_usage_tracking()

    def get_llm_usage_snapshot(self) -> Dict[str, Dict]:
        """Returns per-agent LLM usage summaries for the latest interaction batch."""
        return {
            "supervisor": self.supervisor.get_llm_usage_summary(),
            "qa": self.qa_agent.get_llm_usage_summary(),
            **{
                agent_name: agent.get_llm_usage_summary()
                for agent_name, agent in self.agents.items()
            },
        }

    def get_llm_usage_summary(self) -> Dict[str, object]:
        """
        Returns an aggregated LLM usage summary across the multi-agent system.

        Returns:
            Dict[str, object]: System-wide totals plus per-agent breakdowns.
        """
        by_agent = self.get_llm_usage_snapshot()
        totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        breakdown = []

        for _agent_name, summary in by_agent.items():
            tokens = summary.get("tokens", {})
            totals["input_tokens"] += int(tokens.get("input_tokens", 0) or 0)
            totals["output_tokens"] += int(tokens.get("output_tokens", 0) or 0)
            totals["total_tokens"] += int(tokens.get("total_tokens", 0) or 0)
            breakdown.extend(summary.get("llm_usage_breakdown", []))

        return {
            "call_count": sum(int(summary.get("call_count", 0) or 0) for summary in by_agent.values()),
            "usage_available": any(bool(summary.get("usage_available", False)) for summary in by_agent.values()),
            "tokens": totals,
            "llm_usage_breakdown": breakdown,
            "by_agent": by_agent,
        }


def create_multiagent_assistant() -> MultiAgentAssistant:
    """
    Creates a new Multi-Agent Lisbon Assistant instance.

    Returns:
        MultiAgentAssistant: Configured multi-agent assistant.
    """
    return MultiAgentAssistant()


# ==========================================================================
# Test Block - Comprehensive Multi-Agent System Tests
# ==========================================================================
if __name__ == "__main__":
    import os
    import sys
    from unittest.mock import MagicMock

    # .Fix Windows console encoding
    if sys.platform == "win32":
        with contextlib.suppress(AttributeError):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("MULTI-AGENT SYSTEM - SMOKE TEST SUITE")
    print("=" * 70)

    counters = {"passed": 0, "failed": 0}

    def _check(condition: bool, label: str) -> None:
        if condition:
            counters["passed"] += 1
            print(f"[PASS] {label}")
        else:
            counters["failed"] += 1
            print(f"[FAIL] {label}")

    def _make_usage_summary(
        *,
        model_id: str = "Unknown",
        input_tokens: int = 0,
        output_tokens: int = 0,
        call_count: int = 0,
    ) -> Dict[str, Any]:
        total_tokens = input_tokens + output_tokens
        breakdown = []
        if call_count > 0:
            provider, model = model_id.split("::", 1) if "::" in model_id else ("unknown", model_id)
            breakdown.append(
                {
                    "call_index": 1,
                    "provider": provider,
                    "model": model,
                    "model_id": model_id,
                    "tokens": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                    },
                    "usage_available": True,
                }
            )
        return {
            "call_count": call_count,
            "usage_available": bool(call_count),
            "model_id": model_id,
            "tokens": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
            "llm_usage_breakdown": breakdown,
        }

    def _make_worker_mock(
        *,
        model_id: str = "azure::gpt-5-mini",
        input_tokens: int = 0,
        output_tokens: int = 0,
        call_count: int = 0,
    ):
        worker = MagicMock()
        worker.get_llm_usage_summary = MagicMock(
            return_value=_make_usage_summary(
                model_id=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                call_count=call_count,
            )
        )
        worker.get_tool_calls_log = MagicMock(return_value=[])
        worker.reset_llm_usage_tracking = MagicMock()
        worker.llm_provider = "azure"
        return worker

    try:
        print("\n[OFFLINE] Building execution-summary smoke test...")
        assistant = MultiAgentAssistant.__new__(MultiAgentAssistant)
        assistant.state = {"messages": [], "user_context": None}
        assistant.supervisor = MagicMock()
        assistant.supervisor.get_llm_usage_summary = MagicMock(
            return_value=_make_usage_summary(
                model_id="azure::gpt-5-mini",
                input_tokens=600,
                output_tokens=120,
                call_count=1,
            )
        )
        assistant.qa_agent = MagicMock()
        assistant.qa_agent.get_llm_usage_summary = MagicMock(
            return_value=_make_usage_summary(
                model_id="azure::gpt-5-mini",
                input_tokens=200,
                output_tokens=50,
                call_count=1,
            )
        )
        assistant.agents = {
            "weather": _make_worker_mock(model_id="azure::gpt-5-mini", input_tokens=100, output_tokens=20, call_count=1),
            "transport": _make_worker_mock(),
            "researcher": _make_worker_mock(model_id="azure::claude-haiku-4.5", input_tokens=80, output_tokens=40, call_count=1),
            "planner": _make_worker_mock(),
        }

        original_tracking_status = get_langsmith_request_tracking_status
        globals()["get_langsmith_request_tracking_status"] = lambda: {
            "tracking_state": "disabled",
            "status_label": "disabled",
            "save_attempted": False,
            "persistence_state": "disabled",
            "current_run_attached": False,
            "project_name": None,
            "run_id": None,
            "reason": "LangSmith tracing is disabled by environment",
            "note": "LangSmith tracing is disabled by environment",
        }
        try:
            summary = assistant._collect_execution_summary(
                user_request="Demo summary request",
                routing_reasoning="Demo smoke-test routing",
                agents_to_call=["weather", "researcher"],
                agent_outputs={"weather": "ok", "researcher": "ok"},
                direct_response_used=False,
                workers=["weather", "researcher"],
                run_workers_in_parallel=True,
                qa_result={"complete": True},
                retry_agents_used=[],
                final_repair_ran=False,
                simple_weather_fact_check=None,
                elapsed_time=1.23,
            )
        finally:
            globals()["get_langsmith_request_tracking_status"] = original_tracking_status

        assistant._print_execution_summary(summary)
        _check(summary["execution_type"] == "hybrid", "Execution summary classifies parallel worker runs as hybrid")
        _check(summary["total_cost"]["pricing_complete"] is True, "Execution summary resolves complete pricing")
        _check(summary["langsmith"]["status_label"] == "disabled", "Execution summary includes LangSmith request state")

        if os.getenv("LISBOA_RUN_LIVE_GRAPH_TESTS") == "1":
            print("\n[LIVE] Running optional multi-agent smoke queries...")
            assistant = MultiAgentAssistant()
            live_queries = [
                "Hello!",
                "What's the weather in Lisbon today?",
                "How do I get from Rossio to Sintra by train?",
            ]
            for index, query in enumerate(live_queries, start=1):
                print("\n" + "─" * 70)
                print(f"[LIVE TEST {index}] {query}")
                print("─" * 70)
                live_response = assistant.chat(query, verbose=True)
                print(live_response)
                _check(bool(live_response.strip()), f"Live graph smoke returned content for '{query}'")
                assistant.reset()
        else:
            print("\n[INFO] Live graph smoke skipped. Set LISBOA_RUN_LIVE_GRAPH_TESTS=1 to enable it.")

        print("\n" + "=" * 70)
        print(f"TEST SUMMARY: Passed={counters['passed']} Failed={counters['failed']}")
        print("=" * 70)
        if counters["failed"]:
            raise SystemExit(1)
        print("\n[OK] Multi-agent smoke tests completed!")

    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback

        traceback.print_exc()
        print("\n[Tips]:")
        print("   1. Use the default offline smoke mode for quick validation")
        print("   2. Set LISBOA_RUN_LIVE_GRAPH_TESTS=1 only when live provider checks are intended")
        raise
