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
import re

# Always need as_completed for collecting parallel results
import time as time_module
from concurrent.futures import as_completed
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from agent.agents.base import (
    clean_response,
    is_local_provider,
)
from agent.llm_factory import LLMFactory
from agent.prompts import get_system_prompt
from agent.state import AgentState, create_initial_state
from agent.utils.langgraph_compat import ToolNode
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
    canonicalize_transport_terms,
    enforce_language_labels,
    ensure_response_title,
    final_visual_pass,
    finalize_worker_response,
    format_response,
    generate_response_title,
    reconcile_researcher_place_response,
    resolve_output_language,
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
# Agent Nodes
# ==========================================================================


def _get_tool_call_signature(tool_call) -> str:
    """
    Creates a signature string for a tool call to detect duplicates.

    Args:
        tool_call: A tool call object with 'name' and 'args' attributes.

    Returns:
        str: Unique signature for this tool call.
    """
    try:
        args_str = json.dumps(tool_call.get("args", {}), sort_keys=True)
    except (TypeError, AttributeError):
        args_str = str(getattr(tool_call, "args", {}))

    name = (
        tool_call.get("name", "")
        if isinstance(tool_call, dict)
        else getattr(tool_call, "name", "")
    )
    return f"{name}:{args_str}"


def _get_recent_tool_calls(messages, n: int = 5) -> Set[str]:
    """
    Gets the signatures of recent tool calls from message history.

    Args:
        messages: List of messages.
        n: Number of recent AI messages to check.

    Returns:
        Set[str]: Set of tool call signatures.
    """
    signatures = set()
    ai_msg_count = 0

    for msg in reversed(messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            ai_msg_count += 1
            for tc in msg.tool_calls:
                signatures.add(_get_tool_call_signature(tc))
            if ai_msg_count >= n:
                break

    return signatures


def _check_for_tool_loop(messages, new_tool_calls) -> bool:
    """
    Checks if the new tool calls are duplicates of recent ones.

    Args:
        messages: Current message history.
        new_tool_calls: Tool calls from the latest AI response.

    Returns:
        bool: True if this is a repeat loop, False otherwise.
    """
    if not new_tool_calls:
        return False

    # Get signatures of new tool calls
    new_signatures = {_get_tool_call_signature(tc) for tc in new_tool_calls}

    # Get signatures of recent tool calls
    recent_signatures = _get_recent_tool_calls(messages, n=3)

    # If all new tool calls were already made recently, it's a loop
    return bool(new_signatures) and new_signatures.issubset(recent_signatures)


def _has_tool_results_pending(messages) -> bool:
    """
    Checks if the last message is a tool result that should be summarized.

    Args:
        messages: List of messages.

    Returns:
        bool: True if last message is a ToolMessage.
    """
    if not messages:
        return False
    return isinstance(messages[-1], ToolMessage)


def create_agent_node(llm_with_tools, llm_base=None):
    """
    Creates the main agent node that decides actions.

    Args:
        llm_with_tools: LLM instance with tools bound.
        llm_base: Optional LLM without tools for generating final responses.

    Returns:
        Callable: Agent node function.
    """

    def agent_node(state: AgentState) -> dict:
        """
        Main agent decision node with loop detection.

        Args:
            state (AgentState): Current state.

        Returns:
            dict: Updated state with new messages.
        """
        messages = state["messages"]

        # Add system prompt if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            # Get language from state user_context
            language = "en"  # Default
            user_ctx = state.get("user_context")
            if user_ctx is not None:
                language = user_ctx.get("language", "en")

            system_msg = SystemMessage(content=get_system_prompt(language=language))
            messages = [system_msg] + list(messages)

        # Check if we have tool results that need to be summarized
        # Count consecutive tool messages at the end
        tool_result_count = 0
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                tool_result_count += 1
            else:
                break

        # If we have tool results, add a hint to summarize (helps small LLMs)
        if tool_result_count > 0:
            # Add a subtle prompt to encourage the LLM to respond
            hint_msg = SystemMessage(
                content="IMPORTANT: You have received tool results above. NOW RESPOND TO THE USER with the information. Do NOT call the same tool again."
            )
            messages_with_hint = list(messages) + [hint_msg]
            response = llm_with_tools.invoke(messages_with_hint)
        else:
            response = llm_with_tools.invoke(messages)

        # Check for tool call loops
        if hasattr(response, "tool_calls") and response.tool_calls:
            is_loop = _check_for_tool_loop(messages, response.tool_calls)

            if is_loop:
                # Loop detected! Force the LLM to generate a response without tools
                print("\n⚠️ Tool loop detected - forcing response generation...")

                # Create a message asking the LLM to summarize the tool results
                force_response_msg = SystemMessage(
                    content=(
                        "STOP! You are repeating the same tool call. "
                        "You already have the data from your previous tool call. "
                        "NOW RESPOND TO THE USER with the information you received. "
                        "DO NOT call any more tools."
                    )
                )

                # Try with the hint
                messages_with_force = list(messages) + [force_response_msg]

                # Use base LLM without tools if available, otherwise try again
                if llm_base:
                    response = llm_base.invoke(messages_with_force)
                else:
                    # Remove tool_calls and create a text response from tool results
                    response = _create_fallback_response(messages)

        return {"messages": [response]}

    return agent_node


def _create_fallback_response(messages) -> AIMessage:
    """
    Creates a fallback response when the LLM is stuck in a loop.
    Extracts tool results and formats them as an AI response.

    Args:
        messages: List of messages containing tool results.

    Returns:
        AIMessage: A simple response with the tool results.
    """
    # Find the most recent tool results
    tool_results = []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            tool_results.append(msg.content)
        elif hasattr(msg, "tool_calls") and msg.tool_calls:
            break  # Stop at the tool call, we have all results

    if tool_results:
        # Create a simple response with the tool results
        combined = "\n\n".join(reversed(tool_results))
        return AIMessage(content=f"Here's what I found:\n\n{combined}")
    else:
        return AIMessage(
            content="I apologize, but I'm having trouble processing your request. Could you please try rephrasing your question?"
        )


def should_continue(state: AgentState) -> str:
    """
    Determines whether to continue to tools or end the conversation.

    Args:
        state (AgentState): Current state.

    Returns:
        str: Next node name ('tools' or 'end').
    """
    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM made tool calls, continue to the tools node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Otherwise, end the conversation
    return "end"


# ==========================================================================
# Graph Construction
# ==========================================================================


def build_agent_graph(provider: str = None):
    """
    Builds the LangGraph agent workflow.

    Args:
        provider (str, optional): LLM provider override.

    Returns:
        CompiledGraph: Compiled LangGraph workflow.
    """
    # Initialize LLM
    llm = LLMFactory.get_llm(provider) if provider else LLMFactory.get_llm()

    # Reuse the same LLM instance for fallback (without tools binding)
    llm_base = llm

    # Get tools and bind to LLM
    tools = get_all_tools()
    llm_with_tools = llm.bind_tools(tools)

    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes - pass both LLMs to agent node (with and without tools)
    workflow.add_node("agent", create_agent_node(llm_with_tools, llm_base))
    workflow.add_node("tools", ToolNode(tools))

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional edges
    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "end": END}
    )

    # Tools always return to agent
    workflow.add_edge("tools", "agent")

    # Compile the graph
    return workflow.compile()


# ==========================================================================
# Agent Interface
# ==========================================================================


class LisbonAssistant:
    """
    High-level interface for the Lisbon Urban Assistant.

    Provides a simple API for interacting with the agent.
    """

    def __init__(self, provider: str = None):
        """
        Initializes the assistant.

        Args:
            provider (str, optional): LLM provider override.
        """
        self.graph = build_agent_graph(provider)
        self.state = create_initial_state()

        # Get model info for display - use Config directly instead of creating
        # a third LLM instance (build_agent_graph already creates 2)
        agent_models = Config.get_agent_models()
        sv_config = agent_models.get("supervisor", Config.get_default_agent_model())
        self.model_info = {
            "provider": sv_config.get("provider", Config.MODEL_PROVIDER),
            "model": sv_config.get("model", "Unknown"),
        }
        self.model_name = self.model_info.get("model", "Unknown")

    @traceable(
        name="single_agent_chat",
        run_type="chain",
        tags=["single-agent", "user-query"],
    )
    def chat(
        self,
        message: str,
        language: str = "en",
        verbose: bool = False,
        on_status_change: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Sends a message to the assistant and gets a response.

        Args:
            message (str): User message.
            language (str): Language code ('en' or 'pt').
            verbose (bool): Whether to print verbose output.
            on_status_change (func): Callback for status updates.

        Returns:
            str: Assistant response.
        """
        # Add user message to state
        self.state["messages"].append(HumanMessage(content=message))
        ui_language = language
        effective_language, requires_bilingual_note, detected_language = resolve_output_language(
            user_query=message,
            ui_default=ui_language,
        )

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

        if LANGSMITH_AVAILABLE:
            annotate_current_run(
                metadata={
                    "assistant_mode": "single-agent",
                    "language": effective_language,
                    "ui_language": ui_language,
                    "detected_language": detected_language or effective_language,
                    "requires_bilingual_note": requires_bilingual_note,
                    "request_source": "user_chat",
                }
            )

        # Run the graph with recursion limit to prevent infinite loops
        # Default LangGraph limit is 25, increased for smaller models with tool calling
        result = self.graph.invoke(self.state, config={"recursion_limit": 25})

        # Update state with result
        self.state = result

        # Extract the final response
        last_message = result["messages"][-1]

        if hasattr(last_message, "content"):
            # Clean model-specific artifacts (thinking tags, chat tokens, etc.)
            # Uses clean_response from agent.agents.base, then format for Streamlit
            rendered = format_response(clean_response(last_message.content))
        else:
            rendered = format_response(clean_response(str(last_message)))

        # Phase 1.2: enforce PT/EN label consistency on single-agent output too.
        rendered = enforce_language_labels(rendered, effective_language)

        if requires_bilingual_note and rendered.strip():
            note = build_bilingual_note(detected_language or "und")
            if note and note not in rendered:
                rendered = f"{note}\n\n{rendered}"

        rendered = final_visual_pass(rendered)

        return rendered

    def reset(self):
        """Resets the conversation state."""
        self.state = create_initial_state()

    def get_history(self) -> List:
        """
        Returns the conversation history.

        Returns:
            List: List of messages.
        """
        return self.state["messages"]


# ==========================================================================
# Convenience Functions
# ==========================================================================


def create_assistant(provider: str = None) -> LisbonAssistant:
    """
    Creates a new Lisbon Assistant instance.

    Args:
        provider (str, optional): LLM provider ('openai', 'azure', 'lmstudio').

    Returns:
        LisbonAssistant: Configured assistant instance.
    """
    return LisbonAssistant(provider)


def quick_chat(message: str, provider: str = None) -> str:
    """
    Quick one-off chat without maintaining state.

    Args:
        message (str): User message.
        provider (str, optional): LLM provider.

    Returns:
        str: Assistant response.
    """
    assistant = create_assistant(provider)
    return assistant.chat(message)


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
        """Returns the model name for display (compatibility with V1 app)."""
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

        if "event details (dates, times, ticket prices) should be confirmed at visitlisboa.com" in lowered:
            return None

        if "carris bus route numbers and schedules should be verified at carris.pt" in lowered:
            if language == "pt":
                return "Os números das linhas e os horários da Carris devem ser confirmados em carris.pt, porque os dados GTFS podem não refletir alterações muito recentes."
            return "Carris line numbers and schedules should be confirmed at carris.pt, because GTFS data may miss very recent changes."

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
        pricing_metadata = summary.get("pricing_metadata", {}) if isinstance(summary, dict) else {}
        total_cost = summary.get("total_cost", {}) if isinstance(summary, dict) else {}
        show_detailed_terminal_logs = bool(
            getattr(Config, "SHOW_DETAILED_EXECUTION_LOGS", False)
        )

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

        pricing_snapshot = (
            pricing_metadata.get("pricing_snapshot_date")
            or pricing_metadata.get("pricing_updated_at")
            or "n/a"
        )
        print(
            f"💵  Total Cost: {self._format_usd_cost_label(total_cost)} | "
            f"Pricing Snapshot: {pricing_snapshot}"
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
            ensure_transport_notes_heading,
            infer_researcher_source_kind,
            reconcile_researcher_event_response,
            normalize_transport_notes_block,
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

        if effective_agents:
            title = generate_response_title(effective_agents, message, language)
            final_output = ensure_response_title(formatted, title)
            if "transport" in effective_agents:
                final_output = canonicalize_transport_terms(final_output, language=language)
                final_output = strip_technical_output_artifacts(final_output)
                final_output = ensure_transport_notes_heading(final_output, language=language)
                final_output = normalize_transport_notes_block(final_output)
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
            final_output = final_visual_pass(final_output)

        if agent_outputs:
            source_footer = self._build_combined_source_footer(agent_outputs, language)
            if source_footer:
                footer_line_re = re.compile(r"^(?:[-*•]\s*)?📌\s*\*\*(?:Fonte|Source):\*\*.*$", re.IGNORECASE)
                kept_lines = [line for line in final_output.splitlines() if not footer_line_re.match(line.strip())]
                while kept_lines and not kept_lines[-1].strip():
                    kept_lines.pop()
                final_output = "\n".join(kept_lines).rstrip()
                final_output = f"{final_output}\n\n{source_footer}".strip()
                final_output = final_visual_pass(final_output)
                if "transport" in effective_agents:
                    final_output = enforce_language_labels(final_output, language)
                    final_output = canonicalize_transport_terms(final_output, language=language)
                    final_output = ensure_transport_notes_heading(final_output, language=language)
                    final_output = normalize_transport_notes_block(final_output)
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
            and infer_researcher_source_kind(user_query=message, text=final_output) == "events"
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
            print("=" * 80)
            print("📝 FINAL RESPONSE (Markdown)")
            print("=" * 80)
            print(final_output)
            print("=" * 80 + "\n")

        return final_output

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
                "Keep the same user language, stay strictly grounded in the tool data, "
                "and do not mention QA, validation, or internal checks.\n- "
                + "\n- ".join(agent_specific_feedback)
            )

        return context

    @staticmethod
    def _should_run_final_qa_repair(
        agents_to_call: List[str],
        qa_result: Optional[Dict[str, object]],
    ) -> bool:
        """Returns whether a final QA repair pass is worth running."""
        if not qa_result:
            return False
        if "planner" in agents_to_call:
            return True
        if qa_result.get("needs_repair"):
            return True
        if qa_result.get("missing_data"):
            return True

        fact_check = qa_result.get("fact_check", {})
        if isinstance(fact_check, dict) and fact_check.get("critical_issues"):
            return True

        return False

    @staticmethod
    def _should_block_planner_publication(
        qa_result: Optional[Dict[str, object]],
    ) -> bool:
        """Return whether planner synthesis should be suppressed after QA.

        If the grounded worker evidence is still incomplete after the QA pass,
        publishing a confident itinerary is worse than falling back to the
        structured worker outputs plus explicit caveats.
        """
        if not qa_result:
            return False
        if qa_result.get("complete") is False:
            return True
        if qa_result.get("needs_repair"):
            return True
        if qa_result.get("missing_data"):
            return True

        fact_check = qa_result.get("fact_check", {})
        return bool(
            isinstance(fact_check, dict) and fact_check.get("critical_issues")
        )

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
        effective_language, requires_bilingual_note, detected_language = resolve_output_language(
            user_query=message,
            ui_default=ui_language,
        )

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
        routing = self.supervisor.route(
            message,
            language=effective_language,
            conversation_history=history_for_routing,
        )
        agents_to_call = routing.get("agents", [])
        direct_response = routing.get("direct_response")
        reasoning = routing.get("reasoning", "")

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

        # Map internal agent names to user-friendly display names
        name_map_pt = {
            "weather": "Meteorologia 🌤️",
            "transport": "Transportes 🚇",
            "researcher": "Pesquisa Local 🔎",
            "planner": "Planeador 📅",
        }
        name_map_en = {
            "weather": "Weather 🌤️",
            "transport": "Transport 🚇",
            "researcher": "Local Search 🔎",
            "planner": "Planner 📅",
        }
        name_map = name_map_pt if ui_language == "pt" else name_map_en

        # Notify status: Agents selected
        if agents_to_call:
            # Filter out planner from the "Consulting" list as it runs last
            consulting: list[str] = [
                str(name_map.get(a, a or ""))
                for a in agents_to_call
                if a and a != "planner"
            ]

            if consulting and on_status_change:
                msg = (
                    f"🚀 Vou consultar: {', '.join(consulting)}..."
                    if ui_language == "pt"
                    else f"🚀 Consulting: {', '.join(consulting)}..."
                )
                on_status_change(msg)

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
                friendly_workers = [name_map.get(w, w) for w in workers]
                msg = (
                    f"⏳ A aguardar respostas de: {', '.join(friendly_workers)}..."
                    if ui_language == "pt"
                    else f"⏳ Waiting for: {', '.join(friendly_workers)}..."
                )
                on_status_change(msg)

            # Context for agents: language instruction + minimal follow-up context
            # Workers should focus on the CURRENT query, not be biased by history
            agent_context = f"User language: {effective_language}. Respond in {'Portuguese (PT-PT)' if effective_language == 'pt' else 'English'}."

            # Only add last user message for follow-up context (e.g., "E amanhã?")
            recent_msgs = self.state.get("messages", [])
            if len(recent_msgs) > 1:
                # Find the previous user message for reference
                for msg in reversed(recent_msgs[:-1]):
                    if isinstance(msg, HumanMessage) and msg.content:
                        agent_context += f"\nPrevious user question (for context only): {msg.content[:150]}"
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

                        # Pass verbose=verbose to invoke
                        future_to_agent[
                            executor.submit(
                                self.agents[agent_name].invoke,
                                message,
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
                        output = self.agents[agent_name].invoke(
                            message,
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

        # Step 4: QA Validation (single retry if incomplete)
        skip_qa_for_simple_weather = (
            workers == ["weather"]
            and "planner" not in agents_to_call
            and (
                self.agents["weather"]._is_current_weather_query(message)
                or self.agents["weather"]._is_simple_forecast_query(message)
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

        if agent_outputs and len(workers) > 0 and not skip_qa_for_simple_weather:
            if verbose:
                print("\n   [QA] Validating completeness...")

            if on_status_change:
                msg = (
                    "🔍 A validar completude dos dados..."
                    if ui_language == "pt"
                    else "🔍 Validating data completeness..."
                )
                on_status_change(msg)

            messages_list = self.state.get("messages", [])
            qa_history = (
                [
                    f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: "
                    f"{m.content[:_QA_MSG_PREVIEW_LEN]}"
                    for m in messages_list[:-1][-_QA_HISTORY_WINDOW:]
                ]
                if len(messages_list) > 1 else None
            )
            qa_result = self.qa_agent.validate(
                user_query=message,
                agent_outputs=agent_outputs,
                agents_called=workers,
                language=effective_language,
                user_context=self.state.get("user_context"),
                conversation_history=qa_history,
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

            if retry_agents and (not qa_result["complete"] or qa_result.get("needs_repair")):
                retry_agents_used = list(retry_agents)

                if retry_agents:
                    if verbose:
                        print(f"   [QA RETRY] Calling additional agents: {retry_agents}")

                    if on_status_change:
                        friendly_retry = [name_map.get(a, a) for a in retry_agents]
                        msg = (
                            f"🔄 A recolher dados adicionais: {', '.join(friendly_retry)}..."
                            if ui_language == "pt"
                            else f"🔄 Gathering additional data: {', '.join(friendly_retry)}..."
                        )
                        on_status_change(msg)

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

                    qa_result_2 = self.qa_agent.validate(
                        user_query=message,
                        agent_outputs=agent_outputs,
                        agents_called=workers + retry_agents,
                        language=effective_language,
                        user_context=self.state.get("user_context"),
                        conversation_history=qa_history,
                    )

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
            if isinstance(aoutput, str) and aoutput.startswith("Error:"):
                if verbose:
                    print(f"   [FILTER] Removing failed agent output: {aname}")
                continue
            clean_outputs[aname] = aoutput
        agent_outputs = clean_outputs

        planner_requested = "planner" in agents_to_call
        planner_blocked = planner_requested and self._should_block_planner_publication(
            qa_result
        )
        response_agents_to_call = list(agents_to_call)
        planner_executed = False

        if planner_blocked:
            response_agents_to_call = [
                agent_name for agent_name in agents_to_call if agent_name != "planner"
            ]
            if verbose:
                print(
                    "\n   [QA] Blocking planner synthesis because grounded data is still incomplete"
                )
            if on_status_change:
                on_status_change(
                    "⚠️ A consolidar resposta grounded sem itinerário final..."
                    if effective_language == "pt"
                    else "⚠️ Consolidating a grounded answer without final itinerary synthesis..."
                )

        # Step 6: If Planner was requested and QA did not block publication,
        # synthesize the final response. Otherwise, fall back to the combined
        # worker outputs so caveats remain visible instead of publishing a
        # confident itinerary over incomplete evidence.
        if planner_requested and not planner_blocked:
            if verbose:
                print(
                    f"\n   [AGENT: PLANNER] Synthesizing from {list(agent_outputs.keys())}..."
                )

            if on_status_change:
                on_status_change("✍️ A escrever o itinerário final...")

            response = self.agents["planner"].synthesize(message, agent_outputs)
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

        if self._should_run_final_qa_repair(response_agents_to_call, qa_result):
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
            from agent.agents.planner_agent import enforce_multi_day_quality_mode
            from agent.utils.response_formatter import finalize_worker_response

            response = enforce_multi_day_quality_mode(
                response=response,
                user_message=message,
                language=effective_language,
            )
            response = finalize_worker_response(response, "planner", message, effective_language)
        elif len(response_agents_to_call) == 1 and response_agents_to_call[0] in {"researcher", "transport"}:
            from agent.utils.response_formatter import finalize_worker_response

            response = finalize_worker_response(
                response,
                response_agents_to_call[0],
                message,
                effective_language,
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
            from agent.utils.response_formatter import canonicalize_local_information_terms

            response = canonicalize_local_information_terms(response, effective_language)

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

    @staticmethod
    def _extract_structured_section_parts(text: str) -> tuple[str, List[str], Optional[str]]:
        """Removes per-section source lines while collecting links and timestamps for a combined footer."""
        source_line_re = re.compile(
            r"^(?:[-*•]\s*)?(?:📌\s*)?(?:\*\*)?(?:Fonte|Source)(?:\*\*)?:.*$",
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
                "notes": "### ⚠️ Notas Úteis",
                "source": "📌 **Fonte:**",
                "updated": "**Atualizado:**",
            },
            "en": {
                "weather": "### 🌤️ Weather Snapshot",
                "transport": "### 🚇 Mobility and Connections",
                "researcher": "### 📍 Local Highlights",
                "notes": "### ⚠️ Helpful Notes",
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

        for agent_name, output in (agent_outputs or {}).items():
            if str(agent_name).startswith("_") or not isinstance(output, str):
                continue
            _, links, timestamp = self._extract_structured_section_parts(output)
            for link in links:
                if link not in collected_links:
                    collected_links.append(link)
            if timestamp:
                collected_timestamps.append(timestamp)

        if not collected_links:
            return None

        timestamp = max(collected_timestamps) if collected_timestamps else datetime.now().strftime("%H:%M")
        return (
            f"{label_set['source']} {' | '.join(collected_links)} | "
            f"{label_set['updated']} {timestamp}"
        )

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

        for agent_name, summary in by_agent.items():
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

    def get_history(self) -> List:
        """Returns the conversation history."""
        return self.state["messages"]


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
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

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
