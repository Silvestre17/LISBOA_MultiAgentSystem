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

# Always need as_completed for collecting parallel results
import time as time_module  # For latency tracking
from concurrent.futures import as_completed
from typing import Callable, Dict, List, Optional, Set

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from agent.agents.base import clean_response  # Shared response cleaning utility
from agent.llm_factory import LLMFactory
from agent.prompts import get_system_prompt
from agent.state import AgentState, create_initial_state
from agent.utils.langgraph_compat import ToolNode
from agent.utils.langsmith_tracing import (
    LANGSMITH_AVAILABLE,
    ContextThreadPoolExecutor,
    annotate_current_run,
    get_current_run_tree,
    traceable,
)

# Response formatting for Streamlit rendering
from agent.utils.response_formatter import (
    ensure_response_title,
    format_response,
    generate_response_title,
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

        # Update user language preference in state
        user_ctx = self.state.get("user_context")
        if user_ctx is None:
            from agent.state import UserContext

            user_ctx = UserContext()
            user_ctx["language"] = language
            self.state["user_context"] = user_ctx
        else:
            user_ctx["language"] = language

        if LANGSMITH_AVAILABLE:
            annotate_current_run(
                metadata={
                    "assistant_mode": "single-agent",
                    "language": language,
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
            return format_response(clean_response(last_message.content))

        return format_response(clean_response(str(last_message)))

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
        from langchain_core.messages import HumanMessage

        self.state["messages"].append(HumanMessage(content=message))

        # Update user language preference in state
        user_ctx = self.state.get("user_context")
        if user_ctx is None:
            from agent.state import UserContext

            user_ctx = UserContext()
            user_ctx["language"] = language
            self.state["user_context"] = user_ctx
        else:
            user_ctx["language"] = language

        if LANGSMITH_AVAILABLE:
            annotate_current_run(
                metadata={
                    "assistant_mode": "multi-agent",
                    "language": language,
                    "request_source": "user_chat",
                }
            )

        # Notify status: Routing
        if on_status_change:
            status_msg = (
                "🤔 A analisar o pedido..."
                if language == "pt"
                else "🤔 Analyzing request..."
            )
            on_status_change(status_msg)

        # Step 1: Route the query (with conversation history for follow-up awareness)
        # Exclude the current message (last) from history
        history_for_routing = self.state["messages"][:-1] if len(self.state["messages"]) > 1 else None
        routing = self.supervisor.route(
            message,
            language=language,
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
        name_map = name_map_pt if language == "pt" else name_map_en

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
                    if language == "pt"
                    else f"🚀 Consulting: {', '.join(consulting)}..."
                )
                on_status_change(msg)

        # Step 2: Handle direct response (no agents needed)
        if direct_response and not agents_to_call:
            if verbose:
                print("      Mode: DIRECT RESPONSE (no agents called)")
            return format_response(clean_response(direct_response))

        # Step 3: Execute agents (Parallelized with LangSmith context propagation)
        agent_outputs = {}

        # Identify worker agents (exclude planner which runs last)
        workers = [a for a in agents_to_call if a != "planner" and a in self.agents]

        if workers:
            if verbose:
                print(f"      [PARALLEL] Executing {len(workers)} agents: {workers}")

            if on_status_change:
                friendly_workers = [name_map.get(w, w) for w in workers]
                msg = (
                    f"⏳ A aguardar respostas de: {', '.join(friendly_workers)}..."
                    if language == "pt"
                    else f"⏳ Waiting for: {', '.join(friendly_workers)}..."
                )
                on_status_change(msg)

            # Context for agents: language instruction + minimal follow-up context
            # Workers should focus on the CURRENT query, not be biased by history
            agent_context = f"User language: {language}. Respond in {'Portuguese (PT-PT)' if language == 'pt' else 'English'}."

            # Only add last user message for follow-up context (e.g., "E amanhã?")
            recent_msgs = self.state.get("messages", [])
            if len(recent_msgs) > 1:
                # Find the previous user message for reference
                for msg in reversed(recent_msgs[:-1]):
                    if isinstance(msg, HumanMessage) and msg.content:
                        agent_context += f"\nPrevious user question (for context only): {msg.content[:150]}"
                        break

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
                            agent_context,  # context with language
                            verbose,  # verbose flag
                        )
                    ] = agent_name

                # Collect results as they complete with latency tracking
                for future in as_completed(future_to_agent):
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
                        error_msg = f"Error: {str(e)}"
                        agent_outputs[agent_name] = error_msg
                        if verbose:
                            print(f"   [AGENT: {agent_name.upper()}] Failed: {str(e)}")

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

        if agent_outputs and len(workers) > 0 and not skip_qa_for_simple_weather:
            if verbose:
                print("\n   [QA] Validating completeness...")

            if on_status_change:
                msg = (
                    "🔍 A validar completude dos dados..."
                    if language == "pt"
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
                language=language,
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

            # Single retry: call missing agents or agents needing refinement if QA says incomplete
            if not qa_result["complete"] and qa_result["required_agents"]:
                retry_agents = [
                    a for a in qa_result["required_agents"]
                    if a in self.agents and a != "planner"
                ]

                if retry_agents:
                    if verbose:
                        print(f"   [QA RETRY] Calling additional agents: {retry_agents}")

                    if on_status_change:
                        friendly_retry = [name_map.get(a, a) for a in retry_agents]
                        msg = (
                            f"🔄 A recolher dados adicionais: {', '.join(friendly_retry)}..."
                            if language == "pt"
                            else f"🔄 Gathering additional data: {', '.join(friendly_retry)}..."
                        )
                        on_status_change(msg)

                    # Construct feedback context
                    feedback_context = agent_context
                    if qa_result.get("missing_data"):
                        feedback_context += (
                            f"\n\nIMPORTANT QA FEEDBACK: Your previous search missed the following required data: "
                            f"{', '.join(qa_result['missing_data'])}. "
                        )
                        if qa_result.get("reasoning"):
                            feedback_context += f"Reasoning: {qa_result['reasoning']}. "
                        feedback_context += "Please specifically search for and provide this missing information."

                    # Execute retry agents in parallel
                    with ContextThreadPoolExecutor(max_workers=len(retry_agents)) as executor:
                        retry_futures = {}
                        for agent_name in retry_agents:
                            # Use feedback context if agent was already called
                            ctx = feedback_context if agent_name in workers else agent_context
                            retry_futures[
                                executor.submit(
                                    self.agents[agent_name].invoke,
                                    message,
                                    ctx,
                                    verbose,
                                )
                            ] = agent_name

                        for future in as_completed(retry_futures):
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

                    # Post-retry re-validation (lightweight, no further retries)
                    if verbose:
                        print("   [QA] Post-retry re-validation...")

                    qa_result_2 = self.qa_agent.validate(
                        user_query=message,
                        agent_outputs=agent_outputs,
                        agents_called=workers + retry_agents,
                        language=language,
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
                    merged_fc_disclaimers = list(set(
                        fc1.get("disclaimers", []) + fc2.get("disclaimers", [])
                    ))

                    qa_result = qa_result_2
                    qa_result["disclaimers"] = all_disclaimers
                    if qa_result.get("fact_check"):
                        qa_result["fact_check"]["disclaimers"] = merged_fc_disclaimers

            # Pass QA disclaimers as context for synthesis (internal key, filtered from output)
            all_qa_warnings = qa_result.get("disclaimers", [])
            fc_warns = qa_result.get("fact_check", {})
            if isinstance(fc_warns, dict):
                all_qa_warnings = list(set(all_qa_warnings + fc_warns.get("disclaimers", [])))
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

        # Step 6: If Planner was requested, synthesize final response
        if "planner" in agents_to_call:
            if verbose:
                print(
                    f"\n   [AGENT: PLANNER] Synthesizing from {list(agent_outputs.keys())}..."
                )

            if on_status_change:
                on_status_change("✍️ A escrever o itinerário final...")

            response = self.agents["planner"].synthesize(message, agent_outputs)
        elif agent_outputs:
            # Combine agent outputs if no planner
            response = self._combine_outputs(agent_outputs)
        else:
            # Fallback: Use researcher for general queries
            if verbose:
                print("\n   [FALLBACK] Using researcher agent")
            response = self.agents["researcher"].invoke(message, verbose=verbose)

        formatted = format_response(clean_response(response))
        title = generate_response_title(agents_to_call, message, language)
        return ensure_response_title(formatted, title)

    def _combine_outputs(self, agent_outputs: dict) -> str:
        """
        Combines outputs from multiple agents into a single coherent response
        using LLM synthesis (D3) instead of naive concatenation.

        Args:
            agent_outputs: Dict mapping agent names to their outputs.

        Returns:
            str: Combined, coherent response.
        """
        # Filter out internal keys (QA metadata, etc.) - never expose to user
        filtered = {k: v for k, v in agent_outputs.items() if not k.startswith("_")}

        if not filtered:
            return ""

        # If only one agent responded, return its output directly
        if len(filtered) == 1:
            return list(filtered.values())[0]

        # Use LLM synthesis for multi-agent responses
        try:
            from langchain_core.messages import HumanMessage as HMsg
            from langchain_core.messages import SystemMessage as SysMsg

            sections = []
            for agent_name, output in filtered.items():
                label = {
                    "weather": "Weather Information",
                    "transport": "Transport Information",
                    "researcher": "Places & Attractions",
                }.get(agent_name, agent_name.title())
                sections.append(f"## {label}\n{output}")

            combined_data = "\n\n---\n\n".join(sections)

            # Add QA disclaimers as context for synthesis (if any)
            qa_disclaimers = agent_outputs.get("_qa_disclaimers", [])
            if qa_disclaimers:
                combined_data += "\n\n## Data Limitations\n" + "\n".join(f"- {d}" for d in qa_disclaimers)

            synthesis_prompt = (
                "You are a response synthesizer. Combine the following agent outputs "
                "into a single, coherent, well-organized response. "
                "Preserve ALL factual data from each source. "
                "Use markdown formatting with ### headers and emojis. "
                "Use **bold** for all important information (names, dates, prices, labels, statuses). "
                "Do NOT add information that isn't in the source data. "
                "Make the response flow naturally as if from a single assistant. "
                "Do not mention internal agent names, tool names, quality checks, "
                "disclaimers sections, or any internal system details. "
                "\n\n"
                "RULES:\n"
                "1. MATCH THE USER'S LANGUAGE: Portuguese query = Portuguese response, English query = English response.\n"
                "2. Do not suggest features that don't exist: no 'reservar bilhetes', 'book tickets', "
                "'send reminders', 'set alerts', 'save favorites'. The system cannot do these.\n"
                "3. Do not write closing sections like 'Se quiser, eu posso:' or 'I can also:' offering additional services.\n"
                "4. Do not use ambiguous labels like 'seleção top 5' or 'best picks' unless the user asked for a ranking.\n"
                "5. End with ONE source attribution line. Format: '📌 **Fonte:** [*Name*](url) **| Atualizado:** HH:MM'. Do not duplicate source lines.\n"
                "6. Use **bold** formatting extensively - ALL section headers, operator names, labels, and key values must be bold.\n"
                "7. Every list item must start with `- ` followed by an emoji.\n"
                "8. Source names in 📌 **Fonte** lines must use italic markdown: [*Name*](url), not plain text.\n"
                "9. Do NOT count stops between stations or claim stop positions (e.g., '1ª paragem após X'). Report only origin, destination, and line.\n"
                "10. Do NOT add data not present in the source outputs. If information is missing, omit it rather than inventing it.\n"
                "11. If 'Data Limitations' are listed, mention them naturally (e.g., 'opening hours may vary, check the official website').\n"
                "12. Use `---` horizontal rules ONLY to separate distinct topic sections (e.g., weather from transport from places). Never use them within the same topic.\n"
                "13. Format transport data with correct emoji patterns: 🚇 Metro, 🚌 buses, 🚆 trains, 🚋 trams. Sub-items under each operator MUST be `- ` bullets.\n"
                "14. Preserve the visual hierarchy: operator name with emoji as header, then `- ` bullet items with status emojis (🟢, ⚠️, ❌) and **bold** labels.\n"
                "15. Keep the answer user-facing. Do not add meta commentary about constraints, evaluation, or how the response was produced.\n"
                "16. Do not add closing offers such as 'Se quiser...' or 'I can also...'. End naturally after the useful content and source attribution.\n"
            )

            messages = [
                SysMsg(content=synthesis_prompt),
                HMsg(content=f"Combine these agent outputs into one response:\n\n{combined_data}"),
            ]

            response = self.supervisor._safe_llm_invoke(self.supervisor.llm, messages)
            # Return raw cleaned text - format_response is called by chat()
            return clean_response(response.content)

        except Exception:
            # Fallback to simple concatenation if LLM fails
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
    import sys

    # .Fix Windows console encoding
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    print("=" * 70)
    print("MULTI-AGENT SYSTEM - COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    try:
        # Initialize
        if Config.USE_MULTI_AGENT:
            print("\n[INIT] Initializing Multi-Agent Assistant...")
            assistant = MultiAgentAssistant()
            print("[OK] Ready!")
            print(f"   Supervisor: {assistant.model_info['supervisor']}")
            print(f"   Weather:    {assistant.model_info['weather']}")
            print(f"   Transport:  {assistant.model_info['transport']}")
            print(f"   Researcher: {assistant.model_info['researcher']}")
            print(f"   Planner:    {assistant.model_info['planner']}")
        else:
            print("\n[INIT] Single-Agent mode - switching to Multi-Agent for tests...")
            assistant = MultiAgentAssistant()

        # =================================================================
        # TEST CATEGORIES
        # =================================================================

        test_cases = [
            # ---------------------------------------------------------
            # CATEGORY 1: OFF-TOPIC / GENERAL (No agents needed)
            # ---------------------------------------------------------
            {
                "category": "OFF-TOPIC / GENERAL",
                "description": "Questions unrelated to Lisbon - should respond directly without agents",
                "queries": [
                    ("Hello!", "Greeting - no agents needed"),
                    (
                        "What is the capital of France?",
                        "Off-topic - should decline or answer directly",
                    ),
                    ("Who won the World Cup in 2022?", "Off-topic - not about Lisbon"),
                ],
            },
            # ---------------------------------------------------------
            # CATEGORY 2: SINGLE AGENT - WEATHER ONLY
            # ---------------------------------------------------------
            {
                "category": "SINGLE AGENT - WEATHER",
                "description": "Weather questions - should ONLY call weather agent",
                "queries": [
                    (
                        "What's the weather in Lisbon today?",
                        "Should use only weather agent",
                    ),
                    ("Is it going to rain tomorrow?", "Simple forecast - weather only"),
                ],
            },
            # ---------------------------------------------------------
            # CATEGORY 3: SINGLE AGENT - TRANSPORT ONLY
            # ---------------------------------------------------------
            {
                "category": "SINGLE AGENT - TRANSPORT",
                "description": "Transport questions - should ONLY call transport agent",
                "queries": [
                    ("Is the metro working?", "Metro status - transport only"),
                    ("How do I get from Rossio to Belem?", "Routing - transport only"),
                ],
            },
            # ---------------------------------------------------------
            # CATEGORY 4: SINGLE AGENT - RESEARCHER ONLY
            # ---------------------------------------------------------
            {
                "category": "SINGLE AGENT - RESEARCHER",
                "description": "Places/events questions - should ONLY call researcher agent",
                "queries": [
                    (
                        "What are the best museums in Lisbon?",
                        "Places search - researcher only",
                    ),
                    ("Are there any events today?", "Events search - researcher only"),
                    (
                        "Tell me about the history of Castelo de São Jorge",
                        "History search - researcher only",
                    ),
                ],
            },
            # ---------------------------------------------------------
            # CATEGORY 5: MULTI-AGENT - COMPLEX QUERIES
            # ---------------------------------------------------------
            {
                "category": "MULTI-AGENT - COMPLEX",
                "description": "Complex queries requiring multiple agents + planner",
                "queries": [
                    (
                        "Plan my day visiting museums, considering the weather",
                        "Needs weather + researcher + planner",
                    ),
                    (
                        "Suggest outdoor activities for today based on weather",
                        "Needs weather + researcher + planner",
                    ),
                ],
            },
        ]

        # =================================================================
        # RUN TESTS
        # =================================================================

        total_tests = 0

        for category_data in test_cases:
            print("\n" + "=" * 70)
            print(f"CATEGORY: {category_data['category']}")
            print(f"   {category_data['description']}")
            print("=" * 70)

            for query, expected in category_data["queries"]:
                total_tests += 1
                print(f"\n{'─' * 70}")
                print(f"[TEST {total_tests}]: {query}")
                print(f"   Expected: {expected}")
                print("─" * 70)

                # Run with verbose mode to show routing and agent usage
                response = assistant.chat(query, verbose=True)

                print("\n[RESPONSE]:")

                # Print full response
                print(response)

                # Reset state for next test
                assistant.reset()

        # =================================================================
        # SUMMARY
        # =================================================================
        print("\n" + "=" * 70)
        print(f"TEST SUMMARY: {total_tests} tests completed")
        print("=" * 70)
        print("\nKey observations to verify:")
        print("   1. OFF-TOPIC: Should show 'DIRECT RESPONSE' (no agents called)")
        print("   2. SINGLE AGENT: Should show only ONE agent being called")
        print("   3. MULTI-AGENT: Should show multiple agents + PLANNER synthesizing")
        print("\n[OK] All tests completed!")

    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback

        traceback.print_exc()
        print("\n[Tips]:")
        print("   1. Ensure API keys are set in .env file")
        print("   2. Check LM Studio is running for local models")
        print("   3. Check network connectivity for cloud providers")
