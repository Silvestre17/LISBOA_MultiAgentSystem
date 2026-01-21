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

import os
import sys
import re
import json
from typing import List, Set, Tuple, Callable, Optional

# LangSmith tracing support (optional - graceful fallback if not available)
try:
    from langsmith.run_helpers import traceable, get_current_run_tree
    from langsmith import ContextThreadPoolExecutor
    from langsmith.run_trees import RunTree
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    # Fallback: no-op decorator and standard ThreadPoolExecutor
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def get_current_run_tree():
        return None
    RunTree = None
    from concurrent.futures import ThreadPoolExecutor as ContextThreadPoolExecutor

# Always need as_completed for collecting parallel results
from concurrent.futures import as_completed
import time as time_module  # For latency tracking

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.state import AgentState, create_initial_state
from agent.prompts import get_system_prompt
from agent.llm_factory import LLMFactory
from agent.agents.base import clean_response

# Import tools
from tools.ipma_api import (
    get_weather_warnings,
    get_weather_forecast,
    get_current_weather_summary,
    get_portugal_weather_overview   # Weather for all Portugal locations
)
from tools.transport_api import (
    get_metro_status,
    get_metro_wait_time,           # Real-time metro wait times
    get_metro_line_wait_times,     # Wait times for entire line
    find_nearest_metro,            # Find nearest metro station by GPS
    get_metro_frequency,           # Train frequency schedules
    get_all_metro_stations,        # List all metro stations
    get_carris_metropolitana_alerts,
    get_carris_metropolitana_stop_info,
    search_carris_metropolitana_lines,
    get_train_status,
    get_transport_summary,
    get_route_between_stations,
    find_bus_routes,               # Bus routing between locations
    get_bus_realtime_locations,    # Real-time bus GPS locations
    get_bus_next_departures,       # Bus route schedule/stops
    search_cp_stations             # CP train station search (AML)
)
from tools.dados_abertos import (
    find_nearby_services,
    list_available_datasets,
    get_dataset_details,
    find_place_in_datasets          # Search places by name across datasets
)
from tools.visitlisboa_api import (
    search_cultural_events,
    search_places_attractions,
    get_event_categories,
    get_place_categories,
    search_lisbon_knowledge
)
from tools.carris_api import (
    carris_get_stops,
    carris_get_routes,
    carris_get_next_departures,
    carris_find_routes_between,
    carris_get_realtime_vehicles,
)


# ==========================================================================
# Response Cleaning
# ==========================================================================

def _clean_response(content: str) -> str:
    """
    Cleans model-specific artifacts from the response before displaying to user.
    
    Removes:
        - <think>...</think> blocks (Qwen3 reasoning)
        - <tool_call>...</tool_call> blocks (some models embed these in text)
        - Embedded JSON tool call syntax
        - <|im_start|>, <|im_end|> tokens (chat template artifacts)
        - Leading/trailing whitespace
    
    Args:
        content: Raw response from the LLM.
        
    Returns:
        str: Cleaned response suitable for user display.
    """
    if not content:
        return content
    
    # Remove <think>...</think> blocks (Qwen3 reasoning) - handles multiline
    content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)
    
    # Remove <tool_call>...</tool_call> blocks that some models embed in text
    content = re.sub(r'</?tool_call>\s*', '', content, flags=re.DOTALL)
    
    # Remove embedded JSON tool call syntax (e.g., {"name": "...", "arguments": {...}})
    content = re.sub(r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}', '', content)
    
    # Remove chat template tokens that might leak through
    content = re.sub(r'<\|im_start\|>.*?\n?', '', content)
    content = re.sub(r'<\|im_end\|>\s*', '', content)
    
    # Remove any other common model tokens
    content = re.sub(r'<\|.*?\|>\s*', '', content)
    
    # Clean up excess whitespace
    content = content.strip()
    
    return content



# ==========================================================================
# Tool Configuration
# ==========================================================================

def get_all_tools() -> List:
    """
    Returns all available tools for the agent.
    
    Returns:
        List: List of LangChain tools.
    """
    return [
        # Weather Tools (IPMA)
        get_weather_warnings,
        get_weather_forecast,
        get_current_weather_summary,
        get_portugal_weather_overview,  # Weather for all Portugal
        
        # Transport Tools
        get_metro_status,
        get_metro_wait_time,          # Real-time metro wait times
        get_metro_line_wait_times,    # Wait times for entire line
        find_nearest_metro,           # Find nearest metro by GPS
        get_metro_frequency,          # Train frequency schedules
        get_all_metro_stations,       # List all metro stations
        get_carris_metropolitana_alerts,
        get_carris_metropolitana_stop_info,
        search_carris_metropolitana_lines,
        get_train_status,
        get_transport_summary,
        get_route_between_stations,   # Metro routing assistance
        find_bus_routes,              # Bus routing between locations
        get_bus_realtime_locations,   # Real-time bus GPS locations
        get_bus_next_departures,      # Bus route schedule/stops
        search_cp_stations,           # CP train station search (AML)
        
        # Open Data Tools (Lisboa Aberta)
        find_nearby_services,
        list_available_datasets,
        get_dataset_details,
        find_place_in_datasets,       # Search places by name
        
        # VisitLisboa Tools (Events & Places) - Semantic Search
        search_cultural_events,
        search_places_attractions,
        get_event_categories,
        get_place_categories,
        search_lisbon_knowledge,  # Comprehensive RAG search
        
        # Carris Urban Tools (Lisbon city buses & trams)
        carris_get_stops,             # Search Carris urban stops
        carris_get_routes,            # Get bus/tram routes (701, 15E, etc.)
        carris_get_next_departures,   # Schedule for a stop
        carris_find_routes_between,   # Find routes connecting two areas
        carris_get_realtime_vehicles, # Real-time bus/tram positions
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
        args_str = str(getattr(tool_call, 'args', {}))
    
    name = tool_call.get("name", "") if isinstance(tool_call, dict) else getattr(tool_call, 'name', '')
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
            system_msg = SystemMessage(content=get_system_prompt())
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
        return AIMessage(content="I apologize, but I'm having trouble processing your request. Could you please try rephrasing your question?")


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
    
    # Create a base LLM without tools for fallback responses
    llm_base = LLMFactory.get_llm(provider) if provider else LLMFactory.get_llm()
    
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
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
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
        
        # Get model info for display
        llm = LLMFactory.get_llm(provider) if provider else LLMFactory.get_llm()
        self.model_name = LLMFactory.get_model_info(llm)
    
    def chat(self, message: str) -> str:
        """
        Sends a message to the assistant and gets a response.
        
        Args:
            message (str): User message.
            
        Returns:
            str: Assistant response.
        """
        # Add user message to state
        self.state["messages"].append(HumanMessage(content=message))
        
        # Run the graph with recursion limit to prevent infinite loops
        # Default LangGraph limit is 25, increased for smaller models with tool calling
        result = self.graph.invoke(
            self.state,
            config={"recursion_limit": 25}
        )
        
        # Update state with result
        self.state = result
        
        # Extract the final response
        last_message = result["messages"][-1]
        
        if hasattr(last_message, "content"):
            # Clean model-specific artifacts (thinking tags, chat tokens, etc.)
            return clean_response(last_message.content)
        
        return clean_response(str(last_message))
    
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
        provider (str, optional): LLM provider ('groq', 'google', 'openai', etc.).
        
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
        from agent.agents.supervisor import SupervisorAgent
        from agent.agents.weather_agent import WeatherAgent
        from agent.agents.transport_agent import TransportAgent
        from agent.agents.researcher_agent import ResearcherAgent
        from agent.agents.planner_agent import PlannerAgent
        from agent.state import create_initial_state
        
        # Initialize agents
        self.supervisor = SupervisorAgent()
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
            **{name: agent.get_model_info() for name, agent in self.agents.items()}
        }
    
    @property
    def model_name(self) -> str:
        """Returns the model name for display (compatibility with V1 app)."""
        sv_model = self.model_info.get("supervisor", "Unknown")
        return f"Multi-Agent ({sv_model})"
    
    @traceable(name="multi_agent_chat", run_type="chain")
    def chat(self, message: str, verbose: bool = False, on_status_change: Optional[Callable[[str], None]] = None) -> str:
        """
        Processes a user message using the multi-agent system.
        
        Uses @traceable decorator to create a single parent trace in LangSmith
        that encompasses all agent and tool calls. The ContextThreadPoolExecutor
        ensures proper context propagation across parallel agent executions.
        
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
        
        # Notify status: Routing
        if on_status_change:
            on_status_change("🤔 A analisar o pedido...")
        
        # Step 1: Route the query
        routing = self.supervisor.route(message)
        agents_to_call = routing.get("agents", [])
        direct_response = routing.get("direct_response")
        reasoning = routing.get("reasoning", "")
        
        if verbose:
            print(f"\n   [ROUTING] Supervisor decision:")
            print(f"      Reasoning: {reasoning}")
            print(f"      Agents: {agents_to_call if agents_to_call else 'None (direct response)'}")
        
        # Inject LangSmith metadata and tags based on routing decision
        if LANGSMITH_AVAILABLE and agents_to_call:
            try:
                run_tree = get_current_run_tree()
                if run_tree:
                    # Ensure metadata structure exists
                    if not hasattr(run_tree, 'extra') or run_tree.extra is None:
                        run_tree.extra = {}
                    if 'metadata' not in run_tree.extra:
                        run_tree.extra['metadata'] = {}
                    
                    # Add routing metadata
                    run_tree.extra['metadata']['agents_called'] = agents_to_call
                    run_tree.extra['metadata']['num_agents'] = len(agents_to_call)
                    run_tree.extra['metadata']['supervisor_reasoning'] = reasoning[:200]  # Truncate
                    
                    # Determine query type tags
                    query_tags = []
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
                    
                    run_tree.extra['metadata']['query_tags'] = query_tags
                    
                    # Add tags to run_tree if supported
                    if hasattr(run_tree, 'tags') and run_tree.tags is not None:
                        run_tree.tags.extend(query_tags)
                    elif hasattr(run_tree, 'tags'):
                        run_tree.tags = query_tags
            except Exception:
                pass  # Silently ignore metadata errors
        
        # Notify status: Agents selected
        if agents_to_call:
            # Map internal names to friendly PT names
            name_map = {
                "weather": "Meteorologia 🌤️",
                "transport": "Transportes 🚇",
                "researcher": "Pesquisa Local 🔎",
                "planner": "Planeador 📅"
            }
            # Filter out planner from the "Consulting" list as it runs last
            consulting = [name_map.get(a, a) for a in agents_to_call if a != "planner"]
            
            if consulting and on_status_change:
                on_status_change(f"🚀 Vou consultar: {', '.join(consulting)}...")
        
        # Step 2: Handle direct response (no agents needed)
        if direct_response and not agents_to_call:
            if verbose:
                print(f"      Mode: DIRECT RESPONSE (no agents called)")
            return clean_response(direct_response)
        
        # Step 3: Execute agents (Parallelized with LangSmith context propagation)
        agent_outputs = {}
        
        # Identify worker agents (exclude planner which runs last)
        workers = [a for a in agents_to_call if a != "planner" and a in self.agents]
        
        if workers:
            if verbose:
                print(f"      [PARALLEL] Executing {len(workers)} agents: {workers}")
            
            if on_status_change:
                on_status_change(f"⏳ A aguardar respostas de: {', '.join(workers)}...")
            
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
                    future_to_agent[executor.submit(
                        self.agents[agent_name].invoke, 
                        message, 
                        "",     # context
                        verbose # verbose flag
                    )] = agent_name
                
                # Collect results as they complete with latency tracking
                for future in as_completed(future_to_agent):
                    agent_name = future_to_agent[future]
                    agent_latency = time_module.time() - agent_start_times[agent_name]
                    
                    try:
                        output = future.result()
                        agent_outputs[agent_name] = output
                        
                        # Log latency to LangSmith metadata if available
                        if LANGSMITH_AVAILABLE:
                            try:
                                run_tree = get_current_run_tree()
                                if run_tree:
                                    if not hasattr(run_tree, 'extra') or run_tree.extra is None:
                                        run_tree.extra = {}
                                    if 'metadata' not in run_tree.extra:
                                        run_tree.extra['metadata'] = {}
                                    run_tree.extra['metadata'][f'agent_{agent_name}_latency_ms'] = int(agent_latency * 1000)
                                    run_tree.extra['metadata'][f'agent_{agent_name}_output_chars'] = len(output)
                            except Exception:
                                pass  # Silently ignore metadata errors
                        
                        if verbose:
                            print(f"   [AGENT: {agent_name.upper()}] Finished ({len(output)} chars, {agent_latency:.2f}s)")
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        agent_outputs[agent_name] = error_msg
                        if verbose:
                            print(f"   [AGENT: {agent_name.upper()}] Failed: {str(e)}")
        
        # Step 4: If Planner was requested, synthesize final response
        if "planner" in agents_to_call:
            if verbose:
                print(f"\n   [AGENT: PLANNER] Synthesizing from {list(agent_outputs.keys())}...")
            
            if on_status_change:
                on_status_change("✍️ A escrever o itinerário final...")
                
            response = self.agents["planner"].synthesize(message, agent_outputs)
        elif agent_outputs:
            # Combine agent outputs if no planner
            response = self._combine_outputs(agent_outputs)
        else:
            # Fallback: Use researcher for general queries
            if verbose:
                print(f"\n   [FALLBACK] Using researcher agent")
            response = self.agents["researcher"].invoke(message, verbose=verbose)
        
        return clean_response(response)
    
    def _combine_outputs(self, agent_outputs: dict) -> str:
        """
        Combines outputs from multiple agents into a single response.
        
        Args:
            agent_outputs: Dict mapping agent names to their outputs.
            
        Returns:
            str: Combined response.
        """
        sections = []
        
        if "weather" in agent_outputs:
            sections.append(agent_outputs["weather"])
        
        if "researcher" in agent_outputs:
            sections.append(agent_outputs["researcher"])
        
        if "transport" in agent_outputs:
            sections.append(agent_outputs["transport"])
        
        return "\n\n---\n\n".join(sections)
    
    def reset(self):
        """Resets the conversation state."""
        from agent.state import create_initial_state
        self.state = create_initial_state()
    
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
    from config import Config
    import sys
    
    # Fix Windows console encoding
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
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
                    ("What is the capital of France?", "Off-topic - should decline or answer directly"),
                    ("Who won the World Cup in 2022?", "Off-topic - not about Lisbon"),
                ]
            },
            # ---------------------------------------------------------
            # CATEGORY 2: SINGLE AGENT - WEATHER ONLY
            # ---------------------------------------------------------
            {
                "category": "SINGLE AGENT - WEATHER",
                "description": "Weather questions - should ONLY call weather agent",
                "queries": [
                    ("What's the weather in Lisbon today?", "Should use only weather agent"),
                    ("Is it going to rain tomorrow?", "Simple forecast - weather only"),
                ]
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
                ]
            },
            # ---------------------------------------------------------
            # CATEGORY 4: SINGLE AGENT - RESEARCHER ONLY
            # ---------------------------------------------------------
            {
                "category": "SINGLE AGENT - RESEARCHER",
                "description": "Places/events questions - should ONLY call researcher agent",
                "queries": [
                    ("What are the best museums in Lisbon?", "Places search - researcher only"),
                    ("Are there any events today?", "Events search - researcher only"),
                ]
            },
            # ---------------------------------------------------------
            # CATEGORY 5: MULTI-AGENT - COMPLEX QUERIES
            # ---------------------------------------------------------
            {
                "category": "MULTI-AGENT - COMPLEX",
                "description": "Complex queries requiring multiple agents + planner",
                "queries": [
                    ("Plan my day visiting museums, considering the weather", "Needs weather + researcher + planner"),
                    ("Suggest outdoor activities for today based on weather", "Needs weather + researcher + planner"),
                ]
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
                
                print(f"\n[RESPONSE]:")
                # Truncate long responses
                if len(response) > 600:
                    print(response[:600] + "\n   [...truncated...]")
                else:
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
        print("   3. Check network connectivity for Groq API")
