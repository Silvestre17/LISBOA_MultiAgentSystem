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
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.state import AgentState, create_initial_state
from agent.prompts import get_system_prompt
from agent.llm_factory import LLMFactory

# Import tools
from tools.ipma_api import (
    get_weather_warnings,
    get_weather_forecast,
    get_current_weather_summary
)
from tools.transport_api import (
    get_metro_status,
    get_metro_wait_time,           # Real-time metro wait times
    get_metro_line_wait_times,     # Wait times for entire line
    find_nearest_metro,            # Find nearest metro station by GPS
    get_metro_frequency,           # Train frequency schedules
    get_all_metro_stations,        # List all metro stations
    get_carris_alerts,
    get_carris_stop_info,
    search_carris_lines,
    get_train_status,
    get_transport_summary,
    get_route_between_stations,
    find_bus_routes,               # Bus routing between locations
    get_bus_realtime_locations,    # Real-time bus GPS locations
    get_bus_schedule,              # Bus route schedule/stops
    search_cp_stations             # CP train station search (AML)
)
from tools.dados_abertos import (
    find_nearby_services,
    list_available_datasets,
    get_dataset_details
)
from tools.visitlisboa_api import (
    search_cultural_events,
    search_places_attractions,
    get_event_categories,
    get_place_categories,
    search_lisbon_knowledge
)


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
        # Weather Tools
        get_weather_warnings,
        get_weather_forecast,
        get_current_weather_summary,
        
        # Transport Tools
        get_metro_status,
        get_metro_wait_time,          # Real-time metro wait times
        get_metro_line_wait_times,    # Wait times for entire line
        find_nearest_metro,           # Find nearest metro by GPS
        get_metro_frequency,          # Train frequency schedules
        get_all_metro_stations,       # List all metro stations
        get_carris_alerts,
        get_carris_stop_info,
        search_carris_lines,
        get_train_status,
        get_transport_summary,
        get_route_between_stations,   # Metro routing assistance
        find_bus_routes,              # Bus routing between locations
        get_bus_realtime_locations,   # Real-time bus GPS locations
        get_bus_schedule,             # Bus route schedule/stops
        search_cp_stations,           # CP train station search (AML)
        
        # Open Data Tools (Lisboa Aberta)
        find_nearby_services,
        list_available_datasets,
        get_dataset_details,
        
        # VisitLisboa Tools (Events & Places) - Semantic Search
        search_cultural_events,
        search_places_attractions,
        get_event_categories,
        get_place_categories,
        search_lisbon_knowledge,  # Comprehensive RAG search
    ]


# ==========================================================================
# Agent Nodes
# ==========================================================================

def create_agent_node(llm_with_tools):
    """
    Creates the main agent node that decides actions.
    
    Args:
        llm_with_tools: LLM instance with tools bound.
        
    Returns:
        Callable: Agent node function.
    """
    def agent_node(state: AgentState) -> dict:
        """
        Main agent decision node.
        
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
        
        # Call the LLM
        response = llm_with_tools.invoke(messages)
        
        return {"messages": [response]}
    
    return agent_node


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
    
    # Get tools and bind to LLM
    tools = get_all_tools()
    llm_with_tools = llm.bind_tools(tools)
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", create_agent_node(llm_with_tools))
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
        # Default LangGraph limit is 25, we set 15 for safety
        result = self.graph.invoke(
            self.state,
            config={"recursion_limit": 15}
        )
        
        # Update state with result
        self.state = result
        
        # Extract the final response
        last_message = result["messages"][-1]
        
        if hasattr(last_message, "content"):
            return last_message.content
        
        return str(last_message)
    
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
# Test Block
# ==========================================================================
if __name__ == "__main__":
    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 LangGraph Agent Test\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")
    
    try:
        # Initialize assistant
        print("\n\033[1m🔄 Initializing Lisbon Assistant...\033[0m")
        assistant = create_assistant()
        print(f"\033[1;32m✅ Ready!\033[0m Model: {assistant.model_name}")
        
        # Test queries
        test_queries = [
            "What's the weather like in Lisbon today?",
            "Is the metro working?",
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n\033[1m📝 Test {i}: {query}\033[0m")
            print("-" * 40)
            
            response = assistant.chat(query)
            print(f"\n\033[1m🤖 Response:\033[0m")
            print(response[:1000] + "..." if len(response) > 1000 else response)
            
            # Reset for next test
            assistant.reset()
        
        print(f"\n\033[1;32m✅ All tests completed!\033[0m")
        
    except Exception as e:
        print(f"\n\033[1;31m❌ Error:\033[0m {e}")
        print("\n\033[1m💡 Tips:\033[0m")
        print("   1. Ensure API keys are set in .env file or local LLM are running.")
        print("   2. Check network connectivity.")
