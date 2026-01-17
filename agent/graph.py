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
from typing import List, Set, Tuple

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
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
            return _clean_response(last_message.content)
        
        return _clean_response(str(last_message))
    
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
