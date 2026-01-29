# ==========================================================================
# Master Thesis - Base Agent Utilities
#   - André Filipe Gomes Silvestre, 20240502
# 
#   Shared utilities for all specialized agents.
#   Provides common functionality for tool binding, LLM creation, and
#   response cleaning.
# ==========================================================================

import os
import sys
import re
from typing import List, Dict, Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import Config

# ==========================================================================
# LangSmith Tracing Support
# ==========================================================================
try:
    from langsmith.run_helpers import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    # Fallback: no-op decorator
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# ==========================================================================
# Tool Definitions by Agent
# ==========================================================================

def get_agent_tools(agent_name: str) -> List:
    """
    Returns the tools for a specific agent.
    
    This function lazily imports tools to avoid circular imports
    and only loads what's needed for each agent.
    
    Args:
        agent_name (str): Name of the agent ('weather', 'transport', 'researcher', 'planner')
        
    Returns:
        List: List of LangChain tools for the specified agent.
    """
    if agent_name == "weather":
        from tools.ipma_api import (
            get_weather_warnings,
            get_weather_forecast,
            get_current_weather_summary,
            get_portugal_weather_overview
        )
        return [
            get_weather_warnings,
            get_weather_forecast,
            get_current_weather_summary,
            get_portugal_weather_overview,
        ]
    
    elif agent_name == "transport":
        # Metro de Lisboa (Official API with OAuth2)
        from tools.metrolisboa_api import (
            get_metro_status,
            get_metro_wait_time,
            get_metro_line_wait_times,
            find_nearest_metro,
            get_metro_frequency,
            get_all_metro_stations,
        )
        # Carris Metropolitana (Suburban buses)
        from tools.carrismetropolitana_api import (
            get_carris_metropolitana_alerts,
            get_carris_metropolitana_stop_info,
            search_carris_metropolitana_lines,
            find_bus_routes,
            find_direct_bus_lines,
            get_bus_realtime_locations,
            get_bus_next_departures,
        )
        # CP (Comboios de Portugal) - Trains
        from tools.cp_api import (
            get_train_status,
            search_cp_stations,
            plan_train_trip,
            get_train_schedule,
        )
        # Multi-modal transport routing
        from tools.transport_api import (
            get_transport_summary,
            get_route_between_stations,
        )
        from tools.carris_api import (
            carris_get_stops,
            carris_get_routes,
            carris_get_next_departures,
            carris_find_routes_between,
            carris_get_realtime_vehicles,
            carris_get_arrivals,
            carris_vehicle_eta,
        )
        return [
            get_metro_status,
            get_metro_wait_time,
            get_metro_line_wait_times,
            find_nearest_metro,
            get_metro_frequency,
            get_all_metro_stations,
            get_carris_metropolitana_alerts,
            get_carris_metropolitana_stop_info,
            search_carris_metropolitana_lines,
            find_direct_bus_lines,
            get_train_status,
            plan_train_trip,
            get_train_schedule,
            get_transport_summary,
            get_route_between_stations,
            find_bus_routes,
            get_bus_realtime_locations,
            get_bus_next_departures,
            search_cp_stations,
            carris_get_stops,
            carris_get_routes,
            carris_get_next_departures,
            carris_find_routes_between,
            carris_get_realtime_vehicles,
            carris_get_arrivals,
            carris_vehicle_eta,
        ]
    
    elif agent_name == "researcher":
        from tools.visitlisboa_api import (
            search_cultural_events,
            search_places_attractions,
            get_event_categories,
            get_place_categories,
            search_lisbon_knowledge
        )
        from tools.dados_abertos import (
            find_nearby_services,
            list_available_datasets,
            get_dataset_details,
            find_place_in_datasets
        )
        return [
            search_cultural_events,
            search_places_attractions,
            get_event_categories,
            get_place_categories,
            search_lisbon_knowledge,
            find_nearby_services,
            list_available_datasets,
            get_dataset_details,
            find_place_in_datasets,
        ]
    
    elif agent_name == "planner":
        # Planner has no tools - it synthesizes outputs from other agents
        return []
    
    elif agent_name == "supervisor":
        # Supervisor has no tools - it only routes to other agents
        return []
    
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


# ==========================================================================
# LLM Factory for Agents
# ==========================================================================

def get_agent_llm(agent_name: str) -> BaseChatModel:
    """
    Creates an LLM instance configured for a specific agent.
    
    Uses AGENT_MODELS from config.py for per-agent model configuration.
    Falls back to DEFAULT_AGENT_MODEL if not specified.
    
    Args:
        agent_name (str): Name of the agent.
        
    Returns:
        BaseChatModel: Configured LLM for the agent.
    """
    from agent.llm_factory import LLMFactory
    return LLMFactory.get_agent_llm(agent_name)


# ==========================================================================
# Response Cleaning Utilities
# ==========================================================================

def clean_response(content: str) -> str:
    """
    Cleans model-specific artifacts from the response.
    
    Removes:
        - <think>...</think> blocks (Qwen3 reasoning)
        - <tool_call>...</tool_call> blocks
        - Embedded JSON tool call syntax
        - Chat template tokens
        - Qwen3 "thinking out loud" patterns (e.g., "How do I..." followed by reasoning)
        - Step-by-step internal reasoning ("Step 1:", "Wait -", etc.)
        
    Args:
        content: Raw response from the LLM.
        
    Returns:
        str: Cleaned response suitable for user display.
    """
    if not content:
        return content
    
    # CRITICAL: Detect and remove Qwen3 "thinking out loud" pattern
    # Pattern: Model starts answering a DIFFERENT question and reasons through it
    # Example: "How do I get to airport from Rossio?\n\nWe are in English...\n\nStep-by-step:..."
    
    # FIRST: Check if entire response is a "thinking" block about a wrong question
    # This is the CRITICAL fix for the hallucination bug where the model answers
    # a completely different question than what was asked
    wrong_question_patterns = [
        # Full response is about getting to airport when that wasn't the question
        r'^How do I get to (?:the )?airport.*$',
        # Model "thinking" about the question
        r'^We are in (?:English|Portuguese)\.\s*The user wants to.*$',
        # Internal planning that leaked through
        r'^Step-by-step:.*$',
        # "Note:" at the very start indicates internal reasoning
        r'^Note:.*Rossio is a major station.*$',
        # Important internal marker
        r'^Important:.*(?:is served by|does NOT).*$',
    ]
    
    for pattern in wrong_question_patterns:
        if re.match(pattern, content, flags=re.DOTALL | re.IGNORECASE):
            # The entire response is internal reasoning - return error message
            return "Sorry, I'm having difficulty processing your request. Please try again."
    
    thinking_patterns = [
        # "How do I..." followed by step-by-step reasoning (different question hallucination)
        r'^How do I [^?]+\?\s*(?:\n.*)?We are in (?:English|Portuguese).*$',
        # "Step-by-step:" internal reasoning
        r'Step-by-step:\s*\n.*(?:Check if|If not|Use tools).*',
        # "Wait -" reasoning pattern
        r'\n\s*Wait\s*[-–]\s*.*(?:\n.*)*',
        # "But wait" reasoning pattern  
        r'\n\s*But wait\s*[-–]?\s*.*(?:\n.*)*',
        # "Let me check" / "Let me recheck" internal reasoning
        r'\n\s*Let me (?:check|recheck).*(?:\n.*)*',
        # "Therefore," followed by internal logic
        r'\n\s*Therefore,\s*(?:I must|we must|from|the).*(?:\n.*)*',
        # "So final response:" marker
        r'\n\s*So final response:.*(?:\n.*)*',
        # "Final output:" marker
        r'\n\s*Final output[:\s].*(?:\n.*)*',
        # Checkmarks at the end of reasoning
        r'\n\s*✅\s*(?:Language|No origin|Clear)[^\n]*(?:\n.*)*$',
        # "Ah!" discovery pattern
        r'\n\s*Ah!.*(?:\n.*)*',
        # "So from Rossio to..." planning pattern
        r'\n\s*So from [A-Z][a-z]+ to (?:airport|[A-Z]).*(?:\n.*)*',
        # "But is there a..." questioning pattern
        r'\n\s*But is there a.*(?:\n.*)*',
        # "This is correct and follows all rules" reasoning marker
        r'\n\s*This is correct and follows all rules.*(?:\n.*)*',
        # "No hallucination" marker
        r'\n\s*No hallucination.*(?:\n.*)*',
        # "The CP train lines:" internal knowledge dump
        r'\n\s*The CP train lines:.*(?:\n.*)*',
        # "The metro lines are:" internal knowledge dump
        r'\n\s*The metro lines are:.*(?:\n.*)*',
        # "Actually, the" reasoning
        r'\n\s*Actually, the.*(?:\n.*)*',
        # "The only train line" reasoning
        r'\n\s*The only train line.*(?:\n.*)*',
    ]
    
    for pattern in thinking_patterns:
        content = re.sub(pattern, '', content, flags=re.DOTALL | re.MULTILINE | re.IGNORECASE)
    
    # Remove <think>...</think> blocks (Qwen3 reasoning) - handles multiline
    content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)
    
    # Remove <tool_call>...</tool_call> blocks
    content = re.sub(r'</?tool_call>\s*', '', content, flags=re.DOTALL)
    
    # Remove embedded JSON tool call syntax
    content = re.sub(r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}', '', content)
    
    # Remove chat template tokens
    content = re.sub(r'<\|im_start\|>.*?\n?', '', content)
    content = re.sub(r'<\|im_end\|>\s*', '', content)
    content = re.sub(r'<\|.*?\|>\s*', '', content)
    
    # Remove specific tool artifacts first
    content = re.sub(r'<tool_code>.*?</tool_code>', '', content, flags=re.DOTALL)
    
    # Strip markdown code blocks if the entire content is wrapped
    # e.g. ```markdown ... ``` or ``` ... ```
    # This prevents Streamlit from rendering the whole response as a code block
    content = re.sub(r'^```(?:markdown|text)?\s*\n', '', content, flags=re.IGNORECASE)
    content = re.sub(r'\n\s*```$', '', content)
    
    # Clean up excess whitespace
    content = content.strip()
    
    # Final check: If content is nearly empty after cleaning, return error
    if len(content) < 20:
        return "Sorry, I'm having difficulty processing your request. Please try again."
    
    return content


def parse_json_response(content: str) -> Optional[Dict[str, Any]]:
    """
    Extracts JSON from a response that may contain markdown code blocks.
    
    Args:
        content: Response text that may contain JSON.
        
    Returns:
        Dict or None: Parsed JSON if found, None otherwise.
    """
    import json
    
    if not content:
        return None
    
    # Clean first
    content = clean_response(content)
    
    # Try to find JSON in code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find raw JSON
    json_match = re.search(r'\{[^{}]*"agents"[^{}]*\}', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Try parsing the entire content as JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


# ==========================================================================
# Base Agent Class
# ==========================================================================

class BaseAgent:
    """
    Base class for all specialized agents.
    
    Provides common functionality for:
        - LLM initialization with agent-specific config
        - Tool binding
        - Response cleaning
        - State management
    """
    
    def __init__(self, agent_name: str):
        """
        Initializes the base agent.
        
        Args:
            agent_name: Name of this agent (for config lookup).
        """
        self.agent_name = agent_name
        self.tools = get_agent_tools(agent_name)
        self.llm = get_agent_llm(agent_name)
        
        # Bind tools if this agent has any
        if self.tools:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        else:
            self.llm_with_tools = self.llm
    
    def get_model_info(self) -> Dict[str, Any]:
        """Returns the model info dictionary."""
        from agent.llm_factory import LLMFactory
        return LLMFactory.get_model_info(self.llm)
    
    def clean_response(self, content: str) -> str:
        """Cleans model artifacts from response."""
        return clean_response(content)


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 Base Agent Utilities Test\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")
    
    # Test tool loading
    for agent in ["weather", "transport", "researcher", "planner", "supervisor"]:
        tools = get_agent_tools(agent)
        print(f"\n\033[1m{agent.capitalize()} Agent:\033[0m {len(tools)} tools")
        if tools:
            for t in tools:
                print(f"   - {t.name}")
    
    print(f"\n\033[1;32m✅ Base utilities loaded successfully!\033[0m")
