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
            get_current_weather_summary
        )
        return [
            get_weather_warnings,
            get_weather_forecast,
            get_current_weather_summary,
        ]
    
    elif agent_name == "transport":
        from tools.transport_api import (
            get_metro_status,
            get_metro_wait_time,
            get_metro_line_wait_times,
            find_nearest_metro,
            get_metro_frequency,
            get_all_metro_stations,
            get_carris_alerts,
            get_carris_stop_info,
            search_carris_lines,
            get_train_status,
            get_transport_summary,
            get_route_between_stations,
            find_bus_routes,
            get_bus_realtime_locations,
            get_bus_schedule,
            search_cp_stations
        )
        return [
            get_metro_status,
            get_metro_wait_time,
            get_metro_line_wait_times,
            find_nearest_metro,
            get_metro_frequency,
            get_all_metro_stations,
            get_carris_alerts,
            get_carris_stop_info,
            search_carris_lines,
            get_train_status,
            get_transport_summary,
            get_route_between_stations,
            find_bus_routes,
            get_bus_realtime_locations,
            get_bus_schedule,
            search_cp_stations,
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
            get_dataset_details
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
        
    Args:
        content: Raw response from the LLM.
        
    Returns:
        str: Cleaned response suitable for user display.
    """
    if not content:
        return content
    
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
    
    def get_model_info(self) -> str:
        """Returns the model name being used by this agent."""
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
            for t in tools[:3]:
                print(f"   - {t.name}")
            if len(tools) > 3:
                print(f"   - ... and {len(tools) - 3} more")
    
    print(f"\n\033[1;32m✅ Base utilities loaded successfully!\033[0m")
