# ==========================================================================
# Master Thesis - Multi-Agent Prompts Package
#   - André Filipe Gomes Silvestre, 20240502
# 
#   Specialized prompts for each agent in the Multi-Agent System.
#   Each agent has a focused, concise prompt optimized for its task.
#   
#   Also re-exports the original single-agent system prompt for
#   backward compatibility.
# ==========================================================================

# Import from specialized agent prompts
from agent.prompts.supervisor import SUPERVISOR_PROMPT, get_supervisor_prompt
from agent.prompts.weather import WEATHER_AGENT_PROMPT, get_weather_prompt
from agent.prompts.transport import TRANSPORT_AGENT_PROMPT, get_transport_prompt
from agent.prompts.researcher import RESEARCHER_AGENT_PROMPT, get_researcher_prompt
from agent.prompts.planner import PLANNER_AGENT_PROMPT, get_planner_prompt

# Import from original system prompt module (renamed to _system_prompt.py)
# This maintains backward compatibility with existing code
from agent.prompts._system_prompt import (
    SYSTEM_PROMPT,
    COMPACT_SYSTEM_PROMPT,
    ITINERARY_PLANNING_PROMPT,
    WEATHER_ANALYSIS_PROMPT,
    TRANSPORT_ANALYSIS_PROMPT,
    API_ERROR_RESPONSE,
    NO_DATA_RESPONSE,
    get_system_prompt
)

__all__ = [
    # Original single-agent prompts (backward compatibility)
    "SYSTEM_PROMPT",
    "COMPACT_SYSTEM_PROMPT",
    "ITINERARY_PLANNING_PROMPT",
    "WEATHER_ANALYSIS_PROMPT",
    "TRANSPORT_ANALYSIS_PROMPT",
    "API_ERROR_RESPONSE",
    "NO_DATA_RESPONSE",
    "get_system_prompt",
    
    # Multi-agent specialized prompts
    "SUPERVISOR_PROMPT",
    "get_supervisor_prompt",
    "WEATHER_AGENT_PROMPT", 
    "get_weather_prompt",
    "TRANSPORT_AGENT_PROMPT",
    "get_transport_prompt",
    "RESEARCHER_AGENT_PROMPT",
    "get_researcher_prompt",
    "PLANNER_AGENT_PROMPT",
    "get_planner_prompt",
]
