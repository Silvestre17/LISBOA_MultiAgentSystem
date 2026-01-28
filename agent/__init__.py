# ==========================================================================
# Master Thesis - Agent Package
#   - André Filipe Gomes Silvestre, 20240502
# ==========================================================================

from agent.graph import (
    LisbonAssistant,
    create_assistant,
    quick_chat,
    build_agent_graph,
    get_all_tools,
    MultiAgentAssistant,
    create_multiagent_assistant
)

from agent.state import (
    AgentState,
    UserContext,
    WeatherContext,
    TransportContext,
    PlanItem,
    create_initial_state
)

from agent.prompts import (
    get_system_prompt,
    SYSTEM_PROMPT_EN, SYSTEM_PROMPT_PT,
    ITINERARY_PLANNING_PROMPT,
    WEATHER_ANALYSIS_PROMPT,
    TRANSPORT_ANALYSIS_PROMPT,
    # Multi-Agent Prompts
    SUPERVISOR_PROMPT_EN, SUPERVISOR_PROMPT_PT, get_supervisor_prompt,
    WEATHER_AGENT_PROMPT, get_weather_prompt,
    TRANSPORT_AGENT_PROMPT, get_transport_prompt,
    RESEARCHER_AGENT_PROMPT, get_researcher_prompt,
    PLANNER_AGENT_PROMPT, get_planner_prompt
)

from agent.llm_factory import LLMFactory

__all__ = [
    # Graph
    "LisbonAssistant",
    "create_assistant",
    "quick_chat",
    "build_agent_graph",
    "get_all_tools",
    "MultiAgentAssistant",
    "create_multiagent_assistant",
    
    # State
    "AgentState",
    "UserContext",
    "WeatherContext",
    "TransportContext",
    "PlanItem",
    "create_initial_state",
    
    # Prompts
    "get_system_prompt",
    "SYSTEM_PROMPT_EN",
    "SYSTEM_PROMPT_PT",
    "ITINERARY_PLANNING_PROMPT",
    "WEATHER_ANALYSIS_PROMPT",
    "TRANSPORT_ANALYSIS_PROMPT",
    
    # Multi-Agent Prompts
    "SUPERVISOR_PROMPT_EN", "SUPERVISOR_PROMPT_PT", "get_supervisor_prompt",
    "WEATHER_AGENT_PROMPT", "get_weather_prompt",
    "TRANSPORT_AGENT_PROMPT", "get_transport_prompt",
    "RESEARCHER_AGENT_PROMPT", "get_researcher_prompt",
    "PLANNER_AGENT_PROMPT", "get_planner_prompt",
    
    # LLM
    "LLMFactory"
]
