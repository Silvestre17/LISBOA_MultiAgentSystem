# ==========================================================================
# Master Thesis - Agent Package
#   - André Filipe Gomes Silvestre, 20240502
# ==========================================================================

from agent.graph import (
    LisbonAssistant,
    create_assistant,
    quick_chat,
    build_agent_graph,
    get_all_tools
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
    SYSTEM_PROMPT,
    ITINERARY_PLANNING_PROMPT,
    WEATHER_ANALYSIS_PROMPT,
    TRANSPORT_ANALYSIS_PROMPT
)

from agent.llm_factory import LLMFactory

__all__ = [
    # Graph
    "LisbonAssistant",
    "create_assistant",
    "quick_chat",
    "build_agent_graph",
    "get_all_tools",
    
    # State
    "AgentState",
    "UserContext",
    "WeatherContext",
    "TransportContext",
    "PlanItem",
    "create_initial_state",
    
    # Prompts
    "get_system_prompt",
    "SYSTEM_PROMPT",
    "ITINERARY_PLANNING_PROMPT",
    "WEATHER_ANALYSIS_PROMPT",
    "TRANSPORT_ANALYSIS_PROMPT",
    
    # LLM
    "LLMFactory"
]
