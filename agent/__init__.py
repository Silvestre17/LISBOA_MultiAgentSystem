# ==========================================================================
# Master Thesis - Agent Package
#   - André Filipe Gomes Silvestre, 20240502
# ==========================================================================

import agent.agents as agents
from agent.graph import (
    MultiAgentAssistant,
    create_multiagent_assistant,
    get_all_tools,
)
from agent.llm_factory import LLMFactory
from agent.prompts import (  # Multi-Agent Prompts
    PLANNER_AGENT_PROMPT,
    QA_AGENT_PROMPT_EN,
    QA_AGENT_PROMPT_PT,
    RESEARCHER_AGENT_PROMPT,
    SUPERVISOR_PROMPT_EN,
    SUPERVISOR_PROMPT_PT,
    TRANSPORT_AGENT_PROMPT,
    WEATHER_AGENT_PROMPT,
    get_planner_prompt,
    get_qa_prompt,
    get_researcher_prompt,
    get_supervisor_prompt,
    get_transport_prompt,
    get_weather_prompt,
)
from agent.state import (
    AgentState,
    TransportContext,
    UserContext,
    WeatherContext,
    create_initial_state,
)

__all__ = [
    # Graph
    "get_all_tools",
    "MultiAgentAssistant",
    "create_multiagent_assistant",

    # State
    "AgentState",
    "UserContext",
    "WeatherContext",
    "TransportContext",
    "create_initial_state",

    # Multi-Agent Prompts
    "SUPERVISOR_PROMPT_EN", "SUPERVISOR_PROMPT_PT", "get_supervisor_prompt",
    "WEATHER_AGENT_PROMPT", "get_weather_prompt",
    "TRANSPORT_AGENT_PROMPT", "get_transport_prompt",
    "RESEARCHER_AGENT_PROMPT", "get_researcher_prompt",
    "PLANNER_AGENT_PROMPT", "get_planner_prompt",
    "QA_AGENT_PROMPT_EN", "QA_AGENT_PROMPT_PT", "get_qa_prompt",

    # LLM
    "LLMFactory",

    # Subpackages
    "agents",
]
