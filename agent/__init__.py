# ==========================================================================
# Master Thesis - Agent Package
#   - André Filipe Gomes Silvestre, 20240502
# ==========================================================================

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


def __getattr__(name: str):
    """Lazily expose heavy graph/agent objects without import-time cycles.

    Utility modules under ``agent.utils`` are imported by low-level tools. If the
    package imports ``agent.graph`` eagerly here, those tools can end up importing
    the graph while it is still importing the tools. Keeping graph and agent
    classes lazy preserves the public package API without creating circular
    imports during standalone tool tests.
    """
    if name in {"MultiAgentAssistant", "create_multiagent_assistant", "get_all_tools"}:
        from agent.graph import MultiAgentAssistant, create_multiagent_assistant, get_all_tools

        values = {
            "MultiAgentAssistant": MultiAgentAssistant,
            "create_multiagent_assistant": create_multiagent_assistant,
            "get_all_tools": get_all_tools,
        }
        globals().update(values)
        return values[name]
    if name == "agents":
        import agent.agents as _agents

        globals()["agents"] = _agents
        return _agents
    raise AttributeError(f"module 'agent' has no attribute {name!r}")


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
