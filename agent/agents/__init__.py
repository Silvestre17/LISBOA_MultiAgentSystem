# ==========================================================================
# Master Thesis - Multi-Agent Package
#   - André Filipe Gomes Silvestre, 20240502
# 
#   Specialized agents for the Multi-Agent System.
#   Each agent has focused responsibilities and tools.
# ==========================================================================

from agent.agents.base import BaseAgent, get_agent_tools
from agent.agents.planner_agent import PlannerAgent
from agent.agents.qa_agent import QualityAssuranceAgent
from agent.agents.researcher_agent import ResearcherAgent
from agent.agents.supervisor import SupervisorAgent
from agent.agents.transport_agent import TransportAgent
from agent.agents.weather_agent import WeatherAgent

__all__ = [
    "BaseAgent",
    "get_agent_tools",
    "SupervisorAgent",
    "WeatherAgent",
    "TransportAgent",
    "ResearcherAgent",
    "PlannerAgent",
    "QualityAssuranceAgent",
]
