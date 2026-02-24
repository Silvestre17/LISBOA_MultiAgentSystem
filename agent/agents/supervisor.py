# ==========================================================================
# Master Thesis - Supervisor Agent
#   - André Filipe Gomes Silvestre, 20240502
#
#   Smart router that analyzes user intent and decides which specialized
#   agents to invoke. Only calls agents when necessary.
# ==========================================================================

import os
import re
import sys
from typing import Dict, Any, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agent.agents.base import BaseAgent, parse_json_response, clean_response, traceable
from agent.prompts.supervisor import get_supervisor_prompt


class SupervisorAgent(BaseAgent):
    """
    Supervisor agent that routes queries to specialized agents.

    Responsibilities:
        - Analyze user query intent
        - Decide which agents to call (can be 0, 1, or multiple)
        - Handle simple queries directly without calling agents
        - Return routing decisions as structured JSON
    """

    def __init__(self):
        """Initializes the supervisor agent."""
        super().__init__("supervisor")
        # System prompt is now dynamic per request

    @traceable(name="supervisor_agent", run_type="chain", tags=["sub-agent", "supervisor"])
    def route(self, user_message: str, language: str = "en") -> Dict[str, Any]:
        """
        Analyzes user message and returns routing decision.

        Args:
            user_message: The user's query.
            language: Language code ('en' or 'pt').

        Returns:
            Dict with:
                - reasoning: Why these agents were chosen
                - agents: List of agent names to call (can be empty)
                - direct_response: Response if no agents needed (or None)
        """
        system_prompt = get_supervisor_prompt(language)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        # Get routing decision from LLM
        response = self.llm.invoke(messages)
        content = clean_response(response.content)

        # Parse JSON response
        decision = parse_json_response(content)

        if decision:
            agents = decision.get("agents", [])
            reasoning = decision.get("reasoning", "")

            # Check if this is a planning query that requires weather
            planning_keywords = [
                "plan",
                "itinerary",
                "roteiro",
                "plano",
                "day",
                "dia",
                "schedule",
                "agenda",
                "day trip",
                "passeio",
                "visita",
            ]
            is_planning_query = any(
                kw in user_message.lower() for kw in planning_keywords
            )

            # Force weather agent for near-future planning
            if is_planning_query and self._requires_weather_for_planning(user_message):
                if "weather" not in agents:
                    agents.append("weather")
                    reasoning += " (Added weather agent: planning for near-future date)"

            # Enforce rejection for out of scope queries even if LLM tries to answer
            reasoning_lower = reasoning.lower()
            if not agents and any(k in reasoning_lower for k in ["matemática", "math", "fora de âmbito", "out of scope", "trivia", "trivialidade"]):
                decision["direct_response"] = "Sou um assistente especializado na Área Metropolitana de Lisboa, focado apenas em transportes, meteorologia e locais da capital. Não posso ajudar-te com essa questão! 🏙️" if language == "pt" else "I am a specialized assistant for the Lisbon Metropolitan Area, focused only on transports, weather, and local places. I cannot help you with that query! 🏙️"

            return {
                "reasoning": reasoning,
                "agents": agents,
                "direct_response": decision.get("direct_response"),
            }

        # Fallback: If JSON parsing fails, try to extract intent heuristically
        return self._fallback_routing(user_message, content)

    def _fallback_routing(self, user_message: str, llm_response: str) -> Dict[str, Any]:
        """
        Fallback routing when JSON parsing fails.
        Uses simple keyword matching as backup.

        Args:
            user_message: Original user query.
            llm_response: Raw LLM response that failed to parse.

        Returns:
            Dict with routing decision.
        """
        message_lower = user_message.lower()

        # 1. Check for Out-of-Scope keywords (Locations outside AML)
        # Note: AML includes Sintra, Cascais, Montijo, Setúbal, etc. - these are IN SCOPE!
        forbidden_keywords = [
            "porto",
            "aveiro",
            "braga",
            "coimbra",
            "faro",
            "algarve",
            "évora",
            "madrid",
            "paris",
            "london",
            "barcelona",
        ]
        if any(city in message_lower for city in forbidden_keywords):
            return {
                "reasoning": "Fallback: Detected out-of-scope location (outside AML)",
                "agents": [],
                "direct_response": "Sou especializado na Área Metropolitana de Lisboa! Posso ajudar-te com transportes, locais ou eventos na região da capital? 🏙️",
            }

        # 2. AML locations that ARE in scope - should trigger transport agent
        aml_keywords = [
            "sintra",
            "cascais",
            "oeiras",
            "amadora",
            "loures",
            "odivelas",
            "almada",
            "seixal",
            "barreiro",
            "montijo",
            "alcochete",
            "setúbal",
            "palmela",
            "sesimbra",
            "mafra",
            "vila franca",
        ]
        if any(loc in message_lower for loc in aml_keywords):
            # These are AML locations - use transport agent
            return {
                "reasoning": "Fallback: AML location detected - using transport agent",
                "agents": ["transport"],
                "direct_response": None,
            }

        # Weather keywords
        weather_keywords = [
            "weather",
            "rain",
            "temperature",
            "chover",
            "tempo",
            "meteo",
            "previsão",
            "sol",
            "chuva",
            "temperatura",
            "forecast",
        ]

        # Transport keywords
        transport_keywords = [
            "metro",
            "bus",
            "train",
            "carris",
            "comboio",
            "autocarro",
            "route",
            "rota",
            "como chego",
            "how to get",
            "transporte",
            "ferry",
            "barco",
            "fertagus",
            "cp",
        ]

        # Places/Events keywords
        places_keywords = [
            "museum",
            "restaurant",
            "park",
            "museu",
            "restaurante",
            "parque",
            "visit",
            "visitar",
            "event",
            "evento",
            "attraction",
            "atração",
            "what to do",
            "o que fazer",
            "places",
        ]

        # Itinerary keywords
        itinerary_keywords = [
            "plan",
            "plano",
            "itinerary",
            "itinerário",
            "day",
            "dia",
            "schedule",
            "agenda",
            "roteiro",
        ]

        agents = []

        # Check for keywords
        if any(kw in message_lower for kw in weather_keywords):
            agents.append("weather")
        if any(kw in message_lower for kw in transport_keywords):
            agents.append("transport")
        if any(kw in message_lower for kw in places_keywords):
            agents.append("researcher")
        if any(kw in message_lower for kw in itinerary_keywords):
            # Itinerary needs weather + researcher + planner
            if "weather" not in agents:
                agents.append("weather")
            if "researcher" not in agents:
                agents.append("researcher")
            agents.append("planner")

        # If no specific keywords, check if it's a greeting/simple question
        greeting_keywords = [
            "hello",
            "hi",
            "olá",
            "ola",
            "bom dia",
            "boa tarde",
            "boa noite",
            "obrigado",
            "thanks",
            "help",
            "ajuda",
        ]

        if not agents and any(kw in message_lower for kw in greeting_keywords):
            return {
                "reasoning": "Simple greeting/general query",
                "agents": [],
                "direct_response": "Hello! 👋 I'm the Lisbon Urban Assistant. How can I help you explore Lisbon today?",
            }

        # If still no agents and not a greeting, default to researcher
        if not agents:
            agents = ["researcher"]

        return {
            "reasoning": f"Fallback routing based on keywords: {agents}",
            "agents": agents,
            "direct_response": None,
        }

    def _requires_weather_for_planning(self, user_message: str) -> bool:
        """
        Detects if user is asking to plan for today, tomorrow, this week, or next few days.
        Returns True if weather data should be mandatory for the planning.

        Args:
            user_message: The user's query.

        Returns:
            bool: True if weather should be required for planning.
        """
        message_lower = user_message.lower()

        # Patterns for immediate/near-future planning (requires weather)
        immediate_patterns = [
            # Today
            r"\bhoje\b",
            r"\btoday\b",
            r"\bfor today\b",
            r"\bpara hoje\b",
            # Tomorrow
            r"\bamanh[ãa]\b",
            r"\btomorrow\b",
            r"\bfor tomorrow\b",
            r"\bpara amanh[ãa]\b",
            # This week
            r"\besta semana\b",
            r"\bthis week\b",
            r"\bna semana\b",
            r"\bduring this week\b",
            # Next X days
            r"\bpr[óo]ximos?\s+\d+\s+dias?\b",
            r"\bnext\s+\d+\s+days?\b",
            r"\bpr[óo]ximos?\s+(?:dois|tr[êe]s|quatro|cinco|seis|sete)\s+dias?\b",
            r"\bnext\s+(?:two|three|four|five|six|seven)\s+days?\b",
            r"\b(?:dois|tr[êe]s|quatro|cinco|seis|sete)\s+dias?\b",
            r"\b(?:two|three|four|five|six|seven)\s+days?\b",
            # Day plans
            r"\bday\s+\d+\b",
            r"\b(\d+)[oa]?\s+dia\b",
            # Weekend
            r"\bweekend\b",
            r"\bfim de semana\b",
            r"\bpr[óo]ximo fim de semana\b",
            r"\bnext weekend\b",
            # Now/immediate
            r"\bagora\b",
            r"\bnow\b",
            r"\bcurrently\b",
            r"\batualmente\b",
        ]

        # Check if any immediate pattern matches
        for pattern in immediate_patterns:
            if re.search(pattern, message_lower):
                return True

        return False

    def format_agent_outputs(self, agent_outputs: Dict[str, str]) -> str:
        """
        Formats outputs from multiple agents into a combined context string.

        Args:
            agent_outputs: Dict mapping agent names to their outputs.

        Returns:
            str: Formatted context for the planner or final response.
        """
        if not agent_outputs:
            return ""

        sections = []

        if "weather" in agent_outputs:
            sections.append(f"## 🌤️ Weather Information\n{agent_outputs['weather']}")

        if "transport" in agent_outputs:
            sections.append(
                f"## 🚇 Transport Information\n{agent_outputs['transport']}"
            )

        if "researcher" in agent_outputs:
            sections.append(f"## 🔍 Places & Events\n{agent_outputs['researcher']}")

        return "\n\n---\n\n".join(sections)


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 Supervisor Agent Test\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")

    try:
        supervisor = SupervisorAgent()
        print(
            f"\n\033[1m✅ Supervisor initialized:\033[0m {supervisor.get_model_info()}"
        )

        # Test queries
        test_queries = [
            "Hello!",
            "What's the weather like?",
            "How do I get to Belém?",
            "Recommend some museums",
            "Plan my day visiting museums and considering the weather",
            "Quanto é 2+2?",
            "Que sitios posso visitar no Porto?",
            "Como está o tempo em Aveiro?",
        ]

        for query in test_queries:
            print(f"\n\033[1m📝 Query:\033[0m {query}")
            decision = supervisor.route(query)
            print(f"   \033[1mAgents:\033[0m {decision['agents']}")
            print(f"   \033[1mReason:\033[0m {decision['reasoning']}")
            if decision["direct_response"]:
                print(f"   \033[1mDirect:\033[0m {decision['direct_response']}")

        print("\n\033[1;32m✅ Supervisor agent working!\033[0m")

    except Exception as e:
        print(f"\n\033[1;31m❌ Error:\033[0m {e}")
