# ==========================================================================
# Master Thesis - Planner Agent
#   - André Filipe Gomes Silvestre, 20240502
# 
#   Itinerary synthesis agent. Combines outputs from other agents
#   into coherent travel plans.
# ==========================================================================

import os
import sys
from typing import Dict, Any, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from agent.agents.base import BaseAgent, clean_response, traceable
from agent.prompts.planner import get_planner_prompt


class PlannerAgent(BaseAgent):
    """
    Itinerary planner agent that synthesizes outputs from other agents.
    
    Responsibilities:
        - Combine weather, transport, and places data
        - Apply constraints (mobility, time, weather)
        - Generate coherent, practical itineraries
    
    Note:
        This agent has NO tools - it only synthesizes data from other agents.
    """
    
    def __init__(self):
        """Initializes the planner agent."""
        super().__init__("planner")
        self.system_prompt = get_planner_prompt()
    
    @traceable(name="planner_agent", run_type="chain")
    def invoke(
        self, 
        user_message: str, 
        weather_data: str = "",
        transport_data: str = "",
        places_data: str = "",
        events_data: str = ""
    ) -> str:
        """
        Creates an itinerary from gathered data.
        
        Args:
            user_message: The user's original query.
            weather_data: Output from weather agent.
            transport_data: Output from transport agent.
            places_data: Output from researcher agent (places).
            events_data: Output from researcher agent (events).
            
        Returns:
            str: Formatted itinerary.
        """
        # Build context from agent outputs
        context_parts = []
        
        if weather_data:
            context_parts.append(f"## 🌤️ Weather Data\n{weather_data}")
        
        if places_data:
            context_parts.append(f"## 🏛️ Places & Attractions\n{places_data}")
        
        if events_data:
            context_parts.append(f"## 🎭 Events\n{events_data}")
        
        if transport_data:
            context_parts.append(f"## 🚇 Transport Info\n{transport_data}")
        
        context = "\n\n---\n\n".join(context_parts) if context_parts else "No additional data provided."
        
        messages = [
            SystemMessage(content=self.system_prompt),
            SystemMessage(content=f"# Data from Specialized Agents\n\n{context}"),
            HumanMessage(content=f"Based on the data above, create an itinerary for: {user_message}")
        ]
        
        # Planner has no tools - direct LLM call
        response = self.llm.invoke(messages)
        return clean_response(response.content)
    
    def synthesize(self, user_message: str, agent_outputs: Dict[str, str]) -> str:
        """
        Synthesizes outputs from multiple agents into a response.
        
        Args:
            user_message: Original user query.
            agent_outputs: Dict mapping agent names to their outputs.
            
        Returns:
            str: Synthesized response.
        """
        return self.invoke(
            user_message=user_message,
            weather_data=agent_outputs.get("weather", ""),
            transport_data=agent_outputs.get("transport", ""),
            places_data=agent_outputs.get("researcher", ""),
            events_data=""  # Events come from researcher too
        )


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 Planner Agent Test\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")
    
    try:
        agent = PlannerAgent()
        print(f"\n\033[1m✅ Planner Agent initialized:\033[0m {agent.get_model_info()}")
        print(f"   Tools: {len(agent.tools)} (planner has no tools)")
        
        # Simulate data from other agents
        mock_weather = """
        Today in Lisbon: ☀️ Clear sky
        🌡️ Temperature: 18°C - 24°C
        🌧️ Precipitation: 10% (unlikely)
        🌤️ UV Index: High - bring sunscreen!
        """
        
        mock_places = """
        1. 🏛️ **Mosteiro dos Jerónimos** - UNESCO World Heritage
           📍 Belém | 🕐 10:00-17:00 | 💰 €10
        
        2. 🏛️ **Museu Nacional dos Coches** - World's best carriage collection
           📍 Belém | 🕐 10:00-18:00 | 💰 €8
        
        3. 🎨 **MAAT** - Modern architecture & contemporary art
           📍 Belém | 🕐 11:00-19:00 | 💰 €9
        """
        
        print(f"\n\033[1m📝 Testing with mock data:\033[0m")
        response = agent.invoke(
            user_message="Plan my morning in Belém",
            weather_data=mock_weather,
            places_data=mock_places
        )
        print(f"\n\033[1m🤖 Response:\033[0m")
        print(response[:800] + "..." if len(response) > 800 else response)
        
        print(f"\n\033[1;32m✅ Planner agent working!\033[0m")
        
    except Exception as e:
        print(f"\n\033[1;31m❌ Error:\033[0m {e}")
        import traceback
        traceback.print_exc()
