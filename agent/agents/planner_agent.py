# ==========================================================================
# Master Thesis - Planner Agent
#   - André Filipe Gomes Silvestre, 20240502
# 
#   Itinerary synthesis agent. Combines outputs from other agents
#   into coherent travel plans.
# ==========================================================================

from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

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
        This agent has NO tools. It only synthesizes data gathered by worker
        agents and can surface QA disclaimers in the final planning response.
        In the default runtime, it is invoked only when the supervisor route
        includes the planner. Direct and simple single-domain queries can
        return without using this agent.
    """
    
    def __init__(self):
        """Initializes the planner agent."""
        super().__init__("planner")
        self.system_prompt = get_planner_prompt()
    
    @traceable(name="planner_agent", run_type="chain", tags=["sub-agent", "planner"])
    def invoke(
        self, 
        user_message: str, 
        weather_data: str = "",
        transport_data: str = "",
        places_data: str = "",
        events_data: str = "",
        qa_disclaimers: list[str] | None = None,
    ) -> str:
        """
        Creates an itinerary from gathered data.
        
        Args:
            user_message: The user's original query.
            weather_data: Output from weather agent.
            transport_data: Output from transport agent.
            places_data: Output from researcher agent (places).
            events_data: Output from researcher agent (events).
            qa_disclaimers: Optional list of QA-flagged data limitations.
            
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
        
        # Inject QA disclaimers so the planner transparently communicates limitations
        if qa_disclaimers:
            disclaimer_text = "\n".join(f"- ⚠️ {d}" for d in qa_disclaimers)
            context_parts.append(
                f"## ⚠️ Data Limitations (from QA validation)\n"
                f"Include these caveats in your response where relevant:\n{disclaimer_text}"
            )
        
        context = "\n\n---\n\n".join(context_parts) if context_parts else "No additional data provided."
        
        messages = [
            SystemMessage(content=self.system_prompt),
            SystemMessage(content=f"# Data from Specialized Agents\n\n{context}"),
            HumanMessage(content=f"Based on the data above, create an itinerary for: {user_message}")
        ]
        
        # Planner has no tools - LLM call with retry for Azure content filter
        response = self._safe_llm_invoke(self.llm, messages)
        return clean_response(response.content)
    
    def synthesize(self, user_message: str, agent_outputs: Dict[str, str]) -> str:
        """
        Synthesizes outputs from multiple agents into a response.
        
        Extracts QA disclaimers from internal keys and passes them
        to the planner so data limitations are surfaced to the user.
        
        Args:
            user_message: Original user query.
            agent_outputs: Dict mapping agent names to their outputs.
                May contain '_qa_disclaimers' (list) from QA validation.
            
        Returns:
            str: Synthesized response.
        """
        # Extract QA disclaimers before passing to invoke
        qa_disclaimers = agent_outputs.get("_qa_disclaimers")
        if isinstance(qa_disclaimers, str):
            # Safety: if it was stored as a string, wrap in list
            qa_disclaimers = [qa_disclaimers]

        return self.invoke(
            user_message=user_message,
            weather_data=agent_outputs.get("weather", ""),
            transport_data=agent_outputs.get("transport", ""),
            places_data=agent_outputs.get("researcher", ""),
            events_data="",  # Events come from researcher too
            qa_disclaimers=qa_disclaimers,
        )


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
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
        
        print("\n\033[1m📝 Testing with mock data:\033[0m")
        response = agent.invoke(
            user_message="Plan my morning in Belém",
            weather_data=mock_weather,
            places_data=mock_places
        )
        print("\n\033[1m🤖 Response:\033[0m")
        print(response[:800] + "..." if len(response) > 800 else response)
        
        print("\n\033[1;32m✅ Planner agent working!\033[0m")
        
    except Exception as e:
        print(f"\n\033[1;31m❌ Error:\033[0m {e}")
        import traceback
        traceback.print_exc()
