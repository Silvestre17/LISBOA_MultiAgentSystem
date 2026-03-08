# ==========================================================================
# Master Thesis - Weather Agent
#   - André Filipe Gomes Silvestre, 20240502
#
#   Specialized agent for weather-related queries using IPMA data.
#   Uses BaseAgent.execute_react_loop() for tool execution.
# ==========================================================================

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

from agent.agents.base import BaseAgent, traceable
from agent.prompts.weather import get_weather_prompt
from agent.state import AgentState
from agent.utils.langgraph_compat import ToolNode


class WeatherAgent(BaseAgent):
    """
    Weather specialist agent using IPMA data.

    Tools:
        - get_weather_warnings
        - get_weather_forecast
        - get_current_weather_summary
        - get_portugal_weather_overview

    Notes:
        This worker is used for weather-specific retrieval. The optional
        `context` argument is injected by the orchestrator in multi-agent
        scenarios to preserve language and follow-up hints.
    """

    def __init__(self):
        """Initializes the weather agent."""
        super().__init__("weather")
        self.system_prompt = get_weather_prompt()

    @traceable(name="weather_agent", run_type="chain", tags=["sub-agent", "weather"])
    def invoke(
        self, user_message: str, context: str = "", verbose: bool = False
    ) -> str:
        """
        Processes a weather-related query.

        Args:
            user_message: The user's query.
            context: Additional context from other agents (optional).
            verbose: Whether involved tool calls should be printed.

        Returns:
            str: Weather information response.
        """
        messages = [SystemMessage(content=self.system_prompt)]

        if context:
            messages.append(
                SystemMessage(content=f"Context from other agents:\n{context}")
            )

        messages.append(HumanMessage(content=user_message))

        return self.execute_react_loop(
            messages=messages,
            verbose=verbose,
            max_iterations=5,
            tool_enforcement_msg=(
                "You MUST use a tool (like get_current_weather_summary) to get real data. "
                "Do NOT answer from your knowledge base. Call the tool now."
            ),
        )

    def build_subgraph(self) -> "CompiledStateGraph":
        """
        Builds a LangGraph subgraph for this agent.

        Returns:
            CompiledStateGraph: Compiled subgraph for weather queries.
        """

        def agent_node(state: AgentState) -> dict:
            """Weather agent decision node."""
            messages = list(state["messages"])

            # Add system prompt if not present
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=self.system_prompt)] + messages

            response = self._safe_llm_invoke(self.llm_with_tools, messages)
            return {"messages": [response]}

        def should_continue(state: AgentState) -> str:
            """Determines next step."""
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return "end"

        # Build graph
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", "end": END}
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile()


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 Weather Agent Test\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")

    try:
        agent = WeatherAgent()
        print(f"\n\033[1m✅ Weather Agent initialized:\033[0m {agent.get_model_info()}")
        print(f"   Tools: {[t.name for t in agent.tools]}")

        print("\n\033[1m📝 Testing query:\033[0m 'What is the weather in Lisbon?'")
        response = agent.invoke("What is the weather in Lisbon?")
        print("\n\033[1m🤖 Response:\033[0m")
        print(response)

        print("\n\033[1;32m✅ Weather agent working!\033[0m")

    except Exception as e:
        print(f"\n\033[1;31m❌ Error:\033[0m {e}")
        import traceback

        traceback.print_exc()
