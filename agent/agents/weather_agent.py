# ==========================================================================
# Master Thesis - Weather Agent
#   - André Filipe Gomes Silvestre, 20240502
# 
#   Specialized agent for weather-related queries using IPMA data.
# ==========================================================================

import os
import sys
from typing import Dict, Any, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from agent.agents.base import BaseAgent, clean_response, traceable
from agent.prompts.weather import get_weather_prompt
from agent.state import AgentState


class WeatherAgent(BaseAgent):
    """
    Weather specialist agent using IPMA data.
    
    Tools:
        - get_weather_warnings
        - get_weather_forecast
        - get_current_weather_summary
    """
    
    def __init__(self):
        """Initializes the weather agent."""
        super().__init__("weather")
        self.system_prompt = get_weather_prompt()
    
    @traceable(name="weather_agent", run_type="chain")
    def invoke(self, user_message: str, context: str = "", verbose: bool = False) -> str:
        """
        Processes a weather-related query.
        
        Args:
            user_message: The user's query.
            context: Additional context from other agents (optional).
            verbose: Whether involved tool calls should be printed.
            
        Returns:
            str: Weather information response.
        """
        messages = [
            SystemMessage(content=self.system_prompt),
        ]
        
        if context:
            messages.append(SystemMessage(content=f"Context from other agents:\n{context}"))
        
        messages.append(HumanMessage(content=user_message))
        
        # First call - may request tool use
        response = self.llm_with_tools.invoke(messages)
        
        # ENFORCEMENT: If no tool calls and not a greeting, force tool usage
        if not (hasattr(response, "tool_calls") and response.tool_calls):
            # Check if it looks like a refusal or hallucination
            if verbose:
                print(f"      [DEBUG] No tools called initially. Forcing tool usage...")
            
            messages.append(AIMessage(content=response.content))
            messages.append(HumanMessage(
                content="You MUST use a tool (like get_current_weather_summary) to get real data. "
                        "Do NOT answer from your knowledge base. call the tool now."
            ))
            response = self.llm_with_tools.invoke(messages)
        
        # If tool calls are requested, execute them
        max_iterations = 5
        iteration = 0
        
        while hasattr(response, "tool_calls") and response.tool_calls and iteration < max_iterations:
            messages.append(response)
            
            # Execute tools directly (not using ToolNode to avoid state issues)
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id", f"call_{iteration}")
                
                if verbose:
                    print(f"      [TOOL] Calling {tool_name} with args: {tool_args}")
                
                # Find and execute the tool
                tool_result = None
                for tool in self.tools:
                    if tool.name == tool_name:
                        try:
                            tool_result = tool.invoke(tool_args)
                            if verbose:
                                result_preview = str(tool_result)[:100] + "..." if len(str(tool_result)) > 100 else str(tool_result)
                                print(f"      [TOOL] Result: {result_preview}")
                        except Exception as e:
                            tool_result = f"Error executing {tool_name}: {str(e)}"
                            if verbose:
                                print(f"      [TOOL] Error: {tool_result}")
                        break
                
                if tool_result is None:
                    tool_result = f"Tool '{tool_name}' not found."
                
                # Add tool result as ToolMessage
                messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_id))
            
            # Get next response
            response = self.llm_with_tools.invoke(messages)
            iteration += 1
        
        return clean_response(response.content)
    
    def build_subgraph(self) -> StateGraph:
        """
        Builds a LangGraph subgraph for this agent.
        
        Returns:
            StateGraph: Compiled subgraph for weather queries.
        """
        def agent_node(state: AgentState) -> dict:
            """Weather agent decision node."""
            messages = list(state["messages"])
            
            # Add system prompt if not present
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=self.system_prompt)] + messages
            
            response = self.llm_with_tools.invoke(messages)
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
        workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
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
        
        print(f"\n\033[1m📝 Testing query:\033[0m 'What is the weather in Lisbon?'")
        response = agent.invoke("What is the weather in Lisbon?")
        print(f"\n\033[1m🤖 Response:\033[0m")
        print(response)
        
        print(f"\n\033[1;32m✅ Weather agent working!\033[0m")
        
    except Exception as e:
        print(f"\n\033[1;31m❌ Error:\033[0m {e}")
        import traceback
        traceback.print_exc()
