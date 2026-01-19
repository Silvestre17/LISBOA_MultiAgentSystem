# ==========================================================================
# Master Thesis - Transport Agent
#   - André Filipe Gomes Silvestre, 20240502
# 
#   Specialized agent for transport-related queries.
#   Handles metro, bus, train, and multi-modal routing.
# ==========================================================================

import os
import sys
from typing import Dict, Any, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from agent.agents.base import BaseAgent, clean_response
from agent.prompts.transport import get_transport_prompt
from agent.state import AgentState


class TransportAgent(BaseAgent):
    """
    Transport specialist agent for Lisbon's public transport.
    
    Handles:
        - Metro de Lisboa (status, routing, wait times)
        - Carris Metropolitana (bus routes, alerts)
        - CP trains (suburban lines)
        - Multi-modal routing
    """
    
    def __init__(self):
        """Initializes the transport agent."""
        super().__init__("transport")
        self.system_prompt = get_transport_prompt()
    
    def invoke(self, user_message: str, context: str = "", verbose: bool = False) -> str:
        """
        Processes a transport-related query.
        
        Args:
            user_message: The user's query.
            context: Additional context from other agents (optional).
            verbose: Whether involved tool calls should be printed.
            
        Returns:
            str: Transport information response.
        """
        messages = [
            SystemMessage(content=self.system_prompt),
        ]
        
        if context:
            messages.append(SystemMessage(content=f"Context from other agents:\n{context}"))
        
        messages.append(HumanMessage(content=user_message))
        
        # ReAct loop with tools
        response = self.llm_with_tools.invoke(messages)
        
        # ENFORCEMENT: If no tool calls and not a greeting/thanks, force tool usage
        is_greeting = any(w in user_message.lower() for w in ["hello", "thanks", "obrigado", "tchau"])
        if not (hasattr(response, "tool_calls") and response.tool_calls) and not is_greeting:
            if verbose:
                print(f"      [DEBUG] No tools called initially. Forcing tool usage...")
            messages.append(AIMessage(content=response.content))
            messages.append(HumanMessage(
                content="You MUST use a tool (like get_metro_status or search_transport_routes) to get real data. "
                        "Do NOT answer from your knowledge base. Call the tool now."
            ))
            response = self.llm_with_tools.invoke(messages)
            
        max_iterations = 5  # Reduced from 8 to prevent runaway loops
        iteration = 0
        called_tools = set()  # Track tool signatures to prevent duplicates
        
        while hasattr(response, "tool_calls") and response.tool_calls and iteration < max_iterations:
            messages.append(response)
            
            # Check for duplicate tool calls (loop detection)
            new_calls = []
            duplicate_detected = False
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                # Create signature for this tool call
                import json
                try:
                    args_str = json.dumps(tool_args, sort_keys=True)
                except:
                    args_str = str(tool_args)
                signature = f"{tool_name}:{args_str}"
                
                if signature in called_tools:
                    duplicate_detected = True
                    if verbose:
                        print(f"      [LOOP] Duplicate tool call detected: {tool_name}. Breaking loop.")
                else:
                    called_tools.add(signature)
                    new_calls.append(tool_call)
            
            # If all calls are duplicates, force response generation
            if duplicate_detected and not new_calls:
                if verbose:
                    print(f"      [LOOP] All tool calls are duplicates. Forcing response.")
                # Add a system message to force response
                messages.append(SystemMessage(
                    content="STOP CALLING TOOLS. You have already called these tools. Use the results you have and respond to the user NOW."
                ))
                response = self.llm_with_tools.invoke(messages)
                break
            
            # Execute only non-duplicate tools
            tools_to_execute = new_calls if new_calls else response.tool_calls[:1]  # Fallback to first
            
            for tool_call in tools_to_execute:
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
            
            response = self.llm_with_tools.invoke(messages)
            iteration += 1
        
        # If we hit max iterations, force a response
        if iteration >= max_iterations and hasattr(response, "tool_calls") and response.tool_calls:
            if verbose:
                print(f"      [LIMIT] Max iterations reached. Forcing response.")
            messages.append(response)
            messages.append(SystemMessage(
                content="MAXIMUM TOOL CALLS REACHED. Stop calling tools and respond to the user with the information you have collected."
            ))
            response = self.llm_with_tools.invoke(messages)
        
        return clean_response(response.content)
    
    def build_subgraph(self) -> StateGraph:
        """
        Builds a LangGraph subgraph for this agent.
        
        Returns:
            StateGraph: Compiled subgraph for transport queries.
        """
        def agent_node(state: AgentState) -> dict:
            """Transport agent decision node."""
            messages = list(state["messages"])
            
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
    print("\033[1m🧪 Transport Agent Test\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")
    
    try:
        agent = TransportAgent()
        print(f"\n\033[1m✅ Transport Agent initialized:\033[0m {agent.get_model_info()}")
        print(f"   Tools: {len(agent.tools)} transport tools")
        
        print(f"\n\033[1m📝 Testing query:\033[0m 'Is the metro working?'")
        response = agent.invoke("Is the metro working?")
        print(f"\n\033[1m🤖 Response:\033[0m")
        print(response[:500] + "..." if len(response) > 500 else response)
        
        print(f"\n\033[1;32m✅ Transport agent working!\033[0m")
        
    except Exception as e:
        print(f"\n\033[1;31m❌ Error:\033[0m {e}")
        import traceback
        traceback.print_exc()
