# ==========================================================================
# Master Thesis - Base Agent Utilities
#   - André Filipe Gomes Silvestre, 20240502
#
#   Shared utilities for all specialized agents.
#   Provides common functionality for:
#     - Tool binding and LLM creation
#     - Response cleaning (think tags, JSON artifacts, etc.)
#     - ReAct loop execution (tool calls, parallel execution, loop detection)
#     - Tool registry per agent type
# ==========================================================================

import json
import os
import sys
import re
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config import Config

# ==========================================================================
# LangSmith Tracing Support
# ==========================================================================
try:
    from langsmith.run_helpers import traceable

    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

    # Fallback: no-op decorator
    def traceable(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


# ==========================================================================
# Tool Definitions by Agent
# ==========================================================================


def get_agent_tools(agent_name: str) -> List:
    """
    Returns the tools for a specific agent.

    This function lazily imports tools to avoid circular imports
    and only loads what's needed for each agent.

    Args:
        agent_name (str): Name of the agent ('weather', 'transport', 'researcher', 'planner')

    Returns:
        List: List of LangChain tools for the specified agent.
    """
    if agent_name == "weather":
        from tools.ipma_api import (
            get_weather_warnings,
            get_weather_forecast,
            get_current_weather_summary,
            get_portugal_weather_overview,
        )

        return [
            get_weather_warnings,
            get_weather_forecast,
            get_current_weather_summary,
            get_portugal_weather_overview,
        ]

    elif agent_name == "transport":
        # Metro de Lisboa (Official API with OAuth2)
        from tools.metrolisboa_api import (
            get_metro_status,
            get_metro_wait_time,
            get_metro_line_wait_times,
            find_nearest_metro,
            get_metro_frequency,
            get_all_metro_stations,
        )

        # Carris Metropolitana (Suburban buses)
        from tools.carrismetropolitana_api import (
            get_real_time_bus_positions,
            get_carris_metropolitana_alerts,
            get_carris_metropolitana_stop_info,
            search_carris_metropolitana_lines,
            find_bus_routes,
            find_direct_bus_lines,
            get_bus_realtime_locations,
            get_bus_next_departures,
        )

        # CP (Comboios de Portugal) - Trains
        from tools.cp_api import (
            get_train_status,
            search_cp_stations,
            plan_train_trip,
            get_train_schedule,
            get_cp_routes,
        )

        # Multi-modal transport routing
        from tools.transport_api import (
            get_transport_summary,
            get_route_between_stations,
        )
        from tools.carris_api import (
            carris_get_stops,
            carris_get_routes,
            carris_get_next_departures,
            carris_find_routes_between,
            carris_get_realtime_vehicles,
            carris_get_arrivals,
            carris_vehicle_eta,
        )

        return [
            get_metro_status,
            get_metro_wait_time,
            get_metro_line_wait_times,
            find_nearest_metro,
            get_metro_frequency,
            get_all_metro_stations,
            get_carris_metropolitana_alerts,
            get_carris_metropolitana_stop_info,
            search_carris_metropolitana_lines,
            find_direct_bus_lines,
            get_real_time_bus_positions,
            get_train_status,
            plan_train_trip,
            get_train_schedule,
            get_cp_routes,
            get_transport_summary,
            get_route_between_stations,
            find_bus_routes,
            get_bus_realtime_locations,
            get_bus_next_departures,
            search_cp_stations,
            carris_get_stops,
            carris_get_routes,
            carris_get_next_departures,
            carris_find_routes_between,
            carris_get_realtime_vehicles,
            carris_get_arrivals,
            carris_vehicle_eta,
        ]

    elif agent_name == "researcher":
        from tools.visitlisboa_api import (
            search_cultural_events,
            search_places_attractions,
            get_event_categories,
            get_place_categories,
            search_lisbon_knowledge,
        )
        from tools.dados_abertos import (
            find_nearby_services,
            list_available_datasets,
            get_dataset_details,
            find_place_in_datasets,
        )
        from tools.web_knowledge import search_history_culture

        return [
            search_cultural_events,
            search_places_attractions,
            get_event_categories,
            get_place_categories,
            search_lisbon_knowledge,
            find_nearby_services,
            list_available_datasets,
            get_dataset_details,
            find_place_in_datasets,
            search_history_culture,
        ]

    elif agent_name == "planner":
        # Planner has no tools - it synthesizes outputs from other agents
        return []

    elif agent_name == "supervisor":
        # Supervisor has no tools - it only routes to other agents
        return []

    else:
        raise ValueError(f"Unknown agent: {agent_name}")


# ==========================================================================
# LLM Factory for Agents
# ==========================================================================


def get_agent_llm(agent_name: str) -> BaseChatModel:
    """
    Creates an LLM instance configured for a specific agent.

    Uses AGENT_MODELS from config.py for per-agent model configuration.
    Falls back to DEFAULT_AGENT_MODEL if not specified.

    Args:
        agent_name (str): Name of the agent.

    Returns:
        BaseChatModel: Configured LLM for the agent.
    """
    from agent.llm_factory import LLMFactory

    return LLMFactory.get_agent_llm(agent_name)


# ==========================================================================
# Response Cleaning Utilities
# ==========================================================================


def clean_response(content: str) -> str:
    """
    Cleans model-specific artifacts from the response.

    Removes:
        - <think>...</think> blocks (Qwen3 reasoning)
        - <tool_call>...</tool_call> blocks
        - Embedded JSON tool call syntax
        - Chat template tokens
        - Qwen3 "thinking out loud" patterns (e.g., "How do I..." followed by reasoning)
        - Step-by-step internal reasoning ("Step 1:", "Wait -", etc.)

    Args:
        content: Raw response from the LLM.

    Returns:
        str: Cleaned response suitable for user display.
    """
    # Handle None or empty content
    if content is None:
        return ""
    
    # Handle non-string content (e.g., list from Responses API)
    if not isinstance(content, str):
        # If it's a list, try to extract text from it
        if isinstance(content, list):
            # Common pattern: list of content blocks with 'text' key
            text_parts = []
            for item in content:
                if isinstance(item, dict) and 'text' in item:
                    text_parts.append(item['text'])
                elif isinstance(item, str):
                    text_parts.append(item)
            content = "\n".join(text_parts) if text_parts else str(content)
        else:
            content = str(content)
    
    if not content:
        return ""

    # CRITICAL: Detect and remove Qwen3 "thinking out loud" pattern
    # Pattern: Model starts answering a DIFFERENT question and reasons through it
    # Example: "How do I get to airport from Rossio?\n\nWe are in English...\n\nStep-by-step:..."

    # FIRST: Check if entire response is a "thinking" block about a wrong question
    # This is the CRITICAL fix for the hallucination bug where the model answers
    # a completely different question than what was asked
    wrong_question_patterns = [
        # Full response is about getting to airport when that wasn't the question
        r"^How do I get to (?:the )?airport.*$",
        # Model "thinking" about the question
        r"^We are in (?:English|Portuguese)\.\s*The user wants to.*$",
        # Internal planning that leaked through
        r"^Step-by-step:.*$",
        # "Note:" at the very start indicates internal reasoning
        r"^Note:.*Rossio is a major station.*$",
        # Important internal marker
        r"^Important:.*(?:is served by|does NOT).*$",
    ]

    for pattern in wrong_question_patterns:
        if re.match(pattern, content, flags=re.DOTALL | re.IGNORECASE):
            # The entire response is internal reasoning - return error message
            return "Ocorreu um erro ao processar. / An error occurred while processing."

    thinking_patterns = [
        # "How do I..." followed by step-by-step reasoning (different question hallucination)
        r"^How do I [^?]+\?\s*(?:\n.*)?We are in (?:English|Portuguese).*$",
        # "Step-by-step:" internal reasoning
        r"Step-by-step:\s*\n.*(?:Check if|If not|Use tools).*",
        # "Wait -" reasoning pattern
        r"\n\s*Wait\s*[-–]\s*.*(?:\n.*)*",
        # "But wait" reasoning pattern
        r"\n\s*But wait\s*[-–]?\s*.*(?:\n.*)*",
        # "Let me check" / "Let me recheck" internal reasoning
        r"\n\s*Let me (?:check|recheck).*(?:\n.*)*",
        # "Therefore," followed by internal logic
        r"\n\s*Therefore,\s*(?:I must|we must|from|the).*(?:\n.*)*",
        # "So final response:" marker
        r"\n\s*So final response:.*(?:\n.*)*",
        # "Final output:" marker
        r"\n\s*Final output[:\s].*(?:\n.*)*",
        # Checkmarks at the end of reasoning
        r"\n\s*✅\s*(?:Language|No origin|Clear)[^\n]*(?:\n.*)*$",
        # "Ah!" discovery pattern
        r"\n\s*Ah!.*(?:\n.*)*",
        # "So from Rossio to..." planning pattern
        r"\n\s*So from [A-Z][a-z]+ to (?:airport|[A-Z]).*(?:\n.*)*",
        # "But is there a..." questioning pattern
        r"\n\s*But is there a.*(?:\n.*)*",
        # "This is correct and follows all rules" reasoning marker
        r"\n\s*This is correct and follows all rules.*(?:\n.*)*",
        # "No hallucination" marker
        r"\n\s*No hallucination.*(?:\n.*)*",
        # "The CP train lines:" internal knowledge dump
        r"\n\s*The CP train lines:.*(?:\n.*)*",
        # "The metro lines are:" internal knowledge dump
        r"\n\s*The metro lines are:.*(?:\n.*)*",
        # "Actually, the" reasoning
        r"\n\s*Actually, the.*(?:\n.*)*",
        # "The only train line" reasoning
        r"\n\s*The only train line.*(?:\n.*)*",
    ]

    for pattern in thinking_patterns:
        content = re.sub(
            pattern, "", content, flags=re.DOTALL | re.MULTILINE | re.IGNORECASE
        )

    # Remove <think>...</think> blocks (Qwen3 reasoning) - handles multiline
    content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)

    # Remove <tool_call>...</tool_call> blocks
    content = re.sub(r"</?tool_call>\s*", "", content, flags=re.DOTALL)

    # Remove embedded JSON tool call syntax
    content = re.sub(
        r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}', "", content
    )

    # Remove chat template tokens
    content = re.sub(r"<\|im_start\|>.*?\n?", "", content)
    content = re.sub(r"<\|im_end\|>\s*", "", content)
    content = re.sub(r"<\|.*?\|>\s*", "", content)

    # Remove specific tool artifacts first
    content = re.sub(r"<tool_code>.*?</tool_code>", "", content, flags=re.DOTALL)

    # Strip markdown code blocks if the entire content is wrapped
    # e.g. ```markdown ... ``` or ``` ... ```
    # This prevents Streamlit from rendering the whole response as a code block
    content = re.sub(r"^```(?:markdown|text)?\s*\n", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\n\s*```$", "", content)

    # Clean up excess whitespace
    content = content.strip()

    # Final check: If content is empty after cleaning, return error
    if not content:
        return "Desculpe, tive dificuldades em processar o pedido. / Sorry, I'm having difficulty processing your request."

    # Print markdown to terminal if debugging is enabled
    if Config.SHOW_MARKDOWN_RESPONSE_IN_TERMINAL:
        print("\n" + "=" * 80)
        print("📝 AI RESPONSE (Markdown)")
        print("=" * 80)
        print(content)
        print("=" * 80 + "\n")

    return content


def parse_json_response(content: str) -> Optional[Dict[str, Any]]:
    """
    Extracts JSON from a response that may contain markdown code blocks.

    Args:
        content: Response text that may contain JSON.

    Returns:
        Dict or None: Parsed JSON if found, None otherwise.
    """
    if not content:
        return None

    # Clean first
    content = clean_response(content)

    # Try to find JSON in code blocks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))  # json imported at module top
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON
    json_match = re.search(r'\{[^{}]*"agents"[^{}]*\}', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Try parsing the entire content as JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


# ==========================================================================
# Base Agent Class
# ==========================================================================


class BaseAgent:
    """
    Base class for all specialized agents.

    Provides common functionality for:
        - LLM initialization with agent-specific config
        - Tool binding
        - Response cleaning
        - State management
    """

    def __init__(self, agent_name: str):
        """
        Initializes the base agent.

        Args:
            agent_name: Name of this agent (for config lookup).
        """
        self.agent_name = agent_name
        self.tools = get_agent_tools(agent_name)
        self.llm = get_agent_llm(agent_name)

        # Bind tools if this agent has any
        if self.tools:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        else:
            self.llm_with_tools = self.llm

    def get_model_info(self) -> Dict[str, Any]:
        """Returns the model info dictionary."""
        from agent.llm_factory import LLMFactory

        return LLMFactory.get_model_info(self.llm)

    def clean_response(self, content: str) -> str:
        """Cleans model artifacts from response."""
        return clean_response(content)

    def execute_tool_from_json(self, content: str, verbose: bool = False) -> Optional[str]:
        """
        Execute a tool call if the model returns a JSON tool request in content.

        Supports formats like:
            {"tool_call_name": "get_metro_status", "tool_call_arguments": {}}
            {"name": "get_metro_status", "arguments": {}}

        Returns:
            Tool result as string, or None if no tool call detected.
        """
        parsed = parse_json_response(content)
        if not isinstance(parsed, dict):
            return None

        tool_name = parsed.get("tool_call_name") or parsed.get("name")
        tool_args = (
            parsed.get("tool_call_arguments")
            or parsed.get("arguments")
            or parsed.get("args")
            or {}
        )
        if not tool_name:
            return None

        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    if verbose:
                        print(f"      [TOOL] Calling {tool_name} with args: {tool_args}")
                    return str(tool.invoke(tool_args))
                except Exception as e:
                    return f"Error executing {tool_name}: {str(e)}"

        return f"Tool '{tool_name}' not found."

    def execute_tools_parallel(
        self,
        tool_calls: List[Dict],
        max_workers: int = 4,
        timeout: float = 30.0
    ) -> Dict[str, str]:
        """
        Execute multiple tool calls in parallel for faster processing.
        
        Args:
            tool_calls: List of tool call dicts with 'name', 'args', 'id' keys.
            max_workers: Maximum concurrent workers.
            timeout: Maximum wait time for all tools.
        
        Returns:
            Dict mapping tool call IDs to results.
        """
        if not tool_calls:
            return {}
        
        # Create tool name to object mapping
        tool_map = {tool.name: tool for tool in self.tools}
        results = {}
        
        def execute_single_tool(tool_call: Dict) -> Tuple[str, str]:
            tool_name = tool_call.get('name', '')
            tool_args = tool_call.get('args', {})
            tool_id = tool_call.get('id', f'call_{hash(tool_name)}')
            
            if tool_name not in tool_map:
                return (tool_id, f"Tool '{tool_name}' not found.")
            
            try:
                result = tool_map[tool_name].invoke(tool_args)
                return (tool_id, str(result))
            except Exception as e:
                return (tool_id, f"Error executing {tool_name}: {str(e)}")
        
        # Limit workers to number of tools
        num_workers = min(max_workers, len(tool_calls))
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_tool = {
                executor.submit(execute_single_tool, tc): tc
                for tc in tool_calls
            }
            
            for future in as_completed(future_to_tool, timeout=timeout):
                try:
                    tool_id, result = future.result()
                    results[tool_id] = result
                except Exception as e:
                    tool_call = future_to_tool[future]
                    tool_id = tool_call.get('id', 'unknown')
                    results[tool_id] = f"Execution error: {str(e)}"
        
        return results

    def execute_react_loop(
        self,
        messages: list,
        verbose: bool = False,
        max_iterations: int = 5,
        tool_enforcement_msg: str = "",
    ) -> str:
        """
        Executes the shared ReAct tool-calling loop.

        Handles the full cycle of:
            1. Initial LLM call with tool enforcement
            2. Iterative tool execution (parallel when multiple)
            3. Loop detection (duplicate tool call prevention)
            4. JSON tool call fallback
            5. Response cleaning

        Args:
            messages: Initial message list (system prompt + context + user query).
            verbose: Whether to print debug information.
            max_iterations: Maximum tool-calling iterations.
            tool_enforcement_msg: Custom message to force tool usage if LLM
                doesn't call any tools initially.

        Returns:
            str: Cleaned final response.
        """
        # First LLM call - may request tool use
        response = self.llm_with_tools.invoke(messages)

        # Tool enforcement: force tool usage if LLM doesn't call any tools
        if tool_enforcement_msg and not (
            hasattr(response, "tool_calls") and response.tool_calls
        ):
            if verbose:
                print("      [DEBUG] No tools called initially. Forcing tool usage...")
            messages.append(AIMessage(content=response.content))
            messages.append(HumanMessage(content=tool_enforcement_msg))
            response = self.llm_with_tools.invoke(messages)

        iteration = 0
        called_tools = set()        # Track tool signatures for loop detection
        last_tool_results = []      # Store results for fallback

        while (
            hasattr(response, "tool_calls")
            and response.tool_calls
            and iteration < max_iterations
        ):
            messages.append(response)

            # --- Loop Detection: Check for duplicate tool calls ---
            new_calls = []
            duplicate_detected = False
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                try:
                    args_str = json.dumps(tool_args, sort_keys=True)
                except Exception:
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
                    print("      [LOOP] All tool calls are duplicates. Forcing response.")
                # Return last tool result if available
                if last_tool_results:
                    return clean_response(last_tool_results[-1])
                # Otherwise force LLM to respond using existing data
                messages.append(
                    SystemMessage(
                        content="STOP CALLING TOOLS. You already have the data. Respond to the user NOW."
                    )
                )
                response = self.llm_with_tools.invoke(messages)
                break

            # --- Tool Execution (parallel when >1, sequential otherwise) ---
            tools_to_execute = new_calls if new_calls else response.tool_calls[:1]

            if len(tools_to_execute) > 1:
                if verbose:
                    print(f"      [PARALLEL] Executing {len(tools_to_execute)} tools in parallel...")

                tool_results = self.execute_tools_parallel(tools_to_execute, max_workers=4)

                for tool_call in tools_to_execute:
                    tool_id = tool_call.get("id", f"call_{iteration}")
                    tool_name = tool_call.get("name", "unknown")
                    result = tool_results.get(tool_id, f"Tool '{tool_name}' execution failed.")
                    last_tool_results.append(str(result))

                    if verbose:
                        preview = str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
                        print(f"      [TOOL] {tool_name} Result: {preview}")

                    messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))
            else:
                # Single tool - sequential execution
                for tool_call in tools_to_execute:
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call.get("id", f"call_{iteration}")

                    if verbose:
                        print(f"      [TOOL] Calling {tool_name} with args: {tool_args}")

                    tool_result = None
                    for tool in self.tools:
                        if tool.name == tool_name:
                            try:
                                tool_result = tool.invoke(tool_args)
                                last_tool_results.append(str(tool_result))
                                if verbose:
                                    preview = (
                                        str(tool_result)[:100] + "..."
                                        if len(str(tool_result)) > 100
                                        else str(tool_result)
                                    )
                                    print(f"      [TOOL] Result: {preview}")
                            except Exception as e:
                                tool_result = f"Error executing {tool_name}: {str(e)}"
                                if verbose:
                                    print(f"      [TOOL] Error: {tool_result}")
                            break

                    if tool_result is None:
                        tool_result = f"Tool '{tool_name}' not found."

                    messages.append(
                        ToolMessage(content=str(tool_result), tool_call_id=tool_id)
                    )

            response = self.llm_with_tools.invoke(messages)
            iteration += 1

        # JSON tool call fallback (some models embed tool calls in text)
        json_tool_result = self.execute_tool_from_json(response.content, verbose=verbose)
        if json_tool_result:
            return clean_response(json_tool_result)

        return clean_response(response.content)


# ==========================================================================
# Loop Detection Utilities (Shared across agents)
# ==========================================================================


def detect_tool_loop(messages: List, recent_tool_calls: List, lookback: int = 3) -> bool:
    """
    Detects if recent tool calls are duplicates (potential infinite loop).
    
    This utility is shared across weather_agent, transport_agent, and researcher_agent
    to prevent the LLM from calling the same tool repeatedly with the same arguments.
    
    Args:
        messages: Conversation history.
        recent_tool_calls: Tool calls from the latest AI response.
        lookback: Number of previous AI messages to check (default 3).
        
    Returns:
        bool: True if loop detected (duplicate tool calls), False otherwise.
        
    Example:
        >>> if detect_tool_loop(state["messages"], response.tool_calls):
        >>>     # Force response generation instead of calling tools again
        >>>     pass
    """
    if not recent_tool_calls:
        return False
    
    # Get signatures of recent tool calls
    def _get_signature(tool_call) -> str:
        """Creates unique signature for a tool call."""
        try:
            args_str = json.dumps(tool_call.get("args", {}), sort_keys=True)
        except (TypeError, AttributeError):
            args_str = str(getattr(tool_call, "args", {}))
        
        name = (
            tool_call.get("name", "")
            if isinstance(tool_call, dict)
            else getattr(tool_call, "name", "")
        )
        return f"{name}:{args_str}"
    
    # Get signatures of new tool calls
    new_signatures = {_get_signature(tc) for tc in recent_tool_calls}
    
    # Get signatures of recent tool calls from history
    recent_signatures = set()
    ai_msg_count = 0
    
    for msg in reversed(messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            ai_msg_count += 1
            for tc in msg.tool_calls:
                recent_signatures.add(_get_signature(tc))
            if ai_msg_count >= lookback:
                break
    
    # If all new tool calls were already made recently, it's a loop
    return bool(new_signatures) and new_signatures.issubset(recent_signatures)


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 Base Agent Utilities Test\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")

    # Test tool loading
    for agent in ["weather", "transport", "researcher", "planner", "supervisor"]:
        tools = get_agent_tools(agent)
        print(f"\n\033[1m{agent.capitalize()} Agent:\033[0m {len(tools)} tools")
        if tools:
            for t in tools:
                print(f"   - {t.name}")

    print("\n\033[1;32m✅ Base utilities loaded successfully!\033[0m")
