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
import re
import time as time_module
from concurrent.futures import as_completed
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.utils.langsmith_tracing import ContextThreadPoolExecutor

try:
    from config import Config
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from config import Config


_LOCAL_LLM_PROVIDERS = {"lmstudio"}


# ==========================================================================
# Tool Definitions by Agent
# ==========================================================================


def _normalize_messages_for_lmstudio(messages: List[Any]) -> List[Any]:
    """Collapse multiple system messages into one for LM Studio-compatible chat payloads.

    Some LM Studio prompt templates, including the user-provided
    `qwen/qwen3.5-9b` setup validated on 2026-03-17, fail when the chat payload
    contains more than one ``SystemMessage``. LangChain agents in this
    repository often prepend several system messages, for example for language,
    grounding, and retry instructions. To keep those instructions while
    preserving compatibility, we merge all system-message contents into a
    single leading ``SystemMessage`` and keep every non-system message in order.
    """
    if not messages:
        return messages

    system_contents: List[str] = []
    non_system_messages: List[Any] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            content = str(getattr(message, "content", "") or "").strip()
            if content:
                system_contents.append(content)
            continue
        non_system_messages.append(message)

    if len(system_contents) <= 1:
        return messages

    merged_system_message = SystemMessage(content="\n\n".join(system_contents))
    return [merged_system_message, *non_system_messages]


def is_local_provider(provider: Any) -> bool:
    """Returns whether the provider represents a local runtime that should stay sequential."""
    return str(provider or "").strip().lower() in _LOCAL_LLM_PROVIDERS


def get_agent_tools(agent_name: str) -> List:
    """
    Returns the tools for a specific agent.

    This function lazily imports tools to avoid circular imports
    and only loads what's needed for each agent.

    Args:
        agent_name (str): Name of the agent. Supported values are
            `weather`, `transport`, `researcher`, `planner`, `supervisor`,
            and `qa`.

    Returns:
        List: List of LangChain tools for the specified agent. Tool-less agents
            such as planner, supervisor, and qa return an empty list by design.
    """
    if agent_name == "weather":
        from tools.ipma_api import (
            get_current_weather_summary,
            get_portugal_weather_overview,
            get_weather_forecast,
            get_weather_warnings,
        )

        return [
            get_weather_warnings,
            get_weather_forecast,
            get_current_weather_summary,
            get_portugal_weather_overview,
        ]

    elif agent_name == "transport":
        # Metro de Lisboa (Official API with OAuth2)
        from tools.carris_api import (
            carris_find_routes_between,
            carris_get_arrivals,
            carris_get_next_departures,
            carris_get_realtime_vehicles,
            carris_get_routes,
            carris_get_service_frequency,
            carris_get_stops,
            carris_vehicle_eta,
        )

        # Carris Metropolitana (Suburban buses)
        from tools.carrismetropolitana_api import (
            find_bus_routes,
            find_direct_bus_lines,
            get_bus_next_departures,
            get_bus_realtime_locations,
            get_carris_metropolitana_alerts,
            get_carris_metropolitana_stop_info,
            get_real_time_bus_positions,
            search_carris_metropolitana_lines,
        )

        # CP (Comboios de Portugal) - Trains
        from tools.cp_api import (
            get_cp_routes,
            get_train_frequency,
            get_train_schedule,
            get_train_status,
            plan_train_trip,
            search_cp_stations,
        )
        from tools.metrolisboa_api import (
            find_nearest_metro,
            get_all_metro_stations,
            get_metro_frequency,
            get_metro_line_wait_times,
            get_metro_status,
            get_metro_wait_time,
        )

        # Multi-modal transport routing
        from tools.transport_api import (
            get_route_between_stations,
            get_transport_summary,
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
            get_train_frequency,
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
            carris_get_service_frequency,
        ]

    elif agent_name == "researcher":
        from tools.dados_abertos import (
            find_nearby_services,
            find_place_in_datasets,
            get_dataset_details,
            list_available_datasets,
            list_service_categories,
        )
        from tools.visitlisboa_api import (
            get_event_categories,
            get_place_categories,
            search_cultural_events,
            search_lisbon_knowledge,
            search_places_attractions,
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
            list_service_categories,
            search_history_culture,
        ]

    elif agent_name == "planner":
        # Planner has no tools - it synthesizes outputs from other agents
        return []

    elif agent_name == "supervisor":
        # Supervisor has no tools - it only routes to other agents
        return []

    elif agent_name == "qa":
        # QA agent has no tools - it only validates agent outputs
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


def clean_response(content: str, _print: bool = True) -> str:
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
        _print: Whether to print the markdown to terminal (default True).
            Set to False when called from parse_json_response to avoid
            duplicate prints.

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
    # Remove dangling <think> blocks when the model starts reasoning but never closes the tag.
    content = re.sub(r"<think>.*$", "", content, flags=re.DOTALL)

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

    # Clean first (suppress print to avoid duplicate terminal output)
    content = clean_response(content, _print=False)

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


def _cleaned_response_looks_incomplete(cleaned_content: str, raw_content: str = "") -> bool:
    """Detect whether a cleaned final answer is too incomplete to show to the user.

    This mainly guards local models that occasionally stop after a header or leak
    an unfinished reasoning block after tool execution.
    """
    cleaned = str(cleaned_content or "").strip()
    raw = str(raw_content or "")
    if not cleaned:
        return True

    lowered = cleaned.lower()
    if "an error occurred while processing" in lowered or "having difficulty processing your request" in lowered:
        return True

    if "<think>" in raw and len(cleaned) < 120:
        return True

    first_line = cleaned.splitlines()[0].strip()
    if len(cleaned.splitlines()) <= 2 and (
        re.match(r"^###\s+.+$", first_line)
        or re.match(r"^\*\*[^*]+\*\*$", first_line)
    ):
        return True

    return False


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
        - Shared ReAct loop execution
        - Optional parallel tool execution
        - Safe LLM invocation with Azure content-filter retry handling

    Tool-less agents:
        Planner, Supervisor, and QA intentionally skip tool binding and use the
        base LLM directly.
    """

    def __init__(self, agent_name: str):
        """
        Initializes the base agent.

        Args:
            agent_name: Name of this agent (for config lookup).
        """
        self.agent_name = agent_name
        self.tools = get_agent_tools(agent_name)
        agent_config = Config.AGENT_MODELS().get(
            agent_name, Config.DEFAULT_AGENT_MODEL()
        )
        self.llm_provider = agent_config.get("provider", Config.MODEL_PROVIDER)
        self.llm_model_name = agent_config.get("model", "Unknown")
        self.llm_temperature = agent_config.get("temperature", Config.TEMPERATURE)
        self.llm = get_agent_llm(agent_name)
        self._llm_usage_events: List[Dict[str, Any]] = []
        self._llm_usage_call_index = 0
        self._tool_calls_log: List[Dict[str, Any]] = []

        # Bind tools if this agent has any
        if self.tools:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        else:
            self.llm_with_tools = self.llm

    def init_llm(
        self,
        provider: str = "azure",
        model: str = None,
        temperature: float = 0.0,
    ) -> None:
        """
        Re-initializes the agent's LLM with custom provider/model settings.

        Useful for benchmarking and evaluation where per-run model configuration
        is needed. Rebinds tools automatically.

        Args:
            provider: LLM provider name (e.g. "azure", "lmstudio", "openai").
            model: Model name override (uses provider default if None).
            temperature: Sampling temperature.
        """
        from agent.llm_factory import LLMFactory

        self.llm = LLMFactory.get_llm(
            provider=provider,
            model=model,
            temperature=temperature,
        )
        model_info = LLMFactory.get_model_info(self.llm)
        self.llm_provider = provider
        self.llm_model_name = model_info.get("model", model or self.llm_model_name)
        self.llm_temperature = temperature
        if self.tools:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        else:
            self.llm_with_tools = self.llm

    def reset_llm_usage_tracking(self) -> None:
        """Resets the in-memory LLM and Tool usage tracker for this agent."""
        self._llm_usage_events = []
        self._llm_usage_call_index = 0
        self._tool_calls_log = []

    def get_llm_usage_events(self) -> List[Dict[str, Any]]:
        """Returns a defensive copy of the raw LLM usage events."""
        return deepcopy(self._llm_usage_events)

    def get_tool_calls_log(self) -> List[Dict[str, Any]]:
        """Returns a defensive copy of the logged tool calls."""
        return deepcopy(getattr(self, "_tool_calls_log", []))

    def _invoke_tool(
        self,
        tool: Any,
        args: Optional[dict] = None,
        *,
        tool_name: Optional[str] = None,
        verbose: bool = False,
    ) -> Any:
        """Invoke a loaded tool while recording the call for analytics.

        Args:
            tool: Loaded LangChain tool-like object.
            args: Tool arguments.
            tool_name: Optional explicit tool name override.
            verbose: Whether to print a debug line before execution.

        Returns:
            Any: Raw tool result.
        """
        resolved_args = args if isinstance(args, dict) else {}
        resolved_name = str(tool_name or getattr(tool, "name", "unknown")).strip() or "unknown"
        self._record_tool_call(resolved_name, resolved_args)
        if verbose:
            print(f"      [TOOL] Calling {resolved_name} with args: {resolved_args}")
        return tool.invoke(resolved_args)

    def _invoke_tool_by_name(
        self,
        tool_name: str,
        args: Optional[dict] = None,
        *,
        verbose: bool = False,
    ) -> Any:
        """Resolve a loaded tool by name and invoke it with analytics logging."""
        resolved_name = str(tool_name or "").strip()
        if not resolved_name:
            raise ValueError("Tool name cannot be empty.")

        for tool in self.tools:
            if getattr(tool, "name", "") == resolved_name:
                return self._invoke_tool(tool, args, tool_name=resolved_name, verbose=verbose)

        raise ValueError(f"Tool '{resolved_name}' not found.")

    def _format_tool_result_for_fallback(
        self,
        *,
        tool_name: str,
        tool_args: Optional[dict],
        result: Any,
        language: str,
    ) -> str:
        """Formats a tool result for ReAct fallback paths when a subclass exposes a formatter."""
        raw_result = str(result)
        formatter = getattr(self, "_format_deterministic_tool_result", None)
        if callable(formatter):
            try:
                formatted = formatter(
                    tool_name=tool_name,
                    tool_args=tool_args or {},
                    result=raw_result,
                    language=language,
                )
                if formatted:
                    return clean_response(str(formatted))
            except Exception:
                pass

        return clean_response(raw_result)

    def _record_tool_call(self, tool_name: str, args: dict) -> None:
        """Records a tool call to the agent's internal log."""
        if not hasattr(self, "_tool_calls_log") or self._tool_calls_log is None:
            self._tool_calls_log = []
        self._tool_calls_log.append({
            "tool_name": tool_name,
            "args": deepcopy(args)
        })

    def get_llm_usage_summary(self) -> Dict[str, Any]:
        """
        Returns an aggregated view of tracked LLM token usage for this agent.

        Returns:
            Dict[str, Any]: Summary with total tokens, call count, and the raw
            per-call breakdown.
        """
        totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        for event in self._llm_usage_events:
            tokens = event.get("tokens", {})
            totals["input_tokens"] += int(tokens.get("input_tokens", 0) or 0)
            totals["output_tokens"] += int(tokens.get("output_tokens", 0) or 0)
            totals["total_tokens"] += int(tokens.get("total_tokens", 0) or 0)

        return {
            "agent_name": self.agent_name,
            "provider": self.llm_provider,
            "model": self.llm_model_name,
            "model_id": f"{self.llm_provider}::{self.llm_model_name}",
            "call_count": len(self._llm_usage_events),
            "usage_available": any(event.get("usage_available", False) for event in self._llm_usage_events),
            "tokens": totals,
            "llm_usage_breakdown": self.get_llm_usage_events(),
        }

    def _record_llm_usage(self, llm: Any, response: Any) -> None:
        """
        Records token usage for a single LLM invocation.

        Args:
            llm: LLM instance used for the call.
            response: Raw LLM response object.
        """
        from agent.llm_factory import LLMFactory

        usage = LLMFactory.extract_usage_metadata(response)
        model_info = LLMFactory.get_model_info(llm)
        model_name = model_info.get("model", self.llm_model_name)

        self._llm_usage_call_index += 1
        self._llm_usage_events.append(
            {
                "call_index": self._llm_usage_call_index,
                "agent_name": self.agent_name,
                "provider": self.llm_provider,
                "model": model_name,
                "model_id": f"{self.llm_provider}::{model_name}",
                "tokens": {
                    "input_tokens": int(usage.get("input_tokens", 0) or 0),
                    "output_tokens": int(usage.get("output_tokens", 0) or 0),
                    "total_tokens": int(usage.get("total_tokens", 0) or 0),
                },
                "usage_available": bool(usage.get("usage_available", False)),
            }
        )

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
                    return str(self._invoke_tool(tool, tool_args, tool_name=tool_name, verbose=verbose))
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
            timeout: Maximum wait time passed to `as_completed()` while waiting
                for the batch to finish.

        Returns:
            Dict mapping tool call IDs to results.

        Notes:
            If the overall wait exceeds `timeout`, `as_completed()` may raise a
            timeout-related exception to the caller. Individual tool execution
            errors are captured and returned as error strings.
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
                result = self._invoke_tool(tool_map[tool_name], tool_args, tool_name=tool_name)
                return (tool_id, str(result))
            except Exception as e:
                return (tool_id, f"Error executing {tool_name}: {str(e)}")

        # Limit workers to number of tools
        num_workers = min(max_workers, len(tool_calls))

        with ContextThreadPoolExecutor(max_workers=num_workers) as executor:
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

    def _safe_llm_invoke(self, llm, messages: list, retries: int = 2, verbose: bool = False):
        """
        Invokes the LLM with targeted retry logic for Azure content-filter false positives.

        Azure OpenAI may probabilistically flag benign prompts as "jailbreak".
        This method retries the call with exponential backoff since the same
        request often succeeds on a second attempt.

        Args:
            llm: The LLM instance (with or without tools bound).
            messages: The messages to send.
            retries: Maximum number of retry attempts for the specific content-
                filter patterns handled here.
            verbose: Whether to print debug information.

        Returns:
            The LLM response object.

        Raises:
            The original exception if all retries fail and it's not a
            content filter issue, or re-raises after exhausting retries.
        """
        last_exception = None
        prepared_messages = (
            _normalize_messages_for_lmstudio(messages)
            if getattr(self, "llm_provider", "") == "lmstudio"
            else messages
        )
        for attempt in range(retries + 1):
            try:
                response = llm.invoke(prepared_messages)
                self._record_llm_usage(llm, response)
                return response
            except Exception as e:
                error_str = str(e).lower()
                is_content_filter = (
                    "content_filter" in error_str
                    or "responsibleaipolicyviolation" in error_str
                    or "jailbreak" in error_str
                )
                if is_content_filter and attempt < retries:
                    wait = 1.5 * (attempt + 1)
                    print(f"      [RETRY] Azure content filter triggered (attempt {attempt + 1}/{retries + 1}). Retrying in {wait}s...")
                    time_module.sleep(wait)
                    last_exception = e
                    continue
                raise
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("LLM invoke failed after all retries")

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
            5. Forced response generation when loops are detected
            6. Response cleaning

        Args:
            messages: Initial message list (system prompt + context + user query).
            verbose: Whether to print debug information.
            max_iterations: Maximum tool-calling iterations.
            tool_enforcement_msg: Custom message to force tool usage if LLM
                doesn't call any tools initially.

        Returns:
            str: Cleaned final response or, in certain loop-break cases, the
            latest available tool result converted into user-facing text.
        """
        # First LLM call - may request tool use (with retry for Azure content filter)
        response = self._safe_llm_invoke(self.llm_with_tools, messages, verbose=verbose)

        # Tool enforcement: force tool usage if LLM doesn't call any tools
        if tool_enforcement_msg and not (
            hasattr(response, "tool_calls") and response.tool_calls
        ):
            if verbose:
                print("      [DEBUG] No tools called initially. Forcing tool usage...")
            messages.append(AIMessage(content=response.content))
            messages.append(HumanMessage(content=tool_enforcement_msg))
            response = self._safe_llm_invoke(self.llm_with_tools, messages, verbose=verbose)

        iteration = 0
        called_tools = set()        # Track tool signatures for loop detection
        last_tool_results = []      # Store tool payloads for fallback formatting

        from agent.utils.response_formatter import infer_response_language

        fallback_query = ""
        for message in reversed(messages):
            if isinstance(message, HumanMessage) and getattr(message, "content", None):
                fallback_query = str(message.content)
                break
        fallback_language = infer_response_language(user_query=fallback_query, default="en")

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
                    latest = last_tool_results[-1]
                    return self._format_tool_result_for_fallback(
                        tool_name=latest["name"],
                        tool_args=latest["args"],
                        result=latest["result"],
                        language=fallback_language,
                    )
                # Otherwise force LLM to respond using existing data
                messages.append(
                    SystemMessage(
                        content="STOP CALLING TOOLS. You already have the data. Respond to the user NOW."
                    )
                )
                response = self._safe_llm_invoke(self.llm_with_tools, messages, verbose=verbose)
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
                    tool_args = tool_call.get("args", {})
                    result = tool_results.get(tool_id, f"Tool '{tool_name}' execution failed.")
                    last_tool_results.append(
                        {"name": tool_name, "args": tool_args, "result": str(result)}
                    )

                    if "error" in str(result).lower() or "failed" in str(result).lower():
                        print(f"      [ERROR] Tool {tool_name} failed: {str(result)[:150]}...")

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

                    self._record_tool_call(tool_name, tool_args)
                    tool_result = None
                    for tool in self.tools:
                        if tool.name == tool_name:
                            try:
                                tool_result = tool.invoke(tool_args)
                                last_tool_results.append(
                                    {"name": tool_name, "args": tool_args, "result": str(tool_result)}
                                )
                                if verbose:
                                    preview = (
                                        str(tool_result)[:100] + "..."
                                        if len(str(tool_result)) > 100
                                        else str(tool_result)
                                    )
                                    print(f"      [TOOL] Result: {preview}")
                            except Exception as e:
                                tool_result = f"Error executing {tool_name}: {str(e)}"
                                print(f"      [ERROR] Tool {tool_name} failed: {str(e)[:150]}")
                            break

                    if tool_result is None:
                        tool_result = f"Tool '{tool_name}' not found."

                    messages.append(
                        ToolMessage(content=str(tool_result), tool_call_id=tool_id)
                    )

            response = self._safe_llm_invoke(self.llm_with_tools, messages, verbose=verbose)
            iteration += 1

        # JSON tool call fallback (some models embed tool calls in text)
        json_tool_result = self.execute_tool_from_json(response.content, verbose=verbose)
        if json_tool_result:
            return clean_response(json_tool_result)

        cleaned_response = clean_response(response.content)
        if _cleaned_response_looks_incomplete(cleaned_response, getattr(response, "content", "")) and last_tool_results:
            formatted_tool_results = [
                self._format_tool_result_for_fallback(
                    tool_name=payload["name"],
                    tool_args=payload["args"],
                    result=payload["result"],
                    language=fallback_language,
                )
                for payload in last_tool_results
            ]
            combined_tool_fallback = clean_response("\n\n".join(formatted_tool_results))
            if not _cleaned_response_looks_incomplete(combined_tool_fallback):
                if verbose:
                    print("      [FALLBACK] Final reply looked incomplete. Returning combined tool results instead.")
                return combined_tool_fallback
            if verbose:
                print("      [FALLBACK] Final reply looked incomplete. Returning latest tool result instead.")
            latest = last_tool_results[-1]
            return self._format_tool_result_for_fallback(
                tool_name=latest["name"],
                tool_args=latest["args"],
                result=latest["result"],
                language=fallback_language,
            )

        return cleaned_response


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

    counters = {"passed": 0, "failed": 0}

    def _check(condition: bool, label: str) -> None:
        if condition:
            counters["passed"] += 1
            print(f"   \033[1;32m✅ PASS\033[0m: {label}")
        else:
            counters["failed"] += 1
            print(f"   \033[1;31m❌ FAIL\033[0m: {label}")

    # Test tool loading
    for agent in ["weather", "transport", "researcher", "planner", "supervisor"]:
        tools = get_agent_tools(agent)
        print(f"\n\033[1m{agent.capitalize()} Agent:\033[0m {len(tools)} tools")
        if tools:
            for t in tools:
                print(f"   - {t.name}")

    print("\n\033[1m🔎 Deterministic helper checks:\033[0m")
    _check(is_local_provider("lmstudio") is True, "LM Studio is treated as a local provider")
    _check(is_local_provider("azure") is False, "Azure is not treated as a local provider")
    _check(clean_response("<think>internal</think>Hello Lisbon!") == "Hello Lisbon!", "clean_response strips leaked think blocks")
    _check(
        detect_tool_loop(
            [AIMessage(content="", tool_calls=[{"name": "get_metro_status", "args": {}, "id": "call_1"}])],
            [{"name": "get_metro_status", "args": {}, "id": "call_2"}],
            lookback=1,
        ) is True,
        "Loop detector flags repeated tool calls",
    )

    print(f"\n\033[1mSummary:\033[0m Passed={counters['passed']} Failed={counters['failed']}")
    if counters["failed"]:
        raise SystemExit(1)
    print("\n\033[1;32m✅ Base utilities loaded successfully!\033[0m")
