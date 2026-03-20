# ==========================================================================
# Master Thesis - Weather Agent
#   - André Filipe Gomes Silvestre, 20240502
#
#   Specialized agent for weather-related queries using IPMA data.
#   Uses BaseAgent.execute_react_loop() for tool execution.
# ==========================================================================

import re
import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

from agent.agents.base import BaseAgent, traceable
from agent.prompts.weather import get_weather_prompt
from agent.state import AgentState
from agent.utils.langgraph_compat import ToolNode
from agent.utils.response_formatter import (
    finalize_worker_response,
    infer_response_language,
)


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

    @staticmethod
    def _infer_weather_query_language(user_message: str) -> str:
        """Adds a small PT-PT heuristic for short follow-ups where generic language inference is weak."""
        query = (user_message or "").lower()
        pt_markers = [
            "amanhã",
            "amanha",
            "daqui",
            "semana",
            "previsão",
            "previsao",
            "avisos",
            "hoje",
            "tempo",
            "próxim",
            "proxim",
        ]
        if any(marker in query for marker in pt_markers) or re.search(r"[ãõáéíóúç]", query):
            return "pt"
        return infer_response_language(user_query=user_message, default="en")

    @staticmethod
    def _is_content_filter_error(error: Exception) -> bool:
        """Returns whether an exception is an Azure content-filter false positive."""
        error_str = str(error).lower()
        return (
            "content_filter" in error_str
            or "responsibleaipolicyviolation" in error_str
            or "jailbreak" in error_str
        )

    @staticmethod
    def _build_messages(system_prompt: str, user_message: str, context: str = "") -> list:
        """Builds the message list for a weather invocation."""
        language = WeatherAgent._infer_weather_query_language(user_message)
        language_instruction = (
            "Respond ENTIRELY in Portuguese (PT-PT)."
            if language == "pt"
            else "Respond ENTIRELY in English."
        )

        messages = [
            SystemMessage(content=system_prompt),
            SystemMessage(content=language_instruction),
        ]

        if context:
            messages.append(SystemMessage(content=f"Context from other agents:\n{context}"))

        messages.append(HumanMessage(content=user_message))
        return messages

    def _get_tool_by_name(self, tool_name: str):
        """Returns a loaded tool by name, or None if not found."""
        for tool in self.tools:
            if getattr(tool, "name", "") == tool_name:
                return tool
        return None

    @staticmethod
    def _has_english_language_drift(response: str, language: str) -> bool:
        """Detects when an English weather answer still leaks obvious PT-PT content."""
        if language != "en" or not response:
            return False

        drift_patterns = [
            r"\bsegunda-feira\b",
            r"\bterça-feira\b",
            r"\bquarta-feira\b",
            r"\bquinta-feira\b",
            r"\bsexta-feira\b",
            r"\bsábado\b",
            r"\bdomingo\b",
            r"\bchuva\b",
            r"\baguaceiros\b",
            r"\bfraca\b",
            r"\bnoroeste\b",
            r"\bvista casaco\b",
            r"\bguarda-chuva\b",
        ]
        matches = sum(1 for pattern in drift_patterns if re.search(pattern, response, re.IGNORECASE))
        return matches >= 2

    @staticmethod
    def _is_current_weather_query(user_message: str) -> bool:
        """Detects simple current-weather queries that should use the summary tool directly."""
        query = (user_message or "").lower()
        return bool(
            "right now" in query
            or "current weather summary" in query
            or "current temperature" in query
            or "agora" in query
            or re.search(r"\b(weather|tempo)\b.*\b(today|hoje|now|agora)\b", query)
            or re.search(r"\b(today|hoje)\b.*\b(weather|tempo)\b", query)
        )

    @staticmethod
    def _extract_requested_forecast_days(user_message: str) -> Optional[int]:
        """Extracts the requested forecast window from simple weather queries."""
        query = (user_message or "").lower()

        explicit_days = re.search(r"\b([1-5])\s*(?:-|\s)?\s*(?:day|days|dia|dias)\b", query)
        if explicit_days:
            return max(1, min(int(explicit_days.group(1)), 5))

        if any(term in query for term in ["tomorrow", "amanhã", "amanha"]):
            return 2

        if any(term in query for term in ["week", "semana", "weekend", "fim de semana"]):
            return 5

        if any(term in query for term in ["forecast", "previsão", "previsao", "next days", "próximos dias", "proximos dias"]):
            return 3

        return None

    @staticmethod
    def _extract_explicit_forecast_date(user_message: str) -> Optional[datetime]:
        """Extracts an explicit forecast date from common ISO or DD/MM/YYYY forms."""
        query = user_message or ""
        for pattern, fmt in (
            (r"\b(\d{4}-\d{2}-\d{2})\b", "%Y-%m-%d"),
            (r"\b(\d{2}/\d{2}/\d{4})\b", "%d/%m/%Y"),
        ):
            match = re.search(pattern, query)
            if not match:
                continue
            try:
                return datetime.strptime(match.group(1), fmt)
            except ValueError:
                continue
        return None

    @classmethod
    def _is_beyond_forecast_horizon_query(cls, user_message: str) -> bool:
        """Returns whether the query clearly asks for weather beyond IPMA's 5-day horizon."""
        query = (user_message or "").lower()

        if re.search(r"\b([6-9]|\d{2,})\s*(?:-|\s)?\s*(?:day|days|dia|dias)\b", query):
            return True

        beyond_horizon_patterns = [
            r"\bin\s+(?:a|one)\s+week\b",
            r"\ba\s+week\s+from\s+now\b",
            r"\bnext\s+week\b",
            r"\bdaqui\s+a\s+uma\s+semana\b",
            r"\bpr[oó]xima\s+semana\b",
            r"\bpr[oó]ximos?\s+7\s+dias\b",
        ]
        if any(re.search(pattern, query) for pattern in beyond_horizon_patterns):
            return True

        explicit_date = cls._extract_explicit_forecast_date(user_message)
        if explicit_date is not None:
            delta_days = (explicit_date.date() - datetime.now().date()).days
            if delta_days > 4:
                return True

        return False

    @staticmethod
    def _build_forecast_horizon_limit_message(language: str) -> str:
        """Builds a localized message when the user asks beyond the 5-day forecast horizon."""
        max_supported_date = (datetime.now() + timedelta(days=4)).strftime("%Y-%m-%d")
        if language == "pt":
            return (
                "⚠️ Só tenho previsão meteorológica fiável do IPMA para Lisboa para os próximos 5 dias, "
                f"por isso não consigo confirmar o tempo para esse horizonte. O limite atual vai até {max_supported_date}."
            )
        return (
            "⚠️ I only have reliable IPMA weather forecast data for Lisbon for the next 5 days, "
            f"so I can't confirm the weather for that time horizon. The current reliable limit runs through {max_supported_date}."
        )

    @classmethod
    def _is_simple_forecast_query(cls, user_message: str) -> bool:
        """Detects standalone forecast/warnings queries that can skip free-form synthesis."""
        query = (user_message or "").lower()
        planning_terms = [
            "plan", "itinerary", "roteiro", "plano", "activity", "activities",
            "visit", "visitar", "museum", "museu", "restaurant", "restaurante",
        ]

        if any(term in query for term in planning_terms):
            return False

        return bool(
            cls._extract_requested_forecast_days(user_message)
            or any(term in query for term in ["warning", "warnings", "aviso", "avisos"])
        )

    def _run_direct_tool_fallback(
        self,
        user_message: str,
        *,
        force_forecast_days: Optional[int] = None,
        include_warnings: bool = False,
    ) -> str:
        """
        Runs a deterministic tool-only fallback when Azure blocks weather prompt
        attempts. This preserves real data access without relying on another
        model call.
        """
        language = self._infer_weather_query_language(user_message)
        if self._is_beyond_forecast_horizon_query(user_message):
            return self._build_forecast_horizon_limit_message(language)

        query = user_message.lower()
        requested_forecast_days = force_forecast_days or self._extract_requested_forecast_days(user_message)
        wants_warnings = include_warnings or any(term in query for term in ["warning", "warnings", "aviso", "avisos"])
        wants_forecast = requested_forecast_days is not None
        wants_current = any(term in query for term in ["today", "current", "now", "hoje", "agora"]) or (
            any(term in query for term in ["weather", "tempo"]) and not wants_warnings and not wants_forecast
        )

        sections = []

        if (wants_warnings or (wants_forecast and not wants_current)) and not wants_current:
            warnings_tool = self._get_tool_by_name("get_weather_warnings")
            if warnings_tool:
                sections.append(warnings_tool.invoke({"area": "LSB"}))

        if wants_current:
            current_tool = self._get_tool_by_name("get_current_weather_summary")
            if current_tool:
                sections.append(current_tool.invoke({}))

        forecast_tool = self._get_tool_by_name("get_weather_forecast")
        if forecast_tool and wants_forecast and requested_forecast_days:
            sections.append(forecast_tool.invoke({"days": requested_forecast_days}))

        if not sections:
            current_tool = self._get_tool_by_name("get_current_weather_summary")
            if current_tool:
                sections.append(current_tool.invoke({}))

        if not sections:
            return "Unable to retrieve weather data at the moment."

        return "\n\n---\n\n".join(section for section in sections if section).strip()

    @staticmethod
    def _build_tool_call(name: str, args: dict) -> AIMessage:
        """Creates a deterministic tool call message for the subgraph."""
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": name,
                    "args": args,
                    "id": f"auto_{uuid.uuid4().hex}",
                    "type": "tool_call",
                }
            ],
        )

    @staticmethod
    def _build_language_instruction(language: str) -> str:
        """Builds a compact language instruction for subgraph LLM steps."""
        return (
            "Respond ENTIRELY in Portuguese (PT-PT)."
            if language == "pt"
            else "Respond ENTIRELY in English."
        )

    def _ensure_subgraph_messages(self, messages: list, language: str) -> list:
        """Ensures weather subgraph LLM calls receive system and language instructions."""
        updated_messages = list(messages)
        if not updated_messages or not isinstance(updated_messages[0], SystemMessage):
            updated_messages = [SystemMessage(content=self.system_prompt)] + updated_messages

        if not any(
            isinstance(message, SystemMessage)
            and "Respond ENTIRELY" in str(message.content)
            for message in updated_messages[:3]
        ):
            updated_messages = [
                updated_messages[0],
                SystemMessage(content=self._build_language_instruction(language)),
                *updated_messages[1:],
            ]

        return updated_messages

    @classmethod
    def _build_deterministic_subgraph_tool_call(cls, user_message: str) -> Optional[AIMessage]:
        """Routes obvious weather queries to their canonical tool in the subgraph."""
        query = user_message.lower().strip()

        if "portugal-wide" in query or "portugal wide" in query or "portugal-wide weather overview" in query:
            return cls._build_tool_call("get_portugal_weather_overview", {"day": 0})

        if "warning" in query or "warnings" in query or "avisos" in query:
            return cls._build_tool_call("get_weather_warnings", {"area": "LSB"})

        forecast_days = cls._extract_requested_forecast_days(user_message)
        if forecast_days:
            return cls._build_tool_call("get_weather_forecast", {"days": forecast_days})

        if "current weather summary" in query or "right now" in query or ("weather" in query and "today" in query):
            return cls._build_tool_call("get_current_weather_summary", {})

        return None

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
        # Extract explicit language preference from context if provided
        import re
        language_match = re.search(r"User language:\s*(en|pt)", context, re.IGNORECASE)
        if language_match:
            language = language_match.group(1).lower()
        else:
            language = self._infer_weather_query_language(user_message)
        if self._is_beyond_forecast_horizon_query(user_message):
            return finalize_worker_response(
                self._build_forecast_horizon_limit_message(language),
                agent_name="weather",
                user_query=user_message,
                language=language,
            )

        messages = self._build_messages(self.system_prompt, user_message, context)
        tool_enforcement_msg = (
            "You MUST use a tool (like get_current_weather_summary) to get real data. "
            "Do NOT answer from your knowledge base. Call the tool now."
        )

        if self._is_current_weather_query(user_message) or self._is_simple_forecast_query(user_message):
            response = self._run_direct_tool_fallback(user_message)
            return finalize_worker_response(
                response,
                agent_name="weather",
                user_query=user_message,
                language=language,
            )

        try:
            response = self.execute_react_loop(
                messages=messages,
                verbose=verbose,
                max_iterations=5,
                tool_enforcement_msg=tool_enforcement_msg,
            )
        except Exception as e:
            if not self._is_content_filter_error(e):
                raise

            if verbose:
                print("      [WEATHER] Retrying with safe prompt variant after content filter...")

            safe_messages = self._build_messages(
                get_weather_prompt(safe_mode=True),
                user_message,
                context,
            )
            try:
                response = self.execute_react_loop(
                    messages=safe_messages,
                    verbose=verbose,
                    max_iterations=5,
                    tool_enforcement_msg=tool_enforcement_msg,
                )
            except Exception as safe_error:
                if not self._is_content_filter_error(safe_error):
                    raise

                if verbose:
                    print("      [WEATHER] Falling back to direct tool invocation after repeated content-filter blocks...")

                response = self._run_direct_tool_fallback(user_message)

        if self._has_english_language_drift(response, language):
            if verbose:
                print("      [WEATHER] Detected language drift in EN response, switching to deterministic tool output...")
            response = self._run_direct_tool_fallback(user_message)

        return finalize_worker_response(
            response,
            agent_name="weather",
            user_query=user_message,
            language=language,
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

            user_message = None
            for message in reversed(messages):
                if isinstance(message, HumanMessage) and message.content:
                    user_message = str(message.content)
                    break

            language = self._infer_weather_query_language(user_message or "")

            if user_message and self._is_beyond_forecast_horizon_query(user_message):
                return {
                    "messages": [
                        AIMessage(
                            content=self._build_forecast_horizon_limit_message(language)
                        )
                    ]
                }

            last_message = messages[-1] if messages else None
            if isinstance(last_message, ToolMessage):
                response = self._safe_llm_invoke(
                    self.llm_with_tools,
                    self._ensure_subgraph_messages(messages, language),
                )
                return {"messages": [response]}

            if user_message:
                deterministic_call = self._build_deterministic_subgraph_tool_call(user_message)
                if deterministic_call is not None:
                    return {"messages": [deterministic_call]}

            response = self._safe_llm_invoke(
                self.llm_with_tools,
                self._ensure_subgraph_messages(messages, language),
            )
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
