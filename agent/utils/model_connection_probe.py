# ==========================================================================
# Master Thesis
#   - André Filipe Gomes Silvestre, 20240502
#
# Capability check for visitor model connections.
#
# Sends one minimal tool-calling request through the same chat-model client
# the agents use (streaming, usage reporting and tool binding included), so a
# connection that passes is one the agents can use. LangSmith tracing is off
# for the call, and failures are reduced to reason codes: provider error
# bodies, which can echo part of a key, never reach the UI or the logs.
# ==========================================================================

# Required libraries:
# pip install langchain-core langchain-openai openai

from __future__ import annotations

import logging
from typing import Optional

import openai
from langchain_core.messages import HumanMessage

from agent.llm_factory import LLMConnection, LLMFactory
from agent.utils.langsmith_tracing import tracing_context

logger = logging.getLogger(__name__)

CHECK_TIMEOUT_SECONDS = 60
CHECK_TOOL_NAME = "report_status"
_CHECK_TOOL = {
    "type": "function",
    "function": {
        "name": CHECK_TOOL_NAME,
        "description": "Report that the connection works.",
        "parameters": {
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["ok"]}},
            "required": ["status"],
        },
    },
}
_CHECK_PROMPT = f"Call the {CHECK_TOOL_NAME} tool with status set to ok."


def _failure_reason(exc: Exception) -> str:
    """Map a provider exception to a reason code without reading secrets.

    Args:
        exc: Exception raised by the chat-model call.

    Returns:
        str: ``auth``, ``not_found``, ``rate_limit``, ``connection``,
        ``bad_request`` or ``unknown``.
    """
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return "auth"
    if isinstance(exc, openai.NotFoundError):
        return "not_found"
    if isinstance(exc, openai.RateLimitError):
        return "rate_limit"
    if isinstance(exc, openai.APIConnectionError):
        return "connection"
    if isinstance(exc, (openai.BadRequestError, openai.UnprocessableEntityError)):
        # Some providers, such as Gemini, answer an invalid key with HTTP 400.
        return "auth" if "api key" in str(exc).lower() else "bad_request"
    return "unknown"


def check_tool_calling(connection: LLMConnection) -> Optional[str]:
    """Check that a visitor connection answers with a tool call.

    Args:
        connection: Visitor endpoint, model and key to check.

    Returns:
        Optional[str]: None when the model called the check tool, otherwise a
        reason code: ``auth``, ``not_found``, ``rate_limit``, ``connection``,
        ``bad_request``, ``no_tool_call`` or ``unknown``.
    """
    llm = LLMFactory.get_connection_llm(
        connection,
        timeout=CHECK_TIMEOUT_SECONDS,
        max_retries=1,
    )
    try:
        with tracing_context(enabled=False):
            response = llm.bind_tools([_CHECK_TOOL]).invoke([HumanMessage(content=_CHECK_PROMPT)])
    except Exception as exc:
        reason = _failure_reason(exc)
        logger.warning("Model capability check failed: %s (%s)", reason, type(exc).__name__)
        return reason

    tool_calls = getattr(response, "tool_calls", None) or []
    if not any(call.get("name") == CHECK_TOOL_NAME for call in tool_calls):
        logger.warning("Model capability check failed: no_tool_call")
        return "no_tool_call"
    return None
