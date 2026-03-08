# ==========================================================================
# Master Thesis
#   - André Filipe Gomes Silvestre, 20240502
# 
# LangGraph compatibility helpers for import-time warnings.
# 
# This module centralizes narrow workarounds for third-party LangGraph import
# behavior so the rest of the codebase can keep clean imports.
# ==========================================================================

# Required libraries:
# pip install langgraph

import warnings
from typing import Final

from langgraph.warnings import LangGraphDeprecatedSinceV10

_TOOLNODE_WARNING_PATTERN: Final[str] = (
    r"AgentStatePydantic has been moved to `langchain\.agents`.*"
)


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=_TOOLNODE_WARNING_PATTERN,
        category=LangGraphDeprecatedSinceV10,
    )
    from langgraph.prebuilt import ToolNode as _LangGraphToolNode


ToolNode = _LangGraphToolNode

__all__ = ["ToolNode"]
