# ==========================================================================
# Master Thesis - Structured Planner Support
#   - André Filipe Gomes Silvestre, 20240502
#
#   Public package interface for the structured PlannerAgent support modules:
#   evidence extraction, dataclass contracts, JSON prompt construction, quality
#   gating, and deterministic Markdown rendering.
# ==========================================================================
"""Public exports for the structured planner support package."""

from agent.planning.evidence import EvidenceBundle, build_evidence_bundle
from agent.planning.models import EvidenceCard, PlanBlock, PlanDraft, SourceRef
from agent.planning.prompts import build_structured_plan_messages
from agent.planning.quality import parse_plan_draft_json, validate_plan_draft
from agent.planning.renderer import render_plan_markdown

__all__ = [
    "EvidenceBundle",
    "EvidenceCard",
    "PlanBlock",
    "PlanDraft",
    "SourceRef",
    "build_evidence_bundle",
    "build_structured_plan_messages",
    "parse_plan_draft_json",
    "validate_plan_draft",
    "render_plan_markdown",
]
