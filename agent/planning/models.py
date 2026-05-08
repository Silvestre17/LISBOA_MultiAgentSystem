# ==========================================================================
# Master Thesis - Structured Planner Models
#   - André Filipe Gomes Silvestre, 20240502
#
#   Defines lightweight dataclass contracts for the PlannerAgent JSON synthesis
#   path. These models avoid third-party validation dependencies while giving
#   the deterministic renderer a stable structure for evidence-grounded plans.
# ==========================================================================
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class SourceRef:
    """Public evidence source that may be cited in the final answer.

    Attributes:
        id: Internal source identifier used by evidence cards.
        label_en: English source label for rendered answers.
        label_pt: Portuguese source label for rendered answers.
        url: Public source URL used in source footers.
    """

    id: str
    label_en: str
    label_pt: str
    url: str


@dataclass
class EvidenceCard:
    """Compact evidence item available to PlannerAgent.

    Attributes:
        id: Stable identifier for references inside the planner prompt.
        kind: Evidence category, such as weather, transport, place, or event.
        title: Short human-readable evidence title.
        summary: Compact summary of the evidence item.
        fields: Optional labeled details extracted from worker output.
        source_ids: Public source identifiers supporting this card.
        confidence: Grounding status carried into planning.
        limitations: Card-specific caveats that must not be dropped.
    """

    id: str
    kind: str
    title: str
    summary: str = ""
    fields: Dict[str, str] = field(default_factory=dict)
    source_ids: List[str] = field(default_factory=list)
    confidence: str = "grounded"
    limitations: List[str] = field(default_factory=list)

    def to_prompt_text(self, language: str = "en") -> str:
        """Render this card as compact text for the structured planner prompt.

        Args:
            language: Requested response language. The current card contract is
                language-neutral, but the argument keeps the interface aligned
                with the bundle renderer.

        Returns:
            Plain text representation consumed by the planner prompt.
        """
        lines = [f"- id: {self.id}", f"  kind: {self.kind}", f"  title: {self.title}"]
        if self.summary:
            lines.append(f"  summary: {self.summary}")
        for key, value in self.fields.items():
            if value:
                lines.append(f"  {key}: {value}")
        if self.limitations:
            lines.append("  limitations: " + "; ".join(self.limitations[:3]))
        if self.source_ids:
            lines.append("  source_ids: " + ", ".join(dict.fromkeys(self.source_ids)))
        return "\n".join(lines)


@dataclass
class PlanBlock:
    """One ordered block of an itinerary or mobility plan.

    Attributes:
        title: User-facing block title.
        kind: Block category used by the deterministic renderer.
        purpose: Explanation of why the block belongs in the plan.
        details: Grounded place, event, service, or activity details.
        movement: Grounded transport or walking details.
        weather: Weather adaptations relevant to this block.
        limitations: Block-level caveats or missing confirmations.
        source_ids: Public sources materially used by this block.
    """

    title: str
    kind: str = "activity"
    purpose: str = ""
    details: List[str] = field(default_factory=list)
    movement: List[str] = field(default_factory=list)
    weather: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    source_ids: List[str] = field(default_factory=list)


@dataclass
class PlanDraft:
    """Structured planner output rendered deterministically to Markdown.

    Attributes:
        title: Short answer title.
        direct_answer: First-sentence answer to the user's request.
        constraints_used: User constraints or evidence constraints applied.
        blocks: Ordered itinerary or plan blocks.
        movement_logic: Cross-block movement rationale.
        weather_strategy: Cross-block weather adaptation rationale.
        tips: Practical grounded tips.
        limitations: Global caveats that affect the plan.
        source_ids: Public source identifiers used by the whole draft.
    """

    title: str
    direct_answer: str
    constraints_used: List[str] = field(default_factory=list)
    blocks: List[PlanBlock] = field(default_factory=list)
    movement_logic: List[str] = field(default_factory=list)
    weather_strategy: List[str] = field(default_factory=list)
    tips: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    source_ids: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PlanDraft":
        """Create a PlanDraft from a permissive JSON dictionary.

        Args:
            payload: Parsed model JSON response.

        Returns:
            Sanitized ``PlanDraft`` with unsupported block payloads ignored.
        """
        blocks: List[PlanBlock] = []
        for raw_block in payload.get("blocks") or []:
            if not isinstance(raw_block, dict):
                continue
            blocks.append(
                PlanBlock(
                    title=str(raw_block.get("title") or "").strip(),
                    kind=str(raw_block.get("kind") or "activity").strip() or "activity",
                    purpose=str(raw_block.get("purpose") or "").strip(),
                    details=_string_list(raw_block.get("details")),
                    movement=_string_list(raw_block.get("movement")),
                    weather=_string_list(raw_block.get("weather")),
                    limitations=_string_list(raw_block.get("limitations")),
                    source_ids=_string_list(raw_block.get("source_ids")),
                )
            )
        return cls(
            title=str(payload.get("title") or "").strip(),
            direct_answer=str(payload.get("direct_answer") or "").strip(),
            constraints_used=_string_list(payload.get("constraints_used")),
            blocks=blocks,
            movement_logic=_string_list(payload.get("movement_logic")),
            weather_strategy=_string_list(payload.get("weather_strategy")),
            tips=_string_list(payload.get("tips")),
            limitations=_string_list(payload.get("limitations")),
            source_ids=_string_list(payload.get("source_ids")),
        )


def _string_list(value: Any) -> List[str]:
    """Return a clean list of non-placeholder strings.

    Args:
        value: Scalar or iterable value from a permissive JSON payload.

    Returns:
        Ordered list of non-empty strings after placeholder filtering.
    """
    if value is None:
        return []
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, (list, tuple, set)):
        candidates = list(value)
    else:
        candidates = [value]
    cleaned: List[str] = []
    forbidden = {"", "n/a", "na", "none", "null", "unknown", "not available", "not provided", "+ info"}
    for item in candidates:
        text = str(item or "").strip()
        if not text or text.lower().strip(" .:-_") in forbidden:
            continue
        cleaned.append(text)
    return cleaned
