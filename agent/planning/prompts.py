# ==========================================================================
# Master Thesis - Structured Planner Prompt
#   - André Filipe Gomes Silvestre, 20240502
#
#   Builds the PlannerAgent message pair for JSON-only synthesis. The prompt
#   delegates Markdown, visual formatting, indentation, and source-footers to
#   deterministic code so the LLM focuses on selecting evidence-supported plan
#   content.
# ==========================================================================

from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from agent.planning.evidence import EvidenceBundle


def build_structured_plan_messages(
    *,
    user_message: str,
    language: str,
    evidence: EvidenceBundle,
    conversation_context: str = "",
) -> list:
    """Build messages that ask the planner LLM for JSON, not Markdown.

    Args:
        user_message: Original user request to satisfy.
        language: Detected or requested response language.
        evidence: Structured evidence bundle extracted from worker outputs.
        conversation_context: Optional recent conversation context for
            continuity-sensitive planning turns.

    Returns:
        LangChain system and human messages for structured plan synthesis.
    """
    is_pt = (language or "en").lower().startswith("pt")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    language_rule = "Portuguese from Portugal" if is_pt else "English"
    evidence_text = evidence.to_prompt_text()
    schema = """
{
  "title": "short user-facing title",
  "direct_answer": "one concise answer sentence",
  "constraints_used": ["constraint or preference actually used"],
  "blocks": [
    {
      "title": "evidence-supported place/event/service name or generic local block",
      "kind": "place|museum|culture|event|food|coffee|pastry|transport|walk|service|activity",
      "purpose": "why this block fits the request",
      "details": ["Description: evidence-supported description", "Address: evidence-supported address/map link", "Hours: evidence-supported hours", "Price: evidence-supported price", "Website: evidence-supported official/details link"],
      "movement": ["evidence-supported movement detail or scoped uncertainty"],
      "weather": ["weather adaptation when relevant"],
      "limitations": ["only relevant unconfirmed fields"],
      "source_ids": ["source ids used by this block"]
    }
  ],
  "movement_logic": ["overall movement logic"],
  "weather_strategy": ["weather-aware strategy when relevant"],
  "tips": ["short practical tips supported by evidence"],
  "limitations": ["global limitations"],
  "source_ids": ["source ids materially used"]
}
""".strip()
    # The schema is embedded directly in the prompt to keep the LLM output
    # aligned with the dataclass contract used by the renderer and quality gate.
    system = f"""
You are LISBOA's planning composer. You decide the plan content, but a deterministic renderer will handle all Markdown, emojis, indentation, headings, and source footers.

Return ONLY valid JSON. No Markdown. No prose outside JSON. No code fences.

Language: {language_rule}.
Current date/time for reasoning only: {now}.

Hard rules:
- Use only evidence cards below. Do not invent venues, restaurants, cafes, events, prices, opening hours, tickets, accessibility, live status, or exact routes.
- If an exact transport leg is not evidenced, write a scoped uncertainty in movement or limitations.
- If the user asks for public transport and transport evidence exists, include the line/operator/route detail that is evidenced.
- Do not use live departures as a schedule for a future itinerary unless the user explicitly asks for live/next departures.
- If the user asks for a plan around a named neighbourhood or starting/ending area, prefer evidence located in that area or on the direct route. Do not choose a better-known venue in another district as the cultural stop unless you clearly frame it as an on-the-way detour.
- If events or places appear in the evidence, include their useful fields in details when selected.
- If the user asks for multiple explicit themes and evidence exists for them, include at least one selected block for each requested theme. For example, a plan asking for historical sights and gastronomy must include both a cultural/historical stop and a food/restaurant/pastry block when those evidence cards are available.
- For meal stops, choose restaurants in the same neighbourhood as the adjacent itinerary block whenever evidence allows it. Do not select a restaurant in a distant district such as Parque das Nações for a Belém/centre/Saldanha route unless that district is explicitly part of the user's route.
- For selected places/events, preserve useful evidence fields as detail strings with these labels when present: Description, Address, Hours, Price, Website, When, Venue, Tickets. Omit any missing field; never write N/A, unknown, or + info.
- For time-specific plans, do not choose a place whose evidence says it is closed for that period. If all strong matches are closed or lack hours, either choose a weaker open-ended stop and state the limitation, or frame the closed venue only as exterior/context, not as an enterable visit.
- Avoid static skeletons. Every block must explain purpose plus at least one useful detail, movement, weather adaptation, or limitation.
- Keep one-day plans compact by default, but respect explicit user cardinality such as "5 sites", "3 museums and 1 restaurant", or required waypoints when evidence exists. Do not add extra filler blocks beyond what is useful.
- Use source_ids only from the evidence. If unsure, leave the source_ids list empty and state the limitation.

JSON schema:
{schema}
""".strip()
    context_parts = ["# Evidence cards", evidence_text]
    if conversation_context.strip():
        context_parts.extend(["# Conversation continuity", conversation_context.strip()[:1200]])
    human = "\n\n".join(context_parts) + f"\n\n# User request\n{user_message}\n\nReturn the JSON plan now."
    return [SystemMessage(content=system), HumanMessage(content=human)]
