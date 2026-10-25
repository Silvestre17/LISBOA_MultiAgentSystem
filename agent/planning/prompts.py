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

from agent.planning.brief import PlanBrief, extra_stops_allowed, plan_date
from agent.planning.evidence import EvidenceBundle
from tools.utils import lisbon_now, sunrise_sunset


_BRIEF_PLAN_SCHEMA = """
{
  "title": "short title written entirely in the answer language, naming the area and the part of the day or the number of days (Portuguese: \"Belém à tarde\", \"Lisboa em 2 dias\"; English: \"Belém in the afternoon\", \"Lisbon in 2 days\"), without dashes",
  "direct_answer": "one sentence naming the stops, with where and when (and the transport mode the user asked for)",
  "blocks": [
    {
      "day": 1,
      "title": "HH:MM · exact title of the evidence card",
      "stay_minutes": 60,
      "kind": "museum|culture|viewpoint|garden|beach|food|coffee|pastry|event|place",
      "purpose": "one short sentence on why this stop fits the request",
      "weather": ["only when this stop depends on the weather"],
      "limitations": ["only a real gap for this stop"],
      "source_ids": ["source ids of the card"]
    }
  ],
  "movement_logic": ["leg from the start point to the first stop, from the transport evidence"],
  "weather_strategy": ["forecast for the plan's day and period, and what it changes"],
  "tips": ["at most two practical tips grounded in the evidence"],
  "limitations": ["real gaps only"],
  "source_ids": ["source ids used"]
}
""".strip()


def _sun_line(brief: PlanBrief) -> str:
    """Return the sunrise and sunset of the plan's first day, for the prompt."""
    day = plan_date(brief, lisbon_now().date())
    times = sunrise_sunset(day)
    if not times:
        return ""
    return f"Sun on {day.strftime('%A %Y-%m-%d')} (Lisbon): sunrise {times[0]}, sunset {times[1]}."


def build_brief_plan_messages(
    *,
    user_message: str,
    language: str,
    evidence: EvidenceBundle,
    brief: PlanBrief,
    conversation_context: str = "",
) -> list:
    """Build the planner messages for a request that comes with a planning brief.

    The brief states what the user asked for (start point, areas, time window,
    components with counts, mode, constraints); the evidence cards state what
    the workers found. The LLM chooses and orders the stops and writes the
    text. Code copies each stop's details from its card, computes the legs
    between stops, aligns the times with those legs, and renders the Markdown.

    Args:
        user_message: Original user request.
        language: Output language (``pt`` or ``en``).
        evidence: Evidence cards extracted from worker outputs.
        brief: Structured reading of the request.
        conversation_context: Earlier plan context for revision turns.

    Returns:
        LangChain system and human messages.
    """
    is_pt = (language or "en").lower().startswith("pt")
    now = lisbon_now().strftime("%A %Y-%m-%d %H:%M")
    language_rule = 'European Portuguese (PT-PT), addressing the user as "tu"' if is_pt else "English"
    extras_rule = (
        "The user named fewer stops than this time window holds: add one or two more nearby stops from the cards that fit the area, the audience, and the preferences, so the window is not left empty, but never more stops of a type whose count the user gave (\"two museums\" stays two museums)."
        if extra_stops_allowed(brief)
        else "Do not add stop types the user did not ask for, beyond the meals and the night stop that rule 5 gives the time window."
    )
    system = f"""
You are LISBOA's itinerary planner. You choose and order the stops and write the text; code copies each stop's details from its card, computes the walking or transport legs between consecutive stops, aligns the start times with those legs, and renders the Markdown.

Return ONLY valid JSON, no Markdown, no code fences. Write every text value in {language_rule}.
Current date and time: {now} (Lisbon).
{_sun_line(brief)}

Rules:
1. Stops come only from the place and event evidence cards. Copy the card title exactly into the block title, after the start time: "HH:MM · <card title>". One block per stop, in visiting order. Never invent a venue, restaurant, café, event, price, schedule, or line.
2. Stay in the visit area. When the brief names areas, every stop must be in or next to them: the cards show "Distance: X km from <area>", so prefer cards within about 2 km of the area (the distance is to the area's centre, so 0.3 km is not better than 1 km) and never use a card from another district. When the brief is city-wide, keep the stops in one or two neighbouring areas so the day does not zig-zag across Lisbon. For a general visit to an area, include its best-known landmark from the cards unless the user excluded it, and order the stops so the walk goes in one direction through the area, never back and forth. When the user request fixes the first or the last stop ("começando a visita pelo Padrão dos Descobrimentos"), keep that stop there and order the others from it.
3. Cover every requested component with its count. "2 x viewpoint; 1 x cafe" means two viewpoint stops and one café stop. The "Found for" line of a card says which request it answers. A café or pastry request needs a café or pastelaria card, not a restaurant meal; a garden request needs a garden or park card; a museum request needs a museum card. When no card covers a requested component, do not substitute another type: add a limitation naming the missing component (for example "No bookshop in Chiado was found for this plan.").
4. {extras_rule} When the user travels by metro, prefer stops with a "Nearest metro" station. For a viewpoint, prefer places named "Miradouro" over rooftop bars or hotels, unless the user asked for drinks or a bar.
5. Timing: start every block title with a start time inside the requested window, and give the window its shape. A morning (about 09:30-13:00) or an afternoon (14:00-18:30) holds sights only, with no lunch or dinner unless the user asked for one. An evening (18:30-23:00) is a visit or a viewpoint, dinner, and then one night stop. A full day (09:30-23:00) is sights in the morning, lunch, sights in the afternoon, dinner, and one night stop; with an end time from the user it ends there, and for children it ends after dinner. The night stop is an event card dated on the plan's day with a time after dinner and a venue near the dinner, or else a fado house, bar, or live-music card near the dinner; leave it out when no card fits. Event cards are dated on the plan's day: when one is in or next to the visit area and its time falls inside the window (or it is an exhibition open that day), prefer it to a lesser place for one of the sights, so the plan is not only places; never use an event whose time falls outside the window. A half day is about 4 hours and "N hours" must fit in N hours. Set stay_minutes to a realistic stay (museum 60-90, monument 45-60, viewpoint 20-30, garden 30-45, café 20-30, meal 60-90, night stop 60-90) and leave time for the travel between stops. Lunch starts between 12:30 and 14:00 and dinner between 19:30 and 21:00. For a sunset stop, be at the viewpoint from about 30 minutes before the sunset time given above. Never place a stop at a time its card shows it closed (the card hours are today's). A card with no hours is not closed: choose it on its merits, never skip a landmark for that. Do not write notes about opening hours; code adds them when needed.
6. Getting there: when the brief has a start point, movement_logic[0] is the leg from the start point to the first stop, written from the transport evidence with the line, direction, transfer station, exit station, and duration it shows, e.g. "Cais do Sodré → Oriente: Metro Green Line towards Telheiras, change at Alameda to the Red Line towards Aeroporto, exit at Oriente (~31 min), then walk to the first stop". Never copy next departures, waiting times, or service-hours notices into it. If the transport evidence does not cover that leg, write one sentence saying the exact connection was not confirmed. Do not write the legs between stops, and never write tips or limitations about connections, legs, or transport between stops: code computes and checks them. Add a final return leg only if the brief asks for it.
6b. Several days: when the brief covers more than one day, set "day" (1, 2, ...) on every block, give each day its own time sequence, keep each day in one or two neighbouring areas, and give each day the full-day shape of rule 5 (sights in the morning, lunch, sights in the afternoon, dinner) unless the user gave a shorter window, using the meal cards found for that day's area; never leave a day with only one or two stops while cards remain. For a one-day plan, set "day" to 1.
7. Weather: weather_strategy gives the forecast for the plan's day and period from the weather evidence (conditions, temperatures, rain probability, active warnings) and one consequence for the plan. If the user's premise contradicts the forecast (a "rainy afternoon" with no rain expected), say so plainly and still give the indoor plan the user asked for. For a stop that depends on the weather ("beach only if the weather allows"), decide from the forecast and name the alternative. The forecast gives only the day's minimum and maximum, so never say how the temperature changes during the day (a "cooler start", "warmer later", "temperatures rise in the afternoon"). If there is no weather evidence, leave weather_strategy empty.
8. Live waiting times and live departures are valid only for leaving now; never use them in a plan for a later time or another day.
9. direct_answer is one sentence that summarizes the plan: the area, the time window, the stops by their names (for example "the Oceanário and the Pavilhão do Conhecimento", not "two visits"), and the transport mode only when the user named one. Do not name lines, stations, or stops of the journey there; the legs section shows them.
10. limitations list only real gaps for the user: a requested component that is missing, or a stop whose details must be checked. Never mention tools, agents, searches, cards, evidence, data sources, or the system, and never write in the first person.
10b. Audience: for a wheelchair user or someone with reduced mobility, prefer cards whose Features list Accessibility, keep walks short and flat (use the metro rather than steep walks), avoid viewpoints reached by stairs or steep streets, and add one limitation saying accessibility must be confirmed for stops that do not list it. For children, prefer stops the cards describe as family or interactive.
11. tips are practical advice for the traveller (what to book, bring, or time), never an explanation of why stops were or were not chosen.
12. Write complete sentences with a final full stop. Use only source_ids from the evidence.

JSON schema:
{_BRIEF_PLAN_SCHEMA}
""".strip()
    parts = ["# Planning brief", brief.to_prompt_text(), "# Evidence cards", evidence.to_prompt_text(max_cards=30)]
    if conversation_context.strip():
        parts.extend(["# Earlier plan in this conversation", conversation_context.strip()[:1200]])
    parts.append(f"# User request\n{user_message}\n\nReturn the JSON plan now.")
    return [SystemMessage(content=system), HumanMessage(content="\n\n".join(parts))]


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
- A coffee/cafe/pastry stop is a distinct request from a full meal. If the user asks to add or keep a cafe or pastelaria and cafe/pastry evidence exists, include a separate block with kind "coffee" or "pastry", even when a restaurant or meal block is already present. Do not treat an existing lunch/dinner as satisfying a requested cafe.
- When the conversation continuity says to preserve earlier grounded stops and add a new component, keep the earlier evidenced stops and also add the newly requested component as its own block; do not silently drop either.
- For selected places/events, preserve useful evidence fields as detail strings with these labels when present: Description, Address, Hours, Price, Website, When, Venue, Tickets. Omit any missing field; never write N/A, unknown, or + info.
- For time-specific plans, do not choose a place whose evidence says it is closed for that period. If all strong matches are closed, either choose a weaker open-ended stop and state the limitation, or frame the closed venue only as exterior/context, not as an enterable visit. A place whose evidence gives no hours is not closed: choose it on its merits and leave out the Hours detail.
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
