# ==========================================================================
# Master Thesis - Planning Request Brief
#   - André Filipe Gomes Silvestre, 20240502
#
#   Structured reading of an itinerary request: the start point, the areas
#   where the visits happen, the time window, the requested components with
#   their counts, the transport mode, and the user's constraints. The
#   Supervisor fills it once per planning request, and the brief then steers
#   the Researcher's evidence searches, the Transport worker's route request,
#   and the Planner's synthesis, so every stage reads the request the same way.
# ==========================================================================

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agent.planning.evidence import normalize_text


BRIEF_CONTEXT_PREFIX = "PLAN_BRIEF_JSON:"

COMPONENT_KINDS = (
    "museum",
    "monument",
    "attraction",
    "indoor",
    "viewpoint",
    "garden",
    "beach",
    "cafe",
    "pastry",
    "restaurant",
    "lunch",
    "dinner",
    "market",
    "shopping",
    "event",
    "nightlife",
    "walk",
)
TRANSPORT_MODES = ("metro", "tram", "bus", "train", "walking", "public_transport")
_CITY_WIDE_AREAS = {"lisboa", "lisbon", "lisbon city", "cidade de lisboa", "centro", "center", "centre", "city centre", "city center"}

# Search settings per component: English query, Portuguese query, VisitLisboa category.
_COMPONENT_SEARCH: Dict[str, tuple[str, str, Optional[str]]] = {
    "museum": ("museums", "museus", "Museums"),
    "monument": ("monuments and historic sights", "monumentos e locais históricos", "Museums & Monuments"),
    "attraction": ("top attractions and places to visit", "principais atrações e locais a visitar", None),
    "indoor": ("indoor attractions and museums", "atrações interiores e museus", "Museums & Monuments"),
    "viewpoint": ("viewpoints (miradouros)", "miradouros", "View Points"),
    "garden": ("gardens and parks", "jardins e parques", "Parks & Gardens"),
    "beach": ("beaches", "praias", "Beaches"),
    "cafe": ("cafés and coffee shops", "cafés e pastelarias", "Restaurants"),
    "pastry": ("pastry shops (pastelarias)", "pastelarias", "Restaurants"),
    "restaurant": ("restaurants", "restaurantes", "Restaurants"),
    "lunch": ("restaurants for lunch", "restaurantes para almoço", "Restaurants"),
    "dinner": ("restaurants for dinner", "restaurantes para jantar", "Restaurants"),
    "market": ("markets", "mercados", "Shopping"),
    "shopping": ("shopping streets and shops", "zonas de compras e lojas", "Shopping"),
    "nightlife": ("bars, live music and fado", "bares, música ao vivo e fado", None),
    "walk": ("walking routes and sights", "percursos a pé e locais", None),
}


@dataclass
class PlanComponent:
    """One requested part of an itinerary.

    Attributes:
        kind: Component type from ``COMPONENT_KINDS``.
        count: Number of stops of this type the user asked for; ``0`` when the
            user named the type without a number.
        detail: Qualifier worth keeping, such as "for children" or
            "traditional Portuguese".
    """

    kind: str
    count: int = 0
    detail: str = ""


@dataclass
class PlanBrief:
    """Structured reading of one itinerary request.

    Attributes:
        origin: Where the user starts (station, landmark, or base area).
        areas: Areas where the visits should happen; empty for city-wide plans.
        named_places: Specific venues or landmarks the user wants to visit.
        day: Relative or explicit day ("today", "tomorrow", "this weekend").
        time_window: Part of the day or span ("morning", "afternoon",
            "evening", "sunset", "full day", "4 hours", "2 days").
        start_time: Explicit start time as HH:MM, when given.
        end_time: Explicit end time as HH:MM, when given.
        days: Number of days the plan covers.
        components: Requested stop types with counts.
        transport_mode: Requested mode from ``TRANSPORT_MODES``, or empty.
        audience: Who the plan is for ("children", "wheelchair user").
        preferences: Other constraints worth honouring ("indoor", "budget").
        exclusions: Things or areas the user wants to avoid.
        weather_conditional: Whether part of the plan depends on the weather.
        end_point: Where the plan must end, when requested.
        return_to_origin: Whether the user wants the return leg.
    """

    origin: str = ""
    areas: List[str] = field(default_factory=list)
    named_places: List[str] = field(default_factory=list)
    day: str = ""
    time_window: str = ""
    start_time: str = ""
    end_time: str = ""
    days: int = 1
    components: List[PlanComponent] = field(default_factory=list)
    transport_mode: str = ""
    audience: List[str] = field(default_factory=list)
    preferences: List[str] = field(default_factory=list)
    exclusions: List[str] = field(default_factory=list)
    weather_conditional: bool = False
    end_point: str = ""
    return_to_origin: bool = False

    @property
    def activity_areas(self) -> List[str]:
        """Return named areas, dropping city-wide placeholders such as "Lisbon"."""
        return [area for area in self.areas if normalize_text(area) not in _CITY_WIDE_AREAS]

    def to_context_line(self) -> str:
        """Serialize the brief as one machine-readable context line for workers."""
        return BRIEF_CONTEXT_PREFIX + " " + json.dumps(asdict(self), ensure_ascii=False)

    def to_prompt_text(self) -> str:
        """Render the brief as compact bullet text for LLM prompts."""
        lines: List[str] = []
        if self.origin:
            lines.append(f"- Start point: {self.origin}")
        lines.append(f"- Visit areas: {', '.join(self.activity_areas) if self.activity_areas else 'city-wide (no specific area)'}")
        if self.named_places:
            lines.append("- Places the user named: " + ", ".join(self.named_places))
        when = " ".join(part for part in (self.day, self.time_window) if part)
        if when:
            lines.append(f"- When: {when}")
        if self.start_time or self.end_time:
            lines.append(f"- Time limits: {self.start_time or '?'} to {self.end_time or '?'}")
        if self.days > 1:
            lines.append(f"- Days: {self.days}")
        if self.components:
            parts = []
            for component in self.components:
                label = f"{component.count} x {component.kind}" if component.count else component.kind
                if component.detail:
                    label += f" ({component.detail})"
                parts.append(label)
            lines.append("- Requested components: " + "; ".join(parts))
        if self.transport_mode:
            lines.append(f"- Transport mode: {self.transport_mode.replace('_', ' ')}")
        if self.audience:
            lines.append("- Audience: " + ", ".join(self.audience))
        if self.preferences:
            lines.append("- Preferences: " + ", ".join(self.preferences))
        if self.exclusions:
            lines.append("- Avoid: " + ", ".join(self.exclusions))
        if self.weather_conditional:
            lines.append("- Part of the plan depends on the weather")
        if self.end_point:
            lines.append(f"- End point: {self.end_point}")
        if self.return_to_origin:
            lines.append("- Include the return to the start point")
        return "\n".join(lines)


def build_plan_brief_messages(user_message: str, language: str, conversation_context: str = "") -> list:
    """Build the messages that ask an LLM to read an itinerary request.

    Args:
        user_message: The user's planning request.
        language: Response language of the turn (``pt`` or ``en``).
        conversation_context: Earlier plan context for revision turns.

    Returns:
        LangChain messages whose answer is one JSON object.
    """
    system = f"""
You read Lisbon itinerary requests and return ONE JSON object. No prose, no Markdown.

Fields:
- "origin": where the user starts (the place after "starting from", "from", "partindo de", "saindo de", "a partir de", "a começar no", "parto do", "estou em", or the hotel/base area). Clean place name only, without the transport mode (write "Cais do Sodré", never "Cais do Sodré by train"). Empty string if not given.
- "areas": list of the neighbourhoods, districts, or municipalities where the visits happen, in the order given (e.g. ["Chiado", "Bairro Alto"], ["Belém"], ["Cascais"]). Never put the start point here unless the visits also happen there. Use [] when the plan is city-wide ("in Lisbon", "central Lisbon").
- "named_places": specific venues, landmarks, or complexes the user wants to visit (e.g. ["Jerónimos Monastery"], ["Oceanário"], ["Boca do Inferno"]); [] if none. A venue that also sets the place of the plan ("an afternoon at LX Factory", "a manhã no Oceanário") goes here, and its neighbourhood goes in "areas" when you know it (LX Factory -> Alcântara). Do not repeat a named place in "areas".
- "day": "today", "tomorrow", "this weekend", a weekday, or a date, as stated; "" if not stated.
- "time_window": "morning", "afternoon", "evening", "sunset", "night", "full day", "half day", "N hours", or "N days"; "" if not stated. "meio dia" / "roteiro de meio dia" / "half a day" is "half day" (about four hours), not noon; "ao meio-dia" / "at noon" is start_time "12:00". When the user gives a duration and a part of the day, keep both ("duas horas de manhã" -> "2 hours morning", "3 hours this afternoon" -> "3 hours afternoon").
- "start_time", "end_time": "HH:MM" only when the user gives a clock time; else "".
- "days": number of days (1 unless the user asks for more).
- "components": list of {{"kind", "count", "detail"}} for every requested stop type. kind is one of {", ".join(COMPONENT_KINDS)}. count is the number the user asked for (e.g. "dois miradouros" -> 2, "a museum" -> 1, "two visits" -> attraction 2); use 0 when the user names the type without a number. detail keeps short qualifiers about the stop itself ("traditional", "with sunset view", "by the sea", "only if the weather allows"), not the audience. Indoor visits -> kind "indoor". A named place is also a component of its type (Jerónimos Monastery -> monument). If the user asks to "plan the visits" in an area without naming types, use [{{"kind": "attraction", "count": 0, "detail": ""}}]. How the user moves ("a pé", "on foot", "walking") is the transport mode, never a component. For a specific kind of shop or venue, keep it in detail ("livraria histórica" -> kind "shopping", detail "livraria histórica").
- "transport_mode": one of {", ".join(TRANSPORT_MODES)}, or "" when not stated. "metro"/"de metro" -> metro, "train"/"comboio" -> train, "tram"/"elétrico" -> tram, "bus"/"autocarro" -> bus, "on foot"/"a pé" -> walking, "public transport"/"transportes públicos" -> public_transport.
- "audience": e.g. ["children"], ["wheelchair user"], ["first-time visitor"]; [] if none.
- "preferences": other constraints, e.g. ["indoor visits", "budget", "rainy day"]; [] if none.
- "exclusions": things or areas to avoid; [] if none.
- "weather_conditional": true when part of the plan depends on the weather ("only if the weather allows", "indoor options if it rains", "considering the weather").
- "end_point": where the plan must end, if stated; else "".
- "return_to_origin": true only if the user asks to come back to the start.

When the earlier plan context gives the earlier plan's day and time window and the new request only changes stops ("swap the second visit", "add a café"), keep that day (as an ISO date) and that time window unless the new request states others.
Keep place names as the user wrote them (accents included). The answer language does not matter for this JSON. The user's language is {"Portuguese" if language == "pt" else "English"}.
""".strip()
    parts = []
    if conversation_context.strip():
        parts.append("Earlier plan context (use only to resolve references in the new request):\n" + conversation_context.strip()[:1200])
    parts.append(f"Request: {user_message}")
    return [SystemMessage(content=system), HumanMessage(content="\n\n".join(parts))]


def _clean_text(value: Any, limit: int = 120) -> str:
    """Return a trimmed single-line string."""
    text = re.sub(r"\s+", " ", str(value or "")).strip(" .,;")
    if text.lower() in {"none", "null", "n/a", "unknown"}:
        return ""
    return text[:limit]


def _clean_list(value: Any, limit: int = 6) -> List[str]:
    """Return a list of clean non-empty strings."""
    items = value if isinstance(value, list) else ([value] if value else [])
    output: List[str] = []
    for item in items:
        text = _clean_text(item)
        if text and text not in output:
            output.append(text)
    return output[:limit]


def _clean_clock(value: Any) -> str:
    """Return HH:MM when the value is a valid clock time, else an empty string."""
    match = re.fullmatch(r"\s*(\d{1,2})[:h](\d{2})\s*", str(value or ""))
    if not match:
        return ""
    hour, minute = int(match.group(1)), int(match.group(2))
    if hour > 23 or minute > 59:
        return ""
    return f"{hour:02d}:{minute:02d}"


def plan_brief_from_payload(payload: Dict[str, Any]) -> PlanBrief:
    """Build a sanitized brief from a parsed JSON payload.

    Args:
        payload: Dictionary produced by the brief LLM or by ``asdict``.

    Returns:
        PlanBrief with unknown values dropped.
    """
    components: List[PlanComponent] = []
    for raw in payload.get("components") or []:
        if not isinstance(raw, dict):
            continue
        kind = normalize_text(str(raw.get("kind") or "")).replace(" ", "_")
        if kind in {"coffee", "coffee_shop"}:
            kind = "cafe"
        if kind in {"park", "parks", "gardens"}:
            kind = "garden"
        if kind not in COMPONENT_KINDS:
            continue
        try:
            count = max(0, min(int(raw.get("count") or 0), 8))
        except (TypeError, ValueError):
            count = 0
        components.append(PlanComponent(kind=kind, count=count, detail=_clean_text(raw.get("detail"), 80)))

    mode = normalize_text(str(payload.get("transport_mode") or "")).replace(" ", "_")
    if mode not in TRANSPORT_MODES:
        mode = ""
    try:
        days = max(1, min(int(payload.get("days") or 1), 7))
    except (TypeError, ValueError):
        days = 1

    named_places = _clean_list(payload.get("named_places"), limit=6)
    named_keys = {normalize_text(place) for place in named_places}
    areas = [
        area for area in _clean_list(payload.get("areas"), limit=4)
        if normalize_text(area) not in named_keys
    ]
    return PlanBrief(
        origin=_clean_text(payload.get("origin")),
        areas=areas,
        named_places=named_places,
        day=_clean_text(payload.get("day"), 40),
        time_window=_clean_text(payload.get("time_window"), 40),
        start_time=_clean_clock(payload.get("start_time")),
        end_time=_clean_clock(payload.get("end_time")),
        days=days,
        components=components,
        transport_mode=mode,
        audience=_clean_list(payload.get("audience")),
        preferences=_clean_list(payload.get("preferences")),
        exclusions=_clean_list(payload.get("exclusions")),
        weather_conditional=bool(payload.get("weather_conditional")),
        end_point=_clean_text(payload.get("end_point")),
        return_to_origin=bool(payload.get("return_to_origin")),
    )


def roll_past_window_to_tomorrow(brief: PlanBrief, now: datetime) -> PlanBrief:
    """Move a plan with no day to tomorrow when its part of the day is over.

    "Faz-me um roteiro de meio dia em Belém com dois museus" asked at 20:00,
    or "a sunset walk in Graça" after sunset, is for tomorrow: planned for
    now it would find every museum closed and the sun gone. A day the user
    gave ("hoje", "this morning"), a clock time, or a bare duration ("3
    hours") is left as it is.

    Args:
        brief: Brief read from the request.
        now: Current Lisbon time.

    Returns:
        The same brief, with ``day`` set to "tomorrow" when the window is over.
    """
    if brief.day or brief.start_time:
        return brief
    window = normalize_text(brief.time_window)
    minutes = now.hour * 60 + now.minute
    over = False
    if re.search(r"\b(?:sunset|por do sol)\b", window):
        from tools.utils import sunrise_sunset

        times = sunrise_sunset(now.date())
        if times:
            hour, minute = (int(part) for part in times[1].split(":"))
            over = minutes >= hour * 60 + minute - 10
    elif brief.days > 1:
        over = minutes >= 14 * 60
    elif not window or re.fullmatch(r"\d+\s*(?:h|hours?|horas?)", window):
        # No part of the day ("suggest a walk", "a two-hour walk with a
        # market"): late in the evening, daytime stops are closed, so the plan
        # is for tomorrow unless it asks for something of the evening.
        kinds = {component.kind for component in brief.components}
        over = (
            minutes >= 19 * 60 + 30
            and bool(kinds & _DAYTIME_KINDS)
            and not kinds & {"dinner", "nightlife", "event"}
        )
    else:
        # Latest start (minutes after midnight) at which each part of the day
        # can still be planned for today.
        last_starts = (
            (_MORNING_WINDOW_RE, 12 * 60 + 30),
            (_HALF_DAY_WINDOW_RE, 17 * 60),
            (_FULL_DAY_WINDOW_RE, 14 * 60),
            (_AFTERNOON_WINDOW_RE, 18 * 60 + 30),
            (_EVENING_WINDOW_RE, 22 * 60 + 30),
        )
        # "Morning and afternoon" is over only when its latest part is.
        limits = [limit for pattern, limit in last_starts if pattern.search(window)]
        over = bool(limits) and minutes >= max(limits)
    if over:
        brief.day = "tomorrow"
    return brief


def parse_plan_brief(content: str) -> Optional[PlanBrief]:
    """Parse the LLM answer into a brief.

    Args:
        content: Raw model answer, optionally fenced.

    Returns:
        PlanBrief, or ``None`` when no JSON object can be read.
    """
    text = str(content or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        text = text[start : end + 1]
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return plan_brief_from_payload(payload)


def plan_brief_from_context(context: str) -> Optional[PlanBrief]:
    """Read a brief that the orchestrator placed in a worker context string."""
    for line in str(context or "").splitlines():
        stripped = line.strip()
        if stripped.startswith(BRIEF_CONTEXT_PREFIX):
            return parse_plan_brief(stripped[len(BRIEF_CONTEXT_PREFIX):])
    return None


def strip_plan_brief_from_context(context: str) -> str:
    """Remove the machine-readable brief line before a context reaches an LLM prompt."""
    return "\n".join(
        line for line in str(context or "").splitlines()
        if not line.strip().startswith(BRIEF_CONTEXT_PREFIX)
    )


FOOD_COMPONENT_KINDS = {"cafe", "pastry", "restaurant", "lunch", "dinner"}
# Requests for places off the usual tourist track.
_OFFBEAT_RE = re.compile(
    r"\b(?:fora d[oa]s? (?:habitua\w*|circuitos?|roteiros?)|sitios habituais|diferente|alternativ\w*|"
    r"menos (?:turistic\w*|conhecid\w*)|pouco (?:turistic\w*|conhecid\w*)|escondid\w*|secret\w*|"
    r"off the beaten|lesser[- ]known|less touristy|hidden|unusual|different|non[- ]touristy|"
    r"not (?:the )?(?:usual|touristy)|usual tourist\w*)\b"
)
# Kinds whose search is best driven by the user's own words when a detail is
# given ("livraria histórica" beats "shopping streets and shops").
_DETAIL_DRIVEN_KINDS = {"shopping", "attraction", "market", "nightlife", "indoor", "walk"}
# Parts of the day a time window names, used to size the plan (whole words:
# "amanhã" is not "manhã", and "fim de tarde" is an evening).
_HALF_DAY_WINDOW_RE = re.compile(r"\b(?:half day|meio dia)\b")
_FULL_DAY_WINDOW_RE = re.compile(r"\b(?:full day|whole day|dia inteiro|dia todo|todo o dia|days?|dias?)\b")
_EVENING_WINDOW_RE = re.compile(r"\b(?:evening|night|noite|sunset|por do sol|fim de tarde|fim da tarde)\b")
_DAYTIME_WINDOW_RE = re.compile(r"\b(?:morning|manha|afternoon|(?<!fim de )(?<!fim da )tarde)\b")
_MORNING_WINDOW_RE = re.compile(r"\b(?:morning|manha)\b")
# Stop types that are mostly closed late in the evening.
_DAYTIME_KINDS = {"museum", "monument", "attraction", "indoor", "garden", "beach", "cafe", "pastry", "market", "shopping", "lunch"}
_AFTERNOON_WINDOW_RE = re.compile(r"\b(?:afternoon|(?<!fim de )(?<!fim da )tarde)\b")
_CHILD_AUDIENCE_RE = re.compile(r"child|crian|kid|famil|filh", re.IGNORECASE)
# Weather conditions in a detail ("only if the weather allows", "se não chover");
# whole words, so "contemporary", "train", and "pôr do sol" are not weather.
_WEATHER_DETAIL_RE = re.compile(
    r"\b(?:weather|tempo|chuv\w*|chov\w*|rain\w*|sunny|sunshine|if (?:it(?:'s| is) )?sunny|com sol|se (?:estiver|houver|fizer) sol)\b",
    re.IGNORECASE,
)
_PROXIMITY_DETAIL_RE = re.compile(r"^(?:near(?:by)?|close|perto|pr[oó]ximo)$", re.IGNORECASE)
_PRACTICAL_PREFERENCE_RE = re.compile(
    r"\b(?:rain\w*|chuv\w*|indoor|interior|budget|cheap|barat\w*|econom\w*|accessib\w*|acessib\w*|wheelchair|"
    r"walk\w*|a pe|relax\w*|calm\w*|slow|lento|tourist\w*|turist\w*|crowd\w*|multid\w*|quiet|sosseg\w*|"
    r"kids?|child\w*|crianc\w*|famil\w*|short|curt\w*|free|gratis|gratuit\w*|early|cedo|late|tarde|noite)\b"
)
# A start point that names no place ("my hotel", "home") cannot anchor a search.
_GENERIC_ORIGIN_RE = re.compile(
    r"\b(?:my|meu|minha|hotel|hostel|home|casa|apartment|apartamento|airbnb|here|aqui)\b",
    re.IGNORECASE,
)


def is_full_day_window(time_window: str) -> bool:
    """Return whether a time window spans a whole day ("full day", "2 days"), not a half day."""
    window = normalize_text(time_window)
    return bool(_FULL_DAY_WINDOW_RE.search(window)) and not _HALF_DAY_WINDOW_RE.search(window)


def window_stop_target(brief: PlanBrief) -> int:
    """Return how many non-food stops the brief's time window usually holds.

    Used to add nearby options when the user named fewer stops than the
    window can take, and to tell short plans from full days.
    """
    window = normalize_text(brief.time_window)
    hours = re.search(r"(\d+)\s*(?:h|hours?|horas?)\b", window)
    if hours:
        return max(1, min(int(hours.group(1)) // 2 + 1, 4))
    if _HALF_DAY_WINDOW_RE.search(window):
        return 2
    if _FULL_DAY_WINDOW_RE.search(window):
        return 4
    daytime = bool(_DAYTIME_WINDOW_RE.search(window))
    evening = bool(_EVENING_WINDOW_RE.search(window))
    if daytime and evening:
        return 3
    return 2 if daytime else 1 if evening else 0


def extra_stops_allowed(brief: PlanBrief) -> bool:
    """Return whether nearby extra stops may fill the time window.

    When the user asked for fewer non-food stops than the window usually
    holds ("a day in Cascais with a beach stop", "a morning with children at
    the Oceanário"). A count still limits its own type: "two museums" never
    becomes three museums (the planner prompt says so).
    """
    requested = max(
        len(brief.named_places),
        sum(1 for component in brief.components if component.kind not in FOOD_COMPONENT_KINDS | {"walk"}),
    )
    return requested < window_stop_target(brief) * max(1, brief.days)


def _plan_components_for_search(brief: PlanBrief) -> List[PlanComponent]:
    """Return the components to search, adding the implicit ones a plan needs."""
    components = [component for component in brief.components if component.kind != "event"]
    # "Something different, away from the usual tourist spots" names no stop
    # type; as a search it matches tour companies selling "alternative
    # tourism". The catalogue has no "lesser-known" flag, so the quieter
    # stop types are searched and the planner picks the less obvious ones.
    offbeat = normalize_text(" ".join([*brief.preferences, *brief.exclusions]))
    expanded: List[PlanComponent] = []
    for component in components:
        if component.kind == "attraction" and _OFFBEAT_RE.search(f"{normalize_text(component.detail)} {offbeat}"):
            expanded.extend(PlanComponent(kind=kind) for kind in ("garden", "viewpoint", "museum"))
        else:
            expanded.append(component)
    components = expanded
    # "walk" describes how the user moves, not a stop type, once other stops exist.
    if any(component.kind != "walk" for component in components):
        components = [component for component in components if component.kind != "walk"]
    if not components and not any(component.kind == "event" for component in brief.components):
        components = [PlanComponent(kind="attraction")]
    kinds = {component.kind for component in components}
    if extra_stops_allowed(brief) and "attraction" not in kinds:
        components.append(PlanComponent(kind="attraction"))
        kinds.add("attraction")
    window = normalize_text(brief.time_window)
    if (brief.days > 1 or is_full_day_window(brief.time_window)) and not kinds & {"restaurant", "lunch", "dinner"}:
        components.append(PlanComponent(kind="lunch"))
    # An evening plan built around a visit or an event usually includes dinner.
    if re.search(r"evening|night|sunset|noite|fim de tarde|por do sol", window) and not kinds & FOOD_COMPONENT_KINDS:
        components.append(PlanComponent(kind="dinner"))
    wants_indoor_backup = brief.weather_conditional or any(
        re.search(r"rain|chuva|indoor|interior", item, re.IGNORECASE) for item in brief.preferences
    )
    if wants_indoor_backup and not kinds & {"indoor", "museum"}:
        components.append(PlanComponent(kind="indoor"))
    return components


def plan_brief_search_requests(brief: PlanBrief, language: str) -> List[Dict[str, Any]]:
    """Turn the requested components into area-anchored place searches.

    Named places get an exact lookup. Every other component is searched near
    each visit area. With no area named, a morning or an afternoon is searched
    near the start point, and a full day or an evening across Lisbon. A start
    point never replaces a named area: "an evening in Chiado starting from
    Oriente" is searched in Chiado.

    Args:
        brief: Structured request.
        language: Output language for the tool (``pt`` or ``en``).

    Returns:
        Keyword arguments for ``search_places_attractions``, one per search.
    """
    is_pt = language == "pt"
    requests: List[Dict[str, Any]] = [
        {"query": place, "category": None, "max_results": 1, "specific_lookup": True, "language": language}
        for place in brief.named_places[:4]
    ]
    anchors = brief.activity_areas[:2] or brief.named_places[:1]
    if (
        not anchors
        and brief.origin
        and not _GENERIC_ORIGIN_RE.search(brief.origin)
        and brief.days <= 1
        and (window_stop_target(brief) == 0 or 1 < window_stop_target(brief) < 4)
    ):
        # A morning, an afternoon, or a walk with no time given ("I'm at
        # Entrecampos, suggest a walk") with no area named stays near where the
        # user starts. A full day may cross the city, and evening plans
        # (dinner, music, a sunset) belong to their own areas.
        anchors = [brief.origin]
    anchors = anchors or [""]
    audience = ""
    if any(_CHILD_AUDIENCE_RE.search(item) for item in brief.audience):
        audience = " para crianças" if is_pt else " for children"
    seen: set[tuple[str, str]] = set()
    # An interest among the preferences ("architecture lovers") steers the
    # generic attraction search; practical preferences (rain, budget, pace,
    # accessibility, audience) do not describe what to see.
    theme = next(
        (
            re.sub(r"(?i)\b(?:lovers?|fans?|amantes de|apaixonad[oa]s (?:por|de)|f[aã]s de)\b", " ", preference).strip()
            for preference in brief.preferences
            if preference and not _PRACTICAL_PREFERENCE_RE.search(normalize_text(preference))
        ),
        "",
    )
    for component in _plan_components_for_search(brief):
        query_en, query_pt, category = _COMPONENT_SEARCH.get(component.kind, _COMPONENT_SEARCH["attraction"])
        base = query_pt if is_pt else query_en
        detail = component.detail or (theme if component.kind == "attraction" else "")
        if (
            not detail
            or _WEATHER_DETAIL_RE.search(detail)
            or _PROXIMITY_DETAIL_RE.match(detail)
            or _CHILD_AUDIENCE_RE.search(detail)
            or normalize_text(detail) in {normalize_text(place) for place in brief.named_places}
        ):
            detail = ""
        if detail and component.kind in _DETAIL_DRIVEN_KINDS and len(detail.split()) >= 2:
            base, detail = detail, ""
        component_audience = audience if component.kind not in FOOD_COMPONENT_KINDS else ""
        for anchor in anchors:
            key = (component.kind, normalize_text(anchor))
            if key in seen:
                continue
            seen.add(key)
            proximity = (f" perto de {anchor}" if is_pt else f" near {anchor}") if anchor else (" em Lisboa" if is_pt else " in Lisbon")
            max_results = max(4, min(component.count + 3, 6))
            if component.kind not in FOOD_COMPONENT_KINDS and (brief.days > 1 or window_stop_target(brief) >= 4):
                max_results = 8
            requests.append(
                {
                    "query": f"{base}{component_audience}{(' ' + detail) if detail else ''}{proximity}".strip(),
                    "category": category,
                    "max_results": max_results,
                    "language": language,
                    "_kind": component.kind,
                    "_detail": component.detail,
                    "_anchor": anchor,
                }
            )
    for component in brief.components:
        if component.kind != "event" or not component.detail or _WEATHER_DETAIL_RE.search(component.detail):
            continue
        anchor = anchors[0]
        proximity = (f" perto de {anchor}" if is_pt else f" near {anchor}") if anchor else (" em Lisboa" if is_pt else " in Lisbon")
        requests.append(
            {
                "query": f"{component.detail}{proximity}",
                "category": None,
                "max_results": 4,
                "language": language,
                "_kind": "nightlife",
            }
        )
    return requests[:9]


def plan_brief_event_request(brief: PlanBrief, language: str) -> Optional[Dict[str, Any]]:
    """Return ``search_cultural_events`` arguments when the brief asks for an event."""
    if not any(component.kind == "event" for component in brief.components):
        return None
    detail = next((component.detail for component in brief.components if component.kind == "event" and component.detail), "")
    area = brief.activity_areas[0] if brief.activity_areas else ""
    if language == "pt":
        query = f"eventos {detail} em {area or 'Lisboa'}".replace("  ", " ")
    else:
        query = f"{detail} events in {area or 'Lisbon'}".strip()
    args: Dict[str, Any] = {"query": query, "max_results": 5, "language": language}
    day = normalize_text(brief.day)
    if re.search(r"weekend|fim de semana|sabado|domingo|saturday|sunday", day):
        args["date_filter"] = "this weekend"
    elif re.search(r"tomorrow|amanha", day):
        args["date_filter"] = "tomorrow"
    elif re.search(r"today|hoje|tonight|esta noite", day):
        args["date_filter"] = "today"
    elif re.search(r"week|semana", day):
        args["date_filter"] = "this week"
    return args


def plan_brief_transport_request(brief: PlanBrief, language: str) -> str:
    """Return a focused route question for the Transport worker, or an empty string.

    Args:
        brief: Structured request.
        language: Output language.

    Returns:
        A point-to-point route question from the start point to the first
        visit area, in the requested mode.
    """
    if not brief.origin or brief.transport_mode == "walking":
        return ""
    destination = brief.activity_areas[0] if brief.activity_areas else brief.end_point
    if not destination or normalize_text(destination) == normalize_text(brief.origin):
        return ""
    mode_pt = {
        "metro": " de metro",
        "tram": " de elétrico",
        "bus": " de autocarro",
        "train": " de comboio",
        "public_transport": " de transportes públicos",
    }
    mode_en = {
        "metro": " by metro",
        "tram": " by tram",
        "bus": " by bus",
        "train": " by train",
        "public_transport": " by public transport",
    }
    if language == "pt":
        return f"Como vou de {brief.origin} para {destination}{mode_pt.get(brief.transport_mode, ' de transportes públicos')}?"
    return f"How do I get from {brief.origin} to {destination}{mode_en.get(brief.transport_mode, ' by public transport')}?"


WEEKDAY_NAMES = (
    ("monday", "segunda"),
    ("tuesday", "terca"),
    ("wednesday", "quarta"),
    ("thursday", "quinta"),
    ("friday", "sexta"),
    ("saturday", "sabado"),
    ("sunday", "domingo"),
)


def plan_date(brief: Optional[PlanBrief], today: date) -> date:
    """Return the calendar date of the plan's first day.

    Args:
        brief: Planning brief, or ``None``.
        today: Current Lisbon date.

    Returns:
        ``today`` unless the brief names another day (tomorrow, a weekday,
        the weekend, or a DD/MM or ISO date).
    """
    day = normalize_text(brief.day if brief else "")
    if not day or re.search(r"\b(?:today|hoje|tonight|esta noite|now|agora)\b", day):
        return today
    if re.search(r"day after tomorrow|depois de amanha", day):
        return today + timedelta(days=2)
    if re.search(r"\b(?:tomorrow|amanha)\b", day):
        return today + timedelta(days=1)
    iso = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", day)
    short = re.search(r"\b(\d{1,2})[/-](\d{1,2})\b", day)
    try:
        if iso:
            return date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        if short:
            candidate = date(today.year, int(short.group(2)), int(short.group(1)))
            return candidate if candidate >= today else date(today.year + 1, candidate.month, candidate.day)
    except ValueError:
        return today
    if re.search(r"weekend|fim de semana", day):
        return today if today.weekday() >= 5 else today + timedelta(days=5 - today.weekday())
    for index, names in enumerate(WEEKDAY_NAMES):
        if any(re.search(rf"\b{name}", day) for name in names):
            return today + timedelta(days=(index - today.weekday()) % 7)
    return today
