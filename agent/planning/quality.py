# ==========================================================================
# Master Thesis - Planner JSON Parser and Quality Gate
#   - André Filipe Gomes Silvestre, 20240502
#
#   Parses PlannerAgent JSON output and applies deterministic quality checks
#   before Markdown rendering. The gate blocks unsafe placeholders, internal
#   wording, unsupported named venues, vague public-transport plans, and missing
#   weather adaptation in weather-sensitive requests.
# ==========================================================================

import json
import re
from datetime import timedelta
from itertools import permutations
from typing import Any, Dict, List, Sequence

from agent.planning.brief import (
    WEEKDAY_NAMES,
    extra_stops_allowed,
    full_day_with_evening,
    is_full_day_window,
    plan_date,
    window_stop_target,
)
from agent.planning.evidence import EvidenceBundle, normalize_text
from agent.planning.models import PlanDraft
from tools.place_coordinates import lookup_place_coordinates
from tools.utils import haversine_distance, lisbon_now, sunrise_sunset


PLACEHOLDER_RE = re.compile(
    r"\b(?:n\s*/\s*a|unknown|not available|not provided|TBD|\+ info|null|none)\b",
    re.IGNORECASE,
)
RAW_FIELD_RE = re.compile(r"\b(?:Location|Address|Website|Phone|Category|Description|Morada|Telefone|Categoria|Descrição)\s*:", re.IGNORECASE)
COUNT_TOKEN_RE = (
    r"(?:\d{1,2}|um|uma|one|dois|duas|two|tres|three|quatro|four|"
    r"cinco|five|seis|six|sete|seven|oito|eight)"
)
COUNT_WORDS = {
    "um": 1,
    "uma": 1,
    "one": 1,
    "dois": 2,
    "duas": 2,
    "two": 2,
    "tres": 3,
    "three": 3,
    "quatro": 4,
    "four": 4,
    "cinco": 5,
    "five": 5,
    "seis": 6,
    "six": 6,
    "sete": 7,
    "seven": 7,
    "oito": 8,
    "eight": 8,
}


def parse_plan_draft_json(content: str) -> PlanDraft | None:
    """Parse a model response into a ``PlanDraft`` when valid JSON is present.

    Args:
        content: Raw LLM response, optionally wrapped in a JSON code fence.

    Returns:
        Parsed ``PlanDraft`` when the response contains a JSON object, otherwise
        ``None``.
    """
    if not content:
        return None
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return PlanDraft.from_dict(payload)


def _requested_max_blocks(user_message: str) -> int:
    """Return the maximum block count allowed by explicit user cardinality."""
    normalized = normalize_text(user_message)
    requested = 0
    pattern = re.compile(
        rf"\b(?P<count>{COUNT_TOKEN_RE})\s+"
        r"(?P<unit>museus?|museums?|monumentos?|monuments?|atracoes|attractions?|"
        r"locais|lugares|sitios|sites|places|stops|paragens|restaurantes?|restaurants?|"
        r"food\s+stops?|meal\s+stops?|lunch\s+stops?|dinner\s+stops?|miradouros?|viewpoints?|eventos?|events?)\b",
        re.IGNORECASE,
    )
    for match in pattern.finditer(normalized):
        token = match.group("count")
        count = int(token) if token.isdigit() else COUNT_WORDS.get(token, 0)
        requested += max(0, min(count, 8))
    return max(5, min(8, requested or 5))


def validate_plan_draft(draft: PlanDraft, evidence: EvidenceBundle, user_message: str = "") -> List[str]:
    """Return blocking issues for a structured plan draft.

    Args:
        draft: Structured plan produced by the planner LLM.
        evidence: Evidence bundle available to the planner.
        user_message: Original user request, used for intent-sensitive checks.

    Returns:
        Deduplicated issue labels. An empty list means the draft can be rendered.
    """
    issues: List[str] = []
    if not draft.title:
        issues.append("missing title")
    if not draft.direct_answer:
        issues.append("missing direct answer")
    if len(draft.blocks) < 1:
        issues.append("missing plan blocks")
    if len(draft.blocks) > _requested_max_blocks(user_message):
        issues.append("too many plan blocks")

    user_norm = normalize_text(user_message)
    pt_requested = _user_prefers_portuguese(user_message)
    evidence_titles = [normalize_text(card.title) for card in evidence.cards if card.title]
    evidence_text = normalize_text(" ".join([card.title + " " + card.summary for card in evidence.cards]))
    grounded_itinerary_requested = _query_requests_grounded_itinerary(user_message)
    place_cards = [card for card in evidence.cards if getattr(card, "kind", "") in {"place", "food", "event", "service"}]
    food_cards = [card for card in evidence.cards if _evidence_card_is_food(card)]
    cafe_cards = [card for card in evidence.cards if _evidence_card_is_cafe(card)]
    cultural_cards = [card for card in place_cards if _evidence_card_is_cultural(card)]
    viewpoint_cards = [card for card in place_cards if _evidence_card_is_viewpoint(card)]
    garden_cards = [card for card in place_cards if _evidence_card_is_garden(card)]
    target_area = _extract_single_area_target(user_message)
    known_area_requested = _is_known_compact_area(target_area)
    area_has_evidence = bool(
        target_area
        and any(_evidence_card_matches_area(card, target_area) for card in place_cards)
    )
    matched_place_blocks = 0

    all_text_fields = [draft.title, draft.direct_answer, *draft.constraints_used, *draft.movement_logic, *draft.weather_strategy, *draft.tips, *draft.limitations]
    detail_text_fields: List[str] = []
    for block in draft.blocks:
        all_text_fields.extend([block.title, block.purpose, *block.movement, *block.weather, *block.limitations])
        detail_text_fields.extend(block.details)
        if PLACEHOLDER_RE.search(block.title) or RAW_FIELD_RE.search(block.title):
            issues.append(f"unsafe block title: {block.title[:40]}")
        if _looks_like_unsupported_named_venue(block.title, evidence_titles, user_norm, evidence_text):
            issues.append(f"unsupported venue title: {block.title[:60]}")
        matched_card = _matching_evidence_card(block.title, evidence)
        if matched_card and getattr(matched_card, "kind", "") in {"place", "food", "event", "service"}:
            matched_place_blocks += 1
            if (
                (area_has_evidence or known_area_requested)
                and target_area
                and not _evidence_card_matches_area(matched_card, target_area)
            ):
                issues.append(f"selected stop outside requested area: {block.title[:60]}")
        if (
            grounded_itinerary_requested
            and place_cards
            and _looks_like_generic_itinerary_block_title(block.title)
        ):
            issues.append(f"generic itinerary block instead of evidence card: {block.title[:60]}")
        if (
            matched_card
            and _query_requests_time_specific_visit(user_message)
            and _evidence_card_is_closed(matched_card)
        ):
            issues.append(f"time-specific plan selected closed venue: {block.title[:60]}")
        if _block_time_conflicts_with_supported_hours(block):
            issues.append(f"planned stop outside supported opening hours: {block.title[:60]}")
    for text in all_text_fields:
        if not text:
            continue
        if PLACEHOLDER_RE.search(text):
            issues.append("placeholder leaked into plan")
        if RAW_FIELD_RE.search(text):
            issues.append("raw place-card field leaked into plan")
        if re.search(r"https?\s*\*\*\s*:", text, flags=re.IGNORECASE):
            issues.append("broken URL leaked into plan")
        if re.search(r"\b(?:tool|agent|QA|LangSmith|repository|run id)\b", text, flags=re.IGNORECASE):
            issues.append("internal system wording leaked into plan")
        if pt_requested and _has_pt_language_drift(text):
            issues.append("Portuguese plan contains English scaffold text")

    for text in detail_text_fields:
        if not text:
            continue
        if PLACEHOLDER_RE.search(text):
            issues.append("placeholder leaked into plan")
        if RAW_FIELD_RE.search(text) and not _renderer_can_normalize_field_label(text):
            issues.append("raw place-card field leaked into plan")
        if re.search(r"https?\s*\*\*\s*:", text, flags=re.IGNORECASE):
            issues.append("broken URL leaked into plan")
        if re.search(r"\b(?:tool|agent|QA|LangSmith|repository|run id)\b", text, flags=re.IGNORECASE):
            issues.append("internal system wording leaked into plan")
        if pt_requested and _has_pt_language_drift(text):
            issues.append("Portuguese plan contains English scaffold text")

    if _query_requests_public_transport(user_message):
        movement_text = normalize_text(" ".join([*draft.movement_logic, *[" ".join(block.movement) for block in draft.blocks]]))
        if not re.search(r"\b(?:metro|carris|cp|bus|tram|train|line|linha|route|rota|stop|station|paragem|estacao|unconfirmed|nao confirmad|não confirmad)\b", movement_text):
            issues.append("public transport requested but movement logic is too vague")

    if grounded_itinerary_requested and place_cards and matched_place_blocks < min(2, len(place_cards)):
        issues.append("grounded itinerary did not select enough evidence cards")

    if _query_requests_food_stop(user_message) and food_cards and not _draft_includes_food_stop(draft, food_cards):
        issues.append("requested gastronomy but plan omitted food evidence")

    # A requested cafe/pastry stop is distinct from a meal: an existing lunch must
    # not silently satisfy it. Flag only when cafe/pastry evidence is actually
    # available, so the model is asked to use real grounded options on retry.
    if _query_requests_cafe_stop(user_message) and cafe_cards and not _draft_includes_cafe_stop(draft):
        issues.append("requested cafe/pastry stop but plan omitted cafe evidence")

    if _query_requests_cultural_stop(user_message) and cultural_cards and not _draft_includes_cultural_stop(draft, cultural_cards):
        issues.append("requested cultural stop but plan omitted cultural evidence")

    # Viewpoints and gardens are explicit requested components in many revisions
    # ("mantém o jardim e o miradouro"). Flag only when matching evidence exists,
    # so a dropped-but-available stop is restored on retry instead of replaced by
    # a generic placeholder.
    if _query_requests_viewpoint_stop(user_message) and viewpoint_cards and not _draft_includes_token_stop(draft, _VIEWPOINT_TOKEN_RE):
        issues.append("requested viewpoint but plan omitted viewpoint evidence")

    if _query_requests_garden_stop(user_message) and garden_cards and not _draft_includes_token_stop(draft, _GARDEN_TOKEN_RE):
        issues.append("requested garden but plan omitted garden evidence")

    if re.search(r"\b(?:rain|chuva|weather|tempo|umbrella|guarda chuva|indoor|interior)\b", normalize_text(user_message)):
        weather_text = normalize_text(" ".join([*draft.weather_strategy, *[" ".join(block.weather) for block in draft.blocks]]))
        if not weather_text:
            issues.append("weather-sensitive request without weather strategy")

    return list(dict.fromkeys(issues))


def _looks_like_unsupported_named_venue(title: str, evidence_titles: Sequence[str], user_norm: str, evidence_text: str) -> bool:
    """Return whether a block title appears to invent an unsupported venue.

    Args:
        title: Planner block title to inspect.
        evidence_titles: Normalized titles extracted from evidence cards.
        user_norm: Normalized original user request.
        evidence_text: Normalized aggregate evidence text.

    Returns:
        ``True`` when the title resembles a named venue absent from both user
        request and evidence.
    """
    normalized = normalize_text(title)
    if not normalized or normalized in {"block", "bloco", "plan", "plano"}:
        return False
    generic_tokens = (
        "start", "arrival", "base", "return", "transport", "walking", "coffee", "pastry", "dinner", "lunch", "cultural stop", "indoor backup", "rain backup", "inicio", "chegada", "regresso", "transporte", "jantar", "almoco", "almoço", "paragem cultural",
    )
    if any(token in normalized for token in generic_tokens):
        return False
    if normalized in user_norm or any(part for part in normalized.split() if len(part) > 3 and part in user_norm):
        return False
    if any(normalized in title_norm or title_norm in normalized for title_norm in evidence_titles if title_norm):
        return False
    place_markers = ("museum", "museu", "monastery", "mosteiro", "palace", "palacio", "palácio", "garden", "jardim", "restaurant", "restaurante", "cafe", "café", "pastelaria", "event", "evento")
    has_place_marker = any(marker in normalized for marker in place_markers)
    has_title_case_shape = bool(re.search(r"\b[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]{3,}\b", title or ""))
    if (has_place_marker or has_title_case_shape) and normalized not in evidence_text:
        return True
    return False


def _query_requests_public_transport(user_message: str) -> bool:
    """Return whether the request asks for public transport guidance."""
    normalized = normalize_text(user_message)
    return bool(re.search(r"\b(public transport|transportes publicos|metro|carris|cp|bus|autocarro|comboio|train|tram|eletrico|route|rota|how do i get|como vou|como chego)\b", normalized))


def _query_requests_grounded_itinerary(user_message: str) -> bool:
    """Return whether a user asks for a concrete visit plan with named stops."""
    normalized = normalize_text(user_message)
    return bool(
        re.search(r"\b(plan|itinerary|roteiro|plano|visit|visitar|tour|dia|day)\b", normalized)
        and re.search(
            r"\b(monument|monumento|museum|museu|historic|historico|historia|história|culture|cultura|restaurant|restaurante|food|gastronomy|gastronomia|traditional|tradicional)\b",
            normalized,
        )
    )


def _query_requests_food_stop(user_message: str) -> bool:
    """Return whether the user explicitly asks for food or gastronomy in a plan."""
    normalized = normalize_text(user_message)
    return bool(
        re.search(
            r"\b(?:gastronomy|gastronomia|restaurant|restaurante|food|comida|lunch|almoco|almoço|dinner|jantar|pastry|pastelaria|pastel|traditional cuisine|cozinha tradicional)\b",
            normalized,
        )
    )


_CAFE_TOKEN_RE = re.compile(
    r"\b(?:cafe|cafes|cafetaria|pastelaria|pastelarias|pastel|pasteis|"
    r"nata|natas|brunch|coffee|pastry|pastries)\b"
)


def _query_requests_cafe_stop(user_message: str) -> bool:
    """Return whether the user explicitly asks for a cafe or pastry stop."""
    return bool(_CAFE_TOKEN_RE.search(normalize_text(user_message)))


def _evidence_card_is_cafe(card: Any) -> bool:
    """Return whether an evidence card supports a cafe or pastry stop."""
    if getattr(card, "kind", "") in {"coffee", "pastry"}:
        return True
    return bool(_CAFE_TOKEN_RE.search(_evidence_card_text(card)))


def _draft_includes_cafe_stop(draft: PlanDraft) -> bool:
    """Return whether a plan draft selected a distinct cafe or pastry stop."""
    for block in draft.blocks:
        if getattr(block, "kind", "") in {"coffee", "pastry"}:
            return True
        block_text = normalize_text(
            " ".join([block.title, block.purpose, *block.details, *block.limitations])
        )
        if _CAFE_TOKEN_RE.search(block_text):
            return True
    return False


_VIEWPOINT_TOKEN_RE = re.compile(r"\b(?:miradouro|miradouros|viewpoint|viewpoints|lookout|panoram\w*|vista\s+panoram\w*)\b")
_GARDEN_TOKEN_RE = re.compile(r"\b(?:jardim|jardins|garden|gardens|parque|parques|park|parks)\b")


def _query_requests_viewpoint_stop(user_message: str) -> bool:
    """Return whether the user explicitly asks for a viewpoint/lookout stop."""
    return bool(_VIEWPOINT_TOKEN_RE.search(normalize_text(user_message)))


def _query_requests_garden_stop(user_message: str) -> bool:
    """Return whether the user explicitly asks for a garden or park stop."""
    return bool(_GARDEN_TOKEN_RE.search(normalize_text(user_message)))


def _evidence_card_is_viewpoint(card: Any) -> bool:
    """Return whether an evidence card supports a viewpoint stop."""
    return bool(_VIEWPOINT_TOKEN_RE.search(_evidence_card_text(card)))


def _evidence_card_is_garden(card: Any) -> bool:
    """Return whether an evidence card supports a garden or park stop."""
    return bool(_GARDEN_TOKEN_RE.search(_evidence_card_text(card)))


def _draft_includes_token_stop(draft: PlanDraft, token_re: "re.Pattern[str]") -> bool:
    """Return whether any plan block matches the given category token regex."""
    for block in draft.blocks:
        block_text = normalize_text(
            " ".join([block.title, block.purpose, *block.details, *block.limitations])
        )
        if token_re.search(block_text):
            return True
    return False


def _query_requests_cultural_stop(user_message: str) -> bool:
    """Return whether the user explicitly asks for culture, museums, or heritage."""
    normalized = normalize_text(user_message)
    return bool(
        re.search(
            r"\b(?:museum|museu|museums|museus|monument|monumento|monumentos|historic|historical|"
            r"historico|historica|patrimonio|heritage|culture|cultura|cultural|exhibition|exposicao)\b",
            normalized,
        )
    )


def _extract_single_area_target(user_message: str) -> str:
    """Extract a compact named area when a request is clearly area-bounded."""
    normalized = normalize_text(user_message)
    if not re.search(r"\b(?:mini plan|mini plano|2 horas|two hours|pouco tempo|short time|perto|near|around|em|no|na)\b", normalized):
        return ""

    patterns = (
        r"\b(?:em|no|na|nos|nas|around|near)\s+(?P<area>[a-z0-9][a-z0-9 /'-]{1,60}?)(?:\s+(?:com|sem|para|durante|e|with|without|for|during|and)\b|[,.;]|$)",
        r"\b(?:perto de|perto do|perto da|near)\s+(?P<area>[a-z0-9][a-z0-9 /'-]{1,60}?)(?:\s+(?:com|sem|para|durante|e|with|without|for|during|and)\b|[,.;]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        area = re.sub(r"\s+", " ", match.group("area")).strip(" .:-")
        if area and area not in {"lisboa", "lisbon"}:
            return area

    known_areas = (
        "oriente",
        "parque das nacoes",
        "expo",
        "belem",
        "alfama",
        "baixa",
        "chiado",
        "campo de ourique",
        "avenidas novas",
    )
    for area in known_areas:
        if re.search(rf"\b{re.escape(area)}\b", normalized):
            return area
    return ""


def _evidence_card_text(card: Any) -> str:
    """Return normalized searchable text for an evidence card."""
    fields = getattr(card, "fields", {}) or {}
    return normalize_text(
        " ".join(
            [
                str(getattr(card, "title", "")),
                str(getattr(card, "summary", "")),
                *[str(key) for key in fields.keys()],
                *[str(value) for value in fields.values()],
            ]
        )
    )


def _evidence_card_matches_area(card: Any, target_area: str) -> bool:
    """Return whether an evidence card belongs to the requested compact area."""
    area = normalize_text(target_area)
    if not area:
        return False
    text = _evidence_card_text(card)
    if not text:
        return False

    if area in {"oriente", "parque das nacoes", "expo", "estacao do oriente"}:
        if "museu do oriente" in text:
            return False
        return bool(
            re.search(
                r"\b(?:oriente|parque das nacoes|expo|oceanario|pavilhao do conhecimento|"
                r"centro vasco da gama|vasco da gama|fil|altice arena|alameda dos oceanos|"
                r"rua do bojador|rossio dos olivais|1990|1998)\b",
                text,
            )
        )
    if area == "belem":
        return bool(re.search(r"\b(?:belem|brasilia|jeronimos|padrao|descobrimentos|imperio|india|1400)\b", text))
    if area == "alfama":
        return bool(re.search(r"\b(?:alfama|se de lisboa|catedral de lisboa|santa luzia|portas do sol|mouraria|1100)\b", text))
    return area in text


def _is_known_compact_area(target_area: str) -> bool:
    """Return whether the area has explicit local-planning semantics."""
    area = normalize_text(target_area)
    return area in {
        "oriente",
        "parque das nacoes",
        "expo",
        "estacao do oriente",
        "belem",
        "alfama",
        "baixa",
        "chiado",
        "campo de ourique",
        "avenidas novas",
    }


def _evidence_card_is_food(card: Any) -> bool:
    """Return whether an evidence card supports a food or restaurant stop."""
    if getattr(card, "kind", "") == "food":
        return True
    text = _evidence_card_text(card)
    return bool(
        re.search(
            r"\b(?:restaurant|restaurante|gastronomy|gastronomia|food|cuisine|cozinha|pastelaria|pastry|comida)\b",
            text,
        )
    )


def _evidence_card_is_cultural(card: Any) -> bool:
    """Return whether an evidence card supports a cultural or heritage stop."""
    if _evidence_card_is_food(card):
        return False
    text = _evidence_card_text(card)
    return bool(
        re.search(
            r"\b(?:museum|museu|museums|museus|monument|monumento|monumentos|historic|historical|"
            r"historico|historica|patrimonio|heritage|culture|cultura|cultural|exhibition|exposicao|"
            r"oceanario|pavilhao do conhecimento|castelo|torre|mosteiro)\b",
            text,
        )
    )


def _draft_includes_food_stop(draft: PlanDraft, food_cards: Sequence[Any]) -> bool:
    """Return whether a plan draft selected at least one evidenced food stop."""
    food_titles = [normalize_text(getattr(card, "title", "")) for card in food_cards]
    for block in draft.blocks:
        if getattr(block, "kind", "") in {"food", "coffee", "pastry"}:
            return True
        block_text = normalize_text(
            " ".join([block.title, block.purpose, *block.details, *block.movement, *block.limitations])
        )
        if re.search(r"\b(?:restaurant|restaurante|gastronomy|gastronomia|food|cuisine|cozinha|pastelaria|pastry|almoco|almoço|jantar)\b", block_text):
            return True
        if any(food_title and (food_title in block_text or block_text in food_title) for food_title in food_titles):
            return True
    return False


def _draft_includes_cultural_stop(draft: PlanDraft, cultural_cards: Sequence[Any]) -> bool:
    """Return whether a plan draft selected at least one cultural evidence stop."""
    cultural_titles = [normalize_text(getattr(card, "title", "")) for card in cultural_cards]
    for block in draft.blocks:
        if getattr(block, "kind", "") in {"museum", "culture", "cultural", "monument", "heritage"}:
            return True
        block_text = normalize_text(
            " ".join([block.title, block.purpose, *block.details, *block.movement, *block.limitations])
        )
        if re.search(
            r"\b(?:museum|museu|monument|monumento|historic|historical|historico|historica|"
            r"patrimonio|heritage|culture|cultura|cultural|exhibition|exposicao|oceanario)\b",
            block_text,
        ):
            return True
        if any(title and (title in block_text or block_text in title) for title in cultural_titles):
            return True
    return False


def _block_time_conflicts_with_supported_hours(block: Any) -> bool:
    """Return whether a timed block falls outside its evidenced opening hours."""
    title = str(getattr(block, "title", "") or "")
    scheduled_minutes = _extract_block_start_minutes(title)
    if scheduled_minutes is None:
        return False

    hours_fragments = [
        str(detail)
        for detail in getattr(block, "details", []) or []
        if re.search(r"(?:Hours|Horário|Horario|Horários|Horarios)\s*:\s*\*{0,2}", str(detail), flags=re.IGNORECASE)
    ]
    if not hours_fragments:
        return False
    hours_text = " ".join(hours_fragments)
    if re.search(r"\b(?:closed|fechado|encerrado)\b", normalize_text(hours_text)):
        return True

    intervals = _extract_hour_intervals(hours_text)
    if not intervals:
        return False
    return not any(_minutes_in_interval(scheduled_minutes, start, end) for start, end in intervals)


def _extract_block_start_minutes(title: str) -> int | None:
    """Extract the leading planned time from a rendered planner block title."""
    match = re.match(r"^\s*(?:\D{0,8})?(?P<hour>\d{1,2}):(?P<minute>\d{2})\b", title or "")
    if not match:
        return None
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    if hour > 23 or minute > 59:
        return None
    return hour * 60 + minute


def _extract_hour_intervals(text: str) -> List[tuple[int, int]]:
    """Extract opening-hour intervals from a detail string."""
    intervals: List[tuple[int, int]] = []
    for match in re.finditer(
        r"(?P<start_h>\d{1,2}):(?P<start_m>\d{2})\s*[-–—]\s*(?P<end_h>\d{1,2}):(?P<end_m>\d{2})",
        text or "",
    ):
        start = int(match.group("start_h")) * 60 + int(match.group("start_m"))
        end = int(match.group("end_h")) * 60 + int(match.group("end_m"))
        if end <= start:
            end += 24 * 60
        intervals.append((start, end))
    return intervals


def _minutes_in_interval(minutes: int, start: int, end: int) -> bool:
    """Return whether minutes since midnight falls inside an opening interval."""
    candidate = minutes
    if end > 24 * 60 and candidate < start:
        candidate += 24 * 60
    return start <= candidate < end


_WEEKDAY_PREFIX_RE = re.compile(
    r"^\s*(monday|tuesday|wednesday|thursday|friday|saturday|sunday|segunda|terca|terça|quarta|quinta|sexta|"
    r"sabado|sábado|domingo)(?:-feira)?\s*:",
    re.IGNORECASE,
)
_CLOSED_HOURS_RE = re.compile(
    r"^\s*(?:(?:today|hoje|[a-zà-ú]+(?:-feira)?)\s*:\s*)?(?:closed|fechado|encerrado)\b",
    re.IGNORECASE,
)


def hours_for_plan_day(hours_text: str, plan_day: Any, today: Any) -> str:
    """Return the card hours when they describe the plan's day, else an empty string.

    Cards show today's hours ("Today: 10:00 - 18:00") unless the search named
    a weekday, in which case they show that day's ("Sunday: Closed").
    """
    text = str(hours_text or "").strip()
    if not text:
        return ""
    if " · " in text:
        # "Saturday: 10:00 - 18:00 · Sunday: Closed": the plan day's segment.
        for segment in text.split(" · "):
            if hours_for_plan_day(segment, plan_day, today):
                return segment.strip()
        return ""
    prefix = _WEEKDAY_PREFIX_RE.match(text)
    if prefix:
        folded = normalize_text(prefix.group(1))
        index = next((i for i, names in enumerate(WEEKDAY_NAMES) if folded in names), None)
        return text if index is not None and index == plan_day.weekday() else ""
    return text if plan_day == today else ""


def closed_all_day(hours_text: str) -> bool:
    """Return whether card hours say the place is closed that day ("Today: Closed")."""
    return bool(_CLOSED_HOURS_RE.search(hours_text or ""))


def open_at(hours_text: str, minutes: int) -> bool | None:
    """Return whether card hours cover a time of day.

    Args:
        hours_text: The Hours value of a card ("10:00 - 18:00", "Closed").
        minutes: Time of day in minutes after midnight.

    Returns:
        ``True`` or ``False``, or ``None`` when the hours cannot be read.
    """
    if _CLOSED_HOURS_RE.search(hours_text or ""):
        return False
    intervals = _extract_hour_intervals(hours_text)
    if not intervals:
        return None
    return any(_minutes_in_interval(minutes, start, end) for start, end in intervals)


def minutes_until_closing(hours_text: str, minutes: int) -> int | None:
    """Return how many minutes the place stays open after a time of day.

    ``None`` when the hours cannot be read or no opening interval covers that time.
    """
    for start, end in _extract_hour_intervals(hours_text):
        if _minutes_in_interval(minutes, start, end):
            return end - (minutes if minutes >= start else minutes + 24 * 60)
    return None


def _user_prefers_portuguese(user_message: str) -> bool:
    """Return whether the request is clearly Portuguese."""
    normalized = normalize_text(user_message)
    return bool(
        re.search(
            r"\b(?:cria|d[aá]|quero|roteiro|plano|monumentos|hist[oó]ricos|gastronomia|tradicional|lisboa|hoje|amanh[aã])\b",
            normalized,
        )
    )


def _has_pt_language_drift(text: str) -> bool:
    """Return whether a Portuguese plan still contains English scaffold prose."""
    return bool(
        re.search(
            r"\b(?:morning|lunch|afternoon|dinner|start at|after lunch|from the|if you prefer|"
            r"have lunch|i couldn['’]?t|couldn['’]?t confirm|allow about|good first stop|"
            r"walking is|how to get|bel[eé]m stops|by tram|taxi|rideshare|"
            r"traditional portuguese cuisine|more heritage|traditional meal|live entertainment|real-time entertainment)\b",
            text or "",
            flags=re.IGNORECASE,
        )
    )


def _looks_like_generic_itinerary_block_title(title: str) -> bool:
    """Return whether a block title is only a temporal/area placeholder."""
    normalized = normalize_text(title)
    if not normalized:
        return True
    return bool(
        re.fullmatch(
            r"(?:manha|manhã|morning|almoco|almoço|lunch|tarde|afternoon|jantar|dinner|fim de tarde|evening)(?:\s+[a-z/ -]{0,40})?",
            normalized,
        )
        or normalized in {"baixa", "belem", "belém", "centro", "centro historico", "centro histórico", "baixa chiado"}
    )


def _query_requests_time_specific_visit(user_message: str) -> bool:
    """Return whether a plan implies a time window where closures matter."""
    normalized = normalize_text(user_message)
    return bool(
        re.search(
            r"\b(evening|tonight|night|afternoon|morning|today|tomorrow|"
            r"itinerary|route|day plan|plan|visit|visiting|"
            r"noite|esta noite|fim de tarde|tarde|manha|manhã|hoje|amanha|amanhã|"
            r"roteiro|itinerario|itinerário|plano|planeia|planear|programa|dia|visita|visitar)\b",
            normalized,
        )
    )


def _matching_evidence_card(title: str, evidence: EvidenceBundle) -> Any | None:
    """Return the evidence card that best matches a rendered block title."""
    normalized = normalize_text(title)
    if not normalized:
        return None
    for card in evidence.cards:
        card_title = normalize_text(card.title)
        if not card_title:
            continue
        if normalized == card_title or normalized in card_title or card_title in normalized:
            return card
    return None


def _evidence_card_is_closed(card: Any) -> bool:
    """Return whether a selected evidence card explicitly says the venue is closed."""
    fields = getattr(card, "fields", {}) or {}
    hours = " ".join(
        str(value)
        for key, value in fields.items()
        if normalize_text(str(key)) in {"hours", "horario", "horarios", "today", "hoje"}
    )
    normalized = normalize_text(hours)
    return bool(re.search(r"\b(closed|fechado|encerrado)\b", normalized))


def _renderer_can_normalize_field_label(text: str) -> bool:
    """Return whether a raw field label can be rendered as a semantic bullet."""
    return bool(
        re.match(
            r"^\s*(?:Description|Descrição|Descricao|Address|Morada|Location|Local|Venue|When|Quando|Hours|Horário|Horario|Price|Preço|Preco|Website|Tickets|Bilhetes|Category|Categoria)\s*:",
            text or "",
            flags=re.IGNORECASE,
        )
    )


# ==========================================================================
# Review of plans composed from a planning brief
# ==========================================================================

_BLOCK_TIME_PREFIX_RE = re.compile(r"^\s*\d{1,2}[:h]\d{2}\s*(?:[·\-–—:|]\s*)?")
_INTERNAL_WORDING_RE = re.compile(
    r"\b(?:tools?|agents?|evidence|evid[eê]ncias?|QA|LangSmith|JSON|source_ids?|found for|encontrado para)\b|"
    r"(?<![Ll]isboa )\bcards?\b|\bcart(?:ão|ões|ao|oes)\b|\bfichas?\b",
    re.IGNORECASE,
)
# Cards a traveller uses (transport and payment cards) are not internal words:
# "Compra um cartão Viva Viagem", "Buy a Viva Viagem card", "pay by card".
_PUBLIC_CARD_RE = re.compile(
    r"\b(?:lisboa|viva viagem|navegante|zapping|contactless|credit|debit|bank|payment|travel|transport)\s+cards?\b|"
    r"\bcart(?:ão|ões|ao|oes)\s+(?:de\s+)?(?:lisboa|viva viagem|navegante|zapping|contactless|cr[eé]dito|d[eé]bito|"
    r"banc[aá]ri\w*|multibanco|pagamento|transporte\w*)\b|"
    r"\b(?:by|with a|pay by|pay with|accepts?)\s+cards?\b|\b(?:com|por|aceita\w*)\s+cart(?:ão|ões|ao|oes)\b",
    re.IGNORECASE,
)


def without_public_cards(text: str) -> str:
    """Blank out transport and payment cards ("cartão Viva Viagem", "pay by card").

    They are travel advice, not the planner's evidence cards, so internal-word
    checks run on the text without them.
    """
    return _PUBLIC_CARD_RE.sub(" ", text or "")


def has_internal_wording(text: str) -> bool:
    """Return whether a traveller-facing text names internal machinery."""
    return bool(_INTERNAL_WORDING_RE.search(without_public_cards(text)))


# A stop farther than this from the requested area is flagged when a closer
# card of the same type exists.
_AREA_RADIUS_KM = 2.0
# Neighbourhood scale: beyond this a stop is in another neighbourhood, and a
# card of the same type within the inner radius is inside the named one.
_OUTSIDE_NEIGHBOURHOOD_KM = 0.9
_INSIDE_NEIGHBOURHOOD_KM = 0.5
_KIND_MATCHERS: dict[str, "re.Pattern[str]"] = {
    "museum": re.compile(r"\b(?:museu|museum|museus|museums|ciencia viva|pavilhao do conhecimento|oceanario|aquarium|gulbenkian|maat|mude)\b"),
    "monument": re.compile(r"\b(?:monument|monumento|mosteiro|monastery|torre|tower|castelo|castle|palacio|palace|igreja|church|se de lisboa|cathedral|padrao|convento)\b"),
    "viewpoint": re.compile(r"\b(?:miradouro|miradouros|viewpoint|view points|vista|panoram|rooftop)\b"),
    "garden": re.compile(r"\b(?:jardim|jardins|garden|gardens|parque|park|tapada|mata)\b"),
    "beach": re.compile(r"\b(?:praia|beach|beaches|praias)\b"),
    "cafe": re.compile(r"\b(?:cafe|cafes|cafetaria|coffee|pastelaria|pastry|confeitaria|padaria|bakery|nata|pasteis)\b"),
    "pastry": re.compile(r"\b(?:pastelaria|pastry|confeitaria|padaria|bakery|nata|pasteis|pastel)\b"),
    "restaurant": re.compile(r"\b(?:restaurante|restaurant|tasca|taberna|marisqueira|cervejaria|bistro|food|gastronom|cozinha|cuisine)\b"),
    "event": re.compile(r"\b(?:event|evento|concert|concerto|festival|exhibition|exposicao|espetaculo|show)\b"),
    "market": re.compile(r"\b(?:mercado|market|feira)\b"),
}
_KIND_ALIASES = {"lunch": "restaurant", "dinner": "restaurant", "indoor": "museum"}
_STRICT_KINDS = {"museum", "monument", "viewpoint", "garden", "beach"}
# Details that name a specific kind of place ("livraria histórica", "live
# music"): the chosen stop must be one, or the plan must say none was found.
_DETAIL_TOPICS: dict[str, "re.Pattern[str]"] = {
    "bookshop": re.compile(r"\b(?:livraria|livrarias|bookshop|bookstore|books?|livros|alfarrabista)"),
    # Specific kinds of show come before "live music": a fado show is not the
    # classical concert or the jazz the user asked for.
    "jazz": re.compile(r"\bjazz"),
    "classical music": re.compile(
        r"\b(?:musica classica|classical|orquestra|orchestra|sinfoni|symphon|filarmonic|philharmonic|"
        r"opera|piano|recital|camara|chamber music)"
    ),
    "theatre": re.compile(r"\b(?:teatro|theatre|theater|peca de teatro|play)\b"),
    "live music": re.compile(r"\b(?:fado|jazz|live music|musica ao vivo|concert|concerto|live entertainment)"),
    "market": re.compile(r"\b(?:mercado|market|feira)"),
    "pastry": re.compile(r"\b(?:pastelaria|pastry|pasteis|confeitaria|nata)"),
    "beach": re.compile(r"\b(?:praia|beach)"),
}


def strip_block_time(title: str) -> str:
    """Return a block title without its leading "HH:MM ·" time."""
    return _BLOCK_TIME_PREFIX_RE.sub("", str(title or "")).strip()


def match_block_to_card(title: str, evidence: EvidenceBundle) -> Any | None:
    """Return the place/event card a block title refers to, if any."""
    normalized = normalize_text(strip_block_time(title))
    if not normalized:
        return None
    best = None
    best_len = 0
    for card in evidence.cards:
        if getattr(card, "kind", "") in {"weather", "transport"}:
            continue
        card_title = normalize_text(card.title)
        if not card_title:
            continue
        if normalized == card_title:
            return card
        # Whole words only: the card "Sé" is not inside "Museu Nacional do Azulejo".
        contained = (
            re.search(rf"\b{re.escape(card_title)}\b", normalized)
            or re.search(rf"\b{re.escape(normalized)}\b", card_title)
        )
        if contained and len(card_title) > best_len:
            best, best_len = card, len(card_title)
    return best


def card_matches_kind(card: Any, kind: str) -> bool:
    """Return whether a card (as chosen for a block) satisfies one component type."""
    target = _KIND_ALIASES.get(kind, kind)
    if target in {"attraction", "walk", "shopping", "nightlife"}:
        return True
    pattern = _KIND_MATCHERS.get(target)
    if pattern is None:
        return True
    fields = getattr(card, "fields", {}) or {}
    # A garden returned by a viewpoint search is still a garden: for these
    # types the place's own title and category decide.
    found_for = "" if target in _STRICT_KINDS else str(fields.get("Found for", ""))
    category = str(fields.get("Category", ""))
    if target in {"museum", "monument"}:
        # The combined catalogue label says nothing about which of the two it is.
        category = re.sub(r"(?i)museums?\s*&\s*monuments?|museus?\s*e\s*monumentos?", "", category)
    text = normalize_text(
        " ".join(
            [
                str(getattr(card, "title", "")),
                category,
                found_for,
                str(fields.get("Features", "")),
            ]
        )
    )
    if target == "restaurant" and _KIND_MATCHERS["cafe"].search(text) and not _KIND_MATCHERS["restaurant"].search(text):
        return False
    return bool(pattern.search(text))


_SUNSET_REQUEST_RE = re.compile(r"sunset|por do sol|fim de tarde")


def _sunset_viewpoint_issue(draft: PlanDraft, brief: Any) -> str:
    """Return an issue when a sunset plan leaves its viewpoint before sunset."""
    if not _SUNSET_REQUEST_RE.search(normalize_text(f"{brief.time_window} {' '.join(brief.preferences)}")):
        return ""
    times = sunrise_sunset(plan_date(brief, lisbon_now().date()))
    if not times:
        return ""
    sunset_hour, sunset_minute = (int(part) for part in times[1].split(":"))
    sunset = sunset_hour * 60 + sunset_minute
    for block in draft.blocks:
        start = _extract_block_start_minutes(block.title)
        text = normalize_text(f"{block.kind} {block.title}")
        if start is None or not _KIND_MATCHERS["viewpoint"].search(text):
            continue
        stay = int(getattr(block, "stay_minutes", 0) or 25)
        if start + stay < sunset - 5:
            return (
                f'the sunset is at {times[1]} but "{strip_block_time(block.title)[:50]}" ends at '
                f"{(start + stay) // 60:02d}:{(start + stay) % 60:02d}; start the viewpoint about 30 minutes before "
                "the sunset, stay until it, and schedule dinner after it"
            )
        return ""
    return ""


_VISITLISBOA_URL_RE = re.compile(r"https?://www\.visitlisboa\.com/[^\s)]+")
_MAP_COORDINATES_RE = re.compile(r"query=(-?\d{1,2}\.\d{3,}),(-?\d{1,2}\.\d{3,})")


def card_coordinates(card: Any) -> tuple[float, float] | None:
    """Return a card's stored coordinates: catalogue entry, map pin, or Coordinates field."""
    if card is None:
        return None
    fields = getattr(card, "fields", {}) or {}
    url = _VISITLISBOA_URL_RE.search(" ".join(str(value) for value in fields.values()))
    if url:
        coordinates = lookup_place_coordinates(url.group(0))
        if coordinates:
            return coordinates
    match = re.match(r"\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)", str(fields.get("Coordinates") or "")) or _MAP_COORDINATES_RE.search(
        str(fields.get("Address") or fields.get("Venue") or "")
    )
    return (float(match.group(1)), float(match.group(2))) if match else None


def _zigzag_issue(matched: List[tuple[Any, Any]]) -> str:
    """Return an issue when a day goes to another area and comes back (A → far B → A).

    Inside one area the same back-and-forth counts when the walk out and the
    walk back are each over 800 m and the third stop is next to the first
    (Jerónimos → Torre de Belém → lunch beside the Jerónimos).
    """
    points = [(block, card_coordinates(card)) for block, card in matched]
    for (first, a), (middle, b), (last, c) in zip(points, points[1:], points[2:]):
        if not (a and b and c) or getattr(first, "day", 0) != getattr(last, "day", 0):
            continue
        out, back, across = haversine_distance(*a, *b), haversine_distance(*b, *c), haversine_distance(*a, *c)
        if (out > 2.5 and back > 2.5 and across < 1.5) or (min(out, back) > 0.8 and across < 0.35 * min(out, back)):
            return (
                f'the route goes from "{strip_block_time(first.title)[:40]}" to "{strip_block_time(middle.title)[:40]}" '
                f'and back to "{strip_block_time(last.title)[:40]}"; "{strip_block_time(middle.title)[:40]}" is far from the '
                "stops around it: move it next to stops near it (on another day if needed) or replace it with a card near "
                f'"{strip_block_time(first.title)[:40]}", so each area is visited once'
            )
    return ""


def _visit_order_issue(matched: List[tuple[Any, Any]]) -> str:
    """Return an issue when a run of sights is visited in a needlessly long order.

    Coches → Jerónimos → MAAT → Padrão walks 2.6 km along Belém; Coches →
    MAAT → Padrão → Jerónimos covers the same stops in 1.9 km. Each run of
    sights between meals and events keeps its first stop (where the traveller
    arrives) and the stop that follows the run.
    """
    runs: List[tuple[list, tuple[float, float] | None]] = []
    run: List[tuple[Any, Any, tuple[float, float]]] = []
    previous_day = None
    for block, card in matched:
        day = getattr(block, "day", 0)
        if run and day != previous_day:
            runs.append((run, None))
            run = []
        previous_day = day
        point = card_coordinates(card)
        if getattr(card, "kind", "") == "event" or card_matches_kind(card, "restaurant"):
            runs.append((run, point))
            run = []
        elif point:
            run.append((block, card, point))
        else:
            runs.append((run, None))
            run = []
    runs.append((run, None))
    for run, anchor in runs:
        if not 3 <= len(run) <= 6:
            continue

        def length(order: tuple[tuple[Any, Any, tuple[float, float]], ...]) -> float:
            points = [stop[2] for stop in order] + ([anchor] if anchor else [])
            return sum(haversine_distance(*a, *b) for a, b in zip(points, points[1:]))

        current = length(tuple(run))
        best = min((tuple(run[:1]) + rest for rest in permutations(run[1:])), key=length)
        if current - length(best) > 0.5 and current > 1.2 * length(best):
            order = " → ".join(f'"{strip_block_time(stop[0].title)[:40]}"' for stop in best)
            return (
                f"the stops go back and forth ({current:.1f} km of walking where {length(best):.1f} km is enough); "
                f"visit them in this order: {order}, and shift the times"
            )
    return ""


def _spread_issue(matched: List[tuple[Any, Any]], evidence: EvidenceBundle, brief: Any) -> str:
    """Return an issue when a plan shorter than a day jumps to a far area.

    A morning or an evening has no time for a cross-city trip between two
    stops when a stop of the same type was found close to the previous one.
    Areas and places the user named are kept, however far apart. A plan on
    foot keeps each day's consecutive stops within walking distance.
    """
    if getattr(brief, "transport_mode", "") == "walking":
        for (first_block, first_card), (second_block, second_card) in zip(matched, matched[1:]):
            if getattr(first_block, "day", 0) != getattr(second_block, "day", 0):
                continue
            a, b = card_coordinates(first_card), card_coordinates(second_card)
            if a and b and haversine_distance(*a, *b) > 3.0:
                return (
                    f'"{second_card.title[:60]}" is {haversine_distance(*a, *b):.1f} km from "{first_card.title[:60]}", '
                    "too far for a plan on foot; keep each day's stops within walking distance of each other"
                )
    if brief.days > 1 or len(brief.activity_areas) > 1 or not 0 < window_stop_target(brief) < 4:
        return ""
    named = [normalize_text(place) for place in brief.named_places if place]
    chosen = {id(card) for _block, card in matched}
    for (_first, first_card), (second, second_card) in zip(matched, matched[1:]):
        a, b = card_coordinates(first_card), card_coordinates(second_card)
        if not (a and b) or haversine_distance(*a, *b) <= 4.0:
            continue
        second_title = normalize_text(second_card.title)
        if any(place in second_title or second_title in place for place in named):
            continue
        kind = _block_component_kind(second, second_card)
        closer = [
            card for card in evidence.cards
            if id(card) not in chosen
            and getattr(card, "kind", "") not in {"weather", "transport"}
            and card_matches_kind(card, kind)
            and (coordinates := card_coordinates(card))
            and haversine_distance(*a, *coordinates) <= 2.0
        ]
        if closer:
            return (
                f'"{second_card.title[:60]}" is {haversine_distance(*a, *b):.1f} km from "{first_card.title[:60]}", '
                f'too far for a {brief.time_window or "short plan"}; use a closer stop of the same type, such as "{closer[0].title[:60]}"'
            )
    return ""


def _past_start_issue(draft: PlanDraft, brief: Any) -> str:
    """Return an issue when a plan for today starts before the current time.

    A request at 14:45 with no time given ("suggest a walk") is for now, not
    for 10:00. An explicit clock time, or a part of the day that is already
    over ("this morning" asked at 16:00), is left to the user.
    """
    now = lisbon_now()
    if getattr(brief, "start_time", "") or plan_date(brief, now.date()) != now.date():
        return ""
    # A full day or several days asked in the afternoon may well be for
    # another day; only a plan shorter than a day is moved to now.
    if brief.days > 1 or window_stop_target(brief) >= 4:
        return ""
    now_minutes = now.hour * 60 + now.minute
    window = normalize_text(brief.time_window)
    if re.search(r"\b(?:morning|manha)\b", window) and now_minutes > 13 * 60:
        return ""
    if re.search(r"\b(?:afternoon|tarde)\b", window) and now_minutes > 19 * 60:
        return ""
    starts = [
        minutes
        for block in draft.blocks
        if int(getattr(block, "day", 0) or 1) == 1
        and (minutes := _extract_block_start_minutes(block.title)) is not None
    ]
    if not starts or min(starts) >= now_minutes - 10:
        return ""
    earliest = (now_minutes + 20 + 4) // 5 * 5
    if earliest >= 24 * 60:
        # No start is left today; the plan cannot be moved later than midnight.
        return ""
    return (
        f"the plan is for today and it is now {now_minutes // 60:02d}:{now_minutes % 60:02d}; "
        f"start the first stop at {earliest // 60:02d}:{earliest % 60:02d} or later and shift the others"
    )


def _late_start_issue(draft: PlanDraft, brief: Any) -> str:
    """Return an issue when a plan for now starts hours later without a reason.

    "Estou em Entrecampos, sugere-me um passeio" at 11:13, with no time or part
    of the day given, is for now; a plan that starts at 14:00 leaves the
    morning empty for no reason the user gave.
    """
    now = lisbon_now()
    if getattr(brief, "start_time", "") or brief.time_window or plan_date(brief, now.date()) != now.date():
        return ""
    if brief.days > 1 or window_stop_target(brief) >= 4:
        return ""
    starts = [
        minutes
        for block in draft.blocks
        if int(getattr(block, "day", 0) or 1) == 1
        and (minutes := _extract_block_start_minutes(block.title)) is not None
    ]
    now_minutes = now.hour * 60 + now.minute
    if not starts or min(starts) <= now_minutes + 90:
        return ""
    earliest = (now_minutes + 20 + 4) // 5 * 5
    return (
        f"no time was asked and it is now {now_minutes // 60:02d}:{now_minutes % 60:02d}; "
        f"start the first stop around {earliest // 60:02d}:{earliest % 60:02d} and shift the others"
    )


def _late_full_day_start_issue(draft: PlanDraft, brief: Any) -> str:
    """Return an issue when a full day on a later date starts in the afternoon without a reason.

    "A day in Sintra tomorrow" starting at 14:41 because today's next train
    leaves at 14:01: listed departures are today's, and a full day starts in
    the morning unless the user gave a start time.
    """
    now = lisbon_now()
    if getattr(brief, "start_time", "") or not (brief.days > 1 or window_stop_target(brief) >= 4):
        return ""
    if plan_date(brief, now.date()) == now.date():
        return ""
    first_start: dict[int, int] = {}
    for block in draft.blocks:
        minutes = _extract_block_start_minutes(block.title)
        if minutes is not None:
            day = int(getattr(block, "day", 0) or 1)
            first_start[day] = min(first_start.get(day, minutes), minutes)
    late = [(day, minutes) for day, minutes in sorted(first_start.items()) if minutes > 11 * 60]
    if not late:
        return ""
    day, minutes = late[0]
    return (
        f"day {day} starts at {minutes // 60:02d}:{minutes % 60:02d}, leaving the morning empty; a full day starts around "
        "09:30-10:00 (departure times in the transport data are for today, not for the plan day), so shift the stops earlier"
    )


def _unused_sights_near(card: Any, matched: List[tuple[Any, Any]], evidence: EvidenceBundle, radius_km: float = 1.5) -> List[Any]:
    """Return the unused sight cards within walking reach of a card, nearest first, skipping parts of chosen sites."""
    origin = card_coordinates(card)
    if not origin:
        return []
    chosen = [other for _block, other in matched]
    found: List[tuple[float, Any]] = []
    for other in evidence.cards:
        if (
            getattr(other, "kind", "") != "place"
            or card_matches_kind(other, "restaurant")
            or any(other is item or _same_site(other, item) for item in chosen)
        ):
            continue
        point = card_coordinates(other)
        distance = haversine_distance(*origin, *point) if point else None
        if distance is not None and distance <= radius_km:
            found.append((distance, other))
    return [other for _distance, other in sorted(found, key=lambda item: item[0])]


def _long_gap_issue(matched: List[tuple[Any, Any]], brief: Any, evidence: EvidenceBundle) -> str:
    """Return an issue when a plan leaves a long empty gap between two stops.

    "Uma tarde em Alfama" with a viewpoint at 14:35 and the café at 19:30
    (because it reopens then) is five hours of nothing; the stop should move
    earlier inside its hours or be replaced. A full day allows a longer pause
    (a rest before dinner) but not an empty late afternoon.
    """
    limit = 120 if brief.days > 1 or window_stop_target(brief) >= 4 else 90
    timed = [
        (block, card, minutes)
        for block, card in matched
        if (minutes := _extract_block_start_minutes(block.title)) is not None
    ]
    for (previous, _card, previous_start), (current, _next_card, current_start) in zip(timed, timed[1:]):
        if getattr(previous, "day", 0) != getattr(current, "day", 0):
            continue
        stay = previous.stay_minutes or DEFAULT_STAY_MINUTES.get(previous.kind, 45)
        gap = current_start - (previous_start + stay)
        if gap > limit and current.kind == "event":
            # An event keeps its session time; the gap before it needs a stop.
            return (
                f'there are {gap} minutes with nothing planned between "{strip_block_time(previous.title)[:40]}" and '
                f'the event "{strip_block_time(current.title)[:40]}", whose time is fixed; add a stop in between '
                "or move the earlier stops later"
            )
        if gap > limit and current_start >= 18 * 60 and card_matches_kind(_next_card, "restaurant"):
            # Dinner cannot move into the afternoon; the afternoon needs more
            # stops, unless no card near the earlier one is left (a rest then).
            nearby = _unused_sights_near(_card, matched, evidence)
            if not nearby:
                continue
            names = ", ".join(f'"{card.title[:50]}"' for card in nearby[:2])
            return (
                f'there are {gap} minutes with nothing planned between "{strip_block_time(previous.title)[:40]}" and '
                f'dinner at "{strip_block_time(current.title)[:40]}"; add a nearby card in between, such as {names}, '
                "and keep dinner at its time"
            )
        if gap > limit:
            return (
                f'there are {gap} minutes with nothing planned between "{strip_block_time(previous.title)[:40]}" and '
                f'"{strip_block_time(current.title)[:40]}"; move the later stop earlier inside its opening hours, '
                "add a stop in between or choose another card that is open at that time"
            )
    return ""


def _event_meal_distance_issue(matched: List[tuple[Any, Any]], evidence: EvidenceBundle) -> str:
    """Return an issue when the meal of an event plan is far from the event venue.

    The dinner of an evening around a show belongs near the venue; a
    restaurant 3 km away described as "near the event" is wrong.
    """
    event = next(((block, card) for block, card in matched if getattr(card, "kind", "") == "event"), None)
    if event is None:
        return ""
    event_point = card_coordinates(event[1])
    if not event_point:
        return ""
    for block, card in matched:
        if not card_matches_kind(card, "restaurant"):
            continue
        point = card_coordinates(card)
        if not point or haversine_distance(*event_point, *point) <= 1.5:
            continue
        chosen = {id(other) for _block, other in matched}
        closer = sorted(
            (
                other for other in evidence.cards
                if id(other) not in chosen
                and card_matches_kind(other, "restaurant")
                and card_coordinates(other)
                and haversine_distance(*event_point, *card_coordinates(other)) <= 1.0
            ),
            key=lambda other: haversine_distance(*event_point, *card_coordinates(other)),
        )
        if closer:
            return (
                f'"{card.title[:50]}" is {haversine_distance(*event_point, *point):.1f} km from the event venue; '
                f'use a restaurant near it, such as "{closer[0].title[:50]}"'
            )
    return ""


# Component types that do not narrow which sights the user wants to see.
_GENERAL_COMPONENT_KINDS = {"attraction", "walk", "restaurant", "lunch", "dinner", "cafe", "pastry", "nightlife", "event"}
_TOP_ATTRACTIONS_RE = re.compile(r"top attractions|principais atracoes")


def _landmark_issue(matched: List[tuple[Any, Any]], evidence: EvidenceBundle, brief: Any) -> str:
    """Return an issue when a general visit to one area leaves out its best-known landmark.

    "Plan a visit to Belém" with the Coach Museum and the MAAT but without the
    Jerónimos: the first card of the area's top-attractions search is its
    best-known landmark. Requests for named places or for given sight types
    (museums, viewpoints) choose their own stops and are left alone.
    """
    if (
        len(brief.activity_areas) != 1
        or brief.named_places
        or any(component.kind not in _GENERAL_COMPONENT_KINDS for component in brief.components)
    ):
        return ""
    landmark = next(
        (
            card for card in evidence.cards
            if getattr(card, "kind", "") == "place"
            and _TOP_ATTRACTIONS_RE.search(normalize_text(str((card.fields or {}).get("Found for", ""))))
            and not card_matches_kind(card, "restaurant")
        ),
        None,
    )
    if landmark is None or any(card is landmark or _same_site(card, landmark) for _block, card in matched):
        return ""
    distance = _card_distance_km(landmark)
    title = normalize_text(landmark.title)
    today = lisbon_now().date()
    if (
        (distance is not None and distance > 2.0)
        or closed_all_day(hours_for_plan_day(str(landmark.fields.get("Hours", "")), plan_date(brief, today), today))
        or any(normalize_text(item) and normalize_text(item) in title for item in brief.exclusions)
    ):
        return ""
    sights = [card for _block, card in matched if not card_matches_kind(card, "restaurant")]
    if len(sights) < 2:
        return ""
    return (
        f'a visit to {brief.activity_areas[0]} leaves out "{landmark.title[:60]}", its best-known landmark; '
        "add it, replacing a lesser stop only when the time window has no room, and keep the walk in one direction"
    )


# Earliest end of the last stop for a morning or an afternoon that is not left
# half empty (minutes): about two hours before lunch, or before 19:00.
_PART_OF_DAY_MIN_END = (
    (re.compile(r"\b(?:morning|manha)\b"), 11 * 60 + 30),
    (re.compile(r"\b(?:afternoon|(?<!fim de )(?<!fim da )tarde)\b"), 16 * 60 + 45),
)


def _short_part_of_day_issue(draft: PlanDraft, unused: List[Any], brief: Any) -> str:
    """Return an issue when a morning or an afternoon plan ends hours before that part of the day.

    "Tomorrow afternoon in Belém" with two stops finished by 15:30 leaves the
    rest of the afternoon empty while nearby place cards are unused. Plans
    with an end time, a stated duration, or only the stops the user named are
    left alone.
    """
    window = normalize_text(brief.time_window)
    if (
        brief.days > 1
        or getattr(brief, "end_time", "")
        or _EVENING_RE.search(window)
        or re.search(r"\d+\s*(?:h|hours?|horas?)\b", window)
        or not extra_stops_allowed(brief)
    ):
        return ""
    from agent.utils.geographic_scope import extract_aml_municipality_mentions

    min_end = next((end for pattern, end in _PART_OF_DAY_MIN_END if pattern.search(window)), None)
    # A neighbourhood is walking scale, a municipality is not (as in the review's area checks).
    neighbourhood = any(not extract_aml_municipality_mentions(area) for area in brief.activity_areas)
    radius = _OUTSIDE_NEIGHBOURHOOD_KM if neighbourhood else _AREA_RADIUS_KM
    nearby = [card for card in unused if (_card_distance_km(card) or 0.0) <= radius]
    if min_end is None or not nearby:
        return ""
    ends = [
        start + int(block.stay_minutes or DEFAULT_STAY_MINUTES.get(str(block.kind), 45))
        for block in draft.blocks
        if (start := _extract_block_start_minutes(block.title)) is not None
    ]
    if not ends or max(ends) >= min_end:
        return ""
    names = ", ".join(f'"{card.title[:50]}"' for card in nearby[:3])
    return (
        f"the plan ends around {max(ends) // 60:02d}:{max(ends) % 60:02d}, leaving most of the {brief.time_window} empty; "
        f"add one or two nearby stops from the place cards, such as {names}, keeping the walk in one direction"
    )


def _short_day_issue(draft: PlanDraft, matched: List[tuple[Any, Any]], evidence: EvidenceBundle, brief: Any) -> str:
    """Return an issue when a day ends long before the part of the day it covers.

    A full day (or a day of several) should not end before the afternoon, and
    a morning or an afternoon should not end hours early. Only raised when
    unused place cards could fill the gap.
    """
    chosen = {id(card) for _block, card in matched}
    # "2 museums" stays two museums: a type whose count is reached is not suggested.
    full_kinds = [
        kind for kind, count in requested_counts(brief).items()
        if count and sum(1 for _block, card in matched if card_matches_kind(card, kind)) >= count
    ]
    unused = [
        card for card in evidence.cards
        if id(card) not in chosen
        and getattr(card, "kind", "") == "place"
        and not card_matches_kind(card, "restaurant")
        and not any(card_matches_kind(card, kind) for kind in full_kinds)
        and not any(_same_site(card, other) for _block, other in matched)
    ]
    if not unused:
        return ""
    if brief.days <= 1 and window_stop_target(brief) < 4:
        return _short_part_of_day_issue(draft, unused, brief)
    last_start: dict[int, int] = {}
    for block in draft.blocks:
        minutes = _extract_block_start_minutes(block.title)
        if minutes is not None:
            day = int(getattr(block, "day", 0) or 1)
            last_start[day] = max(last_start.get(day, 0), minutes)
    # A last stop starting before mid-afternoon (often lunch) leaves the day half empty.
    short_days = [day for day, minutes in sorted(last_start.items()) if minutes < 15 * 60]
    if not short_days:
        return ""
    names = ", ".join(f'"{card.title[:50]}"' for card in unused[:3])
    return (
        f"day {short_days[0]} ends before the afternoon although this is a full day; "
        f"add afternoon stops (until about 18:00) from the place cards, such as {names}"
    )


# Title words that do not identify a place on their own.
_GENERIC_TITLE_WORDS = {
    "museu", "museum", "jardim", "jardins", "garden", "gardens", "parque", "park", "miradouro", "viewpoint",
    "restaurante", "restaurant", "cafe", "igreja", "church", "palacio", "palace", "mercado", "market",
    "lisboa", "lisbon", "nacional", "national", "praia", "beach", "casa", "house", "centro", "center", "centre",
}


def _distinctive_title_tokens(title: str) -> set[str]:
    """Return the words of a title that identify a place ("jorge" in "Castelo de São Jorge")."""
    return {token for token in re.findall(r"[a-z0-9]{4,}", normalize_text(title)) if token not in _GENERIC_TITLE_WORDS}


def _same_site(first: Any, second: Any) -> bool:
    """Return whether two cards are parts of one site: within 300 m and sharing a distinctive name word."""
    a, b = card_coordinates(first), card_coordinates(second)
    shared = _distinctive_title_tokens(first.title) & _distinctive_title_tokens(second.title)
    return bool(a and b and shared) and haversine_distance(*a, *b) <= 0.3


def _same_site_issue(matched: List[tuple[Any, Any]], evidence: EvidenceBundle) -> str:
    """Return an issue when two sights are parts of one site while other cards are unused.

    "Castelo de São Jorge", "Miradouro de São Jorge" and "Torre da Igreja do
    Castelo de São Jorge" are one visit of the castle hill, not three stops.
    Two sights within 300 m that share a distinctive name word count as one.
    """
    sights = [card for _block, card in matched if getattr(card, "kind", "") == "place" and not card_matches_kind(card, "restaurant")]
    chosen = {id(card) for _block, card in matched}
    for index, first in enumerate(sights):
        for second in sights[index + 1:]:
            if first is second or not _same_site(first, second):
                continue
            shared = _distinctive_title_tokens(first.title) & _distinctive_title_tokens(second.title)
            alternative = next(
                (
                    card for card in evidence.cards
                    if id(card) not in chosen
                    and getattr(card, "kind", "") == "place"
                    and not card_matches_kind(card, "restaurant")
                    and not _distinctive_title_tokens(card.title) & shared
                    and (_card_distance_km(card) or 0.0) <= _OUTSIDE_NEIGHBOURHOOD_KM
                ),
                None,
            )
            if alternative is None:
                return ""
            return (
                f'"{first.title[:50]}" and "{second.title[:50]}" are one site; keep one and use another card '
                f'instead, such as "{alternative.title[:50]}"'
            )
    return ""


def _direct_answer_issue(draft: PlanDraft, matched: List[tuple[Any, Any]], brief: Any, evidence: EvidenceBundle) -> str:
    """Return an issue when the direct answer names none of the chosen stops, or a card that is no stop.

    Words of the area and the start point do not count: "a day in Sintra"
    does not name the "Palácio Nacional de Sintra".
    """
    answer = normalize_text(draft.direct_answer)
    if not answer or not matched:
        return ""
    area_words = set(normalize_text(" ".join([*brief.activity_areas, brief.origin or ""])).split())
    names_a_stop = False
    for _block, card in matched:
        tokens = [
            token for token in re.findall(r"[a-z0-9]{4,}", normalize_text(card.title))
            if token not in _GENERIC_TITLE_WORDS and token not in area_words
        ]
        if not tokens or any(re.search(rf"\b{re.escape(token)}\b", answer) for token in tokens):
            names_a_stop = True
            break
    if not names_a_stop:
        return "direct_answer names none of the stops; name the main stops in that sentence"
    # "Castelo de São Jorge" in the answer while the stop is the "Miradouro de
    # São Jorge": a card the plan does not use, unless a stop's name holds it.
    chosen_titles = [normalize_text(card.title) for _block, card in matched]
    places = normalize_text(" ".join([*brief.activity_areas, brief.origin or ""]))
    for card in evidence.cards:
        title = normalize_text(card.title)
        if (
            getattr(card, "kind", "") in {"place", "event"}
            and len(title) >= 8
            and title not in places
            and not any(title in chosen for chosen in chosen_titles)
            and re.search(rf"\b{re.escape(title)}\b", answer)
        ):
            return f'direct_answer names "{card.title[:60]}", which is not one of the stops; name only the planned stops'
    return ""


# "rainy afternoon", "chuvosa", and umbrella advice all claim rain.
_RAIN_CLAIM_RE = re.compile(r"\b(?:rain\w*|showers?|chuv\w*|aguaceiros?|chover|umbrella|guarda-chuva)\b")
_NO_RAIN_RE = re.compile(
    r"\b(?:no|not|without|unlikely|dry|sem|nao|improvavel|seca?|premise|premissa|don t|dont|no need|dispensa\w*|desnecessari\w*)\b"
)


def _rain_claim_issue(draft: PlanDraft, weather_cards: List[Any]) -> str:
    """Return an issue when the plan says rain is expected and the forecast says it is not."""
    if not weather_cards:
        return ""
    probabilities = [
        float(value.replace(",", "."))
        for card in weather_cards
        for value in re.findall(r"(\d+(?:[.,]\d+)?)\s*%", " ".join([card.summary, *card.fields.values()]))
    ]
    if not probabilities or max(probabilities) >= 30:
        return ""
    for text in [*draft.weather_strategy, draft.direct_answer, *draft.tips]:
        normalized = normalize_text(text)
        if _RAIN_CLAIM_RE.search(normalized) and not _NO_RAIN_RE.search(normalized):
            return (
                f"the forecast shows at most {max(probabilities):g}% chance of rain, but the plan says rain is expected; "
                "say that no rain is forecast (the rainy premise does not match), do not suggest an umbrella, "
                "and keep the indoor plan"
            )
    return ""


def _card_text(card: Any) -> str:
    """Return a card's title, category, features, and description, normalized."""
    fields = getattr(card, "fields", {}) or {}
    return normalize_text(
        " ".join(
            str(value)
            for value in (
                getattr(card, "title", ""),
                fields.get("Category", ""),
                fields.get("Features", ""),
                fields.get("Description", ""),
                getattr(card, "summary", ""),
            )
        )
    )


# Start-time bounds (minutes) for the parts of the day a brief can name.
_WINDOW_BOUNDS = (
    (re.compile(r"\b(?:morning|manha)\b"), 8 * 60 + 30, 13 * 60),
    (re.compile(r"\b(?:afternoon|tarde)\b"), 13 * 60 + 30, 19 * 60),
    (re.compile(r"\b(?:evening|night|noite)\b"), 17 * 60 + 30, 23 * 60 + 30),
)


def _time_window_issue(draft: PlanDraft, brief: Any) -> str:
    """Return an issue when the stops start outside the part of the day asked for."""
    window = normalize_text(brief.time_window)
    # "2 hours morning" is still a morning; a bare duration matches no bound.
    if not window or re.search(r"full day|dia inteiro|half day|meio dia|sunset|por do sol|fim de tarde|\d+\s*(?:days?|dias?)\b", window):
        return ""
    bounds = next(((start, end) for pattern, start, end in _WINDOW_BOUNDS if pattern.search(window)), None)
    if not bounds:
        return ""
    starts = [minutes for minutes in (_extract_block_start_minutes(block.title) for block in draft.blocks) if minutes is not None]
    if not starts:
        return ""
    if min(starts) < bounds[0] - 30 or min(starts) > bounds[1]:
        return (
            f"the request is for the {brief.time_window}, but the first stop starts at "
            f"{min(starts) // 60:02d}:{min(starts) % 60:02d}; schedule the stops between "
            f"{bounds[0] // 60:02d}:{bounds[0] % 60:02d} and {bounds[1] // 60:02d}:{bounds[1] % 60:02d}"
        )
    return ""


_EVENING_RE = re.compile(r"evening|night|noite|sunset|por do sol|fim de tarde")


# Default stays per block kind when the model gives none (minutes).
# Typical stay per block kind, used when the model gives no stay_minutes.
DEFAULT_STAY_MINUTES = {
    "museum": 75,
    "culture": 60,
    "viewpoint": 25,
    "garden": 40,
    "beach": 90,
    "food": 75,
    "coffee": 25,
    "pastry": 20,
    "event": 90,
    "place": 45,
}


def _estimated_travel_minutes(distance_km: float) -> int:
    """Rough door-to-door minutes: walking up to 1.2 km, public transport beyond."""
    if distance_km <= 1.2:
        return max(3, round(distance_km * 15))
    return round(12 + 3.5 * distance_km)


def _schedule_travel_issues(matched: List[tuple[Any, Any]]) -> List[str]:
    """Return issues when a start time leaves no room for the stay plus the trip there."""
    issues: List[str] = []
    for (previous, previous_card), (current, current_card) in zip(matched, matched[1:]):
        if getattr(previous, "day", 0) != getattr(current, "day", 0):
            continue
        start, next_start = _extract_block_start_minutes(previous.title), _extract_block_start_minutes(current.title)
        a, b = card_coordinates(previous_card), card_coordinates(current_card)
        if start is None or next_start is None or not (a and b):
            continue
        stay = int(getattr(previous, "stay_minutes", 0) or DEFAULT_STAY_MINUTES.get(str(previous.kind), 45))
        travel = _estimated_travel_minutes(haversine_distance(*a, *b))
        if next_start < start + stay + travel - 10:
            arrival = start + stay + travel
            issues.append(
                f'"{strip_block_time(current.title)[:50]}" starts at {current.title[:5].strip()}, but after {stay} min at '
                f'"{strip_block_time(previous.title)[:40]}" and about {travel} min of travel it is reached around '
                f"{arrival // 60:02d}:{arrival % 60:02d}; fix the times, or reorder the stops so nearby ones are together "
                "and the meal falls at a normal time"
            )
    return issues[:3]


def _missing_meal_issues(matched: List[tuple[Any, Any]], evidence: EvidenceBundle, brief: Any) -> List[str]:
    """Return issues for full days without lunch or dinner and evening plans without dinner."""
    food_cards = [
        card for card in evidence.cards
        if getattr(card, "kind", "") not in {"weather", "transport"} and card_matches_kind(card, "restaurant")
    ]
    if not food_cards:
        return []
    window = normalize_text(f"{brief.time_window} {brief.day}")
    issues: List[str] = []
    days: dict[int, List[tuple[Any, Any]]] = {}
    for block, card in matched:
        days.setdefault(int(getattr(block, "day", 0) or 0), []).append((block, card))

    def has_meal(items: List[tuple[Any, Any]], start: int, end: int) -> bool:
        for block, card in items:
            minutes = _extract_block_start_minutes(block.title)
            if minutes is not None and start <= minutes < end and card_matches_kind(card, "restaurant"):
                return True
        return False

    # The day field ('day after tomorrow', 'dia 12') says when, not how long.
    if brief.days > 1 or is_full_day_window(brief.time_window):
        for day, items in sorted(days.items()):
            label = f"day {day}" if day else "the day"
            if not has_meal(items, 11 * 60 + 30, 14 * 60 + 45):
                issues.append(f"{label} has no lunch stop; add one from the restaurant cards near that day's stops")
            # Each day of several keeps the full-day shape, ending with dinner.
            if brief.days > 1 and not (brief.end_time and brief.end_time < "19:30") and not has_meal(items, 18 * 60 + 30, 22 * 60):
                issues.append(f"{label} has no dinner stop; add one from the restaurant cards near that day's last stop")
        if full_day_with_evening(brief) and not has_meal(matched, 18 * 60 + 30, 22 * 60):
            issues.append("the full day has no dinner stop; add one from the restaurant cards near the last afternoon stop")
    elif _EVENING_RE.search(window) and not has_meal(matched, 18 * 60, 23 * 60 + 30):
        issues.append("the evening plan has no dinner stop; add one from the restaurant cards near the other stops")
    return issues


def _block_component_kind(block: Any, card: Any) -> str:
    """Return the component type a chosen block stands for (for the closer-card check)."""
    fields = getattr(card, "fields", {}) or {}
    text = normalize_text(" ".join([str(getattr(block, "kind", "")), str(getattr(card, "title", "")), str(fields.get("Category", ""))]))
    for kind in ("viewpoint", "garden", "museum", "monument", "beach", "pastry", "cafe", "restaurant", "market", "event"):
        if _KIND_MATCHERS[kind].search(text):
            return kind
    return "attraction"


def _card_distance_km(card: Any) -> float | None:
    """Return the distance-from-area value shown on a card, in kilometres."""
    value = str((getattr(card, "fields", {}) or {}).get("Distance", ""))
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*km", value)
    if match:
        return float(match.group(1).replace(",", "."))
    match = re.search(r"(\d+)\s*m\b", value)
    return float(match.group(1)) / 1000.0 if match else None


# Traveller-facing names of the component types, for unmet-request notes.
_COMPONENT_LABELS = {
    "museum": ("museu", "museus", "museum", "museums"),
    "monument": ("monumento", "monumentos", "monument", "monuments"),
    "viewpoint": ("miradouro", "miradouros", "viewpoint", "viewpoints"),
    "garden": ("jardim", "jardins", "garden", "gardens"),
    "beach": ("praia", "praias", "beach", "beaches"),
    "cafe": ("café", "cafés", "café", "cafés"),
    "pastry": ("pastelaria", "pastelarias", "pastry shop", "pastry shops"),
    "restaurant": ("restaurante", "restaurantes", "restaurant", "restaurants"),
    "lunch": ("almoço", "almoços", "lunch stop", "lunch stops"),
    "dinner": ("jantar", "jantares", "dinner stop", "dinner stops"),
    "market": ("mercado", "mercados", "market", "markets"),
    "indoor": ("visita interior", "visitas interiores", "indoor visit", "indoor visits"),
}


_EVENT_TIMING_KEYS = ("Date/Time", "Data/Hora", "When", "Quando", "Schedule", "Horários", "Hours")
_CLOCK_RE = re.compile(r"\b(\d{1,2})[:h](\d{2})\b")
_CLOCK_RANGE_RE = re.compile(r"\b\d{1,2}[:h]\d{2}\s*(?:-|–|—|to|até|às|a)\s*\d{1,2}[:h]\d{2}\b")


def _event_session_issue(block: Any, card: Any, start_minutes: int) -> str:
    """Return an issue when an event is planned at a time that is none of its sessions.

    "Fado in Chiado • 17:30 e 19:00" planned at 18:10 or 20:15 is a time the
    show does not run. Opening hours ("10:00 - 18:00", an exhibition) are not
    sessions and are left to the opening-hours check.
    """
    sessions = _event_sessions(card)
    if not sessions or any(abs(start_minutes - session) <= _SESSION_TOLERANCE_MINUTES for session in sessions):
        return ""
    listed = ", ".join(f"{session // 60:02d}:{session % 60:02d}" for session in sessions[:6])
    return (
        f'"{strip_block_time(block.title)[:50]}" starts at {block.title[:5].strip()}, but its sessions are at {listed}; '
        "start it at one of them and fit the other stops around it"
    )


_SESSION_TOLERANCE_MINUTES = 10


def _event_sessions(card: Any) -> List[int]:
    """Return an event card's session start times in minutes, or [] for opening hours or no times."""
    fields = getattr(card, "fields", {}) or {}
    timing = " ".join(str(fields.get(key) or "") for key in _EVENT_TIMING_KEYS)
    if _CLOCK_RANGE_RE.search(timing):
        return []
    return sorted({int(hour) * 60 + int(minute) for hour, minute in _CLOCK_RE.findall(timing) if int(hour) < 24})


def snap_events_to_sessions(draft: PlanDraft, evidence: EvidenceBundle) -> PlanDraft:
    """Move each event block that is off its sessions to the next listed session.

    The repair round may leave "Fado in Chiado" at 17:45 when it runs at 17:30
    and 19:00; the plan then shows a time the show does not start. The next
    session (or the last one, when all are earlier) is used; the schedule
    alignment that follows pushes later stops back if needed.
    """
    for block in draft.blocks:
        card = match_block_to_card(block.title, evidence)
        if getattr(card, "kind", "") != "event":
            continue
        start = _extract_block_start_minutes(block.title)
        sessions = _event_sessions(card)
        if start is None or not sessions or any(abs(start - session) <= _SESSION_TOLERANCE_MINUTES for session in sessions):
            continue
        target = next((session for session in sessions if session >= start), sessions[-1])
        block.title = re.sub(r"^\s*\d{1,2}[:h]\d{2}", f"{target // 60:02d}:{target % 60:02d}", block.title, count=1)
    return draft


def untimed_event_notes(draft: PlanDraft, evidence: EvidenceBundle, language: str) -> List[str]:
    """Return a note for each planned event whose card gives no session time.

    An event card with dates only ("27 Sep – 5 Oct") gives no hour; the
    time in the plan is a suggestion and the user must check the session.

    Args:
        draft: Final plan.
        evidence: Evidence cards.
        language: Output language.

    Returns:
        One note per such event.
    """
    notes: List[str] = []
    for block in draft.blocks:
        card = match_block_to_card(block.title, evidence)
        if card is None or getattr(card, "kind", "") != "event":
            continue
        fields = getattr(card, "fields", {}) or {}
        timing = " ".join(str(fields.get(key) or "") for key in _EVENT_TIMING_KEYS)
        if _CLOCK_RE.search(timing):
            continue
        name = strip_block_time(block.title)
        notes.append(
            f"Os dados de {name} não indicam a hora da sessão; confirma-a antes de ires e ajusta o resto do plano."
            if language == "pt"
            else f"The data for {name} gives no session time; check it before you go and adjust the rest of the plan."
        )
    return notes


def requested_counts(brief: Any) -> Dict[str, int]:
    """Return the requested number of stops per component kind (0: no number given).

    The brief may list a kind more than once (a "monument" for each named
    monument); the counts add up.
    """
    counts: Dict[str, int] = {}
    for component in brief.components:
        counts[component.kind] = counts.get(component.kind, 0) + int(component.count or 0)
    return counts


def closed_named_place_notes(draft: PlanDraft, evidence: EvidenceBundle, brief: Any, language: str) -> List[str]:
    """Return a note for each place the user named that is closed all day and left out.

    The review moves the plan to an open card; the user still has to hear why
    the place they asked for is not in it.

    Args:
        draft: Final plan.
        evidence: Evidence cards.
        brief: Planning brief, for the named places and the plan's day.
        language: Output language.

    Returns:
        One note per such place.
    """
    today = lisbon_now().date()
    day = plan_date(brief, today)
    used = {id(match_block_to_card(block.title, evidence)) for block in draft.blocks}
    existing = normalize_text(" ".join(draft.limitations))
    notes: List[str] = []
    for place in brief.named_places:
        card = match_block_to_card(place, evidence)
        if card is None or id(card) in used or normalize_text(card.title) in existing:
            continue
        if closed_all_day(hours_for_plan_day(str((card.fields or {}).get("Hours", "")), day, today)):
            notes.append(
                f"{card.title} está fechado nesse dia, por isso não entrou no plano."
                if language == "pt"
                else f"{card.title} is closed that day, so it is not in the plan."
            )
    return notes


def unmet_component_notes(draft: PlanDraft, evidence: EvidenceBundle, brief: Any, language: str) -> List[str]:
    """Return one limitation per requested stop type the final plan does not cover.

    The review sends missing components back to the model once; this runs on
    the final plan, after the repair round and after closed stops are removed,
    so a request that is still unmet is stated rather than silently dropped.
    A type the plan's own limitations already mention is left to them.
    """
    is_pt = language == "pt"
    matched = [
        card for card in (match_block_to_card(block.title, evidence) for block in draft.blocks) if card is not None
    ]
    existing = normalize_text(" ".join(draft.limitations))
    area = brief.activity_areas[0] if brief.activity_areas else ""
    notes: List[str] = []
    for component in brief.components:
        if component.kind == "event":
            # Every card "matches" the event kind; only an event card is one.
            if any(getattr(card, "kind", "") == "event" for card in matched) or re.search(r"\bevent", existing):
                continue
            found_events = any(getattr(card, "kind", "") == "event" for card in evidence.cards)
            if is_pt:
                notes.append(
                    "O plano não inclui nenhum evento: os eventos encontrados para esse dia não encaixavam no horário nem na zona do plano."
                    if found_events
                    else "O plano não inclui nenhum evento: não encontrei eventos confirmados para esse dia."
                )
            else:
                notes.append(
                    "The plan includes no event: the events found for that day did not fit the plan's times or area."
                    if found_events
                    else "The plan includes no event: no confirmed event was found for that day."
                )
            continue
        labels = _COMPONENT_LABELS.get(component.kind)
        if not labels:
            continue
        required = component.count or 1
        found = sum(1 for card in matched if card_matches_kind(card, component.kind))
        if found >= required:
            continue
        singular_pt, plural_pt, singular_en, plural_en = labels
        if any(normalize_text(label) in existing for label in labels):
            continue
        where_pt = f" perto de {area}" if area else ""
        where_en = f" near {area}" if area else ""
        if found:
            notes.append(
                f"Pediste {required} {plural_pt}, mas o plano só inclui {found}: não houve mais opções confirmadas{where_pt} que encaixassem no horário."
                if is_pt
                else f"You asked for {required} {plural_en}, but the plan includes only {found}: no other confirmed option{where_en} fitted the schedule."
            )
        else:
            notes.append(
                f"O plano não inclui {singular_pt if required == 1 else plural_pt}: não houve nenhuma opção confirmada{where_pt} que encaixasse no horário."
                if is_pt
                else f"The plan has no {singular_en if required == 1 else plural_en}: no confirmed option{where_en} fitted the schedule."
            )
    return notes


def review_plan_draft(draft: PlanDraft, evidence: EvidenceBundle, brief: Any) -> tuple[List[str], List[str]]:
    """Check a brief-driven plan against the request and the evidence.

    Args:
        draft: Plan returned by the planner LLM.
        evidence: Evidence available to the planner.
        brief: Planning brief (``PlanBrief``) describing the request.

    Returns:
        ``(critical, repairable)`` issue lists. Critical issues make the
        draft unusable (no stop matches the evidence). Repairable issues are
        sent back to the model once.
    """
    critical: List[str] = []
    repairable: List[str] = []
    if not draft.blocks:
        return ["the plan has no stops"], []
    today = lisbon_now().date()
    day_of_plan = plan_date(brief, today)
    if not draft.direct_answer:
        repairable.append("direct_answer is empty")

    matched: List[tuple[Any, Any]] = []
    # Named neighbourhoods (not whole municipalities such as Cascais or Sintra,
    # whose sights are spread over several kilometres).
    from agent.utils.geographic_scope import extract_aml_municipality_mentions

    neighbourhood_areas = [
        area for area in brief.activity_areas if not extract_aml_municipality_mentions(area)
    ]
    for block in draft.blocks:
        card = match_block_to_card(block.title, evidence)
        if card is None:
            repairable.append(f'"{strip_block_time(block.title)[:60]}" is not an evidence card title; use a card title exactly or drop the block')
            continue
        matched.append((block, card))
        distance = _card_distance_km(card)
        # A sight is never replaced by a restaurant, nor a meal by a sight.
        is_meal = card_matches_kind(card, "restaurant")
        if brief.activity_areas and distance is not None and distance > _AREA_RADIUS_KM:
            # A stop of the same type inside the area, or clearly closer to it.
            closer = sorted(
                (
                    other for other in evidence.cards
                    if other is not card
                    and getattr(other, "kind", "") not in {"weather", "transport"}
                    and (_card_distance_km(other) or 99.0) <= max(_AREA_RADIUS_KM, distance - 0.5)
                    and card_matches_kind(other, _block_component_kind(block, card))
                    and card_matches_kind(other, "restaurant") == is_meal
                    and all(match_block_to_card(other_block.title, evidence) is not other for other_block in draft.blocks)
                ),
                key=lambda other: _card_distance_km(other) or 99.0,
            )
            if closer:
                repairable.append(
                    f'"{card.title[:60]}" is {distance:.1f} km from {", ".join(brief.activity_areas)}; '
                    f'use a closer stop of the same type, such as "{closer[0].title[:60]}"'
                )
        elif neighbourhood_areas and distance is not None and distance > _OUTSIDE_NEIGHBOURHOOD_KM:
            # A neighbourhood is walking scale: a viewpoint 1.1 km away is in
            # another neighbourhood when one of the same type is 400 m away.
            inside = sorted(
                (
                    other for other in evidence.cards
                    if other is not card
                    and getattr(other, "kind", "") not in {"weather", "transport"}
                    and (_card_distance_km(other) or 99.0) <= _INSIDE_NEIGHBOURHOOD_KM
                    and card_matches_kind(other, _block_component_kind(block, card))
                    and card_matches_kind(other, "restaurant") == is_meal
                    and all(match_block_to_card(other_block.title, evidence) is not other for other_block in draft.blocks)
                ),
                key=lambda other: _card_distance_km(other) or 99.0,
            )
            if inside:
                repairable.append(
                    f'"{card.title[:60]}" is {distance:.1f} km from {", ".join(neighbourhood_areas)}, outside the area; '
                    f'use a stop of the same type inside it, such as "{inside[0].title[:60]}"'
                )
        if brief.time_window and not _BLOCK_TIME_PREFIX_RE.match(block.title):
            repairable.append(f'block "{strip_block_time(block.title)[:40]}" has no start time')
        start_minutes = _extract_block_start_minutes(block.title)
        block_day = day_of_plan + timedelta(days=max(0, int(getattr(block, "day", 0) or 1) - 1))
        hours_text = hours_for_plan_day(str((getattr(card, "fields", {}) or {}).get("Hours", "")), block_day, today)
        if start_minutes is not None and closed_all_day(hours_text):
            day_label = f"on day {block.day}" if brief.days > 1 and getattr(block, "day", 0) else "that day"
            other_day = "move it to a day when it is open or choose another card" if brief.days > 1 else "choose another card"
            repairable.append(f'"{card.title[:60]}" is closed all day {day_label} (hours: {hours_text[:40]}); {other_day}')
        elif start_minutes is not None and open_at(hours_text, start_minutes) is False:
            repairable.append(
                f'"{card.title[:60]}" is closed at {block.title[:5].strip()} (hours: {hours_text[:40]}); move it inside its hours or choose another card'
            )
        elif start_minutes is not None and getattr(card, "kind", "") == "event":
            session_issue = _event_session_issue(block, card, start_minutes)
            if session_issue:
                repairable.append(session_issue)
        elif start_minutes is not None:
            # A museum reached at 17:58 that closes at 18:00 is no visit; the
            # real legs then push it past closing.
            visit = min(block.stay_minutes or DEFAULT_STAY_MINUTES.get(block.kind, 45), 30)
            if open_at(hours_text, min(start_minutes + visit, 24 * 60 - 1)) is False:
                repairable.append(
                    f'"{card.title[:60]}" closes too soon after {block.title[:5].strip()} (hours: {hours_text[:40]}); '
                    "move it earlier inside its hours or choose another card"
                )

    if not matched:
        critical.append("no stop matches an evidence card")
    # The same card twice (the castle on both days, one restaurant for two
    # meals): the copy is dropped later and leaves a hole in the day.
    repeated = list(dict.fromkeys(card.title for index, (_block, card) in enumerate(matched) if any(card is other for _b, other in matched[:index])))
    if repeated:
        repairable.append(
            f'"{repeated[0][:60]}" is used for more than one stop; each card is one stop, so use a different card for the repeat'
        )
    direct_issue = _direct_answer_issue(draft, matched, brief, evidence)
    if direct_issue:
        repairable.append(direct_issue)
    short_day_issue = _short_day_issue(draft, matched, evidence, brief) if matched else ""
    if short_day_issue:
        repairable.append(short_day_issue)
    past_start_issue = _past_start_issue(draft, brief)
    if past_start_issue:
        repairable.append(past_start_issue)
    for issue in (
        _late_start_issue(draft, brief),
        _late_full_day_start_issue(draft, brief),
        _long_gap_issue(matched, brief, evidence),
        _landmark_issue(matched, evidence, brief),
        _same_site_issue(matched, evidence),
        _event_meal_distance_issue(matched, evidence),
    ):
        if issue:
            repairable.append(issue)

    for kind, count in requested_counts(brief).items():
        if kind in {"attraction", "walk", "event"}:
            continue
        found = sum(1 for block, card in matched if card_matches_kind(card, kind))
        required = count or 1
        cards_available = [
            card for card in evidence.cards
            if getattr(card, "kind", "") not in {"weather", "transport"}
            and card_matches_kind(card, kind)
        ]
        if found < required and len(cards_available) > found:
            repairable.append(
                f"the request asks for {required} {kind} stop(s) but the plan has {found}; add one from the matching cards"
            )
        if count and kind not in {"lunch", "dinner", "restaurant"} and found > count:
            repairable.append(
                f"the request asks for {count} {kind} stop(s) but the plan has {found}; "
                f"keep {count} and use other kinds of stops for the rest of the time"
            )

    for component in brief.components:
        detail = normalize_text(component.detail)
        topic = next((name for name, pattern in _DETAIL_TOPICS.items() if pattern.search(detail)), "")
        if not topic:
            continue
        pattern = _DETAIL_TOPICS[topic]
        if any(pattern.search(_card_text(card)) for _block, card in matched):
            continue
        candidates = [
            card for card in evidence.cards
            if getattr(card, "kind", "") not in {"weather", "transport"} and pattern.search(_card_text(card))
        ]
        if candidates:
            repairable.append(f'no stop is a {component.detail}; use "{candidates[0].title[:60]}"')
        else:
            repairable.append(
                f"no {component.detail} was found; do not substitute another kind of place, "
                f"drop that stop and add a limitation saying no {component.detail} was found in the area"
            )

    sunset_issue = _sunset_viewpoint_issue(draft, brief)
    if sunset_issue:
        repairable.append(sunset_issue)
    zigzag_issue = _zigzag_issue(matched) or _spread_issue(matched, evidence, brief) or _visit_order_issue(matched)
    if zigzag_issue:
        repairable.append(zigzag_issue)
    repairable.extend(_missing_meal_issues(matched, evidence, brief))
    repairable.extend(_schedule_travel_issues(matched))
    window_issue = _time_window_issue(draft, brief)
    if window_issue:
        repairable.append(window_issue)

    if brief.origin and brief.transport_mode != "walking" and not draft.movement_logic:
        repairable.append("movement_logic must start with the leg from the start point to the first stop")
    weather_cards = [card for card in evidence.cards if card.kind == "weather"]
    rain_issue = _rain_claim_issue(draft, weather_cards)
    if rain_issue:
        repairable.append(rain_issue)
    if weather_cards and (brief.weather_conditional or brief.day or brief.time_window) and not draft.weather_strategy:
        repairable.append("weather_strategy is empty although there is weather evidence")

    texts = [draft.title, draft.direct_answer, *draft.movement_logic, *draft.weather_strategy, *draft.tips, *draft.limitations]
    for block in draft.blocks:
        texts.extend([block.purpose, *block.limitations, *block.weather])
    if any(has_internal_wording(text) for text in texts):
        repairable.append("remove internal words (tools, agents, cards, evidence, searches) from the text")
    return list(dict.fromkeys(critical)), list(dict.fromkeys(repairable))
