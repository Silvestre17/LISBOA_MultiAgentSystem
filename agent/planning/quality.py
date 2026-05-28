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
from typing import Any, List, Sequence

from agent.planning.evidence import EvidenceBundle, normalize_text
from agent.planning.models import PlanDraft


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
