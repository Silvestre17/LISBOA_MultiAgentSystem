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


PLACEHOLDER_RE = re.compile(r"\b(?:N/?A|unknown|not available|not provided|TBD|\+ info|null|none)\b", re.IGNORECASE)
RAW_FIELD_RE = re.compile(r"\b(?:Location|Address|Website|Phone|Category|Description|Morada|Telefone|Categoria|Descrição)\s*:", re.IGNORECASE)


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
    if len(draft.blocks) > 5:
        issues.append("too many plan blocks")

    user_norm = normalize_text(user_message)
    evidence_titles = [normalize_text(card.title) for card in evidence.cards if card.title]
    evidence_text = normalize_text(" ".join([card.title + " " + card.summary for card in evidence.cards]))

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
        if (
            matched_card
            and _query_requests_time_specific_visit(user_message)
            and _evidence_card_is_closed(matched_card)
        ):
            issues.append(f"time-specific plan selected closed venue: {block.title[:60]}")
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

    if _query_requests_public_transport(user_message):
        movement_text = normalize_text(" ".join([*draft.movement_logic, *[" ".join(block.movement) for block in draft.blocks]]))
        if not re.search(r"\b(?:metro|carris|cp|bus|tram|train|line|linha|route|rota|stop|station|paragem|estacao|unconfirmed|nao confirmad|não confirmad)\b", movement_text):
            issues.append("public transport requested but movement logic is too vague")

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


def _query_requests_time_specific_visit(user_message: str) -> bool:
    """Return whether a plan implies a time window where closures matter."""
    normalized = normalize_text(user_message)
    return bool(
        re.search(
            r"\b(evening|tonight|night|afternoon|morning|today|tomorrow|noite|esta noite|fim de tarde|tarde|manha|manhã|hoje|amanha|amanhã)\b",
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
