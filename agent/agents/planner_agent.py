# ==========================================================================
# Master Thesis - Planner Agent
#   - André Filipe Gomes Silvestre, 20240502
#
#   Itinerary synthesis agent. Combines outputs from other agents
#   into coherent travel plans.
# ==========================================================================

import re
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agent.agents.base import BaseAgent, clean_response
from agent.prompts.planner import get_planner_prompt
from agent.utils.langsmith_tracing import traceable
from agent.utils.response_formatter import (
    final_post_qa_guard,
    infer_response_language,
)
from agent.planning import (
    build_evidence_bundle,
    build_structured_plan_messages,
    EvidenceBundle,
    EvidenceCard,
    PlanBlock,
    PlanDraft,
    parse_plan_draft_json,
    render_plan_markdown,
    validate_plan_draft,
)

_PLANNER_FIELD_LABELS = {
    "brief description",
    "description",
    "address",
    "location",
    "opening hours",
    "tip",
    "quick tip",
    "price",
    "prices",
    "source",
    "fonte",
    "updated",
    "atualizado",
    "conditions",
    "final notes",
    "dicas finais",
    "weather data",
    "places & attractions",
    "events",
    "transport info",
    "data limitations",
}
_PLANNER_FIELD_LABEL_WORDS = {
    "address",
    "morada",
    "category",
    "categoria",
    "phone",
    "telefone",
    "website",
    "site",
    "rating",
    "price",
    "preco",
    "preço",
    "source",
    "fonte",
    "updated",
    "atualizado",
    "hours",
    "horario",
    "horário",
}
_PLANNER_INVALID_PLACE_NAME_WORDS = {
    "transport",
    "transportes",
    "metro",
    "carris",
    "cp",
    "train",
    "trains",
    "comboios",
    "bus",
    "buses",
    "autocarro",
    "line",
    "linha",
    "rossio",
    "route",
    "rota",
}
_PLANNER_GENERIC_ACTIVITY_TERMS = (
    "lunch",
    "dinner",
    "breakfast",
    "coffee",
    "break",
    "return",
    "free time",
    "walk",
    "transfer",
    "transport",
)
_PLANNER_PLACE_HINTS = (
    "museum",
    "museu",
    "monastery",
    "mosteiro",
    "castle",
    "castelo",
    "aqueduct",
    "lighthouse",
    "pavilion",
    "pavilh",
    "monument",
    "society",
    "science",
    "sport",
    "geographical",
    "cemetery",
    "maat",
    "mude",
    "gulbenkian",
    "berardo",
    "tower",
    "torre",
)
_PLANNER_ACCESSIBILITY_RE = re.compile(
    r"\b(wheelchair|step[- ]?free|accessible|accessibility|elevator|lift|accessible restroom|cadeira de rodas|acess[ií]vel|elevador|wc adaptado|mobilidade reduzida|curb[- ]?cut)\b",
    re.IGNORECASE,
)
_PLANNER_SOURCE_LINE_RE = re.compile(
    r"^(?:[-*•]\s*)?(?:📌\s*)?(?:\*\*)?(?:Fontes?|Sources?)(?:\*\*)?:.*$",
    re.IGNORECASE,
)


def _normalize_planner_text(text: str) -> str:
    """Normalizes planner text for robust grounding comparisons."""
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    normalized = re.sub(r"[^a-zA-Z0-9\s/-]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip().lower()


def _sanitize_planner_place_name(candidate: str) -> str:
    """Return a clean place name, or an empty string for labels/fragments."""
    cleaned = re.sub(r"\s+", " ", str(candidate or "")).strip().strip("-–—: ")
    cleaned = re.sub(r"^\d{1,2}(?::\d{2})?\s*[·.\-–—]\s*", "", cleaned).strip()
    cleaned = re.sub(
        r"^(?:morning|afternoon|evening|manhã|tarde|noite)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    normalized = _normalize_planner_text(cleaned)
    words = set(normalized.split())
    if (
        not normalized
        or normalized in _PLANNER_FIELD_LABELS
        or normalized.isdigit()
        or re.fullmatch(r"\d+\.?", normalized)
        or words.intersection(_PLANNER_FIELD_LABEL_WORDS)
    ):
        return ""
    if words.intersection(_PLANNER_INVALID_PLACE_NAME_WORDS) and not re.search(
        r"\b(?:museum|museu|monument|monumento|gallery|galeria|palace|palacio|palácio|garden|jardim|centre|centro)\b",
        normalized,
    ):
        return ""
    if len(normalized.split()) > 12:
        return ""
    return cleaned


def _extract_allowed_place_names(text: str) -> List[str]:
    """Extracts grounded POI names from researcher/event outputs."""
    if not text:
        return []

    candidates: List[str] = []
    seen = set()

    for line in text.splitlines():
        for match in re.findall(r"\*\*([^*]+)\*\*", line):
            candidate = _sanitize_planner_place_name(match)
            normalized = _normalize_planner_text(candidate)
            if not normalized:
                continue
            if len(normalized.split()) > 12:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(candidate)

    return candidates


def _query_requests_accessibility(user_message: str) -> bool:
    """Detects whether the user asked for accessibility support."""
    return bool(
        re.search(
            r"\b(wheelchair|accessible|accessibility|step[- ]?free|mobility|reduced mobility|cadeira de rodas|acess[ií]vel|mobilidade reduzida)\b",
            user_message or "",
            re.IGNORECASE,
        )
    )


def _query_requests_public_transport(user_message: str) -> bool:
    """Detect whether the user explicitly wants public-transport planning."""
    normalized = _normalize_planner_text(user_message)
    return bool(
        re.search(
            r"\b(public transport|transportes publicos|metro|bus|autocarro|comboio|train|tram|eletrico|route|rota|how do i get|como vou|como chego)\b",
            normalized,
            re.IGNORECASE,
        )
    )


def _build_public_transport_synthesis_instruction(
    user_message: str,
    transport_data: str,
) -> str:
    """Build a stricter synthesis contract for itinerary requests with transport evidence."""
    if not _query_requests_public_transport(user_message) or not str(transport_data or "").strip():
        return ""

    return (
        "PUBLIC TRANSPORT SYNTHESIS CONTRACT:\n"
        "- The user asked for public transport and transport context is available.\n"
        "- For every movement you include, provide concrete evidence-supported details from the transport context: "
        "operator, line or mode, direction when available, and board/alight or transfer point when available.\n"
        "- If the provided context does not confirm a specific leg, write exactly which leg is unconfirmed.\n"
        "- Do not replace missing route details with vague prose such as 'continue by public transport', "
        "'verify locally', 'use the exact street address', or 'check the most direct connection locally'.\n"
        "- For future-day itineraries, do not present live next departures captured now as tomorrow's times."
    )


def _extract_requested_day_count(user_message: str) -> Optional[int]:
    """Extract the requested itinerary length in days when it is explicit.

    Args:
        user_message: Original planner request.

    Returns:
        Optional[int]: Requested number of days, when it can be inferred.
    """
    query = (user_message or "").lower()
    explicit_match = re.search(r"\b([2-7])\s*(?:day|days|dia|dias)\b", query)
    if explicit_match:
        return int(explicit_match.group(1))

    word_to_days = {
        "two days": 2,
        "three days": 3,
        "four days": 4,
        "five days": 5,
        "six days": 6,
        "seven days": 7,
        "dois dias": 2,
        "três dias": 3,
        "tres dias": 3,
        "quatro dias": 4,
        "cinco dias": 5,
        "seis dias": 6,
        "sete dias": 7,
        "weekend": 2,
        "fim de semana": 2,
    }
    for phrase, count in word_to_days.items():
        if phrase in query:
            return count

    return None


def _build_multi_day_planner_instruction(language: str, requested_days: Optional[int]) -> str:
    """Build a quality-focused instruction for multi-day itinerary requests.

    Args:
        language: Output language code.
        requested_days: Explicit day count, if detected.

    Returns:
        str: Additional planner instruction, or an empty string.
    """
    if not requested_days or requested_days <= 1:
        return ""

    visible_days = min(requested_days, 5)
    if language == "pt":
        return (
            f"MULTI-DAY QUALITY MODE: o pedido é para {requested_days} dias. "
            f"Entrega um plano dia-a-dia para até {visible_days} dias, com detalhe suficiente para ser útil. "
            "Cada dia deve ter uma zona principal, 2-4 paragens suportadas pelos dados quando existirem, "
            "uma opção interior/backup, lógica de deslocação e limites explícitos. "
            "Se o pedido exceder 5 dias, limita a resposta aos primeiros 5 dias e explica brevemente porquê. "
            "Não inventes horários, preços, reservas, acessibilidade ou tempos em tempo real para datas futuras."
        )

    return (
        f"MULTI-DAY QUALITY MODE: the request covers {requested_days} days. "
        f"Deliver a day-by-day plan for up to {visible_days} days, detailed enough to be useful. "
        "Each day must have one main area, 2-4 evidence-supported stops when the data supports them, "
        "one indoor/backup option, movement logic, and explicit limits. "
        "If the request exceeds 5 days, limit the answer to the first 5 days and briefly explain why. "
        "Do not invent opening hours, prices, bookings, accessibility details, or future live transport times."
    )


def _build_multi_day_follow_up_note(language: str, requested_days: Optional[int]) -> Optional[str]:
    """Build a short follow-up note for multi-day planner fallbacks.

    Args:
        language: Output language code.
        requested_days: Explicit day count, if detected.

    Returns:
        Optional[str]: Follow-up note, when relevant.
    """
    if not requested_days or requested_days <= 1:
        return None

    if requested_days > 5:
        if language == "pt":
            return (
                "Para manter a qualidade, o plano fica limitado aos primeiros 5 dias; "
                "dias adicionais devem ser validados numa nova iteração."
            )
        return (
            "To preserve quality, the plan is limited to the first 5 days; "
            "additional days should be validated in a separate iteration."
        )

    if language == "pt":
        return (
            "Este é um framework de planeamento, não uma agenda fechada: horários, bilhetes, "
            "reservas, acessibilidade e transportes futuros devem ser confirmados por dia."
        )

    return (
        "This is a planning framework, not a locked schedule: opening hours, tickets, "
        "bookings, accessibility, and future transport should be confirmed day by day."
    )


def enforce_multi_day_quality_mode(response: str, user_message: str, language: str) -> str:
    """Preserve useful bounded multi-day plans and annotate thin one-day drafts.

    Args:
        response: Planner draft or repaired final response.
        user_message: Original user request.
        language: Output language code.

    Returns:
        str: Multi-day response, with a follow-up note only when needed.
    """
    requested_days = _extract_requested_day_count(user_message)
    if not requested_days or requested_days <= 1:
        return response

    normalized_response = str(response or "").lower()
    has_later_day_sections = bool(
        re.search(
            r"(?mi)^\s*(?:#{1,6}\s*)?(?:[-*]\s*)?(?:📍\s*)?(?:\*\*)?(?:day|dia)\s+2\b",
            str(response or ""),
        )
    )
    if any(
        marker in normalized_response
        for marker in (
            "5-day lisbon itinerary",
            "first 5 days in lisbon",
            "plano de 5 dias em lisboa",
            "primeiros 5 dias em lisboa",
            "plano turístico de **",
            "i’m limiting the",
            "i'm limiting the",
            "vou limitar o pedido",
            "multi-day planning framework",
            "first 5 days planning framework",
            "framework de planeamento multi-dia",
            "framework dos primeiros 5 dias",
            "planning framework",
            "framework de planeamento",
        )
    ) or has_later_day_sections:
        return response

    lines = str(response or "").splitlines()
    normalized_lines = list(lines)

    follow_up_note = _build_multi_day_follow_up_note(language, requested_days)
    if follow_up_note and follow_up_note not in "\n".join(normalized_lines):
        normalized_lines.extend(["", f"- {follow_up_note}"])

    return "\n".join(normalized_lines).strip()


def _context_has_accessibility_data(*texts: str) -> bool:
    """Returns whether accessibility details are explicitly present in context."""
    return any(_PLANNER_ACCESSIBILITY_RE.search(text or "") for text in texts)


def _clean_activity_title(title: str) -> str:
    """Removes itinerary prefixes from activity titles before validation."""
    cleaned = re.sub(
        r"^(start|optional(?:,\s*time-permitting)?|opcional(?:,\s*se houver tempo)?)\s*:\s*",
        "",
        title.strip(),
        flags=re.IGNORECASE,
    )
    return cleaned.strip(" -–—")


def _matches_allowed_place(activity_title: str, allowed_places: List[str]) -> bool:
    """Checks whether an activity title matches one of the allowed POIs."""
    normalized_activity = _normalize_planner_text(activity_title)
    if not normalized_activity:
        return True

    for place in allowed_places:
        normalized_place = _normalize_planner_text(place)
        if not normalized_place:
            continue
        if normalized_place in normalized_activity or normalized_activity in normalized_place:
            return True

    return False


def _find_planner_grounding_issues(
    response: str,
    allowed_places: List[str],
    accessibility_requested: bool,
    accessibility_confirmed: bool,
) -> List[str]:
    """Finds unsupported venue or accessibility claims in planner drafts."""
    issues: List[str] = []

    activity_lines = re.findall(r"^🕐.*?-\s*\*\*(.+?)\*\*", response or "", flags=re.MULTILINE)
    for raw_title in activity_lines:
        title = _clean_activity_title(raw_title)
        normalized_title = _normalize_planner_text(title)
        if not normalized_title:
            continue
        if any(term in normalized_title for term in _PLANNER_GENERIC_ACTIVITY_TERMS):
            continue
        if allowed_places and any(term in normalized_title for term in _PLANNER_PLACE_HINTS):
            if not _matches_allowed_place(title, allowed_places):
                issues.append(f"Unsupported venue mentioned: {title}")

    if accessibility_requested and not accessibility_confirmed and _PLANNER_ACCESSIBILITY_RE.search(response or ""):
        issues.append(
            "Accessibility details were claimed without explicit confirmation in the provided data."
        )

    return issues


def _build_planner_grounding_message(
    allowed_places: List[str],
    accessibility_requested: bool,
    accessibility_confirmed: bool,
) -> str:
    """Builds a strict grounding note for planner synthesis."""
    rules = [
        "GROUNDING RULES:",
        "- You MUST stay supported by the provided agent data.",
        "- Do NOT mention any venue, museum, restaurant, or landmark unless it appears in the provided data.",
        "- If data is missing, say it is not confirmed instead of filling the gap from general knowledge.",
    ]

    if allowed_places:
        rules.append("- Allowed venue names: " + "; ".join(allowed_places[:15]))
        rules.append("- Any venue name not in the allowed list above is forbidden.")

    if accessibility_requested and not accessibility_confirmed:
        rules.append(
            "- Accessibility was requested, but the provided data does NOT confirm wheelchair access. "
            "Do NOT claim step-free access, elevators, accessible toilets, or wheelchair-friendly facilities. "
            "State clearly that accessibility must be confirmed with the official venue/operator."
        )

    return "\n".join(rules)


def _compact_planner_context_block(
    heading: str,
    content: str,
    *,
    max_lines: int = 18,
    max_chars: int = 1400,
) -> str:
    """Trim agent context before planner synthesis to keep the prompt compact and stable."""
    if not content:
        return ""

    kept_lines: List[str] = []
    char_count = 0
    for raw_line in str(content).splitlines():
        stripped = raw_line.strip()
        if not stripped or _PLANNER_SOURCE_LINE_RE.match(stripped):
            continue

        kept_lines.append(raw_line.rstrip())
        char_count += len(raw_line)
        if len(kept_lines) >= max_lines or char_count >= max_chars:
            break

    compact_body = "\n".join(kept_lines).strip()
    if not compact_body:
        return ""
    return f"{heading}\n{compact_body}"


def _build_planner_evidence_packet(
    *,
    user_message: str,
    language: str,
    weather_data: str,
    transport_data: str,
    places_data: str,
    events_data: str,
    qa_disclaimers: list[str] | None,
    conversation_context: str,
) -> str:
    """Build a structured evidence summary for dynamic planner synthesis.

    The packet keeps the LLM path evidence-supported without turning common
    planning requests into static templates. It intentionally summarizes
    specialized-agent evidence and limitations; deterministic builders remain a
    last-resort fallback.
    """
    is_pt = language == "pt"
    requested_days = _extract_requested_day_count(user_message)
    allowed_places = _extract_allowed_place_names("\n".join([places_data or "", events_data or ""]))
    weather_bullets = _extract_weather_fact_bullets(weather_data, language, max_items=5)
    transport_bullets = _extract_planner_fallback_bullets(transport_data, max_items=6)
    place_bullets = _extract_planner_fallback_bullets(places_data, max_items=8)
    event_bullets = _extract_planner_fallback_bullets(events_data, max_items=6)

    def _section(title: str, bullets: List[str], fallback: str) -> str:
        body = "\n".join(bullets) if bullets else f"- {fallback}"
        return f"### {title}\n{body}"

    day_line = (
        f"- **Dias pedidos:** {requested_days or 1}"
        if is_pt
        else f"- **Requested days:** {requested_days or 1}"
    )
    continuity_line = (
        "- **Continuidade:** usa o contexto anterior para evitar repetir zonas/paragens quando possível."
        if is_pt and conversation_context.strip()
        else "- **Continuity:** use prior context to avoid repeating areas/stops when possible."
        if conversation_context.strip()
        else ""
    )
    place_names = "; ".join(allowed_places[:12])

    sections = [
        "## Evidence Packet for Dynamic Planning" if not is_pt else "## Pacote de Evidência para Planeamento Dinâmico",
        "### User Constraints" if not is_pt else "### Restrições do Utilizador",
        "\n".join(line for line in [day_line, continuity_line] if line).strip(),
        _section(
            "Weather Facts" if not is_pt else "Factos Meteorológicos",
            weather_bullets,
            "Weather was not confirmed in the available evidence." if not is_pt else "A meteorologia não foi confirmada na evidência disponível.",
        ),
        _section(
            "Transport Evidence" if not is_pt else "Evidência de Transportes",
            transport_bullets,
            "Transport was not confirmed; do not invent lines, stops, durations, or live departures." if not is_pt else "Os transportes não foram confirmados; não inventes linhas, paragens, durações ou partidas em tempo real.",
        ),
        _section(
            "Evidence-Supported Places" if not is_pt else "Locais Suportados por Evidência",
            place_bullets,
            "No concrete place cards were provided; ask for confirmation or give a scoped limitation instead of inventing venues." if not is_pt else "Não foram fornecidos cartões de locais concretos; assume uma limitação delimitada em vez de inventar locais.",
        ),
    ]

    if place_names:
        sections.append(
            ("### Allowed Venue Names\n- " if not is_pt else "### Nomes de Locais Permitidos\n- ")
            + place_names
        )
    if event_bullets:
        sections.append(
            _section(
                "Evidence-Supported Events" if not is_pt else "Eventos Suportados por Evidência",
                event_bullets,
                "",
            )
        )
    if qa_disclaimers:
        limitation_title = "Data Caveats" if not is_pt else "Cautelas dos Dados"
        sections.append(
            f"### {limitation_title}\n" + "\n".join(f"- {item}" for item in qa_disclaimers)
        )

    return "\n\n".join(section for section in sections if section.strip())


def _extract_plan_constraints(user_message: str, conversation_context: str, language: str) -> List[str]:
    """Extract reusable itinerary constraints from the current and previous turns."""
    combined = _normalize_planner_text(f"{conversation_context}\n{user_message}")
    is_pt = language == "pt"
    constraints: List[str] = []

    def add(pt: str, en: str) -> None:
        value = pt if is_pt else en
        if value not in constraints:
            constraints.append(value)

    if re.search(r"\b(rain|rainy|chuva|guarda chuva|indoor|interior|covered|coberto)\b", combined):
        add("plano seguro para chuva/interior", "rain-safe or indoor backup")
    if re.search(r"\b(low walking|low walk|little walking|pouca caminhada|baixo declive|sem grandes caminhadas|reduced mobility)\b", combined):
        add("pouca caminhada", "low walking")
    if re.search(r"\b(public transport|transportes publicos|metro|carris|cp|bus|tram|autocarro|comboio)\b", combined):
        add("usar transporte público", "use public transport")
    if re.search(r"\b(cheap|cheaper|budget|barato|barata|baixo custo|econ[oó]mico)\b", combined):
        add("manter custos baixos", "keep costs low")
    avoid_match = re.search(
        r"\b(?:do not repeat|avoid|não repetir|nao repetir|evitar)\s+(?P<areas>[A-Za-zÀ-ÿ\s,/&-]+)",
        user_message,
        flags=re.IGNORECASE,
    )
    if avoid_match:
        areas = re.sub(r"\s+", " ", avoid_match.group("areas")).strip(" .?!,;:")
        add(f"evitar {areas}", f"avoid {areas}")
    return constraints[:6]


def _planner_response_matches_schema(response: str) -> bool:
    """Return whether a planner draft follows the required safe plan schema."""
    if not response:
        return False
    headings = [line.strip() for line in response.splitlines() if line.strip().startswith("### ")]
    normalized_headings = [_normalize_planner_text(heading) for heading in headings]
    if len(headings) < 3 or len(normalized_headings) != len(set(normalized_headings)):
        return False
    if re.search(r"(?im)^\s*(?:place cards?|raw cards?|visit suggestions)\s*$", response):
        return False
    if re.search(r"(?mi)^\s*[-*]\s*[🔹•-]?\s*\*\*(?:Category|Description|Location|Transport Note|Area|Why This Day|Categoria|Descrição|Localização)\*\*\s*:", response):
        return False
    if len(re.findall(r"(?im)^\s*(?:[-*]\s*)?📌\s*\*\*(?:Source|Sources|Fonte|Fontes):\*\*", response)) > 1:
        return False
    if re.search(
        r"(?mi)^\s*[-*]\s*(?:[^\w\s]\s*)?\*\*(?:Location|Address|Website|Phone|Morada|Telefone)\*\*\s*:",
        response,
    ):
        return False
    normalized = _normalize_planner_text(response)
    required_en = (
        "direct answer",
        "plan basis",
        "suggested route",
        "how to move",
        "weather adaptation",
        "final notes",
    )
    legacy_required_en = (
        "direct answer",
        "constraints used",
        "plan blocks",
        "movement logic",
        "weather strategy",
        "limitations",
    )
    has_english_schema = all(section in normalized for section in required_en) or all(
        section in normalized for section in legacy_required_en
    ) or all(
        section in normalized
        for section in ("direct answer", "suggested route", "how to move", "final notes")
    )
    has_portuguese_schema = all(
        any(alias in normalized for alias in aliases)
        for aliases in (
            ("resposta direta",),
            ("base do plano",),
            ("roteiro sugerido",),
            ("como te deslocas",),
            ("adaptação ao tempo", "adaptacao ao tempo"),
            ("notas finais",),
        )
    ) or all(
        any(alias in normalized for alias in aliases)
        for aliases in (
            ("resposta direta",),
            ("restrições usadas", "restricoes usadas"),
            ("blocos do plano",),
            ("lógica de movimento", "logica de movimento"),
            ("estratégia meteorológica", "estrategia meteorologica"),
            ("limitações", "limitacoes"),
        )
    ) or all(
        any(alias in normalized for alias in aliases)
        for aliases in (
            ("resposta direta",),
            ("roteiro sugerido",),
            ("como te deslocas",),
            ("notas finais",),
        )
    )
    if not (has_english_schema or has_portuguese_schema):
        return False

    unsafe_patterns = (
        r"\bMuseum:\*\*",
        r"\bRestaurant:\*\*",
        r"\bEvent:\*\*",
        r"\bguaranteed\b",
        r"\bbooked\b",
        r"\breserved\b",
        r"\+ info\b",
        r"not provided",
    )
    if any(re.search(pattern, response, flags=re.IGNORECASE) for pattern in unsafe_patterns):
        return False

    for line in response.splitlines():
        if re.search(r"\bexact\s+(?:route|routes|price|prices|ticket|tickets|weather)\b", line, flags=re.IGNORECASE):
            if not re.search(r"\b(?:not|no|without|unconfirmed|do not|did not|não|nao|sem)\b", line, flags=re.IGNORECASE):
                return False
    return True


def _planner_response_has_minimum_user_value(response: str) -> bool:
    """Return whether a planner answer contains actionable content for the user.

    Planner responses can pass a heading-level schema check while still being
    effectively empty after LLM repair or Markdown normalization. This guard
    rejects those shells before they reach QA or the UI.
    """
    if not response or not response.strip():
        return False

    without_sources = re.sub(
        r"(?im)^\s*(?:[-*•]\s*)?(?:📌\s*)?\**(?:Fonte|Fontes|Source|Sources)\**\s*:.*$",
        "",
        response,
    )
    compact = re.sub(r"\s+", " ", without_sources).strip()
    if len(compact) < 180:
        return False

    normalized = _normalize_planner_text(without_sources)
    if re.search(r"\b(?:dados nao confirmados|data not confirmed)\b", normalized):
        return False

    headings = [
        _normalize_planner_text(line)
        for line in without_sources.splitlines()
        if line.strip().startswith("### ")
    ]
    useful_headings = [
        heading
        for heading in headings
        if not re.search(r"\b(?:fonte|source|sources|fontes)\b", heading)
    ]
    meaningful_bullets: list[str] = []
    item_cards: list[str] = []
    body_lines: list[str] = []

    for raw_line in without_sources.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped == "---" or stripped.startswith("### "):
            continue
        normalized_line = _normalize_planner_text(stripped)
        if re.search(r"\b(?:fonte|source|sources|fontes|resposta direta|direct answer)\b", normalized_line):
            continue
        body_lines.append(stripped)
        if re.match(r"^(?:[-*•]|\d+[.)])\s+", stripped):
            meaningful_bullets.append(stripped)
        elif re.match(r"^\*\*[^*\n]{3,120}\*\*\s*$", stripped):
            item_cards.append(stripped)

    if len(item_cards) >= 1 and (meaningful_bullets or len(body_lines) >= 3):
        return True
    if len(meaningful_bullets) >= 2 and len(useful_headings) >= 2:
        return True
    has_plan_section = any(
        re.search(
            r"\b(?:roteiro sugerido|suggested route|como te deslocas|how to move|adapta[cç]ao ao tempo|weather adaptation|notas finais|final notes)\b",
            heading,
        )
        for heading in useful_headings
    )
    return has_plan_section and len(useful_headings) >= 3 and len(body_lines) >= 3




def _is_oriente_station_nearby_request(user_message: str) -> bool:
    """Return whether Oriente should be treated as station/Parque das Nações locality."""
    normalized = _normalize_planner_text(user_message or "")
    if "oriente" not in normalized or "museu do oriente" in normalized:
        return False
    locality_signals = (
        "arrive", "chego", "chegar", "nearby", "perto", "near", "dinner",
        "jantar", "rain-safe", "cultural", "culture", "indoor", "interior",
        "parque das nacoes", "station", "estacao",
    )
    return any(signal in normalized for signal in locality_signals)


def _oriente_station_locality_cards(language: str) -> list[dict[str, str]]:
    """Return stable Oriente/Parque das Nações anchors without live claims."""
    if language == "pt":
        return [
            {
                "name": "Estação do Oriente / Parque das Nações",
                "category": "âncora local",
                "description": "Usa Oriente como base compacta para uma chegada ao fim do dia, sem atravessar a cidade.",
            },
            {
                "name": "Centro Vasco da Gama",
                "category": "base coberta para jantar",
                "description": "Opção prática e abrigada para procurar restauração sem assumir reserva ou horário específico.",
            },
            {
                "name": "Oceanário / Pavilhão do Conhecimento",
                "category": "referência cultural próxima",
                "description": "Trata como alternativa cultural a confirmar, sem afirmar disponibilidade ou horário de visita.",
            },
        ]
    return [
        {
            "name": "Oriente station / Parque das Nações",
            "category": "local anchor",
            "description": "Use Oriente as the compact arrival base instead of crossing the city.",
        },
        {
            "name": "Centro Vasco da Gama",
            "category": "covered dinner base",
            "description": "A practical sheltered base for finding food without assuming booking or opening-hour confirmation.",
        },
        {
            "name": "Oceanário / Pavilhão do Conhecimento",
            "category": "nearby cultural reference",
            "description": "Treat as a cultural backup to verify, without claiming availability or exact opening hours.",
        },
    ]


def _build_structured_plan_fallback(
    *,
    user_message: str,
    language: str,
    weather_data: str,
    transport_data: str,
    places_data: str,
    events_data: str,
    qa_disclaimers: list[str] | None,
    conversation_context: str = "",
) -> str:
    """Build a conservative itinerary from a small explicit plan schema.

    The fallback avoids prompt-specific templates. It derives duration,
    constraints, blocks, movement, weather strategy, and sources from the user
    request plus available worker evidence.
    """
    is_pt = language == "pt"
    requested_days = _extract_requested_day_count(user_message) or 1
    visible_days = min(requested_days, 5)
    constraints = _extract_plan_constraints(user_message, conversation_context, language)
    combined_places = "\n".join([places_data or "", events_data or ""])
    cards = _extract_visitlisboa_place_cards(combined_places, max_items=10)
    clean_cards = [
        {**card, "name": clean_name}
        for card in cards
        for clean_name in [_sanitize_planner_place_name(card.get("name", ""))]
        if clean_name
    ]
    if _is_oriente_station_nearby_request(user_message):
        clean_cards = [
            card for card in clean_cards
            if "museu do oriente" not in _normalize_planner_text(card.get("name", ""))
        ]
        clean_cards = _oriente_station_locality_cards(language) + clean_cards[:2]
    weather_bullets = _extract_weather_safety_bullets(weather_data, language)
    transport_bullets = _extract_planner_fallback_bullets(transport_data, max_items=4)
    source_line = _build_planner_fallback_source_line(
        language,
        weather_data,
        transport_data,
        places_data,
        events_data,
    )
    if source_line and "carris" not in _normalize_planner_text("\n".join(transport_bullets)):
        source_line = re.sub(r"\s*\|\s*\[\*Carris\*\]\(https://www\.carris\.pt\)", "", source_line)

    transport_limitation = (
        "A ligação entre zonas não ficou confirmada nos dados recolhidos; não inventei linhas, paragens, durações ou partidas."
        if is_pt
        else "The connection between areas was not confirmed in the gathered data; I did not invent lines, stops, durations, or departures."
    )

    route_target = _extract_requested_plan_area(user_message)
    route_origin = _extract_requested_plan_origin(user_message)
    if visible_days == 1 and route_target:
        title = (
            f"### 📅 **Fim de dia descontraído em {route_target}**"
            if is_pt
            else f"### 📅 **Relaxed evening around {route_target}**"
        )
    else:
        title = (
            f"### 📅 **Plano de Lisboa de {visible_days} dia{'s' if visible_days > 1 else ''}**"
            if is_pt
            else f"### 📅 **Lisbon {visible_days}-Day Plan**"
        )
    if requested_days > 5:
        direct = (
            "Vou limitar o pedido aos primeiros 5 dias para manter verificabilidade; rotas, preços, bilhetes, restaurantes e meteorologia futura não devem ser tratados como confirmados."
            if is_pt
            else "I’m limiting this to the first 5 days to keep it verifiable; routes, prices, tickets, restaurants, and future weather are not treated as confirmed."
        )
    elif visible_days > 1:
        direct = (
            "Segue um plano de alto nível por dias, com exemplos ancorados nos dados disponíveis e limitações explícitas."
            if is_pt
            else "Here is a high-level day-by-day plan with examples anchored in the available data and explicit limits."
        )
    else:
        if route_origin and route_target and transport_bullets:
            direct = (
                f"Usa a ligação de transporte público evidenciada a partir de **{route_origin}** para **{route_target}**, acrescenta uma paragem cultural confirmada e mantém a caminhada final curta."
                if is_pt
                else f"Use the evidenced public-transport leg from **{route_origin}** toward **{route_target}**, add one confirmed cultural stop, and keep the final walk short."
            )
        elif route_origin and route_target:
            direct = (
                f"Não ficou confirmada uma ligação concreta entre **{route_origin}** e **{route_target}**; deixo apenas os dados verificáveis recolhidos e a limitação explícita."
                if is_pt
                else f"A concrete connection between **{route_origin}** and **{route_target}** was not confirmed; I am keeping only the verified gathered data and the explicit limitation."
            )
        else:
            direct = (
                "Segue um plano curto e ordenado, sem inventar horários, preços ou disponibilidade."
                if is_pt
                else "Here is a short ordered plan without inventing opening hours, prices, or availability."
            )

    lines: List[str] = [
        title,
        "",
        f"✅ **{'Resposta direta' if is_pt else 'Direct answer'}:** {direct}",
    ]
    if constraints:
        lines.extend([
            "",
            f"💡 **{'Dicas' if is_pt else 'Tips'}:**",
            *[f"    - {constraint}" for constraint in constraints[:3]],
        ])
    lines.extend([
        "",
        ("### 📍 **Roteiro sugerido**" if clean_cards else "### 📍 **Dados confirmados para o plano**")
        if is_pt
        else ("### 📍 **Suggested Route**" if clean_cards else "### 📍 **Confirmed Planning Data**"),
    ])

    if visible_days == 1:
        block_count = 0
        for card in clean_cards[:3]:
            block_count += 1
            lines.extend(["", f"**🏛️ {card['name']}**"])
            if card.get("category"):
                lines.append(f"    - 📝 **{'Categoria' if is_pt else 'Category'}:** {card['category']}")
            if card.get("description"):
                description = _planner_card_description_for_language(card["description"], language)
                if description:
                    lines.append(f"    - 📝 **{'Nota' if is_pt else 'Note'}:** {description[:220].rstrip()}")
        if block_count == 0:
            lines.append(
                "- ⚠️ Não consegui confirmar locais concretos suficientes para publicar um roteiro fechado; mantém o plano como orientação e pede uma zona, tema ou ponto de partida para eu recolher evidência concreta."
                if is_pt
                else "- ⚠️ I could not confirm enough concrete places to publish a fixed itinerary; treat this as guidance and provide an area, theme, or starting point so I can gather concrete evidence."
            )
    else:
        themes_en = [
            "historic core with low walking",
            "riverfront or Belém-style heritage day",
            "modern Lisbon / Oriente-style indoor day",
            "museums and viewpoints with short transfers",
            "flexible rainy backup and cheap-food day",
        ]
        themes_pt = [
            "centro histórico com pouca caminhada",
            "frente ribeirinha ou património tipo Belém",
            "Lisboa moderna / Oriente com opção interior",
            "museus e miradouros com transferes curtos",
            "backup de chuva e comida económica",
        ]
        themes = themes_pt if is_pt else themes_en
        for day_index in range(visible_days):
            anchor = clean_cards[day_index]["name"] if day_index < len(clean_cards) else None
            label = f"Dia {day_index + 1}" if is_pt else f"Day {day_index + 1}"
            theme = themes[day_index % len(themes)]
            if anchor:
                lines.append(f"- **{label}:** {theme}; exemplo ancorado: **{anchor}**." if is_pt else f"- **{label}:** {theme}; evidenced example: **{anchor}**.")
            else:
                lines.append(f"- **{label}:** {theme}; escolher locais concretos só após confirmação." if is_pt else f"- **{label}:** {theme}; choose concrete venues only after confirmation.")

    if transport_bullets:
        lines.extend([
            "",
            "### 🚇 **Como te deslocas**" if is_pt else "### 🚇 **How to move**",
            *transport_bullets[:4],
        ])
    if weather_bullets:
        lines.extend([
            "",
            "### ⛅ **Adaptação ao tempo**" if is_pt else "### ⛅ **Weather adaptation**",
            *weather_bullets[:4],
        ])
    lines.extend([
        "",
        "### ⚠️ **Notas finais**" if is_pt else "### ⚠️ **Final notes**",
    ])
    limitations = [
        "Não confirmei horários, preços, bilhetes, reservas, lotação ou disponibilidade em tempo real."
        if is_pt
        else "I did not confirm opening hours, prices, tickets, bookings, crowding, or real-time availability.",
    ]
    if transport_bullets:
        limitations.append(
            "Não uses partidas em tempo real recolhidas agora como horário para planos futuros."
            if is_pt
            else "Do not use live departures captured now as a schedule for future plans."
        )
    elif _query_requests_public_transport(user_message) or route_origin or route_target:
        limitations.append(transport_limitation)
    if requested_days > 5:
        limitations.append(
            "Limitei a resposta aos primeiros 5 dias; os restantes dias devem ser planeados numa segunda iteração."
            if is_pt
            else "I limited the answer to the first 5 days; remaining days should be planned in a second iteration."
        )
    if qa_disclaimers:
        visible_transport = _normalize_planner_text("\n".join(transport_bullets))
        for item in qa_disclaimers[:3]:
            item_text = str(item or "").strip()
            if not item_text:
                continue
            normalized_item = _normalize_planner_text(item_text)
            if "carris" in normalized_item and "carris" not in visible_transport:
                continue
            limitations.append(item_text)
    lines.extend(f"- {item}" for item in limitations)
    if source_line:
        lines.extend(["", source_line])
    return "\n".join(lines).strip()


def _extract_requested_plan_area(user_message: str) -> str:
    """Extract a named target area from a short planning request."""
    text = str(user_message or "").strip()
    patterns = [
        r"\baround\s+(?P<area>[^,.;]+?)(?:\s+starting\b|\s+from\b|\s+with\b|$)",
        r"\bin\s+(?P<area>[^,.;]+?)(?:\s+starting\b|\s+from\b|\s+with\b|$)",
        r"\bà volta de\s+(?P<area>[^,.;]+?)(?:\s+a partir\b|\s+com\b|$)",
        r"\bem\s+(?P<area>[^,.;]+?)(?:\s+a partir\b|\s+com\b|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            area = re.sub(r"\s+", " ", match.group("area")).strip(" .:-")
            if 2 <= len(area) <= 80:
                return area
    return ""


def _extract_requested_plan_origin(user_message: str) -> str:
    """Extract an origin anchor from a short planning request."""
    text = str(user_message or "").strip()
    patterns = [
        r"\bstarting\s+from\s+(?P<origin>[^,.;]+?)(?:[,.;]?\s+with\b|[,.;]?\s+include\b|$)",
        r"\bfrom\s+(?P<origin>[^,.;]+?)\s+to\b",
        r"\ba partir de\s+(?P<origin>[^,.;]+?)(?:[,.;]?\s+com\b|[,.;]?\s+inclui\b|$)",
        r"\bde\s+(?P<origin>[^,.;]+?)\s+para\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            origin = re.sub(r"\s+", " ", match.group("origin")).strip(" .:-")
            if 2 <= len(origin) <= 80:
                return origin
    return ""


def _clean_planner_card_description(description: str) -> str:
    """Remove nested field labels from fallback place descriptions."""
    text = str(description or "").strip()
    text = re.sub(
        r"^(?:[-*•]\s*)?(?:📝\s*)?\*\*(?:Description|Descrição|Descricao|Note|Nota)\s*:?\*\*\s*:?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^(?:Description|Descrição|Descricao|Note|Nota)\s*:\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip(" -")


def _planner_card_description_for_language(description: str, language: str) -> str:
    """Return a cleaned card description with conservative PT localization."""
    text = _clean_planner_card_description(description)
    if language == "pt":
        fixed_translations = {
            "Typical Portuguese cuisine.": "Cozinha portuguesa típica.",
            "Traditional Portuguese cuisine.": "Cozinha portuguesa tradicional.",
            "Typical Portuguese food.": "Comida portuguesa típica.",
        }
        return fixed_translations.get(text, text)
    return text


def _extract_planner_fallback_bullets(text: str, *, max_items: int = 4) -> List[str]:
    """Extract compact user-facing bullets from worker outputs for deterministic planner fallback."""
    bullets: List[str] = []
    seen = set()
    for raw_line in str(text or "").splitlines():
        stripped = raw_line.strip()
        if (
            not stripped
            or stripped.startswith(("### ", "#### ", "---"))
            or _PLANNER_SOURCE_LINE_RE.match(stripped)
        ):
            continue

        if stripped.startswith(("- ", "* ", "• ")):
            candidate = re.sub(r"^[-*•]\s+", "", stripped)
        elif stripped.startswith(("⛅", "⚠️", "💡", "🚇", "🚌", "📍", "🏛️", "☕", "🌤️")):
            candidate = stripped
        else:
            continue
        candidate = re.sub(r"^#{1,6}\s*", "", candidate).strip()
        if re.fullmatch(r"(?:🚇\s*)?\*\*(?:best transport|melhor transporte)\*\*", candidate, flags=re.IGNORECASE):
            continue
        if re.fullmatch(
            r"[\U0001F300-\U0001FAFF\u2300-\u23FF\u2600-\u27BF\uFE0F\u200D\s]+",
            candidate,
        ):
            continue
        if re.fullmatch(
            r"(?:[\U0001F300-\U0001FAFF\u2300-\u23FF\u2600-\u27BF\uFE0F\u200D]+\s*)?\*\*[A-Za-zÀ-ÿ0-9 /'-]{2,70}:\*\*",
            candidate,
        ):
            continue

        normalized = _normalize_planner_text(candidate)
        if not normalized or normalized in seen:
            continue
        if re.match(
            r"^(?:description|descricao|descrição|category|categoria|address|morada|phone|telefone|website|site|source|fonte)\s*:",
            normalized,
        ):
            continue
        if re.search(r"\b(?:google\.com/maps|tel:|official website\]\(#|not available|n/a)\b", normalized):
            continue
        seen.add(normalized)
        bullets.append(f"- {candidate}")
        if len(bullets) >= max_items:
            break

    return bullets


def _extract_weather_fact_bullets(weather_data: str, language: str, *, max_items: int = 4) -> List[str]:
    """Extract compact weather facts from WeatherAgent/IPMA text."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", str(weather_data or ""))
    if not text.strip():
        return []

    is_pt = language == "pt"
    bullets: List[str] = []
    seen: set[str] = set()

    def _append(label_pt: str, label_en: str, value: str, icon: str = "") -> None:
        value = re.sub(r"\s+", " ", str(value or "")).strip(" .;")
        if not value:
            return
        label = label_pt if is_pt else label_en
        bullet = f"- {icon} **{label}:** {value}." if icon else f"- **{label}:** {value}."
        normalized = _normalize_planner_text(bullet)
        if normalized in seen:
            return
        seen.add(normalized)
        bullets.append(bullet)

    lowered = text.lower()
    if (
        "no active weather warnings" in lowered
        or "sem avisos meteorol" in lowered
        or "sem avisos ativos" in lowered
    ):
        bullets.append(
            "- ✅ **Avisos:** sem avisos meteorológicos ativos para Lisboa."
            if is_pt
            else "- ✅ **Warnings:** no active weather warnings for Lisbon."
        )
        seen.add(_normalize_planner_text(bullets[-1]))

    temperature_match = re.search(
        r"(?:🌡️\s*)?(?:\*\*)?(?:Temperature|Temperatura)?(?:\*\*)?:?\s*"
        r"(?P<min>-?\d+(?:\.\d+)?)°C\s*(?:to|a)\s*(?P<max>-?\d+(?:\.\d+)?)°C",
        text,
        flags=re.IGNORECASE,
    )
    if temperature_match:
        separator = "a" if is_pt else "to"
        _append(
            "Temperatura",
            "Temperature",
            f"{temperature_match.group('min')}°C {separator} {temperature_match.group('max')}°C",
            "🌡️",
        )

    conditions_match = re.search(
        r"(?:☁️|🌤️)\s*(?:\*\*)?(?:Conditions|Condições)(?:\*\*)?:\s*(?P<value>[^\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    if conditions_match:
        _append("Condições", "Conditions", conditions_match.group("value"), "☁️")

    rain_match = re.search(
        r"(?:💧\s*)?(?:\*\*)?(?:Rain probability|Probabilidade de chuva|Rain|Chuva|Precipitation|Precipitação)(?:\*\*)?:\s*(?P<value>[^\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    if rain_match:
        _append("Chuva", "Rain", rain_match.group("value"), "💧")

    wind_match = re.search(
        r"(?:💨\s*)?(?:\*\*)?(?:Wind|Vento)(?:\*\*)?:\s*(?P<value>[^\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    if wind_match:
        _append("Vento", "Wind", wind_match.group("value"), "💨")

    return bullets[:max_items]


def _build_planner_fallback_source_line(
    language: str,
    weather_data: str,
    transport_data: str,
    places_data: str,
    events_data: str,
) -> str:
    """Build a compact, deduplicated source line for deterministic planner fallback."""
    combined = "\n".join([weather_data or "", transport_data or "", places_data or "", events_data or ""]).lower()
    sources: List[str] = []
    if weather_data or "ipma" in combined:
        sources.append("[*IPMA*](https://www.ipma.pt)")
    if "visitlisboa" in combined:
        sources.append(
            "[*VisitLisboa Locais*](https://www.visitlisboa.com/pt-pt/locais)"
            if language == "pt"
            else "[*VisitLisboa Places*](https://www.visitlisboa.com/en/places)"
        )
    if "lisboa aberta" in combined or "dados abertos" in combined:
        sources.append("[*Lisboa Aberta*](https://dados.cm-lisboa.pt)")
    if (
        "metrolisboa" in combined
        or "metro de lisboa" in combined
        or re.search(r"\bmetro\b", combined)
        or re.search(r"\b(?:yellow|blue|green|red)\s+line\b|\blinha\s+(?:amarela|azul|verde|vermelha)\b", combined)
    ):
        sources.append("[*Metro de Lisboa*](https://www.metrolisboa.pt)")
    if "carrismetropolitana" in combined or "carris metropolitana" in combined:
        sources.append("[*Carris Metropolitana*](https://www.carrismetropolitana.pt)")
    if "carris.pt" in combined or "carris urban" in combined or "carris line" in combined or "linha carris" in combined:
        sources.append("[*Carris*](https://www.carris.pt)")
    if "cp.pt" in combined or "cp trains" in combined or "comboios de portugal" in combined:
        sources.append("[*CP*](https://www.cp.pt)")

    timestamp = "**Atualizado:**" if language == "pt" else "**Updated:**"
    source_label = "📌 **Fonte:**" if language == "pt" else "📌 **Source:**"
    if not sources:
        return ""
    return f"{source_label} {' | '.join(sources)} | {timestamp} {datetime.now().strftime('%H:%M')}"


def _extract_weather_safety_bullets(weather_data: str, language: str) -> List[str]:
    """Extract concise rain/warning bullets for planner fallbacks."""
    text = str(weather_data or "")
    if not text.strip():
        return []

    bullets = _extract_weather_fact_bullets(text, language, max_items=3)
    if not bullets:
        bullets.append(
            "- No detailed IPMA forecast facts were available in this run; keep weather-dependent stops flexible."
            if language != "pt"
            else "- Esta execução não trouxe factos IPMA detalhados; mantém flexíveis as paragens dependentes do tempo."
        )

    return bullets[:2]


def _is_full_museum_day_request(user_message: str) -> bool:
    """Return whether the request asks for a full-day museum plan."""
    normalized_query = _normalize_planner_text(user_message or "")
    return bool(
        re.search(r"\b(?:museum|museums|museu|museus)\b", normalized_query)
        and re.search(r"\b(?:full day|day plan|dia inteiro|um dia|amanha|tomorrow)\b", normalized_query)
    )


def _planner_response_has_incomplete_museum_day_blocks(
    user_message: str,
    response: str,
) -> bool:
    """Return whether a museum-day planner draft contains empty generic blocks."""
    if not _is_full_museum_day_request(user_message):
        return False

    normalized_response = _normalize_planner_text(response or "")
    if not normalized_response:
        return True

    placeholder_patterns = [
        r"\bblock\s+\d+\s*[.:·-]*\s*(?:short food|food/coffee|indoor backup|return leg|confirmable cultural stop)\b",
        r"\bbloco\s+\d+\s*[.:·-]*\s*(?:pausa curta|backup interior|regresso|paragem cultural confirmavel)\b",
        r"\bblock\s+\d+\s*[.:·-]*\s*$",
        r"\bbloco\s+\d+\s*[.:·-]*\s*$",
    ]
    if any(re.search(pattern, normalized_response, flags=re.MULTILINE) for pattern in placeholder_patterns):
        return True

    museum_signal_count = len(
        re.findall(
            r"\b(?:museu|museum|maat|gulbenkian|coaches|coches|chiado|belem|bel[eé]m|carris museum|national museum)\b",
            normalized_response,
        )
    )
    has_itinerary_structure = bool(re.search(r"\b(?:plan blocks|recommended itinerary|roteiro recomendado|blocos do plano)\b", normalized_response))
    return has_itinerary_structure and museum_signal_count < 3


def _is_historic_gastronomy_day_request(normalized_query: str) -> bool:
    """Detect one-day history plus traditional food itinerary requests."""
    day_intent = bool(
        re.search(
            r"\b(?:1\s*dia|um\s+dia|1[-\s]*day|one\s+day|full\s+day|dia\s+inteiro|day\s+itinerary|itinerario\s+de\s+1\s+dia|roteiro\s+de\s+1\s+dia)\b",
            normalized_query,
        )
        or re.search(r"\b(?:roteiro|plano|itinerario)\b.{0,40}\b(?:1\s*dia|um\s+dia)\b", normalized_query)
        or re.search(r"\b(?:itinerary|plan)\b.{0,40}\b(?:one\s+day|1[-\s]*day)\b", normalized_query)
    )
    history_intent = bool(
        re.search(
            r"\b(?:historic|historical|historia|historico|historicos|monument|monuments|monumento|monumentos|heritage|patrimonio|museum|museu|cultural)\b",
            normalized_query,
        )
    )
    food_intent = bool(
        re.search(
            r"\b(?:gastronom|traditional|tradicional|restaurant|restaurante|food|comida|almoco|almoço|lunch|jantar|dinner|pastry|pastelaria)\b",
            normalized_query,
        )
    )
    return day_intent and history_intent and food_intent


def _extract_visitlisboa_place_cards(text: str, *, max_items: int = 8) -> List[Dict[str, str]]:
    """Extract lightweight VisitLisboa place cards from gathered researcher text."""
    cards: List[Dict[str, str]] = []
    seen_names: set[str] = set()

    patterns = [
        re.compile(
            r"(?ms)^[ \t]{0,3}[-*]\s+\*\*(?P<icon>[^\w\s*]{0,8})\s*(?P<name>[^*\n]+?)\*\*\s*\n(?P<body>.*?)(?=^[ \t]{0,3}[-*]\s+\*\*[^\n*]{2,140}\*\*\s*$|^\s*\*\*[^\n*]{2,140}\*\*\s*$|^\s*###\s+|\Z)"
        ),
        re.compile(
            r"(?ms)^\s*\*\*(?P<icon>[^\w\s*]{0,8})\s*(?P<name>[^*\n]+?)\*\*\s*\n(?P<body>.*?)(?=^\s*\*\*[^\n*]{2,140}\*\*\s*$|^\s*###\s+|\Z)"
        ),
        re.compile(
            r"(?ms)^\s*\d+\.\s*(?P<icon>[^\w\s*]{0,4})\s*\*\*(?P<name>[^*\n]+)\*\*\s*\n(?P<body>.*?)(?=^\s*\d+\.\s*[^\n]*\*\*|\Z)"
        ),
        re.compile(
            r"(?ms)^###\s*(?P<icon>[^\w\s*]{0,4})\s*(?:\*\*)?(?P<name>[^\n*]+?)(?:\*\*)?\s*\n(?P<body>.*?)(?=^###\s+|\Z)"
        ),
    ]

    for pattern in patterns:
        for match in pattern.finditer(str(text or "")):
            name = _sanitize_planner_place_name(match.group("name"))
            normalized_name = _normalize_planner_text(name)
            if not name or normalized_name in seen_names:
                continue
            if normalized_name in {
                "places and attractions",
                "locais e atracoes",
                "locais e atracoes encontrados em lisboa",
                "events found",
                "eventos encontrados",
            }:
                continue
            body = match.group("body")
            if not re.search(
                r"(?mi)(Description|Descri(?:ç|c)[aã]o|Address|Morada|Category|Categoria|Website|Site|More details|Mais detalhes|Rating|Avalia|Price|Pre(?:ç|c)o|Tickets|Bilhetes|Hours|Hor[aá]rio)",
                body,
            ):
                continue
            description_match = re.search(r"(?mi)^\s*[-*•]?\s*(?:[^\w\s*]{1,8}\s*)?(?:\*\*)?(?:Description|Descri(?:ç|c)[aã]o)\s*:\s*(?:\*\*)?\s*(?P<value>[^\n]+)", body)
            category_match = re.search(r"(?mi)^\s*[-*•]?\s*(?:[^\w\s*]{1,8}\s*)?(?:\*\*)?(?:Category|Categoria)\s*:\s*(?:\*\*)?\s*(?P<value>[^\n]+)", body)
            address_match = re.search(r"(?mi)^\s*[-*•]?\s*(?:[^\w\s*]{1,8}\s*)?(?:\*\*)?(?:Address|Morada|Location|Localiza(?:ç|c)[aã]o)\s*:\s*(?:\*\*)?\s*(?P<value>[^\n]+)", body)
            hours_match = re.search(r"(?mi)^\s*[-*•]?\s*(?:[^\w\s*]{1,8}\s*)?(?:\*\*)?(?:Hours|Hor[aá]rio|Horarios|Horários)\s*:\s*(?:\*\*)?\s*(?P<value>[^\n]+)", body)
            price_match = re.search(r"(?mi)^\s*[-*•]?\s*(?:[^\w\s*]{1,8}\s*)?(?:\*\*)?(?:Price|Pre(?:ç|c)o)\s*:\s*(?:\*\*)?\s*(?P<value>[^\n]+)", body)
            rating_match = re.search(r"(?mi)^\s*[-*•]?\s*(?:[^\w\s*]{1,8}\s*)?(?:\*\*)?(?:Rating|Avalia(?:ç|c)[aã]o)\s*:\s*(?:\*\*)?\s*(?P<value>[^\n]+)", body)
            phone_match = re.search(r"(?mi)^\s*[-*•]?\s*(?:[^\w\s*]{1,8}\s*)?(?:\*\*)?(?:Phone|Telefone)\s*:\s*(?:\*\*)?\s*(?P<value>[^\n]+)", body)
            email_match = re.search(r"(?mi)^\s*[-*•]?\s*(?:[^\w\s*]{1,8}\s*)?(?:\*\*)?(?:Email|E-mail)\s*:\s*(?:\*\*)?\s*(?P<value>[^\n]+)", body)
            url_match = re.search(
                r"(?mi)^\s*[-*•]?\s*(?:[^\w\s*]{1,8}\s*)?"
                r"\*\*(?P<label>Website|Site|More details|Mais detalhes):\*\*\s*"
                r"(?P<value>https?://\S+|\[[^\]]+\]\(https?://[^)]+\))",
                body,
            )
            if not url_match:
                url_match = re.search(
                    r"(?mi)^\s*[-*•]?\s*(?:🔗|🌐)\s*"
                    r"(?P<value>https?://\S+|\[[^\]]+\]\(https?://[^)]+\))",
                    body,
                )
            description = description_match.group("value").strip() if description_match else ""
            if not description:
                for raw_line in body.splitlines():
                    line = raw_line.strip()
                    if not line or re.match(r"^[-*•]?\s*(?:[^\w\s*]{1,8}\s*)?\*\*(?:Category|Categoria|Address|Morada|Location|Localiza(?:ç|c)[aã]o|Hours|Hor[aá]rio|Price|Pre(?:ç|c)o|Rating|Avalia|Phone|Telefone|Website|Site|Tickets|Bilhetes|More details|Mais detalhes)\s*:", line, flags=re.IGNORECASE):
                        continue
                    description = re.sub(r"\s+", " ", line).strip()
                    break
            url = url_match.group("value").strip() if url_match else ""
            url_label = url_match.groupdict().get("label", "").strip() if url_match else ""
            if url and not url_label:
                url_label = "VisitLisboa" if "visitlisboa.com" in url.lower() else "Official website"
            markdown_url_match = re.search(r"\((https?://[^)]+)\)", url)
            if markdown_url_match:
                label_match = re.match(r"\[([^\]]+)\]\(", url)
                if label_match:
                    url_label = label_match.group(1).strip() or url_label
                url = markdown_url_match.group(1)
            cards.append(
                {
                    "name": name,
                    "category": category_match.group("value").strip() if category_match else "",
                    "address": address_match.group("value").strip() if address_match else "",
                    "hours": hours_match.group("value").strip() if hours_match else "",
                    "price": price_match.group("value").strip() if price_match else "",
                    "rating": rating_match.group("value").strip() if rating_match else "",
                    "phone": phone_match.group("value").strip() if phone_match else "",
                    "email": email_match.group("value").strip() if email_match else "",
                    "url": url,
                    "url_label": url_label,
                    "description": description,
                }
            )
            seen_names.add(normalized_name)
            if len(cards) >= max_items:
                return cards
    return cards


def _build_card_based_itinerary_fallback(
    *,
    user_message: str,
    language: str,
    weather_data: str,
    transport_data: str,
    places_data: str,
    events_data: str,
    qa_disclaimers: list[str] | None,
) -> str:
    """Build a generic fallback from evidence cards rather than prompt-specific templates."""
    normalized_query = _normalize_planner_text(user_message)
    if not re.search(
        r"\b(?:plan|itinerary|roteiro|planeia|planear|day|dia|afternoon|evening|museum|museu|visit|visitar|tour)\b",
        normalized_query,
    ):
        return ""
    if (
        re.search(r"\b(?:coffee|cafe|cafes|café|pastelaria)\b", normalized_query)
        and re.search(r"\b(?:cultural|culture|cultura|cultural stop|paragem cultural)\b", normalized_query)
    ):
        return ""

    combined_context = "\n".join([places_data or "", events_data or ""])
    cards = _extract_visitlisboa_place_cards(combined_context, max_items=12)
    cards = [
        {**card, "name": clean_name}
        for card in cards
        for clean_name in [_sanitize_planner_place_name(card.get("name", ""))]
        if clean_name
    ]
    if not cards:
        return ""

    return _build_card_based_renderer_fallback(
        user_message=user_message,
        language=language,
        cards=cards,
        weather_data=weather_data,
        transport_data=transport_data,
        places_data=places_data,
        events_data=events_data,
        qa_disclaimers=qa_disclaimers,
    )


def _build_card_based_renderer_fallback(
    *,
    user_message: str,
    language: str,
    cards: List[Dict[str, str]],
    weather_data: str,
    transport_data: str,
    places_data: str,
    events_data: str,
    qa_disclaimers: list[str] | None,
) -> str:
    """Render card fallback through the deterministic structured planner renderer.

    Args:
        user_message: Original user planning request.
        language: Final response language.
        cards: Place cards extracted from specialized-agent output.
        weather_data: Weather output, if used.
        transport_data: Transport output, if used.
        places_data: Researcher place output.
        events_data: Researcher event output.
        qa_disclaimers: Optional QA limitations to surface.

    Returns:
        Streamlit-safe LISBOA Markdown, or an empty string if no evidence card
        is usable.
    """
    selected_cards = _select_planner_cards_for_request(cards, user_message)
    if not selected_cards:
        return ""
    historic_food_request = _is_historic_gastronomy_day_request(_normalize_planner_text(user_message))
    if historic_food_request:
        selected_cards = _order_historic_food_cards(selected_cards)

    evidence = build_evidence_bundle(
        weather_data=weather_data,
        transport_data=transport_data,
        places_data=places_data,
        events_data=events_data,
        qa_disclaimers=qa_disclaimers,
    )
    is_pt = language == "pt"
    blocks: List[PlanBlock] = []
    for index, card in enumerate(selected_cards[:5], start=1):
        details = (
            _card_details_for_itinerary_block(card, language=language)
            if historic_food_request
            else _card_details_for_plan_block(card, language=language)
        )
        display_name = _planner_card_display_name(card)
        blocks.append(
            PlanBlock(
                title=_itinerary_block_title(
                    display_name or card["name"],
                    card,
                    index=index,
                    historic_food_request=historic_food_request,
                    language=language,
                ),
                kind=_card_kind_for_plan_block(card),
                purpose=_card_purpose_for_plan_block(card, is_pt),
                details=details,
                source_ids=["visitlisboa_places"],
            )
        )

    movement_items = [
        item
        for item in (
            _fallback_bullet_body(bullet)
            for bullet in _extract_planner_fallback_bullets(transport_data, max_items=5)
        )
        if item and not _is_generic_transport_heading(item)
    ]
    weather_items = [
        item
        for item in (
            _fallback_bullet_body(bullet)
            for bullet in _extract_weather_safety_bullets(weather_data, language)
        )
        if item
    ]
    limitation_items = _planner_fallback_limitations(
        language=language,
        transport_data=transport_data,
        qa_disclaimers=qa_disclaimers,
    )
    title = _card_fallback_title(user_message, language)
    direct = _card_fallback_direct_answer(user_message, language)
    source_ids = list(evidence.sources.keys())
    if "visitlisboa_places" not in source_ids:
        source_ids.insert(0, "visitlisboa_places")
    for source_id in ("metro", "carris", "cp", "ipma"):
        if source_id in evidence.sources and source_id not in source_ids:
            source_ids.append(source_id)

    draft = PlanDraft(
        title=title,
        direct_answer=direct,
        blocks=blocks,
        movement_logic=movement_items,
        weather_strategy=weather_items,
        tips=[
            "Mantém 20-30 minutos de margem entre a deslocação e a paragem cultural."
            if is_pt
            else "Keep 20-30 minutes of buffer between the transport leg and the cultural stop."
        ],
        limitations=limitation_items,
        source_ids=source_ids,
    )
    rendered = render_plan_markdown(draft, evidence.sources, language=language)
    return final_post_qa_guard(rendered, language=language)


def _planner_card_display_name(card: Dict[str, str]) -> str:
    """Return a concrete display name for generic researcher section cards."""
    name = _sanitize_planner_place_name(card.get("name", ""))
    normalized_name = _normalize_planner_text(name)
    if name and not re.search(
        r"\b(?:manha|manhã|almoco|almoço|tarde|jantar|fim de tarde|sugestao|sugestão|dica|ordem|roteiro)\b",
        normalized_name,
    ):
        return name

    evidence_text = " ".join(
        str(card.get(key, ""))
        for key in ("description", "category", "address", "url_label")
    )
    patterns = [
        r"\b((?:Museu|Mosteiro|Torre|Castelo|Pal[aá]cio|Padr[aã]o|S[eé]\s+de\s+Lisboa|Igreja|Convento|Parreirinha|Pap[’']A[cç]orda|Past[eé]is de Bel[eé]m)[^,.;\n]{0,70})",
        r"\b((?:Museum|Monastery|Tower|Castle|Palace|Church|Cathedral|Past[eé]is de Bel[eé]m)[^,.;\n]{0,70})",
    ]
    for pattern in patterns:
        match = re.search(pattern, evidence_text)
        if match:
            candidate = _sanitize_planner_place_name(match.group(1))
            if candidate:
                return candidate
    return name


def _order_historic_food_cards(cards: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Order historic-and-food fallback cards into a practical one-day flow."""
    cultural_cards = [card for card in cards if _card_kind_for_plan_block(card) != "food"]
    food_cards = [card for card in cards if _card_kind_for_plan_block(card) == "food"]

    def _zone_score(card: Dict[str, str]) -> int:
        basis = _normalize_planner_text(
            " ".join(str(card.get(key, "")) for key in ("name", "address", "description"))
        )
        if re.search(r"\b(?:carmo|chiado|baixa|rossio|correeiros|douradores)\b", basis):
            return 0
        if re.search(r"\b(?:belem|belem|bras[ií]lia|tejo|descobrimentos)\b", basis):
            return 2
        return 1

    cultural_cards = sorted(cultural_cards, key=lambda card: (_zone_score(card), -_score_historic_plan_card(card)))
    food_cards = sorted(food_cards, key=lambda card: (_zone_score(card), -_score_food_plan_card(card)))

    ordered: List[Dict[str, str]] = []
    if cultural_cards:
        ordered.append(cultural_cards[0])
    if food_cards:
        ordered.append(food_cards[0])
    ordered.extend(cultural_cards[1:4])
    if len(ordered) < 5:
        ordered.extend(food_cards[1:2])

    deduped: List[Dict[str, str]] = []
    seen: set[str] = set()
    for card in ordered:
        key = _normalize_planner_text(_planner_card_display_name(card) or card.get("name", ""))
        if key and key not in seen:
            deduped.append(card)
            seen.add(key)
    return deduped[:5]


def _itinerary_block_title(
    title: str,
    card: Dict[str, str],
    *,
    index: int,
    historic_food_request: bool,
    language: str,
) -> str:
    """Add a light schedule cue to planner fallback blocks when useful."""
    if not historic_food_request:
        return title
    is_pt = language == "pt"
    kind = _card_kind_for_plan_block(card)
    time_labels = ["09:30", "12:45", "15:00", "16:30", "19:00"]
    time_label = time_labels[min(max(index - 1, 0), len(time_labels) - 1)]
    if kind == "food" and index <= 2:
        prefix = "Almoço tradicional" if is_pt else "Traditional lunch"
    elif kind == "food":
        prefix = "Jantar opcional" if is_pt else "Optional dinner"
    else:
        prefix = "Paragem histórica" if is_pt else "Historic stop"
    return f"{time_label} · {prefix}: {title}"


def _select_planner_cards_for_request(cards: List[Dict[str, str]], user_message: str) -> List[Dict[str, str]]:
    """Select the number and type of place cards needed by a fallback plan."""
    if not cards:
        return []
    normalized = _normalize_planner_text(user_message)
    usable_cards = [card for card in cards if not _planner_dict_card_is_closed(card)] or cards
    if re.search(r"\b(?:one|1|uma|um)\s+(?:cultural\s+)?(?:stop|paragem)\b", normalized):
        scored_cards = [
            (score, card)
            for card in usable_cards
            for score in [_score_cultural_stop_card(card, normalized)]
            if score > 0
        ]
        if scored_cards:
            scored_cards.sort(key=lambda item: item[0], reverse=True)
            return [scored_cards[0][1]]
        return usable_cards[:1]
    if (
        re.search(r"\b(?:hist[oó]ric|monument|monumento|patrim[oó]nio|heritage)\b", normalized)
        and re.search(r"\b(?:gastronom|restaurant|restaurante|food|comida|tradicional|almo[cç]o|jantar)\b", normalized)
    ):
        cultural_cards = sorted(
            (
                (score, card)
                for card in usable_cards
                for score in [_score_historic_plan_card(card)]
                if score > 0
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        food_cards = sorted(
            (
                (score, card)
                for card in usable_cards
                for score in [_score_food_plan_card(card)]
                if score > 0
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        selected: List[Dict[str, str]] = []
        for _, card in cultural_cards[:2]:
            selected.append(card)
        if food_cards:
            selected.append(food_cards[0][1])
        for _, card in cultural_cards[2:4]:
            selected.append(card)
        for _, card in food_cards[1:2]:
            selected.append(card)
        deduped: List[Dict[str, str]] = []
        seen: set[str] = set()
        for card in selected:
            key = _normalize_planner_text(_planner_card_display_name(card) or card.get("name", ""))
            if key and key not in seen:
                deduped.append(card)
                seen.add(key)
        if deduped:
            return deduped[:5]
    return usable_cards[:5]


def _planner_dict_card_is_closed(card: Dict[str, str]) -> bool:
    """Return whether a raw planner card says the place is closed."""
    basis = _normalize_planner_text(
        " ".join(str(card.get(key, "")) for key in ("name", "description", "category", "hours"))
    )
    return bool(re.search(r"\b(?:hoje fechado|fechado hoje|today closed|closed today|horario hoje fechado|hours today closed)\b", basis))


def _score_historic_plan_card(card: Dict[str, str]) -> int:
    """Score a card as a historic/cultural itinerary stop."""
    basis = _normalize_planner_text(
        " ".join(str(card.get(key, "")) for key in ("name", "category", "address", "description", "hours"))
    )
    score = 0
    if re.search(r"\b(?:monument|monumento|museu|museum|igreja|church|cathedral|se de lisboa|sé de lisboa|torre|tower|padrao|padrão|mosteiro|monastery|castelo|castle|palacio|palácio|convento|carmo)\b", basis):
        score += 50
    if re.search(r"\b(?:lisboa|belem|belém|baixa|alfama|chiado|carmo|se |sé |brasilia|brasília)\b", basis):
        score += 18
    if re.search(r"\b(?:batalha|alcobaca|alcobaça|tomar|setubal|setúbal|cascais|sintra)\b", basis):
        score -= 80
    if _planner_dict_card_is_closed(card):
        score -= 60
    if card.get("address"):
        score += 5
    return score


def _score_food_plan_card(card: Dict[str, str]) -> int:
    """Score a card as a traditional-food itinerary stop."""
    basis = _normalize_planner_text(
        " ".join(str(card.get(key, "")) for key in ("name", "category", "address", "description", "hours"))
    )
    score = 0
    if re.search(r"\b(?:restaurant|restaurante|cozinha tradicional portuguesa|gastronomia|food|comida|bar|cafe|café)\b", basis):
        score += 50
    if re.search(r"\b(?:tradicional|portuguesa|typical portuguese|cozinha)\b", basis):
        score += 20
    if re.search(r"\b(?:lisboa|baixa|alfama|chiado|prata|douradores|correeiros)\b", basis):
        score += 10
    if _planner_dict_card_is_closed(card):
        score -= 80
    return score


def _score_cultural_stop_card(card: Dict[str, str], normalized_query: str) -> int:
    """Score a place card for a single cultural-stop itinerary."""
    basis = _normalize_planner_text(
        " ".join(str(card.get(key, "")) for key in ("name", "category", "address", "description", "hours"))
    )
    if not basis:
        return 0

    score = 0
    if re.search(r"\b(?:museum|museu|gallery|galeria|monument|monumento|reservoir|reservatorio|patrimonial|heritage)\b", basis):
        score += 50
    if re.search(r"\b(?:view point|viewpoint|miradouro|garden|jardim|fair|feira|neighbourhood|neighborhood|bairro)\b", basis):
        score += 28
    if re.search(r"\b(?:cultural association|associacao cultural|centro cultural|cultural centre|culture centre)\b", basis):
        score += 22
    if re.search(r"\b(?:cultural|culture|cultura)\b", basis):
        score += 8

    if re.search(r"\b(?:hostel|hotel|apartment|apartamento|accommodation|alojamento|embassy|embaixada)\b", basis):
        score -= 45
    if re.search(r"\b(?:open data|dados abertos|estacoes de metro|estacao de metro)\b", basis):
        score -= 20
    if re.search(r"\b(?:today closed|closed today|hoje fechado|fechado hoje)\b", basis):
        score -= 55

    for token in ("principe real", "príncipe real", "saldanha", "chiado", "belem", "belém", "baixa"):
        if token in normalized_query and token in basis:
            score += 12
    if card.get("address"):
        score += 4
    if card.get("url"):
        score += 3
    return score


def _card_kind_for_plan_block(card: Dict[str, str]) -> str:
    """Infer a renderer block kind from a VisitLisboa-style card."""
    basis = _normalize_planner_text(" ".join(str(card.get(key, "")) for key in ("name", "category", "description")))
    if re.search(r"\b(?:restaurant|restaurante|food|gastronomy|gastronomia|wine|bar)\b", basis):
        return "food"
    if re.search(r"\b(?:museum|museu|gallery|galeria|monument|monumento|reservoir|reservatorio|reservatório)\b", basis):
        return "museum"
    return "place"


def _card_purpose_for_plan_block(card: Dict[str, str], is_pt: bool) -> str:
    """Build a concise, grounded purpose line for a fallback place block."""
    category = str(card.get("category") or "").strip()
    if is_pt:
        if category:
            return f"Paragem compacta e verificável para o tema pedido; categoria indicada: {category}."
        return "Paragem compacta e verificável para o tema pedido, limitada aos dados recolhidos."
    if category:
        return f"Compact, evidenced stop for the requested theme; listed category: {category}."
    return "Compact, evidenced stop for the requested theme, limited to the gathered data."


def _card_details_for_plan_block(card: Dict[str, str], *, language: str = "en") -> List[str]:
    """Convert a place card into semantic planner detail fields."""
    details: List[str] = []
    if card.get("description"):
        details.append(f"Description: {_planner_card_description_for_language(card['description'], language)}")
    if card.get("category"):
        details.append(f"Category: {card['category']}")
    if card.get("address"):
        details.append(f"Address: {card['address']}")
    if card.get("hours"):
        details.append(f"Hours: {card['hours']}")
    if card.get("price"):
        details.append(f"Price: {card['price']}")
    if card.get("rating"):
        details.append(f"Rating: {card['rating']}")
    if card.get("phone"):
        details.append(f"Phone: {card['phone']}")
    if card.get("email"):
        details.append(f"Email: {card['email']}")
    if card.get("url"):
        url = str(card["url"])
        url_label = str(card.get("url_label") or "").strip()
        if not url_label:
            url_label = "VisitLisboa" if "visitlisboa.com" in url.lower() else "Official website"
        if url_label.lower() == "visitlisboa" and "visitlisboa.com" not in url.lower():
            url_label = "Website oficial" if language == "pt" else "Official website"
        details.append(f"Website: [{url_label}]({url})")
    return details


def _card_details_for_itinerary_block(card: Dict[str, str], *, language: str = "en") -> List[str]:
    """Convert a place card into concise itinerary fields rather than a raw card."""
    details: List[str] = []
    if card.get("description"):
        details.append(f"Description: {_planner_card_description_for_language(card['description'], language)}")
    if card.get("address"):
        details.append(f"Address: {card['address']}")
    if card.get("hours"):
        details.append(f"Hours: {card['hours']}")
    if card.get("price"):
        details.append(f"Price: {card['price']}")
    if card.get("url"):
        url = str(card["url"])
        url_label = str(card.get("url_label") or "").strip()
        if not url_label:
            url_label = "VisitLisboa" if "visitlisboa.com" in url.lower() else "Official website"
        if url_label.lower() == "visitlisboa" and "visitlisboa.com" not in url.lower():
            url_label = "Website oficial" if language == "pt" else "Official website"
        details.append(f"Website: [{url_label}]({url})")
    return details


def _enrich_plan_draft_from_evidence(
    *,
    draft: PlanDraft,
    evidence: EvidenceBundle,
    user_message: str,
) -> PlanDraft:
    """Restore useful card fields that the planner selected but omitted.

    The LLM chooses the plan structure; this deterministic pass preserves the
    Researcher contract by carrying available address, hours, price, website,
    and ticket fields into selected blocks instead of letting them disappear.
    """
    if not draft.blocks or not evidence.cards:
        return draft

    time_sensitive = bool(
        re.search(
            r"\b(evening|tonight|night|afternoon|morning|today|tomorrow|noite|esta noite|fim de tarde|tarde|manha|manhã|hoje|amanha|amanhã)\b",
            _normalize_planner_text(user_message),
        )
    )
    for block in draft.blocks:
        matched_card = _match_evidence_card_for_block(block.title, evidence.cards)
        if not matched_card:
            continue
        existing_labels = {
            _normalize_planner_text(match.group("label"))
            for detail in block.details
            for match in [re.match(r"^\s*(?P<label>[A-Za-zÀ-ÿ ]{2,30})\s*:", str(detail or ""))]
            if match
        }
        for label in ("Description", "Category", "Address", "Venue", "When", "Hours", "Price", "Rating", "Phone", "Email", "Website", "Tickets"):
            value = _evidence_field_value(matched_card, label)
            if not value:
                continue
            normalized_label = _normalize_planner_text(label)
            if normalized_label not in existing_labels:
                block.details.append(f"{label}: {value}")
                existing_labels.add(normalized_label)
        block.source_ids = list(dict.fromkeys([*block.source_ids, *matched_card.source_ids]))
        if time_sensitive and _evidence_card_is_closed(matched_card):
            warning = (
                "Source lists this stop as closed for the checked period; use it only as exterior/context unless opening is confirmed."
            )
            if not any("closed" in _normalize_planner_text(item) or "fechado" in _normalize_planner_text(item) for item in block.limitations):
                block.limitations.insert(0, warning)
    draft.source_ids = list(dict.fromkeys([*draft.source_ids, *evidence.source_ids()]))
    return draft


def _match_evidence_card_for_block(title: str, cards: List[EvidenceCard]) -> EvidenceCard | None:
    """Find the evidence card that supports a selected plan block."""
    normalized_title = _normalize_planner_text(title)
    if not normalized_title:
        return None
    for card in cards:
        normalized_card = _normalize_planner_text(card.title)
        if not normalized_card:
            continue
        if (
            normalized_title == normalized_card
            or normalized_title in normalized_card
            or normalized_card in normalized_title
        ):
            return card
    return None


def _evidence_field_value(card: EvidenceCard, label: str) -> str:
    """Return a field value from an evidence card using canonical label aliases."""
    wanted = _normalize_planner_text(label)
    aliases = {
        "address": {"address", "morada", "location", "localizacao"},
        "venue": {"venue", "local"},
        "hours": {"hours", "horario", "horarios", "today", "hoje"},
        "price": {"price", "preco"},
        "website": {"website", "url", "more details", "mais detalhes"},
        "tickets": {"tickets", "bilhetes"},
        "category": {"category", "categoria"},
        "description": {"description", "descricao"},
        "rating": {"rating", "avaliacao", "avaliação"},
        "phone": {"phone", "telefone"},
        "email": {"email", "e-mail"},
        "when": {"when", "quando"},
    }
    accepted = aliases.get(wanted, {wanted})
    for raw_label, raw_value in (card.fields or {}).items():
        if _normalize_planner_text(str(raw_label)) in accepted:
            return str(raw_value or "").strip()
    return ""


def _evidence_card_is_closed(card: EvidenceCard) -> bool:
    """Return whether an evidence card explicitly marks the venue as closed."""
    hours = _evidence_field_value(card, "Hours")
    return bool(re.search(r"\b(closed|fechado|encerrado)\b", _normalize_planner_text(hours)))


def _fallback_bullet_body(bullet: str) -> str:
    """Normalize a fallback bullet into a renderer-ready item body."""
    text = str(bullet or "").strip()
    text = re.sub(r"^(?:[-•]\s+|\*\s+)", "", text).strip()
    text = re.sub(r"^[🔹▪️▫️]\s*", "", text).strip()
    text = re.sub(r"^#{1,6}\s*", "", text).strip()
    text = re.sub(r"\*\*([^*:\n]{2,80}):\*\*\s*", r"**\1:** ", text)
    text = re.sub(r"\*\*([^*:\n]{2,80}):\s*([^*]{1,160})\*\*", r"**\1:** \2", text)
    text = re.sub(r"\*\*([^*]+)\*\*\s*:\s*", r"\1: ", text)
    text = re.sub(r"\*\*([^*:]{2,80}):\s*\*\*", r"**\1:**", text)
    text = re.sub(
        r"\b(Transfer at|Continue on|Exit at|Start at|Route|Estimated total time|Nearest metro to [^:]{1,80}|Mudar em|Continuar em|Sair em|Começar em|Comecar em|Rota|Percurso|Tempo estimado):(?=[^\s*])",
        r"\1: ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^\*\*([^*]+)\*\*$", r"\1", text)
    return re.sub(r"\s+", " ", text).strip(" -")


def _is_generic_transport_heading(item: str) -> bool:
    """Return whether a transport item is only a section title."""
    normalized = _normalize_planner_text(item)
    return normalized in {
        "getting there",
        "how to move",
        "main move metro",
        "best transport",
        "melhor transporte",
        "como te deslocas",
        "logica de deslocacao",
    }


def _planner_fallback_limitations(
    *,
    language: str,
    transport_data: str,
    qa_disclaimers: list[str] | None,
) -> List[str]:
    """Build concise fallback limitations without internal QA wording."""
    is_pt = language == "pt"
    limitations = [
        "Horários, bilhetes, reservas e disponibilidade em tempo real só estão confirmados quando aparecem nos detalhes acima."
        if is_pt
        else "Opening hours, tickets, bookings, and live availability are confirmed only when they appear in the details above."
    ]
    if transport_data:
        limitations.append(
            "Para uma viagem futura, confirma partidas e eventuais alterações no operador antes de sair."
            if is_pt
            else "For a future trip, confirm departures and any service changes with the operator before leaving."
        )
    for item in qa_disclaimers or []:
        text = _fallback_bullet_body(str(item))
        if re.search(
            r"\b(?:final response|canonical|worker|agent|qa|repair|schema|should avoid presenting|keeps both|gtfs data|carris line numbers|dados gtfs|numeros das linhas|números das linhas|horarios da carris|horários da carris|carris\.pt)\b",
            text,
            flags=re.IGNORECASE,
        ):
            continue
        if text and len(text) <= 180:
            limitations.append(text)
    return list(dict.fromkeys(limitations))[:4]


def _card_fallback_title(user_message: str, language: str) -> str:
    """Build a specific fallback title from the requested planning intent."""
    normalized = _normalize_planner_text(user_message)
    if _is_historic_gastronomy_day_request(normalized):
        return "Roteiro histórico e gastronómico de 1 dia" if language == "pt" else "One-day history and food itinerary"
    if "principe real" in normalized or "príncipe real" in str(user_message).lower():
        return "Noite descontraída no Príncipe Real" if language == "pt" else "Relaxed evening around Príncipe Real"
    if re.search(r"\b(?:museum|museu|museums|museus)\b", normalized):
        return "Dia de museus em Lisboa" if language == "pt" else "Lisbon museum day"
    if "belem" in normalized:
        return "Plano para Belém" if language == "pt" else "Belém plan"
    return "Roteiro sugerido" if language == "pt" else "Suggested itinerary"


def _card_fallback_direct_answer(
    user_message: str,
    language: str,
) -> str:
    """Build the direct answer for the renderer-based card fallback."""
    is_pt = language == "pt"
    normalized = _normalize_planner_text(user_message)
    if _is_historic_gastronomy_day_request(normalized):
        if is_pt:
            return "Organizei um dia compacto com monumentos históricos e uma pausa gastronómica, usando apenas dados recolhidos e verificáveis."
        return "I organized a compact day with historic stops and one food break, using only gathered and verifiable data."
    if "principe real" in normalized:
        if is_pt:
            return "Parte de Saldanha, usa o metro como eixo principal até à zona da Avenida/Rato e mantém uma única paragem cultural no Príncipe Real."
        return "Start from Saldanha, use the metro as the main public-transport leg toward Avenida/Rato, and keep one cultural stop around Príncipe Real."
    if is_pt:
        return "Segue a ordem abaixo; usei apenas locais e deslocações apoiados pela evidência recolhida."
    return "Follow the order below; I used only places and movement details supported by the gathered evidence."


def _build_specific_planner_fallback(
    *,
    user_message: str,
    language: str,
    weather_data: str,
    transport_data: str,
    places_data: str,
    events_data: str,
    qa_disclaimers: list[str] | None = None,
) -> str:
    """Return a card-based fallback for itinerary shapes with concrete evidence."""
    normalized_query = _normalize_planner_text(user_message)
    if _is_historic_gastronomy_day_request(normalized_query):
        return _build_card_based_itinerary_fallback(
            user_message=user_message,
            language=language,
            weather_data=weather_data,
            transport_data=transport_data,
            places_data=places_data,
            events_data=events_data,
            qa_disclaimers=qa_disclaimers,
        )

    if _is_full_museum_day_request(user_message):
        card_based_fallback = _build_card_based_itinerary_fallback(
            user_message=user_message,
            language=language,
            weather_data=weather_data,
            transport_data=transport_data,
            places_data=places_data,
            events_data=events_data,
            qa_disclaimers=qa_disclaimers,
        )
        return card_based_fallback

    return ""


def _planner_response_requires_fallback(cleaned_response: str) -> bool:
    """Return whether the cleaned planner draft is effectively a failure placeholder."""
    normalized = (cleaned_response or "").strip().lower()
    if not normalized:
        return True
    failure_markers = [
        "desculpe, tive dificuldades em processar o pedido",
        "sorry, i'm having difficulty processing your request",
        "an error occurred while processing",
    ]
    return any(marker in normalized for marker in failure_markers)


def _planner_response_has_markdown_contract_defects(cleaned_response: str) -> bool:
    """Return whether a planner draft is structurally unsafe to render.

    This catches classes of failures observed in live planner synthesis: repeated
    pseudo-card headings, raw place-card fragments injected as itinerary steps,
    and loose key-stop dumps rendered as top-level bullets. These are repaired by
    falling back to the deterministic planner template instead of letting QA
    rewrite a malformed structure after the fact.
    """
    if not cleaned_response:
        return True

    headings = [line.strip() for line in cleaned_response.splitlines() if line.strip().startswith("### ")]
    normalized_headings = [_normalize_planner_text(heading) for heading in headings]
    if len(normalized_headings) != len(set(normalized_headings)):
        return True
    if _has_nested_item_headings_inside_route_section(cleaned_response):
        return True

    raw_lower = (cleaned_response or "").lower()
    normalized = _normalize_planner_text(cleaned_response)
    source_match = re.search(r"(?mi)^\s*📌\s+\*\*(?:Source|Fonte):\*\*.*$", cleaned_response)
    source_line = source_match.group(0) if source_match else ""
    if re.search(r"(?m)^\s*[-*]\s*[\U0001F300-\U0001FAFF\u2300-\u23FF\u2600-\u27BF\uFE0F\u200D\s]+\s*$", cleaned_response):
        return True
    if re.search(
        r"(?mi)^\s*[-*]\s*📝\s+\*\*(?:Descrição|Descricao|Description):\*\*\s*(?:⭐|TripAdvisor|Avalia[cç][aã]o|Rating)\b",
        cleaned_response,
    ):
        return True
    if re.search(r"\b(?:linha\s+15e|linha\s+728|15e|728)\b", raw_lower) and "carris" not in source_line.lower():
        return True
    if re.search(r"(?m)^\s*\d+[.)]\s+", cleaned_response):
        return True
    if len(re.findall(r"(?im)^\s*(?:[-*]\s*)?📌\s*\*\*(?:Source|Sources|Fonte|Fontes):\*\*", cleaned_response)) > 1:
        return True
    defect_patterns = [
        r"\bkey stop cards\b",
        r"\bif the showers continue, this is the best moment\b",
        r"\bphone\b\s*:\s*\+351.*\brating\b",
        r"\bhelpful notes\b",
        r"\bnotas uteis\b",
        r"\bsource footer\b",
        r"\bplace card\b",
        r"\bbest practical version for tonight\b",
        r"\bclosed today\b",
        r"\bnot open today\b",
        r"\bfechado hoje\b",
        r"\bnao abre hoje\b",
        r"\bnão abre hoje\b",
        r"\bfor an \d{1,2}\s*:\s*\d{2}\b",
        r"\bnot near\b.*\btreat it as\b",
        r"\bnot in the same immediate\b",
        r"\bcrossing lisbon just to fit\b",
        r"\bconfirmable cultural stop\b",
        r"\bparagem cultural confirmavel\b",
        r"\bpasseio livre\b",
        r"\bpasseio historico final\b",
        r"\bseguir a ordem indicada\b",
        r"\bpernas exatas\b",
        r"\bno explicit constraints beyond the requested plan\b",
        r"\brestricoes nao especificadas\b",
        r"\buse only evidence cards\b",
        r"\bprefer supported transport evidence\b",
        r"\bdescription\s*:\s+.*\bcategory\s*:",
        r"\baddress\s*:\s+.*google\.com/maps",
        r"\bdescri[cç][aã]o\s*:\s+.*\bcategoria\s*:",
    ]
    unsafe_heading_patterns = [
        r"^###\s*(?:⚠️\s*)?(?:helpful notes|notas úteis|notas uteis|notes)\b",
        r"^###\s+[^\n]*\b\d{1,2}:\d{2}\b\s*[·\-]\s*(?:today|hoje)\s*:",
        r"^###\s+[^\n]*\b(?:hours|horario|horário)\s*:",
        r"^###\s*(?:ℹ️\s*)?(?:note|nota)\s*:?\s*$",
        r"^###\s+🚇\s+\*\*(?!(?:Como te deslocas|How to move)\b)",
        r"^###\s+🚇\s+\*\*(?:Manhã|Meio da manhã|Almoço|Tarde|Jantar|Fim de tarde|Morning|Lunch|Afternoon|Dinner)\b",
    ]
    if any(re.search(pattern, heading, flags=re.IGNORECASE) for heading in headings for pattern in unsafe_heading_patterns):
        return True
    if re.search(r"(?m)^###\s+[^\n]*\b\d{1,2}:\d{2}\b\s*[·\-]\s*(?:today|hoje)\s*:", raw_lower):
        return True
    if re.search(
        r"(?mi)^\s*[-*]\s*(?:[^\w\s]\s*)?\*\*(?:Location|Address|Website|Phone|Category|Description|Morada|Telefone|Categoria|Descri[cç][aã]o)\*\*\s*:",
        cleaned_response,
    ):
        return True
    if re.search(
        r"(?mi)^\s*[-*]\s*(?:[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+\s*)?\*\*(?:Description|Category|Address|Phone|Website|Descrição|Descricao|Categoria|Morada|Telefone)\*\*\s*:",
        cleaned_response,
    ):
        return True
    if re.search(r"(?mi)^\s*[-*]\s*(?:[^\w\s]\s*)?\*\*https\*\*\s*:", cleaned_response):
        return True
    if re.search(r"(?mi)\*\*(?:https?|www)\*\*\s*:\s*//", cleaned_response):
        return True
    if re.search(r"(?mi)^###\s+.*\bscheduled tonight\b", cleaned_response):
        return True
    if re.search(
        r"(?mi)^\*\*🏷️\s*(?:Museum|Place|Manh[aã]|Almo[cç]o|Tarde|Jantar|Fim de tarde|Evening|Morning|Lunch|Afternoon|Dinner)\b",
        cleaned_response,
    ):
        return True
    if re.search(r"(?mi)^###\s+🧭\s+\*\*(?:Plan basis|Base do plano)\*\*", cleaned_response):
        return True
    if re.search(
        r"(?is)^###\s+☔\s+\*\*(?:Weather adaptation|Adaptação ao tempo)\*\*.*?\b(?:weather was not confirmed|weather was not provided|worker output|tempo não confirmado|tempo nao confirmado)\b",
        cleaned_response,
    ):
        return True
    if re.search(r"(?mi)^###\s+🚇\s+\*\*(?:Transport Limits|Limites dos transportes)\*\*", cleaned_response):
        return True
    if re.search(
        r"\*\*(?:Best transport|Best public transport|Estimated travel time|Estimated total time|Route|Trajeto|Percurso|Tempo total estimado|Melhor transporte|Melhor transporte público):\S",
        cleaned_response,
        flags=re.IGNORECASE,
    ):
        return True
    if re.search(
        r"(?is)💡\s*\*\*(?:Tips|Dicas):\*\*\s*\n+\s*[-*]\s*(?:use public transport|usar transportes públicos)\b",
        cleaned_response,
    ):
        return True
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in defect_patterns)


def _has_nested_item_headings_inside_route_section(markdown: str) -> bool:
    """Return whether planner item cards were published as H3 headings."""
    in_route = False
    for raw_line in str(markdown or "").splitlines():
        stripped = raw_line.strip()
        if re.match(r"^###\s+📍\s+(?:\*\*)?(?:Roteiro sugerido|Suggested route)", stripped, flags=re.IGNORECASE):
            in_route = True
            continue
        if in_route and stripped.startswith("### ") and re.match(
            r"^###\s+(?:🏛️|🍽️|☕|🥐|🎭|📍)\s+",
            stripped,
        ):
            return True
        if in_route and stripped.startswith("### ") and not re.match(
            r"^###\s+(?:🏛️|🍽️|☕|🥐|🎭|📍)\s+",
            stripped,
        ):
            in_route = False
    return False


def _planner_response_has_transport_quality_defects(
    cleaned_response: str,
    user_message: str,
    transport_data: str,
) -> bool:
    """Return whether transport-aware planner output hides grounded route gaps."""
    has_route_leg_evidence = bool(
        re.search(
            r"\b(?:liga[cç][oõ]es entre paragens do roteiro|route legs between itinerary stops|carris\s+15e|caminhada curta|short walk)\b",
            _normalize_planner_text(transport_data),
            re.IGNORECASE,
        )
    )
    if (
        not cleaned_response
        or not str(transport_data or "").strip()
        or (
            not _query_requests_public_transport(user_message)
            and not has_route_leg_evidence
        )
    ):
        return False

    normalized = _normalize_planner_text(cleaned_response)
    if has_route_leg_evidence and not re.search(
        r"\b(?:como te deslocas|how to move|carris\s+15e|linha\s+15e|caminhada curta|short walk|route legs|liga[cç][oõ]es)\b",
        normalized,
        re.IGNORECASE,
    ):
        return True

    vague_transport_patterns = (
        r"\bverify locally\b",
        r"\bcheck the most direct (?:bus or metro )?connection locally\b",
        r"\bcontinue by public transport\b",
        r"\buse public transport from\b",
        r"\bcentral lisbon by public transport\b",
        r"\buse the exact street address\b",
        r"\bnearest stop name for the most reliable route\b",
        r"\btransport is available but should be checked\b",
    )
    if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in vague_transport_patterns):
        return True

    live_status_requested = re.search(
        r"\b(?:next|live|real[- ]?time|right now|now|current|pr[oó]xim[oa]s?|tempo real|agora)\b",
        _normalize_planner_text(user_message),
        re.IGNORECASE,
    )
    if (
        not live_status_requested
        and re.search(r"\b(?:departures?|partidas?|live wait|current status|pr[oó]xim[oa]s? partidas?)\b", normalized, re.IGNORECASE)
        and re.search(r"\b\d{1,2}\s+\d{2}\b", normalized)
    ):
        return True

    if re.search(r"\btransport\b", normalized) and not re.search(
        r"\b(line|linha|metro|carris|cp|bus|tram|autocarro|eletrico|green|blue|red|yellow|verde|azul|vermelha|amarela|board|alight|transfer|embarque|saida|paragem|stop|station)\b",
        normalized,
        re.IGNORECASE,
    ):
        return True

    return False


def _append_transport_uncertainty_note(response: str, language: str) -> str:
    """Append a scoped transport note without replacing a useful planner draft."""
    note_heading = "### 🚇 Limites dos transportes" if language == "pt" else "### 🚇 Transport Limits"
    note_line = (
        "- Algumas ligações entre zonas não ficaram totalmente confirmadas nos dados recolhidos; confirma no operador antes de sair e não trates isto como horário de partida em tempo real."
        if language == "pt"
        else "- Some exact public-transport legs were not fully confirmed in the gathered data; check the operator before leaving and do not treat this as a live departure schedule."
    )
    if note_heading.lower() in response.lower():
        return response

    source_match = re.search(r"(?m)^📌\s+\*\*(?:Source|Fonte):\*\*.*$", response or "")
    if source_match:
        before = response[:source_match.start()].rstrip()
        source = response[source_match.start():].strip()
        return f"{before}\n\n---\n\n{note_heading}\n{note_line}\n\n{source}".strip()
    return f"{response.rstrip()}\n\n---\n\n{note_heading}\n{note_line}".strip()


def _add_transport_quality_issue(
    issues: List[str],
    response: str,
    user_message: str,
    transport_data: str,
) -> List[str]:
    """Append a planner transport-quality issue when route synthesis is too vague."""
    if _planner_response_has_transport_quality_defects(response, user_message, transport_data):
        issue = (
            "Public-transport guidance is too vague despite transport context being available. "
            "Use concrete supported operator, line, direction, board/alight, or transfer details; "
            "otherwise mark the exact leg as unconfirmed."
        )
        if issue not in issues:
            issues.append(issue)
    return issues


class PlannerAgent(BaseAgent):
    """
    Itinerary planner agent that synthesizes outputs from other agents.

    Responsibilities:
        - Combine weather, transport, and places data
        - Apply constraints (mobility, time, weather)
        - Generate coherent, practical itineraries

    Note:
        This agent has NO tools. It only synthesizes data gathered by worker
        agents and can surface QA disclaimers in the final planning response.
        In the default runtime, it is invoked only when the supervisor route
        includes the planner. Direct and simple single-domain queries can
        return without using this agent.
    """

    def __init__(self):
        """Initializes the planner agent."""
        super().__init__("planner")
        self.system_prompt = get_planner_prompt(language="en")
        self._system_prompt_dynamic = True

    def _get_runtime_system_prompt(self, language: str) -> str:
        """Return the prompt for the requested language while preserving explicit test overrides."""
        if not getattr(self, "_system_prompt_dynamic", False):
            override_prompt = getattr(self, "system_prompt", "")
            if override_prompt:
                return override_prompt

        prompt = get_planner_prompt(language=language)
        self.system_prompt = prompt
        return prompt

    def _try_structured_json_plan(
        self,
        *,
        user_message: str,
        language: str,
        weather_data: str,
        transport_data: str,
        places_data: str,
        events_data: str,
        qa_disclaimers: list[str] | None,
        conversation_context: str,
    ) -> str:
        """Attempt the new evidence-card -> JSON -> deterministic-renderer path.

        This is the preferred planner path. It keeps the LLM responsible for
        planning decisions while making the visual response deterministic. If
        the model does not return valid or sufficiently grounded JSON, the
        legacy planner path below remains available as a fallback.
        """
        evidence = build_evidence_bundle(
            weather_data=weather_data,
            transport_data=transport_data,
            places_data=places_data,
            events_data=events_data,
            qa_disclaimers=qa_disclaimers,
        )

        # If there is no usable evidence at all, keep the legacy path so its
        # bounded fallbacks can explain the limitation.
        if not evidence.cards:
            return ""

        messages = build_structured_plan_messages(
            user_message=user_message,
            language=language,
            evidence=evidence,
            conversation_context=conversation_context,
        )
        try:
            response = self._safe_llm_invoke(self.llm, messages)
        except Exception:
            return ""

        raw_content = getattr(response, "content", "")
        draft = parse_plan_draft_json(str(raw_content or ""))
        if draft is None:
            draft = parse_plan_draft_json(clean_response(raw_content, _print=False))
        if draft is None:
            return ""

        issues = validate_plan_draft(
            draft,
            evidence,
            user_message=user_message,
        )
        if issues:
            # Give the model one targeted repair attempt. This is not a free-form
            # Markdown repair; it must still satisfy the same JSON contract.
            retry_messages = messages + [
                SystemMessage(
                    content=(
                        "The previous JSON failed validation. Return corrected JSON only. "
                        "Do not add Markdown. Fix these issues: " + "; ".join(issues[:8])
                    )
                )
            ]
            try:
                retry_response = self._safe_llm_invoke(self.llm, retry_messages, retries=1)
                draft = parse_plan_draft_json(str(getattr(retry_response, "content", "") or ""))
            except Exception:
                draft = None
            if draft is None:
                return ""
            issues = validate_plan_draft(
                draft,
                evidence,
                user_message=user_message,
            )
            if issues:
                return ""

        draft = _enrich_plan_draft_from_evidence(
            draft=draft,
            evidence=evidence,
            user_message=user_message,
        )
        rendered = render_plan_markdown(
            draft,
            sources=evidence.sources,
            language=language,
        )
        rendered = final_post_qa_guard(rendered, language=language)
        if not _planner_response_matches_schema(rendered):
            return ""
        if _planner_response_has_markdown_contract_defects(rendered):
            return ""
        if not _planner_response_has_minimum_user_value(rendered):
            return ""
        return rendered.strip()

    @traceable(name="planner_agent", run_type="chain", tags=["sub-agent", "planner"])
    def invoke(
        self,
        user_message: str,
        weather_data: str = "",
        transport_data: str = "",
        places_data: str = "",
        events_data: str = "",
        qa_disclaimers: list[str] | None = None,
        conversation_context: str = "",
    ) -> str:
        """
        Creates an itinerary from gathered data.

        Args:
            user_message: The user's original query.
            weather_data: Output from weather agent.
            transport_data: Output from transport agent.
            places_data: Output from researcher agent (places).
            events_data: Output from researcher agent (events).
            qa_disclaimers: Optional list of QA-flagged data limitations.
            conversation_context: Previous-turn planning context for follow-ups.

        Returns:
            str: Formatted itinerary.
        """
        language = infer_response_language(user_query=user_message, default="en")
        # Preferred path: the planner decides content in JSON, then a deterministic
        # renderer guarantees LISBOA visual structure. This avoids asking the LLM
        # to simultaneously plan and hand-format Markdown. Fallbacks are only
        # used after this evidence-driven synthesis path cannot produce a valid
        # evidence-supported plan.
        structured_plan = self._try_structured_json_plan(
            user_message=user_message,
            language=language,
            weather_data=weather_data,
            transport_data=transport_data,
            places_data=places_data,
            events_data=events_data,
            qa_disclaimers=qa_disclaimers,
            conversation_context=conversation_context,
        )
        if structured_plan:
            if (
                not _planner_response_has_markdown_contract_defects(structured_plan)
                and _planner_response_has_minimum_user_value(structured_plan)
                and not _planner_response_has_transport_quality_defects(
                    structured_plan,
                    user_message,
                    transport_data,
                )
            ):
                return enforce_multi_day_quality_mode(
                    structured_plan,
                    user_message,
                    language,
                )
            structured_plan = ""

        specific_plan = _build_specific_planner_fallback(
            user_message=user_message,
            language=language,
            weather_data=weather_data,
            transport_data=transport_data,
            places_data=places_data,
            events_data=events_data,
        )
        if specific_plan:
            if _planner_response_has_minimum_user_value(specific_plan):
                return enforce_multi_day_quality_mode(
                    specific_plan,
                    user_message,
                    language,
                )

        card_based_plan = _build_card_based_itinerary_fallback(
            user_message=user_message,
            language=language,
            weather_data=weather_data,
            transport_data=transport_data,
            places_data=places_data,
            events_data=events_data,
            qa_disclaimers=qa_disclaimers,
        )
        if card_based_plan:
            if _planner_response_has_minimum_user_value(card_based_plan):
                return enforce_multi_day_quality_mode(
                    card_based_plan,
                    user_message,
                    language,
                )

        # Legacy path: retained as a fallback when the JSON path cannot produce a
        # valid, grounded PlanDraft.

        # Build context from agent outputs
        context_parts = []
        evidence_packet = _build_planner_evidence_packet(
            user_message=user_message,
            language=language,
            weather_data=weather_data,
            transport_data=transport_data,
            places_data=places_data,
            events_data=events_data,
            qa_disclaimers=qa_disclaimers,
            conversation_context=conversation_context,
        )
        if evidence_packet:
            context_parts.append(evidence_packet)
        if conversation_context:
            context_parts.append(
                "## 🔁 Conversation Continuity\n"
                + conversation_context.strip()[:1200]
            )
        if weather_data:
            compact_weather = _compact_planner_context_block(
                "## 🌤️ Weather Data",
                weather_data,
                max_lines=12,
                max_chars=900,
            )
            if compact_weather:
                context_parts.append(compact_weather)

        if places_data:
            compact_places = _compact_planner_context_block(
                "## 🏛️ Places & Attractions",
                places_data,
                max_lines=22,
                max_chars=1600,
            )
            if compact_places:
                context_parts.append(compact_places)

        if events_data:
            compact_events = _compact_planner_context_block(
                "## 🎭 Events",
                events_data,
                max_lines=18,
                max_chars=1400,
            )
            if compact_events:
                context_parts.append(compact_events)

        if transport_data:
            compact_transport = _compact_planner_context_block(
                "## 🚇 Transport Info",
                transport_data,
                max_lines=18,
                max_chars=1400,
            )
            if compact_transport:
                context_parts.append(compact_transport)

        # Inject QA disclaimers so the planner transparently communicates limitations
        if qa_disclaimers:
            disclaimer_text = "\n".join(f"- ⚠️ {d}" for d in qa_disclaimers)
            heading = "## ⚠️ Cautelas dos Dados" if language == "pt" else "## ⚠️ Data Caveats"
            instruction = (
                "Inclui estas cautelas apenas quando forem úteis para o utilizador:"
                if language == "pt"
                else "Include these caveats only where useful for the user:"
            )
            context_parts.append(
                f"{heading}\n"
                f"{instruction}\n{disclaimer_text}"
            )

        context = "\n\n---\n\n".join(context_parts) if context_parts else "No additional data provided."
        allowed_places = _extract_allowed_place_names("\n".join(part for part in [places_data, events_data] if part))
        accessibility_requested = _query_requests_accessibility(user_message)
        accessibility_confirmed = _context_has_accessibility_data(
            places_data,
            events_data,
            transport_data,
        )
        grounding_message = _build_planner_grounding_message(
            allowed_places=allowed_places,
            accessibility_requested=accessibility_requested,
            accessibility_confirmed=accessibility_confirmed,
        )
        requested_days = _extract_requested_day_count(user_message)
        multi_day_instruction = _build_multi_day_planner_instruction(
            language=language,
            requested_days=requested_days,
        )
        public_transport_instruction = _build_public_transport_synthesis_instruction(
            user_message=user_message,
            transport_data=transport_data,
        )

        language_instruction = (
            "Respond ENTIRELY in Portuguese (PT-PT)."
            if language == "pt"
            else "Respond ENTIRELY in English."
        )

        messages = [
            SystemMessage(content=self._get_runtime_system_prompt(language)),
            SystemMessage(content=language_instruction),
            SystemMessage(content=grounding_message),
            *([SystemMessage(content=multi_day_instruction)] if multi_day_instruction else []),
            *([SystemMessage(content=public_transport_instruction)] if public_transport_instruction else []),
            SystemMessage(
                content=(
                    "OUTPUT BUDGET AND STRUCTURE:\n"
                    "- Target 450-650 words for rich cross-domain itineraries; stay shorter for simple requests.\n"
                    "- Use this exact data schema internally: title, direct_answer, plan_basis, route_blocks, movement, weather_adaptation, final_notes.\n"
                    "- Make route_blocks ordered and useful; use at most 4 blocks for one-day plans.\n"
                    "- Include the useful evidence gathered by specialized agents: weather consequence, realistic movement, supported stops, and user preferences.\n"
                    "- Do not include raw place cards, empty sections, malformed labels, or placeholder fields.\n"
                    "- Do not include restaurants, events, exact prices, tickets, exact routes, or exact weather unless confirmed in the provided data.\n"
                    "- Do not include current live departure times in an itinerary unless the user explicitly asked for next departures or live status.\n"
                    "- Prefer short factual sentences over long explanations."
                )
            ),
            SystemMessage(content=f"# Data from Specialized Agents\n\n{context}"),
            HumanMessage(content=f"Based on the data above, create an itinerary for: {user_message}")
        ]

        # Planner has no tools - LLM call with retry for Azure content filter
        try:
            response = self._safe_llm_invoke(self.llm, messages)
            cleaned_response = clean_response(response.content)
            if _is_oriente_station_nearby_request(user_message) and "museu do oriente" in _normalize_planner_text(cleaned_response):
                cleaned_response = _build_structured_plan_fallback(
                    user_message=user_message,
                    language=language,
                    weather_data=weather_data,
                    transport_data=transport_data,
                    places_data=places_data,
                    events_data=events_data,
                    qa_disclaimers=qa_disclaimers,
                    conversation_context=conversation_context,
                )
            if _planner_response_has_incomplete_museum_day_blocks(user_message, cleaned_response):
                cleaned_response = _build_card_based_itinerary_fallback(
                    user_message=user_message,
                    language=language,
                    weather_data=weather_data,
                    transport_data=transport_data,
                    places_data=places_data,
                    events_data=events_data,
                    qa_disclaimers=qa_disclaimers,
                )
        except Exception:
            fallback = _build_specific_planner_fallback(
                user_message=user_message,
                language=language,
                weather_data=weather_data,
                transport_data=transport_data,
                places_data=places_data,
                events_data=events_data,
                qa_disclaimers=qa_disclaimers,
            ) or _build_structured_plan_fallback(
                user_message=user_message,
                language=language,
                weather_data=weather_data,
                transport_data=transport_data,
                places_data=places_data,
                events_data=events_data,
                qa_disclaimers=qa_disclaimers,
                conversation_context=conversation_context,
            )
            return enforce_multi_day_quality_mode(fallback, user_message, language)

        if (
            _planner_response_requires_fallback(cleaned_response)
            or _planner_response_has_markdown_contract_defects(cleaned_response)
            or not _planner_response_matches_schema(cleaned_response)
            or not _planner_response_has_minimum_user_value(cleaned_response)
        ):
            cleaned_response = _build_specific_planner_fallback(
                user_message=user_message,
                language=language,
                weather_data=weather_data,
                transport_data=transport_data,
                places_data=places_data,
                events_data=events_data,
                qa_disclaimers=qa_disclaimers,
            ) or _build_structured_plan_fallback(
                user_message=user_message,
                language=language,
                weather_data=weather_data,
                transport_data=transport_data,
                places_data=places_data,
                events_data=events_data,
                qa_disclaimers=qa_disclaimers,
                conversation_context=conversation_context,
            )

        grounding_issues = _find_planner_grounding_issues(
            cleaned_response,
            allowed_places=allowed_places,
            accessibility_requested=accessibility_requested,
            accessibility_confirmed=accessibility_confirmed,
        )
        grounding_issues = _add_transport_quality_issue(
            grounding_issues,
            cleaned_response,
            user_message,
            transport_data,
        )

        retry_count = 0
        while grounding_issues and retry_count < 2:
            retry_count += 1
            retry_messages = messages + [
                SystemMessage(
                    content=(
                        "Your previous draft violated the grounding rules. Revise it now.\n"
                        "- Remove any unsupported venue names.\n"
                        "- Remove unsupported accessibility claims.\n"
                        "- Keep only facts supported by the provided data.\n"
                        "- Replace vague transport prose with concrete supported line/stop/transfer details, "
                        "or mark the exact leg as unconfirmed."
                    )
                ),
                HumanMessage(
                    content=(
                        "Revise this itinerary draft and fix the grounding issues below.\n\n"
                        "Grounding issues:\n- " + "\n- ".join(grounding_issues) +
                        "\n\nDraft:\n" + cleaned_response
                    )
                ),
            ]
            response = self._safe_llm_invoke(self.llm, retry_messages)
            cleaned_response = clean_response(response.content)
            if _is_oriente_station_nearby_request(user_message) and "museu do oriente" in _normalize_planner_text(cleaned_response):
                cleaned_response = _build_structured_plan_fallback(
                    user_message=user_message,
                    language=language,
                    weather_data=weather_data,
                    transport_data=transport_data,
                    places_data=places_data,
                    events_data=events_data,
                    qa_disclaimers=qa_disclaimers,
                    conversation_context=conversation_context,
                )
            if _planner_response_has_incomplete_museum_day_blocks(user_message, cleaned_response):
                cleaned_response = _build_card_based_itinerary_fallback(
                    user_message=user_message,
                    language=language,
                    weather_data=weather_data,
                    transport_data=transport_data,
                    places_data=places_data,
                    events_data=events_data,
                    qa_disclaimers=qa_disclaimers,
                )
            grounding_issues = _find_planner_grounding_issues(
                cleaned_response,
                allowed_places=allowed_places,
                accessibility_requested=accessibility_requested,
                accessibility_confirmed=accessibility_confirmed,
            )
            grounding_issues = _add_transport_quality_issue(
                grounding_issues,
                cleaned_response,
                user_message,
                transport_data,
            )
            if not _planner_response_matches_schema(cleaned_response):
                grounding_issues.append("Planner response does not follow the required structured schema.")
            if not _planner_response_has_minimum_user_value(cleaned_response):
                grounding_issues.append("Planner response is structurally present but lacks useful plan content.")

        if (
            _planner_response_requires_fallback(cleaned_response)
            or _planner_response_has_markdown_contract_defects(cleaned_response)
            or not _planner_response_matches_schema(cleaned_response)
            or not _planner_response_has_minimum_user_value(cleaned_response)
            or _planner_response_has_incomplete_museum_day_blocks(user_message, cleaned_response)
        ):
            cleaned_response = _build_specific_planner_fallback(
                user_message=user_message,
                language=language,
                weather_data=weather_data,
                transport_data=transport_data,
                places_data=places_data,
                events_data=events_data,
                qa_disclaimers=qa_disclaimers,
            ) or _build_structured_plan_fallback(
                user_message=user_message,
                language=language,
                weather_data=weather_data,
                transport_data=transport_data,
                places_data=places_data,
                events_data=events_data,
                qa_disclaimers=qa_disclaimers,
                conversation_context=conversation_context,
            )
        elif _planner_response_has_transport_quality_defects(
            cleaned_response,
            user_message,
            transport_data,
        ):
            card_fallback = _build_card_based_itinerary_fallback(
                user_message=user_message,
                language=language,
                weather_data=weather_data,
                transport_data=transport_data,
                places_data=places_data,
                events_data=events_data,
                qa_disclaimers=qa_disclaimers,
            )
            cleaned_response = card_fallback or _append_transport_uncertainty_note(
                cleaned_response,
                language,
            )

        if not _planner_response_has_minimum_user_value(cleaned_response):
            rescue_response = _build_structured_plan_fallback(
                user_message=user_message,
                language=language,
                weather_data=weather_data,
                transport_data=transport_data,
                places_data=places_data,
                events_data=events_data,
                qa_disclaimers=qa_disclaimers,
                conversation_context=conversation_context,
            )
            if _planner_response_has_minimum_user_value(rescue_response):
                cleaned_response = rescue_response

        return enforce_multi_day_quality_mode(
            cleaned_response,
            user_message,
            language,
        )

    def synthesize(self, user_message: str, agent_outputs: Dict[str, str]) -> str:
        """
        Synthesizes outputs from multiple agents into a response.

        Extracts QA disclaimers from internal keys and passes them
        to the planner so data limitations are surfaced to the user.

        Args:
            user_message: Original user query.
            agent_outputs: Dict mapping agent names to their outputs.
                May contain '_qa_disclaimers' (list) from QA validation.

        Returns:
            str: Synthesized response.
        """
        # Extract QA disclaimers before passing to invoke
        qa_disclaimers = agent_outputs.get("_qa_disclaimers")
        if isinstance(qa_disclaimers, str):
            # Safety: if it was stored as a string, wrap in list
            qa_disclaimers = [qa_disclaimers]
        conversation_context = agent_outputs.get("_conversation_context", "")
        if not isinstance(conversation_context, str):
            conversation_context = ""

        return self.invoke(
            user_message=user_message,
            weather_data=agent_outputs.get("weather", ""),
            transport_data=agent_outputs.get("transport", ""),
            places_data=agent_outputs.get("researcher", ""),
            events_data="",  # Events come from researcher too
            qa_disclaimers=qa_disclaimers,
            conversation_context=conversation_context,
        )


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    import io
    import os
    import sys
    from types import SimpleNamespace

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 Planner Agent Test\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")
    counters = {"passed": 0, "failed": 0}

    def _check(condition: bool, label: str) -> None:
        if condition:
            counters["passed"] += 1
            print(f"   \033[1;32m✅ PASS\033[0m: {label}")
        else:
            counters["failed"] += 1
            print(f"   \033[1;31m❌ FAIL\033[0m: {label}")

    mock_weather = """
    Today in Lisbon: ☀️ Clear sky
    🌡️ Temperature: 18°C - 24°C
    🌧️ Precipitation: 10% (unlikely)
    🌤️ UV Index: High - bring sunscreen!
    """
    mock_places = """
    1. 🏛️ **Mosteiro dos Jerónimos** - UNESCO World Heritage
       📍 Belém | 🕐 10:00-17:00 | 💰 €10

    2. 🏛️ **Museu Nacional dos Coches** - Carriage collection
       📍 Belém | 🕐 10:00-18:00 | 💰 €8
    """

    print("\n\033[1m📝 Offline deterministic smoke checks:\033[0m")
    _check(_extract_requested_day_count("Plan 3 days in Lisbon for me.") == 3, "Requested day count parser detects multi-day requests")

    bounded = enforce_multi_day_quality_mode(
        response="### 📅 Lisbon Plan\n- 📅 Day 1\n- Jerónimos\n- 📅 Day 2\n- MAAT",
        user_message="Plan 3 days in Lisbon for me.",
        language="en",
    )
    _check("\n- 📅 Day 2" in bounded and "Day 1" in bounded, "Multi-day guard preserves later day sections")

    mocked_agent = PlannerAgent.__new__(PlannerAgent)
    mocked_agent.system_prompt = "PLANNER PROMPT"
    mocked_agent.llm = object()
    mocked_agent._safe_llm_invoke = lambda _llm, _messages: SimpleNamespace(
        content="### 📅 Day 1\n- **Jerónimos Monastery**\n- Short evidence-supported plan"
    )

    mocked_response = mocked_agent.invoke(
        user_message="Plan 3 days in Lisbon for me.",
        weather_data=mock_weather,
        places_data=mock_places,
        events_data="",
        transport_data="🚇 Metro available",
    )
    print("\n\033[1m🤖 Mocked planner response:\033[0m")
    print(mocked_response[:800] + "..." if len(mocked_response) > 800 else mocked_response)
    _check("Day 1" in mocked_response, "Planner invoke works with a mocked LLM response")

    fallback_agent = PlannerAgent.__new__(PlannerAgent)
    fallback_agent.system_prompt = "PLANNER PROMPT"
    fallback_agent.llm = object()

    def _raise_forced_fallback(_llm, _messages):
        raise RuntimeError("forced planner fallback")

    fallback_agent._safe_llm_invoke = _raise_forced_fallback
    fallback_response = fallback_agent.invoke(
        user_message="Plan 3 days in Lisbon for me.",
        weather_data=mock_weather,
        places_data=mock_places,
        events_data="",
        transport_data="🚇 Metro available",
    )
    _check(
        ("Source" in fallback_response or "Fonte" in fallback_response)
        and ("Itinerary" in fallback_response or "Roteiro" in fallback_response),
        "Planner fallback stays user-facing and structured",
    )

    if os.getenv("LISBOA_RUN_LIVE_PLANNER_TESTS") == "1":
        print("\n\033[1m🌐 Optional live planner smoke:\033[0m")
        try:
            live_agent = PlannerAgent()
            live_response = live_agent.invoke(
                user_message="Plan my morning in Belém",
                weather_data=mock_weather,
                places_data=mock_places,
            )
            print(live_response[:800] + "..." if len(live_response) > 800 else live_response)
            _check(bool(live_response.strip()), "Live planner smoke returned content")
        except Exception as exc:
            _check(False, f"Live planner smoke failed: {exc}")
    else:
        print("\n   ℹ️ Live planner smoke skipped. Set LISBOA_RUN_LIVE_PLANNER_TESTS=1 to enable it.")

    print(f"\n\033[1mSummary:\033[0m Passed={counters['passed']} Failed={counters['failed']}")
    if counters["failed"]:
        raise SystemExit(1)
    print("\n\033[1;32m✅ Planner agent smoke test passed!\033[0m")
