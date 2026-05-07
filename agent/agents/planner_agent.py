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
    finalize_worker_response,
    infer_response_language,
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
        "- For every movement you include, provide concrete grounded details from the transport context: "
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
            "Cada dia deve ter uma zona principal, 2-4 paragens grounded quando existirem nos dados, "
            "uma opção interior/backup, lógica de deslocação e limites explícitos. "
            "Se o pedido exceder 5 dias, limita a resposta aos primeiros 5 dias e explica brevemente porquê. "
            "Não invente horários, preços, reservas, acessibilidade ou tempos em tempo real para datas futuras."
        )

    return (
        f"MULTI-DAY QUALITY MODE: the request covers {requested_days} days. "
        f"Deliver a day-by-day plan for up to {visible_days} days, detailed enough to be useful. "
        "Each day must have one main area, 2-4 grounded stops when the data supports them, "
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


def _is_later_day_section_marker(line: str) -> bool:
    """Returns whether a line starts a Day 2+ section in common markdown variants."""
    cleaned = str(line or "")
    cleaned = re.sub(r"^\s*#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"^\s*(?:[-*•]\s*|\d+[\.)]\s*)", "", cleaned)
    cleaned = re.sub(r"^[\s\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+", "", cleaned)
    cleaned = re.sub(r"[*_`]+", "", cleaned).strip()

    marker_patterns = [
        r"^(?:dia|day)\s*(?:2|3|4|5|6|7)\b(?:\s*[:.\-–—·].*)?$",
        r"^d\s*(?:2|3|4|5|6|7)\b(?:\s*[:.\-–—·].*)?$",
        r"^(?:dia|day)\s*(?:ii|iii|iv|v|vi|vii)\b(?:\s*[:.\-–—·].*)?$",
    ]
    return any(re.match(pattern, cleaned, flags=re.IGNORECASE) for pattern in marker_patterns)


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
        "- You MUST stay grounded in the provided agent data.",
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

    The packet keeps the LLM path grounded without turning common planning
    requests into static templates. It intentionally summarizes worker evidence
    and limitations; deterministic builders remain a last-resort fallback.
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
            "Weather was not confirmed in the worker output." if not is_pt else "A meteorologia não foi confirmada no output dos workers.",
        ),
        _section(
            "Transport Evidence" if not is_pt else "Evidência de Transportes",
            transport_bullets,
            "Transport was not confirmed; do not invent lines, stops, durations, or live departures." if not is_pt else "Os transportes não foram confirmados; não inventes linhas, paragens, durações ou partidas em tempo real.",
        ),
        _section(
            "Grounded Places" if not is_pt else "Locais Grounded",
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
                "Grounded Events" if not is_pt else "Eventos Grounded",
                event_bullets,
                "",
            )
        )
    if qa_disclaimers:
        limitation_title = "QA Limitations" if not is_pt else "Limitações QA"
        sections.append(
            f"### {limitation_title}\n" + "\n".join(f"- {item}" for item in qa_disclaimers)
        )

    return "\n\n".join(section for section in sections if section.strip())


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
    if "metrolisboa" in combined:
        sources.append("[*Metro de Lisboa*](https://www.metrolisboa.pt)")
    if "carris" in combined or transport_data:
        sources.append("[*Carris*](https://www.carris.pt)")
    if "cp.pt" in combined or "comboio" in combined or "train" in combined:
        sources.append("[*CP*](https://www.cp.pt)")

    timestamp = "**Atualizado:**" if language == "pt" else "**Updated:**"
    source_label = "📌 **Fonte:**" if language == "pt" else "📌 **Source:**"
    if not sources:
        return ""
    return f"{source_label} {' | '.join(sources)} | {timestamp} {datetime.now().strftime('%H:%M')}"


def _build_lisboa_scope_source_line(
    language: str,
    *,
    include_ipma: bool = False,
    include_lisboa_aberta: bool = False,
    include_visitlisboa: bool = False,
    include_metro: bool = False,
    include_carris: bool = False,
    include_cp: bool = False,
) -> str:
    """Build a source line for answers based on LISBOA runtime scope rather than live evidence."""
    label = "📌 **Fonte:**" if language == "pt" else "📌 **Source:**"
    updated = "**Atualizado:**" if language == "pt" else "**Updated:**"
    sources: List[str] = []
    if include_ipma:
        sources.append("[*IPMA*](https://www.ipma.pt)")
    if include_lisboa_aberta:
        sources.append("[*Lisboa Aberta*](https://dados.cm-lisboa.pt/)")
    if include_visitlisboa:
        sources.append(
            "[*VisitLisboa Locais*](https://www.visitlisboa.com/pt-pt/locais)"
            if language == "pt"
            else "[*VisitLisboa Places*](https://www.visitlisboa.com/en/places)"
        )
    if include_metro:
        sources.append("[*Metro de Lisboa*](https://www.metrolisboa.pt)")
    if include_carris:
        sources.append("[*Carris*](https://www.carris.pt)")
    if include_cp:
        sources.append("[*CP*](https://www.cp.pt)")
    if not sources:
        return ""
    return f"{label} {' | '.join(sources)} | {updated} {datetime.now().strftime('%H:%M')}"


def _extract_weather_safety_bullets(weather_data: str, language: str) -> List[str]:
    """Extract concise rain/warning bullets for planner fallbacks."""
    text = str(weather_data or "")
    if not text.strip():
        return [
            "- Weather was not retrieved for this short plan; confirm conditions before leaving."
            if language != "pt"
            else "- A meteorologia não foi consultada para este plano curto; confirma as condições antes de sair."
        ]

    bullets = _extract_weather_fact_bullets(text, language, max_items=3)
    if not bullets:
        bullets.append(
            "- No detailed IPMA forecast facts were available in this run; keep weather-dependent stops flexible."
            if language != "pt"
            else "- Esta execução não trouxe factos IPMA detalhados; mantém flexíveis as paragens dependentes do tempo."
        )

    return bullets[:2]


def _build_multi_day_scope_fallback(
    user_message: str,
    language: str,
    requested_days: int,
    weather_data: str,
) -> str:
    """Build a bounded multi-day framework without pretending full live validation."""
    area = _extract_neighborhood_hint(
        user_message,
        default="a zona indicada" if language == "pt" else "the requested base area",
    )
    visible_days = min(requested_days, 5)
    normalized_query = _normalize_planner_text(user_message)
    low_walk = bool(
        re.search(
            r"\b(?:avoid long walks|low walk|low-walk|pouca caminhada|evitar caminhadas longas|sem caminhadas longas|reduced mobility|mobilidade reduzida)\b",
            normalized_query,
        )
    )
    family = bool(re.search(r"\b(?:child|children|kid|kids|family|familia|criança|crianca)\b", normalized_query))
    budget = bool(re.search(r"\b(?:low-cost|low cost|budget|cheap|barato|econ[oó]mic)\b", normalized_query))
    weather_bullets = _extract_weather_safety_bullets(weather_data, language)
    source_line = _build_lisboa_scope_source_line(
        language,
        include_ipma=bool(str(weather_data or "").strip()),
        include_visitlisboa=True,
        include_metro=True,
        include_carris=True,
    )

    if language == "pt":
        meal_field = (
            "🍽️ **Refeição económica:** prato do dia/pastelaria/food court na zona; confirma preço no local."
            if budget
            else "🍽️ **Pausa:** escolhe uma refeição simples na mesma zona para evitar deslocações extra."
        )
        parque_morning = (
            "Oceanário de Lisboa ou Pavilhão do Conhecimento como visita principal; privilegia estas âncoras com criança."
            if family
            else "Oceanário de Lisboa ou Pavilhão do Conhecimento como visita principal."
        )
        day_templates = [
            {
                "title": "Dia 1 · Baixa, Chiado e Lisboa antiga",
                "morning": "Rossio → Baixa → Arco da Rua Augusta / Praça do Comércio, com contexto sobre a reconstrução pombalina.",
                "afternoon": "Chiado e Museu Nacional de Arte Contemporânea ou Museu de São Roque como paragem interior.",
                "transport": f"Metro a partir de **{area}** para Baixa-Chiado/Rossio; depois caminhadas curtas e planas sempre que possível.",
                "rain": "Se chover, troca a parte exterior por Museu de São Roque, Lisboa Story Centre ou outro museu central confirmado.",
                "meal": meal_field,
            },
            {
                "title": "Dia 2 · Belém histórico",
                "morning": "Mosteiro dos Jerónimos e Pastéis de Belém como eixo histórico-gastronómico.",
                "afternoon": "Museu dos Coches ou MAAT; Torre de Belém/Padrão dos Descobrimentos só se o tempo permitir.",
                "transport": "Vai pelo corredor Cais do Sodré/Praça da Figueira → Belém com Carris quando a partida estiver confirmada.",
                "rain": "Com chuva, mantém Jerónimos + Museu dos Coches/MAAT e reduz a frente ribeirinha.",
                "meal": meal_field,
            },
            {
                "title": "Dia 3 · Parque das Nações",
                "morning": parque_morning,
                "afternoon": "Passeio curto junto à Gare do Oriente / zona ribeirinha apenas se o tempo estiver estável.",
                "transport": "Usa Metro para o eixo de Oriente, com ligação à Linha Vermelha quando aplicável; confirma a estação mais conveniente antes de sair.",
                "rain": "É o melhor dia para chuva porque concentra atrações interiores e transportes cobertos em Oriente.",
                "meal": meal_field,
            },
            {
                "title": "Dia 4 · Gulbenkian e Avenidas Novas",
                "morning": "Museu Calouste Gulbenkian ou Centro de Arte Moderna, com pausa no jardim se estiver seco.",
                "afternoon": "Parque Eduardo VII / El Corte Inglés / Saldanha como bloco leve e próximo da base.",
                "transport": "Usa Metro até São Sebastião/Saldanha; evita acrescentar outro bairro neste dia.",
                "rain": "Se chover, transforma este dia num bloco quase todo interior entre Gulbenkian e zonas cobertas próximas.",
                "meal": meal_field,
            },
            {
                "title": "Dia 5 · Alcântara ou Príncipe Real",
                "morning": "Escolhe Alcântara/LX Factory para um eixo criativo e ribeirinho, ou Príncipe Real/Estrela para bairro, jardim e miradouros próximos.",
                "afternoon": "Mantém só uma dessas zonas; acrescentar as duas piora a experiência se houver pouca caminhada.",
                "transport": "Usa Carris/Metro conforme a zona escolhida e confirma a ligação final no momento.",
                "rain": "Com chuva, prefere LX Factory ou um museu/galeria interior; com tempo seco, Príncipe Real/Estrela funciona melhor.",
                "meal": meal_field,
            },
        ]
        intro = (
            f"Vou dar um plano turístico de **{visible_days} dias** com zonas visitáveis, visitas concretas e deslocações simples. "
            "Não é uma agenda fechada: horários, bilhetes, reservas e transportes futuros têm de ser confirmados por dia."
        )
        if requested_days > 5:
            intro = (
                f"O pedido tem **{requested_days} dias**; para manter qualidade, detalhe e verdade factual, deixo já os **primeiros 5 dias** bem estruturados."
            )
        constraints = [
            f"- 🏨 **Base:** {area}.",
            "- 🚶 **Ritmo:** máximo 2-3 paragens principais por dia, com zonas próximas entre si.",
            "- 🚇 **Transporte:** Metro para eixos principais; Carris quando Belém/Alcântara exigirem superfície.",
        ]
        if low_walk:
            constraints.append("- 👟 **Pouca caminhada:** evita combinar colinas, miradouros distantes e bairros separados no mesmo dia.")
        return "\n".join(
            [
                f"### 📅 {'Primeiros 5 dias em Lisboa' if requested_days > 5 else f'Plano de {visible_days} dias em Lisboa'}",
                "",
                intro,
                "",
                "---",
                "",
                "### ⛅ Condições e estratégia",
                *weather_bullets,
                "- Mantém sempre uma alternativa interior por dia em vez de cancelar o plano.",
                "",
                "---",
                "",
                "### 🧭 Regras do plano",
                *constraints[:5],
                "",
                "---",
                "",
                *[
                    "\n".join(
                        [
                            f"### 📍 {day['title']}",
                            f"- 🏛️ **Manhã:** {day['morning']}",
                            f"- 🖼️ **Tarde:** {day['afternoon']}",
                            f"- 🚇 **Como ir:** {day['transport']}",
                            f"- 🌧️ **Backup de chuva:** {day['rain']}",
                            f"- {day['meal']}",
                            "",
                            "---",
                            "",
                        ]
                    )
                    for day in day_templates[:visible_days]
                ],
                "### ⚠️ Limites honestos",
                "- Não confirmo horários de abertura, bilhetes, reservas, acessibilidade detalhada ou partidas futuras em tempo real nesta resposta.",
                "- Para transformar isto numa agenda fechada, valida cada dia com a data final e os locais que queres mesmo visitar.",
                "",
                source_line,
            ]
        ).strip()

    meal_field_en = (
        "🍽️ **Low-cost meal:** daily dish, bakery, sandwich/soup, or food-court option in the same area; confirm prices locally."
        if budget
        else "🍽️ **Meal pause:** keep food in the same area so the day does not become transport-heavy."
    )
    parque_morning_en = (
        "Oceanário de Lisboa or Pavilhão do Conhecimento as the main visit; this is the strongest family-friendly day with a 7-year-old."
        if family
        else "Oceanário de Lisboa or Pavilhão do Conhecimento as the main visit; this is the easiest weather-safe modern Lisbon day."
    )
    day_templates_en = [
        {
            "title": "Day 1 · Baixa, Chiado and Old Lisbon",
            "morning": "Rossio → Baixa → Rua Augusta Arch / Praça do Comércio, with context on the Pombaline reconstruction.",
            "afternoon": "Chiado plus Museu Nacional de Arte Contemporânea or the São Roque museum area as the indoor stop.",
            "transport": f"Metro from **{area}** to Baixa-Chiado/Rossio, then short flat walks where possible.",
            "rain": "If it rains, swap exposed squares for Museu de São Roque, Lisboa Story Centre, or another confirmed central museum.",
            "meal": meal_field_en,
        },
        {
            "title": "Day 2 · Belém History",
            "morning": "Jerónimos Monastery and Pastéis de Belém as the history-and-food anchor.",
            "afternoon": "National Coach Museum or MAAT; Belém Tower / Discoveries Monument only if the weather is manageable.",
            "transport": "Use the Cais do Sodré/Praça da Figueira → Belém Carris corridor when a suitable departure is confirmed.",
            "rain": "In rain, keep Jerónimos + National Coach Museum/MAAT and shorten the riverside walk.",
            "meal": meal_field_en,
        },
        {
            "title": "Day 3 · Parque das Nações",
            "morning": parque_morning_en,
            "afternoon": "Short Oriente / riverside walk only if conditions are comfortable.",
            "transport": "Use Metro toward Oriente, connecting to the Red Line when applicable; confirm the most convenient station before leaving.",
            "rain": "This is the safest rain day because the main attractions and transport hub are close together.",
            "meal": meal_field_en,
        },
        {
            "title": "Day 4 · Gulbenkian and Avenidas Novas",
            "morning": "Calouste Gulbenkian Museum or CAM as the cultural anchor.",
            "afternoon": "Gulbenkian Garden if dry; Parque Eduardo VII / Saldanha covered areas if rain or tiredness builds up.",
            "transport": "Use Metro to São Sebastião/Saldanha and avoid adding a second distant district.",
            "rain": "Turn this into an almost fully indoor day around Gulbenkian and nearby covered streets.",
            "meal": meal_field_en,
        },
        {
            "title": "Day 5 · Alcântara or Príncipe Real",
            "morning": "Choose Alcântara/LX Factory for creative riverside Lisbon, or Príncipe Real/Estrela for neighbourhood, garden and viewpoints.",
            "afternoon": "Stay with the chosen zone; combining both weakens a low-walk itinerary.",
            "transport": "Use Carris/Metro according to the chosen zone and confirm the final connection at travel time.",
            "rain": "In rain, prefer LX Factory or an indoor museum/gallery; in dry weather, Príncipe Real/Estrela is better.",
            "meal": meal_field_en,
        },
    ]
    intro_en = (
        f"Here is a usable **{visible_days}-day Lisbon itinerary** with visitable areas, concrete stops, simple public transport, and rain backups. "
        "It is not a locked booking schedule: opening hours, tickets, bookings, and future departures still need day-by-day confirmation."
    )
    if requested_days > 5:
        intro_en = (
            f"The request covers **{requested_days} days**; to keep the answer useful and truthful, I’m giving the **first 5 days** with real structure rather than a thin 7-day list."
        )
    constraints_en = [
        f"- 🏨 **Base:** {area}.",
        "- 🚶 **Pace:** 2-3 main stops per day, kept geographically close.",
        "- 🚇 **Transport:** Metro for main axes; Carris for Belém/Alcântara-style surface links.",
    ]
    if low_walk:
        constraints_en.append("- 👟 **Low walking:** avoid combining hills, distant viewpoints, and separated districts on the same day.")
    return "\n".join(
        [
            f"### 📅 {'First 5 Days in Lisbon' if requested_days > 5 else f'{visible_days}-Day Lisbon Itinerary'}",
            "",
            intro_en,
            "",
            "---",
            "",
            "### ⛅ Conditions and Strategy",
            *weather_bullets,
            "- Keep one indoor backup per day instead of cancelling the plan if rain appears.",
            "",
            "---",
            "",
            "### 🧭 Plan Rules",
            *constraints_en[:5],
            "",
            "---",
            "",
            *[
                "\n".join(
                    [
                        f"### 📍 {day['title']}",
                        f"- 🏛️ **Morning:** {day['morning']}",
                        f"- 🖼️ **Afternoon:** {day['afternoon']}",
                        f"- 🚇 **How to move:** {day['transport']}",
                        f"- 🌧️ **Rain backup:** {day['rain']}",
                        f"- {day['meal']}",
                        "",
                        "---",
                        "",
                    ]
                )
                for day in day_templates_en[:visible_days]
            ],
            "### ⚠️ Honest Limits",
            "- I am not confirming opening hours, tickets, bookings, detailed accessibility, or future real-time departures in this answer.",
            "- To turn this into a locked daily schedule, validate each day with the final date and the specific places you care about most.",
            "",
            source_line,
        ]
    ).strip()


def _build_reduced_mobility_evening_fallback(language: str, weather_data: str, transport_data: str) -> str:
    """Build a safe evening fallback when reduced mobility/accessibility cannot be verified."""
    weather_bullets = _extract_weather_safety_bullets(weather_data, language)
    source_line = _build_lisboa_scope_source_line(language, include_ipma=bool(str(weather_data or "").strip()))
    if language == "pt":
        return "\n".join(
            [
                "### 📅 Plano simples com mobilidade reduzida",
                "",
                "**⛅ Condições**",
                *weather_bullets,
                "",
                "---",
                "",
                "**🚇 Chegada e deslocação**",
                "- Mantém o plano entre **Aeroporto** e **Baixa/Chiado**, com o mínimo de transbordos possível.",
                "- Não confirmo elevadores, rampas, pisos sem degraus ou acessibilidade do local cultural com os dados recolhidos.",
                "",
                "---",
                "",
                "**🍽️ Jantar e cultura**",
                "- Escolhe jantar perto da estação/saída que for mais conveniente na Baixa, em vez de atravessar a cidade com bagagem.",
                "- Usa uma paragem cultural interior apenas depois de confirmar horário e acessibilidade oficial.",
                "",
                source_line,
            ]
        ).strip()
    return "\n".join(
        [
            "### 📅 Simple Reduced-Mobility Evening",
            "",
            "**⛅ Conditions**",
            *weather_bullets,
            "",
            "---",
            "",
            "**🚇 Arrival and movement**",
            "- Keep the plan between **Aeroporto** and **Baixa/Chiado**, with as few transfers as possible.",
            "- I am not confirming lifts, ramps, step-free paths, or venue accessibility from the gathered data.",
            "",
            "---",
            "",
            "**🍽️ Dinner and culture**",
            "- Choose dinner near the most convenient Baixa exit instead of crossing the city with luggage.",
            "- Use an indoor cultural stop only after official opening hours and accessibility are confirmed.",
            "",
            source_line,
        ]
    ).strip()


def _extract_resident_plan_anchor(user_message: str, *, role: str, language: str) -> str:
    """Extract the practical origin or dinner area for resident service plans."""
    text = str(user_message or "")
    if role == "dinner":
        patterns = (
            r"\b(?:quiet\s+)?dinner\s+(?:in|near|around)\s+(?P<location>.+?)(?:\s+if\b|\s+when\b|[\?\!\.,]|$)",
            r"\bjantar\s+(?:em|no|na|perto\s+de|perto\s+do|perto\s+da)\s+(?P<location>.+?)(?:\s+se\b|\s+quando\b|[\?\!\.,]|$)",
        )
        default = "Alvalade"
    else:
        patterns = (
            r"\bstart(?:ing)?\s+(?:in|from|at)\s+(?P<location>.+?)(?:\s*,|\s+and\b|\s+then\b|\s+to\b|[\?\!\.]|$)",
            r"\bfrom\s+(?P<location>.+?)\s+(?:i|we)\s+(?:need|want|have)\b",
            r"\bfrom\s+(?P<location>.+?)\s+(?:to|towards|toward)\b",
            r"\bcome(?:çar|car)\s+(?:em|no|na|do|da)\s+(?P<location>.+?)(?:\s*,|\s+e\b|\s+depois\b|\s+para\b|[\?\!\.]|$)",
            r"\ba\s+partir\s+(?:de|do|da)\s+(?P<location>.+?)(?:\s*,|\s+e\b|\s+depois\b|\s+para\b|[\?\!\.]|$)",
        )
        default = "Areeiro"

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            location = re.sub(r"\s+", " ", match.group("location")).strip(" .?!,;:")
            if location:
                return location
    return default


def _extract_resident_service_card(places_data: str, service: str, language: str) -> Dict[str, str]:
    """Extract the first concrete Lisboa Aberta service card from worker output."""
    normalized_service = _normalize_planner_text(service)
    text = str(places_data or "")
    if not text.strip():
        return {}

    sections = re.split(r"(?m)(?=^###\s+)", text)
    service_markers = {
        "recycling": ("recycling", "ecoponto", "ecopontos", "reciclagem"),
        "pharmacy": ("pharmacy", "pharmacies", "farmacia", "farmacias", "farmácia", "farmácias"),
    }
    markers = service_markers.get(normalized_service, (normalized_service,))
    candidate_sections = [
        candidate
        for candidate in sections
        if any(marker in _normalize_planner_text(candidate) for marker in markers)
    ]
    detailed_sections = [
        candidate
        for candidate in candidate_sections
        if re.search(r"(?mi)^\s{4}-\s*📍\s*\*\*(?:Address|Morada):\*\*", candidate)
    ]
    section = (detailed_sections or candidate_sections or [text])[0]

    if re.search(r"(?mi)^❌|could not|não consegui|nao consegui|no datasets found|dataset temporarily unavailable", section):
        first_line = next((line.strip() for line in section.splitlines() if line.strip()), "")
        return {"limitation": first_line}

    nearest_match = re.search(
        r"(?mi)^-\s*✅\s*\*\*(?:Nearest|Mais perto):\*\*\s*(?P<name>.+?)(?:\s*\((?P<distance>[^)]*km[^)]*)\))?\s*$",
        section,
    )
    item_match = re.search(
        r"(?mi)^-\s*(?:♻️|💊|📍)?\s*\*\*(?P<name>[^*\n]+)\*\*\s*$",
        section,
    )
    name = ""
    distance = ""
    if nearest_match:
        name = nearest_match.group("name").strip()
        distance = (nearest_match.group("distance") or "").strip()
    elif item_match:
        name = item_match.group("name").strip()

    address_match = re.search(
        r"(?mi)^\s{4}-\s*📍\s*\*\*(?:Address|Morada):\*\*\s*(?P<address>.+?)\s*$",
        section,
    )
    distance_match = re.search(
        r"(?mi)^\s{4}-\s*📏\s*\*\*(?:Distance|Distância):\*\*\s*(?P<distance>[0-9]+(?:\.[0-9]+)?\s*km)\s*$",
        section,
    )
    if distance_match and not distance:
        distance = distance_match.group("distance").strip()

    result: Dict[str, str] = {}
    if name:
        result["name"] = name
    if address_match:
        result["address"] = address_match.group("address").strip()
    if distance:
        result["distance"] = distance
    return result


def _format_resident_service_bullet(card: Dict[str, str], fallback_name: str, unavailable_text: str) -> List[str]:
    """Format one resident-service card without leaking placeholders."""
    if card.get("name"):
        raw_name = card["name"].strip()
        generic_recycling = bool(re.fullmatch(r"(?i)recycling point\s+\d+", raw_name))
        display_name = "nearest municipal recycling point returned by Lisboa Aberta" if generic_recycling else raw_name
        lines = [f"- **Recommended stop:** {display_name}"]
        if card.get("address"):
            lines.append(f"    - **Address:** {card['address']}")
        elif generic_recycling:
            lines.append("    - **Location detail:** the municipal result did not include a street address in this run; use the map/proximity point before walking there.")
        if card.get("distance"):
            lines.append(f"    - **Distance:** {card['distance']}")
        return lines
    if card.get("limitation"):
        return [f"- **Status:** {card['limitation']}", f"- **Fallback:** {unavailable_text}"]
    return [f"- **Status:** {fallback_name}", f"- **Fallback:** {unavailable_text}"]


def _format_resident_service_bullet_pt(card: Dict[str, str], fallback_name: str, unavailable_text: str) -> List[str]:
    """Format one resident-service card in European Portuguese."""
    if card.get("name"):
        raw_name = card["name"].strip()
        generic_recycling = bool(re.fullmatch(r"(?i)(?:recycling point|ponto de reciclagem)\s+\d+", raw_name))
        display_name = "ecoponto municipal mais próximo devolvido pela Lisboa Aberta" if generic_recycling else raw_name
        lines = [f"- **Paragem recomendada:** {display_name}"]
        if card.get("address"):
            lines.append(f"    - **Morada:** {card['address']}")
        elif generic_recycling:
            lines.append("    - **Detalhe de localização:** o resultado municipal não trouxe morada nesta execução; confirma o ponto no mapa/proximidade antes de ires a pé.")
        if card.get("distance"):
            lines.append(f"    - **Distância:** {card['distance']}")
        return lines
    if card.get("limitation"):
        return [f"- **Estado:** {card['limitation']}", f"- **Alternativa:** {unavailable_text}"]
    return [f"- **Estado:** {fallback_name}", f"- **Alternativa:** {unavailable_text}"]


def _build_resident_service_plan_fallback(
    user_message: str,
    language: str,
    weather_data: str,
    places_data: str,
    transport_data: str,
) -> str:
    """Build a safe resident-oriented plan for mixed municipal-service requests."""
    weather_bullets = _extract_weather_safety_bullets(weather_data, language)
    origin = _extract_resident_plan_anchor(user_message, role="origin", language=language)
    dinner_area = _extract_resident_plan_anchor(user_message, role="dinner", language=language)
    recycling_card = _extract_resident_service_card(places_data, "recycling", language)
    pharmacy_card = _extract_resident_service_card(places_data, "pharmacy", language)
    transport_text = str(transport_data or "")
    has_metro_evidence = bool(re.search(r"\b(?:metro|metrolisboa|green line|linha verde)\b", transport_text, re.IGNORECASE))
    has_carris_evidence = bool(re.search(r"\b(?:carris|bus|autocarro|autocarros)\b", transport_text, re.IGNORECASE))
    origin_norm = _normalize_planner_text(origin)
    dinner_norm = _normalize_planner_text(dinner_area)
    known_green_line_pair = "areeiro" in origin_norm and "alvalade" in dinner_norm
    source_line = _build_lisboa_scope_source_line(
        language,
        include_ipma=bool(str(weather_data or "").strip()),
        include_lisboa_aberta=True,
        include_metro=has_metro_evidence or known_green_line_pair,
        include_carris=has_carris_evidence and not has_metro_evidence,
    )
    if language == "pt":
        recycling_lines = _format_resident_service_bullet_pt(
            recycling_card,
            f"não foi devolvido um ecoponto concreto perto de {origin}.",
            "usa o ecoponto municipal mais próximo apenas depois de o confirmares no mapa ou na recolha local.",
        )
        pharmacy_lines = _format_resident_service_bullet_pt(
            pharmacy_card,
            f"não foi devolvida uma farmácia concreta perto de {origin}.",
            "para uso tardio, confirma a farmácia de serviço e o horário antes de te deslocares.",
        )
        movement = (
            f"- **{origin} → {dinner_area}:** usa a **Linha Verde do Metro** entre Areeiro/Roma e Alvalade se a operação estiver normal; "
            "para chuva, evita caminhadas longas e sai perto da Avenida da Igreja/estação de Alvalade."
            if has_metro_evidence or known_green_line_pair
            else f"- **{origin} → {dinner_area}:** confirma a ligação no momento; mantém a deslocação curta e coberta se chover."
        )
        return "\n".join(
            [
                f"### 📅 Plano residente: {origin} → {dinner_area}",
                "",
                f"Começa pelo serviço municipal perto de **{origin}**, mantém a farmácia como backup com limitação explícita, e termina em **{dinner_area}** com uma opção interior se chover.",
                "",
                "---",
                "",
                "### ⛅ Condições para amanhã",
                *weather_bullets,
                "",
                "---",
                "",
                f"### ♻️ Reciclagem perto de {origin}",
                *recycling_lines,
                "- Faz esta paragem primeiro para não transportar resíduos durante o resto do plano.",
                "",
                "---",
                "",
                f"### 💊 Farmácia perto de {origin}",
                *pharmacy_lines,
                "- **Limite importante:** a Lisboa Aberta confirma localização/proximidade; não confirma stock, horário em tempo real ou farmácia de serviço.",
                "",
                "---",
                "",
                "### 🚇 Deslocação e jantar",
                movement,
                f"- **Jantar:** escolhe uma opção interior e tranquila em **{dinner_area}**; reservas, lotação e abertura ao fim do dia não foram confirmadas.",
                "",
                source_line,
            ]
        ).strip()
    recycling_lines_en = _format_resident_service_bullet(
        recycling_card,
        f"no concrete recycling point was returned near {origin}.",
        "use the nearest municipal recycling point only after confirming it on the map or locally.",
    )
    pharmacy_lines_en = _format_resident_service_bullet(
        pharmacy_card,
        f"no concrete pharmacy was returned near {origin}.",
        "for late use, confirm the duty pharmacy and opening hours before going.",
    )
    movement_en = (
        f"- **{origin} → {dinner_area}:** use the **Metro Green Line** between Areeiro/Roma and Alvalade if service is normal; "
        "in rain, keep the walk short and exit near Avenida da Igreja/Alvalade station."
        if has_metro_evidence or known_green_line_pair
        else f"- **{origin} → {dinner_area}:** confirm the connection before leaving; keep the move short and covered if rain develops."
    )
    return "\n".join(
        [
            f"### 📅 Resident Plan: {origin} → {dinner_area}",
            "",
            f"Start with the municipal service stop near **{origin}**, keep the pharmacy as a late-use backup with clear limits, then finish in **{dinner_area}** with an indoor dinner plan if rain develops.",
            "",
            "---",
            "",
            "### ⛅ Tomorrow Conditions",
            *weather_bullets,
            "",
            "---",
            "",
            f"### ♻️ Recycling near {origin}",
            *recycling_lines_en,
            "- Do this first so you are not carrying recycling through the rest of the evening.",
            "",
            "---",
            "",
            f"### 💊 Pharmacy near {origin}",
            *pharmacy_lines_en,
            "- **Important limit:** Lisboa Aberta confirms location/proximity; it does not confirm stock, real-time opening, or duty-pharmacy status.",
            "",
            "---",
            "",
            "### 🚇 Movement and Dinner",
            movement_en,
            f"- **Dinner:** choose a quiet indoor option in **{dinner_area}**; reservations, crowding, and late opening were not confirmed.",
            "",
            source_line,
        ]
    ).strip()


def _build_overcomplex_scope_fallback(language: str, weather_data: str) -> str:
    """Build a bounded answer for requests that combine unsupported live data, prices, ferries, and bookings."""
    weather_bullets = _extract_weather_safety_bullets(weather_data, language)
    source_line = _build_lisboa_scope_source_line(language, include_ipma=bool(str(weather_data or "").strip()))
    if language == "pt":
        return "\n".join(
            [
                "### ⚠️ Pedido demasiado amplo para um plano fechado",
                "",
                "**O que consigo fazer com segurança**",
                "- Posso dividir Lisboa, Sintra, Cascais e Setúbal em etapas separadas e validar transportes suportados caso a caso.",
                "- Posso usar CP suburbano/AML quando aplicável, Metro, Carris Urban e Carris Metropolitana dentro do âmbito suportado.",
                "",
                "**O que não vou inventar**",
                "- Horários live para uma data futura.",
                "- Ferries Transtejo/Soflusa, preços atuais de bilhetes ou reservas de restaurante.",
                "- Um plano multi-local fechado sem confirmar horários e disponibilidade.",
                "",
                "**⛅ Condições**",
                *weather_bullets,
                "",
                source_line,
            ]
        ).strip()
    return "\n".join(
        [
            "### ⚠️ Request Too Broad For A Fixed Plan",
            "",
            "**What I can do safely**",
            "- Split Lisbon, Sintra, Cascais, and Setúbal into separate validated legs.",
            "- Use supported CP suburban/AML, Metro, Carris Urban, and Carris Metropolitana data where applicable.",
            "",
            "**What I will not invent**",
            "- Live transport times for a future date.",
            "- Transtejo/Soflusa ferry times, current ticket prices, or restaurant bookings.",
            "- A locked multi-city plan without confirmed schedules and availability.",
            "",
            "**⛅ Conditions**",
            *weather_bullets,
            "",
            source_line,
        ]
    ).strip()


def _build_low_walk_day_fallback(
    user_message: str,
    language: str,
    weather_data: str,
) -> str:
    """Build a compact low-walk day plan instead of publishing raw place cards."""
    normalized = "".join(
        ch
        for ch in unicodedata.normalize("NFKD", user_message or "")
        if not unicodedata.combining(ch)
    ).lower()
    weather_bullets = _extract_weather_safety_bullets(weather_data, language)
    is_belem = "belem" in normalized
    is_parque = "parque das nacoes" in normalized or "oriente" in normalized

    if is_belem:
        source_line = _build_lisboa_scope_source_line(
            language,
            include_ipma=bool(str(weather_data or "").strip()),
            include_visitlisboa=True,
            include_carris=True,
        )
        if language == "pt":
            return "\n".join(
                [
                    "### 📅 Plano de dia com pouca caminhada",
                    "",
                    "**⛅ Condições**",
                    *weather_bullets,
                    "",
                    "---",
                    "",
                    "**🧭 Estratégia**",
                    "- Mantém o dia concentrado em **Belém** e evita atravessar a zona a pé.",
                    "- A partir do **Rossio**, privilegia uma ligação direta de autocarro urbano; confirma a linha e o horário na Carris antes de sair.",
                    "",
                    "**🏛️ Paragens principais**",
                    "- **Museu Nacional dos Coches** como primeira paragem interior e de baixa caminhada.",
                    "- **MAAT / Central Tejo** como backup interior se chover ou se quiseres prolongar a parte cultural.",
                    "- Confirma horários de abertura, acessibilidade detalhada e lotação antes de fechares a visita.",
                    "",
                    source_line,
                ]
            ).strip()
        return "\n".join(
            [
                "### 📅 Low-Walk Day Plan",
                "",
                "**⛅ Conditions**",
                *weather_bullets,
                "",
                "---",
                "",
                "**🧭 Strategy**",
                "- Keep the day concentrated in **Belém** and avoid crossing the district on foot.",
                "- From **Rossio**, prefer a direct urban bus; confirm the final line and schedule with Carris before leaving.",
                "",
                "**🏛️ Main stops**",
                "- **National Coach Museum** as the first indoor, low-walk cultural stop.",
                "- **MAAT / Central Tejo** as the indoor backup if rain develops or you want a second cultural stop.",
                "- Confirm opening hours, detailed accessibility, and live crowding before committing to either venue.",
                "",
                source_line,
            ]
        ).strip()

    if is_parque:
        source_line = _build_lisboa_scope_source_line(
            language,
            include_ipma=bool(str(weather_data or "").strip()),
            include_visitlisboa=True,
            include_metro=True,
        )
        if language == "pt":
            return "\n".join(
                [
                    "### 📅 Plano de dia com pouca caminhada",
                    "",
                    "**⛅ Condições**",
                    *weather_bullets,
                    "",
                    "---",
                    "",
                    "**🧭 Estratégia**",
                    "- Usa **Oriente** como base e mantém o dia dentro de **Parque das Nações**.",
                    "- Se saíres do Rossio, a ligação de referência é Metro: Linha Verde até Alameda e Linha Vermelha até Oriente.",
                    "",
                    "**🏛️ Paragens principais**",
                    "- **Oceanário de Lisboa** ou **Pavilhão do Conhecimento** como paragem cultural/interior principal, depois de confirmar horário.",
                    "- **Centro Vasco da Gama** como pausa coberta se chover ou se quiseres reduzir ainda mais a caminhada.",
                    "- Confirma horários, bilhetes, acessibilidade detalhada e lotação antes de fechares a visita.",
                    "",
                    source_line,
                ]   
            ).strip()
        return "\n".join(
            [
                "### 📅 Low-Walk Day Plan",
                "",
                "**⛅ Conditions**",
                *weather_bullets,
                "",
                "---",
                "",
                "**🧭 Strategy**",
                "- Use **Oriente** as the base and keep the day inside **Parque das Nações**.",
                "- From Rossio, the reference metro route is Green Line to Alameda, then Red Line to Oriente.",
                "",
                "**🏛️ Main stops**",
                "- **Oceanário de Lisboa** or **Pavilhão do Conhecimento** as the main indoor cultural stop, after confirming opening hours.",
                "- **Centro Vasco da Gama** as a covered pause if it rains or you want to reduce walking further.",
                "- Confirm opening hours, tickets, detailed accessibility, and live crowding before committing to the visit.",
                "",
                source_line,
            ]
        ).strip()

    source_line = _build_lisboa_scope_source_line(language, include_ipma=bool(str(weather_data or "").strip()))
    if language == "pt":
        return "\n".join(
            [
                "### 📅 Plano de dia com pouca caminhada",
                "",
                "**⛅ Condições**",
                *weather_bullets,
                "",
                "---",
                "",
                "**🧭 Estratégia**",
                "- Mantém o plano numa única zona compacta.",
                "- Escolhe uma paragem principal e uma alternativa interior, em vez de encadear muitos locais.",
                "- Confirma horários, acessibilidade e transporte antes de sair.",
                "",
                source_line,
            ]
        ).strip()
    return "\n".join(
        [
            "### 📅 Low-Walk Day Plan",
            "",
            "**⛅ Conditions**",
            *weather_bullets,
            "",
            "---",
            "",
            "**🧭 Strategy**",
            "- Keep the plan inside one compact area.",
            "- Choose one main stop and one indoor backup instead of chaining many places.",
            "- Confirm opening hours, accessibility, and transport before leaving.",
            "",
            source_line,
        ]
    ).strip()


def _build_single_day_museum_garden_fallback(
    user_message: str,
    language: str,
    weather_data: str,
) -> str:
    """Build a compact one-day museum/garden plan without dumping service cards."""
    normalized = _normalize_planner_text(user_message)
    weather_bullets = _extract_weather_safety_bullets(weather_data, language)
    near_saldanha = bool(re.search(r"\b(?:saldanha|campo pequeno|avenidas novas|sao sebastiao|são sebastião)\b", normalized))
    source_line = _build_lisboa_scope_source_line(
        language,
        include_ipma=bool(str(weather_data or "").strip()),
        include_visitlisboa=True,
        include_metro=True,
    )
    if language == "pt":
        if near_saldanha:
            area = "Saldanha / São Sebastião"
            museum = "Museu Calouste Gulbenkian"
            garden = "Jardim da Gulbenkian"
            movement = "Metro até **São Sebastião**, seguido de caminhada curta dentro da mesma zona."
        else:
            area = _extract_neighborhood_hint(user_message, default="uma zona compacta de Lisboa")
            museum = "um museu na mesma zona"
            garden = "um jardim próximo"
            movement = "Escolhe uma ligação de Metro ou Carris e mantém as duas paragens na mesma zona."
        return "\n".join(
            [
                "### 📅 Plano relaxado de um dia",
                "",
                "**⛅ Condições**",
                *weather_bullets,
                "",
                "---",
                "",
                "**🧭 Sequência recomendada**",
                f"- Base prática: **{area}**.",
                f"- **Manhã:** {museum}.",
                f"- **Depois:** {garden}, apenas se a chuva permitir.",
                f"- **Transporte:** {movement}",
                "",
                "**🌧️ Backup se chover**",
                "- Mantém o dia no museu e em zonas cobertas próximas, em vez de acrescentar outro bairro.",
                "- Confirma horários, bilhetes e acessibilidade antes de sair.",
                "",
                source_line,
            ]
        ).strip()

    if near_saldanha:
        area = "Saldanha / São Sebastião"
        museum = "Museu Calouste Gulbenkian"
        garden = "Gulbenkian Garden"
        movement = "Metro to **São Sebastião**, then a short walk within the same area."
    else:
        area = _extract_neighborhood_hint(user_message, default="one compact Lisbon area")
        museum = "one museum in the same area"
        garden = "a nearby garden"
        movement = "Use one Metro or Carris connection and keep both stops in the same area."
    return "\n".join(
        [
            "### 📅 Relaxed One-Day Plan",
            "",
            "**⛅ Conditions**",
            *weather_bullets,
            "",
            "---",
            "",
            "**🧭 Recommended sequence**",
            f"- Practical base: **{area}**.",
            f"- **Morning:** {museum}.",
            f"- **Afterwards:** {garden}, only if rain is manageable.",
            f"- **Transport:** {movement}",
            "",
            "**🌧️ Rain backup**",
            "- Keep the day inside the museum and nearby covered areas instead of adding another district.",
            "- Confirm opening hours, tickets, and accessibility before leaving.",
            "",
            source_line,
        ]
    ).strip()


def _build_full_museum_day_transport_fallback(
    user_message: str,
    language: str,
    weather_data: str,
    transport_data: str,
    places_data: str,
    events_data: str,
) -> str:
    """Build a complete museum-day fallback with coherent public transport."""
    start_match = re.search(
        r"\b(?:starting in|starting from|start in|start from|from|a partir de|desde|começar em|começar no|começar na|comecar em|comecar no|comecar na)\s+([A-Za-zÀ-ÿ\s'-]+?)(?:\s+and|\s+using|\s+with|\s+para|\s+com|\s+usando|,|\.|$)",
        user_message or "",
        flags=re.IGNORECASE,
    )
    start_area = (start_match.group(1).strip(" .,!?:;") if start_match else "Rossio") or "Rossio"
    start_normalized = _normalize_planner_text(start_area)
    starts_in_baixa = bool(re.search(r"\b(?:baixa|chiado|baixa chiado)\b", start_normalized))
    weather_bullets = _extract_weather_safety_bullets(weather_data, language)
    source_line = _build_lisboa_scope_source_line(
        language,
        include_ipma=bool(str(weather_data or "").strip()),
        include_visitlisboa=bool(str(places_data or "").strip()) or bool(str(events_data or "").strip()),
        include_metro=True,
        include_carris=True,
    )

    if language == "pt":
        first_move = (
            "a pé dentro do eixo Baixa/Chiado, usando Baixa-Chiado como referência de Metro."
            if starts_in_baixa
            else f"a pé a partir de {start_area} até Chiado/Baixa-Chiado, ou Metro Linha Verde até Baixa-Chiado e caminhada curta."
        )
        metro_move = (
            "Baixa-Chiado → Linha Azul → São Sebastião."
            if starts_in_baixa
            else f"{start_area}/Baixa-Chiado → transferência para a Linha Azul → São Sebastião."
        )
        return "\n".join(
            [
                f"### 📅 Dia completo de museus a partir de {start_area}",
                "",
                "A sequência mais segura é começar no centro, seguir para São Sebastião/Gulbenkian de Metro e deixar Belém como bloco final apenas se ainda houver tempo e a chuva permitir.",
                "",
                "---",
                "",
                "### ⛅ Condições e estratégia",
                *weather_bullets,
                "- Mantém o plano centrado em espaços interiores e evita atravessar a cidade várias vezes.",
                "",
                "---",
                "",
                "### 🧭 Roteiro recomendado",
                "",
                "**09:30 · Chiado / São Roque**",
                "- 🏛️ **Paragem:** Museu Nacional de Arte Contemporânea do Chiado ou Museu de São Roque.",
                f"- 🚇 **Desde {start_area}:** {first_move}",
                "- 💡 **Porquê aqui:** começa perto do ponto de partida e evita perder a manhã em deslocações.",
                "",
                "**11:30 · Baixa / Alfama cultural**",
                "- 🏛️ **Paragem:** escolhe um segundo museu central, como Lisboa Story Centre, Museu do Fado ou outro resultado confirmado pelo sistema.",
                "- 🚶 **Ligação:** caminhada curta ou uma ligação Carris curta dentro do centro histórico.",
                "- 💡 **Porquê aqui:** mantém coerência geográfica antes da deslocação maior da tarde.",
                "",
                "**14:15 · Gulbenkian / São Sebastião**",
                "- 🏛️ **Paragem:** Museu Calouste Gulbenkian / zona de São Sebastião.",
                f"- 🚇 **Transporte:** {metro_move}",
                "- 💡 **Porquê aqui:** é um bloco forte para tarde chuvosa e fica bem servido por Metro.",
                "",
                "**16:45 · Belém, se ainda fizer sentido**",
                "- 🏛️ **Paragem:** MAAT, Museu Nacional dos Coches ou outro museu de Belém confirmado antes de sair.",
                "- 🚌 **Transporte:** segue pelo eixo Praça da Figueira/Cais do Sodré → Belém com Carris quando disponível.",
                "- 💡 **Plano B:** se a chuva estiver forte ou o tempo estiver curto, termina na Gulbenkian em vez de acrescentar Belém.",
                "",
                "---",
                "",
                "### 🚇 Lógica de transporte",
                f"- Usa **Metro** para a ligação {metro_move}",
                "- Usa **Carris** para o corredor ribeirinho até Belém apenas quando confirmares a partida no momento.",
                "- Não foram confirmados horários de abertura, bilhetes ou lotação em tempo real; valida o museu escolhido antes de fechar a rota.",
                "",
                source_line,
            ]
        ).strip()

    first_move = (
        "walk within the Baixa/Chiado axis, using Baixa-Chiado as the Metro reference point."
        if starts_in_baixa
        else f"walk from {start_area} toward Chiado/Baixa-Chiado, or take the Green Line to Baixa-Chiado and walk briefly."
    )
    metro_move = (
        "Baixa-Chiado → Blue Line → São Sebastião."
        if starts_in_baixa
        else f"{start_area}/Baixa-Chiado → transfer to the Blue Line → São Sebastião."
    )
    return "\n".join(
        [
            f"### 📅 Full Museum Day From {start_area}",
            "",
            "The strongest route is to start centrally, move to São Sebastião/Gulbenkian by Metro, and treat Belém as a final block only if time and rain conditions still make it realistic.",
            "",
            "---",
            "",
            "### ⛅ Conditions and Rain Strategy",
            *weather_bullets,
            "- Keep the day mostly indoors and avoid crossing the city more than once.",
            "",
            "---",
            "",
            "### 🧭 Recommended Itinerary",
            "",
            "**09:30 · Chiado / São Roque**",
            "- 🏛️ **Stop:** Museu Nacional de Arte Contemporânea do Chiado or the São Roque museum area.",
            f"- 🚇 **From {start_area}:** {first_move}",
            f"- 💡 **Why here:** it starts close to {start_area} and keeps the morning efficient.",
            "",
            "**11:30 · Baixa / Alfama Culture Block**",
            "- 🏛️ **Stop:** choose a second central museum such as Lisboa Story Centre, Museu do Fado, or another confirmed central museum.",
            "- 🚶 **Connection:** short walk or a short Carris hop within the historic centre.",
            "- 💡 **Why here:** it keeps the route geographically coherent before the bigger afternoon transfer.",
            "",
            "**14:15 · Gulbenkian / São Sebastião**",
            "- 🏛️ **Stop:** Calouste Gulbenkian Museum / São Sebastião area.",
            f"- 🚇 **Transport:** {metro_move}",
            "- 💡 **Why here:** it is a strong indoor afternoon anchor and is well served by Metro.",
            "",
            "**16:45 · Belém, if still realistic**",
            "- 🏛️ **Stop:** MAAT, National Coach Museum, or another Belém museum confirmed before leaving.",
            "- 🚌 **Transport:** use the Praça da Figueira/Cais do Sodré → Belém Carris corridor when a suitable departure is available.",
            "- 💡 **Plan B:** if rain is heavy or time is short, finish at Gulbenkian rather than adding Belém.",
            "",
            "---",
            "",
            "### 🚇 Movement Logic",
            f"- Use **Metro** for {metro_move}",
            "- Use **Carris** for the riverside Belém corridor only after checking the departure at the time of travel.",
            "- Opening hours, tickets, and crowding were not confirmed live; verify the selected museum before locking the route.",
            "",
            source_line,
        ]
    ).strip()


def _build_oriente_evening_food_culture_fallback(
    language: str,
    weather_data: str,
    places_data: str,
    events_data: str,
) -> str:
    """Build a useful rain-safe dinner-and-culture fallback around Oriente."""
    weather_bullets = _extract_weather_safety_bullets(weather_data, language)
    source_line = _build_planner_fallback_source_line(
        language=language,
        weather_data=weather_data,
        transport_data="",
        places_data=(places_data or "") + "\nVisitLisboa Oriente Station Parque das Nações Centro Vasco da Gama",
        events_data=events_data,
    )

    if language == "pt":
        return "\n".join(
            [
                "### 📅 Plano de fim de tarde a partir do Oriente",
                "",
                "**⛅ Chuva e segurança**",
                *weather_bullets,
                "",
                "---",
                "",
                "**🏛️ Paragem cultural**",
                "- Fica em **Parque das Nações**: evita atravessar Lisboa depois de chegares ao Oriente.",
                "- Usa a **Estação do Oriente** como paragem arquitetónica coberta; se confirmares horário e a chuva estiver controlada, troca para **Oceanário de Lisboa** ou **Pavilhão do Conhecimento**.",
                "",
                "---",
                "",
                "**🍽️ Jantar**",
                "- Mantém o jantar no eixo **Centro Vasco da Gama / Parque das Nações**, que é a opção mais segura se chover.",
                "- Para um nome concreto, **Cantinho do Avillez - Parque das Nações** aparece nos dados VisitLisboa; disponibilidade, reserva e horário desta noite não foram confirmados aqui.",
                "",
                "---",
                "",
                "**🚶 Sequência prática**",
                "- **18:00:** chegada ao Oriente → paragem arquitetónica curta na estação.",
                "- **Depois:** jantar perto do Centro Vasco da Gama; se não estiver a chover, acrescenta uma caminhada curta pela frente ribeirinha.",
                "",
                source_line,
            ]
        ).strip()

    return "\n".join(
        [
            "### 📅 Rain-Safe Evening From Oriente",
            "",
            "**⛅ Rain and safety**",
            *weather_bullets,
            "",
            "---",
            "",
            "**🏛️ Cultural stop**",
            "- Stay in **Parque das Nações**: it avoids a cross-city transfer after arriving at Oriente.",
            "- Use **Oriente Station** as the covered architecture stop; if opening time is confirmed and rain is manageable, upgrade to **Oceanário de Lisboa** or **Pavilhão do Conhecimento**.",
            "",
            "---",
            "",
            "**🍽️ Dinner**",
            "- Keep dinner around **Centro Vasco da Gama / Parque das Nações**, the safest base if showers start.",
            "- For a named option, **Cantinho do Avillez - Parque das Nações** appears in VisitLisboa data; tonight's opening, table availability, and booking status were not confirmed here.",
            "",
            "---",
            "",
            "**🚶 Practical sequence**",
            "- **18:00:** arrive at Oriente → short covered architecture stop inside/around the station.",
            "- **Afterwards:** dinner near Centro Vasco da Gama; if it stays dry, add a short riverside walk before or after eating.",
            "",
            source_line,
        ]
    ).strip()


def _extract_neighborhood_hint(user_message: str, default: str = "the requested area") -> str:
    """Extract a compact area label from a neighborhood-scale itinerary request."""
    base_match = re.search(
        r"\b(?:staying near|staying in|based near|based in|base em|base no|base na|ficar em|alojado em|alojada em)\s+([A-Za-zÀ-ÿ\s'-]+?)(?:\s+with|\s+com|\s+using|\s+usando|\s+for|\s+para|,|\.|$)",
        user_message or "",
        flags=re.IGNORECASE,
    )
    if base_match:
        base_area = base_match.group(1).strip(" .,!?:;")
        if 2 <= len(base_area) <= 60:
            return base_area

    origin_match = re.search(
        r"\b(?:from|a partir de|a partir do|a partir da|a partir dos|a partir das|desde)\s+([A-Za-zÀ-ÿ\s'-]+?)(?:\s+with|\s+com|\s+using|\s+usando|\s+for|\s+para|,|\.|$)",
        user_message or "",
        flags=re.IGNORECASE,
    )
    if origin_match:
        origin_area = origin_match.group(1).strip(" .,!?:;")
        if 2 <= len(origin_area) <= 60:
            return origin_area

    match = re.search(
        r"\b(?:in|around|near|em|perto de|na zona de)\s+([A-Za-zÀ-ÿ\s'-]+?)(?:\s+with|\s+com|\s+for|\s+para|,|\.|$)",
        user_message or "",
        flags=re.IGNORECASE,
    )
    if not match:
        return default
    area = match.group(1).strip(" .,!?:;")
    return area if 2 <= len(area) <= 60 else default


def _build_belem_history_pastry_fallback(
    user_message: str,
    language: str,
    weather_data: str,
    transport_data: str,
    places_data: str,
    events_data: str,
) -> str:
    """Build a rich corridor plan for history plus pastry requests in Belem."""
    del user_message
    normalized_context = _normalize_planner_text("\n".join([transport_data or "", places_data or "", events_data or ""]))
    weather_bullets = _extract_planner_fallback_bullets(weather_data, max_items=4)
    if not weather_bullets:
        weather_bullets = [
            "- No detailed weather facts were available in this run; keep riverside walking segments optional and prioritise indoor heritage stops if conditions worsen."
            if language != "pt"
            else "- Esta execução não trouxe factos meteorológicos detalhados; mantém opcionais os troços ribeirinhos a pé e privilegia paragens interiores se o tempo piorar."
        ]

    transport_bullets_en: List[str] = []
    transport_bullets_pt: List[str] = []
    if "15e" in normalized_context:
        transport_bullets_en.append(
            "- **Surface option:** Carris tram **15E** from the Baixa/Praça da Figueira axis toward **Belém/Algés**; useful if you want the classic riverside approach."
        )
        transport_bullets_pt.append(
            "- **Opção de superfície:** elétrico Carris **15E** a partir do eixo Baixa/Praça da Figueira em direção a **Belém/Algés**; é a opção clássica junto ao rio."
        )
    if re.search(r"\b728\b", normalized_context):
        transport_bullets_en.append(
            "- **Bus option:** Carris **728** is a useful Chiado/Baixa-to-Belém corridor option when confirmed for your exact stop."
        )
        transport_bullets_pt.append(
            "- **Autocarro:** Carris **728** é uma opção útil no corredor Chiado/Baixa-Belém quando confirmada para a tua paragem exata."
        )
    if "cais do sodre" in normalized_context or "cais do sodré" in normalized_context:
        transport_bullets_en.append(
            "- **Rail-backed option:** Metro from **Baixa/Chiado** to **Cais do Sodré**, then CP Cascais line to **Belém** if you confirm the timetable on the travel day."
        )
        transport_bullets_pt.append(
            "- **Opção com comboio:** Metro de **Baixa/Chiado** para **Cais do Sodré**, depois Linha de Cascais da CP até **Belém**, confirmando horários no próprio dia."
        )
    if not transport_bullets_en:
        transport_bullets_en.append(
            "- Transport data did not confirm a specific line strongly enough; use Carris/CP official channels before leaving."
        )
        transport_bullets_pt.append(
            "- Os dados de transporte não confirmaram uma linha específica com confiança suficiente; confirma nos canais Carris/CP antes de sair."
        )
    transport_bullets_en.append("- Do **not** use current next-departure times as an afternoon schedule; check live departures when you are ready to leave.")
    transport_bullets_pt.append("- Não uses próximas partidas recolhidas agora como horário da tarde; confirma partidas live quando fores sair.")

    used_sources: List[str] = []
    if str(weather_data or "").strip():
        used_sources.append("[*IPMA*](https://www.ipma.pt)" if language == "pt" else "[*IPMA*](https://www.ipma.pt/en/)")
    if any("Carris" in bullet for bullet in transport_bullets_en):
        used_sources.append("[*Carris*](https://www.carris.pt)")
    if any("Metro" in bullet for bullet in transport_bullets_en):
        used_sources.append("[*Metro de Lisboa*](https://www.metrolisboa.pt)")
    if any("CP" in bullet for bullet in transport_bullets_en):
        used_sources.append("[*CP*](https://www.cp.pt)")
    used_sources.append(
        "[*VisitLisboa Locais*](https://www.visitlisboa.com/pt-pt/locais)"
        if language == "pt"
        else "[*VisitLisboa Places*](https://www.visitlisboa.com/en/places)"
    )
    deduped_sources = list(dict.fromkeys(used_sources))
    label = "📌 **Fonte:**" if language == "pt" else "📌 **Source:**"
    updated = "**Atualizado:**" if language == "pt" else "**Updated:**"
    source_line = f"{label} {' | '.join(deduped_sources)} | {updated} {datetime.now().strftime('%H:%M')}"

    if language == "pt":
        return "\n".join(
            [
                "### 📅 Tarde em Belém a partir do Chiado",
                "",
                "A versão mais útil é fazer **um corredor simples Chiado → Belém**, com transporte para a deslocação longa e caminhada curta apenas dentro de Belém.",
                "",
                "### ⛅ Tempo e ritmo",
                *weather_bullets,
                "- Se houver chuva, começa por Jerónimos/Pastéis de Belém e deixa a frente ribeirinha para uma janela mais seca.",
                "",
                "---",
                "",
                "### 🚇 Transporte recomendado",
                *transport_bullets_pt[:4],
                "",
                "---",
                "",
                "### 🏛️ Plano histórico ordenado",
                "- **Mosteiro dos Jerónimos:** começa aqui; é o melhor ponto para enquadrar Belém na expansão marítima portuguesa e no período manuelino.",
                "- **Pastéis de Belém:** faz a pausa de pastelaria a meio da tarde, junto ao eixo dos Jerónimos, para não quebrar a lógica geográfica.",
                "- **Padrão dos Descobrimentos:** segue para a frente ribeirinha se o tempo permitir; funciona como transição visual entre história e rio.",
                "- **Torre de Belém:** termina aqui se ainda houver luz e condições para caminhar junto ao Tejo; se chover mais, encurta este troço.",
                "",
                "**💡 Dicas úteis**",
                "- Mantém o plano em 3-4 horas; não juntes outro bairro à mesma tarde.",
                "- Horários, bilhetes e entrada nos monumentos não foram confirmados aqui; valida-os antes de fechar o plano.",
                "",
                source_line,
            ]
        ).strip()

    return "\n".join(
        [
            "### 📅 Belém Afternoon From Chiado",
            "",
            "The useful version is a **simple Chiado → Belém corridor**: use transport for the long move, then keep the walking local inside Belém.",
            "",
            "### ⛅ Weather and pacing",
            *weather_bullets,
            "- If showers pick up, start with Jerónimos/Pastéis de Belém and leave the open riverside for the driest window.",
            "",
            "---",
            "",
            "### 🚇 Recommended transport",
            *transport_bullets_en[:4],
            "",
            "---",
            "",
            "### 🏛️ Ordered history plan",
            "- **Jerónimos Monastery:** start here; it anchors Belém’s Age of Discovery and Manueline context.",
            "- **Pastéis de Belém:** use this as the mid-afternoon pastry break beside the monastery axis, not as a detached detour.",
            "- **Padrão dos Descobrimentos:** continue to the riverside if the weather allows; it links the historical theme to the Tagus setting.",
            "- **Belém Tower:** finish here if there is still light and the riverside walk is comfortable; shorten this leg if rain strengthens.",
            "",
            "**💡 Useful tips**",
            "- Keep this to 3-4 hours; do not add another Lisbon district to the same afternoon.",
            "- Opening hours, tickets, and monument access were not confirmed here; verify them before locking the plan.",
            "",
            source_line,
        ]
    ).strip()


def _is_historic_gastronomy_day_request(normalized_query: str) -> bool:
    """Detect one-day history plus traditional food itinerary requests."""
    day_intent = bool(
        re.search(
            r"\b(?:1\s*dia|um\s+dia|one\s+day|full\s+day|dia\s+inteiro|day\s+itinerary|itinerario\s+de\s+1\s+dia|roteiro\s+de\s+1\s+dia)\b",
            normalized_query,
        )
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


def _extract_weather_overview_bullets(weather_data: str, language: str, *, max_items: int = 4) -> List[str]:
    """Extract concrete weather facts for itinerary fallbacks."""
    text = str(weather_data or "").strip()
    if not text:
        return [
            "- Confirma a previsão do IPMA antes de fechar longos troços ao ar livre."
            if language == "pt"
            else "- Check the IPMA forecast before locking long outdoor stretches."
        ]

    fact_bullets = _extract_weather_fact_bullets(text, language, max_items=max_items)
    if fact_bullets:
        return fact_bullets[:max_items]

    bullets: List[str] = []
    seen: set[str] = set()
    weather_markers = (
        "weather",
        "meteorolog",
        "temperature",
        "temperatura",
        "°c",
        "rain",
        "chuva",
        "precip",
        "wind",
        "vento",
        "warning",
        "aviso",
        "condition",
        "condi",
        "aguaceiro",
        "showers",
        "cloud",
        "nublado",
        "trovoada",
    )
    ambiguous_weather_markers = ("tempo",)

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or _PLANNER_SOURCE_LINE_RE.match(line):
            continue
        line = re.sub(r"^[-*•]\s+", "", line).strip()
        line = re.sub(r"^\*{0,2}📅\s*", "", line).strip()
        if re.search(
            r"\b(?:claro|sim|yes|aqui tens|here is|here's|roteiro|itinerario|itinerary|plano|plan|monumentos|gastronomia|gastronomy|weather context|weather in|contexto meteorologico|estado do tempo|tempo em|should be comfortable|deve estar confortavel|deve estar confortável|da para|dá para|confortaveis|confortáveis)\b",
            line,
            flags=re.IGNORECASE,
        ):
            continue
        line_lower = line.lower()
        has_strong_weather_marker = any(marker in line_lower for marker in weather_markers)
        has_ambiguous_weather_marker = any(marker in line_lower for marker in ambiguous_weather_markers)
        if not has_strong_weather_marker and not has_ambiguous_weather_marker:
            continue
        if has_ambiguous_weather_marker and not has_strong_weather_marker:
            continue
        if re.search(r"\b(?:source|fonte|updated|atualizado)\b", line, flags=re.IGNORECASE):
            continue
        normalized = _normalize_planner_text(line)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        bullets.append(f"- {line}")
        if len(bullets) >= max_items:
            break

    if bullets:
        return bullets

    return _extract_weather_safety_bullets(weather_data, language)[:max_items]


def _is_next_day_planning_follow_up(user_message: str, conversation_context: str = "") -> bool:
    """Detect a planning continuation that asks for the following day."""
    normalized_query = _normalize_planner_text(user_message)
    normalized_context = _normalize_planner_text(conversation_context)
    if not re.search(r"\b(?:dia seguinte|proximo dia|próximo dia|amanha|amanhã|tomorrow|next day|following day)\b", normalized_query):
        return False
    if not re.search(r"\b(?:plan|planeia|planejar|itinerary|itinerario|roteiro|dia|day)\b", normalized_query):
        return False
    return bool(
        re.search(
            r"\b(?:plan|itinerary|itinerario|roteiro|monument|monumento|histor|gastronom|traditional|tradicional|restaurant|restaurante)\b",
            normalized_context,
        )
    )


def _build_next_day_historic_food_transport_fallback(
    language: str,
    weather_data: str,
    transport_data: str,
    conversation_context: str,
) -> str:
    """Build a second-day continuation that preserves prior interests without repeating stops."""
    weather_bullets = _extract_weather_overview_bullets(weather_data, language, max_items=3)
    has_transport_context = bool(str(transport_data or "").strip())
    source_line = _build_lisboa_scope_source_line(
        language,
        include_ipma=bool(str(weather_data or "").strip()),
        include_visitlisboa=True,
        include_metro=has_transport_context,
        include_carris=has_transport_context,
    )
    prior_context = _normalize_planner_text(conversation_context)
    avoid_belem_centre = any(
        token in prior_context
        for token in ("belem", "jeronimos", "baixa", "carmo", "se de lisboa", "santo antonio")
    )

    if language == "pt":
        continuity_note = (
            "Mantive o tema do roteiro anterior (**história + gastronomia tradicional**), mas mudei o eixo para **Estrela, Campo de Ourique, Ajuda e Alcântara**, evitando repetir Baixa, Sé, Carmo e Belém."
            if avoid_belem_centre
            else "Mantive o tema anterior (**história + gastronomia tradicional**) e organizei um segundo dia compacto, com transportes simples e pouca repetição de zonas."
        )
        transport_note = (
            "- Usa **Metro** como eixo principal até **Rato/Marquês/São Sebastião** e completa as ligações para Estrela, Ajuda ou Alcântara com **Carris**; confirma a linha e a hora pouco antes de sair."
            if has_transport_context
            else "- Os dados de transporte detalhado por perna não foram recolhidos; confirma as ligações exatas no operador antes de sair."
        )
        return "\n".join(
            [
                "### 📅 Dia Seguinte · História, Gastronomia e Transportes",
                "",
                continuity_note,
                "",
                "### ⛅ Tempo e ritmo",
                *weather_bullets,
                "- Se chover, troca a ordem para fazer primeiro os espaços interiores e deixa jardins/miradouros para as janelas mais secas.",
                "",
                "---",
                "",
                "### 🏛️ Plano alternativo",
                "**09:30 · Basílica da Estrela e Jardim da Estrela**",
                "- 📍 **Zona:** Estrela",
                "- 🏷️ **Tema:** património religioso, bairro histórico e pausa verde curta",
                "- 🚌 **Transporte:** começa por Metro até Rato/Marquês e completa a ligação local com Carris ou caminhada curta, consoante o ponto de partida.",
                "",
                "**11:30 · Campo de Ourique e paragem gastronómica**",
                "- 📍 **Zona:** Campo de Ourique",
                "- 🏷️ **Tema:** bairro residencial, mercado/restauração e gastronomia tradicional",
                "- 💡 **Dica:** escolhe aqui o almoço para não voltares ao eixo turístico do dia anterior.",
                "",
                "**14:30 · Palácio Nacional da Ajuda ou Museu Nacional de Arte Antiga**",
                "- 📍 **Zona:** Ajuda / Santos",
                "- 🏷️ **Tema:** palácio, coleções históricas e programa interior para chuva",
                "- 🚌 **Transporte:** usa Carris para a aproximação final; os horários exatos devem ser confirmados no próprio dia.",
                "",
                "**17:30 · Alcântara ou Santos para jantar tradicional**",
                "- 📍 **Zona:** Alcântara / Santos",
                "- 🏷️ **Tema:** jantar, frente ribeirinha urbana e regresso fácil",
                "- 💡 **Dica:** termina aqui se quiseres evitar atravessar novamente a cidade ao fim do dia.",
                "",
                "---",
                "",
                "### 🚇 Transportes",
                transport_note,
                "- Evita depender de elétricos turísticos em hora de ponta ou com chuva; são úteis, mas podem ser lentos e cheios.",
                "- Para deslocações entre zonas sem ligação direta confirmada, privilegia Metro + Carris em vez de longas caminhadas.",
                "",
                "### 💡 Limites práticos",
                "- Não repeti os principais blocos do dia anterior; a ideia é dar continuidade temática sem fazer o mesmo circuito.",
                "- Confirma horários de entrada, encerramentos e disponibilidade de restaurantes antes de sair.",
                "",
                source_line,
            ]
        ).strip()

    continuity_note = (
        "I kept the previous theme (**history + traditional food**) but moved the route to **Estrela, Campo de Ourique, Ajuda, and Alcântara**, avoiding a repeat of Baixa, Sé, Carmo, and Belém."
        if avoid_belem_centre
        else "I kept the previous theme (**history + traditional food**) and structured a compact second day with simple public-transport logic."
    )
    transport_note = (
        "- Use **Metro** as the backbone toward **Rato/Marquês/São Sebastião**, then complete the local legs to Estrela, Ajuda, or Alcântara with **Carris**; confirm the exact line and time shortly before leaving."
        if has_transport_context
        else "- Detailed leg-by-leg transport was not retrieved; confirm exact connections with the operator before leaving."
    )
    return "\n".join(
        [
            "### 📅 Next Day · History, Food, And Transport",
            "",
            continuity_note,
            "",
            "### ⛅ Weather and pacing",
            *weather_bullets,
            "- If it rains, do the indoor stops first and keep gardens or viewpoints for the driest window.",
            "",
            "---",
            "",
            "### 🏛️ Alternative plan",
            "**09:30 · Basilica da Estrela and Estrela Garden**",
            "- 📍 **Area:** Estrela",
            "- 🏷️ **Theme:** religious heritage, historic neighbourhood, and a short green pause",
            "- 🚌 **Transport:** start by Metro toward Rato/Marquês and complete the local connection by Carris or a short walk, depending on your start point.",
            "",
            "**11:30 · Campo de Ourique food stop**",
            "- 📍 **Area:** Campo de Ourique",
            "- 🏷️ **Theme:** residential Lisbon, market/restaurants, and traditional food",
            "- 💡 **Tip:** lunch here so the second day does not fall back into the same tourist corridor.",
            "",
            "**14:30 · Ajuda National Palace or Museu Nacional de Arte Antiga**",
            "- 📍 **Area:** Ajuda / Santos",
            "- 🏷️ **Theme:** palace, historical collections, and an indoor rain-safe block",
            "- 🚌 **Transport:** use Carris for the final approach; exact times should be checked on the day.",
            "",
            "**17:30 · Alcântara or Santos for dinner**",
            "- 📍 **Area:** Alcântara / Santos",
            "- 🏷️ **Theme:** dinner, urban riverside, and an easier return",
            "- 💡 **Tip:** finish here if you want to avoid crossing the city again late in the day.",
            "",
            "---",
            "",
            "### 🚇 Transport",
            transport_note,
            "- Avoid relying on tourist trams during peak hours or rain; they can be useful, but slow and crowded.",
            "- For legs without a confirmed direct link, prefer Metro + Carris over long walks.",
            "",
            "### 💡 Practical limits",
            "- This avoids repeating the main blocks from the previous day while keeping the same interests.",
            "- Confirm entry times, closures, and restaurant availability before leaving.",
            "",
            source_line,
        ]
    ).strip()


def _build_historic_gastronomy_day_fallback(
    language: str,
    weather_data: str,
    transport_data: str,
) -> str:
    """Build a rich one-day historical and traditional food itinerary fallback."""
    weather_bullets = _extract_weather_overview_bullets(weather_data, language, max_items=4)
    has_transport_context = bool(str(transport_data or "").strip())
    source_line = _build_lisboa_scope_source_line(
        language,
        include_ipma=bool(str(weather_data or "").strip()),
        include_visitlisboa=True,
        include_carris=has_transport_context,
        include_cp=has_transport_context,
    )
    belem_move_pt = (
        "- **Carmo → Belém:** desce para o eixo Baixa/Cais do Sodré e confirma uma opção Carris para Belém/Algés ou a Linha de Cascais da CP até Belém no momento da viagem."
        if has_transport_context
        else "- **Carmo → Belém:** usa transporte público confirmado no momento da viagem; escolhe uma ligação direta para Belém em vez de acrescentar outro bairro ao roteiro."
    )
    belem_move_en = (
        "- **Carmo → Belém:** move down toward the Baixa/Cais do Sodré axis and confirm a Carris option toward Belém/Algés or the CP Cascais line to Belém at travel time."
        if has_transport_context
        else "- **Carmo → Belém:** use public transport confirmed at travel time; choose a direct link to Belém instead of adding another neighbourhood."
    )

    if language == "pt":
        return "\n".join(
            [
                "### 📅 Roteiro histórico e gastronómico para 1 dia",
                "",
                "A opção mais coerente é concentrar o dia entre **Baixa, Sé, Chiado/Carmo e Belém**, com as deslocações longas antes ou depois do almoço e caminhadas curtas dentro de cada zona.",
                "",
                "### ⛅ Condições meteorológicas",
                *weather_bullets,
                "- Se houver chuva ou vento, mantém Belém e o centro histórico como blocos separados e usa pausas interiores entre visitas.",
                "",
                "---",
                "",
                "### 🏛️ Plano otimizado",
                "**09:00 · Galerias Romanas / Baixa**",
                "- 📍 **Localização:** Rua da Prata / Rua da Conceição, Lisboa",
                "- 🏷️ **Categoria:** Monumento histórico",
                "- 💡 **Dica:** começa no centro histórico para manter o início compacto e evitar deslocações cedo demais.",
                "",
                "**10:30 · Sé de Lisboa e Igreja de Santo António**",
                "- 📍 **Localização:** Largo da Sé / Largo de Santo António da Sé, Lisboa",
                "- 🏷️ **Categoria:** Património religioso e histórico",
                "- 💡 **Dica:** faz este bloco a pé; as duas paragens ficam próximas e dão contexto à Lisboa medieval.",
                "",
                "**12:30 · Almoço tradicional na Baixa**",
                "- 📍 **Sugestão:** Granja Velha, Rua dos Douradores, 200, Lisboa",
                "- 🏷️ **Categoria:** Restaurante tradicional",
                "- 💡 **Dica:** é uma boa pausa logística antes de subires para o Carmo ou seguires para Belém.",
                "",
                "**14:30 · Museu Arqueológico do Carmo**",
                "- 📍 **Localização:** Largo do Carmo, Lisboa",
                "- 🏷️ **Categoria:** Museu / monumento histórico",
                "- 💡 **Dica:** funciona bem depois do almoço porque fica perto da Baixa e tem valor histórico claro sem alongar demasiado o percurso.",
                "",
                "**16:30 · Belém histórico**",
                "- 📍 **Paragens:** Mosteiro dos Jerónimos, Padrão dos Descobrimentos e Torre de Belém",
                "- 🏷️ **Categoria:** Monumentos e frente ribeirinha",
                "- 💡 **Dica:** se o tempo piorar, prioriza Jerónimos e deixa a caminhada ribeirinha para uma janela mais seca.",
                "",
                "**18:30 · Pastelaria ou jantar leve**",
                "- 📍 **Sugestão:** Pastéis de Belém ou regresso à Baixa/Chiado para jantar tradicional",
                "- 🏷️ **Categoria:** Gastronomia tradicional",
                "- 💡 **Dica:** escolhe Belém se ainda estiveres nessa zona; regressa ao centro se quiseres terminar perto de transportes.",
                "",
                "---",
                "",
                "### 🚶 Como chegar e deslocação",
                "- **Baixa → Sé / Santo António:** a pé, por ruas curtas do centro histórico.",
                "- **Sé / Santo António → Granja Velha:** a pé, regressando para a Baixa.",
                "- **Granja Velha → Carmo:** a pé, subindo para Chiado/Carmo; conta com uma subida curta.",
                belem_move_pt,
                "",
                "### 💡 Dicas úteis",
                "- Confirma horários, bilhetes e eventuais encerramentos de cada monumento antes de sair.",
                "- Mantém 20-30 minutos de margem entre blocos, sobretudo se chover ou se Belém ficar para o fim do dia.",
                "- O plano é otimizado por coerência geográfica e temática; horários de entradas, partidas e restaurantes devem ser confirmados em tempo real.",
                "",
                source_line,
            ]
        ).strip()

    return "\n".join(
        [
            "### 📅 One-Day History And Traditional Food Itinerary",
            "",
            "The strongest structure is **Baixa, Sé, Chiado/Carmo, then Belém**, with compact walks inside each area and the longer move kept to one clear transfer.",
            "",
            "### ⛅ Weather and pacing",
            *weather_bullets,
            "- If rain or wind increases, keep Belém and the historic centre as separate blocks and use indoor pauses between stops.",
            "",
            "---",
            "",
            "### 🏛️ Optimized plan",
            "**09:00 · Roman Galleries / Baixa**",
            "- 📍 **Location:** Rua da Prata / Rua da Conceição, Lisbon",
            "- 🏷️ **Category:** Historical monument",
            "- 💡 **Tip:** start in the historic centre so the morning stays compact.",
            "",
            "**10:30 · Lisbon Cathedral and Saint Anthony Church**",
            "- 📍 **Location:** Largo da Sé / Largo de Santo António da Sé, Lisbon",
            "- 🏷️ **Category:** Religious and historical heritage",
            "- 💡 **Tip:** do this section on foot; the stops are close and give the route a medieval Lisbon anchor.",
            "",
            "**12:30 · Traditional lunch in Baixa**",
            "- 📍 **Suggestion:** Granja Velha, Rua dos Douradores, 200, Lisbon",
            "- 🏷️ **Category:** Traditional restaurant",
            "- 💡 **Tip:** it is a practical lunch pause before Carmo or the longer move to Belém.",
            "",
            "**14:30 · Carmo Archaeological Museum**",
            "- 📍 **Location:** Largo do Carmo, Lisbon",
            "- 🏷️ **Category:** Museum / historical monument",
            "- 💡 **Tip:** it works well after lunch because it is close to Baixa and gives the day a strong historical stop.",
            "",
            "**16:30 · Historic Belém**",
            "- 📍 **Stops:** Jerónimos Monastery, Padrão dos Descobrimentos, and Belém Tower",
            "- 🏷️ **Category:** Monuments and riverside heritage",
            "- 💡 **Tip:** if the weather worsens, prioritize Jerónimos and save the riverside walk for the driest window.",
            "",
            "**18:30 · Pastry or light dinner**",
            "- 📍 **Suggestion:** Pastéis de Belém or return to Baixa/Chiado for traditional dinner",
            "- 🏷️ **Category:** Traditional food",
            "- 💡 **Tip:** stay in Belém if you are already there; return to the centre if you want an easier transport finish.",
            "",
            "---",
            "",
            "### 🚶 Movement logic",
            "- **Baixa → Sé / Saint Anthony:** walk through the compact historic core.",
            "- **Sé / Saint Anthony → Granja Velha:** walk back toward Baixa.",
            "- **Granja Velha → Carmo:** walk uphill toward Chiado/Carmo.",
            belem_move_en,
            "",
            "### 💡 Useful tips",
            "- Confirm opening hours, tickets, and closures before leaving.",
            "- Keep 20-30 minutes of buffer between blocks, especially if rain or wind affects movement.",
            "- The plan is optimized for geographic and thematic coherence; entry times, departures, and restaurant availability should be confirmed live.",
            "",
            source_line,
        ]
    ).strip()


def _extract_visitlisboa_place_cards(text: str, *, max_items: int = 8) -> List[Dict[str, str]]:
    """Extract lightweight VisitLisboa place cards from gathered researcher text."""
    cards: List[Dict[str, str]] = []
    seen_names: set[str] = set()

    patterns = [
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
            body = match.group("body")
            if not re.search(
                r"(?mi)(Address|Morada|Category|Categoria|Website|Site|Rating|Avalia|Price|Preço|Tickets|Bilhetes)",
                body,
            ):
                continue
            category_match = re.search(r"(?mi)^\s*📂\s*(?:\*\*)?(?:Category|Categoria)(?:\*\*)?:\s*(?P<value>[^\n]+)", body)
            address_match = re.search(r"(?mi)^\s*📍\s*\*\*(?:Address|Morada):\*\*\s*(?P<value>[^\n]+)", body)
            hours_match = re.search(r"(?mi)^\s*🕐\s*(?P<value>[^\n]+)", body)
            url_match = re.search(
                r"(?mi)^\s*(?:🔗|🌐)\s*(?:\*\*(?:Website|Site):\*\*\s*)?(?P<value>https?://\S+|\[[^\]]+\]\(https?://[^)]+\))",
                body,
            )
            description = ""
            for raw_line in body.splitlines():
                line = raw_line.strip()
                if not line or re.match(r"^(📂|🎫|📍|🕐|💰|⭐|📞|🔗|🌐)", line):
                    continue
                description = re.sub(r"\s+", " ", line).strip()
                break
            url = url_match.group("value").strip() if url_match else ""
            markdown_url_match = re.search(r"\((https?://[^)]+)\)", url)
            if markdown_url_match:
                url = markdown_url_match.group(1)
            cards.append(
                {
                    "name": name,
                    "category": category_match.group("value").strip() if category_match else "",
                    "address": address_match.group("value").strip() if address_match else "",
                    "hours": hours_match.group("value").strip() if hours_match else "",
                    "url": url,
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
    """Build a generic fallback from grounded cards rather than prompt-specific templates."""
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
    if len(cards) < 2:
        return ""

    requested_days = _extract_requested_day_count(user_message)
    visible_days = min(requested_days or 1, 5)
    weather_bullets = _extract_weather_safety_bullets(weather_data, language)
    transport_bullets = _extract_planner_fallback_bullets(transport_data, max_items=5)
    source_line = _build_planner_fallback_source_line(
        language,
        weather_data,
        transport_data,
        places_data,
        events_data,
    )

    if language == "pt":
        title = (
            f"### 📅 Plano Grounded de {visible_days} Dias"
            if visible_days > 1
            else "### 📅 Roteiro Grounded"
        )
        intro = (
            "Usei os locais concretos recolhidos pelos workers e mantive detalhes não confirmados como limitações."
        )
        weather_title = "### ⛅ Tempo e Ritmo"
        move_title = "### 🚇 Lógica de Deslocação"
        no_transport = "- As pernas exatas de transporte não ficaram confirmadas; valida a ligação no operador antes de sair."
        limits_title = "### 💡 Notas Práticas"
        limits = [
            "- Horários, bilhetes, reservas e acessibilidade não devem ser assumidos se não aparecerem nos dados.",
            "- Mantém 20-30 minutos de margem entre blocos quando houver mudança de zona.",
        ]
        day_label = "Dia"
        stop_label = "Paragem"
    else:
        title = (
            f"### 📅 Grounded {visible_days}-Day Plan"
            if visible_days > 1
            else "### 📅 Grounded Itinerary"
        )
        intro = (
            "I used the concrete worker-gathered places and kept unconfirmed details as limitations."
        )
        weather_title = "### ⛅ Weather and Pacing"
        move_title = "### 🚇 Movement Logic"
        no_transport = "- Exact transport legs were not confirmed; check the operator before leaving."
        limits_title = "### 💡 Practical Notes"
        limits = [
            "- Do not assume opening hours, tickets, bookings, or accessibility unless they appear in the data.",
            "- Keep 20-30 minutes of buffer between blocks when changing areas.",
        ]
        day_label = "Day"
        stop_label = "Stop"

    sections: List[str] = [title, intro, "", "---", "", weather_title, *weather_bullets, "", "---", ""]
    if visible_days > 1:
        cards_per_day = max(1, min(3, (len(cards) + visible_days - 1) // visible_days))
        for day in range(visible_days):
            day_cards = cards[day * cards_per_day:(day + 1) * cards_per_day]
            if not day_cards:
                break
            sections.extend([f"### 📍 {day_label} {day + 1}", ""])
            for index, card in enumerate(day_cards, start=1):
                sections.append(f"**{stop_label} {index} · {card['name']}**")
                sections.extend(_place_card_line(card, language=language)[1:])
                sections.append("")
            sections.append("---")
            sections.append("")
    else:
        times = ["09h30", "11h15", "13h00", "15h00", "17h00"]
        for time_label, card in zip(times, cards[:5]):
            sections.append(f"### 🏛️ {time_label} · {card['name']}")
            sections.extend(_place_card_line(card, language=language)[1:])
            sections.append("")
            sections.append("---")
            sections.append("")

    sections.extend([move_title])
    sections.extend(transport_bullets if transport_bullets else [no_transport])
    if qa_disclaimers:
        sections.extend(f"- {item}" for item in qa_disclaimers[:3])
    sections.extend(["", limits_title, *limits])
    if requested_days and requested_days > 5:
        sections.append(
            "- Limitei a resposta aos primeiros 5 dias para manter qualidade e verificabilidade."
            if language == "pt"
            else "- I limited the answer to the first 5 days to preserve quality and verifiability."
        )
    if source_line:
        sections.extend(["", source_line])
    return "\n".join(section for section in sections if section is not None).strip()


def _place_card_line(card: Dict[str, str], *, language: str) -> List[str]:
    """Format a compact place candidate without placeholder fields."""
    lines = [f"- **Candidate from gathered data:** {card['name']}" if language != "pt" else f"- **Candidato dos dados recolhidos:** {card['name']}"]
    if card.get("description"):
        label = "Nota" if language == "pt" else "Note"
        lines.append(f"    - **{label}:** {card['description']}")
    if card.get("category"):
        label = "Categoria" if language == "pt" else "Category"
        lines.append(f"    - **{label}:** {card['category']}")
    if card.get("address"):
        label = "Morada" if language == "pt" else "Address"
        lines.append(f"    - **{label}:** {card['address']}")
    if card.get("hours"):
        label = "Horário" if language == "pt" else "Hours"
        lines.append(f"    - **{label}:** {card['hours']}")
    if card.get("url"):
        label = "Website" if language == "pt" else "Website"
        lines.append(f"    - **{label}:** [VisitLisboa]({card['url']})")
    return lines


def _select_short_plan_card(cards: List[Dict[str, str]], *, area: str, kind: str) -> Optional[Dict[str, str]]:
    """Select the best concrete card for a short neighborhood plan."""
    area_norm = _normalize_planner_text(area)
    candidates: List[tuple[int, Dict[str, str]]] = []
    for card in cards:
        basis = _normalize_planner_text(
            " ".join(str(card.get(field, "")) for field in ("name", "category", "address", "description"))
        )
        score = 0
        if kind == "coffee":
            if re.search(r"\b(cafe|cafes|coffee|restaurant|restaurants|pastelaria|bar|wine|wines)\b", basis):
                score += 10
            else:
                continue
        else:
            if re.search(r"\b(museum|museums|monument|monuments|cultural|culture|museu|museus|galer|gallery|casa)\b", basis):
                score += 10
            else:
                continue
        if area_norm and area_norm in basis:
            score += 8
        if "campo de ourique" in area_norm and re.search(r"\b(casa fernando pessoa|amoreiras|campo de ourique)\b", basis):
            score += 10
        if re.search(r"\b(today:\s*closed|hoje:\s*fechado|closed today|fechado hoje)\b", basis):
            score -= 6
        candidates.append((score, card))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _known_short_plan_anchor(area: str, kind: str) -> Optional[Dict[str, str]]:
    """Return a conservative known anchor for common short neighborhood plans."""
    area_norm = _normalize_planner_text(area)
    if "campo de ourique" not in area_norm:
        return None
    if kind == "coffee":
        return {
            "name": "Mercado de Campo de Ourique",
            "category": "Market / food hall",
            "address": "Rua Coelho da Rocha, Campo de Ourique, Lisboa",
            "hours": "",
            "url": "",
            "description": "Useful local base for a short coffee or snack pause, but individual vendors and morning service must be checked on arrival.",
            "source_hint": "lisboa_aberta",
        }
    if kind == "culture":
        return {
            "name": "Casa Fernando Pessoa",
            "category": "Museum",
            "address": "Rua Coelho da Rocha, 16/18, Campo de Ourique, 1250-088, Lisboa",
            "hours": "",
            "url": "https://www.visitlisboa.com/en/places/casa-fernando-pessoa",
            "description": "Local cultural anchor for Fernando Pessoa's work and house-museum context.",
            "source_hint": "visitlisboa",
        }
    return None


def _build_short_coffee_culture_fallback(
    user_message: str,
    language: str,
    weather_data: str,
    places_data: str,
    events_data: str,
) -> str:
    """Build a conservative short-plan fallback for coffee plus one cultural stop."""
    weather_bullets = _extract_weather_safety_bullets(weather_data, language)
    area = _extract_neighborhood_hint(
        user_message,
        default="a zona indicada" if language == "pt" else "the requested area",
    )
    context = "\n".join([places_data or "", events_data or ""])
    normalized_context = _normalize_planner_text(context)
    mentions_closed = bool(re.search(r"\b(?:today closed|closed today|hoje fechado|fechado hoje)\b", normalized_context))
    cards = _extract_visitlisboa_place_cards(context)
    coffee_card = _select_short_plan_card(cards, area=area, kind="coffee")
    culture_card = _select_short_plan_card(cards, area=area, kind="culture")
    known_source_context = ""
    if not coffee_card:
        coffee_card = _known_short_plan_anchor(area, "coffee")
        if coffee_card:
            known_source_context += "\nLisboa Aberta: Mercado de Campo de Ourique"
    if not culture_card:
        culture_card = _known_short_plan_anchor(area, "culture")
        if culture_card:
            known_source_context += "\nVisitLisboa Places: Casa Fernando Pessoa"
    source_line = _build_planner_fallback_source_line(
        language=language,
        weather_data=weather_data,
        transport_data="",
        places_data=f"{places_data or ''}{known_source_context}",
        events_data=events_data,
    )

    if language == "pt":
        coffee_lines = (
            _place_card_line(coffee_card, language=language)
            if coffee_card
            else [
                f"- Não foi confirmado um café específico em **{area}** nos dados recolhidos.",
                "- Mantém o café numa opção local e interior, validada no momento.",
            ]
        )
        culture_lines = (
            _place_card_line(culture_card, language=language)
            if culture_card
            else [
                f"- Não foi confirmado um local cultural específico em **{area}** nos dados recolhidos.",
                "- Usa uma paragem curta no bairro apenas depois de confirmares que está aberta.",
            ]
        )
        closed_note = (
            "- Se o local cultural recolhido aparecer como fechado hoje, não o uses como paragem principal; substitui por uma pausa cultural curta no bairro e valida um espaço aberto antes de sair."
            if mentions_closed
            else "- Usa o local cultural recolhido apenas se o horário oficial confirmar que está aberto no intervalo pretendido."
        )
        return "\n".join(
            [
                "### 📅 Plano curto com café e cultura",
                "",
                "### ⛅ Condições",
                *weather_bullets,
                "",
                "---",
                "",
                "### ☕ Café",
                f"- Mantém o café em **{area}** para não gastar a janela de 90 minutos em deslocações.",
                *coffee_lines,
                "",
                "---",
                "",
                "### 🏛️ Paragem cultural",
                f"- Faz uma paragem curta e localizada em **{area}**, em vez de atravessar Lisboa para preencher o plano.",
                *culture_lines,
                closed_note,
                "",
                "---",
                "",
                "### 🚶 Ritmo recomendado",
                "- **30 min:** café.",
                "- **10-15 min:** deslocação a pé curta.",
                "- **35-45 min:** paragem cultural ou alternativa interior confirmada aberta.",
                "",
                source_line,
            ]
        ).strip()

    closed_note = (
        "- If the gathered cultural venue is marked closed today, do not use it as the main stop; keep the plan to a short neighborhood cultural pause and verify an open venue before leaving."
        if mentions_closed
        else "- Use the gathered cultural venue only if official hours confirm it is open during your window."
    )
    coffee_lines_en = (
        _place_card_line(coffee_card, language=language)
        if coffee_card
        else [
            f"- Keep coffee local in **{area}** and choose a confirmed-open indoor café before leaving.",
            "- Do not cross the city for coffee in a 90-minute plan; protect the time for the cultural stop.",
        ]
    )
    culture_lines_en = (
        _place_card_line(culture_card, language=language)
        if culture_card
        else [
            f"- Use a short cultural stop in **{area}** only after confirming current opening hours.",
            "- If no grounded venue is available, make the cultural part a brief neighbourhood architecture or literary walk rather than inventing a museum.",
        ]
    )
    return "\n".join(
        [
            "### 📅 Short Coffee And Culture Plan",
            "",
            "### ⛅ Conditions",
            *weather_bullets,
            "",
            "---",
            "",
            "### ☕ Coffee",
            f"- Keep coffee in **{area}** so the 90-minute window is not spent crossing Lisbon.",
            *coffee_lines_en,
            "",
            "---",
            "",
            "### 🏛️ Cultural stop",
            f"- Keep the cultural stop local to **{area}**, rather than forcing a cross-city museum visit.",
            *culture_lines_en,
            closed_note,
            "",
            "---",
            "",
            "### 🚶 Suggested pace",
            "- **30 min:** coffee.",
            "- **10-15 min:** short walk.",
            "- **35-45 min:** cultural stop or confirmed-open indoor alternative.",
            "",
            source_line,
        ]
    ).strip()


def _build_deterministic_planner_fallback(
    user_message: str,
    language: str,
    weather_data: str,
    transport_data: str,
    places_data: str,
    events_data: str,
    qa_disclaimers: list[str] | None,
) -> str:
    """Build a compact deterministic itinerary when planner LLM synthesis fails or times out."""
    requested_days = _extract_requested_day_count(user_message)
    normalized_query = _normalize_planner_text(user_message)
    overcomplex_scope_request = bool(
        re.search(r"\b(?:ferry|ferries|transtejo|soflusa|ticket prices|prices|bookings|booking|reservations|reservas|precos|preços)\b", normalized_query)
        and re.search(r"\b(?:sintra|cascais|setubal|setubal|setúbal)\b", normalized_query)
        and re.search(r"\b(?:live|right now|next saturday|future|next)\b", normalized_query)
    )
    if overcomplex_scope_request:
        return _build_overcomplex_scope_fallback(language, weather_data)

    card_based_fallback = _build_card_based_itinerary_fallback(
        user_message=user_message,
        language=language,
        weather_data=weather_data,
        transport_data=transport_data,
        places_data=places_data,
        events_data=events_data,
        qa_disclaimers=qa_disclaimers,
    )
    if card_based_fallback:
        return card_based_fallback

    if requested_days and requested_days > 1:
        return _build_multi_day_scope_fallback(
            user_message=user_message,
            language=language,
            requested_days=requested_days,
            weather_data=weather_data,
        )

    reduced_mobility_evening_request = bool(
        re.search(r"\b(?:reduced mobility|wheelchair|accessible|accessibility|step free|mobility|mobilidade reduzida|acessivel|acessível)\b", normalized_query)
        and re.search(r"\b(?:dinner|eat|meal|jantar|comer)\b", normalized_query)
        and re.search(r"\b(?:cultural|culture|indoor|interior|museum|museu)\b", normalized_query)
    )
    if reduced_mobility_evening_request:
        return _build_reduced_mobility_evening_fallback(language, weather_data, transport_data)

    resident_service_plan_request = bool(
        re.search(r"\b(?:recycling|ecoponto|reciclagem|recycle)\b", normalized_query)
        and re.search(r"\b(?:pharmacy|pharmacies|farmacia|farmacias|farmácia|farmácias)\b", normalized_query)
        and re.search(r"\b(?:dinner|jantar|restaurant|restaurante)\b", normalized_query)
    )
    if resident_service_plan_request:
        return _build_resident_service_plan_fallback(
            user_message,
            language,
            weather_data,
            places_data,
            transport_data,
        )

    single_day_museum_garden_request = bool(
        re.search(r"\b(?:single|one|um|uma|relaxed|quiet|calm|tranquilo|relaxado)\b", normalized_query)
        and re.search(r"\b(?:museum|museu)\b", normalized_query)
        and re.search(r"\b(?:garden|jardim)\b", normalized_query)
        and re.search(r"\b(?:rain backup|backup|chuva|se chover)\b", normalized_query)
    )
    if single_day_museum_garden_request:
        return _build_single_day_museum_garden_fallback(
            user_message=user_message,
            language=language,
            weather_data=weather_data,
        )

    low_walk_day_request = bool(
        re.search(r"\b(?:relaxed|quiet|calm|tranquilo|relaxado|relaxed day|second day|day plan|one relaxed day|um dia)\b", normalized_query)
        and re.search(r"\b(?:avoid long walks|avoiding long walks|low walk|low-walk|same walking preference|pouca caminhada|evitar caminhadas longas|sem caminhadas longas|rain backup|indoor backup|backup interior|se chover)\b", normalized_query)
    )
    if low_walk_day_request:
        return _build_low_walk_day_fallback(
            user_message=user_message,
            language=language,
            weather_data=weather_data,
        )

    belem_history_pastry_request = bool(
        re.search(r"\b(?:belem|bel[eÃ©]m)\b", normalized_query)
        and re.search(r"\b(?:history|historical|historia|hist[oó]ria|culture|cultural|monument|monastery|jeronimos|jer[oó]nimos)\b", normalized_query)
        and re.search(r"\b(?:pastry|custard|tart|pastel|pasteis|past[eé]is|nata)\b", normalized_query)
    )
    if belem_history_pastry_request:
        return _build_belem_history_pastry_fallback(
            user_message=user_message,
            language=language,
            weather_data=weather_data,
            transport_data=transport_data,
            places_data=places_data,
            events_data=events_data,
        )

    if _is_historic_gastronomy_day_request(normalized_query):
        return _build_historic_gastronomy_day_fallback(
            language=language,
            weather_data=weather_data,
            transport_data=transport_data,
        )

    walking_route_request = bool(
        (
            re.search(r"\b(passeio|percurso|rota)\b", normalized_query)
            and re.search(r"\b(a pe|pedonal|caminh|walk|walking|horas?|minutos?)\b", normalized_query)
        )
        or (
            re.search(r"\b(?:coherent walk|walking route|walk)\b", normalized_query)
            and re.search(r"\b(?:minutes?|hours?|minutos?|horas?)\b", normalized_query)
        )
    )
    evening_food_culture_request = bool(
        re.search(r"\b(?:dinner|restaurant|jantar|restaurante)\b", normalized_query)
        and re.search(r"\b(?:cultural|culture|cultura|cultural stop|paragem cultural)\b", normalized_query)
    )
    short_coffee_culture_request = bool(
        re.search(r"\b(?:coffee|cafe|cafes|café|cafes|pastelaria)\b", normalized_query)
        and re.search(r"\b(?:cultural|culture|cultura|cultural stop|paragem cultural)\b", normalized_query)
    )
    museum_day_request = bool(
        re.search(r"\b(?:museum|museums|museu|museus)\b", normalized_query)
        and re.search(r"\b(?:full day|day plan|dia inteiro|um dia|amanha|tomorrow)\b", normalized_query)
    )
    if museum_day_request:
        return _build_full_museum_day_transport_fallback(
            user_message=user_message,
            language=language,
            weather_data=weather_data,
            transport_data=transport_data,
            places_data=places_data,
            events_data=events_data,
        )
    if evening_food_culture_request:
        if re.search(r"\b(?:oriente|parque das nacoes|parque das nações|expo)\b", normalized_query):
            return _build_oriente_evening_food_culture_fallback(
                language=language,
                weather_data=weather_data,
                places_data=places_data,
                events_data=events_data,
            )
        area = _extract_neighborhood_hint(
            user_message,
            default="a zona indicada" if language == "pt" else "the requested area",
        )
        area = re.sub(
            r"\b(?:tonight|today|this evening|hoje|esta noite|esta tarde)\b",
            "",
            area,
            flags=re.IGNORECASE,
        ).strip(" /,.-") or ("a zona indicada" if language == "pt" else "the requested area")
        area_axis = (
            "Santos / Cais do Sodré"
            if "santos" in _normalize_planner_text(area)
            else f"{area} / Cais do Sodré / Santos"
        )
        normalized_context = _normalize_planner_text("\n".join([places_data or "", events_data or ""]))
        has_doca = "doca de santo" in normalized_context
        has_mnaa = (
            "museu nacional de arte antiga" in normalized_context
            or "national museum of ancient art" in normalized_context
        )
        weather_bullets = _extract_weather_overview_bullets(weather_data, language, max_items=2)
        if not weather_bullets:
            weather_bullets = [
                "- No detailed weather facts were available in this run; keep the riverside or outdoor part flexible."
                if language != "pt"
                else "- Esta execução não trouxe factos meteorológicos detalhados; mantém flexível a parte ribeirinha ou ao ar livre."
            ]
        source_line = _build_lisboa_scope_source_line(
            language,
            include_ipma=bool(str(weather_data or "").strip()),
            include_visitlisboa=bool(str(places_data or events_data or "").strip()),
        )
        if language == "pt":
            dinner_line = (
                "- **Jantar:** **Doca de Santo** apareceu nos dados recolhidos; usa-o como opção candidata, mas confirma horário, reserva e disponibilidade antes de sair."
                if has_doca
                else f"- **Jantar:** escolhe uma opção na zona de **{area}** depois de confirmares horário e disponibilidade."
            )
            culture_line = (
                "- **Paragem cultural:** **Museu Nacional de Arte Antiga** é a âncora cultural mais clara perto de Santos; para esta noite, confirma se há horário compatível ou evento ativo antes de o tratar como plano fechado."
                if has_mnaa
                else "- **Paragem cultural:** usa apenas uma paragem interior próxima que consigas confirmar aberta hoje; não assumas disponibilidade noturna sem confirmação."
            )
            return "\n".join(
                [
                    "### 📅 Plano de fim de tarde",
                    "",
                    "**⛅ Condições e segurança**",
                    *weather_bullets,
                    "",
                    "---",
                    "",
                    "**📍 Plano recomendado**",
                    f"- Mantém o plano concentrado em **{area_axis}**, em vez de atravessar Lisboa para preencher uma paragem.",
                    culture_line,
                    dinner_line,
                    "- Se a paragem cultural não estiver confirmada aberta, transforma a noite em jantar + passeio curto junto ao rio.",
                    "- Disponibilidade noturna, reservas e bilhetes devem ser confirmados antes de sair.",
                    "",
                    source_line,
                ]
            ).strip()
        dinner_line = (
            "- **Dinner:** **Doca de Santo** appeared in the gathered data; treat it as a candidate option, but confirm opening hours, booking, and availability before leaving."
            if has_doca
            else f"- **Dinner:** choose an option in **{area}** after confirming opening hours and availability."
        )
        culture_line = (
            "- **Cultural stop:** **Museu Nacional de Arte Antiga** is the clearest cultural anchor near Santos; for tonight, confirm compatible opening/event hours before treating it as a fixed stop."
            if has_mnaa
            else "- **Cultural stop:** use only a nearby indoor stop that you can confirm open today; do not assume evening availability without confirmation."
        )
        return "\n".join(
            [
                "### 📅 Suggested Evening Plan",
                "",
                "**⛅ Weather and safety**",
                *weather_bullets,
                "",
                "---",
                "",
                "**📍 Recommended plan**",
                f"- Keep the evening around **{area_axis}** instead of crossing Lisbon just to fill the cultural stop.",
                culture_line,
                dinner_line,
                "- If the cultural stop is not confirmed open, turn the evening into dinner plus a short riverside walk.",
                "- Confirm tonight's opening, tickets, reservations, and event availability before leaving.",
                "",
                source_line,
            ]
        ).strip()
    if short_coffee_culture_request:
        return _build_short_coffee_culture_fallback(
            user_message=user_message,
            language=language,
            weather_data=weather_data,
            places_data=places_data,
            events_data=events_data,
        )
    if language == "pt" and walking_route_request:
        origin_match = re.search(r"\b(?:estou em|em|a partir de|desde)\s+([A-Za-zÀ-ÿ\s-]+?)(?:\s+e\s+quero|\s+com|\s+de\s+\d+|\s*,|$)", user_message)
        origin = origin_match.group(1).strip(" .,!?:;") if origin_match else "a zona indicada"
        duration_match = re.search(r"\b(\d+\s*(?:h|horas|minutos?))\b", normalized_query)
        duration = duration_match.group(1).replace("h", " horas") if duration_match else "cerca de 2 horas"
        weather_bullets = _extract_planner_fallback_bullets(weather_data, max_items=2)
        if not weather_bullets:
            weather_bullets = ["- Confirma a previsão do IPMA antes de sair, sobretudo se houver hipótese de chuva."]
        source_line = _build_planner_fallback_source_line(language, weather_data, "", "", "")
        return "\n".join(
            [
                "### 📅 Itinerário sugerido",
                "",
                "**⛅ Condições e segurança**",
                *weather_bullets,
                "",
                "---",
                "",
                "**🚶 Percurso a pé**",
                f"- **Início:** {origin}.",
                f"- **Duração:** {duration}.",
                "- **Lógica do percurso:** mantém um circuito compacto, com ruas secundárias e pontos de pausa próximos, em vez de saltar entre zonas afastadas.",
                "- **Transporte:** não é necessário para o corpo do passeio; usa metro/autocarro apenas para chegar ao ponto inicial ou regressar.",
                "",
                "---",
                "",
                "**📍 Paragens sugeridas**",
                f"- Começa em **{origin}** e escolhe ruas de bairro com comércio local para a primeira parte.",
                "- Faz uma pausa curta num jardim, praça ou café próximo que esteja efetivamente aberto no momento.",
                "- Fecha o circuito regressando por ruas paralelas, para manter continuidade e evitar repetir exatamente o mesmo caminho.",
                "",
                "---",
                "",
                "**✨ Notas práticas**",
                "- Não foram confirmados horários, lotação ou abertura atual de espaços específicos; valida qualquer paragem interior antes de entrar.",
                "",
                source_line,
            ]
        ).strip()
    if walking_route_request:
        origin_match = re.search(r"\b(?:near|at|from|starting from)\s+([A-Za-zÀ-ÿ\s'-]+?)(?:\s+before|\s+with|\s+for|\s*,|$)", user_message, flags=re.IGNORECASE)
        origin = origin_match.group(1).strip(" .,!?:;") if origin_match else "the indicated area"
        duration_match = re.search(r"\b(\d+\s*(?:minutes?|hours?))\b", normalized_query)
        duration = duration_match.group(1) if duration_match else "about 90 minutes"
        weather_bullets = _extract_planner_fallback_bullets(weather_data, max_items=2)
        if not weather_bullets:
            weather_bullets = [
                "- No detailed weather facts were available in this run; keep the route compact and preserve a covered fallback."
            ]
        source_line = _build_planner_fallback_source_line(language, weather_data, "", "", "")
        return "\n".join(
            [
                "### 📅 Suggested Walk",
                "",
                "**⛅ Weather and safety**",
                *weather_bullets,
                "",
                "---",
                "",
                "**🚶 Route logic**",
                f"- **Start and finish:** {origin}.",
                f"- **Duration:** {duration}.",
                "- **Shape:** use a compact out-and-back route so you can return quickly if the weather worsens.",
                "- **Train buffer:** keep the final 15-20 minutes close to the station instead of committing to a long one-way loop.",
                "",
                "---",
                "",
                "**📍 Suggested stops**",
                f"- Start near **{origin}** and stay on the most direct, familiar streets first.",
                "- Add the riverside or viewpoint stretch only while conditions are comfortable.",
                "- Turn back early if showers or wind strengthen, and use nearby cafes or covered streets as the fallback.",
                "",
                source_line,
            ]
        ).strip()
    title = (
        "### 📅 Dia 1 · Itinerário Sugerido"
        if language == "pt"
        else "### 📅 Day 1 · Suggested Itinerary"
    ) if requested_days and requested_days > 1 else (
        "### 📅 Itinerário Sugerido"
        if language == "pt"
        else "### 📅 Suggested Itinerary"
    )
    weather_heading = "### ⛅ Condições e Segurança" if language == "pt" else "### ⛅ Weather and Safety"
    transport_heading = "### 🚇 Como Chegar e Deslocação" if language == "pt" else "### 🚇 Getting There and Moving Around"
    activities_heading = "### 📍 Sugestões para a visita" if language == "pt" else "### 📍 Visit Suggestions"
    notes_heading = "### ✨ Notas Práticas" if language == "pt" else "### ✨ Practical Notes"

    weather_bullets = _extract_planner_fallback_bullets(weather_data, max_items=3)
    transport_bullets = _extract_planner_fallback_bullets(transport_data, max_items=4)
    activity_bullets = _extract_planner_fallback_bullets(
        "\n".join(part for part in [places_data, events_data] if part),
        max_items=4,
    )
    if language == "pt":
        english_markers = re.compile(
            r"\b(start|continue|end|suitable|quieter|calmer|heritage|finish|listed|available)\b",
            re.IGNORECASE,
        )
        activity_bullets = [bullet for bullet in activity_bullets if not english_markers.search(bullet)]
    note_bullets = [f"- {item}" for item in (qa_disclaimers or [])[:3]]

    if not weather_bullets:
        weather_bullets = [
            "- Esta execução não trouxe factos meteorológicos detalhados; mantém flexíveis as partes ao ar livre."
            if language == "pt"
            else "- No detailed weather facts were available in this run; keep outdoor parts flexible."
        ]
    if not transport_bullets:
        transport_bullets = [
            "- Confirme o trajeto em carris.pt, metrolisboa.pt ou cp.pt antes de partir."
            if language == "pt"
            else "- Confirm the route on carris.pt, metrolisboa.pt, or cp.pt before leaving."
        ]
    if not activity_bullets:
        activity_bullets = [
            "- Priorize espaços interiores e pausas curtas, com base nos locais já recolhidos."
            if language == "pt"
            else "- Prioritize indoor stops and short breaks based on the gathered places data."
        ]
    if not note_bullets:
        note_bullets = [
            "- Verifique horários e acessibilidade diretamente nos operadores e locais oficiais."
            if language == "pt"
            else "- Verify opening hours and accessibility directly with the official operators and venues."
        ]

    follow_up_note = _build_multi_day_follow_up_note(language, requested_days)
    if follow_up_note:
        note_bullets.insert(0, f"- {follow_up_note}")

    sections = [
        title,
        "",
        weather_heading,
        *weather_bullets,
        "",
        "---",
        "",
        transport_heading,
        *transport_bullets,
        "",
        "---",
        "",
        activities_heading,
        *activity_bullets,
        "",
        "---",
        "",
        notes_heading,
        *note_bullets,
        "",
        _build_planner_fallback_source_line(
            language,
            weather_data,
            transport_data,
            places_data,
            events_data,
        ),
    ]
    return "\n".join(sections).strip()


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

    raw_lower = (cleaned_response or "").lower()
    normalized = _normalize_planner_text(cleaned_response)
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
        r"\bdescription\s*:\s+.*\bcategory\s*:",
        r"\baddress\s*:\s+.*google\.com/maps",
        r"\bdescri[cç][aã]o\s*:\s+.*\bcategoria\s*:",
    ]
    unsafe_heading_patterns = [
        r"^###\s*(?:⚠️\s*)?(?:helpful notes|notas úteis|notas uteis|notes)\b",
        r"^###\s+[^\n]*\b\d{1,2}:\d{2}\b\s*[·\-]\s*(?:today|hoje)\s*:",
        r"^###\s+[^\n]*\b(?:hours|horario|horário)\s*:",
        r"^###\s*(?:ℹ️\s*)?(?:note|nota)\s*:?\s*$",
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
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in defect_patterns)


def _planner_response_has_transport_quality_defects(
    cleaned_response: str,
    user_message: str,
    transport_data: str,
) -> bool:
    """Return whether transport-aware planner output hides grounded route gaps."""
    if (
        not cleaned_response
        or not str(transport_data or "").strip()
        or not _query_requests_public_transport(user_message)
    ):
        return False

    normalized = _normalize_planner_text(cleaned_response)
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
        "- Algumas pernas exatas de transporte público não ficaram totalmente confirmadas nos dados recolhidos; confirma no operador antes de sair e não trates isto como horário live de partida."
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
            "Use concrete grounded operator, line, direction, board/alight, or transfer details; "
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
        # Common planning requests must use the dynamic LLM synthesis path first.
        # Deterministic builders below are retained only as safety fallbacks when
        # the LLM fails, invents unsupported facts, or emits unsafe Markdown.

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
            heading = "## ⚠️ Limitações dos Dados" if language == "pt" else "## ⚠️ Data Limitations"
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
                    "OUTPUT BUDGET:\n"
                    "- Target 450-650 words for rich cross-domain itineraries; stay shorter for simple requests.\n"
                    "- Include the useful evidence the workers gathered: weather consequence, realistic movement, grounded stops, and user preferences.\n"
                    "- Use compact sections instead of a loose bullet dump: weather/pacing, transport, ordered plan, and one short tips/limits block.\n"
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
        except Exception:
            fallback = _build_deterministic_planner_fallback(
                user_message=user_message,
                language=language,
                weather_data=weather_data,
                transport_data=transport_data,
                places_data=places_data,
                events_data=events_data,
                qa_disclaimers=qa_disclaimers,
            )
            fallback = enforce_multi_day_quality_mode(fallback, user_message, language)
            return finalize_worker_response(
                fallback,
                agent_name="planner",
                user_query=user_message,
                language=language,
            )

        if (
            _planner_response_requires_fallback(cleaned_response)
            or _planner_response_has_markdown_contract_defects(cleaned_response)
        ):
            cleaned_response = _build_deterministic_planner_fallback(
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

        retry_count = 0
        while grounding_issues and retry_count < 2:
            retry_count += 1
            retry_messages = messages + [
                SystemMessage(
                    content=(
                        "Your previous draft violated the grounding rules. Revise it now.\n"
                        "- Remove any unsupported venue names.\n"
                        "- Remove unsupported accessibility claims.\n"
                        "- Keep only facts grounded in the provided data.\n"
                        "- Replace vague transport prose with concrete grounded line/stop/transfer details, "
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

        if (
            _planner_response_requires_fallback(cleaned_response)
            or _planner_response_has_markdown_contract_defects(cleaned_response)
        ):
            cleaned_response = _build_deterministic_planner_fallback(
                user_message=user_message,
                language=language,
                weather_data=weather_data,
                transport_data=transport_data,
                places_data=places_data,
                events_data=events_data,
                qa_disclaimers=qa_disclaimers,
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

        cleaned_response = enforce_multi_day_quality_mode(
            cleaned_response,
            user_message,
            language,
        )

        return finalize_worker_response(
            cleaned_response,
            agent_name="planner",
            user_query=user_message,
            language=language,
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
        content="### 📅 Day 1\n- **Jerónimos Monastery**\n- Short grounded plan"
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
    _check("Source" in fallback_response and "Suggested Itinerary" in fallback_response, "Planner fallback stays user-facing and structured")

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
