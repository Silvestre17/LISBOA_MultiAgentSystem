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

from agent.agents.base import BaseAgent, clean_response, traceable
from agent.prompts.planner import get_planner_prompt
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
    r"^(?:[-*•]\s*)?(?:📌\s*)?(?:\*\*)?(?:Fonte|Source)(?:\*\*)?:.*$",
    re.IGNORECASE,
)


def _normalize_planner_text(text: str) -> str:
    """Normalizes planner text for robust grounding comparisons."""
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    normalized = re.sub(r"[^a-zA-Z0-9\s/-]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip().lower()


def _extract_allowed_place_names(text: str) -> List[str]:
    """Extracts grounded POI names from researcher/event outputs."""
    if not text:
        return []

    candidates: List[str] = []
    seen = set()

    for line in text.splitlines():
        for match in re.findall(r"\*\*([^*]+)\*\*", line):
            candidate = match.strip().strip("-–—: ")
            normalized = _normalize_planner_text(candidate)
            if not normalized or normalized in _PLANNER_FIELD_LABELS:
                continue
            if normalized.isdigit() or re.fullmatch(r"\d+\.?", normalized):
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

    if language == "pt":
        return (
            f"MULTI-DAY QUALITY MODE: o pedido é para {requested_days} dias. "
            "Entregue apenas o Dia 1 em detalhe agora, com sequência geograficamente otimizada e transportes claros. "
            "Não esboce os restantes dias com baixa confiança. Termine com uma nota curta a explicar que os dias seguintes devem ser pedidos depois de validar o Dia 1."
        )

    return (
        f"MULTI-DAY QUALITY MODE: the request covers {requested_days} days. "
        "Deliver only Day 1 in full detail now, with geographically optimized sequencing and clear transport guidance. "
        "Do not sketch later days with low confidence. Finish with one short note explaining that the remaining days should be requested after Day 1 is confirmed."
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

    if language == "pt":
        return (
            f"Para manter a qualidade num pedido de {requested_days} dias, este primeiro bloco cobre apenas o Dia 1. "
            "Se a estrutura estiver alinhada, peça-me depois o Dia 2."
        )

    return (
        f"To keep quality high across a {requested_days}-day request, this first block only covers Day 1. "
        "If the structure fits what you want, ask me next for Day 2."
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
    """Clamp multi-day requests to a high-quality Day 1 response.

    Args:
        response: Planner draft or repaired final response.
        user_message: Original user request.
        language: Output language code.

    Returns:
        str: Day 1-focused response for explicit multi-day requests.
    """
    requested_days = _extract_requested_day_count(user_message)
    if not requested_days or requested_days <= 1:
        return response

    lines = str(response or "").splitlines()
    trimmed_lines: List[str] = []
    truncated = False
    truncation_index: Optional[int] = None

    for index, line in enumerate(lines):
        if _is_later_day_section_marker(line):
            truncated = True
            truncation_index = index
            break
        trimmed_lines.append(line)

    normalized_lines = list(trimmed_lines if truncated else lines)
    replacement_title = (
        "### 📅 Dia 1 · Itinerário Sugerido"
        if language == "pt"
        else "### 📅 Day 1 · Suggested Itinerary"
    )
    title_updated = False
    for index, line in enumerate(normalized_lines):
        if line.strip().startswith("### ") or line.strip().startswith("## "):
            normalized_lines[index] = replacement_title
            title_updated = True
            break
    if not title_updated:
        normalized_lines = [replacement_title, "", *normalized_lines]

    if truncated and truncation_index is not None:
        preserved_tail = [
            line
            for line in lines[truncation_index:]
            if _PLANNER_SOURCE_LINE_RE.match(line.strip())
        ]
        if preserved_tail:
            normalized_lines.extend(["", *preserved_tail])

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
        seen.add(normalized)
        bullets.append(f"- {candidate}")
        if len(bullets) >= max_items:
            break

    return bullets


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
    if "visitlisboa" in combined or places_data or events_data:
        sources.append("[*VisitLisboa*](https://www.visitlisboa.com)")
    if "metrolisboa" in combined:
        sources.append("[*Metro de Lisboa*](https://www.metrolisboa.pt)")
    if "carris" in combined or transport_data:
        sources.append("[*Carris*](https://www.carris.pt)")
    if "cp.pt" in combined or "comboio" in combined or "train" in combined:
        sources.append("[*CP*](https://www.cp.pt)")

    timestamp = "**Atualizado:**" if language == "pt" else "**Updated:**"
    source_label = "📌 **Fonte:**" if language == "pt" else "📌 **Source:**"
    return f"{source_label} {' | '.join(sources)} | {timestamp} {datetime.now().strftime('%H:%M')}"


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
    note_bullets = [f"- {item}" for item in (qa_disclaimers or [])[:3]]

    if not weather_bullets:
        weather_bullets = [
            "- Confirme a previsão do IPMA antes de sair."
            if language == "pt"
            else "- Check the latest IPMA forecast before leaving."
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
        self.system_prompt = get_planner_prompt()

    @traceable(name="planner_agent", run_type="chain", tags=["sub-agent", "planner"])
    def invoke(
        self,
        user_message: str,
        weather_data: str = "",
        transport_data: str = "",
        places_data: str = "",
        events_data: str = "",
        qa_disclaimers: list[str] | None = None,
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

        Returns:
            str: Formatted itinerary.
        """
        # Build context from agent outputs
        context_parts = []
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
            context_parts.append(
                f"## ⚠️ Data Limitations (from QA validation)\n"
                f"Include these caveats in your response where relevant:\n{disclaimer_text}"
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
        language = infer_response_language(user_query=user_message, default="en")
        requested_days = _extract_requested_day_count(user_message)
        multi_day_instruction = _build_multi_day_planner_instruction(
            language=language,
            requested_days=requested_days,
        )

        language_instruction = (
            "Respond ENTIRELY in Portuguese (PT-PT)."
            if language == "pt"
            else "Respond ENTIRELY in English."
        )

        messages = [
            SystemMessage(content=self.system_prompt),
            SystemMessage(content=language_instruction),
            SystemMessage(content=grounding_message),
            *([SystemMessage(content=multi_day_instruction)] if multi_day_instruction else []),
            SystemMessage(
                content=(
                    "OUTPUT BUDGET:\n"
                    "- Maximum 4 itinerary/activity cards.\n"
                    "- Maximum 2 bullets per card.\n"
                    "- Maximum 320 words total.\n"
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

        if _planner_response_requires_fallback(cleaned_response):
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

        retry_count = 0
        while grounding_issues and retry_count < 2:
            retry_count += 1
            retry_messages = messages + [
                SystemMessage(
                    content=(
                        "Your previous draft violated the grounding rules. Revise it now.\n"
                        "- Remove any unsupported venue names.\n"
                        "- Remove unsupported accessibility claims.\n"
                        "- Keep only facts grounded in the provided data."
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

        if _planner_response_requires_fallback(cleaned_response):
            cleaned_response = _build_deterministic_planner_fallback(
                user_message=user_message,
                language=language,
                weather_data=weather_data,
                transport_data=transport_data,
                places_data=places_data,
                events_data=events_data,
                qa_disclaimers=qa_disclaimers,
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

        return self.invoke(
            user_message=user_message,
            weather_data=agent_outputs.get("weather", ""),
            transport_data=agent_outputs.get("transport", ""),
            places_data=agent_outputs.get("researcher", ""),
            events_data="",  # Events come from researcher too
            qa_disclaimers=qa_disclaimers,
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

    clamped = enforce_multi_day_quality_mode(
        response="### 📅 Lisbon Plan\n- 📅 Day 1\n- Jerónimos\n- 📅 Day 2\n- MAAT",
        user_message="Plan 3 days in Lisbon for me.",
        language="en",
    )
    _check("\n- 📅 Day 2" not in clamped and "Day 1" in clamped, "Multi-day clamp removes later day sections")

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
