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
from typing import Callable, Dict, List, Optional, Sequence
from urllib.parse import quote_plus

from langchain_core.messages import HumanMessage, SystemMessage

from agent.agents.base import BaseAgent, clean_response
from agent.prompts.planner import get_planner_prompt
from agent.utils.langsmith_tracing import traceable
from agent.utils.response_formatter import (
    final_post_qa_guard,
    infer_response_language,
)
from agent.planning.evidence import SOURCE_CATALOG
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
# Also accept common mojibake encodings of "→" from copied transcripts.
_PLANNER_ROUTE_ARROW_RE = re.compile(
    r"\s*(?:\u2192|->|\u00e2\u2020\u2019|\u00c3\u00a2\u00e2\u20ac\u00a0\u00e2\u20ac\u2122)\s*"
)


def _normalize_planner_text(text: str) -> str:
    """Normalizes planner text for robust grounding comparisons."""
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    normalized = re.sub(r"[^a-zA-Z0-9\s/-]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip().lower()


def _planner_text_has_route_arrow(text: str) -> bool:
    """Return whether text contains a common route arrow representation."""
    return bool(_PLANNER_ROUTE_ARROW_RE.search(text or ""))


_PLANNER_ANCHOR_KEYWORD_RE = re.compile(
    r"\b(?:rua|avenida|av|largo|praca|calcada|travessa|estrada|campo|"
    r"museu|palacio|castelo|mosteiro|torre|padrao|catedral|igreja|capela|"
    r"jardim|parque|miradouro|mercado|fundacao|universidade|faculdade|"
    r"hospital|farmacia|estacao|aeroporto|terminal|centro\s+comercial|"
    r"shopping|livraria|teatro|coliseu|oceanario|maat|lx\s*factory)\b",
    re.IGNORECASE,
)
_PLANNER_GENERIC_ANCHOR_RE = re.compile(
    r"\b(?:monumentos?|atracoes?|atra[cç][oõ]es?|museus?|restaurantes?|"
    r"miradouros?|viewpoints?|view\s+points?|views?|vistas?|vista|lookouts?|lookout|"
    r"gastronomia|comida|cozinha|almo[cç]o|almoco|almo[cç]ar|almocar|"
    r"jantar|refei[cç][aã]o|refeicao|refei[cç][oõ]es|refeicoes|"
    r"eventos?|cultura|historicos?|tradicional|imperdiveis|locais|sitios|"
    r"places|sights|restaurants?|food|meals?|lunch|dinner|culture|events?)\b",
    re.IGNORECASE,
)
_PLANNER_ANCHOR_LIST_SPLIT_RE = re.compile(
    r"\s*(?:[,;]|\+|/|\b(?:e|and)\b\s+"
    r"(?=(?:rua|avenida|av\.?|largo|praca|praça|museu|palacio|palácio|castelo|"
    r"mosteiro|torre|padrao|padrão|catedral|igreja|capela|jardim|parque|"
    r"miradouro|mercado|fundacao|fundação|universidade|faculdade|teatro|"
    r"coliseu|oceanario|oceanário|maat|lx|[A-Z0-9]{2,})\b))\s*",
    re.IGNORECASE,
)
_PLANNER_PROCEDURAL_ANCHOR_RE = re.compile(
    r"\b(?:how\s+(?:i|we|to)\s+(?:get|go|travel)|how\s+to\s+(?:get|go|travel)|"
    r"get\s+there|go\s+there|travel\s+there|como\s+(?:vou|vamos|chego|chegar|ir)|"
    r"como\s+(?:me|nos)?\s*desloc|rota|percurso|trajeto|transportes?)\b",
    re.IGNORECASE,
)


def _planner_anchor_part_is_generic_component(part: str) -> bool:
    """Return whether a phrase is a requested component, not a place anchor."""
    normalized = _normalize_planner_text(part)
    if not normalized:
        return True
    normalized = re.sub(
        r"^(?:\d{1,2}|um|uma|one|dois|duas|two|tres|three|quatro|four|cinco|five)\s+",
        "",
        normalized,
    )
    normalized = re.sub(r"\b(?:pela|na|no|da|do|de|em|with|for|the)\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return bool(
        re.fullmatch(
            r"(?:museus?|museums?|galerias?|galleries|monumentos?|monuments?|"
            r"miradouros?|viewpoints?|view\s+points?|views?|vistas?|vista|lookouts?|lookout|"
            r"restaurantes?|restaurants?|gastronomia|comida|cozinha|food|meal|meals|"
            r"almo[cç]o|almoco|almo[cç]ar|almocar|lunch|jantar|dinner|"
            r"cidade|city|tradicional|traditional|cultural|cultura|historico|historicos|historic|local|locais)"
            r"(?:\s+(?:cidade|city|tradicional|traditional|cultural|cultura|historico|historicos|historic|local|locais))*",
            normalized,
        )
    )


_PLANNER_ANCHOR_CONSTRAINT_TAIL_RE = re.compile(
    r"\s+(?:e\s+com|and\s+with|com|with|inclui|include|including|"
    r"pass(?:a(?:r|ndo)?|e(?:m)?)\s+(?:por|pelo|pela)|via|through|"
    r"para\s+(?:almo[cç]ar|almocar|almo[cç]o|almoco|jantar|comer|"
    r"restaurantes?|s[ií]tio|sitio|paragem)|"
    r"for\s+(?:lunch|dinner|food|restaurants?|a\s+place|a\s+stop)|"
    r"durante|during|às|as|at)\b.*$",
    re.IGNORECASE,
)
_PLANNER_EMBEDDED_START_ANCHOR_RE = re.compile(
    r"\b(?:come[cç](?:ar|a|e|ando)|iniciar|iniciando|starting|start)\s+"
    r"(?:no|na|nos|nas|em|at|from|in)\s+(?P<place>[^,.;]+)",
    re.IGNORECASE,
)


def _trim_planner_anchor_constraint_tail(value: str) -> str:
    """Remove trailing planning constraints from a candidate place name.

    The helper is intentionally generic: it strips clauses such as "com
    almoço" or "with lunch" after extraction, without maintaining a curated
    list of venues.
    """
    cleaned = re.sub(r"\s+", " ", str(value or "")).strip(" .:-")
    previous = None
    while cleaned and previous != cleaned:
        previous = cleaned
        cleaned = _PLANNER_ANCHOR_CONSTRAINT_TAIL_RE.sub("", cleaned).strip(" .:-")
    return cleaned


def _clean_requested_anchor_fragment(fragment: str) -> str:
    """Return a compact candidate place name from a user-request fragment."""
    cleaned = re.sub(r"\([^)]*\)", " ", str(fragment or ""))
    cleaned = re.sub(
        r"^\s*(?:(?:roteiro|itiner[aá]rio|plano|route|itinerary|plan)\s+)?"
        r"(?:de\s+|for\s+)?(?:meio\s+dia|half[-\s]?day|\d+\s*(?:h|hora|horas|hours?))\s+"
        r"(?:de|desde|from)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    if _PLANNER_PROCEDURAL_ANCHOR_RE.search(_normalize_planner_text(cleaned)):
        return ""
    colon_match = re.match(r"^\s*(?P<prefix>[^:]{2,80})\s*:\s*(?P<place>.+)$", cleaned)
    if colon_match:
        prefix_key = _normalize_planner_text(colon_match.group("prefix"))
        if re.search(
            r"\b(?:pontos?|zonas?|areas?|areas?|locais|sitios|sítios|paragens?|"
            r"stops?|places?|spots?|areas?|waypoints?|distantes?|diferentes?|seguintes?)\b",
            prefix_key,
        ):
            cleaned = colon_match.group("place")
    embedded_start = _PLANNER_EMBEDDED_START_ANCHOR_RE.search(cleaned)
    if embedded_start:
        cleaned = embedded_start.group("place")
    cleaned = re.sub(
        r"^\s*(?:come\S*|iniciar|inicia|starting|start|termin\S*|acab\S*|ending|end)\s+"
        r"(?:no|na|nos|nas|em|at|from|in|the)?\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^\s*(?:o|a|os|as|um|uma|uns|umas|no|na|nos|nas|em|at|from|the|"
        r"de|do|da|dos|das)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s+(?:e|and)\s+(?:termin\S*|acab\S*|ending|end)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s+(?:e|and)\s+(?:mant[eé]m|mantem|preserva|keep|keeping|passa|passar|via)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s+(?:termin\S*|acab\S*|ending|finish(?:ing)?|end)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s+(?:as|às|at|pelas|por\s+volta|durante|during|com|with|inclui|include|"
        r"including|pass(?:a(?:r|ndo)?|e(?:m)?)\s+(?:por|pelo|pela)|via|through|depois|then|"
        r"sem|using|usando|how|como|transportes?|rota|route)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s+(?:no|na|num|numa|em|in)\s+"
        r"(?:roteiro|itinerario|itinerário|plano|itinerary|plan)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = _trim_planner_anchor_constraint_tail(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip(" .:-")


def _requested_anchor_fragment_is_specific(fragment: str) -> bool:
    """Return whether a candidate is likely a specific place, not a category."""
    cleaned = _clean_requested_anchor_fragment(fragment)
    normalized = _normalize_planner_text(cleaned)
    if not (2 <= len(cleaned) <= 90 and normalized):
        return False
    if re.search(
        r"^(?:almoco|almoço|almoçar|almocar|jantar|dinner|lunch|"
        r"sitio\s+para|sítio\s+para|paragem\s+para|stop\s+for)\b",
        normalized,
    ):
        return False
    if _PLANNER_PROCEDURAL_ANCHOR_RE.search(normalized):
        return False
    if re.search(
        r"\b(?:meio\s+dia|half\s+day|itinerario|roteiro|plano|organiza|"
        r"planeia|planejar|planning|itinerary|route\s+plan)\b",
        normalized,
    ):
        return False
    if re.search(r"\b(?:hotel|hoteis|hotéis|alojamento|alojamentos|accommodation)\b", normalized):
        return False
    if _PLANNER_GENERIC_ANCHOR_RE.fullmatch(normalized):
        return False
    generic_parts = [
        part.strip()
        for part in re.split(r"\s+(?:e|and)\s+", normalized)
        if part.strip()
    ]
    if generic_parts and all(_PLANNER_GENERIC_ANCHOR_RE.fullmatch(part) for part in generic_parts):
        return False
    counted_generic = re.match(
        r"^(?:\d{1,2}|um|uma|one|dois|duas|two|tres|three|quatro|four|cinco|five)\s+(.+)$",
        normalized,
    )
    if counted_generic:
        counted_parts = [
            part.strip()
            for part in re.split(r"\s+(?:e|and)\s+", counted_generic.group(1).strip())
            if part.strip()
        ]
        if counted_parts and all(_PLANNER_GENERIC_ANCHOR_RE.fullmatch(part) for part in counted_parts):
            return False
    component_parts = [
        part.strip()
        for part in re.split(r"\s+(?:e|and)\s+", normalized)
        if part.strip()
    ]
    if component_parts and all(_planner_anchor_part_is_generic_component(part) for part in component_parts):
        return False
    if _PLANNER_ANCHOR_KEYWORD_RE.search(normalized):
        return True
    if re.search(r"\b[A-Z0-9]{2,}\b", cleaned):
        return True
    if len(normalized.split()) >= 2 and re.search(r"\b[A-Z][A-Za-z0-9-]{2,}", cleaned):
        return True
    return len(normalized.split()) == 1 and cleaned[:1].isupper() and len(normalized) >= 4


def _split_requested_anchor_fragment(fragment: str) -> List[str]:
    """Split a user-provided place list into specific place candidates."""
    raw_parts = []
    for part in _PLANNER_ANCHOR_LIST_SPLIT_RE.split(str(fragment or "")):
        cleaned_part = _clean_requested_anchor_fragment(part)
        if not cleaned_part:
            continue
        route_split = re.split(
            r"\s+(?:to|para|ate|até)\s+",
            cleaned_part,
            maxsplit=1,
            flags=re.IGNORECASE,
        )
        if (
            len(route_split) == 2
            and _requested_anchor_fragment_is_specific(route_split[0].strip(" .:-"))
            and _requested_anchor_fragment_is_specific(route_split[1].strip(" .:-"))
        ):
            raw_parts.append(route_split[0].strip(" .:-"))
            continue
        conjoined_parts = re.split(
            r"\s+(?:e|and)\s+(?=[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ0-9])",
            cleaned_part,
        )
        if len(conjoined_parts) <= 1:
            raw_parts.append(cleaned_part)
            continue
        expanded_parts = [candidate.strip(" .:-") for candidate in conjoined_parts if candidate.strip(" .:-")]
        starts_with_specific_venue_type = bool(re.match(
            r"(?i)^\s*(?:museu|museum|pal[aá]cio|palace|mosteiro|monastery|"
            r"igreja|church|teatro|theatre|theater|restaurante|restaurant)\b",
            cleaned_part,
        ))
        if starts_with_specific_venue_type and not any(
            _PLANNER_ANCHOR_KEYWORD_RE.search(_normalize_planner_text(candidate))
            for candidate in expanded_parts[1:]
        ):
            raw_parts.append(cleaned_part)
        else:
            raw_parts.extend(expanded_parts)
    raw_parts = [part for part in raw_parts if part]
    if len(raw_parts) > 1:
        anchored_parts = [
            part
            for part in raw_parts
            if _PLANNER_ANCHOR_KEYWORD_RE.search(_normalize_planner_text(part))
            or re.fullmatch(r"[A-Z0-9][A-Z0-9 -]{1,20}", part.strip())
        ]
        if (
            len(anchored_parts) == 1
            and anchored_parts[0] == raw_parts[0]
            and not re.search(r"[,;/]|\s+(?:e|and)\s+", str(fragment or ""), flags=re.IGNORECASE)
        ):
            whole = _clean_requested_anchor_fragment(fragment)
            if _requested_anchor_fragment_is_specific(whole):
                return [whole]

    candidates: List[str] = []
    for cleaned in raw_parts:
        if _requested_anchor_fragment_is_specific(cleaned):
            candidates.append(cleaned)
    return candidates


def _extract_requested_anchor_phrases(user_message: str) -> List[str]:
    """Extract specific requested place names without relying on a fixed gazetteer."""
    text = str(user_message or "").strip()
    if not text:
        return []

    labels: List[str] = []
    seen: set[str] = set()

    def add_fragment(fragment: str) -> None:
        for candidate in _split_requested_anchor_fragment(fragment):
            key = _normalize_planner_text(candidate)
            if key and key not in seen:
                seen.add(key)
                labels.append(candidate)

    endpoint_patterns = [
        r"\bfrom\s+(?P<origin>[^,.;]+?)\s+(?:via|through)\s+(?P<waypoint>[^,.;]+?)\s+to\s+(?P<destination>[^,.;]+)",
        r"\bde\s+(?P<origin>[^,.;]+?)\s+(?:via|por|pela|pelo)\s+(?P<waypoint>[^,.;]+?)\s+(?:para|ate|até)\s+(?P<destination>[^,.;]+)",
        r"\bde\s+(?P<origin>[^,.;]+?)\s+(?:para|ate|até)\s+(?P<destination>[^,.;]+)",
        r"\bfrom\s+(?P<origin>[^,.;]+?)\s+to\s+(?P<destination>[^,.;]+)",
    ]
    for pattern in endpoint_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            add_fragment(match.group("origin"))
            if "waypoint" in match.groupdict():
                add_fragment(match.group("waypoint"))
            add_fragment(match.group("destination"))

    single_place_patterns = [
        r"\b(?:come\S*|iniciar|inicia|starting|start)\s+(?:no|na|em|at|from)\s+(?P<place>[^,.;]+)",
        r"\b(?:termin\S*|acab\S*|ending|end)\s+(?:no|na|em|at|in)\s+(?P<place>[^,.;]+)",
        r"\ba\s+partir\s+d\S*\s+(?P<place>[^,.;]+)",
        r"\bfrom\s+(?P<place>[^,.;]+?)(?:\s+(?:through|via|ending|end|finish|to)\b|[,.;]|$)",
    ]
    for pattern in single_place_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            add_fragment(match.group("place"))

    list_patterns = [
        r"\b(?:visitar|visit|inclui\S*|include|including|pass(?:a(?:r|ndo)?|e(?:m)?)\s+(?:por|pela|pelo)|"
        r"pass\s+through|through|via|faz(?:er)?\s+escala\s+(?:em|no|na)|stop\s+at|"
        r"walk\s+through|walk\s+around|caminhada\s+(?:por|pela|pelo)|passeio\s+(?:por|pela|pelo))\s+"
        r"(?P<places>[^.;]+)",
        r"\b(?:com|with)\s+(?P<places>[^.;]+)",
    ]
    for pattern in list_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            add_fragment(match.group("places"))

    normalized_text = _normalize_planner_text(text)
    if re.search(r"\b(?:comeca|comece|comecar|comecando|inicia|iniciar|iniciando|start|starting)\b", normalized_text):
        origin = _extract_requested_plan_origin(text)
        if origin:
            add_fragment(origin)

    end_area = _extract_requested_plan_area(text)
    end_key = _normalize_planner_text(end_area)
    if end_key and len(labels) > 1:
        end_labels = [label for label in labels if _normalize_planner_text(label) == end_key]
        if end_labels:
            labels = [label for label in labels if _normalize_planner_text(label) != end_key]
            labels.extend(end_labels)

    return labels


def _requested_anchor_time_constraints(user_message: str) -> List[tuple[str, str]]:
    """Extract explicit waypoint time constraints from the user's request."""
    text = str(user_message or "")
    if not text.strip():
        return []

    time_re = r"(?P<time>\d{1,2}(?:[:hH]\d{0,2})?)"
    patterns = [
        (
            r"\b(?:pass(?:a(?:r|ndo)?|e(?:m)?)|visita(?:r|ndo)?|visite(?:m)?|visit|stop(?:ping)?|via)\s+"
            r"(?:(?:por|pelo|pela|em|no|na|at)\s+)?(?P<place>[^,.;]+?)\s+"
            r"(?:as|às|pelas|por\s+volta\s+d(?:as?|e)?|around|at)\s+"
            + time_re
        ),
        (
            r"\b(?:almo[cç](?:ar|a|o|ando)?|lunch|jantar|dinner)\s+"
            r"(?:(?:em|no|na|at)\s+)?(?P<place>[^,.;]+?)\s+"
            r"(?:as|às|pelas|por\s+volta\s+d(?:as?|e)?|around|at)\s+"
            + time_re
        ),
        (
            r"\b(?:almo[cç](?:ar|a|o|ando)?|lunch|jantar|dinner)\s+"
            r"(?:as|às|pelas|por\s+volta\s+d(?:as?|e)?|around|at)\s+"
            + time_re
            + r"\s+(?:em|no|na|at)\s+(?P<place>[^,.;]+)"
        ),
        (
            r"\b(?:reserva|booking|reservation|reserved|booked)\s+"
            r"(?:as|às|pelas|por\s+volta\s+d(?:as?|e)?|around|at)\s+"
            + time_re
            + r"\s+(?:em|no|na|at)\s+(?P<place>[^,.;]+)"
        ),
        (
            r"\b(?:mant[eé]m|mantem|preserva|keep|keeping)\s+"
            r"(?:(?:o|a|os|as|the)\s+)?(?P<place>[^,.;]+?)\s+"
            r"(?:as|às|pelas|por\s+volta\s+d(?:as?|e)?|around|at|by)\s+"
            + time_re
        ),
        (
            r"\b(?:reserva|booking|reservation|reserved|booked)\s+"
            r"(?:(?:em|no|na|at)\s+)?(?P<place>[^,.;]+?)\s+"
            r"(?:as|às|pelas|por\s+volta\s+d(?:as?|e)?|around|at)\s+"
            + time_re
        ),
        (
            r"\b(?:estar|chegar|arrive|be)\s+(?:(?:em|no|na|at)\s+)"
            r"(?P<place>[^,.;]+?)\s+"
            r"(?:até|ate|by|as|às|at)\s+"
            + time_re
        ),
        (
            r"\b(?:as|às|at)\s+"
            + time_re
            + r"\s+(?:pass(?:a(?:r|ndo)?|e(?:m)?)\s+(?:por|pelo|pela)|visita(?:r|ndo)?|visite(?:m)?|visit|stop\s+at)\s+"
            r"(?P<place>[^,.;]+)"
        ),
        (
            r"\b(?:as|às|at)\s+"
            + time_re
            + r"\s+(?:almo[cç](?:ar|a|o)?|lunch|jantar|dinner)\s+"
            r"(?:(?:em|no|na|at)\s+)?(?P<place>[^,.;]+)"
        ),
    ]

    constraints: List[tuple[str, str]] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            place = _clean_requested_anchor_fragment(match.group("place"))
            if not _requested_anchor_fragment_is_specific(place):
                continue
            time_label = _normalize_requested_time_label(match.group("time"))
            if not time_label:
                continue
            key = f"{_normalize_planner_text(place)}|{time_label}"
            if key in seen:
                continue
            seen.add(key)
            constraints.append((place, time_label))
    return constraints[:5]


def _normalize_requested_time_label(value: str) -> str:
    """Normalize a user-supplied hour into HH:MM."""
    match = re.match(r"^\s*(?P<hour>\d{1,2})(?:[:hH](?P<minute>\d{0,2}))?\s*$", str(value or ""))
    if not match:
        return ""
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or "00")
    if hour > 23 or minute > 59:
        return ""
    return f"{hour:02d}:{minute:02d}"


def _requested_time_label_for_card(card: Dict[str, str], user_message: str) -> str:
    """Return the explicit requested time for a selected card, if any."""
    for label, time_label in _requested_anchor_time_constraints(user_message):
        if _card_kind_for_plan_block(card) == "food":
            title_key = _normalize_planner_text(_planner_card_display_name(card) or card.get("name", ""))
            label_key = _normalize_planner_text(label)
            direct_food_time_match = bool(
                title_key
                and label_key
                and (
                    title_key == label_key
                    or title_key in label_key
                    or (len(label_key.split()) >= 2 and label_key in title_key)
                )
            )
            if not direct_food_time_match:
                continue
        if _planner_card_matches_requested_label(card, label):
            return time_label
    return ""


_PLANNER_CENTRAL_AREA_RE = re.compile(
    r"\b(?:se de lisboa|catedral de lisboa|carmo|baixa|chiado|rossio|"
    r"praca do comercio|terreiro do paco|alfama|mouraria|castelo|"
    r"bacalhoeiros|douradores|correeiros|santa cruz do castelo)\b",
    re.IGNORECASE,
)
_PLANNER_BELEM_AREA_RE = re.compile(
    r"\b(?:belem|torre de belem|padrao dos descobrimentos|jeronimos|"
    r"mosteiro dos jeronimos|brasilia|imperio|doca do bom sucesso|bom sucesso)\b",
    re.IGNORECASE,
)


def _planner_area_is_broad_city(area: str) -> bool:
    """Return whether an extracted area is too broad for same-area walking."""
    normalized = _normalize_planner_text(area)
    return normalized in {
        "lisboa",
        "lisbon",
        "cidade de lisboa",
        "city of lisbon",
        "centro de lisboa",
        "central lisbon",
    }


_PREVIOUS_PLACE_SET_REFERENCE_RE = re.compile(
    r"\b(?:estes|estas|esses|essas|desses|dessas|these|those|previous|above|listed|"
    r"locais\s+(?:anteriores|acima|listados)|lugares\s+(?:anteriores|acima|listados)|"
    r"places\s+(?:above|listed|mentioned)|que\s+(?:disseste|mencionaste|indicaste))\b",
    re.IGNORECASE,
)


def _query_references_previous_place_set(user_message: str) -> bool:
    """Return whether the request refers to places from the previous answer."""
    normalized_query = _normalize_planner_text(user_message)
    return bool(_PREVIOUS_PLACE_SET_REFERENCE_RE.search(normalized_query))


def _previous_context_place_labels(conversation_context: str, max_items: int = 8) -> List[str]:
    """Extract visible place names from previous-answer context."""
    if not conversation_context:
        return []
    marker_match = re.search(
        r"(?is)Previous referenced places:\s*(?P<section>.*?)(?:\n\s*\nPrevious final plan excerpt:|\Z)",
        conversation_context,
    )
    if marker_match:
        conversation_context = marker_match.group("section")

    labels: List[str] = []
    seen: set[str] = set()

    def add_label(raw_label: str) -> None:
        label = _sanitize_planner_place_name(raw_label)
        key = _normalize_planner_text(label)
        if not key or key in seen:
            return
        if _planner_text_is_negative_result(label):
            return
        if re.search(
            r"\b(?:resposta direta|direct answer|fonte|source|dicas|tips|notas finais|final notes|"
            r"roteiro sugerido|suggested route|locais e atracoes|places and attractions)\b",
            key,
            flags=re.IGNORECASE,
        ):
            return
        seen.add(key)
        labels.append(label)

    for card in _extract_visitlisboa_place_cards(conversation_context, max_items=max_items, language="pt"):
        add_label(_planner_card_display_name(card) or card.get("name", ""))
        if len(labels) >= max_items:
            return labels

    card_title_re = re.compile(
        r"(?m)^\s*[-*]\s+\*\*(?:[\U0001F300-\U0001FAFF\u2300-\u27BF\uFE0F\u200D]+\s*)?"
        r"(?P<title>[^*\n]{2,120})\*\*\s*$"
    )
    for match in card_title_re.finditer(conversation_context):
        add_label(match.group("title"))
        if len(labels) >= max_items:
            break
    return labels


def _requested_anchor_labels(user_message: str, evidence_data: str = "") -> List[str]:
    """Extract explicit Lisbon anchors that should remain visible in a plan."""
    normalized_query = _normalize_planner_text(user_message)
    if not normalized_query:
        return []

    labels: List[str] = []
    seen: set[str] = set()
    exact_evidence_names: set[str] = set()
    excluded_keys = {
        _normalize_planner_text(area)
        for area in _extract_excluded_plan_areas(user_message)
        if _normalize_planner_text(area)
    }
    origin_label = _normalize_planner_text(_extract_requested_plan_origin(user_message))
    start_as_origin_only = bool(
        origin_label
        and re.search(
            r"\b(?:comeca|comece|comecar|comecando|inicia|iniciar|iniciando|"
            r"start|starting|a\s+partir|desde|from|hotel)\b",
            normalized_query,
        )
        and not re.search(
            rf"\b(?:visitar|visit|inclui|include|including|pass(?:a(?:r|ndo)?|e(?:m)?)\s+(?:por|pelo|pela)|via|pass\s+through)\b"
            rf"[^.;]*\b{re.escape(origin_label)}\b",
            normalized_query,
        )
    )
    explicit_start_end_sequence = bool(
        origin_label
        and (
            (
                re.search(
                    r"\b(?:comeca|comece|comecar|comecando|inicia|iniciar|iniciando|start|starting)\b",
                    normalized_query,
                )
                and re.search(
                    r"\b(?:termina|termine|terminar|acaba|acabe|acabar|end|ending|finish)\b",
                    normalized_query,
                )
            )
            or re.search(
                r"\b(?:from\s+.+\s+to|de\s+.+\s+(?:para|ate))\b",
                normalized_query,
            )
        )
    )

    def add_label(label: str) -> None:
        key = _normalize_planner_text(label)
        if key and any(key == excluded or excluded in key or key in excluded for excluded in excluded_keys):
            return
        if key and key not in seen:
            seen.add(key)
            labels.append(label)

    if evidence_data:
        for card in _extract_visitlisboa_place_cards(evidence_data, max_items=24, language="pt"):
            display_name = _planner_card_display_name(card) or card.get("name", "")
            normalized_name = _normalize_planner_text(display_name)
            if normalized_name:
                exact_evidence_names.add(normalized_name)
            if len(normalized_name.split()) < 2:
                continue
            if re.search(rf"\b{re.escape(normalized_name)}\b", normalized_query):
                add_label(display_name)

    if evidence_data and _query_references_previous_place_set(user_message):
        for label in _previous_context_place_labels(evidence_data):
            add_label(label)

    for label in _extract_requested_anchor_phrases(user_message):
        normalized_label = _normalize_planner_text(label)
        if (
            start_as_origin_only
            and not explicit_start_end_sequence
            and normalized_label == origin_label
            and normalized_label not in exact_evidence_names
        ):
            continue
        add_label(label)
    for label, _time_label in _requested_anchor_time_constraints(user_message):
        add_label(label)

    return labels


def _response_mentions_requested_anchor(response: str, label: str) -> bool:
    """Return whether a final response still mentions a requested anchor."""
    normalized_response = _normalize_planner_text(response)
    normalized_label = _normalize_planner_text(label)
    return bool(normalized_label and re.search(rf"\b{re.escape(normalized_label)}\b", normalized_response))


def _planner_response_missing_requested_stops(
    response: str,
    user_message: str,
    evidence_data: str = "",
) -> bool:
    """Return whether a planner answer omitted places explicitly requested by the user."""
    normalized_query = _normalize_planner_text(user_message)
    if not re.search(r"\b(?:roteiro|plano|itinerario|plan|itinerary|visitar|visit|dia|day)\b", normalized_query):
        return False
    requested_labels = _requested_anchor_labels(user_message, evidence_data)
    if not requested_labels:
        return False
    return any(
        not _response_mentions_requested_anchor(response, label)
        for label in requested_labels
    )


def _query_requests_movement_details(user_message: str) -> bool:
    """Return whether the user explicitly wants movement/transport guidance in a plan."""
    normalized = _normalize_planner_text(user_message)
    return bool(
        re.search(
            r"\b(?:como\s+(?:me|te)?\s*deslocas?|deslocacoes|deslocacao|deslocar|transportes?|"
            r"percurso|trajeto|rota|route|movement|how to move|get around|getting around|"
            r"otimizad[oa]s?|optimized|optimised|optimization|optimisation|"
            r"baixa\s+caminhada|pouca\s+caminhada|pouco\s+andar|low\s+walking|minimal\s+walking|"
            r"inclui\s+(?:como|transportes?|deslocacoes)|include\s+(?:transport|movement|how to))\b",
            normalized,
            flags=re.IGNORECASE,
        )
        or _query_requests_public_transport(user_message)
    )


def _query_requests_stop_by_stop_movement(user_message: str) -> bool:
    """Return whether the user asks for movement between itinerary stops."""
    normalized = _normalize_planner_text(user_message)
    return bool(
        re.search(
            r"\b(?:"
            r"(?:transportes?|deslocacoes|deslocacao|ligacoes|percurso|trajeto|route|movement)\s+entre\s+"
            r"(?:todas\s+as\s+)?(?:paragens|pontos|stops)|"
            r"entre\s+(?:todas\s+as\s+)?(?:paragens|pontos|stops)|"
            r"de\s+paragem\s+em\s+paragem|cada\s+paragem|cada\s+ponto|"
            r"between\s+(?:all\s+)?(?:itinerary\s+)?stops|"
            r"each\s+stop|every\s+stop|stop[-\s]?to[-\s]?stop"
            r")\b",
            normalized,
            flags=re.IGNORECASE,
        )
    )


def _query_requests_custard_tart_stop(user_message: str) -> bool:
    """Return whether the user explicitly asked for a pastel de nata stop."""
    normalized = _normalize_planner_text(user_message)
    return bool(
        re.search(
            r"\b(?:pastel(?:\s+de\s+nata)?|pasteis(?:\s+de\s+nata)?|nata|"
            r"custard(?:\s+tarts?)?|tarts?)\b",
            normalized,
        )
    )


def _query_requests_food_stop(user_message: str) -> bool:
    """Return whether the plan request explicitly asks for a meal or food stop."""
    normalized = _normalize_planner_text(user_message)
    return bool(
        _query_requests_custard_tart_stop(normalized)
        or re.search(
            r"\b(?:gastronom\w*|restaurants?|restaurantes?|food|comida|tradicional|"
            r"almo[cç]o|almoco|almo[cç]ar|comer|refei[cç][aã]o|meal|lunch|"
            r"jantar|dinner|cozinha|cafe|coffee|pastelaria|pastry|padaria|"
            r"brunch|pequeno\s+almoco|breakfast)\b",
            normalized,
        )
    )


def _query_requests_cultural_stop(user_message: str) -> bool:
    """Return whether the plan request explicitly asks for a cultural stop."""
    normalized = _normalize_planner_text(user_message)
    return bool(
        re.search(
            r"\b(?:museus?|museum|museums|monumentos?|monuments?|hist[oó]ric\w*|"
            r"heritage|patrim[oó]nio|cultural|culture|cultura|exposi[cç][aã]o|"
            r"exhibition|galeria|gallery|ocean[aá]rio|aquarium|pavilh[aã]o\s+do\s+conhecimento)\b",
            normalized,
        )
    )


def _query_requests_cafe_stop(user_message: str) -> bool:
    """Return whether the user specifically asked for a cafe or pastry stop."""
    normalized = _normalize_planner_text(user_message)
    return bool(
        _query_requests_custard_tart_stop(normalized)
        or re.search(
            r"\b(?:cafe|coffee|pastelaria|pastry|padaria|brunch|"
            r"pequeno\s+almoco|breakfast)\b",
            normalized,
        )
    )


def _query_requests_morning_window(user_message: str) -> bool:
    """Return whether the requested plan is anchored to the morning."""
    normalized = _normalize_planner_text(user_message)
    return bool(re.search(r"\b(?:manha|morning|antes\s+do\s+almoco|before\s+lunch)\b", normalized))


def _query_requests_low_walk_plan(user_message: str) -> bool:
    """Return whether the plan should avoid unnecessary walking or spread."""
    normalized = _normalize_planner_text(user_message)
    return bool(
        re.search(
            r"\b(?:pouca\s+caminhada|pouco\s+andar|caminhadas?\s+curtas?|"
            r"pouca\s+distancia|baixa\s+distancia|low\s+walk|less\s+walking|"
            r"desloca\w*\s+curtas?|transfer(?:e)?s?\s+curtos?|short\s+transfers?|"
            r"short\s+walks?|minimal\s+walking|reduced\s+mobility|"
            r"sem\s+grandes\s+caminhadas?|sem\s+muita\s+caminhada|"
            r"evita(?:r)?\s+(?:muita\s+)?caminhada|evita(?:r)?\s+caminhar|"
            r"avoid(?:ing)?\s+(?:too\s+much\s+)?walking|avoid(?:ing)?\s+long\s+walks?|"
            r"pouco\s+tempo\s+a\s+pe|pouco\s+tempo\s+a\s+p[eé])\b",
            normalized,
        )
    )


def _query_requests_walking_only_plan(user_message: str) -> bool:
    """Return whether the user requested a plan primarily done on foot."""
    normalized = _normalize_planner_text(user_message)
    if not normalized or _query_requests_public_transport(user_message):
        return False
    if _query_requests_low_walk_plan(user_message) or re.search(
        r"\b(?:evita(?:r)?|avoid(?:ing)?|sem|without|menos|less|minimal|pouca|pouco|reduced)\b"
        r".{0,45}\b(?:caminh\w*|walk(?:ing|s)?)\b",
        normalized,
    ):
        return False
    return bool(
        re.search(
            r"\b(?:a\s+p[eé]|a\s+pe|walking\s+tour|walking\s+itinerary|"
            r"walk\s+(?:through|around|in)|on\s+foot|walkable|caminhar|"
            r"caminhada|passeio\s+a\s+p[eé])\b",
            normalized,
        )
    )


def _query_requests_architecture_theme(user_message: str) -> bool:
    """Return whether the plan is explicitly themed around architecture."""
    normalized = _normalize_planner_text(user_message)
    return bool(re.search(r"\b(?:arquitetura|arquitectura|architecture|architectural)\b", normalized))


def _planner_walking_only_guidance(language: str) -> List[str]:
    """Return compact movement guidance for plans requested as walking-only."""
    if language == "pt":
        return [
            "🚶 **Plano a pé:** mantém as paragens agrupadas em zonas próximas e evita atravessar a cidade a pé no mesmo bloco.",
            "🧭 **Ritmo:** mantém 20-30 min de margem entre paragens e evita atravessar a cidade a pé no mesmo bloco.",
        ]
    return [
        "🚶 **Walking plan:** keep the stops grouped around nearby streets and avoid crossing the whole city on foot in the same block.",
        "🧭 **Pace:** keep 20-30 min between stops and avoid crossing the whole city on foot in the same block.",
    ]


def _query_describes_single_area_plan(user_message: str) -> bool:
    """Return whether the request asks for a compact plan inside one named area."""
    normalized = _normalize_planner_text(user_message)
    if _query_has_explicit_anchor_sequence(user_message):
        return False
    has_area_anchor = bool(_extract_compact_plan_area_anchor(user_message))
    return bool(
        _query_requests_low_walk_plan(user_message)
        or (has_area_anchor and _query_requests_walking_only_plan(user_message))
        or re.search(r"\b(?:em|no|na|nos|nas|in|around)\s+[^,.;]{2,80}\s+(?:com|with)\b", normalized)
        or (
            has_area_anchor
            and re.search(
                r"\b(?:mini\s*plano|mini\s*plan|plano\s+curto|short\s+plan|"
                r"manh(?:a)?\s+curta|short\s+morning|tarde\s+curta|short\s+afternoon|"
                r"2\s+horas?|two\s+hours|pouco\s+tempo|short\s+time)\b",
                normalized,
            )
        )
        or (
            has_area_anchor
            and re.search(
                r"\b(?:perto\s+d(?:e|o|a|os|as)|near|around|junto\s+a|"
                r"come[cç]ar|come[cç]ando|iniciar|iniciando|start|starting|"
                r"meio\s+dia|half\s+day|[2-5]\s+horas?|"
                r"paragens?.{0,40}(?:comer|refei[cç][aã]o|almo[cç]o|jantar)|"
                r"stops?.{0,40}(?:eat|food|meal|lunch|dinner))\b",
                normalized,
            )
        )
    )


def _extract_requested_plan_duration_minutes(user_message: str) -> Optional[int]:
    """Extract an explicit bounded-plan duration from the user request.

    The result is only used for relative time budgeting inside an itinerary. It
    is not a clock time and does not imply provider-confirmed availability.
    """
    normalized = _normalize_planner_text(user_message)
    if not normalized:
        return None

    if (
        re.search(r"\b(?:meio\s+dia|half\s+day|half-day)\b", normalized)
        and re.search(r"\b(?:roteiro|itinerario|itinerary|plano|plan|visitar|visit|paragens|stops)\b", normalized)
    ):
        return 4 * 60

    hour_minute_match = re.search(
        r"\b(?P<hours>\d{1,2})\s*h(?:\s*(?P<minutes>\d{1,2}))?\b",
        normalized,
    )
    if hour_minute_match:
        hours = int(hour_minute_match.group("hours"))
        minutes = int(hour_minute_match.group("minutes") or 0)
        prefix = normalized[max(0, hour_minute_match.start() - 24):hour_minute_match.start()]
        if hours > 8 or re.search(r"\b(?:as|pelas|at|around|volta\s+d(?:as?|e)?)\s*$", prefix):
            return None
        return hours * 60 + minutes

    hour_match = re.search(
        r"\b(?P<hours>\d{1,2})\s*(?:hora|horas|hour|hours)\b"
        r"(?:\s*(?:e|and)?\s*(?P<minutes>\d{1,2})\s*(?:min|mins|minuto|minutos|minute|minutes))?",
        normalized,
    )
    if hour_match:
        hours = int(hour_match.group("hours"))
        minutes = int(hour_match.group("minutes") or 0)
        return hours * 60 + minutes

    minute_match = re.search(
        r"\b(?P<minutes>\d{2,3})\s*(?:min|mins|minuto|minutos|minute|minutes)\b",
        normalized,
    )
    if minute_match:
        return int(minute_match.group("minutes"))

    return None


def _planner_time_allocations_for_cards(
    cards: Sequence[Dict[str, str]],
    total_minutes: Optional[int],
) -> List[int]:
    """Allocate a relative time budget across selected itinerary cards.

    Args:
        cards: Ordered planner cards selected for the route.
        total_minutes: Explicit duration requested by the user, if any.

    Returns:
        Minute allocations rounded to five-minute increments. Empty when the
        request does not provide a usable bounded duration.
    """
    if not total_minutes or total_minutes < 30 or not cards:
        return []

    visible_cards = list(cards[:5])
    transfer_buffer = min(25, max(0, (len(visible_cards) - 1) * 10))
    usable_minutes = max(20 * len(visible_cards), total_minutes - transfer_buffer)
    weights: List[float] = []
    minimums: List[int] = []
    for card in visible_cards:
        kind = _card_kind_for_plan_block(card)
        if kind == "food":
            weights.append(0.75)
            minimums.append(30)
        elif kind in {"coffee", "pastry"}:
            weights.append(0.45)
            minimums.append(20)
        else:
            weights.append(1.0)
            minimums.append(35)

    total_weight = sum(weights) or float(len(visible_cards))
    allocations = [
        max(minimum, int(round((usable_minutes * weight / total_weight) / 5.0) * 5))
        for weight, minimum in zip(weights, minimums)
    ]
    difference = usable_minutes - sum(allocations)
    if allocations:
        target_index = next(
            (
                index
                for index, card in enumerate(visible_cards)
                if _card_kind_for_plan_block(card) != "food"
            ),
            len(allocations) - 1,
        )
        allocations[target_index] = max(20, allocations[target_index] + difference)
    return allocations


def _planner_explicit_start_minutes(user_message: str) -> int | None:
    """Return an explicit itinerary start time if the user supplied one."""
    normalized = _normalize_planner_text(user_message)
    explicit_start = re.search(
        r"\b(?:a\s+partir\s+d(?:as?|e)|desde\s+as|come[cç]ar\s+as|come[cç]ando\s+as|"
        r"start(?:ing)?\s+at|from)\s+(?P<hour>\d{1,2})(?:[:h](?P<minute>\d{2}))?\b",
        normalized,
    )
    if explicit_start:
        hour = int(explicit_start.group("hour"))
        minute = int(explicit_start.group("minute") or 0)
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return hour * 60 + minute
    return None


def _planner_default_start_minutes(user_message: str) -> int:
    """Infer a practical default start time for bounded itinerary schedules."""
    explicit_start = _planner_explicit_start_minutes(user_message)
    if explicit_start is not None:
        return explicit_start
    normalized = _normalize_planner_text(user_message)
    if re.search(r"\b(?:tarde|afternoon)\b", normalized):
        return 14 * 60
    if re.search(r"\b(?:noite|evening|jantar|dinner)\b", normalized):
        return 18 * 60
    if re.search(r"\b(?:almoco|lunch)\b", normalized):
        return 10 * 60
    return 9 * 60 + 30


def _planner_schedule_labels_for_cards(
    cards: Sequence[Dict[str, str]],
    allocations: Sequence[int],
    user_message: str,
) -> List[str]:
    """Return human-readable suggested time windows for bounded plans."""
    if not cards or not allocations:
        return []
    current_minutes = _planner_default_start_minutes(user_message)
    explicit_start = _planner_explicit_start_minutes(user_message)
    if explicit_start is None and re.search(r"\b(?:almoco|almocar|lunch)\b", _normalize_planner_text(user_message)):
        first_food_index = next(
            (
                index
                for index, card in enumerate(cards[: len(allocations)])
                if _card_kind_for_plan_block(card) == "food"
            ),
            -1,
        )
        if first_food_index >= 0:
            minutes_before_food = sum(allocation for allocation in allocations[:first_food_index] if allocation > 0)
            current_minutes = max(9 * 60, 12 * 60 - minutes_before_food)
    if explicit_start is None:
        timed_anchor_index = next(
            (
                index
                for index, card in enumerate(cards[: len(allocations)])
                if _requested_time_label_for_card(card, user_message)
            ),
            -1,
        )
        if timed_anchor_index >= 0:
            timed_label = _requested_time_label_for_card(cards[timed_anchor_index], user_message)
            timed_minutes = _time_label_to_minutes(timed_label)
            if timed_minutes is not None:
                minutes_before_anchor = sum(
                    allocation
                    for allocation in allocations[:timed_anchor_index]
                    if allocation > 0
                )
                anchor_allocation = allocations[timed_anchor_index] if timed_anchor_index < len(allocations) else 0
                anchor_offset = min(15, max(0, anchor_allocation // 3))
                current_minutes = max(
                    8 * 60,
                    timed_minutes - anchor_offset - minutes_before_anchor,
                )
    labels: List[str] = []
    for allocation in list(allocations)[: len(cards)]:
        if allocation <= 0:
            labels.append("")
            continue
        end_minutes = current_minutes + allocation
        labels.append(
            f"{(current_minutes // 60) % 24:02d}:{current_minutes % 60:02d}-"
            f"{(end_minutes // 60) % 24:02d}:{end_minutes % 60:02d}"
        )
        current_minutes = end_minutes
    return labels


def _query_requests_budget_food(user_message: str) -> bool:
    """Return whether the user explicitly asks for a low-cost meal stop."""
    normalized_query = _normalize_planner_text(user_message)
    has_food_context = bool(
        re.search(
            r"\b(?:restaurante|restaurantes|restaurant|restaurants|comer|food|meal|"
            r"almo[cç]o|almoco|almo[cç]ar|almocar|jantar|dinner|lunch)\b",
            normalized_query,
        )
    )
    return bool(
        re.search(
            r"\b(?:barat\w*|econ[oó]mic\w*|baixo\s+custo|low\s+cost|cheap|budget|affordable|"
            r"under\s+20|menos\s+de\s+20)\b",
            normalized_query,
        )
        or (
            has_food_context
            and re.search(
                r"\b(?:ate|max(?:imo)?|maximum|below|under|menos\s+de)\s*\d{1,3}\b(?!\s*h)",
                normalized_query,
            )
        )
    )


def _food_card_budget_rank(card: Dict[str, str]) -> int:
    """Rank food cards by explicit price evidence; lower is cheaper."""
    raw_basis = " ".join(
        str(card.get(key, "") or "")
        for key in ("name", "category", "description", "features", "price")
    ).lower()
    raw_basis = unicodedata.normalize("NFKD", raw_basis)
    raw_basis = "".join(char for char in raw_basis if not unicodedata.combining(char))
    basis = _normalize_planner_text(
        " ".join(
            str(card.get(key, "") or "")
            for key in ("name", "category", "description", "features", "price")
        )
    )
    if re.search(r"(?:<|menos\s+de|under|ate|até)\s*20\s*(?:€|eur|euros?|\?)?", raw_basis):
        return 0
    if re.search(r"\b(?:low\s+cost|baixo\s+custo|barat\w*|cheap|budget)\b", basis):
        return 0
    if re.search(r"20\s*(?:€|eur|euros?|\?)?\s*(?:a|to|-)\s*50\s*(?:€|eur|euros?|\?)?", raw_basis):
        return 1
    if re.search(r"(?:>|mais\s+de|over)\s*50\s*(?:€|eur|euros?|\?)?", raw_basis):
        return 2
    return 3


def _score_food_budget_fit(card: Dict[str, str], user_message: str) -> int:
    """Score food cards against explicit low-cost meal preferences."""
    if not _query_requests_budget_food(user_message):
        return 0
    rank = _food_card_budget_rank(card)
    if rank == 0:
        return 180
    if rank == 1:
        return -180
    if rank == 2:
        return -320
    return -90


def _food_card_matches_requested_context(card: Dict[str, str], user_message: str) -> bool:
    """Return whether a food card fits explicit cafe/time-of-day constraints."""
    normalized_query = _normalize_planner_text(user_message)
    basis = _normalize_planner_text(
        " ".join(
            str(card.get(key, ""))
            for key in ("name", "category", "description", "features", "hours", "price")
        )
    )
    if _query_requests_budget_food(user_message) and _food_card_budget_rank(card) != 0:
        return False
    if re.search(r"\b(?:tradicional|traditional|tipic\w*|typical|portuguesa|portuguese)\b", normalized_query):
        if not re.search(
            r"\b(?:tradicional|traditional|tipic\w*|typical|portuguesa|portuguese|"
            r"cozinha\s+portuguesa|portuguese\s+cuisine|local\s+cuisine)\b",
            basis,
        ):
            return False
    if _query_requests_cafe_stop(normalized_query):
        food_break_pattern = (
            r"\b(?:pastel(?:\s+de\s+nata)?|pasteis(?:\s+de\s+nata)?|nata|custard(?:\s+tarts?)?|"
            r"tarts?|cafe|coffee|pastelaria|pastry|padaria|bakery|brunch|pequeno\s+almoco|breakfast)\b"
            if _query_requests_custard_tart_stop(normalized_query)
            else r"\b(?:cafe|coffee|pastelaria|pastry|padaria|brunch|pequeno\s+almoco|breakfast)\b"
        )
        if not re.search(food_break_pattern, basis):
            return False
    if _query_requests_morning_window(normalized_query) and card.get("hours"):
        morning_probe_minutes = (9 * 60 + 30, 10 * 60 + 30, 11 * 60 + 30)
        if not any(_card_open_at_minutes(card, minutes) for minutes in morning_probe_minutes):
            return False
    return True


def _query_has_explicit_anchor_sequence(user_message: str) -> bool:
    """Return whether the user requested a concrete ordered sequence of places."""
    normalized = _normalize_planner_text(user_message)
    if _query_has_explicit_start_end_constraint(user_message):
        return True
    requested_labels = _requested_anchor_labels(user_message)
    if len(requested_labels) < 2:
        return False
    return bool(
        re.search(r"\bde\s+.+\s+(?:para|ate)\s+.+", normalized)
        or re.search(r"\bfrom\s+.+\s+to\s+.+", normalized)
        or re.search(r"\b(?:pass(?:a(?:r|ndo)?|e(?:m)?)\s+(?:por|pelo|pela)|via|through|pass\s+through|stop\s+at)\b", normalized)
        or (
            re.search(r"\b(?:comeca|comece|comecar|comecando|inicia|iniciar|iniciando|start|starting)\b", normalized)
            and re.search(r"\b(?:termina|termine|terminar|acaba|acabe|acabar|end|ending|finish)\b", normalized)
        )
        or (
            len(requested_labels) >= 3
            and re.search(r"\b(?:com|with|inclui|include|visitar|visit)\b", normalized)
            and re.search(r"\b(?:roteiro|itinerario|itinerary|plano|plan|dia|day|paragens|stops|transportes?|transport)\b", normalized)
        )
    )


def _query_has_explicit_start_end_constraint(user_message: str) -> bool:
    """Return whether a plan names both a starting point and an ending area."""
    normalized = _normalize_planner_text(user_message)
    if re.search(r"\bfrom\s+[^,.;]+?\s+to\s+[^,.;]+", normalized):
        return bool(_extract_requested_plan_origin(user_message) and _extract_requested_plan_area(user_message))
    if not (
        re.search(r"\b(?:comeca|comece|comecar|comecando|inicia|iniciar|iniciando|start|starting)\b", normalized)
        and re.search(r"\b(?:termina|termine|terminar|terminando|acaba|acabe|acabar|acabando|end|ending|finish|finishing)\b", normalized)
    ):
        return False
    return bool(_extract_requested_plan_origin(user_message) and _extract_requested_plan_area(user_message))


def _requested_sequence_transport_limitation_bullets(user_message: str, language: str) -> List[str]:
    """Build scoped movement limitations for an explicit requested place sequence."""
    labels = _requested_anchor_labels(user_message)
    origin = labels[0] if labels else ""
    destination = labels[-1] if len(labels) >= 2 else ""
    if language == "pt":
        if origin and destination:
            return [
                f"⚠️ **{origin} → {destination}:** a ligação concreta não ficou confirmada nos dados recolhidos; não inventei linhas, paragens, durações ou partidas."
            ]
        return [
            "⚠️ As ligações exatas entre estas paragens não ficaram confirmadas nos dados recolhidos; não inventei linhas, paragens, durações ou partidas."
        ]
    if origin and destination:
        return [
            f"⚠️ **{origin} → {destination}:** the concrete connection was not confirmed in the gathered data; I did not invent lines, stops, durations, or departures."
        ]
    return [
        "⚠️ Exact legs between these stops were not confirmed in the gathered data; I did not invent lines, stops, durations, or departures."
    ]


def _text_mentions_central_and_belem(text: str) -> bool:
    """Return whether text mentions both central Lisbon and Belém-area anchors."""
    normalized = _normalize_planner_text(text)
    return bool(_PLANNER_CENTRAL_AREA_RE.search(normalized) and _PLANNER_BELEM_AREA_RE.search(normalized))


def _has_concrete_cross_zone_movement(response: str) -> bool:
    """Return whether the response includes a concrete central-Lisbon to Belém leg."""
    normalized_response = _normalize_planner_text(response)
    if not _text_mentions_central_and_belem(normalized_response):
        return False
    concrete_mode_re = re.compile(
        r"\b(?:carris\s+\d{2,4}[a-z]?|\d{1,4}e|\d{3,4}[a-z]?|linha\s+(?:\d{2,4}[a-z]?|verde|azul|amarela|vermelha|de\s+(?:cascais|sintra|azambuja|sado))|"
        r"line\s+(?:\d{2,4}[a-z]?|green|blue|yellow|red)|cp\s+linha|comboio\s+linha|train\s+line)\b",
        re.IGNORECASE,
    )
    for raw_line in str(response or "").splitlines():
        normalized_line = _normalize_planner_text(raw_line)
        if not (_planner_text_has_route_arrow(raw_line) or " para " in f" {normalized_line} " or " to " in f" {normalized_line} "):
            continue
        if (
            _PLANNER_CENTRAL_AREA_RE.search(normalized_line)
            and _PLANNER_BELEM_AREA_RE.search(normalized_line)
            and concrete_mode_re.search(normalized_line)
        ):
            return True
    return bool(
        concrete_mode_re.search(normalized_response)
        and _PLANNER_CENTRAL_AREA_RE.search(normalized_response)
        and _PLANNER_BELEM_AREA_RE.search(normalized_response)
    )


def _has_explicit_cross_zone_limitation(response: str) -> bool:
    """Return whether the answer states the exact central-Belém leg is unconfirmed."""
    normalized_response = _normalize_planner_text(response)
    return bool(
        _text_mentions_central_and_belem(normalized_response)
        and re.search(
            r"\b(?:nao confirmad\w*|unconfirmed|sem ligacao confirmad\w*|sem ligacao concreta|did not confirm|not confirmed)\b",
            normalized_response,
            flags=re.IGNORECASE,
        )
    )


def _planner_response_missing_requested_movement(
    response: str,
    user_message: str,
    transport_data: str,
) -> bool:
    """Return whether a plan drops requested movement details between requested areas."""
    if not _query_requests_movement_details(user_message):
        return False
    if _query_has_explicit_anchor_sequence(user_message) and not re.search(
        r"\b(?:como\s+te\s+deslocas|how\s+to\s+move|transporte|transport|"
        r"liga[cç][aã]o|ligacao|percurso|trajeto|route)\b",
        _normalize_planner_text(response),
        flags=re.IGNORECASE,
    ):
        return True
    if _query_has_explicit_anchor_sequence(user_message):
        movement_text = _planner_movement_section_text(response)
        requested_labels = _requested_anchor_labels(user_message)
        if len(requested_labels) >= 2:
            stop_by_stop_movement = _query_requests_stop_by_stop_movement(user_message)
            origin = _normalize_planner_text(requested_labels[0])
            destination = _normalize_planner_text(requested_labels[-1])
            normalized_movement = _normalize_planner_text(movement_text)
            if (
                not stop_by_stop_movement
                and (
                    not normalized_movement
                    or origin not in normalized_movement
                    or destination not in normalized_movement
                )
            ):
                return True
            has_explicit_limitation = bool(
                re.search(
                    r"\b(?:nao ficou confirmad\w*|não ficou confirmad\w*|not confirmed|"
                    r"sem ligacao confirmad\w*|sem ligação confirmad\w*|nao inventei|não inventei|did not invent)\b",
                    normalized_movement,
                    flags=re.IGNORECASE,
                )
            )
            has_concrete_leg = any(
                _planner_transport_bullet_is_actionable(line)
                and not re.search(r"\b(?:percurso base|base route)\b", _normalize_planner_text(line))
                for line in movement_text.splitlines()
            )
            if not has_explicit_limitation and not has_concrete_leg:
                return True
            movement_leg_lines = [
                line for line in movement_text.splitlines()
                if _planner_text_has_route_arrow(line)
            ]
            if stop_by_stop_movement:
                route_blocks = _planner_response_route_blocks(response)
                required_legs = min(4, max(0, len(route_blocks) - 1))
                covered_legs = [
                    line for line in movement_leg_lines
                    if not re.search(r"\b(?:percurso base|base route)\b", _normalize_planner_text(line))
                ]
                if required_legs >= 2:
                    return len(covered_legs) < required_legs
            if (
                movement_leg_lines
                and not _query_requests_food_stop(user_message)
                and not any(
                    _movement_item_matches_requested_sequence(line, user_message)
                    for line in movement_leg_lines
                )
            ):
                return True
    normalized_query = _normalize_planner_text(user_message)
    if not _text_mentions_central_and_belem(normalized_query):
        return False

    transport_has_cross_zone_evidence = bool(
        _text_mentions_central_and_belem(transport_data)
        or re.search(r"\bcarris\b", _normalize_planner_text(transport_data), flags=re.IGNORECASE)
    )
    if transport_has_cross_zone_evidence:
        return not (
            _has_concrete_cross_zone_movement(response)
            or _has_explicit_cross_zone_limitation(response)
        )
    return not (
        _has_concrete_cross_zone_movement(response)
        or _has_explicit_cross_zone_limitation(response)
    )


def _planner_movement_section_text(response: str) -> str:
    """Extract only the movement section from a planner response."""
    lines: list[str] = []
    in_movement = False
    for raw_line in str(response or "").splitlines():
        stripped = raw_line.strip()
        normalized = _normalize_planner_text(stripped)
        if re.match(r"^###\s+", stripped) and re.search(
            r"\b(?:como te deslocas|how to move)\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            in_movement = True
            lines.append(stripped)
            continue
        if in_movement and stripped.startswith("### "):
            break
        if in_movement and _PLANNER_SOURCE_LINE_RE.match(stripped):
            break
        if in_movement:
            lines.append(stripped)
    return "\n".join(lines).strip()


def _movement_item_matches_requested_sequence(item: str, user_message: str) -> bool:
    """Return whether a movement item belongs to the user's explicit X-to-Y pair."""
    requested_labels = _requested_anchor_labels(user_message)
    if len(requested_labels) < 2:
        return True
    normalized_item = _normalize_planner_text(item)
    origin = _normalize_planner_text(requested_labels[0])
    destination = _normalize_planner_text(requested_labels[-1])
    return bool(origin and destination and origin in normalized_item and destination in normalized_item)


def _movement_item_is_self_referential_origin(item: str, user_message: str) -> bool:
    """Return whether a movement line loops from an accommodation origin to itself."""
    origin = _normalize_planner_text(_extract_requested_plan_origin(user_message))
    if not origin or not _planner_text_has_route_arrow(item):
        return False
    parts = _PLANNER_ROUTE_ARROW_RE.split(str(item or ""), maxsplit=1)
    if len(parts) < 2:
        return False
    left = _normalize_planner_text(re.sub(r"^[-*]\s*(?:\*\*)?", "", parts[0]))
    right = _normalize_planner_text(re.split(r":|\*\*", parts[1], maxsplit=1)[0])
    if not left or not right:
        return False
    if origin in left and right == origin:
        return True
    return "hotel" in origin and origin in left and right in {"hotel", origin}


def _planner_response_violates_requested_start(response: str, user_message: str) -> bool:
    """Return whether a plan ignores an explicitly requested starting anchor."""
    origin = _extract_requested_plan_origin(user_message)
    if not origin:
        normalized_query = _normalize_planner_text(user_message)
        has_start_verb = bool(
            re.search(r"\b(?:comeca|comece|comecar|comecando|inicia|iniciar|iniciando|start|starting)\b", normalized_query)
        )
        if has_start_verb:
            requested_labels = _extract_requested_anchor_phrases(user_message)
            origin = requested_labels[0] if requested_labels else ""
    normalized_origin = _normalize_planner_text(origin)
    if not normalized_origin:
        return False
    normalized_response = _normalize_planner_text(response)
    if re.search(
        rf"\b{re.escape(normalized_origin)}\b.{{0,80}}\b(?:primeira|first|comeca|start|segue|walk|caminhada|ligacao|leg)\b",
        normalized_response,
    ):
        return False

    in_route_section = False
    for raw_line in str(response or "").splitlines():
        stripped = raw_line.strip()
        normalized_line = _normalize_planner_text(stripped)
        if re.search(r"\b(?:roteiro sugerido|suggested route)\b", normalized_line):
            in_route_section = True
            continue
        if in_route_section and stripped.startswith("### "):
            return False
        if not in_route_section:
            continue
        if not re.match(r"^[-*]\s+\*\*.+\*\*", stripped):
            continue
        first_title = re.sub(r"^[-*]\s+\*\*", "", stripped)
        first_title = re.sub(r"\*\*.*$", "", first_title)
        return normalized_origin not in _normalize_planner_text(first_title)
    return False


def _planner_response_has_unrequested_sequence_stops(response: str, user_message: str) -> bool:
    """Return whether a strict X-to-Y plan adds visible stops the user did not ask for."""
    if not _query_has_explicit_anchor_sequence(user_message):
        return False
    requested_labels = _requested_anchor_labels(user_message)
    if len(requested_labels) < 2:
        return False
    allowed_keys = [_normalize_planner_text(label) for label in requested_labels if _normalize_planner_text(label)]
    allow_food = _query_requests_food_stop(user_message)

    in_route_section = False
    visible_titles: List[str] = []
    for raw_line in str(response or "").splitlines():
        stripped = raw_line.strip()
        normalized_line = _normalize_planner_text(stripped)
        if re.search(r"\b(?:roteiro sugerido|suggested route)\b", normalized_line):
            in_route_section = True
            continue
        if in_route_section and stripped.startswith("### "):
            break
        if not in_route_section:
            continue
        if raw_line[:1].isspace():
            continue
        match = re.match(r"^(?:[-*]\s+)?\*\*(?P<title>[^*]+)\*\*", stripped)
        if match:
            title = re.sub(r"^[^\wÀ-ÿ]+", "", match.group("title")).strip()
            if re.match(
                r"^\s*(?:morada|address|descri[cç][aã]o|description|pre[cç]o|price|"
                r"hor[aá]rio|hours|website|bilhetes|tickets|categoria|category)\s*:",
                title,
                flags=re.IGNORECASE,
            ):
                continue
            if title:
                visible_titles.append(title)

    for title in visible_titles:
        normalized_title = _normalize_planner_text(title)
        if any(key and (key in normalized_title or normalized_title in key) for key in allowed_keys):
            continue
        if allow_food and re.search(r"\b(?:almoco|almo[cç]o|jantar|restaurant|restaurante|food|comida)\b", normalized_title):
            continue
        return True
    return False


def _query_requests_central_corridor_plan(user_message: str) -> bool:
    """Return whether the user constrains a plan to Lisbon's central corridor."""
    normalized = _normalize_planner_text(user_message)
    if not normalized:
        return False
    central_hits = sum(
        bool(re.search(rf"\b{term}\b", normalized))
        for term in ("baixa", "chiado", "alfama", "rossio", "carmo", "mouraria")
    )
    return central_hits >= 2


def _planner_local_area_profile(user_message: str) -> tuple[str, str, tuple[str, ...]]:
    """Return a requested compact-area profile and far-area blockers."""
    normalized = _normalize_planner_text(user_message)
    target_area = _normalize_planner_text(
        _extract_requested_plan_area(user_message)
        or _extract_compact_plan_area_anchor(user_message)
    )
    probe = f"{normalized} {target_area}"
    if not (
        _query_describes_single_area_plan(user_message)
        or _query_requests_central_corridor_plan(user_message)
        or re.search(r"\b(?:mini\s*plano|mini\s*plan|2\s+horas?|two\s+hours|pouco\s+tempo|short\s+time)\b", normalized)
    ):
        return "", "", ()

    if _query_requests_central_corridor_plan(user_message):
        return (
            "central_corridor",
            "Baixa / Chiado / Alfama",
            (
                "belem",
                "torre de belem",
                "padrao dos descobrimentos",
                "mosteiro dos jeronimos",
                "jeronimos",
                "oriente",
                "parque das nacoes",
                "expo",
                "marvila",
                "1950",
                "oeiras",
                "almada",
                "sintra",
                "cascais",
                "alcantara",
            ),
        )

    if re.search(r"\b(?:oriente|parque das nacoes|expo|estacao do oriente|station oriente)\b", probe):
        return (
            "oriente",
            "Oriente / Parque das Nações",
            (
                "belem",
                "torre de belem",
                "padrao dos descobrimentos",
                "mosteiro dos jeronimos",
                "jeronimos",
                "carmo",
                "chiado",
                "baixa",
                "alfama",
            ),
        )
    if re.search(r"\b(?:belem|torre de belem|padrao dos descobrimentos|jeronimos|mosteiro dos jeronimos)\b", probe):
        return (
            "belem",
            "Belém",
            ("oriente", "parque das nacoes", "expo", "alfama", "carmo", "chiado", "baixa"),
        )
    if re.search(r"\b(?:alfama|se de lisboa|catedral de lisboa|santa luzia|portas do sol)\b", probe):
        return (
            "alfama",
            "Alfama",
            ("belem", "torre de belem", "padrao dos descobrimentos", "oriente", "parque das nacoes", "expo"),
        )
    if re.search(
        r"\b(?:marques|marques de pombal|marques pombal|avenida da liberdade|parque eduardo vii|"
        r"saldanha|picoas|sao sebastiao|gulbenkian)\b",
        probe,
    ):
        return (
            "marques_de_pombal",
            "Marquês de Pombal / Avenida da Liberdade / Saldanha",
            (
                "oeiras",
                "2784",
                "palacio marques de pombal",
                "madragoa",
                "esperanca",
                "madre de deus",
                "azulejo",
                "ajuda",
                "janelas verdes",
                "1249",
                "1349",
                "belem",
                "torre de belem",
                "padrao dos descobrimentos",
                "oriente",
                "parque das nacoes",
                "expo",
                "marvila",
                "1950",
                "almada",
                "sintra",
                "cascais",
            ),
        )
    if (
        _query_requests_low_walk_plan(user_message)
        and re.search(r"\b(?:lisboa|lisbon)\b", normalized)
        and re.search(
            r"\b(?:historic|historical|historia|historico|historicos|monument|monuments|"
            r"monumento|monumentos|heritage|patrimonio|cultural|gastronom|tradicional|"
            r"traditional|almoco|lunch)\b",
            normalized,
        )
    ):
        return (
            "central_corridor",
            "Baixa / Chiado / Alfama",
            (
                "belem",
                "torre de belem",
                "padrao dos descobrimentos",
                "mosteiro dos jeronimos",
                "jeronimos",
                "oriente",
                "parque das nacoes",
                "expo",
                "marvila",
                "1950",
                "oeiras",
                "almada",
                "sintra",
                "cascais",
                "alcantara",
            ),
        )
    return "", "", ()


def _planner_response_has_local_area_drift(response: str, user_message: str) -> bool:
    """Return whether a compact local plan includes clearly distant anchors."""
    area_key, _area_label, blockers = _planner_local_area_profile(user_message)
    if not area_key or not blockers:
        return False
    normalized_response = _normalize_planner_text(response)
    if area_key == "oriente" and "museu do oriente" in normalized_response:
        return True
    if any(re.search(rf"\b{re.escape(blocker)}\b", normalized_response) for blocker in blockers):
        return True

    area_allowed_re = {
        "oriente": re.compile(
            r"\b(?:oriente|parque\s+das\s+nacoes|expo|oceanario|pavilhao\s+do\s+conhecimento|"
            r"centro\s+vasco\s+da\s+gama|vasco\s+da\s+gama|fil|altice\s+arena|"
            r"alameda\s+dos\s+oceanos|rua\s+do\s+bojador|rossio\s+dos\s+olivais|1990|1998)\b"
        ),
        "belem": re.compile(
            r"\b(?:belem|brasilia|jeronimos|padrao|descobrimentos|torre\s+de\s+belem|imperio|india|1400)\b"
        ),
        "alfama": re.compile(
            r"\b(?:alfama|se\s+de\s+lisboa|catedral\s+de\s+lisboa|santa\s+luzia|portas\s+do\s+sol|mouraria|1100)\b"
        ),
        "central_corridor": re.compile(
            r"\b(?:baixa|chiado|alfama|rossio|carmo|mouraria|se\s+de\s+lisboa|"
            r"catedral\s+de\s+lisboa|praca\s+do\s+comercio|terreiro\s+do\s+paco|"
            r"restauradores|santa\s+justa|1200|1100|1150)\b"
        ),
        "marques_de_pombal": re.compile(
            r"\b(?:marques\s+de\s+pombal|marques\s+pombal|avenida\s+da\s+liberdade|"
            r"parque\s+eduardo\s+vii|rua\s+de\s+santa\s+marta|rua\s+rosa\s+araujo|"
            r"rodrigues\s+sampaio|gulbenkian|berna|tomas\s+ribeiro|tom[aá]s\s+ribeiro|"
            r"sao\s+pedro\s+de\s+alcantara|s\s+pedro\s+de\s+alcantara|torel|"
            r"1250|1150|1050|1067|1070)\b"
        ),
    }
    allowed_re = area_allowed_re.get(area_key)
    if not allowed_re:
        return False

    in_route_section = False
    current_block: list[str] = []
    card_blocks: list[str] = []
    card_heading_re = re.compile(r"^[-*]\s+\*\*(?:[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D]+\s+)?[^*\n]+\*\*")
    for raw_line in str(response or "").splitlines():
        stripped = raw_line.strip()
        normalized_line = _normalize_planner_text(stripped)
        if re.search(r"\b(?:roteiro sugerido|suggested route)\b", normalized_line):
            in_route_section = True
            continue
        if in_route_section and stripped.startswith("### "):
            break
        if not in_route_section:
            continue
        if card_heading_re.match(stripped):
            if current_block:
                card_blocks.append("\n".join(current_block))
            current_block = [stripped]
            continue
        if current_block and (raw_line[:1].isspace() or stripped.startswith("- ")):
            current_block.append(stripped)
    if current_block:
        card_blocks.append("\n".join(current_block))

    for block in card_blocks:
        normalized_block = _normalize_planner_text(block)
        if re.search(r"\b(?:almoco|almoço|jantar|refeicao|refeição|meal)\b", normalized_block) and area_key in normalized_block:
            continue
        location_evidence_lines = []
        for line_index, raw_block_line in enumerate(block.splitlines()):
            normalized_block_line = _normalize_planner_text(raw_block_line)
            if line_index == 0 or re.search(
                r"\b(?:morada|address|venue|local|categoria|category)\b",
                normalized_block_line,
            ):
                location_evidence_lines.append(raw_block_line)
        location_evidence = _normalize_planner_text("\n".join(location_evidence_lines))
        if not allowed_re.search(location_evidence):
            return True
    return False


def _planner_response_mixes_distant_walking_areas(response: str, user_message: str) -> bool:
    """Return whether one day of a walking plan mixes distant Lisbon areas."""
    if (
        not _query_requests_walking_only_plan(user_message)
        or _query_has_explicit_anchor_sequence(user_message)
        or (_extract_requested_day_count(user_message) or 1) <= 1
    ):
        return False

    day_chunks = re.split(
        r"(?im)^\s*[-*]?\s*\*\*(?:dia|day)\s+\d+[^*\n]*\*\*.*$",
        str(response or ""),
    )
    if len(day_chunks) <= 1:
        return False
    for chunk in day_chunks[1:]:
        normalized = _normalize_planner_text(chunk)
        if _PLANNER_CENTRAL_AREA_RE.search(normalized) and _PLANNER_BELEM_AREA_RE.search(normalized):
            return True
    return False


def _planner_text_is_negative_result(text: str) -> bool:
    """Return whether text is a no-result diagnostic, not place evidence."""
    normalized = _normalize_planner_text(text)
    return bool(
        re.search(
            r"\b(?:nao\s+(?:foram\s+)?encontrad\w*|nao\s+encontrei|"
            r"i\s+could\s+not\s+find|no\s+matching|no\s+results?)\b",
            normalized,
        )
    )


def _sanitize_planner_place_name(candidate: str) -> str:
    """Return a clean place name, or an empty string for labels/fragments."""
    cleaned = re.sub(r"\s+", " ", str(candidate or "")).strip().strip("-–—: ")
    parts = [part.strip() for part in cleaned.split("|")]
    if len(parts) == 2:
        left = _normalize_planner_text(parts[0])
        right = _normalize_planner_text(parts[1])
        generic_suffixes = {
            "restaurante",
            "restaurantes",
            "restaurant",
            "restaurants",
            "food restaurants",
            "food and restaurants",
            "evento",
            "event",
            "events",
            "museu",
            "museum",
            "monumento",
            "monument",
        }
        if left == right or right in generic_suffixes:
            cleaned = parts[0]
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
        or (len(normalized) < 3 and normalized not in {"se"})
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
            r"\b(public transport|transportes publicos|metro|bus|autocarro|comboio|train|tram|eletrico|"
            r"route|rota|percurso|trajeto|deslocacao|deslocacoes|desloca[cç][aã]o|desloca[cç][oõ]es|"
            r"how do i get|como vou|como chego)\b",
            normalized,
            re.IGNORECASE,
        )
    )


def _query_requests_specific_transport_mode(user_message: str) -> bool:
    """Return whether the user named a public-transport operator or mode."""
    normalized = _normalize_planner_text(user_message)
    return bool(
        re.search(
            r"\b(?:metro|metro de lisboa|carris|cp|comboio|comboios|train|trains|"
            r"bus|buses|autocarro|autocarros|tram|eletrico|electrico|"
            r"transportes publicos|transporte publico|public transport)\b",
            normalized,
            re.IGNORECASE,
        )
    )


def _query_requests_live_transport_status(user_message: str) -> bool:
    """Detect whether the user explicitly asks for live or next-departure transport data."""
    normalized = _normalize_planner_text(user_message)
    if re.search(r"\b(?:agora|tempo real|em direto|live|real[- ]?time|right now|current)\b", normalized, re.IGNORECASE):
        return True
    return bool(
        re.search(
            r"\bproxim[oa]s?\s+(?:metro|metros|autocarro|autocarros|bus|comboio|comboios|train|trains|partida|partidas|saida|saidas|chegada|chegadas)\b",
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
        "- Preserve the user's stated origin and target in the movement section when they are present in the request.\n"
        "- Do not replace missing route details with vague prose such as 'continue by public transport', "
        "'verify locally', 'use the exact street address', or 'check the most direct connection locally'.\n"
        "- Do not present live next departures unless the user explicitly asked for next departures, live status, or current waiting times.\n"
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


# Generic placeholder/marketing-style phrases that the LLM sometimes pads the
# "How to move" / "Como te deslocas" section with when grounded route legs are
# scarce. Each phrase is brochure-style and lacks any concrete origin,
# destination, line, or operator. The sanitizer below strips bullets that
# match any of these AND lack a concrete `→` arrow / time / line marker.
_PLANNER_MOVEMENT_PLACEHOLDER_PHRASES = (
    r"rota\s+otimizada",
    r"rotas?\s+mais\s+convenientes?",
    r"(?:metro|carris|cp)\s*[,/]\s*(?:metro|carris|cp)\s*(?:[,/]\s*(?:metro|carris|cp))?\s+(?:mais\s+)?conven",
    r"tempo\s+estimado\s+de\s+desloca",
    r"liga[cç][oõ]es?\s+diretas?(?!\s*[:→])",
    r"transportes?\s+mais\s+convenientes?",
    r"optimi[sz]ed\s+route",
    r"most\s+convenient\s+(?:line|route|transport|mode|public\s+transport)",
    r"estimated\s+travel\s+time(?!\s*:?\s*\d)",
    r"direct\s+connections?(?!\s*[:→])",
    r"ponto\s+de\s+origem",
    r"consoante\s+a\s+zona",
    r"dependendo\s+da\s+zona",
    r"conforme\s+a\s+zona",
    r"se\s+estiver(?:em)?\s+na\s+rede",
    r"entre\s+os\s+dois\s+locais",
)


def _movement_bullet_has_concrete_signal(text: str) -> bool:
    """Return True when a movement bullet carries a concrete grounded signal.

    Concrete signals are: an origin→destination arrow, a clock/time pattern
    (e.g. ``10 min``, ``12:34``), a named line (``Linha Verde``, ``15E``,
    ``736``), an action verb anchored to a place (``Embarque em``,
    ``Transbordo em``), or a walking distance like ``300 m``. Bullets without
    any of these are treated as brochure padding.
    """
    if _planner_text_has_route_arrow(text):
        return True
    if re.search(r"\b\d{1,3}\s*(?:min|m|km)\b", text, re.IGNORECASE):
        return True
    if re.search(r"\b\d{1,2}[:h]\d{2}\b", text):
        return True
    normalized = _normalize_planner_text(text)
    if (
        re.search(r"\b(?:caminhada|a pe|walk|walking)\b", normalized)
        and re.search(r"\b(?:entre|between)\b", normalized)
    ):
        return True
    if re.search(r"\b(?:linha)\s+(?:verde|azul|amarela|vermelha|green|blue|yellow|red|de\s+\w+)\b", text, re.IGNORECASE):
        return True
    if re.search(r"\b(?:linha\s+)?\d{1,3}[A-Za-z]?\b(?:\s*(?:para|to|→))?", text) and re.search(
        r"\b(?:carris|metro|cp|comboio|bus|autocarro|tram|el[eé]ctrico|eletrico)\b", text, re.IGNORECASE
    ):
        return True
    if re.search(
        r"\b(?:embarque|board|transbordo|transfer|sair|exit|sai\s+em|continuar|continue)\s+(?:em|at|na|no|in)\s+\w",
        text,
        re.IGNORECASE,
    ):
        return True
    return False


def _planner_text_area_bucket(text: str) -> str:
    """Infer a coarse Lisbon area bucket from visible planner text."""
    normalized = _normalize_planner_text(text)
    if _PLANNER_BELEM_AREA_RE.search(normalized):
        return "belem"
    if re.search(r"\b(?:parque das nacoes|oriente|expo|oceanario|alameda dos oceanos)\b", normalized):
        return "parque_nacoes"
    if re.search(r"\b(?:saldanha|avenidas novas|picoas|marques de pombal|avenida da liberdade)\b", normalized):
        return "avenidas"
    if _PLANNER_CENTRAL_AREA_RE.search(normalized):
        return "central"
    return ""


def _planner_short_walk_item_is_unsafe(text: str) -> bool:
    """Return whether a walking bullet overstates a broad or cross-zone leg."""
    normalized = _normalize_planner_text(text)
    if not re.search(r"\b(?:caminhada curta|short walk|short walking)\b", normalized):
        return False
    if re.search(r"\b(?:zona|area|eixo)\s+(?:de\s+|da\s+|do\s+)?(?:lisboa|lisbon)\b", normalized):
        return True
    if not _planner_text_has_route_arrow(text):
        return False
    endpoints = _PLANNER_ROUTE_ARROW_RE.split(text, maxsplit=1)
    if len(endpoints) != 2:
        return False
    origin_zone = _planner_text_area_bucket(endpoints[0])
    destination_zone = _planner_text_area_bucket(re.split(r"\s*:\s*", endpoints[1], maxsplit=1)[0])
    return bool(origin_zone and destination_zone and origin_zone != destination_zone)


def _planner_movement_bullet_is_generic_operator_advice(text: str) -> bool:
    """Return whether a movement bullet names operators without a real leg."""
    normalized = _normalize_planner_text(text)
    if not normalized:
        return True
    if _movement_bullet_has_concrete_signal(text):
        return False

    has_operator = re.search(
        r"\b(?:metro|metro de lisboa|carris|carris urbana?|carris metropolitana|cp|comboio|autocarro|bus|tram|eletrico|electrico|transporte publico|public transport)\b",
        normalized,
    )
    if not has_operator:
        return False

    generic_markers = (
        "ponto de origem",
        "primeiro local",
        "zona",
        "consoante",
        "dependendo",
        "conforme",
        "se estiver",
        "se estiverem",
        "entre os dois locais",
        "na rede",
        "according to your area",
        "depending on",
        "if they are on the network",
        "between the two venues",
        "nao detalha",
        "nao confirma",
        "sem horarios",
        "sem horario",
        "not detailed",
        "not confirmed",
        "without live times",
    )
    return any(marker in normalized for marker in generic_markers)


def _planner_response_contains_generic_movement_advice(markdown: str) -> bool:
    """Return whether the rendered movement section still contains vague advice."""
    if not markdown:
        return False

    section_heading_re = re.compile(r"^\s*###\s+🚇\s+\*\*(?:Como te deslocas|How to move)\*\*\s*$")
    next_heading_re = re.compile(r"^\s*###\s+\S")
    section_break_re = re.compile(r"^\s*---\s*$")
    bullet_re = re.compile(r"^\s*[-*•]\s+")

    in_movement = False
    for line in markdown.splitlines():
        if section_heading_re.match(line):
            in_movement = True
            continue
        if not in_movement:
            continue
        if next_heading_re.match(line) or section_break_re.match(line):
            in_movement = False
            continue
        if bullet_re.match(line):
            stripped = bullet_re.sub("", line).strip().rstrip(";").rstrip()
            if _planner_movement_bullet_is_generic_operator_advice(stripped):
                return True
    return False


def _strip_planner_movement_placeholders(response: str) -> str:
    """Remove placeholder/marketing-style bullets from the movement section.

    The Planner LLM sometimes pads "Como te deslocas" / "How to move" with
    generic brochure bullets such as ``- 🗺️ rota otimizada entre os pontos
    que queres visitar;`` when route evidence is thin. These bullets carry
    no operational value and degrade the visual contract. This helper drops
    them while preserving any bullet that contains a concrete signal.
    """
    if not response:
        return response

    placeholder_re = re.compile(
        r"(?i)\b(?:" + "|".join(_PLANNER_MOVEMENT_PLACEHOLDER_PHRASES) + r")\b",
    )
    section_heading_re = re.compile(
        r"^\s*###\s+🚇\s+\*\*(?:Como te deslocas|How to move)\*\*\s*$"
    )
    next_heading_re = re.compile(r"^\s*###\s+\S")
    section_break_re = re.compile(r"^\s*---\s*$")
    bullet_re = re.compile(r"^\s*[-*•]\s+")

    lines = response.splitlines()
    out: list[str] = []
    in_movement = False

    for line in lines:
        if section_heading_re.match(line):
            in_movement = True
            out.append(line)
            continue
        if in_movement:
            if next_heading_re.match(line) or section_break_re.match(line):
                in_movement = False
                out.append(line)
                continue
            if bullet_re.match(line):
                stripped = bullet_re.sub("", line).strip().rstrip(";").rstrip()
                if (
                    _planner_movement_bullet_is_generic_operator_advice(stripped)
                    or (placeholder_re.search(stripped) and not _movement_bullet_has_concrete_signal(stripped))
                ):
                    # Drop placeholder bullet entirely.
                    continue
        out.append(line)

    return "\n".join(out)


def _planner_route_card_terms(response: str) -> set[str]:
    """Extract normalized stop names that are visible in the rendered route."""
    terms: set[str] = set()
    route_section = False
    for raw_line in str(response or "").splitlines():
        stripped = raw_line.strip()
        normalized_line = _normalize_planner_text(stripped)
        if re.search(r"\b(?:roteiro sugerido|suggested route)\b", normalized_line):
            route_section = True
            continue
        if route_section and stripped.startswith("### "):
            break
        if not route_section:
            continue
        if re.search(r"\b(?:como te deslocas|how to move)\b", normalized_line):
            break
        match = re.match(
            r"^[-*]\s+\*\*(?:[^\w\s*]{0,8}\s*)?(?:[^:*]{2,80}:\s*)?(?P<name>[^*\n]{3,160})\*\*",
            stripped,
        )
        if not match:
            continue
        name = re.sub(r"\b\d{1,2}:\d{2}\b", " ", match.group("name"))
        if ":" in name:
            name = name.rsplit(":", 1)[1]
        name = re.sub(r"\s+", " ", name).strip(" .:-")
        normalized = _normalize_planner_text(name)
        if len(normalized) >= 4:
            terms.add(normalized)
    return terms


def _planner_visible_route_card_count(response: str) -> int:
    """Count visible itinerary cards in the rendered route section."""
    route_section = False
    count = 0
    for raw_line in str(response or "").splitlines():
        stripped = raw_line.strip()
        normalized_line = _normalize_planner_text(stripped)
        if re.search(r"\b(?:roteiro sugerido|suggested route)\b", normalized_line):
            route_section = True
            continue
        if route_section and stripped.startswith("### "):
            break
        if not route_section or raw_line[:1].isspace():
            continue
        if re.match(r"^[-*]\s+\*\*(?:[^\w\s*]{0,8}\s*)?(?:[^:*]{2,80}:\s*)?[^*\n]{3,160}\*\*", stripped):
            count += 1
    return count


def _ensure_partial_planner_movement_limitation(response: str, user_message: str, language: str) -> str:
    """Add a limitation when only part of an itinerary's movement legs are evidenced."""
    if not response or not (_query_requests_movement_details(user_message) or _query_requests_public_transport(user_message)):
        return response
    route_count = _planner_visible_route_card_count(response)
    if route_count < 3:
        return response
    movement_text = _planner_movement_section_text(response)
    if not movement_text:
        return response
    normalized_movement = _normalize_planner_text(movement_text)
    if re.search(r"\b(?:restantes paragens|remaining stops|selected stops|paragens selecionadas|nao inventei|não inventei)\b", normalized_movement):
        return response
    evidenced_legs = sum(
        1
        for line in movement_text.splitlines()
        if _planner_text_has_route_arrow(line)
    )
    if evidenced_legs >= route_count - 1:
        return response
    limitation = (
        "- ⚠️ As ligações exatas entre as restantes paragens do roteiro não ficaram confirmadas nos dados recolhidos; não inventei linhas, paragens ou durações."
        if language == "pt"
        else "- ⚠️ Exact legs between the remaining itinerary stops were not confirmed in the gathered data; I did not invent lines, stops, or durations."
    )
    lines = str(response).splitlines()
    output: List[str] = []
    in_movement = False
    inserted = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"^###\s+.*\*\*(?:Como te deslocas|How to move)\*\*\s*$", stripped, flags=re.IGNORECASE):
            in_movement = True
            output.append(line)
            continue
        if in_movement and (stripped.startswith("### ") or stripped == "---" or _PLANNER_SOURCE_LINE_RE.match(stripped)):
            if not inserted:
                output.append(limitation)
                inserted = True
            in_movement = False
        output.append(line)
    if in_movement and not inserted:
        output.append(limitation)
    return "\n".join(output)


def _planner_selected_card_context(cards: List[Dict[str, str]]) -> str:
    """Build a minimal route-section context from selected planner cards."""
    lines = ["### 📍 **Roteiro sugerido**"]
    for card in cards:
        name = _planner_card_display_name(card) or str(card.get("name") or "").strip()
        if not name:
            continue
        lines.append(f"- **{name}**")
    return "\n".join(lines)


def _planner_requested_movement_terms(user_message: str, response: str) -> set[str]:
    """Build terms that make a movement bullet relevant to the current plan."""
    terms = _planner_route_card_terms(response)
    excluded_terms = {
        _normalize_planner_text(area)
        for area in _extract_excluded_plan_areas(user_message)
        if _normalize_planner_text(area)
    }
    for value in (
        _extract_requested_plan_origin(user_message),
        _extract_requested_plan_area(user_message),
    ):
        normalized = _normalize_planner_text(value)
        if normalized in {"lisbon", "lisboa"} or normalized in excluded_terms:
            continue
        if len(normalized) >= 3:
            terms.add(normalized)

    normalized_query = _normalize_planner_text(user_message)
    if ("parque das nacoes" in normalized_query or "oriente" in normalized_query) and not excluded_terms.intersection(
        {"parque das nacoes", "oriente"}
    ):
        terms.update({"parque das nacoes", "oriente", "fil", "rua do bojador", "expo", "expo 98", "vasco da gama"})
    if "belem" in normalized_query and "belem" not in excluded_terms:
        terms.update({"belem", "jeronimos", "imperio", "brasilia"})
    if "alfama" in normalized_query and "alfama" not in excluded_terms:
        terms.update({"alfama", "se de lisboa", "santa luzia", "portas do sol"})

    return {term for term in terms if len(term) >= 3 and term not in excluded_terms}


def _planner_movement_item_is_relevant(
    item: str,
    user_message: str,
    response: str,
) -> bool:
    """Return whether a movement bullet belongs to the current itinerary."""
    terms = _planner_requested_movement_terms(user_message, response)
    if not terms:
        return True
    normalized_item = _normalize_planner_text(item)
    if not normalized_item:
        return False

    matched_terms = [term for term in terms if term in normalized_item]
    if _planner_text_has_route_arrow(item):
        endpoints = _PLANNER_ROUTE_ARROW_RE.split(item, maxsplit=1)
        if len(endpoints) == 2:
            endpoint_matches = [
                any(term in _normalize_planner_text(endpoint) for term in terms)
                for endpoint in endpoints
            ]
            return sum(1 for matched in endpoint_matches if matched) >= 2
    if len(matched_terms) >= 2:
        return True
    origin = _normalize_planner_text(_extract_requested_plan_origin(user_message))
    target = _normalize_planner_text(_extract_requested_plan_area(user_message))
    if origin and origin in normalized_item:
        if target and target in normalized_item:
            return True
        return any(term in normalized_item for term in terms if term != origin)
    return False


def _planner_movement_pair_key(item: str) -> str:
    """Return a stable origin-target key for a movement bullet."""
    body = _fallback_bullet_body(item)
    bold_match = re.search(r"\*\*([^*\n]*?(?:→|->)[^*\n]*?)\*\*", body)
    segment = bold_match.group(1) if bold_match else body
    endpoints = _PLANNER_ROUTE_ARROW_RE.split(segment, maxsplit=1)
    if len(endpoints) != 2:
        return ""
    origin = _normalize_planner_text(_planner_compact_movement_name(endpoints[0]))
    destination = _normalize_planner_text(
        _planner_compact_movement_name(
            re.split(r"\s*:\s*", endpoints[1], maxsplit=1)[0]
        )
    )
    if not origin or not destination:
        return ""
    return f"{origin}->{destination}"


def _remove_empty_planner_movement_sections(response: str) -> str:
    """Remove movement headings left empty after route-leg filtering."""
    if not response:
        return response
    cleaned = re.sub(
        r"(?ms)\n---\s*\n+###\s+.*\*\*(?:Como te deslocas|How to move)\*\*\s*"
        r"(?=\n---\s*\n|\n###\s+|(?:\n)?📌\s+\*\*(?:Fonte|Source):|\Z)",
        "",
        response,
    )
    cleaned = re.sub(
        r"(?ms)^###\s+.*\*\*(?:Como te deslocas|How to move)\*\*\s*"
        r"(?=\n---\s*\n|\n###\s+|(?:\n)?📌\s+\*\*(?:Fonte|Source):|\Z)",
        "",
        cleaned,
    )
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _strip_irrelevant_planner_movement_items(response: str, user_message: str, language: str) -> str:
    """Remove movement bullets whose endpoints do not belong to the current plan."""
    if not str(response or "").strip():
        return response
    lines = str(response).splitlines()
    out: List[str] = []
    in_movement = False
    kept_in_section = False
    removed_in_section = False
    seen_movement_pairs: set[str] = set()

    def fallback_line() -> str:
        if language == "pt":
            return "- ⚠️ A ligação pedida não ficou confirmada de forma suficientemente específica nos dados recolhidos; não inventei linhas, paragens ou durações."
        return "- ⚠️ The requested movement leg was not confirmed specifically enough in the gathered data; I did not invent lines, stops, or durations."

    def close_section_if_needed() -> None:
        nonlocal kept_in_section, removed_in_section
        if in_movement and removed_in_section and not kept_in_section and _query_requests_movement_details(user_message):
            out.append(fallback_line())
            kept_in_section = True

    for line in lines:
        stripped = line.strip()
        movement_heading_match = re.match(
            r"^(?:###\s+.*\*\*|[-*]\s+\*\*\s*(?:🚇\s*)?)(?:Como te deslocas|How to move)\*\*",
            stripped,
            flags=re.IGNORECASE,
        )
        if movement_heading_match:
            in_movement = True
            kept_in_section = False
            removed_in_section = False
            seen_movement_pairs = set()
            heading = "### 🚇 **Como te deslocas**" if language == "pt" else "### 🚇 **How to move**"
            out.append(heading)
            continue
        if in_movement and (stripped.startswith("### ") or stripped == "---"):
            close_section_if_needed()
            in_movement = False
            out.append(line)
            continue
        if in_movement and _PLANNER_SOURCE_LINE_RE.match(stripped):
            close_section_if_needed()
            in_movement = False
            out.append(line)
            continue
        if in_movement and re.match(r"^[-*]\s+", stripped):
            body = _fallback_bullet_body(stripped)
            if _movement_item_is_self_referential_origin(body, user_message):
                removed_in_section = True
                continue
            if _planner_short_walk_item_is_unsafe(body):
                removed_in_section = True
                continue
            if (
                _planner_transport_bullet_is_actionable(body)
                and not _planner_movement_item_is_relevant(body, user_message, response)
            ):
                removed_in_section = True
                continue
            pair_key = _planner_movement_pair_key(stripped)
            if pair_key and pair_key in seen_movement_pairs:
                removed_in_section = True
                continue
            if pair_key:
                seen_movement_pairs.add(pair_key)
            kept_in_section = True
        out.append(line)

    close_section_if_needed()
    return _remove_empty_planner_movement_sections("\n".join(out))


def _enforce_walking_only_movement(response: str, user_message: str, language: str) -> str:
    """Keep walking-only plans from leaking bus, metro, or CP route bullets."""
    if not _query_requests_walking_only_plan(user_message):
        return response

    is_pt = language == "pt"
    fallback_items = _planner_walking_only_guidance(language)
    output: List[str] = []
    in_movement = False
    kept_in_section = False

    def append_fallback_if_needed() -> None:
        nonlocal kept_in_section
        if in_movement and not kept_in_section:
            output.extend(fallback_items)
            kept_in_section = True

    for raw_line in str(response or "").splitlines():
        stripped = raw_line.strip()
        heading_match = re.match(
            r"^###\s+.*\*\*(?:Como te deslocas|How to move)\*\*\s*$",
            stripped,
            flags=re.IGNORECASE,
        )
        if heading_match:
            append_fallback_if_needed()
            in_movement = True
            kept_in_section = False
            output.append("### 🚶 **Como te deslocas**" if is_pt else "### 🚶 **How to move**")
            continue

        if in_movement and (stripped.startswith("### ") or _PLANNER_SOURCE_LINE_RE.match(stripped)):
            append_fallback_if_needed()
            in_movement = False

        if in_movement and re.match(r"^[-*]\s+", stripped):
            normalized = _normalize_planner_text(stripped)
            has_transport_operator = bool(
                re.search(
                    r"\b(?:carris|metro|cp|comboio|comboios|train|autocarro|autocarros|bus|tram|eletrico|"
                    r"linha\s+\d{1,4}[a-z]?)\b",
                    normalized,
                )
            )
            has_walking_signal = bool(
                re.search(r"\b(?:pe|walk|walking|caminhada|caminhar|andar)\b", normalized)
            )
            if has_transport_operator and not has_walking_signal:
                continue
            kept_in_section = True

        output.append(raw_line)

    append_fallback_if_needed()
    return "\n".join(output)


def enforce_multi_day_quality_mode(response: str, user_message: str, language: str) -> str:
    """Preserve useful bounded multi-day plans and annotate thin one-day drafts.

    Args:
        response: Planner draft or repaired final response.
        user_message: Original user request.
        language: Output language code.

    Returns:
        str: Multi-day response, with a follow-up note only when needed.
    """
    response = _strip_planner_movement_placeholders(response)
    response = _strip_irrelevant_planner_movement_items(response, user_message, language)
    response = _enforce_walking_only_movement(response, user_message, language)
    requested_days = _extract_requested_day_count(user_message)
    if not requested_days or requested_days <= 1:
        return response

    normalized_response = str(response or "").lower()
    has_requested_day_sections = _response_has_requested_day_sections(response, user_message)
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
    ) or has_requested_day_sections:
        return response

    lines = str(response or "").splitlines()
    normalized_lines = list(lines)

    follow_up_note = _build_multi_day_follow_up_note(language, requested_days)
    if follow_up_note and follow_up_note not in "\n".join(normalized_lines):
        normalized_lines.extend(["", f"- {follow_up_note}"])

    return "\n".join(normalized_lines).strip()


def _response_has_requested_day_sections(response: str, user_message: str) -> bool:
    """Return whether every explicitly requested visible day is present."""
    requested_days = _extract_requested_day_count(user_message)
    if not requested_days or requested_days <= 1:
        return True
    visible_days = min(requested_days, 5)
    text = str(response or "")
    for day_index in range(1, visible_days + 1):
        if not re.search(
            rf"(?mi)^\s*(?:#{{1,6}}\s*)?(?:[-*]\s*)?(?:[📅📍🗓️]\s*)?(?:\*\*)?(?:day|dia)\s+{day_index}\b",
            text,
        ):
            return False
    return True


def _planner_response_missing_requested_day_sections(response: str, user_message: str) -> bool:
    """Return whether a multi-day request lost its visible day-by-day structure."""
    requested_days = _extract_requested_day_count(user_message)
    return bool(requested_days and requested_days > 1 and not _response_has_requested_day_sections(response, user_message))


def _ensure_multi_day_response_quality(
    response: str,
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
    """Replace thin multi-day drafts with a bounded day-by-day fallback."""
    bounded = enforce_multi_day_quality_mode(response, user_message, language)
    bounded = _enforce_missing_event_filter_limitation(
        bounded,
        user_message=user_message,
        language=language,
        places_data=places_data,
        events_data=events_data,
    )
    if not _planner_response_missing_requested_day_sections(bounded, user_message):
        return bounded

    fallback = _build_structured_plan_fallback(
        user_message=user_message,
        language=language,
        weather_data=weather_data,
        transport_data=transport_data,
        places_data=places_data,
        events_data=events_data,
        qa_disclaimers=qa_disclaimers,
        conversation_context=conversation_context,
    )
    fallback = _strip_unrequested_live_departure_lines(fallback, user_message)
    fallback = _ensure_requested_origin_target_in_transport_section(
        fallback,
        user_message,
        language,
        transport_data,
    )
    if _planner_response_has_minimum_user_value(fallback) and _response_has_requested_day_sections(fallback, user_message):
        fallback = enforce_multi_day_quality_mode(fallback, user_message, language)
        return _enforce_missing_event_filter_limitation(
            fallback,
            user_message=user_message,
            language=language,
            places_data=places_data,
            events_data=events_data,
        )
    return bounded


def _enforce_missing_event_filter_limitation(
    response: str,
    *,
    user_message: str,
    language: str,
    places_data: str,
    events_data: str,
) -> str:
    """Surface no-result event filters when a mixed plan falls back to food only."""
    normalized_query = _normalize_planner_text(user_message)
    if not _is_event_planning_request(normalized_query):
        return response
    if not re.search(r"\b(?:gratuit|gratis|free)\b", normalized_query):
        return response
    if not re.search(r"\b(?:hoje|today|esta noite|tonight)\b", normalized_query):
        return response

    normalized_response = _normalize_planner_text(response)
    if re.search(r"\b(?:vegetariano|vegetariana|vegetarian|vegan|vegano|vegana)\b", normalized_query):
        response = re.sub(r"\bJantar tradicional:", "Jantar vegetariano:", response)
        response = re.sub(r"\bTraditional dinner:", "Vegetarian dinner:", response)
        normalized_response = _normalize_planner_text(response)
    context = _normalize_planner_text("\n".join([places_data or "", events_data or ""]))
    context_reports_no_event = bool(
        re.search(r"\b(?:nao encontrei eventos|sem eventos|no events|no confirmed events|did not find events)\b", context)
    )
    event_stop_probe = re.sub(
        r"✅\s+\*\*(?:Resposta direta|Direct answer):\*\*[^\n]+",
        "",
        response,
        count=1,
    )
    normalized_event_stop_probe = _normalize_planner_text(event_stop_probe)
    response_has_event_stop = bool(
        re.search(
            r"\b(?:evento cultural|evento gratuito|concerto|festival|teatro|exposicao|exposição|cultural event|free event|concert|theatre|theater|exhibition)\b",
            normalized_event_stop_probe,
        )
    )
    if response_has_event_stop and not context_reports_no_event:
        return response

    if re.search(r"\b(?:nao encontrei evento|sem evento gratuito|no confirmed free event|no free event)\b", normalized_response):
        return response

    is_pt = language == "pt"
    note = (
        "- Não encontrei evento gratuito com data confirmada para hoje nos dados consultados; não inventei uma alternativa como evento confirmado."
        if is_pt
        else "- I did not find a free event with a confirmed date for today in the gathered data; I did not invent an alternative as confirmed."
    )
    response = re.sub(
        r"✅\s+\*\*(Resposta direta|Direct answer):\*\*[^\n]+",
        (
            "✅ **Resposta direta:** não encontrei um evento gratuito confirmado para hoje; mantive apenas os restantes pontos suportados pelos dados."
            if is_pt
            else "✅ **Direct answer:** I did not find a confirmed free event for today; I kept only the remaining data-supported parts."
        ),
        response,
        count=1,
    )
    final_notes_re = r"(?m)^(###\s+⚠️\s+\*\*(?:Notas finais|Final notes)\*\*\s*)$"
    if re.search(final_notes_re, response):
        return re.sub(final_notes_re, rf"\1\n{note}", response, count=1)

    source_match = re.search(r"(?m)^📌\s+\*\*(?:Fonte|Source):\*\*", response)
    heading = "### ⚠️ **Notas finais**" if is_pt else "### ⚠️ **Final notes**"
    block = f"\n\n---\n\n{heading}\n{note}\n\n"
    if source_match:
        return response[:source_match.start()].rstrip() + block + response[source_match.start():].lstrip()
    return response.rstrip() + block.rstrip()


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
    if re.search(r"\b(low walking|low walk|little walking|pouca caminhada|baixo declive|pouca subida|sem grandes caminhadas|reduced mobility|ritmo calmo|senior|sénior)\b", combined):
        add("ritmo calmo e poucas subidas", "slow pace and low-gradient walking")
    if re.search(r"\b(public transport|transportes publicos|metro|carris|cp|bus|tram|autocarro|comboio)\b", combined):
        add("usar transporte público", "use public transport")
    if re.search(r"\b(cheap|cheaper|budget|barato|barata|baixo custo|econ[oó]mico)\b", combined) or _query_requests_budget_food(user_message):
        add("manter custos baixos", "keep costs low")
    for place, time_label in _requested_anchor_time_constraints(user_message):
        if place and time_label:
            add(f"passar por {place} às {time_label}", f"be at {place} at {time_label}")
    deadline_match = re.search(
        r"\b(?:termin\S*|acab\S*|finish(?:ing)?|end(?:ing)?)\s+"
        r"(?:até|ate|by|as|às|at)(?:\s+(?:as|às))?\s+"
        r"(?P<time>\d{1,2}(?:[:hH]\d{0,2})?)",
        user_message,
        flags=re.IGNORECASE,
    )
    if deadline_match:
        deadline = _normalize_requested_time_label(deadline_match.group("time").rstrip("hH"))
        if deadline:
            add(f"terminar até {deadline}", f"finish by {deadline}")
    for area in _extract_excluded_plan_areas(user_message):
        label = area.title()
        add(f"evitar {label}", f"avoid {label}")
    category_labels = {
        "museum": ("evitar museus", "avoid museums"),
        "monument": ("evitar monumentos", "avoid monuments"),
        "food": ("evitar restaurantes", "avoid restaurants"),
        "viewpoint": ("evitar miradouros", "avoid viewpoints"),
        "event": ("evitar eventos", "avoid events"),
    }
    for category in sorted(_extract_excluded_plan_categories(user_message)):
        pt_label, en_label = category_labels.get(category, ("evitar categoria pedida", "avoid requested category"))
        add(pt_label, en_label)
    avoid_match = re.search(
        r"\b(?:do not repeat|avoid|não repetir|nao repetir|evitar)\s+(?P<areas>[A-Za-zÀ-ÿ\s,/&-]+)",
        user_message,
        flags=re.IGNORECASE,
    )
    if avoid_match:
        areas = re.sub(r"\s+", " ", avoid_match.group("areas")).strip(" .?!,;:")
        areas = re.split(
            r"\b(?:or[cç]amento|budget|come[cç]ar|terminar|ritmo|pouca|muita\s+chuva|chuva|weather)\b",
            areas,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" .?!,;:")
        normalized_areas = _normalize_planner_text(areas)
        if areas and not re.search(r"\b(?:chuva|weather|or[cç]amento|budget|ritmo|come[cç]ar|terminar)\b", normalized_areas):
            add(f"evitar {areas}", f"avoid {areas}")
    return constraints[:6]


_PLANNER_NON_AREA_EXCLUSION_RE = re.compile(
    r"\b(?:caminh|walk|metro|autocar|bus|comboio|train|eletrico|tram|"
    r"chuva|rain|preco|price|orcamento|budget|caro|barat|tempo|time|"
    r"horario|schedule|subida|escadas|stairs|muito|grandes|longas?|"
    r"museus?|museums?|restaurantes?|restaurants?|pagos?|paid|gratuitos?|free)\b",
    re.IGNORECASE,
)


def _extract_excluded_plan_areas(user_message: str) -> List[str]:
    """Extract explicit negative area constraints from a planning request."""
    normalized = _normalize_planner_text(user_message)
    if not normalized:
        return []

    exclusions: List[str] = []
    patterns = (
        r"\b(?:do not repeat|dont repeat|avoid|exclude|excluding|sem repetir|nao repetir|"
        r"evita(?:r)?|exclui(?:r)?)\s+(?P<areas>[a-z0-9][a-z0-9 /&-]{1,100})",
        r"\b(?:sem|without)\s+(?P<areas>[a-z0-9][a-z0-9 /&-]{1,80})",
    )
    stop_re = re.compile(
        r"\b(?:com|with|mas|but|evitando|avoiding|para|porque|because|"
        r"quero|queria|gostava|i want|i would|terminar|acabar|finish|end|passar|passe|passem|via)\b",
        re.IGNORECASE,
    )
    split_re = re.compile(r"\s*(?:,|/|\+|\band\b|\bor\b|\be\b|\bou\b|&)\s*", re.IGNORECASE)

    for pattern in patterns:
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
            raw = stop_re.split(match.group("areas"), maxsplit=1)[0]
            raw = re.sub(r"\b(?:zonas?|areas?|bairros?|neighbourhoods?|neighborhoods?)\b", " ", raw)
            for piece in split_re.split(raw):
                cleaned = re.sub(r"\s+", " ", piece).strip(" -")
                cleaned = re.sub(
                    r"^(?:pass(?:ar|e|em)\s+por|ir\s+(?:a|ao|aos|à|as|às)|visitar|visit(?:ar)?|"
                    r"go\s+to|pass\s+through|stop\s+at)\s+",
                    "",
                    cleaned,
                    flags=re.IGNORECASE,
                ).strip(" -")
                if not cleaned or len(cleaned.split()) > 5:
                    continue
                if _PLANNER_NON_AREA_EXCLUSION_RE.search(cleaned):
                    continue
                if cleaned not in exclusions:
                    exclusions.append(cleaned)
    return exclusions[:8]


def _extract_excluded_plan_categories(user_message: str) -> set[str]:
    """Extract explicit category exclusions such as "sem museus"."""
    normalized = _normalize_planner_text(user_message)
    excluded: set[str] = set()
    exclusion_prefix = r"(?:sem|without|avoid|evitar|excluir|exclude|nao\s+quero|não\s+quero)"
    if re.search(rf"\b{exclusion_prefix}\s+(?:museus?|museums?|galerias?|galleries)\b", normalized):
        excluded.add("museum")
    if re.search(rf"\b{exclusion_prefix}\s+(?:monumentos?|monuments?)\b", normalized):
        excluded.add("monument")
    if re.search(rf"\b{exclusion_prefix}\s+(?:restaurantes?|restaurants?|cafes?|pastelarias?)\b", normalized):
        excluded.add("food")
    if re.search(rf"\b{exclusion_prefix}\s+(?:miradouros?|viewpoints?)\b", normalized):
        excluded.add("viewpoint")
    if re.search(rf"\b{exclusion_prefix}\s+(?:eventos?|events?)\b", normalized):
        excluded.add("event")
    return excluded


def _planner_card_matches_excluded_category(card: Dict[str, str], excluded_category: str) -> bool:
    """Return whether a card conflicts with an explicit category exclusion."""
    if excluded_category == "food":
        return _card_kind_for_plan_block(card) == "food"
    if excluded_category == "event":
        return _card_kind_for_plan_block(card) == "event"
    if excluded_category == "viewpoint":
        return _planner_card_is_viewpoint(card)
    if excluded_category in {"museum", "monument"}:
        return _planner_card_matches_requested_count_type(card, excluded_category)
    return False


def _planner_card_matches_excluded_area(card: Dict[str, str], excluded_area: str) -> bool:
    """Return whether a planner card conflicts with an explicit area exclusion."""
    normalized_area = _normalize_planner_text(excluded_area)
    if not normalized_area:
        return False
    if _planner_card_matches_area(card, normalized_area):
        return True
    basis = _normalize_planner_text(
        " ".join(
            str(card.get(key, ""))
            for key in (
                "name",
                "category",
                "address",
                "venue",
                "description",
                "features",
                "url",
                "details_url",
            )
        )
    )
    return bool(len(normalized_area) >= 3 and re.search(rf"\b{re.escape(normalized_area)}\b", basis))


def _filter_planner_cards_for_request_constraints(
    cards: List[Dict[str, str]],
    user_message: str,
) -> List[Dict[str, str]]:
    """Apply user area/exclusion constraints before itinerary card selection."""
    if not cards:
        return []

    filtered = list(cards)
    excluded_areas = _extract_excluded_plan_areas(user_message)
    if excluded_areas:
        filtered = [
            card
            for card in filtered
            if not any(_planner_card_matches_excluded_area(card, area) for area in excluded_areas)
        ]
    excluded_categories = _extract_excluded_plan_categories(user_message)
    if excluded_categories:
        filtered = [
            card
            for card in filtered
            if not any(_planner_card_matches_excluded_category(card, category) for category in excluded_categories)
        ]

    target_area = _extract_compact_plan_area_anchor(user_message)
    if target_area and _query_describes_single_area_plan(user_message):
        area_cards = [
            card for card in filtered
            if _planner_card_matches_area(card, target_area)
        ]
        if area_cards:
            open_area_cultural_cards = [
                card for card in area_cards
                if _card_kind_for_plan_block(card) not in {"food", "event"}
                and not _planner_dict_card_is_closed(card)
            ]
            area_has_cultural = any(
                _card_kind_for_plan_block(card) not in {"food", "event"}
                and not _planner_dict_card_is_closed(card)
                for card in area_cards
            )
            original_has_cultural = any(
                _card_kind_for_plan_block(card) not in {"food", "event"}
                and not _planner_dict_card_is_closed(card)
                for card in filtered
            )
            if not _planner_cards_satisfy_requested_counts(area_cards, user_message):
                supplemental_count_cards = [
                    card for card in filtered
                    if card not in area_cards
                    and any(
                        _planner_card_matches_requested_count_type(card, count_type)
                        for count_type, requested_count in _requested_plan_type_counts(user_message).items()
                        if count_type in {"museum", "monument", "viewpoint", "event"}
                        and requested_count > 0
                    )
                    and not _planner_dict_card_is_closed(card)
                ]
                supplemental_count_cards = sorted(
                    supplemental_count_cards,
                    key=lambda card: _score_card_for_requested_count_type(card, "total", user_message),
                    reverse=True,
                )
                filtered = _dedupe_planner_cards([
                    *area_cards,
                    *supplemental_count_cards,
                    *filtered,
                ])
            elif _query_requests_food_stop(user_message) and not area_has_cultural and original_has_cultural:
                nearby_cultural_cards = [
                    card for card in filtered
                    if _card_kind_for_plan_block(card) not in {"food", "event"}
                    and not _planner_dict_card_is_closed(card)
                ]
                filtered = _dedupe_planner_cards([*nearby_cultural_cards, *area_cards, *filtered])
            elif (
                _query_requests_food_stop(user_message)
                and len(open_area_cultural_cards) < 2
                and original_has_cultural
            ):
                normalized_query = _normalize_planner_text(user_message)
                supplemental_cultural_cards = sorted(
                    [
                        card for card in filtered
                        if card not in area_cards
                        and _card_kind_for_plan_block(card) not in {"food", "event"}
                        and not _planner_dict_card_is_closed(card)
                        and _compact_central_plan_far_area_penalty(card, user_message) < 100
                    ],
                    key=lambda card: _score_local_area_plan_card(card, normalized_query, user_message),
                    reverse=True,
                )
                supplemental_cultural_cards = [
                    card for card in supplemental_cultural_cards
                    if _score_local_area_plan_card(card, normalized_query, user_message) >= 0
                ]
                filtered = _dedupe_planner_cards([
                    *area_cards,
                    *supplemental_cultural_cards[: max(0, 2 - len(open_area_cultural_cards))],
                    *filtered,
                ])
            else:
                filtered = area_cards

    area_key, _area_label, blockers = _planner_local_area_profile(user_message)
    if area_key and blockers:
        blocker_cards = [
            card for card in filtered
            if any(
                re.search(
                    rf"\b{re.escape(blocker)}\b",
                    _normalize_planner_text(
                        " ".join(
                            str(card.get(key, ""))
                            for key in ("name", "address", "description", "category", "url", "details_url")
                        )
                    ),
                )
                for blocker in blockers
            )
        ]
        if blocker_cards and len(blocker_cards) < len(filtered):
            filtered = [card for card in filtered if card not in blocker_cards]

    return filtered


def _planner_response_uses_excluded_area(response: str, user_message: str) -> bool:
    """Return whether a rendered plan still includes explicitly excluded areas."""
    excluded_areas = _extract_excluded_plan_areas(user_message)
    if not response or not excluded_areas:
        return False

    route_chunks: List[str] = []
    in_relevant_section = False
    for raw_line in str(response or "").splitlines():
        stripped = raw_line.strip()
        normalized_line = _normalize_planner_text(stripped)
        if re.search(r"\b(?:roteiro sugerido|suggested route|como te deslocas|how to move)\b", normalized_line):
            in_relevant_section = True
            continue
        if in_relevant_section and stripped.startswith("### "):
            in_relevant_section = False
        if in_relevant_section and stripped:
            route_chunks.append(stripped)

    relevant_text = _normalize_planner_text("\n".join(route_chunks))
    if not relevant_text:
        return False
    return any(
        re.search(rf"\b{re.escape(_normalize_planner_text(area))}\b", relevant_text)
        for area in excluded_areas
        if _normalize_planner_text(area)
    )


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

    plan_title_without_body = bool(
        re.search(r"\b(?:roteiro sugerido|suggested route|suggested itinerary|itinerario sugerido)\b", normalized)
        and not re.search(
            r"(?m)^\s*[-*•]\s+\*\*(?:[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+\s+)?[^*\n]{3,120}\*\*",
            without_sources,
        )
    )
    if plan_title_without_body:
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
    has_route_heading = any(
        re.search(r"\b(?:roteiro sugerido|suggested route|recommended route|route plan)\b", heading)
        for heading in useful_headings
    )
    non_actionable_heading_re = re.compile(
        r"\b(?:dicas|tips|notas finais|final notes|fonte|source|sources|fontes|"
        r"como te deslocas|how to move|adapta[cç]ao ao tempo|weather adaptation|"
        r"base do plano|plan basis|resposta direta|direct answer)\b"
    )
    actionable_section_headings = [
        heading
        for heading in useful_headings[1:]
        if not non_actionable_heading_re.search(heading)
    ]
    route_card_count = 0
    route_content_bullets = 0
    in_route_section = False
    for raw_line in without_sources.splitlines():
        stripped = raw_line.strip()
        normalized_line = _normalize_planner_text(stripped)
        if re.search(r"\b(?:roteiro sugerido|suggested route|recommended route|route plan)\b", normalized_line):
            in_route_section = True
            continue
        if in_route_section and stripped.startswith("### "):
            in_route_section = False
        if not in_route_section or raw_line[:1].isspace():
            continue
        if re.match(r"^\s*[-*•]\s+\*\*(?![^*\n]{0,80}:\*\*)[^*\n]{3,140}\*\*\s*$", stripped):
            route_card_count += 1
            continue
        if re.match(r"^\s*[-*•]\s+", stripped) and not re.search(
            r"\b(?:morada|address|descri[cç][aã]o|description|pre[cç]o|price|"
            r"hor[aá]rio|hours|website|bilhetes|tickets|categoria|category|"
            r"telefone|phone|email|mais detalhes|more details)\b",
            normalized_line,
        ):
            route_content_bullets += 1

    plan_like_shell = bool(
        re.search(
            r"\b(?:roteiro|itinerario|itinerary|planner?|plano|planear|planning|"
            r"tarde|manha|manh[aã]|dia|day)\b",
            normalized,
        )
    )
    has_actionable_plan_body = bool(
        route_card_count > 0
        or route_content_bullets >= 2
        or actionable_section_headings
    )
    if plan_like_shell and not has_actionable_plan_body:
        return False

    has_grounded_place_field = bool(
        re.search(
            r"\b(?:morada|address|mais detalhes|more details|website|site oficial|official website|"
            r"categoria|category|bilhetes|tickets|preco|preco|price|horario|hours|"
            r"rua|avenida|av\.|largo|praca|pra[cç]a)\b",
            normalized,
        )
    )
    if has_route_heading and not has_grounded_place_field:
        return False
    if has_route_heading and re.search(
        r"\b(?:viewpoint time|museum stop|food stop|meal stop|optional stop|late afternoon|"
        r"relaxed return|return to (?:the )?(?:center|centre|centro)|choose one (?:viewpoint|museum|place|stop)|"
        r"local a confirmar|paragem a confirmar|paragem opcional)\b",
        normalized,
    ):
        return False
    if re.search(
        r"\b(?:rossio|baixa|chiado|carmo|marques|marques de pombal)\s*(?:->|→)\s*(?:lisbon|lisboa)\b|"
        r"\b(?:lisbon|lisboa)\s*(?:->|→)\s*(?:viewpoint time|museum stop|food stop|late afternoon)\b",
        normalized,
    ):
        return False

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
    if has_actionable_plan_body and len(meaningful_bullets) >= 2 and len(useful_headings) >= 2:
        return True
    has_plan_section = any(
        re.search(
            r"\b(?:roteiro sugerido|suggested route|como te deslocas|how to move)\b",
            heading,
        )
        for heading in useful_headings
    )
    return has_plan_section and has_actionable_plan_body and len(useful_headings) >= 3 and len(body_lines) >= 3


def _planner_response_missing_requested_food_stop(response: str, user_message: str) -> bool:
    """Return whether a food itinerary request lacks a concrete meal stop.

    A planner draft can be structurally valid while still dropping or misusing
    a requested gastronomy, cafe, lunch, dinner, or restaurant component.
    """
    normalized_query = _normalize_planner_text(user_message)
    if not re.search(r"\b(?:plan|planeia|planear|plano|itinerary|itinerario|roteiro|programa|day|dia|manha|morning)\b", normalized_query):
        return False
    if not re.search(
        r"\b(?:gastronom\w*|restaurants?|restaurantes?|food|comida|comer|refei[cç][aã]o|meal|"
        r"tradicional|almoco|almoço|almocar|almoçar|"
        r"lunch|jantar|dinner|cozinha|cafe|coffee|pastelaria|pastry|brunch|"
        r"pastel|pasteis|nata|custard|tarts?)\b",
        normalized_query,
    ):
        return False

    without_sources = re.sub(
        r"(?im)^\s*(?:[-*•]\s*)?(?:📌\s*)?\**(?:Fonte|Fontes|Source|Sources)\**\s*:.*$",
        "",
        response or "",
    )
    normalized_response = _normalize_planner_text(without_sources)
    if re.search(
        r"\b(?:nenhum\s+restaurante\s+especifico\s+ficou\s+confirmado|"
        r"pausa\s+gastronomica\s+pedida\s+pelo\s+utilizador|"
        r"no\s+specific\s+restaurant\s+was\s+confirmed|"
        r"food\s+break\s+requested\s+by\s+the\s+user)\b",
        normalized_response,
    ):
        return True
    explicitly_requested_meals: list[str] = []
    if re.search(r"\b(?:almoco|almoço|almocar|almoçar|lunch)\b", normalized_query):
        explicitly_requested_meals.append("lunch")
    if re.search(r"\b(?:jantar|dinner)\b", normalized_query):
        explicitly_requested_meals.append("dinner")
    if len(explicitly_requested_meals) > 1:
        meal_patterns = {
            "lunch": r"\b(?:almoco|almoço|almocar|almoçar|lunch)\b",
            "dinner": r"\b(?:jantar|dinner)\b",
        }
        return any(
            not re.search(meal_patterns[meal_kind], normalized_response)
            for meal_kind in explicitly_requested_meals
        )

    if _query_requests_custard_tart_stop(user_message):
        if re.search(r"\b(?:hoje\s+fechado|fechado\s+hoje|today\s+closed|closed\s+today)\b", normalized_response):
            return True
        return not bool(
            re.search(
                r"\b(?:pastel(?:\s+de\s+nata)?|pasteis(?:\s+de\s+nata)?|nata|"
                r"custard(?:\s+tarts?)?|tarts?|pastelaria|pastry|pausa\s+gastronomica|food\s+break)\b",
                normalized_response,
            )
        )
    if _query_requests_cafe_stop(user_message):
        if re.search(r"\b(?:hoje\s+fechado|fechado\s+hoje|today\s+closed|closed\s+today)\b", normalized_response):
            return True
        return not bool(
            re.search(
                r"\b(?:cafe|coffee|pastelaria|pastry|brunch|pausa\s+gastronomica|food\s+break)\b",
                normalized_response,
            )
        )
    return not bool(
        re.search(
            r"\b(?:almoco|almoço|jantar|lunch|dinner|restaurante|restaurant|cozinha\s+tradicional|traditional\s+(?:lunch|dinner)|optional\s+dinner)\b",
            normalized_response,
        )
        or "🍽️" in without_sources
    )


def _planner_response_loses_transport_leg_evidence(response: str, transport_data: str) -> bool:
    """Return whether planner synthesis dropped concrete route-leg evidence."""
    if not response or not transport_data:
        return False

    def _route_leg_count(value: str) -> int:
        count = 0
        for raw_line in str(value or "").splitlines():
            line = raw_line.strip()
            if not re.match(r"^[-*•]\s+", line):
                continue
            normalized = _normalize_planner_text(line)
            if _planner_text_has_route_arrow(line) and re.search(
                r"\b(?:caminhada|walk|carris|metro|cp|comboio|autocarro|bus|tram|linha|line)\b",
                normalized,
            ):
                count += 1
        return count

    evidence_count = _route_leg_count(transport_data)
    if evidence_count < 2:
        return False
    response_count = _route_leg_count(response)
    return response_count < min(evidence_count, 3)


def _is_oriente_station_nearby_request(user_message: str) -> bool:
    """Return whether Oriente should be treated as station/Parque das Nações locality."""
    normalized = _normalize_planner_text(user_message or "")
    if "oriente" not in normalized or "museu do oriente" in normalized:
        return False
    if re.search(r"\b(?:terminar|acabar|finish|end)\b", normalized):
        return False
    locality_signals = (
        "arrive", "chego", "chegar", "nearby", "perto", "near", "dinner",
        "jantar", "rain-safe", "cultural", "culture", "indoor", "interior",
        "parque das nacoes", "station", "estacao", "chuva", "rain", "museu",
        "museum", "mini plano", "mini plan", "2 horas", "two hours",
        "pouco tempo", "short time", "pouco tempo a pe", "short walk",
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
                "source_id": "local_context",
            },
            {
                "name": "Centro Vasco da Gama",
                "category": "base coberta para jantar",
                "description": "Opção prática e abrigada para procurar restauração sem assumir reserva ou horário específico.",
                "source_id": "local_context",
            },
            {
                "name": "Oceanário / Pavilhão do Conhecimento",
                "category": "referência cultural próxima",
                "description": "Trata como alternativa cultural a confirmar, sem afirmar disponibilidade ou horário de visita.",
                "source_id": "local_context",
            },
        ]
    return [
        {
            "name": "Oriente station / Parque das Nações",
            "category": "local anchor",
            "description": "Use Oriente as the compact arrival base instead of crossing the city.",
            "source_id": "local_context",
        },
        {
            "name": "Centro Vasco da Gama",
            "category": "covered dinner base",
            "description": "A practical sheltered base for finding food without assuming booking or opening-hour confirmation.",
            "source_id": "local_context",
        },
        {
            "name": "Oceanário / Pavilhão do Conhecimento",
            "category": "nearby cultural reference",
            "description": "Treat as a cultural backup to verify, without claiming availability or exact opening hours.",
            "source_id": "local_context",
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
    prior_place_context = conversation_context if _query_references_previous_place_set(user_message) else ""
    combined_places = "\n".join([prior_place_context, places_data or "", events_data or ""])
    requested_label_count = len(_requested_anchor_labels(user_message, combined_places))
    card_limit = 18 if requested_days > 1 else 12
    if _query_requests_food_stop(user_message):
        card_limit = max(card_limit, 24)
    if requested_label_count >= 3:
        card_limit = max(card_limit, 48)
    cards = _extract_visitlisboa_place_cards(combined_places, max_items=card_limit, language=language)
    clean_cards = [
        {**card, "name": clean_name}
        for card in cards
        for clean_name in [_sanitize_planner_place_name(card.get("name", ""))]
        if clean_name and not _planner_card_is_synthetic_plan_heading({**card, "name": clean_name})
    ]
    if _is_oriente_station_nearby_request(user_message):
        clean_cards = [
            card for card in clean_cards
            if "museu do oriente" not in _normalize_planner_text(card.get("name", ""))
        ]
        clean_cards = _oriente_station_locality_cards(language) + clean_cards[:8]
    clean_cards = [
        card for card in clean_cards
        if not _planner_card_is_low_fit_infrastructure(card, user_message)
        and not _planner_card_is_synthetic_plan_heading(card)
    ]
    if not _is_event_planning_request(_normalize_planner_text(user_message)):
        clean_cards = [
            card for card in clean_cards
            if not _planner_card_is_event_result(card)
        ]
    target_area = _extract_compact_plan_area_anchor(user_message)
    strict_same_area_context = bool(
        target_area
        and re.search(
            r"\b(?:zona\s+anterior|previous\s+area|restri[cç][aã]o\s+de\s+zona|area\s+constraint)\b",
            _normalize_planner_text(user_message),
        )
    )
    if strict_same_area_context:
        area_cards = [
            card for card in clean_cards
            if _planner_card_matches_area(card, target_area)
        ]
        clean_cards = area_cards
    excluded_areas = _extract_excluded_plan_areas(user_message)
    if excluded_areas:
        clean_cards = [
            card for card in clean_cards
            if not any(_planner_card_matches_excluded_area(card, area) for area in excluded_areas)
        ]
    selected_cards = _select_planner_cards_for_request(clean_cards, user_message)
    requested_labels = _requested_anchor_labels(user_message, combined_places)
    strict_requested_sequence = (
        _query_has_explicit_anchor_sequence(user_message)
        or _query_has_explicit_start_end_constraint(user_message)
    )
    if strict_requested_sequence:
        ordered_labels = _requested_anchor_labels(user_message, combined_places)
        seen_ordered = {_normalize_planner_text(label) for label in ordered_labels}
        requested_labels = [
            *ordered_labels,
            *[
                label for label in requested_labels
                if _normalize_planner_text(label) not in seen_ordered
            ],
        ]
    requested_cards = _requested_anchor_cards_in_order(requested_labels, clean_cards, language)
    if not selected_cards and clean_cards:
        candidate_pool = clean_cards
        if target_area:
            area_pool = [
                card for card in clean_cards
                if _planner_card_matches_area(card, target_area)
            ]
            if area_pool:
                candidate_pool = area_pool
        selected_cards = [
            card for card in candidate_pool
            if _card_kind_for_plan_block(card) != "food"
        ][:5] or candidate_pool[:5]
        selected_cards = _insert_requested_food_stop_if_needed(
            selected_cards,
            clean_cards,
            user_message,
            language,
        )[:6]
        selected_cards = _insert_requested_cultural_stop_if_needed(
            selected_cards,
            clean_cards,
            user_message,
            language,
        )[:6]
    if selected_cards:
        if strict_requested_sequence and requested_cards:
            selected_cards = _insert_requested_food_stop_if_needed(
                requested_cards[:6],
                clean_cards,
                user_message,
                language,
            )[:6]
        elif re.search(
            r"\b(?:gastronom\w*|restaurants?|restaurantes?|food|comida|tradicional|almo[cç]o|jantar)\b",
            _normalize_planner_text(user_message),
        ) and not any(_card_kind_for_plan_block(card) == "food" for card in selected_cards):
            food_candidates = sorted(
                [
                    card for card in clean_cards
                    if _card_kind_for_plan_block(card) == "food"
                    and _food_card_matches_requested_context(card, user_message)
                ],
                key=_score_food_plan_card,
                reverse=True,
            )
            if food_candidates:
                selected_cards = [*selected_cards[:4], food_candidates[0]]
            else:
                meal_card = _requested_meal_placeholder_card(user_message, language)
                if meal_card:
                    selected_cards = [*selected_cards[:1], meal_card, *selected_cards[1:]]
        if requested_cards and _query_references_previous_place_set(user_message):
            requested_limit = 8 if len(_requested_meal_kinds(user_message)) > 1 else 6
            selected_cards = requested_cards[:requested_limit]
        elif requested_cards and not strict_requested_sequence:
            selected_cards = _dedupe_planner_cards([*requested_cards, *selected_cards])[:6]
        selected_cards = _insert_requested_cultural_stop_if_needed(
            selected_cards,
            clean_cards,
            user_message,
            language,
        )[:6]
        selected_cards = _move_requested_origin_card_first(selected_cards, user_message)
        selected_cards = _drop_origin_name_collision_cards(selected_cards, user_message)
        selected_keys = {
            _normalize_planner_text(_planner_card_display_name(card) or card.get("name", ""))
            for card in selected_cards
        }
        if strict_requested_sequence and selected_cards:
            clean_cards = selected_cards
        else:
            clean_cards = selected_cards + [
                card for card in clean_cards
                if _normalize_planner_text(_planner_card_display_name(card) or card.get("name", "")) not in selected_keys
            ]
    elif requested_cards:
        selected_cards = _insert_requested_food_stop_if_needed(
            requested_cards[:6],
            clean_cards,
            user_message,
            language,
        )[:6]
        selected_cards = _insert_requested_cultural_stop_if_needed(
            selected_cards,
            clean_cards,
            user_message,
            language,
        )[:6]
        clean_cards = _drop_origin_name_collision_cards(
            _move_requested_origin_card_first(selected_cards, user_message),
            user_message,
        )
    clean_cards = _dedupe_planner_cards(clean_cards)
    clean_cards = _limit_cards_for_user_cardinality(clean_cards, user_message)
    clean_cards = _arrange_cards_for_multi_day_plan(clean_cards, user_message, visible_days)
    weather_bullets = _extract_weather_safety_bullets(weather_data, language)
    walking_only_plan = _query_requests_walking_only_plan(user_message)
    transport_bullets = [
        item
        for item in (
            _fallback_bullet_body(bullet)
            for bullet in _extract_planner_fallback_bullets(transport_data, max_items=6)
        )
        if item
        and not _is_generic_transport_heading(item)
        and not _is_planner_transport_status_summary(item)
        and _planner_transport_bullet_is_actionable(item)
    ][:4]
    if strict_requested_sequence:
        transport_bullets = [
            item for item in transport_bullets
            if _movement_item_matches_requested_sequence(item, user_message)
        ]
    if walking_only_plan:
        transport_bullets = _planner_walking_only_guidance(language)
    if not transport_bullets and _query_requests_movement_details(user_message):
        transport_bullets = _requested_sequence_transport_limitation_bullets(user_message, language)

    transport_limitation = (
        "A ligação entre zonas não ficou confirmada nos dados recolhidos; não inventei linhas, paragens, durações ou partidas."
        if is_pt
        else "The connection between areas was not confirmed in the gathered data; I did not invent lines, stops, durations, or departures."
    )

    route_target = _extract_requested_plan_area(user_message)
    route_origin = _extract_requested_plan_origin(user_message)
    if visible_days > 1:
        transport_bullets = _build_multi_day_movement_guidance(
            user_message=user_message,
            language=language,
            route_origin=route_origin,
            route_target=route_target,
            transport_bullets=transport_bullets,
        )
    source_line = _build_planner_fallback_source_line(
        language,
        weather_data,
        transport_data,
        places_data,
        events_data,
        include_transport_sources=bool(transport_bullets),
    )
    source_line = _filter_planner_fallback_transport_sources(source_line, transport_bullets)
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
    if (visible_days != 1 or not route_target) and walking_only_plan:
        if _query_requests_architecture_theme(user_message):
            title = (
                f"### 📅 **Plano de arquitetura a pé de {visible_days} dia{'s' if visible_days > 1 else ''}**"
                if is_pt
                else f"### 📅 **{visible_days}-Day Architecture Walking Plan**"
            )
        else:
            title = (
                f"### 📅 **Plano a pé de {visible_days} dia{'s' if visible_days > 1 else ''}**"
                if is_pt
                else f"### 📅 **{visible_days}-Day Walking Plan**"
            )
    if requested_days > 5:
        direct = (
            "Vou limitar o pedido aos primeiros 5 dias para manter verificabilidade; rotas, preços, bilhetes, restaurantes e meteorologia futura não devem ser tratados como confirmados."
            if is_pt
            else "I’m limiting this to the first 5 days to keep it verifiable; routes, prices, tickets, restaurants, and future weather are not treated as confirmed."
        )
    elif visible_days > 1:
        if walking_only_plan and _query_requests_architecture_theme(user_message):
            direct = (
                "Segue um plano por dias focado em arquitetura e pensado para ser feito sobretudo a pé, com exemplos ancorados nos dados disponíveis."
                if is_pt
                else "Here is a day-by-day architecture plan designed primarily for walking, with examples anchored in the available data."
            )
        elif walking_only_plan:
            direct = (
                "Segue um plano por dias pensado para ser feito sobretudo a pé, com exemplos ancorados nos dados disponíveis e limitações explícitas."
                if is_pt
                else "Here is a day-by-day plan designed primarily for walking, with examples anchored in the available data and explicit limits."
            )
        else:
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
            f"### 💡 **{'Dicas' if is_pt else 'Tips'}**",
            *[f"- {constraint}" for constraint in constraints[:3]],
        ])
    lines.extend([
        "",
        ("### 📍 **Roteiro sugerido**" if clean_cards else "### 📍 **Dados confirmados para o plano**")
        if is_pt
        else ("### 📍 **Suggested Route**" if clean_cards else "### 📍 **Confirmed Planning Data**"),
    ])

    if visible_days == 1:
        block_count = 0
        visible_card_count = max(3, _requested_plan_required_component_count(user_message))
        if _query_has_explicit_anchor_sequence(user_message):
            visible_card_count = max(
                visible_card_count,
                len(_requested_anchor_labels(user_message)) + _requested_plan_required_component_count(user_message),
            )
        visible_cards = clean_cards[: min(8, visible_card_count)]
        visible_cards = _move_requested_end_card_last(visible_cards, user_message)
        visible_cards = _cluster_visible_cards_by_requested_route_areas(visible_cards, user_message)
        visible_cards = _position_requested_meal_cards_for_plan_window(visible_cards, user_message)
        visible_cards = _position_compact_local_food_stop(visible_cards, user_message)
        visible_cards = _move_requested_end_card_last(visible_cards, user_message)
        time_allocations = _planner_time_allocations_for_cards(
            visible_cards,
            _extract_requested_plan_duration_minutes(user_message),
        )
        for card in visible_cards:
            block_count += 1
            display_name = _localize_planner_display_title(
                _planner_card_display_name(card) or card["name"],
                language,
            )
            icon = "🍽️" if _card_kind_for_plan_block(card) == "food" else "🏛️"
            lines.extend(["", f"**{icon} {display_name}**"])
            if len(time_allocations) >= block_count and time_allocations[block_count - 1] > 0:
                lines.append(
                    f"    - ⏱️ **{'Tempo sugerido' if is_pt else 'Suggested time'}:** "
                    f"~{time_allocations[block_count - 1]} min"
                )
            lines.extend(
                _structured_card_detail_lines(
                    card,
                    language=language,
                    user_message=user_message,
                    indent="    ",
                    max_items=7,
                )
            )
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
            "Lisboa histórica com pouca caminhada",
            "frente ribeirinha ou património tipo Belém",
            "Lisboa moderna / Oriente com opção interior",
            "museus e miradouros com transferes curtos",
            "backup de chuva e comida económica",
        ]
        themes = themes_pt if is_pt else themes_en
        day_card_groups = _group_cards_for_multi_day_plan(clean_cards, user_message, visible_days)
        for day_index in range(visible_days):
            label = f"Dia {day_index + 1}" if is_pt else f"Day {day_index + 1}"
            theme = themes[day_index % len(themes)]
            day_cards = day_card_groups[day_index] if day_index < len(day_card_groups) else []
            if day_cards:
                lines.append(f"- **{label}:** {theme}." if is_pt else f"- **{label}:** {theme}.")
                for card in day_cards[:3]:
                    kind = _card_kind_for_plan_block(card)
                    icon = "🍽️" if kind == "food" else "🏛️"
                    display_name = _localize_planner_display_title(
                        _planner_card_display_name(card) or card["name"],
                        language,
                    )
                    lines.append(f"    - {icon} **{display_name}**")
                    lines.extend(
                        _structured_card_detail_lines(
                            card,
                            language=language,
                            user_message=user_message,
                            indent="        ",
                            max_items=6,
                        )
                    )
            else:
                lines.append(f"- **{label}:** {theme}; escolher locais concretos só após confirmação." if is_pt else f"- **{label}:** {theme}; choose concrete venues only after confirmation.")

    if transport_bullets:
        movement_heading = (
            "### 🚶 **Como te deslocas**" if walking_only_plan and is_pt
            else "### 🚶 **How to move**" if walking_only_plan
            else "### 🚇 **Como te deslocas**" if is_pt
            else "### 🚇 **How to move**"
        )
        lines.extend([
            "",
            movement_heading,
            *[
                item if item.lstrip().startswith(("-", "*")) else f"- {item}"
                for item in transport_bullets[:4]
            ],
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


def _clean_extracted_plan_area(area: str) -> str:
    """Remove origin/constraint tails from an extracted planning area."""
    cleaned = re.sub(r"\s+", " ", area or "").strip(" .:-")
    cleaned = re.sub(
        r"^\s*(?:o|a|the|meu|minha|my)?\s*"
        r"(?:hotel|alojamento|base|accommodation)\s+"
        r"(?:em|no|na|near|around|at|in|perto\s+d(?:e|o|a|os|as))\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip(" .:-")
    cleaned = re.sub(
        r"\s+(?:starting|start|beginning|begin)\s+(?:from|at|in)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip(" .:-")
    cleaned = re.sub(
        r"\s+a\s+partir\s+d(?:e|o|a|os|as)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip(" .:-")
    cleaned = re.sub(
        r"\s+(?:termin\S*|acab\S*|ending|finish(?:ing)?|end)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip(" .:-")
    cleaned = re.sub(
        r"\s+(?:e|and)\s+"
        r"(?:evit\S*|avoid(?:ing)?|sem|without|com|with|inclui\S*|include|including)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip(" .:-")
    cleaned = re.sub(
        r"\s+(?:e|and)\s+(?:mant[eé]m|mantem|preserva|keep|keeping|passa|passar|via)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip(" .:-")
    return _trim_planner_anchor_constraint_tail(cleaned)


def _extract_requested_plan_area(user_message: str) -> str:
    """Extract a named target area from a short planning request."""
    text = str(user_message or "").strip()
    ending_patterns = [
        r"\b(?:termin\S*|acab\S*|finishing|ending|finish|end)\s+(?:até|ate|by|as|às|at)(?:\s+(?:as|às))?\s+\d{1,2}(?:[:hH]\d{0,2})?\s+(?:em|no|na|near|around|at|in)\s+(?P<area>[^,.;]+?)(?:\s+(?:e|and|com|with|passa|passar|via)\b|[,.;]|$)",
        r"\b(?:termin\S*|acab\S*|finishing|ending|finish|end)\s+(?:perto\s+d(?:e|o|a|os|as)|em|no|na|near|around|at|in)\s+(?P<area>[^,.;]+?)(?:\s+(?:e|and|mant[eé]m|mantem|preserva|keep|keeping|com|with|passa|passar|via)\b|\s+no\s+segundo\s+dia|\s+on\s+day\s+\d+|[,.;]|$)",
        r"\b(?:termin\S*|acab\S*|finishing|ending|finish|end)\s+(?:(?:o|a|no|na|on|the)\s+)?(?:primeiro|segundo|terceiro|\d+)(?:\s+dia|\s+day)?\s+(?:perto\s+d(?:e|o|a|os|as)|em|no|na|near|around|at|in)\s+(?P<area>[^,.;]+?)(?:[,.;]|$)",
    ]
    for pattern in ending_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            area = _clean_extracted_plan_area(match.group("area"))
            if 2 <= len(area) <= 80:
                normalized_area = _normalize_planner_text(area)
                if "baixa" in normalized_area and "chiado" in normalized_area:
                    return "Baixa e Chiado"
                return area

    from_to_match = re.search(
        r"\bfrom\s+[^,.;]+?\s+(?:via|through)\s+[^,.;]+?\s+to\s+(?P<area>[^,.;]+?)(?:\s+(?:with|including|and|for)\b|[,.;]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if from_to_match:
        area = _clean_extracted_plan_area(from_to_match.group("area"))
        if 2 <= len(area) <= 80:
            return area
    from_to_match = re.search(
        r"\bfrom\s+[^,.;]+?\s+to\s+(?P<area>[^,.;]+?)(?:\s+(?:with|including|via|through|and|for)\b|[,.;]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if from_to_match:
        area = _clean_extracted_plan_area(from_to_match.group("area"))
        if 2 <= len(area) <= 80:
            return area
    pt_from_to_match = re.search(
        r"\bde\s+[^,.;]+?\s+(?:via|por|pela|pelo)\s+[^,.;]+?\s+(?:para|ate|até)\s+(?P<area>[^,.;]+?)"
        r"(?:\s+(?:com|inclui|incluindo|and|e|for|para\s+almo[cç]ar)\b|[,.;]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if pt_from_to_match:
        area = _clean_extracted_plan_area(pt_from_to_match.group("area"))
        if 2 <= len(area) <= 80:
            return area
    pt_from_to_match = re.search(
        r"\bde\s+[^,.;]+?\s+(?:para|ate|até)\s+(?P<area>[^,.;]+?)"
        r"(?:\s+(?:com|inclui|incluindo|via|through|and|e|for|para\s+almo[cç]ar)\b|[,.;]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if pt_from_to_match:
        area = _clean_extracted_plan_area(pt_from_to_match.group("area"))
        if 2 <= len(area) <= 80:
            return area

    patterns = [
        r"\b(?:in|around)\s+(?P<area>[^,.;]+?)\s+(?:starting|start|beginning|begin)\s+(?:from|at|in)\b",
        r"\b(?:em|no|na|nos|nas)\s+(?P<area>[^,.;]+?)\s+a\s+partir\s+d(?:e|o|a|os|as)\b",
        r"\b(?:zona\s+anterior|previous\s+area|restri[cç][aã]o\s+de\s+zona|area\s+constraint)\s*:\s*\*{0,2}(?P<area>[^,.;\n*]+)",
        r"\b(?:come[cç]ar|come[cç]ando|come[cç]a|iniciar|iniciando|start|starting|begin|beginning)\s+(?:em|no|na|nos|nas|at|from|in)\s+(?P<area>[^,.;]+?)(?:\s+(?:com|with|e|and|para|for)\b|[,.;]|$)",
        r"\b(?:perto\s+d(?:e|o|a|os|as)|junto\s+a|near|around|close\s+to)\s+(?P<area>[^,.;]+?)(?:\s+(?:com|with|e|and|para|for|sem|without)\b|[,.;]|$)",
        r"\b(?:termin\S*|acab\S*|finishing|ending|finish|end)\s+(?:perto\s+d(?:e|o|a|os|as)|em|no|na|near|around|at|in)\s+(?P<area>[^,.;]+?)(?:\s+(?:e|and|mant[eé]m|mantem|preserva|keep|keeping|com|with|passa|passar|via)\b|\s+no\s+segundo\s+dia|\s+on\s+day\s+\d+|[,.;]|$)",
        r"\b(?:termin\S*|acab\S*|finishing|ending|finish|end)\s+(?:(?:o|a|no|na|on|the)\s+)?(?:primeiro|segundo|terceiro|\d+)(?:\s+dia|\s+day)?\s+(?:perto\s+d(?:e|o|a|os|as)|em|no|na|near|around|at|in)\s+(?P<area>[^,.;]+?)(?:[,.;]|$)",
        r"\b(?:through|via|por|pela|pelo)\s+(?P<area>[^,.;]+?)(?:\s+(?:with|com|sem|without|para|for)\b|[,.;]|$)",
        r"\baround\s+(?P<area>[^,.;]+?)(?:\s+starting\b|\s+from\b|\s+with\b|$)",
        r"\bin\s+(?P<area>[^,.;]+?)(?:\s+starting\b|\s+from\b|\s+with\b|$)",
        r"\bà volta de\s+(?P<area>[^,.;]+?)(?:\s+a partir\b|\s+com\b|$)",
        r"\b(?:em|no|na|nos|nas)\s+(?P<area>[^,.;]+?)(?:\s+a partir\b|\s+com\b|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            area = _clean_extracted_plan_area(match.group("area"))
            if 2 <= len(area) <= 80:
                normalized_area = _normalize_planner_text(area)
                if "baixa" in normalized_area and "chiado" in normalized_area:
                    return "Baixa e Chiado"
                return area
    return ""


def _arrange_cards_for_multi_day_plan(
    cards: List[Dict[str, str]],
    user_message: str,
    visible_days: int,
) -> List[Dict[str, str]]:
    """Reserve explicitly requested end-area cards for the final day."""
    if visible_days <= 1 or not cards:
        return cards

    normalized_query = _normalize_planner_text(user_message)
    if (
        _query_requests_walking_only_plan(user_message)
        and not _query_has_explicit_anchor_sequence(user_message)
        and not re.search(r"\b(?:terminar|terminando|acabar|acabando|finish|finishing|end|ending)\b", normalized_query)
    ):
        indexed_cards = list(enumerate(cards))
        sorted_cards = [
            card for _index, card in sorted(
                indexed_cards,
                key=lambda item: (_planner_card_area_bucket(item[1]), item[0]),
            )
        ]
        if [card for _index, card in indexed_cards] != sorted_cards:
            return sorted_cards

    if not re.search(r"\b(?:terminar|terminando|acabar|acabando|finish|finishing|end|ending)\b", normalized_query):
        return cards
    target_area = _extract_compact_plan_area_anchor(user_message)
    if not target_area:
        return cards

    target_cards = [card for card in cards if _planner_card_matches_area(card, target_area)]
    other_cards = [card for card in cards if card not in target_cards]
    if not target_cards:
        return cards

    cards_per_day = max(1, min(3, (len(cards) + visible_days - 1) // visible_days))
    final_day_slots = min(3, cards_per_day)
    wants_food = bool(
        re.search(
            r"\b(?:gastronom\w*|restaurants?|restaurantes?|food|comida|tradicional|almo[cç]o|jantar)\b",
            normalized_query,
        )
    )
    cultural_target = sorted(
        [card for card in target_cards if _card_kind_for_plan_block(card) != "food"],
        key=_score_historic_plan_card,
        reverse=True,
    )
    food_target = sorted(
        [card for card in target_cards if _card_kind_for_plan_block(card) == "food"],
        key=_score_food_plan_card,
        reverse=True,
    )
    if wants_food and final_day_slots >= 2:
        final_day_cards = cultural_target[: final_day_slots - 1] + food_target[:1]
    else:
        final_day_cards = cultural_target[:final_day_slots]
    for card in cultural_target + food_target + target_cards:
        if len(final_day_cards) >= final_day_slots:
            break
        if card not in final_day_cards:
            final_day_cards.append(card)
    earlier_cards = list(other_cards)
    deferred_target_cards = [card for card in target_cards if card not in final_day_cards]
    target_prefix_len = max(0, cards_per_day * (visible_days - 1))
    arranged_cards = (
        earlier_cards[:target_prefix_len]
        + final_day_cards
        + earlier_cards[target_prefix_len:]
        + deferred_target_cards
    )
    visible_card_count = max(1, cards_per_day * visible_days)
    if wants_food and not any(
        _card_kind_for_plan_block(card) == "food"
        for card in arranged_cards[:visible_card_count]
    ):
        food_candidates = sorted(
            [card for card in cards if _card_kind_for_plan_block(card) == "food"],
            key=lambda card: (
                0 if _planner_card_matches_area(card, target_area) else 1,
                -_score_food_plan_card(card),
            ),
        )
        if food_candidates:
            food_card = food_candidates[0]
            arranged_cards = [card for card in arranged_cards if card is not food_card]
            final_day_start = max(0, cards_per_day * (visible_days - 1))
            if _planner_card_matches_area(food_card, target_area):
                insert_index = min(final_day_start + max(0, final_day_slots - 1), len(arranged_cards))
            else:
                insert_index = min(max(0, cards_per_day - 1), len(arranged_cards))
            if insert_index < visible_card_count and len(arranged_cards) >= visible_card_count:
                displaced_card = arranged_cards.pop(insert_index)
                arranged_cards.insert(insert_index, food_card)
                arranged_cards.insert(min(visible_card_count, len(arranged_cards)), displaced_card)
            else:
                arranged_cards.insert(insert_index, food_card)
    return arranged_cards


def _planner_card_area_bucket(card: Dict[str, str]) -> int:
    """Return a rough Lisbon area bucket for walkable multi-day grouping."""
    basis = _normalize_planner_text(
        " ".join(
            str(card.get(key, ""))
            for key in ("name", "address", "description", "category")
        )
    )
    if _PLANNER_CENTRAL_AREA_RE.search(basis):
        return 0
    if _PLANNER_BELEM_AREA_RE.search(basis):
        return 1
    if re.search(r"\b(?:parque das nacoes|oriente|olivais|expo)\b", basis):
        return 2
    return 3


def _group_cards_for_multi_day_plan(
    cards: List[Dict[str, str]],
    user_message: str,
    visible_days: int,
) -> List[List[Dict[str, str]]]:
    """Group planner place cards by day while preserving requested final areas."""
    if visible_days <= 0:
        return []
    if not cards:
        return [[] for _ in range(visible_days)]

    cards_per_day = max(1, min(3, (len(cards) + visible_days - 1) // visible_days))
    default_groups = [
        cards[day_index * cards_per_day : (day_index + 1) * cards_per_day]
        for day_index in range(visible_days)
    ]

    normalized_query = _normalize_planner_text(user_message)
    if not re.search(r"\b(?:terminar|terminando|acabar|acabando|finish|finishing|end|ending)\b", normalized_query):
        return default_groups

    target_area = _extract_compact_plan_area_anchor(user_message)
    if not target_area:
        return default_groups

    target_cards = [card for card in cards if _planner_card_matches_area(card, target_area)]
    if not target_cards:
        return default_groups

    final_day_slots = min(3, cards_per_day)
    wants_food = bool(
        re.search(
            r"\b(?:gastronom\w*|restaurants?|restaurantes?|food|comida|tradicional|almo[cç]o|jantar)\b",
            normalized_query,
        )
    )
    cultural_target = sorted(
        [card for card in target_cards if _card_kind_for_plan_block(card) != "food"],
        key=_score_historic_plan_card,
        reverse=True,
    )
    food_target = sorted(
        [card for card in target_cards if _card_kind_for_plan_block(card) == "food"],
        key=_score_food_plan_card,
        reverse=True,
    )
    if wants_food and food_target and final_day_slots >= 2:
        final_day_cards = cultural_target[: final_day_slots - 1] + food_target[:1]
    else:
        final_day_cards = cultural_target[:final_day_slots]

    for card in cultural_target + food_target + target_cards:
        if len(final_day_cards) >= final_day_slots:
            break
        if card not in final_day_cards:
            final_day_cards.append(card)

    earlier_pool = [card for card in cards if card not in target_cards]
    earlier_groups = [
        earlier_pool[day_index * cards_per_day : (day_index + 1) * cards_per_day]
        for day_index in range(max(0, visible_days - 1))
    ]
    while len(earlier_groups) < max(0, visible_days - 1):
        earlier_groups.append([])

    visible_cards = [card for group in earlier_groups for card in group] + final_day_cards
    if wants_food and not any(_card_kind_for_plan_block(card) == "food" for card in visible_cards):
        food_candidates = sorted(
            [card for card in earlier_pool if _card_kind_for_plan_block(card) == "food"],
            key=_score_food_plan_card,
            reverse=True,
        )
        if food_candidates and earlier_groups:
            food_card = food_candidates[0]
            first_group = [card for card in earlier_groups[0] if card is not food_card]
            if len(first_group) >= cards_per_day:
                first_group[-1] = food_card
            else:
                first_group.append(food_card)
            earlier_groups[0] = first_group[:cards_per_day]
        elif food_target:
            food_card = food_target[0]
            final_day_cards = [card for card in final_day_cards if card is not food_card]
            if len(final_day_cards) >= final_day_slots:
                final_day_cards[-1] = food_card
            else:
                final_day_cards.append(food_card)

    return [*earlier_groups, final_day_cards][:visible_days]


def _planner_card_matches_area(card: Dict[str, str], area: str) -> bool:
    """Return whether a card belongs to a named planning area."""
    normalized_area = _normalize_planner_text(area)
    if not normalized_area:
        return False
    if _planner_area_is_broad_city(normalized_area):
        return False
    distance_basis = _normalize_planner_text(str(card.get("distance") or ""))
    if distance_basis and normalized_area in distance_basis:
        distance_km = _planner_card_distance_km(card)
        return distance_km is None or distance_km <= 2.5
    basis = _normalize_planner_text(
        " ".join(
            str(card.get(key, ""))
            for key in ("name", "category", "address", "venue", "description", "url", "details_url")
        )
    )
    if normalized_area == "belem":
        return bool(
            re.search(
                r"\b(?:belem|bel[eé]m|belem|brasilia|bras[ií]lia|jeronimos|jer[oó]nimos|padrao|padr[aã]o|descobrimentos|torre\s+de\s+belem|imp[eé]rio|india|[ií]ndia)\b",
                basis,
            )
        )
    if normalized_area == "alfama":
        return bool(
            re.search(
                r"\b(?:alfama|se\s+de\s+lisboa|catedral\s+de\s+lisboa|largo\s+da\s+se|"
                r"santa\s+luzia|portas\s+do\s+sol|judiaria|sao\s+joao\s+da\s+praca|"
                r"terreiro\s+do\s+trigo|espirito\s+santo|castelo\s+de\s+sao\s+jorge|mouraria)\b",
                basis,
            )
        )
    if normalized_area in {"oriente", "parque das nacoes", "expo", "estacao do oriente"}:
        if "museu do oriente" in basis:
            return False
        return bool(
            re.search(
                r"\b(?:oriente|parque\s+das\s+nacoes|expo|oceanario|pavilhao\s+do\s+conhecimento|"
                r"centro\s+vasco\s+da\s+gama|vasco\s+da\s+gama|fil|altice\s+arena|"
                r"rua\s+do\s+bojador|1990|1998)\b",
                basis,
            )
        )
    if normalized_area in {"marques de pombal", "marques pombal", "marques"}:
        if re.search(r"\b(?:oeiras|2784|palacio\s+marques\s+de\s+pombal)\b", basis):
            return False
        return bool(
            re.search(
                r"\b(?:marques\s+de\s+pombal|marques\s+pombal|avenida\s+da\s+liberdade|"
                r"parque\s+eduardo\s+vii|rua\s+de\s+santa\s+marta|rua\s+rosa\s+araujo|"
                r"rodrigues\s+sampaio|sao\s+pedro\s+de\s+alcantara|s\s+pedro\s+de\s+alcantara|"
                r"torel|1250|1150|1050)\b",
                basis,
            )
        )
    if "baixa" in normalized_area and "chiado" in normalized_area:
        return bool(
            re.search(
                r"\b(?:baixa|chiado|carmo|rossio|rua\s+augusta|rua\s+do\s+carmo|"
                r"largo\s+do\s+carmo|garrett|camoes|prata|conceicao|concepcao|"
                r"elevador\s+de\s+santa\s+justa|1200|1100)\b",
                basis,
            )
        )
    if normalized_area == "chiado":
        return bool(
            re.search(
                r"\b(?:chiado|carmo|largo\s+do\s+carmo|garrett|camoes|camões|"
                r"ivens|serpa\s+pinto|capelo|rua\s+do\s+carmo|1200)\b",
                basis,
            )
        )
    if normalized_area in {"cais do sodre", "cais do sodré", "sodre", "sodre"}:
        return bool(
            re.search(
                r"\b(?:cais\s+do\s+sodre|cais\s+do\s+sodré|sodre|sodré|"
                r"ribeira\s+nova|corpo\s+santo|mercado\s+da\s+ribeira|"
                r"rua\s+do\s+alecrim|1200-376|1200-450)\b",
                basis,
            )
        )
    return normalized_area in basis


def _planner_card_distance_km(card: Dict[str, str]) -> float | None:
    """Return a parsed proximity distance in kilometres, when a card has one."""
    text = unicodedata.normalize("NFKD", str(card.get("distance") or ""))
    text = "".join(char for char in text if not unicodedata.combining(char)).lower()
    if not text:
        return None
    km_match = re.search(r"\b(?P<value>\d+(?:[\.,]\d+)?)\s*km\b", text)
    if km_match:
        return float(km_match.group("value").replace(",", "."))
    meter_match = re.search(r"\b(?P<value>\d+(?:[\.,]\d+)?)\s*m\b", text)
    if meter_match:
        return float(meter_match.group("value").replace(",", ".")) / 1000.0
    return None


def _planner_same_area_walking_items(
    cards: List[Dict[str, str]],
    user_message: str,
    language: str,
) -> List[str]:
    """Build walking movement bullets when the requested plan stays in one area."""
    if _query_requests_specific_transport_mode(user_message):
        return []
    target_area = _extract_compact_plan_area_anchor(user_message)
    if not target_area:
        return []
    if _planner_area_is_broad_city(target_area):
        return []
    area_cards = [card for card in cards if _planner_card_matches_area(card, target_area)]
    if len(area_cards) < 2:
        return []

    is_pt = language == "pt"
    area_label = target_area.strip()
    items: List[str] = []
    for origin, destination in zip(area_cards, area_cards[1:4]):
        origin_name = _planner_compact_movement_name(
            _planner_card_display_name(origin) or str(origin.get("name") or "").strip()
        )
        destination_name = _planner_compact_movement_name(
            _planner_card_display_name(destination) or str(destination.get("name") or "").strip()
        )
        if not origin_name or not destination_name or origin_name == destination_name:
            continue
        if is_pt:
            items.append(
                f"🚶 **{origin_name} → {destination_name}:** caminhada curta na zona de {area_label}; "
                "mantém esta ligação a pé se o tempo permitir."
            )
        else:
            items.append(
                f"🚶 **{origin_name} → {destination_name}:** short walk in the {area_label} area; "
                "keep this as a walking leg if conditions allow."
            )
    return items[:3]


def _planner_origin_to_first_stop_item(
    cards: List[Dict[str, str]],
    user_message: str,
    language: str,
) -> str:
    """Build a first-leg note when a plan has an explicit start anchor."""
    if not cards:
        return ""
    origin = _extract_requested_plan_origin(user_message)
    if not origin:
        return ""
    first_card = cards[0]
    first_name = _planner_compact_movement_name(
        _planner_card_display_name(first_card) or str(first_card.get("name") or "").strip()
    )
    if not first_name:
        return ""
    normalized_origin = _normalize_planner_text(origin)
    normalized_first_name = _normalize_planner_text(first_name)
    if normalized_origin and normalized_origin == normalized_first_name:
        return ""
    basis = _normalize_planner_text(
        " ".join(str(first_card.get(key, "")) for key in ("name", "address", "description"))
    )
    if normalized_origin in basis or (
        normalized_origin == "rossio"
        and re.search(r"\b(?:portas\s+de\s+santo\s+antao|restauradores|baixa|chiado)\b", basis)
    ):
        if language == "pt":
            return f"🚶 **{origin} → {first_name}:** caminhada curta na zona de partida; confirma no local se preferires evitar a pé."
        return f"🚶 **{origin} → {first_name}:** short walk from the starting area; check locally if you prefer to avoid walking."
    if _planner_card_matches_area(first_card, origin) or (
        (distance_km := _planner_card_distance_km(first_card)) is not None and distance_km <= 1.5
    ):
        if language == "pt":
            return f"🚶 **{origin} → {first_name}:** começa em {origin} e segue para a primeira paragem a pé, mantendo margem para orientação local."
        return f"🚶 **{origin} → {first_name}:** start at {origin} and walk to the first stop, keeping a little buffer for local wayfinding."
    if _query_requests_public_transport(user_message):
        if language == "pt":
            return f"🚇 **{origin} → {first_name}:** começa em {origin}; a ligação pública exata para esta primeira paragem não ficou confirmada nos dados recolhidos."
        return f"🚇 **{origin} → {first_name}:** start from {origin}; the exact public-transport leg to this first stop was not confirmed in the gathered data."
    if language == "pt":
        return f"🚶 **{origin} → {first_name}:** começa em {origin}; a ligação exata para esta primeira paragem não ficou confirmada nos dados recolhidos."
    return f"🚶 **{origin} → {first_name}:** start from {origin}; the exact first leg was not confirmed in the gathered data."


def _planner_consecutive_stop_transition_items(
    cards: List[Dict[str, str]],
    existing_items: List[str],
    user_message: str,
    language: str,
) -> List[str]:
    """Build scoped movement notes between consecutive visible itinerary stops."""
    visible_cards = [
        card for card in cards
        if str(card.get("source_id") or "").strip() != "user_request"
    ]
    if len(visible_cards) < 2:
        return []

    existing_text = _normalize_planner_text("\n".join(existing_items))
    is_pt = language == "pt"
    wants_public_transport = _query_requests_public_transport(user_message)
    items: List[str] = []
    for origin_card, destination_card in zip(visible_cards, visible_cards[1:5]):
        origin_name = _planner_compact_movement_name(
            _planner_card_display_name(origin_card) or str(origin_card.get("name") or "").strip()
        )
        destination_name = _planner_compact_movement_name(
            _planner_card_display_name(destination_card) or str(destination_card.get("name") or "").strip()
        )
        if not origin_name or not destination_name or origin_name == destination_name:
            continue
        normalized_origin = _normalize_planner_text(origin_name)
        normalized_destination = _normalize_planner_text(destination_name)
        if normalized_origin in existing_text and normalized_destination in existing_text:
            continue
        if wants_public_transport:
            items.append(
                (
                    f"🚇 **{origin_name} → {destination_name}:** a ligação de transporte público "
                    "exata entre estas paragens não ficou confirmada nos dados recolhidos; "
                    "não inventei linha, paragem ou duração."
                )
                if is_pt
                else (
                    f"🚇 **{origin_name} → {destination_name}:** the exact public-transport leg "
                    "between these stops was not confirmed in the gathered data; "
                    "I did not invent a line, stop, or duration."
                )
            )
        else:
            items.append(
                (
                    f"🚶 **{origin_name} → {destination_name}:** faz esta transição como "
                    "deslocação curta/local entre paragens do roteiro; confirma no mapa a melhor rua de ligação no momento."
                )
                if is_pt
                else (
                    f"🚶 **{origin_name} → {destination_name}:** treat this as a short/local transition "
                    "between itinerary stops; check the best walking link on the map when you go."
                )
            )
    return items


def _planner_compact_movement_name(name: str) -> str:
    """Return a concise POI label for movement bullets."""
    cleaned = re.sub(r"\s+", " ", str(name or "")).strip(" .:-")
    parts = [part.strip(" .:-") for part in cleaned.split("|") if part.strip(" .:-")]
    if len(parts) >= 2:
        return parts[0]
    return cleaned


def _query_requests_return_to_origin(user_message: str) -> bool:
    """Return whether the user explicitly asks to return to the base/origin."""
    normalized = _normalize_planner_text(user_message)
    return bool(
        re.search(
            r"\b(?:regress\w*|voltar|volta|retornar|return(?:ing)?|back)\b"
            r".{0,80}\b(?:hotel|base|alojamento|accommodation|origem|origin|ponto de partida)\b",
            normalized,
        )
        or re.search(
            r"\b(?:hotel|base|alojamento|accommodation|origem|origin|ponto de partida)\b"
            r".{0,80}\b(?:regress\w*|voltar|volta|retornar|return(?:ing)?|back)\b",
            normalized,
        )
    )


def _query_requests_return_to_hotel(user_message: str) -> bool:
    """Return whether the user asks to return to an accommodation anchor."""
    normalized = _normalize_planner_text(user_message)
    return bool(
        re.search(
            r"\b(?:regress\w*|voltar|volta|retornar|return(?:ing)?|back)\b"
            r".{0,80}\b(?:hotel|base|alojamento|accommodation)\b",
            normalized,
        )
        or re.search(
            r"\b(?:hotel|base|alojamento|accommodation)\b"
            r".{0,80}\b(?:regress\w*|voltar|volta|retornar|return(?:ing)?|back)\b",
            normalized,
        )
    )


def _extract_requested_hotel_anchor(user_message: str, language: str) -> str:
    """Extract a concrete hotel/accommodation location when the user provides one."""
    text = str(user_message or "").strip()
    patterns = [
        r"\b(?:hotel|alojamento|accommodation)\s+(?:no|na|em|in|near|perto\s+d(?:e|o|a|os|as)|at)\s+(?P<area>[^,.;]+)",
        r"\b(?:meu|minha|my|the)\s+(?:hotel|alojamento|accommodation)\s+(?:no|na|em|in|near|perto\s+d(?:e|o|a|os|as)|at)\s+(?P<area>[^,.;]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        area = re.sub(r"\s+", " ", match.group("area")).strip(" .:-")
        area = re.sub(
            r"\s+(?:and|e)\s+(?:return|regress|voltar|volta|retornar|back)\b.*$",
            "",
            area,
            flags=re.IGNORECASE,
        ).strip(" .:-")
        if area and _normalize_planner_text(area) not in {"meu", "minha", "my", "the"}:
            return f"hotel em {area}" if language == "pt" else f"hotel in {area}"
    return ""


def _planner_requested_return_target(user_message: str, language: str) -> str:
    """Return the requested return target without conflating origin and hotel."""
    if _query_requests_return_to_hotel(user_message):
        hotel_anchor = _extract_requested_hotel_anchor(user_message, language)
        if hotel_anchor:
            return hotel_anchor
        origin = _extract_requested_plan_origin(user_message)
        if re.search(r"\b(?:hotel|alojamento|accommodation)\b", origin, flags=re.IGNORECASE):
            return origin
        return (
            "hotel (localização não indicada)"
            if language == "pt"
            else "your hotel (location not specified)"
        )

    origin = _extract_requested_plan_origin(user_message)
    normalized = _normalize_planner_text(user_message)
    if not origin and "hotel" in normalized and "saldanha" in normalized:
        return "hotel no Saldanha" if language == "pt" else "hotel in Saldanha"
    return origin


def _planner_return_target_is_unknown_hotel(user_message: str, target: str) -> bool:
    """Return whether the return target is a hotel without a usable location."""
    return _query_requests_return_to_hotel(user_message) and bool(
        re.search(
            r"\b(?:localizacao nao indicada|location not specified)\b",
            _normalize_planner_text(target),
        )
    )


def _planner_return_to_origin_item(
    cards: List[Dict[str, str]],
    user_message: str,
    language: str,
) -> str:
    """Build a grounded return-leg limitation when the user asks to return to base."""
    if not cards or not _query_requests_return_to_origin(user_message):
        return ""
    target = _planner_requested_return_target(user_message, language)
    if not target:
        return ""
    last_card = cards[-1]
    last_name = _planner_card_display_name(last_card) or str(last_card.get("name") or "").strip()
    if not last_name:
        return ""
    if _planner_return_target_is_unknown_hotel(user_message, target):
        if language == "pt":
            return (
                f"🚇 **{last_name} → {target}:** inclui o regresso pedido ao hotel, mas a "
                "localização do hotel não foi indicada; não usei o ponto de partida como se fosse o hotel "
                "e não inventei linhas, paragens ou horários."
            )
        return (
            f"🚇 **{last_name} → {target}:** includes the requested return to the hotel, but the "
            "hotel location was not provided; I did not treat the starting point as the hotel "
            "or invent lines, stops, or schedules."
        )
    if language == "pt":
        return (
            f"🚇 **{last_name} → {target}:** regresso ao ponto indicado; a ligação exata "
            "não ficou confirmada nos dados recolhidos, por isso não inventei linhas, paragens ou horários."
        )
    return (
        f"🚇 **{last_name} → {target}:** return to the requested point; the exact leg was not "
        "confirmed in the gathered data, so I did not invent lines, stops, or schedules."
    )


def _extract_requested_plan_origin(user_message: str) -> str:
    """Extract an origin anchor from a short planning request."""
    text = str(user_message or "").strip()
    patterns = [
        r"\b(?:come[cç](?:ar|a|e|ando)|iniciar|iniciando|starting|start)\s+(?:perto\s+d(?:e|o|a|os|as)|junto\s+a|near|close\s+to|no|na|em|at|from|in)\s+(?P<origin>[^,.;]+?)(?:[,.;]|\s+às\b|\s+as\b|\s+at\b|$)",
        r"\bfrom\s+(?P<origin>[^,.;]+?)\s+to\b",
        r"\bstarting\s+from\s+(?P<origin>[^,.;]+?)(?:[,.;]?\s+with\b|[,.;]?\s+include\b|$)",
        r"\b(?:how\s+(?:i|we|to)\s+)?(?:get|go|travel)\s+there\s+from\s+(?P<origin>[^,.;]+?)(?:[,.;]|$)",
        r"\bfrom\s+(?P<origin>[^,.;]+?)(?:[,.;]?\s+(?:with|including|and|for|through|via|ending|end|finish)\b|[,.;]|$)",
        r"\bdesde\s+(?:o|a|os|as)?\s*(?P<origin>[^,.;]+?)(?:[,.;]?\s+com\b|[,.;]?\s+inclui\b|$)",
        r"\bcomo\s+(?:vou|vamos|chego|chegar|ir)\s+(?:la|lá|ali|a[ií])\s+(?:desde|a\s+partir\s+d(?:e|o|a|os|as))\s+(?P<origin>[^,.;]+?)(?:[,.;]|$)",
        r"\bcomo\s+(?:chegar|ir|vou|vamos)\s+(?:desde|a\s+partir\s+d(?:e|o|a|os|as))\s+(?P<origin>[^,.;]+?)(?:[,.;]|$)",
        r"\ba partir d(?:e|o|a|os|as)\s+(?P<origin>[^,.;]+?)(?:[,.;]?\s+com\b|[,.;]?\s+inclui\b|$)",
        r"\bde\s+(?P<origin>[^,.;]+?)\s+para\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            origin = re.sub(r"\s+", " ", match.group("origin")).strip(" .:-")
            origin = re.sub(
                r"\s+(?:e|and)\s+(?:termin\S*|acab\S*|regress\S*|voltar|volta|retornar|return(?:ing)?|back|ending|end)\b.*$",
                "",
                origin,
                flags=re.IGNORECASE,
            ).strip(" .:-")
            origin = re.sub(
                r"\s+(?:and\s+using|using|with|including|and\s+public\s+transport|"
                r"e\s+usando|usando|com\s+transporte|e\s+transporte)\b.*$",
                "",
                origin,
                flags=re.IGNORECASE,
            ).strip(" .:-")
            origin = _clean_requested_anchor_fragment(origin)
            origin = _trim_planner_anchor_constraint_tail(origin)
            if 2 <= len(origin) <= 80:
                return origin
    return ""


def _extract_compact_plan_area_anchor(user_message: str) -> str:
    """Return the locality anchor for compact plans without forcing a destination.

    In prompts such as "plano curto a partir do Rossio com almoço", the
    starting point is also the practical search area. Treating it as missing
    lets broad city-level evidence displace nearby stops.
    """
    area = _extract_requested_plan_area(user_message)
    if area:
        return area

    normalized = _normalize_planner_text(user_message)
    if _query_has_explicit_start_end_constraint(user_message) or _query_has_explicit_anchor_sequence(user_message):
        return ""

    origin = _extract_requested_plan_origin(user_message)
    if not origin:
        return ""
    if re.search(
        r"\b(?:a\s+partir|desde|from|come[cç]ar|come[cç]ando|iniciar|iniciando|"
        r"start|starting|plano\s+curto|short\s+plan|meio\s+dia|half\s+day|"
        r"[2-5]\s+horas?|[2-5]\s+hours?)\b",
        normalized,
    ):
        return origin
    return ""


def _planner_target_is_belem(target: str) -> bool:
    """Return whether the requested final area is the Belém riverfront cluster."""
    normalized_target = _normalize_planner_text(target)
    return bool(
        re.search(
            r"\b(?:belem|bel[eé]m|brasilia|bras[ií]lia|jeronimos|jer[oó]nimos|"
            r"padrao|padr[aã]o|descobrimentos|torre\s+de\s+belem)\b",
            normalized_target,
        )
    )


def _filter_planner_fallback_transport_sources(source_line: str, transport_bullets: List[str]) -> str:
    """Keep transport source entries only when the movement bullets use them."""
    if not source_line:
        return source_line

    movement_text = _normalize_planner_text("\n".join(transport_bullets))
    source_entries = {
        "Metro de Lisboa": "https://www.metrolisboa.pt",
        "Carris Metropolitana": "https://www.carrismetropolitana.pt",
        "Carris": "https://www.carris.pt",
        "CP": "https://www.cp.pt",
    }
    used_entries = {
        "Metro de Lisboa": bool(
            re.search(
                r"\b(?:metro|linha\s+(?:azul|verde|amarela|vermelha)|"
                r"(?:blue|green|yellow|red)\s+line)\b",
                movement_text,
            )
        ),
        "Carris Metropolitana": bool(re.search(r"\bcarris\s+metropolitana\b", movement_text)),
        "Carris": bool(
            re.search(
                r"\b(?:carris(?!\s+metropolitana)|linha\s+\d{1,4}[a-z]?|el[eé]trico|electrico|tram|autocarro)\b",
                movement_text,
            )
        ),
        "CP": bool(re.search(r"\b(?:cp|comboio|comboios|train|linha\s+de\s+cascais|cascais\s+line)\b", movement_text)),
    }

    filtered = source_line
    for label, url in source_entries.items():
        if used_entries[label]:
            continue
        entry = rf"\[\*{re.escape(label)}\*\]\({re.escape(url)}\)"
        filtered = re.sub(rf"\s*\|\s*{entry}", "", filtered)
        filtered = re.sub(rf"{entry}\s*\|\s*", "", filtered)
        filtered = re.sub(entry, "", filtered)

    if "[*" not in filtered:
        return ""
    for label, url in source_entries.items():
        if not used_entries[label]:
            continue
        entry_text = f"[*{label}*]({url})"
        if entry_text in filtered:
            continue
        timestamp_match = re.search(r"\s+\|\s+\*\*(?:Atualizado|Updated):\*\*", filtered)
        if timestamp_match:
            insert_at = timestamp_match.start()
            filtered = f"{filtered[:insert_at]} | {entry_text}{filtered[insert_at:]}"
        else:
            filtered = f"{filtered} | {entry_text}"
    filtered = re.sub(r":\s*\|\s*", ": ", filtered)
    filtered = re.sub(r"\s*\|\s*\|", " |", filtered)
    filtered = re.sub(r"\s{2,}", " ", filtered)
    return filtered.strip()


def _build_multi_day_movement_guidance(
    *,
    user_message: str,
    language: str,
    route_origin: str,
    route_target: str,
    transport_bullets: List[str],
) -> List[str]:
    """Build movement guidance for multi-day frameworks without stale card legs."""
    is_pt = language == "pt"
    normalized_query = _normalize_planner_text(user_message)
    if not re.search(
        r"\b(?:terminar|termina|terminando|termine|acabar|acaba|acabando|acabe|"
        r"finish|finishes|finishing|ends?|ending|ultimo\s+dia|last\s+day)\b",
        normalized_query,
    ):
        return transport_bullets[:2]

    origin_sentence = route_origin or ("a zona inicial" if is_pt else "the starting area")
    origin_route = route_origin or ("Zona inicial" if is_pt else "Starting area")
    target = route_target
    if not target:
        return transport_bullets[:2]

    if is_pt:
        guidance = [
            "🧭 **Dia 1:** mantém as deslocações curtas dentro da zona central e evita encadear subidas quando houver chuva.",
            f"🚇 **Dia final:** reserva {target} para o fim; confirma no próprio dia a melhor ligação entre {origin_sentence} e {target}, porque partidas e alterações operacionais são temporais.",
        ]
    else:
        guidance = [
            "🧭 **Day 1:** keep transfers short within the central area and avoid chaining uphill walks when rain is likely.",
            f"🚇 **Final day:** reserve {target} for the end; confirm the best connection between {origin_sentence} and {target} on the day itself because departures and operational changes are time-sensitive.",
        ]

    normalized_target = _normalize_planner_text(target)
    actionable_bullets = []
    for item in transport_bullets:
        if (
            not item
            or _is_generic_transport_heading(item)
            or _is_planner_transport_status_summary(item)
            or not _planner_transport_bullet_is_actionable(item)
        ):
            continue
        if normalized_target and normalized_target not in _normalize_planner_text(item):
            continue
        actionable_bullets.append(item)
        if len(actionable_bullets) >= 3:
            break

    if _planner_target_is_belem(target):
        if actionable_bullets:
            return guidance + actionable_bullets
        if is_pt:
            guidance.extend(
                [
                    f"🚇 **{origin_route} → Belém:** não há uma perna de transporte confirmada nos dados recolhidos; escolhe a ligação no operador no próprio dia antes de fechares o horário.",
                    "🚶 **Dentro de Belém:** agrupa apenas paragens com moradas confirmadas no mesmo eixo ribeirinho; se a distância real for incerta, deixa margem para transporte curto ou táxi.",
                ]
            )
        else:
            guidance.extend(
                [
                    f"🚇 **{origin_route} → Belém:** no confirmed transport leg is available in the gathered data; choose the operator connection on the day before fixing the timetable.",
                    "🚶 **Within Belém:** group only stops with confirmed addresses on the same riverfront axis; if the real distance is uncertain, leave room for a short ride or taxi.",
                ]
            )
    return guidance + actionable_bullets


def _strip_unrequested_live_departure_lines(response: str, user_message: str) -> str:
    """Remove live departure rows from itinerary prose when the user did not ask for live waits."""
    if not response or _query_requests_live_transport_status(user_message):
        return response

    cleaned_lines: list[str] = []
    skip_departure_rows = False
    departure_label = re.compile(
        r"\b(?:pr[oó]ximas?\s+(?:sa[ií]das?|partidas?|metros?)|next\s+(?:departures?|metros?))\b",
        re.IGNORECASE,
    )
    time_row = re.compile(r"^\s*[-*•]?\s*(?:\*\*)?\d{1,2}[:h]\d{2}\b")
    for line in response.splitlines():
        stripped = line.strip()
        if departure_label.search(stripped):
            skip_departure_rows = True
            continue
        if re.search(r"\b(?:tempo real|real[- ]?time|live)\b", stripped, re.IGNORECASE):
            continue
        if skip_departure_rows:
            if not stripped or stripped.startswith("### ") or re.match(r"^\s*---\s*$", stripped):
                skip_departure_rows = False
            elif time_row.search(stripped):
                continue
            elif re.search(r"\b\d{1,2}[:h]\d{2}\b", stripped) and re.search(r"\b(?:paragem|stop|departure|partida|saida)\b", stripped, re.IGNORECASE):
                continue
            else:
                skip_departure_rows = False
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def _extract_requested_origin_target_transport_bullet(
    transport_data: str,
    origin: str,
    target: str,
    language: str,
) -> str:
    """Summarize a confirmed transport leg for the requested origin and target."""
    text = str(transport_data or "")
    normalized_text = _normalize_planner_text(text)
    normalized_origin = _normalize_planner_text(origin)
    normalized_target = _normalize_planner_text(target)
    if not text or not normalized_origin or not normalized_target:
        return ""
    if normalized_origin not in normalized_text or normalized_target not in normalized_text:
        return ""

    existing_leg = _extract_existing_origin_target_transport_line(
        text,
        normalized_origin,
        normalized_target,
    )
    if existing_leg:
        return existing_leg

    route_block = _extract_origin_target_route_block(text, normalized_origin, normalized_target)
    parse_text = route_block or text

    carris_leg = _extract_requested_origin_target_carris_bullet(
        parse_text,
        origin,
        target,
        language,
    )
    if carris_leg:
        return carris_leg

    transfer_required = bool(
        re.search(
            r"\b(?:Transfer Required|Transfer at|Transfer to|Transbordo|Transfer[êe]ncia|Mudar em)\b",
            parse_text,
            flags=re.IGNORECASE,
        )
    )
    line_match = re.search(
        r"(?:Take|Apanha|Toma|Usa)\s+\*\*(?P<line>[^*]+?)\s+Line\*\*|"
        r"(?:Linha|line)\s+\*\*(?P<line_alt>[^*]+)\*\*",
        parse_text,
        flags=re.IGNORECASE,
    )
    if not line_match:
        line_match = re.search(
            r"\b(?P<line>Vermelha|Verde|Azul|Amarela|Red|Green|Blue|Yellow)\s+Line\b",
            parse_text,
            flags=re.IGNORECASE,
        )
    if not line_match:
        line_match = re.search(
            r"\*\*(?:Line|Linha):\*\*\s*(?:[^\w\s]\s*)?(?P<line>Vermelha|Verde|Azul|Amarela|Red|Green|Blue|Yellow)\b",
            parse_text,
            flags=re.IGNORECASE,
        )
    if not line_match:
        line_match = re.search(
            r"\b(?:Linha\s+(?P<line_pt>Vermelha|Verde|Azul|Amarela)|"
            r"(?P<line_en>Red|Green|Blue|Yellow)\s+Line)\b",
            parse_text,
            flags=re.IGNORECASE,
        )
    time_match = re.search(
        r"(?:Estimated travel time|Tempo total estimado|Tempo estimado)[^:\n]*:\s*\*\*(?P<time>[^\n*]+)\*\*",
        parse_text,
        flags=re.IGNORECASE,
    )
    if not time_match:
        time_match = re.search(
            r"\*\*(?:Estimated total time|Estimated travel time|Tempo total estimado|Tempo estimado):\*\*\s*(?P<time>[^\n]+)",
            parse_text,
            flags=re.IGNORECASE,
        )
    board_match = re.search(
        r"(?:Board at|Embarque na estação|Embarca em)\s+\*\*(?P<board>[^*]+)\*\*",
        parse_text,
        flags=re.IGNORECASE,
    )
    if not board_match:
        board_match = re.search(
            r"(?:Board at|Embarque na esta[çc][aã]o|Embarca em)\s+(?:station\s+)?(?P<board>[A-Za-zÀ-ÿ .'-]{2,80})",
            parse_text,
            flags=re.IGNORECASE,
        )
    exit_matches = list(re.finditer(
        r"(?:Exit at|Saia na estação|Sai em)\s+\*\*(?P<exit>[^*]+)\*\*",
        parse_text,
        flags=re.IGNORECASE,
    ))
    exit_match = exit_matches[-1] if exit_matches else None
    if not exit_match:
        fallback_exit_matches = list(re.finditer(
            r"(?:Exit at|Saia na esta[çc][aã]o|Sai em)\s+(?:station\s+)?(?P<exit>[A-Za-zÀ-ÿ .'-]{2,80})",
            parse_text,
            flags=re.IGNORECASE,
        ))
        exit_match = fallback_exit_matches[-1] if fallback_exit_matches else None

    line_value = (
        (
            line_match.groupdict().get("line")
            or line_match.groupdict().get("line_alt")
            or line_match.groupdict().get("line_pt")
            or line_match.groupdict().get("line_en")
            or ""
        ).strip()
        if line_match
        else ""
    )
    if transfer_required:
        line_value = ""
    time_value = time_match.group("time").strip() if time_match else ""
    if time_value and not re.search(r"\d", time_value):
        time_value = ""
    board = board_match.group("board").strip() if board_match else origin
    board = re.sub(r"\s*\([^)]*\)\s*$", "", board).strip()
    exit_station = exit_match.group("exit").strip() if exit_match else ""

    if not line_value and not time_value and not exit_station:
        return ""

    if language == "pt":
        mode = "Metro com transbordo" if transfer_required else f"Metro Linha {line_value}" if line_value else "Metro"
        parts = [f"🚇 **{origin} → {target}:** {mode}"]
        if board:
            parts.append(f"desde **{board}**")
        if exit_station:
            parts.append(f"até **{exit_station}**")
        if time_value:
            parts.append(f"(**{time_value}**)")
        if exit_station and _normalize_planner_text(exit_station) != normalized_target:
            parts.append(f"e segue a pé até **{target}**")
        return " ".join(parts).strip() + "."

    english_line_names = {
        "Vermelha": "Red",
        "Verde": "Green",
        "Azul": "Blue",
        "Amarela": "Yellow",
    }
    line_value = english_line_names.get(line_value, line_value)
    mode = "Metro with transfer" if transfer_required else f"Metro {line_value} Line" if line_value else "Metro"
    parts = [f"🚇 **{origin} → {target}:** take the {mode}"]
    if board:
        parts.append(f"from **{board}**")
    if exit_station:
        parts.append(f"to **{exit_station}**")
    if time_value:
        parts.append(f"(**{time_value}**)")
    if exit_station and _normalize_planner_text(exit_station) != normalized_target:
        parts.append(f"then walk to **{target}**")
    return " ".join(parts).strip() + "."


def _extract_origin_target_route_block(text: str, normalized_origin: str, normalized_target: str) -> str:
    """Return the route-output block whose header names the requested endpoints."""
    if not text or not normalized_origin or not normalized_target:
        return ""

    blocks = re.split(
        r"(?m)(?=^(?:🗺️\s+\*\*(?:Route|Rota|Trajeto|Percurso)\s*:|"
        r"###\s+.*\*\*[^*\n]*(?:→|->)[^*\n]*\*\*))",
        str(text),
    )
    for block in blocks:
        stripped = block.strip()
        if not stripped:
            continue
        header = stripped.splitlines()[0]
        normalized_header = _normalize_planner_text(header)
        if (
            normalized_origin in normalized_header
            and normalized_target in normalized_header
            and _planner_text_has_route_arrow(header)
        ):
            return stripped
    return ""


def _extract_requested_sequence_transport_bullets(
    transport_data: str,
    user_message: str,
    language: str,
) -> List[str]:
    """Extract confirmed transport bullets for adjacent user-requested anchors."""
    if not transport_data or not _query_has_explicit_anchor_sequence(user_message):
        return []

    labels = _requested_anchor_labels(user_message)
    if len(labels) < 2:
        return []

    bullets: List[str] = []
    seen: set[str] = set()
    for origin, target in zip(labels, labels[1:]):
        bullet = _extract_requested_origin_target_transport_bullet(
            transport_data,
            origin,
            target,
            language,
        )
        if not bullet:
            continue
        normalized_bullet = _normalize_planner_text(bullet)
        if normalized_bullet in seen:
            continue
        seen.add(normalized_bullet)
        bullets.append(bullet)
    return bullets


def _extract_existing_origin_target_transport_line(
    text: str,
    normalized_origin: str,
    normalized_target: str,
) -> str:
    """Return an existing summarized movement line for the requested pair."""
    for raw_line in str(text or "").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        normalized_line = _normalize_planner_text(stripped)
        if normalized_origin not in normalized_line or normalized_target not in normalized_line:
            continue
        if not _planner_text_has_route_arrow(stripped):
            continue
        if re.search(
            r"\b(?:op[cç][oõ]es carris|carris\s+\d{1,4}[a-z]?|linha\s+\d{1,4}[a-z]?|"
            r"metro|cp|comboio|autocarro|bus|tram|el[eé]trico|eletrico|caminhada|walk)\b",
            normalized_line,
            flags=re.IGNORECASE,
        ):
            return re.sub(r"^\s*[-*•]\s*", "", stripped).strip()
    return ""


def _extract_requested_origin_target_carris_bullet(
    text: str,
    origin: str,
    target: str,
    language: str,
) -> str:
    """Extract one concrete Carris/Carris-like route leg from route evidence.

    The Carris route tool returns a plain-text block such as ``15E: para
    Algés`` followed by boarding/alighting stops, optional live departures, and
    travel time. Planner synthesis needs this converted into one compact,
    user-facing movement bullet instead of falling back to an unconfirmed-leg
    warning.
    """
    if not (
        re.search(r"\b(?:direct routes found|rotas diretas|routes:|carris urban|autocarros|buses)\b", text, re.IGNORECASE)
        or re.search(r"(?m)^\s*[-*]\s*\*\*\d{1,4}[A-Za-z]?\*\*\s*:", text)
    ):
        return ""

    lines = text.splitlines()
    candidates: list[dict[str, str]] = []
    current_mode = ""
    route_line_re = re.compile(
        r"^\s*(?:[-*]\s*)?(?:\*\*)?(?P<line>\d{1,4}[A-Za-z]?)(?:\*\*)?\s*:\s*(?P<headsign>[^\n]+?)\s*$",
        flags=re.IGNORECASE,
    )
    for index, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if re.match(r"^(?:TRAMS?|EL[ÉE]TRICOS?|ELECTRICOS?|BUSES|AUTOCARROS?)$", stripped, flags=re.IGNORECASE):
            current_mode = stripped.lower()
            continue
        match = route_line_re.match(stripped)
        if not match:
            continue
        block = "\n".join(lines[index: index + 8])
        stops_match = re.search(
            r"(?:\*\*)?Stops:\*{0,2}\s*board at\s+(?:\*\*)?(?P<board>[^;*\n]+)(?:\*\*)?;\s*leave at\s+(?:\*\*)?(?P<leave>[^*\n]+)(?:\*\*)?",
            block,
            flags=re.IGNORECASE,
        )
        if not stops_match:
            stops_match = re.search(
                r"(?:\*\*)?Paragens:\*{0,2}\s*apanh[ae]\s+em\s+(?:\*\*)?(?P<board>[^;*\n]+)(?:\*\*)?;\s*sai\s+em\s+(?:\*\*)?(?P<leave>[^*\n]+)(?:\*\*)?",
                block,
                flags=re.IGNORECASE,
            )
        walk_match = re.search(
            r"(?:\*\*)?(?:Final walk|Caminhada final):\*{0,2}\s*(?P<walk>~?\s*\d+\s*min[^\n.]*)",
            block,
            flags=re.IGNORECASE,
        )
        next_match = re.search(
            r"(?:\*\*)?(?:Next|Pr[oó]ximas partidas):\*{0,2}\s*(?P<next>[^\n]+)",
            block,
            flags=re.IGNORECASE,
        )
        time_match = re.search(
            r"(?m)^\s*(?P<time>~\s*\d+\s*min)\s*(?:travel|de viagem)\s*$",
            block,
            flags=re.IGNORECASE,
        )
        if not time_match:
            time_match = re.search(
                r"(?:\*\*)?(?:Tempo estimado|Estimated time|Estimated travel time):\*{0,2}\s*(?P<time>~?\s*\d+\s*min)",
                block,
                flags=re.IGNORECASE,
            )
        candidates.append({
            "line": match.group("line").strip(),
            "headsign": match.group("headsign").strip(),
            "board": stops_match.group("board").strip() if stops_match else "",
            "leave": re.sub(r"\.\s*$", "", stops_match.group("leave").strip()) if stops_match else "",
            "walk": re.sub(r"\s+", " ", walk_match.group("walk")).strip(" *") if walk_match else "",
            "next": re.sub(r"\s+", " ", next_match.group("next")).strip(" *") if next_match else "",
            "time": re.sub(r"\s+", " ", time_match.group("time")).strip() if time_match else "",
            "mode": current_mode,
        })

    if not candidates:
        return ""

    def score(candidate: dict[str, str]) -> tuple[int, int, int]:
        return (
            1 if candidate.get("next") else 0,
            1 if candidate.get("time") else 0,
            1 if candidate.get("board") and candidate.get("leave") else 0,
        )

    best = sorted(candidates, key=score, reverse=True)[0]
    line = best.get("line", "")
    headsign = re.sub(r"^(?:para|to)\s+", "", best.get("headsign", ""), flags=re.IGNORECASE).strip()
    board = best.get("board", "")
    leave = best.get("leave", "")
    walk = best.get("walk", "")
    next_departures = best.get("next", "")
    travel_time = re.sub(r"(?i)(\d)\s*min\b", r"\1 min", best.get("time", ""))

    if language == "pt":
        board = re.sub(r"^(?:stop|paragem)\s+", "", board, flags=re.IGNORECASE).strip()
        leave = re.sub(r"^(?:stop|paragem)\s+", "", leave, flags=re.IGNORECASE).strip()
        walk_pt = re.sub(r"\bto destination\b", "até ao destino", walk, flags=re.IGNORECASE)
        parts = [f"🚌 **{origin} → {target}:** Carris **{line}**"]
        if headsign:
            parts.append(f"para **{headsign}**")
        if board and leave:
            parts.append(f"; apanha em **{board}** e sai em **{leave}**")
        if walk_pt:
            parts.append(f"; caminhada final {walk_pt}")
        if travel_time:
            parts.append(f"(**{travel_time}**)")
        sentence = " ".join(parts).replace(" ** ;", " **;").replace("** ;", "**;").strip() + "."
        if next_departures:
            next_departures_pt = re.sub(r"\(stop\s+", "(paragem ", next_departures, flags=re.IGNORECASE)
            sentence += f" Próximas partidas: {next_departures_pt}."
        return sentence

    parts = [f"🚌 **{origin} → {target}:** take Carris **{line}**"]
    if headsign:
        parts.append(f"towards **{headsign}**")
    if board and leave:
        parts.append(f"; board at **{board}** and leave at **{leave}**")
    if walk:
        parts.append(f"; final walk {walk}")
    if travel_time:
        parts.append(f"(**{travel_time}**)")
    sentence = " ".join(parts).replace(" ** ;", " **;").replace("** ;", "**;").strip() + "."
    if next_departures:
        sentence += f" Next departures: {next_departures}."
    return sentence


def _repair_visible_transport_sources(response: str) -> str:
    """Keep planner transport sources aligned with visible movement content."""
    source_match = re.search(r"(?mi)^\s*📌\s+\*\*(?:Source|Fonte):\*\*.*$", response or "")
    if not source_match:
        return response

    source_line = source_match.group(0)
    body = (response[: source_match.start()] + response[source_match.end() :]).strip()
    normalized_body = _normalize_planner_text(body)
    is_pt = bool(re.search(r"\*\*Fonte:\*\*", source_line, flags=re.IGNORECASE))
    updated_match = re.search(
        r"\|\s*\*\*(?:Updated|Atualizado):\*\*\s*(?P<time>\d{1,2}:\d{2})",
        source_line,
        flags=re.IGNORECASE,
    )
    updated_time = updated_match.group("time") if updated_match else datetime.now().strftime("%H:%M")

    existing_labels = {
        "Metro de Lisboa": "metrolisboa.pt" in source_line,
        "Carris": "carris.pt" in source_line and "carrismetropolitana.pt" not in source_line,
        "Carris Metropolitana": "carrismetropolitana.pt" in source_line,
        "CP": "cp.pt" in source_line,
        "VisitLisboa Events": "visitlisboa.com" in source_line and re.search(r"\bevent", source_line, flags=re.IGNORECASE),
        "VisitLisboa Places": "visitlisboa.com" in source_line and re.search(r"\b(?:places|locais)", source_line, flags=re.IGNORECASE),
        "IPMA": "ipma.pt" in source_line,
    }

    visible_entries: list[str] = []
    if existing_labels["IPMA"] and re.search(r"\b(?:weather|tempo|chuva|temperatura|avisos)\b", normalized_body):
        visible_entries.append("[*IPMA*](https://www.ipma.pt)")
    if (
        existing_labels["Metro de Lisboa"]
        or re.search(r"\b(?:metro|linha\s+(?:vermelha|verde|azul|amarela)|(?:red|green|blue|yellow)\s+line)\b", normalized_body)
    ):
        if re.search(r"\b(?:metro|linha\s+(?:vermelha|verde|azul|amarela)|(?:red|green|blue|yellow)\s+line)\b", normalized_body):
            visible_entries.append("[*Metro de Lisboa*](https://www.metrolisboa.pt)")
    visible_carris_movement = bool(re.search(
        r"\b(?:op[cç][oõ]es carris|carris\s+\d{1,4}[a-z]?|autocarro|autocarros|bus|buses|tram|el[eé]trico|eletrico)\b",
        normalized_body,
    ))
    if (existing_labels["Carris"] or visible_carris_movement) and re.search(
        r"\b(?:carris|autocarro|autocarros|bus|buses|tram|trams|eletrico|eletricos|el[eé]trico|linha\s+\d{2,4}[a-z]?)\b",
        normalized_body,
    ):
        visible_entries.append("[*Carris*](https://www.carris.pt)")
    if existing_labels["Carris Metropolitana"] and "carris metropolitana" in normalized_body:
        visible_entries.append("[*Carris Metropolitana*](https://www.carrismetropolitana.pt)")
    if existing_labels["CP"] and re.search(r"\b(?:cp|comboio|comboios|train|trains)\b", normalized_body):
        visible_entries.append("[*CP*](https://www.cp.pt)")
    if existing_labels["VisitLisboa Events"]:
        visible_entries.append(
            "[*VisitLisboa Eventos*](https://www.visitlisboa.com/pt-pt/eventos)"
            if is_pt
            else "[*VisitLisboa Events*](https://www.visitlisboa.com/en/events)"
        )
    if existing_labels["VisitLisboa Places"]:
        visible_entries.append(
            "[*VisitLisboa Locais*](https://www.visitlisboa.com/pt-pt/locais)"
            if is_pt
            else "[*VisitLisboa Places*](https://www.visitlisboa.com/en/places)"
        )

    deduped_entries = list(dict.fromkeys(visible_entries))
    if not deduped_entries:
        return response

    label = "Fonte" if is_pt else "Source"
    updated_label = "Atualizado" if is_pt else "Updated"
    new_line = f"📌 **{label}:** {' | '.join(deduped_entries)} | **{updated_label}:** {updated_time}"
    return response[: source_match.start()] + new_line + response[source_match.end() :]


def _repair_planner_address_map_links(response: str) -> str:
    """Rebuild Google Maps links from the visible address text."""
    if not response:
        return response

    address_line_re = re.compile(
        r"(?m)^(?P<prefix>\s*[-*]\s*(?:📍\s+)?\*\*(?:Address|Morada):\*\*\s*)"
        r"\[(?P<address>[^\]]{4,240})\]\(https://www\.google\.com/maps/search/\?api=1&query=[^)]+\)"
        r"(?P<suffix>\s*)$",
        flags=re.IGNORECASE,
    )

    def replace(match: re.Match[str]) -> str:
        address = re.sub(r"\s+", " ", match.group("address")).strip()
        if not address:
            return match.group(0)
        url = "https://www.google.com/maps/search/?api=1&query=" + quote_plus(address)
        return f"{match.group('prefix')}[{address}]({url}){match.group('suffix')}"

    repaired = address_line_re.sub(replace, response)
    truncated_address_line_re = re.compile(
        r"(?m)^(?P<prefix>\s*[-*]\s*(?:📍\s+)?\*\*(?:Address|Morada):\*\*\s*)"
        r"\[(?P<address>[^\]]{4,240})\]\(https://www\.google\.com/maps/search/\?api=1&query=[^\n)]*"
        r"(?P<suffix>\s*)$",
        flags=re.IGNORECASE,
    )
    return truncated_address_line_re.sub(replace, repaired)


def _planner_origin_target_leg_has_movement_detail(line: str) -> bool:
    """Return whether an origin-target movement bullet contains usable operational detail."""
    normalized = _normalize_planner_text(line)
    return bool(
        re.search(
            r"\b(?:ate|to|exit|sai|saia|alight|walk|segue|station|estacao|paragem|stop|"
            r"direcao|direction|duracao|duration|tempo|time|min)\b",
            normalized,
            flags=re.IGNORECASE,
        )
    )


def _remove_same_endpoint_transport_warnings(response: str, endpoint: str) -> str:
    """Remove empty movement warnings for origin and target resolved to one place."""
    endpoint_norm = _normalize_planner_text(endpoint)
    if not response or not endpoint_norm:
        return response

    kept_lines: List[str] = []
    for raw_line in response.splitlines():
        normalized_line = _normalize_planner_text(raw_line)
        same_endpoint_warning = (
            _planner_text_has_route_arrow(raw_line)
            and endpoint_norm in normalized_line
            and (
                "ligacao concreta nao ficou confirmada" in normalized_line
                or "concrete connection was not confirmed" in normalized_line
                or "exact leg" in normalized_line
            )
        )
        generic_unconfirmed_leg = (
            "ligacoes exatas entre estas paragens nao ficaram confirmadas" in normalized_line
            or "exact legs between these stops were not confirmed" in normalized_line
        )
        if same_endpoint_warning or generic_unconfirmed_leg:
            continue
        kept_lines.append(raw_line)

    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(kept_lines)).strip()
    cleaned = re.sub(
        r"(?ms)\n?---\s*\n\s*###\s+.*\*\*(?:Como te deslocas|How to move)\*\*\s*(?=\n\s*---|\n\s*###|\Z)",
        "",
        cleaned,
    )
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _ensure_requested_origin_target_in_transport_section(
    response: str,
    user_message: str,
    language: str,
    transport_data: str,
) -> str:
    """Ensure planner movement guidance keeps the user's stated origin and target visible."""
    if not response or not str(transport_data or "").strip():
        return response
    if (
        _query_has_explicit_anchor_sequence(user_message)
        and len(_requested_anchor_labels(user_message, transport_data)) > 2
    ):
        return response

    origin = _extract_requested_plan_origin(user_message)
    target = _extract_requested_plan_area(user_message)
    if not origin or not target:
        return response
    origin_norm = _normalize_planner_text(origin)
    target_norm = _normalize_planner_text(target)
    if origin_norm and target_norm and origin_norm == target_norm:
        return _remove_same_endpoint_transport_warnings(response, origin)
    if target_norm in {"lisbon", "lisboa"} or re.match(r"^(?:lisbon|lisboa)\s+(?:museums?|museus?|viewpoints?|miradouros?)\b", target_norm):
        return response
    if _query_has_explicit_start_end_constraint(user_message):
        return _ensure_requested_start_end_in_transport_section(
            response,
            user_message,
            language,
            transport_data,
        )

    movement_heading_pattern = r"(?m)^(###\s+.*\*\*(?:Como te deslocas|How to move)\*\*\s*)$"
    normalized_response = _normalize_planner_text(response)
    has_movement_section = bool(re.search(movement_heading_pattern, response))
    movement_text = _planner_movement_section_text(response)
    has_confirmed_origin_target_leg = any(
        _planner_transport_bullet_is_actionable(line)
        and _normalize_planner_text(origin) in _normalize_planner_text(line)
        and _normalize_planner_text(target) in _normalize_planner_text(line)
        and not re.search(r"\b(?:base route|percurso base)\b", _normalize_planner_text(line))
        and _planner_origin_target_leg_has_movement_detail(line)
        for line in movement_text.splitlines()
    )
    if (
        has_confirmed_origin_target_leg
        and _normalize_planner_text(origin) in normalized_response
        and _normalize_planner_text(target) in normalized_response
        and (has_movement_section or not _query_requests_movement_details(user_message))
    ):
        return _move_origin_target_movement_bullet_first(response, origin, target)

    confirmed_leg = _extract_requested_origin_target_transport_bullet(
        transport_data,
        origin,
        target,
        language,
    )
    if confirmed_leg:
        response = re.sub(
            r"(?mi)^\s*[-*]\s*(?:🗺️\s*)?\*\*(?:Base route|Percurso base):\*\*.*(?:public-transport leg|transporte p[úu]blico|indicad[ao]|confirmed|confirmada).*$\n?",
            "",
            response,
        ).strip()
        origin_norm = _normalize_planner_text(origin)
        target_norm = _normalize_planner_text(target)
        kept_lines: list[str] = []
        for raw_line in response.splitlines():
            normalized_line = _normalize_planner_text(raw_line)
            is_same_leg = (
                _planner_transport_bullet_is_actionable(raw_line)
                and origin_norm in normalized_line
                and target_norm in normalized_line
                and not _planner_origin_target_leg_has_movement_detail(raw_line)
            )
            if not is_same_leg:
                kept_lines.append(raw_line)
        response = re.sub(r"\n{3,}", "\n\n", "\n".join(kept_lines)).strip()

    if language == "pt":
        heading = "### 🚇 **Como te deslocas**"
        bullet = f"- 🗺️ **Percurso base:** começa em **{origin}** e segue para **{target}** com a ligação de transporte público confirmada abaixo."
        final_notes_pattern = r"(?m)^###\s+.*\*\*Notas finais\*\*"
    else:
        heading = "### 🚇 **How to move**"
        bullet = f"- 🗺️ **Base route:** start from **{origin}** and continue toward **{target}** using the confirmed public-transport leg below."
        final_notes_pattern = r"(?m)^###\s+.*\*\*Final notes\*\*"

    movement_heading_pattern = r"(?m)^(###\s+.*\*\*(?:Como te deslocas|How to move)\*\*\s*)$"
    if language == "pt":
        heading = "### 🚇 **Como te deslocas**"
        bullet = (
            f"- {confirmed_leg}"
            if confirmed_leg
            else f"- ⚠️ **{origin} → {target}:** a ligação concreta não ficou confirmada nos dados recolhidos; não inventei linhas, paragens, durações ou partidas."
        )
        final_notes_pattern = r"(?m)^###\s+.*\*\*Notas finais\*\*"
    else:
        heading = "### 🚇 **How to move**"
        bullet = (
            f"- {confirmed_leg}"
            if confirmed_leg
            else f"- ⚠️ **{origin} → {target}:** the concrete connection was not confirmed in the gathered data; I did not invent lines, stops, durations, or departures."
        )
        final_notes_pattern = r"(?m)^###\s+.*\*\*Final notes\*\*"

    if re.search(movement_heading_pattern, response):
        return re.sub(movement_heading_pattern, rf"\1\n{bullet}", response, count=1).strip()

    if re.search(final_notes_pattern, response):
        return re.sub(final_notes_pattern, f"{heading}\n{bullet}\n\n---\n\n\\g<0>", response, count=1).strip()

    return f"{response.rstrip()}\n\n---\n\n{heading}\n{bullet}".strip()


def _planner_last_visible_stop_from_response(response: str, language: str) -> str:
    """Extract the last top-level itinerary stop visible in a planner response."""
    candidates: List[str] = []
    in_movement = False
    for raw_line in str(response or "").splitlines():
        stripped = raw_line.strip()
        normalized = _normalize_planner_text(stripped)
        if re.match(r"^###\s+", stripped) and re.search(
            r"\b(?:como te deslocas|how to move)\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            in_movement = True
            continue
        if in_movement and stripped.startswith("### "):
            in_movement = False
        if in_movement:
            continue
        if not re.match(r"^[-*]\s+\*\*.+\*\*\s*$", stripped):
            continue
        title = re.sub(r"^[-*]\s+", "", stripped)
        title = re.sub(r"^\*\*|\*\*$", "", title).strip()
        title = re.sub(r"^[^\wÀ-ÿ]+", "", title).strip()
        if "→" in title or "->" in title:
            continue
        if re.search(
            r"\b(?:resposta direta|direct answer|preco|price|morada|address|horario|hours|fonte|source)\b",
            _normalize_planner_text(title),
        ):
            continue
        if " · " in title:
            title = title.split(" · ", 1)[1].strip()
        title = re.sub(r"^\d{1,2}:\d{2}\s*(?:·|-)\s*", "", title).strip()
        if ":" in title:
            title = title.rsplit(":", 1)[-1].strip()
        if title:
            candidates.append(title)

    if candidates:
        return candidates[-1]
    return "última paragem" if language == "pt" else "last stop"


def _planner_first_visible_stop_from_response(response: str, language: str) -> str:
    """Extract the first top-level itinerary stop visible in a planner response."""
    candidates: List[str] = []
    in_movement = False
    for raw_line in str(response or "").splitlines():
        stripped = raw_line.strip()
        normalized = _normalize_planner_text(stripped)
        if re.match(r"^###\s+", stripped) and re.search(
            r"\b(?:como te deslocas|how to move)\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            in_movement = True
            continue
        if in_movement and stripped.startswith("### "):
            in_movement = False
        if in_movement:
            continue
        if not re.match(r"^[-*]\s+\*\*.+\*\*\s*$", stripped):
            continue
        title = re.sub(r"^[-*]\s+", "", stripped)
        title = re.sub(r"^\*\*|\*\*$", "", title).strip()
        title = re.sub(r"^[^\wÀ-ÿ]+", "", title).strip()
        if "â†’" in title or "->" in title:
            continue
        if re.search(
            r"\b(?:resposta direta|direct answer|preco|price|morada|address|horario|hours|fonte|source)\b",
            _normalize_planner_text(title),
        ):
            continue
        if " Â· " in title:
            title = title.split(" Â· ", 1)[1].strip()
        title = re.sub(r"^\d{1,2}:\d{2}\s*(?:·|-)\s*", "", title).strip()
        if ":" in title:
            title = title.rsplit(":", 1)[-1].strip()
        if title:
            candidates.append(title)

    if candidates:
        return candidates[0]
    return "primeira paragem" if language == "pt" else "first stop"


def _strip_direct_start_end_movement_collapse(response: str, origin: str, target: str) -> str:
    """Remove direct start-to-end movement lines from multi-stop start/end plans."""
    if not response or not origin or not target:
        return response

    origin_norm = _normalize_planner_text(origin)
    target_norm = _normalize_planner_text(target)
    output: List[str] = []
    in_movement = False
    for raw_line in response.splitlines():
        stripped = raw_line.strip()
        normalized = _normalize_planner_text(stripped)
        if re.match(r"^###\s+", stripped) and re.search(
            r"\b(?:como te deslocas|how to move)\b",
            normalized,
        ):
            in_movement = True
            output.append(raw_line)
            continue
        if in_movement and stripped.startswith("### "):
            in_movement = False
        if (
            in_movement
            and origin_norm
            and target_norm
            and origin_norm in normalized
            and target_norm in normalized
            and ("→" in raw_line or "â†’" in raw_line or "->" in raw_line)
        ):
            continue
        output.append(raw_line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(output)).strip()


def _extract_strict_origin_target_transport_bullet(
    transport_data: str,
    origin: str,
    target: str,
) -> str:
    """Return only a transport line that explicitly names both leg endpoints."""
    normalized_origin = _normalize_planner_text(origin)
    normalized_target = _normalize_planner_text(target)
    if not normalized_origin or not normalized_target:
        return ""
    return _extract_existing_origin_target_transport_line(
        transport_data,
        normalized_origin,
        normalized_target,
    )


def _strip_generic_start_end_movement_warning(response: str) -> str:
    """Remove broad movement warnings once specific start/end bullets exist."""
    if not response:
        return response
    output: List[str] = []
    in_movement = False
    for raw_line in response.splitlines():
        stripped = raw_line.strip()
        normalized = _normalize_planner_text(stripped)
        if re.match(r"^###\s+", stripped) and re.search(
            r"\b(?:como te deslocas|how to move)\b",
            normalized,
        ):
            in_movement = True
            output.append(raw_line)
            continue
        if in_movement and stripped.startswith("### "):
            in_movement = False
        if (
            in_movement
            and "→" not in raw_line
            and "â†’" not in raw_line
            and "->" not in raw_line
            and re.search(
                r"\b(?:ligacao pedida|ligacao concreta|requested connection|exact leg)\b",
                normalized,
            )
            and re.search(r"\b(?:nao ficou confirmad|not confirmed|nao inventei|did not invent)\b", normalized)
        ):
            continue
        output.append(raw_line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(output)).strip()


def _ensure_requested_start_end_in_transport_section(
    response: str,
    user_message: str,
    language: str,
    transport_data: str,
) -> str:
    """Keep explicit start and end constraints visible without collapsing the plan."""
    if not response or not _query_has_explicit_start_end_constraint(user_message):
        return response

    origin = _extract_requested_plan_origin(user_message)
    target = _extract_requested_plan_area(user_message)
    if not origin or not target:
        return response

    response = _strip_direct_start_end_movement_collapse(response, origin, target)
    movement_text = _planner_movement_section_text(response)
    normalized_movement = _normalize_planner_text(movement_text)
    first_stop = _planner_first_visible_stop_from_response(response, language)
    last_stop = _planner_last_visible_stop_from_response(response, language)
    origin_norm = _normalize_planner_text(origin)
    target_norm = _normalize_planner_text(target)
    first_stop_norm = _normalize_planner_text(first_stop)
    last_stop_norm = _normalize_planner_text(last_stop)
    generic_first_stop = first_stop_norm in {"primeira paragem", "first stop"}
    generic_last_stop = last_stop_norm in {"ultima paragem", "last stop"}

    bullets: List[str] = []
    if (
        origin_norm
        and first_stop_norm
        and not generic_first_stop
        and first_stop_norm != origin_norm
        and origin_norm not in normalized_movement
    ):
        confirmed_start = _extract_strict_origin_target_transport_bullet(
            transport_data,
            origin,
            first_stop,
        )
        if confirmed_start:
            bullets.append(f"- {confirmed_start}")
        elif language == "pt":
            bullets.append(
                f"- ⚠️ **{origin} → {first_stop}:** começa no ponto indicado; a ligação concreta para a primeira paragem não ficou confirmada nos dados recolhidos."
            )
        else:
            bullets.append(
                f"- ⚠️ **{origin} → {first_stop}:** start from the requested point; the concrete leg to the first stop was not confirmed in the gathered data."
            )

    has_final_leg = bool(
        target_norm
        and last_stop_norm
        and target_norm in normalized_movement
        and last_stop_norm in normalized_movement
        and ("â†’" in movement_text or "->" in movement_text)
    )
    if not has_final_leg and target_norm not in _normalize_planner_text(last_stop):
        if generic_last_stop:
            confirmed_end = ""
        else:
            confirmed_end = _extract_strict_origin_target_transport_bullet(
                transport_data,
                last_stop,
                target,
            )
        if confirmed_end:
            bullets.append(f"- {confirmed_end}")
        elif not generic_last_stop and language == "pt":
            bullets.append(
                f"- ⚠️ **{last_stop} → {target}:** termina no ponto indicado; a ligação concreta não ficou confirmada nos dados recolhidos, por isso não inventei linhas, paragens ou horários."
            )
        elif not generic_last_stop:
            bullets.append(
                f"- ⚠️ **{last_stop} → {target}:** finish at the requested point; the concrete leg was not confirmed in the gathered data, so I did not invent lines, stops, or schedules."
            )

    if not bullets:
        return response

    response = _strip_generic_start_end_movement_warning(response)
    heading = "### 🚇 **Como te deslocas**" if language == "pt" else "### 🚇 **How to move**"
    final_notes_pattern = (
        r"(?m)^###\s+.*\*\*Notas finais\*\*"
        if language == "pt"
        else r"(?m)^###\s+.*\*\*Final notes\*\*"
    )
    section_re = re.compile(
        r"(?ms)(?P<header>^###\s+.*\*\*(?:Como te deslocas|How to move)\*\*\s*(?:\n|$))"
        r"(?P<body>.*?)(?=(?:\n---\s*\n|\n###\s+|\nðŸ“Œ\s+\*\*(?:Fonte|Source):|\Z))"
    )

    def append_to_movement(match: re.Match[str]) -> str:
        body = match.group("body").rstrip()
        existing = _normalize_planner_text(body)
        additions = [
            bullet for bullet in bullets
            if _normalize_planner_text(bullet) not in existing
        ]
        if not additions:
            return match.group(0)
        return f"{match.group('header')}{body}\n" + "\n".join(additions) + "\n"

    if section_re.search(response):
        return section_re.sub(append_to_movement, response, count=1).strip()

    if re.search(final_notes_pattern, response):
        return re.sub(
            final_notes_pattern,
            f"{heading}\n" + "\n".join(bullets) + "\n\n---\n\n\\g<0>",
            response,
            count=1,
        ).strip()

    return f"{response.rstrip()}\n\n---\n\n{heading}\n" + "\n".join(bullets)


def _planner_response_has_return_to_origin_movement(
    response: str,
    user_message: str,
    language: str,
) -> bool:
    """Return whether the movement section explicitly includes the requested return."""
    if not _query_requests_return_to_origin(user_message):
        return True

    target = _planner_requested_return_target(user_message, language)
    target_norm = _normalize_planner_text(target)
    movement_text = _planner_movement_section_text(response) or str(response or "")

    for line in movement_text.splitlines():
        normalized_line = _normalize_planner_text(line)
        if "→" not in line and "->" not in line:
            continue
        if target_norm and target_norm not in normalized_line:
            continue
        if re.search(r"\b(?:regress\w*|volta|voltar|retorno|return|back)\b", normalized_line):
            return True
    return False


def _ensure_requested_return_to_origin_in_transport_section(
    response: str,
    user_message: str,
    language: str,
) -> str:
    """Ensure explicit return-to-origin requests remain visible in final plans."""
    if (
        not response
        or not _query_requests_return_to_origin(user_message)
        or _planner_response_has_return_to_origin_movement(response, user_message, language)
    ):
        return response

    target = _planner_requested_return_target(user_message, language)
    if not target:
        target = "ponto indicado" if language == "pt" else "requested point"

    last_stop = _planner_last_visible_stop_from_response(response, language)
    if language == "pt":
        heading = "### 🚇 **Como te deslocas**"
        if _planner_return_target_is_unknown_hotel(user_message, target):
            bullet = (
                f"- 🚇 **{last_stop} → {target}:** inclui o regresso pedido ao hotel, mas a "
                "localização do hotel não foi indicada; não usei o ponto de partida como se fosse o hotel "
                "e não inventei linhas, paragens ou horários."
            )
        else:
            bullet = (
                f"- 🚇 **{last_stop} → {target}:** inclui o regresso ao ponto indicado; "
                "a ligação exata não ficou confirmada nos dados recolhidos, por isso não inventei "
                "linhas, paragens ou horários."
            )
        final_notes_pattern = r"(?m)^###\s+.*\*\*Notas finais\*\*"
    else:
        heading = "### 🚇 **How to move**"
        if _planner_return_target_is_unknown_hotel(user_message, target):
            bullet = (
                f"- 🚇 **{last_stop} → {target}:** includes the requested return to the hotel, but the "
                "hotel location was not provided; I did not treat the starting point as the hotel "
                "or invent lines, stops, or schedules."
            )
        else:
            bullet = (
                f"- 🚇 **{last_stop} → {target}:** includes the return to the requested point; "
                "the exact leg was not confirmed in the gathered data, so I did not invent "
                "lines, stops, or schedules."
            )
        final_notes_pattern = r"(?m)^###\s+.*\*\*Final notes\*\*"

    section_re = re.compile(
        r"(?ms)(?P<header>^###\s+.*\*\*(?:Como te deslocas|How to move)\*\*\s*\n)"
        r"(?P<body>.*?)(?=(?:\n---\s*\n|\n###\s+|\n📌\s+\*\*(?:Fonte|Source):|\Z))"
    )

    def append_to_movement(match: re.Match[str]) -> str:
        body = match.group("body").rstrip()
        if _normalize_planner_text(bullet) in _normalize_planner_text(body):
            return match.group(0)
        return f"{match.group('header')}{body}\n{bullet}\n"

    if section_re.search(response):
        return section_re.sub(append_to_movement, response, count=1).strip()

    if re.search(final_notes_pattern, response):
        return re.sub(final_notes_pattern, f"{heading}\n{bullet}\n\n---\n\n\\g<0>", response, count=1).strip()

    return f"{response.rstrip()}\n\n---\n\n{heading}\n{bullet}".strip()


def _move_origin_target_movement_bullet_first(response: str, origin: str, target: str) -> str:
    """Move the user's requested origin-target movement leg to the top."""
    if not response or not origin or not target:
        return response

    origin_norm = _normalize_planner_text(origin)
    target_norm = _normalize_planner_text(target)
    section_re = re.compile(
        r"(?ms)(?P<header>^###\s+.*\*\*(?:Como te deslocas|How to move)\*\*\s*\n)"
        r"(?P<body>.*?)(?=\n---\s*\n|\n###\s+|\Z)"
    )

    def replace(match: re.Match[str]) -> str:
        body_lines = match.group("body").splitlines()
        target_lines: List[str] = []
        other_lines: List[str] = []
        for line in body_lines:
            normalized_line = _normalize_planner_text(line)
            is_requested_leg = (
                _planner_transport_bullet_is_actionable(line)
                and origin_norm in normalized_line
                and target_norm in normalized_line
            )
            if is_requested_leg:
                target_lines.append(line)
            else:
                other_lines.append(line)
        if not target_lines:
            return match.group(0)
        reordered_lines = [*target_lines, *other_lines]
        while reordered_lines and not reordered_lines[-1].strip():
            reordered_lines.pop()
        return f"{match.group('header')}{chr(10).join(reordered_lines)}"

    return section_re.sub(replace, response, count=1)


def _clean_planner_card_description(description: str) -> str:
    """Remove nested field labels from fallback place descriptions."""
    text = str(description or "").strip()
    if _planner_text_is_negative_result(text):
        return ""
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
            "Restaurant opened on 11 January 1993. Portuguese cuisine.": (
                "Restaurante aberto em 11 de janeiro de 1993, com cozinha portuguesa."
            ),
            "Portuguese cuisine.": "Cozinha portuguesa.",
            "Light meals just off Avenida da Liberdade.": "Refeições leves junto à Avenida da Liberdade.",
        }
        fixed = fixed_translations.get(text)
        if fixed:
            return fixed
        text = re.sub(
            r"\b(Cozinha portuguesa t\S*pica|Cozinha portuguesa tradicional)\s+cuisine\b\.?",
            r"\1.",
            text,
            flags=re.IGNORECASE,
        )
        normalized = _normalize_planner_text(text)
        if re.search(
            r"\b(?:the|and|with|for|from|good|major|museum|building|architecture|"
            r"architectural|urban|design|heritage|historic|river|landmark)\b",
            normalized,
        ) and not re.search(
            r"\b(?:com|para|por|uma|um|museu|edificio|edifício|arquitetura|"
            r"arquitectura|patrimonio|património|historico|histórico)\b",
            normalized,
        ):
            if re.search(r"\b(?:architecture|architectural|design|building|urban)\b", normalized):
                return "Paragem relevante para arquitetura, design ou contexto urbano, confirmada nos dados recolhidos."
            if re.search(r"\b(?:museum|heritage|historic|cultural|landmark)\b", normalized):
                return "Paragem cultural ou patrimonial confirmada nos dados recolhidos."
            if re.search(r"\b(?:restaurant|cuisine|food|dining)\b", normalized):
                return "Paragem gastronómica confirmada nos dados recolhidos."
            return ""
        return text
    return text


def _planner_detail_value_for_language(value: str, label_key: str, language: str) -> str:
    """Localize small recurring detail values in deterministic planner cards."""
    cleaned = str(value or "").strip()
    if language != "pt" or not cleaned:
        return cleaned
    if label_key == "price":
        cleaned = re.sub(r"\bGratuito\s+with\s+Lisbon\s+Card\b", "Gratuito com Lisboa Card", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bFree\s+with\s+Lisbon\s+Card\b", "Gratuito com Lisboa Card", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bwith\s+Lisbon\s+Card\b", "com Lisboa Card", cleaned, flags=re.IGNORECASE)
    return cleaned


def _structured_card_detail_lines(
    card: Dict[str, str],
    *,
    language: str,
    user_message: str = "",
    indent: str,
    max_items: int = 6,
) -> List[str]:
    """Render useful Researcher card fields inside a structured fallback plan."""
    details = _card_details_for_plan_block(card, language=language)
    if not details:
        return []

    is_pt = language == "pt"
    label_map = {
        "description": ("📝", "Descrição" if is_pt else "Description"),
        "category": ("🏷️", "Categoria" if is_pt else "Category"),
        "when": ("🕒", "Quando" if is_pt else "When"),
        "duration": ("⏱️", "Duração" if is_pt else "Duration"),
        "venue": ("📍", "Local" if is_pt else "Venue"),
        "address": ("📍", "Morada" if is_pt else "Address"),
        "hours": ("🕒", "Horário" if is_pt else "Hours"),
        "price": ("💶", "Preço" if is_pt else "Price"),
        "features": ("✨", "Características" if is_pt else "Features"),
        "rating": ("⭐", "Avaliação" if is_pt else "Rating"),
        "website": ("🌐", "Website"),
        "tickets": ("🎟️", "Bilhetes" if is_pt else "Tickets"),
        "more details": ("🔗", "Mais detalhes" if is_pt else "More details"),
    }
    priority = [
        "description",
        "when",
        "duration",
        "venue",
        "address",
        "hours",
        "price",
        "features",
        "rating",
        "tickets",
        "website",
        "more details",
        "category",
    ]
    parsed: Dict[str, str] = {}
    for detail in details:
        match = re.match(r"^\s*(?P<label>[A-Za-zÀ-ÿ ]{2,30})\s*:\s*(?P<value>.+)$", str(detail or "").strip())
        if not match:
            continue
        key = _normalize_planner_text(match.group("label"))
        value = _clean_structured_card_detail_value(match.group("value"), key)
        if value and key not in parsed:
            parsed[key] = value

    lines: List[str] = []
    suppress_today_hours = _query_requests_future_plan(user_message)
    for key in priority:
        value = parsed.get(key, "")
        if not value:
            continue
        if key == "hours" and suppress_today_hours and re.match(r"(?i)^\s*(?:today|hoje)\s*:", value):
            continue
        if key == "category" and len(lines) >= 2:
            continue
        value = _planner_detail_value_for_language(value, key, language)
        if not value:
            continue
        icon, label = label_map.get(key, ("📝", key.title()))
        lines.append(f"{indent}- {icon} **{label}:** {value}")
        if len(lines) >= max_items:
            break
    return lines


def _clean_structured_card_detail_value(value: str, label_key: str) -> str:
    """Return a display-safe card field value or an empty string."""
    cleaned = re.sub(r"\s+", " ", str(value or "")).strip(" .;")
    if not cleaned:
        return ""
    normalized = _normalize_planner_text(cleaned)
    if normalized in {
        "n/a",
        "na",
        "unknown",
        "not available",
        "nao disponivel",
        "não disponível",
        "indisponivel",
        "indisponível",
        "+ info",
        "#",
    }:
        return ""
    if label_key == "address" and normalized in {"lisboa", "lisbon"}:
        return ""
    if label_key == "description" and _planner_text_is_negative_result(cleaned):
        return ""
    if label_key == "description" and re.match(
        r"^(?:[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+\s*)?"
        r"(?:\*\*)?(?:Rating|Avalia[cç][aã]o|Features|Caracter[ií]sticas|"
        r"Caracteristicas|Price|Pre[cç]o|Hours|Hor[aá]rio)\s*:",
        cleaned,
        flags=re.IGNORECASE,
    ):
        return ""
    if label_key in {"website", "tickets", "more details"}:
        if not re.search(r"\]\(https?://[^)]+\)", cleaned) and not re.search(r"^https?://", cleaned):
            return ""
    return cleaned


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
    *,
    include_transport_sources: bool = True,
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
    if include_transport_sources and (
        "metrolisboa" in combined
        or "metro de lisboa" in combined
        or re.search(r"\bmetro\b", combined)
        or re.search(r"\b(?:yellow|blue|green|red)\s+line\b|\blinha\s+(?:amarela|azul|verde|vermelha)\b", combined)
    ):
        sources.append("[*Metro de Lisboa*](https://www.metrolisboa.pt)")
    if include_transport_sources and ("carrismetropolitana" in combined or "carris metropolitana" in combined):
        sources.append("[*Carris Metropolitana*](https://www.carrismetropolitana.pt)")
    if include_transport_sources and (
        "carris.pt" in combined
        or "carris urban" in combined
        or "carris line" in combined
        or "linha carris" in combined
    ):
        sources.append("[*Carris*](https://www.carris.pt)")
    if include_transport_sources and ("cp.pt" in combined or "cp trains" in combined or "comboios de portugal" in combined):
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


def _query_requests_future_plan(user_message: str) -> bool:
    """Return whether the requested plan is for a future day, not today."""
    normalized_query = _normalize_planner_text(user_message or "")
    return bool(
        re.search(
            r"\b(?:tomorrow|amanha|amanhã|next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|weekend)|"
            r"proxim[ao]\s+(?:segunda|terca|terça|quarta|quinta|sexta|sabado|sábado|domingo|semana|fim\s+de\s+semana)|"
            r"\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b",
            normalized_query,
        )
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
            r"\b(?:gastronom\w*|traditional|tradicional|restaurants?|restaurantes?|food|comida|"
            r"almoco|almoço|lunch|jantar|dinner|pastry|pastelaria|pastel|pasteis|nata|"
            r"custard|tarts?|cafe|coffee)\b",
            normalized_query,
        )
    )
    return day_intent and history_intent and food_intent


def _requested_event_count(user_message: str, *, default: int = 2) -> int:
    """Extract the number of requested event stops from a planning prompt."""
    normalized = _normalize_planner_text(user_message)
    numeric_match = re.search(r"\b(?P<count>[1-5])\s+(?:eventos?|events?)\b", normalized)
    if numeric_match:
        return max(1, min(5, int(numeric_match.group("count"))))
    word_counts = {
        "um": 1,
        "uma": 1,
        "one": 1,
        "dois": 2,
        "duas": 2,
        "two": 2,
        "tres": 3,
        "três": 3,
        "three": 3,
    }
    for word, count in word_counts.items():
        if re.search(rf"\b{re.escape(word)}\s+(?:eventos?|events?)\b", normalized):
            return count
    return default


def _is_event_planning_request(normalized_query: str) -> bool:
    """Return whether a prompt asks the planner to include events."""
    plan_intent = bool(
        re.search(
            r"\b(?:plan|plano|planear|planeia|itinerary|itinerario|roteiro|programa|dia|day|fim de semana|weekend)\b",
            normalized_query,
        )
    )
    event_intent = bool(
        re.search(
            r"\b(?:evento|eventos|event|events|concerto|concert|festival|teatro|exposicao|exposição|musica|música)\b",
            normalized_query,
        )
    )
    return plan_intent and event_intent


def _is_event_food_plan_request(normalized_query: str) -> bool:
    """Return whether a prompt asks for events plus a food stop."""
    food_intent = bool(
        re.search(
            r"\b(?:gastronom\w*|restaurants?|restaurantes?|food|comida|tradicional|"
            r"almo[cç]o|jantar|dinner|cafe|coffee|pastelaria|pastry|pastel|pasteis|"
            r"nata|custard|tarts?)\b",
            normalized_query,
        )
    )
    return _is_event_planning_request(normalized_query) and food_intent


def _extract_visitlisboa_place_cards(
    text: str,
    *,
    max_items: int = 8,
    language: str = "en",
) -> List[Dict[str, str]]:
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
                "planning evidence",
                "evidencia para planeamento",
                "evidência para planeamento",
                "places and attractions",
                "locais e atracoes",
                "locais e atracoes encontrados em lisboa",
                "events found",
                "eventos encontrados",
                "restaurants",
                "restaurantes",
                "restaurantes encontrados",
                "eventos culturais",
                "cultural events",
            }:
                continue
            body = match.group("body")
            if not re.search(
                r"(?mi)(Description|Descri(?:ç|c)[aã]o|Address|Morada|Category|Categoria|Website|Site|More details|Mais detalhes|Features|Caracter(?:í|i)sticas|Rating|Avalia|Price|Pre(?:ç|c)o|Tickets|Bilhetes|Hours|Hor[aá]rio|When|Quando|Data/Hora|Data e hora|Duration|Dura(?:ç|c)[aã]o|Venue|Local)",
                body,
            ):
                continue
            description_match = re.search(r"(?mi)^\s*[-*•]?\s*(?:[^\w\s*]{1,8}\s*)?(?:\*\*)?(?:Description|Descri(?:ç|c)[aã]o)\s*:\s*(?:\*\*)?\s*(?P<value>[^\n]+)", body)
            category_match = re.search(r"(?mi)^\s*[-*•]?\s*(?:[^\w\s*]{1,8}\s*)?(?:\*\*)?(?:Category|Categoria)\s*:\s*(?:\*\*)?\s*(?P<value>[^\n]+)", body)
            when_match = re.search(r"(?mi)^\s*[-*•]?\s*(?:[^\w\s*]{1,8}\s*)?(?:\*\*)?(?:When|Quando|Data/Hora|Data e hora|Date/Time)\s*:\s*(?:\*\*)?\s*(?P<value>[^\n]+)", body)
            duration_match = re.search(r"(?mi)^\s*[-*•]?\s*(?:[^\w\s*]{1,8}\s*)?(?:\*\*)?(?:Duration|Dura(?:ç|c)[aã]o)\s*:\s*(?:\*\*)?\s*(?P<value>[^\n]+)", body)
            venue_match = re.search(r"(?mi)^\s*[-*•]?\s*(?:[^\w\s*]{1,8}\s*)?(?:\*\*)?(?:Venue|Local)\s*:\s*(?:\*\*)?\s*(?P<value>[^\n]+)", body)
            address_match = re.search(r"(?mi)^\s*[-*•]?\s*(?:[^\w\s*]{1,8}\s*)?(?:\*\*)?(?:Address|Morada|Location|Localiza(?:ç|c)[aã]o)\s*:\s*(?:\*\*)?\s*(?P<value>[^\n]+)", body)
            hours_match = re.search(r"(?mi)^\s*[-*•]?\s*(?:[^\w\s*]{1,8}\s*)?(?:\*\*)?(?:Hours|Hor[aá]rio|Horarios|Horários)\s*:\s*(?:\*\*)?\s*(?P<value>[^\n]+)", body)
            price_match = re.search(r"(?mi)^\s*[-*•]?\s*(?:[^\w\s*]{1,8}\s*)?(?:\*\*)?(?:Price|Pre(?:ç|c)o)\s*:\s*(?:\*\*)?\s*(?P<value>[^\n]+)", body)
            features_match = re.search(r"(?mi)^\s*[-*•]?\s*(?:[^\w\s*]{1,8}\s*)?(?:\*\*)?(?:Features|Caracter(?:í|i)sticas)\s*:\s*(?:\*\*)?\s*(?P<value>[^\n]+)", body)
            rating_match = re.search(r"(?mi)^\s*[-*•]?\s*(?:[^\w\s*]{1,8}\s*)?(?:\*\*)?(?:Rating|Avalia(?:ç|c)[aã]o)\s*:\s*(?:\*\*)?\s*(?P<value>[^\n]+)", body)
            phone_match = re.search(r"(?mi)^\s*[-*•]?\s*(?:[^\w\s*]{1,8}\s*)?(?:\*\*)?(?:Phone|Telefone)\s*:\s*(?:\*\*)?\s*(?P<value>[^\n]+)", body)
            email_match = re.search(r"(?mi)^\s*[-*•]?\s*(?:[^\w\s*]{1,8}\s*)?(?:\*\*)?(?:Email|E-mail)\s*:\s*(?:\*\*)?\s*(?P<value>[^\n]+)", body)
            distance_match = re.search(r"(?mi)^\s*[-*•]?\s*(?:[^\w\s*]{1,8}\s*)?(?:\*\*)?(?:Distance|Dist\S{0,12}ncia)\s*:\s*(?:\*\*)?\s*(?P<value>[^\n]+)", body)
            url_entries: List[tuple[str, str, str]] = []
            url_field_re = re.compile(
                r"(?mi)^\s*[-*•]?\s*(?:[^\w\s*]{1,8}\s*)?"
                r"\*\*(?P<label>Website|Site|More details|Mais detalhes|Tickets|Bilhetes):\*\*\s*"
                r"(?P<value>https?://\S+|\[[^\]]+\]\(https?://[^)]+\))"
            )
            icon_url_re = re.compile(
                r"(?mi)^\s*[-*•]?\s*(?P<icon>🔗|🌐|🎟️)\s*"
                r"(?P<value>https?://\S+|\[[^\]]+\]\(https?://[^)]+\))"
            )
            for match in url_field_re.finditer(body):
                url_entries.append((match.group("label").strip(), match.group("value").strip(), ""))
            for match in icon_url_re.finditer(body):
                icon = match.group("icon")
                if icon == "🎟️":
                    label = "Tickets"
                elif icon == "🔗":
                    label = "More details"
                else:
                    label = "Website"
                url_entries.append((label, match.group("value").strip(), match.group("icon")))
            description = description_match.group("value").strip() if description_match else ""
            if not description:
                for raw_line in body.splitlines():
                    line = raw_line.strip()
                    if (
                        not line
                        or re.match(
                            r"^[-*•]?\s*(?:[^\w\s*]{1,8}\s*)?\*\*"
                            r"(?:Category|Categoria|Address|Morada|Location|Localiza(?:ç|c)[aã]o|"
                            r"Hours|Hor[aá]rio|Price|Pre(?:ç|c)o|Features|Caracter(?:í|i)sticas|"
                            r"Rating|Avalia(?:ç|c)[aã]o|Avaliacao|Phone|Telefone|Email|E-mail|"
                            r"Website|Site|Tickets|Bilhetes|More details|Mais detalhes)\s*:",
                            line,
                            flags=re.IGNORECASE,
                        )
                        or re.search(r"\*\*[^*\n]{2,40}:\*\*", line)
                        or re.match(
                            r"^[-*•]?\s*(?:🌐|🔗|🎟️)\s*(?:https?://\S+|\[[^\]]+\]\(https?://[^)]+\))\s*$",
                            line,
                        )
                        or _planner_text_is_negative_result(line)
                    ):
                        continue
                    description = re.sub(r"\s+", " ", line).strip()
                    break
            website_url = ""
            website_label = ""
            tickets_url = ""
            tickets_label = ""
            details_url = ""
            details_label = ""
            for field_label, raw_url, _icon in url_entries:
                normalized_url = raw_url
                value_label = ""
                markdown_url_match = re.search(r"\((https?://[^)]+)\)", normalized_url)
                if markdown_url_match:
                    label_match = re.match(r"\[([^\]]+)\]\(", normalized_url)
                    if label_match:
                        value_label = label_match.group(1).strip()
                    normalized_url = markdown_url_match.group(1)
                field_key = _normalize_planner_text(field_label)
                if "ticket" in field_key or "bilhet" in field_key:
                    if not tickets_url:
                        tickets_url = normalized_url
                        tickets_label = value_label or ("Comprar bilhetes" if language == "pt" else "Tickets")
                elif "visitlisboa.com" in normalized_url.lower() or "detail" in field_key or "detalh" in field_key:
                    if not details_url:
                        details_url = normalized_url
                        details_label = value_label or "VisitLisboa"
                elif not website_url:
                    website_url = normalized_url
                    website_label = value_label or ("Website oficial" if language == "pt" else "Official website")
            url = details_url or website_url
            url_label = details_label if details_url else website_label
            cards.append(
                {
                    "name": name,
                    "category": category_match.group("value").strip() if category_match else "",
                    "when": when_match.group("value").strip() if when_match else "",
                    "duration": duration_match.group("value").strip() if duration_match else "",
                    "venue": venue_match.group("value").strip() if venue_match else "",
                    "address": address_match.group("value").strip() if address_match else "",
                    "hours": hours_match.group("value").strip() if hours_match else "",
                    "price": price_match.group("value").strip() if price_match else "",
                    "features": features_match.group("value").strip() if features_match else "",
                    "rating": rating_match.group("value").strip() if rating_match else "",
                    "phone": phone_match.group("value").strip() if phone_match else "",
                    "email": email_match.group("value").strip() if email_match else "",
                    "distance": distance_match.group("value").strip() if distance_match else "",
                    "url": url,
                    "url_label": url_label,
                    "website_url": website_url,
                    "website_label": website_label,
                    "tickets_url": tickets_url,
                    "tickets_label": tickets_label,
                    "details_url": details_url,
                    "details_label": details_label,
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
    conversation_context: str = "",
) -> str:
    """Build a generic fallback from evidence cards rather than prompt-specific templates."""
    if (_extract_requested_day_count(user_message) or 1) > 1:
        return ""

    normalized_query = _normalize_planner_text(user_message)
    if not re.search(
        r"\b(?:plan|plano|itinerary|itinerario|roteiro|planeia|planear|programa|day|dia|afternoon|evening|museum|museu|visit|visitar|tour|evento|eventos|events?)\b",
        normalized_query,
    ):
        return ""
    prior_place_context = conversation_context if _query_references_previous_place_set(user_message) else ""
    combined_context = "\n".join([prior_place_context, places_data or "", events_data or ""])
    requested_label_count = len(_requested_anchor_labels(user_message, combined_context))
    requested_stop_count = _requested_plan_required_component_count(user_message)
    card_limit = 24 if _query_requests_food_stop(user_message) or requested_stop_count >= 4 else 12
    if requested_label_count >= 3:
        card_limit = max(card_limit, 48)
    if requested_stop_count >= 5:
        card_limit = max(card_limit, 48)
    cards = _extract_visitlisboa_place_cards(combined_context, max_items=card_limit, language=language)
    cards = [
        {**card, "name": clean_name}
        for card in cards
        for clean_name in [_sanitize_planner_place_name(card.get("name", ""))]
        if clean_name and not _planner_card_is_synthetic_plan_heading({**card, "name": clean_name})
    ]
    if not _is_event_planning_request(normalized_query):
        cards = [
            card for card in cards
            if not _planner_card_is_event_result(card)
        ]
    if _is_oriente_station_nearby_request(user_message):
        cards = [
            card for card in cards
            if "museu do oriente" not in _normalize_planner_text(card.get("name", ""))
        ]
        cards = _dedupe_planner_cards([*_oriente_station_locality_cards(language), *cards])
    cards = _filter_planner_cards_for_request_constraints(cards, user_message)
    target_area = _extract_compact_plan_area_anchor(user_message)
    strict_same_area_context = bool(
        target_area
        and re.search(
            r"\b(?:zona\s+anterior|previous\s+area|restri[cç][aã]o\s+de\s+zona|area\s+constraint)\b",
            normalized_query,
        )
    )
    if strict_same_area_context:
        area_cards = [
            card for card in cards
            if _planner_card_matches_area(card, target_area)
        ]
        cards = area_cards
    complete_cards = [
        card
        for card in cards
        if str(card.get("address") or "").strip()
        or str(card.get("url") or card.get("details_url") or "").strip()
        or _card_kind_for_plan_block(card) == "event"
        or str(card.get("source_id") or "").strip()
    ]
    if complete_cards:
        cards = complete_cards
    if not cards:
        return ""

    return _build_card_based_renderer_fallback(
        user_message=user_message,
        language=language,
        cards=cards,
        weather_data=weather_data,
        transport_data=transport_data,
        places_data=combined_context if prior_place_context else places_data,
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
    normalized_query = _normalize_planner_text(user_message)
    cards = _filter_planner_cards_for_request_constraints(cards, user_message)
    if not _is_event_planning_request(normalized_query):
        cards = [
            card for card in cards
            if not _planner_card_is_event_result(card)
        ]
    if not cards:
        return ""
    requested_labels = _requested_anchor_labels(user_message, "\n".join([places_data or "", events_data or ""]))
    strict_requested_sequence = _query_has_explicit_anchor_sequence(user_message)
    selection_limit = 8 if (
        _query_references_previous_place_set(user_message)
        or len(_requested_meal_kinds(user_message)) > 1
    ) else 6
    selection_limit = _selection_limit_for_requested_cardinality(user_message, selection_limit)
    if strict_requested_sequence:
        ordered_labels = _extract_requested_anchor_phrases(user_message)
        seen_ordered = {_normalize_planner_text(label) for label in ordered_labels}
        requested_labels = [
            *ordered_labels,
            *[
                label for label in requested_labels
                if _normalize_planner_text(label) not in seen_ordered
            ],
        ]
    requested_cards = _requested_anchor_cards_in_order(requested_labels, cards, language)
    selected_cards = _select_planner_cards_for_request(cards, user_message)
    if not selected_cards and requested_cards:
        selected_cards = requested_cards[:selection_limit]
    if not selected_cards:
        return ""
    if strict_requested_sequence and requested_cards:
        selected_cards = _insert_requested_food_stop_if_needed(
            _dedupe_planner_cards([*requested_cards, *selected_cards])[:selection_limit],
            cards,
            user_message,
            language,
        )[:selection_limit]
    elif requested_cards:
        if _query_references_previous_place_set(user_message):
            selected_cards = requested_cards[:selection_limit]
        else:
            selected_cards = _dedupe_planner_cards([*requested_cards, *selected_cards])[:selection_limit]
    historic_food_request = _is_historic_gastronomy_day_request(normalized_query)
    event_food_request = _is_event_food_plan_request(normalized_query)
    if _query_requests_food_stop(user_message):
        selected_cards = [
            card for card in selected_cards
            if _card_kind_for_plan_block(card) != "food"
            or _food_card_matches_requested_context(card, user_message)
        ]
    if _query_requests_food_stop(user_message) and sum(
        1 for card in selected_cards if _card_kind_for_plan_block(card) == "food"
    ) < max(1, len(_requested_meal_kinds(user_message))):
        selected_cards = _insert_requested_food_stop_if_needed(
            selected_cards,
            cards,
            user_message,
            language,
        )[:selection_limit]
    selected_cards = _ensure_selected_cards_satisfy_requested_counts(
        selected_cards,
        cards,
        user_message,
    )[:selection_limit]
    selected_cards = _insert_requested_cultural_stop_if_needed(
        selected_cards,
        cards,
        user_message,
        language,
    )[:selection_limit]
    type_placeholders = _requested_type_placeholder_cards(selected_cards, user_message, language)
    if type_placeholders:
        selected_cards = _dedupe_planner_cards([*type_placeholders, *selected_cards])[:selection_limit]
    if historic_food_request and not strict_requested_sequence:
        selected_cards = _order_historic_food_cards(selected_cards)
    selected_cards = _limit_cards_for_user_cardinality(selected_cards, user_message)
    selected_cards = _move_requested_origin_card_first(selected_cards, user_message)
    selected_cards = _drop_origin_name_collision_cards(selected_cards, user_message)
    selected_cards = _move_requested_end_card_last(selected_cards, user_message)
    compact_duration_minutes = _extract_requested_plan_duration_minutes(user_message)
    if (
        selected_cards
        and (
            (compact_duration_minutes is not None and compact_duration_minutes <= 300)
            or re.search(r"\b(?:pequeno|curto|compacto|meio\s+dia|half\s+day|short|small|compact)\b", normalized_query)
        )
        and len(selected_cards) > 3
        and _requested_plan_required_component_count(user_message) <= 3
        and not (
            strict_requested_sequence
            and _requested_plan_total_stop_count(user_message) > 0
        )
    ):
        compact_cards: List[Dict[str, str]] = []
        first_food = next((card for card in selected_cards if _card_kind_for_plan_block(card) == "food"), None)
        first_place = next((card for card in selected_cards if _card_kind_for_plan_block(card) != "food"), None)
        if first_place:
            compact_cards.append(first_place)
        if first_food:
            compact_cards.append(first_food)
        for card in selected_cards:
            if len(compact_cards) >= 3:
                break
            if card not in compact_cards:
                compact_cards.append(card)
        selected_cards = _dedupe_planner_cards(compact_cards)

    evidence = build_evidence_bundle(
        weather_data=weather_data,
        transport_data=transport_data,
        places_data=places_data,
        events_data=events_data,
        qa_disclaimers=qa_disclaimers,
    )
    is_pt = language == "pt"
    blocks: List[PlanBlock] = []
    block_limit = _selection_limit_for_requested_cardinality(user_message, 8 if selection_limit > 6 else 5)
    required_visible_cards = _requested_plan_required_component_count(user_message) + len(
        _requested_anchor_labels(user_message, "\n".join([places_data or "", events_data or ""]))
    )
    if required_visible_cards > 0:
        block_limit = min(8, max(block_limit, required_visible_cards))
    visible_cards = selected_cards[:block_limit]
    end_area = _extract_requested_plan_area(user_message)
    end_key = _normalize_planner_text(end_area)
    if end_key:
        exact_end_visible = any(
            _normalize_planner_text(_planner_card_display_name(card) or card.get("name", "")) == end_key
            for card in visible_cards
        )
        if not exact_end_visible:
            exact_end_card = next(
                (
                    card
                    for card in selected_cards
                    if _normalize_planner_text(_planner_card_display_name(card) or card.get("name", "")) == end_key
                ),
                None,
            )
            if exact_end_card:
                visible_cards = [
                    *[card for card in visible_cards[: max(0, block_limit - 1)] if card is not exact_end_card],
                    exact_end_card,
                ]
            elif not any(_planner_card_matches_area(card, end_area) for card in visible_cards):
                area_end_card = next(
                    (
                        card
                        for card in selected_cards[block_limit:]
                        if _planner_card_matches_area(card, end_area)
                    ),
                    None,
                )
                if area_end_card:
                    visible_cards = [*visible_cards[: max(0, block_limit - 1)], area_end_card]
    visible_cards = _move_requested_end_card_last(visible_cards, user_message)
    if _query_treats_start_anchor_as_origin_only(user_message):
        origin_key = _normalize_planner_text(_extract_requested_plan_origin(user_message))

        def is_origin_anchor(card: Dict[str, str]) -> bool:
            title_key = _normalize_planner_text(_planner_card_display_name(card) or card.get("name", ""))
            return (
                bool(origin_key)
                and title_key == origin_key
                and str(card.get("source_id") or "").strip() == "user_request"
            )

        if any(is_origin_anchor(card) for card in visible_cards):
            visible_cards = [card for card in visible_cards if not is_origin_anchor(card)]
            for candidate in selected_cards:
                if len(visible_cards) >= block_limit:
                    break
                if is_origin_anchor(candidate) or candidate in visible_cards:
                    continue
                visible_cards.append(candidate)
            visible_cards = _move_requested_end_card_last(visible_cards, user_message)
    visible_cards = _limit_visible_cards_to_requested_anchor_sequence(
        visible_cards,
        selected_cards,
        requested_labels,
        user_message,
        language,
    )
    visible_cards = _position_requested_meal_cards_for_plan_window(visible_cards, user_message)
    visible_cards = _position_compact_local_food_stop(visible_cards, user_message)
    visible_cards = _limit_visible_cards_for_requested_type_counts(visible_cards, user_message)
    visible_cards = _cluster_visible_cards_by_requested_route_areas(visible_cards, user_message)
    visible_cards = _position_requested_meal_cards_for_plan_window(visible_cards, user_message)
    visible_cards = _position_compact_local_food_stop(visible_cards, user_message)
    visible_cards = _move_requested_end_card_last(visible_cards, user_message)
    time_allocations = _planner_time_allocations_for_cards(
        visible_cards,
        _extract_requested_plan_duration_minutes(user_message),
    )
    schedule_labels = _planner_schedule_labels_for_cards(
        visible_cards,
        time_allocations,
        user_message,
    )
    for index, card in enumerate(visible_cards, start=1):
        details = (
            _card_details_for_itinerary_block(card, language=language)
            if historic_food_request or event_food_request
            else _card_details_for_plan_block(card, language=language)
        )
        if len(time_allocations) >= index and time_allocations[index - 1] > 0:
            details = [
                f"Suggested time: ~{time_allocations[index - 1]} min",
                *details,
            ]
        if len(schedule_labels) >= index and schedule_labels[index - 1]:
            details = [
                f"Suggested schedule: {schedule_labels[index - 1]}",
                *details,
            ]
        display_name = _planner_card_display_name(card)
        requested_time_label = _requested_time_label_for_card(card, user_message)
        block_title = _itinerary_block_title(
            display_name or card["name"],
            card,
            index=index,
            historic_food_request=historic_food_request,
            event_food_request=event_food_request,
            language=language,
        )
        if requested_time_label:
            block_title = re.sub(r"^\s*\d{1,2}:\d{2}\s*(?:[-\u00b7]|\u00b7)\s*", "", block_title).strip()
            block_title = f"{requested_time_label} \u00b7 {block_title}"
            details = [
                (
                    f"Hora pedida pelo utilizador: {requested_time_label}"
                    if is_pt
                    else f"User-requested time: {requested_time_label}"
                ),
                *details,
            ]
        blocks.append(
            PlanBlock(
                title=block_title,
                kind=_card_kind_for_plan_block(card),
                purpose=_card_purpose_for_plan_block(card, is_pt),
                details=details,
                source_ids=[source_id for source_id in [_card_source_id_for_plan_block(card)] if source_id],
            )
        )

    walking_only_plan = _query_requests_walking_only_plan(user_message)
    movement_items = [] if walking_only_plan else [
        item
        for item in (
            _fallback_bullet_body(bullet)
            for bullet in _extract_planner_fallback_bullets(transport_data, max_items=5)
        )
        if item
        and not _is_generic_transport_heading(item)
        and not _is_planner_transport_status_summary(item)
        and _planner_transport_bullet_is_actionable(item)
    ]
    sequence_movement_items: List[str] = []
    if strict_requested_sequence:
        sequence_movement_items = _extract_requested_sequence_transport_bullets(
            transport_data,
            user_message,
            language,
        )
        movement_items = [
            item for item in movement_items
            if _movement_item_matches_requested_sequence(item, user_message)
        ]
        if sequence_movement_items:
            movement_items = list(dict.fromkeys([*sequence_movement_items, *movement_items]))
    strict_sequence_limitations = (
        _requested_sequence_transport_limitation_bullets(user_message, language)
        if strict_requested_sequence and _query_requests_movement_details(user_message)
        else []
    )
    if strict_requested_sequence and not movement_items and strict_sequence_limitations:
        movement_items = strict_sequence_limitations
    same_area_walking_items = _planner_same_area_walking_items(
        visible_cards,
        user_message,
        language,
    )
    first_leg_item = _planner_origin_to_first_stop_item(
        visible_cards,
        user_message,
        language,
    )
    if first_leg_item and not any(
        _normalize_planner_text(_extract_requested_plan_origin(user_message))
        in _normalize_planner_text(item)
        for item in movement_items
    ):
        movement_items = [first_leg_item, *movement_items]
    if same_area_walking_items and (
        _query_requests_movement_details(user_message)
        or _query_requests_low_walk_plan(user_message)
    ) and not strict_requested_sequence:
        if movement_items and _extract_requested_plan_origin(user_message):
            movement_items = list(dict.fromkeys([*movement_items, *same_area_walking_items]))
        else:
            movement_items = same_area_walking_items
    if walking_only_plan:
        movement_items = _planner_walking_only_guidance(language)
    elif not movement_items:
        sequence_limitations = _requested_sequence_transport_limitation_bullets(user_message, language)
        if _query_requests_movement_details(user_message) and sequence_limitations:
            movement_items = sequence_limitations
        elif _query_requests_public_transport(user_message):
            movement_items = [
                (
                    "As ligações exatas entre estas paragens não ficaram confirmadas nos dados recolhidos; "
                    "não inventei linhas, paragens ou durações."
                )
                if is_pt
                else (
                    "Exact legs between these stops were not confirmed in the gathered data; "
                    "I did not invent lines, stops, or durations."
                )
            ]
        else:
            movement_items = [
                (
                    "As ligações exatas entre estas paragens não ficaram confirmadas nos dados recolhidos; "
                    "não inventei linhas, paragens ou durações."
                )
                if is_pt
                else (
                    "Exact legs between these stops were not confirmed in the gathered data; "
                    "I did not invent lines, stops, or durations."
                )
            ]
    if movement_items:
        selected_context = _planner_selected_card_context(visible_cards)
        filtered_movement_items: List[str] = []
        for item in movement_items:
            if _movement_item_is_self_referential_origin(item, user_message):
                continue
            if (
                _planner_text_has_route_arrow(item)
                and _planner_transport_bullet_is_actionable(item)
                and not _planner_movement_item_is_relevant(item, user_message, selected_context)
            ):
                continue
            filtered_movement_items.append(item)
        if filtered_movement_items:
            movement_items = filtered_movement_items
        elif same_area_walking_items and not strict_requested_sequence:
            movement_items = same_area_walking_items
        elif _query_requests_movement_details(user_message) or _query_requests_public_transport(user_message):
            movement_items = strict_sequence_limitations or [
                (
                    "As ligações exatas entre as paragens selecionadas não ficaram confirmadas nos dados recolhidos; "
                    "não inventei linhas, paragens ou durações."
                )
                if is_pt
                else (
                    "Exact legs between the selected stops were not confirmed in the gathered data; "
                    "I did not invent lines, stops, or durations."
                )
            ]
        else:
            movement_items = []
    consecutive_transition_items = _planner_consecutive_stop_transition_items(
        visible_cards,
        movement_items,
        user_message,
        language,
    )
    if consecutive_transition_items:
        movement_items = list(dict.fromkeys([*movement_items, *consecutive_transition_items]))
    return_item = _planner_return_to_origin_item(
        selected_cards,
        user_message,
        language,
    )
    if return_item and not any(
        re.search(r"\b(?:regresso|return|back)\b", _normalize_planner_text(item))
        for item in movement_items
    ):
        movement_items.append(return_item)
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
    block_source_ids = [
        source_id
        for block in blocks
        for source_id in (block.source_ids or [])
        if source_id in SOURCE_CATALOG
    ]
    source_ids = list(dict.fromkeys(block_source_ids))
    if movement_items:
        source_ids.extend(
            source_id for source_id in ("metro", "carris", "carris_metropolitana", "cp")
            if source_id in evidence.sources
        )
    if weather_items and "ipma" in evidence.sources:
        source_ids.append("ipma")
    source_ids = list(dict.fromkeys(source_ids))
    if walking_only_plan:
        source_ids = [
            source_id for source_id in source_ids
            if source_id not in {"metro", "carris", "carris_metropolitana", "cp"}
        ]
    for source_id in source_ids:
        if source_id not in evidence.sources and source_id in SOURCE_CATALOG:
            evidence.sources[source_id] = SOURCE_CATALOG[source_id]

    if _query_requests_walking_only_plan(user_message):
        tip_text = (
            "Mantém 20-30 minutos de margem entre paragens para observação, descanso e fotografias."
            if is_pt
            else "Keep 20-30 minutes of buffer between stops for observation, rest, and photos."
        )
    else:
        tip_text = (
            "Mantém 20-30 minutos de margem entre a deslocação e a paragem cultural."
            if is_pt
            else "Keep 20-30 minutes of buffer between the transport leg and the cultural stop."
        )

    draft = PlanDraft(
        title=title,
        direct_answer=direct,
        blocks=blocks,
        movement_logic=movement_items,
        weather_strategy=weather_items,
        tips=[tip_text],
        limitations=limitation_items,
        source_ids=source_ids,
    )
    rendered = render_plan_markdown(draft, evidence.sources, language=language)
    rendered = _strip_unrequested_live_departure_lines(rendered, user_message)
    rendered = _ensure_requested_origin_target_in_transport_section(
        rendered,
        user_message,
        language,
        transport_data,
    )
    if strict_sequence_limitations and not sequence_movement_items and not any(
        _normalize_planner_text(item) in _normalize_planner_text(rendered)
        for item in strict_sequence_limitations
    ):
        movement_heading = "### 🚇 **Como te deslocas**" if is_pt else "### 🚇 **How to move**"
        movement_section = "\n\n---\n\n" + "\n".join(
            [movement_heading, "", *[f"- {item}" for item in strict_sequence_limitations]]
        )
        if re.search(r"\n\n---\n\n###\s+💡", rendered):
            rendered = re.sub(r"\n\n---\n\n###\s+💡", f"{movement_section}\n\n---\n\n### 💡", rendered, count=1)
        elif re.search(r"\n\n###\s+⚠️", rendered):
            rendered = re.sub(r"\n\n###\s+⚠️", f"{movement_section}\n\n### ⚠️", rendered, count=1)
        else:
            rendered = f"{rendered.rstrip()}{movement_section}"
    guarded = final_post_qa_guard(rendered, language=language)
    if not source_ids:
        guarded = "\n".join(
            line for line in guarded.splitlines()
            if not re.match(
                r"^\s*(?:[-*]\s*)?📌\s*\*\*(?:Fonte|Source|Fontes|Sources):\*\*",
                line,
                flags=re.IGNORECASE,
            )
        ).strip()
        guarded = re.sub(
            r"(?:\n\n---)?\n\n###\s+⚠️\s+\*\*(?:Notas finais|Final notes)\*\*\s*$",
            "",
            guarded,
            flags=re.IGNORECASE,
        ).strip()
    return guarded


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
        if re.search(r"\b(?:se|largo da se|catedral|carmo|chiado|baixa|rossio|correeiros|douradores)\b", basis):
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

    deduped: List[Dict[str, str]] = []
    seen: set[str] = set()
    for card in ordered:
        key = _normalize_planner_text(_planner_card_display_name(card) or card.get("name", ""))
        if key and key not in seen:
            deduped.append(card)
            seen.add(key)
    return deduped[:5]


def _card_source_id_for_plan_block(card: Dict[str, str]) -> str:
    """Return the VisitLisboa source id that materially supports a card."""
    if str(card.get("source_id") or "").strip() == "user_request":
        return ""
    return "visitlisboa_events" if _card_kind_for_plan_block(card) == "event" else "visitlisboa_places"


def _itinerary_block_title(
    title: str,
    card: Dict[str, str],
    *,
    index: int,
    historic_food_request: bool,
    event_food_request: bool = False,
    language: str,
) -> str:
    """Add a light schedule cue to planner fallback blocks when useful."""
    if not historic_food_request and not event_food_request:
        return title
    is_pt = language == "pt"
    title = _localize_planner_display_title(title, language)
    original_title_key = unicodedata.normalize("NFKD", str(title or ""))
    original_title_key = original_title_key.encode("ascii", "ignore").decode("ascii").lower()
    explicit_meal_kind = (
        "lunch" if re.search(r"^\s*(?:almo[cç]o|lunch)\s*:", original_title_key)
        else "dinner" if re.search(r"^\s*(?:jantar|dinner)\s*:", original_title_key)
        else ""
    )
    title = re.sub(
        r"^\s*(?:Almo[cç]o|Jantar|Lunch|Dinner)\s*:\s*",
        "",
        title,
        flags=re.IGNORECASE,
    ).strip()
    kind = _card_kind_for_plan_block(card)
    if event_food_request:
        if kind == "food":
            prefix = "Jantar tradicional" if is_pt else "Traditional dinner"
            return f"{prefix}: {title}"
        if kind == "event":
            event_time = _event_card_time_sort_key(card)
            prefix = "Evento cultural" if is_pt else "Cultural event"
            if event_time < 24 * 60:
                return f"{_minutes_to_time_label(event_time)} · {prefix}: {title}"
            return f"{prefix}: {title}"

    time_labels = ["09:30", "12:45", "15:00", "16:30", "18:00"]
    time_label = time_labels[min(max(index - 1, 0), len(time_labels) - 1)]
    if kind == "food" and explicit_meal_kind == "lunch":
        time_label = "12:45"
    elif kind == "food" and explicit_meal_kind == "dinner":
        time_label = "19:30"
    elif kind == "food" and index <= 2:
        time_label = "12:45"
    elif kind == "food" and index >= 4:
        time_label = "19:30"
    time_label = _adjust_time_label_for_card_hours(time_label, card)
    time_minutes = _time_label_to_minutes(time_label)
    if kind == "food" and explicit_meal_kind == "lunch":
        prefix = "Almoço tradicional" if is_pt else "Traditional lunch"
    elif kind == "food" and explicit_meal_kind == "dinner":
        prefix = "Jantar opcional" if is_pt else "Optional dinner"
    elif kind == "food" and index <= 2:
        if time_minutes is not None and time_minutes >= 18 * 60:
            prefix = "Jantar opcional" if is_pt else "Optional dinner"
        else:
            prefix = "Almoço tradicional" if is_pt else "Traditional lunch"
    elif kind == "food":
        prefix = "Jantar opcional" if is_pt else "Optional dinner"
    else:
        prefix = "Paragem histórica" if is_pt else "Historic stop"
    return f"{time_label} · {prefix}: {title}"


def _localize_planner_display_title(title: str, language: str) -> str:
    """Localize safe English place-type prefixes in PT itinerary headings."""
    cleaned = str(title or "").strip()
    parts = [part.strip() for part in cleaned.split("|")]
    if len(parts) == 2 and _normalize_planner_text(parts[0]) == _normalize_planner_text(parts[1]):
        cleaned = parts[0]
    if language != "pt" or not cleaned:
        return cleaned
    cleaned = re.sub(r"\bSé de Lisboa\s*\|\s*Lisbon Cathedral\b", "Sé de Lisboa", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bTower of Bel[eé]m\b", "Torre de Belém", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bBel[eé]m Tower\b", "Torre de Belém", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bMonument to the Discoveries\b", "Padrão dos Descobrimentos", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bJer[oó]nimos Monastery\b", "Mosteiro dos Jerónimos", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bNational Museum of Contempor[aâ]neo Art - Museu do Chiado\b", "Museu Nacional de Arte Contemporânea do Chiado", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bJewish Cultural Center\b", "Centro Cultural Judaico", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bCultural Center\b", "Centro Cultural", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bRestaurant\b", "Restaurante", cleaned, flags=re.IGNORECASE)

    replacements = (
        (r"^Chapel of\s+", "Capela de "),
        (r"^Church of\s+", "Igreja de "),
        (r"^Cathedral of\s+", "Catedral de "),
        (r"^Monastery of\s+", "Mosteiro de "),
        (r"^National Palace of\s+", "Palácio Nacional de "),
        (r"^Palace of\s+", "Palácio de "),
    )
    for pattern, replacement in replacements:
        if re.search(pattern, cleaned, flags=re.IGNORECASE):
            return re.sub(pattern, replacement, cleaned, count=1, flags=re.IGNORECASE)
    return cleaned


def _select_planner_cards_for_request(cards: List[Dict[str, str]], user_message: str) -> List[Dict[str, str]]:
    """Select the number and type of place cards needed by a fallback plan."""
    if not cards:
        return []
    normalized = _normalize_planner_text(user_message)
    usable_cards = [card for card in cards if not _planner_dict_card_is_closed(card)] or cards
    filtered_cards = [
        card for card in usable_cards
        if not _planner_card_is_low_fit_infrastructure(card, user_message)
        and not _planner_card_is_synthetic_plan_heading(card)
    ]
    if filtered_cards:
        usable_cards = filtered_cards
    target_area = _extract_compact_plan_area_anchor(user_message)
    if target_area and _query_describes_single_area_plan(user_message):
        original_usable_cards = list(usable_cards)
        area_cards = [
            card for card in usable_cards
            if _planner_card_matches_area(card, target_area)
        ]
        if area_cards:
            open_area_cultural_cards = [
                card for card in area_cards
                if _card_kind_for_plan_block(card) not in {"food", "event"}
                and not _planner_dict_card_is_closed(card)
            ]
            area_has_cultural = bool(open_area_cultural_cards)
            original_has_cultural = any(
                _card_kind_for_plan_block(card) not in {"food", "event"}
                and not _planner_dict_card_is_closed(card)
                for card in original_usable_cards
            )
            if not _planner_cards_satisfy_requested_counts(area_cards, user_message):
                supplemental_count_cards = [
                    card for card in original_usable_cards
                    if card not in area_cards
                    and any(
                        _planner_card_matches_requested_count_type(card, count_type)
                        for count_type, requested_count in _requested_plan_type_counts(user_message).items()
                        if count_type in {"museum", "monument", "viewpoint", "event"}
                        and requested_count > 0
                    )
                    and not _planner_dict_card_is_closed(card)
                ]
                supplemental_count_cards = sorted(
                    supplemental_count_cards,
                    key=lambda card: _score_card_for_requested_count_type(card, "total", user_message),
                    reverse=True,
                )
                usable_cards = _dedupe_planner_cards([
                    *area_cards,
                    *supplemental_count_cards,
                    *original_usable_cards,
                ])
            elif _query_requests_food_stop(user_message) and not area_has_cultural and original_has_cultural:
                nearby_cultural_cards = [
                    card for card in original_usable_cards
                    if _card_kind_for_plan_block(card) not in {"food", "event"}
                    and not _planner_dict_card_is_closed(card)
                ]
                usable_cards = _dedupe_planner_cards([*nearby_cultural_cards, *area_cards, *original_usable_cards])
            elif (
                _query_requests_food_stop(user_message)
                and len(open_area_cultural_cards) < 2
                and original_has_cultural
            ):
                supplemental_cultural_cards = sorted(
                    [
                        card for card in original_usable_cards
                        if card not in area_cards
                        and _card_kind_for_plan_block(card) not in {"food", "event"}
                        and not _planner_dict_card_is_closed(card)
                        and _compact_central_plan_far_area_penalty(card, user_message) < 100
                    ],
                    key=lambda card: _score_local_area_plan_card(card, normalized, user_message),
                    reverse=True,
                )
                supplemental_cultural_cards = [
                    card for card in supplemental_cultural_cards
                    if _score_local_area_plan_card(card, normalized, user_message) >= 0
                ]
                usable_cards = _dedupe_planner_cards([
                    *area_cards,
                    *supplemental_cultural_cards[: max(0, 2 - len(open_area_cultural_cards))],
                    *original_usable_cards,
                ])
            else:
                usable_cards = area_cards
    requested_count_cards = _select_cards_by_requested_counts(usable_cards, user_message)
    if requested_count_cards:
        return requested_count_cards
    single_cultural_stop_request = bool(
        re.search(r"\b(?:one|1|uma|um)\s+(?:cultural\s+)?(?:stop|paragem)\b", normalized)
        and not re.search(
            r"\b(?:para\s+comer|onde\s+comer|comer|food|meal|to\s+eat|for\s+food|"
            r"refeicao|refei[cç][aã]o|almoco|almo[cç]o|jantar|lunch|dinner)\b",
            normalized,
        )
    )
    if single_cultural_stop_request:
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
    if _is_event_planning_request(normalized):
        event_count = _requested_event_count(user_message)
        target_area = _extract_requested_plan_area(user_message)
        event_cards = [
            card for card in usable_cards
            if _card_kind_for_plan_block(card) == "event"
        ]
        if target_area:
            area_event_cards = [
                card for card in event_cards
                if _planner_card_matches_area(card, target_area)
            ]
            if area_event_cards:
                event_cards = area_event_cards
        event_cards = sorted(
            event_cards,
            key=lambda card: (
                _event_card_time_sort_key(card),
                -_score_event_plan_card(card),
            ),
        )
        selected = event_cards[:event_count]
        if re.search(r"\b(?:hist[oó]ric|monument|monumento|patrim[oó]nio|heritage|museu|museum)\b", normalized):
            cultural_cards = sorted(
                [
                    card for card in usable_cards
                    if _card_kind_for_plan_block(card) != "food"
                    and _card_kind_for_plan_block(card) != "event"
                ],
                key=_score_historic_plan_card,
                reverse=True,
            )
            if cultural_cards:
                selected = _dedupe_planner_cards([*selected, cultural_cards[0]])
        if _is_event_food_plan_request(normalized):
            food_cards = sorted(
                [
                    card for card in usable_cards
                    if _card_kind_for_plan_block(card) == "food"
                    and _food_card_matches_requested_context(card, user_message)
                ],
                key=_score_food_plan_card,
                reverse=True,
            )
            if food_cards:
                selected = _insert_food_card_into_event_plan(selected, food_cards[0])
        if selected:
            return _dedupe_planner_cards(selected)[:5]
    if _query_requests_architecture_theme(user_message):
        architecture_cards = sorted(
            (
                (score, card)
                for card in usable_cards
                if _card_kind_for_plan_block(card) not in {"food", "event"}
                for score in [_score_architecture_plan_card(card)]
                if score >= 55
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        if architecture_cards:
            return _dedupe_planner_cards([card for _score, card in architecture_cards])[:5]
    if _is_full_museum_day_request(user_message):
        museum_cards = sorted(
            (
                (score, card)
                for card in usable_cards
                if _card_kind_for_plan_block(card) == "museum"
                for score in [_score_historic_plan_card(card)]
                if score > 0
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        if museum_cards:
            selected = [card for _score, card in museum_cards]
            if re.search(r"\b(?:viewpoints?|view\s+points?|miradouros?|lookout|vista)\b", normalized):
                viewpoint_cards = sorted(
                    (
                        (score, card)
                        for card in usable_cards
                        for basis in [
                            _normalize_planner_text(
                                " ".join(
                                    str(card.get(key, ""))
                                    for key in ("name", "category", "address", "description", "url")
                                )
                            )
                        ]
                        if re.search(r"\b(?:viewpoints?|view\s+points?|miradouros?|lookout|vista|panoramic)\b", basis)
                        for score in [_score_cultural_stop_card(card, normalized)]
                        if score > 0
                    ),
                    key=lambda item: item[0],
                    reverse=True,
                )
                if viewpoint_cards:
                    selected = _dedupe_planner_cards([*selected[:2], viewpoint_cards[0][1], *selected[2:]])
            return _dedupe_planner_cards(selected)[:5]
    if (
        re.search(r"\b(?:hist[oó]ric|monument|monumento|patrim[oó]nio|heritage)\b", normalized)
        and re.search(r"\b(?:gastronom\w*|restaurants?|restaurantes?|food|comida|tradicional|almo[cç]o|jantar)\b", normalized)
    ):
        cultural_cards = sorted(
            (
                (score, card)
                for card in usable_cards
                for score in [_score_historic_plan_card(card)]
                if score >= 40
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        food_cards = sorted(
            (
                (score, card)
                for card in usable_cards
                if _food_card_matches_requested_context(card, user_message)
                for score in [_score_food_plan_card(card)]
                if score >= 60
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
        deduped = _dedupe_planner_cards(selected)
        deduped = _limit_cards_for_user_cardinality(deduped, user_message)
        if deduped:
            return deduped[:5]
    if _query_requests_food_stop(user_message):
        compact_food_plan = bool(
            re.search(
                r"\b(?:pequeno|curto|compacto|meio\s+dia|half\s+day|short|small|compact)\b",
                normalized,
            )
            and _requested_plan_required_component_count(user_message) <= 3
        )

        def local_fit_score(card: Dict[str, str]) -> int:
            return (
                _score_local_area_plan_card(card, normalized, user_message)
                + _requested_sequence_area_fit_score(card, user_message)
                - _compact_central_plan_far_area_penalty(card, user_message)
            )

        scored_cards = sorted(
            usable_cards,
            key=local_fit_score,
            reverse=True,
        )
        cultural_scored_cards = [
            (local_fit_score(card), card)
            for card in scored_cards
            if _card_kind_for_plan_block(card) not in {"food", "event"}
        ]
        if compact_food_plan:
            primary_cultural = [card for _score, card in cultural_scored_cards[:1]]
            additional_cultural = [
                card for score, card in cultural_scored_cards[1:]
                if (
                    score >= 25
                    or ((dist := _planner_card_distance_km(card)) is not None and dist <= 2.5)
                    or _planner_card_matches_area(card, _extract_compact_plan_area_anchor(user_message))
                )
            ]
            cultural_cards = [*primary_cultural, *additional_cultural]
        else:
            cultural_cards = [card for _score, card in cultural_scored_cards]
        food_cards = sorted(
            [
                card for card in scored_cards
                if _card_kind_for_plan_block(card) == "food"
                and _food_card_matches_requested_context(card, user_message)
            ],
            key=lambda card: _score_food_plan_card(card)
            + _score_food_card_for_meal_context(card, "lunch", user_message, cultural_cards[:2])
            + _requested_sequence_area_fit_score(card, user_message),
            reverse=True,
        )
        if cultural_cards and food_cards:
            if compact_food_plan:
                mixed_cards = _dedupe_planner_cards([
                    cultural_cards[0],
                    food_cards[0],
                    *cultural_cards[1:2],
                ])
            else:
                mixed_cards = _dedupe_planner_cards([
                    *cultural_cards[:2],
                    food_cards[0],
                    *cultural_cards[2:4],
                ])
            mixed_cards = _limit_cards_for_user_cardinality(mixed_cards, user_message)
            if mixed_cards:
                return mixed_cards[:5]
    if _query_describes_single_area_plan(user_message):
        scored_cards = sorted(
            usable_cards,
            key=lambda card: _score_local_area_plan_card(card, normalized, user_message),
            reverse=True,
        )
        if _query_requests_food_stop(user_message):
            cultural_cards = [
                card for card in scored_cards
                if _card_kind_for_plan_block(card) != "food"
            ]
            food_cards = sorted(
                [
                    card for card in scored_cards
                    if _card_kind_for_plan_block(card) == "food"
                    and _food_card_matches_requested_context(card, user_message)
                ],
                key=lambda card: _score_food_plan_card(card)
                + _score_food_card_for_meal_context(card, "lunch", user_message, cultural_cards[:1]),
                reverse=True,
            )
            if cultural_cards and food_cards:
                mixed_cards = _dedupe_planner_cards([
                    cultural_cards[0],
                    food_cards[0],
                    *cultural_cards[1:3],
                    *scored_cards,
                ])
                return _limit_cards_for_user_cardinality(mixed_cards, user_message)[:5]
        return _limit_cards_for_user_cardinality(scored_cards, user_message)[:5]
    return usable_cards[:5]


def _limit_food_cards_for_plan(cards: List[Dict[str, str]], *, max_food: int) -> List[Dict[str, str]]:
    """Limit restaurant cards in mixed itineraries while preserving order."""
    output: List[Dict[str, str]] = []
    food_count = 0
    for card in cards:
        if _card_kind_for_plan_block(card) == "food":
            food_count += 1
            if food_count > max_food:
                continue
        output.append(card)
    return output


_PLANNER_COUNT_WORDS = {
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
_PLANNER_COUNT_TOKEN_RE = (
    r"(?:\d{1,2}|um|uma|one|dois|duas|two|tres|three|quatro|four|"
    r"cinco|five|seis|six|sete|seven|oito|eight)"
)
_PLANNER_COUNT_UNIT_RE = re.compile(
    rf"\b(?P<count>{_PLANNER_COUNT_TOKEN_RE})\s+"
    r"(?:(?:numero|number)\s+of\s+|(?:numero|n)\s+de\s+)?"
    r"(?P<unit>museus?|museums?|galerias?|galleries|monumentos?|monuments?|"
    r"atracoes|attractions?|locais|lugares|sitios|sites|places|stops|paragens|pois?|"
    r"restaurantes?|restaurants?|cafes?|cafeterias?|pastelarias?|food\s+stops?|meal\s+stops?|lunch\s+stops?|dinner\s+stops?|"
    r"miradouros?|viewpoints?|view\s+points?|views?|vistas?|lookouts?|eventos?|events?)\b",
    re.IGNORECASE,
)


def _planner_count_to_int(value: str) -> int:
    """Convert a small user-facing count token into an integer."""
    token = _normalize_planner_text(value)
    if token.isdigit():
        return int(token)
    return _PLANNER_COUNT_WORDS.get(token, 0)


def _requested_plan_type_counts(user_message: str) -> Dict[str, int]:
    """Return explicit user cardinalities by itinerary stop type."""
    normalized = _normalize_planner_text(user_message)
    if not normalized:
        return {}

    counts: Dict[str, int] = {}
    for match in _PLANNER_COUNT_UNIT_RE.finditer(normalized):
        count = _planner_count_to_int(match.group("count"))
        if count <= 0:
            continue
        count = min(count, 8)
        unit = _normalize_planner_text(match.group("unit"))
        if re.search(r"\b(?:restaurantes?|restaurants?|cafes?|cafeterias?|pastelarias?|food|meal|lunch|dinner)\b", unit):
            count_type = "food"
        elif re.search(r"\b(?:museus?|museums?|galerias?|galleries)\b", unit):
            count_type = "museum"
        elif re.search(r"\b(?:monumentos?|monuments?)\b", unit):
            count_type = "monument"
        elif re.search(r"\b(?:miradouros?|viewpoints?|view point|views?|vistas?|lookouts?)\b", unit):
            count_type = "viewpoint"
        elif re.search(r"\b(?:eventos?|events?)\b", unit):
            count_type = "event"
        else:
            count_type = "total"
        counts[count_type] = max(counts.get(count_type, 0), count)
    return counts


def _requested_plan_total_stop_count(user_message: str) -> int:
    """Return the explicit total stop count implied by the request."""
    counts = _requested_plan_type_counts(user_message)
    if not counts:
        return 0
    specific_total = sum(value for key, value in counts.items() if key != "total")
    return max(counts.get("total", 0), specific_total)


def _requested_plan_required_component_count(user_message: str) -> int:
    """Return the minimum visible stops needed to satisfy explicit components."""
    counts = _requested_plan_type_counts(user_message)
    requested_total = _requested_plan_total_stop_count(user_message)
    if _query_requests_food_stop(user_message) and counts.get("food", 0) <= 0:
        requested_total += 1
    return min(8, requested_total)


def _selection_limit_for_requested_cardinality(user_message: str, default: int) -> int:
    """Expand card selection limits when the user asks for more stops."""
    requested_total = _requested_plan_required_component_count(user_message)
    if requested_total <= 0:
        return default
    return min(8, max(default, requested_total))


def _planner_card_named_content_basis(card: Dict[str, str]) -> str:
    """Return a matching basis that excludes broad VisitLisboa category labels."""
    return _normalize_planner_text(
        " ".join(str(card.get(key, "")) for key in ("name", "description", "features", "url", "details_url"))
    )


def _planner_card_matches_requested_count_type(card: Dict[str, str], count_type: str) -> bool:
    """Return whether a candidate card satisfies a requested stop type."""
    kind = _card_kind_for_plan_block(card)
    if count_type == "food":
        return kind == "food"
    if count_type == "event":
        return kind == "event"
    if count_type == "viewpoint":
        return _planner_card_is_viewpoint(card)
    if count_type == "museum":
        named_basis = _planner_card_named_content_basis(card)
        return bool(
            kind == "museum"
            and re.search(
                r"\b(?:museus?|museums?|galerias?|galleries|oceanario|aquarium|pavilhao\s+do\s+conhecimento)\b",
                named_basis,
            )
        )
    if count_type == "monument":
        named_basis = _planner_card_named_content_basis(card)
        return bool(
            kind == "museum"
            and re.search(
                r"\b(?:monumentos?|monuments?|mosteiro|monastery|torre|tower|padrao|cathedral|catedral|igreja|church|castelo|castle|palacio|palace)\b",
                named_basis,
            )
        )
    if count_type == "cultural":
        return kind not in {"food", "event"}
    return True


def _planner_cards_satisfy_requested_counts(cards: List[Dict[str, str]], user_message: str) -> bool:
    """Return whether a candidate pool can satisfy explicit typed counts."""
    counts = {
        key: value
        for key, value in _requested_plan_type_counts(user_message).items()
        if key in {"museum", "monument", "viewpoint", "event"} and value > 0
    }
    if not counts:
        return True
    return all(
        sum(
            1 for card in cards
            if _planner_card_matches_requested_count_type(card, count_type)
            and not _planner_dict_card_is_closed(card)
        ) >= requested_count
        for count_type, requested_count in counts.items()
    )


def _score_planner_card_opening_fit(card: Dict[str, str], user_message: str) -> int:
    """Score whether a card's opening hours fit the likely itinerary window."""
    if not str(card.get("hours") or "").strip():
        return 0
    normalized = _normalize_planner_text(user_message)
    if re.search(r"\b(?:noite|evening|jantar|dinner)\b", normalized):
        probe_minutes = (18 * 60, 19 * 60, 20 * 60)
    elif re.search(r"\b(?:tarde|afternoon)\b", normalized):
        probe_minutes = (14 * 60, 15 * 60, 16 * 60)
    elif re.search(r"\b(?:almoco|almocar|lunch)\b", normalized):
        probe_minutes = (10 * 60 + 30, 11 * 60 + 30, 12 * 60, 13 * 60 + 30)
    else:
        probe_minutes = (10 * 60, 11 * 60 + 30, 14 * 60 + 30)
    return 0 if any(_card_open_at_minutes(card, minutes) for minutes in probe_minutes) else -160


def _score_card_for_requested_count_type(card: Dict[str, str], count_type: str, user_message: str) -> int:
    """Score a candidate card for a requested stop type."""
    normalized = _normalize_planner_text(user_message)
    compact_far_penalty = _compact_central_plan_far_area_penalty(card, user_message)
    sequence_fit_score = _requested_sequence_area_fit_score(card, user_message)
    opening_fit_score = _score_planner_card_opening_fit(card, user_message)
    if count_type == "food":
        return _score_food_plan_card(card, user_message) + sequence_fit_score + opening_fit_score
    if count_type == "viewpoint":
        return (
            _score_cultural_stop_card(card, normalized)
            + (80 if _planner_card_is_viewpoint(card) else -40)
            + sequence_fit_score
            + opening_fit_score
            - compact_far_penalty
        )
    if count_type == "event":
        return _score_event_plan_card(card)
    if count_type in {"museum", "monument", "cultural"}:
        return (
            _score_local_area_plan_card(card, normalized, user_message)
            + _score_historic_plan_card(card)
            + sequence_fit_score
            + opening_fit_score
            - compact_far_penalty
        )
    if _query_requests_food_stop(user_message) and _card_kind_for_plan_block(card) == "food":
        return _score_food_plan_card(card, user_message) + sequence_fit_score + opening_fit_score
    if _card_kind_for_plan_block(card) == "food":
        return -30
    if _card_kind_for_plan_block(card) == "event" and not _is_event_planning_request(normalized):
        return -60
    return _score_local_area_plan_card(card, normalized, user_message) + sequence_fit_score - compact_far_penalty


def _compact_central_plan_far_area_penalty(card: Dict[str, str], user_message: str) -> int:
    """Penalize far-flung cards for short plans constrained to central Lisbon."""
    normalized_query = _normalize_planner_text(user_message)
    if not re.search(r"\b(?:curt[ao]|compact[ao]|meio\s+dia|half\s+day|short|small|tarde|afternoon)\b", normalized_query):
        return 0
    if not re.search(
        r"\b(?:chiado|cais\s+do\s+sodre|sodre|baixa|rossio|marques|marques\s+de\s+pombal|"
        r"saldanha|picoas|avenida|liberdade|carmo|santos)\b",
        normalized_query,
    ):
        return 0
    basis = _normalize_planner_text(
        " ".join(str(card.get(key, "")) for key in ("name", "category", "address", "description", "features", "url", "details_url"))
    )
    if re.search(
        r"\b(?:madre\s+de\s+deus|azulejo|madragoa|esperanca|janelas\s+verdes|ajuda|1249|1349|"
        r"marvila|belem|belem|benfica|fronteira|campo\s+pequeno|"
        r"parque\s+das\s+nacoes|oriente|expo|lumiar|ajuda)\b",
        basis,
    ):
        return 140
    return 0


def _requested_sequence_area_fit_score(card: Dict[str, str], user_message: str) -> int:
    """Score how well a card fits the user's explicit route anchors.

    The score is derived from the anchors in the current request, not from
    prompt-specific cases. It helps compact plans with start/pass-through/end
    constraints choose nearby evidence instead of high-profile but distant
    venues.
    """
    if not _query_has_explicit_anchor_sequence(user_message):
        return 0

    anchors = _requested_anchor_labels(user_message)
    if not anchors:
        return 0

    basis = _normalize_planner_text(
        " ".join(
            str(card.get(key, ""))
            for key in ("name", "category", "address", "description", "features", "url", "details_url")
        )
    )
    if not basis:
        return 0

    score = 0
    matched_anchor = False
    for anchor in anchors:
        if _planner_card_matches_area(card, anchor):
            matched_anchor = True
            score += 170
        elif _normalize_planner_text(anchor) in basis:
            matched_anchor = True
            score += 120

    route_mentions_central_axis = any(
        re.search(
            r"\b(?:marques|marques\s+de\s+pombal|avenida|liberdade|baixa|chiado|carmo|"
            r"cais\s+do\s+sodre|sodre|rossio)\b",
            _normalize_planner_text(anchor),
        )
        for anchor in anchors
    )
    if route_mentions_central_axis:
        if _PLANNER_CENTRAL_AREA_RE.search(basis) or re.search(r"\b(?:1200|1100|1150|1250)\b", basis):
            score += 55
        if not matched_anchor and re.search(
            r"\b(?:belem|belem|ajuda|benfica|lumiar|marvila|parque\s+das\s+nacoes|"
            r"oriente|olivais|campo\s+pequeno|madre\s+de\s+deus)\b",
            basis,
        ):
            score -= 120

    return score


def _select_cards_by_requested_counts(cards: List[Dict[str, str]], user_message: str) -> List[Dict[str, str]]:
    """Select evidence cards that satisfy explicit user cardinalities."""
    counts = _requested_plan_type_counts(user_message)
    if not counts:
        return []

    selected: List[Dict[str, str]] = []
    requested_types = ["museum", "monument", "viewpoint", "event", "food"]
    for count_type in requested_types:
        requested_count = counts.get(count_type, 0)
        if requested_count <= 0:
            continue
        pool = [
            card for card in cards
            if card not in selected and _planner_card_matches_requested_count_type(card, count_type)
        ]
        pool = sorted(
            pool,
            key=lambda card: _score_card_for_requested_count_type(card, count_type, user_message),
            reverse=True,
        )
        selected.extend(pool[:requested_count])

    requested_total = _requested_plan_total_stop_count(user_message)
    if counts.get("total", 0) and len(selected) < requested_total:
        filler_pool = [
            card for card in cards
            if card not in selected
            and _planner_card_matches_requested_count_type(card, "cultural")
        ]
        if _query_requests_food_stop(user_message):
            filler_pool = [
                card for card in cards
                if card not in selected
                and _card_kind_for_plan_block(card) != "event"
            ]
        filler_pool = sorted(
            filler_pool,
            key=lambda card: _score_card_for_requested_count_type(card, "total", user_message),
            reverse=True,
        )
        selected.extend(filler_pool[: max(0, requested_total - len(selected))])

    return _dedupe_planner_cards(selected)[: min(8, max(requested_total, len(selected)))]


def _ensure_selected_cards_satisfy_requested_counts(
    selected_cards: List[Dict[str, str]],
    all_cards: List[Dict[str, str]],
    user_message: str,
) -> List[Dict[str, str]]:
    """Add grounded cards when an earlier ordering step dropped typed counts."""
    counts = {
        key: value
        for key, value in _requested_plan_type_counts(user_message).items()
        if key in {"museum", "monument", "viewpoint", "event"} and value > 0
    }
    if not counts:
        return selected_cards

    result = list(selected_cards)
    for count_type, requested_count in counts.items():
        current_count = sum(
            1 for card in result
            if _planner_card_matches_requested_count_type(card, count_type)
        )
        if current_count >= requested_count:
            continue
        used_keys = {
            _normalize_planner_text(_planner_card_display_name(card) or card.get("name", ""))
            for card in result
        }
        candidates = sorted(
            [
                card for card in all_cards
                if _planner_card_matches_requested_count_type(card, count_type)
                and _normalize_planner_text(_planner_card_display_name(card) or card.get("name", "")) not in used_keys
                and not _planner_dict_card_is_closed(card)
            ],
            key=lambda card: _score_card_for_requested_count_type(card, count_type, user_message),
            reverse=True,
        )
        for card in candidates[: max(0, requested_count - current_count)]:
            result.append(card)
            used_keys.add(_normalize_planner_text(_planner_card_display_name(card) or card.get("name", "")))
    return _dedupe_planner_cards(result)


def _requested_food_option_limit(user_message: str) -> int:
    """Return the maximum number of food stops implied by the user's wording."""
    if not _query_requests_food_stop(user_message):
        return 0

    requested_food = _requested_plan_type_counts(user_message).get("food", 0)
    if requested_food > 0:
        return requested_food

    normalized = _normalize_planner_text(user_message)
    explicit_count = re.search(
        r"\b(?P<count>[2-5])\s+"
        r"(?:opcoes?\s+gastronomicas?|opções?\s+gastronómicas?|restaurantes?|restaurants?|food\s+stops?|meal\s+stops?)\b",
        normalized,
    )
    if explicit_count:
        return min(5, int(explicit_count.group("count")))

    asks_for_alternatives = bool(
        re.search(
            r"\b(?:varias|várias|algumas|alternativas?|opcoes|opções|options|several|multiple|lista|list)\b",
            normalized,
        )
    )
    if asks_for_alternatives and not re.search(r"\b(?:uma|um|one|1)\b", normalized):
        return 3

    return 1


def _limit_food_cards_for_user_request(cards: List[Dict[str, str]], user_message: str) -> List[Dict[str, str]]:
    """Apply user-stated food cardinality in mixed plans without harming food-only lists."""
    max_food = max(_requested_food_option_limit(user_message), len(_requested_meal_kinds(user_message)))
    if max_food <= 0:
        return cards
    if not any(_card_kind_for_plan_block(card) != "food" for card in cards):
        return cards
    return _limit_food_cards_for_plan(cards, max_food=max_food)


def _requested_cultural_stop_limit(user_message: str) -> int:
    """Return the maximum number of cultural stops explicitly requested by the user."""
    counts = _requested_plan_type_counts(user_message)
    if any(counts.get(key, 0) > 0 for key in ("museum", "monument", "viewpoint", "event")):
        return 0
    explicit_cultural = counts.get("total", 0) if _query_requests_cultural_stop(user_message) else 0
    if explicit_cultural > 0:
        return explicit_cultural

    normalized = _normalize_planner_text(user_message)
    if re.search(
        r"\b(?:um|uma|one|1)\s+"
        r"(?:monumento|monument|museu|museum|atracao|atração|attraction|"
        r"paragem\s+historica|paragem\s+histórica|historic\s+stop)\b",
        normalized,
    ):
        return 1
    return 0


def _limit_cultural_cards_for_user_request(cards: List[Dict[str, str]], user_message: str) -> List[Dict[str, str]]:
    """Apply explicit cultural-stop cardinality while preserving food and events."""
    max_cultural = _requested_cultural_stop_limit(user_message)
    if max_cultural <= 0:
        return cards

    output: List[Dict[str, str]] = []
    cultural_count = 0
    for card in cards:
        kind = _card_kind_for_plan_block(card)
        if kind not in {"food", "event"}:
            cultural_count += 1
            if cultural_count > max_cultural:
                continue
        output.append(card)
    return output


def _requested_viewpoint_stop_limit(user_message: str) -> int:
    """Return the maximum number of viewpoint stops explicitly requested by the user."""
    requested_viewpoints = _requested_plan_type_counts(user_message).get("viewpoint", 0)
    if requested_viewpoints > 0:
        return requested_viewpoints

    normalized = _normalize_planner_text(user_message)
    if re.search(r"\b(?:um|uma|one|1)\s+(?:miradouro|viewpoint|view\s+point|view|vista|lookout)\b", normalized):
        return 1
    return 0


def _planner_card_is_viewpoint(card: Dict[str, str]) -> bool:
    """Return whether a planner card is a viewpoint/lookout stop."""
    strong_basis = _normalize_planner_text(
        " ".join(str(card.get(key, "")) for key in ("name", "category", "url", "details_url"))
    )
    if re.search(r"\b(?:miradouro|viewpoint|view\s+point|lookout)\b", strong_basis):
        return True
    description_basis = _normalize_planner_text(
        " ".join(str(card.get(key, "")) for key in ("description", "features"))
    )
    return bool(
        _card_kind_for_plan_block(card) != "food"
        and re.search(r"\b(?:panoramic\s+view|city\s+view|vista\s+panoramica|vista\s+sobre\s+a\s+cidade)\b", description_basis)
    )


def _limit_viewpoint_cards_for_user_request(cards: List[Dict[str, str]], user_message: str) -> List[Dict[str, str]]:
    """Apply explicit viewpoint cardinality while preserving non-viewpoint stops."""
    max_viewpoints = _requested_viewpoint_stop_limit(user_message)
    if max_viewpoints <= 0:
        return cards

    output: List[Dict[str, str]] = []
    viewpoint_count = 0
    for card in cards:
        if _planner_card_is_viewpoint(card):
            viewpoint_count += 1
            if viewpoint_count > max_viewpoints:
                continue
        output.append(card)
    return output


def _limit_specific_count_cards_for_user_request(cards: List[Dict[str, str]], user_message: str) -> List[Dict[str, str]]:
    """Apply explicit typed cardinalities while preserving other requested stops."""
    counts = _requested_plan_type_counts(user_message)
    typed_limits = {
        key: value
        for key, value in counts.items()
        if key in {"museum", "monument", "event", "viewpoint"} and value > 0
    }
    if not typed_limits:
        return cards

    seen_counts = {key: 0 for key in typed_limits}
    output: List[Dict[str, str]] = []
    for card in cards:
        matched_limited_type = ""
        for count_type in typed_limits:
            if _planner_card_matches_requested_count_type(card, count_type):
                matched_limited_type = count_type
                break
        if matched_limited_type:
            seen_counts[matched_limited_type] += 1
            if seen_counts[matched_limited_type] > typed_limits[matched_limited_type]:
                continue
        output.append(card)
    return output


def _limit_visible_cards_for_requested_type_counts(
    cards: List[Dict[str, str]],
    user_message: str,
) -> List[Dict[str, str]]:
    """Apply explicit typed counts after sequence/end positioning has run."""
    if not cards:
        return cards
    counts = {
        key: value
        for key, value in _requested_plan_type_counts(user_message).items()
        if key in {"museum", "monument", "event", "viewpoint"} and value > 0
    }
    if not counts:
        return cards

    timed_labels = {
        _normalize_planner_text(label)
        for label, _time_label in _requested_anchor_time_constraints(user_message)
        if label
    }
    end_area = _extract_requested_plan_area(user_message)
    end_key = _normalize_planner_text(end_area)

    def is_protected(card: Dict[str, str]) -> bool:
        title_key = _normalize_planner_text(_planner_card_display_name(card) or card.get("name", ""))
        if end_key and (
            title_key == end_key
            or _planner_card_matches_area(card, end_area)
        ):
            return True
        return any(
            label and (
                label == title_key
                or _planner_card_matches_requested_label(card, label)
            )
            for label in timed_labels
        )

    keep_flags = [True] * len(cards)
    for count_type, limit in counts.items():
        matching_indices = [
            index
            for index, card in enumerate(cards)
            if keep_flags[index] and _planner_card_matches_requested_count_type(card, count_type)
        ]
        if len(matching_indices) <= limit:
            continue
        removable = [
            index for index in matching_indices
            if not is_protected(cards[index])
        ]
        for index in reversed(removable):
            if len([idx for idx in matching_indices if keep_flags[idx]]) <= limit:
                break
            keep_flags[index] = False

    return [card for card, keep in zip(cards, keep_flags) if keep]


def _limit_cards_for_user_cardinality(cards: List[Dict[str, str]], user_message: str) -> List[Dict[str, str]]:
    """Respect explicit user cardinality for mixed itinerary card selection."""
    limited_cards = _limit_food_cards_for_user_request(cards, user_message)
    limited_cards = _limit_viewpoint_cards_for_user_request(limited_cards, user_message)
    limited_cards = _limit_specific_count_cards_for_user_request(limited_cards, user_message)
    return _limit_cultural_cards_for_user_request(limited_cards, user_message)


def _dedupe_planner_cards(cards: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Deduplicate planner cards by title and stable place identifiers."""
    deduped: List[Dict[str, str]] = []
    seen: set[str] = set()
    for card in cards:
        title_key = _normalize_planner_text(_planner_card_display_name(card) or card.get("name", ""))
        address_key = _normalize_planner_text(str(card.get("address") or ""))
        url_key = _normalize_planner_text(str(card.get("website_url") or card.get("details_url") or card.get("url") or ""))
        keys = [title_key, url_key]
        if title_key and address_key:
            keys.append(f"{title_key}|{address_key}")
        keys = [key for key in keys if key and key not in {"lisboa", "lisbon"}]
        if keys and not any(key in seen for key in keys):
            deduped.append(card)
            seen.update(keys)
    return deduped


def _planner_card_matches_requested_label(card: Dict[str, str], requested_label: str) -> bool:
    """Return whether a place card represents a user-requested anchor label."""
    requested_key = _normalize_planner_text(requested_label)
    if not requested_key or (
        len(requested_key.split()) < 2
        and not _requested_anchor_fragment_is_specific(requested_label)
    ):
        return False
    title_key = _normalize_planner_text(_planner_card_display_name(card) or card.get("name", ""))
    if title_key and (title_key == requested_key or requested_key in title_key or title_key in requested_key):
        return True
    basis = _normalize_planner_text(
        " ".join(
            str(value or "")
            for value in (
                card.get("address"),
                card.get("description"),
                card.get("category"),
                card.get("url_label"),
            )
        )
    )
    return bool(basis and requested_key in basis)


def _requested_anchor_cards_in_order(
    requested_labels: List[str],
    existing_cards: List[Dict[str, str]],
    language: str,
) -> List[Dict[str, str]]:
    """Return evidence or conservative placeholder cards in the user's requested order."""
    ordered_cards: List[Dict[str, str]] = []
    used_keys: set[str] = set()
    placeholder_by_key = {
        _normalize_planner_text(card.get("name", "")): card
        for card in _requested_anchor_placeholder_cards(requested_labels, existing_cards, language)
    }

    for label in requested_labels:
        requested_key = _normalize_planner_text(label)
        if not requested_key or requested_key in used_keys:
            continue
        matched_card = next(
            (
                card for card in existing_cards
                if _planner_card_matches_requested_label(card, label)
                and _normalize_planner_text(_planner_card_display_name(card) or card.get("name", "")) not in used_keys
            ),
            None,
        )
        if matched_card is None:
            matched_card = placeholder_by_key.get(requested_key)
        elif (
            _planner_dict_card_is_closed(matched_card)
            and not re.search(
                r"\b(?:museu|museum|monumento|monument|restaurante|restaurant|"
                r"igreja|church|castelo|castle|palacio|palace|teatro|theatre|theater)\b",
                _normalize_planner_text(label),
            )
        ):
            matched_card = placeholder_by_key.get(requested_key) or matched_card
        if matched_card:
            ordered_cards.append(matched_card)
            used_keys.add(_normalize_planner_text(_planner_card_display_name(matched_card) or matched_card.get("name", "")))
    return ordered_cards


def _limit_visible_cards_to_requested_anchor_sequence(
    visible_cards: List[Dict[str, str]],
    candidate_cards: List[Dict[str, str]],
    requested_labels: List[str],
    user_message: str,
    language: str,
) -> List[Dict[str, str]]:
    """Keep explicit multi-anchor plans focused on the user's requested sequence."""
    requested_counts = _requested_plan_type_counts(user_message)
    total_only_count_matches_anchor_sequence = bool(
        set(requested_counts) == {"total"}
        and requested_counts.get("total", 0) <= len(requested_labels)
    )
    if (
        not _query_has_explicit_anchor_sequence(user_message)
        or len(requested_labels) < 2
        or (requested_counts and not total_only_count_matches_anchor_sequence)
    ):
        return visible_cards

    all_candidates = _dedupe_planner_cards([*visible_cards, *candidate_cards])
    limited: List[Dict[str, str]] = []

    def already_used(card: Dict[str, str]) -> bool:
        key = _normalize_planner_text(_planner_card_display_name(card) or card.get("name", ""))
        return bool(key) and any(
            key == _normalize_planner_text(_planner_card_display_name(existing) or existing.get("name", ""))
            for existing in limited
        )

    for label in requested_labels:
        matched_card = next(
            (
                card for card in all_candidates
                if _card_kind_for_plan_block(card) != "food"
                and not already_used(card)
                and _planner_card_matches_requested_label(card, label)
            ),
            None,
        )
        if matched_card is None:
            matched_card = next(
                (
                    card for card in all_candidates
                    if _card_kind_for_plan_block(card) != "food"
                    and not already_used(card)
                    and _planner_card_matches_area(card, label)
                ),
                None,
            )
        if matched_card is None:
            placeholder = _requested_anchor_cards_in_order([label], all_candidates, language)
            matched_card = placeholder[0] if placeholder else None
        if matched_card and not already_used(matched_card):
            limited.append(matched_card)

    if not limited:
        return visible_cards

    if _query_requests_food_stop(user_message):
        meal_kinds = _requested_meal_kinds(user_message) or ["lunch"]
        food_candidates = [
            card for card in all_candidates
            if _card_kind_for_plan_block(card) == "food"
            and _food_card_matches_requested_context(card, user_message)
        ]
        labelled_food_cards: List[Dict[str, str]] = []
        for meal_kind in meal_kinds[:2]:
            best_food = max(
                food_candidates,
                key=lambda card: _score_food_card_for_meal_context(card, meal_kind, user_message, limited),
                default=None,
            )
            if best_food:
                labelled_food_cards.append(
                    _label_food_card_for_meal(
                        best_food,
                        meal_kind,
                        language,
                        force_label=len(meal_kinds) > 1 or not _food_card_has_meal_label(best_food),
                    )
                )
        for food_card in labelled_food_cards:
            insert_at = min(max(1, len(limited) - 1), len(limited))
            if not already_used(food_card):
                limited.insert(insert_at, food_card)

    return _dedupe_planner_cards(limited)


def _food_card_has_meal_label(card: Dict[str, str]) -> bool:
    """Return whether a food card is already explicitly tied to lunch or dinner."""
    basis = _normalize_planner_text(
        " ".join(
            str(card.get(key, "") or "")
            for key in ("name", "title", "description", "purpose", "category")
        )
    )
    return bool(re.search(r"\b(?:almoco|almoço|almocar|almoçar|lunch|jantar|dinner)\b", basis))


def _label_food_card_for_meal(
    card: Dict[str, str],
    meal_kind: str,
    language: str,
    force_label: bool,
) -> Dict[str, str]:
    """Return a copy of a restaurant card labelled for a requested meal slot."""
    if not force_label:
        return card
    label = (
        "Jantar" if language == "pt" and meal_kind == "dinner"
        else "Almoço" if language == "pt"
        else "Dinner" if meal_kind == "dinner"
        else "Lunch"
    )
    display_name = _planner_card_display_name(card) or card.get("name", "")
    display_name = re.sub(
        r"^\s*(?:Almo[cç]o|Jantar|Lunch|Dinner)\s*:\s*",
        "",
        display_name,
        flags=re.IGNORECASE,
    ).strip()
    labelled = dict(card)
    labelled["name"] = f"{label}: {display_name}" if display_name else label
    return labelled


def _score_food_card_for_meal_context(
    card: Dict[str, str],
    meal_kind: str,
    user_message: str,
    planned_cards: List[Dict[str, str]],
) -> int:
    """Score a restaurant by how well it fits the route area for a meal slot."""
    card_basis = _normalize_planner_text(
        " ".join(str(card.get(key, "") or "") for key in ("name", "address", "description", "features", "distance", "hours"))
    )
    route_cards = [
        item for item in planned_cards
        if _card_kind_for_plan_block(item) != "food"
    ]
    plan_basis = _normalize_planner_text(
        " ".join(
            [user_message]
            + [
                " ".join(str(item.get(key, "") or "") for key in ("name", "address", "description", "category"))
                for item in route_cards
            ]
        )
    )
    if not card_basis:
        return 0

    score = 0
    traditional_requested = _query_requests_traditional_portuguese_food(plan_basis)
    if traditional_requested:
        if _food_text_has_traditional_portuguese_marker(card_basis):
            score += 220
        elif _food_text_has_non_traditional_cuisine_marker(card_basis):
            score -= 240

    if _planner_dict_card_is_closed(card):
        score -= 300
    card_is_belem = bool(_PLANNER_BELEM_AREA_RE.search(card_basis))
    card_is_central = bool(_PLANNER_CENTRAL_AREA_RE.search(card_basis))
    card_is_saldanha_axis = bool(
        re.search(r"\b(?:saldanha|avenidas novas|picoas|republica|marques|liberdade|duque de avila)\b", card_basis)
    )
    card_is_remote_east = bool(re.search(r"\b(?:parque das nacoes|oriente|olivais|expo)\b", card_basis))

    plan_has_belem = bool(_PLANNER_BELEM_AREA_RE.search(plan_basis))
    plan_has_central = bool(_PLANNER_CENTRAL_AREA_RE.search(plan_basis))
    plan_mentions_remote_east = bool(re.search(r"\b(?:parque das nacoes|oriente|olivais|expo)\b", plan_basis))
    hotel_or_end_near_saldanha = bool(re.search(r"\b(?:hotel|saldanha|avenidas novas|picoas)\b", plan_basis))
    preferred_area_key, _preferred_area_label, _preferred_blockers = _planner_local_area_profile(user_message)

    if meal_kind == "lunch" and _query_has_explicit_anchor_sequence(user_message):
        anchors = _requested_anchor_labels(user_message)
        anchor_basis = _normalize_planner_text(" ".join(anchors))
        if any(_planner_card_matches_area(card, anchor) for anchor in anchors):
            score += 180
        elif card_is_saldanha_axis and not re.search(
            r"\b(?:marques|saldanha|picoas|avenida\s+da\s+liberdade|liberdade)\b",
            anchor_basis,
        ):
            score -= 180
        elif not (card_is_belem or card_is_central):
            score -= 60

    if meal_kind == "lunch" and preferred_area_key in {"central_corridor", "alfama"}:
        if card_is_central:
            score += 130
        if card_is_belem:
            score -= 130
        if card_is_remote_east:
            score -= 100
        if card_is_saldanha_axis and not hotel_or_end_near_saldanha:
            score -= 80

    if meal_kind == "lunch" and preferred_area_key == "marques_de_pombal":
        if re.search(r"\b(?:marques\s+de\s+pombal|marques\s+pombal|avenida\s+da\s+liberdade|1250)\b", card_basis):
            score += 140
        if re.search(r"\bdistancia\b.*\bmarques\s+de\s+pombal\b|\bmarques\s+de\s+pombal\b", card_basis):
            score += 80
        distance_km = _planner_card_distance_km(card)
        if distance_km is not None:
            if distance_km <= 1.5:
                score += 180
            elif distance_km <= 2.5:
                score += 80
            else:
                score -= 190
        if card_is_belem or card_is_remote_east:
            score -= 130

    if meal_kind == "lunch" and plan_has_belem:
        if card_is_belem:
            score += 100
        elif card_is_central and plan_has_central:
            score += 55
        elif card_is_saldanha_axis and not hotel_or_end_near_saldanha:
            score -= 140
        elif card_is_central:
            score -= 20
        else:
            score -= 45
        if card_is_remote_east and not plan_mentions_remote_east:
            score -= 70
    elif meal_kind == "dinner" and hotel_or_end_near_saldanha:
        if re.search(r"\b(?:saldanha|avenidas novas|picoas|republica|duque de avila)\b", card_basis):
            score += 120
        elif card_is_central:
            score -= 20
        if card_is_belem:
            score -= 45
        if card_is_remote_east and not plan_mentions_remote_east:
            score -= 60
    elif card_is_remote_east and not plan_mentions_remote_east and (plan_has_belem or card_is_central):
        score -= 35

    return score


def _query_requests_traditional_portuguese_food(text: str) -> bool:
    """Return whether a planning request asks for Portuguese/traditional food."""
    normalized = _normalize_planner_text(text)
    return bool(
        re.search(
            r"\b(?:gastronomia\s+tradicional|cozinha\s+tradicional|comida\s+tradicional|"
            r"comida\s+portuguesa|cozinha\s+portuguesa|pratos?\s+portugues(?:es|as)|"
            r"almo[cç]o\s+tradicional|jantar\s+tradicional|refei[cç][aã]o\s+tradicional|"
            r"restaurante\s+tradicional|tradicional\s+portuguesa|portuguesa\s+tradicional|"
            r"tasca|taberna|fado|"
            r"traditional\s+portuguese|typical\s+portuguese|portuguese\s+cuisine|local\s+cuisine)\b",
            normalized,
        )
    )


def _food_text_has_traditional_portuguese_marker(text: str) -> bool:
    """Return whether a restaurant card carries Portuguese/traditional food signals."""
    normalized = _normalize_planner_text(text)
    return bool(
        re.search(
            r"\b(?:cozinha\s+tradicional\s+portuguesa|gastronomia\s+tradicional|"
            r"cozinha\s+portuguesa|comida\s+portuguesa|tradicional\s+portuguesa|"
            r"portuguesa\s+tradicional|traditional\s+portuguese|typical\s+portuguese|"
            r"portuguese\s+cuisine|portuguese\s+food|tasca|taberna|fado|petiscos)\b",
            normalized,
        )
    )


def _food_text_has_non_traditional_cuisine_marker(text: str) -> bool:
    """Return whether a restaurant card mainly signals a non-Portuguese cuisine."""
    normalized = _normalize_planner_text(text)
    return bool(
        re.search(
            r"\b(?:international|internacional|fusion|italian|italiana|pizzaria|pizza|"
            r"sushi|japanese|japonesa|asian|asiatica|burger|hamburger|brasserie|"
            r"french|francesa|mexican|mexicana|indian|indiana)\b",
            normalized,
        )
    )


def _insert_requested_food_stop_if_needed(
    selected_cards: List[Dict[str, str]],
    all_cards: List[Dict[str, str]],
    user_message: str,
    language: str,
) -> List[Dict[str, str]]:
    """Insert grounded or explicit meal stops when the user asked for food."""
    if not _query_requests_food_stop(user_message):
        return selected_cards

    requested_meals = _requested_meal_kinds(user_message)
    required_food_count = max(1, len(requested_meals))
    result = list(selected_cards)
    if len(requested_meals) > 1:
        result = [
            card for card in result
            if _card_kind_for_plan_block(card) != "food"
            or _food_card_has_meal_label(card)
        ]
    budget_requested = _query_requests_budget_food(user_message)
    existing_food_count = sum(1 for card in selected_cards if _card_kind_for_plan_block(card) == "food")
    if existing_food_count >= required_food_count and len(result) == len(selected_cards):
        if budget_requested and any(
            _card_kind_for_plan_block(card) == "food" and _food_card_budget_rank(card) != 0
            for card in result
        ):
            result = [card for card in result if _card_kind_for_plan_block(card) != "food"]
            existing_food_count = 0
        else:
            return selected_cards
    if existing_food_count >= required_food_count and len(result) == len(selected_cards):
        return selected_cards

    selected_keys = {
        _normalize_planner_text(_planner_card_display_name(card) or card.get("name", ""))
        for card in result
    }
    food_candidates = sorted(
        [
            card for card in all_cards
            if _card_kind_for_plan_block(card) == "food"
            and _food_card_matches_requested_context(card, user_message)
            and _normalize_planner_text(_planner_card_display_name(card) or card.get("name", "")) not in selected_keys
        ],
        key=lambda card: _score_food_plan_card(card, user_message)
        + _score_food_card_for_meal_context(card, requested_meals[0] if requested_meals else "lunch", user_message, result),
        reverse=True,
    )
    used_food_keys = set(selected_keys)
    for meal_kind in requested_meals:
        if sum(1 for card in result if _card_kind_for_plan_block(card) == "food") >= required_food_count:
            break
        meal_card: Dict[str, str] = {}
        available_candidates = sorted(
            [
                candidate for candidate in food_candidates
                if _normalize_planner_text(_planner_card_display_name(candidate) or candidate.get("name", "")) not in used_food_keys
            ],
            key=lambda candidate: _score_food_plan_card(candidate, user_message)
            + _score_food_card_for_meal_context(candidate, meal_kind, user_message, result),
            reverse=True,
        )
        for candidate in available_candidates:
            candidate_key = _normalize_planner_text(_planner_card_display_name(candidate) or candidate.get("name", ""))
            if candidate_key and candidate_key not in used_food_keys:
                meal_card = _label_food_card_for_meal(candidate, meal_kind, language, len(requested_meals) > 1)
                used_food_keys.add(candidate_key)
                break
        if not meal_card:
            meal_card = _requested_meal_placeholder_card(user_message, language, meal_kind=meal_kind)
        if not meal_card:
            continue
        insert_at = len(result) if meal_kind == "dinner" else min(1, len(result))
        result = _dedupe_planner_cards([*result[:insert_at], meal_card, *result[insert_at:]])
    return result


def _requested_cultural_placeholder_card(user_message: str, language: str) -> Dict[str, str]:
    """Build an explicit cultural slot when no concrete cultural stop was confirmed."""
    if not _query_requests_cultural_stop(user_message):
        return {}
    requested_area = _extract_requested_plan_area(user_message).strip()
    area = requested_area or ("Lisboa" if language == "pt" else "Lisbon")
    if language == "pt":
        return {
            "name": f"Paragem cultural em {area}",
            "category": "Paragem cultural a confirmar",
            "description": (
                "Paragem pedida pelo utilizador; os dados recolhidos não confirmaram "
                "um museu, monumento ou espaço cultural específico adequado nesta zona."
            ),
            "source_id": "user_request",
        }
    return {
        "name": f"Cultural stop in {area}",
        "category": "Cultural stop to confirm",
        "description": (
            "Stop requested by the user; the gathered data did not confirm a specific "
            "museum, monument, or cultural venue that fits this area."
        ),
        "source_id": "user_request",
    }


def _planner_card_is_requested_cultural_fit(card: Dict[str, str], user_message: str) -> bool:
    """Return whether a card can satisfy a requested cultural or museum stop."""
    normalized_query = _normalize_planner_text(user_message)
    kind = _card_kind_for_plan_block(card)
    if kind == "food":
        return False
    if kind in {"museum", "event"}:
        return True
    return _score_cultural_stop_card(card, normalized_query) > 0 or _score_historic_plan_card(card) > 0


def _insert_requested_cultural_stop_if_needed(
    selected_cards: List[Dict[str, str]],
    all_cards: List[Dict[str, str]],
    user_message: str,
    language: str,
) -> List[Dict[str, str]]:
    """Insert one cultural stop when the user asked for culture and selection omitted it."""
    if not _query_requests_cultural_stop(user_message) or any(
        _planner_card_is_requested_cultural_fit(card, user_message)
        for card in selected_cards
    ):
        return selected_cards

    selected_keys = {
        _normalize_planner_text(_planner_card_display_name(card) or card.get("name", ""))
        for card in selected_cards
    }
    target_area = _extract_requested_plan_area(user_message)
    candidates = [
        card for card in all_cards
        if _planner_card_is_requested_cultural_fit(card, user_message)
        and _normalize_planner_text(_planner_card_display_name(card) or card.get("name", "")) not in selected_keys
    ]
    if target_area:
        area_candidates = [
            card for card in candidates
            if _planner_card_matches_area(card, target_area)
        ]
        if area_candidates:
            candidates = area_candidates
    candidates = sorted(
        candidates,
        key=lambda card: (
            _planner_card_matches_area(card, target_area) if target_area else False,
            _score_local_area_plan_card(card, _normalize_planner_text(user_message), user_message),
            _score_historic_plan_card(card),
        ),
        reverse=True,
    )
    cultural_card = candidates[0] if candidates else _requested_cultural_placeholder_card(user_message, language)
    if not cultural_card:
        return selected_cards
    insert_at = 0
    return _dedupe_planner_cards([*selected_cards[:insert_at], cultural_card, *selected_cards[insert_at:]])


def _query_treats_start_anchor_as_origin_only(user_message: str) -> bool:
    """Return whether a requested start point is an origin, not a visit stop."""
    normalized_query = _normalize_planner_text(user_message)
    return bool(
        re.search(
            r"\b(?:comeca|comece|comecar|comecando|inicia|iniciar|iniciando|start|starting)\b",
            normalized_query,
        )
        and not re.search(
            r"\b(?:visitar|visit|inclui|include|including|pass(?:ar|e|em)\s+por|pass\s+through)\b",
            normalized_query,
        )
    )


def _position_requested_meal_cards_for_plan_window(
    cards: List[Dict[str, str]],
    user_message: str,
) -> List[Dict[str, str]]:
    """Place requested meal stops in a plausible itinerary slot without hard-coded venues."""
    if len(cards) < 2:
        return cards
    normalized_query = _normalize_planner_text(user_message)
    if not re.search(r"\b(?:almoco|almocar|lunch)\b", normalized_query):
        return cards
    result = list(cards)
    food_index = next(
        (
            index
            for index, card in enumerate(result)
            if _card_kind_for_plan_block(card) == "food"
        ),
        -1,
    )
    if food_index < 0:
        return cards
    end_area = _extract_requested_plan_area(user_message)
    end_key = _normalize_planner_text(end_area)
    protected_last = bool(
        end_key
        and result
        and (
            _normalize_planner_text(_planner_card_display_name(result[-1]) or result[-1].get("name", "")) == end_key
            or _planner_card_matches_area(result[-1], end_area)
        )
    )
    movable_end = len(result) - 1 if protected_last else len(result)
    first_non_food_index = next(
        (
            index
            for index, card in enumerate(result[:movable_end])
            if _card_kind_for_plan_block(card) != "food"
        ),
        -1,
    )
    if protected_last and end_area and _planner_card_matches_area(result[food_index], end_area):
        food_card = result.pop(food_index)
        insert_at = max(1, len(result) - 1)
        result.insert(insert_at, food_card)
        return result
    if first_non_food_index < 0 or food_index > first_non_food_index:
        return cards
    food_card = result.pop(food_index)
    if protected_last and end_area and _planner_card_matches_area(food_card, end_area):
        insert_at = max(1, len(result) - 1)
        result.insert(insert_at, food_card)
        return result
    insert_after = first_non_food_index
    if food_index < first_non_food_index:
        insert_after -= 1
    insert_at = min(insert_after + 1, len(result) - (1 if protected_last else 0))
    result.insert(max(0, insert_at), food_card)
    return result


def _position_compact_local_food_stop(
    cards: List[Dict[str, str]],
    user_message: str,
) -> List[Dict[str, str]]:
    """Place a nearby meal stop early in compact local plans.

    Compact half-day plans with an origin anchor should avoid sending the user
    past a nearby requested lunch stop and then backtracking. This only moves a
    grounded food card when it is clearly close to the requested compact area.
    """
    if len(cards) < 3 or not _query_requests_food_stop(user_message):
        return cards
    if _query_has_explicit_anchor_sequence(user_message):
        return cards
    target_area = _extract_compact_plan_area_anchor(user_message)
    if not target_area:
        return cards
    if not (
        _extract_requested_plan_duration_minutes(user_message)
        or re.search(
            r"\b(?:curt[ao]|compact[ao]|meio\s+dia|half\s+day|short|small|pequeno)\b",
            _normalize_planner_text(user_message),
        )
    ):
        return cards

    food_index = next(
        (index for index, card in enumerate(cards) if _card_kind_for_plan_block(card) == "food"),
        -1,
    )
    if food_index <= 1:
        return cards

    food_card = cards[food_index]
    distance_km = _planner_card_distance_km(food_card)
    area_key, _area_label, _blockers = _planner_local_area_profile(user_message)
    food_basis = _normalize_planner_text(
        " ".join(
            str(food_card.get(key, "") or "")
            for key in ("name", "address", "description", "features", "distance")
        )
    )
    near_requested_area = bool(
        _planner_card_matches_area(food_card, target_area)
        or (distance_km is not None and distance_km <= 1.5)
        or (
            area_key == "marques_de_pombal"
            and re.search(
                r"\b(?:marques|saldanha|picoas|avenida\s+da\s+liberdade|"
                r"tomas\s+ribeiro|tom[aá]s\s+ribeiro|1050|1067|1070|1250)\b",
                food_basis,
            )
        )
    )
    if not near_requested_area:
        return cards

    reordered = list(cards)
    food = reordered.pop(food_index)
    reordered.insert(1, food)
    return reordered


def _cluster_visible_cards_by_requested_route_areas(
    cards: List[Dict[str, str]],
    user_message: str,
) -> List[Dict[str, str]]:
    """Group visible stops around explicit route anchors without dropping evidence.

    This keeps flexible LLM-selected evidence, but avoids route backtracking when
    the user gives anchors such as "pass by X at 11:30" and "end in Y".
    """
    if len(cards) < 3:
        return cards

    end_area = _extract_requested_plan_area(user_message)
    end_key = _normalize_planner_text(end_area)
    origin = _extract_requested_plan_origin(user_message)
    origin_key = _normalize_planner_text(origin)
    origin_only = _query_treats_start_anchor_as_origin_only(user_message)

    anchor_areas: List[str] = []
    seen_areas: set[str] = set()

    def add_anchor_area(label: str) -> None:
        area_key = _normalize_planner_text(label)
        if not area_key or area_key in seen_areas:
            return
        if _planner_area_is_broad_city(area_key):
            return
        if end_key and area_key == end_key:
            return
        if origin_only and origin_key and area_key == origin_key:
            return
        if not any(_planner_card_matches_area(card, label) for card in cards):
            return
        seen_areas.add(area_key)
        anchor_areas.append(label)

    if origin and not origin_only:
        add_anchor_area(origin)
    for label, _time_label in _requested_anchor_time_constraints(user_message):
        add_anchor_area(label)
    for label in _requested_anchor_labels(user_message):
        add_anchor_area(label)

    if not anchor_areas and not end_key:
        return cards

    remaining = list(cards)

    def pop_matching(predicate: Callable[[Dict[str, str]], bool]) -> List[Dict[str, str]]:
        matched: List[Dict[str, str]] = []
        kept: List[Dict[str, str]] = []
        for card in remaining:
            if predicate(card):
                matched.append(card)
            else:
                kept.append(card)
        remaining[:] = kept
        return matched

    ordered: List[Dict[str, str]] = []
    for area in anchor_areas:
        ordered.extend(
            pop_matching(
                lambda card, area=area: _card_kind_for_plan_block(card) != "food"
                and _planner_card_matches_area(card, area)
            )
        )

    end_cards: List[Dict[str, str]] = []
    if end_key:
        end_cards = pop_matching(lambda card: _planner_card_matches_area(card, end_area))

    return _dedupe_planner_cards([*ordered, *remaining, *end_cards])


def _move_requested_origin_card_first(cards: List[Dict[str, str]], user_message: str) -> List[Dict[str, str]]:
    """Move the explicitly requested starting anchor to the first visible stop."""
    if not cards:
        return cards
    origin = _extract_requested_plan_origin(user_message)
    normalized_query = _normalize_planner_text(user_message)
    if not origin and re.search(r"\b(?:comeca|comece|comecar|comecando|inicia|iniciar|iniciando|start|starting)\b", normalized_query):
        requested_labels = _extract_requested_anchor_phrases(user_message)
        origin = requested_labels[0] if requested_labels else ""
    normalized_origin = _normalize_planner_text(origin)
    if not normalized_origin:
        return cards
    start_as_origin_only = _query_treats_start_anchor_as_origin_only(user_message)
    for index, card in enumerate(cards):
        title_key = _normalize_planner_text(_planner_card_display_name(card) or card.get("name", ""))
        if start_as_origin_only and title_key != normalized_origin:
            continue
        basis = _normalize_planner_text(
            " ".join(
                str(value or "")
                for value in (
                    _planner_card_display_name(card),
                    card.get("name"),
                    card.get("address"),
                    card.get("description"),
                )
            )
        )
        if normalized_origin and normalized_origin in basis:
            return [card, *cards[:index], *cards[index + 1:]]
    return cards


def _move_requested_end_card_last(cards: List[Dict[str, str]], user_message: str) -> List[Dict[str, str]]:
    """Move an explicitly requested ending anchor to the final visible stop."""
    if not cards:
        return cards
    normalized_query = _normalize_planner_text(user_message)
    if not re.search(r"\b(?:termina|termine|terminar|terminando|acaba|acabe|acabar|acabando|end|ending|finish|finishing)\b", normalized_query):
        return cards
    end_area = _extract_requested_plan_area(user_message)
    end_key = _normalize_planner_text(end_area)
    if not end_key:
        return cards
    for index in range(len(cards) - 1, -1, -1):
        card = cards[index]
        title_key = _normalize_planner_text(_planner_card_display_name(card) or card.get("name", ""))
        if title_key == end_key:
            return [*cards[:index], *cards[index + 1:], card]
    for index in range(len(cards) - 1, -1, -1):
        card = cards[index]
        if _planner_card_matches_area(card, end_area):
            return [*cards[:index], *cards[index + 1:], card]
    return cards


def _drop_origin_name_collision_cards(
    cards: List[Dict[str, str]],
    user_message: str,
) -> List[Dict[str, str]]:
    """Remove cards that only match a start point by name collision.

    A starting point such as a square, station, or avenue is an origin unless
    the user explicitly asks to visit/include it. If a VisitLisboa card merely
    contains the same name but points to another municipality, it should not
    become a stop in the plan.
    """
    if not cards:
        return cards
    normalized_query = _normalize_planner_text(user_message)
    origin = _extract_requested_plan_origin(user_message)
    normalized_origin = _normalize_planner_text(origin)
    if not normalized_origin:
        return cards
    start_as_origin_only = bool(
        re.search(r"\b(?:comeca|comece|comecar|comecando|inicia|iniciar|iniciando|start|starting)\b", normalized_query)
        and not re.search(
            r"\b(?:visitar|visit|inclui|include|including|pass(?:ar|e|em)\s+por|pass\s+through)\b",
            normalized_query,
        )
    )
    if not start_as_origin_only:
        return cards

    outlying_municipality_re = re.compile(
        r"\b(?:oeiras|cascais|sintra|setubal|setúbal|almada|seixal|montijo|"
        r"barreiro|loures|amadora|odivelas|mafra|vila\s+franca|alcochete|moita|palmela|sesimbra)\b"
    )
    filtered: List[Dict[str, str]] = []
    for card in cards:
        title_key = _normalize_planner_text(_planner_card_display_name(card) or card.get("name", ""))
        address_key = _normalize_planner_text(str(card.get("address") or ""))
        if (
            title_key
            and normalized_origin in title_key
            and title_key != normalized_origin
            and outlying_municipality_re.search(address_key)
            and not outlying_municipality_re.search(normalized_origin)
        ):
            continue
        filtered.append(card)
    return filtered


def _planner_card_is_synthetic_plan_heading(card: Dict[str, str]) -> bool:
    """Return whether a card is a generated itinerary heading rather than a real place."""
    name = _normalize_planner_text(_planner_card_display_name(card) or str(card.get("name") or ""))
    if not name:
        return False
    return bool(
        re.match(r"^(?:day|dia)\s+\d+\b", name)
        or re.search(
            r"\b(?:age of discoveries|historic center and urban layers|urban layers|"
            r"itinerario sugerido|roteiro sugerido|suggested itinerary)\b",
            name,
        )
    )


def _event_card_time_sort_key(card: Dict[str, str]) -> int:
    """Return a rough sortable minute-of-day for event cards."""
    text = _normalize_planner_text(" ".join(str(card.get(key, "")) for key in ("when", "hours", "description")))
    match = re.search(r"\b(?P<hour>\d{1,2})[:h](?P<minute>\d{2})\b", text)
    if not match:
        return 24 * 60
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    if hour > 23 or minute > 59:
        return 24 * 60
    return hour * 60 + minute


def _score_event_plan_card(card: Dict[str, str]) -> int:
    """Score an event card for event-planning fallbacks."""
    basis = _normalize_planner_text(
        " ".join(str(card.get(key, "")) for key in ("name", "category", "when", "duration", "address", "url"))
    )
    score = 0
    if "visitlisboa.com/en/events" in basis or "visitlisboa.com/pt-pt/eventos" in basis:
        score += 40
    if re.search(r"\b(?:evento|event|concerto|concert|festival|teatro|exposicao|exposição|musica|música|feira|dance|dança)\b", basis):
        score += 35
    if card.get("when"):
        score += 25
    if card.get("address") or card.get("venue"):
        score += 10
    if card.get("tickets_url"):
        score += 5
    return score


def _insert_food_card_into_event_plan(
    event_cards: List[Dict[str, str]],
    food_card: Dict[str, str],
) -> List[Dict[str, str]]:
    """Insert a food card at a natural point in an event-oriented plan."""
    if not event_cards:
        return [food_card]
    output = list(event_cards)
    first_evening_index = next(
        (
            index
            for index, card in enumerate(output)
            if _event_card_time_sort_key(card) >= 20 * 60
        ),
        -1,
    )
    if first_evening_index > 0:
        output.insert(first_evening_index, food_card)
    elif first_evening_index == 0:
        output.insert(0, food_card)
    else:
        output.append(food_card)
    return output


def _planner_dict_card_is_closed(card: Dict[str, str]) -> bool:
    """Return whether a raw planner card says the place is closed."""
    basis = _normalize_planner_text(
        " ".join(str(card.get(key, "")) for key in ("name", "description", "category", "hours"))
    )
    return bool(
        re.search(
            r"\b(?:hoje fechado|fechado hoje|today closed|closed today|"
            r"temporarily closed|temporary closure|temporarily unavailable|"
            r"encerrado temporariamente|temporariamente encerrado|"
            r"horario hoje fechado|hours today closed)\b",
            basis,
        )
    )


def _planner_card_is_low_fit_infrastructure(card: Dict[str, str], user_message: str) -> bool:
    """Filter infrastructure landmarks from museum/history plans unless requested."""
    basis = _normalize_planner_text(
        " ".join(str(card.get(key, "")) for key in ("name", "category", "address", "description"))
    )
    normalized_query = _normalize_planner_text(user_message)
    strict_historic_request = bool(
        re.search(r"\b(?:historic\w*|historico\w*|hist[oó]ric\w*|monument\w*|patrim[oó]ni\w*|heritage)\b", normalized_query)
    )
    if strict_historic_request and not re.search(r"\b(?:fado|music|m[uú]sica|show|concerto|concert)\b", normalized_query):
        is_experience_or_show = bool(
            re.search(r"\b(?:living\s+experience|immersive|imersiv\w*|experience|experi[eê]ncia|fado|music|m[uú]sica|show|concert|concerto)\b", basis)
        )
        has_strong_landmark_signal = bool(
            re.search(
                r"\b(?:igreja|church|cathedral|catedral|se de lisboa|sé de lisboa|capela|chapel|"
                r"torre|tower|padrao|padrão|descobrimentos|mosteiro|monastery|castelo|castle|"
                r"palacio|palácio|palace|convento|convent|memorial|forte|fortress)\b",
                basis,
            )
        )
        if is_experience_or_show and not has_strong_landmark_signal:
            return True
    is_weak_statuary = bool(
        re.search(r"\b(?:estatuaria|estatu[aá]ria|statue|sculpture|escultura)\b", basis)
        and not re.search(r"\b(?:monument|monumento|memorial|padrao|padrão|descobrimentos)\b", basis)
    )
    if is_weak_statuary:
        if re.search(r"\b(?:estatuaria|estatu[aá]ria|statue|sculpture|escultura|public art|arte publica)\b", normalized_query):
            return False
        return bool(
            re.search(
                r"\b(?:museu|museum|historic|historico|historia|monument|monumento|gastronom|roteiro|itinerario)\b",
                normalized_query,
            )
        )

    is_hotel_or_rooftop = bool(
        re.search(
            r"\b(?:hotel|hoteis|hot[eé]is|hostel|accommodation|alojamento|rooftop|terrace|terra[cç]o|"
            r"business\s+center|rooms?|quartos?|rent-a-car|lockers)\b",
            basis,
        )
    )
    if is_hotel_or_rooftop and re.search(
        r"\b(?:museu|museum|historic|historico|historia|monument|monumento|patrim[oó]nio|heritage|gastronom|roteiro|itinerario)\b",
        normalized_query,
    ):
        return not re.search(
            r"\b(?:rooftop|terrace|terra[cç]o|miradouro|viewpoint|view\s+point)\b",
            normalized_query,
        )

    if not re.search(r"\b(?:bridge|ponte|25\s+de\s+abril|aqueduct|aqueduto)\b", basis):
        return False

    if re.search(r"\b(?:bridge|ponte|aqueduct|aqueduto|engineering|engenharia|architecture|arquitetura)\b", normalized_query):
        return False

    return bool(
        re.search(
            r"\b(?:museu|museum|historic|historico|historia|monument|monumento|gastronom|senior|ritmo|subida|chuva)\b",
            normalized_query,
        )
    )


def _score_historic_plan_card(card: Dict[str, str]) -> int:
    """Score a card as a historic/cultural itinerary stop."""
    basis = _normalize_planner_text(
        " ".join(str(card.get(key, "")) for key in ("name", "category", "address", "description", "features", "hours"))
    )
    score = 0
    if re.search(r"\b(?:monument|monumento|museu|museum|igreja|church|cathedral|se de lisboa|sé de lisboa|torre|tower|padrao|padrão|mosteiro|monastery|castelo|castle|palacio|palácio|convento|carmo|ocean[aá]rio|aquarium|pavilh[aã]o\s+do\s+conhecimento)\b", basis):
        score += 50
    if re.search(r"\b(?:estatuaria|estatu[aá]ria|statue|sculpture|escultura)\b", basis):
        score -= 45
        if not re.search(r"\b(?:monument|monumento|memorial|padrao|padrão|descobrimentos)\b", basis):
            score -= 60
    if re.search(r"\b(?:bridge|ponte|25\s+de\s+abril|aqueduct|aqueduto)\b", basis):
        score -= 45
    if re.search(r"\b(?:lisboa|belem|belém|baixa|alfama|chiado|carmo|se |sé |brasilia|brasília)\b", basis):
        score += 18
    if re.search(r"\b(?:batalha|alcobaca|alcobaça|tomar|setubal|setúbal|cascais|sintra)\b", basis):
        score -= 80
    if _planner_dict_card_is_closed(card):
        score -= 60
    if card.get("address"):
        score += 5
    elif not (card.get("url") or card.get("details_url")):
        score -= 25
    return score


def _score_architecture_plan_card(card: Dict[str, str]) -> int:
    """Score a card for architecture-themed itineraries."""
    basis = _normalize_planner_text(
        " ".join(str(card.get(key, "")) for key in ("name", "category", "address", "description", "features", "hours"))
    )
    score = _score_historic_plan_card(card)
    if re.search(
        r"\b(?:arquitetura|arquitectura|architecture|architectural|design|urbanismo|urban|"
        r"edificio|edificio|building|fachada|facade|manuelino|manueline|azulejo|tile|"
        r"mosteiro|monastery|torre|tower|se de lisboa|cathedral|igreja|church|palacio|palace|"
        r"convento|convent|maat|mude|museu do design|pavilhao|pavilhao|aqueduto|aqueduct)\b",
        basis,
    ):
        score += 45
    if re.search(r"\b(?:fado|music|musica|música|living experience|restaurant|restaurante|bar)\b", basis):
        score -= 45
    if _planner_dict_card_is_closed(card):
        score -= 15
    return score


def _score_food_plan_card(card: Dict[str, str], user_message: str = "") -> int:
    """Score a card as a traditional-food itinerary stop."""
    basis = _normalize_planner_text(
        " ".join(str(card.get(key, "")) for key in ("name", "category", "address", "description", "features", "hours", "distance"))
    )
    score = 0
    if re.search(r"\b(?:restaurants?|restaurantes?|cozinha tradicional portuguesa|gastronomia|food|comida|bar|cafe|café)\b", basis):
        score += 50
    if re.search(r"\b(?:tradicional|portuguesa|typical portuguese|cozinha)\b", basis):
        score += 20
    if re.search(r"\b(?:italian|italiana|pizzaria|pizza|burger|hamburger|sushi|fusion|international|internacional)\b", basis):
        score -= 35
    if _query_requests_traditional_portuguese_food(user_message):
        if _food_text_has_traditional_portuguese_marker(basis):
            score += 120
        elif _food_text_has_non_traditional_cuisine_marker(basis):
            score -= 160
    if re.search(r"\b(?:lisboa|baixa|alfama|chiado|prata|douradores|correeiros|belem|bel[eé]m|brasilia|bras[ií]lia)\b", basis):
        score += 10
    if card.get("address") or card.get("url") or card.get("details_url") or card.get("website_url"):
        score += 12
    if str(card.get("source_id") or "").strip() in {"local_context", "user_request"}:
        score -= 35
    score += _score_food_budget_fit(card, user_message)
    if _card_open_at_minutes(card, 12 * 60 + 45):
        score += 30
    elif card.get("hours"):
        score -= 20
    if _planner_dict_card_is_closed(card):
        score -= 80
    return score


def _time_label_to_minutes(time_label: str) -> int | None:
    """Convert an HH:MM label into minutes since midnight."""
    match = re.match(r"^\s*(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*$", time_label or "")
    if not match:
        return None
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    if hour > 23 or minute > 59:
        return None
    return hour * 60 + minute


def _minutes_to_time_label(minutes: int) -> str:
    """Convert minutes since midnight to an HH:MM label."""
    bounded = minutes % (24 * 60)
    return f"{bounded // 60:02d}:{bounded % 60:02d}"


def _card_hour_intervals(card: Dict[str, str]) -> List[tuple[int, int]]:
    """Extract opening intervals from a planner card's hours field."""
    hours_text = str(card.get("hours") or "")
    intervals: List[tuple[int, int]] = []
    for match in re.finditer(
        r"(?P<start_h>\d{1,2}):(?P<start_m>\d{2})\s*[-–—]\s*(?P<end_h>\d{1,2}):(?P<end_m>\d{2})",
        hours_text,
    ):
        start = int(match.group("start_h")) * 60 + int(match.group("start_m"))
        end = int(match.group("end_h")) * 60 + int(match.group("end_m"))
        if end <= start:
            end += 24 * 60
        intervals.append((start, end))
    return intervals


def _card_open_at_minutes(card: Dict[str, str], minutes: int) -> bool:
    """Return whether a planner card is open at a given time."""
    intervals = _card_hour_intervals(card)
    if not intervals:
        return False
    for start, end in intervals:
        candidate = minutes
        if end > 24 * 60 and candidate < start:
            candidate += 24 * 60
        if start <= candidate < end:
            return True
    return False


def _adjust_time_label_for_card_hours(time_label: str, card: Dict[str, str]) -> str:
    """Shift a fallback itinerary time to the next evidenced opening interval."""
    minutes = _time_label_to_minutes(time_label)
    intervals = _card_hour_intervals(card)
    if minutes is None or not intervals or _card_open_at_minutes(card, minutes):
        return time_label
    future_starts = [start for start, _ in intervals if start > minutes]
    if future_starts:
        return _minutes_to_time_label(min(future_starts))
    feasible_previous_starts = [
        max(start, end - 60)
        for start, end in intervals
        if end <= minutes and end - start >= 30
    ]
    if feasible_previous_starts:
        return _minutes_to_time_label(max(feasible_previous_starts))
    return time_label


def _score_cultural_stop_card(card: Dict[str, str], normalized_query: str) -> int:
    """Score a place card for a single cultural-stop itinerary."""
    basis = _normalize_planner_text(
        " ".join(str(card.get(key, "")) for key in ("name", "category", "address", "description", "hours"))
    )
    if not basis:
        return 0

    score = 0
    if re.search(r"\b(?:museum|museu|gallery|galeria|monument|monumento|reservoir|reservatorio|patrimonial|heritage|ocean[aá]rio|aquarium|pavilh[aã]o\s+do\s+conhecimento)\b", basis):
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


def _score_local_area_plan_card(card: Dict[str, str], normalized_query: str, user_message: str) -> int:
    """Score a card for compact same-area plans with explicit stop types."""
    basis = _normalize_planner_text(
        " ".join(str(card.get(key, "")) for key in ("name", "category", "address", "description", "features", "hours", "distance"))
    )
    score = _score_cultural_stop_card(card, normalized_query)
    target_area = _extract_compact_plan_area_anchor(user_message)
    if target_area and _planner_card_matches_area(card, target_area):
        score += 35
        normalized_target_area = _normalize_planner_text(target_area)
        if (
            normalized_target_area in {"marques de pombal", "marques pombal", "marques"}
            and re.search(r"\b(?:marques\s+de\s+pombal|parque\s+eduardo\s+vii)\b", basis)
        ):
            score += 35
    area_key, _area_label, _blockers = _planner_local_area_profile(user_message)
    if area_key == "marques_de_pombal" and re.search(
        r"\b(?:saldanha|picoas|sao\s+sebastiao|avenida\s+da\s+liberdade|liberdade|"
        r"barata\s+salgueiro|salitre|parque\s+eduardo\s+vii|gulbenkian|berna|"
        r"restauradores|1250|1269|1050|1067)\b",
        basis,
    ):
        score += 45
    distance_km = _planner_card_distance_km(card)
    if distance_km is not None and _card_kind_for_plan_block(card) != "food":
        if distance_km <= 0.75:
            score += 55
        elif distance_km <= 1.5:
            score += 30
        elif distance_km > 3.0:
            score -= 70
    if re.search(r"\b(?:open\s+data|dados\s+abertos)\b", basis) and re.search(
        r"\b(?:roteiro|itinerario|itinerary|plano|plan|meio\s+dia|half\s+day)\b",
        normalized_query,
    ):
        score -= 40
    if _query_requests_food_stop(user_message) and _card_kind_for_plan_block(card) == "food":
        if _food_card_matches_requested_context(card, user_message):
            score += 95
        else:
            score -= 70
        if distance_km is not None:
            if distance_km <= 1.5:
                score += 90
            elif distance_km > 2.5:
                score -= 130
    if re.search(r"\b(?:miradouro|viewpoint|view\s+point|lookout|vista)\b", normalized_query):
        if re.search(r"\b(?:miradouro|viewpoint|view\s+point|lookout|vista|santa\s+luzia|portas\s+do\s+sol)\b", basis):
            score += 85
        else:
            score -= 8
    if re.search(r"\b(?:igreja|church|capela|chapel|catedral|cathedral|se)\b", normalized_query):
        if re.search(r"\b(?:igreja|church|capela|chapel|catedral|cathedral|se\s+de\s+lisboa)\b", basis):
            score += 85
        else:
            score -= 8
    if re.search(r"\b(?:museu|museum|cultural|culture|interior|indoor|exposicao|exhibition)\b", normalized_query):
        if re.search(
            r"\b(?:museu|museum|oceanario|aquarium|pavilhao\s+do\s+conhecimento|knowledge\s+pavilion|"
            r"galeria|gallery|exhibition|exposicao|cultural)\b",
            basis,
        ):
            score += 85
        elif _card_kind_for_plan_block(card) != "food":
            score -= 20
    if _query_requests_cafe_stop(user_message):
        if _card_kind_for_plan_block(card) == "food" and _food_card_matches_requested_context(card, user_message):
            score += 70
        elif _card_kind_for_plan_block(card) == "food":
            score -= 90
    if _planner_dict_card_is_closed(card):
        score -= 60
    return score


def _card_kind_for_plan_block(card: Dict[str, str]) -> str:
    """Infer a renderer block kind from a VisitLisboa-style card."""
    basis = _normalize_planner_text(
        " ".join(
            str(card.get(key, ""))
            for key in ("name", "category", "description", "features", "when", "duration", "url", "details_url")
        )
    )
    if _planner_card_is_event_result(card):
        return "event"
    if re.search(
        r"\b(?:restaurants?|restaurantes?|food|gastronomy|gastronomia|wine|bar|"
        r"almoco|almo[cç]o|lunch|jantar|dinner|meal|refeicao|refei[cç]ao|"
        r"cafe|coffee|pastelaria|pastry|pastel|pasteis|nata|custard|tarts?|"
        r"padaria|brunch|pequeno\s+almoco|breakfast)\b",
        basis,
    ):
        return "food"
    if re.search(r"\b(?:event|evento|events|eventos)\b", basis):
        return "event"
    if re.search(
        r"\b(?:museum|museu|gallery|galeria|monument|monumento|reservoir|reservatorio|reservatório|ocean[aá]rio|aquarium|pavilh[aã]o\s+do\s+conhecimento|"
        r"mosteiro|monastery|torre|tower|padrao|padrão|cathedral|catedral|sé de lisboa|se de lisboa|"
        r"chapel|capela|igreja|church|castelo|castle|palacio|palácio)\b",
        basis,
    ):
        return "museum"
    return "place"


def _planner_card_is_event_result(card: Dict[str, str]) -> bool:
    """Return whether a card is an actual event result rather than a fixed place.

    Event cards may appear in the same evidence context as place cards when a
    broad researcher call returns mixed evidence. Non-event itineraries should
    not absorb those cards just because they also have an address.
    """
    url_basis = _normalize_planner_text(
        " ".join(
            str(card.get(key, ""))
            for key in ("url", "details_url", "website_url", "source_url")
        )
    )
    if re.search(r"\bvisitlisboa com (?:en )?events\b|\beventos\b", url_basis):
        return True

    if str(card.get("when") or "").strip() or str(card.get("duration") or "").strip():
        return True

    category_basis = _normalize_planner_text(str(card.get("category") or ""))
    if re.search(
        r"\b(?:eventos?|events?|concertos?|concerts?|festivais|festivals|"
        r"teatro|theatre|theater|feiras?|fairs?)\b",
        category_basis,
    ):
        return True

    return False


def _requested_anchor_placeholder_cards(
    requested_labels: List[str],
    existing_cards: List[Dict[str, str]],
    language: str,
) -> List[Dict[str, str]]:
    """Create conservative cards for user-requested anchors missing from evidence."""
    existing_names = {
        _normalize_planner_text(_planner_card_display_name(card) or card.get("name", ""))
        for card in existing_cards
    }
    placeholders: List[Dict[str, str]] = []
    seen: set[str] = set()
    is_pt = language == "pt"
    for label in requested_labels:
        cleaned = _sanitize_planner_place_name(label)
        key = _normalize_planner_text(cleaned)
        if not key or key in existing_names or key in seen:
            continue
        has_specific_place_type = bool(re.search(
            r"\b(?:museu|museum|monumento|monument|mosteiro|monastery|torre|tower|"
            r"castelo|castle|palacio|palace|palacio|igreja|church|capela|chapel|"
            r"catedral|cathedral|restaurante|restaurant|pastelaria|cafe|café|"
            r"jardim|garden|miradouro|viewpoint|teatro|theatre|theater)\b",
            key,
        ))
        is_generic_route_anchor = bool(
            not has_specific_place_type
            and _requested_anchor_fragment_is_specific(cleaned)
            and key not in {"lisboa", "lisbon", "aml", "centro", "centro de lisboa"}
        )
        if not has_specific_place_type and not is_generic_route_anchor:
            continue
        seen.add(key)
        placeholders.append(
            {
                "name": cleaned,
                "category": "Ponto de passagem pedido" if is_generic_route_anchor and is_pt
                else "Requested waypoint" if is_generic_route_anchor
                else "Paragem pedida" if is_pt
                else "Requested stop",
                "description": (
                    "Âncora indicada pelo utilizador; usei-a para respeitar o percurso, sem a tratar como local oficial confirmado."
                    if is_generic_route_anchor and is_pt
                    else "User-provided route anchor; used to respect the requested path without treating it as a confirmed official place."
                    if is_generic_route_anchor
                    else
                    "Ponto indicado pelo utilizador; os detalhes oficiais não ficaram confirmados nos dados recolhidos."
                    if is_pt
                    else "User-requested stop; official details were not confirmed in the gathered evidence."
                ),
                "source_id": "user_request",
            }
        )
    return placeholders


def _requested_type_placeholder_cards(
    selected_cards: List[Dict[str, str]],
    user_message: str,
    language: str,
) -> List[Dict[str, str]]:
    """Build conservative placeholders for requested stop types missing from evidence."""
    normalized = _normalize_planner_text(user_message)
    area = _extract_requested_plan_area(user_message).strip()
    area_label = area or ("Lisboa" if language == "pt" else "Lisbon")
    is_pt = language == "pt"

    selected_basis = _normalize_planner_text(
        " ".join(
            " ".join(str(card.get(key, "")) for key in ("name", "category", "address", "description"))
            for card in selected_cards
        )
    )
    placeholders: List[Dict[str, str]] = []

    def append_placeholder(name_pt: str, name_en: str, category_pt: str, category_en: str, description_pt: str, description_en: str) -> None:
        placeholders.append(
            {
                "name": name_pt if is_pt else name_en,
                "category": category_pt if is_pt else category_en,
                "description": description_pt if is_pt else description_en,
                "source_id": "user_request",
            }
        )

    if (
        re.search(r"\b(?:miradouro|viewpoint|view\s+point|lookout|vista)\b", normalized)
        and not re.search(r"\b(?:miradouro|viewpoint|view\s+point|lookout|vista)\b", selected_basis)
    ):
        append_placeholder(
            f"Miradouro em {area_label}",
            f"Viewpoint in {area_label}",
            "Miradouro",
            "Viewpoint",
            (
                "Paragem pedida pelo utilizador; os dados recolhidos não confirmaram "
                "um miradouro específico nesta zona."
            ),
            (
                "Stop requested by the user; the gathered data did not confirm "
                "a specific viewpoint in this area."
            ),
        )

    if (
        re.search(r"\b(?:igreja|church|capela|chapel|catedral|cathedral|se\s+de\s+lisboa)\b", normalized)
        and not re.search(r"\b(?:igreja|church|capela|chapel|catedral|cathedral|se\s+de\s+lisboa)\b", selected_basis)
    ):
        append_placeholder(
            f"Igreja em {area_label}",
            f"Church in {area_label}",
            "Igreja / monumento",
            "Church / monument",
            (
                "Paragem pedida pelo utilizador; os dados recolhidos não confirmaram "
                "uma igreja específica com detalhes suficientes nesta zona."
            ),
            (
                "Stop requested by the user; the gathered data did not confirm "
                "a specific church with enough details in this area."
            ),
        )

    return placeholders


def _requested_meal_kinds(user_message: str) -> List[str]:
    """Return the distinct meal kinds explicitly requested by the user."""
    normalized = _normalize_planner_text(user_message)
    kinds: List[str] = []
    if re.search(r"\b(?:almoco|almoço|almocar|almoçar|lunch)\b", normalized):
        kinds.append("lunch")
    if re.search(r"\b(?:jantar|dinner)\b", normalized):
        kinds.append("dinner")
    return kinds or ["dinner" if re.search(r"\b(?:jantar|dinner)\b", normalized) else "lunch"]


def _requested_meal_placeholder_card(
    user_message: str,
    language: str,
    meal_kind: str = "",
) -> Dict[str, str]:
    """Build an explicit meal slot when the user requested one but no restaurant was confirmed."""
    normalized = _normalize_planner_text(user_message)
    if not _query_requests_food_stop(user_message):
        return {}

    is_pt = language == "pt"
    dinner = meal_kind == "dinner" or (
        not meal_kind and bool(re.search(r"\b(?:jantar|dinner)\b", normalized))
    )
    cafe_stop = _query_requests_cafe_stop(user_message)
    requested_area = _extract_requested_plan_area(user_message).strip()
    excluded_areas = set(_extract_excluded_plan_areas(user_message))
    if requested_area:
        area_pt = requested_area
        area_en = requested_area
    elif re.search(r"\b(?:baixa|chiado|rossio|carmo)\b", normalized):
        area_pt = "Baixa"
        area_en = "Baixa"
    elif re.search(r"\b(?:belem|bel[eé]m)\b", normalized):
        area_pt = "Belém"
        area_en = "Belém"
    else:
        area_pt = "Lisboa"
        area_en = "Lisbon"
    if _normalize_planner_text(area_pt) in excluded_areas:
        area_pt = "Lisboa"
        area_en = "Lisbon"

    if cafe_stop:
        if _query_requests_custard_tart_stop(user_message):
            if is_pt:
                return {
                    "name": f"Pastel de nata em {area_pt}",
                    "category": "Pausa gastronómica",
                    "description": (
                        "Paragem pedida pelo utilizador; os dados recolhidos não confirmaram "
                        "uma pastelaria específica para pastel de nata nessa zona."
                    ),
                    "source_id": "user_request",
                }
            return {
                "name": f"Custard tart stop in {area_en}",
                "category": "Food break",
                "description": (
                    "Stop requested by the user; the gathered data did not confirm a specific "
                    "custard tart or pastry shop in this area."
                ),
                "source_id": "user_request",
            }
        if is_pt:
            return {
                "name": f"Café tradicional em {area_pt}",
                "category": "Pausa gastronómica",
                "description": (
                    "Pausa pedida pelo utilizador; os dados recolhidos não confirmaram "
                    "um café ou pastelaria específico e adequado ao horário nessa zona."
                ),
                "source_id": "user_request",
            }
        return {
            "name": f"Traditional cafe in {area_en}",
            "category": "Food break",
            "description": (
                "Stop requested by the user; the gathered data did not confirm a specific "
                "cafe or pastry shop that fits the requested time in this area."
            ),
            "source_id": "user_request",
        }

    if is_pt:
        meal_label = "Jantar" if dinner else "Almoço"
        return {
            "name": f"{meal_label} em {area_pt}",
            "category": "Refeição",
            "address": f"{area_pt}, Lisboa" if area_pt != "Lisboa" else "Lisboa",
            "description": (
                "Pausa gastronómica pedida pelo utilizador; nenhum restaurante específico "
                "ficou confirmado nos dados recolhidos."
            ),
            "source_id": "user_request",
        }

    meal_label = "Dinner" if dinner else "Lunch"
    return {
        "name": f"{meal_label} in {area_en}",
        "category": "Meal stop",
        "address": f"{area_en}, Lisbon" if area_en != "Lisbon" else "Lisbon",
        "description": (
            "Meal stop requested by the user; no specific restaurant was confirmed "
            "in the gathered data."
        ),
        "source_id": "user_request",
    }


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


def _append_card_link_details(details: List[str], card: Dict[str, str], *, language: str = "en") -> None:
    """Append website and VisitLisboa details links without collapsing them into one field."""
    website_url = str(card.get("website_url") or "").strip()
    tickets_url = str(card.get("tickets_url") or "").strip()
    details_url = str(card.get("details_url") or "").strip()
    legacy_url = str(card.get("url") or "").strip()

    if not website_url and legacy_url and "visitlisboa.com" not in legacy_url.lower():
        website_url = legacy_url
    if not details_url and legacy_url and "visitlisboa.com" in legacy_url.lower():
        details_url = legacy_url

    if website_url:
        website_label = str(card.get("website_label") or "").strip()
        if not website_label or website_label.lower() == "visitlisboa":
            website_label = "Website oficial" if language == "pt" else "Official website"
        details.append(f"Website: [{website_label}]({website_url})")

    if tickets_url:
        tickets_label = str(card.get("tickets_label") or "").strip() or ("Comprar bilhetes" if language == "pt" else "Tickets")
        details.append(f"Tickets: [{tickets_label}]({tickets_url})")

    if details_url and details_url != website_url:
        details_label = str(card.get("details_label") or "").strip() or "VisitLisboa"
        if _normalize_planner_text(details_label) in {"more details", "mais detalhes"}:
            details_label = "VisitLisboa"
        details_field = "Mais detalhes" if language == "pt" else "More details"
        details.append(f"{details_field}: [{details_label}]({details_url})")


def _planner_details_match_key(value: str) -> str:
    """Return a stable key for matching visible cards to VisitLisboa evidence."""
    normalized = _normalize_planner_text(value)
    normalized = re.sub(
        r"\b(?:restaurant|restaurants|restaurante|restaurantes|"
        r"coffee shop|coffee|cafe|cafes|cafetaria|bar|pastelaria)\b",
        " ",
        normalized,
    )
    normalized = re.sub(
        r"\b(?:paragem historica|historic stop|almoco tradicional|traditional lunch|"
        r"almoco|lunch|jantar opcional|optional dinner|jantar|dinner|"
        r"pausa gastronomica|food break)\b",
        " ",
        normalized,
    )
    return re.sub(r"\s+", " ", normalized).strip()


def _planner_visible_card_title_key(line: str) -> str:
    """Extract the comparable title key from a rendered itinerary card line."""
    match = re.match(r"^\s*[-*]\s+\*\*(?P<title>[^*\n]{3,180})\*\*", line or "")
    if not match:
        return ""
    title = re.sub(r"\b\d{1,2}:\d{2}\b", " ", match.group("title"))
    title = re.sub(r"^[^\w]+", " ", title)
    if ":" in title:
        title = title.rsplit(":", 1)[1]
    return _planner_details_match_key(title)


def _repair_planner_visitlisboa_details_links(
    response: str,
    *,
    places_data: str,
    language: str,
) -> str:
    """Restore missing VisitLisboa details links for visible planner cards.

    QA/final repair may occasionally keep a restaurant/place card but drop the
    `Mais detalhes`/`More details` row. When the selected venue is present in
    the Researcher evidence with a VisitLisboa URL, restore that row without
    inventing any new source fields.
    """
    if not response or not places_data:
        return response

    cards = _extract_visitlisboa_place_cards(places_data, max_items=32, language=language)
    details_by_key: Dict[str, tuple[str, str]] = {}
    for card in cards:
        details_url = str(card.get("details_url") or "").strip()
        if not details_url:
            legacy_url = str(card.get("url") or "").strip()
            if "visitlisboa.com" in legacy_url.lower():
                details_url = legacy_url
        if "visitlisboa.com" not in details_url.lower():
            continue
        for raw_name in {
            str(card.get("name") or "").strip(),
            _planner_card_display_name(card),
            _localize_planner_display_title(_planner_card_display_name(card), language),
        }:
            key = _planner_details_match_key(raw_name)
            if len(key) >= 4:
                details_by_key.setdefault(key, (details_url, str(card.get("details_label") or "VisitLisboa")))

    if not details_by_key:
        return response

    lines = response.splitlines()
    output = list(lines)
    inserted_offset = 0
    for index, line in enumerate(lines):
        title_key = _planner_visible_card_title_key(line)
        if not title_key:
            continue

        matched_details: tuple[str, str] | None = None
        for evidence_key, details in details_by_key.items():
            if evidence_key in title_key or title_key in evidence_key:
                matched_details = details
                break
        if not matched_details:
            continue

        block_end = index + 1
        while block_end < len(lines):
            candidate = lines[block_end]
            stripped = candidate.strip()
            if re.match(r"^\s*[-*]\s+\*\*[^*\n]{3,180}\*\*", candidate) or stripped.startswith("### "):
                break
            if _PLANNER_SOURCE_LINE_RE.match(stripped):
                break
            block_end += 1

        block_text = "\n".join(lines[index:block_end])
        normalized_block = _normalize_planner_text(block_text)
        if (
            "ponto indicado pelo utilizador" in normalized_block
            or "ancora indicada pelo utilizador" in normalized_block
            or "user provided point" in normalized_block
            or "user provided route anchor" in normalized_block
            or "paragem pedida" in normalized_block
            or "ponto de passagem pedido" in normalized_block
            or "requested stop" in normalized_block
            or "requested waypoint" in normalized_block
        ):
            continue
        if re.search(r"\b(?:Mais detalhes|More details)\b|visitlisboa\.com/(?:en|pt-pt)/places", block_text, re.IGNORECASE):
            continue

        insert_at = block_end + inserted_offset
        while insert_at > index + inserted_offset + 1 and not output[insert_at - 1].strip():
            insert_at -= 1

        details_url, details_label = matched_details
        if _normalize_planner_text(details_label) in {"more details", "mais detalhes"}:
            details_label = "VisitLisboa"
        field_label = "Mais detalhes" if language == "pt" else "More details"
        output.insert(insert_at, f"    - 🔗 **{field_label}:** [{details_label}]({details_url})")
        inserted_offset += 1

    return "\n".join(output)


def _planner_text_is_internal_context_marker(text: str) -> bool:
    """Return whether text is orchestration context that must not be shown."""
    normalized = _normalize_planner_text(text)
    return any(
        marker in normalized
        for marker in (
            "previous final plan excerpt",
            "previous referenced places",
            "previous planning request",
            "continuity requirement",
            "current follow up request",
            "current follow-up request",
        )
    )


def _card_details_for_plan_block(card: Dict[str, str], *, language: str = "en") -> List[str]:
    """Convert a place card into semantic planner detail fields."""
    details: List[str] = []
    if card.get("description"):
        description = _planner_card_description_for_language(card["description"], language)
        if description and not _planner_text_is_internal_context_marker(description):
            details.append(f"Description: {description}")
    if card.get("category"):
        details.append(f"Category: {card['category']}")
    if card.get("when"):
        details.append(f"When: {card['when']}")
    if card.get("duration"):
        details.append(f"Duration: {card['duration']}")
    if card.get("venue"):
        details.append(f"Venue: {card['venue']}")
    if card.get("address"):
        details.append(f"Address: {card['address']}")
    if card.get("hours"):
        details.append(f"Hours: {card['hours']}")
    if card.get("price"):
        details.append(f"Price: {card['price']}")
    if card.get("features"):
        details.append(f"Features: {card['features']}")
    if card.get("rating"):
        details.append(f"Rating: {card['rating']}")
    if card.get("phone"):
        details.append(f"Phone: {card['phone']}")
    if card.get("email"):
        details.append(f"Email: {card['email']}")
    _append_card_link_details(details, card, language=language)
    return details


def _card_details_for_itinerary_block(card: Dict[str, str], *, language: str = "en") -> List[str]:
    """Convert a place card into concise itinerary fields rather than a raw card."""
    details: List[str] = []
    if card.get("description"):
        description = _planner_card_description_for_language(card["description"], language)
        if description and not _planner_text_is_internal_context_marker(description):
            details.append(f"Description: {description}")
    if card.get("when"):
        details.append(f"When: {card['when']}")
    if card.get("duration"):
        details.append(f"Duration: {card['duration']}")
    if card.get("venue"):
        details.append(f"Venue: {card['venue']}")
    if card.get("address"):
        details.append(f"Address: {card['address']}")
    if card.get("hours"):
        details.append(f"Hours: {card['hours']}")
    if card.get("price"):
        details.append(f"Price: {card['price']}")
    if card.get("features"):
        details.append(f"Features: {card['features']}")
    _append_card_link_details(details, card, language=language)
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

    _replace_meal_blocks_with_contextual_evidence(draft, evidence, user_message)

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


def _replace_meal_blocks_with_contextual_evidence(
    draft: PlanDraft,
    evidence: EvidenceBundle,
    user_message: str,
) -> None:
    """Replace weak meal selections with better area-matched evidence cards."""
    if not _query_requests_food_stop(user_message):
        return
    all_food_candidates = [card for card in evidence.cards if card.kind == "food"]
    food_candidates = [card for card in all_food_candidates if not _evidence_card_is_closed(card)] or all_food_candidates
    if not food_candidates:
        return

    planned_card_dicts = [
        _evidence_card_to_planner_card(card)
        for card in evidence.cards
        if card.kind != "food"
    ]
    is_pt = bool(re.search(r"\b(?:almo[cç]o|jantar|roteiro|amanh[aã]|hotel)\b", user_message, flags=re.IGNORECASE))
    for block in draft.blocks:
        block_key = _normalize_planner_text(" ".join([block.title, block.kind, block.purpose]))
        meal_kind = ""
        if re.search(r"\b(?:jantar|dinner)\b", block_key):
            meal_kind = "dinner"
        elif re.search(r"\b(?:almoco|almocar|lunch)\b", block_key):
            meal_kind = "lunch"
        elif block.kind == "food":
            meal_kind = "dinner" if re.search(r"\b(?:jantar|dinner)\b", _normalize_planner_text(user_message)) else "lunch"
        if not meal_kind:
            continue

        current_card = _match_evidence_card_for_block(block.title, food_candidates)
        current_score = _score_evidence_food_card(current_card, meal_kind, user_message, planned_card_dicts)
        best_card = max(
            food_candidates,
            key=lambda card: _score_evidence_food_card(card, meal_kind, user_message, planned_card_dicts),
        )
        best_score = _score_evidence_food_card(best_card, meal_kind, user_message, planned_card_dicts)
        if best_score <= current_score + 20:
            continue

        label = (
            "Jantar" if is_pt and meal_kind == "dinner"
            else "Almoço" if is_pt
            else "Dinner" if meal_kind == "dinner"
            else "Lunch"
        )
        block.title = f"{label}: {best_card.title}"
        block.kind = "food"
        block.details = _details_from_evidence_card(best_card, language="pt" if is_pt else "en")
        block.source_ids = list(dict.fromkeys([*block.source_ids, *best_card.source_ids]))


def _repair_meal_locality_in_response(
    response: str,
    *,
    user_message: str,
    places_data: str,
    language: str,
) -> str:
    """Replace meal blocks that drift away from the planned route area.

    This is a post-QA safety net for cases where a repair pass rewrites a
    structured plan as Markdown and selects a restaurant away from the itinerary
    corridor despite better restaurant evidence being available.
    """
    if not response or not places_data or not _query_requests_food_stop(user_message):
        return response or ""

    evidence = build_evidence_bundle(places_data=places_data)
    all_food_candidates = [card for card in evidence.cards if card.kind == "food"]
    food_candidates = [card for card in all_food_candidates if not _evidence_card_is_closed(card)] or all_food_candidates
    if not food_candidates:
        return response

    planned_card_dicts = [
        _evidence_card_to_planner_card(card)
        for card in evidence.cards
        if card.kind != "food"
    ]
    scoring_context = "\n".join([user_message, response])
    is_pt = (language or "").lower().startswith("pt")

    meal_block_re = re.compile(
        r"(?m)^(?P<head>(?:-\s*)?\*\*(?:🏷️\s*)?(?P<title>[^*\n]*(?:Almoço|Lunch|Jantar|Dinner)[^*\n]*)\*\*\s*\n)"
        r"(?P<body>(?:(?!^(?:-\s*)?\*\*(?:🏷️\s*)?[^*\n]+\*\*\s*$|^###\s+|^---\s*$|^📌\s+)[^\n]*\n?)*)",
        re.IGNORECASE | re.MULTILINE,
    )

    def replacement(match: re.Match[str]) -> str:
        title = match.group("title").strip()
        body = match.group("body") or ""
        title_key = _normalize_planner_text(title)
        meal_kind = "dinner" if re.search(r"\b(?:jantar|dinner)\b", title_key) else "lunch"
        current_card = _match_evidence_card_for_block(title, food_candidates)
        current_score = _score_evidence_food_card(
            current_card,
            meal_kind,
            scoring_context,
            planned_card_dicts,
        )
        if current_card is None:
            current_dict = {
                "name": title,
                "description": body,
                "address": body,
                "features": body,
                "category": "Restaurante" if is_pt else "Restaurant",
            }
            current_score = _score_food_plan_card(current_dict) + _score_food_card_for_meal_context(
                current_dict,
                meal_kind,
                scoring_context,
                planned_card_dicts,
            )

        context_basis = _normalize_planner_text(scoring_context)
        query_basis = _normalize_planner_text(user_message)
        preferred_area_key, _preferred_area_label, _preferred_blockers = _planner_local_area_profile(user_message)

        def card_basis(card: EvidenceCard) -> str:
            """Return searchable text for meal locality checks."""
            return _normalize_planner_text(
                " ".join([card.title, card.summary, " ".join(card.fields.values())])
            )

        locality_candidates = []
        if meal_kind == "lunch" and preferred_area_key in {"central_corridor", "alfama"}:
            locality_candidates = [
                card for card in food_candidates
                if _PLANNER_CENTRAL_AREA_RE.search(card_basis(card))
                and not _PLANNER_BELEM_AREA_RE.search(card_basis(card))
            ]
        elif meal_kind == "lunch" and preferred_area_key == "marques_de_pombal":
            locality_candidates = [
                card for card in food_candidates
                if re.search(
                    r"\b(?:marques\s+de\s+pombal|marques\s+pombal|avenida\s+da\s+liberdade|1250)\b",
                    card_basis(card),
                )
            ]
        elif (
            meal_kind == "lunch"
            and _PLANNER_BELEM_AREA_RE.search(context_basis)
            and _PLANNER_BELEM_AREA_RE.search(query_basis)
        ):
            locality_candidates = [
                card for card in food_candidates
                if _PLANNER_BELEM_AREA_RE.search(card_basis(card))
            ]
        elif meal_kind == "dinner" and re.search(r"\b(?:hotel|saldanha|avenidas novas|picoas)\b", context_basis):
            locality_candidates = [
                card for card in food_candidates
                if re.search(r"\b(?:saldanha|avenidas novas|picoas|republica|duque de avila)\b", card_basis(card))
            ]
        if not locality_candidates and meal_kind == "lunch" and (
            _PLANNER_CENTRAL_AREA_RE.search(context_basis)
            or _PLANNER_BELEM_AREA_RE.search(context_basis)
        ):
            locality_candidates = [
                card for card in food_candidates
                if (
                    _PLANNER_CENTRAL_AREA_RE.search(card_basis(card))
                    or _PLANNER_BELEM_AREA_RE.search(card_basis(card))
                )
                and not re.search(
                    r"\b(?:saldanha|avenidas novas|picoas|republica|duque de avila)\b",
                    card_basis(card),
                )
            ]

        candidate_pool = locality_candidates or food_candidates
        best_card = max(
            candidate_pool,
            key=lambda card: _score_evidence_food_card(
                card,
                meal_kind,
                scoring_context,
                planned_card_dicts,
            ),
        )
        best_score = _score_evidence_food_card(
            best_card,
            meal_kind,
            scoring_context,
            planned_card_dicts,
        )
        current_basis = _normalize_planner_text(
            " ".join(
                [
                    title,
                    body,
                    current_card.title if current_card else "",
                    current_card.summary if current_card else "",
                    " ".join(current_card.fields.values()) if current_card else "",
                ]
            )
        )
        best_basis = card_basis(best_card)
        locality_override = bool(
            meal_kind == "lunch"
            and preferred_area_key in {"central_corridor", "alfama"}
            and _PLANNER_CENTRAL_AREA_RE.search(best_basis)
            and not _PLANNER_CENTRAL_AREA_RE.search(current_basis)
        ) or bool(
            meal_kind == "lunch"
            and _PLANNER_BELEM_AREA_RE.search(context_basis)
            and _PLANNER_BELEM_AREA_RE.search(query_basis)
            and _PLANNER_BELEM_AREA_RE.search(best_basis)
            and not _PLANNER_BELEM_AREA_RE.search(current_basis)
        ) or bool(
            meal_kind == "dinner"
            and re.search(r"\b(?:hotel|saldanha|avenidas novas|picoas)\b", context_basis)
            and re.search(r"\b(?:saldanha|avenidas novas|picoas|republica|duque de avila)\b", best_basis)
            and not re.search(r"\b(?:saldanha|avenidas novas|picoas|republica|duque de avila)\b", current_basis)
        )
        if not locality_override and best_score <= current_score + 20:
            return match.group(0)

        from agent.planning.renderer import _format_detail_bullet

        label = (
            "Jantar" if is_pt and meal_kind == "dinner"
            else "Almoço" if is_pt
            else "Dinner" if meal_kind == "dinner"
            else "Lunch"
        )
        lines = [f"- **🏷️ {label}: {best_card.title}**"]
        for detail in _details_from_evidence_card(best_card, language=language):
            lines.append(_format_detail_bullet(detail, is_pt))
        return "\n".join(lines).rstrip() + "\n\n"

    repaired = meal_block_re.sub(replacement, response)
    if repaired != response:
        repaired = _strip_irrelevant_planner_movement_items(repaired, user_message, language)
        repaired = _repair_planner_visitlisboa_details_links(
            repaired,
            places_data=places_data,
            language=language,
        )
    return repaired


def _evidence_card_to_planner_card(card: EvidenceCard) -> Dict[str, str]:
    """Convert an EvidenceCard into the lightweight dict used by scorers."""
    return {
        "name": card.title,
        "category": _evidence_field_value(card, "Category"),
        "address": _evidence_field_value(card, "Address"),
        "description": _evidence_field_value(card, "Description") or card.summary,
        "features": _evidence_field_value(card, "Features"),
        "hours": _evidence_field_value(card, "Hours"),
        "rating": _evidence_field_value(card, "Rating"),
        "distance": _evidence_field_value(card, "Distance"),
    }


def _score_evidence_food_card(
    card: EvidenceCard | None,
    meal_kind: str,
    user_message: str,
    planned_cards: List[Dict[str, str]],
) -> int:
    """Score a food evidence card for a requested meal slot."""
    if card is None:
        return -999
    card_dict = _evidence_card_to_planner_card(card)
    return _score_food_plan_card(card_dict, user_message) + _score_food_card_for_meal_context(
        card_dict,
        meal_kind,
        user_message,
        planned_cards,
    )


def _details_from_evidence_card(card: EvidenceCard, *, language: str = "en") -> List[str]:
    """Build canonical detail strings from an evidence card."""
    details: List[str] = []
    for label in (
        "Description",
        "Category",
        "Address",
        "Distance",
        "Hours",
        "Price",
        "Features",
        "Rating",
        "Phone",
        "Email",
        "Website",
        "Tickets",
        "More details",
        "Mais detalhes",
    ):
        value = _evidence_field_value(card, label)
        if value and label == "Description":
            value = _planner_card_description_for_language(value, language)
        if value:
            details.append(f"{label}: {value}")
    return details


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
        "features": {"features", "caracteristicas"},
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
    text = re.sub(r"^([^*\n:]{2,90}):\*\*\s*([^*\n]{1,220})\*\*$", r"**\1:** \2", text)
    text = re.sub(r"\*\*([^*:\n]{2,80}):\*\*\s*", r"**\1:** ", text)
    text = re.sub(r"\*\*([^*:\n]{2,80}):\s*([^*]{1,160})\*\*", r"**\1:** \2", text)
    text = re.sub(r"\*\*([^*]+)\*\*\s*:\s*", r"\1: ", text)
    text = re.sub(r"\*\*([^*:]{2,80}):\s*\*\*", r"**\1:**", text)
    if text.count("**") % 2 == 1:
        text = re.sub(r"\*\*\s*$", "", text).strip()
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
        "metro de lisboa",
        "carris",
        "cp",
    }


def _is_planner_transport_status_summary(item: str) -> bool:
    """Return whether a transport bullet is a network status summary, not a leg."""
    normalized = _normalize_planner_text(item)
    if not normalized:
        return True
    if re.search(r"\b(?:estado geral|general status|circulacao normal em todas as linhas|all lines normal)\b", normalized):
        return True
    if re.fullmatch(r"(?:amarela|azul|verde|vermelha|yellow|blue|green|red)\s*ok", normalized):
        return True
    if re.search(r"\b(?:linha\s+)?(?:amarela|azul|verde|vermelha|yellow|blue|green|red)\b.*\bok\b", normalized):
        return True
    if re.search(
        r"\b(?:linha\s+)?(?:amarela|azul|verde|vermelha|yellow|blue|green|red)\b"
        r".*\b(?:circulacao\s+normal|normal\s+service)\b",
        normalized,
    ):
        return True
    if not _planner_text_has_route_arrow(item) and re.search(
        r"\b(?:servico em funcionamento|serviço em funcionamento|veiculos em servico|veículos em serviço|"
        r"alertas ativos|alertas activos|atrasos|comboios a circular|vehicles in service|active alerts|delays)\b",
        normalized,
    ):
        return True
    return False


def _planner_transport_bullet_is_actionable(item: str) -> bool:
    """Return whether a movement bullet contains route-level user value."""
    normalized = _normalize_planner_text(item)
    if not normalized:
        return False
    if item.count("**") % 2:
        return False
    if re.search(
        r"(?i)\b(?:embarque em|board at|saia em|exit at|partida|departure):\S",
        item,
    ):
        return False
    if _planner_movement_bullet_is_generic_operator_advice(item):
        return False
    if _planner_text_has_route_arrow(item):
        return True
    if re.search(r"\b(?:linha|line)\s+(?:\d{1,3}[a-z]?|azul|verde|amarela|vermelha|blue|green|yellow|red)\b", normalized):
        return True
    if re.search(r"\b(?:nao confirmad|não confirmad|unconfirmed|sem ligacao confirmada|sem ligação confirmada)\b", normalized):
        return True
    if re.search(r"\b\d{1,3}\s*(?:min|km|m)\b", normalized):
        return True
    if (
        re.search(r"\b(?:carris|metro|cp|comboio|autocarro|bus|tram|eletrico|el[eé]trico)\b", normalized)
        and re.search(
            r"\b(?:para|from|to|entre|between|apanha|board|sair|exit|paragem|stop|estacao|estação|station|"
            r"transbordo|transfer|diret[ao]s?|direct|opcoes carris|opções carris)\b",
            normalized,
        )
        and _movement_bullet_has_concrete_signal(item)
    ):
        return True
    if re.match(r"^\d{1,2}\s*[:h]\s*\d{2}\b", normalized):
        return False
    return False


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
    target_area = _extract_requested_plan_area(user_message)
    if target_area and _query_requests_architecture_theme(user_message) and _query_requests_walking_only_plan(user_message):
        area_label = target_area if language == "pt" else re.sub(r"\s+e\s+", " and ", target_area)
        return (
            f"Caminhada de arquitetura em {area_label}"
            if language == "pt"
            else f"Architecture walk in {area_label}"
        )
    if target_area and _query_requests_morning_window(user_message):
        return f"Manhã em {target_area}" if language == "pt" else f"Morning in {target_area}"
    if target_area and re.search(r"\b(?:tarde|afternoon)\b", normalized):
        return f"Tarde em {target_area}" if language == "pt" else f"Afternoon in {target_area}"
    if _query_requests_cafe_stop(user_message):
        return "Plano cultural com pausa gastronómica" if language == "pt" else "Cultural plan with a food break"
    if _query_requests_cultural_stop(user_message) and _query_requests_food_stop(user_message):
        return "Plano cultural com pausa gastronómica" if language == "pt" else "Cultural plan with a food break"
    if _is_historic_gastronomy_day_request(normalized):
        return "Roteiro histórico e gastronómico de 1 dia" if language == "pt" else "One-day history and food itinerary"
    if _is_event_food_plan_request(normalized):
        if re.search(r"\b(?:vegetariano|vegetariana|vegetarian|vegan|vegano|vegana)\b", normalized):
            return "Plano cultural com jantar vegetariano" if language == "pt" else "Cultural plan with vegetarian dinner"
        return "Plano cultural com jantar tradicional" if language == "pt" else "Cultural plan with traditional dinner"
    if _is_event_planning_request(normalized):
        return "Plano de eventos culturais" if language == "pt" else "Cultural events plan"
    if "principe real" in normalized or "príncipe real" in str(user_message).lower():
        return "Noite descontraída no Príncipe Real" if language == "pt" else "Relaxed evening around Príncipe Real"
    if re.search(r"\b(?:museum|museu|museums|museus)\b", normalized):
        return "Dia de museus em Lisboa" if language == "pt" else "Lisbon museum day"
    excluded_areas = set(_extract_excluded_plan_areas(user_message))
    if "belem" in normalized and "belem" not in excluded_areas:
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
    if _is_event_food_plan_request(normalized):
        if is_pt:
            if re.search(r"\b(?:vegetariano|vegetariana|vegetarian|vegan|vegano|vegana)\b", normalized):
                return "Organizei a tarde com base nos dados confirmados; se não houver evento gratuito confirmado, deixo essa limitação explícita e mantenho apenas o jantar vegetariano suportado."
            return "Organizei os eventos e o jantar com os dados confirmados; onde não há ligação concreta, deixo a limitação explícita."
        return "I ordered the events and dinner from confirmed data; where no concrete connection is available, I keep the limitation explicit."
    if _is_event_planning_request(normalized):
        if is_pt:
            return "Selecionei eventos suportados pelos dados recolhidos, sem inventar disponibilidade, bilhetes ou horários."
        return "I selected events supported by the gathered data without inventing availability, tickets, or schedules."
    target_area = _extract_requested_plan_area(user_message)
    if target_area and _query_requests_architecture_theme(user_message) and _query_requests_walking_only_plan(user_message):
        area_label = target_area if is_pt else re.sub(r"\s+e\s+", " and ", target_area)
        if is_pt:
            return f"Organizei uma caminhada curta em {area_label} com foco em arquitetura, usando locais confirmados e limitações explícitas quando os detalhes não ficaram confirmados."
        return f"I organized a short architecture walk in {area_label}, using confirmed places and explicit limitations where details were not confirmed."
    if re.search(r"\b(?:chuva|chover|rain|interiores?|indoor|cobert[oa]s?|covered)\b", normalized):
        if target_area:
            if is_pt:
                return f"Adaptei o roteiro para chuva, mantendo apenas opções interiores ou cobertas confirmadas na zona de {target_area}."
            return f"I adapted the itinerary for rain, keeping only confirmed indoor or covered options in {target_area}."
        if is_pt:
            return "Adaptei o roteiro para chuva, priorizando opções interiores ou cobertas confirmadas."
        return "I adapted the itinerary for rain, prioritizing confirmed indoor or covered options."
    if target_area and _query_requests_low_walk_plan(user_message):
        if is_pt:
            return f"Organizei uma proposta compacta em {target_area}, priorizando paragens próximas e limitações explícitas quando os dados não confirmam um detalhe."
        return f"I organized a compact proposal in {target_area}, prioritizing nearby stops and explicit limitations where the gathered data does not confirm a detail."
    if _query_requests_walking_only_plan(user_message):
        if _query_requests_architecture_theme(user_message):
            if is_pt:
                return "Organizei o plano para ser feito sobretudo a pé e com foco em arquitetura, usando apenas locais confirmados nos dados recolhidos."
            return "I organized the plan to be done mostly on foot with an architecture focus, using only places confirmed in the gathered data."
        if is_pt:
            return "Organizei o plano para ser feito sobretudo a pé, usando apenas locais confirmados nos dados recolhidos."
        return "I organized the plan to be done mostly on foot, using only places confirmed in the gathered data."
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
    conversation_context: str = "",
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
            conversation_context=conversation_context,
        )

    if _is_event_planning_request(normalized_query):
        return _build_card_based_itinerary_fallback(
            user_message=user_message,
            language=language,
            weather_data=weather_data,
            transport_data=transport_data,
            places_data=places_data,
            events_data=events_data,
            qa_disclaimers=qa_disclaimers,
            conversation_context=conversation_context,
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
            conversation_context=conversation_context,
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

    visible_lines = [line.strip() for line in cleaned_response.splitlines() if line.strip()]
    if any(line.count("**") % 2 for line in visible_lines):
        return True
    if any(
        re.search(
            r"(?i)\b(?:embarque em|board at|saia em|exit at|partida|departure|chegada|arrival):\S",
            line,
        )
        for line in visible_lines
    ):
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
    if any(line.count("**") % 2 == 1 for line in cleaned_response.splitlines()):
        return True
    if any(
        "](" in line and ")" not in line[line.rfind("](") + 2 :]
        for line in cleaned_response.splitlines()
    ):
        return True
    if _planner_response_contains_generic_movement_advice(cleaned_response):
        return True
    if re.search(
        r"(?mis)^\s*[-*]?\s*\*\*🏷️[^\n]*(?:Paragem hist[oó]rica|Historic stop):[^\n]+\*\*\s*"
        r"\n\s*[-*]\s*🏷️\s+\*\*(?:Categoria|Category):\*\*[^\n]*"
        r"(?=\n\s*(?:[-*]?\s*\*\*🏷️|---|###|📌|$))",
        cleaned_response,
    ):
        return True
    if re.search(r"(?m)^\s*[-*]\s*[\U0001F300-\U0001FAFF\u2300-\u23FF\u2600-\u27BF\uFE0F\u200D\s]+\s*$", cleaned_response):
        return True
    if re.search(
        r"(?mi)^\s*[-*]\s*📝\s+\*\*(?:Descrição|Descricao|Description):\*\*\s*(?:⭐|TripAdvisor|Avalia[cç][aã]o|Rating)\b",
        cleaned_response,
    ):
        return True
    if re.search(r"(?mi)^###\s+🏛️\s+\*\*(?:Evid[eê]ncia para planeamento|Planning evidence)\*\*", cleaned_response):
        return True
    if re.search(
        r"(?mi)^\s*[-*]?\s*(?:\*\*)?[^\n]*?(?:Paragem hist[oó]rica|Historic stop):\s*[^*\n]*(?:Restaurante|Restaurant)\b",
        cleaned_response,
    ):
        return True
    if re.search(r"\b(?:paragem hist.?rica|historic stop)\b.{0,90}\b(?:restaurante|restaurant)\b", normalized):
        return True
    if re.search(
        r"(?mi)^\s*[-*]\s*(?:[^\w\s*]+\s*)?\*\*(?:Carris Urbana?|Carris Metropolitana(?:\s*\([^)]*\))?|Metro de Lisboa|CP)\*\*\s*$",
        cleaned_response,
    ):
        return True
    if (
        re.search(r"\b(?:linha|autocarro|bus|tram|el[eé]trico|electrico)\s+\d{1,4}[a-z]?\b", raw_lower)
        and "carris" not in source_line.lower()
    ):
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
        r"\bhoje fechado\b",
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
    if re.search(
        r"(?mi)^[-*]\s*(?:[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+\s*)?\*\*(?:Description|Category|Address|Phone|Website|Hours|Price|Rating|Features|Descrição|Descricao|Categoria|Morada|Telefone|Horário|Horario|Preço|Preco|Avaliação|Avaliacao|Características|Caracteristicas)\s*:\*\*",
        cleaned_response,
    ):
        return True
    if re.search(r"\*\*[^*\n]*[A-Za-zÀ-ÿ][^*\n]{0,50}:\S[^*\n]*\*\*", cleaned_response):
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


def _planner_response_missing_requested_plan_components(
    cleaned_response: str,
    user_message: str,
) -> bool:
    """Return whether a mixed plan collapsed requested components into too few cards."""
    normalized_query = _normalize_planner_text(user_message)
    required_components = 0
    if _is_event_planning_request(normalized_query):
        required_components += 1
    if re.search(
        r"\b(?:gastronom\w*|restaurants?|restaurantes?|food|comida|tradicional|almoco|almoço|"
        r"jantar|dinner|cafe|coffee|pastelaria|pastry|pastel|pasteis|nata|custard|tarts?)\b",
        normalized_query,
    ):
        required_components += 1
    if re.search(
        r"\b(?:historic|historical|historico|historico|monument|monumento|patrimonio|museu|museum|heritage)\b",
        normalized_query,
    ):
        required_components += 1
    if required_components < 2:
        return False

    in_route = False
    route_card_count = 0
    route_lines: List[str] = []
    for raw_line in str(cleaned_response or "").splitlines():
        stripped = raw_line.strip()
        if re.match(r"^###\s+📍\s+(?:\*\*)?(?:Roteiro sugerido|Suggested route)", stripped, flags=re.IGNORECASE):
            in_route = True
            continue
        if in_route and stripped.startswith("### "):
            break
        if in_route and stripped:
            route_lines.append(stripped)
        if in_route and re.match(r"^(?:[-*]\s+)?\*\*[^*\n]{3,180}\*\*\s*$", stripped):
            route_card_count += 1
    if route_card_count < required_components:
        return True

    route_text = _normalize_planner_text("\n".join(route_lines))
    if _query_requests_food_stop(user_message) and not re.search(
        r"\b(?:restaurante|restaurant|almo[cç]o|lunch|jantar|dinner|cafe|coffee|"
        r"pastelaria|pastry|pastel|pasteis|nata|custard|tarts?|gastronom)\b",
        route_text,
    ):
        return True
    if _query_requests_cultural_stop(user_message) and not re.search(
        r"\b(?:museu|museum|monumento|monument|hist[oó]ric|heritage|patrim[oó]nio|cultural|culture|"
        r"exposi[cç][aã]o|exhibition|galeria|gallery|ocean[aá]rio|aquarium|pavilh[aã]o\s+do\s+conhecimento)\b",
        route_text,
    ):
        return True
    if _is_event_planning_request(normalized_query) and not re.search(
        r"\b(?:evento|event|concerto|concert|festival|teatro|theatre|theater|m[uú]sica|music)\b",
        route_text,
    ):
        return True
    return False


def _planner_response_route_blocks(response: str) -> List[str]:
    """Extract top-level itinerary blocks from the user-facing route section."""
    blocks: List[List[str]] = []
    current: List[str] = []
    in_route = False

    for raw_line in str(response or "").splitlines():
        stripped = raw_line.strip()
        normalized_line = _normalize_planner_text(stripped)
        if re.search(r"\b(?:roteiro sugerido|suggested route)\b", normalized_line):
            in_route = True
            continue
        if in_route and stripped.startswith("### "):
            break
        if not in_route:
            continue
        if not stripped:
            if current:
                current.append(raw_line)
            continue
        is_top_level_title = bool(
            raw_line[:1] not in {" ", "\t"}
            and re.match(r"^(?:[-*]\s+)?\*\*[^*\n]{2,180}\*\*", stripped)
        )
        if is_top_level_title:
            if current:
                blocks.append(current)
            current = [raw_line]
        elif current:
            current.append(raw_line)

    if current:
        blocks.append(current)
    return ["\n".join(block).strip() for block in blocks if "\n".join(block).strip()]


def _planner_response_block_title(block: str) -> str:
    """Return the visible title for an itinerary block."""
    first_line = str(block or "").splitlines()[0] if str(block or "").splitlines() else ""
    match = re.match(r"^(?:[-*]\s+)?\*\*(?P<title>[^*\n]{2,180})\*\*", first_line.strip())
    if not match:
        return ""
    title = re.sub(r"^[^\wÀ-ÿ0-9]+", "", match.group("title")).strip()
    title = re.sub(r"^\d{1,2}:\d{2}\s*[·.-]\s*", "", title).strip()
    return title


def _merge_planner_movement_sections(response: str, language: str) -> str:
    """Merge duplicated planner movement sections into one Streamlit-safe block."""
    if not str(response or "").strip():
        return response
    lines = str(response).splitlines()

    def is_movement_heading(line: str) -> bool:
        return bool(
            re.match(
                r"^###\s+.*\*\*(?:Como te deslocas|How to move)\*\*\s*$",
                line.strip(),
                flags=re.IGNORECASE,
            )
        )

    sections: List[tuple[int, int, List[str]]] = []
    for index, line in enumerate(lines):
        if not is_movement_heading(line):
            continue
        start = index
        previous = index - 1
        while previous >= 0 and not lines[previous].strip():
            previous -= 1
        if previous >= 0 and lines[previous].strip() == "---":
            start = previous
        end = index + 1
        while end < len(lines):
            stripped = lines[end].strip()
            if (
                stripped == "---"
                or stripped.startswith("### ")
                or _PLANNER_SOURCE_LINE_RE.match(stripped)
            ):
                break
            end += 1
        sections.append((start, end, lines[index + 1:end]))

    if len(sections) <= 1:
        return response

    heading = "### 🚇 **Como te deslocas**" if language == "pt" else "### 🚇 **How to move**"
    bullets: List[str] = []
    seen: set[str] = set()
    for _start, _end, body_lines in sections:
        for raw_line in body_lines:
            stripped = raw_line.strip()
            if not re.match(r"^[-*]\s+", stripped):
                continue
            key = _normalize_planner_text(stripped)
            if key and key not in seen:
                seen.add(key)
                bullets.append(stripped)

    if not bullets:
        return response

    merged_section = f"{heading}\n\n" + "\n".join(bullets)
    first_start = sections[0][0]
    last_end = sections[-1][1]
    before = "\n".join(lines[:first_start]).rstrip()
    after = "\n".join(lines[last_end:]).lstrip()
    prefix = f"{before}\n\n---\n\n" if before else ""
    suffix = f"\n\n{after}" if after else ""
    return f"{prefix}{merged_section}{suffix}".strip()


def _ensure_stop_by_stop_movement_in_response(
    response: str,
    user_message: str,
    language: str,
) -> str:
    """Ensure explicit stop-by-stop movement requests cover consecutive stops."""
    cleaned = _merge_planner_movement_sections(response, language)
    movement_text = _planner_movement_section_text(cleaned)
    if not _query_requests_stop_by_stop_movement(user_message) and not movement_text:
        return cleaned

    blocks = _planner_response_route_blocks(cleaned)
    title_blocks = [
        (title, block)
        for block in blocks
        for title in [_planner_response_block_title(block)]
        if title
    ]
    titles = [title for title, _block in title_blocks]
    if len(titles) < 2:
        return cleaned

    existing_bullets = [
        line.strip()
        for line in movement_text.splitlines()
        if re.match(r"^[-*]\s+", line.strip())
    ]
    normalized_query = _normalize_planner_text(user_message)
    wants_public_transport = _query_requests_public_transport(user_message) or bool(
        re.search(r"\btransportes?\b", normalized_query)
    )
    is_pt = language == "pt"
    requested_areas = [
        area for area in [
            *[label for label, _time_label in _requested_anchor_time_constraints(user_message)],
            *_requested_anchor_labels(user_message),
            _extract_requested_plan_area(user_message),
        ]
        if _normalize_planner_text(area) and not _planner_area_is_broad_city(_normalize_planner_text(area))
    ]

    def block_matches_area(block: str, title: str, area: str) -> bool:
        return _planner_card_matches_area(
            {
                "name": title,
                "address": block,
                "description": block,
                "category": block,
            },
            area,
        )

    def existing_bullet_for_pair(
        origin: str,
        origin_block: str,
        destination: str,
        destination_block: str,
        used_indexes: set[int],
    ) -> str:
        origin_key = _normalize_planner_text(origin)
        destination_key = _normalize_planner_text(destination)
        for index, bullet in enumerate(existing_bullets):
            if index in used_indexes:
                continue
            if not _planner_text_has_route_arrow(bullet):
                continue
            bullet_key = _normalize_planner_text(bullet)
            if origin_key and destination_key and origin_key in bullet_key and destination_key in bullet_key:
                used_indexes.add(index)
                return bullet

        origin_areas = [
            area for area in requested_areas
            if block_matches_area(origin_block, origin, area)
        ]
        destination_areas = [
            area for area in requested_areas
            if block_matches_area(destination_block, destination, area)
        ]
        if not origin_areas or not destination_areas:
            return ""

        origin_area_keys = {_normalize_planner_text(area) for area in origin_areas}
        destination_area_keys = {_normalize_planner_text(area) for area in destination_areas}
        if origin_area_keys & destination_area_keys:
            return ""
        for index, bullet in enumerate(existing_bullets):
            if index in used_indexes:
                continue
            if not _planner_text_has_route_arrow(bullet):
                continue
            bullet_key = _normalize_planner_text(bullet)
            if any(key and key in bullet_key for key in origin_area_keys) and any(
                key and key in bullet_key for key in destination_area_keys
            ):
                used_indexes.add(index)
                return bullet
        return ""

    ordered_bullets: List[str] = []
    seen_bullets: set[str] = set()
    used_bullet_indexes: set[int] = set()
    if title_blocks:
        first_title, first_block = title_blocks[0]
        first_leg_item = _planner_origin_to_first_stop_item(
            [
                {
                    "name": first_title,
                    "address": first_block,
                    "description": first_block,
                    "category": first_block,
                }
            ],
            user_message,
            language,
        )
        if first_leg_item:
            first_leg_bullet = (
                first_leg_item
                if re.match(r"^[-*]\s+", first_leg_item.strip())
                else f"- {first_leg_item}"
            )
            first_leg_key = _normalize_planner_text(first_leg_bullet)
            requested_origin_key = _normalize_planner_text(_extract_requested_plan_origin(user_message))
            first_title_key = _normalize_planner_text(first_title)
            already_has_first_leg = False
            if requested_origin_key and first_title_key:
                for bullet in existing_bullets:
                    if not _planner_text_has_route_arrow(bullet):
                        continue
                    bullet_key = _normalize_planner_text(bullet)
                    origin_pos = bullet_key.find(requested_origin_key)
                    first_pos = bullet_key.find(first_title_key)
                    if origin_pos >= 0 and first_pos >= 0 and origin_pos < first_pos:
                        already_has_first_leg = True
                        break
            if first_leg_key and not already_has_first_leg:
                seen_bullets.add(first_leg_key)
                ordered_bullets.append(first_leg_bullet)

    for (origin, origin_block), (destination, destination_block) in zip(title_blocks, title_blocks[1:5]):
        existing = existing_bullet_for_pair(
            origin,
            origin_block,
            destination,
            destination_block,
            used_bullet_indexes,
        )
        if existing:
            bullet_key = _normalize_planner_text(existing)
            if bullet_key and bullet_key not in seen_bullets:
                seen_bullets.add(bullet_key)
                ordered_bullets.append(existing)
            continue
        origin_key = _normalize_planner_text(origin)
        destination_key = _normalize_planner_text(destination)
        if origin_key and destination_key and any(
            origin_key in _normalize_planner_text(bullet)
            and destination_key in _normalize_planner_text(bullet)
            for bullet in ordered_bullets
        ):
            continue
        origin_area_keys = {
            _normalize_planner_text(area)
            for area in requested_areas
            if block_matches_area(origin_block, origin, area)
        }
        destination_area_keys = {
            _normalize_planner_text(area)
            for area in requested_areas
            if block_matches_area(destination_block, destination, area)
        }
        same_requested_area = bool(origin_area_keys & destination_area_keys)
        if wants_public_transport and not same_requested_area:
            ordered_bullets.append(
                (
                    f"- 🚇 **{origin} → {destination}:** a ligação de transporte público exata "
                    "entre estas paragens não ficou confirmada nos dados recolhidos; não inventei linha, paragem ou duração."
                )
                if is_pt
                else (
                    f"- 🚇 **{origin} → {destination}:** the exact public-transport leg between these stops "
                    "was not confirmed in the gathered data; I did not invent a line, stop, or duration."
                )
            )
        else:
            ordered_bullets.append(
                (
                    f"- 🚶 **{origin} → {destination}:** faz esta transição como deslocação curta/local; "
                    "confirma no mapa a melhor rua de ligação no momento."
                )
                if is_pt
                else (
                    f"- 🚶 **{origin} → {destination}:** treat this as a short/local transition; "
                    "check the best walking link on the map when you go."
                )
            )

    if not ordered_bullets:
        return cleaned

    heading = "### 🚇 **Como te deslocas**" if is_pt else "### 🚇 **How to move**"
    section_re = re.compile(
        r"(?ms)(?P<header>^###\s+.*\*\*(?:Como te deslocas|How to move)\*\*\s*\n)"
        r"(?P<body>.*?)(?=(?:\n---\s*\n|\n###\s+|\n📌\s+\*\*(?:Fonte|Source):|\Z))",
        flags=re.MULTILINE,
    )

    def replace_section(match: re.Match[str]) -> str:
        return f"{match.group('header')}\n" + "\n".join(ordered_bullets) + "\n"

    if section_re.search(cleaned):
        return section_re.sub(replace_section, cleaned, count=1).strip()

    movement_section = f"{heading}\n\n" + "\n".join(ordered_bullets)
    insert_pattern = (
        r"(?m)^###\s+.*\*\*Notas finais\*\*"
        if is_pt
        else r"(?m)^###\s+.*\*\*Final notes\*\*"
    )
    if re.search(insert_pattern, cleaned):
        return re.sub(insert_pattern, f"{movement_section}\n\n---\n\n\\g<0>", cleaned, count=1).strip()
    source_match = re.search(r"(?m)^📌\s+\*\*(?:Fonte|Source):\*\*.*$", cleaned)
    if source_match:
        before = cleaned[:source_match.start()].rstrip()
        source = cleaned[source_match.start():].strip()
        return f"{before}\n\n---\n\n{movement_section}\n\n{source}".strip()
    return f"{cleaned.rstrip()}\n\n---\n\n{movement_section}".strip()


def _planner_response_block_matches_requested_type(block: str, count_type: str) -> bool:
    """Return whether a rendered itinerary block satisfies a requested count type."""
    basis = _normalize_planner_text(block)
    basis_without_broad_category = _normalize_planner_text(
        "\n".join(
            line
            for line in block.splitlines()
            if not re.search(
                r"\*\*(?:categoria|category)\s*:?\*\*.*\b(?:museums?\s*&\s*monuments?|museus?\s*&\s*monumentos?)\b",
                line,
                flags=re.IGNORECASE,
            )
        )
    )
    if count_type == "food":
        return bool(
            re.search(
                r"\b(?:restaurante|restaurant|almo[cç]o|almoco|lunch|jantar|dinner|"
                r"cafe|caf[eé]|coffee|pastelaria|pastry|gastronom|comida|food|meal)\b",
                basis,
            )
        )
    if count_type == "museum":
        return bool(
            re.search(
                r"\b(?:museu|museus|museum|museums|galeria|gallery|galleries|"
                r"oceanario|ocean[aá]rio|aquarium|pavilhao do conhecimento|pavilh[aã]o do conhecimento)\b",
                basis_without_broad_category,
            )
        )
    if count_type == "monument":
        return bool(
            re.search(
                r"\b(?:monumento|monument|mosteiro|monastery|torre|tower|padrao|padr[aã]o|"
                r"catedral|cathedral|igreja|church|castelo|castle|palacio|pal[aá]cio)\b",
                basis,
            )
        )
    if count_type == "viewpoint":
        return bool(re.search(r"\b(?:miradouro|viewpoint|lookout|vista|view)\b", basis))
    if count_type == "event":
        return bool(
            re.search(
                r"\b(?:evento|event|concerto|concert|festival|teatro|theatre|theater|"
                r"musica|m[uú]sica|desporto|sport)\b",
                basis,
            )
        )
    return True


def _planner_response_missing_requested_counts(response: str, user_message: str) -> bool:
    """Return whether the final itinerary violates explicit user cardinalities."""
    counts = _requested_plan_type_counts(user_message)
    if not counts:
        return False
    blocks = _planner_response_route_blocks(response)
    if not blocks:
        # Explicit counts need a parseable itinerary body. If the route section
        # cannot be read, retry instead of accepting a plan that may have
        # ignored constraints such as "2 museums" or "1 viewpoint".
        return True
    minimum_count_request = bool(
        re.search(r"\b(?:pelo\s+menos|no\s+minimo|no\s+m[ií]nimo|at\s+least|minimum)\b", _normalize_planner_text(user_message))
    )

    for count_type, requested_count in counts.items():
        if requested_count <= 0:
            continue
        if count_type == "total":
            if len(blocks) < requested_count or (not minimum_count_request and len(blocks) > requested_count):
                return True
            continue
        matching_blocks = [
            block for block in blocks
            if _planner_response_block_matches_requested_type(block, count_type)
        ]
        if len(matching_blocks) < requested_count:
            return True
        if not minimum_count_request and len(matching_blocks) > requested_count:
            return True
    return False


def _planner_route_block_ranges(response: str) -> List[tuple[int, int, str]]:
    """Return line ranges for visible route blocks in the suggested route."""
    lines = str(response or "").splitlines()
    ranges: List[tuple[int, int, str]] = []
    in_route = False
    route_end = len(lines)
    current_start: int | None = None

    for index, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        normalized_line = _normalize_planner_text(stripped)
        if re.search(r"\b(?:roteiro sugerido|suggested route)\b", normalized_line):
            in_route = True
            continue
        if in_route and stripped.startswith("### "):
            route_end = index
            break
        if not in_route:
            continue
        is_top_level_title = bool(
            raw_line[:1] not in {" ", "\t"}
            and re.match(r"^(?:[-*]\s+)?\*\*[^*\n]{2,180}\*\*", stripped)
        )
        if is_top_level_title:
            if current_start is not None:
                ranges.append((current_start, index, "\n".join(lines[current_start:index]).strip()))
            current_start = index

    if current_start is not None:
        ranges.append((current_start, route_end, "\n".join(lines[current_start:route_end]).strip()))
    return ranges


def _planner_response_block_is_protected(block: str, user_message: str, *, is_last: bool) -> bool:
    """Return whether a route block must stay despite typed count caps."""
    normalized_block = _normalize_planner_text(block)
    title_key = _normalize_planner_text(_planner_response_block_title(block))
    for label, time_label in _requested_anchor_time_constraints(user_message):
        label_key = _normalize_planner_text(label)
        if label_key and label_key in normalized_block and time_label in block:
            return True

    end_area = _extract_requested_plan_area(user_message)
    end_key = _normalize_planner_text(end_area)
    if (
        is_last
        and end_key
        and (
            end_key in title_key
            or title_key in end_key
            or end_key in normalized_block
        )
    ):
        return True
    return False


def _repair_response_requested_type_counts(response: str, user_message: str) -> str:
    """Remove extra typed itinerary blocks while preserving explicit anchors."""
    counts = {
        key: value
        for key, value in _requested_plan_type_counts(user_message).items()
        if key in {"museum", "monument", "event", "viewpoint"} and value > 0
    }
    if not counts or not response:
        return response or ""
    if re.search(r"\b(?:pelo\s+menos|no\s+minimo|no\s+m[ií]nimo|at\s+least|minimum)\b", _normalize_planner_text(user_message)):
        return response

    block_ranges = _planner_route_block_ranges(response)
    if not block_ranges:
        return response

    remove_line_indexes: set[int] = set()
    last_block_index = len(block_ranges) - 1
    for count_type, requested_count in counts.items():
        matching = [
            (index, start, end, block)
            for index, (start, end, block) in enumerate(block_ranges)
            if _planner_response_block_matches_requested_type(block, count_type)
        ]
        if len(matching) <= requested_count:
            continue
        removable = [
            (index, start, end, block)
            for index, start, end, block in matching
            if not _planner_response_block_is_protected(
                block,
                user_message,
                is_last=index == last_block_index,
            )
        ]
        current_count = len(matching)
        for _index, start, end, _block in reversed(removable):
            if current_count <= requested_count:
                break
            remove_line_indexes.update(range(start, end))
            current_count -= 1

    if not remove_line_indexes:
        return response

    repaired_lines = [
        line for index, line in enumerate(str(response).splitlines())
        if index not in remove_line_indexes
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(repaired_lines)).strip()


def _planner_response_violates_requested_end(response: str, user_message: str) -> bool:
    """Return whether an explicit ending anchor is not the final visible stop."""
    normalized_query = _normalize_planner_text(user_message)
    if not re.search(
        r"\b(?:termina|termine|terminar|terminando|acaba|acabe|acabar|acabando|end|ending|finish|finishing)\b",
        normalized_query,
    ) and not re.search(r"\bfrom\s+[^,.;]+?\s+to\s+[^,.;]+", normalized_query):
        return False
    requested_end = _extract_requested_plan_area(user_message)
    requested_key = _normalize_planner_text(requested_end)
    if not requested_key:
        return False
    blocks = _planner_response_route_blocks(response)
    if not blocks:
        # An explicit start/end request needs a visible ordered itinerary. If
        # the route section cannot be parsed, accept no silent success here:
        # let the planner/QA retry instead of shipping an uncheckable plan.
        return True
    last_title = _normalize_planner_text(_planner_response_block_title(blocks[-1]))
    return bool(
        last_title
        and requested_key
        and requested_key not in last_title
        and last_title not in requested_key
    )


def _planner_response_has_closed_timed_stop(response: str, user_message: str) -> bool:
    """Return whether a user-timed stop is rendered as closed."""
    if not _requested_anchor_time_constraints(user_message):
        return False
    for block in _planner_response_route_blocks(response):
        normalized_block = _normalize_planner_text(block)
        has_requested_time = bool(
            re.search(r"\b(?:hora pedida pelo utilizador|requested time)\b", normalized_block)
            or re.search(r"^\s*[-*]\s+\*\*\s*\d{1,2}:\d{2}\b", block)
        )
        if has_requested_time and re.search(r"\b(?:hoje fechado|fechado hoje|today closed|closed today)\b", normalized_block):
            return True
    return False


def _planner_response_has_unusable_closed_stop(response: str, user_message: str) -> bool:
    """Return whether a plan includes a closed visit/meal stop as usable."""
    if re.search(r"\b(?:exterior|outside|fachada|facade)\b", _normalize_planner_text(user_message)):
        return False
    for block in _planner_response_route_blocks(response):
        normalized_block = _normalize_planner_text(block)
        if not re.search(r"\b(?:hoje fechado|fechado hoje|today closed|closed today)\b", normalized_block):
            continue
        if re.search(r"\b(?:restaurante|restaurant|almoco|lunch|jantar|dinner|museu|museum|visita|visit)\b", normalized_block):
            return True
    return False


def _planner_response_missing_requested_time_constraints(response: str, user_message: str) -> bool:
    """Return whether requested pass-through times are absent from the route."""
    time_constraints = _requested_anchor_time_constraints(user_message)
    if not time_constraints:
        return False
    blocks = _planner_response_route_blocks(response)
    if not blocks:
        # Explicit cardinalities are only meaningful if the final answer has a
        # parseable itinerary body. Otherwise the plan may look fluent while
        # silently ignoring requested counts such as "2 museums".
        return True
    for label, time_label in time_constraints:
        normalized_label = _normalize_planner_text(label)
        if not normalized_label or not time_label:
            continue
        matching_blocks = [
            block for block in blocks
            if normalized_label in _normalize_planner_text(block)
        ]
        if not matching_blocks:
            return True
        if not any(time_label in block for block in matching_blocks):
            return True
    return False


def _planner_response_violates_explicit_preference_contract(response: str, user_message: str) -> bool:
    """Return whether a plan ignores explicit stop counts, end, or timed anchors."""
    return bool(
        _planner_response_missing_requested_counts(response, user_message)
        or _planner_response_violates_requested_end(response, user_message)
        or _planner_response_has_closed_timed_stop(response, user_message)
        or _planner_response_has_unusable_closed_stop(response, user_message)
        or _planner_response_missing_requested_time_constraints(response, user_message)
    )


def _planner_response_has_low_fit_infrastructure_stop(response: str, user_message: str) -> bool:
    """Return whether a plan labels hotels/rooftops as cultural stops without being asked."""
    normalized_query = _normalize_planner_text(user_message)
    if not re.search(
        r"\b(?:museu|museum|historic|historico|historia|monument|monumento|patrim[oó]nio|heritage|gastronom|roteiro|itinerario)\b",
        normalized_query,
    ):
        return False
    if re.search(
        r"\b(?:rooftop|terrace|terra[cç]o|miradouro|viewpoint|view\s+point)\b",
        normalized_query,
    ):
        return False
    strict_historic_request = bool(
        re.search(r"\b(?:historic\w*|historico\w*|hist[oó]ric\w*|monument\w*|patrim[oó]ni\w*|heritage)\b", normalized_query)
    )
    allows_music_experiences = bool(
        re.search(r"\b(?:fado|music|m[uú]sica|show|concerto|concert)\b", normalized_query)
    )

    in_route_section = False
    for raw_line in str(response or "").splitlines():
        stripped = raw_line.strip()
        normalized_line = _normalize_planner_text(stripped)
        if re.search(r"\b(?:roteiro sugerido|suggested route)\b", normalized_line):
            in_route_section = True
            continue
        if in_route_section and stripped.startswith("### "):
            break
        if not in_route_section or raw_line[:1].isspace():
            continue
        match = re.match(r"^(?:[-*]\s+)?\*\*(?P<title>[^*]+)\*\*", stripped)
        if not match:
            continue
        title = _normalize_planner_text(match.group("title"))
        if re.search(
            r"\b(?:hotel|hostel|accommodation|alojamento|rooftop|terrace|terra[cç]o|business\s+center|rent-a-car|lockers)\b",
            title,
        ):
            return True
        if strict_historic_request and not allows_music_experiences and re.search(
            r"\b(?:living\s+experience|immersive|imersiv\w*|experience|experi[eê]ncia|fado|music|m[uú]sica|show|concert|concerto)\b",
            title,
        ):
            return True
    return False


def _planner_response_has_transport_quality_defects(
    cleaned_response: str,
    user_message: str,
    transport_data: str,
) -> bool:
    """Return whether transport-aware planner output hides grounded route gaps."""
    has_route_leg_evidence = bool(
        re.search(
            r"\b(?:liga[cç][oõ]es entre paragens do roteiro|route legs between itinerary stops|carris\s+\d{1,4}[a-z]?|caminhada curta|short walk)\b",
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
        r"\b(?:como te deslocas|how to move|carris\s+\d{1,4}[a-z]?|linha\s+\d{1,4}[a-z]?|caminhada curta|short walk|route legs|liga[cç][oõ]es)\b",
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
        r"\busa transporte publico\b.*\bligacao exata\b.*\bconfirmad",
        r"\bligacao exata\b.*\bconfirmad",
        r"\bligacao exacta\b.*\bconfirmad",
        r"\bdeve ser confirmada perto da hora\b",
        r"\bdeve ser confirmado perto da hora\b",
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
        and re.search(r"\b\d{1,2}(?::|h|\s+)\d{2}\b", normalized)
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
    if _planner_response_missing_requested_movement(response, user_message, transport_data):
        issue = (
            "The user explicitly requested movement between itinerary areas, but the draft omits "
            "the required concrete cross-area leg or exact unconfirmed-leg limitation."
        )
        if issue not in issues:
            issues.append(issue)
    return issues


def _add_requested_stop_issue(
    issues: List[str],
    response: str,
    user_message: str,
    evidence_data: str,
) -> List[str]:
    """Append planner issues when explicit user-requested anchors or order disappear."""
    if _planner_response_missing_requested_stops(response, user_message, evidence_data):
        issue = (
            "The draft omits at least one explicit place or area requested by the user. "
            "Preserve all requested anchors that are supported by the gathered evidence."
        )
        if issue not in issues:
            issues.append(issue)
    if _planner_response_violates_requested_start(response, user_message):
        issue = (
            "The draft ignores the user's requested starting anchor. "
            "Make the first itinerary stop match the requested start when it is available."
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
        issues = validate_plan_draft(
            draft,
            evidence,
            user_message=user_message,
        )
        if issues:
            retry_messages = messages + [
                SystemMessage(
                    content=(
                        "The previous JSON failed validation after evidence fields such as hours, prices, or links were restored. "
                        "Return corrected JSON only. Do not add Markdown. Ensure every timed stop falls within its evidenced opening hours; "
                        "if that is impossible, change the time, choose another evidenced stop, or state a scoped limitation. "
                        "Fix these issues: " + "; ".join(issues[:8])
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
            issues = validate_plan_draft(
                draft,
                evidence,
                user_message=user_message,
            )
            if issues:
                return ""
        rendered = render_plan_markdown(
            draft,
            sources=evidence.sources,
            language=language,
        )
        rendered = final_post_qa_guard(rendered, language=language)
        if not _planner_response_matches_schema(rendered):
            return ""
        if _planner_response_missing_requested_counts(rendered, user_message):
            return ""
        if _planner_response_has_markdown_contract_defects(rendered):
            return ""
        if _planner_response_has_local_area_drift(rendered, user_message):
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
        output_language: str | None = None,
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
        language = output_language if output_language in {"pt", "en"} else infer_response_language(user_query=user_message, default="en")
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
                and not _planner_response_missing_requested_movement(
                    structured_plan,
                    user_message,
                    transport_data,
                )
                and not _planner_response_missing_requested_stops(
                    structured_plan,
                    user_message,
                    "\n".join([places_data or "", events_data or ""]),
                )
                and not _planner_response_violates_requested_start(structured_plan, user_message)
                and not _planner_response_has_local_area_drift(structured_plan, user_message)
                and not _planner_response_violates_explicit_preference_contract(structured_plan, user_message)
                and not _planner_response_mixes_distant_walking_areas(structured_plan, user_message)
                and not _planner_response_uses_excluded_area(structured_plan, user_message)
            ):
                finalized_plan = _ensure_multi_day_response_quality(
                    structured_plan,
                    user_message=user_message,
                    language=language,
                    weather_data=weather_data,
                    transport_data=transport_data,
                    places_data=places_data,
                    events_data=events_data,
                    qa_disclaimers=qa_disclaimers,
                    conversation_context=conversation_context,
                )
                finalized_plan = _repair_response_requested_type_counts(finalized_plan, user_message)
                finalized_plan = _ensure_stop_by_stop_movement_in_response(
                    finalized_plan,
                    user_message,
                    language,
                )
                if not _planner_response_violates_explicit_preference_contract(finalized_plan, user_message):
                    return finalized_plan
            structured_plan = ""

        specific_plan = _build_specific_planner_fallback(
            user_message=user_message,
            language=language,
            weather_data=weather_data,
            transport_data=transport_data,
            places_data=places_data,
            events_data=events_data,
            qa_disclaimers=qa_disclaimers,
            conversation_context=conversation_context,
        )
        if specific_plan:
            if (
                _planner_response_has_minimum_user_value(specific_plan)
                and not _planner_response_violates_requested_start(specific_plan, user_message)
                and not _planner_response_has_local_area_drift(specific_plan, user_message)
                and not _planner_response_violates_explicit_preference_contract(specific_plan, user_message)
                and not _planner_response_missing_requested_food_stop(specific_plan, user_message)
                and not _planner_response_missing_requested_movement(
                    specific_plan,
                    user_message,
                    transport_data,
                )
            ):
                finalized_plan = _ensure_multi_day_response_quality(
                    specific_plan,
                    user_message=user_message,
                    language=language,
                    weather_data=weather_data,
                    transport_data=transport_data,
                    places_data=places_data,
                    events_data=events_data,
                    qa_disclaimers=qa_disclaimers,
                    conversation_context=conversation_context,
                )
                finalized_plan = _repair_response_requested_type_counts(finalized_plan, user_message)
                finalized_plan = _ensure_stop_by_stop_movement_in_response(
                    finalized_plan,
                    user_message,
                    language,
                )
                if not _planner_response_violates_explicit_preference_contract(finalized_plan, user_message):
                    return finalized_plan

        card_based_plan = _build_card_based_itinerary_fallback(
            user_message=user_message,
            language=language,
            weather_data=weather_data,
            transport_data=transport_data,
            places_data=places_data,
            events_data=events_data,
            qa_disclaimers=qa_disclaimers,
            conversation_context=conversation_context,
        )
        if card_based_plan:
            if (
                _planner_response_has_minimum_user_value(card_based_plan)
                and not _planner_response_violates_requested_start(card_based_plan, user_message)
                and not _planner_response_has_local_area_drift(card_based_plan, user_message)
                and not _planner_response_violates_explicit_preference_contract(card_based_plan, user_message)
                and not _planner_response_missing_requested_food_stop(card_based_plan, user_message)
            ):
                finalized_plan = _ensure_multi_day_response_quality(
                    card_based_plan,
                    user_message=user_message,
                    language=language,
                    weather_data=weather_data,
                    transport_data=transport_data,
                    places_data=places_data,
                    events_data=events_data,
                    qa_disclaimers=qa_disclaimers,
                    conversation_context=conversation_context,
                )
                finalized_plan = _repair_response_requested_type_counts(finalized_plan, user_message)
                finalized_plan = _ensure_stop_by_stop_movement_in_response(
                    finalized_plan,
                    user_message,
                    language,
                )
                if not _planner_response_violates_explicit_preference_contract(finalized_plan, user_message):
                    return finalized_plan

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
                    conversation_context=conversation_context,
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
                conversation_context=conversation_context,
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
            fallback = _strip_unrequested_live_departure_lines(fallback, user_message)
            fallback = _ensure_requested_origin_target_in_transport_section(
                fallback,
                user_message,
                language,
                transport_data,
            )
            return _ensure_multi_day_response_quality(
                fallback,
                user_message=user_message,
                language=language,
                weather_data=weather_data,
                transport_data=transport_data,
                places_data=places_data,
                events_data=events_data,
                qa_disclaimers=qa_disclaimers,
                conversation_context=conversation_context,
            )

        if (
            _planner_response_requires_fallback(cleaned_response)
            or _planner_response_has_markdown_contract_defects(cleaned_response)
            or not _planner_response_matches_schema(cleaned_response)
            or not _planner_response_has_minimum_user_value(cleaned_response)
            or _planner_response_missing_requested_day_sections(cleaned_response, user_message)
            or _planner_response_missing_requested_movement(cleaned_response, user_message, transport_data)
            or _planner_response_missing_requested_food_stop(cleaned_response, user_message)
            or _planner_response_missing_requested_plan_components(cleaned_response, user_message)
            or _planner_response_missing_requested_stops(
                cleaned_response,
                user_message,
                "\n".join([places_data or "", events_data or ""]),
            )
            or _planner_response_violates_requested_start(cleaned_response, user_message)
            or _planner_response_violates_explicit_preference_contract(cleaned_response, user_message)
            or _planner_response_has_local_area_drift(cleaned_response, user_message)
            or _planner_response_mixes_distant_walking_areas(cleaned_response, user_message)
            or _planner_response_uses_excluded_area(cleaned_response, user_message)
        ):
            cleaned_response = _build_specific_planner_fallback(
                user_message=user_message,
                language=language,
                weather_data=weather_data,
                transport_data=transport_data,
                places_data=places_data,
                events_data=events_data,
                qa_disclaimers=qa_disclaimers,
                conversation_context=conversation_context,
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
        grounding_issues = _add_requested_stop_issue(
            grounding_issues,
            cleaned_response,
            user_message,
            "\n".join([places_data or "", events_data or ""]),
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
            grounding_issues = _add_requested_stop_issue(
                grounding_issues,
                cleaned_response,
                user_message,
                "\n".join([places_data or "", events_data or ""]),
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
            or _planner_response_missing_requested_day_sections(cleaned_response, user_message)
            or _planner_response_missing_requested_movement(cleaned_response, user_message, transport_data)
            or _planner_response_missing_requested_food_stop(cleaned_response, user_message)
            or _planner_response_missing_requested_plan_components(cleaned_response, user_message)
            or _planner_response_missing_requested_stops(
                cleaned_response,
                user_message,
                "\n".join([places_data or "", events_data or ""]),
            )
            or _planner_response_violates_requested_start(cleaned_response, user_message)
            or _planner_response_has_local_area_drift(cleaned_response, user_message)
            or _planner_response_mixes_distant_walking_areas(cleaned_response, user_message)
            or _planner_response_has_unrequested_sequence_stops(cleaned_response, user_message)
            or _planner_response_has_low_fit_infrastructure_stop(cleaned_response, user_message)
            or _planner_response_has_incomplete_museum_day_blocks(user_message, cleaned_response)
            or _planner_response_uses_excluded_area(cleaned_response, user_message)
        ):
            cleaned_response = _build_specific_planner_fallback(
                user_message=user_message,
                language=language,
                weather_data=weather_data,
                transport_data=transport_data,
                places_data=places_data,
                events_data=events_data,
                qa_disclaimers=qa_disclaimers,
                conversation_context=conversation_context,
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
                conversation_context=conversation_context,
            )
            if _planner_response_missing_requested_food_stop(cleaned_response, user_message):
                card_meal_repair = _build_card_based_itinerary_fallback(
                    user_message=user_message,
                    language=language,
                    weather_data=weather_data,
                    transport_data=transport_data,
                    places_data=places_data,
                    events_data=events_data,
                    qa_disclaimers=qa_disclaimers,
                    conversation_context=conversation_context,
                )
                if card_meal_repair and not _planner_response_missing_requested_food_stop(
                    card_meal_repair,
                    user_message,
                ):
                    cleaned_response = card_meal_repair
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

        cleaned_response = _strip_unrequested_live_departure_lines(cleaned_response, user_message)
        cleaned_response = _strip_planner_movement_placeholders(cleaned_response)
        cleaned_response = _strip_irrelevant_planner_movement_items(
            cleaned_response,
            user_message,
            language,
        )
        cleaned_response = _ensure_partial_planner_movement_limitation(
            cleaned_response,
            user_message,
            language,
        )
        cleaned_response = _enforce_walking_only_movement(cleaned_response, user_message, language)
        cleaned_response = _ensure_requested_origin_target_in_transport_section(
            cleaned_response,
            user_message,
            language,
            transport_data,
        )
        cleaned_response = _ensure_requested_return_to_origin_in_transport_section(
            cleaned_response,
            user_message,
            language,
        )
        cleaned_response = _repair_planner_address_map_links(cleaned_response)
        cleaned_response = _repair_visible_transport_sources(cleaned_response)
        if _planner_response_has_markdown_contract_defects(cleaned_response):
            card_repair = _build_card_based_itinerary_fallback(
                user_message=user_message,
                language=language,
                weather_data=weather_data,
                transport_data=transport_data,
                places_data=places_data,
                events_data=events_data,
                qa_disclaimers=qa_disclaimers,
                conversation_context=conversation_context,
            )
            if card_repair and not _planner_response_has_markdown_contract_defects(card_repair):
                cleaned_response = card_repair
        if _planner_response_missing_requested_food_stop(cleaned_response, user_message):
            card_meal_repair = _build_card_based_itinerary_fallback(
                user_message=user_message,
                language=language,
                weather_data=weather_data,
                transport_data=transport_data,
                places_data=places_data,
                events_data=events_data,
                qa_disclaimers=qa_disclaimers,
                conversation_context=conversation_context,
            )
            if card_meal_repair and not _planner_response_missing_requested_food_stop(
                card_meal_repair,
                user_message,
            ):
                cleaned_response = card_meal_repair

        cleaned_response = _ensure_requested_return_to_origin_in_transport_section(
            cleaned_response,
            user_message,
            language,
        )

        cleaned_response = _repair_response_requested_type_counts(cleaned_response, user_message)
        final_response = _ensure_multi_day_response_quality(
            cleaned_response,
            user_message=user_message,
            language=language,
            weather_data=weather_data,
            transport_data=transport_data,
            places_data=places_data,
            events_data=events_data,
            qa_disclaimers=qa_disclaimers,
            conversation_context=conversation_context,
        )
        final_response = _repair_response_requested_type_counts(final_response, user_message)
        return _ensure_stop_by_stop_movement_in_response(final_response, user_message, language)

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
        output_language = str(agent_outputs.get("_language") or "").strip().lower()

        return self.invoke(
            user_message=user_message,
            weather_data=agent_outputs.get("weather", ""),
            transport_data=agent_outputs.get("transport", ""),
            places_data=agent_outputs.get("researcher", ""),
            events_data="",  # Events come from researcher too
            qa_disclaimers=qa_disclaimers,
            conversation_context=conversation_context,
            output_language=output_language if output_language in {"pt", "en"} else None,
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
